#!/usr/bin/env python3
"""
Hotword training API service.

POST a wake word -> trains a local openWakeWord ONNX model using Piper (bulk
accents) + F5-TTS (human/cloned voices) -> download the model.

Runs in the API venv (fastapi/uvicorn). Heavy work is shelled out to GPU venvs
by pipeline.py. One training job runs at a time (single GPU), the rest queue.
"""
import os
import time
import json
import uuid
import queue
import threading
import subprocess

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel

import pipeline as P

VERSION = "1.0"
STATE = P.STATE
JOBS = P.JOBS
os.makedirs(STATE, exist_ok=True)
os.makedirs(JOBS, exist_ok=True)

API_KEY_FILE = os.path.join(STATE, "api_key")
API_KEY = open(API_KEY_FILE).read().strip() if os.path.exists(API_KEY_FILE) else None
LAST_ACTIVITY = os.path.join(STATE, "last_activity")

app = FastAPI(title="Hotword Training Service", version=VERSION)

# ---- single-worker job queue ----
_q: "queue.Queue[str]" = queue.Queue()


def _worker():
    while True:
        job_id = _q.get()
        try:
            job_dir = os.path.join(JOBS, job_id)
            st = P.get_status(job_dir) or {}
            P.run_job(job_id, st.get("params", {}))
        except Exception as e:  # already recorded in status.json
            print(f"[worker] job {job_id} failed: {e}", flush=True)
        finally:
            _q.task_done()


threading.Thread(target=_worker, daemon=True).start()


def touch_activity():
    try:
        with open(LAST_ACTIVITY, "w") as f:
            f.write(str(int(time.time())))
    except Exception:
        pass


@app.middleware("http")
async def activity_mw(request: Request, call_next):
    touch_activity()
    return await call_next(request)


def require_key(request: Request):
    if API_KEY:
        if request.headers.get("x-api-key") != API_KEY:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


# ---- models ----
class TrainReq(BaseModel):
    wake_word: str
    n_samples: int = 5000
    n_samples_val: int = 1000
    steps: int = 20000
    f5_samples: int = 150
    use_f5: bool = True
    custom_negative_phrases: list[str] = []
    target_false_positives_per_hour: float = 0.2


def _data_ready():
    ok = os.path.exists(P.NEG_FEATURES) and os.path.exists(P.VAL_FEATURES)
    ok = ok and os.path.isdir(P.RIR_DIR) and len(os.listdir(P.RIR_DIR)) > 0
    ok = ok and os.path.isdir(P.BG_DIR) and len(os.listdir(P.BG_DIR)) > 0
    return ok


def _enqueue(params) -> str:
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    job_dir = os.path.join(JOBS, job_id)
    os.makedirs(job_dir, exist_ok=True)
    P.set_status(job_dir, job_id=job_id, wake_word=params["wake_word"],
                 state="queued", stage="queued", progress=0, params=params,
                 created=P._now())
    _q.put(job_id)
    return job_id


# ---- endpoints ----
@app.get("/health")
def health():
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
    except Exception:
        gpu = "none"
    return {"status": "ok", "version": VERSION, "gpu": gpu,
            "data_ready": _data_ready(),
            "queue_depth": _q.qsize(),
            "auth_required": bool(API_KEY)}


@app.post("/train")
def train(req: TrainReq, _=Depends(require_key)):
    if not req.wake_word.strip():
        raise HTTPException(400, "wake_word is required")
    if not _data_ready():
        raise HTTPException(503, "training data not ready yet (downloads still running)")
    job_id = _enqueue(req.model_dump())
    return {"job_id": job_id, "state": "queued",
            "poll": f"/jobs/{job_id}", "model": f"/jobs/{job_id}/model"}


@app.post("/train-clone")
def train_clone(
    wake_word: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(""),
    n_samples: int = Form(5000),
    steps: int = Form(20000),
    f5_samples: int = Form(300),
    _=Depends(require_key),
):
    """Train with a custom reference voice to clone (multipart upload)."""
    if not _data_ready():
        raise HTTPException(503, "training data not ready yet")
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    job_dir = os.path.join(JOBS, job_id)
    os.makedirs(job_dir, exist_ok=True)
    ref_path = os.path.join(job_dir, "ref_voice.wav")
    with open(ref_path, "wb") as f:
        f.write(ref_audio.file.read())
    params = {"wake_word": wake_word, "n_samples": n_samples, "steps": steps,
              "f5_samples": f5_samples, "use_f5": True,
              "ref_audio": ref_path, "ref_text": ref_text}
    P.set_status(job_dir, job_id=job_id, wake_word=wake_word, state="queued",
                 stage="queued", progress=0, params=params, created=P._now())
    _q.put(job_id)
    return {"job_id": job_id, "state": "queued", "poll": f"/jobs/{job_id}"}


@app.get("/jobs")
def list_jobs():
    out = []
    for jid in sorted(os.listdir(JOBS), reverse=True):
        st = P.get_status(os.path.join(JOBS, jid))
        if st:
            out.append({k: st.get(k) for k in
                        ("job_id", "wake_word", "state", "stage", "progress", "updated")})
    return out


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job_dir = os.path.join(JOBS, job_id)
    st = P.get_status(job_dir)
    if not st:
        raise HTTPException(404, "no such job")
    log_tail = ""
    logp = os.path.join(job_dir, "job.log")
    if os.path.exists(logp):
        with open(logp) as f:
            log_tail = "".join(f.readlines()[-25:])
    st["log_tail"] = log_tail
    return st


@app.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
def job_log(job_id: str):
    logp = os.path.join(JOBS, job_id, "job.log")
    if not os.path.exists(logp):
        raise HTTPException(404, "no log")
    return open(logp).read()


@app.get("/jobs/{job_id}/model")
def job_model(job_id: str):
    st = P.get_status(os.path.join(JOBS, job_id))
    if not st:
        raise HTTPException(404, "no such job")
    if st.get("state") != "done" or not st.get("model_file"):
        raise HTTPException(409, f"model not ready (state={st.get('state')})")
    mf = st["model_file"]
    if not os.path.exists(mf):
        raise HTTPException(410, "model file missing")
    return FileResponse(mf, filename=os.path.basename(mf),
                        media_type="application/octet-stream")


@app.get("/")
def root():
    return JSONResponse({"service": "hotword-training", "version": VERSION,
                         "docs": "/docs", "health": "/health"})
