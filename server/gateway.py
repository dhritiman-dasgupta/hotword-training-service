#!/usr/bin/env python3
"""
Always-on gateway API (runs on the small t3.micro).

It is the public training API. It powers the GPU spot instance ON when a training
job arrives, proxies the job to the GPU's training API, caches the job status and the
resulting ONNX locally, and powers the GPU OFF again when no jobs are running — so the
expensive GPU only runs during actual training while this endpoint stays reachable 24/7.

Exposed (behind Caddy) at https://<host>/api  -> 127.0.0.1:8001
"""
import os
import json
import time
import threading

import boto3
import requests
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

REGION = "ap-south-1"
GPU_INSTANCE = "i-089730e889eb1fb3d"
GPU_API = "http://13.200.68.170:8000"
CACHE = "/opt/gateway/cache"
os.makedirs(CACHE, exist_ok=True)

API_KEY = open("/opt/gateway/api_key").read().strip()
GPU_API_KEY = open("/opt/gateway/gpu_api_key").read().strip()
GHEAD = {"X-API-Key": GPU_API_KEY}

ec2 = boto3.client("ec2", region_name=REGION)
app = FastAPI(title="Hotword Gateway", version="1.0")
_power_lock = threading.Lock()


# ---------- GPU power control ----------
def gpu_state():
    r = ec2.describe_instances(InstanceIds=[GPU_INSTANCE])
    return r["Reservations"][0]["Instances"][0]["State"]["Name"]


def gpu_api_ok(timeout=4):
    try:
        return requests.get(f"{GPU_API}/health", headers=GHEAD, timeout=timeout).ok
    except Exception:
        return False


def ensure_gpu_running(max_wait=240):
    with _power_lock:
        st = gpu_state()
        if st in ("stopping",):
            ec2.get_waiter("instance_stopped").wait(InstanceIds=[GPU_INSTANCE])
            st = "stopped"
        if st in ("stopped",):
            ec2.start_instances(InstanceIds=[GPU_INSTANCE])
        if st != "running":
            ec2.get_waiter("instance_running").wait(InstanceIds=[GPU_INSTANCE])
            ec2.get_waiter("instance_status_ok").wait(InstanceIds=[GPU_INSTANCE])
    # wait for the training API to answer
    deadline = time.time() + max_wait
    while time.time() < deadline:
        if gpu_api_ok():
            return True
        time.sleep(5)
    return gpu_api_ok()


def gpu_jobs_active():
    try:
        jobs = requests.get(f"{GPU_API}/jobs", headers=GHEAD, timeout=8).json()
        return any(j.get("state") in ("queued", "running") for j in jobs)
    except Exception:
        return False


def maybe_stop_gpu():
    with _power_lock:
        if not gpu_jobs_active() and gpu_state() == "running":
            ec2.stop_instances(InstanceIds=[GPU_INSTANCE])
            return True
    return False


# ---------- local cache ----------
def cache_status(job_id, status):
    json.dump(status, open(os.path.join(CACHE, job_id + ".json"), "w"))


def read_cached_status(job_id):
    p = os.path.join(CACHE, job_id + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def cached_model(job_id):
    p = os.path.join(CACHE, job_id + ".onnx")
    return p if os.path.exists(p) else None


# ---------- background watcher ----------
def watch(job_id):
    """Poll GPU until job ends; cache status + model; then power GPU down if idle."""
    while True:
        try:
            s = requests.get(f"{GPU_API}/jobs/{job_id}", headers=GHEAD, timeout=10).json()
        except Exception:
            time.sleep(10)
            if not gpu_api_ok():
                break
            continue
        cache_status(job_id, s)
        if s.get("state") in ("done", "error"):
            if s.get("state") == "done":
                try:
                    r = requests.get(f"{GPU_API}/jobs/{job_id}/model", headers=GHEAD, timeout=60)
                    if r.ok:
                        open(os.path.join(CACHE, job_id + ".onnx"), "wb").write(r.content)
                except Exception:
                    pass
            break
        time.sleep(15)
    time.sleep(5)
    maybe_stop_gpu()


# ---------- auth ----------
def require_key(request: Request):
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(401, "invalid or missing X-API-Key")


class TrainReq(BaseModel):
    wake_word: str
    n_samples: int = 5000
    n_samples_val: int = 1000
    steps: int = 20000
    f5_samples: int = 150
    use_f5: bool = True
    custom_negative_phrases: list[str] = []
    target_false_positives_per_hour: float = 0.2


# ---------- endpoints ----------
@app.get("/health")
def health():
    st = gpu_state()
    return {"gateway": "ok", "gpu_instance": GPU_INSTANCE, "gpu_state": st,
            "gpu_api_reachable": gpu_api_ok() if st == "running" else False}


@app.post("/train")
def train(req: TrainReq, _=Depends(require_key)):
    if not ensure_gpu_running():
        raise HTTPException(503, "could not bring GPU instance online; try again shortly")
    r = requests.post(f"{GPU_API}/train", headers=GHEAD, json=req.model_dump(), timeout=30)
    if not r.ok:
        raise HTTPException(r.status_code, r.text)
    data = r.json()
    threading.Thread(target=watch, args=(data["job_id"],), daemon=True).start()
    return data


@app.post("/train-clone")
def train_clone(wake_word: str = Form(...), ref_audio: UploadFile = File(...),
                ref_text: str = Form(""), n_samples: int = Form(5000),
                steps: int = Form(20000), f5_samples: int = Form(300), _=Depends(require_key)):
    if not ensure_gpu_running():
        raise HTTPException(503, "could not bring GPU instance online")
    files = {"ref_audio": (ref_audio.filename, ref_audio.file.read(), "audio/wav")}
    form = {"wake_word": wake_word, "ref_text": ref_text, "n_samples": str(n_samples),
            "steps": str(steps), "f5_samples": str(f5_samples)}
    r = requests.post(f"{GPU_API}/train-clone", headers=GHEAD, data=form, files=files, timeout=60)
    if not r.ok:
        raise HTTPException(r.status_code, r.text)
    data = r.json()
    threading.Thread(target=watch, args=(data["job_id"],), daemon=True).start()
    return data


@app.get("/jobs")
def jobs():
    if gpu_state() == "running" and gpu_api_ok():
        try:
            return requests.get(f"{GPU_API}/jobs", headers=GHEAD, timeout=10).json()
        except Exception:
            pass
    out = []
    for f in sorted(os.listdir(CACHE), reverse=True):
        if f.endswith(".json"):
            s = json.load(open(os.path.join(CACHE, f)))
            out.append({k: s.get(k) for k in ("job_id", "wake_word", "state", "stage", "progress")})
    return out


@app.get("/jobs/{job_id}")
def job(job_id: str):
    cached = read_cached_status(job_id)
    if cached and cached.get("state") in ("done", "error"):
        return cached
    if gpu_state() == "running" and gpu_api_ok():
        try:
            s = requests.get(f"{GPU_API}/jobs/{job_id}", headers=GHEAD, timeout=10).json()
            cache_status(job_id, s)
            return s
        except Exception:
            pass
    if cached:
        return cached
    raise HTTPException(404, "unknown job (and GPU instance is offline)")


@app.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
def job_log(job_id: str):
    if gpu_state() == "running" and gpu_api_ok():
        r = requests.get(f"{GPU_API}/jobs/{job_id}/log", headers=GHEAD, timeout=10)
        return r.text
    s = read_cached_status(job_id)
    return (s or {}).get("log_tail", "GPU offline and no cached log")


@app.get("/jobs/{job_id}/model")
def job_model(job_id: str):
    p = cached_model(job_id)
    if p:
        return FileResponse(p, filename=f"{job_id}.onnx", media_type="application/octet-stream")
    if gpu_state() == "running" and gpu_api_ok():
        r = requests.get(f"{GPU_API}/jobs/{job_id}/model", headers=GHEAD, timeout=60)
        if r.ok:
            open(os.path.join(CACHE, job_id + ".onnx"), "wb").write(r.content)
            return FileResponse(os.path.join(CACHE, job_id + ".onnx"),
                                filename=f"{job_id}.onnx", media_type="application/octet-stream")
        raise HTTPException(r.status_code, r.text)
    raise HTTPException(409, "model not cached and GPU instance is offline")


@app.post("/gpu/stop")
def gpu_stop(_=Depends(require_key)):
    return {"stopped": maybe_stop_gpu(), "state": gpu_state()}


@app.get("/")
def root():
    return {"service": "hotword-gateway", "endpoints": ["/health", "/train", "/jobs/{id}", "/jobs/{id}/model"]}
