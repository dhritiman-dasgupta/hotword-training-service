#!/usr/bin/env python3
"""
Hotword training pipeline orchestrator.

Chains the stages:
  1. Piper synthetic generation   (openWakeWord train.py --generate_clips)
  2. F5-TTS human/cloned voices   (f5_generate.py, dropped into positive dirs)
  3. Augmentation (RIR + noise)   (train.py --augment_clips)
  4. Train + export ONNX          (train.py --train_model)

Runs in the API venv; shells out to the train/tts venvs. No heavy imports here.
"""
import os
import re
import json
import time
import shutil
import subprocess
import datetime

import yaml

# ---- fixed paths on the instance ----
HOME = "/opt/hotword"
TRAIN_PY = f"{HOME}/openWakeWord/openwakeword/train.py"
TRAIN_PYTHON = f"{HOME}/venvs/train/bin/python"
TTS_PYTHON = f"{HOME}/venvs/tts/bin/python"
F5_GEN = f"{HOME}/server/f5_generate.py"
PSG = f"{HOME}/piper-sample-generator-oww"
DATA = f"{HOME}/data"
JOBS = f"{HOME}/jobs"
STATE = f"{HOME}/state"

NEG_FEATURES = f"{DATA}/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
VAL_FEATURES = f"{DATA}/validation_set_features.npy"
RIR_DIR = f"{DATA}/mit_rirs"
BG_DIR = f"{DATA}/fma"


def _now():
    return datetime.datetime.utcnow().isoformat() + "Z"


def slugify(text):
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s or "wakeword"


def set_status(job_dir, **kw):
    path = os.path.join(job_dir, "status.json")
    cur = {}
    if os.path.exists(path):
        with open(path) as f:
            cur = json.load(f)
    cur.update(kw)
    cur["updated"] = _now()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cur, f, indent=2)
    os.replace(tmp, path)
    return cur


def get_status(job_dir):
    path = os.path.join(job_dir, "status.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _log(job_dir, msg):
    with open(os.path.join(job_dir, "job.log"), "a") as f:
        f.write(f"[{_now()}] {msg}\n")


def _run(job_dir, cmd, cwd=HOME, env_extra=None):
    """Run a subprocess, streaming combined output to the job log. Returns exit code."""
    _log(job_dir, "RUN: " + " ".join(cmd))
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    logf = open(os.path.join(job_dir, "job.log"), "a")
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=logf,
                         stderr=subprocess.STDOUT, bufsize=1)
    p.wait()
    logf.flush()
    logf.close()
    _log(job_dir, f"EXIT {p.returncode}: {cmd[0]} ...")
    return p.returncode


def build_config(job_dir, params):
    phrase = params["wake_word"]
    name = slugify(phrase)
    work = os.path.join(job_dir, "work")
    os.makedirs(work, exist_ok=True)
    cfg = {
        "model_name": name,
        "target_phrase": [phrase],
        "custom_negative_phrases": params.get("custom_negative_phrases", []),
        "n_samples": int(params.get("n_samples", 5000)),
        "n_samples_val": int(params.get("n_samples_val", 1000)),
        "tts_batch_size": int(params.get("tts_batch_size", 50)),
        "augmentation_batch_size": int(params.get("augmentation_batch_size", 16)),
        "piper_sample_generator_path": PSG,
        "output_dir": work,
        "rir_paths": [RIR_DIR],
        "background_paths": [BG_DIR],
        "background_paths_duplication_rate": [1],
        "false_positive_validation_data_path": VAL_FEATURES,
        "augmentation_rounds": int(params.get("augmentation_rounds", 1)),
        "feature_data_files": {"ACAV100M_sample": NEG_FEATURES},
        "batch_n_per_class": {"ACAV100M_sample": 1024, "adversarial_negative": 50, "positive": 50},
        "model_type": "dnn",
        "layer_size": int(params.get("layer_size", 32)),
        "steps": int(params.get("steps", 20000)),
        "max_negative_weight": int(params.get("max_negative_weight", 1500)),
        "target_false_positives_per_hour": float(params.get("target_false_positives_per_hour", 0.2)),
    }
    cfg_path = os.path.join(job_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path, cfg, name, work


def run_job(job_id, params):
    """Full pipeline for one job. Updates status.json at each stage."""
    job_dir = os.path.join(JOBS, job_id)
    os.makedirs(job_dir, exist_ok=True)
    try:
        set_status(job_dir, job_id=job_id, wake_word=params["wake_word"],
                   state="running", stage="config", progress=5, error=None,
                   params=params, started=_now())
        cfg_path, cfg, name, work = build_config(job_dir, params)
        _log(job_dir, f"config written: {cfg_path}")

        # preflight: required data present?
        missing = [p for p in (NEG_FEATURES, VAL_FEATURES) if not os.path.exists(p)]
        if not os.path.isdir(RIR_DIR) or not os.path.isdir(BG_DIR):
            missing.append("rir/background dirs")
        if missing:
            raise RuntimeError(f"required training data missing: {missing}. "
                               f"Run data setup first.")

        common = [TRAIN_PYTHON, TRAIN_PY, "--training_config", cfg_path]

        # 1) Piper generation
        set_status(job_dir, stage="generate_piper", progress=15)
        rc = _run(job_dir, common + ["--generate_clips"])
        if rc != 0:
            raise RuntimeError("piper generation failed (see job.log)")

        # 2) F5-TTS human/cloned voices -> positive dirs
        if params.get("use_f5", True):
            set_status(job_dir, stage="generate_f5", progress=40)
            pos_train = os.path.join(work, name, "positive_train")
            pos_test = os.path.join(work, name, "positive_test")
            os.makedirs(pos_train, exist_ok=True)
            os.makedirs(pos_test, exist_ok=True)
            f5_n = int(params.get("f5_samples", 150))
            ref_audio = params.get("ref_audio", "") or ""
            ref_text = params.get("ref_text", "") or ""
            base = [TTS_PYTHON, F5_GEN, "--text", params["wake_word"]]
            if ref_audio:
                base += ["--ref-audio", ref_audio, "--ref-text", ref_text]
            _run(job_dir, base + ["--out-dir", pos_train, "--n", str(f5_n), "--prefix", "f5tr"])
            _run(job_dir, base + ["--out-dir", pos_test, "--n", str(max(20, f5_n // 5)), "--prefix", "f5te"])
        else:
            _log(job_dir, "F5 generation disabled by request")

        # 3) Augmentation
        set_status(job_dir, stage="augment", progress=60)
        rc = _run(job_dir, common + ["--augment_clips", "--overwrite"])
        if rc != 0:
            raise RuntimeError("augmentation failed (see job.log)")

        # 4) Train + export ONNX
        set_status(job_dir, stage="train", progress=75)
        rc = _run(job_dir, common + ["--train_model"])
        if rc != 0:
            raise RuntimeError("training failed (see job.log)")

        # collect model
        onnx_src = os.path.join(work, name + ".onnx")
        if not os.path.exists(onnx_src):
            # some versions write into the model subdir
            alt = os.path.join(work, name, name + ".onnx")
            onnx_src = alt if os.path.exists(alt) else onnx_src
        if not os.path.exists(onnx_src):
            raise RuntimeError("training finished but no .onnx produced")
        out_model = os.path.join(job_dir, name + ".onnx")
        shutil.copy2(onnx_src, out_model)

        set_status(job_dir, state="done", stage="done", progress=100,
                   model_file=out_model, model_name=name, finished=_now())
        _log(job_dir, f"DONE -> {out_model}")
        return out_model
    except Exception as e:
        _log(job_dir, f"ERROR: {e}")
        set_status(job_dir, state="error", error=str(e), finished=_now())
        raise


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--wake-word", required=True)
    ap.add_argument("--job-id", default="cli_" + str(int(time.time())))
    ap.add_argument("--n-samples", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--f5-samples", type=int, default=150)
    ap.add_argument("--no-f5", action="store_true")
    a = ap.parse_args()
    m = run_job(a.job_id, {
        "wake_word": a.wake_word, "n_samples": a.n_samples, "steps": a.steps,
        "f5_samples": a.f5_samples, "use_f5": not a.no_f5,
    })
    print("MODEL:", m)
