# Hotword Detector — API Documentation

Two parts:

1. **Training API** — POST a wake word, get a trained `.onnx` wake-word detector.
2. **Inference** — run that `.onnx` to detect the wake word (Python, browser, or raw ONNX).

---

# Part 1 — Training API (generate the ONNX)

**Base URL:** `https://65-2-7-128.sslip.io/api`  (always-on gateway, HTTPS)
**Auth:** every request needs header `X-API-Key: <key>`
The key lives in `deploy/api_key.txt`.
**Interactive docs (OpenAPI/Swagger):** `https://65-2-7-128.sslip.io/api/docs`

### Power management (automatic — you don't start/stop anything)
The base URL is an always-on gateway (small t3.micro). The GPU instance is **off** by
default. When you `POST /train`, the gateway **boots the GPU automatically**, runs the job,
caches the result, then **shuts the GPU down** once no jobs remain. Just call the API.

- The first `/train` after the GPU has been idle takes **~90–150 s extra** while it boots.
- `/jobs/{id}` and `/jobs/{id}/model` are served from the gateway's cache **even after the
  GPU is powered off** — so polling and downloads never wake the GPU unnecessarily.
- `POST /gpu/stop` forces the GPU off immediately (no-op if a job is running).
- **Direct/advanced:** `http://13.200.68.170:8000` talks straight to the GPU box, but is only
  reachable while it happens to be running. Prefer the gateway.

---

## Quickstart (full flow)

```bash
KEY=<your-api-key>
BASE=https://65-2-7-128.sslip.io/api

# 1. submit a wake word
JOB=$(curl -s -X POST $BASE/train \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"wake_word":"hey kiki"}' | jq -r .job_id)

# 2. poll until done
while [ "$(curl -s $BASE/jobs/$JOB -H "X-API-Key: $KEY" | jq -r .state)" = "running" ]; do sleep 15; done

# 3. download the ONNX detector
curl -s $BASE/jobs/$JOB/model -H "X-API-Key: $KEY" -o hey_kiki.onnx
```

---

## Endpoints

### `GET /health`
No side effects. Returns service status.
```json
{ "status":"ok", "version":"1.0", "gpu":"Tesla T4, 15360 MiB",
  "data_ready":true, "queue_depth":0, "auth_required":true }
```
`data_ready:false` means the training datasets are still downloading — wait before training.

---

### `POST /train`
Queue a training job. Body is JSON.

| Field | Type | Default | Description |
|---|---|---|---|
| `wake_word` | string | **required** | The phrase to detect, e.g. `"hey kiki"` |
| `n_samples` | int | 5000 | Positive samples to synthesize (Piper). More = better, slower |
| `n_samples_val` | int | 1000 | Validation positives |
| `steps` | int | 20000 | Training steps |
| `f5_samples` | int | 150 | Extra human/cloned-voice samples from F5-TTS |
| `use_f5` | bool | true | Include F5-TTS samples |
| `custom_negative_phrases` | string[] | [] | Phrases to explicitly NOT trigger on |
| `target_false_positives_per_hour` | float | 0.2 | Tuning target |

**Response** `200`:
```json
{ "job_id":"20260614-021500-ab12cd", "state":"queued",
  "poll":"/jobs/20260614-021500-ab12cd", "model":"/jobs/20260614-021500-ab12cd/model" }
```

**Example — Python**
```python
import requests, time
BASE="https://65-2-7-128.sslip.io/api"; H={"X-API-Key":"<key>"}
jid = requests.post(f"{BASE}/train", headers=H,
                    json={"wake_word":"hey kiki","n_samples":20000,"steps":50000}).json()["job_id"]
while requests.get(f"{BASE}/jobs/{jid}", headers=H).json()["state"] == "running":
    time.sleep(20)
open("hey_kiki.onnx","wb").write(requests.get(f"{BASE}/jobs/{jid}/model", headers=H).content)
```

**Example — JavaScript (fetch)**
```js
const BASE="https://65-2-7-128.sslip.io/api", H={"X-API-Key":"<key>","Content-Type":"application/json"};
const { job_id } = await (await fetch(`${BASE}/train`,{method:"POST",headers:H,
  body:JSON.stringify({wake_word:"hey kiki"})})).json();
let s; do { await new Promise(r=>setTimeout(r,15000));
  s = await (await fetch(`${BASE}/jobs/${job_id}`,{headers:H})).json();
} while (s.state==="running");
const blob = await (await fetch(`${BASE}/jobs/${job_id}/model`,{headers:H})).blob();
```

---

### `POST /train-clone`  (clone a specific voice)
`multipart/form-data`. Biases the model toward a voice you upload.

| Field | Type | Description |
|---|---|---|
| `wake_word` | form text | required |
| `ref_audio` | file (wav) | 5–15 s clean clip of the target voice |
| `ref_text` | form text | transcript of the reference clip (improves cloning) |
| `n_samples`, `steps`, `f5_samples` | form text | optional, same as `/train` |

```bash
curl -X POST $BASE/train-clone -H "X-API-Key: $KEY" \
  -F wake_word="hey kiki" -F ref_audio=@myvoice.wav \
  -F ref_text="hello this is my voice"
```

---

### `GET /jobs`
List all jobs (id, wake_word, state, stage, progress, updated).

### `GET /jobs/{id}`
Full job status:
```json
{ "job_id":"...", "wake_word":"hey kiki", "state":"running",
  "stage":"train", "progress":75, "error":null, "log_tail":"...last 25 log lines..." }
```
**states:** `queued` → `running` → `done` | `error`
**stages:** `config` → `generate_piper` → `generate_f5` → `augment` → `train` → `done`

### `GET /jobs/{id}/log`
Full training log (plain text).

### `GET /jobs/{id}/model`
Downloads the trained `.onnx`. `409` if the model isn't ready yet.

---

## Errors
| Code | Meaning |
|---|---|
| `401` | missing/invalid `X-API-Key` |
| `503` | training data not ready yet (`data_ready:false`) |
| `409` | model requested but job not `done` |
| `404` | unknown job id |

## Tuning
- Quick test: `n_samples=2000, steps=5000` (~5–10 min).
- Production: `n_samples=20000+, steps=50000` (~1–2 h). Bigger = higher recall, fewer false triggers.
- Wake words of 3–4 syllables ("hey kiki") work better than single common words.

---

# Part 2 — Inference (run the ONNX)

The output is a standard **openWakeWord** model. It is **not** a single end-to-end ONNX:
detection uses a 3-stage pipeline (two shared feature models + your wake-word model).
You can use the high-level library (easy) or the raw ONNX graph (full control).

## A) Python (recommended)

```bash
pip install openwakeword onnxruntime sounddevice numpy
```

**Live mic:**
```python
from openwakeword.model import Model
import sounddevice as sd, numpy as np

oww = Model(wakeword_models=["hey_kiki.onnx"], inference_framework="onnx")
name = list(oww.models.keys())[0]

def cb(indata, frames, t, status):
    pcm = (indata[:,0]*32767).astype(np.int16)
    score = oww.predict(pcm)[name]
    if score > 0.5:
        print("DETECTED", round(score,3))

with sd.InputStream(channels=1, samplerate=16000, blocksize=1280,
                    dtype="float32", callback=cb):
    input("listening… enter to quit\n")
```

**Score a file** (16 kHz mono, 1280-sample frames):
```python
import numpy as np, scipy.io.wavfile as wav
from openwakeword.model import Model
sr, data = wav.read("clip16k.wav")               # must be 16 kHz, 16-bit
oww = Model(wakeword_models=["hey_kiki.onnx"], inference_framework="onnx")
name = list(oww.models.keys())[0]
scores = [oww.predict(data[i:i+1280].astype(np.int16))[name]
          for i in range(0, len(data)-1280, 1280)]
print("peak confidence:", max(scores))
```

## B) Browser (onnxruntime-web)
A full working client-side implementation is deployed and open-source:
- Live app: `https://65-2-7-128.sslip.io/`
- Code: `github.com/dhritiman-dasgupta/hey-kiki-wakeword-web` (`pipeline.js` is a self-contained, dependency-free port)

Load `melspectrogram.onnx`, `embedding_model.onnx`, and your `hey_kiki.onnx` with
`onnxruntime-web`, capture mic at 16 kHz, and feed 1280-sample chunks through the pipeline
below.

## C) Raw ONNX spec (for custom integrations)
Three ONNX models run in series. Audio is **16 kHz mono, 16-bit PCM** (values in int16
range, i.e. float samples × 32767 — **not** normalized to [-1,1]).

| Model | Input (name, shape, dtype) | Output | Notes |
|---|---|---|---|
| `melspectrogram.onnx` | `input` `[1, N]` f32 | `[1,1,frames,32]` | `frames ≈ ceil(N/160 - 3)`. **Apply transform `x/10 + 2`** to every value. |
| `embedding_model.onnx` | `input_1` `[1,76,32,1]` f32 | `[…,96]` | 76 mel frames → one 96-dim embedding |
| `hey_kiki.onnx` (yours) | `[1,16,96]` f32 | `[1,1]` | 16 embeddings → score in **[0,1]** (sigmoid applied) |

**Streaming algorithm** (per 1280-sample / 80 ms chunk):
1. Append chunk to a raw-audio ring buffer.
2. Run `melspectrogram` on the **last 1760 samples** (`1280 + 480`) → ~8 new mel frames; apply `x/10+2`; append to a mel buffer (init the mel buffer with 76 rows of `1.0`).
3. Take the **last 76 mel frames** → run `embedding_model` → append the 96-dim vector to a feature buffer.
4. Once ≥16 features exist, take the **last 16** → run your wake-word model → **score**.
5. Trigger when `score ≥ threshold` (start at `0.5`); add a refractory period (~1.5 s) to avoid repeats.

This algorithm is implemented and **numerically verified to match openWakeWord's Python
reference to < 1e-7** in `web/pipeline.js` (browser) and `web-verify/` (Node).

---

## Appendix — get the feature models
`melspectrogram.onnx` and `embedding_model.onnx` are shared across all wake words:
- Python: `python -c "import openwakeword; openwakeword.utils.download_models()"`
  (saved under the installed `openwakeword/resources/models/`)
- Or copy them from this project's `web/models/`.
