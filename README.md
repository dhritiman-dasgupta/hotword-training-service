# Hotword Training Service

A self-hosted **wake-word ("hotword") training service**. You call an HTTP API with a
wake word (e.g. `"hey kiki"`); the service synthesizes thousands of training samples
across many voices and accents using **Piper** + **F5-TTS** (including voice cloning),
augments them with room reverb and background noise, trains a compact **openWakeWord**
detector on a GPU, and returns a **portable ONNX model** that runs the wake-word
detector fully offline on any mic.

> **In one line:** `POST /train {"wake_word": "..."}` → wait → `GET /jobs/<id>/model` → `your_word.onnx`

**Full API reference (generate + inference):** see [`API_DOCS.md`](API_DOCS.md).

A pre-trained demo model (`models/hey_kiki.onnx`) is included.

---

## How it works

```
  wake word
     |
     +--> [Piper]      904 LibriTTS speakers, varied speed/pitch   --+
     +--> [F5-TTS]     human-like + voice-cloned samples (+ your ref) |
                                                                      v
                       [Augment]   MIT room impulse responses + FMA background noise
                                                                      v
                       [Train]     openWakeWord DNN on Google's frozen speech embeddings
                                                                      v
                       your_word.onnx   -->   runs locally via onnxruntime
```

- **Piper** (`piper-sample-generator`, LibriTTS-R generator) — bulk positive samples with
  wide speaker/accent/speed diversity, fast on GPU.
- **F5-TTS** — very human-like and **voice-cloned** samples. Upload a reference clip via
  `/train-clone` to clone a specific voice.
- **openWakeWord** — trains a small classifier on top of Google's frozen speech-embedding
  model. Output is a portable `.onnx` that also works with **Home Assistant**, **Rhasspy**,
  and the openWakeWord library directly.

> TFLite export is intentionally skipped (it needs an ancient tensorflow/onnx_tf stack).
> ONNX covers local/desktop/server inference. A microcontroller (ESP32) build would be a
> separate microWakeWord project.

---

## Using the API

The service is an HTTP API. Send `X-API-Key: <YOUR_API_KEY>` on every request (the key is
generated on the instance at `/opt/hotword/state/api_key` and mirrored to the
git-ignored `deploy/api_key.txt` — never commit it). Base URL is the instance's IP on
port `8000`. Interactive docs at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | status, GPU, whether training data is ready |
| `POST` | `/train` | `{ "wake_word": "...", "n_samples": 5000, "steps": 20000, "use_f5": true }` -> `{job_id}` |
| `POST` | `/train-clone` | multipart: `wake_word`, `ref_audio` (wav), `ref_text` — clone a specific voice |
| `GET`  | `/jobs` | list jobs |
| `GET`  | `/jobs/{id}` | job status + last log lines |
| `GET`  | `/jobs/{id}/log` | full training log |
| `GET`  | `/jobs/{id}/model` | download the trained `.onnx` |

### curl
```bash
KEY=<YOUR_API_KEY>
BASE=http://<INSTANCE_IP>:8000

# 1) submit
curl -s -X POST $BASE/train -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"wake_word":"hey kiki"}'
# -> {"job_id":"20260613-...","state":"queued",...}

# 2) poll until state == "done"
curl -s $BASE/jobs/<job_id> -H "X-API-Key: $KEY"

# 3) download the ONNX
curl -s $BASE/jobs/<job_id>/model -H "X-API-Key: $KEY" -o hey_kiki.onnx
```

### Python (submit → wait → save)
```python
import requests, time
BASE = "http://<INSTANCE_IP>:8000"
H = {"X-API-Key": "<YOUR_API_KEY>"}

jid = requests.post(f"{BASE}/train", headers=H, json={"wake_word": "hey kiki"}).json()["job_id"]
while requests.get(f"{BASE}/jobs/{jid}", headers=H).json()["state"] == "running":
    time.sleep(15)
open("hey_kiki.onnx", "wb").write(requests.get(f"{BASE}/jobs/{jid}/model", headers=H).content)
```

**Quality knobs** (POST body): `n_samples` (positives, default 5000), `steps` (default
20000), `f5_samples`, `use_f5`. For a production-quality model use `n_samples: 20000+` and
`steps: 50000` (takes ~1–2 h on the T4 GPU).

---

## Using the trained model

The `.onnx` is a standard openWakeWord model. Feed it 16 kHz mono mic audio in 80 ms
(1280-sample) chunks; it returns a score 0–1; trigger when it crosses ~0.5.

### Ready-made listener (any machine with a mic + Python 3.8+)
```bash
cd client
pip install -r requirements.txt
python listen.py --model ../models/hey_kiki.onnx --threshold 0.5
```
Prints a live confidence bar and `DETECTED` when the wake word is spoken.

### In your own code
```python
from openwakeword.model import Model
import sounddevice as sd, numpy as np

oww = Model(wakeword_models=["hey_kiki.onnx"], inference_framework="onnx")

def cb(indata, frames, t, status):
    pcm = (indata[:, 0] * 32767).astype(np.int16)
    if oww.predict(pcm)["hey_kiki"] > 0.5:
        print("wake word detected!")        # <-- your action here

with sd.InputStream(channels=1, samplerate=16000, blocksize=1280,
                    dtype="float32", callback=cb):
    input("listening... press Enter to quit\n")
```
`pip install openwakeword onnxruntime sounddevice numpy`

---

## Repository layout

```
hotword-service/
├─ server/                FastAPI service that runs ON the GPU instance
│  ├─ app.py              API: /train, /train-clone, /jobs, /jobs/{id}/model
│  ├─ pipeline.py         orchestrates Piper -> F5 -> augment -> train -> ONNX
│  ├─ f5_generate.py      F5-TTS human/cloned-voice sample generation
│  └─ watchdog.py         idle auto-shutdown
├─ client/
│  ├─ listen.py           local offline mic detector
│  └─ requirements.txt
├─ deploy/                provisioning + setup scripts (see "Self-host" below)
│  ├─ setup_stack.sh      installs Piper + F5-TTS + openWakeWord into 3 venvs
│  ├─ dl_features.sh      downloads the 17 GB openWakeWord negatives + validation
│  ├─ dl_data.py          downloads MIT RIRs + FMA background audio
│  ├─ hotword-api.service / hotword-watchdog.service   systemd units
│  └─ *.json              EC2 block-device / tags / spot market config
├─ models/
│  └─ hey_kiki.onnx       included demo model
├─ hw.ps1                 Windows control script (start/stop/train/download)
└─ README.md
```

Secrets (`deploy/hotword-key.pem`, `deploy/api_key.txt`, instance/EIP IDs) are
**git-ignored** — see `.gitignore`.

---

## Self-host / reproduce

The service was provisioned on AWS as a g4dn.2xlarge **persistent Spot** instance from the
*Deep Learning Base GPU AMI (Ubuntu 22.04)*. To rebuild on a fresh GPU box:

```bash
# on the instance (Ubuntu 22.04 + NVIDIA driver)
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1 git python3.10-venv
git clone https://github.com/rhasspy/piper-sample-generator           /opt/hotword/...
git clone https://github.com/dscripka/openWakeWord                     /opt/hotword/openWakeWord
git clone https://github.com/dscripka/piper-sample-generator           /opt/hotword/piper-sample-generator-oww
bash deploy/setup_stack.sh        # creates venvs/{api,train,tts}, installs everything
bash deploy/dl_features.sh        # 17 GB negatives + validation features
python deploy/dl_data.py          # MIT RIRs + FMA background (run in the train venv)
sudo cp deploy/hotword-*.service /etc/systemd/system/ && sudo systemctl enable --now hotword-api hotword-watchdog
```

Pinned versions that matter in the train venv: `numpy<2`, `pyarrow==12.0.1`,
`setuptools<81` (webrtcvad needs `pkg_resources`). An upstream openWakeWord bug
(`train.py` argparse `default="False"`, a truthy string, forcing TFLite conversion) is
patched to `default=False`.

---

## Operations (current deployment)

| Resource | Value |
|---|---|
| EC2 instance | g4dn.2xlarge persistent Spot (T4 GPU, 8 vCPU, 32 GB), ap-south-1 |
| AMI | Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) |
| API | `http://<INSTANCE_IP>:8000` (Elastic IP, stable across stop/start) |
| Auto-stop | stops after **5 min idle** (won't stop while a job runs); spot interruption also stops, not terminates |
| Cost | ~$0.24/hr spot only while running; ~$10/mo EBS + ~$3.6/mo Elastic IP while stopped |

- **Restart after auto-stop:** `aws ec2 start-instances --instance-ids <ID> --region ap-south-1`
  (systemd brings the API + watchdog back up automatically). The API is unreachable while stopped.
- **Logs:** `sudo journalctl -u hotword-api -f` / `-u hotword-watchdog -f`.
- **Change idle timeout:** edit `IDLE_TIMEOUT_SECONDS` in the watchdog unit, then
  `sudo systemctl daemon-reload && sudo systemctl restart hotword-watchdog`.

## Tips for a good wake word

- 3–4 syllables, distinct phonetics ("hey kiki", "hey jarvis") trigger less falsely than
  single common words ("computer").
- Use `/train-clone` with a 5–15 s clean reference clip (wav + transcript) to bias the
  model toward a specific voice.

## Credits

Built on [openWakeWord](https://github.com/dscripka/openWakeWord),
[piper-sample-generator](https://github.com/rhasspy/piper-sample-generator), and
[F5-TTS](https://github.com/SWivid/F5-TTS).
