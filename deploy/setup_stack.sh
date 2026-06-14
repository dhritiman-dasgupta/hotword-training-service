#!/bin/bash
set -e
cd /opt/hotword
PSG=/opt/hotword/piper-sample-generator-oww

echo "[setup] apt extras $(date)"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q espeak-ng libspeexdsp-dev >/dev/null

echo "[setup] ===== TRAIN/OWW venv ===== $(date)"
source venvs/train/bin/activate
pip install -q --upgrade pip wheel setuptools
echo "[setup] torch cu121 (train)"
pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "[setup] openwakeword + training deps"
pip install -q openwakeword
pip install -q "speechbrain==0.5.16" torchinfo torchmetrics audiomentations torch-audiomentations mutagen acoustics pronouncing "datasets==2.14.6" "deep-phonemizer==0.0.19" onnx pyyaml tqdm scipy scikit-learn
# deep-phonemizer (module `dp`) is required by generate_adversarial_texts for words not in
# the CMU pronunciation dictionary (e.g. uncommon names). Without it generation crashes.
echo "[setup] piper fork deps"
pip install -q espeak-phonemizer webrtcvad
echo "[setup] downloading oww pretrained feature models"
python -c "from openwakeword.utils import download_models; download_models()"
echo "[setup] sanity import oww.data"
python -c "import openwakeword.data; print('oww.data OK')"
echo "[setup] sanity import piper fork generate_samples"
( cd $PSG && python -c "import sys; sys.path.insert(0,'.'); import generate_samples; print('generate_samples OK')" ) || echo "WARN: generate_samples import failed"
deactivate

echo "[setup] piper generator model $(date)"
mkdir -p $PSG/models
[ -f $PSG/models/en-us-libritts-high.pt ] || wget -q -O $PSG/models/en-us-libritts-high.pt "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt"
ls -lh $PSG/models

echo "[setup] ===== TTS venv (F5-TTS) ===== $(date)"
source venvs/tts/bin/activate
pip install -q --upgrade pip wheel setuptools
pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q f5-tts soundfile
python -c "import f5_tts; print('f5_tts OK')" || echo "WARN: f5_tts import failed"
deactivate

echo "[setup] ===== API venv ===== $(date)"
source venvs/api/bin/activate
pip install -q --upgrade pip
pip install -q fastapi "uvicorn[standard]" python-multipart pyyaml boto3 soundfile requests
deactivate

echo "SETUP_STACK_DONE $(date)"
