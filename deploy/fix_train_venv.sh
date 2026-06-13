#!/bin/bash
source /opt/hotword/venvs/train/bin/activate
pip install -q "setuptools<81"
cd /opt/hotword/piper-sample-generator-oww
python -c "import sys; sys.path.insert(0,'.'); import generate_samples; print('GENERATE_SAMPLES_OK')"
