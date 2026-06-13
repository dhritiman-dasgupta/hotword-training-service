#!/bin/bash
NB=/opt/hotword/openWakeWord/notebooks/automatic_model_training.ipynb
echo "=== notebook lines mentioning rir / mit / background / scaper / datasets / wget / zip ==="
grep -noE "[^\"]*(mit_rir|MIT|rir|background_clips|scaper|load_dataset|datasets|wget|unzip|\.zip|fma|fsd50k|genspeech)[^\"]*" "$NB" | sed 's/\\n//g' | head -80
echo
echo "=== generate_samples.py signature (1-60) ==="
sed -n '1,60p' /opt/hotword/piper-sample-generator-oww/generate_samples.py
echo
echo "=== train.py onnx export lines (880-960) ==="
sed -n '880,960p' /opt/hotword/openWakeWord/openwakeword/train.py
