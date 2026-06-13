#!/bin/bash
echo "=== FULL custom_model.yml ==="
cat /opt/hotword/openWakeWord/examples/custom_model.yml
echo
echo "=== train.py imports (1-24) ==="
sed -n '1,24p' /opt/hotword/openWakeWord/openwakeword/train.py
echo
echo "=== generate_adversarial_texts location ==="
grep -rn "def generate_adversarial_texts" /opt/hotword/openWakeWord/ 2>/dev/null
echo
echo "=== openwakeword install extras (pyproject/setup) ==="
cat /opt/hotword/openWakeWord/setup.py 2>/dev/null
cat /opt/hotword/openWakeWord/pyproject.toml 2>/dev/null
cat /opt/hotword/openWakeWord/setup.cfg 2>/dev/null
echo
echo "=== clone dscripka piper-sample-generator fork ==="
[ -d /opt/hotword/piper-sample-generator-oww ] || git clone -q https://github.com/dscripka/piper-sample-generator /opt/hotword/piper-sample-generator-oww
ls /opt/hotword/piper-sample-generator-oww
echo "--- fork requirements ---"
cat /opt/hotword/piper-sample-generator-oww/requirements.txt 2>/dev/null
echo "--- fork has generate_samples.py? ---"
ls /opt/hotword/piper-sample-generator-oww/generate_samples.py 2>/dev/null && grep -n "def generate_samples\|def generate_adversarial_texts" /opt/hotword/piper-sample-generator-oww/*.py
