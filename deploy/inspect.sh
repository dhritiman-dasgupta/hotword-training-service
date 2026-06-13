#!/bin/bash
echo "=== oww requirements ==="
cat /opt/hotword/openWakeWord/requirements.txt
echo
echo "=== oww notebooks ls ==="
ls /opt/hotword/openWakeWord/notebooks
echo
echo "=== train.py top-level defs ==="
grep -nE "^def |^class |argparse|add_argument|training_config" /opt/hotword/openWakeWord/openwakeword/train.py | head -80
echo
echo "=== download URLs referenced in notebooks ==="
grep -rhoE "https?://[A-Za-z0-9./_~?=&%-]+" /opt/hotword/openWakeWord/notebooks/ 2>/dev/null | sort -u | head -60
echo
echo "=== example training config yaml (if any) ==="
find /opt/hotword/openWakeWord -name "*.yml" -o -name "*.yaml" 2>/dev/null | head
echo
echo "=== how notebook invokes training ==="
grep -rhnE "train.py|train_model|download_|\.npy|features|piper_sample_generator|augment" /opt/hotword/openWakeWord/notebooks/*.ipynb 2>/dev/null | head -60
