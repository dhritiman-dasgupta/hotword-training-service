#!/bin/bash
# Fix upstream bug: default="False" (truthy string) -> default=False (real bool)
sed -i 's/default="False"/default=False/g' /opt/hotword/openWakeWord/openwakeword/train.py
echo "patched defaults:"
grep -n 'default=False' /opt/hotword/openWakeWord/openwakeword/train.py | head

echo "=== verify the already-trained onnx loads in the openWakeWord runtime ==="
ONNX=/opt/hotword/jobs/20260613-080010-d87115/work/hey_kiki.onnx
ls -la "$ONNX" 2>/dev/null || echo "onnx missing"
/opt/hotword/venvs/train/bin/python - "$ONNX" <<'PY'
import sys
from openwakeword.model import Model
m = Model(wakeword_models=[sys.argv[1]], inference_framework="onnx")
print("LOADED OK, models:", list(m.models.keys()))
import numpy as np
# feed 1 second of silence to confirm predict() runs
for _ in range(20):
    s = m.predict(np.zeros(1280, dtype=np.int16))
print("predict() runs; keys:", list(s.keys()))
PY
