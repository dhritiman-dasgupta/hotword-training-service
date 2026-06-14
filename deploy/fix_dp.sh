#!/bin/bash
source /opt/hotword/venvs/train/bin/activate
pip install -q "deep-phonemizer==0.0.19"
python -c "from dp.phonemizer import Phonemizer; print('dp import OK')"
python - <<'PY'
from openwakeword.data import generate_adversarial_texts
t = generate_adversarial_texts("hey nikhilesh", N=5, include_partial_phrase=1.0, include_input_words=0.2)
print("adversarial gen OK; samples:", t[:3])
PY
echo "DP_FIX_DONE"
