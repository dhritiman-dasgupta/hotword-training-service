import sys
import numpy as np
import scipy.io.wavfile as wav
from openwakeword.model import Model

wav_path, model_path = sys.argv[1], sys.argv[2]
sr, data = wav.read(wav_path)
assert sr == 16000, f"expected 16kHz, got {sr}"
if data.ndim > 1:
    data = data.mean(axis=1)
pcm = data.astype(np.int16)

oww = Model(wakeword_models=[model_path], inference_framework="onnx")
name = list(oww.models.keys())[0]

scores = []
for i in range(0, len(pcm) - 1280, 1280):
    s = oww.predict(pcm[i:i + 1280])[name]
    scores.append(s)

scores = np.array(scores) if scores else np.array([0.0])
dur = len(pcm) / 16000.0
peak_idx = int(scores.argmax())
print(f"model        : {name}")
print(f"duration     : {dur:.2f} s")
print(f"frames       : {len(scores)} (80ms each)")
print(f"MAX score    : {scores.max():.4f}  at t={peak_idx*0.08:.2f}s")
print(f"mean score   : {scores.mean():.4f}")
print(f"frames>0.5   : {int((scores>0.5).sum())}")
print(f"detected     : {'YES' if scores.max() >= 0.5 else 'NO'} (threshold 0.5)")
# show the top moments
top = scores.argsort()[-5:][::-1]
print("top frames   :", ", ".join(f"{t*0.08:.2f}s={scores[t]:.3f}" for t in sorted(top)))
