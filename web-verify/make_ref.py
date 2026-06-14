"""Decode the test clip to 16k mono WAV and dump openWakeWord per-frame scores as
ground truth for verifying the JS port. Also dumps melspec/embedding shapes."""
import json
import sys
import numpy as np
import av
import scipy.io.wavfile as wav
import openwakeword
from openwakeword.model import Model

AAC = sys.argv[1]
MODEL = sys.argv[2]
OUT_WAV = sys.argv[3]
OUT_JSON = sys.argv[4]

# decode -> 16k mono s16
container = av.open(AAC)
res = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
chunks = []
def collect(frames):
    if frames is None: return
    if not isinstance(frames, list): frames = [frames]
    for fr in frames: chunks.append(fr.to_ndarray().reshape(-1))
for frame in container.decode(audio=0):
    collect(res.resample(frame))
collect(res.resample(None))
pcm = np.concatenate(chunks).astype(np.int16)
wav.write(OUT_WAV, 16000, pcm)
print("wrote", OUT_WAV, "samples:", len(pcm))

openwakeword.utils.download_models()
oww = Model(wakeword_models=[MODEL], inference_framework="onnx")
name = list(oww.models.keys())[0]

scores = []
for i in range(0, len(pcm) - 1280, 1280):
    scores.append(float(oww.predict(pcm[i:i+1280])[name]))

json.dump({"model": name, "n_samples": int(len(pcm)), "scores": scores}, open(OUT_JSON, "w"), indent=1)
print("frames:", len(scores), "max:", round(max(scores), 5))
