"""Score an audio file (any format incl. AAC/m4a/mp3/wav) against a wake-word ONNX model.
Usage: python score_file.py <audio_file> <model.onnx>
"""
import sys
import numpy as np
import av
import openwakeword
from openwakeword.model import Model


def decode_to_16k_mono(path):
    container = av.open(path)
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []

    def collect(frames):
        if frames is None:
            return
        if not isinstance(frames, list):
            frames = [frames]
        for fr in frames:
            chunks.append(fr.to_ndarray().reshape(-1))

    for frame in container.decode(audio=0):
        collect(resampler.resample(frame))
    collect(resampler.resample(None))  # flush
    if not chunks:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(chunks).astype(np.int16)


def main():
    audio_path, model_path = sys.argv[1], sys.argv[2]
    try:
        openwakeword.utils.download_models()
    except Exception as e:
        print("warn: base model download:", e)

    pcm = decode_to_16k_mono(audio_path)
    oww = Model(wakeword_models=[model_path], inference_framework="onnx")
    name = list(oww.models.keys())[0]

    scores = []
    for i in range(0, len(pcm) - 1280, 1280):
        scores.append(oww.predict(pcm[i:i + 1280])[name])
    scores = np.array(scores) if scores else np.array([0.0])

    dur = len(pcm) / 16000.0
    peak = int(scores.argmax())
    print("=" * 44)
    print(f"file        : {audio_path.split(chr(92))[-1]}")
    print(f"model       : {name}")
    print(f"duration    : {dur:.2f} s   ({len(scores)} x 80ms frames)")
    print(f"MAX CONF    : {scores.max():.4f}  at t={peak*0.08:.2f}s")
    print(f"mean conf   : {scores.mean():.4f}")
    print(f"frames>0.5  : {int((scores>0.5).sum())}")
    print(f"DETECTED    : {'YES' if scores.max() >= 0.5 else 'NO'}  (threshold 0.5)")
    top = sorted(scores.argsort()[-6:][::-1])
    print("timeline    :", ", ".join(f"{t*0.08:.1f}s={scores[t]:.3f}" for t in top))
    print("=" * 44)


if __name__ == "__main__":
    main()
