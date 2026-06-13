#!/usr/bin/env python3
"""
Local hotword detector — listens on the microphone and fires when your trained
wake word is spoken. Runs fully offline using the ONNX model from the service.

Setup (any machine with a mic + Python 3.8+):
    pip install -r requirements.txt
Run:
    python listen.py --model hey_kiki.onnx --threshold 0.5

The model is a standard openWakeWord model, so it also works with Home Assistant,
Rhasspy, and the openWakeWord library directly.
"""
import argparse
import time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the trained .onnx model")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="detection score threshold (0-1)")
    ap.add_argument("--refractory", type=float, default=2.0,
                    help="seconds to wait after a detection before firing again")
    ap.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    ap.add_argument("--device", default=None, help="input device index or name")
    args = ap.parse_args()

    import sounddevice as sd
    if args.list_devices:
        print(sd.query_devices())
        return

    from openwakeword.model import Model
    import openwakeword
    try:
        openwakeword.utils.download_models()  # base melspectrogram + embedding models
    except Exception:
        pass

    oww = Model(wakeword_models=[args.model], inference_framework="onnx")
    name = list(oww.models.keys())[0]
    print(f"Loaded model '{name}'. Listening... (Ctrl+C to stop)")

    SR = 16000
    BLOCK = 1280  # 80 ms frames, as openWakeWord expects
    last_fire = 0.0

    def callback(indata, frames, time_info, status):
        nonlocal last_fire
        pcm = (indata[:, 0] * 32767).astype(np.int16)
        scores = oww.predict(pcm)
        score = scores.get(name, 0.0)
        if score >= args.threshold and (time.time() - last_fire) > args.refractory:
            last_fire = time.time()
            print(f"\n🔔 DETECTED '{name}'  (score={score:.3f})  {time.strftime('%H:%M:%S')}")
        else:
            bar = "#" * int(score * 30)
            print(f"\r{score:5.3f} |{bar:<30}|", end="", flush=True)

    with sd.InputStream(channels=1, samplerate=SR, blocksize=BLOCK,
                        dtype="float32", device=args.device, callback=callback):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
