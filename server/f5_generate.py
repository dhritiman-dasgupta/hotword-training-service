#!/usr/bin/env python3
"""
Generate human-like / voice-cloned positive samples of a wake word using F5-TTS.

Runs inside the F5-TTS venv (/opt/hotword/venvs/tts). Output is 16 kHz mono 16-bit
WAV, written into the target directory so openWakeWord's augment/train steps pick
them up alongside the Piper-generated clips.

This is best-effort: if F5-TTS fails to load or a reference is bad, we generate as
many clips as we can and exit 0 so the overall pipeline still trains on Piper data.
"""
import argparse
import os
import sys
import glob
import json
import traceback

import numpy as np
import soundfile as sf


def log(msg):
    print(f"[f5] {msg}", flush=True)


def resample_to_16k_mono(wav, sr):
    """Resample a float waveform to 16 kHz mono using torchaudio."""
    import torch
    import torchaudio
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    t = torch.from_numpy(wav.astype(np.float32))
    if sr != 16000:
        t = torchaudio.functional.resample(t, sr, 16000)
    # peak normalize a touch to avoid clipping, then to int16
    peak = float(t.abs().max()) or 1.0
    t = t / peak * 0.95
    return (t.numpy() * 32767.0).astype(np.int16)


def _f5_root():
    import f5_tts
    f = getattr(f5_tts, "__file__", None)
    if f:
        return os.path.dirname(f)
    p = list(getattr(f5_tts, "__path__", []))
    return p[0] if p else None


def discover_bundled_refs():
    """Find reference voices bundled with the f5_tts package.

    basic_ref_en.wav has a known transcript (reliable). The multi/* voices have
    no transcript, so we pass ref_text="" -> F5 auto-transcribes if ASR is
    available, otherwise those calls fail gracefully and we still have basic.
    """
    refs = []
    try:
        root = _f5_root()
        if not root:
            return refs
        ex = os.path.join(root, "infer", "examples")
        # Only WAV refs with a known transcript are reliable: the bundled .flac
        # voices need a newer ffmpeg than the base image ships (torchcodec can't
        # decode them), and empty ref_text would force ASR. So we use the WAV one
        # here; broader cloned-voice diversity comes via the /train-clone upload.
        basic = os.path.join(ex, "basic", "basic_ref_en.wav")
        if os.path.exists(basic):
            refs.append((basic, "Some call me nature, others call me mother nature."))
    except Exception as e:
        log(f"could not discover bundled refs: {e}")
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="wake word / phrase to synthesize")
    ap.add_argument("--out-dir", required=True, help="directory to write 16k wavs into")
    ap.add_argument("--n", type=int, default=120, help="approx total clips to generate")
    ap.add_argument("--ref-audio", default="", help="optional user reference wav to clone")
    ap.add_argument("--ref-text", default="", help="transcript of the user reference (optional)")
    ap.add_argument("--prefix", default="f5", help="output filename prefix")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build the reference voice list
    voices = []
    if args.ref_audio and os.path.exists(args.ref_audio):
        voices.append((args.ref_audio, args.ref_text))  # user voice first
    voices.extend(discover_bundled_refs())
    if not voices:
        log("no reference voices available; skipping F5 generation")
        return 0
    log(f"{len(voices)} reference voice(s) available")

    try:
        from f5_tts.api import F5TTS
    except Exception as e:
        log(f"f5_tts import failed ({e}); skipping F5 generation")
        return 0

    try:
        model = F5TTS()  # default model + vocoder, GPU if available
    except Exception as e:
        log(f"F5TTS init failed ({e}); skipping")
        traceback.print_exc()
        return 0

    speeds = [0.85, 1.0, 1.15, 1.3]
    per_voice = max(1, args.n // len(voices))
    made = 0
    for vi, (ref_wav, ref_text) in enumerate(voices):
        for k in range(per_voice):
            speed = speeds[k % len(speeds)]
            out_path = os.path.join(args.out_dir, f"{args.prefix}_{vi}_{k}.wav")
            try:
                wav, sr, _ = model.infer(
                    ref_file=ref_wav,
                    ref_text=ref_text,
                    gen_text=args.text,
                    speed=speed,
                    remove_silence=False,
                )
                pcm = resample_to_16k_mono(np.asarray(wav), int(sr))
                sf.write(out_path, pcm, 16000, subtype="PCM_16")
                made += 1
            except Exception as e:
                log(f"infer failed (voice {vi}, k {k}): {e}")
                if made == 0 and k == 0 and vi == 0:
                    traceback.print_exc()
    log(f"generated {made} F5 clips into {args.out_dir}")
    print(json.dumps({"f5_clips": made}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
