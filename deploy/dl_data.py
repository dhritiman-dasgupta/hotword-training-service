"""Download MIT room impulse responses + FMA background music for openWakeWord augmentation.
Run inside the train venv (needs datasets, scipy, numpy)."""
import os
import numpy as np
import scipy.io.wavfile
from tqdm import tqdm
import datasets

DATA = "/opt/hotword/data"
rir_dir = os.path.join(DATA, "mit_rirs")
fma_dir = os.path.join(DATA, "fma")
os.makedirs(rir_dir, exist_ok=True)
os.makedirs(fma_dir, exist_ok=True)

# --- MIT environmental impulse responses ---
if len(os.listdir(rir_dir)) < 50:
    print("[data] downloading MIT RIRs ...", flush=True)
    rir = datasets.load_dataset("davidscripka/MIT_environmental_impulse_responses",
                                split="train", streaming=True)
    n = 0
    for row in tqdm(rir):
        name = row["audio"]["path"].split("/")[-1]
        scipy.io.wavfile.write(os.path.join(rir_dir, name), 16000,
                               (row["audio"]["array"] * 32767).astype(np.int16))
        n += 1
    print(f"[data] RIRs written: {n}", flush=True)
else:
    print("[data] RIRs already present, skipping", flush=True)

# --- FMA background music (~2 hours of 30s clips) ---
TARGET_CLIPS = 240
if len(os.listdir(fma_dir)) < TARGET_CLIPS * 0.9:
    print("[data] downloading FMA background ...", flush=True)
    fma = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
    fma = iter(fma.cast_column("audio", datasets.Audio(sampling_rate=16000)))
    for i in tqdm(range(TARGET_CLIPS)):
        try:
            row = next(fma)
        except StopIteration:
            break
        name = row["audio"]["path"].split("/")[-1]
        if not name.endswith(".wav"):
            name = name.rsplit(".", 1)[0] + ".wav"
        scipy.io.wavfile.write(os.path.join(fma_dir, name), 16000,
                               (row["audio"]["array"] * 32767).astype(np.int16))
    print(f"[data] FMA clips: {len(os.listdir(fma_dir))}", flush=True)
else:
    print("[data] FMA already present, skipping", flush=True)

print("DL_DATA_DONE", flush=True)
