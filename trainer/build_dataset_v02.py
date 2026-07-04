#!/usr/bin/env python3
from pathlib import Path
import json
import sys
import numpy as np

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("Installe les dépendances dans le venv :")
    print("source .venv/bin/activate")
    print("pip install librosa soundfile numpy")
    sys.exit(1)

ROOT = Path(".")
BREAKS_DIR = ROOT / "breaks"
DATASET_DIR = ROOT / "dataset"
SLICES_DIR = DATASET_DIR / "slices"

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".mp3"}

def norm(y):
    m = np.max(np.abs(y)) if len(y) else 0
    return y / m * 0.95 if m > 0 else y

def classify_slice(y, sr):
    if len(y) < 64:
        return "ghost"

    rms = float(np.sqrt(np.mean(y * y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    dur = len(y) / sr

    if rms < 0.018:
        return "ghost"
    if centroid < 1350 and rms > 0.035:
        return "kick"
    if centroid > 4200 and dur < 0.25:
        return "hat"
    if 1300 <= centroid <= 5200 and rms > 0.028:
        return "snare"
    if centroid > 3800:
        return "hat"
    return "perc"

def main():
    DATASET_DIR.mkdir(exist_ok=True)
    SLICES_DIR.mkdir(exist_ok=True)

    breaks = sorted([
        p for p in BREAKS_DIR.rglob("*")
        if p.suffix.lower() in AUDIO_EXTS and not p.name.endswith(".asd")
    ])

    print(f"{len(breaks)} breaks trouvés dans {BREAKS_DIR}")

    manifest = []
    slice_manifest = []

    for bi, path in enumerate(breaks):
        print(f"[{bi+1}/{len(breaks)}] {path.name}")

        y, sr = librosa.load(path, sr=44100, mono=True)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.asarray(tempo).reshape(-1)[0])

        onset_samples = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            units="samples",
            backtrack=True,
            delta=0.16
        )

        points = sorted(set([0] + [int(x) for x in onset_samples] + [len(y)]))

        break_id = f"break_{bi:04d}"
        out_break_dir = SLICES_DIR / break_id
        out_break_dir.mkdir(parents=True, exist_ok=True)

        local_count = 0

        for si in range(len(points) - 1):
            start = points[si]
            end = points[si + 1]

            if end - start < int(sr * 0.035):
                continue

            end = min(end, start + int(sr * 0.7))
            chunk = y[start:end].copy()

            fade = min(int(sr * 0.004), len(chunk) // 4)
            if fade > 1:
                ramp = np.linspace(0, 1, fade)
                chunk[:fade] *= ramp
                chunk[-fade:] *= ramp[::-1]

            label = classify_slice(chunk, sr)
            slice_name = f"{break_id}_slice_{local_count:04d}_{label}.wav"
            slice_path = out_break_dir / slice_name

            sf.write(slice_path, norm(chunk), sr)

            slice_manifest.append({
                "break_id": break_id,
                "source": str(path),
                "slice_file": str(slice_path),
                "label": label,
                "start_sample": start,
                "end_sample": end,
                "duration": len(chunk) / sr,
            })

            local_count += 1

        manifest.append({
            "id": break_id,
            "file": str(path),
            "estimated_bpm": tempo,
            "slice_count": local_count,
        })

    (DATASET_DIR / "breaks_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    (DATASET_DIR / "slices_manifest.json").write_text(
        json.dumps(slice_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("")
    print("Dataset créé.")
    print(f"Breaks : {len(manifest)}")
    print(f"Slices : {len(slice_manifest)}")
    print("Fichiers :")
    print("  dataset/breaks_manifest.json")
    print("  dataset/slices_manifest.json")
    print("  dataset/slices/")

if __name__ == "__main__":
    main()
