#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Find Drum Hits v01

But :
Trouver les vraies slices kick / hihat / snare dans un break.

Cette étape ne génère PAS de variations.
Elle sert à vérifier la matière première :
    exports/find_drum_hits_v01/<break>/
        kick/
        snare/
        hat/
        other/
        metadata_find_drum_hits_v01.json

Usage :
    python find_drum_hits_v01.py --source "Amen"
    python find_drum_hits_v01.py --source "London"
    python find_drum_hits_v01.py --source "Camo"

Plus de slices candidates :
    python find_drum_hits_v01.py --source "Amen" --max-hits 24

Découpe plus sensible :
    python find_drum_hits_v01.py --source "Amen" --delta 0.05

Découpe moins sensible :
    python find_drum_hits_v01.py --source "Amen" --delta 0.12
"""

from pathlib import Path
import argparse
import json
import sys
import warnings

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:
    print("librosa manquant : pip install librosa soundfile numpy")
    sys.exit(1)


BREAKS_DIR = Path("breaks")
OUT_DIR = Path("exports/find_drum_hits_v01")
SR = 44100
AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".mp3"}


def find_source(name):
    files = sorted(
        p for p in BREAKS_DIR.rglob("*")
        if p.suffix.lower() in AUDIO_EXTS and not p.name.endswith(".asd")
    )
    matches = [p for p in files if name.lower() in p.name.lower()]
    if not matches:
        print(f"Aucun break trouvé pour : {name}")
        sys.exit(1)
    return matches[0]


def normalize(y, peak=0.95):
    m = np.max(np.abs(y)) if len(y) else 0
    return y if m <= 1e-9 else y / m * peak


def fade(y, ms=2):
    if len(y) < 16:
        return y
    n = min(int(SR * ms / 1000), len(y) // 4)
    if n <= 1:
        return y
    y = y.copy()
    ramp = np.linspace(0, 1, n)
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def load_audio(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    return normalize(y), sr


def safe_centroid(y):
    if len(y) < 64:
        return 0.0
    n_fft = min(2048, max(64, 2 ** int(np.floor(np.log2(max(64, len(y)))))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=n_fft)))


def features(y):
    if len(y) < 64:
        return {
            "rms": 0.0,
            "peak": 0.0,
            "centroid": 0.0,
            "zcr": 0.0,
            "low": 0.0,
            "lowmid": 0.0,
            "mid": 0.0,
            "high": 0.0,
            "duration": len(y) / SR,
        }

    rms = float(np.sqrt(np.mean(y * y)))
    peak = float(np.max(np.abs(y)))
    centroid = safe_centroid(y)
    duration = len(y) / SR

    try:
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    except Exception:
        zcr = 0.0

    yn = normalize(y, peak=1.0)
    spec = np.abs(np.fft.rfft(yn))
    freqs = np.fft.rfftfreq(len(yn), 1.0 / SR)
    total = np.sum(spec) + 1e-9

    low = float(np.sum(spec[(freqs >= 20) & (freqs < 180)]) / total)
    lowmid = float(np.sum(spec[(freqs >= 180) & (freqs < 700)]) / total)
    mid = float(np.sum(spec[(freqs >= 700) & (freqs < 3000)]) / total)
    high = float(np.sum(spec[(freqs >= 3000) & (freqs < 12000)]) / total)

    return {
        "rms": rms,
        "peak": peak,
        "centroid": centroid,
        "zcr": zcr,
        "low": low,
        "lowmid": lowmid,
        "mid": mid,
        "high": high,
        "duration": duration,
    }


def detect_slices(y, delta=0.08, min_ms=35, max_ms=420):
    onsets = librosa.onset.onset_detect(
        y=y,
        sr=SR,
        units="samples",
        backtrack=True,
        delta=delta,
        wait=1,
        pre_max=3,
        post_max=3,
        pre_avg=8,
        post_avg=8,
    )

    points = sorted(set([0] + [int(x) for x in onsets] + [len(y)]))

    min_len = int(SR * min_ms / 1000)
    max_len = int(SR * max_ms / 1000)

    slices = []

    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]

        if end - start < min_len:
            continue

        end = min(end, start + max_len)
        audio = fade(y[start:end].copy(), ms=2)
        f = features(audio)

        slices.append({
            "index": len(slices),
            "start": int(start),
            "end": int(end),
            "start_ms": float(start / SR * 1000),
            "end_ms": float(end / SR * 1000),
            "duration": float((end - start) / SR),
            "features": f,
            "audio": audio,
        })

    return slices


def score_kick(s):
    f = s["features"]
    return (
        f["low"] * 5.0
        + f["lowmid"] * 2.0
        + f["rms"] * 5.0
        + f["peak"] * 2.2
        + max(0.0, 2600.0 - f["centroid"]) / 2600.0
        - f["high"] * 1.2
        - f["zcr"] * 1.5
    )


def score_snare(s):
    f = s["features"]
    centroid_ok = 1.0 if 800 <= f["centroid"] <= 6500 else 0.0
    return (
        f["mid"] * 3.0
        + f["lowmid"] * 1.4
        + f["high"] * 0.9
        + f["rms"] * 4.0
        + f["peak"] * 1.8
        + centroid_ok
        - f["low"] * 0.6
    )


def score_hat(s):
    f = s["features"]
    return (
        f["high"] * 5.0
        + f["centroid"] / 8500.0
        + f["zcr"] * 1.5
        + f["rms"] * 1.0
        - f["low"] * 2.0
        - f["lowmid"] * 0.7
    )


def assign_labels(slices, max_hits=16):
    """
    Classement relatif, avec exclusion :
    - les meilleurs kicks ne peuvent pas devenir hats
    - les meilleurs snares ne remplacent pas les kicks
    """
    for s in slices:
        s["scores"] = {
            "kick": score_kick(s),
            "snare": score_snare(s),
            "hat": score_hat(s),
        }
        s["label"] = "other"

    kick_rank = sorted(slices, key=lambda s: s["scores"]["kick"], reverse=True)
    snare_rank = sorted(slices, key=lambda s: s["scores"]["snare"], reverse=True)
    hat_rank = sorted(slices, key=lambda s: s["scores"]["hat"], reverse=True)

    n = len(slices)
    kick_n = min(max_hits, max(2, n // 5))
    snare_n = min(max_hits, max(2, n // 4))
    hat_n = min(max_hits * 2, max(4, n // 2))

    for s in kick_rank[:kick_n]:
        s["label"] = "kick"

    for s in snare_rank:
        if len([x for x in slices if x["label"] == "snare"]) >= snare_n:
            break
        if s["label"] != "kick":
            s["label"] = "snare"

    for s in hat_rank:
        if len([x for x in slices if x["label"] == "hat"]) >= hat_n:
            break
        if s["label"] not in ["kick", "snare"]:
            s["label"] = "hat"

    return slices


def export_results(slices, outdir, source):
    outdir.mkdir(parents=True, exist_ok=True)

    for label in ["kick", "snare", "hat", "other"]:
        (outdir / label).mkdir(exist_ok=True)

    exported = []

    for s in slices:
        label = s["label"]
        f = s["features"]
        sc = s["scores"]

        name = (
            f"{label}_{s['index']:03d}_"
            f"k{sc['kick']:.2f}_s{sc['snare']:.2f}_h{sc['hat']:.2f}_"
            f"{int(s['start_ms'])}ms.wav"
        )
        name = name.replace("-", "m")

        wav = outdir / label / name
        sf.write(wav, normalize(s["audio"]), SR)

        exported.append({
            "index": int(s["index"]),
            "label": label,
            "wav": str(wav),
            "start_ms": float(s["start_ms"]),
            "end_ms": float(s["end_ms"]),
            "duration": float(s["duration"]),
            "scores": {
                "kick": float(sc["kick"]),
                "snare": float(sc["snare"]),
                "hat": float(sc["hat"]),
            },
            "features": {k: float(v) for k, v in f.items()},
        })

    metadata = {
        "version": "find_drum_hits_v01",
        "source": str(source),
        "slice_count": len(slices),
        "counts": {
            "kick": len([s for s in slices if s["label"] == "kick"]),
            "snare": len([s for s in slices if s["label"] == "snare"]),
            "hat": len([s for s in slices if s["label"] == "hat"]),
            "other": len([s for s in slices if s["label"] == "other"]),
        },
        "exports": exported,
    }

    (outdir / "metadata_find_drum_hits_v01.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Amen")
    parser.add_argument("--delta", type=float, default=0.08)
    parser.add_argument("--min-ms", type=float, default=35.0)
    parser.add_argument("--max-ms", type=float, default=420.0)
    parser.add_argument("--max-hits", type=int, default=16)
    args = parser.parse_args()

    source = find_source(args.source)
    print("Source :", source)

    y, sr = load_audio(source)
    slices = detect_slices(
        y,
        delta=args.delta,
        min_ms=args.min_ms,
        max_ms=args.max_ms,
    )

    if not slices:
        print("Aucune slice détectée.")
        sys.exit(1)

    slices = assign_labels(slices, max_hits=args.max_hits)

    safe = source.stem.replace(" ", "_").replace("'", "")
    outdir = OUT_DIR / safe
    metadata = export_results(slices, outdir, source)

    print("")
    print("Slices :", len(slices))
    print("Répartition :")
    for label, count in metadata["counts"].items():
        print(f"  {label:6}: {count}")

    print("")
    print("Dossier :", outdir)
    print("Écoute surtout :")
    print(" ", outdir / "kick")
    print(" ", outdir / "snare")
    print(" ", outdir / "hat")


if __name__ == "__main__":
    main()
