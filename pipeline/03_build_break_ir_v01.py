#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_build_break_ir_v01.py

Break IR v01 :
Convertit un break WAV en représentation intermédiaire propre.

Entrée :
    breaks/Amen....wav

Sorties :
    dataset/break_ir/<break>_break_ir_v01.json
    dataset/break_ir/<break>/hits/kick/*.wav
    dataset/break_ir/<break>/hits/snare/*.wav
    dataset/break_ir/<break>/hits/hat/*.wav
    dataset/break_ir/<break>/hits/other/*.wav

But :
Créer une base stable pour la suite :
- extraction de hits
- classification kick/snare/hat/other
- quantification sur une grille 16
- objet JSON lisible par renderer / IA / éditeur

Usage :
    python pipeline/03_build_break_ir_v01.py --source "Amen"
    python pipeline/03_build_break_ir_v01.py --source "Camo"
    python pipeline/03_build_break_ir_v01.py --source "London"

Options :
    python pipeline/03_build_break_ir_v01.py --source "Amen" --steps 16
    python pipeline/03_build_break_ir_v01.py --source "Amen" --delta 0.06
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
    print("librosa manquant : python -m pip install librosa soundfile numpy")
    sys.exit(1)


BREAKS_DIR = Path("breaks")
OUT_DIR = Path("dataset/break_ir")
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


def safe_name(path):
    return path.stem.replace(" ", "_").replace("'", "").replace("/", "_")


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


def detect_hits(y, delta=0.08, min_ms=35, max_ms=420):
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

    hits = []

    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]

        if end - start < min_len:
            continue

        end = min(end, start + max_len)
        audio = fade(y[start:end].copy(), ms=2)
        f = features(audio)

        hits.append({
            "id": f"hit_{len(hits):03d}",
            "index": len(hits),
            "start_sample": int(start),
            "end_sample": int(end),
            "start_ms": float(start / SR * 1000),
            "end_ms": float(end / SR * 1000),
            "duration": float((end - start) / SR),
            "features": f,
            "instrument": "other",
            "audio": audio,
        })

    return hits


def score_kick(hit):
    f = hit["features"]
    return (
        f["low"] * 5.0
        + f["lowmid"] * 2.0
        + f["rms"] * 5.0
        + f["peak"] * 2.2
        + max(0.0, 2600.0 - f["centroid"]) / 2600.0
        - f["high"] * 1.2
        - f["zcr"] * 1.5
    )


def score_snare(hit):
    f = hit["features"]
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


def score_hat(hit):
    f = hit["features"]
    return (
        f["high"] * 5.0
        + f["centroid"] / 8500.0
        + f["zcr"] * 1.5
        + f["rms"] * 1.0
        - f["low"] * 2.0
        - f["lowmid"] * 0.7
    )


def classify_hits(hits, max_hits=16):
    for hit in hits:
        hit["scores"] = {
            "kick": score_kick(hit),
            "snare": score_snare(hit),
            "hat": score_hat(hit),
        }
        hit["instrument"] = "other"

    kick_rank = sorted(hits, key=lambda h: h["scores"]["kick"], reverse=True)
    snare_rank = sorted(hits, key=lambda h: h["scores"]["snare"], reverse=True)
    hat_rank = sorted(hits, key=lambda h: h["scores"]["hat"], reverse=True)

    n = len(hits)
    kick_n = min(max_hits, max(2, n // 5))
    snare_n = min(max_hits, max(2, n // 4))
    hat_n = min(max_hits * 2, max(4, n // 2))

    for h in kick_rank[:kick_n]:
        h["instrument"] = "kick"

    for h in snare_rank:
        if len([x for x in hits if x["instrument"] == "snare"]) >= snare_n:
            break
        if h["instrument"] != "kick":
            h["instrument"] = "snare"

    for h in hat_rank:
        if len([x for x in hits if x["instrument"] == "hat"]) >= hat_n:
            break
        if h["instrument"] not in ["kick", "snare"]:
            h["instrument"] = "hat"

    return hits


def quantize_hits(hits, total_ms, steps=16):
    if total_ms <= 0:
        total_ms = 1.0

    events = []

    for hit in hits:
        pos = hit["start_ms"] / total_ms
        step = int(round(pos * (steps - 1)))
        step = max(0, min(steps - 1, step))

        velocity = float(min(1.0, max(0.05, hit["features"]["rms"] * 8.0)))

        events.append({
            "step": int(step),
            "instrument": hit["instrument"],
            "hit_id": hit["id"],
            "velocity": velocity,
            "length": float(hit["duration"]),
            "start_ms": float(hit["start_ms"]),
        })

    events.sort(key=lambda e: (e["step"], e["start_ms"]))
    return events


def export_hits(hits, outdir):
    hit_dir = outdir / "hits"
    for label in ["kick", "snare", "hat", "other"]:
        (hit_dir / label).mkdir(parents=True, exist_ok=True)

    exported_hits = []

    for hit in hits:
        label = hit["instrument"]
        wav_name = f"{hit['id']}_{label}_{int(hit['start_ms'])}ms.wav"
        wav_path = hit_dir / label / wav_name
        sf.write(wav_path, normalize(hit["audio"]), SR)

        exported = {
            "id": hit["id"],
            "index": int(hit["index"]),
            "instrument": label,
            "audio_path": str(wav_path),
            "start_sample": int(hit["start_sample"]),
            "end_sample": int(hit["end_sample"]),
            "start_ms": float(hit["start_ms"]),
            "end_ms": float(hit["end_ms"]),
            "duration": float(hit["duration"]),
            "features": {k: float(v) for k, v in hit["features"].items()},
            "scores": {k: float(v) for k, v in hit["scores"].items()},
        }
        exported_hits.append(exported)

    return exported_hits


def render_reconstruction(hits, outdir):
    chunks = [h["audio"] for h in hits]
    if not chunks:
        return None

    audio = normalize(np.concatenate(chunks))
    wav = outdir / "reconstruction_from_hits.wav"
    sf.write(wav, audio, SR)
    return str(wav)


def build_ir(source, y, hits, exported_hits, events, reconstruction, steps):
    counts = {
        "kick": len([h for h in hits if h["instrument"] == "kick"]),
        "snare": len([h for h in hits if h["instrument"] == "snare"]),
        "hat": len([h for h in hits if h["instrument"] == "hat"]),
        "other": len([h for h in hits if h["instrument"] == "other"]),
    }

    return {
        "version": "break_ir_v01",
        "source": str(source),
        "sample_rate": SR,
        "duration": float(len(y) / SR),
        "steps_per_bar": int(steps),
        "counts": counts,
        "reconstruction": reconstruction,
        "hits": exported_hits,
        "pattern": {
            "steps": int(steps),
            "events": events,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Amen")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--delta", type=float, default=0.08)
    parser.add_argument("--min-ms", type=float, default=35.0)
    parser.add_argument("--max-ms", type=float, default=420.0)
    parser.add_argument("--max-hits", type=int, default=16)
    args = parser.parse_args()

    source = find_source(args.source)
    print("Source :", source)

    y, sr = load_audio(source)
    hits = detect_hits(
        y,
        delta=args.delta,
        min_ms=args.min_ms,
        max_ms=args.max_ms,
    )

    if not hits:
        print("Aucun hit détecté.")
        sys.exit(1)

    hits = classify_hits(hits, max_hits=args.max_hits)

    safe = safe_name(source)
    outdir = OUT_DIR / safe
    outdir.mkdir(parents=True, exist_ok=True)

    exported_hits = export_hits(hits, outdir)
    events = quantize_hits(hits, total_ms=len(y) / SR * 1000, steps=args.steps)
    reconstruction = render_reconstruction(hits, outdir)

    ir = build_ir(
        source=source,
        y=y,
        hits=hits,
        exported_hits=exported_hits,
        events=events,
        reconstruction=reconstruction,
        steps=args.steps,
    )

    ir_path = OUT_DIR / f"{safe}_break_ir_v01.json"
    ir_path.write_text(json.dumps(ir, indent=2), encoding="utf-8")

    print("")
    print("Break IR créé :", ir_path)
    print("Dossier hits  :", outdir / "hits")
    print("Reconstruction:", reconstruction)
    print("")
    print("Répartition :")
    for label, count in ir["counts"].items():
        print(f"  {label:6}: {count}")


if __name__ == "__main__":
    main()
