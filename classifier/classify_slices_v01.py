#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Drum Classifier v01 - classify_slices_v01.py

But :
Analyser toutes les slices déjà créées dans dataset/slices_manifest.json,
les classifier, calculer leurs caractéristiques audio, et créer une
bibliothèque rangée de samples.

Entrée :
    dataset/slices_manifest.json
    dataset/slices/**/*.wav

Sorties :
    dataset/drum_library/drum_library_v01.json
    dataset/drum_library/report_v01.txt
    dataset/drum_library/samples/
        kick/
        snare/
        ghost/
        hat/
        perc/

Usage :
    python classifier/classify_slices_v01.py

Options :
    python classifier/classify_slices_v01.py --copy
    python classifier/classify_slices_v01.py --limit 500
"""

from pathlib import Path
import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:
    print("librosa manquant.")
    print("Installe dans le venv : pip install librosa soundfile numpy")
    sys.exit(1)

DATASET = Path("dataset")
SLICES_MANIFEST = DATASET / "slices_manifest.json"

OUT_DIR = DATASET / "drum_library"
SAMPLES_DIR = OUT_DIR / "samples"
LIBRARY_JSON = OUT_DIR / "drum_library_v01.json"
REPORT_TXT = OUT_DIR / "report_v01.txt"

LABELS = ["kick", "snare", "ghost", "hat", "perc"]


def safe_float(x):
    try:
        if math.isnan(float(x)) or math.isinf(float(x)):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def load_manifest():
    if not SLICES_MANIFEST.exists():
        print("Manquant :", SLICES_MANIFEST)
        print("Lance d'abord le build dataset.")
        sys.exit(1)

    return json.loads(SLICES_MANIFEST.read_text(encoding="utf-8"))


def read_audio(path):
    y, sr = sf.read(path, dtype="float32")

    if y.ndim > 1:
        y = y.mean(axis=1)

    if len(y) == 0:
        return np.zeros(1, dtype=np.float32), sr

    return y.astype(np.float32), sr


def normalize_for_analysis(y):
    m = np.max(np.abs(y)) if len(y) else 0
    if m <= 1e-9:
        return y
    return y / m


def audio_features(y, sr):
    y = np.asarray(y, dtype=np.float32)
    yn = normalize_for_analysis(y)

    duration = len(y) / sr
    rms = safe_float(np.sqrt(np.mean(y * y))) if len(y) else 0.0
    peak = safe_float(np.max(np.abs(y))) if len(y) else 0.0

    if len(yn) < 128:
        return {
            "duration": duration,
            "rms": rms,
            "peak": peak,
            "centroid": 0.0,
            "bandwidth": 0.0,
            "rolloff": 0.0,
            "zcr": 0.0,
            "flatness": 0.0,
            "low_energy": 0.0,
            "mid_energy": 0.0,
            "high_energy": 0.0,
            "attack": 0.0,
            "tail_ratio": 0.0,
        }

    centroid = safe_float(np.mean(librosa.feature.spectral_centroid(y=yn, sr=sr)))
    bandwidth = safe_float(np.mean(librosa.feature.spectral_bandwidth(y=yn, sr=sr)))
    rolloff = safe_float(np.mean(librosa.feature.spectral_rolloff(y=yn, sr=sr)))
    zcr = safe_float(np.mean(librosa.feature.zero_crossing_rate(yn)))
    flatness = safe_float(np.mean(librosa.feature.spectral_flatness(y=yn)))

    # Energie par bandes
    spec = np.abs(np.fft.rfft(yn))
    freqs = np.fft.rfftfreq(len(yn), 1.0 / sr)

    total_spec = np.sum(spec) + 1e-9
    low_energy = safe_float(np.sum(spec[(freqs >= 20) & (freqs < 180)]) / total_spec)
    mid_energy = safe_float(np.sum(spec[(freqs >= 180) & (freqs < 2500)]) / total_spec)
    high_energy = safe_float(np.sum(spec[(freqs >= 2500) & (freqs < 12000)]) / total_spec)

    # Attaque : énergie premiers 25 ms / énergie totale
    attack_len = max(1, int(sr * 0.025))
    attack = safe_float(np.sqrt(np.mean(y[:attack_len] * y[:attack_len])) / (rms + 1e-9))

    # Tail ratio : énergie fin / énergie début
    head_len = max(1, int(sr * 0.060))
    tail_len = max(1, int(sr * 0.120))
    head_rms = safe_float(np.sqrt(np.mean(y[:head_len] * y[:head_len]))) if len(y) else 0.0
    tail_rms = safe_float(np.sqrt(np.mean(y[-tail_len:] * y[-tail_len:]))) if len(y) else 0.0
    tail_ratio = safe_float(tail_rms / (head_rms + 1e-9))

    return {
        "duration": safe_float(duration),
        "rms": rms,
        "peak": peak,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "zcr": zcr,
        "flatness": flatness,
        "low_energy": low_energy,
        "mid_energy": mid_energy,
        "high_energy": high_energy,
        "attack": attack,
        "tail_ratio": tail_ratio,
    }


def classify(features):
    """
    Classifier heuristique v01.
    Il n'est pas parfait, mais il est beaucoup plus riche que l'ancien.

    kick :
        grave, low_energy haut, centroid bas, rms/attack présents

    snare :
        medium + énergie, centroid moyen, bruit mais pas trop aigu

    ghost :
        faible rms ou faible peak, souvent petit coup proche snare

    hat :
        centroid élevé, high_energy/zcr élevés, souvent court

    perc :
        reste
    """

    rms = features["rms"]
    peak = features["peak"]
    duration = features["duration"]
    centroid = features["centroid"]
    zcr = features["zcr"]
    low = features["low_energy"]
    mid = features["mid_energy"]
    high = features["high_energy"]
    flatness = features["flatness"]
    attack = features["attack"]

    # Silence / ghost très faible
    if rms < 0.010 or peak < 0.035:
        return "ghost", 0.72

    # Kick
    kick_score = 0.0
    if low > 0.28:
        kick_score += 0.35
    if centroid < 1500:
        kick_score += 0.25
    if rms > 0.020:
        kick_score += 0.15
    if attack > 1.15:
        kick_score += 0.10
    if duration > 0.055:
        kick_score += 0.05
    if high < 0.38:
        kick_score += 0.10

    # Hat
    hat_score = 0.0
    if centroid > 4200:
        hat_score += 0.35
    if high > 0.45:
        hat_score += 0.25
    if zcr > 0.10:
        hat_score += 0.15
    if duration < 0.26:
        hat_score += 0.10
    if low < 0.18:
        hat_score += 0.10
    if flatness > 0.08:
        hat_score += 0.05

    # Snare
    snare_score = 0.0
    if 1100 <= centroid <= 5200:
        snare_score += 0.28
    if mid > 0.30:
        snare_score += 0.24
    if rms > 0.018:
        snare_score += 0.16
    if 0.035 <= duration <= 0.45:
        snare_score += 0.10
    if 0.04 <= zcr <= 0.22:
        snare_score += 0.10
    if high > 0.18:
        snare_score += 0.07
    if low < 0.36:
        snare_score += 0.05

    # Perc
    perc_score = 0.35
    if mid > 0.25:
        perc_score += 0.15
    if 0.06 <= duration <= 0.50:
        perc_score += 0.10
    if 1500 <= centroid <= 6500:
        perc_score += 0.10

    scores = {
        "kick": kick_score,
        "snare": snare_score,
        "hat": hat_score,
        "perc": perc_score,
    }

    label = max(scores.items(), key=lambda x: x[1])[0]
    confidence = max(scores.values())

    # Ghost peut aussi être une snare douce
    if rms < 0.018 and label in ["snare", "perc"]:
        return "ghost", max(0.55, confidence * 0.85)

    # Cas clash kick/snare : si très grave, préfère kick
    if label == "snare" and low > 0.38 and centroid < 1800:
        return "kick", max(confidence, 0.65)

    # Cas snare/hat : si très aigu et très court, préfère hat
    if label == "snare" and centroid > 5000 and duration < 0.20:
        return "hat", max(confidence, 0.68)

    return label, safe_float(confidence)


def brightness_tag(features):
    c = features["centroid"]
    if c < 1500:
        return "dark"
    if c < 3500:
        return "medium"
    if c < 6500:
        return "bright"
    return "very_bright"


def length_tag(features):
    d = features["duration"]
    if d < 0.08:
        return "short"
    if d < 0.22:
        return "medium"
    return "long"


def energy_tag(features):
    rms = features["rms"]
    if rms < 0.015:
        return "soft"
    if rms < 0.035:
        return "medium"
    return "hard"


def copy_or_link_sample(src, dst, copy_files=True):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return

    if copy_files:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except Exception:
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--copy", action="store_true",
                        help="Copie les fichiers dans dataset/drum_library/samples. Par défaut: symlinks si possible.")
    parser.add_argument("--no-files", action="store_true",
                        help="N'écrit que le JSON/report, sans copier/linker les samples.")
    args = parser.parse_args()

    manifest = load_manifest()

    if args.limit:
        manifest = manifest[:args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    library = []
    counts = Counter()
    by_source = defaultdict(Counter)
    missing = 0

    for i, item in enumerate(manifest, start=1):
        src_path = Path(item["slice_file"])

        if not src_path.exists():
            missing += 1
            continue

        try:
            y, sr = read_audio(src_path)
        except Exception as e:
            print("Erreur lecture :", src_path, e)
            continue

        features = audio_features(y, sr)
        label, confidence = classify(features)

        counts[label] += 1
        by_source[item.get("source", "unknown")][label] += 1

        bright = brightness_tag(features)
        length = length_tag(features)
        energy = energy_tag(features)

        sample_name = f"{label}_{counts[label]:05d}_{energy}_{bright}_{length}.wav"
        out_rel = Path("samples") / label / sample_name
        out_abs = OUT_DIR / out_rel

        if not args.no_files:
            copy_or_link_sample(src_path, out_abs, copy_files=args.copy)

        row = {
            "id": f"{label}_{counts[label]:05d}",
            "label": label,
            "confidence": round(confidence, 5),
            "source_slice": str(src_path),
            "library_file": str(out_abs),
            "library_rel": str(out_rel),
            "source_break": item.get("source", ""),
            "break_id": item.get("break_id", ""),
            "tags": {
                "energy": energy,
                "brightness": bright,
                "length": length,
            },
            "features": {k: round(safe_float(v), 6) for k, v in features.items()},
            "original_manifest": item,
        }

        library.append(row)

        if i % 500 == 0:
            print(f"{i}/{len(manifest)} slices analysées")

    payload = {
        "version": "drum_library_v01",
        "count": len(library),
        "missing": missing,
        "label_counts": dict(counts),
        "labels": LABELS,
        "samples": library,
    }

    LIBRARY_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("DRUM LIBRARY v01")
    lines.append("================")
    lines.append("")
    lines.append(f"Slices analysées : {len(library)}")
    lines.append(f"Fichiers manquants : {missing}")
    lines.append("")
    lines.append("Répartition :")
    for label in LABELS:
        lines.append(f"  {label:8} {counts[label]}")
    lines.append("")

    lines.append("Exemples par label :")
    for label in LABELS:
        lines.append("")
        lines.append(f"[{label}]")
        examples = [x for x in library if x["label"] == label][:12]
        for ex in examples:
            f = ex["features"]
            tags = ex["tags"]
            lines.append(
                f"  {Path(ex['library_file']).name} "
                f"conf={ex['confidence']:.3f} "
                f"rms={f['rms']:.4f} "
                f"centroid={f['centroid']:.0f} "
                f"low={f['low_energy']:.2f} "
                f"high={f['high_energy']:.2f} "
                f"tags={tags['energy']}/{tags['brightness']}/{tags['length']}"
            )

    lines.append("")
    lines.append("Top sources par quantité :")
    for source, c in list(by_source.items())[:30]:
        total = sum(c.values())
        lines.append(f"  {Path(source).name} total={total} {dict(c)}")

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("")
    print("Drum Library créée :")
    print(" ", LIBRARY_JSON)
    print(" ", REPORT_TXT)
    print("")
    print("Répartition :")
    for label in LABELS:
        print(f"  {label:8} {counts[label]}")


if __name__ == "__main__":
    main()
