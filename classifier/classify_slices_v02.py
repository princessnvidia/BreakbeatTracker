#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Drum Classifier v02 - classify_slices_v02.py

Améliorations vs v01 :
- détecte enfin les ghost snares
- récupère plus de kicks
- évite les warnings n_fft trop grand sur les samples courts
- ajoute des tags plus utiles pour SampleBrain
- garde un rapport clair sur la répartition

Entrée :
    dataset/slices_manifest.json

Sorties :
    dataset/drum_library_v02/drum_library_v02.json
    dataset/drum_library_v02/report_v02.txt
    dataset/drum_library_v02/samples/
        kick/
        snare/
        ghost/
        hat/
        perc/

Usage :
    python classifier/classify_slices_v02.py
    python classifier/classify_slices_v02.py --limit 500
    python classifier/classify_slices_v02.py --copy
"""

from pathlib import Path
import argparse
import json
import math
import shutil
import sys
import warnings
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

OUT_DIR = DATASET / "drum_library_v02"
SAMPLES_DIR = OUT_DIR / "samples"
LIBRARY_JSON = OUT_DIR / "drum_library_v02.json"
REPORT_TXT = OUT_DIR / "report_v02.txt"

LABELS = ["kick", "snare", "ghost", "hat", "perc"]


def safe_float(x):
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return 0.0
        return x
    except Exception:
        return 0.0


def load_manifest():
    if not SLICES_MANIFEST.exists():
        print("Manquant :", SLICES_MANIFEST)
        sys.exit(1)
    return json.loads(SLICES_MANIFEST.read_text(encoding="utf-8"))


def read_audio(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) == 0:
        y = np.zeros(1, dtype=np.float32)
    return y.astype(np.float32), sr


def normalize_for_analysis(y):
    m = np.max(np.abs(y)) if len(y) else 0
    if m <= 1e-9:
        return y
    return y / m


def librosa_feature_safe(func, y, sr=None):
    """
    Évite les warnings n_fft sur très petits samples.
    """
    if len(y) < 64:
        return 0.0

    n_fft = min(2048, max(64, 2 ** int(np.floor(np.log2(max(64, len(y)))))))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if sr is None:
                val = func(y=y, n_fft=n_fft)
            else:
                val = func(y=y, sr=sr, n_fft=n_fft)
        return safe_float(np.mean(val))
    except Exception:
        return 0.0


def audio_features(y, sr):
    y = np.asarray(y, dtype=np.float32)
    yn = normalize_for_analysis(y)

    duration = len(y) / sr
    rms = safe_float(np.sqrt(np.mean(y * y))) if len(y) else 0.0
    peak = safe_float(np.max(np.abs(y))) if len(y) else 0.0

    centroid = librosa_feature_safe(librosa.feature.spectral_centroid, yn, sr)
    bandwidth = librosa_feature_safe(librosa.feature.spectral_bandwidth, yn, sr)
    rolloff = librosa_feature_safe(librosa.feature.spectral_rolloff, yn, sr)

    try:
        zcr = safe_float(np.mean(librosa.feature.zero_crossing_rate(yn)))
    except Exception:
        zcr = 0.0

    try:
        flatness = librosa_feature_safe(librosa.feature.spectral_flatness, yn, None)
    except Exception:
        flatness = 0.0

    spec = np.abs(np.fft.rfft(yn))
    freqs = np.fft.rfftfreq(len(yn), 1.0 / sr)
    total_spec = np.sum(spec) + 1e-9

    low_energy = safe_float(np.sum(spec[(freqs >= 20) & (freqs < 180)]) / total_spec)
    low_mid_energy = safe_float(np.sum(spec[(freqs >= 180) & (freqs < 700)]) / total_spec)
    mid_energy = safe_float(np.sum(spec[(freqs >= 700) & (freqs < 2500)]) / total_spec)
    high_energy = safe_float(np.sum(spec[(freqs >= 2500) & (freqs < 12000)]) / total_spec)

    attack_len = max(1, int(sr * 0.020))
    attack_rms = safe_float(np.sqrt(np.mean(y[:attack_len] * y[:attack_len]))) if len(y) else 0.0
    attack = safe_float(attack_rms / (rms + 1e-9))

    head_len = max(1, int(sr * 0.060))
    tail_len = max(1, int(sr * 0.120))
    head_rms = safe_float(np.sqrt(np.mean(y[:head_len] * y[:head_len]))) if len(y) else 0.0
    tail_rms = safe_float(np.sqrt(np.mean(y[-tail_len:] * y[-tail_len:]))) if len(y) else 0.0
    tail_ratio = safe_float(tail_rms / (head_rms + 1e-9))

    # "body" snare/kick : énergie du milieu par rapport aux aigus
    body_ratio = safe_float((low_energy + low_mid_energy + mid_energy) / (high_energy + 1e-9))

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
        "low_mid_energy": low_mid_energy,
        "mid_energy": mid_energy,
        "high_energy": high_energy,
        "body_ratio": body_ratio,
        "attack": attack,
        "tail_ratio": tail_ratio,
    }


def score_labels(f):
    rms = f["rms"]
    peak = f["peak"]
    dur = f["duration"]
    c = f["centroid"]
    zcr = f["zcr"]
    low = f["low_energy"]
    lowmid = f["low_mid_energy"]
    mid = f["mid_energy"]
    high = f["high_energy"]
    flat = f["flatness"]
    attack = f["attack"]
    body = f["body_ratio"]

    kick = 0.0
    if low > 0.16: kick += 0.26
    if low + lowmid > 0.34: kick += 0.22
    if c < 1850: kick += 0.20
    if body > 1.2: kick += 0.12
    if rms > 0.014: kick += 0.08
    if attack > 1.05: kick += 0.07
    if high < 0.50: kick += 0.05

    snare = 0.0
    if 900 <= c <= 5600: snare += 0.23
    if lowmid + mid > 0.30: snare += 0.22
    if rms > 0.014: snare += 0.14
    if 0.035 <= dur <= 0.50: snare += 0.10
    if 0.035 <= zcr <= 0.25: snare += 0.08
    if high > 0.12: snare += 0.08
    if peak > 0.045: snare += 0.08
    if flat > 0.015: snare += 0.04

    hat = 0.0
    if c > 3800: hat += 0.28
    if high > 0.36: hat += 0.24
    if zcr > 0.075: hat += 0.16
    if dur < 0.30: hat += 0.10
    if low < 0.18: hat += 0.10
    if flat > 0.04: hat += 0.07
    if body < 1.35: hat += 0.05

    perc = 0.30
    if lowmid + mid > 0.22: perc += 0.13
    if 0.04 <= dur <= 0.55: perc += 0.11
    if 1200 <= c <= 7000: perc += 0.08
    if peak > 0.030: perc += 0.05

    # Ghost = souvent une snare/perc douce, pas juste silence.
    ghost = 0.0
    if rms < 0.020: ghost += 0.25
    if peak < 0.095: ghost += 0.18
    if 0.025 <= dur <= 0.30: ghost += 0.13
    if 900 <= c <= 6000: ghost += 0.16
    if lowmid + mid > 0.18: ghost += 0.14
    if high < 0.70: ghost += 0.08
    if zcr < 0.22: ghost += 0.06

    return {
        "kick": kick,
        "snare": snare,
        "ghost": ghost,
        "hat": hat,
        "perc": perc,
    }


def classify(features):
    scores = score_labels(features)

    rms = features["rms"]
    peak = features["peak"]
    c = features["centroid"]
    high = features["high_energy"]
    low = features["low_energy"]
    lowmid = features["low_mid_energy"]
    mid = features["mid_energy"]
    dur = features["duration"]

    # Règles de correction avant max :
    # 1. Hat très évident
    if c > 5200 and high > 0.42 and low < 0.20:
        return "hat", max(scores["hat"], 0.78), scores

    # 2. Kick évident, même si centroid un peu haut
    if (low + lowmid) > 0.42 and c < 2300 and peak > 0.040:
        return "kick", max(scores["kick"], 0.72), scores

    # 3. Ghost snare doux : l'ancien classifier en mettait 0
    if (
        rms < 0.022
        and peak < 0.120
        and 800 <= c <= 6200
        and (lowmid + mid) > 0.18
        and high < 0.78
    ):
        return "ghost", max(scores["ghost"], 0.70), scores

    label = max(scores.items(), key=lambda x: x[1])[0]
    confidence = scores[label]

    # Corrections après max :
    if label == "snare":
        if rms < 0.020 and peak < 0.110:
            return "ghost", max(scores["ghost"], confidence * 0.90), scores

    if label == "perc":
        if rms < 0.018 and 800 <= c <= 6000:
            return "ghost", max(scores["ghost"], confidence * 0.85), scores

    if label == "hat":
        # Un hat faible reste hat, pas ghost, si très aigu.
        if c < 3800 and rms < 0.018:
            return "ghost", max(scores["ghost"], confidence * 0.80), scores

    # Encore plus de kicks : si grave mais classé perc/snare
    if label in ["snare", "perc"] and (low + lowmid) > 0.40 and c < 2500 and high < 0.45:
        return "kick", max(scores["kick"], 0.66), scores

    return label, confidence, scores


def tag_value(value, thresholds, names):
    for threshold, name in zip(thresholds, names):
        if value < threshold:
            return name
    return names[-1]


def tags_for(features, label):
    energy = tag_value(features["rms"], [0.012, 0.025, 0.045], ["very_soft", "soft", "medium", "hard"])
    brightness = tag_value(features["centroid"], [1500, 3500, 6500], ["dark", "medium", "bright", "very_bright"])
    length = tag_value(features["duration"], [0.07, 0.18, 0.38], ["tiny", "short", "medium", "long"])

    body = "thin"
    if features["body_ratio"] > 2.2:
        body = "thick"
    elif features["body_ratio"] > 1.2:
        body = "body"

    return {
        "energy": energy,
        "brightness": brightness,
        "length": length,
        "body": body,
        "family": label,
    }


def copy_or_link(src, dst, copy_files):
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
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--no-files", action="store_true")
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
        src = Path(item["slice_file"])

        if not src.exists():
            missing += 1
            continue

        try:
            y, sr = read_audio(src)
            feats = audio_features(y, sr)
            label, confidence, scores = classify(feats)
        except Exception as e:
            print("Erreur analyse :", src, e)
            continue

        counts[label] += 1
        by_source[item.get("source", "unknown")][label] += 1

        tags = tags_for(feats, label)
        sample_name = (
            f"{label}_{counts[label]:05d}_"
            f"{tags['energy']}_{tags['brightness']}_{tags['length']}.wav"
        )

        rel = Path("samples") / label / sample_name
        dst = OUT_DIR / rel

        if not args.no_files:
            copy_or_link(src, dst, copy_files=args.copy)

        row = {
            "id": f"{label}_{counts[label]:05d}",
            "label": label,
            "confidence": round(safe_float(confidence), 5),
            "scores": {k: round(safe_float(v), 5) for k, v in scores.items()},
            "source_slice": str(src),
            "library_file": str(dst),
            "library_rel": str(rel),
            "source_break": item.get("source", ""),
            "break_id": item.get("break_id", ""),
            "tags": tags,
            "features": {k: round(safe_float(v), 6) for k, v in feats.items()},
            "original_manifest": item,
        }
        library.append(row)

        if i % 500 == 0:
            print(f"{i}/{len(manifest)} slices analysées")

    payload = {
        "version": "drum_library_v02",
        "count": len(library),
        "missing": missing,
        "label_counts": dict(counts),
        "labels": LABELS,
        "samples": library,
    }
    LIBRARY_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("DRUM LIBRARY v02")
    lines.append("================")
    lines.append("")
    lines.append(f"Slices analysées : {len(library)}")
    lines.append(f"Fichiers manquants : {missing}")
    lines.append("")
    lines.append("Répartition :")
    for label in LABELS:
        pct = counts[label] / max(1, len(library)) * 100
        lines.append(f"  {label:8} {counts[label]:5d}  {pct:5.1f}%")
    lines.append("")

    lines.append("Exemples par label :")
    for label in LABELS:
        lines.append("")
        lines.append(f"[{label}]")
        examples = [x for x in library if x["label"] == label][:16]
        for ex in examples:
            f = ex["features"]
            t = ex["tags"]
            sc = ex["scores"]
            lines.append(
                f"  {Path(ex['library_file']).name} "
                f"conf={ex['confidence']:.3f} "
                f"rms={f['rms']:.4f} peak={f['peak']:.3f} "
                f"centroid={f['centroid']:.0f} "
                f"low={f['low_energy']:.2f} lowmid={f['low_mid_energy']:.2f} "
                f"mid={f['mid_energy']:.2f} high={f['high_energy']:.2f} "
                f"tags={t['energy']}/{t['brightness']}/{t['length']}/{t['body']} "
                f"scores=K{sc['kick']:.2f} S{sc['snare']:.2f} g{sc['ghost']:.2f} H{sc['hat']:.2f} p{sc['perc']:.2f}"
            )

    lines.append("")
    lines.append("Top sources par quantité :")
    for source, c in sorted(by_source.items(), key=lambda x: sum(x[1].values()), reverse=True)[:40]:
        lines.append(f"  {Path(source).name} total={sum(c.values())} {dict(c)}")

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("")
    print("Drum Library v02 créée :")
    print(" ", LIBRARY_JSON)
    print(" ", REPORT_TXT)
    print("")
    print("Répartition :")
    for label in LABELS:
        print(f"  {label:8} {counts[label]}")


if __name__ == "__main__":
    main()
