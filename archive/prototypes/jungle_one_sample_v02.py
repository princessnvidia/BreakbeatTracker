#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jungle One Sample v02

Version corrigée pour Camo :
- utilise UN SEUL fichier break source
- pas de drum library
- pas de samples d'autres breaks
- si la classification ne trouve pas de kick/snare/hat,
  elle choisit les meilleurs candidats RELATIFS dans ce même break

Usage :
    python jungle_one_sample_v02.py --source "Camo" --count 32 --mutation 0.30 --bpm 150
"""

from pathlib import Path
import argparse
import random
import sys
import json
import warnings

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:
    print("librosa manquant : pip install librosa soundfile numpy")
    sys.exit(1)


BREAKS_DIR = Path("breaks")
OUT_DIR = Path("exports/jungle_one_sample_v02")

SR = 44100
STEPS = 32
AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".mp3"}

BASE_16 = "K..S..KSK..S..KS"
BASE_32 = BASE_16 + BASE_16

SEEDS = [
    BASE_32,
    "K..S..KSK..S.gKSK..S..KSK..S.gKS",
    "K.gS..KSK..S..KSK.gS..KSK..S..KS",
    "K..S.KKSK.gS..KSK..S..KSK.gS.gKS",
    "K..S..KSK.gS..KSK..S.KKSK..S.gKS",
]


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
    if m <= 1e-9:
        return y
    return y / m * peak


def load_audio(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    return normalize(y), sr


def safe_centroid(y, sr):
    if len(y) < 64:
        return 0.0
    n_fft = min(2048, max(64, 2 ** int(np.floor(np.log2(max(64, len(y)))))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft)))


def slice_features(y, sr):
    if len(y) < 64:
        return {"rms": 0.0, "peak": 0.0, "centroid": 0.0, "low": 0.0, "mid": 0.0, "high": 0.0, "duration": len(y) / sr}

    rms = float(np.sqrt(np.mean(y * y)))
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    centroid = safe_centroid(y, sr)
    duration = len(y) / sr

    yn = normalize(y, peak=1.0)
    spec = np.abs(np.fft.rfft(yn))
    freqs = np.fft.rfftfreq(len(yn), 1.0 / sr)
    total = np.sum(spec) + 1e-9

    low = float(np.sum(spec[(freqs >= 20) & (freqs < 250)]) / total)
    mid = float(np.sum(spec[(freqs >= 250) & (freqs < 3000)]) / total)
    high = float(np.sum(spec[(freqs >= 3000) & (freqs < 12000)]) / total)

    return {
        "rms": rms,
        "peak": peak,
        "centroid": centroid,
        "low": low,
        "mid": mid,
        "high": high,
        "duration": duration,
    }


def detect_slices(y, sr):
    onsets = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        units="samples",
        backtrack=True,
        delta=0.08,
        wait=1,
    )

    points = sorted(set([0] + [int(x) for x in onsets] + [len(y)]))

    slices = []

    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]

        if end - start < int(sr * 0.020):
            continue

        end = min(end, start + int(sr * 0.55))
        chunk = y[start:end].copy()

        analysis = chunk[:min(len(chunk), int(sr * 0.25))]
        feats = slice_features(analysis, sr)

        label = "perc"

        # labels bruts, mais le kit sera choisi ensuite par score relatif
        if feats["centroid"] > 3600 and feats["high"] > 0.25:
            label = "hat"
        elif feats["low"] > 0.10 and feats["centroid"] < 3000:
            label = "kick"
        elif 700 <= feats["centroid"] <= 6500:
            label = "snare"

        slices.append({
            "index": len(slices),
            "start": start,
            "end": end,
            "audio": chunk,
            "label": label,
            "features": feats,
        })

    return slices


def score_kick(s):
    f = s["features"]
    return (
        f["low"] * 3.0
        + f["rms"] * 5.0
        + f["peak"] * 2.0
        + max(0.0, 3000.0 - f["centroid"]) / 3000.0
        - f["high"] * 0.8
    )


def score_snare(s):
    f = s["features"]
    mid_bonus = f["mid"] * 2.0
    centroid_bonus = 1.0 if 700 <= f["centroid"] <= 6500 else 0.0
    return (
        f["rms"] * 5.0
        + f["peak"] * 1.5
        + mid_bonus
        + centroid_bonus
        + f["high"] * 0.4
    )


def score_hat(s):
    f = s["features"]
    return (
        f["high"] * 3.0
        + f["centroid"] / 8000.0
        + max(0.0, 0.35 - f["duration"])
        - f["low"] * 0.8
    )


def choose_kit_relative(slices):
    if len(slices) < 3:
        print("Pas assez de slices dans ce sample.")
        sys.exit(1)

    # Tout est choisi parmi CE break, même si les labels sont mauvais.
    kick_candidates = sorted(slices, key=score_kick, reverse=True)
    snare_candidates = sorted(slices, key=score_snare, reverse=True)
    hat_candidates = sorted(slices, key=score_hat, reverse=True)

    kick = kick_candidates[0]

    # évite de prendre exactement la même slice pour snare
    snare = next((s for s in snare_candidates if s["index"] != kick["index"]), snare_candidates[0])

    used = {kick["index"], snare["index"]}
    hat = next((s for s in hat_candidates if s["index"] not in used), hat_candidates[0])

    # ghost = snare plus douce, si possible différente
    ghost_pool = sorted(
        [s for s in snare_candidates if s["index"] not in used and s["index"] != hat["index"]],
        key=lambda s: s["features"]["rms"]
    )
    ghost = ghost_pool[0] if ghost_pool else snare

    return {
        "kick": kick,
        "snare": snare,
        "ghost": ghost,
        "hat": hat,
    }


def fade(y, sr, ms=3):
    if len(y) < 16:
        return y
    n = min(int(sr * ms / 1000), len(y) // 4)
    if n <= 1:
        return y
    y = y.copy()
    ramp = np.linspace(0, 1, n)
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def trim(y, role, step_samples, sr):
    max_steps = {
        "kick": 2.0,
        "snare": 2.0,
        "ghost": 0.75,
        "hat": 0.55,
    }[role]
    return fade(y[:int(step_samples * max_steps)], sr)


def mutate_pattern(mutation=0.30, hat_density=0.72):
    grid = list(random.choice(SEEDS))

    base_kicks = [0, 6, 8, 14, 16, 22, 24, 30]
    base_snares = [3, 7, 11, 15, 19, 23, 27, 31]

    for i in base_kicks:
        if random.random() < 0.92:
            grid[i] = "K"
    for i in base_snares:
        if random.random() < 0.92:
            grid[i] = "S"

    for i in [2, 4, 5, 10, 12, 13, 18, 20, 21, 26, 28, 29]:
        if grid[i] == "." and random.random() < mutation * 0.75:
            grid[i] = "g"

    for i in [1, 5, 9, 12, 17, 21, 25, 28]:
        if grid[i] == "." and random.random() < mutation * 0.25:
            grid[i] = "K"

    for i in range(STEPS):
        if grid[i] == ".":
            if i % 2 == 0 and random.random() < hat_density:
                grid[i] = "H"
            elif i % 2 == 1 and random.random() < hat_density * 0.35:
                grid[i] = "H"

    return "".join(grid[:STEPS])


def render(grid, kit, bpm, swing, humanize):
    step_samples = int(SR * (60.0 / bpm / 4.0))
    out = np.zeros(step_samples * STEPS + SR, dtype=np.float32)

    audio = {
        role: trim(kit[role]["audio"], role, step_samples, SR)
        for role in kit
    }

    gains = {"K": 0.95, "S": 0.88, "g": 0.22, "H": 0.18}
    role_for = {"K": "kick", "S": "snare", "g": "ghost", "H": "hat"}

    for step, ch in enumerate(grid):
        if ch not in role_for:
            continue

        role = role_for[ch]
        y = audio[role]
        gain = gains[ch]

        if ch in "gH":
            gain *= random.uniform(0.92, 1.06)
        else:
            gain *= random.uniform(0.98, 1.02)

        start = step * step_samples

        if swing > 0 and step % 2 == 1:
            start += int(step_samples * swing * 0.35)

        if ch in "gH":
            start += random.randint(-int(step_samples * humanize), int(step_samples * humanize))

        if start < 0:
            continue

        end = min(start + len(y), len(out))
        if end > start:
            out[start:end] += y[:end-start] * gain

    return normalize(out)


def split_grid(grid):
    return "|".join(grid[i:i+8] for i in range(0, 32, 8))


def layers_text(grid):
    rows = {"KICK ": "", "SNARE": "", "GHOST": "", "HAT  ": ""}
    for ch in grid:
        rows["KICK "] += "K" if ch == "K" else "."
        rows["SNARE"] += "S" if ch == "S" else "."
        rows["GHOST"] += "g" if ch == "g" else "."
        rows["HAT  "] += "H" if ch == "H" else "."

    lines = ["12345678|12345678|12345678|12345678"]
    for k, v in rows.items():
        lines.append(f"{k}: {split_grid(v)}")
    lines.append("FULL : " + split_grid(grid))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Camo")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--bpm", type=float, default=150.0)
    parser.add_argument("--mutation", type=float, default=0.30)
    parser.add_argument("--hat-density", type=float, default=0.72)
    parser.add_argument("--swing", type=float, default=0.025)
    parser.add_argument("--humanize", type=float, default=0.004)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    src = find_source(args.source)
    print("Source unique :", src)

    y, sr = load_audio(src)
    slices = detect_slices(y, sr)

    counts = {}
    for s in slices:
        counts[s["label"]] = counts.get(s["label"], 0) + 1

    print("Slices détectées uniquement depuis ce fichier :")
    for k in ["kick", "snare", "hat", "perc"]:
        print(f"  {k:8}: {counts.get(k, 0)}")

    kit = choose_kit_relative(slices)

    print("")
    print("Kit unique choisi par score relatif :")
    for role, s in kit.items():
        f = s["features"]
        print(
            f"  {role:6}: slice={s['index']} raw_label={s['label']} "
            f"rms={f['rms']:.4f} peak={f['peak']:.3f} "
            f"centroid={f['centroid']:.0f} low={f['low']:.2f} high={f['high']:.2f}"
        )

    safe = src.stem.replace(" ", "_").replace("'", "")
    outdir = OUT_DIR / safe
    outdir.mkdir(parents=True, exist_ok=True)

    report = []

    for i in range(1, args.count + 1):
        grid = mutate_pattern(args.mutation, args.hat_density)
        audio = render(grid, kit, args.bpm, args.swing, args.humanize)

        wav = outdir / f"{safe}_one_sample_jungle_v02_{i:03d}_{int(args.bpm)}bpm.wav"
        sf.write(wav, audio, SR)

        report.append(f"VARIATION {i:03d} -> {wav}")
        report.append("BASE : K..S|..KS|K..S|..KS")
        report.append(layers_text(grid))
        report.append("")

        print("Export :", wav)

    (outdir / "patterns_one_sample_jungle_v02.txt").write_text("\n".join(report), encoding="utf-8")

    meta = {
        "version": "jungle_one_sample_v02",
        "source": str(src),
        "source_only": True,
        "base": "K..S|..KS|K..S|..KS",
        "slice_counts_raw": counts,
        "kit": {
            role: {
                "slice_index": s["index"],
                "raw_label": s["label"],
                "features": s["features"],
            }
            for role, s in kit.items()
        },
    }
    (outdir / "metadata_one_sample_jungle_v02.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("")
    print("Dossier :", outdir)


if __name__ == "__main__":
    main()
