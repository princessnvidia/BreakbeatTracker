#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tracker Lane Base v03

Tracker automatique one-sample avec logique "comme la photo" :
- une seule source audio
- macro-découpes longues sur grille
- pas de superposition
- base jungle explicite :
      K..S|..KS|K..S|..KS
- les hats remplissent les trous
- variations contrôlées autour de cette base
- export WAV + TXT

Différence avec tracker_one_sample_v01/v02 :
Les anciennes versions découpaient aux transients, donc les slices étaient trop courtes.
Cette v03 découpe le break en cellules musicales plus longues.

Usage :
    python tracker_lane_base_v03.py --source "Camo"

    python tracker_lane_base_v03.py \
        --source "Camo" \
        --count 32 \
        --mutation 0.25 \
        --bpm 150

Si c'est encore trop haché :
    python tracker_lane_base_v03.py --source "Camo" --cell-ms 180

Si c'est trop lent/long :
    python tracker_lane_base_v03.py --source "Camo" --cell-ms 120
"""

from pathlib import Path
import argparse
import json
import random
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
OUT_DIR = Path("exports/tracker_lane_base_v03")

SR = 44100
AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".mp3"}

# 16 steps : K..S|..KS|K..S|..KS
BASE_16 = "K..S..KSK..S..KS"

ALLOWED = set(".KSgH")


def find_source(name):
    files = sorted(
        p for p in BREAKS_DIR.rglob("*")
        if p.suffix.lower() in AUDIO_EXTS and not p.name.endswith(".asd")
    )
    matches = [p for p in files if name.lower() in p.name.lower()]
    if not matches:
        print(f"Aucun break trouvé pour : {name}")
        print("Exemple : ls breaks | grep -i Camo")
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


def fade(y, ms=3):
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


def safe_centroid(y):
    if len(y) < 64:
        return 0.0
    n_fft = min(2048, max(64, 2 ** int(np.floor(np.log2(max(64, len(y)))))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=n_fft)))


def cell_features(y):
    if len(y) < 64:
        return {
            "rms": 0.0,
            "peak": 0.0,
            "centroid": 0.0,
            "low": 0.0,
            "mid": 0.0,
            "high": 0.0,
        }

    rms = float(np.sqrt(np.mean(y * y)))
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    centroid = safe_centroid(y)

    yn = normalize(y, peak=1.0)
    spec = np.abs(np.fft.rfft(yn))
    freqs = np.fft.rfftfreq(len(yn), 1.0 / SR)
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
    }


def make_macro_cells(y, cell_samples):
    """
    Découpe en cellules longues régulières.
    C'est volontairement PAS une découpe transient.
    """
    cells = []
    total = len(y)
    idx = 0
    start = 0

    while start < total:
        end = min(start + cell_samples, total)
        chunk = y[start:end].copy()

        if len(chunk) < int(cell_samples * 0.35):
            break

        # pad à taille fixe pour logique tracker
        if len(chunk) < cell_samples:
            chunk = np.concatenate([chunk, np.zeros(cell_samples - len(chunk), dtype=np.float32)])

        chunk = fade(chunk, ms=3)
        feats = cell_features(chunk)

        cells.append({
            "index": idx,
            "start": int(start),
            "end": int(end),
            "audio": chunk,
            "features": feats,
        })

        idx += 1
        start += cell_samples

    return cells


def score_kick(cell):
    f = cell["features"]
    return (
        f["low"] * 3.2
        + f["rms"] * 4.0
        + f["peak"] * 1.8
        + max(0.0, 3000.0 - f["centroid"]) / 3000.0
        - f["high"] * 0.7
    )


def score_snare(cell):
    f = cell["features"]
    centroid_ok = 1.0 if 700 <= f["centroid"] <= 6500 else 0.0
    return (
        f["mid"] * 2.0
        + f["rms"] * 4.0
        + f["peak"] * 1.4
        + f["high"] * 0.45
        + centroid_ok
    )


def score_hat(cell):
    f = cell["features"]
    return (
        f["high"] * 3.0
        + f["centroid"] / 9000.0
        - f["low"] * 0.9
        + f["peak"] * 0.3
    )


def choose_kit(cells):
    """
    Choisit un petit kit depuis les macro-cells du même break.
    Ce sont des cellules longues, donc plus proche de la découpe tracker.
    """
    if len(cells) < 4:
        print("Pas assez de macro-cells.")
        sys.exit(1)

    kick = max(cells, key=score_kick)

    snare_candidates = sorted(cells, key=score_snare, reverse=True)
    snare = next((c for c in snare_candidates if c["index"] != kick["index"]), snare_candidates[0])

    used = {kick["index"], snare["index"]}

    hat_candidates = sorted(cells, key=score_hat, reverse=True)
    hat = next((c for c in hat_candidates if c["index"] not in used), hat_candidates[0])

    ghost_candidates = sorted(
        [c for c in snare_candidates if c["index"] not in used and c["index"] != hat["index"]],
        key=lambda c: c["features"]["rms"]
    )
    ghost = ghost_candidates[0] if ghost_candidates else snare

    return {
        "kick": kick,
        "snare": snare,
        "ghost": ghost,
        "hat": hat,
    }


def make_base(length=16):
    if length <= 16:
        return BASE_16[:length].ljust(length, ".")
    reps = (length + len(BASE_16) - 1) // len(BASE_16)
    return (BASE_16 * reps)[:length]


def mutate_base_pattern(length=16, mutation=0.20, hat_density=0.85):
    """
    Base stricte K..S|..KS puis mutations contrôlées.
    On ne laisse pas le modèle partir en vrille.
    """
    grid = list(make_base(length))

    # Hats dans les silences : comme une piste du haut dans la photo.
    for i in range(length):
        if grid[i] == ".":
            if i % 2 == 0 and random.random() < hat_density:
                grid[i] = "H"
            elif i % 2 == 1 and random.random() < hat_density * 0.45:
                grid[i] = "H"

    # Ghosts près des snares.
    snare_positions = [i for i, ch in enumerate(grid) if ch == "S"]
    for s in snare_positions:
        for offset in [-1, 1, 2]:
            i = s + offset
            if 0 <= i < length and grid[i] == "H" and random.random() < mutation * 0.70:
                grid[i] = "g"

    # Kicks de relance très rares.
    relance_spots = [1, 5, 9, 13, 17, 21, 25, 29]
    for i in relance_spots:
        if i < length and grid[i] == "H" and random.random() < mutation * 0.22:
            grid[i] = "K"

    # Fills fin de phrase, mais toujours lisibles.
    if length >= 16 and random.random() < mutation:
        start = length - random.choice([4, 8])
        start = max(0, start)
        fill_mode = random.choice(["ghosts", "repeat_snare", "hat_roll"])

        if fill_mode == "ghosts":
            for i in range(start, length):
                if grid[i] in ["H", "."] and random.random() < 0.45:
                    grid[i] = "g"

        elif fill_mode == "repeat_snare":
            for i in range(start, length):
                if i % 2 == 1 and grid[i] in ["H", "."]:
                    grid[i] = "g"

        elif fill_mode == "hat_roll":
            for i in range(start, length):
                if grid[i] == ".":
                    grid[i] = "H"

    return "".join(grid)


def render_pattern(grid, kit, cell_samples):
    """
    Une seule timeline.
    À chaque step : une cellule jouée.
    Pas de superposition.
    """
    chunks = []

    role_for = {
        "K": "kick",
        "S": "snare",
        "g": "ghost",
        "H": "hat",
        ".": None,
    }

    gain_for = {
        "K": 0.95,
        "S": 0.88,
        "g": 0.28,
        "H": 0.20,
    }

    for ch in grid:
        role = role_for.get(ch)
        if role is None:
            chunks.append(np.zeros(cell_samples, dtype=np.float32))
            continue

        audio = kit[role]["audio"].copy()

        # assure taille fixe
        if len(audio) > cell_samples:
            audio = audio[:cell_samples]
        elif len(audio) < cell_samples:
            audio = np.concatenate([audio, np.zeros(cell_samples - len(audio), dtype=np.float32)])

        audio = fade(audio, ms=3)
        audio = audio * gain_for[ch]

        chunks.append(audio)

    return normalize(np.concatenate(chunks))


def split_grid(grid):
    return "|".join(grid[i:i+4] for i in range(0, len(grid), 4))


def layers_text(grid):
    rows = {
        "HAT  ": "",
        "KICK ": "",
        "SNARE": "",
        "GHOST": "",
    }

    for ch in grid:
        rows["HAT  "] += "H" if ch == "H" else "."
        rows["KICK "] += "K" if ch == "K" else "."
        rows["SNARE"] += "S" if ch == "S" else "."
        rows["GHOST"] += "g" if ch == "g" else "."

    lines = []
    lines.append("GROUP: " + "|".join(["1234"] * (len(grid) // 4)))
    lines.append("BASE : K..S|..KS|K..S|..KS")
    for name, row in rows.items():
        lines.append(f"{name}: {split_grid(row)}")
    lines.append("FULL : " + split_grid(grid))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Camo")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--bpm", type=float, default=150.0)
    parser.add_argument("--cell-ms", type=float, default=None,
                        help="Durée d'une cellule en ms. Par défaut calculée depuis BPM en double-croches.")
    parser.add_argument("--mutation", type=float, default=0.20)
    parser.add_argument("--hat-density", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    source = find_source(args.source)
    print("Source unique :", source)

    y, sr = load_audio(source)

    if args.cell_ms is None:
        # double-croche à BPM donné
        cell_samples = int(SR * (60.0 / args.bpm / 4.0))
    else:
        cell_samples = int(SR * args.cell_ms / 1000.0)

    print(f"Cellule tracker : {cell_samples / SR * 1000:.1f} ms")

    cells = make_macro_cells(y, cell_samples)

    if not cells:
        print("Aucune macro-cell créée.")
        sys.exit(1)

    print(f"Macro-cells : {len(cells)}")

    kit = choose_kit(cells)

    print("")
    print("Kit choisi depuis les macro-cells du même break :")
    for role, c in kit.items():
        f = c["features"]
        print(
            f"  {role:6}: cell={c['index']:03d} "
            f"rms={f['rms']:.4f} peak={f['peak']:.3f} "
            f"centroid={f['centroid']:.0f} low={f['low']:.2f} high={f['high']:.2f}"
        )

    safe = source.stem.replace(" ", "_").replace("'", "")
    outdir = OUT_DIR / safe
    outdir.mkdir(parents=True, exist_ok=True)

    report = []
    renders = []

    base_grid = make_base(args.steps)
    base_audio = render_pattern(base_grid, kit, cell_samples)

    base_wav = outdir / f"{safe}_lane_base_v03_base.wav"
    sf.write(base_wav, base_audio, SR)

    report.append(f"SOURCE: {source}")
    report.append(f"CELL_MS: {cell_samples / SR * 1000:.1f}")
    report.append(f"STEPS: {args.steps}")
    report.append("")
    report.append("BASE RENDER")
    report.append(str(base_wav))
    report.append(layers_text(base_grid))
    report.append("")

    for i in range(1, args.count + 1):
        grid = mutate_base_pattern(
            length=args.steps,
            mutation=args.mutation,
            hat_density=args.hat_density,
        )

        audio = render_pattern(grid, kit, cell_samples)

        wav = outdir / f"{safe}_lane_base_v03_{i:03d}.wav"
        sf.write(wav, audio, SR)

        renders.append({
            "index": i,
            "wav": str(wav),
            "grid": grid,
        })

        report.append(f"VARIATION {i:03d} -> {wav}")
        report.append(layers_text(grid))
        report.append("")

        print("Export :", wav)

    metadata = {
        "version": "tracker_lane_base_v03",
        "source": str(source),
        "source_only": True,
        "logic": "single tracker timeline, macro-cells, no overlap",
        "base": "K..S|..KS|K..S|..KS",
        "cell_ms": cell_samples / SR * 1000,
        "steps": args.steps,
        "bpm": args.bpm,
        "mutation": args.mutation,
        "hat_density": args.hat_density,
        "base_wav": str(base_wav),
        "kit": {
            role: {
                "cell_index": c["index"],
                "start": c["start"],
                "end": c["end"],
                "features": c["features"],
            }
            for role, c in kit.items()
        },
        "renders": renders,
    }

    (outdir / "tracker_lane_base_report_v03.txt").write_text("\n".join(report), encoding="utf-8")
    (outdir / "tracker_lane_base_metadata_v03.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("")
    print("Base   :", base_wav)
    print("Dossier:", outdir)


if __name__ == "__main__":
    main()
