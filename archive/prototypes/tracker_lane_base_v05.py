#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tracker Lane Base v05

Objectif :
Corriger le problème des découpes pas au bon endroit.

Changement principal :
- ajout d'un offset de départ réglable : --offset-ms
- ajout d'une commande de preview des cellules : --preview
- exporte chaque cellule découpée pour vérifier si les cuts tombent juste
- garde la logique tracker :
    une cellule = une slice
    pas de superposition
    un seul break source

Usage preview :
    python tracker_lane_base_v05.py --source "Camo" --preview --offset-ms 0 --cell-ms 200
    python tracker_lane_base_v05.py --source "Camo" --preview --offset-ms 35 --cell-ms 200
    python tracker_lane_base_v05.py --source "Camo" --preview --offset-ms 70 --cell-ms 200

Usage rendu :
    python tracker_lane_base_v05.py --source "Camo" --offset-ms 35 --cell-ms 200 --count 32

Base :
    K..S|..KS|K..S|..KS
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
OUT_DIR = Path("exports/tracker_lane_base_v05")

SR = 44100
AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".mp3"}
BASE_16 = "K..S..KSK..S..KS"


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


def load_audio(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    return normalize(y), sr


def fade(y, ms=1.5):
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
        return {"rms": 0, "peak": 0, "centroid": 0, "low": 0, "mid": 0, "high": 0}

    rms = float(np.sqrt(np.mean(y * y)))
    peak = float(np.max(np.abs(y)))
    centroid = safe_centroid(y)

    yn = normalize(y, peak=1.0)
    spec = np.abs(np.fft.rfft(yn))
    freqs = np.fft.rfftfreq(len(yn), 1.0 / SR)
    total = np.sum(spec) + 1e-9

    low = float(np.sum(spec[(freqs >= 20) & (freqs < 250)]) / total)
    mid = float(np.sum(spec[(freqs >= 250) & (freqs < 3000)]) / total)
    high = float(np.sum(spec[(freqs >= 3000) & (freqs < 12000)]) / total)

    return {"rms": rms, "peak": peak, "centroid": centroid, "low": low, "mid": mid, "high": high}


def make_cells(y, cell_samples, offset_samples):
    cells = []
    start = offset_samples
    idx = 0

    while start < len(y):
        end = min(start + cell_samples, len(y))
        chunk = y[start:end].copy()

        if len(chunk) < int(cell_samples * 0.40):
            break

        if len(chunk) < cell_samples:
            chunk = np.concatenate([chunk, np.zeros(cell_samples - len(chunk), dtype=np.float32)])

        chunk = fade(chunk, ms=1.5)

        cells.append({
            "index": idx,
            "start": int(start),
            "end": int(end),
            "start_ms": float(start / SR * 1000),
            "end_ms": float(end / SR * 1000),
            "audio": chunk,
            "features": cell_features(chunk),
        })

        idx += 1
        start += cell_samples

    return cells


def score_kick(c):
    f = c["features"]
    return f["low"] * 3.2 + f["rms"] * 4.0 + f["peak"] * 1.8 + max(0, 3000 - f["centroid"]) / 3000 - f["high"] * 0.7


def score_snare(c):
    f = c["features"]
    ok = 1.0 if 700 <= f["centroid"] <= 6500 else 0.0
    return f["mid"] * 2.0 + f["rms"] * 4.0 + f["peak"] * 1.4 + f["high"] * 0.45 + ok


def score_hat(c):
    f = c["features"]
    return f["high"] * 4.0 + f["centroid"] / 8000.0 + f["rms"] * 2.0 - f["low"] * 0.7


def choose_kit(cells):
    if len(cells) < 4:
        print("Pas assez de cellules.")
        sys.exit(1)

    kick = max(cells, key=score_kick)

    snares = sorted(cells, key=score_snare, reverse=True)
    snare = next((c for c in snares if c["index"] != kick["index"]), snares[0])

    used = {kick["index"], snare["index"]}

    hats = sorted(cells, key=score_hat, reverse=True)
    hat = next((c for c in hats if c["index"] not in used), hats[0])

    ghosts = sorted(
        [c for c in snares if c["index"] not in used and c["index"] != hat["index"]],
        key=lambda c: c["features"]["rms"]
    )
    ghost = ghosts[0] if ghosts else snare

    return {"kick": kick, "snare": snare, "ghost": ghost, "hat": hat}


def make_base(length):
    reps = (length + len(BASE_16) - 1) // len(BASE_16)
    return (BASE_16 * reps)[:length]


def mutate_base(length, mutation=0.20, hat_density=0.92):
    grid = list(make_base(length))

    for i in range(length):
        if grid[i] == "." and random.random() < hat_density:
            grid[i] = "H"

    snares = [i for i, ch in enumerate(grid) if ch == "S"]
    for s in snares:
        for off in [-1, 1]:
            i = s + off
            if 0 <= i < length and grid[i] == "H" and random.random() < mutation * 0.45:
                grid[i] = "g"

    for i in [1, 5, 9, 13, 17, 21, 25, 29]:
        if i < length and grid[i] == "H" and random.random() < mutation * 0.15:
            grid[i] = "K"

    return "".join(grid)


def split_grid(grid):
    return "|".join(grid[i:i+4] for i in range(0, len(grid), 4))


def layers_text(grid):
    rows = {"HAT  ": "", "KICK ": "", "SNARE": "", "GHOST": ""}
    for ch in grid:
        rows["HAT  "] += "H" if ch == "H" else "."
        rows["KICK "] += "K" if ch == "K" else "."
        rows["SNARE"] += "S" if ch == "S" else "."
        rows["GHOST"] += "g" if ch == "g" else "."
    lines = ["GROUP: " + "|".join(["1234"] * (len(grid) // 4))]
    lines.append("BASE : K..S|..KS|K..S|..KS")
    for k, v in rows.items():
        lines.append(f"{k}: {split_grid(v)}")
    lines.append("FULL : " + split_grid(grid))
    return "\n".join(lines)


def render(grid, kit, cell_samples, hat_boost):
    chunks = []
    role_for = {"K": "kick", "S": "snare", "g": "ghost", "H": "hat", ".": None}
    gain_for = {"K": 0.95, "S": 0.88, "g": 0.30, "H": hat_boost}

    for ch in grid:
        role = role_for.get(ch)
        if role is None:
            chunks.append(np.zeros(cell_samples, dtype=np.float32))
            continue

        audio = kit[role]["audio"].copy()

        if len(audio) > cell_samples:
            audio = audio[:cell_samples]
        elif len(audio) < cell_samples:
            audio = np.concatenate([audio, np.zeros(cell_samples - len(audio), dtype=np.float32)])

        audio = fade(audio, ms=1.5)
        chunks.append(audio * gain_for[ch])

    return normalize(np.concatenate(chunks))


def export_preview(cells, outdir, preview_count):
    preview_dir = outdir / "preview_cells"
    preview_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.append("CELL | START_MS | END_MS | RMS | CENTROID | LOW | HIGH")
    rows.append("------------------------------------------------------")

    for c in cells[:preview_count]:
        wav = preview_dir / f"cell_{c['index']:03d}_{int(c['start_ms'])}ms.wav"
        sf.write(wav, normalize(c["audio"]), SR)

        f = c["features"]
        rows.append(
            f"{c['index']:04d} | {c['start_ms']:8.1f} | {c['end_ms']:7.1f} | "
            f"{f['rms']:.4f} | {f['centroid']:.0f} | {f['low']:.2f} | {f['high']:.2f}"
        )

    (preview_dir / "preview_cells_report.txt").write_text("\n".join(rows), encoding="utf-8")
    print("Preview exportée :", preview_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Camo")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--bpm", type=float, default=150.0)
    parser.add_argument("--cell-ms", type=float, default=200.0)
    parser.add_argument("--offset-ms", type=float, default=0.0)
    parser.add_argument("--mutation", type=float, default=0.20)
    parser.add_argument("--hat-density", type=float, default=0.92)
    parser.add_argument("--hat-boost", type=float, default=0.55)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    src = find_source(args.source)
    print("Source unique :", src)

    y, sr = load_audio(src)

    cell_samples = int(SR * args.cell_ms / 1000.0)
    offset_samples = int(SR * args.offset_ms / 1000.0)

    print(f"Cellule tracker : {args.cell_ms:.1f} ms")
    print(f"Offset départ   : {args.offset_ms:.1f} ms")

    cells = make_cells(y, cell_samples, offset_samples)
    print("Macro-cells :", len(cells))

    safe = src.stem.replace(" ", "_").replace("'", "")
    outdir = OUT_DIR / safe / f"offset_{int(args.offset_ms)}ms_cell_{int(args.cell_ms)}ms"
    outdir.mkdir(parents=True, exist_ok=True)

    if args.preview:
        export_preview(cells, outdir, args.preview_count)
        print("")
        print("Écoute les fichiers dans preview_cells.")
        print("Quand cell_000 tombe sur le bon début, relance sans --preview.")
        return

    kit = choose_kit(cells)

    print("")
    print("Kit choisi :")
    for role, c in kit.items():
        f = c["features"]
        print(
            f"  {role:6}: cell={c['index']:03d} start={c['start_ms']:.1f}ms "
            f"rms={f['rms']:.4f} centroid={f['centroid']:.0f} low={f['low']:.2f} high={f['high']:.2f}"
        )

    report = []

    base = make_base(args.steps)
    base_audio = render(base, kit, cell_samples, args.hat_boost)
    base_wav = outdir / f"{safe}_lane_base_v05_base.wav"
    sf.write(base_wav, base_audio, SR)

    report.append(f"SOURCE: {src}")
    report.append(f"CELL_MS: {args.cell_ms}")
    report.append(f"OFFSET_MS: {args.offset_ms}")
    report.append(f"HAT_BOOST: {args.hat_boost}")
    report.append("")
    report.append("BASE")
    report.append(str(base_wav))
    report.append(layers_text(base))
    report.append("")

    renders = []

    for i in range(1, args.count + 1):
        grid = mutate_base(args.steps, args.mutation, args.hat_density)
        audio = render(grid, kit, cell_samples, args.hat_boost)

        wav = outdir / f"{safe}_lane_base_v05_{i:03d}.wav"
        sf.write(wav, audio, SR)

        renders.append({"index": i, "wav": str(wav), "grid": grid})

        report.append(f"VARIATION {i:03d} -> {wav}")
        report.append(layers_text(grid))
        report.append("")

        print("Export :", wav)

    metadata = {
        "version": "tracker_lane_base_v05",
        "source": str(src),
        "source_only": True,
        "logic": "tracker cells with adjustable offset, no overlap",
        "base": "K..S|..KS|K..S|..KS",
        "cell_ms": args.cell_ms,
        "offset_ms": args.offset_ms,
        "steps": args.steps,
        "mutation": args.mutation,
        "hat_density": args.hat_density,
        "hat_boost": args.hat_boost,
        "base_wav": str(base_wav),
        "kit": {
            role: {
                "cell_index": c["index"],
                "start_ms": c["start_ms"],
                "end_ms": c["end_ms"],
                "features": c["features"],
            }
            for role, c in kit.items()
        },
        "renders": renders,
    }

    (outdir / "tracker_lane_base_report_v05.txt").write_text("\n".join(report), encoding="utf-8")
    (outdir / "tracker_lane_base_metadata_v05.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("")
    print("Base   :", base_wav)
    print("Dossier:", outdir)


if __name__ == "__main__":
    main()
