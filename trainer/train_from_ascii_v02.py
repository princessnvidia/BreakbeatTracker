#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GrooveBrain v02 - train_from_ascii_v02.py

But :
Apprendre une grammaire breakbeat à partir de :
1. Patterns fondamentaux écrits à la main
2. Transcriptions ASCII de tes breaks

Entrée :
    dataset/ascii_transcriptions/ascii_transcriptions_v01.json

Sortie :
    dataset/models/groovebrain_ascii_v02.json

Idée :
On apprend d'abord la base :
    K.S.|.KS.|K.S.|.KS.

Puis on ajoute progressivement :
    ghost notes
    hats
    variations syncopées
    patterns issus de tes 200 breaks
"""

from pathlib import Path
import json
import sys
from collections import Counter, defaultdict

DATASET = Path("dataset")
ASCII_JSON = DATASET / "ascii_transcriptions" / "ascii_transcriptions_v01.json"
MODEL_DIR = DATASET / "models"
MODEL_FILE = MODEL_DIR / "groovebrain_ascii_v02.json"

STEPS = 32
SYMBOLS = [".", "K", "S", "g", "H", "p"]

# Patterns fondamentaux, volontairement répétés/pondérés.
# Format full : 32 caractères.
# K = kick, S = snare, g = ghost, H = hat, p = perc, . = silence
CORE_PATTERNS = [
    # Base demandée : K.S.|.KS.|K.S.|.KS.
    "K.S..KS.K.S..KS.K.S..KS.K.S..KS.",
    "K.S..KS.K.S..KS.K.S..KS.K.S..KS.",
    "K.S..KS.K.S..KS.K.S..KS.K.S..KS.",
    "K.S..KS.K.S..KS.K.S..KS.K.S..KS.",
    "K.S..KS.K.S..KS.K.S..KS.K.S..KS.",

    # Backbeat simple.
    "K.......S.......K.......S.......",
    "K.....K.S.......K...K...S.......",
    "K.......S...K...K.......S...K...",

    # Breakbeat simple avec ghost.
    "K.g..KS.K.g..KS.K.g..KS.K.g..KS.",
    "K...g...S..g....K...g...S..g....",
    "K..g....S.g.....K..g....S.g.....",

    # Trip-hop / downtempo.
    "K.....K.S.......K...K...S.......",
    "K.......S..g....K.....K.S.g.....",
    "K..g....S.......K.g.....S..g....",

    # Roller léger.
    "K...K...S.g.....K...K...S.g.g...",
    "K.g.K...S...g...K.g.K...S...g...",

    # Plus syncopé.
    "K..gK...S.g.....K.g.K...S...g...",
    "K.g...K.S..g....K...g.K.S.g.....",
]


def clean_grid(grid):
    grid = "".join(ch for ch in grid if ch in SYMBOLS)
    if len(grid) < STEPS:
        grid = grid + "." * (STEPS - len(grid))
    if len(grid) > STEPS:
        grid = grid[:STEPS]
    return grid


def load_transcriptions():
    if not ASCII_JSON.exists():
        print("Fichier manquant :", ASCII_JSON)
        print("Lance d'abord : python trainer/transcribe_breaks_to_ascii_v01.py")
        sys.exit(1)

    data = json.loads(ASCII_JSON.read_text(encoding="utf-8"))
    rows = []

    for t in data.get("transcriptions", []):
        full = clean_grid(t.get("full", "." * STEPS))
        layers = t.get("layers", {})
        rows.append({
            "name": t.get("name", "unknown"),
            "source": t.get("source", ""),
            "full": full,
            "layers": {
                "kick": clean_grid(layers.get("kick", "." * STEPS)),
                "snare": clean_grid(layers.get("snare", "." * STEPS)),
                "ghost": clean_grid(layers.get("ghost", "." * STEPS)),
                "hat": clean_grid(layers.get("hat", "." * STEPS)),
                "perc": clean_grid(layers.get("perc", "." * STEPS)),
            },
        })

    return rows


def norm_counter(counter):
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def add_grid_to_stats(grid, weight, stats):
    step_counts = stats["step_counts"]
    transition_counts = stats["transition_counts"]
    motif4_counts = stats["motif4_counts"]
    motif8_counts = stats["motif8_counts"]
    start_counts = stats["start_counts"]

    grid = clean_grid(grid)

    start_counts[grid[0]] += weight

    for i, sym in enumerate(grid):
        step_counts[i][sym] += weight

    for a, b in zip(grid, grid[1:] + grid[:1]):
        transition_counts[a][b] += weight

    for i in range(0, STEPS, 4):
        motif4_counts[grid[i:i+4]] += weight

    for i in range(0, STEPS, 8):
        motif8_counts[grid[i:i+8]] += weight


def main():
    transcriptions = load_transcriptions()

    stats = {
        "step_counts": [Counter() for _ in range(STEPS)],
        "transition_counts": defaultdict(Counter),
        "motif4_counts": Counter(),
        "motif8_counts": Counter(),
        "start_counts": Counter(),
    }

    # 1. Patterns fondamentaux très pondérés.
    for grid in CORE_PATTERNS:
        add_grid_to_stats(grid, weight=12, stats=stats)

    # 2. Transcriptions de tes breaks, moins pondérées pour ne pas casser la base.
    for t in transcriptions:
        add_grid_to_stats(t["full"], weight=2, stats=stats)

        # On ajoute aussi les couches instrumentales avec poids faible.
        # Ça aide à apprendre les placements séparément.
        for layer_name, layer_grid in t["layers"].items():
            add_grid_to_stats(layer_grid, weight=1, stats=stats)

    model = {
        "version": "groovebrain_ascii_v02",
        "steps": STEPS,
        "symbols": SYMBOLS,
        "core_patterns": CORE_PATTERNS,
        "transcription_count": len(transcriptions),
        "start_probs": norm_counter(stats["start_counts"]),
        "step_probs": [norm_counter(c) for c in stats["step_counts"]],
        "transition_probs": {
            k: norm_counter(v)
            for k, v in stats["transition_counts"].items()
        },
        "motif4_probs": norm_counter(stats["motif4_counts"]),
        "motif8_probs": norm_counter(stats["motif8_counts"]),
        "top_motif4": stats["motif4_counts"].most_common(80),
        "top_motif8": stats["motif8_counts"].most_common(80),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Modèle créé :", MODEL_FILE)
    print("Transcriptions apprises :", len(transcriptions))
    print("")
    print("Top motifs 8 pas :")
    for motif, count in stats["motif8_counts"].most_common(16):
        print(f"  {motif} x{count}")


if __name__ == "__main__":
    main()
