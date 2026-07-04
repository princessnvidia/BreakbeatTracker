#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GrooveBrain v01 - train_groove_markov_v01.py

But :
Apprendre un modèle de groove depuis les partitions extraites.

Ce n'est pas encore un Transformer.
C'est un modèle Markov + statistiques de placement :
- fréquences de symboles par step
- transitions entre symboles
- motifs de 4 pas
- motifs de 8 pas

Entrée :
    dataset/partitions/partitions_v01.json

Sortie :
    dataset/models/groovebrain_markov_v01.json
"""

from pathlib import Path
import json
import sys
from collections import Counter, defaultdict

DATASET = Path("dataset")
PARTITIONS_FILE = DATASET / "partitions" / "partitions_v01.json"
MODEL_DIR = DATASET / "models"
MODEL_FILE = MODEL_DIR / "groovebrain_markov_v01.json"

STEPS = 32
SYMBOLS = [".", "K", "S", "g", "H", "p"]


def load_partitions():
    if not PARTITIONS_FILE.exists():
        print("Fichier manquant :", PARTITIONS_FILE)
        print("Lance d'abord : python trainer/extract_partitions_v01.py")
        sys.exit(1)
    return json.loads(PARTITIONS_FILE.read_text(encoding="utf-8"))["partitions"]


def norm_counter(c):
    total = sum(c.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in c.items()}


def main():
    partitions = load_partitions()

    step_counts = [Counter() for _ in range(STEPS)]
    transition_counts = defaultdict(Counter)
    motif4_counts = Counter()
    motif8_counts = Counter()
    start_counts = Counter()

    for p in partitions:
        grid = p["grid"]
        if len(grid) != STEPS:
            continue

        start_counts[grid[0]] += 1

        for i, sym in enumerate(grid):
            step_counts[i][sym] += 1

        for a, b in zip(grid, grid[1:] + grid[:1]):
            transition_counts[a][b] += 1

        for i in range(0, STEPS, 4):
            motif4_counts[grid[i:i+4]] += 1

        for i in range(0, STEPS, 8):
            motif8_counts[grid[i:i+8]] += 1

    model = {
        "version": "groovebrain_markov_v01",
        "steps": STEPS,
        "symbols": SYMBOLS,
        "partition_count": len(partitions),
        "start_probs": norm_counter(start_counts),
        "step_probs": [norm_counter(c) for c in step_counts],
        "transition_probs": {k: norm_counter(v) for k, v in transition_counts.items()},
        "motif4_probs": norm_counter(motif4_counts),
        "motif8_probs": norm_counter(motif8_counts),
        "top_motif4": motif4_counts.most_common(50),
        "top_motif8": motif8_counts.most_common(50),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Modèle GrooveBrain créé :", MODEL_FILE)
    print("Partitions apprises :", len(partitions))
    print("")
    print("Motifs 8 pas les plus fréquents :")
    for motif, count in motif8_counts.most_common(12):
        print(f"  {motif}  x{count}")


if __name__ == "__main__":
    main()
