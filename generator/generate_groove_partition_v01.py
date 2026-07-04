#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GrooveBrain v01 - generate_groove_partition_v01.py

But :
Générer une PARTITION de breakbeat propre depuis le modèle appris.

Cette version ne génère pas encore l'audio.
Elle génère une grille 32 pas musicale :
    K = kick
    S = snare
    g = ghost snare
    H = hat
    p = perc
    . = vide

Usage :
    python generator/generate_groove_partition_v01.py
    python generator/generate_groove_partition_v01.py --variations 16 --syncopation 0.8
"""

from pathlib import Path
import argparse
import json
import random
import sys

DATASET = Path("dataset")
MODEL_FILE = DATASET / "models" / "groovebrain_markov_v01.json"

STEPS = 32
SYMBOLS = [".", "K", "S", "g", "H", "p"]


def load_model():
    if not MODEL_FILE.exists():
        print("Fichier manquant :", MODEL_FILE)
        print("Lance :")
        print("  python trainer/extract_partitions_v01.py")
        print("  python trainer/train_groove_markov_v01.py")
        sys.exit(1)
    return json.loads(MODEL_FILE.read_text(encoding="utf-8"))


def weighted_choice(probs, fallback="."):
    if not probs:
        return fallback
    items = list(probs.items())
    total = sum(float(v) for _, v in items)
    if total <= 0:
        return fallback
    r = random.random() * total
    acc = 0.0
    for k, v in items:
        acc += float(v)
        if r <= acc:
            return k
    return items[-1][0]


def reinforce_breakbeat_rules(grid, syncopation=0.65, density=0.70, ghost_density=0.65):
    """
    Nettoie la grille pour éviter le fouillis :
    - snares lisibles sur 9 et 25
    - kicks limités
    - ghosts autour des snares
    - hats pas trop envahissants
    """

    grid = list(grid)

    # Snare backbeat stable
    grid[8] = "S"
    grid[24] = "S"

    # Évite les snares fortes aléatoires trop nombreuses
    for i, s in enumerate(grid):
        if s == "S" and i not in [8, 24]:
            if random.random() < 0.70:
                grid[i] = "g"

    # Limite le nombre de kicks
    kick_positions = [i for i, s in enumerate(grid) if s == "K"]
    max_kicks = random.choice([4, 5, 6, 7])

    if len(kick_positions) > max_kicks:
        random.shuffle(kick_positions)
        for i in kick_positions[max_kicks:]:
            grid[i] = "."

    # Kick de départ fréquent
    if random.random() < 0.85:
        grid[0] = "K"

    # Ajout de kicks syncopés contrôlés
    sync_kicks = [3, 6, 10, 11, 14, 18, 20, 22, 27, 29]
    for i in sync_kicks:
        if grid[i] == "." and random.random() < 0.10 + syncopation * 0.12:
            if sum(1 for x in grid if x == "K") < max_kicks:
                grid[i] = "K"

    # Ghosts autour des snares
    ghost_spots = [5, 6, 7, 11, 12, 13, 21, 22, 23, 26, 27, 28, 30, 31]
    for i in ghost_spots:
        if grid[i] == "." and random.random() < ghost_density * 0.45:
            grid[i] = "g"

    # Hats : si vide sur croches, on peut remplir légèrement
    for i in range(STEPS):
        if grid[i] == ".":
            if i % 2 == 0 and random.random() < density * 0.45:
                grid[i] = "H"
            elif i % 2 == 1 and random.random() < density * syncopation * 0.20:
                grid[i] = "H"

    # Évite trop de percs
    for i, s in enumerate(grid):
        if s == "p" and random.random() < 0.65:
            grid[i] = "."

    return "".join(grid)


def generate_grid(model, syncopation=0.65, density=0.70, ghost_density=0.65):
    grid = []

    current = weighted_choice(model.get("start_probs", {}), fallback="K")

    for step in range(STEPS):
        step_probs = model["step_probs"][step]
        trans_probs = model["transition_probs"].get(current, {})

        mixed = {}

        for s in SYMBOLS:
            a = step_probs.get(s, 0.0)
            b = trans_probs.get(s, 0.0)
            mixed[s] = a * 0.65 + b * 0.35

        # Encourage les syncopes sur les off-steps
        if step % 4 not in [0]:
            mixed["g"] = mixed.get("g", 0) * (1 + syncopation * 0.7)
            mixed["K"] = mixed.get("K", 0) * (1 + syncopation * 0.25)

        # Densité globale
        if density < 1.0:
            mixed["."] = mixed.get(".", 0) + (1.0 - density) * 0.35

        current = weighted_choice(mixed, fallback=".")
        grid.append(current)

    grid = "".join(grid)
    return reinforce_breakbeat_rules(
        grid,
        syncopation=syncopation,
        density=density,
        ghost_density=ghost_density,
    )


def split_grid(grid):
    return "|".join(grid[i:i+8] for i in range(0, STEPS, 8))


def grid_to_layers(grid):
    layers = {
        "KICK ": "",
        "SNARE": "",
        "GHOST": "",
        "HAT  ": "",
        "PERC ": "",
    }

    for s in grid:
        layers["KICK "] += "K" if s == "K" else "."
        layers["SNARE"] += "S" if s == "S" else "."
        layers["GHOST"] += "g" if s == "g" else "."
        layers["HAT  "] += "H" if s == "H" else "."
        layers["PERC "] += "p" if s == "p" else "."

    return layers


def print_grid(grid, idx):
    print(f"VARIATION {idx:03d}")
    print("12345678|12345678|12345678|12345678")
    layers = grid_to_layers(grid)
    for name, row in layers.items():
        print(f"{name}: {split_grid(row)}")
    print("FULL : " + split_grid(grid))
    print("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variations", type=int, default=12)
    ap.add_argument("--syncopation", type=float, default=0.70)
    ap.add_argument("--density", type=float, default=0.70)
    ap.add_argument("--ghost-density", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    model = load_model()

    outdir = Path("exports/groovebrain_partitions")
    outdir.mkdir(parents=True, exist_ok=True)

    all_text = []

    for i in range(1, args.variations + 1):
        grid = generate_grid(
            model,
            syncopation=args.syncopation,
            density=args.density,
            ghost_density=args.ghost_density,
        )

        print_grid(grid, i)

        text = []
        text.append(f"VARIATION {i:03d}")
        text.append("12345678|12345678|12345678|12345678")
        for name, row in grid_to_layers(grid).items():
            text.append(f"{name}: {split_grid(row)}")
        text.append("FULL : " + split_grid(grid))
        text.append("")
        all_text.extend(text)

    outfile = outdir / "generated_partitions_v01.txt"
    outfile.write_text("\n".join(all_text), encoding="utf-8")
    print("Partitions exportées :", outfile)


if __name__ == "__main__":
    main()
