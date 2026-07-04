#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BreakBrain - generate_from_placement_model_v01.py

Génère des breaks propres à partir d'un modèle de placement appris
sur tout le dataset.

Idée :
- le dataset apprend où tombent généralement kick/snare/ghost/hat/perc
- le générateur crée d'abord une grille 32 pas propre
- ensuite seulement il choisit des slices audio
- par défaut, les slices viennent d'un seul break source pour éviter le collage fouillis

Usage :
    python generator/generate_from_placement_model_v01.py --source-name "London" --variations 12
    python generator/generate_from_placement_model_v01.py --source-name "Night" --density 0.8 --syncopation 0.7
    python generator/generate_from_placement_model_v01.py --any-source
"""

from pathlib import Path
import argparse
import json
import random
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf

DATASET = Path("dataset")
EXPORTS = Path("exports")
MODEL_PATH = DATASET / "models" / "placement_model_v01.json"
SLICES_MANIFEST = DATASET / "slices_manifest.json"

SR = 44100
BPM = 150
STEPS = 32

LABELS = ["kick", "snare", "ghost", "hat", "perc"]


def normalize(y):
    m = np.max(np.abs(y)) if len(y) else 0
    return y if m <= 0 else y / m * 0.95


def load_json(path):
    if not path.exists():
        print("Fichier manquant :", path)
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def group_slices(items):
    by_break = defaultdict(list)
    for item in items:
        by_break[item["break_id"]].append(item)
    return by_break


def pick_source_break(items, source_name=None, any_source=False):
    by_break = group_slices(items)

    if any_source:
        candidates = list(by_break.keys())
    else:
        if not source_name:
            source_name = "London"

        candidates = []
        for break_id, evs in by_break.items():
            src = evs[0].get("source", "")
            if source_name.lower() in src.lower():
                candidates.append(break_id)

    if not candidates:
        print("Aucun break source trouvé.")
        print("Essaie par exemple : --source-name London")
        sys.exit(1)

    # on préfère un break avec assez de slices
    candidates = sorted(
        candidates,
        key=lambda bid: len(by_break[bid]),
        reverse=True
    )

    chosen = candidates[0]
    return chosen, by_break[chosen]


def make_pools(events):
    pools = {label: [] for label in LABELS}
    for e in events:
        label = e.get("label", "perc")
        if label not in pools:
            label = "perc"
        pools[label].append(e)
    return pools


def weighted_bool(prob):
    return random.random() < max(0.0, min(1.0, prob))


def learned_pattern(model, density=0.75, syncopation=0.55, ghost_density=0.65):
    """
    Génère d'abord une grille propre.
    Les gros kicks/snares restent stables.
    Les ghosts/hats peuvent être syncopés.
    """

    grid = {
        "kick": ["." for _ in range(STEPS)],
        "snare": ["." for _ in range(STEPS)],
        "ghost": ["." for _ in range(STEPS)],
        "hat": ["." for _ in range(STEPS)],
        "perc": ["." for _ in range(STEPS)],
    }

    probs = model["step_label_probs"]

    # 1. Snare backbeat : on force une structure lisible.
    # Sur 32 steps : 9 et 25 = backbeat classique.
    for step in [8, 24]:
        grid["snare"][step] = "S"

    # 2. Kicks appris.
    # On évite de mettre trop de kicks partout.
    kick_probs = probs.get("kick", [0] * STEPS)
    kick_candidates = sorted(
        range(STEPS),
        key=lambda i: kick_probs[i],
        reverse=True
    )

    # On garde les meilleurs placements mais avec une limite.
    max_kicks = random.choice([4, 5, 6, 7])
    kicks = 0

    for step in kick_candidates:
        if step in [8, 24]:
            continue

        p = kick_probs[step] * density

        # syncopes : on encourage les steps hors temps forts
        if step % 4 != 0:
            p *= 1.0 + syncopation * 0.8
        else:
            p *= 0.75

        if weighted_bool(p) and kicks < max_kicks:
            grid["kick"][step] = "K"
            kicks += 1

    # toujours au moins un kick au début
    if random.random() < 0.85:
        grid["kick"][0] = "K"

    # 3. Ghost snares apprises, autour des snares et dans les trous.
    ghost_probs = probs.get("ghost", [0] * STEPS)
    snare_probs = probs.get("snare", [0] * STEPS)

    ghost_candidates = list(range(STEPS))
    random.shuffle(ghost_candidates)

    for step in ghost_candidates:
        if grid["snare"][step] != ".":
            continue

        near_snare = min(abs(step - 8), abs(step - 24), abs(step - 40)) <= 4
        p = max(ghost_probs[step], snare_probs[step] * 0.45)
        p *= ghost_density

        if near_snare:
            p *= 1.45

        if step % 2 == 1:
            p *= 1.0 + syncopation * 0.8

        if weighted_bool(p):
            grid["ghost"][step] = "g"

    # 4. Hats : continuité mais moins fouillis que toutes les slices.
    hat_probs = probs.get("hat", [0] * STEPS)

    for step in range(STEPS):
        p = hat_probs[step] * density

        # croches lisibles
        if step % 2 == 0:
            p = max(p, 0.55)

        # syncopes entre les temps
        if step % 4 in [1, 3]:
            p += syncopation * 0.22

        if weighted_bool(p):
            grid["hat"][step] = "H"

    # 5. Percs rares, en décoration.
    perc_probs = probs.get("perc", [0] * STEPS)
    for step in range(STEPS):
        p = perc_probs[step] * 0.18 * density
        if step % 2 == 1:
            p *= 1.0 + syncopation
        if weighted_bool(p):
            grid["perc"][step] = "p"

    return grid


def ascii_grid(grid):
    def row(label):
        s = "".join(grid[label])
        return "|".join(s[i:i+8] for i in range(0, STEPS, 8))

    return "\n".join([
        "12345678|12345678|12345678|12345678",
        "KICK : " + row("kick"),
        "SNARE: " + row("snare"),
        "GHOST: " + row("ghost"),
        "HAT  : " + row("hat"),
        "PERC : " + row("perc"),
    ])


def choose(pools, label, fallbacks):
    if pools.get(label):
        return random.choice(pools[label])
    for fb in fallbacks:
        if pools.get(fb):
            return random.choice(pools[fb])
    return None


def read_slice(item):
    y, sr = sf.read(item["slice_file"], dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y


def render(grid, pools, out_wav):
    step_samples = int(SR * (60.0 / BPM / 4.0))
    total = step_samples * STEPS
    out = np.zeros(total + SR, dtype=np.float32)

    for step in range(STEPS):
        events = []

        if grid["kick"][step] == "K":
            s = choose(pools, "kick", ["perc", "snare", "ghost", "hat"])
            if s:
                events.append((s, 0.95, 2.4))

        if grid["snare"][step] == "S":
            s = choose(pools, "snare", ["perc", "ghost", "kick", "hat"])
            if s:
                events.append((s, 0.88, 2.2))

        if grid["ghost"][step] == "g":
            s = choose(pools, "ghost", ["snare", "perc", "hat"])
            if s:
                events.append((s, 0.28, 1.4))

        if grid["hat"][step] == "H":
            s = choose(pools, "hat", ["ghost", "perc"])
            if s:
                events.append((s, 0.26, 0.9))

        if grid["perc"][step] == "p":
            s = choose(pools, "perc", ["ghost", "hat"])
            if s:
                events.append((s, 0.22, 1.1))

        start = step * step_samples

        for item, gain, max_steps in events:
            y = read_slice(item)
            max_len = int(step_samples * max_steps)
            y = y[:max_len] * gain

            end = min(start + len(y), len(out))
            if end > start:
                out[start:end] += y[:end-start]

    sf.write(out_wav, normalize(out), SR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-name", default="London",
                    help="Nom du break dont on utilise les slices audio.")
    ap.add_argument("--any-source", action="store_true",
                    help="Choisit automatiquement un break source.")
    ap.add_argument("--variations", type=int, default=12)
    ap.add_argument("--density", type=float, default=0.75)
    ap.add_argument("--syncopation", type=float, default=0.65)
    ap.add_argument("--ghost-density", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    model = load_json(MODEL_PATH)
    slices = load_json(SLICES_MANIFEST)

    break_id, source_events = pick_source_break(
        slices,
        source_name=args.source_name,
        any_source=args.any_source
    )

    pools = make_pools(source_events)
    source = source_events[0].get("source", break_id)

    safe = Path(source).stem.replace(" ", "_").replace("'", "")
    outdir = EXPORTS / f"placement_{safe}"
    outdir.mkdir(parents=True, exist_ok=True)

    print("Modèle :", MODEL_PATH)
    print("Source audio :", source)
    print("Break ID :", break_id)
    print("Slices source :")
    for label in LABELS:
        print(f"  {label:6}: {len(pools.get(label, []))}")
    print("")

    for i in range(1, args.variations + 1):
        grid = learned_pattern(
            model,
            density=args.density,
            syncopation=args.syncopation,
            ghost_density=args.ghost_density,
        )

        wav = outdir / f"{safe}_placement_v01_{i:03d}.wav"
        txt = outdir / f"{safe}_placement_v01_{i:03d}.txt"

        render(grid, pools, wav)
        txt.write_text(ascii_grid(grid) + "\n", encoding="utf-8")

        print(f"Variation {i:03d} :", wav)
        print(ascii_grid(grid))
        print("")

    print("Terminé :", outdir)


if __name__ == "__main__":
    main()
