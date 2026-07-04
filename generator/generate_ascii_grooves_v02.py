#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GrooveBrain v02 - generate_ascii_grooves_v02.py

But :
Générer des grooves ASCII propres à partir du modèle v02.

Sortie :
    exports/groovebrain_ascii_v02/generated_ascii_grooves_v02.txt
    exports/groovebrain_ascii_v02/generated_ascii_grooves_v02.json

Cette version force une base solide :
    K.S.|.KS.|K.S.|.KS.

Puis ajoute :
    ghosts
    hats
    kicks syncopés
    variations contrôlées
"""

from pathlib import Path
import argparse
import json
import random
import sys

DATASET = Path("dataset")
MODEL_FILE = DATASET / "models" / "groovebrain_ascii_v02.json"
OUT_DIR = Path("exports/groovebrain_ascii_v02")
OUT_TXT = OUT_DIR / "generated_ascii_grooves_v02.txt"
OUT_JSON = OUT_DIR / "generated_ascii_grooves_v02.json"

STEPS = 32
SYMBOLS = [".", "K", "S", "g", "H", "p"]

BASE_GRID = "K.S..KS.K.S..KS.K.S..KS.K.S..KS."


def load_model():
    if not MODEL_FILE.exists():
        print("Fichier manquant :", MODEL_FILE)
        print("Lance : python trainer/train_from_ascii_v02.py")
        sys.exit(1)
    return json.loads(MODEL_FILE.read_text(encoding="utf-8"))


def weighted_choice(probs, fallback="."):
    if not probs:
        return fallback

    items = [(k, float(v)) for k, v in probs.items() if float(v) > 0]
    if not items:
        return fallback

    total = sum(v for _, v in items)
    r = random.random() * total
    acc = 0

    for k, v in items:
        acc += v
        if r <= acc:
            return k

    return items[-1][0]


def split32(row):
    return "|".join(row[i:i+8] for i in range(0, STEPS, 8))


def grid_to_layers(grid):
    layers = {
        "kick": ["."] * STEPS,
        "snare": ["."] * STEPS,
        "ghost": ["."] * STEPS,
        "hat": ["."] * STEPS,
        "perc": ["."] * STEPS,
    }

    for i, s in enumerate(grid):
        if s == "K":
            layers["kick"][i] = "K"
        elif s == "S":
            layers["snare"][i] = "S"
        elif s == "g":
            layers["ghost"][i] = "g"
        elif s == "H":
            layers["hat"][i] = "H"
        elif s == "p":
            layers["perc"][i] = "p"

    return {k: "".join(v) for k, v in layers.items()}


def generate_from_model(model, base_strength=0.75, syncopation=0.55, ghost_density=0.55, hat_density=0.55):
    """
    On part de la base, puis on remplace certains steps
    par des probabilités apprises.

    base_strength proche de 1 = reste très proche de K.S.|.KS.
    base_strength plus bas = plus libre.
    """

    grid = list(BASE_GRID)

    step_probs = model["step_probs"]
    transition_probs = model["transition_probs"]

    prev = grid[0]

    for i in range(STEPS):
        # On protège les snares/kicks structurants en partie.
        protected = BASE_GRID[i] in ["K", "S"]

        if protected and random.random() < base_strength:
            prev = grid[i]
            continue

        p_change = (1.0 - base_strength)

        if BASE_GRID[i] == ".":
            p_change += 0.20

        if i % 4 not in [0]:
            p_change += syncopation * 0.18

        if random.random() < p_change:
            local = dict(step_probs[i])
            trans = transition_probs.get(prev, {})

            mixed = {}
            for s in SYMBOLS:
                mixed[s] = local.get(s, 0.0) * 0.70 + trans.get(s, 0.0) * 0.30

            # contrôle esthétique
            if i % 4 not in [0]:
                mixed["g"] = mixed.get("g", 0.0) * (1.0 + syncopation)
                mixed["K"] = mixed.get("K", 0.0) * (1.0 + syncopation * 0.45)

            mixed["H"] = mixed.get("H", 0.0) * hat_density
            mixed["g"] = mixed.get("g", 0.0) * ghost_density

            # évite trop de percs
            mixed["p"] = mixed.get("p", 0.0) * 0.35

            new = weighted_choice(mixed, fallback=grid[i])
            grid[i] = new

        prev = grid[i]

    return cleanup_grid("".join(grid), syncopation, ghost_density, hat_density)


def cleanup_grid(grid, syncopation, ghost_density, hat_density):
    grid = list(grid[:STEPS].ljust(STEPS, "."))

    # Garde la base snare/kick demandée lisible.
    # Sur K.S.|.KS. répété :
    # K positions 0,5,8,13,16,21,24,29
    # S positions 2,6,10,14,18,22,26,30
    # On évite que tout soit supprimé.
    base_kicks = [0, 5, 8, 13, 16, 21, 24, 29]
    base_snares = [2, 6, 10, 14, 18, 22, 26, 30]

    for i in base_kicks:
        if random.random() < 0.55:
            grid[i] = "K"

    for i in base_snares:
        if random.random() < 0.55:
            grid[i] = "S"

    # Limite les kicks.
    kicks = [i for i, s in enumerate(grid) if s == "K"]
    max_kicks = random.choice([5, 6, 7, 8])
    if len(kicks) > max_kicks:
        random.shuffle(kicks)
        for i in kicks[max_kicks:]:
            grid[i] = "."

    # Limite les snares fortes, sinon ça mitraille.
    snares = [i for i, s in enumerate(grid) if s == "S"]
    max_snares = random.choice([4, 5, 6, 7, 8])
    if len(snares) > max_snares:
        random.shuffle(snares)
        for i in snares[max_snares:]:
            grid[i] = "g"

    # Ajoute ghosts syncopés dans les trous.
    ghost_spots = [1, 3, 4, 7, 9, 11, 12, 15, 17, 19, 20, 23, 25, 27, 28, 31]
    for i in ghost_spots:
        if grid[i] == "." and random.random() < ghost_density * 0.35:
            grid[i] = "g"

    # Hats comme couche légère uniquement sur les trous.
    for i in range(STEPS):
        if grid[i] == ".":
            if i % 2 == 0 and random.random() < hat_density * 0.45:
                grid[i] = "H"
            elif i % 2 == 1 and random.random() < hat_density * syncopation * 0.22:
                grid[i] = "H"

    # Percs rares.
    for i, s in enumerate(grid):
        if s == "p" and random.random() < 0.70:
            grid[i] = "."

    return "".join(grid)


def render_text(grooves):
    lines = []

    for i, g in enumerate(grooves, start=1):
        layers = grid_to_layers(g)

        lines.append(f"VARIATION {i:03d}")
        lines.append("12345678|12345678|12345678|12345678")
        lines.append("KICK : " + split32(layers["kick"]))
        lines.append("SNARE: " + split32(layers["snare"]))
        lines.append("GHOST: " + split32(layers["ghost"]))
        lines.append("HAT  : " + split32(layers["hat"]))
        lines.append("PERC : " + split32(layers["perc"]))
        lines.append("FULL : " + split32(g))
        lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variations", type=int, default=24)
    ap.add_argument("--base-strength", type=float, default=0.78)
    ap.add_argument("--syncopation", type=float, default=0.60)
    ap.add_argument("--ghost-density", type=float, default=0.55)
    ap.add_argument("--hat-density", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    model = load_model()

    grooves = []
    for _ in range(args.variations):
        g = generate_from_model(
            model,
            base_strength=args.base_strength,
            syncopation=args.syncopation,
            ghost_density=args.ghost_density,
            hat_density=args.hat_density,
        )
        grooves.append(g)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_TXT.write_text(render_text(grooves), encoding="utf-8")

    payload = {
        "version": "generated_ascii_grooves_v02",
        "steps": STEPS,
        "base_grid": BASE_GRID,
        "grooves": [
            {
                "index": i,
                "full": g,
                "layers": grid_to_layers(g),
            }
            for i, g in enumerate(grooves, start=1)
        ],
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Grooves générés :")
    print(OUT_TXT)
    print(OUT_JSON)
    print("")
    print(render_text(grooves[:5]))


if __name__ == "__main__":
    main()
