#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Groove Critic v01 - score_grooves_v01.py

But :
Noter automatiquement les grooves ASCII générés par GrooveGPT/GrooveBrain.

Entrées possibles :
    exports/groovegpt_v01/groovegpt_generated_v01.json
    exports/groovebrain_ascii_v02/generated_ascii_grooves_v02.json

Sorties :
    exports/groovecritic_v01/scored_grooves_v01.json
    exports/groovecritic_v01/scored_grooves_v01.txt
    exports/groovecritic_v01/best_grooves_v01.json
    exports/groovecritic_v01/best_grooves_v01.txt

Notation :
    - backbeat
    - kick groove
    - ghost notes
    - hats
    - syncopation
    - anti-fouillis
    - structure de base K.S.|.KS.|K.S.|.KS.

Usage :
    python critic/score_grooves_v01.py

    python critic/score_grooves_v01.py \
        --input exports/groovegpt_v01/groovegpt_generated_v01.json \
        --top 50
"""

from pathlib import Path
import argparse
import json
import math
import sys

DEFAULT_INPUTS = [
    Path("exports/groovegpt_v01/groovegpt_generated_v01.json"),
    Path("exports/groovebrain_ascii_v02/generated_ascii_grooves_v02.json"),
]

OUT_DIR = Path("exports/groovecritic_v01")
SCORED_JSON = OUT_DIR / "scored_grooves_v01.json"
SCORED_TXT = OUT_DIR / "scored_grooves_v01.txt"
BEST_JSON = OUT_DIR / "best_grooves_v01.json"
BEST_TXT = OUT_DIR / "best_grooves_v01.txt"

STEPS = 32
ALLOWED = set(".KSgHp")

# Base demandée par Vio :
# K.S.|.KS.|K.S.|.KS. répétée sur 32 pas.
BASE_GRID = "K.S..KS.K.S..KS.K.S..KS.K.S..KS."

BASE_KICKS = [i for i, ch in enumerate(BASE_GRID) if ch == "K"]
BASE_SNARES = [i for i, ch in enumerate(BASE_GRID) if ch == "S"]

# Snares classiques en backbeat sur grille 32.
# On accepte aussi la base K.S.|.KS. qui a plus de snares.
BACKBEAT_SNARES = [8, 24]


def clean_grid(grid):
    grid = "".join(ch for ch in str(grid) if ch in ALLOWED)
    grid = grid[:STEPS].ljust(STEPS, ".")
    return grid


def split32(row):
    row = clean_grid(row)
    return "|".join(row[i:i+8] for i in range(0, STEPS, 8))


def grid_to_layers(grid):
    grid = clean_grid(grid)

    layers = {
        "kick": ["."] * STEPS,
        "snare": ["."] * STEPS,
        "ghost": ["."] * STEPS,
        "hat": ["."] * STEPS,
        "perc": ["."] * STEPS,
    }

    for i, ch in enumerate(grid):
        if ch == "K":
            layers["kick"][i] = "K"
        elif ch == "S":
            layers["snare"][i] = "S"
        elif ch == "g":
            layers["ghost"][i] = "g"
        elif ch == "H":
            layers["hat"][i] = "H"
        elif ch == "p":
            layers["perc"][i] = "p"

    return {k: "".join(v) for k, v in layers.items()}


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, float(x)))


def closeness_to_targets(positions, targets, radius=1):
    """
    Score : est-ce que les coups sont proches des positions attendues ?
    """
    if not targets:
        return 0.0

    hits = 0

    for target in targets:
        ok = False
        for p in positions:
            if abs(p - target) <= radius:
                ok = True
                break
        if ok:
            hits += 1

    return hits / len(targets)


def count_runs(grid, symbol):
    """
    Compte les longues répétitions du même symbole.
    Utile pour détecter les mitraillettes.
    """
    longest = 0
    current = 0

    for ch in grid:
        if ch == symbol:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def score_backbeat(grid):
    snares = [i for i, ch in enumerate(grid) if ch == "S"]

    if not snares:
        return 0.0, {
            "snare_count": 0,
            "classic_backbeat": 0.0,
            "base_snare_match": 0.0,
        }

    classic = closeness_to_targets(snares, BACKBEAT_SNARES, radius=1)
    base_match = closeness_to_targets(snares, BASE_SNARES, radius=0)

    # On accepte deux philosophies :
    # - backbeat classique peu chargé
    # - base K.S.|.KS. plus dense
    score = max(classic, base_match * 0.90)

    snare_count = len(snares)

    if snare_count < 2:
        score *= 0.45
    elif snare_count > 10:
        score *= 0.55
    elif snare_count > 8:
        score *= 0.75

    return clamp(score), {
        "snare_count": snare_count,
        "classic_backbeat": round(classic, 4),
        "base_snare_match": round(base_match, 4),
    }


def score_kicks(grid):
    kicks = [i for i, ch in enumerate(grid) if ch == "K"]
    kick_count = len(kicks)

    if kick_count == 0:
        return 0.0, {"kick_count": 0, "base_kick_match": 0.0}

    base_match = closeness_to_targets(kicks, BASE_KICKS, radius=0)

    # quantité idéale approximative
    if 4 <= kick_count <= 8:
        amount = 1.0
    elif 3 <= kick_count <= 10:
        amount = 0.75
    else:
        amount = 0.45

    # bonus si kick au début
    start_bonus = 1.0 if grid[0] == "K" else 0.80

    # bonus syncopation : kicks hors gros temps
    off_kicks = sum(1 for i in kicks if i % 4 not in [0])
    off_ratio = off_kicks / max(1, kick_count)
    sync_bonus = 0.65 + off_ratio * 0.55

    score = amount * start_bonus * sync_bonus

    # On veut garder la base comme référence, mais pas forcer 100%.
    score = score * 0.70 + base_match * 0.30

    return clamp(score), {
        "kick_count": kick_count,
        "base_kick_match": round(base_match, 4),
        "off_kick_ratio": round(off_ratio, 4),
    }


def score_ghosts(grid):
    ghosts = [i for i, ch in enumerate(grid) if ch == "g"]
    ghost_count = len(ghosts)

    if ghost_count == 0:
        return 0.25, {"ghost_count": 0, "near_snare_ratio": 0.0}

    # Bon nombre de ghosts : assez vivant, pas mitraillette.
    if 3 <= ghost_count <= 10:
        amount = 1.0
    elif 1 <= ghost_count <= 14:
        amount = 0.75
    else:
        amount = 0.45

    snares = [i for i, ch in enumerate(grid) if ch == "S"]

    near = 0
    for g in ghosts:
        if any(abs(g - s) <= 2 for s in snares):
            near += 1

    near_ratio = near / max(1, ghost_count)

    # Les ghosts sont surtout intéressants proches des snares.
    score = amount * (0.45 + near_ratio * 0.75)

    return clamp(score), {
        "ghost_count": ghost_count,
        "near_snare_ratio": round(near_ratio, 4),
    }


def score_hats(grid):
    hats = [i for i, ch in enumerate(grid) if ch == "H"]
    hat_count = len(hats)

    if hat_count == 0:
        return 0.35, {"hat_count": 0, "even_hat_ratio": 0.0}

    if 4 <= hat_count <= 16:
        amount = 1.0
    elif 2 <= hat_count <= 22:
        amount = 0.75
    else:
        amount = 0.45

    even_hats = sum(1 for i in hats if i % 2 == 0)
    even_ratio = even_hats / max(1, hat_count)

    # On aime une base en croches, mais avec des syncopes.
    continuity = 1.0 - abs(even_ratio - 0.65)

    longest = count_runs(grid, "H")
    if longest >= 6:
        run_penalty = 0.55
    elif longest >= 4:
        run_penalty = 0.75
    else:
        run_penalty = 1.0

    score = amount * continuity * run_penalty

    return clamp(score), {
        "hat_count": hat_count,
        "even_hat_ratio": round(even_ratio, 4),
        "longest_hat_run": longest,
    }


def score_syncopation(grid):
    hits = [(i, ch) for i, ch in enumerate(grid) if ch != "."]

    if not hits:
        return 0.0, {"sync_hit_ratio": 0.0, "total_hits": 0}

    # Off-grid simple : pas strictement sur 1/5/9/13...
    sync_hits = 0
    for i, ch in hits:
        if i % 4 not in [0]:
            if ch in "KgpH":
                sync_hits += 1

    ratio = sync_hits / len(hits)

    # Trop syncopé peut aussi perdre le pulse.
    ideal = 0.48
    score = 1.0 - abs(ratio - ideal) / ideal

    return clamp(score), {
        "sync_hit_ratio": round(ratio, 4),
        "total_hits": len(hits),
    }


def score_clutter(grid):
    total_hits = sum(1 for ch in grid if ch != ".")
    density = total_hits / STEPS

    if 0.30 <= density <= 0.62:
        density_score = 1.0
    elif 0.20 <= density <= 0.75:
        density_score = 0.75
    else:
        density_score = 0.40

    # pénalité si trop de coups forts contigus
    strong = "".join("X" if ch in "KS" else "." for ch in grid)
    longest_strong = count_runs(strong, "X")

    if longest_strong >= 4:
        strong_penalty = 0.55
    elif longest_strong >= 3:
        strong_penalty = 0.75
    else:
        strong_penalty = 1.0

    score = density_score * strong_penalty

    return clamp(score), {
        "density": round(density, 4),
        "total_hits": total_hits,
        "longest_strong_run": longest_strong,
    }


def score_base_structure(grid):
    matches = sum(1 for a, b in zip(grid, BASE_GRID) if a == b and a in "KS")
    target = sum(1 for ch in BASE_GRID if ch in "KS")

    exact = matches / max(1, target)

    # aussi score de rôle : si K/S remplacés par ghost, c'est moins grave.
    role_matches = 0
    for a, b in zip(grid, BASE_GRID):
        if b == ".":
            continue
        if a == b:
            role_matches += 1
        elif b == "S" and a == "g":
            role_matches += 0.45
        elif b == "K" and a in ["p", "H"]:
            role_matches += 0.20

    role = role_matches / max(1, target)

    score = exact * 0.65 + role * 0.35

    return clamp(score), {
        "base_exact": round(exact, 4),
        "base_role": round(role, 4),
    }


def score_groove(grid):
    grid = clean_grid(grid)

    backbeat, backbeat_info = score_backbeat(grid)
    kicks, kicks_info = score_kicks(grid)
    ghosts, ghosts_info = score_ghosts(grid)
    hats, hats_info = score_hats(grid)
    sync, sync_info = score_syncopation(grid)
    clutter, clutter_info = score_clutter(grid)
    base, base_info = score_base_structure(grid)

    # Pondération v01
    final = (
        backbeat * 0.20 +
        kicks    * 0.18 +
        ghosts   * 0.15 +
        hats     * 0.10 +
        sync     * 0.15 +
        clutter  * 0.07 +
        base     * 0.15
    )

    details = {
        "final": round(final, 5),
        "backbeat": round(backbeat, 5),
        "kicks": round(kicks, 5),
        "ghosts": round(ghosts, 5),
        "hats": round(hats, 5),
        "syncopation": round(sync, 5),
        "clutter": round(clutter, 5),
        "base_structure": round(base, 5),
        "info": {
            "backbeat": backbeat_info,
            "kicks": kicks_info,
            "ghosts": ghosts_info,
            "hats": hats_info,
            "syncopation": sync_info,
            "clutter": clutter_info,
            "base_structure": base_info,
        },
    }

    return final, details


def load_grooves(input_path):
    if not input_path.exists():
        print("Input introuvable :", input_path)
        sys.exit(1)

    data = json.loads(input_path.read_text(encoding="utf-8"))

    grooves = []

    for item in data.get("grooves", []):
        full = clean_grid(item.get("full", ""))
        grooves.append({
            "source_index": item.get("index", len(grooves) + 1),
            "full": full,
            "layers": grid_to_layers(full),
        })

    if not grooves:
        print("Aucun groove trouvé dans :", input_path)
        sys.exit(1)

    return grooves


def find_default_input():
    for p in DEFAULT_INPUTS:
        if p.exists():
            return p
    print("Aucun input par défaut trouvé.")
    print("Cherché :")
    for p in DEFAULT_INPUTS:
        print(" ", p)
    sys.exit(1)


def render_groove_block(item, rank=None):
    full = item["full"]
    layers = grid_to_layers(full)
    score = item["score"]["final"]

    title = f"RANK {rank:03d}" if rank is not None else "GROOVE"
    lines = [
        f"{title} | score={score:.5f} | source_index={item.get('source_index')}",
        "12345678|12345678|12345678|12345678",
        "KICK : " + split32(layers["kick"]),
        "SNARE: " + split32(layers["snare"]),
        "GHOST: " + split32(layers["ghost"]),
        "HAT  : " + split32(layers["hat"]),
        "PERC : " + split32(layers["perc"]),
        "FULL : " + split32(full),
        (
            "SUB  : "
            f"backbeat={item['score']['backbeat']:.3f} "
            f"kicks={item['score']['kicks']:.3f} "
            f"ghosts={item['score']['ghosts']:.3f} "
            f"hats={item['score']['hats']:.3f} "
            f"sync={item['score']['syncopation']:.3f} "
            f"clean={item['score']['clutter']:.3f} "
            f"base={item['score']['base_structure']:.3f}"
        ),
        "",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None,
                        help="JSON de grooves à noter.")
    parser.add_argument("--top", type=int, default=50,
                        help="Nombre de meilleurs grooves à garder.")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else find_default_input()

    grooves = load_grooves(input_path)

    scored = []
    for g in grooves:
        final, details = score_groove(g["full"])
        scored.append({
            **g,
            "score": details,
        })

    scored.sort(key=lambda x: x["score"]["final"], reverse=True)
    best = scored[:args.top]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": "groovecritic_v01",
        "input": str(input_path),
        "count": len(scored),
        "scored": scored,
    }

    SCORED_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    best_payload = {
        "version": "groovecritic_best_v01",
        "input": str(input_path),
        "top": args.top,
        "count": len(best),
        "grooves": [
            {
                "index": i,
                "full": item["full"],
                "layers": grid_to_layers(item["full"]),
                "score": item["score"],
            }
            for i, item in enumerate(best, start=1)
        ],
    }

    BEST_JSON.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    SCORED_TXT.write_text(
        "\n".join(render_groove_block(item, rank=i) for i, item in enumerate(scored, start=1)),
        encoding="utf-8"
    )

    BEST_TXT.write_text(
        "\n".join(render_groove_block(item, rank=i) for i, item in enumerate(best, start=1)),
        encoding="utf-8"
    )

    print("Grooves notés :", len(scored))
    print("Input :", input_path)
    print("")
    print("Exports :")
    print(" ", SCORED_JSON)
    print(" ", SCORED_TXT)
    print(" ", BEST_JSON)
    print(" ", BEST_TXT)
    print("")
    print("Top 5 :")
    for i, item in enumerate(best[:5], start=1):
        print(render_groove_block(item, rank=i))


if __name__ == "__main__":
    main()
