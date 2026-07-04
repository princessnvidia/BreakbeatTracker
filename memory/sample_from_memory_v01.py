#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory Sampler v01 - sample_from_memory_v01.py

But :
Utiliser Groove Memory pour sélectionner des grooves par style.

Entrée :
    memory/groove_memory_v01.json

Sorties :
    memory/memory_selection_v01.json
    memory/memory_selection_v01.txt
    exports/memory_sampler_v01/training_corpus_from_memory_v01.json
    exports/memory_sampler_v01/training_corpus_from_memory_v01.txt

Exemples :
    python memory/sample_from_memory_v01.py --tag syncopated --count 80
    python memory/sample_from_memory_v01.py --tag groovy --tag base --count 120
    python memory/sample_from_memory_v01.py --near "London" --count 60
    python memory/sample_from_memory_v01.py --min-sync 0.55 --max-density 0.65 --count 100
"""

from pathlib import Path
import argparse
import json
import random
import sys

MEMORY_JSON = Path("memory/groove_memory_v01.json")

OUT_MEMORY_JSON = Path("memory/memory_selection_v01.json")
OUT_MEMORY_TXT = Path("memory/memory_selection_v01.txt")

OUT_DIR = Path("exports/memory_sampler_v01")
OUT_CORPUS_JSON = OUT_DIR / "training_corpus_from_memory_v01.json"
OUT_CORPUS_TXT = OUT_DIR / "training_corpus_from_memory_v01.txt"

STEPS = 32
ALLOWED = set(".KSgHp")


def clean_grid(grid):
    grid = "".join(ch for ch in str(grid) if ch in ALLOWED)
    return grid[:STEPS].ljust(STEPS, ".")


def split32(grid):
    grid = clean_grid(grid)
    return "|".join(grid[i:i+8] for i in range(0, STEPS, 8))


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


def load_memory():
    if not MEMORY_JSON.exists():
        print("Mémoire introuvable :", MEMORY_JSON)
        print("Lance d'abord : python memory/build_groove_memory_v01.py")
        sys.exit(1)
    return json.loads(MEMORY_JSON.read_text(encoding="utf-8"))


def item_score_for_query(item, tags, min_sync, max_sync, min_density, max_density):
    summary = item.get("summary", {})
    style_tags = set(summary.get("style_tags", []))

    score = 0.0

    for tag in tags:
        if tag in style_tags:
            score += 1.0

    sync = float(summary.get("syncopation", 0.0))
    density = float(summary.get("density", 0.0))

    if min_sync is not None and sync < min_sync:
        return None
    if max_sync is not None and sync > max_sync:
        return None
    if min_density is not None and density < min_density:
        return None
    if max_density is not None and density > max_density:
        return None

    # bonus naturel pour grooves utilisables
    if 0.30 <= density <= 0.65:
        score += 0.35
    if 0.45 <= sync <= 0.72:
        score += 0.35

    # bonus si vient du critic/self-improve
    source_type = item.get("source_type", "")
    if source_type in ["selected", "self_improve_best"]:
        score += 0.50

    # bonus si score critic existe
    critic_score = item.get("score")
    if isinstance(critic_score, dict):
        score += float(critic_score.get("final", 0.0))

    return score


def find_near_items(memory, query, count):
    items = memory["items"]
    neighbors = memory.get("neighbors", [])

    query = query.lower()
    roots = []

    for item in items:
        hay = (item.get("name", "") + " " + item.get("audio_source", "") + " " + item.get("id", "")).lower()
        if query in hay:
            roots.append(item["id"])

    if not roots:
        print("Aucun groove proche trouvé pour :", query)
        return []

    selected_ids = []
    for row in neighbors:
        if row["id"] in roots:
            selected_ids.append(row["id"])
            for n in row.get("neighbors", []):
                selected_ids.append(n["id"])

    # unique en gardant ordre
    seen = set()
    unique = []
    by_id = {item["id"]: item for item in items}
    for gid in selected_ids:
        if gid in seen or gid not in by_id:
            continue
        seen.add(gid)
        unique.append(by_id[gid])

    return unique[:count]


def select_items(memory, args):
    items = memory["items"]

    if args.near:
        near = find_near_items(memory, args.near, args.count)
        if near:
            return near

    scored = []

    for item in items:
        s = item_score_for_query(
            item,
            tags=args.tag or [],
            min_sync=args.min_sync,
            max_sync=args.max_sync,
            min_density=args.min_density,
            max_density=args.max_density,
        )

        if s is None:
            continue

        scored.append((s, item))

    if not scored:
        print("Aucun item ne correspond aux filtres.")
        sys.exit(1)

    # Tri + petite randomisation contrôlée
    scored.sort(key=lambda x: x[0], reverse=True)

    top_pool = scored[:max(args.count * 4, args.count)]
    random.shuffle(top_pool)
    top_pool.sort(key=lambda x: x[0] + random.random() * args.randomness, reverse=True)

    return [item for _, item in top_pool[:args.count]]


def render_txt(items):
    lines = []

    for i, item in enumerate(items, start=1):
        summary = item.get("summary", {})
        tags = "/".join(summary.get("style_tags", []))
        full = item["full"]
        layers = grid_to_layers(full)

        lines.append(f"MEMORY PICK {i:04d} | {item.get('name')} | {tags}")
        lines.append(
            f"density={summary.get('density')} "
            f"sync={summary.get('syncopation')} "
            f"base={summary.get('base_match')} "
            f"K={summary.get('kick_count')} "
            f"S={summary.get('snare_count')} "
            f"g={summary.get('ghost_count')} "
            f"H={summary.get('hat_count')}"
        )
        lines.append("12345678|12345678|12345678|12345678")
        lines.append("KICK : " + split32(layers["kick"]))
        lines.append("SNARE: " + split32(layers["snare"]))
        lines.append("GHOST: " + split32(layers["ghost"]))
        lines.append("HAT  : " + split32(layers["hat"]))
        lines.append("PERC : " + split32(layers["perc"]))
        lines.append("FULL : " + split32(full))
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", action="append", default=[],
                        help="Tag style: sparse, medium, dense, straight, groovy, syncopated, base, broken_base, free")
    parser.add_argument("--near", default=None,
                        help="Chercher autour d'un nom/source, ex: London, Amen, Night")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--min-sync", type=float, default=None)
    parser.add_argument("--max-sync", type=float, default=None)
    parser.add_argument("--min-density", type=float, default=None)
    parser.add_argument("--max-density", type=float, default=None)
    parser.add_argument("--randomness", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    memory = load_memory()
    selected = select_items(memory, args)

    OUT_MEMORY_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": "memory_selection_v01",
        "query": {
            "tag": args.tag,
            "near": args.near,
            "count": args.count,
            "min_sync": args.min_sync,
            "max_sync": args.max_sync,
            "min_density": args.min_density,
            "max_density": args.max_density,
        },
        "count": len(selected),
        "items": selected,
    }

    OUT_MEMORY_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MEMORY_TXT.write_text(render_txt(selected), encoding="utf-8")

    corpus = {
        "version": "training_corpus_from_memory_v01",
        "count": len(selected),
        "grooves": [
            {
                "index": i,
                "full": item["full"],
                "layers": grid_to_layers(item["full"]),
                "source_memory_id": item["id"],
                "source_name": item.get("name", ""),
                "summary": item.get("summary", {}),
                "score": item.get("score", None),
            }
            for i, item in enumerate(selected, start=1)
        ],
    }

    OUT_CORPUS_JSON.write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_CORPUS_TXT.write_text(render_txt(selected), encoding="utf-8")

    print("Sélection mémoire :", len(selected))
    print("Exports :")
    print(" ", OUT_MEMORY_JSON)
    print(" ", OUT_MEMORY_TXT)
    print(" ", OUT_CORPUS_JSON)
    print(" ", OUT_CORPUS_TXT)
    print("")
    print(render_txt(selected[:5]))


if __name__ == "__main__":
    main()
