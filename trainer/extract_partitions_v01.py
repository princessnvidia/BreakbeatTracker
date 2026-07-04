#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GrooveBrain v01 - extract_partitions_v01.py

But :
Transformer ton dataset de slices en "partitions" propres.

Entrée :
    dataset/slices_manifest.json

Sortie :
    dataset/partitions/partitions_v01.json

Chaque break devient une grille 32 pas :
    K = kick
    S = snare forte
    g = ghost snare
    H = hat
    p = percussion
    . = vide

Cette étape ne génère PAS d'audio.
Elle apprend la structure rythmique.
"""

from pathlib import Path
import json
import sys
from collections import Counter, defaultdict

DATASET = Path("dataset")
SLICES_MANIFEST = DATASET / "slices_manifest.json"
OUT_DIR = DATASET / "partitions"
OUT_FILE = OUT_DIR / "partitions_v01.json"

STEPS = 32


def load_manifest():
    if not SLICES_MANIFEST.exists():
        print("Fichier manquant :", SLICES_MANIFEST)
        print("Relance d'abord le script de build dataset.")
        sys.exit(1)
    return json.loads(SLICES_MANIFEST.read_text(encoding="utf-8"))


def group_by_break(items):
    groups = defaultdict(list)
    for item in items:
        groups[item["break_id"]].append(item)
    return groups


def quantize(start_sample, break_start, break_end):
    length = max(1, break_end - break_start)
    pos = (start_sample - break_start) / length
    step = int(round(pos * (STEPS - 1)))
    return max(0, min(STEPS - 1, step))


def event_symbol(label, rms=None):
    if label == "kick":
        return "K"
    if label == "snare":
        return "S"
    if label == "ghost":
        return "g"
    if label == "hat":
        return "H"
    if label == "perc":
        return "p"
    return "p"


def merge_symbol(existing, new):
    """
    Priorité quand plusieurs coups tombent sur le même step.
    On garde la structure lisible :
    K et S dominent, puis g, H, p.
    """
    priority = {
        ".": 0,
        "p": 1,
        "H": 2,
        "g": 3,
        "K": 4,
        "S": 5,
    }
    return new if priority.get(new, 0) >= priority.get(existing, 0) else existing


def main():
    items = load_manifest()
    groups = group_by_break(items)

    partitions = []
    label_stats = Counter()

    for break_id, events in sorted(groups.items()):
        events = sorted(events, key=lambda x: x["start_sample"])

        if len(events) < 4:
            continue

        break_start = min(e["start_sample"] for e in events)
        break_end = max(e["end_sample"] for e in events)
        source = events[0].get("source", "")

        grid = ["."] * STEPS
        layers = {
            "kick": ["."] * STEPS,
            "snare": ["."] * STEPS,
            "ghost": ["."] * STEPS,
            "hat": ["."] * STEPS,
            "perc": ["."] * STEPS,
        }

        quantized_events = []

        for e in events:
            label = e.get("label", "perc")
            symbol = event_symbol(label)
            step = quantize(e["start_sample"], break_start, break_end)

            if label not in layers:
                label = "perc"

            if symbol == "K":
                layers["kick"][step] = "K"
            elif symbol == "S":
                layers["snare"][step] = "S"
            elif symbol == "g":
                layers["ghost"][step] = "g"
            elif symbol == "H":
                layers["hat"][step] = "H"
            else:
                layers["perc"][step] = "p"

            grid[step] = merge_symbol(grid[step], symbol)
            label_stats[label] += 1

            quantized_events.append({
                "step": step,
                "label": label,
                "symbol": symbol,
                "duration": e.get("duration", e.get("duration_sec", 0)),
                "source_slice": e.get("slice_file", ""),
            })

        # Filtre : il faut au moins une snare ou un kick
        if "K" not in grid and "S" not in grid:
            continue

        partitions.append({
            "break_id": break_id,
            "source": source,
            "steps": STEPS,
            "grid": "".join(grid),
            "layers": {k: "".join(v) for k, v in layers.items()},
            "events": quantized_events,
            "label_count": dict(Counter(e["label"] for e in quantized_events)),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": "partitions_v01",
        "steps": STEPS,
        "partition_count": len(partitions),
        "label_stats": dict(label_stats),
        "partitions": partitions,
    }

    OUT_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Partitions créées :", OUT_FILE)
    print("Nombre de partitions :", len(partitions))
    print("Stats labels :", dict(label_stats))

    print("")
    print("Exemples :")
    for p in partitions[:5]:
        print(p["break_id"], Path(p["source"]).name)
        print(" ", p["grid"][:8] + "|" + p["grid"][8:16] + "|" + p["grid"][16:24] + "|" + p["grid"][24:32])


if __name__ == "__main__":
    main()
