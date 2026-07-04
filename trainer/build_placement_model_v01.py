#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BreakBrain - build_placement_model_v01.py

Objectif :
Apprendre une "forme de placement" à partir de tout le dataset.

Au lieu de générer en random :
- on lit dataset/slices_manifest.json
- on regroupe les slices par break
- on quantifie chaque slice sur une grille 32 pas
- on apprend où tombent les kicks / snares / hats / ghosts / percs
- on sauvegarde un modèle statistique dans dataset/models/placement_model_v01.json

Ce n'est pas encore un réseau neuronal, mais c'est la bonne base :
un modèle de groove appris depuis tout ton dataset.
"""

from pathlib import Path
import json
import math
from collections import Counter, defaultdict

DATASET = Path("dataset")
MODEL_DIR = DATASET / "models"
SLICES_MANIFEST = DATASET / "slices_manifest.json"

STEPS = 32

LABELS = ["kick", "snare", "ghost", "hat", "perc"]


def load_slices():
    if not SLICES_MANIFEST.exists():
        raise SystemExit("Manquant : dataset/slices_manifest.json")
    return json.loads(SLICES_MANIFEST.read_text(encoding="utf-8"))


def group_by_break(items):
    groups = defaultdict(list)
    for item in items:
        groups[item["break_id"]].append(item)
    return groups


def quantize_step(start_sample, break_start, break_end):
    length = max(1, break_end - break_start)
    pos = (start_sample - break_start) / length
    step = int(round(pos * (STEPS - 1)))
    return max(0, min(STEPS - 1, step))


def normalize_counter(counter):
    total = sum(counter.values())
    if total <= 0:
        return {k: 0.0 for k in counter}
    return {k: v / total for k, v in counter.items()}


def main():
    items = load_slices()
    groups = group_by_break(items)

    step_label_counts = {label: [0 for _ in range(STEPS)] for label in LABELS}
    transition_counts = {label: Counter() for label in LABELS}
    break_summaries = []

    total_events = 0

    for break_id, events in groups.items():
        events = sorted(events, key=lambda x: x["start_sample"])

        if not events:
            continue

        break_start = min(e["start_sample"] for e in events)
        break_end = max(e["end_sample"] for e in events)
        source = events[0].get("source", "")

        grid_labels = [[] for _ in range(STEPS)]

        for e in events:
            label = e.get("label", "perc")
            if label not in LABELS:
                label = "perc"

            step = quantize_step(e["start_sample"], break_start, break_end)
            step_label_counts[label][step] += 1
            grid_labels[step].append(label)
            total_events += 1

        flat = []
        for step, labels in enumerate(grid_labels):
            for label in labels:
                flat.append((step, label))

        for a, b in zip(flat, flat[1:]):
            a_step, a_label = a
            b_step, b_label = b
            delta = (b_step - a_step) % STEPS
            transition_counts[a_label][f"{delta}:{b_label}"] += 1

        label_count = Counter(e.get("label", "perc") for e in events)

        break_summaries.append({
            "break_id": break_id,
            "source": source,
            "event_count": len(events),
            "label_count": dict(label_count),
        })

    # Probabilité brute par label et par step
    step_label_probs = {}
    for label in LABELS:
        counts = step_label_counts[label]
        max_count = max(counts) if counts else 1
        if max_count <= 0:
            step_label_probs[label] = [0.0 for _ in range(STEPS)]
        else:
            step_label_probs[label] = [c / max_count for c in counts]

    # Probabilité globale "quelle famille tombe sur ce step"
    step_family_counts = []
    for step in range(STEPS):
        c = Counter()
        for label in LABELS:
            c[label] = step_label_counts[label][step]
        step_family_counts.append(dict(c))

    step_family_probs = []
    for c in step_family_counts:
        total = sum(c.values())
        if total <= 0:
            step_family_probs.append({label: 0.0 for label in LABELS})
        else:
            step_family_probs.append({label: c.get(label, 0) / total for label in LABELS})

    model = {
        "version": "placement_model_v01",
        "steps": STEPS,
        "labels": LABELS,
        "total_breaks": len(groups),
        "total_events": total_events,
        "step_label_counts": step_label_counts,
        "step_label_probs": step_label_probs,
        "step_family_probs": step_family_probs,
        "transition_probs": {
            label: normalize_counter(counter)
            for label, counter in transition_counts.items()
        },
        "break_summaries": break_summaries,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / "placement_model_v01.json"
    out.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Modèle de placement créé :", out)
    print("Breaks appris :", len(groups))
    print("Events appris :", total_events)
    print("")
    print("Steps forts par famille :")

    for label in LABELS:
        ranked = sorted(
            enumerate(step_label_counts[label], start=1),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        pretty = ", ".join(f"{step}:{count}" for step, count in ranked if count > 0)
        print(f"  {label:6} -> {pretty}")


if __name__ == "__main__":
    main()
