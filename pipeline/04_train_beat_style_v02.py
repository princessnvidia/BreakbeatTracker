#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_train_beat_style_v02.py

Transforme les beats sauvegardés en mémoire de génération.

Ce script scanne :
- dataset/**/*.json
- dataset/**/*.jsonl

Il cherche des patterns contenant des notes avec :
- x_step
- pair
- lane éventuellement
- length éventuellement

Puis il écrit :
    dataset/learning/beat_style_model_v02.json

Le modèle apprend :
- quelle slice apparaît souvent à chaque slot
- quelles slices s'enchaînent souvent
- quels beats ont été sauvegardés/validés
"""

from pathlib import Path
import argparse
import json
import math
from collections import defaultdict, Counter
from datetime import datetime


DATASET_DIR = Path("dataset")
OUT_MODEL = Path("dataset/learning/beat_style_model_v01.json")

HIT_SPACING_STEPS = 2
DEFAULT_SLOTS = 16


def is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False


def as_int(x, default=0):
    try:
        return int(round(float(x)))
    except Exception:
        return default


def guess_safe(data, path):
    candidates = [
        data.get("safe") if isinstance(data, dict) else None,
        data.get("break") if isinstance(data, dict) else None,
        data.get("source") if isinstance(data, dict) else None,
        data.get("project", {}).get("safe") if isinstance(data, dict) and isinstance(data.get("project"), dict) else None,
    ]

    for c in candidates:
        if c:
            return str(c)

    name = path.stem

    for suffix in [
        "_tracker_app_edit_v49",
        "_tracker_app_edit_v48",
        "_tracker_app_edit_v47",
        "_tracker_app_edit_v46",
        "_tracker_app_edit_v45",
        "_tracker_app_edit_v44",
        "_latest_pattern",
        "_pattern",
    ]:
        name = name.replace(suffix, "")

    return name


def looks_like_note(item):
    if not isinstance(item, dict):
        return False

    if "pair" not in item:
        return False

    if "x_step" not in item:
        return False

    if not is_number(item.get("pair")):
        return False

    if not is_number(item.get("x_step")):
        return False

    return True


def normalize_note(item):
    x_step = as_int(item.get("x_step"), 0)
    pair = as_int(item.get("pair"), 0)
    lane = as_int(item.get("lane"), pair)
    length = as_int(item.get("length"), 2)

    slot = int(round(x_step / float(HIT_SPACING_STEPS)))

    return {
        "x_step": x_step,
        "slot": slot,
        "pair": pair,
        "lane": lane,
        "length": max(1, length),
    }


def extract_patterns_from_obj(obj, path):
    """
    Retourne une liste de patterns.
    Chaque pattern = dict safe + notes.
    """
    out = []

    if not isinstance(obj, dict):
        return out

    safe = guess_safe(obj, path)

    candidate_keys = [
        "pattern",
        "items",
        "notes",
        "hits",
        "events",
    ]

    for key in candidate_keys:
        value = obj.get(key)

        if not isinstance(value, list):
            continue

        notes = [normalize_note(x) for x in value if looks_like_note(x)]

        if len(notes) >= 2:
            notes = sorted(notes, key=lambda n: (n["x_step"], n["lane"], n["pair"]))
            out.append({
                "safe": safe,
                "source_file": str(path),
                "key": key,
                "notes": notes,
                "reason": obj.get("reason") or obj.get("save_reason") or obj.get("event_type") or "",
                "time": obj.get("time") or obj.get("saved_at") or "",
            })

    # Certains fichiers peuvent être directement une liste dans "blocks",
    # mais on évite les pair_blocks de slicing : eux ont source_start_sample.
    blocks = obj.get("blocks")

    if isinstance(blocks, list):
        if any(isinstance(b, dict) and "source_start_sample" in b for b in blocks):
            pass
        else:
            notes = [normalize_note(x) for x in blocks if looks_like_note(x)]
            if len(notes) >= 2:
                notes = sorted(notes, key=lambda n: (n["x_step"], n["lane"], n["pair"]))
                out.append({
                    "safe": safe,
                    "source_file": str(path),
                    "key": "blocks",
                    "notes": notes,
                    "reason": obj.get("reason") or "",
                    "time": obj.get("time") or "",
                })

    return out


def load_json_patterns(path):
    patterns = []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return patterns

    if isinstance(data, dict):
        patterns.extend(extract_patterns_from_obj(data, path))

    elif isinstance(data, list):
        notes = [normalize_note(x) for x in data if looks_like_note(x)]
        if len(notes) >= 2:
            notes = sorted(notes, key=lambda n: (n["x_step"], n["lane"], n["pair"]))
            patterns.append({
                "safe": path.stem,
                "source_file": str(path),
                "key": "root_list",
                "notes": notes,
                "reason": "",
                "time": "",
            })

    return patterns


def load_jsonl_patterns(path):
    patterns = []

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return patterns

    for line in lines:
        line = line.strip()

        if not line:
            continue

        try:
            data = json.loads(line)
        except Exception:
            continue

        if isinstance(data, dict):
            patterns.extend(extract_patterns_from_obj(data, path))

            # Corrections : after peut être une note positive.
            after = data.get("after")
            if looks_like_note(after):
                safe = guess_safe(data, path)
                patterns.append({
                    "safe": safe,
                    "source_file": str(path),
                    "key": "correction_after",
                    "notes": [normalize_note(after)],
                    "reason": data.get("event_type", "correction_after"),
                    "time": data.get("time", ""),
                })

    return patterns


def pattern_weight(pattern):
    """
    Pondération simple :
    - beat complet sauvegardé = fort
    - correction individuelle = faible
    - validation/save = encore plus fort
    """
    key = str(pattern.get("key", ""))
    reason = str(pattern.get("reason", "")).lower()

    if key == "correction_after":
        return 1.0

    weight = 4.0

    if "save" in reason or "validation" in reason or "valid" in reason:
        weight += 3.0

    if "randomize" in reason:
        weight -= 1.0

    return max(1.0, weight)


def sort_counter_key(value):
    """
    Tri robuste :
    - nombres : 0,1,2,3...
    - textes/chemins : ordre alphabétique
    """
    s = str(value)

    try:
        return (0, int(s))
    except Exception:
        return (1, s)


def counter_to_dict(counter):
    return {
        str(k): float(v)
        for k, v in sorted(counter.items(), key=lambda kv: sort_counter_key(kv[0]))
    }


def nested_counter_to_dict(d):
    out = {}

    for k, counter in d.items():
        out[str(k)] = counter_to_dict(counter)

    return out


def train(patterns, slots=DEFAULT_SLOTS):
    by_break = {}
    global_slot_pair_counts = defaultdict(Counter)
    global_pair_counts = Counter()
    global_transition_counts = defaultdict(Counter)
    global_length_counts = Counter()

    examples = []

    for pattern in patterns:
        safe = pattern["safe"]
        notes = pattern["notes"]

        if safe not in by_break:
            by_break[safe] = {
                "slot_pair_counts": defaultdict(Counter),
                "pair_counts": Counter(),
                "transition_counts": defaultdict(Counter),
                "length_counts": Counter(),
                "examples_count": 0,
                "notes_count": 0,
                "source_files": Counter(),
            }

        bucket = by_break[safe]
        weight = pattern_weight(pattern)

        bucket["examples_count"] += 1
        bucket["notes_count"] += len(notes)
        bucket["source_files"][pattern["source_file"]] += 1

        examples.append({
            "safe": safe,
            "source_file": pattern["source_file"],
            "key": pattern["key"],
            "notes": len(notes),
            "weight": weight,
        })

        # Notes individuelles.
        for note in notes:
            slot = int(note["slot"]) % slots
            pair = int(note["pair"])
            length = int(note["length"])

            bucket["slot_pair_counts"][slot][pair] += weight
            bucket["pair_counts"][pair] += weight
            bucket["length_counts"][length] += weight

            global_slot_pair_counts[slot][pair] += weight
            global_pair_counts[pair] += weight
            global_length_counts[length] += weight

        # Transitions dans l'ordre.
        ordered = sorted(notes, key=lambda n: (n["slot"], n["x_step"]))

        for a, b in zip(ordered, ordered[1:]):
            pa = int(a["pair"])
            pb = int(b["pair"])

            bucket["transition_counts"][pa][pb] += weight
            global_transition_counts[pa][pb] += weight

    model = {
        "version": "beat_style_model_v02",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "slots": slots,
        "hit_spacing_steps": HIT_SPACING_STEPS,
        "patterns_count": len(patterns),
        "examples": examples[-200:],
        "global": {
            "slot_pair_counts": nested_counter_to_dict(global_slot_pair_counts),
            "pair_counts": counter_to_dict(global_pair_counts),
            "transition_counts": nested_counter_to_dict(global_transition_counts),
            "length_counts": counter_to_dict(global_length_counts),
        },
        "breaks": {},
    }

    for safe, bucket in by_break.items():
        model["breaks"][safe] = {
            "examples_count": bucket["examples_count"],
            "notes_count": bucket["notes_count"],
            "slot_pair_counts": nested_counter_to_dict(bucket["slot_pair_counts"]),
            "pair_counts": counter_to_dict(bucket["pair_counts"]),
            "transition_counts": nested_counter_to_dict(bucket["transition_counts"]),
            "length_counts": counter_to_dict(bucket["length_counts"]),
            "source_files": counter_to_dict(bucket["source_files"]),
        }

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument("--out", default=str(OUT_MODEL))
    parser.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out = Path(args.out)

    if not dataset.exists():
        raise SystemExit(f"ERREUR: dossier introuvable: {dataset}")

    patterns = []

    for path in sorted(dataset.rglob("*.json")):
        # Évite de réentraîner sur le modèle lui-même.
        if path.name == out.name:
            continue

        # Les pair_blocks décrivent des slices, pas des beats.
        if path.name.endswith("_pair_blocks_v02.json"):
            continue

        patterns.extend(load_json_patterns(path))

    for path in sorted(dataset.rglob("*.jsonl")):
        patterns.extend(load_jsonl_patterns(path))

    model = train(patterns, slots=args.slots)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")

    print("OK train beat style")
    print("Patterns trouvés :", len(patterns))
    print("Breaks appris :", len(model["breaks"]))
    print("Model :", out)
    print("")

    for safe, bucket in sorted(model["breaks"].items(), key=lambda kv: kv[1]["examples_count"], reverse=True):
        print(
            f"{safe} | examples={bucket['examples_count']} | notes={bucket['notes_count']}"
        )


if __name__ == "__main__":
    main()
