#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SampleBrain v01 - render_grooves_v01.py

But :
Transformer les grooves ASCII en vrais fichiers WAV à partir de Drum Library v03.

Entrées possibles :
    exports/groovegpt_v01/groovegpt_generated_v01.json
    exports/groovebrain_ascii_v02/generated_ascii_grooves_v02.json
    exports/groovecritic_v01/best_grooves_v01.json
    exports/self_improve_v01/best_for_training_v01.json
    exports/memory_sampler_v01/training_corpus_from_memory_v01.json

Bibliothèque :
    dataset/drum_library_v03/drum_library_v03.json

Sorties :
    exports/samplebrain_v01/*.wav
    exports/samplebrain_v01/render_report_v01.txt

Usage :
    python samplebrain/render_grooves_v01.py

    python samplebrain/render_grooves_v01.py \
        --input exports/groovecritic_v01/best_grooves_v01.json \
        --count 24 \
        --bpm 150

    python samplebrain/render_grooves_v01.py \
        --input exports/self_improve_v01/best_for_training_v01.json \
        --kit-source "London" \
        --count 16
"""

from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import soundfile as sf

DRUM_LIBRARY = Path("dataset/drum_library_v03/drum_library_v03.json")

DEFAULT_INPUTS = [
    Path("exports/groovecritic_v01/best_grooves_v01.json"),
    Path("exports/self_improve_v01/best_for_training_v01.json"),
    Path("exports/memory_sampler_v01/training_corpus_from_memory_v01.json"),
    Path("exports/groovegpt_v01/groovegpt_generated_v01.json"),
    Path("exports/groovebrain_ascii_v02/generated_ascii_grooves_v02.json"),
]

OUT_DIR = Path("exports/samplebrain_v01")
REPORT_TXT = OUT_DIR / "render_report_v01.txt"

STEPS = 32
SR = 44100
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


def normalize_audio(y, peak=0.95):
    m = np.max(np.abs(y)) if len(y) else 0
    if m <= 1e-9:
        return y
    return y / m * peak


def load_json(path):
    if not path.exists():
        print("Fichier introuvable :", path)
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def find_default_input():
    for path in DEFAULT_INPUTS:
        if path.exists():
            return path
    print("Aucun fichier de grooves trouvé.")
    print("Cherché :")
    for p in DEFAULT_INPUTS:
        print(" ", p)
    sys.exit(1)


def load_grooves(path):
    data = load_json(path)
    grooves = []

    for i, item in enumerate(data.get("grooves", []), start=1):
        full = clean_grid(item.get("full", ""))
        grooves.append({
            "index": item.get("index", i),
            "full": full,
            "layers": grid_to_layers(full),
            "score": item.get("score", None),
        })

    if not grooves:
        print("Aucun groove dans :", path)
        sys.exit(1)

    return grooves


def load_library(kit_source=None):
    data = load_json(DRUM_LIBRARY)

    pools = {
        "kick": [],
        "snare": [],
        "snare_soft": [],
        "hat": [],
        "perc": [],
    }

    for sample in data.get("samples", []):
        label = sample.get("label")
        if label not in ["kick", "snare", "hat", "perc"]:
            continue

        if kit_source:
            src = sample.get("source_break", "")
            if kit_source.lower() not in src.lower():
                continue

        role = sample.get("tags", {}).get("role", label)

        if label == "snare" and role == "snare_soft":
            pools["snare_soft"].append(sample)

        pools[label].append(sample)

    # Si kit-source trop restrictif, on fallback global.
    if kit_source and (len(pools["kick"]) == 0 or len(pools["snare"]) == 0 or len(pools["hat"]) == 0):
        print(f"Kit source '{kit_source}' trop incomplet, fallback library complète.")
        return load_library(kit_source=None)

    return pools


def choose_sample(pools, label, prefer_soft=False):
    if label == "ghost":
        if prefer_soft and pools.get("snare_soft"):
            return random.choice(pools["snare_soft"])
        if pools.get("snare"):
            # Choisit plutôt dans les snares les plus soft.
            sorted_snares = sorted(
                pools["snare"],
                key=lambda s: s.get("features", {}).get("rms", 1.0)
            )
            top = sorted_snares[:max(1, len(sorted_snares)//3)]
            return random.choice(top)

    if label == "snare" and pools.get("snare"):
        # Pour snare principale, évite les snares trop faibles si possible.
        sorted_snares = sorted(
            pools["snare"],
            key=lambda s: s.get("features", {}).get("rms", 0.0),
            reverse=True
        )
        top = sorted_snares[:max(1, len(sorted_snares)//2)]
        return random.choice(top)

    if label in pools and pools[label]:
        return random.choice(pools[label])

    fallbacks = {
        "kick": ["perc", "snare", "hat"],
        "snare": ["perc", "hat", "kick"],
        "ghost": ["snare", "perc", "hat"],
        "hat": ["perc", "snare"],
        "perc": ["hat", "snare", "kick"],
    }

    for fb in fallbacks.get(label, []):
        if pools.get(fb):
            return random.choice(pools[fb])

    return None


def read_sample(sample):
    path = Path(sample["library_file"])
    if not path.exists():
        path = Path(sample["source_slice"])

    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)

    return y, sr


def apply_fade(y, fade_ms=3):
    if len(y) < 16:
        return y

    fade = min(int(SR * fade_ms / 1000), len(y) // 4)
    if fade <= 1:
        return y

    y = y.copy()
    ramp = np.linspace(0, 1, fade)
    y[:fade] *= ramp
    y[-fade:] *= ramp[::-1]
    return y


def trim_for_role(y, role, step_samples):
    max_steps = {
        "kick": 2.4,
        "snare": 2.2,
        "ghost": 0.95,
        "hat": 0.70,
        "perc": 1.0,
    }.get(role, 1.0)

    y = y[:int(step_samples * max_steps)]
    return apply_fade(y)


def event_gain(role):
    if role == "kick":
        return random.uniform(0.85, 1.00)
    if role == "snare":
        return random.uniform(0.78, 0.95)
    if role == "ghost":
        return random.uniform(0.18, 0.34)
    if role == "hat":
        return random.uniform(0.16, 0.28)
    if role == "perc":
        return random.uniform(0.18, 0.32)
    return 0.5


def render_groove(groove, pools, bpm, swing=0.0, humanize=0.025):
    step_samples = int(SR * (60.0 / bpm / 4.0))
    total = step_samples * STEPS
    out = np.zeros(total + SR, dtype=np.float32)

    grid = clean_grid(groove["full"])

    events = []

    for step, ch in enumerate(grid):
        if ch == ".":
            continue
        if ch == "K":
            events.append((step, "kick"))
        elif ch == "S":
            events.append((step, "snare"))
        elif ch == "g":
            events.append((step, "ghost"))
        elif ch == "H":
            events.append((step, "hat"))
        elif ch == "p":
            events.append((step, "perc"))

    # Ordre stable dans le mix.
    priority = {"kick": 0, "snare": 1, "ghost": 2, "hat": 3, "perc": 4}
    events.sort(key=lambda x: (x[0], priority[x[1]]))

    used = []

    for step, role in events:
        sample = choose_sample(pools, role, prefer_soft=True)
        if sample is None:
            continue

        y, sr = read_sample(sample)

        # On suppose 44100 car nos builders écrivent à 44100.
        # Si sr différent, on ignore pour v01.
        y = trim_for_role(y, role, step_samples)

        gain = event_gain(role)

        start = step * step_samples

        # Swing : décale légèrement les steps impairs.
        if swing > 0 and step % 2 == 1:
            start += int(step_samples * swing * 0.35)

        # Humanize : surtout petits éléments.
        if role in ["ghost", "hat", "perc"]:
            start += random.randint(
                -int(step_samples * humanize),
                int(step_samples * humanize)
            )

        if start < 0:
            continue

        end = min(start + len(y), len(out))
        if end > start:
            out[start:end] += y[:end-start] * gain

        used.append({
            "step": step,
            "role": role,
            "sample_id": sample.get("id"),
            "sample": sample.get("library_rel"),
            "gain": round(gain, 4),
        })

    return normalize_audio(out), used


def render_txt_report(rows):
    lines = []
    for row in rows:
        g = row["groove"]
        layers = grid_to_layers(g["full"])
        lines.append(f"RENDER {row['index']:03d} -> {row['wav']}")
        lines.append("12345678|12345678|12345678|12345678")
        lines.append("KICK : " + split32(layers["kick"]))
        lines.append("SNARE: " + split32(layers["snare"]))
        lines.append("GHOST: " + split32(layers["ghost"]))
        lines.append("HAT  : " + split32(layers["hat"]))
        lines.append("PERC : " + split32(layers["perc"]))
        lines.append("FULL : " + split32(g["full"]))
        lines.append(f"events={len(row['used_samples'])}")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--bpm", type=float, default=150.0)
    parser.add_argument("--kit-source", default=None,
                        help="Optionnel : utiliser les slices d'un break source, ex: London.")
    parser.add_argument("--swing", type=float, default=0.05)
    parser.add_argument("--humanize", type=float, default=0.020)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    input_path = Path(args.input) if args.input else find_default_input()

    grooves = load_grooves(input_path)
    pools = load_library(kit_source=args.kit_source)

    print("Input grooves :", input_path)
    print("Drum library :", DRUM_LIBRARY)
    print("Pools :")
    for k, v in pools.items():
        print(f"  {k:12}: {len(v)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for i, groove in enumerate(grooves[:args.count], start=1):
        audio, used = render_groove(
            groove,
            pools,
            bpm=args.bpm,
            swing=args.swing,
            humanize=args.humanize,
        )

        suffix = ""
        if args.kit_source:
            suffix = "_" + args.kit_source.replace(" ", "_")

        wav = OUT_DIR / f"samplebrain_v01{suffix}_{i:03d}_{int(args.bpm)}bpm.wav"
        sf.write(wav, audio, SR)

        rows.append({
            "index": i,
            "wav": str(wav),
            "groove": groove,
            "used_samples": used,
        })

        print("Export :", wav)

    REPORT_TXT.write_text(render_txt_report(rows), encoding="utf-8")

    json_report = OUT_DIR / "render_report_v01.json"
    json_report.write_text(json.dumps({
        "version": "samplebrain_v01",
        "input": str(input_path),
        "bpm": args.bpm,
        "kit_source": args.kit_source,
        "renders": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("")
    print("Rapports :")
    print(" ", REPORT_TXT)
    print(" ", json_report)


if __name__ == "__main__":
    main()
