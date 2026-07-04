#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SampleBrain v02 - render_grooves_fixed_kit_v02.py

Objectif :
Rendre les grooves ASCII en audio, mais avec un kit FIXE par rendu.

Différence majeure avec v01 :
- v01 choisissait un sample différent à chaque événement -> fouillis/collage
- v02 choisit au début :
    1 kick
    1 snare principale
    1 snare douce pour les ghosts
    1 hat
    1 perc optionnel
  puis réutilise ces mêmes samples sur toute la boucle

Entrées :
    dataset/drum_library_v03/drum_library_v03.json
    exports/groovecritic_v01/best_grooves_v01.json
    ou autre JSON de grooves

Sorties :
    exports/samplebrain_v02_fixed_kit/*.wav
    exports/samplebrain_v02_fixed_kit/render_report_v02.txt

Usage :
    python samplebrain/render_grooves_fixed_kit_v02.py

    python samplebrain/render_grooves_fixed_kit_v02.py \
        --input exports/groovecritic_v01/best_grooves_v01.json \
        --count 24 \
        --bpm 150

    python samplebrain/render_grooves_fixed_kit_v02.py \
        --input exports/groovecritic_v01/best_grooves_v01.json \
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

OUT_DIR = Path("exports/samplebrain_v02_fixed_kit")
REPORT_TXT = OUT_DIR / "render_report_v02.txt"

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

        pools[label].append(sample)

        if label == "snare" and role == "snare_soft":
            pools["snare_soft"].append(sample)

    if kit_source and (len(pools["kick"]) == 0 or len(pools["snare"]) == 0 or len(pools["hat"]) == 0):
        print(f"Kit source '{kit_source}' trop incomplet, fallback library complète.")
        return load_library(kit_source=None)

    return pools


def rms_of(sample):
    return sample.get("features", {}).get("rms", 0.0)


def centroid_of(sample):
    return sample.get("features", {}).get("centroid", 0.0)


def duration_of(sample):
    return sample.get("features", {}).get("duration", 0.0)


def choose_weighted_top(samples, key_func, reverse=True, top_ratio=0.45):
    if not samples:
        return None
    sorted_samples = sorted(samples, key=key_func, reverse=reverse)
    n = max(1, int(len(sorted_samples) * top_ratio))
    return random.choice(sorted_samples[:n])


def choose_fixed_kit(pools):
    """
    Crée un kit cohérent :
    - kick plutôt fort/grave
    - snare main plutôt forte
    - ghost = snare douce si dispo, sinon même snare main baissée
    - hat plutôt court/aigu
    """
    kit = {}

    kit["kick"] = choose_weighted_top(
        pools.get("kick", []),
        key_func=lambda s: rms_of(s) + max(0, 2500 - centroid_of(s)) / 2500,
        reverse=True,
        top_ratio=0.50,
    )

    kit["snare"] = choose_weighted_top(
        pools.get("snare", []),
        key_func=lambda s: rms_of(s),
        reverse=True,
        top_ratio=0.50,
    )

    if pools.get("snare_soft"):
        kit["ghost"] = choose_weighted_top(
            pools["snare_soft"],
            key_func=lambda s: rms_of(s),
            reverse=False,
            top_ratio=0.70,
        )
    elif pools.get("snare"):
        # prend une snare faible de la pool
        kit["ghost"] = choose_weighted_top(
            pools["snare"],
            key_func=lambda s: rms_of(s),
            reverse=False,
            top_ratio=0.35,
        )
    else:
        kit["ghost"] = kit.get("snare")

    kit["hat"] = choose_weighted_top(
        pools.get("hat", []),
        key_func=lambda s: centroid_of(s) - duration_of(s) * 2000,
        reverse=True,
        top_ratio=0.45,
    )

    if pools.get("perc"):
        kit["perc"] = random.choice(pools["perc"])
    else:
        kit["perc"] = kit.get("hat") or kit.get("snare")

    # Fallbacks extrêmes
    if kit["kick"] is None:
        kit["kick"] = kit.get("snare") or kit.get("hat")
    if kit["snare"] is None:
        kit["snare"] = kit.get("kick") or kit.get("hat")
    if kit["ghost"] is None:
        kit["ghost"] = kit.get("snare")
    if kit["hat"] is None:
        kit["hat"] = kit.get("snare") or kit.get("kick")
    if kit["perc"] is None:
        kit["perc"] = kit.get("hat") or kit.get("snare") or kit.get("kick")

    return kit


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
        "ghost": 0.90,
        "hat": 0.65,
        "perc": 0.95,
    }.get(role, 1.0)

    y = y[:int(step_samples * max_steps)]
    return apply_fade(y)


def base_gain(role):
    return {
        "kick": 0.95,
        "snare": 0.88,
        "ghost": 0.24,
        "hat": 0.20,
        "perc": 0.22,
    }.get(role, 0.5)


def role_from_symbol(ch):
    return {
        "K": "kick",
        "S": "snare",
        "g": "ghost",
        "H": "hat",
        "p": "perc",
    }.get(ch)


def render_groove(groove, kit, bpm, swing=0.0, humanize=0.012):
    step_samples = int(SR * (60.0 / bpm / 4.0))
    total = step_samples * STEPS
    out = np.zeros(total + SR, dtype=np.float32)

    grid = clean_grid(groove["full"])

    # Précharge les samples du kit une seule fois.
    kit_audio = {}
    for role, sample in kit.items():
        if sample is None:
            continue
        y, sr = read_sample(sample)
        kit_audio[role] = trim_for_role(y, role, step_samples)

    used = []

    for step, ch in enumerate(grid):
        role = role_from_symbol(ch)
        if role is None:
            continue

        if role not in kit_audio:
            continue

        y = kit_audio[role]

        # Le même sample, avec micro variations de volume très faibles seulement.
        gain = base_gain(role)
        if role in ["ghost", "hat", "perc"]:
            gain *= random.uniform(0.86, 1.08)
        else:
            gain *= random.uniform(0.96, 1.03)

        start = step * step_samples

        if swing > 0 and step % 2 == 1:
            start += int(step_samples * swing * 0.35)

        # Humanisation très légère, pas de décalage violent.
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
            "sample_id": kit[role].get("id") if kit.get(role) else None,
            "sample": kit[role].get("library_rel") if kit.get(role) else None,
            "gain": round(gain, 4),
        })

    return normalize_audio(out), used


def render_txt_report(rows):
    lines = []
    for row in rows:
        g = row["groove"]
        layers = grid_to_layers(g["full"])
        lines.append(f"RENDER {row['index']:03d} -> {row['wav']}")
        lines.append("KIT:")
        for role, sample in row["kit"].items():
            if sample:
                lines.append(f"  {role:6}: {sample.get('id')} | {sample.get('library_rel')}")
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
                        help="Optionnel : utiliser les samples d'un break source, ex: London.")
    parser.add_argument("--one-kit", action="store_true",
                        help="Utilise le même kit pour tous les rendus, au lieu d'un kit fixe différent par rendu.")
    parser.add_argument("--swing", type=float, default=0.04)
    parser.add_argument("--humanize", type=float, default=0.010)
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

    global_kit = choose_fixed_kit(pools) if args.one_kit else None

    rows = []

    for i, groove in enumerate(grooves[:args.count], start=1):
        kit = global_kit if global_kit else choose_fixed_kit(pools)

        audio, used = render_groove(
            groove,
            kit,
            bpm=args.bpm,
            swing=args.swing,
            humanize=args.humanize,
        )

        suffix = ""
        if args.kit_source:
            suffix += "_" + args.kit_source.replace(" ", "_")
        if args.one_kit:
            suffix += "_onekit"

        wav = OUT_DIR / f"samplebrain_v02_fixed{suffix}_{i:03d}_{int(args.bpm)}bpm.wav"
        sf.write(wav, audio, SR)

        rows.append({
            "index": i,
            "wav": str(wav),
            "groove": groove,
            "kit": kit,
            "used_samples": used,
        })

        print("Export :", wav)

    REPORT_TXT.write_text(render_txt_report(rows), encoding="utf-8")

    json_report = OUT_DIR / "render_report_v02.json"
    json_report.write_text(json.dumps({
        "version": "samplebrain_v02_fixed_kit",
        "input": str(input_path),
        "bpm": args.bpm,
        "kit_source": args.kit_source,
        "one_kit": args.one_kit,
        "renders": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("")
    print("Rapports :")
    print(" ", REPORT_TXT)
    print(" ", json_report)


if __name__ == "__main__":
    main()
