#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JungleBrain Camo v01 - render_camo_jungle_v01.py

But :
Tester une vraie grammaire jungle UNIQUEMENT avec les samples du Camo Break.

Base principale :
    K..S|..KS|K..S|..KS

Sur 16 pas :
    K..S..KSK..S..KS

Sur 32 pas :
    K..S..KSK..S..KSK..S..KSK..S..KS

Entrée :
    dataset/drum_library_v03/drum_library_v03.json

Sorties :
    exports/junglebrain_camo_v01/*.wav
    exports/junglebrain_camo_v01/patterns_camo_jungle_v01.txt
    exports/junglebrain_camo_v01/render_report_camo_jungle_v01.json

Usage :
    python junglebrain/render_camo_jungle_v01.py

Options :
    python junglebrain/render_camo_jungle_v01.py --count 32 --mutation 0.35 --bpm 150
    python junglebrain/render_camo_jungle_v01.py --kit-source "Camo" --one-kit
"""

from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import soundfile as sf

DRUM_LIBRARY = Path("dataset/drum_library_v03/drum_library_v03.json")

OUT_DIR = Path("exports/junglebrain_camo_v01")
PATTERNS_TXT = OUT_DIR / "patterns_camo_jungle_v01.txt"
REPORT_JSON = OUT_DIR / "render_report_camo_jungle_v01.json"

STEPS = 32
SR = 44100
ALLOWED = set(".KSgHp")

BASE_16 = "K..S..KSK..S..KS"
BASE_32 = BASE_16 + BASE_16

JUNGLE_SEEDS = [
    BASE_32,
    "K.gS..KSK..S.gKSK.gS..KSK..S.gKS",
    "K..S.KKSK.gS..KSK..S..KSK.gS.KKS",
    "K..S..KSK..Sg.KSK.gS..KSK..S..KS",
    "K..S..KSK.gS..KSK..S.KKSK..S.gKS",
    "K..S..KSK..S..KSK..S..KSK.gS.gKS",
]


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


def sample_matches_source(sample, kit_source):
    src = sample.get("source_break", "")
    return kit_source.lower() in src.lower()


def make_pools(samples):
    pools = {"kick": [], "snare": [], "hat": [], "perc": []}
    for s in samples:
        label = s.get("label")
        if label in pools:
            pools[label].append(s)
    return pools


def load_camo_pools(kit_source="Camo", allow_global_fallback=False):
    data = load_json(DRUM_LIBRARY)
    all_samples = data.get("samples", [])

    source_samples = [s for s in all_samples if sample_matches_source(s, kit_source)]

    if not source_samples:
        print(f"Aucun sample trouvé pour '{kit_source}'.")
        print("Regarde dataset/drum_library_v03/report_v03.txt pour le nom exact.")
        sys.exit(1)

    pools = make_pools(source_samples)

    print(f"Samples source locked pour '{kit_source}' :")
    for k, v in pools.items():
        print(f"  {k:8}: {len(v)}")

    missing = [k for k in ["kick", "snare", "hat"] if len(pools[k]) == 0]

    if missing and not allow_global_fallback:
        print("")
        print("Classes manquantes dans ce break :", ", ".join(missing))
        print("Relance avec --allow-global-fallback si tu veux compléter,")
        print("mais ça ne sera plus 100% Camo.")
        sys.exit(1)

    if missing and allow_global_fallback:
        global_pools = make_pools(all_samples)
        for k in missing:
            pools[k] = global_pools[k]

    return pools


def rms_of(sample):
    return sample.get("features", {}).get("rms", 0.0)


def centroid_of(sample):
    return sample.get("features", {}).get("centroid", 0.0)


def duration_of(sample):
    return sample.get("features", {}).get("duration", 0.0)


def choose_top(samples, key_func, reverse=True, top_ratio=0.65):
    if not samples:
        return None
    ordered = sorted(samples, key=key_func, reverse=reverse)
    n = max(1, int(len(ordered) * top_ratio))
    return random.choice(ordered[:n])


def choose_kit(pools):
    kit = {}

    kit["kick"] = choose_top(
        pools["kick"],
        lambda s: rms_of(s) + max(0, 2600 - centroid_of(s)) / 2600,
        reverse=True,
        top_ratio=0.80,
    )

    kit["snare"] = choose_top(
        pools["snare"],
        lambda s: rms_of(s),
        reverse=True,
        top_ratio=0.70,
    )

    # Ghost = snare douce du même break, pas classe séparée
    kit["ghost"] = choose_top(
        pools["snare"],
        lambda s: rms_of(s),
        reverse=False,
        top_ratio=0.45,
    ) or kit["snare"]

    kit["hat"] = choose_top(
        pools["hat"],
        lambda s: centroid_of(s) - duration_of(s) * 1800,
        reverse=True,
        top_ratio=0.75,
    )

    kit["perc"] = random.choice(pools["perc"]) if pools.get("perc") else kit["hat"]

    return kit


def mutate_jungle_grid(mutation=0.28, hat_density=0.75):
    grid = list(random.choice(JUNGLE_SEEDS))

    # Snares structurelles de la base K..S..KS...
    base_snares = [3, 7, 11, 15, 19, 23, 27, 31]
    base_kicks = [0, 6, 8, 14, 16, 22, 24, 30]

    # Garde fortement la base
    for i in base_kicks:
        if random.random() < 0.88:
            grid[i] = "K"
    for i in base_snares:
        if random.random() < 0.88:
            grid[i] = "S"

    # Mutations : ghost autour des snares
    ghost_spots = [2, 4, 5, 10, 12, 13, 18, 20, 21, 26, 28, 29]
    for i in ghost_spots:
        if grid[i] == "." and random.random() < mutation * 0.85:
            grid[i] = "g"

    # Mutations : kicks de relance, mais pas trop
    kick_spots = [1, 5, 9, 12, 17, 21, 25, 28]
    for i in kick_spots:
        if grid[i] == "." and random.random() < mutation * 0.38:
            grid[i] = "K"

    # Hats/rides en croches dans les trous
    for i in range(STEPS):
        if grid[i] == ".":
            if i % 2 == 0 and random.random() < hat_density:
                grid[i] = "H"
            elif i % 2 == 1 and random.random() < hat_density * 0.42:
                grid[i] = "H"

    # Fill fin de boucle
    if random.random() < mutation:
        for i in [28, 29, 30]:
            if grid[i] == "." or grid[i] == "H":
                grid[i] = random.choice(["g", "H", "g"])

    # Anti-fouillis : limite les kicks hors base
    kicks = [i for i, ch in enumerate(grid) if ch == "K"]
    if len(kicks) > 10:
        extra = [i for i in kicks if i not in base_kicks]
        random.shuffle(extra)
        for i in extra[10-len(base_kicks):]:
            if i not in base_kicks:
                grid[i] = "."

    return "".join(grid[:STEPS])


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
        "kick": 2.0,
        "snare": 2.0,
        "ghost": 0.75,
        "hat": 0.55,
        "perc": 0.80,
    }.get(role, 1.0)
    return apply_fade(y[:int(step_samples * max_steps)])


def role_from_symbol(ch):
    return {"K": "kick", "S": "snare", "g": "ghost", "H": "hat", "p": "perc"}.get(ch)


def base_gain(role):
    return {
        "kick": 0.95,
        "snare": 0.86,
        "ghost": 0.22,
        "hat": 0.17,
        "perc": 0.20,
    }.get(role, 0.5)


def render(grid, kit, bpm=150, swing=0.025, humanize=0.004):
    step_samples = int(SR * (60.0 / bpm / 4.0))
    total = step_samples * STEPS
    out = np.zeros(total + SR, dtype=np.float32)

    kit_audio = {}
    for role, sample in kit.items():
        if sample is None:
            continue
        y, sr = read_sample(sample)
        kit_audio[role] = trim_for_role(y, role, step_samples)

    used = []

    for step, ch in enumerate(grid):
        role = role_from_symbol(ch)
        if role is None or role not in kit_audio:
            continue

        y = kit_audio[role]
        gain = base_gain(role)

        # même sample, très petites variations seulement
        if role in ["ghost", "hat", "perc"]:
            gain *= random.uniform(0.90, 1.06)
        else:
            gain *= random.uniform(0.98, 1.02)

        start = step * step_samples

        if swing > 0 and step % 2 == 1:
            start += int(step_samples * swing * 0.35)

        if role in ["ghost", "hat"]:
            start += random.randint(-int(step_samples * humanize), int(step_samples * humanize))

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


def render_txt(rows):
    lines = []
    for row in rows:
        grid = row["grid"]
        layers = grid_to_layers(grid)
        lines.append(f"JUNGLE {row['index']:03d} -> {row['wav']}")
        lines.append("BASE : K..S|..KS|K..S|..KS")
        lines.append("12345678|12345678|12345678|12345678")
        lines.append("KICK : " + split32(layers["kick"]))
        lines.append("SNARE: " + split32(layers["snare"]))
        lines.append("GHOST: " + split32(layers["ghost"]))
        lines.append("HAT  : " + split32(layers["hat"]))
        lines.append("PERC : " + split32(layers["perc"]))
        lines.append("FULL : " + split32(grid))
        lines.append("KIT:")
        for role, sample in row["kit"].items():
            if sample:
                lines.append(f"  {role:6}: {sample.get('id')} | {sample.get('library_rel')}")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit-source", default="Camo")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--bpm", type=float, default=150.0)
    parser.add_argument("--mutation", type=float, default=0.30)
    parser.add_argument("--hat-density", type=float, default=0.78)
    parser.add_argument("--one-kit", action="store_true", default=True)
    parser.add_argument("--allow-global-fallback", action="store_true")
    parser.add_argument("--swing", type=float, default=0.025)
    parser.add_argument("--humanize", type=float, default=0.004)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pools = load_camo_pools(args.kit_source, allow_global_fallback=args.allow_global_fallback)

    global_kit = choose_kit(pools)

    rows = []

    for i in range(1, args.count + 1):
        kit = global_kit if args.one_kit else choose_kit(pools)
        grid = mutate_jungle_grid(mutation=args.mutation, hat_density=args.hat_density)
        audio, used = render(grid, kit, bpm=args.bpm, swing=args.swing, humanize=args.humanize)

        suffix = args.kit_source.replace(" ", "_")
        wav = OUT_DIR / f"junglebrain_camo_v01_{suffix}_{i:03d}_{int(args.bpm)}bpm.wav"
        sf.write(wav, audio, SR)

        rows.append({
            "index": i,
            "wav": str(wav),
            "grid": grid,
            "kit": kit,
            "used_samples": used,
        })

        print("Export :", wav)

    PATTERNS_TXT.write_text(render_txt(rows), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps({
        "version": "junglebrain_camo_v01",
        "kit_source": args.kit_source,
        "base": "K..S|..KS|K..S|..KS",
        "bpm": args.bpm,
        "mutation": args.mutation,
        "renders": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("")
    print("Patterns :", PATTERNS_TXT)
    print("Report   :", REPORT_JSON)


if __name__ == "__main__":
    main()
