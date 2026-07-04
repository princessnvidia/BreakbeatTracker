#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SampleBrain v01 - render_partition_to_audio_v01.py

But :
Transformer les partitions GrooveBrain en audio.

Entrées :
    dataset/slices_manifest.json
    exports/groovebrain_partitions/generated_partitions_v01.txt

Sorties :
    exports/samplebrain_audio/samplebrain_render_001.wav
    exports/samplebrain_audio/samplebrain_render_002.wav
    ...

Principe :
- GrooveBrain génère une partition propre : K / S / g / H / p
- SampleBrain choisit des samples dans le dataset
- On rend l'audio à 150 BPM

Important :
Cette v01 utilise encore les slices existantes, mais de manière plus propre :
- un seul événement par step/instrument
- longueurs contrôlées
- volumes contrôlés
- pas de mutation chaotique
"""

from pathlib import Path
import argparse
import json
import random
import re
import sys

import numpy as np
import soundfile as sf

DATASET = Path("dataset")
SLICES_MANIFEST = DATASET / "slices_manifest.json"

DEFAULT_PARTITIONS = Path("exports/groovebrain_partitions/generated_partitions_v01.txt")
OUT_DIR = Path("exports/samplebrain_audio")

SR = 44100
STEPS = 32

SYMBOL_TO_LABEL = {
    "K": "kick",
    "S": "snare",
    "g": "ghost",
    "H": "hat",
    "p": "perc",
}


def normalize(y):
    m = np.max(np.abs(y)) if len(y) else 0
    return y if m <= 0 else y / m * 0.95


def load_manifest():
    if not SLICES_MANIFEST.exists():
        print("Fichier manquant :", SLICES_MANIFEST)
        sys.exit(1)
    return json.loads(SLICES_MANIFEST.read_text(encoding="utf-8"))


def make_pools(items, source_name=None):
    pools = {label: [] for label in ["kick", "snare", "ghost", "hat", "perc"]}

    for item in items:
        source = item.get("source", "")
        if source_name and source_name.lower() not in source.lower():
            continue

        label = item.get("label", "perc")
        if label not in pools:
            label = "perc"

        pools[label].append(item)

    return pools


def pools_are_usable(pools):
    # Il faut au moins kick ou snare, et quelques hats/ghost/percs.
    return sum(len(v) for v in pools.values()) > 0 and (
        len(pools.get("kick", [])) > 0 or len(pools.get("snare", [])) > 0
    )


def parse_partitions(path):
    if not path.exists():
        print("Fichier partition introuvable :", path)
        print("Lance d'abord : python generator/generate_groove_partition_v01.py")
        sys.exit(1)

    text = path.read_text(encoding="utf-8").splitlines()

    partitions = []
    current = None

    for line in text:
        line = line.rstrip("\n")

        if line.startswith("VARIATION"):
            if current:
                partitions.append(current)
            current = {"name": line.strip(), "layers": {}}
            continue

        if current is None:
            continue

        # Format :
        # KICK : K.......|...
        m = re.match(r"^(KICK|SNARE|GHOST|HAT|PERC)\s*:\s*([KSHgp\.\|]+)", line)
        if m:
            name = m.group(1).lower()
            row = m.group(2).replace("|", "")
            if len(row) == STEPS:
                current["layers"][name] = row

    if current:
        partitions.append(current)

    return partitions


def choose_sample(pools, label):
    fallbacks = {
        "kick": ["kick", "perc", "snare", "ghost", "hat"],
        "snare": ["snare", "perc", "ghost", "kick", "hat"],
        "ghost": ["ghost", "snare", "perc", "hat"],
        "hat": ["hat", "ghost", "perc"],
        "perc": ["perc", "ghost", "hat", "snare"],
    }

    for key in fallbacks[label]:
        if pools.get(key):
            return random.choice(pools[key])
    return None


def read_sample(item):
    y, sr = sf.read(item["slice_file"], dtype="float32")

    if y.ndim > 1:
        y = y.mean(axis=1)

    # Si sample pas à 44100, on évite resampling pour l'instant.
    # La majorité de notre dataset est écrite en 44100 par le builder.
    return y


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


def trim_sample(y, label, step_samples):
    # Pour éviter le fouillis, on limite chaque famille.
    max_steps = {
        "kick": 2.4,
        "snare": 2.2,
        "ghost": 1.1,
        "hat": 0.75,
        "perc": 1.0,
    }.get(label, 1.0)

    max_len = int(step_samples * max_steps)
    y = y[:max_len]
    return apply_fade(y)


def render_partition(part, pools, bpm, out_wav):
    step_samples = int(SR * (60.0 / bpm / 4.0))
    total = step_samples * STEPS

    out = np.zeros(total + SR, dtype=np.float32)

    layers = part["layers"]

    events = []

    def add_layer(layer_name, symbol, label, gain):
        row = layers.get(layer_name, "." * STEPS)
        for step, char in enumerate(row):
            if char == symbol:
                events.append((step, label, gain))

    add_layer("kick", "K", "kick", 0.95)
    add_layer("snare", "S", "snare", 0.88)
    add_layer("ghost", "g", "ghost", 0.25)
    add_layer("hat", "H", "hat", 0.22)
    add_layer("perc", "p", "perc", 0.20)

    # On rend dans un ordre stable : kick/snare avant petits éléments.
    priority = {"kick": 0, "snare": 1, "ghost": 2, "hat": 3, "perc": 4}
    events.sort(key=lambda x: (x[0], priority.get(x[1], 9)))

    for step, label, gain in events:
        item = choose_sample(pools, label)
        if item is None:
            continue

        y = read_sample(item)
        y = trim_sample(y, label, step_samples)

        # Variation douce de volume, pas de mutation temporelle chaotique.
        if label == "ghost":
            local_gain = gain * random.uniform(0.65, 1.15)
        elif label == "hat":
            local_gain = gain * random.uniform(0.75, 1.05)
        else:
            local_gain = gain * random.uniform(0.92, 1.05)

        start = step * step_samples

        # Micro-humanisation très faible seulement sur ghosts/hats.
        if label in ["ghost", "hat", "perc"]:
            start += random.randint(-int(step_samples * 0.08), int(step_samples * 0.08))

        if start < 0:
            continue

        end = min(start + len(y), len(out))
        if end > start:
            out[start:end] += y[:end-start] * local_gain

    sf.write(out_wav, normalize(out), SR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", default=str(DEFAULT_PARTITIONS))
    ap.add_argument("--source-name", default=None,
                    help="Optionnel : utiliser les slices d'un break précis, ex: London, Night, Amen.")
    ap.add_argument("--bpm", type=float, default=150.0)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    items = load_manifest()

    pools = make_pools(items, source_name=args.source_name)

    if not pools_are_usable(pools):
        print("Pools inutilisables avec source-name =", args.source_name)
        print("Essaie sans --source-name ou avec un autre nom de break.")
        sys.exit(1)

    print("Pools samples :")
    for k, v in pools.items():
        print(f"  {k:6}: {len(v)}")

    partitions = parse_partitions(Path(args.partitions))

    if not partitions:
        print("Aucune partition trouvée dans :", args.partitions)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    count = min(args.count, len(partitions))

    for i, part in enumerate(partitions[:count], start=1):
        suffix = ""
        if args.source_name:
            suffix = "_" + args.source_name.replace(" ", "_")

        out = OUT_DIR / f"samplebrain_render{suffix}_{i:03d}.wav"
        render_partition(part, pools, args.bpm, out)
        print("Export :", out)

    print("")
    print("Terminé :", OUT_DIR)


if __name__ == "__main__":
    main()
