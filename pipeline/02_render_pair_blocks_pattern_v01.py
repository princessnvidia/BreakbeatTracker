#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_render_pair_blocks_pattern_v01.py

Rend un break complet à partir des 8 blocs générés par :

    python pipeline/01_find_pair_blocks_v01.py --source "Amen" --block-ms 220
ou
    python pipeline/01_find_pair_blocks_v01.py --source "Amen" --block-ms 280

Pattern verrouillé :
    kick hihat snare hihat hihat kick snare hihat

En notation :
    K H S H | H K S H

Mais comme on travaille avec 8 blocs audio de 2 cases :
    pair_00 -> K
    pair_01 -> H
    pair_02 -> S
    pair_03 -> H
    pair_04 -> H
    pair_05 -> K
    pair_06 -> S
    pair_07 -> H

Ce script :
- lit dataset/pair_blocks/<break>_pair_blocks_v01.json
- charge les 8 blocs WAV
- rend une base complète de plusieurs répétitions
- génère des variations propres en répétant/remplaçant des blocs du même rôle
- pas de .txt
- WAV + metadata JSON seulement

Usage :
    python pipeline/02_render_pair_blocks_pattern_v01.py --source "Amen"

Plus long :
    python pipeline/02_render_pair_blocks_pattern_v01.py --source "Amen" --bars 16 --count 16

Plus de variations :
    python pipeline/02_render_pair_blocks_pattern_v01.py --source "Amen" --mutation 0.25 --count 32
"""

from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import soundfile as sf


PAIR_BLOCKS_DIR = Path("dataset/pair_blocks")
OUT_DIR = Path("exports/pair_blocks_pattern_v01")
SR = 44100

SHEET8 = [
    ("kick", 0),
    ("hat", 1),
    ("snare", 2),
    ("hat", 3),
    ("hat", 4),
    ("kick", 5),
    ("snare", 6),
    ("hat", 7),
]


def safe_source_name(source_query):
    files = sorted(PAIR_BLOCKS_DIR.glob("*_pair_blocks_v01.json"))
    matches = [p for p in files if source_query.lower() in p.name.lower()]

    if not matches:
        print(f"Aucun pair_blocks JSON trouvé pour : {source_query}")
        print("Lance d'abord :")
        print(f'  python pipeline/01_find_pair_blocks_v01.py --source "{source_query}" --block-ms 220')
        sys.exit(1)

    return matches[0]


def normalize(y, peak=0.95):
    m = np.max(np.abs(y)) if len(y) else 0
    return y if m <= 1e-9 else y / m * peak


def fade(y, ms=2):
    if len(y) < 16:
        return y
    n = min(int(SR * ms / 1000), len(y) // 4)
    if n <= 1:
        return y
    y = y.copy()
    ramp = np.linspace(0, 1, n)
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def load_wav(path):
    audio, sr = sf.read(path, always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    if sr != SR:
        print(f"Sample rate inattendu {sr} pour {path}, attendu {SR}")
        sys.exit(1)

    return fade(normalize(audio), ms=2)


def load_pair_blocks(pair_json):
    meta = json.loads(pair_json.read_text(encoding="utf-8"))

    blocks = []

    for block in meta["blocks"]:
        audio_path = Path(block["audio_path"])

        if not audio_path.exists():
            print("Bloc introuvable :", audio_path)
            sys.exit(1)

        audio = load_wav(audio_path)

        blocks.append({
            "pair": int(block["pair"]),
            "audio_path": str(audio_path),
            "audio": audio,
            "duration_ms": float(block["duration_ms"]),
            "role": SHEET8[int(block["pair"])][0],
        })

    if len(blocks) != 8:
        print(f"Erreur : attendu 8 blocs, trouvé {len(blocks)}.")
        sys.exit(1)

    blocks = sorted(blocks, key=lambda b: b["pair"])
    return meta, blocks


def render_order(blocks, order):
    chunks = [blocks[i]["audio"] for i in order]
    return normalize(np.concatenate(chunks))


def base_order_for_bars(bars):
    return list(range(8)) * bars


def role_pools():
    pools = {"kick": [], "snare": [], "hat": []}
    for role, pair in SHEET8:
        pools[role].append(pair)
    return pools


def mutate_bar_order(base_bar, mutation=0.16, fill_chance=0.18):
    """
    Mutations propres :
    - la position garde son rôle musical
    - une position kick choisit parmi les pairs kick : 0 ou 5
    - une position snare choisit parmi les pairs snare : 2 ou 6
    - une position hat choisit parmi les pairs hat : 1,3,4,7
    - fill final léger sur les hats de fin
    """
    pools = role_pools()
    out = base_bar[:]

    for pos, pair in enumerate(base_bar):
        role = SHEET8[pos][0]

        if random.random() > mutation:
            continue

        candidates = pools[role]

        if candidates:
            out[pos] = random.choice(candidates)

    # fill final sur les deux derniers slots, en gardant les rôles
    if random.random() < fill_chance:
        for pos in [6, 7]:
            role = SHEET8[pos][0]
            candidates = pools[role]
            if candidates and random.random() < 0.55:
                out[pos] = random.choice(candidates)

    return out


def make_long_order(bars=8, mutation=0.16, fill_chance=0.18):
    order = []
    bar_orders = []

    base_bar = list(range(8))

    for bar in range(1, bars + 1):
        local_mut = mutation
        local_fill = fill_chance

        # début de phrase plus stable
        if bar in [1, 5, 9, 13]:
            local_mut *= 0.40
            local_fill *= 0.30

        # fin de 4 mesures : plus de fill
        if bar % 4 == 0:
            local_mut *= 1.25
            local_fill *= 1.7

        bar_order = mutate_bar_order(
            base_bar,
            mutation=local_mut,
            fill_chance=local_fill,
        )

        order.extend(bar_order)
        bar_orders.append({
            "bar": bar,
            "order": [int(x) for x in bar_order],
            "mutation": float(local_mut),
            "fill_chance": float(local_fill),
        })

    return order, bar_orders


def export_ableton_samples(blocks, outdir):
    """
    Prépare déjà des samples nommés pour un futur Drum Rack.
    MIDI notes standards :
      36 kick
      37 hat
      38 snare
      39 hat
      ...
    """
    samples_dir = outdir / "Samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    note_map = [36, 37, 38, 39, 40, 41, 42, 43]
    exported = []

    for block in blocks:
        pair = block["pair"]
        role = block["role"]
        note = note_map[pair]
        wav = samples_dir / f"{note}_pair_{pair:02d}_{role}.wav"
        sf.write(wav, normalize(block["audio"]), SR)

        exported.append({
            "pair": int(pair),
            "role": role,
            "midi_note": int(note),
            "audio_path": str(wav),
        })

    return exported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Amen")
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--mutation", type=float, default=0.16)
    parser.add_argument("--fill-chance", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    pair_json = safe_source_name(args.source)
    meta, blocks = load_pair_blocks(pair_json)

    safe = pair_json.stem.replace("_pair_blocks_v01", "")
    outdir = OUT_DIR / safe / f"{args.bars}bars"
    outdir.mkdir(parents=True, exist_ok=True)

    print("Pair blocks :", pair_json)
    print("Pattern     : kick hihat snare hihat | hihat kick snare hihat")

    ableton_samples = export_ableton_samples(blocks, outdir)

    # base entière
    base_order = base_order_for_bars(args.bars)
    base_audio = render_order(blocks, base_order)

    base_wav = outdir / f"{safe}_pattern_base_{args.bars}bars.wav"
    sf.write(base_wav, base_audio, SR)

    print("Base :", base_wav)

    renders = []

    for i in range(1, args.count + 1):
        order, bar_orders = make_long_order(
            bars=args.bars,
            mutation=args.mutation,
            fill_chance=args.fill_chance,
        )

        audio = render_order(blocks, order)

        wav = outdir / f"{safe}_pattern_variation_{args.bars}bars_{i:03d}.wav"
        sf.write(wav, audio, SR)

        renders.append({
            "index": int(i),
            "wav": str(wav),
            "order": [int(x) for x in order],
            "bars": bar_orders,
        })

        print("Export :", wav)

    metadata = {
        "version": "pair_blocks_pattern_v01",
        "source_pair_blocks": str(pair_json),
        "source_audio": meta.get("source"),
        "pattern_words": ["kick", "hihat", "snare", "hihat", "hihat", "kick", "snare", "hihat"],
        "sheet8": [
            {"role": role, "pair": int(pair)}
            for role, pair in SHEET8
        ],
        "bars": int(args.bars),
        "mutation": float(args.mutation),
        "fill_chance": float(args.fill_chance),
        "base_wav": str(base_wav),
        "ableton_samples": ableton_samples,
        "renders": renders,
    }

    metadata_path = outdir / "metadata_pair_blocks_pattern_v01.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("")
    print("Dossier :", outdir)
    print("Samples Ableton :", outdir / "Samples")
    print("Metadata :", metadata_path)


if __name__ == "__main__":
    main()
