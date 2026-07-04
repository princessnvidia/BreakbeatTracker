#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_render_pair_blocks_pattern_v02.py

Renderer pour pair_blocks_v02.

Entrée :
    dataset/pair_blocks_v02/<break>_pair_blocks_v02.json

Pattern de base :
    kick hihat snare hihat | hihat kick snare hihat
    soit :
    0 1 2 3 | 4 5 6 7

Règle demandée :
- chaque boucle de 8 blocs a une nouvelle variation
- il y a 4 familles de variations possibles pour chaque boucle
- pas de .txt
- WAV + metadata JSON seulement

Usage :
    python pipeline/02_render_pair_blocks_pattern_v02.py --source "Amen"

Batch :
    for b in "Amen" "London" "Stepper" "Massive" "Camo"; do
      python pipeline/01_find_pair_blocks_v02.py --source "$b"
      python pipeline/02_render_pair_blocks_pattern_v02.py --source "$b" --loops 8 --count 8
    done
"""

from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import soundfile as sf


PAIR_BLOCKS_DIR = Path("dataset/pair_blocks_v02")
OUT_DIR = Path("exports/pair_blocks_pattern_v02")
SR = 44100

BASE8 = [0, 1, 2, 3, 4, 5, 6, 7]

ROLE_BY_PAIR = {
    0: "kick",
    1: "hat",
    2: "snare",
    3: "hat",
    4: "hat",
    5: "kick",
    6: "snare",
    7: "hat",
}

POOLS = {
    "kick": [0, 5],
    "snare": [2, 6],
    "hat": [1, 3, 4, 7],
}


def find_pair_json(source_query):
    files = sorted(PAIR_BLOCKS_DIR.glob("*_pair_blocks_v02.json"))
    matches = [p for p in files if source_query.lower() in p.name.lower()]

    if not matches:
        print(f"Aucun pair_blocks_v02 JSON trouvé pour : {source_query}")
        print("Lance d'abord :")
        print(f'  python pipeline/01_find_pair_blocks_v02.py --source "{source_query}"')
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

        pair = int(block["pair"])

        blocks.append({
            "pair": pair,
            "role": ROLE_BY_PAIR.get(pair, "hat"),
            "audio_path": str(audio_path),
            "audio": load_wav(audio_path),
            "duration_ms": float(block["duration_ms"]),
            "start_ms": float(block["start_ms"]),
            "end_ms": float(block["end_ms"]),
        })

    blocks = sorted(blocks, key=lambda b: b["pair"])

    if len(blocks) != 8:
        print(f"Erreur : attendu 8 blocs, trouvé {len(blocks)}.")
        sys.exit(1)

    return meta, blocks


def render_order(blocks, order):
    chunks = [blocks[i]["audio"] for i in order]
    return normalize(np.concatenate(chunks))


def variation_1_same_role_swap():
    """
    Variation 1 :
    remplace quelques slots par un autre bloc du même rôle.
    Très propre.
    """
    order = BASE8[:]

    for pos in range(8):
        role = ROLE_BY_PAIR[pos]
        if random.random() < 0.35:
            order[pos] = random.choice(POOLS[role])

    return order, "same_role_swap"


def variation_2_second_half_answer():
    """
    Variation 2 :
    première moitié stable, deuxième moitié répond différemment.
    """
    order = BASE8[:]

    for pos in [4, 5, 6, 7]:
        role = ROLE_BY_PAIR[pos]
        if random.random() < 0.55:
            order[pos] = random.choice(POOLS[role])

    return order, "second_half_answer"


def variation_3_hat_shuffle():
    """
    Variation 3 :
    kicks/snares gardent l'ossature, hats bougent.
    """
    order = BASE8[:]

    for pos in [1, 3, 4, 7]:
        if random.random() < 0.75:
            order[pos] = random.choice(POOLS["hat"])

    return order, "hat_shuffle"


def variation_4_fill_end():
    """
    Variation 4 :
    fill de fin sur les deux derniers blocs,
    sans casser le rôle snare/hat.
    """
    order = BASE8[:]

    # garde l'idée snare puis hat, mais change leurs sources
    order[6] = random.choice(POOLS["snare"])
    order[7] = random.choice(POOLS["hat"])

    # parfois petit appel juste avant
    if random.random() < 0.45:
        order[5] = random.choice(POOLS["kick"])
    if random.random() < 0.55:
        order[4] = random.choice(POOLS["hat"])

    return order, "fill_end"


VARIATIONS = [
    variation_1_same_role_swap,
    variation_2_second_half_answer,
    variation_3_hat_shuffle,
    variation_4_fill_end,
]


def make_phrase(loops=8):
    """
    Chaque boucle de 8 reçoit une variation.
    On alterne les 4 familles pour éviter que tout soit pareil.
    """
    full_order = []
    loop_infos = []

    for loop in range(loops):
        fn = VARIATIONS[loop % len(VARIATIONS)]

        # première boucle très proche de la base
        if loop == 0:
            order = BASE8[:]
            name = "base"
        else:
            order, name = fn()

        full_order.extend(order)

        loop_infos.append({
            "loop": int(loop + 1),
            "variation": name,
            "order": [int(x) for x in order],
        })

    return full_order, loop_infos


def export_ableton_samples(blocks, outdir):
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
    parser.add_argument("--loops", type=int, default=8, help="Nombre de boucles de 8 blocs.")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    pair_json = find_pair_json(args.source)
    meta, blocks = load_pair_blocks(pair_json)

    safe = pair_json.stem.replace("_pair_blocks_v02", "")
    outdir = OUT_DIR / safe / f"{args.loops}loops"
    outdir.mkdir(parents=True, exist_ok=True)

    print("Pair blocks :", pair_json)
    print("Pattern     : kick hihat snare hihat | hihat kick snare hihat")
    print("Règle       : une nouvelle variation à chaque boucle de 8, 4 familles")

    ableton_samples = export_ableton_samples(blocks, outdir)

    base_order = BASE8 * args.loops
    base_audio = render_order(blocks, base_order)
    base_wav = outdir / f"{safe}_base_{args.loops}loops.wav"
    sf.write(base_wav, base_audio, SR)

    print("Base :", base_wav)

    renders = []

    for i in range(1, args.count + 1):
        order, loop_infos = make_phrase(loops=args.loops)
        audio = render_order(blocks, order)

        wav = outdir / f"{safe}_variation_{args.loops}loops_{i:03d}.wav"
        sf.write(wav, audio, SR)

        renders.append({
            "index": int(i),
            "wav": str(wav),
            "order": [int(x) for x in order],
            "loops": loop_infos,
        })

        print("Export :", wav)

    metadata = {
        "version": "pair_blocks_pattern_v02",
        "source_pair_blocks": str(pair_json),
        "source_audio": meta.get("source"),
        "pattern_words": ["kick", "hihat", "snare", "hihat", "hihat", "kick", "snare", "hihat"],
        "base8": BASE8,
        "roles": {str(k): v for k, v in ROLE_BY_PAIR.items()},
        "variation_families": [
            "same_role_swap",
            "second_half_answer",
            "hat_shuffle",
            "fill_end",
        ],
        "loops": int(args.loops),
        "base_wav": str(base_wav),
        "ableton_samples": ableton_samples,
        "renders": renders,
    }

    metadata_path = outdir / "metadata_pair_blocks_pattern_v02.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("")
    print("Dossier :", outdir)
    print("Samples Ableton :", outdir / "Samples")
    print("Metadata :", metadata_path)


if __name__ == "__main__":
    main()
