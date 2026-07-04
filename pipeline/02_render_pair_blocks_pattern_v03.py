#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_render_pair_blocks_pattern_v03.py

Renderer pour pair_blocks_v02 avec cohérence des snares.

Amélioration v03 :
- chaque phrase choisit une snare principale
- les snares restent majoritairement répétées au même endroit/source
- variations de snare autorisées seulement parfois :
    * réponse en deuxième moitié
    * fill en fin de boucle
- les hats/kicks peuvent varier plus librement
- chaque boucle de 8 blocs reçoit toujours une variation
- 4 familles de variations

Entrée :
    dataset/pair_blocks_v02/<break>_pair_blocks_v02.json

Pattern :
    kick hihat snare hihat | hihat kick snare hihat
    0    1     2     3    | 4     5    6     7

Usage :
    python pipeline/02_render_pair_blocks_pattern_v03.py --source "Amen"

Plus cohérent :
    python pipeline/02_render_pair_blocks_pattern_v03.py --source "Amen" --snare-stability 0.92

Plus de variations snare :
    python pipeline/02_render_pair_blocks_pattern_v03.py --source "Amen" --snare-stability 0.70
"""

from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import soundfile as sf


PAIR_BLOCKS_DIR = Path("dataset/pair_blocks_v02")
OUT_DIR = Path("exports/pair_blocks_pattern_v03")
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


def choose_snare_memory():
    """
    Choisit une snare principale et une snare de réponse.
    """
    main = random.choice(POOLS["snare"])
    alt = 6 if main == 2 else 2
    return {
        "main": main,
        "alt": alt,
    }


def apply_snare_coherence(order, snare_memory, loop_index, snare_stability, fill=False):
    """
    Force la cohérence des positions snare :
    positions 2 et 6.

    - position 2 : snare principale quasi toujours
    - position 6 : snare principale ou réponse, selon stabilité/fill
    """
    # Snare forte principale, très stable
    if random.random() < snare_stability:
        order[2] = snare_memory["main"]
    else:
        order[2] = random.choice(POOLS["snare"])

    # Snare de réponse : répétition majoritaire, variation parfois
    if fill:
        # en fill, la fin peut répondre
        order[6] = random.choice([snare_memory["main"], snare_memory["alt"], snare_memory["alt"]])
    else:
        if random.random() < snare_stability:
            order[6] = snare_memory["main"]
        else:
            order[6] = snare_memory["alt"]

    # Tous les 4 loops, légère réponse autorisée
    if loop_index % 4 == 0 and random.random() < 0.45:
        order[6] = snare_memory["alt"]

    return order


def variation_1_same_role_swap(snare_memory, loop_index, snare_stability):
    order = BASE8[:]

    for pos in [0, 1, 3, 4, 5, 7]:
        role = ROLE_BY_PAIR[pos]
        if random.random() < 0.35:
            order[pos] = random.choice(POOLS[role])

    order = apply_snare_coherence(
        order,
        snare_memory,
        loop_index,
        snare_stability,
        fill=False,
    )

    return order, "same_role_swap_snare_locked"


def variation_2_second_half_answer(snare_memory, loop_index, snare_stability):
    order = BASE8[:]

    for pos in [4, 5, 7]:
        role = ROLE_BY_PAIR[pos]
        if random.random() < 0.55:
            order[pos] = random.choice(POOLS[role])

    order = apply_snare_coherence(
        order,
        snare_memory,
        loop_index,
        snare_stability,
        fill=False,
    )

    # réponse de snare parfois en deuxième moitié
    if random.random() < 0.35:
        order[6] = snare_memory["alt"]

    return order, "second_half_answer_snare_response"


def variation_3_hat_shuffle(snare_memory, loop_index, snare_stability):
    order = BASE8[:]

    for pos in [1, 3, 4, 7]:
        if random.random() < 0.75:
            order[pos] = random.choice(POOLS["hat"])

    # kicks stables, snares stables
    order[0] = 0 if random.random() < 0.70 else 5
    order[5] = 5 if random.random() < 0.70 else 0

    order = apply_snare_coherence(
        order,
        snare_memory,
        loop_index,
        snare_stability,
        fill=False,
    )

    return order, "hat_shuffle_snare_locked"


def variation_4_fill_end(snare_memory, loop_index, snare_stability):
    order = BASE8[:]

    for pos in [4, 5, 7]:
        role = ROLE_BY_PAIR[pos]
        if random.random() < 0.60:
            order[pos] = random.choice(POOLS[role])

    order = apply_snare_coherence(
        order,
        snare_memory,
        loop_index,
        snare_stability,
        fill=True,
    )

    return order, "fill_end_snare_response"


VARIATIONS = [
    variation_1_same_role_swap,
    variation_2_second_half_answer,
    variation_3_hat_shuffle,
    variation_4_fill_end,
]


def make_phrase(loops=8, snare_stability=0.85):
    """
    Chaque loop de 8 blocs reçoit une variation,
    mais toutes partagent une mémoire de snare.
    """
    full_order = []
    loop_infos = []
    snare_memory = choose_snare_memory()

    for loop in range(loops):
        if loop == 0:
            order = BASE8[:]
            name = "base_snare_memory"
            order = apply_snare_coherence(
                order,
                snare_memory,
                loop + 1,
                snare_stability=1.0,
                fill=False,
            )
        else:
            fn = VARIATIONS[(loop - 1) % len(VARIATIONS)]
            order, name = fn(
                snare_memory,
                loop + 1,
                snare_stability,
            )

        full_order.extend(order)

        loop_infos.append({
            "loop": int(loop + 1),
            "variation": name,
            "snare_main": int(snare_memory["main"]),
            "snare_alt": int(snare_memory["alt"]),
            "order": [int(x) for x in order],
        })

    return full_order, loop_infos, snare_memory


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
    parser.add_argument("--loops", type=int, default=8)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--snare-stability", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    pair_json = find_pair_json(args.source)
    meta, blocks = load_pair_blocks(pair_json)

    safe = pair_json.stem.replace("_pair_blocks_v02", "")
    outdir = OUT_DIR / safe / f"{args.loops}loops_snare{args.snare_stability}"
    outdir.mkdir(parents=True, exist_ok=True)

    print("Pair blocks :", pair_json)
    print("Pattern     : kick hihat snare hihat | hihat kick snare hihat")
    print("Snare       : mémoire + répétition cohérente")

    ableton_samples = export_ableton_samples(blocks, outdir)

    base_order = BASE8 * args.loops
    base_audio = render_order(blocks, base_order)
    base_wav = outdir / f"{safe}_base_{args.loops}loops.wav"
    sf.write(base_wav, base_audio, SR)

    print("Base :", base_wav)

    renders = []

    for i in range(1, args.count + 1):
        order, loop_infos, snare_memory = make_phrase(
            loops=args.loops,
            snare_stability=args.snare_stability,
        )
        audio = render_order(blocks, order)

        wav = outdir / f"{safe}_snarecoherent_{args.loops}loops_{i:03d}.wav"
        sf.write(wav, audio, SR)

        renders.append({
            "index": int(i),
            "wav": str(wav),
            "snare_memory": {
                "main": int(snare_memory["main"]),
                "alt": int(snare_memory["alt"]),
            },
            "order": [int(x) for x in order],
            "loops": loop_infos,
        })

        print("Export :", wav)

    metadata = {
        "version": "pair_blocks_pattern_v03_snare_coherent",
        "source_pair_blocks": str(pair_json),
        "source_audio": meta.get("source"),
        "pattern_words": ["kick", "hihat", "snare", "hihat", "hihat", "kick", "snare", "hihat"],
        "base8": BASE8,
        "roles": {str(k): v for k, v in ROLE_BY_PAIR.items()},
        "snare_rule": {
            "main_positions": [2, 6],
            "snare_pool": POOLS["snare"],
            "stability": float(args.snare_stability),
            "description": "one main snare repeated across loops, alt snare only for response/fill",
        },
        "loops": int(args.loops),
        "base_wav": str(base_wav),
        "ableton_samples": ableton_samples,
        "renders": renders,
    }

    metadata_path = outdir / "metadata_pair_blocks_pattern_v03.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("")
    print("Dossier :", outdir)
    print("Samples Ableton :", outdir / "Samples")
    print("Metadata :", metadata_path)


if __name__ == "__main__":
    main()
