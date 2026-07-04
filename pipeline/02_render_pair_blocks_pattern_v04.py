#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_render_pair_blocks_pattern_v04.py

Renderer pour pair_blocks_v02.

Base conservée depuis v02 :
    kick hihat snare hihat | hihat kick snare hihat

Nouveauté v04 :
- revient au feeling plus libre de v02
- ajoute des blocs hybrides de 2 cases :
    KH = kick + hihat
    SH = snare + hihat
    HH = hihat + hihat
- les kicks peuvent devenir :
    case 1 = kick
    case 2 = hihat
- les snares gardent une cohérence de placement, sans être figées
- chaque boucle de 8 blocs a une variation différente
- pas de .txt
- WAV + metadata JSON seulement

Usage :
    python pipeline/02_render_pair_blocks_pattern_v04.py --source "Amen" --loops 8 --count 16

Plus de blocs KH :
    python pipeline/02_render_pair_blocks_pattern_v04.py --source "Amen" --kh-chance 0.45

Plus sage :
    python pipeline/02_render_pair_blocks_pattern_v04.py --source "Amen" --mutation 0.12 --kh-chance 0.20
"""

from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import soundfile as sf


PAIR_BLOCKS_DIR = Path("dataset/pair_blocks_v02")
OUT_DIR = Path("exports/pair_blocks_pattern_v04")
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


def half_audio(audio, part):
    mid = len(audio) // 2
    if part == "first":
        return audio[:mid]
    return audio[mid:]


def make_hybrid_block(blocks, left_pair, right_pair):
    """
    Crée un bloc 2 cases :
      première moitié = left_pair
      deuxième moitié = right_pair
    """
    left = half_audio(blocks[left_pair]["audio"], "first")
    right = half_audio(blocks[right_pair]["audio"], "second")
    out = np.concatenate([left, right])
    return fade(normalize(out), ms=2)


def render_tokens(blocks, tokens):
    """
    Token possible :
      int -> bloc entier
      {"type":"hybrid", "left":0, "right":1, "name":"KH"}
    """
    chunks = []

    for token in tokens:
        if isinstance(token, int):
            chunks.append(blocks[token]["audio"])
        else:
            chunks.append(make_hybrid_block(blocks, token["left"], token["right"]))

    return normalize(np.concatenate(chunks))


def token_to_meta(token):
    if isinstance(token, int):
        return {"type": "pair", "pair": int(token), "role": ROLE_BY_PAIR.get(token, "hat")}
    return {
        "type": "hybrid",
        "name": token["name"],
        "left": int(token["left"]),
        "right": int(token["right"]),
    }


def choose(role):
    return random.choice(POOLS[role])


def kh_token():
    return {
        "type": "hybrid",
        "name": "KH",
        "left": choose("kick"),
        "right": choose("hat"),
    }


def sh_token():
    return {
        "type": "hybrid",
        "name": "SH",
        "left": choose("snare"),
        "right": choose("hat"),
    }


def hh_token():
    return {
        "type": "hybrid",
        "name": "HH",
        "left": choose("hat"),
        "right": choose("hat"),
    }


def base_loop_tokens():
    return BASE8[:]


def variation_1_same_role_swap(mutation, kh_chance):
    tokens = base_loop_tokens()

    for pos in range(8):
        role = ROLE_BY_PAIR[pos]

        if role == "kick" and random.random() < kh_chance:
            tokens[pos] = kh_token()
            continue

        if role == "snare" and random.random() < kh_chance * 0.45:
            tokens[pos] = sh_token()
            continue

        if random.random() < mutation:
            tokens[pos] = choose(role)

    return tokens, "same_role_swap_plus_kh"


def variation_2_second_half_answer(mutation, kh_chance):
    tokens = base_loop_tokens()

    for pos in [4, 5, 6, 7]:
        role = ROLE_BY_PAIR[pos]

        if role == "kick" and random.random() < kh_chance:
            tokens[pos] = kh_token()
        elif role == "snare" and random.random() < kh_chance * 0.45:
            tokens[pos] = sh_token()
        elif role == "hat" and random.random() < mutation:
            tokens[pos] = choose("hat")
        elif random.random() < mutation:
            tokens[pos] = choose(role)

    return tokens, "second_half_answer_plus_kh"


def variation_3_hat_shuffle(mutation, kh_chance):
    tokens = base_loop_tokens()

    for pos in [1, 3, 4, 7]:
        if random.random() < 0.70:
            tokens[pos] = choose("hat")
        if random.random() < mutation * 0.55:
            tokens[pos] = hh_token()

    # kicks peuvent devenir KH
    for pos in [0, 5]:
        if random.random() < kh_chance:
            tokens[pos] = kh_token()
        elif random.random() < mutation:
            tokens[pos] = choose("kick")

    # snares peu modifiées
    for pos in [2, 6]:
        if random.random() < mutation * 0.35:
            tokens[pos] = choose("snare")

    return tokens, "hat_shuffle_plus_kh"


def variation_4_fill_end(mutation, kh_chance):
    tokens = base_loop_tokens()

    # fin de boucle plus travaillée
    for pos in [4, 5, 6, 7]:
        role = ROLE_BY_PAIR[pos]

        if role == "kick":
            tokens[pos] = kh_token() if random.random() < kh_chance * 1.25 else choose("kick")
        elif role == "snare":
            tokens[pos] = sh_token() if random.random() < kh_chance * 0.65 else choose("snare")
        elif role == "hat":
            if random.random() < 0.45:
                tokens[pos] = hh_token()
            else:
                tokens[pos] = choose("hat")

    # garde le début lisible
    for pos in [0, 1, 2, 3]:
        role = ROLE_BY_PAIR[pos]
        if role == "kick" and random.random() < kh_chance * 0.55:
            tokens[pos] = kh_token()
        elif random.random() < mutation * 0.35:
            tokens[pos] = choose(role)

    return tokens, "fill_end_plus_kh"


VARIATIONS = [
    variation_1_same_role_swap,
    variation_2_second_half_answer,
    variation_3_hat_shuffle,
    variation_4_fill_end,
]


def soften_snare_placement(tokens, snare_repeat=0.78):
    """
    Cohérence légère :
    - positions 2 et 6 restent des snares la majorité du temps
    - source de snare peut varier
    - pas de mémoire stricte
    """
    for pos in [2, 6]:
        if random.random() < snare_repeat:
            if isinstance(tokens[pos], dict):
                # si c'est SH, c'est déjà OK
                continue
            tokens[pos] = choose("snare")
    return tokens


def make_phrase(loops=8, mutation=0.18, kh_chance=0.30, snare_repeat=0.78):
    full_tokens = []
    loop_infos = []

    for loop in range(loops):
        if loop == 0:
            tokens = base_loop_tokens()
            name = "base"
        else:
            fn = VARIATIONS[(loop - 1) % len(VARIATIONS)]
            tokens, name = fn(mutation, kh_chance)
            tokens = soften_snare_placement(tokens, snare_repeat)

        full_tokens.extend(tokens)

        loop_infos.append({
            "loop": int(loop + 1),
            "variation": name,
            "tokens": [token_to_meta(t) for t in tokens],
        })

    return full_tokens, loop_infos


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

    # Exporte aussi des hybrides utiles pour Ableton.
    hybrid_dir = outdir / "Samples_Hybrid"
    hybrid_dir.mkdir(parents=True, exist_ok=True)

    hybrids = []
    templates = [
        ("KH", choose("kick"), choose("hat")),
        ("SH", choose("snare"), choose("hat")),
        ("HH", choose("hat"), choose("hat")),
    ]

    for idx, (name, left, right) in enumerate(templates):
        wav = hybrid_dir / f"hybrid_{idx:02d}_{name}_pair{left}_pair{right}.wav"
        sf.write(wav, make_hybrid_block(blocks, left, right), SR)
        hybrids.append({
            "name": name,
            "left": int(left),
            "right": int(right),
            "audio_path": str(wav),
        })

    return exported, hybrids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Amen")
    parser.add_argument("--loops", type=int, default=8)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--mutation", type=float, default=0.18)
    parser.add_argument("--kh-chance", type=float, default=0.30)
    parser.add_argument("--snare-repeat", type=float, default=0.78)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    pair_json = find_pair_json(args.source)
    meta, blocks = load_pair_blocks(pair_json)

    safe = pair_json.stem.replace("_pair_blocks_v02", "")
    outdir = OUT_DIR / safe / f"{args.loops}loops_kh{args.kh_chance}"
    outdir.mkdir(parents=True, exist_ok=True)

    print("Pair blocks :", pair_json)
    print("Pattern     : kick hihat snare hihat | hihat kick snare hihat")
    print("Nouveau     : blocs KH / SH / HH, snare cohérente légère")

    ableton_samples, hybrid_samples = export_ableton_samples(blocks, outdir)

    base_tokens = BASE8 * args.loops
    base_audio = render_tokens(blocks, base_tokens)
    base_wav = outdir / f"{safe}_base_{args.loops}loops.wav"
    sf.write(base_wav, base_audio, SR)

    print("Base :", base_wav)

    renders = []

    for i in range(1, args.count + 1):
        tokens, loop_infos = make_phrase(
            loops=args.loops,
            mutation=args.mutation,
            kh_chance=args.kh_chance,
            snare_repeat=args.snare_repeat,
        )

        audio = render_tokens(blocks, tokens)

        wav = outdir / f"{safe}_kh_variation_{args.loops}loops_{i:03d}.wav"
        sf.write(wav, audio, SR)

        renders.append({
            "index": int(i),
            "wav": str(wav),
            "tokens": [token_to_meta(t) for t in tokens],
            "loops": loop_infos,
        })

        print("Export :", wav)

    metadata = {
        "version": "pair_blocks_pattern_v04_kh_blocks",
        "source_pair_blocks": str(pair_json),
        "source_audio": meta.get("source"),
        "pattern_words": ["kick", "hihat", "snare", "hihat", "hihat", "kick", "snare", "hihat"],
        "base8": BASE8,
        "roles": {str(k): v for k, v in ROLE_BY_PAIR.items()},
        "hybrid_blocks": ["KH", "SH", "HH"],
        "mutation": float(args.mutation),
        "kh_chance": float(args.kh_chance),
        "snare_repeat": float(args.snare_repeat),
        "loops": int(args.loops),
        "base_wav": str(base_wav),
        "ableton_samples": ableton_samples,
        "hybrid_samples": hybrid_samples,
        "renders": renders,
    }

    metadata_path = outdir / "metadata_pair_blocks_pattern_v04.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("")
    print("Dossier :", outdir)
    print("Samples Ableton :", outdir / "Samples")
    print("Hybrides :", outdir / "Samples_Hybrid")
    print("Metadata :", metadata_path)


if __name__ == "__main__":
    main()
