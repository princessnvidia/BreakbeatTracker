#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_find_pair_blocks_v01.py

Découpe dédiée au mode tracker :
- grille 16
- découpe en 8 blocs de 2 cases
- exactement 8 WAV
- cherche automatiquement le meilleur offset
- vérifie les doublons audio
- pas de génération, juste diagnostic propre

Sorties :
    dataset/pair_blocks/<break>_pair_blocks_v01.json
    dataset/pair_blocks/<break>/pair_blocks/*.wav
    dataset/pair_blocks/<break>/pair_blocks_reconstruction.wav

Usage :
    python pipeline/01_find_pair_blocks_v01.py --source "Amen"

Si les blocs sont trop courts/longs :
    python pipeline/01_find_pair_blocks_v01.py --source "Amen" --block-ms 220
    python pipeline/01_find_pair_blocks_v01.py --source "Amen" --block-ms 280

Tester plusieurs offsets :
    python pipeline/01_find_pair_blocks_v01.py --source "Amen" --offset-search-ms 300
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:
    print("librosa manquant : python -m pip install librosa soundfile numpy")
    sys.exit(1)


BREAKS_DIR = Path("breaks")
OUT_DIR = Path("dataset/pair_blocks")
SR = 44100
AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".mp3"}


def find_source(name):
    files = sorted(
        p for p in BREAKS_DIR.rglob("*")
        if p.suffix.lower() in AUDIO_EXTS and not p.name.endswith(".asd")
    )
    matches = [p for p in files if name.lower() in p.name.lower()]
    if not matches:
        print(f"Aucun break trouvé pour : {name}")
        sys.exit(1)
    return matches[0]


def safe_name(path):
    return path.stem.replace(" ", "_").replace("'", "").replace("/", "_")


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


def load_audio(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    return normalize(y), sr


def rms(y):
    if len(y) == 0:
        return 0.0
    return float(np.sqrt(np.mean(y * y)))


def audio_similarity(a, b):
    """
    Similarité simple entre deux blocs.
    1.0 = quasi identique.
    """
    n = min(len(a), len(b))
    if n <= 8:
        return 0.0

    a = a[:n]
    b = b[:n]

    ar = np.sqrt(np.mean(a * a)) + 1e-9
    br = np.sqrt(np.mean(b * b)) + 1e-9

    a = a / ar
    b = b / br

    corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return abs(corr)


def onset_score(y, cuts):
    env = librosa.onset.onset_strength(y=y, sr=SR)
    env_samples = librosa.frames_to_samples(np.arange(len(env)))

    score = 0.0

    for c in cuts:
        win = int(SR * 0.035)
        idx = np.where((env_samples >= c - win) & (env_samples <= c + win))[0]
        if len(idx):
            score += float(np.max(env[idx]))

        local = y[c:min(len(y), c + int(SR * 0.030))]
        if len(local):
            score += rms(local) * 6.0

    return score


def make_blocks(y, offset_samples, block_samples):
    blocks = []

    for pair in range(8):
        start = offset_samples + pair * block_samples
        end = start + block_samples

        if start >= len(y):
            audio = np.zeros(block_samples, dtype=np.float32)
        else:
            audio = y[start:min(end, len(y))].copy()
            if len(audio) < block_samples:
                audio = np.concatenate([audio, np.zeros(block_samples - len(audio), dtype=np.float32)])

        audio = fade(audio, ms=2)

        blocks.append({
            "pair": pair,
            "start_sample": int(start),
            "end_sample": int(end),
            "start_ms": float(start / SR * 1000),
            "end_ms": float(end / SR * 1000),
            "duration_ms": float(block_samples / SR * 1000),
            "rms": rms(audio),
            "audio": audio,
        })

    return blocks


def duplicate_report(blocks, threshold=0.985):
    duplicates = []

    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            sim = audio_similarity(blocks[i]["audio"], blocks[j]["audio"])
            if sim >= threshold:
                duplicates.append({
                    "a": int(i),
                    "b": int(j),
                    "similarity": float(sim),
                })

    return duplicates


def score_candidate(y, offset_samples, block_samples, duplicate_penalty=True):
    cuts = [offset_samples + i * block_samples for i in range(8)]
    score = onset_score(y, cuts)

    end = offset_samples + 8 * block_samples
    if end > len(y):
        score -= (end - len(y)) / SR * 50.0

    if duplicate_penalty:
        blocks = make_blocks(y, offset_samples, block_samples)
        dups = duplicate_report(blocks, threshold=0.985)
        score -= len(dups) * 20.0

    return score


def auto_find_pair_blocks(y, block_ms=None, min_block_ms=150, max_block_ms=360, offset_search_ms=300, step_ms=5):
    step_samples = int(SR * step_ms / 1000)
    max_offset = int(SR * offset_search_ms / 1000)

    if block_ms is not None:
        block_values = [int(SR * block_ms / 1000)]
    else:
        block_values = list(range(
            int(SR * min_block_ms / 1000),
            int(SR * max_block_ms / 1000) + 1,
            step_samples
        ))

    best = None

    for block_samples in block_values:
        for offset_samples in range(0, max_offset + 1, step_samples):
            score = score_candidate(y, offset_samples, block_samples)

            if best is None or score > best["score"]:
                best = {
                    "score": float(score),
                    "offset_samples": int(offset_samples),
                    "block_samples": int(block_samples),
                    "offset_ms": float(offset_samples / SR * 1000),
                    "block_ms": float(block_samples / SR * 1000),
                }

    return best


def export_blocks(blocks, outdir):
    block_dir = outdir / "pair_blocks"
    block_dir.mkdir(parents=True, exist_ok=True)

    exported = []

    for block in blocks:
        wav = block_dir / f"pair_{block['pair']:02d}_{int(block['start_ms'])}ms.wav"
        sf.write(wav, normalize(block["audio"]), SR)

        exported.append({
            "pair": int(block["pair"]),
            "audio_path": str(wav),
            "start_sample": int(block["start_sample"]),
            "end_sample": int(block["end_sample"]),
            "start_ms": float(block["start_ms"]),
            "end_ms": float(block["end_ms"]),
            "duration_ms": float(block["duration_ms"]),
            "rms": float(block["rms"]),
        })

    recon = normalize(np.concatenate([b["audio"] for b in blocks]))
    recon_path = outdir / "pair_blocks_reconstruction.wav"
    sf.write(recon_path, recon, SR)

    return exported, str(recon_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Amen")
    parser.add_argument("--block-ms", type=float, default=None)
    parser.add_argument("--min-block-ms", type=float, default=150.0)
    parser.add_argument("--max-block-ms", type=float, default=360.0)
    parser.add_argument("--offset-search-ms", type=float, default=300.0)
    parser.add_argument("--step-ms", type=float, default=5.0)
    parser.add_argument("--duplicate-threshold", type=float, default=0.985)
    args = parser.parse_args()

    source = find_source(args.source)
    print("Source :", source)

    y, sr = load_audio(source)
    safe = safe_name(source)

    best = auto_find_pair_blocks(
        y,
        block_ms=args.block_ms,
        min_block_ms=args.min_block_ms,
        max_block_ms=args.max_block_ms,
        offset_search_ms=args.offset_search_ms,
        step_ms=args.step_ms,
    )

    blocks = make_blocks(y, best["offset_samples"], best["block_samples"])
    duplicates = duplicate_report(blocks, threshold=args.duplicate_threshold)

    outdir = OUT_DIR / safe
    outdir.mkdir(parents=True, exist_ok=True)

    exported, reconstruction = export_blocks(blocks, outdir)

    metadata = {
        "version": "pair_blocks_v01",
        "source": str(source),
        "sample_rate": SR,
        "rule": "grid16 split into 8 pair blocks, 2 cases per block",
        "best": best,
        "duplicates": duplicates,
        "duplicate_threshold": args.duplicate_threshold,
        "reconstruction": reconstruction,
        "blocks": exported,
    }

    json_path = OUT_DIR / f"{safe}_pair_blocks_v01.json"
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("")
    print("Pair blocks créés :", json_path)
    print("Dossier blocs     :", outdir / "pair_blocks")
    print("Reconstruction    :", reconstruction)
    print("")
    print(f"Offset : {best['offset_ms']:.1f} ms")
    print(f"Bloc   : {best['block_ms']:.1f} ms")
    print(f"Score  : {best['score']:.3f}")
    print(f"Doublons détectés : {len(duplicates)}")

    if duplicates:
        print("Doublons :")
        for d in duplicates:
            print(f"  pair {d['a']} ~= pair {d['b']} sim={d['similarity']:.3f}")


if __name__ == "__main__":
    main()
