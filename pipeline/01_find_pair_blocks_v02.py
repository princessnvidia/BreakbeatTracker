#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_find_pair_blocks_v02.py

Découpe dédiée tracker, version v02 :
- détecte automatiquement la zone active du break
- ignore les pauses/silences au début et à la fin
- découpe uniquement la zone utile en 8 blocs de 2 cases
- exactement 8 WAV
- vérifie les doublons audio
- exporte reconstruction + metadata

Usage :
    python pipeline/01_find_pair_blocks_v02.py --source "Amen"

Si la zone active est trop courte/longue :
    python pipeline/01_find_pair_blocks_v02.py --source "Amen" --trim-threshold 0.025
    python pipeline/01_find_pair_blocks_v02.py --source "Amen" --trim-threshold 0.010

Si tu veux forcer une taille de bloc :
    python pipeline/01_find_pair_blocks_v02.py --source "Amen" --block-ms 220

Sorties :
    dataset/pair_blocks_v02/<break>_pair_blocks_v02.json
    dataset/pair_blocks_v02/<break>/pair_blocks/*.wav
    dataset/pair_blocks_v02/<break>/pair_blocks_reconstruction.wav
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
OUT_DIR = Path("dataset/pair_blocks_v02")
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


def envelope_rms(y, frame_ms=10, hop_ms=5):
    frame = max(1, int(SR * frame_ms / 1000))
    hop = max(1, int(SR * hop_ms / 1000))

    values = []
    starts = []

    for start in range(0, max(1, len(y) - frame), hop):
        chunk = y[start:start + frame]
        values.append(rms(chunk))
        starts.append(start)

    if not values:
        return np.array([rms(y)]), np.array([0])

    return np.array(values), np.array(starts)


def detect_active_region(y, threshold=0.020, pad_ms=25, min_region_ms=800):
    """
    Détecte la zone utile du break en se basant sur l'enveloppe RMS.
    threshold est relatif au pic d'enveloppe :
      0.020 = très sensible
      0.050 = plus strict
    """
    env, starts = envelope_rms(y)

    peak_env = float(np.max(env)) if len(env) else 0.0
    if peak_env <= 1e-9:
        return 0, len(y), {
            "peak_env": peak_env,
            "threshold_abs": 0.0,
            "reason": "silent_or_empty",
        }

    threshold_abs = peak_env * threshold
    active_idx = np.where(env >= threshold_abs)[0]

    if len(active_idx) == 0:
        return 0, len(y), {
            "peak_env": peak_env,
            "threshold_abs": threshold_abs,
            "reason": "no_active_frame",
        }

    start = int(starts[active_idx[0]])
    end = int(starts[active_idx[-1]])

    pad = int(SR * pad_ms / 1000)
    start = max(0, start - pad)
    end = min(len(y), end + pad)

    min_len = int(SR * min_region_ms / 1000)
    if end - start < min_len:
        center = (start + end) // 2
        start = max(0, center - min_len // 2)
        end = min(len(y), start + min_len)

    return start, end, {
        "peak_env": peak_env,
        "threshold_abs": threshold_abs,
        "reason": "ok",
    }


def audio_similarity(a, b):
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


def make_blocks(y, active_start, active_end, block_samples):
    blocks = []

    for pair in range(8):
        start = active_start + pair * block_samples
        end = start + block_samples

        if start >= active_end:
            audio = np.zeros(block_samples, dtype=np.float32)
        else:
            audio = y[start:min(end, active_end)].copy()
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


def export_active_region(y, active_start, active_end, outdir):
    active = y[active_start:active_end].copy()
    wav = outdir / "active_region.wav"
    sf.write(wav, normalize(active), SR)
    return str(wav)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Amen")
    parser.add_argument("--block-ms", type=float, default=None)
    parser.add_argument("--trim-threshold", type=float, default=0.020)
    parser.add_argument("--trim-pad-ms", type=float, default=25.0)
    parser.add_argument("--min-region-ms", type=float, default=800.0)
    parser.add_argument("--duplicate-threshold", type=float, default=0.985)
    args = parser.parse_args()

    source = find_source(args.source)
    print("Source :", source)

    y, sr = load_audio(source)
    safe = safe_name(source)

    outdir = OUT_DIR / safe
    outdir.mkdir(parents=True, exist_ok=True)

    active_start, active_end, active_info = detect_active_region(
        y,
        threshold=args.trim_threshold,
        pad_ms=args.trim_pad_ms,
        min_region_ms=args.min_region_ms,
    )

    active_len = active_end - active_start

    if args.block_ms is None:
        block_samples = max(1, active_len // 8)
        block_ms = block_samples / SR * 1000
    else:
        block_ms = args.block_ms
        block_samples = int(SR * block_ms / 1000)

    blocks = make_blocks(y, active_start, active_end, block_samples)
    duplicates = duplicate_report(blocks, threshold=args.duplicate_threshold)

    exported, reconstruction = export_blocks(blocks, outdir)
    active_region_wav = export_active_region(y, active_start, active_end, outdir)

    metadata = {
        "version": "pair_blocks_v02",
        "source": str(source),
        "sample_rate": SR,
        "rule": "trim active region, then split into 8 pair blocks for grid16",
        "active_region": {
            "start_sample": int(active_start),
            "end_sample": int(active_end),
            "start_ms": float(active_start / SR * 1000),
            "end_ms": float(active_end / SR * 1000),
            "duration_ms": float(active_len / SR * 1000),
            "audio_path": active_region_wav,
            "info": active_info,
        },
        "block_ms": float(block_ms),
        "duplicates": duplicates,
        "duplicate_threshold": args.duplicate_threshold,
        "reconstruction": reconstruction,
        "blocks": exported,
    }

    json_path = OUT_DIR / f"{safe}_pair_blocks_v02.json"
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("")
    print("Pair blocks v02 créés :", json_path)
    print("Zone active           :", active_region_wav)
    print("Dossier blocs         :", outdir / "pair_blocks")
    print("Reconstruction        :", reconstruction)
    print("")
    print(f"Active start : {active_start / SR * 1000:.1f} ms")
    print(f"Active end   : {active_end / SR * 1000:.1f} ms")
    print(f"Active dur   : {active_len / SR * 1000:.1f} ms")
    print(f"Bloc         : {block_ms:.1f} ms")
    print(f"Doublons     : {len(duplicates)}")

    if duplicates:
        for d in duplicates:
            print(f"  pair {d['a']} ~= pair {d['b']} sim={d['similarity']:.3f}")


if __name__ == "__main__":
    main()
