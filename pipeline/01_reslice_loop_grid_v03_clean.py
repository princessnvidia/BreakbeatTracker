#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_reslice_loop_grid_v03_clean.py

Découpe propre sans logique kick/snare/hat.

But :
- choisir une vraie portion de loop
- découper en cellules régulières
- écrire dataset/pair_blocks_v02/*_pair_blocks_v02.json
- l'app utilisera juste sample 0, sample 1, sample 2...

Usage simple :
    python pipeline/01_reslice_loop_grid_v03_clean.py --source "amen"

Avec correction manuelle de début/fin :
    python pipeline/01_reslice_loop_grid_v03_clean.py --source "amen" --start-ms 120 --end-ms 1850

Décalage global :
    python pipeline/01_reslice_loop_grid_v03_clean.py --source "amen" --shift-ms -5

Découpe en 8 cellules par défaut.
"""

from pathlib import Path
import argparse
import json
import shutil
import sys
from datetime import datetime

import numpy as np
import soundfile as sf


SR = 44100

PAIR_BLOCKS_DIR = Path("dataset/pair_blocks_v02")
OUT_DIR = Path("dataset/loop_grid_clean_v03")

SOURCE_DIRS = [
    Path("dataset/source_audio"),
    Path("dataset/sources"),
    Path("dataset/audio"),
    Path("dataset/raw_audio"),
    Path("dataset"),
    Path("audio"),
    Path("samples"),
    Path("."),
]


def sanitize_name(name):
    out = []

    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")

    return "".join(out).strip("_") or "break"


def resample_linear(y, src_sr, dst_sr=SR):
    y = np.asarray(y, dtype=np.float32)

    if src_sr == dst_sr:
        return y

    if len(y) <= 1:
        return y

    duration = len(y) / float(src_sr)
    new_len = max(1, int(round(duration * dst_sr)))

    old_x = np.linspace(0.0, 1.0, len(y), endpoint=False)
    new_x = np.linspace(0.0, 1.0, new_len, endpoint=False)

    return np.interp(new_x, old_x, y).astype(np.float32)


def load_audio(path):
    audio, sr = sf.read(path, always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)
    audio = resample_linear(audio, sr, SR)

    # DC offset seulement. Pas de normalisation, pas de fade.
    audio = audio - float(np.mean(audio))

    return audio.astype(np.float32)


def normalize_for_preview(y, peak=0.95):
    y = np.asarray(y, dtype=np.float32)

    if len(y) == 0:
        return y

    m = float(np.max(np.abs(y)))

    if m <= 1e-9:
        return y

    return (y / m * peak).astype(np.float32)


def find_existing_pair_json(source):
    if not PAIR_BLOCKS_DIR.exists():
        return None

    files = sorted(PAIR_BLOCKS_DIR.glob("*_pair_blocks_v02.json"))
    matches = [p for p in files if source.lower() in p.name.lower()]

    return matches[0] if matches else None


def find_source_audio(source):
    old_json = find_existing_pair_json(source)

    if old_json:
        try:
            data = json.loads(old_json.read_text(encoding="utf-8"))

            for key in ("source_audio", "source", "audio_path"):
                value = data.get(key)

                if not value:
                    continue

                p = Path(value)

                if p.exists():
                    return p, old_json

                p2 = Path(".") / value

                if p2.exists():
                    return p2, old_json

        except Exception:
            pass

    exts = [".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp3"]

    for folder in SOURCE_DIRS:
        if not folder.exists():
            continue

        for ext in exts:
            for path in sorted(folder.rglob(f"*{ext}")):
                if source.lower() in path.name.lower():
                    return path, old_json

    print(f"Impossible de trouver l'audio source pour : {source}")
    print("")
    print("Astuce : regarde les vrais noms disponibles avec :")
    print("find dataset audio samples -type f \\( -iname '*.wav' -o -iname '*.flac' -o -iname '*.mp3' \\) 2>/dev/null")
    sys.exit(1)


def auto_bounds(audio, pad_ms=0.0):
    """
    Détection simple du début/fin utile.
    Elle ne prétend pas connaître le groove, elle enlève juste le silence évident.
    """
    if len(audio) < 2048:
        return 0, len(audio)

    frame = 1024
    hop = 256

    env = []
    starts = []

    abs_audio = np.abs(audio)

    for start in range(0, len(abs_audio) - frame, hop):
        chunk = abs_audio[start:start + frame]
        env.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
        starts.append(start)

    if not env:
        return 0, len(audio)

    env = np.asarray(env, dtype=np.float32)
    starts = np.asarray(starts, dtype=np.int64)

    peak = float(np.max(env))
    med = float(np.median(env))

    if peak <= 1e-9:
        return 0, len(audio)

    threshold = max(med * 3.0, peak * 0.035)
    active = np.where(env >= threshold)[0]

    if len(active) == 0:
        return 0, len(audio)

    start = int(starts[int(active[0])])
    end = int(starts[int(active[-1])] + frame)

    pad = int(SR * pad_ms / 1000.0)

    start = max(0, start - pad)
    end = min(len(audio), end + pad)

    if end <= start:
        return 0, len(audio)

    return start, end


def backup_if_exists(path):
    if not path.exists():
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak_{stamp}")
    shutil.copy2(path, backup)

    return backup


def process_one(source, cells=8, start_ms=None, end_ms=None, shift_ms=0.0, auto_trim=False):
    source_path, old_json = find_source_audio(source)
    audio = load_audio(source_path)

    safe = sanitize_name(source_path.stem)
    source_ms = len(audio) / SR * 1000.0

    if auto_trim:
        auto_start, auto_end = auto_bounds(audio, pad_ms=0.0)
    else:
        auto_start, auto_end = 0, len(audio)

    if start_ms is None:
        start = auto_start
    else:
        start = int(round(start_ms * SR / 1000.0))

    if end_ms is None:
        end = auto_end
    else:
        end = int(round(end_ms * SR / 1000.0))

    shift = int(round(shift_ms * SR / 1000.0))
    start += shift
    end += shift

    start = max(0, min(len(audio) - 1, start))
    end = max(start + 1, min(len(audio), end))

    loop = audio[start:end].astype(np.float32)
    loop_ms = len(loop) / SR * 1000.0
    cell_len_float = len(loop) / float(cells)
    cell_ms = loop_ms / float(cells)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pair_dir = OUT_DIR / safe
    pair_dir.mkdir(parents=True, exist_ok=True)

    preview_wav = pair_dir / f"{safe}_loop_preview_clean_v03.wav"
    sf.write(preview_wav, normalize_for_preview(loop), SR)

    blocks = []

    for i in range(cells):
        a = int(round(i * cell_len_float))
        b = int(round((i + 1) * cell_len_float))

        a = max(0, min(len(loop) - 1, a))
        b = max(a + 1, min(len(loop), b))

        cell = loop[a:b].astype(np.float32)

        wav_path = pair_dir / f"{safe}_sample_{i:03d}.wav"
        sf.write(wav_path, cell, SR)

        blocks.append({
            "pair": i,
            "name": f"sample {i}",
            "audio_path": str(wav_path),
            "duration_ms": round(len(cell) / SR * 1000.0, 4),
            "source_start_ms": round((start + a) / SR * 1000.0, 4),
            "source_end_ms": round((start + b) / SR * 1000.0, 4),
            "cell_index": i,
            "cell_count": cells,
            "display_role": "sample",
            "role_guess": "sample",
            "manual_role": "sample",
            "formal_role": "sample",
        })

    pair_json = PAIR_BLOCKS_DIR / f"{safe}_pair_blocks_v02.json"
    PAIR_BLOCKS_DIR.mkdir(parents=True, exist_ok=True)

    backup = backup_if_exists(pair_json)

    data = {
        "version": "pair_blocks_v02_loop_grid_clean_v03_no_roles",
        "source_audio": str(source_path),
        "safe": safe,
        "sample_rate": SR,
        "source_duration_ms": round(source_ms, 4),
        "loop_start_ms": round(start / SR * 1000.0, 4),
        "loop_end_ms": round(end / SR * 1000.0, 4),
        "loop_duration_ms": round(loop_ms, 4),
        "cells": cells,
        "cell_ms": round(cell_ms, 6),
        "step_ms": round(cell_ms, 6),
        "display_mode": "plain_samples_no_kick_hat_snare",
        "shift_ms": shift_ms,
        "auto_trim": bool(auto_trim),
        "old_pair_json": str(old_json) if old_json else None,
        "backup_pair_json": str(backup) if backup else None,
        "preview_wav": str(preview_wav),
        "blocks": blocks,
    }

    pair_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    report = pair_dir / f"{safe}_clean_v03_report.txt"
    lines = []
    lines.append(f"Clean grid slice v03: {safe}")
    lines.append(f"Source: {source_path}")
    lines.append(f"Source duration: {source_ms:.2f} ms")
    lines.append(f"Loop start: {start / SR * 1000.0:.2f} ms")
    lines.append(f"Loop end: {end / SR * 1000.0:.2f} ms")
    lines.append(f"Loop duration: {loop_ms:.2f} ms")
    lines.append(f"Cells: {cells}")
    lines.append(f"Cell ms: {cell_ms:.4f}")
    lines.append(f"Preview: {preview_wav}")
    lines.append(f"Pair JSON: {pair_json}")
    if backup:
        lines.append(f"Backup old JSON: {backup}")
    lines.append("")
    for block in blocks:
        lines.append(
            f"sample {block['pair']:02d} | "
            f"{block['source_start_ms']:9.2f} -> {block['source_end_ms']:9.2f} ms | "
            f"dur {block['duration_ms']:9.2f} ms | {block['audio_path']}"
        )

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("OK découpe propre sans rôles")
    print("Source :", source_path)
    print("Preview loop :", preview_wav)
    print("JSON :", pair_json)
    print("Rapport :", report)
    if backup:
        print("Backup ancien JSON :", backup)
    print("")
    print(f"Loop start/end : {start / SR * 1000.0:.2f} ms -> {end / SR * 1000.0:.2f} ms")
    print(f"Cellule : {cell_ms:.4f} ms")
    print("")
    for block in blocks:
        print(
            f"sample {block['pair']:02d} | "
            f"{block['source_start_ms']:9.2f} -> {block['source_end_ms']:9.2f} ms | "
            f"dur {block['duration_ms']:9.2f} ms"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--start-ms", type=float, default=None)
    parser.add_argument("--end-ms", type=float, default=None)
    parser.add_argument("--shift-ms", type=float, default=0.0)
    parser.add_argument("--auto-trim", action="store_true")
    args = parser.parse_args()

    if args.cells < 2:
        raise SystemExit("--cells doit être >= 2")

    process_one(
        source=args.source,
        cells=args.cells,
        start_ms=args.start_ms,
        end_ms=args.end_ms,
        shift_ms=args.shift_ms,
        auto_trim=args.auto_trim,
    )


if __name__ == "__main__":
    main()
