#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_reslice_break8_grid_v02_exact.py

Découpe break8 propre :
- PAS de trim onset
- PAS de fade sur chaque slice
- PAS de normalisation par slice
- découpe exacte de la loop en 8 cellules
- pattern formel : kick hat snare hat hat kick snare hat
- écrit un pair_blocks_v02 compatible tracker

Usage :
    cd ~/Applications/BreakbeatAI
    python pipeline/01_reslice_break8_grid_v02_exact.py --source "amen"

Si tout est légèrement en retard :
    python pipeline/01_reslice_break8_grid_v02_exact.py --source "amen" --shift-ms -3

Si tout est légèrement trop tôt :
    python pipeline/01_reslice_break8_grid_v02_exact.py --source "amen" --shift-ms 3
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

PAIR_BLOCKS_V02 = Path("dataset/pair_blocks_v02")
OUT_DIR = Path("dataset/pair_blocks_break8_grid")

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

ROLES = ["kick", "hat", "snare", "hat", "hat", "kick", "snare", "hat"]


def resample_linear(y, src_sr, dst_sr=SR):
    if src_sr == dst_sr:
        return y.astype(np.float32)

    duration = len(y) / float(src_sr)
    new_len = max(1, int(round(duration * dst_sr)))

    old_x = np.linspace(0, 1, len(y), endpoint=False)
    new_x = np.linspace(0, 1, new_len, endpoint=False)

    return np.interp(new_x, old_x, y).astype(np.float32)


def load_audio_raw(path):
    audio, sr = sf.read(path, always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)
    audio = resample_linear(audio, sr, SR)

    # On enlève juste le DC offset, mais on ne normalise pas.
    audio = audio - float(np.mean(audio))

    return audio.astype(np.float32)


def find_existing_pair_json(source):
    files = sorted(PAIR_BLOCKS_V02.glob("*_pair_blocks_v02.json"))
    matches = [p for p in files if source.lower() in p.name.lower()]
    return matches[0] if matches else None


def find_source_audio(source):
    old_json = find_existing_pair_json(source)

    if old_json:
        try:
            data = json.loads(old_json.read_text(encoding="utf-8"))
            for c in [data.get("source_audio"), data.get("source"), data.get("audio_path")]:
                if not c:
                    continue

                p = Path(c)
                if p.exists():
                    return p, old_json

                p2 = Path(".") / c
                if p2.exists():
                    return p2, old_json
        except Exception:
            pass

    exts = [".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp3"]

    for d in SOURCE_DIRS:
        if not d.exists():
            continue

        for ext in exts:
            files = sorted(d.rglob(f"*{ext}"))
            matches = [p for p in files if source.lower() in p.name.lower()]
            if matches:
                return matches[0], old_json

    print(f"Impossible de trouver l'audio source pour : {source}")
    sys.exit(1)


def safe_stem(path):
    out = []
    for ch in path.stem:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "break"


def backup_if_exists(path):
    if not path.exists():
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak_{stamp}")
    shutil.copy2(path, backup)
    return backup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="amen")
    parser.add_argument("--shift-ms", type=float, default=0.0)
    args = parser.parse_args()

    source_path, old_json = find_source_audio(args.source)
    audio = load_audio_raw(source_path)

    safe = safe_stem(source_path)
    source_len = len(audio)
    source_ms = source_len / SR * 1000.0

    pairs = 8
    cycle_steps = 16
    steps_per_pair = 2

    cell_len = source_len / pairs
    step_ms = source_ms / cycle_steps
    shift_samples = int(round(args.shift_ms * SR / 1000.0))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pair_dir = OUT_DIR / safe
    pair_dir.mkdir(parents=True, exist_ok=True)

    blocks = []

    for i in range(pairs):
        role = ROLES[i]

        start = int(round(i * cell_len)) + shift_samples
        end = int(round((i + 1) * cell_len)) + shift_samples

        start = max(0, min(source_len - 1, start))
        end = max(start + 1, min(source_len, end))

        # Découpe exacte, pas de fade, pas de trim, pas de normalisation.
        cell = audio[start:end].astype(np.float32)

        wav_path = pair_dir / f"{safe}_pair_{i:03d}_{role}_exact.wav"
        sf.write(wav_path, cell, SR)

        blocks.append({
            "pair": i,
            "audio_path": str(wav_path),
            "duration_ms": round(len(cell) / SR * 1000.0, 4),
            "source_start_ms": round(start / SR * 1000.0, 4),
            "source_end_ms": round(end / SR * 1000.0, 4),
            "formal_position": i,
            "formal_position_in_cycle": i,
            "formal_cycle_length": pairs,
            "formal_role": role,
            "manual_role": role,
            "role_guess": role,
            "role_confidence": 1.0,
            "role_source": "break8_exact_grid_no_trim_no_fade",
        })

    out_json = PAIR_BLOCKS_V02 / f"{safe}_pair_blocks_v02.json"
    backup = backup_if_exists(out_json)

    data = {
        "version": "pair_blocks_v02_break8_grid_v02_exact",
        "source_audio": str(source_path),
        "source_duration_ms": round(source_ms, 4),
        "sample_rate": SR,
        "safe": safe,
        "cycle_steps": cycle_steps,
        "pairs": pairs,
        "steps_per_pair": steps_per_pair,
        "step_ms": round(step_ms, 6),
        "pattern": ROLES,
        "pattern_text": " ".join(ROLES),
        "rule": "exact grid cut: no onset trim, no per-slice fade, no per-slice normalization",
        "shift_ms": args.shift_ms,
        "old_pair_json": str(old_json) if old_json else None,
        "backup_pair_json": str(backup) if backup else None,
        "blocks": blocks,
    }

    out_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    report = OUT_DIR / f"{safe}_break8_grid_v02_exact_report.txt"
    lines = []
    lines.append(f"Break8 exact grid reslice: {safe}")
    lines.append(f"Source: {source_path}")
    lines.append(f"Source duration: {source_ms:.2f} ms")
    lines.append(f"Step ms: {step_ms:.4f}")
    lines.append(f"Pattern: {' '.join(ROLES)}")
    lines.append(f"Shift ms: {args.shift_ms}")
    lines.append("No trim onset, no slice fade, no slice normalize.")
    lines.append(f"Output JSON: {out_json}")
    if backup:
        lines.append(f"Backup old JSON: {backup}")
    lines.append("")
    for b in blocks:
        lines.append(
            f"pair {b['pair']:02d} {b['formal_role']:5s} | "
            f"start {b['source_start_ms']:8.2f} ms | "
            f"dur {b['duration_ms']:8.2f} ms | "
            f"{b['audio_path']}"
        )

    report.write_text("\\n".join(lines) + "\\n", encoding="utf-8")

    print("OK reslice exact sans saccades")
    print("Source :", source_path)
    print("JSON :", out_json)
    print("Rapport :", report)
    if backup:
        print("Backup ancien JSON :", backup)
    print("")
    for b in blocks:
        print(
            f"pair {b['pair']:02d} -> {b['formal_role']:5s} | "
            f"start {b['source_start_ms']:8.2f} ms | "
            f"dur {b['duration_ms']:8.2f} ms"
        )


if __name__ == "__main__":
    main()
