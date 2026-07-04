#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_reslice_break8_grid_v01.py

Reslice formel d'un break en 8 samples selon la grammaire :

    kick hat snare hat hat kick snare hat

But :
- Ne plus dépendre d'une détection d'onset bancale.
- Découper une loop complète en 8 cellules musicales propres.
- Chaque cellule = 2 steps sur une loop de 16 steps.
- Recaler l'attaque au début de chaque sample pour enlever l'impression de retard.
- Écrire un pair_blocks_v02 compatible avec le tracker.

Usage :
    cd ~/Applications/BreakbeatAI
    python pipeline/01_reslice_break8_grid_v01.py --source "amen"

Options utiles :
    --no-trim-onset
        garde le début exact de chaque cellule, sans recaler l'attaque.

    --shift-ms -5
        décale toute la grille un peu plus tôt.

    --shift-ms 5
        décale toute la grille un peu plus tard.
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


def normalize(y, peak=0.95):
    y = np.asarray(y, dtype=np.float32)

    if len(y) == 0:
        return y

    m = float(np.max(np.abs(y)))
    if m <= 1e-9:
        return y

    return (y / m * peak).astype(np.float32)


def fade(y, ms=2.0):
    y = np.asarray(y, dtype=np.float32)

    if len(y) < 32:
        return y

    n = min(int(SR * ms / 1000), len(y) // 4)
    if n <= 2:
        return y

    out = y.copy()
    ramp = np.linspace(0, 1, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def resample_linear(y, src_sr, dst_sr=SR):
    if src_sr == dst_sr:
        return y.astype(np.float32)

    duration = len(y) / float(src_sr)
    new_len = max(1, int(round(duration * dst_sr)))

    old_x = np.linspace(0, 1, len(y), endpoint=False)
    new_x = np.linspace(0, 1, new_len, endpoint=False)

    return np.interp(new_x, old_x, y).astype(np.float32)


def load_audio(path):
    audio, sr = sf.read(path, always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)
    audio = resample_linear(audio, sr, SR)
    audio = audio - float(np.mean(audio))

    return normalize(audio)


def find_existing_pair_json(source):
    files = sorted(PAIR_BLOCKS_V02.glob("*_pair_blocks_v02.json"))
    matches = [p for p in files if source.lower() in p.name.lower()]
    return matches[0] if matches else None


def find_source_audio(source):
    # 1) Si l'ancien pair_blocks connaît la source, on l'utilise.
    old_json = find_existing_pair_json(source)
    if old_json:
        try:
            data = json.loads(old_json.read_text(encoding="utf-8"))
            candidates = [
                data.get("source_audio"),
                data.get("source"),
                data.get("audio_path"),
            ]

            for c in candidates:
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

    # 2) Sinon recherche par nom dans les dossiers connus.
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
    print("Dossiers cherchés :")
    for d in SOURCE_DIRS:
        print(" -", d)
    sys.exit(1)


def safe_stem(path):
    name = path.stem
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "break"


def detect_onset_offset(y, max_ms=45.0, preroll_ms=1.5):
    """
    Cherche le premier vrai transient dans une cellule.
    Ça enlève le petit silence/retard au début du sample.
    """
    y = np.asarray(y, dtype=np.float32)

    max_n = min(len(y), int(SR * max_ms / 1000))
    if max_n < 128:
        return 0

    seg = y[:max_n]
    abs_y = np.abs(seg)

    frame = 128
    hop = 16

    env = []
    starts = []

    for start in range(0, max(1, len(abs_y) - frame), hop):
        chunk = abs_y[start:start + frame]
        env.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
        starts.append(start)

    if not env:
        return 0

    env = np.asarray(env, dtype=np.float32)
    starts = np.asarray(starts, dtype=np.int64)

    peak = float(np.max(env))
    if peak <= 1e-8:
        return 0

    noise = float(np.median(env[:min(8, len(env))]))
    threshold = max(noise * 2.5, peak * 0.12)

    idxs = np.where(env >= threshold)[0]
    if len(idxs) == 0:
        return 0

    onset = int(starts[int(idxs[0])])
    preroll = int(SR * preroll_ms / 1000)

    return max(0, onset - preroll)


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
    parser.add_argument("--no-trim-onset", action="store_true")
    args = parser.parse_args()

    source_path, old_json = find_source_audio(args.source)
    audio = load_audio(source_path)

    safe = safe_stem(source_path)
    source_len = len(audio)
    source_ms = source_len / SR * 1000.0

    cycle_steps = 16
    pairs = 8
    steps_per_pair = 2

    step_samples = source_len / cycle_steps
    shift_samples = int(round(args.shift_ms * SR / 1000.0))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pair_dir = OUT_DIR / safe
    pair_dir.mkdir(parents=True, exist_ok=True)

    blocks = []

    for i in range(pairs):
        role = ROLES[i]

        start_f = i * steps_per_pair * step_samples
        end_f = (i + 1) * steps_per_pair * step_samples

        start = int(round(start_f)) + shift_samples
        end = int(round(end_f)) + shift_samples

        start = max(0, min(source_len - 1, start))
        end = max(start + 1, min(source_len, end))

        raw = audio[start:end].copy()

        onset_offset = 0
        if not args.no_trim_onset:
            onset_offset = detect_onset_offset(raw)
            raw = raw[onset_offset:]

        raw = fade(normalize(raw), ms=2.0)

        wav_path = pair_dir / f"{safe}_pair_{i:03d}_{role}.wav"
        sf.write(wav_path, raw, SR)

        duration_ms = len(raw) / SR * 1000.0
        start_ms = (start + onset_offset) / SR * 1000.0
        end_ms = end / SR * 1000.0

        blocks.append({
            "pair": i,
            "audio_path": str(wav_path),
            "duration_ms": round(duration_ms, 4),
            "source_start_ms": round(start_ms, 4),
            "source_end_ms": round(end_ms, 4),
            "grid_start_ms_before_onset_trim": round(start / SR * 1000.0, 4),
            "grid_end_ms": round(end / SR * 1000.0, 4),
            "onset_trim_ms": round(onset_offset / SR * 1000.0, 4),
            "formal_position": i,
            "formal_position_in_cycle": i,
            "formal_cycle_length": pairs,
            "formal_role": role,
            "manual_role": role,
            "role_guess": role,
            "role_confidence": 1.0,
            "role_source": "break8_grid_reslice",
        })

    out_json = PAIR_BLOCKS_V02 / f"{safe}_pair_blocks_v02.json"
    backup = backup_if_exists(out_json)

    data = {
        "version": "pair_blocks_v02_break8_grid_reslice_v01",
        "source_audio": str(source_path),
        "source_duration_ms": round(source_ms, 4),
        "sample_rate": SR,
        "safe": safe,
        "cycle_steps": cycle_steps,
        "pairs": pairs,
        "steps_per_pair": steps_per_pair,
        "step_ms": round(source_ms / cycle_steps, 6),
        "pattern": ROLES,
        "pattern_text": " ".join(ROLES),
        "rule": "fixed grid: one source loop / 16 steps / 8 pairs",
        "shift_ms": args.shift_ms,
        "trim_onset": not args.no_trim_onset,
        "old_pair_json": str(old_json) if old_json else None,
        "backup_pair_json": str(backup) if backup else None,
        "blocks": blocks,
    }

    out_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    report = OUT_DIR / f"{safe}_break8_grid_reslice_v01_report.txt"
    lines = []
    lines.append(f"Break8 grid reslice: {safe}")
    lines.append(f"Source: {source_path}")
    lines.append(f"Source duration: {source_ms:.2f} ms")
    lines.append(f"Step ms: {source_ms / cycle_steps:.4f}")
    lines.append(f"Pattern: {' '.join(ROLES)}")
    lines.append(f"Trim onset: {not args.no_trim_onset}")
    lines.append(f"Shift ms: {args.shift_ms}")
    lines.append(f"Output JSON: {out_json}")
    if backup:
        lines.append(f"Backup old JSON: {backup}")
    lines.append("")
    for b in blocks:
        lines.append(
            f"pair {b['pair']:02d} {b['formal_role']:5s} | "
            f"start {b['source_start_ms']:8.2f} ms | "
            f"dur {b['duration_ms']:8.2f} ms | "
            f"trim {b['onset_trim_ms']:6.2f} ms | "
            f"{b['audio_path']}"
        )

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("OK reslice break8 à la grille")
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
            f"dur {b['duration_ms']:8.2f} ms | trim {b['onset_trim_ms']:6.2f} ms"
        )


if __name__ == "__main__":
    main()
