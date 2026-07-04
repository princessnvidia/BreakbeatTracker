#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_autoslice_break_index_v04.py

Auto-slice index-only pour BreakbeatAI.

N'écrit PAS de WAV par slice.
Garde un seul fichier audio source.
Écrit seulement un JSON avec :
- source_audio
- source_start_sample
- source_end_sample
- duration_ms
- scores d'onset

Usage :
    python pipeline/01_autoslice_break_index_v04.py --source "amen"
    python pipeline/01_autoslice_break_index_v04.py --source "camo"

Options utiles :
    --slices 8
    --min-sep-ms 45
    --threshold 0.28
    --start-ms 100 --end-ms 1850
    --auto-trim
    --mode onset
    --mode grid
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
OUT_DIR = Path("dataset/slice_indexes_v04")

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

    audio = audio - float(np.mean(audio))

    return audio.astype(np.float32)


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
    print("Liste les fichiers audio disponibles avec :")
    print("find dataset audio samples -type f \\( -iname '*.wav' -o -iname '*.flac' -o -iname '*.mp3' \\) 2>/dev/null")
    sys.exit(1)


def normalize01(x):
    x = np.asarray(x, dtype=np.float32)

    if len(x) == 0:
        return x

    x = x - float(np.min(x))
    m = float(np.max(x))

    if m <= 1e-9:
        return np.zeros_like(x)

    return x / m


def smooth(x, n=5):
    x = np.asarray(x, dtype=np.float32)

    if n <= 1 or len(x) < n:
        return x

    k = np.ones(n, dtype=np.float32) / float(n)

    return np.convolve(x, k, mode="same").astype(np.float32)


def auto_bounds(audio, pad_ms=0.0):
    if len(audio) < 2048:
        return 0, len(audio)

    frame = 1024
    hop = 256
    abs_audio = np.abs(audio)

    env = []
    starts = []

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


def onset_score(audio):
    """
    Score de transient :
    - spectral flux large bande
    - flux basse fréquence pour mieux capter les kicks
    - variation RMS

    Pas besoin de scipy/librosa.
    """
    frame = 1024
    hop = 128

    if len(audio) < frame * 2:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.int64)

    window = np.hanning(frame).astype(np.float32)
    freqs = np.fft.rfftfreq(frame, d=1.0 / SR)

    low_mask = (freqs >= 35) & (freqs <= 220)
    mid_mask = (freqs >= 220) & (freqs <= 2500)
    high_mask = (freqs >= 2500) & (freqs <= 12000)

    mags = []
    rms = []
    starts = []

    for start in range(0, len(audio) - frame, hop):
        chunk = audio[start:start + frame] * window
        mag = np.abs(np.fft.rfft(chunk)).astype(np.float32)

        mags.append(mag)
        rms.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
        starts.append(start)

    mags = np.asarray(mags, dtype=np.float32)
    rms = np.asarray(rms, dtype=np.float32)
    starts = np.asarray(starts, dtype=np.int64)

    if len(mags) < 3:
        return np.zeros(1, dtype=np.float32), starts

    diff = np.maximum(0.0, mags[1:] - mags[:-1])

    broad_flux = np.sum(diff, axis=1)
    low_flux = np.sum(diff[:, low_mask], axis=1)
    mid_flux = np.sum(diff[:, mid_mask], axis=1)
    high_flux = np.sum(diff[:, high_mask], axis=1)

    rms_diff = np.maximum(0.0, rms[1:] - rms[:-1])

    # Pondération volontairement kick-friendly.
    score = (
        0.75 * normalize01(broad_flux)
        + 1.35 * normalize01(low_flux)
        + 0.35 * normalize01(mid_flux)
        + 0.20 * normalize01(high_flux)
        + 0.45 * normalize01(rms_diff)
    )

    score = smooth(normalize01(score), n=5)
    score_starts = starts[1:]

    return score.astype(np.float32), score_starts.astype(np.int64)


def pick_onsets(audio, slices=8, threshold=0.28, min_sep_ms=45.0):
    """
    Retourne des cut points sample-index relatifs à la loop :
    [0, cut1, cut2, ..., len(audio)]

    On prend les meilleurs pics d'onset, puis on trie.
    Si pas assez de pics, fallback grille régulière.
    """
    if slices < 2:
        return [0, len(audio)], [], "too_few_slices"

    score, starts = onset_score(audio)

    if len(score) < 3:
        cuts = grid_cuts(len(audio), slices)
        return cuts, [], "fallback_grid_short_audio"

    min_sep = int(round(min_sep_ms * SR / 1000.0))
    threshold_abs = float(np.max(score)) * float(threshold)

    candidates = []

    for i in range(1, len(score) - 1):
        if score[i] < threshold_abs:
            continue

        if score[i] >= score[i - 1] and score[i] >= score[i + 1]:
            candidates.append({
                "sample": int(starts[i]),
                "score": float(score[i]),
            })

    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)

    chosen = []

    for cand in candidates:
        sample = int(cand["sample"])

        if sample <= 0 or sample >= len(audio) - 1:
            continue

        too_close = False

        for old in chosen:
            if abs(sample - int(old["sample"])) < min_sep:
                too_close = True
                break

        if too_close:
            continue

        chosen.append(cand)

        if len(chosen) >= slices - 1:
            break

    chosen = sorted(chosen, key=lambda c: int(c["sample"]))

    if len(chosen) < slices - 1:
        cuts = grid_cuts(len(audio), slices)
        return cuts, chosen, "fallback_grid_not_enough_onsets"

    cuts = [0] + [int(c["sample"]) for c in chosen] + [len(audio)]

    # Sécurité : si une cellule est trop petite, fallback grid.
    min_cell = int(round(25 * SR / 1000.0))
    bad = False

    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < min_cell:
            bad = True
            break

    if bad:
        cuts = grid_cuts(len(audio), slices)
        return cuts, chosen, "fallback_grid_tiny_cell"

    return cuts, chosen, "onset"


def grid_cuts(length, slices):
    cuts = []

    for i in range(slices + 1):
        cuts.append(int(round(i * length / float(slices))))

    cuts[0] = 0
    cuts[-1] = int(length)

    return cuts


def backup_if_exists(path):
    if not path.exists():
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak_{stamp}")
    shutil.copy2(path, backup)

    return backup


def normalize_for_preview(y, peak=0.95):
    y = np.asarray(y, dtype=np.float32)

    m = float(np.max(np.abs(y))) if len(y) else 0.0

    if m <= 1e-9:
        return y

    return (y / m * peak).astype(np.float32)


def write_preview_wav(path, audio):
    sf.write(path, normalize_for_preview(audio), SR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--slices", type=int, default=8)
    parser.add_argument("--mode", choices=["onset", "grid"], default="onset")
    parser.add_argument("--threshold", type=float, default=0.28)
    parser.add_argument("--min-sep-ms", type=float, default=45.0)
    parser.add_argument("--start-ms", type=float, default=None)
    parser.add_argument("--end-ms", type=float, default=None)
    parser.add_argument("--shift-ms", type=float, default=0.0)
    parser.add_argument("--auto-trim", action="store_true")
    args = parser.parse_args()

    source_path, old_json = find_source_audio(args.source)
    full_audio = load_audio(source_path)

    safe = sanitize_name(source_path.stem)

    if args.auto_trim:
        start, end = auto_bounds(full_audio, pad_ms=0.0)
    else:
        start, end = 0, len(full_audio)

    if args.start_ms is not None:
        start = int(round(args.start_ms * SR / 1000.0))

    if args.end_ms is not None:
        end = int(round(args.end_ms * SR / 1000.0))

    shift = int(round(args.shift_ms * SR / 1000.0))
    start += shift
    end += shift

    start = max(0, min(len(full_audio) - 1, start))
    end = max(start + 1, min(len(full_audio), end))

    loop = full_audio[start:end].astype(np.float32)

    if args.mode == "grid":
        cuts = grid_cuts(len(loop), args.slices)
        chosen = []
        slice_method = "grid"
    else:
        cuts, chosen, slice_method = pick_onsets(
            loop,
            slices=args.slices,
            threshold=args.threshold,
            min_sep_ms=args.min_sep_ms,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slice_dir = OUT_DIR / safe
    slice_dir.mkdir(parents=True, exist_ok=True)

    preview_wav = slice_dir / f"{safe}_source_loop_preview_v04.wav"
    write_preview_wav(preview_wav, loop)

    blocks = []

    for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        source_a = int(start + a)
        source_b = int(start + b)
        duration_ms = (source_b - source_a) / SR * 1000.0

        blocks.append({
            "pair": i,
            "name": f"slice {i}",
            "source_audio": str(source_path),
            "source_start_sample": source_a,
            "source_end_sample": source_b,
            "source_start_ms": round(source_a / SR * 1000.0, 4),
            "source_end_ms": round(source_b / SR * 1000.0, 4),
            "duration_ms": round(duration_ms, 4),
            "cell_index": i,
            "cell_count": args.slices,
            "slice_storage": "index_only",
            "audio_path": None,
        })

    pair_json = PAIR_BLOCKS_DIR / f"{safe}_pair_blocks_v02.json"
    PAIR_BLOCKS_DIR.mkdir(parents=True, exist_ok=True)

    backup = backup_if_exists(pair_json)

    loop_ms = len(loop) / SR * 1000.0
    median_cell_ms = float(np.median([b["duration_ms"] for b in blocks])) if blocks else loop_ms / max(1, args.slices)

    data = {
        "version": "pair_blocks_v02_index_only_autoslice_v04",
        "storage": "index_only_single_source_audio",
        "source_audio": str(source_path),
        "safe": safe,
        "sample_rate": SR,
        "source_duration_ms": round(len(full_audio) / SR * 1000.0, 4),
        "loop_start_sample": int(start),
        "loop_end_sample": int(end),
        "loop_start_ms": round(start / SR * 1000.0, 4),
        "loop_end_ms": round(end / SR * 1000.0, 4),
        "loop_duration_ms": round(loop_ms, 4),
        "slices": args.slices,
        "step_ms": round(median_cell_ms, 6),
        "slice_method": slice_method,
        "threshold": args.threshold,
        "min_sep_ms": args.min_sep_ms,
        "mode": args.mode,
        "auto_trim": bool(args.auto_trim),
        "shift_ms": args.shift_ms,
        "old_pair_json": str(old_json) if old_json else None,
        "backup_pair_json": str(backup) if backup else None,
        "preview_wav": str(preview_wav),
        "chosen_onsets": [
            {
                "source_sample": int(start + c["sample"]),
                "source_ms": round((start + c["sample"]) / SR * 1000.0, 4),
                "score": round(float(c["score"]), 6),
            }
            for c in chosen
        ],
        "blocks": blocks,
    }

    pair_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    report = slice_dir / f"{safe}_index_only_v04_report.txt"
    lines = []
    lines.append(f"Index-only autoslice v04: {safe}")
    lines.append(f"Source: {source_path}")
    lines.append(f"Storage: one source audio + JSON slice indexes")
    lines.append(f"Method: {slice_method}")
    lines.append(f"Loop: {start / SR * 1000.0:.2f} ms -> {end / SR * 1000.0:.2f} ms")
    lines.append(f"Slices: {args.slices}")
    lines.append(f"JSON: {pair_json}")
    lines.append(f"Preview: {preview_wav}")
    if backup:
        lines.append(f"Backup old JSON: {backup}")
    lines.append("")
    for b in blocks:
        lines.append(
            f"slice {b['pair']:02d} | "
            f"{b['source_start_ms']:9.2f} -> {b['source_end_ms']:9.2f} ms | "
            f"dur {b['duration_ms']:9.2f} ms"
        )

    report.write_text("\\n".join(lines) + "\\n", encoding="utf-8")

    print("OK auto-slice index-only")
    print("Source :", source_path)
    print("JSON :", pair_json)
    print("Preview :", preview_wav)
    print("Rapport :", report)
    print("Méthode :", slice_method)
    if backup:
        print("Backup ancien JSON :", backup)
    print("")
    for b in blocks:
        print(
            f"slice {b['pair']:02d} | "
            f"{b['source_start_ms']:9.2f} -> {b['source_end_ms']:9.2f} ms | "
            f"dur {b['duration_ms']:9.2f} ms"
        )


if __name__ == "__main__":
    main()
