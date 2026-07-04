#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_m8_style_autoslice_index_v10.py

Slicer façon M8 pour BreakbeatAI.

Architecture :
- 1 seul fichier audio source
- JSON avec slice markers start/end
- pas de WAV par slice

Modes :
- auto    : transients
- grid    : divisions égales
- silence : coupe par silence
- hybrid  : grille musicale + snap sur transients proches
- markers : points manuels en ms, façon lazy chop exporté

Exemples :
    python pipeline/01_m8_style_autoslice_index_v10.py --source "camo"
    python pipeline/01_m8_style_autoslice_index_v10.py --source "camo" --mode hybrid --grid 16
    python pipeline/01_m8_style_autoslice_index_v10.py --source "camo" --mode auto --max-slices 24
    python pipeline/01_m8_style_autoslice_index_v10.py --source "camo" --mode grid --grid 16
    python pipeline/01_m8_style_autoslice_index_v10.py --source "camo" --mode markers --markers-ms "0,190,388,580"
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
OUT_DIR = Path("dataset/slice_indexes_v10")

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
    print("Liste les fichiers audio disponibles avec :")
    print("find dataset audio samples -type f \\( -iname '*.wav' -o -iname '*.flac' -o -iname '*.mp3' \\) 2>/dev/null")
    sys.exit(1)


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

    return start, max(start + 1, end)


def onset_score(audio):
    frame = 1024
    hop = 96

    if len(audio) < frame * 2:
        return np.zeros(1, dtype=np.float32), np.asarray([0], dtype=np.int64)

    window = np.hanning(frame).astype(np.float32)
    freqs = np.fft.rfftfreq(frame, d=1.0 / SR)

    sub_mask = (freqs >= 35) & (freqs <= 90)
    low_mask = (freqs >= 90) & (freqs <= 220)
    mid_mask = (freqs >= 220) & (freqs <= 2500)
    high_mask = (freqs >= 2500) & (freqs <= 9000)
    air_mask = (freqs >= 9000) & (freqs <= 16000)

    mags = []
    rms = []
    zcr = []
    starts = []

    for start in range(0, len(audio) - frame, hop):
        chunk = audio[start:start + frame]
        win = chunk * window
        mag = np.abs(np.fft.rfft(win)).astype(np.float32)

        mags.append(mag)
        rms.append(float(np.sqrt(np.mean(win * win) + 1e-12)))
        zcr.append(float(np.mean(np.abs(np.diff(np.signbit(chunk).astype(np.float32))))))
        starts.append(start)

    mags = np.asarray(mags, dtype=np.float32)
    rms = np.asarray(rms, dtype=np.float32)
    zcr = np.asarray(zcr, dtype=np.float32)
    starts = np.asarray(starts, dtype=np.int64)

    if len(mags) < 3:
        return np.zeros(1, dtype=np.float32), starts

    diff = np.maximum(0.0, mags[1:] - mags[:-1])

    sub_flux = np.sum(diff[:, sub_mask], axis=1)
    low_flux = np.sum(diff[:, low_mask], axis=1)
    mid_flux = np.sum(diff[:, mid_mask], axis=1)
    high_flux = np.sum(diff[:, high_mask], axis=1)
    air_flux = np.sum(diff[:, air_mask], axis=1)
    broad_flux = np.sum(diff, axis=1)

    rms_diff = np.maximum(0.0, rms[1:] - rms[:-1])
    zcr_now = zcr[1:]

    score = (
        0.75 * normalize01(broad_flux)
        + 1.05 * normalize01(sub_flux + low_flux)
        + 0.55 * normalize01(mid_flux)
        + 0.80 * normalize01(high_flux + air_flux)
        + 0.45 * normalize01(rms_diff)
        + 0.20 * normalize01(zcr_now)
    )

    score = smooth(normalize01(score), n=5)

    return score.astype(np.float32), starts[1:].astype(np.int64)


def hat_score(audio):
    frame = 1024
    hop = 64

    if len(audio) < frame * 2:
        return np.zeros(1, dtype=np.float32), np.asarray([0], dtype=np.int64)

    window = np.hanning(frame).astype(np.float32)
    freqs = np.fft.rfftfreq(frame, d=1.0 / SR)

    low_mask = (freqs >= 35) & (freqs <= 300)
    mid_mask = (freqs >= 300) & (freqs <= 2500)
    high_mask = (freqs >= 2500) & (freqs <= 9000)
    air_mask = (freqs >= 9000) & (freqs <= 16000)

    mags = []
    zcr = []
    starts = []

    for start in range(0, len(audio) - frame, hop):
        chunk = audio[start:start + frame]
        win = chunk * window
        mag = np.abs(np.fft.rfft(win)).astype(np.float32)

        mags.append(mag)
        zcr.append(float(np.mean(np.abs(np.diff(np.signbit(chunk).astype(np.float32))))))
        starts.append(start)

    mags = np.asarray(mags, dtype=np.float32)
    zcr = np.asarray(zcr, dtype=np.float32)
    starts = np.asarray(starts, dtype=np.int64)

    if len(mags) < 3:
        return np.zeros(1, dtype=np.float32), starts

    diff = np.maximum(0.0, mags[1:] - mags[:-1])

    low_flux = np.sum(diff[:, low_mask], axis=1)
    mid_flux = np.sum(diff[:, mid_mask], axis=1)
    high_flux = np.sum(diff[:, high_mask], axis=1)
    air_flux = np.sum(diff[:, air_mask], axis=1)

    score = (
        1.9 * normalize01(high_flux)
        + 1.7 * normalize01(air_flux)
        + 0.7 * normalize01(zcr[1:])
        - 0.9 * normalize01(low_flux)
        - 0.25 * normalize01(mid_flux)
    )

    score = smooth(normalize01(score), n=3)

    return score.astype(np.float32), starts[1:].astype(np.int64)


def peaks_from_curve(curve, starts, threshold=0.15, min_sep_ms=24.0, max_peaks=64, kind="auto"):
    if len(curve) < 3:
        return []

    threshold_abs = float(np.max(curve)) * float(threshold)
    min_sep = int(round(min_sep_ms * SR / 1000.0))

    peaks = []

    for i in range(1, len(curve) - 1):
        if curve[i] < threshold_abs:
            continue

        if curve[i] >= curve[i - 1] and curve[i] >= curve[i + 1]:
            peaks.append({
                "sample": int(starts[i]),
                "score": float(curve[i]),
                "kind": kind,
            })

    peaks = sorted(peaks, key=lambda p: p["score"], reverse=True)

    chosen = []

    for peak in peaks:
        sample = int(peak["sample"])

        if sample <= 0 or sample >= starts[-1]:
            continue

        if any(abs(sample - int(old["sample"])) < min_sep for old in chosen):
            continue

        chosen.append(peak)

        if max_peaks > 0 and len(chosen) >= max_peaks:
            break

    return sorted(chosen, key=lambda p: int(p["sample"]))


def grid_markers(length, grid):
    grid = max(1, int(grid))
    return [
        {
            "sample": int(round(i * length / float(grid))),
            "score": 1.0,
            "kind": "grid",
        }
        for i in range(grid)
    ]


def silence_markers(audio, threshold=0.035, min_silence_ms=35.0, min_sep_ms=45.0):
    frame = 512
    hop = 128

    if len(audio) < frame * 2:
        return [{"sample": 0, "score": 1.0, "kind": "start"}]

    abs_audio = np.abs(audio)

    env = []
    starts = []

    for start in range(0, len(abs_audio) - frame, hop):
        chunk = abs_audio[start:start + frame]
        env.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
        starts.append(start)

    env = normalize01(np.asarray(env, dtype=np.float32))
    starts = np.asarray(starts, dtype=np.int64)

    silent = env < threshold
    min_silence_frames = max(1, int(round((min_silence_ms / 1000.0 * SR) / hop)))

    markers = [{"sample": 0, "score": 1.0, "kind": "start"}]
    last_marker = 0

    i = 0
    while i < len(silent):
        if not silent[i]:
            i += 1
            continue

        j = i
        while j < len(silent) and silent[j]:
            j += 1

        if j - i >= min_silence_frames and j < len(starts):
            marker = int(starts[j])
            if marker - last_marker >= int(round(min_sep_ms * SR / 1000.0)):
                markers.append({
                    "sample": marker,
                    "score": 1.0,
                    "kind": "silence",
                })
                last_marker = marker

        i = j + 1

    return markers


def parse_markers_ms(markers_ms):
    values = []

    for part in str(markers_ms).split(","):
        part = part.strip()
        if not part:
            continue

        values.append(float(part))

    return values


def nearest_peak(sample, peaks, window_samples):
    best = None
    best_dist = None

    for peak in peaks:
        dist = abs(int(peak["sample"]) - int(sample))

        if dist > window_samples:
            continue

        if best is None or dist < best_dist:
            best = peak
            best_dist = dist

    return best


def merge_markers(markers, min_sep_ms=12.0, max_markers=128):
    min_sep = int(round(min_sep_ms * SR / 1000.0))

    markers = sorted(markers, key=lambda m: (int(m["sample"]), -float(m.get("score", 0.0))))

    out = []

    for marker in markers:
        sample = int(marker["sample"])

        if sample < 0:
            continue

        if out and abs(sample - int(out[-1]["sample"])) < min_sep:
            if float(marker.get("score", 0.0)) > float(out[-1].get("score", 0.0)):
                out[-1] = marker
            continue

        out.append(marker)

        if max_markers > 0 and len(out) >= max_markers:
            break

    if not out or int(out[0]["sample"]) > int(20 * SR / 1000.0):
        out = [{"sample": 0, "score": 1.0, "kind": "start"}] + out

    return out


def build_markers(audio, args):
    main_curve, main_starts = onset_score(audio)
    hat_curve, hat_starts = hat_score(audio)

    main_peaks = peaks_from_curve(
        main_curve,
        main_starts,
        threshold=args.sensitivity,
        min_sep_ms=args.min_sep_ms,
        max_peaks=args.max_slices,
        kind="auto",
    )

    hat_peaks = peaks_from_curve(
        hat_curve,
        hat_starts,
        threshold=args.hat_sensitivity,
        min_sep_ms=args.hat_min_sep_ms,
        max_peaks=args.hat_rescue,
        kind="hat",
    )

    if args.mode == "grid":
        return merge_markers(
            grid_markers(len(audio), args.grid),
            min_sep_ms=1.0,
            max_markers=args.max_slices,
        )

    if args.mode == "silence":
        return merge_markers(
            silence_markers(
                audio,
                threshold=args.silence_threshold,
                min_silence_ms=args.min_silence_ms,
                min_sep_ms=args.min_sep_ms,
            ),
            min_sep_ms=args.min_sep_ms,
            max_markers=args.max_slices,
        )

    if args.mode == "markers":
        markers = []
        for ms in parse_markers_ms(args.markers_ms):
            markers.append({
                "sample": int(round(ms * SR / 1000.0)),
                "score": 1.0,
                "kind": "manual_marker",
            })
        return merge_markers(markers, min_sep_ms=1.0, max_markers=args.max_slices)

    if args.mode == "auto":
        return merge_markers(
            main_peaks + hat_peaks,
            min_sep_ms=args.merge_sep_ms,
            max_markers=args.max_slices,
        )

    # HYBRID :
    # Grille musicale régulière, chaque ligne est snap à un transient proche.
    # Puis on ajoute quelques hats autour si demandés.
    grid = grid_markers(len(audio), args.grid)
    window_samples = int(round(args.snap_ms * SR / 1000.0))

    snapped = []

    for marker in grid:
        peak = nearest_peak(marker["sample"], main_peaks + hat_peaks, window_samples)

        if peak is None:
            snapped.append(marker)
        else:
            m = dict(peak)
            m["kind"] = "hybrid_snap_" + str(peak.get("kind", "auto"))
            m["grid_sample"] = int(marker["sample"])
            snapped.append(m)

    # On garde le côté M8 : markers simples, mais on injecte quelques hats discrets.
    extra_hats = []
    for peak in sorted(hat_peaks, key=lambda p: p["score"], reverse=True):
        if len(extra_hats) >= args.hat_rescue:
            break
        h = dict(peak)
        h["kind"] = "hat_rescue"
        extra_hats.append(h)

    return merge_markers(
        snapped + extra_hats,
        min_sep_ms=args.merge_sep_ms,
        max_markers=args.max_slices,
    )


def backup_if_exists(path):
    if not path.exists():
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak_{stamp}")
    shutil.copy2(path, backup)

    return backup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)

    parser.add_argument("--mode", choices=["auto", "grid", "silence", "hybrid", "markers"], default="hybrid")

    parser.add_argument("--target-bpm", type=float, default=155.0)
    parser.add_argument("--bars", type=float, default=2.0)
    parser.add_argument("--beats-per-bar", type=float, default=4.0)

    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--max-slices", type=int, default=32)

    parser.add_argument("--sensitivity", type=float, default=0.16)
    parser.add_argument("--min-sep-ms", type=float, default=28.0)

    parser.add_argument("--hat-sensitivity", type=float, default=0.045)
    parser.add_argument("--hat-min-sep-ms", type=float, default=12.0)
    parser.add_argument("--hat-rescue", type=int, default=8)

    parser.add_argument("--snap-ms", type=float, default=55.0)
    parser.add_argument("--merge-sep-ms", type=float, default=10.0)

    parser.add_argument("--silence-threshold", type=float, default=0.035)
    parser.add_argument("--min-silence-ms", type=float, default=35.0)

    parser.add_argument("--markers-ms", default="")

    parser.add_argument("--min-len-ms", type=float, default=35.0)
    parser.add_argument("--max-len-ms", type=float, default=900.0)

    parser.add_argument("--start-ms", type=float, default=None)
    parser.add_argument("--end-ms", type=float, default=None)
    parser.add_argument("--shift-ms", type=float, default=0.0)
    parser.add_argument("--auto-trim", action="store_true")

    args = parser.parse_args()

    source_path, old_json = find_source_audio(args.source)
    full_audio = load_audio(source_path)

    safe = sanitize_name(source_path.stem)

    if args.auto_trim:
        loop_start, loop_end = auto_bounds(full_audio, pad_ms=0.0)
    else:
        loop_start, loop_end = 0, len(full_audio)

    if args.start_ms is not None:
        loop_start = int(round(args.start_ms * SR / 1000.0))

    if args.end_ms is not None:
        loop_end = int(round(args.end_ms * SR / 1000.0))

    shift = int(round(args.shift_ms * SR / 1000.0))
    loop_start += shift
    loop_end += shift

    loop_start = max(0, min(len(full_audio) - 1, loop_start))
    loop_end = max(loop_start + 1, min(len(full_audio), loop_end))

    loop = full_audio[loop_start:loop_end].astype(np.float32)

    markers = build_markers(loop, args)
    markers = sorted(markers, key=lambda m: int(m["sample"]))

    min_len = int(round(args.min_len_ms * SR / 1000.0))
    max_len = int(round(args.max_len_ms * SR / 1000.0))

    blocks = []

    for i, marker in enumerate(markers):
        a = int(marker["sample"])

        if i + 1 < len(markers):
            b = int(markers[i + 1]["sample"])
        else:
            b = len(loop)

        b = max(a + min_len, b)
        b = min(len(loop), a + max_len, b)

        if b <= a:
            continue

        source_a = int(loop_start + a)
        source_b = int(loop_start + b)

        blocks.append({
            "pair": len(blocks),
            "name": f"slice {len(blocks)}",
            "role": "slice",
            "manual_role": "slice",
            "formal_role": "slice",
            "source_audio": str(source_path),
            "source_start_sample": source_a,
            "source_end_sample": source_b,
            "source_start_ms": round(source_a / SR * 1000.0, 4),
            "source_end_ms": round(source_b / SR * 1000.0, 4),
            "duration_ms": round((source_b - source_a) / SR * 1000.0, 4),
            "slice_storage": "index_only",
            "audio_path": None,
            "marker_kind": marker.get("kind", "unknown"),
            "marker_score": round(float(marker.get("score", 0.0)), 6),
            "grid_sample": marker.get("grid_sample"),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = OUT_DIR / safe
    out_dir.mkdir(parents=True, exist_ok=True)

    preview_wav = out_dir / f"{safe}_source_loop_preview_v10.wav"
    sf.write(preview_wav, normalize_for_preview(loop), SR)

    pair_json = PAIR_BLOCKS_DIR / f"{safe}_pair_blocks_v02.json"
    PAIR_BLOCKS_DIR.mkdir(parents=True, exist_ok=True)

    backup = backup_if_exists(pair_json)

    target_loop_ms = args.bars * args.beats_per_bar * 60000.0 / args.target_bpm
    target_loop_samples = int(round(target_loop_ms * SR / 1000.0))

    data = {
        "version": "pair_blocks_v02_m8_style_markers_index_only_v10",
        "storage": "index_only_single_source_audio",
        "slice_method": f"m8_style_{args.mode}_v10",
        "source_audio": str(source_path),
        "safe": safe,
        "sample_rate": SR,
        "source_duration_ms": round(len(full_audio) / SR * 1000.0, 4),

        "loop_start_sample": int(loop_start),
        "loop_end_sample": int(loop_end),
        "loop_start_ms": round(loop_start / SR * 1000.0, 4),
        "loop_end_ms": round(loop_end / SR * 1000.0, 4),
        "loop_duration_ms": round(len(loop) / SR * 1000.0, 4),

        "target_bpm": args.target_bpm,
        "target_bars": args.bars,
        "target_beats_per_bar": args.beats_per_bar,
        "target_loop_ms": round(target_loop_ms, 6),
        "target_loop_samples": target_loop_samples,
        "target_step_ms": round(60000.0 / (args.target_bpm * 4.0), 6),

        "mode": args.mode,
        "grid": args.grid,
        "max_slices": args.max_slices,
        "sensitivity": args.sensitivity,
        "min_sep_ms": args.min_sep_ms,
        "hat_sensitivity": args.hat_sensitivity,
        "hat_min_sep_ms": args.hat_min_sep_ms,
        "hat_rescue": args.hat_rescue,
        "snap_ms": args.snap_ms,

        "slice_count": len(blocks),
        "old_pair_json": str(old_json) if old_json else None,
        "backup_pair_json": str(backup) if backup else None,
        "preview_wav": str(preview_wav),
        "blocks": blocks,
    }

    pair_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    report = out_dir / f"{safe}_m8_style_v10_report.txt"
    lines = []
    lines.append(f"M8-style slicer v10: {safe}")
    lines.append(f"Source: {source_path}")
    lines.append(f"Mode: {args.mode}")
    lines.append(f"Target BPM: {args.target_bpm}")
    lines.append(f"Target loop: {target_loop_ms:.3f} ms")
    lines.append(f"Slices: {len(blocks)}")
    lines.append(f"JSON: {pair_json}")
    lines.append(f"Preview: {preview_wav}")

    if backup:
        lines.append(f"Backup old JSON: {backup}")

    lines.append("")
    for b in blocks:
        lines.append(
            f"slice {b['pair']:02d} | "
            f"{b['source_start_ms']:9.2f} -> {b['source_end_ms']:9.2f} ms | "
            f"dur {b['duration_ms']:9.2f} ms | "
            f"{b['marker_kind']} | score {b['marker_score']}"
        )

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("OK M8-style slicer v10")
    print("Source :", source_path)
    print("Mode :", args.mode)
    print("JSON :", pair_json)
    print("Preview :", preview_wav)
    print("Rapport :", report)
    print("Slices :", len(blocks))
    print("Target BPM :", args.target_bpm)
    print("Target loop ms :", round(target_loop_ms, 3))
    if backup:
        print("Backup ancien JSON :", backup)
    print("")
    for b in blocks:
        print(
            f"slice {b['pair']:02d} | "
            f"{b['source_start_ms']:9.2f} -> {b['source_end_ms']:9.2f} ms | "
            f"{b['duration_ms']:9.2f} ms | {b['marker_kind']}"
        )


if __name__ == "__main__":
    main()
