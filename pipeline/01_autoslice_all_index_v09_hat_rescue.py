#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_autoslice_all_index_v09_hat_rescue.py

Autoslice index-only : ajoute TOUS les hits/transients détectés dans le break.

- Ne génère PAS de WAV par slice.
- Garde un seul fichier audio source.
- Écrit seulement source_start_sample / source_end_sample dans le JSON.
- L'app affiche ensuite une ligne par slice.

Usage :
    python pipeline/01_autoslice_all_index_v09_hat_rescue.py --source "camo"
    python pipeline/01_autoslice_all_index_v09_hat_rescue.py --source "amen"

Réglages utiles :
    --threshold 0.12
    --min-sep-ms 25
    --max-slices 64
    --auto-trim
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
OUT_DIR = Path("dataset/slice_indexes_v09")

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
        0.85 * normalize01(broad_flux)
        + 1.20 * normalize01(sub_flux + low_flux)
        + 0.65 * normalize01(mid_flux)
        + 0.85 * normalize01(high_flux + air_flux)
        + 0.55 * normalize01(rms_diff)
        + 0.25 * normalize01(zcr_now)
    )

    score = smooth(normalize01(score), n=5)
    return score.astype(np.float32), starts[1:].astype(np.int64)


def hat_score(audio):
    """
    v09 : courbe dédiée hi-hat.
    On cherche les attaques hautes fréquences, même si elles sont plus faibles
    que kick/snare/crash.
    """
    frame = 1024
    hop = 64

    if len(audio) < frame * 2:
        return np.zeros(1, dtype=np.float32), np.asarray([0], dtype=np.int64)

    window = np.hanning(frame).astype(np.float32)
    freqs = np.fft.rfftfreq(frame, d=1.0 / SR)

    low_mask = (freqs >= 35) & (freqs <= 280)
    mid_mask = (freqs >= 280) & (freqs <= 2500)
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

    zcr_now = zcr[1:]

    score = (
        1.8 * normalize01(high_flux)
        + 1.6 * normalize01(air_flux)
        + 0.7 * normalize01(zcr_now)
        - 0.8 * normalize01(low_flux)
        - 0.25 * normalize01(mid_flux)
    )

    score = smooth(normalize01(score), n=3)
    return score.astype(np.float32), starts[1:].astype(np.int64)


def peaks_from_curve(curve, starts, threshold=0.10, min_sep_ms=18.0, max_peaks=32, kind="peak"):
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

        too_close = False
        for old in chosen:
            if abs(sample - int(old["sample"])) < min_sep:
                too_close = True
                break

        if too_close:
            continue

        chosen.append(peak)

        if max_peaks > 0 and len(chosen) >= max_peaks:
            break

    return sorted(chosen, key=lambda p: int(p["sample"]))


def merge_main_and_hat_peaks(main_peaks, hat_peaks, max_slices=24, min_merge_ms=12.0, hat_rescue=8):
    """
    Merge intelligent :
    - on garde d'abord les meilleurs pics généraux
    - puis on force quelques hats dédiés, même s'ils sont faibles
    - on évite juste les doublons trop proches
    """
    min_merge = int(round(min_merge_ms * SR / 1000.0))

    chosen = []

    # 1. Main peaks.
    for peak in sorted(main_peaks, key=lambda p: p["score"], reverse=True):
        if len(chosen) >= max_slices:
            break

        sample = int(peak["sample"])

        if any(abs(sample - int(old["sample"])) < min_merge for old in chosen):
            continue

        chosen.append(peak)

    # 2. Hat rescue protégés.
    rescued = 0
    for peak in sorted(hat_peaks, key=lambda p: p["score"], reverse=True):
        if len(chosen) >= max_slices:
            break

        if rescued >= hat_rescue:
            break

        sample = int(peak["sample"])

        if any(abs(sample - int(old["sample"])) < min_merge for old in chosen):
            continue

        peak = dict(peak)
        peak["kind"] = "hat_rescue"
        chosen.append(peak)
        rescued += 1

    # 3. Si on n'a pas rempli, on ajoute encore des hats.
    for peak in sorted(hat_peaks, key=lambda p: p["score"], reverse=True):
        if len(chosen) >= max_slices:
            break

        sample = int(peak["sample"])

        if any(abs(sample - int(old["sample"])) < min_merge for old in chosen):
            continue

        peak = dict(peak)
        peak["kind"] = "hat_extra"
        chosen.append(peak)

    chosen = sorted(chosen, key=lambda p: int(p["sample"]))

    if not chosen or chosen[0]["sample"] > int(20 * SR / 1000.0):
        chosen = [{"sample": 0, "score": 1.0, "kind": "start"}] + chosen

    return chosen[:max_slices]


def pick_all_peaks(
    audio,
    threshold=0.16,
    min_sep_ms=32.0,
    max_slices=24,
    hat_threshold=0.055,
    hat_min_sep_ms=14.0,
    hat_rescue=8,
):
    """
    v09 : détection complète + secours hi-hat.
    """
    main_curve, main_starts = onset_score(audio)
    hats_curve, hats_starts = hat_score(audio)

    main_peaks = peaks_from_curve(
        main_curve,
        main_starts,
        threshold=threshold,
        min_sep_ms=min_sep_ms,
        max_peaks=max_slices,
        kind="main",
    )

    hat_peaks = peaks_from_curve(
        hats_curve,
        hats_starts,
        threshold=hat_threshold,
        min_sep_ms=hat_min_sep_ms,
        max_peaks=max(16, hat_rescue * 3),
        kind="hat",
    )

    return merge_main_and_hat_peaks(
        main_peaks=main_peaks,
        hat_peaks=hat_peaks,
        max_slices=max_slices,
        min_merge_ms=10.0,
        hat_rescue=hat_rescue,
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
    parser.add_argument("--threshold", type=float, default=0.16)
    parser.add_argument("--min-sep-ms", type=float, default=32.0)
    parser.add_argument("--max-slices", type=int, default=24)
    parser.add_argument("--hat-threshold", type=float, default=0.055)
    parser.add_argument("--hat-min-sep-ms", type=float, default=14.0)
    parser.add_argument("--hat-rescue", type=int, default=8)
    parser.add_argument("--min-len-ms", type=float, default=50.0)
    parser.add_argument("--max-len-ms", type=float, default=700.0)
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

    peaks = pick_all_peaks(
        loop,
        threshold=args.threshold,
        min_sep_ms=args.min_sep_ms,
        max_slices=args.max_slices,
        hat_threshold=args.hat_threshold,
        hat_min_sep_ms=args.hat_min_sep_ms,
        hat_rescue=args.hat_rescue,
    )

    if len(peaks) < 2:
        # fallback 8 divisions seulement si rien n'est détecté
        peaks = []
        for i in range(8):
            peaks.append({
                "sample": int(round(i * len(loop) / 8.0)),
                "score": 0.1,
            })

    min_len = int(round(args.min_len_ms * SR / 1000.0))
    max_len = int(round(args.max_len_ms * SR / 1000.0))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = OUT_DIR / safe
    out_dir.mkdir(parents=True, exist_ok=True)

    preview_wav = out_dir / f"{safe}_source_loop_preview_v09.wav"
    sf.write(preview_wav, normalize_for_preview(loop), SR)

    blocks = []

    for i, peak in enumerate(peaks):
        a = int(peak["sample"])

        if i + 1 < len(peaks):
            b = int(peaks[i + 1]["sample"])
        else:
            b = len(loop)

        b = max(a + min_len, b)
        b = min(len(loop), a + max_len, b)

        if b <= a:
            continue

        source_a = int(start + a)
        source_b = int(start + b)

        duration_ms = (source_b - source_a) / SR * 1000.0

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
            "duration_ms": round(duration_ms, 4),
            "slice_storage": "index_only",
            "audio_path": None,
            "onset_score": round(float(peak.get("score", 0.0)), 6),
            "detect_kind": peak.get("kind", "main"),
        })

    pair_json = PAIR_BLOCKS_DIR / f"{safe}_pair_blocks_v02.json"
    PAIR_BLOCKS_DIR.mkdir(parents=True, exist_ok=True)

    backup = backup_if_exists(pair_json)

    median_ms = float(np.median([b["duration_ms"] for b in blocks])) if blocks else 120.0

    data = {
        "version": "pair_blocks_v02_hat_rescue_index_only_v09",
        "storage": "index_only_single_source_audio",
        "source_audio": str(source_path),
        "safe": safe,
        "sample_rate": SR,
        "source_duration_ms": round(len(full_audio) / SR * 1000.0, 4),
        "loop_start_sample": int(start),
        "loop_end_sample": int(end),
        "loop_start_ms": round(start / SR * 1000.0, 4),
        "loop_end_ms": round(end / SR * 1000.0, 4),
        "loop_duration_ms": round(len(loop) / SR * 1000.0, 4),
        "slice_count": len(blocks),
        "step_ms": round(median_ms, 6),
        "slice_method": "hat_rescue_index_only_v09",
        "threshold": args.threshold,
        "min_sep_ms": args.min_sep_ms,
        "hat_threshold": args.hat_threshold,
        "hat_min_sep_ms": args.hat_min_sep_ms,
        "hat_rescue": args.hat_rescue,
        "min_len_ms": args.min_len_ms,
        "max_len_ms": args.max_len_ms,
        "max_slices": args.max_slices,
        "auto_trim": bool(args.auto_trim),
        "shift_ms": args.shift_ms,
        "old_pair_json": str(old_json) if old_json else None,
        "backup_pair_json": str(backup) if backup else None,
        "preview_wav": str(preview_wav),
        "blocks": blocks,
    }

    pair_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    report = out_dir / f"{safe}_all_slices_v09_report.txt"
    lines = []
    lines.append(f"Musical16 index-only v09: {safe}")
    lines.append(f"Source: {source_path}")
    lines.append(f"Storage: one source audio + JSON slice indexes")
    lines.append(f"Detected slices: {len(blocks)}")
    lines.append(f"JSON: {pair_json}")
    lines.append(f"Preview: {preview_wav}")
    if backup:
        lines.append(f"Backup old JSON: {backup}")
    lines.append("")

    for b in blocks:
        lines.append(
            f"slice {b['pair']:02d} | "
            f"{b['source_start_ms']:9.2f} -> {b['source_end_ms']:9.2f} ms | "
            f"dur {b['duration_ms']:9.2f} ms | onset {b['onset_score']} | {b.get('detect_kind', 'main')}"
        )

    report.write_text("\\n".join(lines) + "\\n", encoding="utf-8")

    print("OK autoslice hat-rescue index-only v09")
    print("Source :", source_path)
    print("JSON :", pair_json)
    print("Preview :", preview_wav)
    print("Rapport :", report)
    print("Slices détectées :", len(blocks))
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
