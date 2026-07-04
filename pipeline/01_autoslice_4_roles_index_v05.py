#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_autoslice_4_roles_index_v05.py

Auto-slice index-only en 4 sons :
    0 = snare
    1 = hat
    2 = kick
    3 = crash

Ne génère PAS de WAV par slice.
Garde un seul fichier audio source.
Le JSON contient uniquement les points start/end.

Usage :
    python pipeline/01_autoslice_4_roles_index_v05.py --source "amen"
    python pipeline/01_autoslice_4_roles_index_v05.py --source "camo"

Réglages utiles :
    --threshold 0.20
    --min-sep-ms 35
    --auto-trim
    --start-ms 100 --end-ms 1800
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
OUT_DIR = Path("dataset/slice_indexes_v05")

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

ROLE_ORDER = ["snare", "hat", "kick", "crash"]


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


def normalize_for_preview(y, peak=0.95):
    y = np.asarray(y, dtype=np.float32)

    if len(y) == 0:
        return y

    m = float(np.max(np.abs(y)))

    if m <= 1e-9:
        return y

    return (y / m * peak).astype(np.float32)


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


def onset_curve(audio):
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

    score = (
        0.80 * normalize01(broad_flux)
        + 1.30 * normalize01(low_flux)
        + 0.50 * normalize01(mid_flux)
        + 0.35 * normalize01(high_flux)
        + 0.50 * normalize01(rms_diff)
    )

    score = smooth(normalize01(score), n=5)
    score_starts = starts[1:]

    return score.astype(np.float32), score_starts.astype(np.int64)


def pick_candidates(audio, threshold=0.22, min_sep_ms=35.0, max_candidates=24):
    score, starts = onset_curve(audio)

    if len(score) < 3:
        return []

    threshold_abs = float(np.max(score)) * float(threshold)
    min_sep = int(round(min_sep_ms * SR / 1000.0))

    peaks = []

    for i in range(1, len(score) - 1):
        if score[i] < threshold_abs:
            continue

        if score[i] >= score[i - 1] and score[i] >= score[i + 1]:
            peaks.append({
                "sample": int(starts[i]),
                "score": float(score[i]),
            })

    peaks = sorted(peaks, key=lambda p: p["score"], reverse=True)

    chosen = []

    for peak in peaks:
        sample = int(peak["sample"])

        if sample <= 0 or sample >= len(audio) - 1:
            continue

        too_close = False
        for old in chosen:
            if abs(sample - int(old["sample"])) < min_sep:
                too_close = True
                break

        if too_close:
            continue

        chosen.append(peak)

        if len(chosen) >= max_candidates:
            break

    chosen = sorted(chosen, key=lambda p: int(p["sample"]))

    return chosen


def spectral_features(y):
    y = np.asarray(y, dtype=np.float32)

    if len(y) < 256:
        y = np.pad(y, (0, max(0, 256 - len(y))))

    # Fenêtre courte pour l'attaque.
    attack = y[:min(len(y), int(SR * 0.080))]
    full = y[:min(len(y), int(SR * 0.700))]

    def band_energy(sig, low, high):
        if len(sig) < 256:
            sig = np.pad(sig, (0, 256 - len(sig)))

        n = min(2048, len(sig))
        sig = sig[:n] * np.hanning(n)
        mag = np.abs(np.fft.rfft(sig)).astype(np.float32)
        freqs = np.fft.rfftfreq(n, d=1.0 / SR)
        mask = (freqs >= low) & (freqs <= high)

        if not np.any(mask):
            return 0.0

        return float(np.sum(mag[mask] ** 2))

    low = band_energy(attack, 35, 180)
    low2 = band_energy(attack, 180, 320)
    mid = band_energy(attack, 320, 2500)
    high = band_energy(attack, 2500, 12000)

    total = low + low2 + mid + high + 1e-9

    rms_attack = float(np.sqrt(np.mean(attack * attack) + 1e-12))
    rms_full = float(np.sqrt(np.mean(full * full) + 1e-12))

    # Tail pour distinguer crash vs hat.
    tail_start = int(SR * 0.120)
    tail_end = min(len(y), int(SR * 0.700))

    if tail_end > tail_start:
        tail = y[tail_start:tail_end]
        tail_rms = float(np.sqrt(np.mean(tail * tail) + 1e-12))
    else:
        tail_rms = 0.0

    zcr = 0.0
    if len(attack) > 2:
        zcr = float(np.mean(np.abs(np.diff(np.signbit(attack).astype(np.float32)))))

    return {
        "low_ratio": low / total,
        "low2_ratio": low2 / total,
        "mid_ratio": mid / total,
        "high_ratio": high / total,
        "rms_attack": rms_attack,
        "rms_full": rms_full,
        "tail_rms": tail_rms,
        "tail_ratio": tail_rms / (rms_attack + 1e-9),
        "zcr": zcr,
    }


def candidate_window(audio, cand_sample, next_sample=None):
    start = int(cand_sample)

    if next_sample is None:
        end = min(len(audio), start + int(SR * 0.900))
    else:
        end = int(next_sample)

    end = max(start + 256, min(len(audio), end))

    return audio[start:end].astype(np.float32), start, end


def score_roles(features, onset_strength):
    low = features["low_ratio"]
    mid = features["mid_ratio"]
    high = features["high_ratio"]
    tail = features["tail_ratio"]
    zcr = features["zcr"]
    rms = features["rms_attack"]

    # Scores heuristiques.
    kick = (
        2.6 * low
        + 0.9 * features["low2_ratio"]
        + 0.6 * onset_strength
        + 0.4 * rms
        - 0.7 * high
        - 0.4 * tail
    )

    snare = (
        1.4 * mid
        + 0.9 * high
        + 0.8 * zcr
        + 0.6 * onset_strength
        - 0.8 * low
        - 0.2 * tail
    )

    hat = (
        2.1 * high
        + 0.9 * zcr
        + 0.4 * onset_strength
        - 0.8 * low
        - 0.9 * tail
    )

    crash = (
        1.5 * high
        + 1.8 * tail
        + 0.6 * zcr
        + 0.2 * onset_strength
        - 0.5 * low
    )

    return {
        "kick": float(kick),
        "snare": float(snare),
        "hat": float(hat),
        "crash": float(crash),
    }


def fallback_grid_candidates(audio):
    points = []
    for i in range(8):
        points.append({
            "sample": int(round(i * len(audio) / 8.0)),
            "score": 0.1,
        })
    return points


def choose_four_roles(audio, threshold=0.22, min_sep_ms=35.0):
    candidates = pick_candidates(
        audio,
        threshold=threshold,
        min_sep_ms=min_sep_ms,
        max_candidates=32,
    )

    if len(candidates) < 4:
        candidates = fallback_grid_candidates(audio)

    candidates = sorted(candidates, key=lambda c: int(c["sample"]))

    enriched = []

    for idx, cand in enumerate(candidates):
        next_sample = None
        if idx + 1 < len(candidates):
            next_sample = int(candidates[idx + 1]["sample"])

        y, start, end = candidate_window(audio, int(cand["sample"]), next_sample)
        feat = spectral_features(y)
        role_scores = score_roles(feat, float(cand.get("score", 0.0)))

        enriched.append({
            "candidate_index": idx,
            "start": int(start),
            "end": int(end),
            "onset_score": float(cand.get("score", 0.0)),
            "features": feat,
            "role_scores": role_scores,
        })

    selected = {}
    used = set()

    # Ordre volontaire :
    # - kick d'abord pour garantir qu'on trouve le grave si présent
    # - crash avant hat car crash peut être confondu avec hat
    selection_order = ["kick", "snare", "crash", "hat"]

    for role in selection_order:
        ranked = sorted(
            enriched,
            key=lambda c: c["role_scores"][role],
            reverse=True,
        )

        chosen = None

        for cand in ranked:
            if cand["candidate_index"] in used:
                continue

            chosen = cand
            break

        if chosen is None:
            chosen = ranked[0]

        selected[role] = chosen
        used.add(chosen["candidate_index"])

    # Sortie dans l'ordre demandé par toi.
    return {
        role: selected[role]
        for role in ROLE_ORDER
    }, enriched


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
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--min-sep-ms", type=float, default=35.0)
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

    selected, enriched = choose_four_roles(
        loop,
        threshold=args.threshold,
        min_sep_ms=args.min_sep_ms,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = OUT_DIR / safe
    out_dir.mkdir(parents=True, exist_ok=True)

    preview_wav = out_dir / f"{safe}_source_loop_preview_v05.wav"
    sf.write(preview_wav, normalize_for_preview(loop), SR)

    blocks = []

    for pair, role in enumerate(ROLE_ORDER):
        cand = selected[role]

        source_a = int(start + cand["start"])
        source_b = int(start + cand["end"])

        duration_ms = (source_b - source_a) / SR * 1000.0

        blocks.append({
            "pair": pair,
            "name": role,
            "role": role,
            "manual_role": role,
            "formal_role": role,
            "source_audio": str(source_path),
            "source_start_sample": source_a,
            "source_end_sample": source_b,
            "source_start_ms": round(source_a / SR * 1000.0, 4),
            "source_end_ms": round(source_b / SR * 1000.0, 4),
            "duration_ms": round(duration_ms, 4),
            "slice_storage": "index_only",
            "audio_path": None,
            "onset_score": round(float(cand["onset_score"]), 6),
            "role_scores": {
                k: round(float(v), 6)
                for k, v in cand["role_scores"].items()
            },
            "features": {
                k: round(float(v), 6)
                for k, v in cand["features"].items()
            },
        })

    pair_json = PAIR_BLOCKS_DIR / f"{safe}_pair_blocks_v02.json"
    PAIR_BLOCKS_DIR.mkdir(parents=True, exist_ok=True)

    backup = backup_if_exists(pair_json)

    median_ms = float(np.median([b["duration_ms"] for b in blocks])) if blocks else 120.0

    data = {
        "version": "pair_blocks_v02_4_roles_index_only_v05",
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
        "slice_count": 4,
        "roles": ROLE_ORDER,
        "step_ms": round(median_ms, 6),
        "slice_method": "onset_spectral_role_classifier_v05",
        "threshold": args.threshold,
        "min_sep_ms": args.min_sep_ms,
        "auto_trim": bool(args.auto_trim),
        "shift_ms": args.shift_ms,
        "old_pair_json": str(old_json) if old_json else None,
        "backup_pair_json": str(backup) if backup else None,
        "preview_wav": str(preview_wav),
        "blocks": blocks,
    }

    pair_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    report = out_dir / f"{safe}_4_roles_v05_report.txt"
    lines = []
    lines.append(f"4-role index-only autoslice v05: {safe}")
    lines.append(f"Source: {source_path}")
    lines.append("Roles: snare, hat, kick, crash")
    lines.append("Storage: one source audio + JSON start/end indexes")
    lines.append(f"JSON: {pair_json}")
    lines.append(f"Preview: {preview_wav}")
    if backup:
        lines.append(f"Backup old JSON: {backup}")
    lines.append("")
    for b in blocks:
        lines.append(
            f"pair {b['pair']} {b['role']:6s} | "
            f"{b['source_start_ms']:9.2f} -> {b['source_end_ms']:9.2f} ms | "
            f"dur {b['duration_ms']:9.2f} ms | "
            f"scores {b['role_scores']}"
        )

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("OK auto-slice 4 rôles index-only")
    print("Source :", source_path)
    print("JSON :", pair_json)
    print("Preview :", preview_wav)
    print("Rapport :", report)
    if backup:
        print("Backup ancien JSON :", backup)
    print("")
    for b in blocks:
        print(
            f"pair {b['pair']} = {b['role']:6s} | "
            f"{b['source_start_ms']:9.2f} -> {b['source_end_ms']:9.2f} ms | "
            f"dur {b['duration_ms']:9.2f} ms"
        )


if __name__ == "__main__":
    main()
