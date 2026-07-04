#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_autoslice_4_roles_index_v06_rolecurves.py

Slicer index-only 4 rôles avec courbes séparées :
    pair 0 = snare
    pair 1 = hat
    pair 2 = kick
    pair 3 = crash

Différence avec v05 :
- le hat n'est plus choisi parmi les gros transients généraux
- il a sa propre détection haute fréquence
- utile pour Camo où le vrai hi-hat est plus discret

Ne génère PAS de WAV par slice.
Le JSON pointe vers le fichier source + start/end samples.
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
OUT_DIR = Path("dataset/slice_indexes_v06")

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

ROLE_LENGTH_MS = {
    "snare": 210.0,
    "hat": 95.0,
    "kick": 230.0,
    "crash": 520.0,
}


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

    if end <= start:
        return 0, len(audio)

    return start, end


def make_role_curves(audio):
    frame = 1024
    hop = 96

    if len(audio) < frame * 2:
        starts = np.asarray([0], dtype=np.int64)
        zero = np.asarray([0.0], dtype=np.float32)
        return {"snare": zero, "hat": zero, "kick": zero, "crash": zero}, starts

    window = np.hanning(frame).astype(np.float32)
    freqs = np.fft.rfftfreq(frame, d=1.0 / SR)

    sub_mask = (freqs >= 35) & (freqs <= 90)
    low_mask = (freqs >= 90) & (freqs <= 220)
    low2_mask = (freqs >= 220) & (freqs <= 420)
    mid_mask = (freqs >= 420) & (freqs <= 2500)
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
        zero = np.zeros(max(1, len(mags)), dtype=np.float32)
        return {"snare": zero, "hat": zero, "kick": zero, "crash": zero}, starts

    diff = np.maximum(0.0, mags[1:] - mags[:-1])

    sub_flux = np.sum(diff[:, sub_mask], axis=1)
    low_flux = np.sum(diff[:, low_mask], axis=1)
    low2_flux = np.sum(diff[:, low2_mask], axis=1)
    mid_flux = np.sum(diff[:, mid_mask], axis=1)
    high_flux = np.sum(diff[:, high_mask], axis=1)
    air_flux = np.sum(diff[:, air_mask], axis=1)
    broad_flux = np.sum(diff, axis=1)

    rms_diff = np.maximum(0.0, rms[1:] - rms[:-1])
    zcr_now = zcr[1:]
    score_starts = starts[1:]

    kick_curve = (
        1.8 * normalize01(sub_flux)
        + 1.4 * normalize01(low_flux)
        + 0.7 * normalize01(low2_flux)
        + 0.4 * normalize01(rms_diff)
        + 0.2 * normalize01(broad_flux)
        - 0.35 * normalize01(high_flux)
    )

    snare_curve = (
        1.2 * normalize01(mid_flux)
        + 0.9 * normalize01(high_flux)
        + 0.6 * normalize01(broad_flux)
        + 0.4 * normalize01(rms_diff)
        - 0.45 * normalize01(sub_flux + low_flux)
    )

    # Le point important : hat = courbe haute fréquence dédiée,
    # plus sensible, avec pénalité sur le grave.
    hat_curve = (
        1.7 * normalize01(high_flux)
        + 1.3 * normalize01(air_flux)
        + 0.6 * normalize01(zcr_now)
        + 0.25 * normalize01(rms_diff)
        - 0.65 * normalize01(sub_flux + low_flux)
        - 0.20 * normalize01(mid_flux)
    )

    crash_curve = (
        1.3 * normalize01(high_flux)
        + 1.2 * normalize01(air_flux)
        + 0.4 * normalize01(broad_flux)
        + 0.4 * normalize01(zcr_now)
        - 0.30 * normalize01(sub_flux + low_flux)
    )

    curves = {
        "kick": smooth(normalize01(kick_curve), 5),
        "snare": smooth(normalize01(snare_curve), 5),
        "hat": smooth(normalize01(hat_curve), 3),
        "crash": smooth(normalize01(crash_curve), 7),
    }

    return curves, score_starts


def pick_peaks_for_role(curve, starts, threshold, min_sep_ms, max_candidates=16):
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
                "curve_score": float(curve[i]),
            })

    peaks = sorted(peaks, key=lambda p: p["curve_score"], reverse=True)

    chosen = []

    for peak in peaks:
        sample = int(peak["sample"])

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

    return sorted(chosen, key=lambda p: int(p["sample"]))


def band_energy(sig, low, high):
    sig = np.asarray(sig, dtype=np.float32)

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


def spectral_features(y):
    y = np.asarray(y, dtype=np.float32)

    attack = y[:min(len(y), int(SR * 0.080))]
    full = y[:min(len(y), int(SR * 0.700))]

    sub = band_energy(attack, 35, 90)
    low = band_energy(attack, 90, 220)
    low2 = band_energy(attack, 220, 420)
    mid = band_energy(attack, 420, 2500)
    high = band_energy(attack, 2500, 9000)
    air = band_energy(attack, 9000, 16000)

    total = sub + low + low2 + mid + high + air + 1e-9

    rms_attack = float(np.sqrt(np.mean(attack * attack) + 1e-12))
    rms_full = float(np.sqrt(np.mean(full * full) + 1e-12))

    tail_start = int(SR * 0.120)
    tail_end = min(len(y), int(SR * 0.700))

    if tail_end > tail_start:
        tail = y[tail_start:tail_end]
        tail_rms = float(np.sqrt(np.mean(tail * tail) + 1e-12))
    else:
        tail_rms = 0.0

    if len(attack) > 2:
        zcr = float(np.mean(np.abs(np.diff(np.signbit(attack).astype(np.float32)))))
    else:
        zcr = 0.0

    return {
        "sub_ratio": sub / total,
        "low_ratio": low / total,
        "low2_ratio": low2 / total,
        "mid_ratio": mid / total,
        "high_ratio": high / total,
        "air_ratio": air / total,
        "rms_attack": rms_attack,
        "rms_full": rms_full,
        "tail_rms": tail_rms,
        "tail_ratio": tail_rms / (rms_attack + 1e-9),
        "zcr": zcr,
    }


def role_score(role, features, curve_score):
    sub = features["sub_ratio"]
    low = features["low_ratio"]
    low2 = features["low2_ratio"]
    mid = features["mid_ratio"]
    high = features["high_ratio"]
    air = features["air_ratio"]
    tail = features["tail_ratio"]
    zcr = features["zcr"]
    rms = features["rms_attack"]

    if role == "kick":
        return (
            2.5 * sub
            + 2.0 * low
            + 0.8 * low2
            + 0.6 * curve_score
            + 0.3 * rms
            - 0.8 * high
            - 0.6 * air
            - 0.4 * tail
        )

    if role == "snare":
        return (
            1.5 * mid
            + 0.9 * high
            + 0.6 * air
            + 0.7 * zcr
            + 0.5 * curve_score
            - 0.8 * sub
            - 0.6 * low
            - 0.25 * tail
        )

    if role == "hat":
        return (
            2.3 * high
            + 2.0 * air
            + 0.9 * zcr
            + 0.7 * curve_score
            - 1.1 * sub
            - 0.9 * low
            - 0.45 * mid
            - 1.2 * tail
        )

    if role == "crash":
        return (
            1.6 * high
            + 1.7 * air
            + 2.0 * tail
            + 0.5 * zcr
            + 0.4 * curve_score
            - 0.7 * sub
            - 0.6 * low
        )

    return 0.0


def choose_role(audio, role, peaks, used_samples, role_len_ms):
    role_len = int(round(role_len_ms * SR / 1000.0))

    scored = []

    for peak in peaks:
        start = int(peak["sample"])
        end = min(len(audio), start + role_len)

        if end <= start + 128:
            continue

        y = audio[start:end]
        features = spectral_features(y)
        score = role_score(role, features, float(peak["curve_score"]))

        # Évite de sélectionner exactement le même transient pour deux rôles.
        for used in used_samples:
            if abs(start - used) < int(35 * SR / 1000.0):
                score -= 2.5

        scored.append({
            "role": role,
            "start": start,
            "end": end,
            "curve_score": float(peak["curve_score"]),
            "role_score": float(score),
            "features": features,
        })

    if not scored:
        # fallback tout simple
        start = 0
        end = min(len(audio), start + role_len)
        features = spectral_features(audio[start:end])
        return {
            "role": role,
            "start": start,
            "end": end,
            "curve_score": 0.0,
            "role_score": 0.0,
            "features": features,
        }, []

    scored = sorted(scored, key=lambda c: c["role_score"], reverse=True)

    return scored[0], scored


def choose_four_roles(audio, args):
    curves, starts = make_role_curves(audio)

    peaks_by_role = {
        "kick": pick_peaks_for_role(curves["kick"], starts, args.threshold, args.min_sep_ms, 20),
        "snare": pick_peaks_for_role(curves["snare"], starts, args.threshold, args.min_sep_ms, 20),
        "hat": pick_peaks_for_role(curves["hat"], starts, args.hat_threshold, args.hat_min_sep_ms, 24),
        "crash": pick_peaks_for_role(curves["crash"], starts, args.crash_threshold, args.min_sep_ms, 20),
    }

    selected = {}
    ranked = {}
    used_samples = []

    # Crash avant hat, car crash peut ressembler à un hat long.
    # Hat ensuite utilise une pénalité de tail pour éviter les crashs.
    order = ["kick", "snare", "crash", "hat"]

    for role in order:
        chosen, role_ranked = choose_role(
            audio=audio,
            role=role,
            peaks=peaks_by_role[role],
            used_samples=used_samples,
            role_len_ms=getattr(args, f"{role}_len_ms"),
        )

        selected[role] = chosen
        ranked[role] = role_ranked
        used_samples.append(int(chosen["start"]))

    return selected, ranked, peaks_by_role


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

    parser.add_argument("--threshold", type=float, default=0.18)
    parser.add_argument("--hat-threshold", type=float, default=0.08)
    parser.add_argument("--crash-threshold", type=float, default=0.12)

    parser.add_argument("--min-sep-ms", type=float, default=35.0)
    parser.add_argument("--hat-min-sep-ms", type=float, default=22.0)

    parser.add_argument("--snare-len-ms", type=float, default=ROLE_LENGTH_MS["snare"])
    parser.add_argument("--hat-len-ms", type=float, default=ROLE_LENGTH_MS["hat"])
    parser.add_argument("--kick-len-ms", type=float, default=ROLE_LENGTH_MS["kick"])
    parser.add_argument("--crash-len-ms", type=float, default=ROLE_LENGTH_MS["crash"])

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

    selected, ranked, peaks_by_role = choose_four_roles(loop, args)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = OUT_DIR / safe
    out_dir.mkdir(parents=True, exist_ok=True)

    preview_wav = out_dir / f"{safe}_source_loop_preview_v06.wav"
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
            "curve_score": round(float(cand["curve_score"]), 6),
            "role_score": round(float(cand["role_score"]), 6),
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
        "version": "pair_blocks_v02_4_roles_index_only_v06_rolecurves",
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
        "slice_method": "role_specific_onset_curves_v06",
        "threshold": args.threshold,
        "hat_threshold": args.hat_threshold,
        "crash_threshold": args.crash_threshold,
        "min_sep_ms": args.min_sep_ms,
        "hat_min_sep_ms": args.hat_min_sep_ms,
        "auto_trim": bool(args.auto_trim),
        "shift_ms": args.shift_ms,
        "old_pair_json": str(old_json) if old_json else None,
        "backup_pair_json": str(backup) if backup else None,
        "preview_wav": str(preview_wav),
        "blocks": blocks,
    }

    pair_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    report = out_dir / f"{safe}_4_roles_v06_report.txt"

    lines = []
    lines.append(f"4-role index-only autoslice v06 rolecurves: {safe}")
    lines.append(f"Source: {source_path}")
    lines.append("pair 0 = snare")
    lines.append("pair 1 = hat")
    lines.append("pair 2 = kick")
    lines.append("pair 3 = crash")
    lines.append(f"JSON: {pair_json}")
    lines.append(f"Preview: {preview_wav}")
    if backup:
        lines.append(f"Backup old JSON: {backup}")
    lines.append("")

    lines.append("SELECTED:")
    for b in blocks:
        lines.append(
            f"pair {b['pair']} {b['role']:6s} | "
            f"{b['source_start_ms']:9.2f} -> {b['source_end_ms']:9.2f} ms | "
            f"dur {b['duration_ms']:9.2f} ms | "
            f"score {b['role_score']:8.4f} | features {b['features']}"
        )

    lines.append("")
    lines.append("TOP HAT CANDIDATES:")
    for idx, cand in enumerate(ranked.get("hat", [])[:12]):
        lines.append(
            f"hat #{idx:02d} | "
            f"{(start + cand['start']) / SR * 1000.0:9.2f} ms | "
            f"score {cand['role_score']:8.4f} | "
            f"curve {cand['curve_score']:8.4f} | "
            f"features { {k: round(float(v), 4) for k, v in cand['features'].items()} }"
        )

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("OK auto-slice 4 rôles index-only v06")
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
            f"dur {b['duration_ms']:9.2f} ms | "
            f"score {b['role_score']:8.4f}"
        )


if __name__ == "__main__":
    main()
