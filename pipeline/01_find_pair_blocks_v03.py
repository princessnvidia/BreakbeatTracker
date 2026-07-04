#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_find_pair_blocks_v03.py — BreakbeatAI better slicer

Usage:
  cd ~/Applications/BreakbeatAI
  python pipeline/01_find_pair_blocks_v03.py --source "Amen" --compat-v02

Output:
  dataset/pair_blocks_v03/<safe>_pair_blocks_v03.json
  dataset/pair_blocks_v03/<safe>_pair_000.wav ...
  if --compat-v02:
    backup old v02 json, then write a v02-compatible json pointing to v03 slices.
"""

from pathlib import Path
import argparse, json, re, shutil, sys
import numpy as np
import soundfile as sf

SR = 44100
OUT_V03 = Path("dataset/pair_blocks_v03")
OUT_V02 = Path("dataset/pair_blocks_v02")
AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp3", ".m4a"}
SEARCH_DIRS = [
    Path("dataset/source_audio"), Path("dataset/sources"), Path("dataset/audio"),
    Path("dataset/raw_audio"), Path("dataset"), Path("audio"), Path("samples"), Path(".")
]


def slug(s):
    s = Path(str(s)).stem.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "source"


def norm(y, peak=0.98):
    m = float(np.max(np.abs(y))) if len(y) else 0.0
    return y if m < 1e-9 else (y / m * peak).astype(np.float32)


def rms(y):
    return float(np.sqrt(np.mean(y * y) + 1e-12)) if len(y) else 0.0


def resample_linear(y, old_sr, new_sr):
    if old_sr == new_sr:
        return y.astype(np.float32)
    dur = len(y) / old_sr
    old_x = np.linspace(0, dur, len(y), endpoint=False)
    new_len = max(1, int(round(dur * new_sr)))
    new_x = np.linspace(0, dur, new_len, endpoint=False)
    return np.interp(new_x, old_x, y).astype(np.float32)


def load_audio(path):
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != SR:
        print(f"[slicer v03] resample {sr} -> {SR}")
        y = resample_linear(y, sr, SR)
        sr = SR
    y = y - float(np.mean(y))
    return norm(y), sr


def find_source(query):
    p = Path(query).expanduser()
    if p.exists() and p.is_file():
        return p

    for folder, suffix in [(OUT_V02, "*_pair_blocks_v02.json"), (OUT_V03, "*_pair_blocks_v03.json")]:
        for jp in sorted(folder.glob(suffix)):
            if query.lower() not in jp.name.lower():
                continue
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                continue
            for key in ("source_audio", "source", "input_audio"):
                val = data.get(key)
                if isinstance(val, str) and Path(val).exists():
                    return Path(val)

    matches = []
    q = query.lower()
    for base in SEARCH_DIRS:
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in AUDIO_EXTS:
                continue
            if q in f.name.lower() and "pair_blocks" not in str(f) and "_pair_" not in f.name:
                matches.append(f)

    matches = sorted(set(matches), key=lambda x: (len(str(x)), str(x)))
    if not matches:
        print(f"[slicer v03] source introuvable: {query}")
        print('Donne un chemin direct, ex: --source "dataset/source_audio/Amen.wav"')
        sys.exit(1)

    if len(matches) > 1:
        print("[slicer v03] plusieurs sources possibles, je prends la première:")
        for m in matches[:8]:
            print("  -", m)

    return matches[0]


def frames(y, size, hop):
    if len(y) < size:
        z = np.zeros(size, dtype=np.float32)
        z[:len(y)] = y
        return z[None, :]
    n = 1 + (len(y) - size) // hop
    shape = (n, size)
    strides = (y.strides[0] * hop, y.strides[0])
    return np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides).copy()


def smooth(x, n):
    n = max(1, int(n))
    if n <= 1:
        return x.astype(np.float32)
    k = np.ones(n, dtype=np.float32) / n
    return np.convolve(x, k, mode="same").astype(np.float32)


def robust01(x):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    lo, hi = np.percentile(x, [5, 95])
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def onset_envelope(y, sr, frame_size=1024, hop=128):
    fr = frames(y, frame_size, hop)
    win = np.hanning(frame_size).astype(np.float32)
    mag = np.abs(np.fft.rfft(fr * win[None, :], axis=1)).astype(np.float32)

    r = np.sqrt(np.mean(fr * fr, axis=1) + 1e-12)
    log_r = np.log1p(100 * r)
    rms_diff = np.maximum(np.diff(log_r, prepend=log_r[0]), 0)

    mag_n = mag / (mag.sum(axis=1, keepdims=True) + 1e-9)
    flux = np.maximum(np.diff(mag_n, axis=0, prepend=mag_n[:1]), 0).sum(axis=1)

    freqs = np.fft.rfftfreq(frame_size, 1 / sr)
    hi = mag[:, freqs >= 4500].sum(axis=1)
    hi_diff = np.maximum(np.diff(np.log1p(hi), prepend=np.log1p(hi[0])), 0)

    env = 0.50 * robust01(flux) + 0.30 * robust01(rms_diff) + 0.20 * robust01(hi_diff)
    return robust01(smooth(env, 3))


def pick_peaks(env, sr, hop, threshold=0.15, local_ms=220, min_sep_ms=45):
    if len(env) < 3:
        return []

    rad = max(3, int((local_ms / 1000 * sr) / hop))
    mean = smooth(env, rad * 2 + 1)
    mean2 = smooth(env * env, rad * 2 + 1)
    std = np.sqrt(np.maximum(mean2 - mean * mean, 0))
    gate = mean + 0.65 * std + threshold

    candidates = []
    for i in range(1, len(env) - 1):
        if env[i] >= env[i - 1] and env[i] >= env[i + 1] and env[i] >= gate[i]:
            candidates.append(i)

    min_sep = max(1, int((min_sep_ms / 1000 * sr) / hop))
    chosen = []
    for idx in sorted(candidates, key=lambda i: float(env[i]), reverse=True):
        if all(abs(idx - c) >= min_sep for c in chosen):
            chosen.append(idx)
    return sorted(chosen)


def refine_start(y, rough, sr):
    a = max(0, rough - int(0.030 * sr))
    b = min(len(y), rough + int(0.010 * sr))
    seg = np.abs(y[a:b]).astype(np.float32)
    if len(seg) < 8:
        return max(0, rough)
    e = smooth(seg, max(3, int(0.0015 * sr)))
    pk = float(e.max())
    if pk < 1e-7:
        return max(0, rough)
    th = max(pk * 0.12, float(np.median(e)) + 0.7 * float(np.std(e)))
    pk_i = int(np.argmax(e))
    start = pk_i
    for i in range(0, pk_i + 1):
        if e[i] >= th:
            start = i
            break
    return max(0, a + start - int(0.003 * sr))


def merge_close(onsets, y, sr, min_gap_ms=32):
    if not onsets:
        return []
    gap = int(sr * min_gap_ms / 1000)
    out = [int(onsets[0])]
    for o in onsets[1:]:
        o = int(o)
        prev = out[-1]
        if o - prev < gap:
            win = int(0.008 * sr)
            p1 = float(np.max(np.abs(y[prev:min(len(y), prev + win)])))
            p2 = float(np.max(np.abs(y[o:min(len(y), o + win)])))
            if p2 > p1 * 1.15:
                out[-1] = o
        else:
            out.append(o)
    return out


def fade(y, sr, ms=2):
    if len(y) < 16:
        return y.astype(np.float32)
    n = max(1, min(int(sr * ms / 1000), len(y) // 4))
    r = np.linspace(0, 1, n, dtype=np.float32)
    z = y.astype(np.float32).copy()
    z[:n] *= r
    z[-n:] *= r[::-1]
    return z


def trim_tail(y, sr, min_ms=35, max_ms=420):
    min_len = int(sr * min_ms / 1000)
    max_len = min(len(y), int(sr * max_ms / 1000))
    z = y[:max_len]
    if len(z) <= min_len:
        return z
    size = max(64, int(0.008 * sr))
    hop = size // 2
    fr = frames(z, size, hop)
    e = np.sqrt(np.mean(fr * fr, axis=1) + 1e-12)
    th = max(float(e.max()) * 0.035, float(np.percentile(e, 20)) * 1.2)
    min_frame = max(0, (min_len - size) // hop)
    quiet = 0
    end_frame = len(e) - 1
    for i in range(min_frame, len(e)):
        if e[i] < th:
            quiet += 1
            if quiet >= 3:
                end_frame = i
                break
        else:
            quiet = 0
    end = min(len(z), max(min_len, end_frame * hop + size))
    return z[:end]


def role_guess(y, sr):
    n = min(len(y), int(0.18 * sr))
    if n < 64:
        return {"role_guess": "hat", "role_confidence": 0.0}
    seg = y[:n] * np.hanning(n).astype(np.float32)
    mag = np.abs(np.fft.rfft(seg)).astype(np.float32)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    total = float(mag.sum() + 1e-9)
    low = float(mag[freqs < 180].sum() / total)
    mid = float(mag[(freqs >= 900) & (freqs < 4500)].sum() / total)
    high = float(mag[freqs >= 4500].sum() / total)
    centroid = float((freqs * mag).sum() / total)

    if low > 0.33 and centroid < 2400:
        role, conf = "kick", min(1.0, 0.45 + low)
    elif high > 0.35 and centroid > 3800:
        role, conf = "hat", min(1.0, 0.45 + high)
    else:
        role, conf = "snare", min(1.0, 0.35 + mid + 0.5 * high)

    return {
        "role_guess": role,
        "role_confidence": round(conf, 4),
        "centroid_hz": round(centroid, 2),
        "low_ratio": round(low, 4),
        "mid_ratio": round(mid, 4),
        "high_ratio": round(high, 4),
    }


def build_slices(y, sr, onsets, min_ms, max_ms, tail_margin_ms, min_rms):
    if not onsets:
        onsets = [0]
    onsets = sorted(set(max(0, min(len(y) - 1, int(o))) for o in onsets))

    if onsets[0] > int(0.040 * sr) and rms(y[:onsets[0]]) > min_rms * 1.35:
        onsets.insert(0, 0)

    min_len = int(sr * min_ms / 1000)
    max_len = int(sr * max_ms / 1000)
    tail = int(sr * tail_margin_ms / 1000)

    out = []
    for i, start in enumerate(onsets):
        if i + 1 < len(onsets):
            end = max(start + min_len, onsets[i + 1] - tail)
        else:
            end = len(y)
        end = min(len(y), start + max_len, max(start + min_len, end))
        clip = y[start:end].astype(np.float32)
        if rms(clip) < min_rms:
            continue
        clip = trim_tail(clip, sr, min_ms, max_ms)
        clip = norm(fade(clip, sr), 0.95)
        out.append({
            "pair": len(out),
            "start_sample": int(start),
            "end_sample": int(start + len(clip)),
            "start_sec": round(start / sr, 6),
            "end_sec": round((start + len(clip)) / sr, 6),
            "duration_ms": round(len(clip) / sr * 1000, 3),
            "rms": round(rms(clip), 6),
            "audio": clip,
            **role_guess(clip, sr),
        })
    return out


def write_outputs(source, safe, y, sr, slices, args):
    OUT_V03.mkdir(parents=True, exist_ok=True)
    blocks = []

    for s in slices:
        wav = OUT_V03 / f"{safe}_pair_{s['pair']:03d}.wav"
        sf.write(wav, s["audio"], sr)
        blocks.append({
            "pair": s["pair"],
            "audio_path": str(wav),
            "duration_ms": s["duration_ms"],
            "start_sample": s["start_sample"],
            "end_sample": s["end_sample"],
            "start_sec": s["start_sec"],
            "end_sec": s["end_sec"],
            "rms": s["rms"],
            "role_guess": s["role_guess"],
            "role_confidence": s["role_confidence"],
            "centroid_hz": s.get("centroid_hz"),
            "low_ratio": s.get("low_ratio"),
            "mid_ratio": s.get("mid_ratio"),
            "high_ratio": s.get("high_ratio"),
        })

    data = {
        "version": "pair_blocks_v03_better_transient_slicer",
        "source": str(source),
        "source_audio": str(source),
        "safe": safe,
        "sample_rate": sr,
        "source_duration_sec": round(len(y) / sr, 6),
        "params": vars(args),
        "blocks": blocks,
    }

    json_v03 = OUT_V03 / f"{safe}_pair_blocks_v03.json"
    json_v03.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    report = OUT_V03 / f"{safe}_slicer_v03_report.txt"
    counts = {}
    for b in blocks:
        counts[b["role_guess"]] = counts.get(b["role_guess"], 0) + 1
    lines = [
        "BreakbeatAI slicer v03 report",
        f"source: {source}",
        f"slices: {len(blocks)}",
        f"role guesses: {counts}",
        "",
    ]
    for b in blocks:
        lines.append(
            f"{b['pair']:03d} | {b['start_sec']:7.4f}s -> {b['end_sec']:7.4f}s | "
            f"{b['duration_ms']:7.2f}ms | {b['role_guess']:5s} {b['role_confidence']}"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_v02 = None
    if args.compat_v02:
        OUT_V02.mkdir(parents=True, exist_ok=True)
        json_v02 = OUT_V02 / f"{safe}_pair_blocks_v02.json"
        if json_v02.exists():
            bak = json_v02.with_suffix(json_v02.suffix + ".bak")
            shutil.copy2(json_v02, bak)
            print("[slicer v03] backup ancien v02:", bak)
        compat = dict(data)
        compat["version"] = "pair_blocks_v02_compat_from_v03"
        compat["v03_json"] = str(json_v03)
        json_v02.write_text(json.dumps(compat, indent=2, ensure_ascii=False), encoding="utf-8")

    return json_v03, json_v02, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--compat-v02", action="store_true")
    ap.add_argument("--frame-size", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--min-sep-ms", type=float, default=45.0)
    ap.add_argument("--min-slice-ms", type=float, default=35.0)
    ap.add_argument("--max-slice-ms", type=float, default=420.0)
    ap.add_argument("--tail-margin-ms", type=float, default=4.0)
    ap.add_argument("--min-rms", type=float, default=0.006)
    args = ap.parse_args()

    source = find_source(args.source)
    safe = slug(source.name)
    print("[slicer v03] source:", source)
    print("[slicer v03] safe:", safe)

    y, sr = load_audio(source)
    print(f"[slicer v03] duration: {len(y)/sr:.3f}s")

    env = onset_envelope(y, sr, args.frame_size, args.hop)
    peak_frames = pick_peaks(env, sr, args.hop, args.threshold, min_sep_ms=args.min_sep_ms)
    rough = [p * args.hop for p in peak_frames]
    refined = [refine_start(y, r, sr) for r in rough]
    refined = merge_close(refined, y, sr, min_gap_ms=args.min_sep_ms * 0.70)

    slices = build_slices(
        y, sr, refined,
        min_ms=args.min_slice_ms,
        max_ms=args.max_slice_ms,
        tail_margin_ms=args.tail_margin_ms,
        min_rms=args.min_rms,
    )

    if not slices:
        print("[slicer v03] aucun slice. Essaie:")
        print('python pipeline/01_find_pair_blocks_v03.py --source "Amen" --threshold 0.10 --compat-v02')
        sys.exit(1)

    json_v03, json_v02, report = write_outputs(source, safe, y, sr, slices, args)

    counts = {}
    for s in slices:
        counts[s["role_guess"]] = counts.get(s["role_guess"], 0) + 1

    print("[slicer v03] terminé")
    print("[slicer v03] slices:", len(slices))
    print("[slicer v03] role guesses:", counts)
    print("[slicer v03] json v03:", json_v03)
    if json_v02:
        print("[slicer v03] compat tracker v02:", json_v02)
    print("[slicer v03] report:", report)
    print("")
    print("Puis lance:")
    print(f'python pipeline/03_tracker_editor_app_v08.py --source "{safe}"')


if __name__ == "__main__":
    main()
