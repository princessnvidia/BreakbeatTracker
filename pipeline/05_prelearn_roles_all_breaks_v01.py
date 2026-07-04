#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
import wave

import numpy as np


PROJECT = Path(".").resolve()
PAIR_DIR = PROJECT / "dataset" / "pair_blocks_v02"
BREAK_DIR = PROJECT / "breaks"
LEARNING_DIR = PROJECT / "dataset" / "learning"
OVERRIDES_PATH = LEARNING_DIR / "break_role_overrides_v01.json"
REPORT_CSV = LEARNING_DIR / "role_detection_report_v83.csv"


def slug(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown_break"


def norm(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_overrides():
    if not OVERRIDES_PATH.exists():
        return {
            "version": "break_role_overrides_v01",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "breaks": {},
        }

    try:
        data = load_json(OVERRIDES_PATH)
    except Exception:
        data = {
            "version": "break_role_overrides_v01",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "breaks": {},
        }

    if "breaks" not in data or not isinstance(data["breaks"], dict):
        data["breaks"] = {}

    return data


def pairblock_source_name(path):
    name = Path(path).name
    if name.endswith("_pair_blocks_v02.json"):
        name = name[:-len("_pair_blocks_v02.json")]
    return name


def extract_blocks(data):
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        return []

    for key in [
        "pair_blocks",
        "blocks",
        "pairs",
        "slices",
        "items",
        "data",
    ]:
        value = data.get(key)
        if isinstance(value, list):
            return value

    if any(k in data for k in ["source_start_sample", "start_sample", "pair"]):
        return [data]

    return []


def first_existing_path(candidates):
    for p in candidates:
        if not p:
            continue
        p = Path(str(p)).expanduser()

        if p.is_absolute() and p.exists():
            return p

        q = PROJECT / p
        if q.exists():
            return q

        q = BREAK_DIR / p.name
        if q.exists():
            return q

        q = BREAK_DIR / str(p)
        if q.exists():
            return q

    return None


def resolve_source_audio(block, root_data):
    values = []

    for obj in [block, root_data]:
        if isinstance(obj, dict):
            for key in [
                "source_audio",
                "audio_path",
                "source_path",
                "file",
                "path",
                "filename",
            ]:
                if obj.get(key):
                    values.append(obj.get(key))

    return first_existing_path(values)


def try_soundfile(path):
    try:
        import soundfile as sf
    except Exception:
        return None

    try:
        y, sr = sf.read(str(path), always_2d=False, dtype="float32")
    except Exception:
        return None

    y = np.asarray(y, dtype=np.float32)

    if y.ndim == 2:
        y = y.mean(axis=1)

    return y, int(sr)


def try_wave(path):
    try:
        with wave.open(str(path), "rb") as wf:
            sr = int(wf.getframerate())
            channels = int(wf.getnchannels())
            sampwidth = int(wf.getsampwidth())
            frames = wf.readframes(wf.getnframes())
    except Exception:
        return None

    if sampwidth == 1:
        arr = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
    elif sampwidth == 2:
        arr = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        vals = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        vals = np.where(vals & 0x800000, vals - 0x1000000, vals)
        arr = vals.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        arr = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        return None

    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)

    return arr.astype(np.float32), sr


def ffprobe_sr(path):
    if shutil.which("ffprobe") is None:
        return None

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate",
        "-of", "default=nw=1:nk=1",
        str(path),
    ]

    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return int(out)
    except Exception:
        return None


def try_ffmpeg(path, target_sr=None):
    if shutil.which("ffmpeg") is None:
        return None

    sr = target_sr or ffprobe_sr(path) or 44100

    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(sr),
        "pipe:1",
    ]

    try:
        raw = subprocess.check_output(cmd)
    except Exception:
        return None

    y = np.frombuffer(raw, dtype=np.float32).copy()
    return y, int(sr)


AUDIO_CACHE = {}


def read_audio(path, preferred_sr=None):
    path = Path(path).resolve()
    key = (str(path), preferred_sr)

    if key in AUDIO_CACHE:
        return AUDIO_CACHE[key]

    for loader in [
        try_soundfile,
        try_wave,
    ]:
        result = loader(path)
        if result is not None:
            AUDIO_CACHE[key] = result
            return result

    result = try_ffmpeg(path, preferred_sr)

    if result is None:
        raise RuntimeError(f"Impossible de lire audio: {path}")

    AUDIO_CACHE[key] = result
    return result


def get_num(obj, keys, default=None):
    if not isinstance(obj, dict):
        return default

    for key in keys:
        if key in obj and obj[key] is not None:
            try:
                return float(obj[key])
            except Exception:
                pass

    return default


def get_pair(block, index):
    value = get_num(block, ["pair", "slice", "slice_index", "index", "id"], default=index)
    return int(value)


def extract_segment(block, root_data, index):
    audio_path = resolve_source_audio(block, root_data)

    if audio_path is None:
        raise RuntimeError("source_audio introuvable")

    preferred_sr = get_num(block, ["sample_rate", "source_sample_rate", "sr"], default=None)

    if preferred_sr is None and isinstance(root_data, dict):
        preferred_sr = get_num(root_data, ["sample_rate", "source_sample_rate", "sr"], default=None)

    y, sr = read_audio(audio_path, int(preferred_sr) if preferred_sr else None)

    start = get_num(block, ["source_start_sample", "start_sample", "start", "start_frame"], default=None)
    end = get_num(block, ["source_end_sample", "end_sample", "end", "end_frame"], default=None)

    if start is None:
        start_ms = get_num(block, ["source_start_ms", "start_ms"], default=None)
        if start_ms is not None:
            start = int(float(start_ms) * sr / 1000.0)

    if end is None:
        end_ms = get_num(block, ["source_end_ms", "end_ms"], default=None)
        if end_ms is not None:
            end = int(float(end_ms) * sr / 1000.0)

    if start is None:
        start = 0

    if end is None:
        dur_ms = get_num(block, ["duration_ms", "dur_ms"], default=None)
        if dur_ms is not None:
            end = int(start + float(dur_ms) * sr / 1000.0)
        else:
            end = min(len(y), int(start + 0.35 * sr))

    start = int(max(0, min(len(y), start)))
    end = int(max(start + 1, min(len(y), end)))

    return y[start:end].astype(np.float32), sr, audio_path


def band_energy(y, sr, low, high):
    y = np.asarray(y, dtype=np.float32)

    if len(y) <= 0:
        return 0.0

    if len(y) < 256:
        y = np.pad(y, (0, 256 - len(y)))

    n = min(len(y), 4096)

    if n <= 0:
        return 0.0

    chunk = y[:n] * np.hanning(n).astype(np.float32)
    mag = np.abs(np.fft.rfft(chunk)).astype(np.float32)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))

    mask = (freqs >= low) & (freqs <= high)

    if not np.any(mask):
        return 0.0

    return float(np.sum(mag[mask] ** 2))


def analyse_slice(y, sr):
    y = np.asarray(y, dtype=np.float32)

    if len(y) <= 0:
        y = np.zeros(256, dtype=np.float32)

    full = y[:min(len(y), int(sr * 0.600))]
    attack = y[:min(len(y), int(sr * 0.090))]
    tail = y[min(len(y), int(sr * 0.120)):min(len(y), int(sr * 0.420))]

    if len(attack) <= 0:
        attack = full

    rms = float(np.sqrt(np.mean(full * full) + 1e-12))
    peak = float(np.max(np.abs(full)) + 1e-12)

    attack_rms = float(np.sqrt(np.mean(attack * attack) + 1e-12))

    if len(tail) > 8:
        tail_rms = float(np.sqrt(np.mean(tail * tail) + 1e-12))
    else:
        tail_rms = 0.0

    tail_ratio = tail_rms / (attack_rms + 1e-9)

    if len(attack) > 3:
        zcr = float(np.mean(np.abs(np.diff(np.signbit(attack).astype(np.float32)))))
    else:
        zcr = 0.0

    sub = band_energy(attack, sr, 35, 90)
    low = band_energy(attack, sr, 90, 220)
    lowmid = band_energy(attack, sr, 220, 650)
    mid = band_energy(attack, sr, 650, 2800)
    high = band_energy(attack, sr, 2800, 9000)
    air = band_energy(attack, sr, 9000, 16000)

    total = sub + low + lowmid + mid + high + air + 1e-9

    sub_r = sub / total
    low_r = low / total
    lowmid_r = lowmid / total
    mid_r = mid / total
    high_r = high / total
    air_r = air / total

    centroid = (
        60 * sub_r
        + 150 * low_r
        + 420 * lowmid_r
        + 1500 * mid_r
        + 5500 * high_r
        + 12000 * air_r
    )
    centroid_norm = min(1.0, centroid / 12000.0)

    energy_boost = min(1.0, rms * 10.0)

    kick_score = (
        3.8 * sub_r
        + 2.8 * low_r
        + 0.6 * lowmid_r
        + 0.45 * energy_boost
        - 1.2 * high_r
        - 1.0 * air_r
        - 0.30 * zcr
    )

    snare_score = (
        1.0 * lowmid_r
        + 2.3 * mid_r
        + 1.7 * high_r
        + 0.85 * air_r
        + 0.70 * zcr
        + 0.30 * tail_ratio
        - 1.15 * sub_r
        - 0.55 * low_r
    )

    hat_score = (
        2.7 * high_r
        + 2.8 * air_r
        + 1.15 * zcr
        + 0.75 * centroid_norm
        - 1.45 * sub_r
        - 0.85 * low_r
        - 0.35 * tail_ratio
    )

    ghost_score = (
        0.75 * snare_score
        + 0.35 * high_r
        + 0.25 * zcr
        - 0.55 * energy_boost
    )

    silence_score = 1.0 if peak < 0.012 or rms < 0.004 else 0.0

    return {
        "rms": rms,
        "peak": peak,
        "zcr": zcr,
        "tail_ratio": tail_ratio,
        "sub_r": sub_r,
        "low_r": low_r,
        "lowmid_r": lowmid_r,
        "mid_r": mid_r,
        "high_r": high_r,
        "air_r": air_r,
        "kick": float(kick_score),
        "snare": float(snare_score),
        "hat": float(hat_score),
        "ghost_snare": float(ghost_score),
        "silence": float(silence_score),
    }


def pick_top(scored, role, n, avoid=None, bad=None):
    avoid = set(int(x) for x in (avoid or []))
    bad = set(int(x) for x in (bad or []))

    rows = []

    for row in scored:
        pair = int(row["pair"])

        if pair in avoid or pair in bad:
            continue

        rows.append((float(row["scores"].get(role, 0.0)), pair))

    rows.sort(reverse=True)

    out = []

    for score, pair in rows:
        if pair not in out:
            out.append(pair)

        if len(out) >= n:
            break

    return out


def detect_roles_for_pairblock(path, force=False):
    root_data = load_json(path)
    blocks = extract_blocks(root_data)
    break_name = pairblock_source_name(path)

    scored = []
    skipped = 0
    source_audio = None

    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            skipped += 1
            continue

        pair = get_pair(block, index)

        try:
            y, sr, audio_path = extract_segment(block, root_data, index)
            source_audio = str(audio_path)
            scores = analyse_slice(y, sr)
        except Exception as exc:
            skipped += 1
            print(f"[skip] {break_name} pair={pair}: {exc}")
            continue

        guessed = max(
            ["kick", "snare", "hat", "ghost_snare"],
            key=lambda role: scores.get(role, 0.0),
        )

        scored.append({
            "break": break_name,
            "pair": int(pair),
            "guessed": guessed,
            "scores": scores,
        })

    if not scored:
        return {
            "break": break_name,
            "source_audio": source_audio,
            "roles": {
                "kick": [],
                "snare": [],
                "hat": [],
                "ghost_snare": [],
                "bad": [],
            },
            "scored": [],
            "skipped": skipped,
        }

    bad = sorted(set(
        int(row["pair"])
        for row in scored
        if row["scores"].get("silence", 0.0) >= 1.0
    ))

    kick = pick_top(scored, "kick", 5, bad=bad)
    snare = pick_top(scored, "snare", 5, avoid=kick[:2], bad=bad)
    hat = pick_top(scored, "hat", 8, avoid=set(kick[:2]) | set(snare[:2]), bad=bad)
    ghost = pick_top(scored, "ghost_snare", 6, avoid=kick[:1], bad=bad)

    # Fallbacks pour ne jamais laisser un break sans rôle.
    all_pairs = [int(row["pair"]) for row in scored if int(row["pair"]) not in bad]

    if not kick and all_pairs:
        kick = all_pairs[:1]

    if not snare and all_pairs:
        snare = all_pairs[:1]

    if not hat and all_pairs:
        hat = all_pairs[:1]

    if not ghost:
        ghost = snare[:3] or all_pairs[:1]

    return {
        "break": break_name,
        "source_audio": source_audio,
        "roles": {
            "kick": sorted(set(kick)),
            "snare": sorted(set(snare)),
            "hat": sorted(set(hat)),
            "ghost_snare": sorted(set(ghost)),
            "bad": sorted(set(bad)),
        },
        "scored": scored,
        "skipped": skipped,
    }


def merge_roles(existing_roles, auto_roles, force=False):
    out = {}

    for role in ["kick", "snare", "hat", "ghost_snare", "bad"]:
        old = existing_roles.get(role, []) if isinstance(existing_roles, dict) else []
        old = [int(x) for x in old if str(x).lstrip("-").isdigit()]
        auto = [int(x) for x in auto_roles.get(role, [])]

        if force:
            out[role] = sorted(set(auto))
        else:
            if old:
                out[role] = sorted(set(old))
            else:
                out[role] = sorted(set(auto))

    return out


def write_report(rows):
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "break",
        "pair",
        "guessed",
        "kick",
        "snare",
        "hat",
        "ghost_snare",
        "rms",
        "peak",
        "sub_r",
        "low_r",
        "mid_r",
        "high_r",
        "air_r",
        "zcr",
        "tail_ratio",
    ]

    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            scores = row.get("scores", {})
            writer.writerow({
                "break": row.get("break"),
                "pair": row.get("pair"),
                "guessed": row.get("guessed"),
                "kick": round(float(scores.get("kick", 0.0)), 6),
                "snare": round(float(scores.get("snare", 0.0)), 6),
                "hat": round(float(scores.get("hat", 0.0)), 6),
                "ghost_snare": round(float(scores.get("ghost_snare", 0.0)), 6),
                "rms": round(float(scores.get("rms", 0.0)), 6),
                "peak": round(float(scores.get("peak", 0.0)), 6),
                "sub_r": round(float(scores.get("sub_r", 0.0)), 6),
                "low_r": round(float(scores.get("low_r", 0.0)), 6),
                "mid_r": round(float(scores.get("mid_r", 0.0)), 6),
                "high_r": round(float(scores.get("high_r", 0.0)), 6),
                "air_r": round(float(scores.get("air_r", 0.0)), 6),
                "zcr": round(float(scores.get("zcr", 0.0)), 6),
                "tail_ratio": round(float(scores.get("tail_ratio", 0.0)), 6),
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="Nom d'un break, ex: Camo_Break_-_3A")
    parser.add_argument("--all", action="store_true", help="Analyse tous les breaks")
    parser.add_argument("--force", action="store_true", help="Écrase les labels existants")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pair_files = sorted(PAIR_DIR.glob("*_pair_blocks_v02.json"))

    if args.source:
        target = norm(args.source)
        pair_files = [
            p for p in pair_files
            if norm(pairblock_source_name(p)) == target or target in norm(pairblock_source_name(p))
        ]

    if args.limit:
        pair_files = pair_files[:args.limit]

    if not pair_files:
        print("Aucun pair_block trouvé.")
        sys.exit(1)

    overrides = load_overrides()
    all_report_rows = []

    print("")
    print("[v83-prelearn] fichiers à analyser:", len(pair_files))
    print("[v83-prelearn] force:", args.force)
    print("[v83-prelearn] dry-run:", args.dry_run)
    print("")

    for i, path in enumerate(pair_files, start=1):
        result = detect_roles_for_pairblock(path, force=args.force)

        break_name = result["break"]
        auto_roles = result["roles"]

        all_report_rows.extend(result["scored"])

        br = overrides["breaks"].setdefault(break_name, {})
        old_roles = br.get("roles", {})

        merged = merge_roles(old_roles, auto_roles, force=args.force)

        br["roles"] = merged
        br["auto_roles_v83"] = auto_roles
        br["role_detector_v83"] = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_audio": result.get("source_audio"),
            "skipped_slices": result.get("skipped", 0),
            "force": bool(args.force),
        }

        br["score_preview_v83"] = [
            {
                "pair": row["pair"],
                "guessed": row["guessed"],
                "kick": round(float(row["scores"].get("kick", 0.0)), 4),
                "snare": round(float(row["scores"].get("snare", 0.0)), 4),
                "hat": round(float(row["scores"].get("hat", 0.0)), 4),
                "ghost_snare": round(float(row["scores"].get("ghost_snare", 0.0)), 4),
            }
            for row in result["scored"][:64]
        ]

        print(f"[{i:03d}/{len(pair_files):03d}] {break_name}")
        print("  kick       :", merged["kick"])
        print("  snare      :", merged["snare"])
        print("  hat        :", merged["hat"])
        print("  ghost_snare:", merged["ghost_snare"])
        print("  bad        :", merged["bad"])
        print("")

    overrides["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    overrides["role_detector_v83_global"] = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pairblocks_scanned": len(pair_files),
        "force": bool(args.force),
    }

    if not args.dry_run:
        save_json(OVERRIDES_PATH, overrides)
        write_report(all_report_rows)

        print("[OK] rôles écrits dans :", OVERRIDES_PATH)
        print("[OK] rapport écrit dans:", REPORT_CSV)
    else:
        print("[dry-run] aucun fichier écrit.")

    print("")
    print("Relance ensuite l'app v82/v81 et Generate Candidate utilisera ces rôles.")
    print("")


if __name__ == "__main__":
    main()
