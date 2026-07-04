#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import json
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
REPORT_CSV = LEARNING_DIR / "role_detection_report_v84.csv"


def slug(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown_break"


def norm(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_overrides():
    data = load_json(OVERRIDES_PATH, None)
    if not isinstance(data, dict):
        data = {
            "version": "break_role_overrides_v01",
            "breaks": {},
        }

    data.setdefault("breaks", {})
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

    for key in ["pair_blocks", "blocks", "pairs", "slices", "items", "data"]:
        value = data.get(key)
        if isinstance(value, list):
            return value

    return []


def find_path(value):
    if not value:
        return None

    p = Path(str(value)).expanduser()

    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(PROJECT / p)
        candidates.append(BREAK_DIR / p)
        candidates.append(BREAK_DIR / p.name)

    for c in candidates:
        if c.exists():
            return c

    return None


def resolve_audio(block, root):
    for obj in [block, root]:
        if not isinstance(obj, dict):
            continue

        for key in ["source_audio", "audio_path", "source_path", "file", "path", "filename"]:
            p = find_path(obj.get(key))
            if p:
                return p

    return None


AUDIO_CACHE = {}


def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sw == 1:
        y = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        y = (y - 128.0) / 128.0
    elif sw == 2:
        y = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        vals = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        vals = np.where(vals & 0x800000, vals - 0x1000000, vals)
        y = vals.astype(np.float32) / 8388608.0
    elif sw == 4:
        y = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError("format wav non supporté")

    if ch > 1:
        y = y.reshape(-1, ch).mean(axis=1)

    return y.astype(np.float32), int(sr)


def read_soundfile(path):
    import soundfile as sf
    y, sr = sf.read(str(path), always_2d=False, dtype="float32")
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y, int(sr)


def read_ffmpeg(path):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg introuvable")

    sr = 44100
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

    raw = subprocess.check_output(cmd)
    y = np.frombuffer(raw, dtype=np.float32).copy()
    return y, sr


def read_audio(path):
    path = Path(path).resolve()

    if path in AUDIO_CACHE:
        return AUDIO_CACHE[path]

    loaders = []

    try:
        import soundfile  # noqa
        loaders.append(read_soundfile)
    except Exception:
        pass

    loaders.append(read_wav)
    loaders.append(read_ffmpeg)

    last = None
    for loader in loaders:
        try:
            result = loader(path)
            AUDIO_CACHE[path] = result
            return result
        except Exception as exc:
            last = exc

    raise RuntimeError(f"audio illisible {path}: {last}")


def get_num(obj, keys, default=None):
    if not isinstance(obj, dict):
        return default

    for key in keys:
        if key not in obj:
            continue

        try:
            return float(obj[key])
        except Exception:
            pass

    return default


def extract_segment(block, root, index):
    audio_path = resolve_audio(block, root)
    if audio_path is None:
        raise RuntimeError("source_audio introuvable")

    y, sr = read_audio(audio_path)

    start = get_num(block, ["source_start_sample", "start_sample", "start", "start_frame"], None)
    end = get_num(block, ["source_end_sample", "end_sample", "end", "end_frame"], None)

    if start is None:
        start_ms = get_num(block, ["source_start_ms", "start_ms"], None)
        if start_ms is not None:
            start = int(start_ms * sr / 1000.0)

    if end is None:
        end_ms = get_num(block, ["source_end_ms", "end_ms"], None)
        if end_ms is not None:
            end = int(end_ms * sr / 1000.0)

    if start is None:
        start = 0

    if end is None:
        dur_ms = get_num(block, ["duration_ms", "dur_ms"], 280.0)
        end = start + int(dur_ms * sr / 1000.0)

    start = int(max(0, min(len(y), start)))
    end = int(max(start + 1, min(len(y), end)))

    return y[start:end].astype(np.float32), sr, audio_path


def band_energy(y, sr, low, high):
    y = np.asarray(y, dtype=np.float32)

    if len(y) < 512:
        y = np.pad(y, (0, 512 - len(y)))

    n = min(len(y), 4096)
    if n <= 0:
        return 0.0

    w = np.hanning(n).astype(np.float32)
    chunk = y[:n] * w
    mag = np.abs(np.fft.rfft(chunk)).astype(np.float32)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    mask = (freqs >= low) & (freqs <= high)

    if not np.any(mask):
        return 0.0

    return float(np.sum(mag[mask] ** 2))


def spectral_features(y, sr):
    y = np.asarray(y, dtype=np.float32)

    if len(y) == 0:
        y = np.zeros(512, dtype=np.float32)

    full = y[:min(len(y), int(sr * 0.55))]
    attack = y[:min(len(y), int(sr * 0.08))]
    body = y[min(len(y), int(sr * 0.03)):min(len(y), int(sr * 0.18))]
    tail = y[min(len(y), int(sr * 0.14)):min(len(y), int(sr * 0.42))]

    if len(attack) == 0:
        attack = full

    rms = float(np.sqrt(np.mean(full * full) + 1e-12))
    peak = float(np.max(np.abs(full)) + 1e-12)
    attack_rms = float(np.sqrt(np.mean(attack * attack) + 1e-12))

    if len(body) > 16:
        body_rms = float(np.sqrt(np.mean(body * body) + 1e-12))
    else:
        body_rms = 0.0

    if len(tail) > 16:
        tail_rms = float(np.sqrt(np.mean(tail * tail) + 1e-12))
    else:
        tail_rms = 0.0

    tail_ratio = tail_rms / (attack_rms + 1e-9)
    body_ratio = body_rms / (attack_rms + 1e-9)

    if len(attack) > 4:
        zcr = float(np.mean(np.abs(np.diff(np.signbit(attack).astype(np.float32)))))
    else:
        zcr = 0.0

    sub = band_energy(attack, sr, 35, 95)
    low = band_energy(attack, sr, 95, 220)
    lowmid = band_energy(attack, sr, 220, 650)
    mid = band_energy(attack, sr, 650, 2600)
    high = band_energy(attack, sr, 2600, 8500)
    air = band_energy(attack, sr, 8500, 16000)

    total = sub + low + lowmid + mid + high + air + 1e-12

    sub_r = sub / total
    low_r = low / total
    lowmid_r = lowmid / total
    mid_r = mid / total
    high_r = high / total
    air_r = air / total

    low_total = sub_r + low_r
    mid_total = lowmid_r + mid_r
    high_total = high_r + air_r

    centroid = (
        65 * sub_r
        + 150 * low_r
        + 420 * lowmid_r
        + 1400 * mid_r
        + 5200 * high_r
        + 12000 * air_r
    ) / 12000.0

    return {
        "rms": rms,
        "peak": peak,
        "attack_rms": attack_rms,
        "body_ratio": body_ratio,
        "tail_ratio": tail_ratio,
        "zcr": zcr,
        "sub_r": sub_r,
        "low_r": low_r,
        "lowmid_r": lowmid_r,
        "mid_r": mid_r,
        "high_r": high_r,
        "air_r": air_r,
        "low_total": low_total,
        "mid_total": mid_total,
        "high_total": high_total,
        "centroid": centroid,
    }


def robust_z(values):
    arr = np.asarray(values, dtype=np.float32)

    if len(arr) == 0:
        return []

    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)) + 1e-9)

    return [float((x - med) / (1.4826 * mad + 1e-9)) for x in arr]


def score_rows(rows):
    """
    Détection améliorée :
    - scores bruts par slice
    - normalisation relative dans le break
    - marges de confiance
    """
    raw = []

    for row in rows:
        f = row["features"]

        kick_raw = (
            3.8 * f["sub_r"]
            + 2.7 * f["low_r"]
            + 0.8 * f["attack_rms"]
            - 1.2 * f["high_total"]
            - 0.7 * f["zcr"]
            - 0.5 * f["tail_ratio"]
        )

        snare_raw = (
            1.2 * f["lowmid_r"]
            + 2.4 * f["mid_r"]
            + 1.7 * f["high_r"]
            + 0.9 * f["air_r"]
            + 0.8 * f["zcr"]
            + 0.5 * f["tail_ratio"]
            - 1.4 * f["sub_r"]
        )

        hat_raw = (
            2.7 * f["high_r"]
            + 2.6 * f["air_r"]
            + 1.2 * f["zcr"]
            + 0.9 * f["centroid"]
            - 1.6 * f["low_total"]
            - 0.8 * f["tail_ratio"]
        )

        ghost_raw = (
            0.75 * snare_raw
            + 0.35 * f["high_r"]
            + 0.25 * f["zcr"]
            - 0.65 * f["attack_rms"]
        )

        raw.append({
            "kick": float(kick_raw),
            "snare": float(snare_raw),
            "hat": float(hat_raw),
            "ghost_snare": float(ghost_raw),
        })

    for role in ["kick", "snare", "hat", "ghost_snare"]:
        zs = robust_z([r[role] for r in raw])
        for row, z in zip(rows, zs):
            row["scores"][role] = z

    for row in rows:
        f = row["features"]
        scores = row["scores"]

        # Garde-fous anti "n'importe quel sample"
        if f["peak"] < 0.012 or f["rms"] < 0.004:
            row["bad_reason"] = "near_silence"
            scores["kick"] -= 3
            scores["snare"] -= 3
            scores["hat"] -= 3
            scores["ghost_snare"] -= 3

        # Un hat ne doit pas être trop grave.
        if f["low_total"] > 0.55:
            scores["hat"] -= 2.0

        # Un kick ne doit pas être surtout aigu.
        if f["high_total"] > 0.55:
            scores["kick"] -= 2.0

        # Une snare trop sub devient suspecte.
        if f["sub_r"] > 0.48:
            scores["snare"] -= 1.5

        sorted_roles = sorted(
            ["kick", "snare", "hat", "ghost_snare"],
            key=lambda r: scores.get(r, -99),
            reverse=True,
        )

        best = sorted_roles[0]
        second = sorted_roles[1]

        row["guessed"] = best
        row["confidence"] = float(scores[best] - scores[second])
        row["best_score"] = float(scores[best])


def pick_role(rows, role, n, avoid=None, min_score=0.15, min_margin=-0.30):
    avoid = set(int(x) for x in (avoid or []))
    candidates = []

    for row in rows:
        pair = int(row["pair"])

        if pair in avoid:
            continue

        if row.get("bad_reason"):
            continue

        score = float(row["scores"].get(role, -99))
        margin = float(row.get("confidence", 0.0)) if row.get("guessed") == role else score - max(
            float(row["scores"].get(r, -99))
            for r in ["kick", "snare", "hat", "ghost_snare"]
            if r != role
        )

        if score < min_score:
            continue

        if margin < min_margin:
            continue

        candidates.append((score + 0.25 * margin, pair))

    candidates.sort(reverse=True)

    out = []
    for score, pair in candidates:
        if pair not in out:
            out.append(pair)
        if len(out) >= n:
            break

    return out


def detect_pairblock(path):
    root = load_json(path, None)
    blocks = extract_blocks(root)
    break_name = pairblock_source_name(path)

    rows = []
    skipped = 0
    source_audio = None

    for i, block in enumerate(blocks):
        if not isinstance(block, dict):
            skipped += 1
            continue

        pair = int(get_num(block, ["pair", "slice", "slice_index", "index", "id"], i))

        try:
            y, sr, audio_path = extract_segment(block, root, i)
            source_audio = str(audio_path)
            features = spectral_features(y, sr)
        except Exception as exc:
            print(f"[skip] {break_name} pair={pair}: {exc}")
            skipped += 1
            continue

        rows.append({
            "break": break_name,
            "pair": pair,
            "features": features,
            "scores": {},
            "guessed": "",
            "confidence": 0.0,
            "best_score": 0.0,
            "bad_reason": "",
        })

    if not rows:
        return break_name, source_audio, [], {
            "kick": [],
            "snare": [],
            "hat": [],
            "ghost_snare": [],
            "bad": [],
        }, skipped

    score_rows(rows)

    bad = sorted(set(
        int(r["pair"]) for r in rows
        if r.get("bad_reason") == "near_silence"
    ))

    kick = pick_role(rows, "kick", 5, avoid=bad, min_score=0.10, min_margin=-0.15)
    snare = pick_role(rows, "snare", 5, avoid=set(bad) | set(kick[:1]), min_score=0.10, min_margin=-0.25)
    hat = pick_role(rows, "hat", 8, avoid=set(bad) | set(kick[:1]) | set(snare[:1]), min_score=0.10, min_margin=-0.25)
    ghost = pick_role(rows, "ghost_snare", 6, avoid=set(bad) | set(kick[:1]), min_score=-0.10, min_margin=-0.50)

    roles = {
        "kick": sorted(set(kick)),
        "snare": sorted(set(snare)),
        "hat": sorted(set(hat)),
        "ghost_snare": sorted(set(ghost)),
        "bad": bad,
    }

    return break_name, source_audio, rows, roles, skipped


def role_confidence(rows, roles):
    out = {}

    for role, pairs in roles.items():
        if role == "bad":
            continue

        role_rows = [r for r in rows if int(r["pair"]) in set(pairs)]
        out[role] = [
            {
                "pair": int(r["pair"]),
                "score": round(float(r["scores"].get(role, 0.0)), 5),
                "guessed": r.get("guessed", ""),
                "confidence": round(float(r.get("confidence", 0.0)), 5),
            }
            for r in role_rows
        ]

    return out


def merge_roles(old_roles, auto_roles, force=False):
    out = {}

    for role in ["kick", "snare", "hat", "ghost_snare", "bad"]:
        old = old_roles.get(role, []) if isinstance(old_roles, dict) else []
        old = [int(x) for x in old if str(x).lstrip("-").isdigit()]
        auto = [int(x) for x in auto_roles.get(role, [])]

        if force or not old:
            out[role] = sorted(set(auto))
        else:
            out[role] = sorted(set(old))

    return out


def write_report(all_rows):
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "break", "pair", "guessed", "confidence", "best_score", "bad_reason",
        "kick", "snare", "hat", "ghost_snare",
        "rms", "peak", "sub_r", "low_r", "mid_r", "high_r", "air_r",
        "low_total", "high_total", "zcr", "tail_ratio", "centroid",
    ]

    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for row in all_rows:
            feat = row["features"]
            scores = row["scores"]
            w.writerow({
                "break": row["break"],
                "pair": row["pair"],
                "guessed": row["guessed"],
                "confidence": round(float(row["confidence"]), 6),
                "best_score": round(float(row["best_score"]), 6),
                "bad_reason": row.get("bad_reason", ""),
                "kick": round(float(scores.get("kick", 0)), 6),
                "snare": round(float(scores.get("snare", 0)), 6),
                "hat": round(float(scores.get("hat", 0)), 6),
                "ghost_snare": round(float(scores.get("ghost_snare", 0)), 6),
                "rms": round(float(feat.get("rms", 0)), 6),
                "peak": round(float(feat.get("peak", 0)), 6),
                "sub_r": round(float(feat.get("sub_r", 0)), 6),
                "low_r": round(float(feat.get("low_r", 0)), 6),
                "mid_r": round(float(feat.get("mid_r", 0)), 6),
                "high_r": round(float(feat.get("high_r", 0)), 6),
                "air_r": round(float(feat.get("air_r", 0)), 6),
                "low_total": round(float(feat.get("low_total", 0)), 6),
                "high_total": round(float(feat.get("high_total", 0)), 6),
                "zcr": round(float(feat.get("zcr", 0)), 6),
                "tail_ratio": round(float(feat.get("tail_ratio", 0)), 6),
                "centroid": round(float(feat.get("centroid", 0)), 6),
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(PAIR_DIR.glob("*_pair_blocks_v02.json"))

    if args.source:
        target = norm(args.source)
        files = [
            p for p in files
            if target in norm(pairblock_source_name(p))
            or norm(pairblock_source_name(p)) in target
        ]

    if args.limit:
        files = files[:args.limit]

    if not files:
        print("Aucun pair_block trouvé.")
        sys.exit(1)

    overrides = load_overrides()
    all_rows = []

    print("")
    print("[v84-detect] fichiers :", len(files))
    print("[v84-detect] force    :", args.force)
    print("[v84-detect] dry-run  :", args.dry_run)
    print("")

    for i, path in enumerate(files, 1):
        break_name, source_audio, rows, auto_roles, skipped = detect_pairblock(path)
        all_rows.extend(rows)

        br = overrides["breaks"].setdefault(break_name, {})
        old_roles = br.get("roles", {})
        merged = merge_roles(old_roles, auto_roles, force=args.force)

        br["roles"] = merged
        br["auto_roles_v84"] = auto_roles
        br["role_confidence_v84"] = role_confidence(rows, merged)
        br["role_detector_v84"] = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_audio": source_audio,
            "skipped": skipped,
            "force": bool(args.force),
            "note": "Détection robuste kick/snare/hat avec scores normalisés par break.",
        }

        print(f"[{i:03d}/{len(files):03d}] {break_name}")
        print("  kick :", merged["kick"])
        print("  snare:", merged["snare"])
        print("  hat  :", merged["hat"])
        print("  ghost:", merged["ghost_snare"])
        print("  bad  :", merged["bad"])
        print("")

    overrides["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    overrides["role_detector_v84_global"] = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": len(files),
        "force": bool(args.force),
    }

    if not args.dry_run:
        save_json(OVERRIDES_PATH, overrides)
        write_report(all_rows)
        print("[OK] écrit :", OVERRIDES_PATH)
        print("[OK] csv   :", REPORT_CSV)
    else:
        print("[dry-run] rien écrit.")


if __name__ == "__main__":
    main()
