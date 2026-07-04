#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_train_beat_roles_v03.py

Trainer role-aware pour BreakbeatAI.

Ancien principe :
    position -> numéro de slice

Nouveau principe :
    position -> rôle musical
    rôle musical -> meilleure slice du break courant

Le modèle apprend :
- à quelles cases arrivent les snares
- à quelles cases arrivent les kicks
- à quelles cases arrivent les hats
- quelles longueurs sont utilisées
- quelles transitions de rôles marchent

Sortie :
    dataset/learning/beat_role_model_v01.json
"""

from pathlib import Path
import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import soundfile as sf


SR = 44100
DATASET_DIR = Path("dataset")
PAIR_BLOCKS_DIR = Path("dataset/pair_blocks_v02")
OUT_MODEL = Path("dataset/learning/beat_role_model_v01.json")

ROLE_NAMES = ["kick", "snare", "hat", "other"]


def sort_key(value):
    s = str(value)
    try:
        return (0, int(s))
    except Exception:
        return (1, s)


def counter_to_dict(counter):
    return {
        str(k): float(v)
        for k, v in sorted(counter.items(), key=lambda kv: sort_key(kv[0]))
    }


def nested_counter_to_dict(d):
    out = {}
    for k, counter in sorted(d.items(), key=lambda kv: sort_key(kv[0])):
        out[str(k)] = counter_to_dict(counter)
    return out


def triple_counter_to_dict(d):
    out = {}
    for a, sub in sorted(d.items(), key=lambda kv: sort_key(kv[0])):
        out[str(a)] = {}
        for b, counter in sorted(sub.items(), key=lambda kv: sort_key(kv[0])):
            out[str(a)][str(b)] = counter_to_dict(counter)
    return out


def safe_name(text):
    out = []
    for ch in str(text):
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
    y, sr = sf.read(path, always_2d=False)

    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)

    y = y.astype(np.float32)
    y = resample_linear(y, sr, SR)
    y = y - float(np.mean(y))

    return y.astype(np.float32)


def band_energy(y, low, high):
    y = np.asarray(y, dtype=np.float32)

    if len(y) < 256:
        y = np.pad(y, (0, 256 - len(y)))

    n = min(len(y), 4096)
    chunk = y[:n] * np.hanning(n).astype(np.float32)

    mag = np.abs(np.fft.rfft(chunk)).astype(np.float32)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)

    mask = (freqs >= low) & (freqs <= high)

    if not np.any(mask):
        return 0.0

    return float(np.sum(mag[mask] ** 2))


def classify_audio_role(y):
    """
    Classe une slice en kick/snare/hat/other.
    Ce n'est pas un label parfait, c'est un repère d'apprentissage.
    """
    y = np.asarray(y, dtype=np.float32)

    if len(y) == 0:
        return "other", {"kick": 0.0, "snare": 0.0, "hat": 0.0, "other": 1.0}

    attack = y[:min(len(y), int(SR * 0.100))]
    full = y[:min(len(y), int(SR * 0.500))]

    sub = band_energy(attack, 35, 90)
    low = band_energy(attack, 90, 250)
    lowmid = band_energy(attack, 250, 700)
    mid = band_energy(attack, 700, 2800)
    high = band_energy(attack, 2800, 9000)
    air = band_energy(attack, 9000, 16000)

    total = sub + low + lowmid + mid + high + air + 1e-9

    sub_r = sub / total
    low_r = low / total
    lowmid_r = lowmid / total
    mid_r = mid / total
    high_r = high / total
    air_r = air / total

    rms = float(np.sqrt(np.mean(full * full) + 1e-12))

    if len(attack) > 3:
        zcr = float(np.mean(np.abs(np.diff(np.signbit(attack).astype(np.float32)))))
    else:
        zcr = 0.0

    tail_start = min(len(y), int(SR * 0.120))
    tail_end = min(len(y), int(SR * 0.450))

    if tail_end > tail_start:
        tail = y[tail_start:tail_end]
        tail_rms = float(np.sqrt(np.mean(tail * tail) + 1e-12))
    else:
        tail_rms = 0.0

    tail_ratio = tail_rms / (rms + 1e-9)

    kick_score = (
        3.2 * sub_r
        + 2.5 * low_r
        + 0.7 * lowmid_r
        + 0.25 * rms
        - 1.0 * high_r
        - 0.9 * air_r
    )

    snare_score = (
        1.1 * lowmid_r
        + 2.1 * mid_r
        + 1.4 * high_r
        + 0.8 * air_r
        + 0.9 * zcr
        - 1.1 * sub_r
        - 0.7 * low_r
    )

    hat_score = (
        2.6 * high_r
        + 2.3 * air_r
        + 1.1 * zcr
        - 1.4 * sub_r
        - 1.0 * low_r
        - 0.35 * mid_r
        - 0.45 * tail_ratio
    )

    other_score = (
        0.5 * tail_ratio
        + 0.2 * rms
    )

    scores = {
        "kick": float(kick_score),
        "snare": float(snare_score),
        "hat": float(hat_score),
        "other": float(other_score),
    }

    role = max(scores, key=scores.get)

    # Sécurité : si c'est très confus, on laisse "other".
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if sorted_scores[0][1] < 0.12:
        role = "other"

    return role, scores


def load_pair_projects():
    projects = {}

    if not PAIR_BLOCKS_DIR.exists():
        return projects

    for path in sorted(PAIR_BLOCKS_DIR.glob("*_pair_blocks_v02.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        safe = data.get("safe") or safe_name(path.stem.replace("_pair_blocks_v02", ""))
        data["_json_path"] = str(path)
        projects[str(safe)] = data

    return projects


def find_project_for_safe(projects, safe, path=None):
    if safe in projects:
        return projects[safe]

    low = str(safe).lower()

    for psafe, data in projects.items():
        if low and low in psafe.lower():
            return data

    if path:
        full = str(path).lower()
        for psafe, data in projects.items():
            if psafe.lower() in full:
                return data

    return None


def get_source_audio_for_project(project, cache):
    source_audio = project.get("source_audio")

    if not source_audio:
        return None

    source_path = Path(source_audio)

    if not source_path.exists():
        source_path = Path(".") / source_audio

    if not source_path.exists():
        return None

    key = str(source_path.resolve())

    if key not in cache:
        cache[key] = load_audio(source_path)

    return cache[key]


def classify_project_pairs(project, audio_cache):
    """
    Retourne :
        pair -> {role, scores}
    """
    out = {}

    audio = get_source_audio_for_project(project, audio_cache)

    if audio is None:
        return out

    for block in project.get("blocks", []):
        try:
            pair = int(block.get("pair"))
            a = int(block.get("source_start_sample"))
            b = int(block.get("source_end_sample"))
        except Exception:
            continue

        a = max(0, min(len(audio) - 1, a))
        b = max(a + 1, min(len(audio), b))

        y = audio[a:b]
        role, scores = classify_audio_role(y)

        out[pair] = {
            "role": role,
            "scores": scores,
            "source_start_ms": block.get("source_start_ms"),
            "source_end_ms": block.get("source_end_ms"),
        }

    return out


def is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False


def looks_like_note(item):
    if not isinstance(item, dict):
        return False

    if "pair" not in item or "x_step" not in item:
        return False

    return is_number(item.get("pair")) and is_number(item.get("x_step"))


def normalize_note(item):
    x_step = int(round(float(item.get("x_step", 0))))
    pair = int(round(float(item.get("pair", 0))))
    length = int(round(float(item.get("length", 2))))

    x_step = max(0, min(31, x_step))
    length = max(1, min(8, length))

    return {
        "x_step": x_step,
        "pair": pair,
        "length": length,
        "lane": int(round(float(item.get("lane", pair)))) if is_number(item.get("lane", pair)) else pair,
    }


def guess_safe(data, path):
    if isinstance(data, dict):
        for key in ("safe", "break", "source"):
            if data.get(key):
                return str(data.get(key))

        project = data.get("project")
        if isinstance(project, dict):
            if project.get("safe"):
                return str(project.get("safe"))

    name = Path(path).stem

    for suffix in (
        "_tracker_app_edit_v58",
        "_tracker_app_edit_v57",
        "_tracker_app_edit_v56",
        "_tracker_app_edit_v55",
        "_tracker_app_edit_v54",
        "_latest_pattern",
        "_pattern",
    ):
        name = name.replace(suffix, "")

    return name


def extract_patterns_from_obj(obj, path):
    patterns = []

    if not isinstance(obj, dict):
        return patterns

    safe = guess_safe(obj, path)

    for key in ("pattern", "items", "notes", "hits", "events"):
        value = obj.get(key)

        if not isinstance(value, list):
            continue

        notes = [normalize_note(x) for x in value if looks_like_note(x)]

        if len(notes) >= 1:
            patterns.append({
                "safe": safe,
                "source_file": str(path),
                "key": key,
                "notes": sorted(notes, key=lambda n: (n["x_step"], n["pair"])),
                "reason": obj.get("reason") or obj.get("save_reason") or obj.get("event_type") or "",
                "time": obj.get("time") or obj.get("saved_at") or "",
            })

    blocks = obj.get("blocks")

    if isinstance(blocks, list):
        # pair_blocks = description de slices, pas beat.
        if not any(isinstance(x, dict) and "source_start_sample" in x for x in blocks):
            notes = [normalize_note(x) for x in blocks if looks_like_note(x)]

            if len(notes) >= 1:
                patterns.append({
                    "safe": safe,
                    "source_file": str(path),
                    "key": "blocks",
                    "notes": sorted(notes, key=lambda n: (n["x_step"], n["pair"])),
                    "reason": obj.get("reason") or "",
                    "time": obj.get("time") or "",
                })

    return patterns


def load_json_patterns(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    patterns = []

    if isinstance(data, dict):
        patterns.extend(extract_patterns_from_obj(data, path))

    elif isinstance(data, list):
        notes = [normalize_note(x) for x in data if looks_like_note(x)]
        if notes:
            patterns.append({
                "safe": safe_name(path.stem),
                "source_file": str(path),
                "key": "root_list",
                "notes": sorted(notes, key=lambda n: (n["x_step"], n["pair"])),
                "reason": "",
                "time": "",
            })

    return patterns


def load_jsonl_patterns(path):
    out = []

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return out

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        out.extend(extract_patterns_from_obj(data, path))

        after = data.get("after")
        if looks_like_note(after):
            out.append({
                "safe": guess_safe(data, path),
                "source_file": str(path),
                "key": "correction_after",
                "notes": [normalize_note(after)],
                "reason": data.get("event_type", "correction_after"),
                "time": data.get("time", ""),
            })

    return out


def pattern_weight(pattern):
    key = str(pattern.get("key", ""))
    reason = str(pattern.get("reason", "")).lower()

    if key == "correction_after":
        return 1.0

    w = 4.0

    if "save" in reason or "validation" in reason or "valid" in reason:
        w += 5.0

    if "generate" in reason:
        w -= 1.0

    return max(1.0, w)


def train_role_model(patterns, projects):
    audio_cache = {}
    pair_role_cache = {}

    global_step_role_counts = defaultdict(Counter)
    global_role_counts = Counter()
    global_role_transition_counts = defaultdict(Counter)
    global_step_role_length_counts = defaultdict(lambda: defaultdict(Counter))

    by_break = {}
    examples = []
    skipped = 0

    for pattern in patterns:
        safe = pattern["safe"]
        project = find_project_for_safe(projects, safe, path=pattern["source_file"])

        if project is None:
            skipped += 1
            continue

        project_safe = project.get("safe", safe)
        project_key = str(project_safe)

        if project_key not in pair_role_cache:
            pair_role_cache[project_key] = classify_project_pairs(project, audio_cache)

        pair_roles = pair_role_cache[project_key]

        if project_key not in by_break:
            by_break[project_key] = {
                "step_role_counts": defaultdict(Counter),
                "role_counts": Counter(),
                "role_transition_counts": defaultdict(Counter),
                "step_role_length_counts": defaultdict(lambda: defaultdict(Counter)),
                "pair_role_votes": defaultdict(Counter),
                "examples_count": 0,
                "notes_count": 0,
                "source_files": Counter(),
            }

        bucket = by_break[project_key]
        w = pattern_weight(pattern)

        role_notes = []

        for note in pattern["notes"]:
            pair = int(note["pair"])
            role_info = pair_roles.get(pair)

            if role_info is None:
                role = "other"
            else:
                role = role_info["role"]

            x_step = int(note["x_step"])
            length = int(note["length"])

            role_notes.append({
                "x_step": x_step,
                "pair": pair,
                "role": role,
                "length": length,
            })

            bucket["step_role_counts"][x_step][role] += w
            bucket["role_counts"][role] += w
            bucket["step_role_length_counts"][x_step][role][length] += w
            bucket["pair_role_votes"][role][pair] += w

            global_step_role_counts[x_step][role] += w
            global_role_counts[role] += w
            global_step_role_length_counts[x_step][role][length] += w

        role_notes = sorted(role_notes, key=lambda n: (n["x_step"], n["role"]))

        for a, b in zip(role_notes, role_notes[1:]):
            ra = a["role"]
            rb = b["role"]

            bucket["role_transition_counts"][ra][rb] += w
            global_role_transition_counts[ra][rb] += w

        bucket["examples_count"] += 1
        bucket["notes_count"] += len(role_notes)
        bucket["source_files"][pattern["source_file"]] += 1

        examples.append({
            "safe": project_key,
            "source_file": pattern["source_file"],
            "key": pattern["key"],
            "reason": pattern.get("reason", ""),
            "weight": w,
            "notes": role_notes,
        })

    model = {
        "version": "beat_role_model_v01",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "grid_steps": 32,
        "roles": ROLE_NAMES,
        "patterns_found": len(patterns),
        "patterns_used": len(examples),
        "patterns_skipped_no_pair_project": skipped,
        "global": {
            "step_role_counts": nested_counter_to_dict(global_step_role_counts),
            "role_counts": counter_to_dict(global_role_counts),
            "role_transition_counts": nested_counter_to_dict(global_role_transition_counts),
            "step_role_length_counts": triple_counter_to_dict(global_step_role_length_counts),
        },
        "breaks": {},
        "pair_role_cache": {},
        "examples": examples[-200:],
    }

    for safe, pair_map in pair_role_cache.items():
        model["pair_role_cache"][safe] = {
            str(pair): {
                "role": info["role"],
                "scores": {
                    k: round(float(v), 6)
                    for k, v in info["scores"].items()
                },
            }
            for pair, info in sorted(pair_map.items(), key=lambda kv: int(kv[0]))
        }

    for safe, bucket in by_break.items():
        model["breaks"][safe] = {
            "examples_count": bucket["examples_count"],
            "notes_count": bucket["notes_count"],
            "step_role_counts": nested_counter_to_dict(bucket["step_role_counts"]),
            "role_counts": counter_to_dict(bucket["role_counts"]),
            "role_transition_counts": nested_counter_to_dict(bucket["role_transition_counts"]),
            "step_role_length_counts": triple_counter_to_dict(bucket["step_role_length_counts"]),
            "pair_role_votes": nested_counter_to_dict(bucket["pair_role_votes"]),
            "source_files": counter_to_dict(bucket["source_files"]),
        }

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument("--out", default=str(OUT_MODEL))
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out = Path(args.out)

    if not dataset.exists():
        raise SystemExit(f"ERREUR: dataset introuvable : {dataset}")

    projects = load_pair_projects()

    if not projects:
        raise SystemExit("ERREUR: aucun pair_blocks_v02 trouvé dans dataset/pair_blocks_v02/")

    patterns = []

    for path in sorted(dataset.rglob("*.json")):
        name = path.name.lower()

        if "pair_blocks" in name:
            continue

        if "beat_style_model" in name or "beat_role_model" in name:
            continue

        patterns.extend(load_json_patterns(path))

    for path in sorted(dataset.rglob("*.jsonl")):
        patterns.extend(load_jsonl_patterns(path))

    model = train_role_model(patterns, projects)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")

    print("OK train beat roles")
    print("Patterns trouvés :", model["patterns_found"])
    print("Patterns utilisés :", model["patterns_used"])
    print("Patterns ignorés sans pair_blocks :", model["patterns_skipped_no_pair_project"])
    print("Breaks appris :", len(model["breaks"]))
    print("Model :", out)
    print("")

    print("Global role counts:")
    for role, count in model["global"]["role_counts"].items():
        print(f"  {role}: {count}")

    print("")
    for safe, bucket in sorted(model["breaks"].items(), key=lambda kv: kv[1]["examples_count"], reverse=True):
        print(f"{safe} | examples={bucket['examples_count']} | notes={bucket['notes_count']}")

        step_roles = bucket["step_role_counts"]
        important = []

        for step, roles in step_roles.items():
            best_role = max(roles.items(), key=lambda kv: float(kv[1]))[0]
            important.append((int(step), best_role, roles[best_role]))

        important = sorted(important, key=lambda x: x[0])
        print("  positions:", ", ".join(f"{s}:{r}" for s, r, _ in important[:32]))


if __name__ == "__main__":
    main()
