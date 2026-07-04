#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_quarantine_bad_slices_v01.py

Met en quarantaine des slices qui contiennent un son parasite.

Usage :
    python pipeline/02_quarantine_bad_slices_v01.py --source "camo" --pairs "1,9,10" --export-only
    python pipeline/02_quarantine_bad_slices_v01.py --source "camo" --pairs "1,9,10" --remove

Effet --remove :
- backup du JSON actuel
- les slices demandées sont déplacées dans ignored_blocks
- les autres slices sont réindexées pair 0,1,2...
- l'app ne les affichera plus
"""

from pathlib import Path
import argparse
import json
import shutil
from datetime import datetime

import numpy as np
import soundfile as sf


SR = 44100
PAIR_BLOCKS_DIR = Path("dataset/pair_blocks_v02")
EXPORT_DIR = Path("dataset/debug_slices")


def sanitize_name(name):
    out = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "break"


def parse_pairs(text):
    pairs = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        pairs.append(int(part))
    return sorted(set(pairs))


def find_pair_json(source):
    files = sorted(PAIR_BLOCKS_DIR.glob("*_pair_blocks_v02.json"))
    matches = [p for p in files if source.lower() in p.name.lower()]

    if not matches:
        raise SystemExit(f"ERREUR: aucun pair_blocks_v02 JSON trouvé pour source={source!r}")

    return matches[0]


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


def normalize(y, peak=0.95):
    y = np.asarray(y, dtype=np.float32)

    if len(y) == 0:
        return y

    m = float(np.max(np.abs(y)))

    if m <= 1e-9:
        return y

    return (y / m * peak).astype(np.float32)


def backup(path):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = path.with_suffix(path.suffix + f".bak_quarantine_{stamp}")
    shutil.copy2(path, dst)
    return dst


def export_slices(data, pair_json, pairs):
    source_audio = data.get("source_audio")
    if not source_audio:
        raise SystemExit("ERREUR: source_audio manquant dans le JSON.")

    source_path = Path(source_audio)

    if not source_path.exists():
        source_path = Path(".") / source_audio

    if not source_path.exists():
        raise SystemExit(f"ERREUR: source audio introuvable: {source_audio}")

    audio = load_audio(source_path)

    safe = data.get("safe") or sanitize_name(pair_json.stem)
    out_dir = EXPORT_DIR / safe
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks = data.get("blocks", [])
    by_pair = {int(b["pair"]): b for b in blocks}

    print("")
    print("EXPORT SLICES")
    print("Source audio :", source_path)
    print("Dossier :", out_dir)
    print("")

    for pair in pairs:
        if pair not in by_pair:
            print(f"slice {pair}: absente du JSON")
            continue

        block = by_pair[pair]
        a = int(block.get("source_start_sample", 0))
        b = int(block.get("source_end_sample", a + 1))

        a = max(0, min(len(audio) - 1, a))
        b = max(a + 1, min(len(audio), b))

        y = audio[a:b]
        out = out_dir / f"slice_{pair:02d}_{sanitize_name(block.get('marker_kind', block.get('name', 'slice')))}.wav"

        sf.write(out, normalize(y), SR)

        print(
            f"slice {pair:02d} -> {out} | "
            f"{block.get('source_start_ms')} ms -> {block.get('source_end_ms')} ms | "
            f"dur {block.get('duration_ms')} ms | "
            f"{block.get('marker_kind', block.get('detect_kind', ''))}"
        )


def remove_pairs(data, pairs):
    old_blocks = data.get("blocks", [])

    removed = []
    kept = []

    for block in old_blocks:
        pair = int(block.get("pair", -1))

        if pair in pairs:
            block = dict(block)
            block["quarantine_reason"] = "manual_bad_slice_poinnn"
            block["original_pair"] = pair
            removed.append(block)
        else:
            kept.append(dict(block))

    new_blocks = []

    for new_pair, block in enumerate(kept):
        old_pair = int(block.get("pair", new_pair))
        block["original_pair"] = old_pair
        block["pair"] = new_pair

        name = block.get("name", "")
        if not name or name.startswith("slice "):
            block["name"] = f"slice {new_pair}"

        new_blocks.append(block)

    old_ignored = data.get("ignored_blocks", [])
    if not isinstance(old_ignored, list):
        old_ignored = []

    data["ignored_blocks"] = old_ignored + removed
    data["blocks"] = new_blocks
    data["slice_count"] = len(new_blocks)
    data["quarantine_version"] = "quarantine_bad_slices_v01"
    data["quarantined_pairs_latest"] = pairs
    data["quarantined_count_latest"] = len(removed)

    return data, removed, new_blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()

    pairs = parse_pairs(args.pairs)
    pair_json = find_pair_json(args.source)

    data = json.loads(pair_json.read_text(encoding="utf-8"))

    print("JSON :", pair_json)
    print("Pairs ciblées :", pairs)

    export_slices(data, pair_json, pairs)

    if args.export_only and not args.remove:
        print("")
        print("Mode export-only : aucun JSON modifié.")
        return

    if not args.remove:
        print("")
        print("Aucune suppression demandée. Ajoute --remove pour modifier le JSON.")
        return

    bak = backup(pair_json)

    new_data, removed, kept = remove_pairs(data, pairs)

    pair_json.write_text(json.dumps(new_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print("")
    print("QUARANTAINE OK")
    print("Backup :", bak)
    print("JSON réécrit :", pair_json)
    print("Slices retirées :", [b.get("original_pair", b.get("pair")) for b in removed])
    print("Slices restantes :", len(kept))
    print("")
    print("Mapping actuel :")
    for b in kept:
        print(
            f"new pair {int(b['pair']):02d} <- old pair {int(b.get('original_pair', b['pair'])):02d} | "
            f"{b.get('source_start_ms')} ms -> {b.get('source_end_ms')} ms | "
            f"{b.get('marker_kind', b.get('detect_kind', ''))}"
        )


if __name__ == "__main__":
    main()
