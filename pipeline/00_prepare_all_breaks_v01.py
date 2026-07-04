#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
00_prepare_all_breaks_v01.py

Prépare tous les breaks audio du projet pour BreakbeatAI.

Ce script :
- cherche tous les .wav/.flac/.mp3/.aiff/.ogg
- ignore les previews, debug, learning, pair_blocks, exports
- lance l'autoslicer M8-style v10 pour chaque break
- crée les JSON index-only dans dataset/pair_blocks_v02/

Usage :
    python pipeline/00_prepare_all_breaks_v01.py --dry-run
    python pipeline/00_prepare_all_breaks_v01.py

Options utiles :
    --target-bpm 155
    --mode hybrid
    --grid 16
    --max-slices 32
"""

from pathlib import Path
import argparse
import subprocess
import sys
import json
import time


AUDIO_EXTS = {
    ".wav",
    ".flac",
    ".mp3",
    ".aif",
    ".aiff",
    ".ogg",
    ".m4a",
}

SEARCH_DIRS = [
    Path("audio"),
    Path("samples"),
    Path("breaks"),
    Path("dataset"),
]

EXCLUDE_DIR_TOKENS = [
    "pair_blocks_v02",
    "learning",
    "debug",
    "debug_slices",
    "preview",
    "previews",
    "exports",
    "render",
    "renders",
    "tmp",
    "cache",
    "dirty_backup",
    "backup",
    "quarantine",
    "ignored",
]

EXCLUDE_FILE_TOKENS = [
    "_preview",
    "_audition",
    "_live",
    "loop32",
    "tracker_app",
    "latest_pattern",
    "slice_index",
    "debug",
]


def is_audio_file(path):
    return path.is_file() and path.suffix.lower() in AUDIO_EXTS


def should_exclude(path):
    low_parts = [p.lower() for p in path.parts]
    low_name = path.name.lower()

    for token in EXCLUDE_DIR_TOKENS:
        if token.lower() in low_parts:
            return True

    full = str(path).lower()

    for token in EXCLUDE_DIR_TOKENS:
        if f"/{token.lower()}/" in full:
            return True

    for token in EXCLUDE_FILE_TOKENS:
        if token.lower() in low_name:
            return True

    return False


def find_slicer():
    preferred = Path("pipeline/01_m8_style_autoslice_index_v10.py")

    if preferred.exists():
        return preferred

    candidates = sorted(
        Path("pipeline").glob("01_*autoslice*.py")
    )

    if candidates:
        return candidates[-1]

    raise SystemExit("ERREUR: aucun autoslicer trouvé dans pipeline/.")


def find_audio_files():
    found = []

    for root in SEARCH_DIRS:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not is_audio_file(path):
                continue

            if should_exclude(path):
                continue

            found.append(path)

    # Déduplication par chemin réel.
    unique = {}
    for path in found:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)

        unique[key] = path

    return sorted(unique.values(), key=lambda p: str(p).lower())


def run_slicer_for_file(slicer, audio_path, args):
    cmd = [
        sys.executable,
        str(slicer),
        "--source",
        str(audio_path),
        "--mode",
        args.mode,
        "--grid",
        str(args.grid),
        "--target-bpm",
        str(args.target_bpm),
        "--max-slices",
        str(args.max_slices),
        "--sensitivity",
        str(args.sensitivity),
        "--hat-sensitivity",
        str(args.hat_sensitivity),
        "--hat-rescue",
        str(args.hat_rescue),
        "--snap-ms",
        str(args.snap_ms),
    ]

    print("")
    print("────────────────────────────────────────")
    print("BREAK :", audio_path)
    print("CMD   :", " ".join(cmd))
    print("────────────────────────────────────────")

    if args.dry_run:
        return True

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode == 0:
        return True

    # Fallback : certains anciens slicers cherchent par nom au lieu du chemin complet.
    fallback_source = audio_path.stem

    print("")
    print("[WARN] Échec avec chemin complet, tentative avec le nom :", fallback_source)

    fallback_cmd = list(cmd)
    fallback_cmd[fallback_cmd.index("--source") + 1] = fallback_source

    result2 = subprocess.run(
        fallback_cmd,
        text=True,
        capture_output=True,
        check=False,
    )

    if result2.stdout:
        print(result2.stdout)

    if result2.stderr:
        print(result2.stderr)

    return result2.returncode == 0


def list_pair_blocks():
    out = []

    pair_dir = Path("dataset/pair_blocks_v02")

    if not pair_dir.exists():
        return out

    for path in sorted(pair_dir.glob("*_pair_blocks_v02.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            blocks = data.get("blocks", [])
            safe = data.get("safe", path.stem.replace("_pair_blocks_v02", ""))
            source_audio = data.get("source_audio", "")
            out.append({
                "safe": safe,
                "blocks": len(blocks),
                "path": str(path),
                "source_audio": source_audio,
            })
        except Exception:
            out.append({
                "safe": path.stem.replace("_pair_blocks_v02", ""),
                "blocks": "?",
                "path": str(path),
                "source_audio": "",
            })

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--target-bpm", type=float, default=155.0)
    parser.add_argument("--max-slices", type=int, default=32)
    parser.add_argument("--sensitivity", type=float, default=0.16)
    parser.add_argument("--hat-sensitivity", type=float, default=0.04)
    parser.add_argument("--hat-rescue", type=int, default=8)
    parser.add_argument("--snap-ms", type=float, default=55.0)

    args = parser.parse_args()

    slicer = find_slicer()
    audio_files = find_audio_files()

    print("")
    print("BreakbeatAI — prepare all breaks v01")
    print("Slicer :", slicer)
    print("Mode   :", args.mode)
    print("BPM    :", args.target_bpm)
    print("Audio trouvés :", len(audio_files))

    if args.dry_run:
        print("")
        print("DRY RUN : rien ne sera modifié.")

    for i, path in enumerate(audio_files, start=1):
        print(f"[{i}/{len(audio_files)}] {path}")

    if not audio_files:
        print("")
        print("Aucun audio trouvé dans :")
        for root in SEARCH_DIRS:
            print(" -", root)
        return

    ok = 0
    fail = 0

    start = time.time()

    for path in audio_files:
        success = run_slicer_for_file(slicer, path, args)

        if success:
            ok += 1
        else:
            fail += 1

    elapsed = time.time() - start

    print("")
    print("════════════════════════════════════════")
    print("RÉSUMÉ")
    print("OK     :", ok)
    print("FAIL   :", fail)
    print("Durée  :", round(elapsed, 2), "sec")
    print("════════════════════════════════════════")
    print("")

    pair_blocks = list_pair_blocks()

    print("Pair blocks disponibles :")
    for item in pair_blocks:
        print(
            f" - {item['safe']} | slices={item['blocks']} | {item['path']}"
        )

    print("")
    print("Pour lancer un break :")
    print('  python pipeline/03_tracker_editor_app_v65_cross_song_role_template.py --source "nom_du_break"')
    print("")
    print("Exemple :")
    if pair_blocks:
        print(f'  python pipeline/03_tracker_editor_app_v65_cross_song_role_template.py --source "{pair_blocks[0]["safe"]}"')


if __name__ == "__main__":
    main()
