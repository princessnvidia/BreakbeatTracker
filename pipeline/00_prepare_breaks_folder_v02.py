#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import subprocess
import sys
import json
import re
import time

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".aif", ".aiff", ".ogg", ".m4a"}


def version_num(path):
    m = re.search(r"_v(\d+)", path.name)
    return int(m.group(1)) if m else -1


def find_slicer():
    preferred = Path("pipeline/01_m8_style_autoslice_index_v10.py")
    if preferred.exists():
        return preferred

    candidates = sorted(Path("pipeline").glob("01_*autoslice*.py"), key=version_num)
    if candidates:
        return candidates[-1]

    raise SystemExit("ERREUR: aucun autoslicer trouvé dans pipeline/.")


def safe_name(path):
    out = []
    for ch in path.stem:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "break"


def find_audio_files(break_dir):
    files = []
    for p in break_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        files.append(p)
    return sorted(files, key=lambda x: str(x).lower())


def existing_pair_blocks(audio_path):
    pair_dir = Path("dataset/pair_blocks_v02")
    if not pair_dir.exists():
        return []

    guessed = safe_name(audio_path).lower()
    matches = []

    for p in pair_dir.glob("*_pair_blocks_v02.json"):
        if guessed in p.name.lower():
            matches.append(p)
            continue

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        source_audio = str(data.get("source_audio", ""))
        if source_audio and Path(source_audio).name == audio_path.name:
            matches.append(p)

    return matches


def run_slicer(slicer, audio_path, args):
    attempts = [
        str(audio_path),
        audio_path.name,
        audio_path.stem,
    ]

    seen = set()
    attempts = [x for x in attempts if not (x in seen or seen.add(x))]

    for source_arg in attempts:
        cmd = [
            sys.executable,
            str(slicer),
            "--source",
            source_arg,
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
        print("AUDIO :", audio_path)
        print("TRY   :", source_arg)
        print("CMD   :", " ".join(cmd))
        print("────────────────────────────────────────")

        if args.dry_run:
            return True

        result = subprocess.run(cmd, text=True, capture_output=True, check=False)

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr)

        if result.returncode == 0:
            return True

        print("[WARN] tentative échouée, fallback suivant...")

    return False


def write_library_index():
    pair_dir = Path("dataset/pair_blocks_v02")
    entries = []

    for p in sorted(pair_dir.glob("*_pair_blocks_v02.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            safe = data.get("safe") or p.name.replace("_pair_blocks_v02.json", "")
            blocks = data.get("blocks", []) or []
            source_audio = data.get("source_audio", "")
        except Exception:
            safe = p.name.replace("_pair_blocks_v02.json", "")
            blocks = []
            source_audio = ""

        entries.append({
            "safe": safe,
            "slices": len(blocks),
            "path": str(p),
            "source_audio": source_audio,
        })

    out = Path("dataset/learning/break_library_v01.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "version": "break_library_v01",
                "count": len(entries),
                "entries": entries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return out, entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--break-dir", default="$HOME/Applications/BreakbeatAI/breaks")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--target-bpm", type=float, default=155.0)
    parser.add_argument("--max-slices", type=int, default=32)
    parser.add_argument("--sensitivity", type=float, default=0.16)
    parser.add_argument("--hat-sensitivity", type=float, default=0.04)
    parser.add_argument("--hat-rescue", type=int, default=8)
    parser.add_argument("--snap-ms", type=float, default=55.0)

    args = parser.parse_args()

    break_dir = Path(args.break_dir.replace("$HOME", str(Path.home()))).expanduser()

    if not break_dir.exists():
        raise SystemExit(f"ERREUR: dossier breaks introuvable : {break_dir}")

    slicer = find_slicer()
    audio_files = find_audio_files(break_dir)

    print("")
    print("BreakbeatAI — import dossier breaks v02")
    print("Dossier breaks :", break_dir)
    print("Slicer         :", slicer)
    print("Audios trouvés :", len(audio_files))
    print("Dry-run        :", args.dry_run)
    print("Force          :", args.force)
    print("")

    for i, audio in enumerate(audio_files, start=1):
        existing = existing_pair_blocks(audio)
        status = "EXISTE" if existing else "À FAIRE"
        print(f"{i:03d}. {status:7s} {audio}")

        for p in existing:
            print(f"       -> {p}")

    if args.dry_run:
        print("")
        print("DRY RUN terminé : rien n'a été modifié.")
        return

    ok = 0
    skipped = 0
    fail = 0
    start = time.time()

    for audio in audio_files:
        existing = existing_pair_blocks(audio)

        if existing and not args.force:
            print("")
            print("[SKIP] déjà préparé :", audio)
            skipped += 1
            continue

        success = run_slicer(slicer, audio, args)

        if success:
            ok += 1
        else:
            fail += 1

    index_path, entries = write_library_index()

    print("")
    print("════════════════════════════════════════")
    print("IMPORT TERMINÉ")
    print("OK        :", ok)
    print("SKIPPED   :", skipped)
    print("FAIL      :", fail)
    print("Durée     :", round(time.time() - start, 2), "sec")
    print("Index     :", index_path)
    print("PairBlocks:", len(entries))
    print("════════════════════════════════════════")
    print("")

    for e in entries:
        print(f" - {e['safe']} | slices={e['slices']} | {e['path']}")


if __name__ == "__main__":
    main()
