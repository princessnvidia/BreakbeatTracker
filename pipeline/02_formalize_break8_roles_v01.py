#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_formalize_break8_roles_v01.py

Formalise les rôles d'une découpe de break selon une grammaire 8 temps :

    kick hat snare hat hat kick snare hat

Au lieu d'essayer de reconnaître le son par fréquence, on utilise la position
du slice dans le break :

    pair_index % 8

Pattern :
    0 -> kick
    1 -> hat
    2 -> snare
    3 -> hat
    4 -> hat
    5 -> kick
    6 -> snare
    7 -> hat

Usage :
    cd ~/Applications/BreakbeatAI
    python pipeline/02_formalize_break8_roles_v01.py --source "amen"

Options :
    --no-write-role-guess
        N'écrit pas role_guess/manual_role dans le JSON pair_blocks.
        Écrit seulement formal_role/formal_position_8.

    --pattern "kick,hat,snare,hat,hat,kick,snare,hat"
        Remplace le pattern par défaut.
"""

from pathlib import Path
import argparse
import json
import shutil
import sys
from datetime import datetime


PAIR_BLOCKS_DIR = Path("dataset/pair_blocks_v02")
ROLE_MAPS_DIR = Path("dataset/role_maps")

DEFAULT_PATTERN = ["kick", "hat", "snare", "hat", "hat", "kick", "snare", "hat"]
VALID_ROLES = {"kick", "snare", "hat", "unknown", "placer"}


def clean_role(role):
    role = str(role).strip().lower()

    aliases = {
        "hihat": "hat",
        "hi-hat": "hat",
        "hi_hat": "hat",
        "hh": "hat",
        "bd": "kick",
        "bassdrum": "kick",
        "bass_drum": "kick",
        "sd": "snare",
        "rim": "snare",
        "clap": "snare",
        "?": "unknown",
        "none": "unknown",
        "a_classer": "unknown",
        "à classer": "unknown",
    }

    role = aliases.get(role, role)

    if role not in VALID_ROLES:
        raise ValueError(f"Rôle invalide : {role}")

    return role


def parse_pattern(pattern_text):
    if not pattern_text:
        return list(DEFAULT_PATTERN)

    parts = [clean_role(x) for x in pattern_text.replace(" ", "").split(",") if x.strip()]

    if not parts:
        raise ValueError("Pattern vide.")

    return parts


def find_pair_json(source_query):
    files = sorted(PAIR_BLOCKS_DIR.glob("*_pair_blocks_v02.json"))
    matches = [p for p in files if source_query.lower() in p.name.lower()]

    if not matches:
        print(f"Aucun pair_blocks_v02 JSON trouvé pour : {source_query}")
        print(f"Regarde dans : {PAIR_BLOCKS_DIR}")
        sys.exit(1)

    return matches[0]


def safe_name(pair_json):
    return pair_json.stem.replace("_pair_blocks_v02", "")


def backup_file(path):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".formal_break8_backup_{stamp}")
    shutil.copy2(path, backup)
    return backup


def formalize(pair_json, pattern, write_role_guess=True):
    data = json.loads(pair_json.read_text(encoding="utf-8"))
    blocks = data.get("blocks", [])

    if not blocks:
        raise RuntimeError(f"Aucun block dans {pair_json}")

    blocks_sorted = sorted(blocks, key=lambda b: int(b.get("pair", 0)))

    safe = safe_name(pair_json)
    backup = backup_file(pair_json)

    role_map = {
        "version": "formal_break8_roles_v01",
        "source_pair_json": str(pair_json),
        "backup_pair_json": str(backup),
        "safe": safe,
        "pattern": pattern,
        "pattern_text": " ".join(pattern),
        "rule": "role = pattern[pair_order_index % len(pattern)]",
        "write_role_guess": bool(write_role_guess),
        "assignments": [],
    }

    for order_index, block in enumerate(blocks_sorted):
        pair = int(block.get("pair", order_index))
        pos = order_index % len(pattern)
        role = pattern[pos]

        old_role_guess = block.get("role_guess")
        old_manual_role = block.get("manual_role")
        old_confidence = block.get("role_confidence")

        block["formal_position"] = int(order_index)
        block["formal_position_in_cycle"] = int(pos)
        block["formal_cycle_length"] = int(len(pattern))
        block["formal_role"] = role
        block["formal_role_source"] = "formal_break8_pattern"
        block["formal_pattern"] = pattern

        if write_role_guess:
            block["previous_role_guess_before_formal"] = old_role_guess
            block["previous_manual_role_before_formal"] = old_manual_role
            block["previous_role_confidence_before_formal"] = old_confidence

            # On pose cette grammaire comme vérité de structure.
            block["manual_role"] = role
            block["role_guess"] = role
            block["role_confidence"] = 1.0
            block["role_source"] = "formal_break8_pattern"

        role_map["assignments"].append({
            "order_index": int(order_index),
            "pair": int(pair),
            "cycle_position": int(pos),
            "role": role,
            "audio_path": block.get("audio_path"),
            "duration_ms": block.get("duration_ms"),
            "old_role_guess": old_role_guess,
            "old_manual_role": old_manual_role,
            "old_role_confidence": old_confidence,
        })

    # On remet les blocks dans l'ordre original du JSON, mais comme les objets ont été modifiés
    # par référence, les annotations sont bien conservées.
    data["formal_break8"] = {
        "version": "formal_break8_roles_v01",
        "pattern": pattern,
        "pattern_text": " ".join(pattern),
        "rule": "role = pattern[pair_order_index % len(pattern)]",
    }

    pair_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    ROLE_MAPS_DIR.mkdir(parents=True, exist_ok=True)
    role_map_path = ROLE_MAPS_DIR / f"{safe}_formal_break8_roles_v01.json"
    role_map_path.write_text(json.dumps(role_map, indent=2, ensure_ascii=False), encoding="utf-8")

    report_path = ROLE_MAPS_DIR / f"{safe}_formal_break8_roles_v01.txt"
    lines = []
    lines.append(f"Formal break8 roles for {safe}")
    lines.append(f"Pattern: {' '.join(pattern)}")
    lines.append(f"Rule: role = pattern[pair_order_index % {len(pattern)}]")
    lines.append(f"Updated JSON: {pair_json}")
    lines.append(f"Backup: {backup}")
    lines.append("")
    for a in role_map["assignments"]:
        lines.append(
            f"pair {a['pair']:02d} | order {a['order_index']:02d} | "
            f"pos {a['cycle_position']} -> {a['role']}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return backup, role_map_path, report_path, role_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="amen")
    parser.add_argument(
        "--pattern",
        default=",".join(DEFAULT_PATTERN),
        help="Ex: kick,hat,snare,hat,hat,kick,snare,hat",
    )
    parser.add_argument(
        "--no-write-role-guess",
        action="store_true",
        help="N'écrit pas role_guess/manual_role, ajoute seulement formal_role.",
    )
    args = parser.parse_args()

    pattern = parse_pattern(args.pattern)
    pair_json = find_pair_json(args.source)

    backup, role_map_path, report_path, role_map = formalize(
        pair_json=pair_json,
        pattern=pattern,
        write_role_guess=not args.no_write_role_guess,
    )

    print("OK formalisation break8")
    print("Pattern :", " ".join(pattern))
    print("JSON mis à jour :", pair_json)
    print("Backup :", backup)
    print("Role map JSON :", role_map_path)
    print("Rapport :", report_path)
    print("")
    for a in role_map["assignments"]:
        print(
            f"pair {a['pair']:02d} | order {a['order_index']:02d} | "
            f"pos {a['cycle_position']} -> {a['role']}"
        )


if __name__ == "__main__":
    main()
