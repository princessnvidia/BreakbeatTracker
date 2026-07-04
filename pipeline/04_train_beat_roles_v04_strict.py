#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_train_beat_roles_v04_strict.py

Trainer strict pour BreakbeatAI.

But :
- apprendre case -> rôle musical
- ne PAS apprendre les brouillons/latest/générations IA/corrections isolées

Il réutilise le moteur audio/role de :
    pipeline/04_train_beat_roles_v03.py

Sortie :
    dataset/learning/beat_role_model_v01.json
"""

from pathlib import Path
import importlib.util
import json
from datetime import datetime


BASE_TRAINER = Path("pipeline/04_train_beat_roles_v03.py")
DATASET_DIR = Path("dataset")
OUT_MODEL = Path("dataset/learning/beat_role_model_v01.json")


def load_base_trainer():
    if not BASE_TRAINER.exists():
        raise SystemExit(
            "ERREUR: pipeline/04_train_beat_roles_v03.py introuvable. "
            "Il faut d'abord avoir créé le trainer v03."
        )

    spec = importlib.util.spec_from_file_location("beat_roles_v03", BASE_TRAINER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def reject_path(path):
    s = str(path).lower()
    name = path.name.lower()

    bad_tokens = [
        "pair_blocks",
        "beat_role_model",
        "beat_style_model",
        "latest_pattern",
        "latest",
        "preview",
        "audition",
        "debug",
        "debug_slices",
        "dirty_backup",
        "backup",
        "quarantine",
        "ignored",
        "slice_index",
    ]

    for tok in bad_tokens:
        if tok in s or tok in name:
            return True

    return False


def pattern_is_strict_valid(pattern):
    """
    On garde uniquement :
    - patterns complets
    - fichiers de sauvegarde d'édition / validation
    - pas les corrections isolées
    - pas les générations IA brutes
    """
    path = str(pattern.get("source_file", "")).lower()
    name = Path(path).name.lower()
    key = str(pattern.get("key", "")).lower()
    reason = str(pattern.get("reason", "")).lower()
    notes = pattern.get("notes") or []

    if len(notes) < 4:
        return False

    if key == "correction_after":
        return False

    if "jsonl" in name:
        return False

    bad_reason_tokens = [
        "generate",
        "generated",
        "ai_generated",
        "v59_generate",
        "v60_generate",
        "v61_generate",
        "v62_generate",
        "latest",
        "preview",
        "debug",
    ]

    for tok in bad_reason_tokens:
        if tok in reason:
            return False

    good_tokens = [
        "tracker_app_edit",
        "validation",
        "valid",
        "save",
        "saved",
        "manual",
        "human",
    ]

    if any(tok in name for tok in good_tokens):
        return True

    if any(tok in reason for tok in good_tokens):
        return True

    return False


def collect_strict_patterns(base, dataset):
    all_patterns = []
    kept = []
    rejected = []

    for path in sorted(dataset.rglob("*.json")):
        if reject_path(path):
            continue

        patterns = base.load_json_patterns(path)
        all_patterns.extend(patterns)

        for pat in patterns:
            if pattern_is_strict_valid(pat):
                kept.append(pat)
            else:
                rejected.append(pat)

    return all_patterns, kept, rejected


def main():
    base = load_base_trainer()

    if not DATASET_DIR.exists():
        raise SystemExit(f"ERREUR: dataset introuvable : {DATASET_DIR}")

    projects = base.load_pair_projects()

    if not projects:
        raise SystemExit("ERREUR: aucun pair_blocks_v02 trouvé dans dataset/pair_blocks_v02/")

    all_patterns, kept, rejected = collect_strict_patterns(base, DATASET_DIR)

    model = base.train_role_model(kept, projects)

    model["training_policy"] = "strict_validated_only_v04"
    model["strict_created_at"] = datetime.now().isoformat(timespec="seconds")
    model["patterns_found_before_strict_filter"] = len(all_patterns)
    model["patterns_kept_after_strict_filter"] = len(kept)
    model["patterns_rejected_after_strict_filter"] = len(rejected)

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_MODEL.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")

    print("OK train beat roles STRICT v04")
    print("Patterns trouvés avant filtre :", len(all_patterns))
    print("Patterns gardés stricts       :", len(kept))
    print("Patterns rejetés              :", len(rejected))
    print("Breaks appris                 :", len(model.get("breaks", {})))
    print("Model                         :", OUT_MODEL)
    print("")

    if len(kept) == 0:
        print("ATTENTION: aucun beat validé gardé.")
        print("C'est normal si tu n'as pas encore fait Save validation sur un bon beat.")
        print("La v63 utilisera quand même une grammaire safe kick/snare/hat.")

    print("")
    print("Patterns gardés :")
    for pat in kept[-20:]:
        print(
            " -",
            pat.get("source_file"),
            "| notes=",
            len(pat.get("notes") or []),
            "| reason=",
            pat.get("reason", "")
        )


if __name__ == "__main__":
    main()
