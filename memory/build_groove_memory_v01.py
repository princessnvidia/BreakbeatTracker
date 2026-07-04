#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Groove Memory v01 - build_groove_memory_v01.py

But :
Créer une mémoire de grooves pour BreakBrain.

Entrées utilisées si présentes :
    dataset/ascii_transcriptions/ascii_transcriptions_v01.json
    exports/groovecritic_v01/best_grooves_v01.json
    exports/self_improve_v01/best_for_training_v01.json
    exports/groovegpt_v01/groovegpt_generated_v01.json
    exports/groovebrain_ascii_v02/generated_ascii_grooves_v02.json

Sorties :
    memory/groove_memory_v01.json
    memory/groove_neighbors_v01.txt
    memory/groove_graph_v01.json
    memory/groove_memory_report_v01.txt

Usage :
    python memory/build_groove_memory_v01.py
    python memory/build_groove_memory_v01.py --top-k 8
"""

from pathlib import Path
import argparse
import json
import math
import sys
from collections import Counter, defaultdict

import numpy as np

STEPS = 32
ALLOWED = set(".KSgHp")
SYMBOLS = [".", "K", "S", "g", "H", "p"]

BASE_GRID = "K.S..KS.K.S..KS.K.S..KS.K.S..KS."

INPUTS = [
    Path("dataset/ascii_transcriptions/ascii_transcriptions_v01.json"),
    Path("exports/groovecritic_v01/best_grooves_v01.json"),
    Path("exports/self_improve_v01/best_for_training_v01.json"),
    Path("exports/groovegpt_v01/groovegpt_generated_v01.json"),
    Path("exports/groovebrain_ascii_v02/generated_ascii_grooves_v02.json"),
]

OUT_DIR = Path("memory")
MEMORY_JSON = OUT_DIR / "groove_memory_v01.json"
NEIGHBORS_TXT = OUT_DIR / "groove_neighbors_v01.txt"
GRAPH_JSON = OUT_DIR / "groove_graph_v01.json"
REPORT_TXT = OUT_DIR / "groove_memory_report_v01.txt"


def clean_grid(grid):
    grid = "".join(ch for ch in str(grid) if ch in ALLOWED)
    return grid[:STEPS].ljust(STEPS, ".")


def split32(grid):
    grid = clean_grid(grid)
    return "|".join(grid[i:i+8] for i in range(0, STEPS, 8))


def grid_to_layers(grid):
    grid = clean_grid(grid)
    layers = {
        "kick": ["."] * STEPS,
        "snare": ["."] * STEPS,
        "ghost": ["."] * STEPS,
        "hat": ["."] * STEPS,
        "perc": ["."] * STEPS,
    }
    for i, ch in enumerate(grid):
        if ch == "K":
            layers["kick"][i] = "K"
        elif ch == "S":
            layers["snare"][i] = "S"
        elif ch == "g":
            layers["ghost"][i] = "g"
        elif ch == "H":
            layers["hat"][i] = "H"
        elif ch == "p":
            layers["perc"][i] = "p"
    return {k: "".join(v) for k, v in layers.items()}


def load_items_from_file(path):
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    items = []

    if "transcriptions" in data:
        for i, row in enumerate(data["transcriptions"], start=1):
            items.append({
                "id": f"{path.stem}_{i:04d}",
                "name": row.get("name", f"transcription_{i:04d}"),
                "source_type": "transcription",
                "source_file": str(path),
                "audio_source": row.get("source", ""),
                "full": clean_grid(row.get("full", "")),
                "score": None,
            })

    if "grooves" in data:
        source_type = "generated"
        if "critic" in path.stem or "best" in path.stem:
            source_type = "selected"
        if "self" in str(path):
            source_type = "self_improve_best"

        for i, row in enumerate(data["grooves"], start=1):
            items.append({
                "id": f"{path.stem}_{i:04d}",
                "name": row.get("name", f"{path.stem}_{i:04d}"),
                "source_type": source_type,
                "source_file": str(path),
                "audio_source": row.get("source", ""),
                "full": clean_grid(row.get("full", "")),
                "score": row.get("score", None),
            })

    return items


def feature_vector(grid):
    grid = clean_grid(grid)
    counts = Counter(grid)
    hits = [i for i, ch in enumerate(grid) if ch != "."]

    vec = []

    # Densités
    vec.append(len(hits) / STEPS)
    for sym in ["K", "S", "g", "H", "p"]:
        vec.append(counts[sym] / STEPS)

    # Syncopation
    if hits:
        vec.append(sum(1 for i in hits if i % 4 != 0) / len(hits))
        vec.append(sum(1 for i in hits if i % 2 == 1) / len(hits))
    else:
        vec += [0.0, 0.0]

    # Match base
    targets = [i for i, ch in enumerate(BASE_GRID) if ch in "KS"]
    exact = sum(1 for i in targets if grid[i] == BASE_GRID[i]) / max(1, len(targets))
    role = 0.0
    for i in targets:
        if grid[i] == BASE_GRID[i]:
            role += 1.0
        elif BASE_GRID[i] == "S" and grid[i] == "g":
            role += 0.45
        elif BASE_GRID[i] == "K" and grid[i] in "Hp":
            role += 0.20
    role /= max(1, len(targets))
    vec += [exact, role]

    # Positions one-hot par instrument
    for sym in ["K", "S", "g", "H", "p"]:
        vec.extend([1.0 if ch == sym else 0.0 for ch in grid])

    # Quartiers
    for start in range(0, STEPS, 8):
        chunk = grid[start:start+8]
        vec.append(sum(1 for ch in chunk if ch != ".") / 8)
        for sym in ["K", "S", "g", "H", "p"]:
            vec.append(chunk.count(sym) / 8)

    # Motifs 4 pas compactés : densités par bloc
    for start in range(0, STEPS, 4):
        chunk = grid[start:start+4]
        vec.append(sum(1 for ch in chunk if ch != ".") / 4)

    # Relations
    def positions(sym):
        return [i for i, ch in enumerate(grid) if ch == sym]

    def nearest_mean(a, b):
        if not a or not b:
            return 1.0
        vals = []
        for x in a:
            vals.append(min(abs(x-y) for y in b) / STEPS)
        return sum(vals) / len(vals)

    kicks = positions("K")
    snares = positions("S")
    ghosts = positions("g")
    hats = positions("H")

    vec += [
        nearest_mean(kicks, snares),
        nearest_mean(ghosts, snares),
        nearest_mean(kicks, ghosts),
        nearest_mean(hats, snares),
    ]

    return np.array(vec, dtype=np.float32)


def normalize_matrix(vectors):
    X = np.stack(vectors, axis=0)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    return (X - mean) / std


def cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def summarize(grid):
    grid = clean_grid(grid)
    hits = [i for i, ch in enumerate(grid) if ch != "."]
    counts = Counter(grid)

    density = len(hits) / STEPS
    sync = 0.0
    if hits:
        sync = sum(1 for i in hits if i % 4 != 0) / len(hits)

    base_targets = [i for i, ch in enumerate(BASE_GRID) if ch in "KS"]
    base_match = sum(1 for i in base_targets if grid[i] == BASE_GRID[i]) / max(1, len(base_targets))

    if density < 0.30:
        density_label = "sparse"
    elif density < 0.58:
        density_label = "medium"
    else:
        density_label = "dense"

    if sync < 0.40:
        sync_label = "straight"
    elif sync < 0.62:
        sync_label = "groovy"
    else:
        sync_label = "syncopated"

    if base_match > 0.65:
        base_label = "base"
    elif base_match > 0.35:
        base_label = "broken_base"
    else:
        base_label = "free"

    return {
        "density": round(density, 4),
        "syncopation": round(sync, 4),
        "base_match": round(base_match, 4),
        "kick_count": counts["K"],
        "snare_count": counts["S"],
        "ghost_count": counts["g"],
        "hat_count": counts["H"],
        "perc_count": counts["p"],
        "style_tags": [density_label, sync_label, base_label],
    }


def build_neighbors(items, Z, top_k):
    neighbors = []
    for i, item in enumerate(items):
        sims = []
        for j, other in enumerate(items):
            if i == j:
                continue
            sims.append((cosine(Z[i], Z[j]), j))
        sims.sort(key=lambda x: x[0], reverse=True)
        neighbors.append({
            "id": item["id"],
            "name": item["name"],
            "full": item["full"],
            "neighbors": [
                {
                    "id": items[j]["id"],
                    "name": items[j]["name"],
                    "similarity": round(float(sim), 5),
                    "full": items[j]["full"],
                }
                for sim, j in sims[:top_k]
            ],
        })
    return neighbors


def build_graph(items, neighbors, threshold):
    nodes = []
    edges = []

    for item in items:
        nodes.append({
            "id": item["id"],
            "name": item["name"],
            "source_type": item["source_type"],
            "summary": item["summary"],
            "full": item["full"],
        })

    seen = set()
    for row in neighbors:
        src = row["id"]
        for n in row["neighbors"]:
            dst = n["id"]
            sim = n["similarity"]
            if sim < threshold:
                continue
            key = tuple(sorted([src, dst]))
            if key in seen:
                continue
            seen.add(key)
            edges.append({
                "source": src,
                "target": dst,
                "similarity": sim,
            })

    return {"nodes": nodes, "edges": edges}


def render_neighbors_txt(neighbors, limit=80):
    lines = []
    for row in neighbors[:limit]:
        lines.append(f"{row['name']} | {split32(row['full'])}")
        for n in row["neighbors"]:
            lines.append(f"  -> {n['similarity']:.4f} | {n['name']} | {split32(n['full'])}")
        lines.append("")
    return "\n".join(lines)


def render_report(items, graph):
    lines = []
    lines.append("GROOVE MEMORY v01")
    lines.append("=================")
    lines.append("")
    lines.append(f"Grooves mémorisés : {len(items)}")
    lines.append(f"Noeuds graphe     : {len(graph['nodes'])}")
    lines.append(f"Liens graphe      : {len(graph['edges'])}")
    lines.append("")

    source_counts = Counter(item["source_type"] for item in items)
    lines.append("Sources :")
    for k, v in source_counts.most_common():
        lines.append(f"  {k:20} {v}")
    lines.append("")

    tag_counts = Counter()
    for item in items:
        for tag in item["summary"]["style_tags"]:
            tag_counts[tag] += 1

    lines.append("Tags :")
    for k, v in tag_counts.most_common():
        lines.append(f"  {k:20} {v}")
    lines.append("")

    lines.append("Exemples mémoire :")
    for item in items[:20]:
        s = item["summary"]
        lines.append(
            f"{item['name']} | {'/'.join(s['style_tags'])} | "
            f"K={s['kick_count']} S={s['snare_count']} g={s['ghost_count']} H={s['hat_count']} | "
            f"{split32(item['full'])}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--graph-threshold", type=float, default=0.72)
    args = parser.parse_args()

    raw = []
    for path in INPUTS:
        loaded = load_items_from_file(path)
        if loaded:
            print(f"{path}: {len(loaded)} grooves")
        raw.extend(loaded)

    # dédoublonnage par grille complète + source type
    seen = set()
    items = []
    for item in raw:
        key = (item["full"], item["source_type"])
        if key in seen:
            continue
        seen.add(key)
        item["layers"] = grid_to_layers(item["full"])
        item["summary"] = summarize(item["full"])
        items.append(item)

    if not items:
        print("Aucun groove trouvé.")
        print("Lance d'abord transcribe/groovegpt/critic/self_improve.")
        sys.exit(1)

    vectors = [feature_vector(item["full"]) for item in items]
    Z = normalize_matrix(vectors)

    for i, item in enumerate(items):
        item["embedding"] = [round(float(x), 6) for x in Z[i].tolist()]

    neighbors = build_neighbors(items, Z, args.top_k)
    graph = build_graph(items, neighbors, threshold=args.graph_threshold)

    memory = {
        "version": "groove_memory_v01",
        "count": len(items),
        "steps": STEPS,
        "items": items,
        "neighbors": neighbors,
        "graph": {
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_JSON.write_text(json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8")
    GRAPH_JSON.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    NEIGHBORS_TXT.write_text(render_neighbors_txt(neighbors), encoding="utf-8")
    REPORT_TXT.write_text(render_report(items, graph), encoding="utf-8")

    print("")
    print("Mémoire créée :")
    print(" ", MEMORY_JSON)
    print(" ", NEIGHBORS_TXT)
    print(" ", GRAPH_JSON)
    print(" ", REPORT_TXT)


if __name__ == "__main__":
    main()
