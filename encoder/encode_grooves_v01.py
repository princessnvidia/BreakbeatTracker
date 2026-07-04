#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Groove Encoder v01 - encode_grooves_v01.py

But :
Créer un "espace latent" simple pour les grooves ASCII.

Entrées :
    dataset/ascii_transcriptions/ascii_transcriptions_v01.json
    exports/groovecritic_v01/best_grooves_v01.json
    exports/groovegpt_v01/groovegpt_generated_v01.json
    exports/groovebrain_ascii_v02/generated_ascii_grooves_v02.json

Sorties :
    exports/grooveencoder_v01/groove_embeddings_v01.json
    exports/grooveencoder_v01/nearest_neighbors_v01.txt
    exports/grooveencoder_v01/style_clusters_v01.txt

Idée :
Chaque groove devient un vecteur :
- densité globale
- quantité de kicks/snares/ghosts/hats/percs
- positions des instruments
- syncopation
- match avec la base K.S.|.KS.|K.S.|.KS.
- motifs 4 et 8 pas

Ce n'est pas encore un autoencoder neuronal.
C'est un encoder analytique propre, parfait pour commencer.
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

DEFAULT_INPUTS = [
    Path("dataset/ascii_transcriptions/ascii_transcriptions_v01.json"),
    Path("exports/groovecritic_v01/best_grooves_v01.json"),
    Path("exports/groovegpt_v01/groovegpt_generated_v01.json"),
    Path("exports/groovebrain_ascii_v02/generated_ascii_grooves_v02.json"),
]

OUT_DIR = Path("exports/grooveencoder_v01")
EMBEDDINGS_JSON = OUT_DIR / "groove_embeddings_v01.json"
NEIGHBORS_TXT = OUT_DIR / "nearest_neighbors_v01.txt"
CLUSTERS_TXT = OUT_DIR / "style_clusters_v01.txt"


def clean_grid(grid):
    grid = "".join(ch for ch in str(grid) if ch in ALLOWED)
    return grid[:STEPS].ljust(STEPS, ".")


def split32(grid):
    grid = clean_grid(grid)
    return "|".join(grid[i:i+8] for i in range(0, STEPS, 8))


def load_from_file(path):
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []

    # format transcriptions
    if "transcriptions" in data:
        for idx, item in enumerate(data["transcriptions"], start=1):
            full = clean_grid(item.get("full", ""))
            rows.append({
                "id": f"{path.stem}_{idx:04d}",
                "name": item.get("name", f"transcription_{idx:04d}"),
                "source_file": str(path),
                "audio_source": item.get("source", ""),
                "full": full,
            })

    # format generated/best grooves
    if "grooves" in data:
        for idx, item in enumerate(data["grooves"], start=1):
            full = clean_grid(item.get("full", ""))
            rows.append({
                "id": f"{path.stem}_{idx:04d}",
                "name": f"{path.stem}_{idx:04d}",
                "source_file": str(path),
                "audio_source": item.get("source", ""),
                "full": full,
                "score": item.get("score", None),
            })

    return rows


def one_hot_positions(grid, symbol):
    return [1.0 if ch == symbol else 0.0 for ch in grid]


def rhythm_features(grid):
    grid = clean_grid(grid)
    hits = [i for i, ch in enumerate(grid) if ch != "."]
    total_hits = len(hits)

    counts = Counter(grid)

    feat = []

    # 1. Densités globales
    feat.append(total_hits / STEPS)
    for sym in ["K", "S", "g", "H", "p"]:
        feat.append(counts[sym] / STEPS)

    # 2. Syncopation
    if total_hits:
        off_hits = sum(1 for i in hits if i % 4 not in [0])
        weak_hits = sum(1 for i in hits if i % 2 == 1)
        feat.append(off_hits / total_hits)
        feat.append(weak_hits / total_hits)
    else:
        feat += [0.0, 0.0]

    # 3. Base match K.S.|.KS.
    ks_targets = [i for i, ch in enumerate(BASE_GRID) if ch in "KS"]
    exact = sum(1 for i in ks_targets if grid[i] == BASE_GRID[i]) / max(1, len(ks_targets))
    role = 0.0
    for i in ks_targets:
        if grid[i] == BASE_GRID[i]:
            role += 1.0
        elif BASE_GRID[i] == "S" and grid[i] == "g":
            role += 0.45
        elif BASE_GRID[i] == "K" and grid[i] in "Hp":
            role += 0.20
    role /= max(1, len(ks_targets))
    feat += [exact, role]

    # 4. Positions one-hot par instrument
    for sym in ["K", "S", "g", "H", "p"]:
        feat.extend(one_hot_positions(grid, sym))

    # 5. Motifs par quart de grille
    for start in range(0, STEPS, 8):
        chunk = grid[start:start+8]
        chunk_hits = sum(1 for ch in chunk if ch != ".")
        feat.append(chunk_hits / 8)
        for sym in ["K", "S", "g", "H", "p"]:
            feat.append(chunk.count(sym) / 8)

    # 6. Distances moyennes kick/snare
    kicks = [i for i, ch in enumerate(grid) if ch == "K"]
    snares = [i for i, ch in enumerate(grid) if ch == "S"]
    ghosts = [i for i, ch in enumerate(grid) if ch == "g"]

    def mean_nearest(a, b):
        if not a or not b:
            return 1.0
        vals = []
        for x in a:
            vals.append(min(abs(x-y) for y in b) / STEPS)
        return float(sum(vals) / len(vals))

    feat.append(mean_nearest(kicks, snares))
    feat.append(mean_nearest(ghosts, snares))
    feat.append(mean_nearest(kicks, ghosts))

    return np.array(feat, dtype=np.float32)


def normalize_vectors(vectors):
    X = np.stack(vectors, axis=0)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    Z = (X - mean) / std
    return Z, mean, std


def cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def nearest_neighbors(items, Z, top_k=5):
    rows = []
    for i, item in enumerate(items):
        sims = []
        for j, other in enumerate(items):
            if i == j:
                continue
            sims.append((cosine(Z[i], Z[j]), j))
        sims.sort(reverse=True, key=lambda x: x[0])
        rows.append({
            "index": i,
            "id": item["id"],
            "name": item["name"],
            "full": item["full"],
            "neighbors": [
                {
                    "similarity": round(sim, 5),
                    "id": items[j]["id"],
                    "name": items[j]["name"],
                    "full": items[j]["full"],
                }
                for sim, j in sims[:top_k]
            ],
        })
    return rows


def simple_cluster(items, Z):
    """
    Cluster analytique simple, pas de sklearn.
    On classe par densité + syncopation + base match.
    """
    clusters = defaultdict(list)

    for item in items:
        grid = item["full"]
        total = sum(1 for ch in grid if ch != ".")
        density = total / STEPS

        hits = [i for i, ch in enumerate(grid) if ch != "."]
        sync = 0.0
        if hits:
            sync = sum(1 for i in hits if i % 4 not in [0]) / len(hits)

        base_match = sum(
            1 for a, b in zip(grid, BASE_GRID)
            if a == b and a in "KS"
        ) / max(1, sum(1 for ch in BASE_GRID if ch in "KS"))

        if density < 0.30:
            dens = "sparse"
        elif density < 0.58:
            dens = "medium"
        else:
            dens = "dense"

        if sync < 0.35:
            syn = "straight"
        elif sync < 0.58:
            syn = "groovy"
        else:
            syn = "syncopated"

        if base_match > 0.65:
            base = "base"
        elif base_match > 0.35:
            base = "broken_base"
        else:
            base = "free"

        key = f"{dens}_{syn}_{base}"
        clusters[key].append(item)

    return clusters


def render_neighbors(neighbors, limit=40):
    lines = []
    for row in neighbors[:limit]:
        lines.append(f"{row['name']}")
        lines.append("  " + split32(row["full"]))
        for n in row["neighbors"]:
            lines.append(f"  -> {n['similarity']:.4f} {n['name']} {split32(n['full'])}")
        lines.append("")
    return "\n".join(lines)


def render_clusters(clusters):
    lines = []
    for key, items in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True):
        lines.append(f"CLUSTER {key} | count={len(items)}")
        for item in items[:12]:
            lines.append(f"  {item['name']}  {split32(item['full'])}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--neighbors-limit", type=int, default=60)
    args = ap.parse_args()

    items = []
    for path in DEFAULT_INPUTS:
        items.extend(load_from_file(path))

    # dédoublonnage léger par full + name
    seen = set()
    cleaned = []
    for item in items:
        key = (item["full"], item["name"])
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    items = cleaned

    if not items:
        print("Aucun groove trouvé.")
        print("Lance d'abord les transcriptions et/ou GrooveGPT.")
        sys.exit(1)

    vectors = [rhythm_features(item["full"]) for item in items]
    Z, mean, std = normalize_vectors(vectors)

    neighbors = nearest_neighbors(items, Z, top_k=args.top_k)
    clusters = simple_cluster(items, Z)

    payload = {
        "version": "grooveencoder_v01",
        "count": len(items),
        "vector_size": int(len(vectors[0])),
        "items": [
            {
                **item,
                "embedding": [round(float(x), 6) for x in Z[i].tolist()],
            }
            for i, item in enumerate(items)
        ],
        "neighbors": neighbors,
        "clusters": {
            k: [item["id"] for item in v]
            for k, v in clusters.items()
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    NEIGHBORS_TXT.write_text(render_neighbors(neighbors, limit=args.neighbors_limit), encoding="utf-8")
    CLUSTERS_TXT.write_text(render_clusters(clusters), encoding="utf-8")

    print("Grooves encodés :", len(items))
    print("Taille vecteur :", len(vectors[0]))
    print("")
    print("Exports :")
    print(" ", EMBEDDINGS_JSON)
    print(" ", NEIGHBORS_TXT)
    print(" ", CLUSTERS_TXT)
    print("")
    print("Clusters :")
    for k, v in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {k:28} {len(v)}")


if __name__ == "__main__":
    main()
