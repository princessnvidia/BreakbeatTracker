#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GrooveGPT v01 - generate_groovegpt_v01.py

But :
Générer des grooves ASCII avec le modèle entraîné.

Entrée :
    models/groovegpt_v01.pt

Sorties :
    exports/groovegpt_v01/groovegpt_generated_v01.txt
    exports/groovegpt_v01/groovegpt_generated_v01.json

Usage :
    python generator/generate_groovegpt_v01.py --count 24 --temperature 0.85
"""

from pathlib import Path
import argparse
import json
import random
import sys

import torch
import torch.nn as nn

MODEL_PATH = Path("models/groovegpt_v01.pt")
OUT_DIR = Path("exports/groovegpt_v01")
OUT_TXT = OUT_DIR / "groovegpt_generated_v01.txt"
OUT_JSON = OUT_DIR / "groovegpt_generated_v01.json"

STEPS = 32


class GrooveGPT(nn.Module):
    def __init__(self, vocab_size, emb=64, hidden=128, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.gru = nn.GRU(
            emb,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.0,
        )
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        z = self.emb(x)
        out, h = self.gru(z, h)
        logits = self.head(out)
        return logits, h


def split32(row):
    row = row[:STEPS].ljust(STEPS, ".")
    return "|".join(row[i:i+8] for i in range(0, STEPS, 8))


def grid_to_layers(grid):
    layers = {
        "kick": ["."] * STEPS,
        "snare": ["."] * STEPS,
        "ghost": ["."] * STEPS,
        "hat": ["."] * STEPS,
        "perc": ["."] * STEPS,
    }

    for i, s in enumerate(grid[:STEPS]):
        if s == "K":
            layers["kick"][i] = "K"
        elif s == "S":
            layers["snare"][i] = "S"
        elif s == "g":
            layers["ghost"][i] = "g"
        elif s == "H":
            layers["hat"][i] = "H"
        elif s == "p":
            layers["perc"][i] = "p"

    return {k: "".join(v) for k, v in layers.items()}


def cleanup_grid(grid, base_strength=0.45):
    """
    Nettoyage musical léger :
    - longueur 32
    - garde des kicks/snares structurants
    - limite le mitraillage
    """
    allowed = set(".KSgHp")
    grid = "".join(ch for ch in grid if ch in allowed)
    grid = list(grid[:STEPS].ljust(STEPS, "."))

    base = "K.S..KS.K.S..KS.K.S..KS.K.S..KS."

    for i, ch in enumerate(base[:STEPS]):
        if ch in "KS" and random.random() < base_strength:
            grid[i] = ch

    kicks = [i for i, s in enumerate(grid) if s == "K"]
    if len(kicks) > 8:
        random.shuffle(kicks)
        for i in kicks[8:]:
            grid[i] = "."

    snares = [i for i, s in enumerate(grid) if s == "S"]
    if len(snares) > 8:
        random.shuffle(snares)
        for i in snares[8:]:
            grid[i] = "g"

    # Si trop vide, remet quelques hats.
    filled = sum(1 for x in grid if x != ".")
    if filled < 10:
        for i in range(0, STEPS, 2):
            if grid[i] == "." and random.random() < 0.45:
                grid[i] = "H"

    return "".join(grid)


def sample_next(logits, temperature=0.85, top_k=5):
    logits = logits / max(0.05, temperature)

    if top_k and top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.numel()))
        filtered = torch.full_like(logits, -1e9)
        filtered[indices] = values
        logits = filtered

    probs = torch.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return int(idx.item())


def generate_one(model, token_to_id, id_to_token, device, temperature, top_k, prompt=None):
    model.eval()

    text = "^"
    if prompt:
        text += prompt

    ids = [token_to_id.get(ch, token_to_id["."]) for ch in text]
    x = torch.tensor([[ids[0]]], dtype=torch.long, device=device)
    h = None

    # Prime hidden state avec le prompt.
    for idx in ids[1:]:
        logits, h = model(x, h)
        x = torch.tensor([[idx]], dtype=torch.long, device=device)

    generated = prompt or ""

    while len(generated.replace("|", "")) < STEPS:
        logits, h = model(x, h)
        next_id = sample_next(logits[0, -1], temperature=temperature, top_k=top_k)
        ch = id_to_token[str(next_id)] if str(next_id) in id_to_token else id_to_token[next_id]

        if ch == "$":
            break
        if ch == "^":
            continue

        generated += ch
        x = torch.tensor([[next_id]], dtype=torch.long, device=device)

    grid = generated.replace("|", "")
    return cleanup_grid(grid)


def render_text(grooves):
    lines = []

    for i, grid in enumerate(grooves, start=1):
        layers = grid_to_layers(grid)

        lines.append(f"VARIATION {i:03d}")
        lines.append("12345678|12345678|12345678|12345678")
        lines.append("KICK : " + split32(layers["kick"]))
        lines.append("SNARE: " + split32(layers["snare"]))
        lines.append("GHOST: " + split32(layers["ghost"]))
        lines.append("HAT  : " + split32(layers["hat"]))
        lines.append("PERC : " + split32(layers["perc"]))
        lines.append("FULL : " + split32(grid))
        lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--prompt", default=None,
                    help="Début optionnel, ex: 'K.S.|.KS.'")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if not MODEL_PATH.exists():
        print("Modèle manquant :", MODEL_PATH)
        print("Lance d'abord : python trainer/train_groovegpt_v01.py --epochs 120")
        sys.exit(1)

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    tokens = ckpt["tokens"]
    token_to_id = ckpt["token_to_id"]
    id_to_token = ckpt["id_to_token"]
    cfg = ckpt["config"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GrooveGPT(
        vocab_size=len(tokens),
        emb=cfg["emb"],
        hidden=cfg["hidden"],
        layers=cfg["layers"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    grooves = []
    for _ in range(args.count):
        g = generate_one(
            model,
            token_to_id,
            id_to_token,
            device,
            temperature=args.temperature,
            top_k=args.top_k,
            prompt=args.prompt,
        )
        grooves.append(g)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_TXT.write_text(render_text(grooves), encoding="utf-8")

    payload = {
        "version": "groovegpt_generated_v01",
        "grooves": [
            {
                "index": i,
                "full": g,
                "layers": grid_to_layers(g),
            }
            for i, g in enumerate(grooves, start=1)
        ],
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Exports :")
    print(" ", OUT_TXT)
    print(" ", OUT_JSON)
    print("")
    print(render_text(grooves[:5]))


if __name__ == "__main__":
    main()
