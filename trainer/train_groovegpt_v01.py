#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GrooveGPT v01 - train_groovegpt_v01.py

But :
Entraîner un petit modèle GRU sur tes grooves ASCII.

Entrée prioritaire :
    exports/groovebrain_ascii_v02/generated_ascii_grooves_v02.json

Entrée fallback :
    dataset/ascii_transcriptions/ascii_transcriptions_v01.json

Sortie :
    models/groovegpt_v01.pt

Tokens :
    . = silence
    K = kick
    S = snare
    g = ghost snare
    H = hat
    p = percussion
    | = séparation toutes les 8 steps
    ^ = début
    $ = fin

Usage :
    python trainer/train_groovegpt_v01.py --epochs 120
"""

from pathlib import Path
import argparse
import json
import random
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "groovegpt_v01.pt"

GENERATED_JSON = Path("exports/groovebrain_ascii_v02/generated_ascii_grooves_v02.json")
TRANSCRIPTIONS_JSON = Path("dataset/ascii_transcriptions/ascii_transcriptions_v01.json")

TOKENS = ["^", "$", ".", "K", "S", "g", "H", "p", "|"]
TOKEN_TO_ID = {t: i for i, t in enumerate(TOKENS)}
ID_TO_TOKEN = {i: t for t, i in TOKEN_TO_ID.items()}

STEPS = 32


def split_grid(grid):
    grid = grid[:STEPS].ljust(STEPS, ".")
    return "|".join(grid[i:i+8] for i in range(0, STEPS, 8))


def clean_grid(grid):
    allowed = set(".KSgHp")
    grid = "".join(ch for ch in grid if ch in allowed)
    grid = grid[:STEPS].ljust(STEPS, ".")
    return grid


def load_sequences():
    sequences = []

    if GENERATED_JSON.exists():
        data = json.loads(GENERATED_JSON.read_text(encoding="utf-8"))
        for g in data.get("grooves", []):
            full = clean_grid(g.get("full", ""))
            sequences.append(split_grid(full))

    if TRANSCRIPTIONS_JSON.exists():
        data = json.loads(TRANSCRIPTIONS_JSON.read_text(encoding="utf-8"))
        for t in data.get("transcriptions", []):
            full = clean_grid(t.get("full", ""))
            sequences.append(split_grid(full))

    # Base fondamentale répétée pour bien ancrer la grammaire.
    base = "K.S..KS.K.S..KS.K.S..KS.K.S..KS."
    base = clean_grid(base)
    for _ in range(300):
        sequences.append(split_grid(base))

    # Variations manuelles simples.
    manual = [
        "K.S..KS.K.S..KS.K.S..KS.K.S..KS.",
        "K.g..KS.K.g..KS.K.g..KS.K.g..KS.",
        "K.S.gKS.K.S..KS.K.S.gKS.K.S..KS.",
        "K.H.S.H.K.HS.H.K.H.S.H.K.HS.H.",
        "K...g...S..g....K...g...S..g....",
        "K.....K.S.......K...K...S.......",
        "K..gK...S.g.....K.g.K...S...g...",
    ]
    for row in manual:
        for _ in range(80):
            sequences.append(split_grid(clean_grid(row)))

    # Dédupliquer un peu, puis remélanger avec pondération gardée.
    random.shuffle(sequences)

    if not sequences:
        print("Aucune séquence trouvée.")
        sys.exit(1)

    return sequences


class GrooveDataset(Dataset):
    def __init__(self, sequences):
        self.samples = []
        for seq in sequences:
            text = "^" + seq + "$"
            ids = [TOKEN_TO_ID[ch] for ch in text if ch in TOKEN_TO_ID]
            if len(ids) >= 4:
                x = torch.tensor(ids[:-1], dtype=torch.long)
                y = torch.tensor(ids[1:], dtype=torch.long)
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)

    xpad = torch.full((len(xs), max_len), TOKEN_TO_ID["$"], dtype=torch.long)
    ypad = torch.full((len(xs), max_len), TOKEN_TO_ID["$"], dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        xpad[i, :len(x)] = x
        ypad[i, :len(y)] = y

    return xpad, ypad


class GrooveGPT(nn.Module):
    def __init__(self, vocab_size, emb=64, hidden=128, layers=2, dropout=0.15):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.gru = nn.GRU(
            emb,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        z = self.emb(x)
        out, h = self.gru(z, h)
        logits = self.head(out)
        return logits, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    args = ap.parse_args()

    sequences = load_sequences()
    ds = GrooveDataset(sequences)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Séquences:", len(sequences))
    print("Samples:", len(ds))

    model = GrooveGPT(
        vocab_size=len(TOKENS),
        emb=args.emb,
        hidden=args.hidden,
        layers=args.layers,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0

        for x, y in dl:
            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, len(TOKENS)), y.reshape(-1))

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.item())
            count += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:04d} | loss {total/max(1,count):.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    torch.save({
        "version": "groovegpt_v01",
        "model_state": model.state_dict(),
        "tokens": TOKENS,
        "token_to_id": TOKEN_TO_ID,
        "id_to_token": ID_TO_TOKEN,
        "config": {
            "emb": args.emb,
            "hidden": args.hidden,
            "layers": args.layers,
        },
        "steps": STEPS,
    }, MODEL_PATH)

    print("Modèle sauvegardé :", MODEL_PATH)


if __name__ == "__main__":
    main()
