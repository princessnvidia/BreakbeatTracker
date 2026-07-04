#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import argparse, json, random, sys, subprocess
import torch
import torch.nn as nn

MODEL_PATH = Path('models/groovegpt_v01.pt')
CRITIC_PATH = Path('critic/score_grooves_v01.py')
OUT_DIR = Path('exports/self_improve_v01')
POOL_JSON = OUT_DIR / 'generated_pool_v01.json'
POOL_TXT = OUT_DIR / 'generated_pool_v01.txt'
BEST_JSON = OUT_DIR / 'best_for_training_v01.json'
BEST_TXT = OUT_DIR / 'best_for_training_v01.txt'
CRITIC_OUT_JSON = Path('exports/groovecritic_v01/scored_grooves_v01.json')
STEPS = 32
ALLOWED = set('.KSgHp')

class GrooveGPT(nn.Module):
    def __init__(self, vocab_size, emb=64, hidden=128, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.gru = nn.GRU(emb, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)
    def forward(self, x, h=None):
        z = self.emb(x)
        out, h = self.gru(z, h)
        return self.head(out), h

def clean_grid(grid):
    grid = ''.join(ch for ch in str(grid) if ch in ALLOWED)
    return grid[:STEPS].ljust(STEPS, '.')

def split32(grid):
    grid = clean_grid(grid)
    return '|'.join(grid[i:i+8] for i in range(0, STEPS, 8))

def grid_to_layers(grid):
    grid = clean_grid(grid)
    layers = {k: ['.'] * STEPS for k in ['kick', 'snare', 'ghost', 'hat', 'perc']}
    for i, ch in enumerate(grid):
        if ch == 'K': layers['kick'][i] = 'K'
        elif ch == 'S': layers['snare'][i] = 'S'
        elif ch == 'g': layers['ghost'][i] = 'g'
        elif ch == 'H': layers['hat'][i] = 'H'
        elif ch == 'p': layers['perc'][i] = 'p'
    return {k: ''.join(v) for k, v in layers.items()}

def sample_next(logits, temperature=0.95, top_k=6):
    logits = logits / max(0.05, temperature)
    if top_k and top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.numel()))
        filtered = torch.full_like(logits, -1e9)
        filtered[indices] = values
        logits = filtered
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())

def cleanup_grid(grid, base_strength=0.35):
    grid = list(clean_grid(grid))
    base = 'K.S..KS.K.S..KS.K.S..KS.K.S..KS.'
    for i, ch in enumerate(base):
        if ch in 'KS' and random.random() < base_strength:
            grid[i] = ch
    kicks = [i for i, ch in enumerate(grid) if ch == 'K']
    if len(kicks) > 9:
        random.shuffle(kicks)
        for i in kicks[9:]: grid[i] = '.'
    snares = [i for i, ch in enumerate(grid) if ch == 'S']
    if len(snares) > 9:
        random.shuffle(snares)
        for i in snares[9:]: grid[i] = 'g'
    if sum(1 for ch in grid if ch != '.') < 9:
        for i in range(0, STEPS, 2):
            if grid[i] == '.' and random.random() < 0.35:
                grid[i] = 'H'
    return ''.join(grid)

def load_model():
    if not MODEL_PATH.exists():
        print('Modèle manquant:', MODEL_PATH)
        sys.exit(1)
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    tokens = ckpt['tokens']
    token_to_id = ckpt['token_to_id']
    raw_id_to_token = ckpt['id_to_token']
    id_to_token = {}
    for k, v in raw_id_to_token.items():
        try: id_to_token[int(k)] = v
        except Exception: id_to_token[k] = v
    cfg = ckpt['config']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GrooveGPT(len(tokens), cfg['emb'], cfg['hidden'], cfg['layers']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, token_to_id, id_to_token, device

def generate_one(model, token_to_id, id_to_token, device, temperature, top_k, base_strength):
    start_id = token_to_id['^']
    x = torch.tensor([[start_id]], dtype=torch.long, device=device)
    h = None
    generated = ''
    while len(generated.replace('|', '')) < STEPS:
        logits, h = model(x, h)
        next_id = sample_next(logits[0, -1], temperature, top_k)
        ch = id_to_token.get(next_id, '.')
        if ch == '$': break
        if ch == '^': continue
        generated += ch
        x = torch.tensor([[next_id]], dtype=torch.long, device=device)
    return cleanup_grid(generated.replace('|', ''), base_strength)

def render_txt(grooves):
    lines = []
    for item in grooves:
        idx = item['index']; grid = item['full']; layers = grid_to_layers(grid); score = item.get('score')
        title = f'VARIATION {idx:04d}'
        if score: title += f" | score={score.get('final',0):.5f}"
        lines += [title, '12345678|12345678|12345678|12345678', 'KICK : '+split32(layers['kick']), 'SNARE: '+split32(layers['snare']), 'GHOST: '+split32(layers['ghost']), 'HAT  : '+split32(layers['hat']), 'PERC : '+split32(layers['perc']), 'FULL : '+split32(grid), '']
    return '\n'.join(lines)

def run_critic(top):
    if not CRITIC_PATH.exists():
        print('Critic introuvable:', CRITIC_PATH)
        sys.exit(1)
    subprocess.run([sys.executable, str(CRITIC_PATH), '--input', str(POOL_JSON), '--top', str(top)], check=True)
    if not CRITIC_OUT_JSON.exists():
        print('Sortie critic manquante:', CRITIC_OUT_JSON)
        sys.exit(1)
    return json.loads(CRITIC_OUT_JSON.read_text(encoding='utf-8'))['scored']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=1000)
    ap.add_argument('--top', type=int, default=120)
    ap.add_argument('--temperature', type=float, default=0.95)
    ap.add_argument('--top-k', type=int, default=6)
    ap.add_argument('--base-strength', type=float, default=0.35)
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed); torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, token_to_id, id_to_token, device = load_model()
    print('Device:', device)
    print('Génération pool:', args.pool)
    seen, grooves, attempts = set(), [], 0
    while len(grooves) < args.pool and attempts < args.pool * 6:
        attempts += 1
        g = generate_one(model, token_to_id, id_to_token, device, args.temperature, args.top_k, args.base_strength)
        if g in seen: continue
        seen.add(g)
        grooves.append({'index': len(grooves)+1, 'full': g, 'layers': grid_to_layers(g)})
        if len(grooves) % 100 == 0: print(' ', len(grooves), 'grooves')
    POOL_JSON.write_text(json.dumps({'version':'self_improve_pool_v01','count':len(grooves),'grooves':grooves}, indent=2, ensure_ascii=False), encoding='utf-8')
    POOL_TXT.write_text(render_txt(grooves), encoding='utf-8')
    scored = run_critic(args.top)
    scored.sort(key=lambda x: x['score']['final'], reverse=True)
    best = [{'index': i, 'full': item['full'], 'layers': grid_to_layers(item['full']), 'score': item['score']} for i, item in enumerate(scored[:args.top], start=1)]
    BEST_JSON.write_text(json.dumps({'version':'self_improve_best_for_training_v01','source_pool':str(POOL_JSON),'top':args.top,'count':len(best),'grooves':best}, indent=2, ensure_ascii=False), encoding='utf-8')
    BEST_TXT.write_text(render_txt(best), encoding='utf-8')
    print('Pool:', POOL_JSON)
    print('Best:', BEST_JSON)
    print(render_txt(best[:5]))

if __name__ == '__main__': main()
