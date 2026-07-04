#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jungle Grammar v04 Backbone

Correction :
- récupère une vraie logique d'ossature :
      KICK -> SNARE -> KICK -> SNARE
- le break est d'abord découpé
- chaque slice est classée relativement en kick/snare/hat/other
- la phrase générée respecte une colonne vertébrale K S K S
- les variations restent autour de cette ossature
- pas de .txt par variation
- WAV + metadata global seulement

Usage :
    python jungle_grammar_v04_backbone.py --source "London" --count 32
    python jungle_grammar_v04_backbone.py --source "Amen" --count 32
    python jungle_grammar_v04_backbone.py --source "Stepper" --count 32

Plus de variations :
    python jungle_grammar_v04_backbone.py --source "London" --bars 8 --mutation 0.22 --count 32
"""

from pathlib import Path
import argparse
import json
import random
import sys
import warnings

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:
    print("librosa manquant : pip install librosa soundfile numpy")
    sys.exit(1)


BREAKS_DIR = Path("breaks")
OUT_DIR = Path("exports/jungle_grammar_v04_backbone")
SR = 44100
AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".mp3"}


def find_source(name):
    files = sorted(
        p for p in BREAKS_DIR.rglob("*")
        if p.suffix.lower() in AUDIO_EXTS and not p.name.endswith(".asd")
    )
    matches = [p for p in files if name.lower() in p.name.lower()]
    if not matches:
        print(f"Aucun break trouvé pour : {name}")
        sys.exit(1)
    return matches[0]


def normalize(y, peak=0.95):
    m = np.max(np.abs(y)) if len(y) else 0
    return y if m <= 1e-9 else y / m * peak


def fade(y, ms=2):
    if len(y) < 16:
        return y
    n = min(int(SR * ms / 1000), len(y) // 4)
    if n <= 1:
        return y
    y = y.copy()
    ramp = np.linspace(0, 1, n)
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def load_audio(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    return normalize(y), sr


def clean_cuts(cuts, total_len, min_samples):
    cuts = sorted(set(int(c) for c in cuts if 0 <= int(c) < total_len))
    if not cuts or cuts[0] != 0:
        cuts = [0] + cuts
    if cuts[-1] != total_len:
        cuts.append(total_len)

    cleaned = [cuts[0]]
    for c in cuts[1:]:
        if c - cleaned[-1] >= min_samples or c == total_len:
            cleaned.append(c)

    if cleaned[-1] != total_len:
        cleaned.append(total_len)

    return cleaned


def safe_centroid(y):
    if len(y) < 64:
        return 0.0
    n_fft = min(2048, max(64, 2 ** int(np.floor(np.log2(max(64, len(y)))))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=n_fft)))


def features(y):
    if len(y) < 64:
        return {"rms": 0, "peak": 0, "centroid": 0, "low": 0, "mid": 0, "high": 0}

    rms = float(np.sqrt(np.mean(y * y)))
    peak = float(np.max(np.abs(y)))
    centroid = safe_centroid(y)

    yn = normalize(y, peak=1.0)
    spec = np.abs(np.fft.rfft(yn))
    freqs = np.fft.rfftfreq(len(yn), 1.0 / SR)
    total = np.sum(spec) + 1e-9

    low = float(np.sum(spec[(freqs >= 20) & (freqs < 250)]) / total)
    mid = float(np.sum(spec[(freqs >= 250) & (freqs < 3000)]) / total)
    high = float(np.sum(spec[(freqs >= 3000) & (freqs < 12000)]) / total)

    return {"rms": rms, "peak": peak, "centroid": centroid, "low": low, "mid": mid, "high": high}


def cuts_to_slices(y, cuts):
    slices = []
    for i in range(len(cuts) - 1):
        start = cuts[i]
        end = cuts[i + 1]
        if end <= start:
            continue

        audio = fade(y[start:end].copy(), ms=2)
        f = features(audio)

        slices.append({
            "index": i,
            "start": int(start),
            "end": int(end),
            "start_ms": float(start / SR * 1000),
            "end_ms": float(end / SR * 1000),
            "duration": float((end - start) / SR),
            "features": f,
            "label": "other",
            "audio": audio,
        })
    return slices


def best_grid_offset(y, target_cuts, search_ms=250):
    slice_len = len(y) / target_cuts
    env = librosa.onset.onset_strength(y=y, sr=SR)
    env_samples = librosa.frames_to_samples(np.arange(len(env)))

    best = None
    step = int(SR * 5 / 1000)
    max_offset = int(SR * search_ms / 1000)

    for offset in range(0, max_offset + 1, step):
        score = 0.0
        for i in range(target_cuts):
            c = int(offset + i * slice_len)
            win = int(SR * 0.035)
            idx = np.where((env_samples >= c - win) & (env_samples <= c + win))[0]
            if len(idx):
                score += float(np.max(env[idx]))
        if best is None or score > best[0]:
            best = (score, offset)

    return best[1] if best else 0


def hybrid_cuts(y, target_cuts, min_ms):
    offset = best_grid_offset(y, target_cuts)
    usable = max(1, len(y) - offset)
    slice_len = usable / target_cuts

    onsets = librosa.onset.onset_detect(
        y=y,
        sr=SR,
        units="samples",
        backtrack=True,
        delta=0.06,
        wait=1,
    )

    cuts = []
    snap_window = int(SR * 0.045)

    for i in range(target_cuts):
        grid = int(offset + i * slice_len)
        nearby = [o for o in onsets if abs(o - grid) <= snap_window]
        cut = min(nearby, key=lambda o: abs(o - grid)) if nearby else grid
        cuts.append(cut)

    return clean_cuts([0] + cuts, len(y), int(SR * min_ms / 1000))


def onset_strong_cuts(y, target_cuts, min_ms):
    env = librosa.onset.onset_strength(y=y, sr=SR)
    samples = librosa.frames_to_samples(np.arange(len(env)))

    peaks = librosa.util.peak_pick(
        env,
        pre_max=3,
        post_max=3,
        pre_avg=8,
        post_avg=8,
        delta=0.08,
        wait=1,
    )

    ranked = sorted(peaks, key=lambda i: env[i], reverse=True)
    chosen = sorted(samples[i] for i in ranked[:max(1, target_cuts - 1)])
    return clean_cuts([0] + chosen, len(y), int(SR * min_ms / 1000))


def get_cuts(y, method, target_cuts, min_ms):
    if method == "onset_strong":
        return onset_strong_cuts(y, target_cuts, min_ms)
    return hybrid_cuts(y, target_cuts, min_ms)


def score_kick(s):
    f = s["features"]
    return (
        f["low"] * 3.2
        + f["rms"] * 4.0
        + f["peak"] * 2.0
        + max(0, 3000 - f["centroid"]) / 3000
        - f["high"] * 0.7
    )


def score_snare(s):
    f = s["features"]
    centroid_ok = 1.0 if 700 <= f["centroid"] <= 6500 else 0.0
    return (
        f["mid"] * 2.0
        + f["rms"] * 4.0
        + f["peak"] * 1.6
        + f["high"] * 0.5
        + centroid_ok
    )


def score_hat(s):
    f = s["features"]
    return (
        f["high"] * 3.2
        + f["centroid"] / 8500
        - f["low"] * 0.9
        + f["rms"] * 0.8
    )


def classify_slices(slices):
    if not slices:
        return slices

    # Classement relatif : chaque break a sa propre couleur.
    kick_rank = sorted(slices, key=score_kick, reverse=True)
    snare_rank = sorted(slices, key=score_snare, reverse=True)
    hat_rank = sorted(slices, key=score_hat, reverse=True)

    for s in slices:
        s["label"] = "other"

    # Beaucoup de hats, quelques kicks/snares.
    for s in hat_rank[:max(2, len(slices)//3)]:
        s["label"] = "hat"

    for s in kick_rank[:max(2, len(slices)//5)]:
        s["label"] = "kick"

    for s in snare_rank[:max(2, len(slices)//4)]:
        if s["label"] != "kick":
            s["label"] = "snare"

    # Sécurité : au moins un kick/snare/hat.
    if not [s for s in slices if s["label"] == "kick"]:
        kick_rank[0]["label"] = "kick"
    if not [s for s in slices if s["label"] == "snare"]:
        for s in snare_rank:
            if s["label"] != "kick":
                s["label"] = "snare"
                break
    if not [s for s in slices if s["label"] == "hat"]:
        for s in hat_rank:
            if s["label"] not in ["kick", "snare"]:
                s["label"] = "hat"
                break

    return slices


def pool_indices(slices, label):
    return [s["index"] for s in slices if s["label"] == label]


def choose_from(pool, fallback):
    if pool:
        return random.choice(pool)
    return random.choice(fallback)


def make_backbone_bar(slices, base=None, mutation=0.12):
    """
    Crée une boucle de 16 événements avec une ossature K S K S.
    Positions structurelles :
      0  = kick
      4  = snare
      8  = kick
      12 = snare

    Le reste est rempli avec hats/ghosts/nearby slices.
    """
    all_idx = [s["index"] for s in slices]
    kicks = pool_indices(slices, "kick")
    snares = pool_indices(slices, "snare")
    hats = pool_indices(slices, "hat")
    others = pool_indices(slices, "other")

    seq = []

    for pos in range(16):
        if pos in [0, 8]:
            idx = choose_from(kicks, all_idx)
        elif pos in [4, 12]:
            idx = choose_from(snares, all_idx)
        else:
            # hats en priorité, sinon slice proche de la base si disponible
            idx = choose_from(hats or others, all_idx)

        seq.append(idx)

    # Garde un peu de l'ordre naturel du break sur certaines positions
    if base and random.random() < 0.65:
        for pos in [2, 6, 10, 14]:
            if pos < len(base) and random.random() > mutation:
                seq[pos] = base[pos % len(base)]

    # Variations contrôlées
    for pos in range(16):
        if random.random() > mutation:
            continue

        if pos in [0, 4, 8, 12]:
            # ne casse pas l'ossature
            continue

        action = random.choice(["repeat_prev", "repeat_next", "same_label", "keep"])

        if action == "repeat_prev" and pos > 0:
            seq[pos] = seq[pos - 1]
        elif action == "repeat_next" and pos < 15:
            seq[pos] = seq[pos + 1]
        elif action == "same_label":
            current_label = slices[seq[pos] % len(slices)]["label"]
            pool = pool_indices(slices, current_label)
            if pool:
                seq[pos] = random.choice(pool)

    # Petit fill final mais sans casser 12 = snare
    if random.random() < mutation:
        for pos in [13, 14, 15]:
            if random.random() < 0.55:
                pool = hats + snares + others
                seq[pos] = choose_from(pool, all_idx)

    return seq


def make_longform(slices, bars=8, mutation=0.12):
    natural_base = [s["index"] for s in slices]
    phrase = []
    plans = []

    for bar in range(1, bars + 1):
        local_mut = mutation
        if bar in [1, 5, 9, 13]:
            local_mut *= 0.45
        if bar % 4 == 0:
            local_mut *= 1.35

        seq = make_backbone_bar(slices, base=natural_base, mutation=local_mut)
        phrase.extend(seq)
        plans.append({
            "bar": bar,
            "mutation": round(local_mut, 4),
            "sequence": [int(x) for x in seq],
        })

    return phrase, plans


def render_sequence(slices, seq):
    chunks = [slices[i % len(slices)]["audio"] for i in seq]
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    return normalize(np.concatenate(chunks))


def export_preview(slices, outdir):
    p = outdir / "slices_preview"
    p.mkdir(parents=True, exist_ok=True)
    for s in slices:
        wav = p / f"slice_{s['index']:03d}_{s['label']}_{int(s['start_ms'])}ms.wav"
        sf.write(wav, normalize(s["audio"]), SR)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="London")
    parser.add_argument("--method", choices=["hybrid", "onset_strong"], default="hybrid")
    parser.add_argument("--target-cuts", type=int, default=16)
    parser.add_argument("--min-ms", type=float, default=45.0)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--mutation", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    source = find_source(args.source)
    print("Source :", source)

    y, sr = load_audio(source)
    cuts = get_cuts(y, args.method, args.target_cuts, args.min_ms)
    slices = classify_slices(cuts_to_slices(y, cuts))

    safe = source.stem.replace(" ", "_").replace("'", "")
    outdir = OUT_DIR / safe / f"{args.method}_{args.target_cuts}cuts_{args.bars}bars"
    outdir.mkdir(parents=True, exist_ok=True)

    preview_dir = export_preview(slices, outdir)

    print("Labels :")
    for label in ["kick", "snare", "hat", "other"]:
        print(f"  {label:6}: {len(pool_indices(slices, label))}")

    renders = []

    for i in range(1, args.count + 1):
        seq, plans = make_longform(
            slices,
            bars=args.bars,
            mutation=args.mutation,
        )

        audio = render_sequence(slices, seq)
        wav = outdir / f"{safe}_backbone_{args.bars}bars_{i:03d}.wav"
        sf.write(wav, audio, SR)

        renders.append({
            "index": int(i),
            "wav": str(wav),
            "sequence": [int(x) for x in seq],
            "bars": plans,
        })

        print("Export :", wav)

    metadata = {
        "version": "jungle_grammar_v04_backbone",
        "source": str(source),
        "source_only": True,
        "no_overlap": True,
        "no_txt_variations": True,
        "backbone": "kick snare kick snare",
        "structural_positions_16": {
            "0": "kick",
            "4": "snare",
            "8": "kick",
            "12": "snare",
        },
        "method": args.method,
        "target_cuts": args.target_cuts,
        "min_ms": args.min_ms,
        "bars": args.bars,
        "mutation": args.mutation,
        "preview_dir": str(preview_dir),
        "slices": [
            {
                "index": int(s["index"]),
                "label": s["label"],
                "start_ms": float(s["start_ms"]),
                "end_ms": float(s["end_ms"]),
                "duration": float(s["duration"]),
                "features": s["features"],
            }
            for s in slices
        ],
        "renders": renders,
    }

    (outdir / "metadata_jungle_grammar_v04_backbone.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("")
    print("Dossier :", outdir)
    print("Preview :", preview_dir)


if __name__ == "__main__":
    main()
