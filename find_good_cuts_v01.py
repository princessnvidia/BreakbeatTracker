#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Find Good Cuts v01

But :
Trouver les bonnes découpes d'un break automatiquement.

Le script exporte plusieurs propositions de découpe :
1. onset_backtrack : détection transient + backtrack
2. onset_strong    : seulement les attaques les plus fortes
3. grid_snap       : grille régulière calée sur le meilleur onset
4. hybrid          : grille + correction vers l'onset le plus proche

Ensuite tu écoutes les previews et on garde la meilleure méthode.

Usage :
    python find_good_cuts_v01.py --source "Camo"

Plus de détail :
    python find_good_cuts_v01.py --source "Camo" --target-cuts 16

Si les cuts sont trop courts :
    python find_good_cuts_v01.py --source "Camo" --min-ms 80

Sorties :
    exports/find_good_cuts_v01/Camo.../
        01_onset_backtrack/
        02_onset_strong/
        03_grid_snap/
        04_hybrid/
        comparison_report.txt
"""

from pathlib import Path
import argparse
import json
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
OUT_DIR = Path("exports/find_good_cuts_v01")
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


def cuts_to_slices(y, cuts):
    slices = []
    for i in range(len(cuts) - 1):
        start = cuts[i]
        end = cuts[i + 1]
        if end <= start:
            continue
        audio = fade(y[start:end].copy(), ms=2)
        slices.append({
            "index": i,
            "start": int(start),
            "end": int(end),
            "start_ms": float(start / SR * 1000),
            "end_ms": float(end / SR * 1000),
            "duration": float((end - start) / SR),
            "audio": audio,
        })
    return slices


def onset_backtrack_cuts(y, min_ms):
    onset_samples = librosa.onset.onset_detect(
        y=y,
        sr=SR,
        units="samples",
        backtrack=True,
        delta=0.07,
        wait=1,
        pre_max=3,
        post_max=3,
        pre_avg=8,
        post_avg=8,
    )
    min_samples = int(SR * min_ms / 1000)
    return clean_cuts([0] + list(onset_samples), len(y), min_samples)


def onset_strong_cuts(y, target_cuts, min_ms):
    env = librosa.onset.onset_strength(y=y, sr=SR)
    frames = np.arange(len(env))
    samples = librosa.frames_to_samples(frames)

    # pics candidats
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

    min_samples = int(SR * min_ms / 1000)
    return clean_cuts([0] + chosen, len(y), min_samples)


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


def grid_snap_cuts(y, target_cuts):
    offset = best_grid_offset(y, target_cuts)
    usable = max(1, len(y) - offset)
    slice_len = usable / target_cuts
    cuts = [int(offset + i * slice_len) for i in range(target_cuts)]
    return clean_cuts([0] + cuts, len(y), 1)


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
        if nearby:
            # prend l'onset le plus proche de la grille
            cut = min(nearby, key=lambda o: abs(o - grid))
        else:
            cut = grid

        cuts.append(cut)

    min_samples = int(SR * min_ms / 1000)
    return clean_cuts([0] + cuts, len(y), min_samples)


def render_reconstruction(slices):
    if not slices:
        return np.zeros(1, dtype=np.float32)
    return normalize(np.concatenate([s["audio"] for s in slices]))


def export_method(method_dir, method_name, y, cuts):
    method_dir.mkdir(parents=True, exist_ok=True)

    slices = cuts_to_slices(y, cuts)

    slice_dir = method_dir / "slices"
    slice_dir.mkdir(exist_ok=True)

    rows = []
    rows.append(f"METHOD: {method_name}")
    rows.append(f"SLICE COUNT: {len(slices)}")
    rows.append("")
    rows.append("IDX | START_MS | END_MS | DUR | FILE")
    rows.append("-----------------------------------")

    for s in slices:
        wav = slice_dir / f"slice_{s['index']:03d}_{int(s['start_ms'])}ms.wav"
        sf.write(wav, normalize(s["audio"]), SR)
        rows.append(
            f"{s['index']:03d} | {s['start_ms']:8.1f} | {s['end_ms']:8.1f} | "
            f"{s['duration']:.3f}s | {wav}"
        )

    recon = render_reconstruction(slices)
    recon_wav = method_dir / f"{method_name}_reconstruction.wav"
    sf.write(recon_wav, recon, SR)

    report = method_dir / "cuts_report.txt"
    report.write_text("\n".join(rows), encoding="utf-8")

    metadata = {
        "method": method_name,
        "slice_count": len(slices),
        "reconstruction": str(recon_wav),
        "cuts": [
            {
                "index": s["index"],
                "start_ms": s["start_ms"],
                "end_ms": s["end_ms"],
                "duration": s["duration"],
            }
            for s in slices
        ],
    }
    (method_dir / "cuts_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "method": method_name,
        "slice_count": len(slices),
        "reconstruction": str(recon_wav),
        "report": str(report),
        "dir": str(method_dir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Camo")
    parser.add_argument("--target-cuts", type=int, default=16,
                        help="Nombre cible de slices/cases. Pour 8 blocs de 2 cases, teste aussi 8.")
    parser.add_argument("--min-ms", type=float, default=45.0)
    parser.add_argument("--strong-target", type=int, default=None)
    args = parser.parse_args()

    source = find_source(args.source)
    print("Source :", source)

    y, sr = load_audio(source)

    safe = source.stem.replace(" ", "_").replace("'", "")
    outdir = OUT_DIR / safe
    outdir.mkdir(parents=True, exist_ok=True)

    strong_target = args.strong_target or args.target_cuts

    methods = []

    cuts1 = onset_backtrack_cuts(y, min_ms=args.min_ms)
    methods.append(("01_onset_backtrack", cuts1))

    cuts2 = onset_strong_cuts(y, target_cuts=strong_target, min_ms=args.min_ms)
    methods.append(("02_onset_strong", cuts2))

    cuts3 = grid_snap_cuts(y, target_cuts=args.target_cuts)
    methods.append(("03_grid_snap", cuts3))

    cuts4 = hybrid_cuts(y, target_cuts=args.target_cuts, min_ms=args.min_ms)
    methods.append(("04_hybrid", cuts4))

    summaries = []

    for name, cuts in methods:
        summary = export_method(outdir / name, name, y, cuts)
        summaries.append(summary)
        print(f"{name}: {summary['slice_count']} slices -> {summary['reconstruction']}")

    report_lines = []
    report_lines.append(f"SOURCE: {source}")
    report_lines.append(f"TARGET_CUTS: {args.target_cuts}")
    report_lines.append(f"MIN_MS: {args.min_ms}")
    report_lines.append("")
    report_lines.append("Écoute d'abord les reconstructions :")
    report_lines.append("")

    for s in summaries:
        report_lines.append(f"{s['method']} | slices={s['slice_count']} | {s['reconstruction']}")

    report_lines.append("")
    report_lines.append("Puis écoute les slices individuelles dans chaque dossier /slices.")
    report_lines.append("La meilleure méthode sera celle où les slices commencent naturellement sur les coups.")

    (outdir / "comparison_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    (outdir / "comparison_metadata.json").write_text(json.dumps({
        "version": "find_good_cuts_v01",
        "source": str(source),
        "target_cuts": args.target_cuts,
        "min_ms": args.min_ms,
        "methods": summaries,
    }, indent=2), encoding="utf-8")

    print("")
    print("Rapport :", outdir / "comparison_report.txt")
    print("Dossier :", outdir)


if __name__ == "__main__":
    main()
