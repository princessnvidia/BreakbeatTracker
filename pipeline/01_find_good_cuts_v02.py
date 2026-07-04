#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_find_good_cuts_v02.py

Découpe v2 pour BreakbeatAI :
- sur-détection volontaire des attaques
- nettoyage des onsets trop proches
- fusion intelligente des micro-slices
- export de previews
- sortie JSON réutilisable par Break IR

Objectif :
Obtenir plus de vrais hits qu'en v01, surtout sur Amen/Think/Camo.

Sorties :
    dataset/cuts/<break>_cuts_v02.json
    dataset/cuts/<break>/preview_slices/*.wav
    dataset/cuts/<break>/reconstruction_cuts_v02.wav

Usage :
    python pipeline/01_find_good_cuts_v02.py --source "Amen"
    python pipeline/01_find_good_cuts_v02.py --source "Camo"
    python pipeline/01_find_good_cuts_v02.py --source "London"

Plus sensible :
    python pipeline/01_find_good_cuts_v02.py --source "Amen" --delta 0.035

Moins sensible :
    python pipeline/01_find_good_cuts_v02.py --source "Amen" --delta 0.07

Moins de fusion :
    python pipeline/01_find_good_cuts_v02.py --source "Amen" --merge-ms 35

Plus de fusion :
    python pipeline/01_find_good_cuts_v02.py --source "Amen" --merge-ms 70
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
    print("librosa manquant : python -m pip install librosa soundfile numpy")
    sys.exit(1)


BREAKS_DIR = Path("breaks")
OUT_DIR = Path("dataset/cuts")
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


def safe_name(path):
    return path.stem.replace(" ", "_").replace("'", "").replace("/", "_")


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


def rms(y):
    if len(y) == 0:
        return 0.0
    return float(np.sqrt(np.mean(y * y)))


def peak(y):
    if len(y) == 0:
        return 0.0
    return float(np.max(np.abs(y)))


def onset_candidates(y, delta=0.045):
    """
    Sur-détection volontaire.
    On veut trop de points puis fusionner après.
    """
    onsets = librosa.onset.onset_detect(
        y=y,
        sr=SR,
        units="samples",
        backtrack=True,
        delta=delta,
        wait=1,
        pre_max=2,
        post_max=2,
        pre_avg=4,
        post_avg=4,
    )

    # Ajoute aussi les pics avec une méthode plus permissive.
    env = librosa.onset.onset_strength(y=y, sr=SR)
    peaks = librosa.util.peak_pick(
        env,
        pre_max=2,
        post_max=2,
        pre_avg=4,
        post_avg=4,
        delta=max(0.02, delta * 0.75),
        wait=1,
    )
    peak_samples = librosa.frames_to_samples(peaks)

    points = sorted(set([0] + [int(x) for x in onsets] + [int(x) for x in peak_samples] + [len(y)]))
    return points


def dedupe_points(points, min_gap_samples):
    if not points:
        return []

    points = sorted(set(int(p) for p in points))
    out = [points[0]]

    for p in points[1:]:
        if p - out[-1] >= min_gap_samples or p == points[-1]:
            out.append(p)

    if out[-1] != points[-1]:
        out.append(points[-1])

    return out


def slice_stats(y, start, end):
    audio = y[start:end]
    return {
        "start": int(start),
        "end": int(end),
        "duration_ms": float((end - start) / SR * 1000),
        "rms": rms(audio),
        "peak": peak(audio),
    }


def build_initial_segments(y, points, min_ms=18, max_ms=360):
    min_len = int(SR * min_ms / 1000)
    max_len = int(SR * max_ms / 1000)

    segments = []

    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]

        if end <= start:
            continue

        if end - start < min_len:
            continue

        if end - start > max_len:
            # Si une zone est trop longue, on la coupe en morceaux réguliers.
            pos = start
            while pos < end:
                sub_end = min(end, pos + max_len)
                if sub_end - pos >= min_len:
                    segments.append(slice_stats(y, pos, sub_end))
                pos = sub_end
        else:
            segments.append(slice_stats(y, start, end))

    return segments


def merge_segments(y, segments, merge_ms=55, min_energy_ratio=0.18):
    """
    Fusionne les micro-segments trop proches/faibles.
    Idée :
    - si segment très court et peu énergique, il appartient probablement au hit précédent
    - si deux segments sont séparés de moins que merge_ms, on fusionne
    """
    if not segments:
        return []

    merge_samples = int(SR * merge_ms / 1000)

    merged = []
    current = dict(segments[0])

    global_peak = max((s["peak"] for s in segments), default=1e-9)
    energy_threshold = global_peak * min_energy_ratio

    for seg in segments[1:]:
        gap = seg["start"] - current["end"]
        current_len = current["end"] - current["start"]
        seg_len = seg["end"] - seg["start"]

        seg_is_weak = seg["peak"] < energy_threshold
        current_is_tiny = current_len < merge_samples
        seg_is_tiny = seg_len < merge_samples

        should_merge = (
            gap <= merge_samples
            and (
                seg_is_weak
                or current_is_tiny
                or seg_is_tiny
            )
        )

        if should_merge:
            current["end"] = seg["end"]
            audio = y[current["start"]:current["end"]]
            current["duration_ms"] = float((current["end"] - current["start"]) / SR * 1000)
            current["rms"] = rms(audio)
            current["peak"] = peak(audio)
        else:
            merged.append(current)
            current = dict(seg)

    merged.append(current)
    return merged


def final_cleanup(y, segments, min_ms=35, max_ms=420):
    """
    Dernier nettoyage :
    - élimine les segments minuscules
    - recoupe les segments vraiment trop longs
    """
    min_len = int(SR * min_ms / 1000)
    max_len = int(SR * max_ms / 1000)

    cleaned = []

    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        length = end - start

        if length < min_len:
            if cleaned:
                cleaned[-1]["end"] = end
                audio = y[cleaned[-1]["start"]:cleaned[-1]["end"]]
                cleaned[-1]["duration_ms"] = float((cleaned[-1]["end"] - cleaned[-1]["start"]) / SR * 1000)
                cleaned[-1]["rms"] = rms(audio)
                cleaned[-1]["peak"] = peak(audio)
            continue

        if length > max_len:
            pos = start
            while pos < end:
                sub_end = min(end, pos + max_len)
                if sub_end - pos >= min_len:
                    cleaned.append(slice_stats(y, pos, sub_end))
                pos = sub_end
        else:
            cleaned.append(slice_stats(y, start, end))

    return cleaned


def segments_to_cuts(segments, total_len):
    if not segments:
        return [0, total_len]

    cuts = [0]

    for seg in segments:
        if seg["start"] not in cuts:
            cuts.append(seg["start"])

    if total_len not in cuts:
        cuts.append(total_len)

    cuts = sorted(set(cuts))
    return cuts


def export_preview(y, segments, outdir):
    preview = outdir / "preview_slices"
    preview.mkdir(parents=True, exist_ok=True)

    exported = []

    for i, seg in enumerate(segments):
        audio = fade(y[seg["start"]:seg["end"]].copy(), ms=2)
        wav = preview / f"slice_{i:03d}_{int(seg['duration_ms'])}ms.wav"
        sf.write(wav, normalize(audio), SR)

        exported.append({
            "index": int(i),
            "audio_path": str(wav),
            "start_sample": int(seg["start"]),
            "end_sample": int(seg["end"]),
            "start_ms": float(seg["start"] / SR * 1000),
            "end_ms": float(seg["end"] / SR * 1000),
            "duration_ms": float(seg["duration_ms"]),
            "rms": float(seg["rms"]),
            "peak": float(seg["peak"]),
        })

    if segments:
        recon = np.concatenate([
            fade(y[s["start"]:s["end"]].copy(), ms=2)
            for s in segments
        ])
    else:
        recon = np.zeros(1, dtype=np.float32)

    recon_path = outdir / "reconstruction_cuts_v02.wav"
    sf.write(recon_path, normalize(recon), SR)

    return exported, str(recon_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Amen")
    parser.add_argument("--delta", type=float, default=0.045)
    parser.add_argument("--dedupe-ms", type=float, default=18.0)
    parser.add_argument("--merge-ms", type=float, default=55.0)
    parser.add_argument("--min-ms", type=float, default=35.0)
    parser.add_argument("--max-ms", type=float, default=420.0)
    parser.add_argument("--min-energy-ratio", type=float, default=0.18)
    args = parser.parse_args()

    source = find_source(args.source)
    print("Source :", source)

    y, sr = load_audio(source)
    safe = safe_name(source)

    outdir = OUT_DIR / safe
    outdir.mkdir(parents=True, exist_ok=True)

    raw_points = onset_candidates(y, delta=args.delta)
    points = dedupe_points(raw_points, int(SR * args.dedupe_ms / 1000))

    initial = build_initial_segments(
        y,
        points,
        min_ms=max(8, args.min_ms * 0.45),
        max_ms=args.max_ms,
    )

    merged = merge_segments(
        y,
        initial,
        merge_ms=args.merge_ms,
        min_energy_ratio=args.min_energy_ratio,
    )

    final = final_cleanup(
        y,
        merged,
        min_ms=args.min_ms,
        max_ms=args.max_ms,
    )

    cuts = segments_to_cuts(final, len(y))
    exported, reconstruction = export_preview(y, final, outdir)

    metadata = {
        "version": "cuts_v02",
        "source": str(source),
        "sample_rate": SR,
        "duration": float(len(y) / SR),
        "parameters": {
            "delta": args.delta,
            "dedupe_ms": args.dedupe_ms,
            "merge_ms": args.merge_ms,
            "min_ms": args.min_ms,
            "max_ms": args.max_ms,
            "min_energy_ratio": args.min_energy_ratio,
        },
        "counts": {
            "raw_points": len(raw_points),
            "deduped_points": len(points),
            "initial_segments": len(initial),
            "final_segments": len(final),
        },
        "cuts_samples": [int(c) for c in cuts],
        "cuts_ms": [float(c / SR * 1000) for c in cuts],
        "reconstruction": reconstruction,
        "slices": exported,
    }

    cuts_path = OUT_DIR / f"{safe}_cuts_v02.json"
    cuts_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("")
    print("Cuts v02 créé :", cuts_path)
    print("Preview       :", outdir / "preview_slices")
    print("Reconstruction:", reconstruction)
    print("")
    print("Counts :")
    for k, v in metadata["counts"].items():
        print(f"  {k:17}: {v}")


if __name__ == "__main__":
    main()
