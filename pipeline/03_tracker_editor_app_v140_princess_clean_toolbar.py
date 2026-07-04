import traceback
import wave
import re
import random
from collections import Counter
import math
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BreakbeatAI Tracker Editor v140 — slice index

- Un seul fichier audio source par break.
- Les slices sont des index start/end dans le JSON.
- Pas de WAV généré par slice.
- Affichage simple : slice 0, slice 1, slice 2...
- 32 positions.
- Randomize loop64.
- Chaque correction est sauvegardée et influence les prochains Randomize.
"""

from pathlib import Path
import argparse
import json
import shutil
import subprocess
import time
import os
import signal
import atexit
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import soundfile as sf


SR = 44100

PAIR_BLOCKS_DIR = Path("dataset/pair_blocks_v02")
OUT_DIR = Path("dataset/tracker_edits")
VALIDATED_DIR = Path("dataset/validated_patterns")

LEARNING_DIR = Path("dataset/learning")
CORRECTIONS_DIR = LEARNING_DIR / "corrections"
LATEST_DIR = LEARNING_DIR / "latest"
MEMORY_PATH = LEARNING_DIR / "breakbeatai_slice_index_memory.json"

# v44 :
# 1 hit audio = 2 cases visuelles.
# On place donc les hits sur 0, 2, 4, 6...
HIT_LENGTH_STEPS = 2
HIT_SPACING_STEPS = 2
HIT_SLOTS = 16

# v44 : grille verrouillée.
# 32 cases, 4 cases = 1 temps, step_ms = 96.774 => 155 BPM.
LOCKED_GRID_STEPS = 64
LOCKED_HIT_SLOTS = 16

# v44 : tempo cible.
# Grille : 4 cases = 1 temps.
# BPM = 60000 / (step_ms * 4)
TARGET_BPM = 155.0
TARGET_STEP_MS = 60000.0 / (TARGET_BPM * 4.0)


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def safe_name(pair_json):
    return pair_json.stem.replace("_pair_blocks_v02", "")


def find_pair_json(source_query):
    files = sorted(PAIR_BLOCKS_DIR.glob("*_pair_blocks_v02.json"))
    matches = [p for p in files if source_query.lower() in p.name.lower()]

    if not matches:
        print(f"Aucun pair_blocks_v02 JSON trouvé pour : {source_query}")
        print("")
        print("Crée d'abord un index de slice avec :")
        print(f'python pipeline/01_autoslice_break_index_v04.py --source "{source_query}"')
        raise SystemExit(1)

    return matches[0]


def list_break_jsons():
    out = {}

    for path in sorted(PAIR_BLOCKS_DIR.glob("*_pair_blocks_v02.json")):
        out[safe_name(path)] = path

    return out


def normalize(y, peak=0.95):
    y = np.asarray(y, dtype=np.float32)

    if len(y) == 0:
        return y

    m = float(np.max(np.abs(y)))

    if m <= 1e-9:
        return y

    return (y / m * peak).astype(np.float32)


def fit_audio_no_speed(audio, target_len):
    """
    Fallback sans accélération :
    - trop long : coupe
    - trop court : silence
    """
    audio = np.asarray(audio, dtype=np.float32)
    target_len = int(target_len)

    if target_len <= 1:
        return audio[:1].copy() if len(audio) else np.zeros(1, dtype=np.float32)

    if len(audio) == 0:
        return np.zeros(target_len, dtype=np.float32)

    if len(audio) > target_len:
        fitted = audio[:target_len].copy()
        fade_len = min(48, target_len // 8)

        if fade_len > 2:
            fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            fitted[-fade_len:] *= fade

        return fitted.astype(np.float32)

    if len(audio) < target_len:
        fitted = np.zeros(target_len, dtype=np.float32)
        fitted[:len(audio)] = audio
        return fitted.astype(np.float32)

    return audio.copy().astype(np.float32)


def make_atempo_chain(tempo):
    """
    ffmpeg atempo garde le pitch.
    tempo > 1 = raccourcit
    tempo < 1 = rallonge
    """
    tempo = float(tempo)

    factors = []

    while tempo > 2.0:
        factors.append(2.0)
        tempo /= 2.0

    while tempo < 0.5:
        factors.append(0.5)
        tempo /= 0.5

    factors.append(tempo)

    return ",".join(f"atempo={f:.8f}" for f in factors)


def warp_audio_to_length_external(audio, target_len, cache_key=None):
    """
    v44 : warp pitch-preserve.

    Priorité :
    1. rubberband-cli si installé
    2. ffmpeg atempo si installé
    3. fallback sans accélération

    Important :
    - on ne fait plus de np.interp direct comme v41
    - donc plus de son accéléré/pitché
    """
    audio = np.asarray(audio, dtype=np.float32)
    target_len = int(target_len)

    if target_len <= 1:
        return audio[:1].copy() if len(audio) else np.zeros(1, dtype=np.float32)

    if len(audio) <= 1:
        return np.zeros(target_len, dtype=np.float32)

    if len(audio) == target_len:
        return audio.copy().astype(np.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUT_DIR / "_warp_cache_v44"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if cache_key is None:
        cache_key = f"anon_{len(audio)}_{target_len}_{time.time_ns()}"

    in_wav = tmp_dir / f"{cache_key}_in.wav"
    out_wav = tmp_dir / f"{cache_key}_out.wav"

    sf.write(in_wav, normalize(audio), SR)

    rubberband = shutil.which("rubberband")
    ffmpeg = shutil.which("ffmpeg")

    # Durée cible / durée source.
    ratio = float(target_len) / float(len(audio))

    if rubberband:
        try:
            # rubberband -t ratio = time stretch ratio.
            cmd = [
                rubberband,
                "-q",
                "-t",
                f"{ratio:.8f}",
                str(in_wav),
                str(out_wav),
            ]

            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            y, sr = sf.read(out_wav, always_2d=False)

            if getattr(y, "ndim", 1) > 1:
                y = y.mean(axis=1)

            y = resample_linear(y.astype(np.float32), sr, SR)
            return fit_audio_no_speed(y, target_len)

        except Exception as exc:
            print(f"[v140] rubberband échec, fallback ffmpeg/no-speed : {exc}")

    if ffmpeg:
        try:
            # ffmpeg atempo utilise tempo = source_duration / target_duration.
            tempo = float(len(audio)) / float(target_len)
            atempo = make_atempo_chain(tempo)

            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(in_wav),
                "-filter:a",
                atempo,
                str(out_wav),
            ]

            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            y, sr = sf.read(out_wav, always_2d=False)

            if getattr(y, "ndim", 1) > 1:
                y = y.mean(axis=1)

            y = resample_linear(y.astype(np.float32), sr, SR)
            return fit_audio_no_speed(y, target_len)

        except Exception as exc:
            print(f"[v140] ffmpeg atempo échec, fallback no-speed : {exc}")

    return fit_audio_no_speed(audio, target_len)


def resample_linear(y, src_sr, dst_sr=SR):
    y = np.asarray(y, dtype=np.float32)

    if src_sr == dst_sr:
        return y

    if len(y) <= 1:
        return y

    duration = len(y) / float(src_sr)
    new_len = max(1, int(round(duration * dst_sr)))

    old_x = np.linspace(0.0, 1.0, len(y), endpoint=False)
    new_x = np.linspace(0.0, 1.0, new_len, endpoint=False)

    return np.interp(new_x, old_x, y).astype(np.float32)


def load_source_audio(path):
    audio, sr = sf.read(path, always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)
    audio = resample_linear(audio, sr, SR)

    audio = audio - float(np.mean(audio))

    return audio.astype(np.float32)


def now_stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def file_stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def load_memory():
    LEARNING_DIR.mkdir(parents=True, exist_ok=True)

    if MEMORY_PATH.exists():
        try:
            return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "version": "breakbeatai_slice_index_memory_v01",
        "description": "Mémoire des corrections avec slices index-only.",
        "breaks": {},
        "global": {},
        "events_total": 0,
        "updated_at": None,
    }


def save_memory(memory):
    LEARNING_DIR.mkdir(parents=True, exist_ok=True)
    memory["updated_at"] = now_stamp()
    MEMORY_PATH.write_text(json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8")


def memory_bucket(memory, safe, x_step):
    local_pos = str(safe_int(x_step) % 8)

    memory.setdefault("breaks", {})
    memory.setdefault("global", {})

    break_bucket = memory["breaks"].setdefault(safe, {}).setdefault(local_pos, {
        "positive_pair_counts": {},
        "negative_pair_counts": {},
        "events": 0,
    })

    global_bucket = memory["global"].setdefault(local_pos, {
        "positive_pair_counts": {},
        "negative_pair_counts": {},
        "events": 0,
    })

    return break_bucket, global_bucket


def inc_count(d, key, amount=1):
    key = str(key)
    d[key] = int(d.get(key, 0)) + int(amount)


class SliceIndexTracker:
    def __init__(self, root, pair_json):
        self.root = root
        self.pair_json = pair_json
        self.project = self.load_project(pair_json)

        self.blocks = self.project["blocks"]
        self.block_by_pair = {int(b["pair"]): b for b in self.blocks}
        self.pair_values = [int(b["pair"]) for b in self.blocks]
        self.pair_to_lane = {int(pair): i for i, pair in enumerate(self.pair_values)}

        self.source_audio_cache = {}
        self.warp_cache = {}

        self.step_count = LOCKED_GRID_STEPS
        self.left_width = 112
        self.step_width = 36
        self.row_height = 38
        self.case_height = 29
        self.case_y_padding = 4
        self.max_case_length = 4

        self.default_step_ms = self.guess_step_ms()

        self.canvas_width = self.left_width + self.step_width * self.step_count
        self.canvas_height_total = self.row_height * max(1, len(self.pair_values))
        self.visible_canvas_height = min(620, max(220, self.canvas_height_total))

        self.pattern = self.build_initial_pattern(randomize=False)
        self.selected_id = self.pattern[0]["id"] if self.pattern else None

        self.drag_mode = None
        self.drag_before = None
        self.drag_created_id = None
        self.drag_start_x = 0
        self.drag_start_step = 0
        self.drag_start_len = 0

        self.looping = False
        self.loop_after_id = None
        self.play_process = None
        self.audition_process = None
        self.external_player = self.detect_external_player()

        self.loop_target_sec = 180.0
        self.loop_max_repeats = 256

        self.playhead_after_id = None
        self.loop_started_at = None
        self.current_loop_sec = 0.0

        self.root.title("BreakbeatAI Tracker Editor v140 — slice index")
        self.root.geometry("1480x880")
        self.root.configure(bg="#111018")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        atexit.register(self.shutdown_audio)

        self.build_ui()
        self.draw()
        self.refresh_panel()
        self.bind_keys()

        self.root.after(200, self.force_keyboard_focus)

    def detect_external_player(self):
        for name in ["pw-play", "paplay", "aplay", "ffplay"]:
            path = shutil.which(name)

            if path:
                print(f"[v140] Backend audio : {name} -> {path}")
                return name

        print("[v140] Aucun backend système trouvé, fallback sounddevice.")
        return None

    def load_project(self, pair_json):
        meta = json.loads(pair_json.read_text(encoding="utf-8"))
        safe = safe_name(pair_json)

        blocks = []
        source_audio = meta.get("source_audio")

        for block in meta.get("blocks", []):
            pair = int(block["pair"])

            block_source = block.get("source_audio") or source_audio
            audio_path = block.get("audio_path")

            if block.get("source_start_sample") is not None and block.get("source_end_sample") is not None:
                storage = "index_only"
            elif audio_path:
                storage = "audio_path"
            else:
                storage = "unknown"

            blocks.append({
                "pair": pair,
                "name": block.get("name") or f"slice {pair}",
                "storage": storage,
                "source_audio": block_source,
                "source_start_sample": block.get("source_start_sample"),
                "source_end_sample": block.get("source_end_sample"),
                "audio_path": audio_path,
                "duration_ms": float(block.get("duration_ms", 0.0)),
                "source_start_ms": block.get("source_start_ms"),
                "source_end_ms": block.get("source_end_ms"),
            })

        blocks = sorted(blocks, key=lambda b: int(b["pair"]))

        if not blocks:
            raise RuntimeError(f"Aucun block dans {pair_json}")

        return {
            "safe": safe,
            "source_audio": source_audio,
            "source_pair_json": str(pair_json),
            "storage": meta.get("storage", "mixed"),
            "slice_method": meta.get("slice_method"),
            "source_duration_ms": meta.get("source_duration_ms"),
            "loop_duration_ms": meta.get("loop_duration_ms"),
            "step_ms_from_json": meta.get("step_ms"),
            "blocks": blocks,
        }

    def guess_step_ms(self):
        """
        v44 : BPM réellement verrouillé à 155.
        4 cases = 1 temps.
        step_ms = 60000 / (155 * 4) = 96.774 ms.
        """
        return round(TARGET_STEP_MS, 4)

    def get_step_ms(self):
        try:
            value = float(self.step_ms_var.get())
        except Exception:
            value = self.default_step_ms

        return max(10.0, min(2000.0, value))

    def get_step_samples(self):
        return max(1, int(round(SR * self.get_step_ms() / 1000.0)))

    def get_loop_samples(self):
        return int(self.step_count * self.get_step_samples())

    def lane_to_pair(self, lane_index):
        lane_index = max(0, min(len(self.pair_values) - 1, int(lane_index)))
        return int(self.pair_values[lane_index])

    def pair_color(self, pair):
        palette = [
            "#ff9bd2",
            "#cba6f7",
            "#89b4fa",
            "#94e2d5",
            "#a6e3a1",
            "#f9e2af",
            "#fab387",
            "#f38ba8",
            "#eba0ac",
            "#b4befe",
            "#74c7ec",
            "#f5c2e7",
        ]

        return palette[int(pair) % len(palette)]

    def get_source_audio(self, path):
        path = str(path)

        if path not in self.source_audio_cache:
            self.source_audio_cache[path] = load_source_audio(Path(path))

        return self.source_audio_cache[path]

    def get_audio(self, pair):
        pair = int(pair)
        block = self.block_by_pair[pair]

        if block["storage"] == "index_only":
            source_audio = self.get_source_audio(block["source_audio"])
            a = int(block["source_start_sample"])
            b = int(block["source_end_sample"])

            a = max(0, min(len(source_audio) - 1, a))
            b = max(a + 1, min(len(source_audio), b))

            return source_audio[a:b].astype(np.float32)

        if block["storage"] == "audio_path" and block.get("audio_path"):
            return load_source_audio(block["audio_path"])

        raise RuntimeError(f"Slice {pair} illisible : {block}")

    def pair_weight_from_memory(self, pair, x_step):
        memory = load_memory()
        safe = self.project["safe"]
        local_pos = str(safe_int(x_step) % 8)
        pair_key = str(pair)

        weight = 1.0

        try:
            g = memory.get("global", {}).get(local_pos, {})
            weight += int(g.get("positive_pair_counts", {}).get(pair_key, 0)) * 2.0
            weight -= int(g.get("negative_pair_counts", {}).get(pair_key, 0)) * 2.5
        except Exception:
            pass

        try:
            b = memory.get("breaks", {}).get(safe, {}).get(local_pos, {})
            weight += int(b.get("positive_pair_counts", {}).get(pair_key, 0)) * 5.0
            weight -= int(b.get("negative_pair_counts", {}).get(pair_key, 0)) * 5.0
        except Exception:
            pass

        return max(0.05, float(weight))

    def pick_pair_for_pos(self, pos, randomize=False, rng=None):
        pool = list(self.pair_values)

        if not pool:
            return 0

        if not randomize and not MEMORY_PATH.exists():
            return int(pool[int(pos) % len(pool)])

        weights = [self.pair_weight_from_memory(pair, pos) for pair in pool]

        if randomize and rng is not None:
            weights = np.asarray(weights, dtype=np.float64)
            weights = weights / weights.sum()

            return int(rng.choice(pool, p=weights))

        best_pair = int(pool[0])
        best_weight = -1.0

        for pair, weight in zip(pool, weights):
            if weight > best_weight:
                best_pair = int(pair)
                best_weight = float(weight)

        if best_weight > 1.0:
            return best_pair

        return int(pool[int(pos) % len(pool)])

    def build_initial_pattern(self, randomize=False):
        rng = np.random.default_rng(int(time.time_ns() % (2**32))) if randomize else None
        pattern = []

        total_slots = LOCKED_HIT_SLOTS
        max_items = min(len(self.pair_values), LOCKED_HIT_SLOTS)

        print(
            f"[v140] Génération pattern | break={self.project['safe']} | "
            f"slices={len(self.pair_values)} | slots={total_slots} | randomize={randomize}"
        )

        for slot in range(max_items):
            x_step = slot * HIT_SPACING_STEPS

            if randomize:
                pair = self.pick_pair_for_pos(slot, randomize=True, rng=rng)
            else:
                pair = int(self.pair_values[slot])

            lane = self.pair_to_lane.get(pair, 0)

            pattern.append({
                "id": len(pattern),
                "x_step": x_step,
                "lane": lane,
                "pair": pair,
                "length": HIT_LENGTH_STEPS,
                "variation_bar": x_step // 8,
                "variation_pos": x_step % 8,
                "hit_slot": slot,
                "randomized": bool(randomize),
            })

            print(f"[v140] slot {slot:02d} | step {x_step:02d} -> slice {pair:02d}")

        return pattern

    def build_ui(self):
        main = tk.Frame(self.root, bg="#111018")
        main.pack(fill="both", expand=True, padx=14, pady=14)

        title = tk.Label(
            main,
            text="BreakbeatAI v140 — Princess clean toolbar",
            bg="#111018",
            fg="#ff7acc",
            font=("Sans", 20, "bold"),
        )
        title.pack(anchor="w")

        backend = self.external_player if self.external_player else "sounddevice"
        subtitle = tk.Label(
            main,
            text=(
                f"Backend audio : {backend} | BPM cible : {TARGET_BPM:.1f} | "
                f"Step ms : {TARGET_STEP_MS:.3f} | méthode : {self.project.get('slice_method')}"
            ),
            bg="#111018",
            fg="#b9acc8",
        )
        subtitle.pack(anchor="w", pady=(0, 10))

        break_frame = tk.Frame(main, bg="#15111f")
        break_frame.pack(fill="x", pady=(0, 8))

        self.break_jsons = list_break_jsons()

        tk.Label(
            break_frame,
            text="Break",
            bg="#15111f",
            fg="#ff7acc",
            font=("Sans", 10, "bold"),
        ).pack(side="left", padx=(8, 6))

        self.break_var = tk.StringVar(value=self.project["safe"])
        self.break_box = ttk.Combobox(
            break_frame,
            textvariable=self.break_var,
            values=list(self.break_jsons.keys()),
            width=36,
            state="readonly",
        )
        self.break_box.pack(side="left", padx=4)
        self.break_box.bind("<<ComboboxSelected>>", self.switch_break)

        tk.Button(break_frame, text="Charger break", command=self.switch_break, bg="#30283f", fg="#f5eefe").pack(side="left", padx=4)
        tk.Button(break_frame, text="Randomize loop64", command=self.randomize_pattern, bg="#50315f", fg="#f5eefe").pack(side="left", padx=4)
        tk.Button(break_frame, text="Refresh breaks", command=self.refresh_breaks, bg="#30283f", fg="#f5eefe").pack(side="left", padx=4)

        tk.Label(
            break_frame,
            text="Import auto-slice : un seul audio source, les coupes vivent dans le JSON.",
            bg="#15111f",
            fg="#b9acc8",
        ).pack(side="left", padx=12)

        canvas_frame = tk.Frame(main, bg="#111018")
        canvas_frame.pack(fill="x")

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_width,
            height=self.visible_canvas_height,
            bg="#202020",
            highlightthickness=1,
            highlightbackground="#41334f",
            takefocus=True,
            yscrollincrement=20,
        )
        self.canvas.pack(side="left", fill="x", expand=False)

        self.scrollbar_y = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar_y.pack(side="left", fill="y")

        self.canvas.configure(
            yscrollcommand=self.scrollbar_y.set,
            scrollregion=(0, 0, self.canvas_width, self.canvas_height_total),
        )

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)
        self.canvas.bind("<Button-5>", self.on_mousewheel)

        panel = tk.Frame(main, bg="#1b1824")
        panel.pack(fill="x", pady=(12, 0))

        self.info_label = tk.Label(panel, text="", bg="#1b1824", fg="#f5eefe")
        self.info_label.grid(row=0, column=0, columnspan=14, sticky="w", padx=10, pady=8)

        tk.Label(panel, text="Slice", bg="#1b1824", fg="#b9acc8").grid(row=1, column=0, padx=5)

        self.pair_var = tk.IntVar(value=self.pair_values[0])
        self.pair_box = ttk.Combobox(panel, textvariable=self.pair_var, values=self.pair_values, width=8, state="readonly")
        self.pair_box.grid(row=1, column=1, padx=5)
        self.pair_box.bind("<<ComboboxSelected>>", lambda e: self.set_pair(int(self.pair_var.get())))

        tk.Label(panel, text="Ligne", bg="#1b1824", fg="#b9acc8").grid(row=1, column=2, padx=5)

        self.lane_var = tk.IntVar(value=1)
        self.lane_box = ttk.Combobox(panel, textvariable=self.lane_var, values=[i + 1 for i in range(len(self.pair_values))], width=6, state="readonly")
        self.lane_box.grid(row=1, column=3, padx=5)
        self.lane_box.bind("<<ComboboxSelected>>", self.set_lane_from_choice)

        tk.Label(panel, text="Step ms", bg="#1b1824", fg="#b9acc8").grid(row=1, column=4, padx=5)

        self.step_ms_var = tk.StringVar(value=str(self.default_step_ms))
        self.step_ms_spin = tk.Spinbox(panel, from_=10, to=2000, increment=1, textvariable=self.step_ms_var, width=8, bg="#30283f", fg="#f5eefe", insertbackground="#f5eefe")
        self.step_ms_spin.grid(row=1, column=5, padx=4)

        self.clip_var = tk.BooleanVar(value=False)
        tk.Checkbutton(panel, text="couper à la case", variable=self.clip_var, bg="#1b1824", fg="#f5eefe", selectcolor="#30283f", activebackground="#1b1824", activeforeground="#f5eefe").grid(row=1, column=6, padx=4)

        self.warp_var = tk.BooleanVar(value=True)
        tk.Checkbutton(panel, text="warp pitch-preserve", variable=self.warp_var, bg="#1b1824", fg="#f5eefe", selectcolor="#30283f", activebackground="#1b1824", activeforeground="#f5eefe").grid(row=1, column=7, padx=4)

        tk.Button(panel, text="Play slice", command=self.play_selected_pair, bg="#30283f", fg="#f5eefe").grid(row=1, column=8, padx=4)
        tk.Button(panel, text="Loop / Space", command=self.toggle_loop, bg="#30283f", fg="#f5eefe").grid(row=1, column=9, padx=4)
        tk.Button(panel, text="Render preview", command=self.render_preview_only, bg="#30283f", fg="#f5eefe").grid(row=1, column=10, padx=4)
        tk.Button(panel, text="Save validation", command=self.save, bg="#30513f", fg="#f5eefe").grid(row=1, column=11, padx=4)
        tk.Button(panel, text="Delete", command=self.delete_selected, bg="#4a2630", fg="#f5eefe").grid(row=1, column=12, padx=4)
        tk.Button(panel, text="Reset", command=self.reset, bg="#30283f", fg="#f5eefe").grid(row=1, column=13, padx=4)

        self.output_label = tk.Label(
            panel,
            text="v44 : tempo par défaut 155 BPM ; clic case = son immédiat ; fermeture = stop audio.",
            bg="#1b1824",
            fg="#77f5b5",
            justify="left",
        )
        self.output_label.grid(row=2, column=0, columnspan=14, sticky="w", padx=10, pady=8)

    def bind_keys(self):
        self.root.bind_all("<space>", self.toggle_loop_event)
        self.root.bind_all("<KeyPress-space>", self.toggle_loop_event)
        self.root.bind_all("<Delete>", lambda e: self.delete_selected())
        self.root.bind_all("<Left>", lambda e: self.move_selected(-HIT_SPACING_STEPS, 0))
        self.root.bind_all("<Right>", lambda e: self.move_selected(HIT_SPACING_STEPS, 0))
        self.root.bind_all("<Up>", lambda e: self.move_selected(0, -1))
        self.root.bind_all("<Down>", lambda e: self.move_selected(0, 1))

    def force_keyboard_focus(self):
        try:
            self.root.focus_force()
            self.canvas.focus_set()
        except Exception:
            pass

    def refresh_breaks(self):
        self.break_jsons = list_break_jsons()
        values = list(self.break_jsons.keys())
        self.break_box.configure(values=values)

        if values and self.break_var.get() not in values:
            self.break_var.set(values[0])

        self.output_label.config(text=f"Breaks détectés : {len(values)}")

    def switch_break(self, event=None):
        safe = self.break_var.get()

        if safe not in self.break_jsons:
            self.output_label.config(text=f"Break introuvable : {safe}")
            return

        self.stop_playhead()
        self.stop_audio()
        self.looping = False

        self.pair_json = self.break_jsons[safe]
        self.project = self.load_project(self.pair_json)

        self.blocks = self.project["blocks"]
        self.block_by_pair = {int(b["pair"]): b for b in self.blocks}
        self.pair_values = [int(b["pair"]) for b in self.blocks]
        self.pair_to_lane = {int(pair): i for i, pair in enumerate(self.pair_values)}
        self.source_audio_cache = {}
        self.warp_cache = {}

        self.default_step_ms = self.guess_step_ms()
        self.step_ms_var.set(str(self.default_step_ms))

        self.canvas_height_total = self.row_height * max(1, len(self.pair_values))
        self.step_count = LOCKED_GRID_STEPS
        self.canvas_width = self.left_width + self.step_width * self.step_count
        self.canvas.configure(
            width=self.canvas_width,
            scrollregion=(0, 0, self.canvas_width, self.canvas_height_total),
            height=min(620, max(220, self.canvas_height_total)),
        )

        self.pair_box.configure(values=self.pair_values)
        self.lane_box.configure(values=[i + 1 for i in range(len(self.pair_values))])

        self.pattern = self.build_initial_pattern(randomize=False)
        self.selected_id = self.pattern[0]["id"] if self.pattern else None

        self.draw()
        self.refresh_panel()
        self.write_latest_pattern(reason="switch_break")

        self.output_label.config(text=f"Break chargé : {safe}")

    def on_mousewheel(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(3, "units")
        else:
            direction = -1 if event.delta > 0 else 1
            self.canvas.yview_scroll(direction * 3, "units")

        return "break"

    def step_to_x(self, step):
        return self.left_width + float(step) * self.step_width

    def snap_step(self, step, length=None):
        """
        v44 : les hits font 2 cases.
        On force donc le placement sur les colonnes paires.
        """
        if length is None:
            length = HIT_LENGTH_STEPS

        step = int(round(float(step)))
        step = int(round(step / HIT_SPACING_STEPS) * HIT_SPACING_STEPS)

        return max(0, min(self.step_count - int(length), step))

    def x_to_step(self, x):
        if x < self.left_width:
            return 0

        raw = int((x - self.left_width) // self.step_width)
        return self.snap_step(raw, HIT_LENGTH_STEPS)

    def y_to_lane(self, y):
        return max(0, min(len(self.pair_values) - 1, int(y // self.row_height)))

    def selected(self):
        for item in self.pattern:
            if int(item["id"]) == int(self.selected_id):
                return item

        return None

    def new_id(self):
        return max([int(i["id"]) for i in self.pattern], default=-1) + 1

    def cell_snapshot(self, item):
        if item is None:
            return None

        lane_index = max(0, min(len(self.pair_values) - 1, safe_int(item.get("lane"))))
        pair = self.lane_to_pair(lane_index)

        return {
            "id": safe_int(item.get("id")),
            "x_step": safe_int(item.get("x_step")),
            "lane": lane_index,
            "pair": int(pair),
            "length": safe_int(item.get("length"), 1),
            "variation_bar": safe_int(item.get("x_step")) // 8,
            "variation_pos": safe_int(item.get("x_step")) % 8,
        }

    def selected_snapshot(self):
        return self.cell_snapshot(self.selected())

    def record_correction(self, event_type, before=None, after=None):
        if before == after:
            return

        CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        LEARNING_DIR.mkdir(parents=True, exist_ok=True)

        safe = self.project["safe"]

        event = {
            "version": "breakbeatai_slice_index_correction_v44",
            "time": now_stamp(),
            "safe": safe,
            "event_type": event_type,
            "before": before,
            "after": after,
            "step_ms": self.get_step_ms(),
            "target_bpm": TARGET_BPM,
            "target_step_ms": TARGET_STEP_MS,
        }

        corrections_path = CORRECTIONS_DIR / f"{safe}_slice_index_corrections_v44.jsonl"

        with corrections_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\\n")

        memory = load_memory()
        memory["events_total"] = int(memory.get("events_total", 0)) + 1

        if after is not None:
            break_bucket, global_bucket = memory_bucket(memory, safe, after.get("x_step", 0))
            inc_count(break_bucket["positive_pair_counts"], after.get("pair"))
            inc_count(global_bucket["positive_pair_counts"], after.get("pair"))
            break_bucket["events"] = int(break_bucket.get("events", 0)) + 1
            global_bucket["events"] = int(global_bucket.get("events", 0)) + 1

        if before is not None and after is not None:
            if str(before.get("pair")) != str(after.get("pair")):
                break_bucket, global_bucket = memory_bucket(memory, safe, before.get("x_step", 0))
                inc_count(break_bucket["negative_pair_counts"], before.get("pair"))
                inc_count(global_bucket["negative_pair_counts"], before.get("pair"))

        save_memory(memory)
        latest_path = self.write_latest_pattern(reason=event_type)

        self.output_label.config(
            text=(
                f"Auto-learn sauvegardé : {event_type}\\n"
                f"Correction : {corrections_path}\\n"
                f"Mémoire : {MEMORY_PATH}\\n"
                f"Latest : {latest_path}"
            )
        )

        print(f"[v140] correction sauvegardée : {event_type}")

    def draw(self):
        self.canvas.delete("all")
        self.canvas.configure(scrollregion=(0, 0, self.canvas_width, self.canvas_height_total))

        for lane_index, pair in enumerate(self.pair_values):
            y0 = lane_index * self.row_height
            y1 = y0 + self.row_height
            row_fill = "#252525" if lane_index % 2 == 0 else "#202020"

            self.canvas.create_rectangle(0, y0, self.canvas_width, y1, fill=row_fill, outline="#343434")
            self.canvas.create_rectangle(0, y0, self.left_width, y1, fill="#17131f", outline="#343044")

            box_size = 28
            box_x0 = 16
            box_y0 = y0 + (self.row_height - box_size) / 2
            box_x1 = box_x0 + box_size
            box_y1 = box_y0 + box_size

            self.canvas.create_rectangle(box_x0, box_y0, box_x1, box_y1, fill=self.pair_color(pair), outline="#f5eefe")
            self.canvas.create_text((box_x0 + box_x1) / 2, (box_y0 + box_y1) / 2, text=str(pair), fill="#1a0d14", font=("Sans", 11, "bold"))
            self.canvas.create_text(box_x1 + 34, (box_y0 + box_y1) / 2, text=f"slice {pair}", fill="#f5eefe", font=("Sans", 9, "bold"))

        for step in range(self.step_count):
            x0 = self.step_to_x(step)
            x1 = self.step_to_x(step + 1)
            self.canvas.create_text((x0 + x1) / 2, 10, text=str(step), fill="#f5eefe", font=("Sans", 8, "bold"))

        for step in range(self.step_count + 1):
            x = self.step_to_x(step)

            if step % 8 == 0:
                color = "#ff7acc"
                width = 3
            elif step % 4 == 0:
                color = "#6c6c6c"
                width = 2
            else:
                color = "#393939"
                width = 1

            self.canvas.create_line(x, 0, x, self.canvas_height_total, fill=color, width=width)

        self.canvas.create_line(self.left_width, 0, self.left_width, self.canvas_height_total, fill="#888888", width=2)

        for item in sorted(self.pattern, key=lambda e: (int(e["x_step"]), int(e["lane"]), int(e["id"]))):
            self.draw_case(item)

    def draw_case(self, item):
        lane_index = max(0, min(len(self.pair_values) - 1, int(item.get("lane", 0))))
        pair = self.lane_to_pair(lane_index)

        item["lane"] = lane_index
        item["pair"] = pair

        x0 = self.step_to_x(int(item["x_step"]))
        x1 = self.step_to_x(int(item["x_step"]) + int(item["length"]))

        y0 = lane_index * self.row_height + self.case_y_padding
        y1 = y0 + self.case_height

        outline = "#77f5b5" if int(item["id"]) == int(self.selected_id) else "#ffc0cf"
        width = 3 if int(item["id"]) == int(self.selected_id) else 1
        tags = ("case", f"id_{item['id']}")

        self.canvas.create_rectangle(x0, y0, x1, y1, fill=self.pair_color(pair), outline=outline, width=width, tags=tags)
        self.canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=str(pair), fill="#1a0d14", font=("Sans", 9, "bold"), tags=tags)

        if int(item["id"]) == int(self.selected_id):
            self.canvas.create_rectangle(x0, y0, x0 + 5, y1, fill="#77f5b5", outline="", tags=tags)
            self.canvas.create_rectangle(x1 - 5, y0, x1, y1, fill="#77f5b5", outline="", tags=tags)

    def get_item_at(self, x, y):
        found = self.canvas.find_overlapping(x, y, x, y)

        for obj in reversed(found):
            for tag in self.canvas.gettags(obj):
                if tag.startswith("id_"):
                    return int(tag.replace("id_", ""))

        return None

    def on_click(self, event):
        self.force_keyboard_focus()

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        item_id = self.get_item_at(x, y)

        if item_id is None:
            if x >= self.left_width:
                step = self.x_to_step(x)
                lane_index = self.y_to_lane(y)
                pair = self.lane_to_pair(lane_index)

                new_item = {
                    "id": self.new_id(),
                    "x_step": step,
                    "lane": lane_index,
                    "pair": pair,
                    "length": HIT_LENGTH_STEPS,
                    "variation_bar": step // 8,
                    "variation_pos": step % 8,
                }

                self.pattern.append(new_item)
                self.selected_id = new_item["id"]

                self.drag_mode = "move"
                self.drag_created_id = new_item["id"]
                self.drag_before = None
                self.drag_start_x = x
                self.drag_start_step = step
                self.drag_start_len = HIT_LENGTH_STEPS

                self.draw()
                self.refresh_panel()
                self.audition_selected_case()

            return

        self.selected_id = item_id
        item = self.selected()

        self.drag_created_id = None
        self.drag_before = self.selected_snapshot()

        x0 = self.step_to_x(item["x_step"])
        x1 = self.step_to_x(item["x_step"] + item["length"])

        if abs(x - x0) <= 8:
            self.drag_mode = "resize_left"
        elif abs(x - x1) <= 8:
            self.drag_mode = "resize_right"
        else:
            self.drag_mode = "move"

        self.drag_start_x = x
        self.drag_start_step = item["x_step"]
        self.drag_start_len = item["length"]

        self.draw()
        self.refresh_panel()
        self.audition_selected_case()

    def on_drag(self, event):
        item = self.selected()

        if item is None or self.drag_mode is None:
            return

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        delta_steps = round((x - self.drag_start_x) / self.step_width)

        if self.drag_mode == "move":
            item["x_step"] = self.snap_step(self.drag_start_step + delta_steps, item["length"])
            lane_index = self.y_to_lane(y)
            item["lane"] = lane_index
            item["pair"] = self.lane_to_pair(lane_index)
            item["variation_bar"] = item["x_step"] // 8
            item["variation_pos"] = item["x_step"] % 8

        elif self.drag_mode == "resize_right":
            new_len = self.drag_start_len + delta_steps
            new_len = max(HIT_LENGTH_STEPS, min(self.max_case_length, new_len))
            item["length"] = min(self.step_count - item["x_step"], new_len)

        elif self.drag_mode == "resize_left":
            old_end = self.drag_start_step + self.drag_start_len
            new_start = self.drag_start_step + delta_steps
            new_start = max(0, min(old_end - 1, new_start))
            new_len = old_end - new_start

            if new_len > self.max_case_length:
                new_start = old_end - self.max_case_length
                new_len = self.max_case_length

            item["x_step"] = self.snap_step(new_start, new_len)
            item["length"] = max(HIT_LENGTH_STEPS, new_len)
            item["variation_bar"] = item["x_step"] // 8
            item["variation_pos"] = item["x_step"] % 8

        self.draw()
        self.refresh_panel()

    def on_release(self, event):
        before = self.drag_before
        created_id = self.drag_created_id

        self.drag_mode = None

        after = self.selected_snapshot()

        if created_id is not None:
            self.record_correction("add_case", before=None, after=after)
        elif before != after:
            self.record_correction("drag_or_resize_case", before=before, after=after)

        self.drag_before = None
        self.drag_created_id = None

    def refresh_panel(self):
        item = self.selected()

        if item is None:
            self.info_label.config(text="Aucune case sélectionnée")
            return

        lane_index = max(0, min(len(self.pair_values) - 1, int(item.get("lane", 0))))
        pair = self.lane_to_pair(lane_index)

        item["lane"] = lane_index
        item["pair"] = pair

        block = self.block_by_pair[pair]
        duration = block.get("duration_ms", 0.0)
        cell_ms = int(item["length"]) * self.get_step_ms()

        if block["storage"] == "index_only":
            src = f"{block.get('source_start_ms')}→{block.get('source_end_ms')} ms"
        else:
            src = str(block.get("audio_path"))

        self.info_label.config(
            text=(
                f"case {item['id']} | slice {pair} | "
                f"bar {int(item['x_step']) // 8 + 1}/4 pos {int(item['x_step']) % 8} | "
                f"ligne {lane_index + 1}/{len(self.pair_values)} | "
                f"step {item['x_step']} | length {item['length']} | "
                f"cellule {cell_ms:.1f} ms | slice source {duration:.1f} ms | {src}"
            )
        )

        self.pair_var.set(pair)
        self.lane_var.set(lane_index + 1)

    def set_pair(self, pair):
        item = self.selected()

        if item is None:
            return

        before = self.selected_snapshot()

        pair = int(pair)
        lane = self.pair_to_lane.get(pair, 0)

        item["pair"] = pair
        item["lane"] = lane

        self.draw()
        self.refresh_panel()

        after = self.selected_snapshot()
        self.record_correction("change_pair", before=before, after=after)

    def set_lane_from_choice(self, event=None):
        item = self.selected()

        if item is None:
            return

        before = self.selected_snapshot()

        try:
            lane_index = int(self.lane_var.get()) - 1
        except Exception:
            lane_index = 0

        lane_index = max(0, min(len(self.pair_values) - 1, lane_index))

        item["lane"] = lane_index
        item["pair"] = self.lane_to_pair(lane_index)

        self.draw()
        self.refresh_panel()

        after = self.selected_snapshot()
        self.record_correction("change_lane", before=before, after=after)

    def move_selected(self, dx, dy):
        item = self.selected()

        if item is None:
            return

        before = self.selected_snapshot()

        if dx:
            item["x_step"] = self.snap_step(item["x_step"] + dx, item["length"])
            item["variation_bar"] = item["x_step"] // 8
            item["variation_pos"] = item["x_step"] % 8

        if dy:
            lane_index = max(0, min(len(self.pair_values) - 1, int(item.get("lane", 0)) + dy))
            item["lane"] = lane_index
            item["pair"] = self.lane_to_pair(lane_index)

        self.draw()
        self.refresh_panel()

        after = self.selected_snapshot()
        self.record_correction("keyboard_move_case", before=before, after=after)

    def delete_selected(self):
        before = self.selected_snapshot()

        if self.selected_id is None:
            return

        self.pattern = [i for i in self.pattern if int(i["id"]) != int(self.selected_id)]
        self.selected_id = self.pattern[0]["id"] if self.pattern else None

        self.draw()
        self.refresh_panel()

        if before is not None:
            self.record_correction("delete_case", before=before, after=None)

    def reset(self):
        self.stop_playhead()
        self.stop_audio()
        self.looping = False

        self.pattern = self.build_initial_pattern(randomize=False)
        self.selected_id = self.pattern[0]["id"] if self.pattern else None

        self.draw()
        self.refresh_panel()
        self.write_latest_pattern(reason="reset")
        self.output_label.config(text="Reset : loop64 slice-index regénérée.")

    def randomize_pattern(self):
        self.stop_playhead()
        self.stop_audio()
        self.looping = False

        self.pattern = self.build_initial_pattern(randomize=True)
        self.selected_id = self.pattern[0]["id"] if self.pattern else None

        self.draw()
        self.refresh_panel()

        latest = self.write_latest_pattern(reason="randomize_loop64")
        self.output_label.config(text=f"Randomize OK. Mémoire appliquée.\\nLatest : {latest}")

    def stop_audition(self):
        """
        v44 : stoppe seulement le petit player d'écoute au clic,
        sans forcément couper la loop principale.
        """
        if getattr(self, "audition_process", None) is not None:
            proc = self.audition_process

            try:
                if proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except Exception:
                        proc.terminate()

                    time.sleep(0.02)

                    if proc.poll() is None:
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except Exception:
                            proc.kill()
            except Exception:
                pass

            self.audition_process = None

    def play_audition_array(self, audio, label="audition"):
        """
        v44 : lecture courte au clic.
        On écrit juste un petit WAV temporaire de preview, puis pw-play le lit.
        Ça ne crée pas de slices permanentes : les slices restent index-only.
        """
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        audio = normalize(audio)
        audition_wav = OUT_DIR / f"{self.project['safe']}_v140_audition.wav"
        sf.write(audition_wav, audio, SR)

        self.stop_audition()

        duration_ms = int(len(audio) / SR * 1000)

        if self.external_player:
            try:
                self.audition_process = subprocess.Popen(
                    self.external_command(audition_wav),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                self.output_label.config(text=f"{label} joué au clic — {duration_ms} ms")
                return duration_ms
            except Exception as exc:
                print(f"[v140] Erreur audition {self.external_player} : {exc}")

        try:
            import sounddevice as sd
            sd.play(audio, SR)
            self.output_label.config(text=f"{label} joué au clic — {duration_ms} ms")
            return duration_ms
        except Exception as exc:
            messagebox.showwarning("Audio", f"Erreur audio audition : {exc}")
            return 0

    def audition_selected_case(self):
        """
        v44 : clic sur une case = on joue la slice de cette case.
        """
        item = self.selected()

        if item is None:
            return

        lane_index = max(0, min(len(self.pair_values) - 1, int(item.get("lane", 0))))
        pair = self.lane_to_pair(lane_index)

        item["pair"] = pair
        item["lane"] = lane_index

        audio = self.get_audio(pair)

        if hasattr(self, "warp_var") and self.warp_var.get():
            target_len = max(1, int(item.get("length", HIT_LENGTH_STEPS)) * self.get_step_samples())
            audio = self.warp_slice_audio(item["pair"], audio, target_len)

        self.play_audition_array(audio, label=f"Case {item['id']} / slice {pair}")

    def shutdown_audio(self):
        """
        v44 : arrêt dur de l'audio.
        Important parce que pw-play/paplay/aplay/ffplay peuvent continuer
        après fermeture de la fenêtre si on ne les tue pas explicitement.
        """
        self.stop_audition()

        if self.play_process is not None:
            proc = self.play_process

            try:
                if proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except Exception:
                        proc.terminate()

                    time.sleep(0.05)

                    if proc.poll() is None:
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except Exception:
                            proc.kill()
            except Exception:
                pass

            self.play_process = None

        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

    def stop_audio(self):
        self.shutdown_audio()

    def on_close(self):
        """
        v44 : quand tu fermes la fenêtre, on coupe la loop, le playhead
        et le player externe avant de détruire Tk.
        """
        print("[v140] Fermeture : arrêt audio + destruction fenêtre")

        self.looping = False

        if self.loop_after_id:
            try:
                self.root.after_cancel(self.loop_after_id)
            except Exception:
                pass

            self.loop_after_id = None

        self.stop_playhead()
        self.shutdown_audio()

        try:
            self.root.destroy()
        except Exception:
            pass

    def external_command(self, wav_path):
        if self.external_player == "pw-play":
            return ["pw-play", str(wav_path)]
        if self.external_player == "paplay":
            return ["paplay", str(wav_path)]
        if self.external_player == "aplay":
            return ["aplay", "-q", str(wav_path)]
        if self.external_player == "ffplay":
            return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(wav_path)]

        return None

    def play_audio_array(self, audio, label="audio"):
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        audio = normalize(audio)
        live_wav = OUT_DIR / f"{self.project['safe']}_v140_live.wav"
        sf.write(live_wav, audio, SR)

        self.stop_audio()

        duration_ms = int(len(audio) / SR * 1000)

        if self.external_player:
            try:
                self.play_process = subprocess.Popen(
                    self.external_command(live_wav),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                self.output_label.config(text=f"{label} lancé avec {self.external_player} — {duration_ms} ms")
                return duration_ms
            except Exception as exc:
                print(f"[v140] Erreur backend {self.external_player} : {exc}")

        try:
            import sounddevice as sd
            sd.play(audio, SR)
            return duration_ms
        except Exception as exc:
            messagebox.showwarning("Audio", f"Erreur audio : {exc}")
            return 0

    def warp_slice_audio(self, pair, audio, target_len):
        """
        v44 : cache par slice + durée cible.
        Évite de recalculer rubberband/ffmpeg à chaque lecture.
        """
        pair = int(pair)
        target_len = int(target_len)
        key = (self.project["safe"], pair, target_len, "pitch_preserve_v44")

        if key in self.warp_cache:
            return self.warp_cache[key].copy()

        cache_key = f"{self.project['safe']}_slice{pair}_len{target_len}"
        warped = warp_audio_to_length_external(audio, target_len, cache_key=cache_key)

        self.warp_cache[key] = warped.astype(np.float32)

        return warped.copy()

    def play_selected_pair(self):
        item = self.selected()

        if item is None:
            return

        audio = self.get_audio(item["pair"])

        if hasattr(self, "warp_var") and self.warp_var.get():
            target_len = max(1, int(item.get("length", HIT_LENGTH_STEPS)) * self.get_step_samples())
            audio = self.warp_slice_audio(item["pair"], audio, target_len)

        self.play_audio_array(audio, label=f"Slice {item['pair']}")

    def render_audio_with_timeline(self):
        step_samples = self.get_step_samples()
        loop_samples = self.get_loop_samples()

        out = np.zeros(loop_samples, dtype=np.float32)
        timeline = []

        for item in sorted(self.pattern, key=lambda e: (int(e["x_step"]), int(e["id"]))):
            lane_index = max(0, min(len(self.pair_values) - 1, int(item.get("lane", 0))))
            pair = self.lane_to_pair(lane_index)
            item["pair"] = pair

            start = int(item["x_step"]) * step_samples
            max_len = max(1, int(item["length"]) * step_samples)

            if start >= loop_samples:
                continue

            audio = self.get_audio(pair)

            # v44 :
            # warp pitch-preserve = garde la vitesse originale.
            # Si la slice est trop longue : coupe.
            # Si elle est trop courte : ajoute du silence.
            if hasattr(self, "warp_var") and self.warp_var.get():
                audio = self.warp_slice_audio(pair, audio, max_len)
            elif self.clip_var.get():
                audio = audio[:max_len]

            end = min(loop_samples, start + len(audio))
            take = max(0, end - start)

            if take <= 0:
                continue

            out[start:end] += audio[:take]

            timeline.append({
                "id": int(item["id"]),
                "pair": int(pair),
                "x_step": int(item["x_step"]),
                "length": int(item["length"]),
                "lane": int(lane_index),
                "start_sec": start / SR,
                "end_sec": end / SR,
            })

        return normalize(out), timeline

    def render_audio(self):
        audio, _timeline = self.render_audio_with_timeline()
        return audio

    def render_preview_file(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        wav = OUT_DIR / f"{self.project['safe']}_tracker_app_v140_preview.wav"
        sf.write(wav, self.render_audio(), SR)
        return wav

    def render_preview_only(self):
        wav = self.render_preview_file()
        self.output_label.config(text=f"Preview : {wav}")

    def build_gapless_loop_buffer(self):
        one_loop, _timeline = self.render_audio_with_timeline()

        if len(one_loop) <= 1:
            return one_loop, 1, 0, 0.0

        one_loop_sec = len(one_loop) / SR
        repeats = int(np.ceil(self.loop_target_sec / one_loop_sec))
        repeats = max(2, min(self.loop_max_repeats, repeats))

        long_loop = np.tile(one_loop, repeats).astype(np.float32)

        return long_loop, repeats, int(one_loop_sec * 1000), one_loop_sec

    def draw_playhead(self, x):
        self.canvas.delete("playhead")
        x = max(self.left_width, min(self.canvas_width, x))

        self.canvas.create_line(x, 0, x, self.canvas_height_total, fill="#77f5b5", width=3, tags=("playhead",))
        self.canvas.create_polygon(x - 7, 0, x + 7, 0, x, 13, fill="#77f5b5", outline="", tags=("playhead",))

    def update_playhead(self):
        if not self.looping:
            self.canvas.delete("playhead")
            return

        if self.loop_started_at is None or self.current_loop_sec <= 0:
            return

        elapsed = (time.monotonic() - self.loop_started_at) % self.current_loop_sec
        step_pos = elapsed / max(1e-9, self.get_step_ms() / 1000.0)
        x = self.step_to_x(step_pos)

        self.draw_playhead(x)
        self.playhead_after_id = self.root.after(33, self.update_playhead)

    def start_playhead(self, loop_sec):
        self.stop_playhead(clear=False)
        self.current_loop_sec = float(loop_sec)
        self.loop_started_at = time.monotonic()
        self.update_playhead()

    def stop_playhead(self, clear=True):
        if self.playhead_after_id is not None:
            try:
                self.root.after_cancel(self.playhead_after_id)
            except Exception:
                pass

        self.playhead_after_id = None
        self.loop_started_at = None

        if clear:
            self.canvas.delete("playhead")

    def loop_tick(self):
        if not self.looping:
            return

        audio, repeats, one_loop_ms, one_loop_sec = self.build_gapless_loop_buffer()
        duration_ms = self.play_audio_array(audio, label=f"Loop gapless x{repeats}")

        if duration_ms <= 0:
            self.looping = False
            self.stop_playhead()
            return

        self.start_playhead(one_loop_sec)
        self.output_label.config(
            text=f"Loop : motif {one_loop_ms} ms x{repeats}. Step={self.get_step_ms():.2f} ms."
        )
        self.loop_after_id = self.root.after(max(1000, duration_ms + 20), self.loop_tick)

    def toggle_loop_event(self, event=None):
        print("[v140] Espace détecté")
        self.toggle_loop()
        return "break"

    def toggle_loop(self):
        if self.looping:
            self.looping = False

            if self.loop_after_id:
                try:
                    self.root.after_cancel(self.loop_after_id)
                except Exception:
                    pass

                self.loop_after_id = None

            self.stop_playhead()
            self.stop_audio()
            self.output_label.config(text="Loop arrêtée.")
            return

        self.stop_playhead()
        self.looping = True
        self.output_label.config(text="Lancement loop...")
        self.loop_tick()

    def clean_pattern(self):
        out = []

        for item in sorted(self.pattern, key=lambda e: (int(e["x_step"]), int(e["id"]))):
            lane_index = max(0, min(len(self.pair_values) - 1, int(item.get("lane", 0))))
            pair = self.lane_to_pair(lane_index)
            block = self.block_by_pair[pair]

            out.append({
                "id": int(item["id"]),
                "x_step": int(item["x_step"]),
                "lane": int(lane_index),
                "pair": int(pair),
                "slice_label": f"slice {pair}",
                "length": int(item["length"]),
                "variation_bar": int(item["x_step"]) // 8,
                "variation_pos": int(item["x_step"]) % 8,
                "slice_storage": block["storage"],
                "source_audio": block.get("source_audio"),
                "source_start_sample": block.get("source_start_sample"),
                "source_end_sample": block.get("source_end_sample"),
                "timeline_start_sec": round(int(item["x_step"]) * self.get_step_samples() / SR, 6),
                "cell_duration_sec": round(int(item["length"]) * self.get_step_samples() / SR, 6),
            })

        return out

    def write_latest_pattern(self, reason="correction"):
        LATEST_DIR.mkdir(parents=True, exist_ok=True)

        path = LATEST_DIR / f"{self.project['safe']}_latest_slice_index_v140.json"

        data = {
            "version": "breakbeatai_latest_slice_index_v140",
            "reason": reason,
            "updated_at": now_stamp(),
            "safe": self.project["safe"],
            "source_pair_json": self.project.get("source_pair_json"),
            "source_audio": self.project.get("source_audio"),
            "step_count": self.step_count,
            "locked_bpm": TARGET_BPM,
            "locked_step_ms": TARGET_STEP_MS,
            "locked_grid_steps": LOCKED_GRID_STEPS,
            "locked_hit_slots": LOCKED_HIT_SLOTS,
            "hit_length_steps": HIT_LENGTH_STEPS,
            "hit_spacing_steps": HIT_SPACING_STEPS,
            "hit_slots": HIT_SLOTS,
            "step_ms": self.get_step_ms(),
            "target_bpm": TARGET_BPM,
            "target_step_ms": TARGET_STEP_MS,
            "storage": "index_only_preferred",
            "warp_exact": bool(getattr(self, "warp_var", tk.BooleanVar(value=True)).get()),
            "warp_mode": "pitch_preserve_rubberband_or_ffmpeg_v44",
            "pattern": self.clean_pattern(),
        }

        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def save(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        VALIDATED_DIR.mkdir(parents=True, exist_ok=True)

        wav = self.render_preview_file()

        tracker_path = OUT_DIR / f"{self.project['safe']}_tracker_app_edit_v140.json"

        data = {
            "version": "tracker_app_edit_v44_slice_index",
            "storage": "index_only_single_source_audio",
            "warp_exact": bool(getattr(self, "warp_var", tk.BooleanVar(value=True)).get()),
            "warp_mode": "pitch_preserve_rubberband_or_ffmpeg_v44",
            "learning_memory": str(MEMORY_PATH),
            "step_ms": self.get_step_ms(),
            "target_bpm": TARGET_BPM,
            "target_step_ms": TARGET_STEP_MS,
            "step_count": self.step_count,
            "locked_bpm": TARGET_BPM,
            "locked_step_ms": TARGET_STEP_MS,
            "locked_grid_steps": LOCKED_GRID_STEPS,
            "locked_hit_slots": LOCKED_HIT_SLOTS,
            "hit_length_steps": HIT_LENGTH_STEPS,
            "hit_spacing_steps": HIT_SPACING_STEPS,
            "hit_slots": HIT_SLOTS,
            "source_pair_json": self.project["source_pair_json"],
            "source_audio": self.project["source_audio"],
            "safe": self.project["safe"],
            "pattern": self.clean_pattern(),
            "preview_wav": str(wav),
        }

        tracker_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        stamp = file_stamp()
        validated_path = VALIDATED_DIR / f"{self.project['safe']}_validated_slice_index_loop64_v140_{stamp}.json"

        validated = {
            "version": "breakbeatai_validated_slice_index_loop64_v140",
            "purpose": "human validated loop64 using source audio + slice indexes",
            "source_tracker_json": str(tracker_path),
            "source_audio": self.project["source_audio"],
            "safe": self.project["safe"],
            "step_count": self.step_count,
            "locked_bpm": TARGET_BPM,
            "locked_step_ms": TARGET_STEP_MS,
            "locked_grid_steps": LOCKED_GRID_STEPS,
            "locked_hit_slots": LOCKED_HIT_SLOTS,
            "hit_length_steps": HIT_LENGTH_STEPS,
            "hit_spacing_steps": HIT_SPACING_STEPS,
            "hit_slots": HIT_SLOTS,
            "step_ms": self.get_step_ms(),
            "target_bpm": TARGET_BPM,
            "target_step_ms": TARGET_STEP_MS,
            "pattern": self.clean_pattern(),
            "preview_wav": str(wav),
        }

        validated_path.write_text(json.dumps(validated, indent=2, ensure_ascii=False), encoding="utf-8")
        latest = self.write_latest_pattern(reason="save_validation")

        self.output_label.config(
            text=(
                f"OK sauvegardé\\n"
                f"Tracker : {tracker_path}\\n"
                f"Validation IA : {validated_path}\\n"
                f"Latest : {latest}\\n"
                f"Preview : {wav}"
            )
        )



# ---------------------------------------------------------------------
# v44 EXTENSION : statut compact, sans casser la v39
# ---------------------------------------------------------------------

def v44_set_status(self, text):
    """
    Statut compact :
    - remplace les retours ligne par " | "
    - coupe les chemins trop longs
    - évite que le panel pousse les boutons hors du cadre
    """
    text = str(text).replace("\n", " | ").replace("\\n", " | ")

    max_len = 180
    if len(text) > max_len:
        text = text[:max_len - 1] + "…"

    try:
        self.output_label.config(text=text)
    except Exception:
        pass


_old_v44_build_ui = SliceIndexTracker.build_ui


def v44_build_ui(self):
    _old_v44_build_ui(self)

    try:
        self.output_label.config(
            text="v44 : grille verrouillée 32 cases / 16 hits max / 155 BPM réel.",
            width=130,
            height=2,
            wraplength=1180,
            anchor="w",
            justify="left",
        )
    except Exception:
        pass


def v44_record_correction(self, event_type, before=None, after=None):
    """
    Même apprentissage que v39, mais message UI compact.
    Les chemins complets restent dans le terminal et dans dataset/learning/.
    """
    if before == after:
        return

    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    LEARNING_DIR.mkdir(parents=True, exist_ok=True)

    safe = self.project["safe"]

    event = {
        "version": "breakbeatai_slice_index_correction_v44",
        "time": now_stamp(),
        "safe": safe,
        "event_type": event_type,
        "before": before,
        "after": after,
        "step_ms": self.get_step_ms(),
    }

    corrections_path = CORRECTIONS_DIR / f"{safe}_slice_index_corrections_v44.jsonl"

    with corrections_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    memory = load_memory()
    memory["events_total"] = int(memory.get("events_total", 0)) + 1

    if after is not None:
        break_bucket, global_bucket = memory_bucket(memory, safe, after.get("x_step", 0))
        inc_count(break_bucket["positive_pair_counts"], after.get("pair"))
        inc_count(global_bucket["positive_pair_counts"], after.get("pair"))
        break_bucket["events"] = int(break_bucket.get("events", 0)) + 1
        global_bucket["events"] = int(global_bucket.get("events", 0)) + 1

    if before is not None and after is not None:
        if str(before.get("pair")) != str(after.get("pair")):
            break_bucket, global_bucket = memory_bucket(memory, safe, before.get("x_step", 0))
            inc_count(break_bucket["negative_pair_counts"], before.get("pair"))
            inc_count(global_bucket["negative_pair_counts"], before.get("pair"))

    save_memory(memory)
    latest_path = self.write_latest_pattern(reason=event_type)

    self.set_status(
        f"Auto-learn sauvegardé : {event_type} | correction #{memory.get('events_total', 0)} | dataset/learning/"
    )

    print(f"[v140] correction sauvegardée : {event_type}")
    print(f"[v140] correction path : {corrections_path}")
    print(f"[v140] latest path : {latest_path}")


def v44_randomize_pattern(self):
    self.stop_playhead()
    self.stop_audio()
    self.looping = False

    self.pattern = self.build_initial_pattern(randomize=True)
    self.selected_id = self.pattern[0]["id"] if self.pattern else None

    self.draw()
    self.refresh_panel()

    self.write_latest_pattern(reason="randomize_loop64")
    self.set_status("Randomize OK | mémoire appliquée | latest sauvegardé")


def v44_render_preview_only(self):
    wav = self.render_preview_file()
    self.set_status(f"Preview rendue : {wav.name}")


def v44_switch_break(self, event=None):
    safe = self.break_var.get()

    if safe not in self.break_jsons:
        self.set_status(f"Break introuvable : {safe}")
        return

    self.stop_playhead()
    self.stop_audio()
    self.looping = False

    self.pair_json = self.break_jsons[safe]
    self.project = self.load_project(self.pair_json)

    self.blocks = self.project["blocks"]
    self.block_by_pair = {int(b["pair"]): b for b in self.blocks}
    self.pair_values = [int(b["pair"]) for b in self.blocks]
    self.pair_to_lane = {int(pair): i for i, pair in enumerate(self.pair_values)}
    self.source_audio_cache = {}

    self.default_step_ms = self.guess_step_ms()
    self.step_ms_var.set(str(self.default_step_ms))

    self.canvas_height_total = self.row_height * max(1, len(self.pair_values))
    self.canvas.configure(
        scrollregion=(0, 0, self.canvas_width, self.canvas_height_total),
        height=min(620, max(220, self.canvas_height_total)),
    )

    self.pair_box.configure(values=self.pair_values)
    self.lane_box.configure(values=[i + 1 for i in range(len(self.pair_values))])

    self.pattern = self.build_initial_pattern(randomize=False)
    self.selected_id = self.pattern[0]["id"] if self.pattern else None

    self.draw()
    self.refresh_panel()
    self.write_latest_pattern(reason="switch_break")

    self.set_status(f"Break chargé : {safe}")


def v44_refresh_breaks(self):
    self.break_jsons = list_break_jsons()
    values = list(self.break_jsons.keys())
    self.break_box.configure(values=values)

    if values and self.break_var.get() not in values:
        self.break_var.set(values[0])

    self.set_status(f"Breaks détectés : {len(values)}")


# Monkey patch final.
SliceIndexTracker.set_status = v44_set_status
SliceIndexTracker.build_ui = v44_build_ui
SliceIndexTracker.record_correction = v44_record_correction
SliceIndexTracker.randomize_pattern = v44_randomize_pattern
SliceIndexTracker.render_preview_only = v44_render_preview_only
SliceIndexTracker.switch_break = v44_switch_break
SliceIndexTracker.refresh_breaks = v44_refresh_breaks



# ---------------------------------------------------------------------
# v45 EXTENSION : flèches haut/bas = joue la slice d'arrivée
# ---------------------------------------------------------------------

_old_v45_move_selected = SliceIndexTracker.move_selected


def v45_move_selected(self, dx, dy):
    before = self.selected_snapshot()

    result = _old_v45_move_selected(self, dx, dy)

    after = self.selected_snapshot()

    # Seulement quand on change de ligne avec haut/bas.
    # Gauche/droite ne rejoue pas, pour éviter de spammer pendant le placement rythmique.
    if dy != 0 and before != after:
        try:
            self.audition_selected_case()
        except Exception as exc:
            print(f"[v140] audition après flèche haut/bas impossible : {exc}")

    return result


_old_v45_build_ui = SliceIndexTracker.build_ui


def v45_build_ui(self):
    _old_v45_build_ui(self)

    try:
        self.set_status(
            "v45 : flèches haut/bas = play de la slice d’arrivée | grille 155 BPM."
        )
    except Exception:
        try:
            self.output_label.config(
                text="v45 : flèches haut/bas = play de la slice d’arrivée | grille 155 BPM."
            )
        except Exception:
            pass


SliceIndexTracker.move_selected = v45_move_selected
SliceIndexTracker.build_ui = v45_build_ui



# ---------------------------------------------------------------------
# v46 EXTENSION : logique M8 + warp global 155 BPM
# ---------------------------------------------------------------------

LOCKED_GRID_STEPS = 64
LOCKED_HIT_SLOTS = 16
LOCKED_TARGET_BPM = 155.0
LOCKED_STEP_MS = 60000.0 / (LOCKED_TARGET_BPM * 4.0)
LOCKED_LOOP_SAMPLES = int(round((LOCKED_GRID_STEPS * LOCKED_STEP_MS / 1000.0) * SR))


def v46_fit_audio_no_speed(audio, target_len):
    audio = np.asarray(audio, dtype=np.float32)
    target_len = int(target_len)

    if target_len <= 1:
        return audio[:1].copy() if len(audio) else np.zeros(1, dtype=np.float32)

    if len(audio) == 0:
        return np.zeros(target_len, dtype=np.float32)

    if len(audio) > target_len:
        y = audio[:target_len].copy()
        fade_len = min(64, target_len // 8)
        if fade_len > 2:
            y[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        return y.astype(np.float32)

    if len(audio) < target_len:
        y = np.zeros(target_len, dtype=np.float32)
        y[:len(audio)] = audio
        return y.astype(np.float32)

    return audio.copy().astype(np.float32)


def v46_make_atempo_chain(tempo):
    tempo = float(tempo)
    factors = []

    while tempo > 2.0:
        factors.append(2.0)
        tempo /= 2.0

    while tempo < 0.5:
        factors.append(0.5)
        tempo /= 0.5

    factors.append(tempo)

    return ",".join(f"atempo={f:.8f}" for f in factors)


def v46_warp_loop_pitch_preserve(audio, target_len, cache_key):
    """
    Warp GLOBAL de la boucle, pas warp par slice.
    C'est le point central de la v46.

    1. On prend la boucle source complète.
    2. On la stretch vers 32 cases à 155 BPM.
    3. Les slices sont extraites APRÈS ce warp global.
    """
    audio = np.asarray(audio, dtype=np.float32)
    target_len = int(target_len)

    if len(audio) <= 1:
        return np.zeros(target_len, dtype=np.float32)

    if abs(len(audio) - target_len) <= 2:
        return v46_fit_audio_no_speed(audio, target_len)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = OUT_DIR / "_global_warp_cache_v46"
    cache_dir.mkdir(parents=True, exist_ok=True)

    in_wav = cache_dir / f"{cache_key}_in.wav"
    out_wav = cache_dir / f"{cache_key}_out.wav"

    sf.write(in_wav, normalize(audio), SR)

    rubberband = shutil.which("rubberband")
    ffmpeg = shutil.which("ffmpeg")

    ratio = float(target_len) / float(len(audio))

    if rubberband:
        try:
            subprocess.run(
                [
                    rubberband,
                    "-q",
                    "-t",
                    f"{ratio:.8f}",
                    str(in_wav),
                    str(out_wav),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            y, sr = sf.read(out_wav, always_2d=False)
            if getattr(y, "ndim", 1) > 1:
                y = y.mean(axis=1)

            y = resample_linear(y.astype(np.float32), sr, SR)
            return v46_fit_audio_no_speed(y, target_len)

        except Exception as exc:
            print(f"[v140] rubberband global warp échec : {exc}")

    if ffmpeg:
        try:
            tempo = float(len(audio)) / float(target_len)
            atempo = v46_make_atempo_chain(tempo)

            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(in_wav),
                    "-filter:a",
                    atempo,
                    str(out_wav),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            y, sr = sf.read(out_wav, always_2d=False)
            if getattr(y, "ndim", 1) > 1:
                y = y.mean(axis=1)

            y = resample_linear(y.astype(np.float32), sr, SR)
            return v46_fit_audio_no_speed(y, target_len)

        except Exception as exc:
            print(f"[v140] ffmpeg global warp échec : {exc}")

    print("[v140] Aucun warp pitch-preserve dispo, fallback fit no-speed.")
    return v46_fit_audio_no_speed(audio, target_len)


_old_v46_guess_step_ms = SliceIndexTracker.guess_step_ms


def v46_guess_step_ms(self):
    return round(LOCKED_STEP_MS, 4)


_old_v46_build_initial_pattern = SliceIndexTracker.build_initial_pattern


def v46_build_initial_pattern(self, randomize=False):
    rng = np.random.default_rng(int(time.time_ns() % (2**32))) if randomize else None

    self.step_count = LOCKED_GRID_STEPS
    pattern = []

    max_items = min(len(self.pair_values), LOCKED_HIT_SLOTS)

    print(
        f"[v140] M8 pattern | break={self.project['safe']} | "
        f"slices={len(self.pair_values)} | slots={LOCKED_HIT_SLOTS} | bpm={LOCKED_TARGET_BPM}"
    )

    for slot in range(max_items):
        x_step = slot * HIT_SPACING_STEPS

        if randomize:
            pair = self.pick_pair_for_pos(slot, randomize=True, rng=rng)
        else:
            pair = int(self.pair_values[slot])

        lane = self.pair_to_lane.get(pair, 0)

        pattern.append({
            "id": len(pattern),
            "x_step": x_step,
            "lane": lane,
            "pair": pair,
            "length": HIT_LENGTH_STEPS,
            "variation_bar": x_step // 8,
            "variation_pos": x_step % 8,
            "hit_slot": slot,
            "randomized": bool(randomize),
        })

        print(f"[v140] slot {slot:02d} | step {x_step:02d} -> slice {pair:02d}")

    return pattern


def v46_get_raw_source_loop(self):
    source_audio = self.project.get("source_audio")
    if not source_audio:
        raise RuntimeError("source_audio manquant dans le JSON")

    full = self.get_source_audio(source_audio)

    meta_path = Path(self.project["source_pair_json"])
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    a = int(meta.get("loop_start_sample", 0))
    b = int(meta.get("loop_end_sample", len(full)))

    a = max(0, min(len(full) - 1, a))
    b = max(a + 1, min(len(full), b))

    return full[a:b].astype(np.float32), a, b


def v46_get_warped_loop(self):
    if not hasattr(self, "_v46_global_warp_cache"):
        self._v46_global_warp_cache = {}

    key = (
        self.project["safe"],
        str(self.project.get("source_pair_json")),
        LOCKED_TARGET_BPM,
        LOCKED_LOOP_SAMPLES,
        "global_pitch_preserve_v46",
    )

    if key in self._v46_global_warp_cache:
        return self._v46_global_warp_cache[key]

    raw_loop, loop_start, loop_end = v46_get_raw_source_loop(self)

    cache_key = f"{self.project['safe']}_global155_{loop_start}_{loop_end}_{LOCKED_LOOP_SAMPLES}"

    warped = v46_warp_loop_pitch_preserve(
        raw_loop,
        LOCKED_LOOP_SAMPLES,
        cache_key=cache_key,
    )

    self._v46_global_warp_cache[key] = {
        "audio": warped.astype(np.float32),
        "source_loop_start": int(loop_start),
        "source_loop_end": int(loop_end),
        "source_loop_len": int(loop_end - loop_start),
        "warped_len": int(len(warped)),
    }

    print(
        f"[v140] global warp OK | source_len={loop_end-loop_start} | "
        f"target_len={len(warped)} | bpm={LOCKED_TARGET_BPM}"
    )

    return self._v46_global_warp_cache[key]


_old_v46_get_audio = SliceIndexTracker.get_audio


def v46_get_audio(self, pair):
    """
    v46 :
    - si global warp ON : on extrait les slices depuis la boucle déjà warpée à 155 BPM
    - sinon : fallback get_audio original
    """
    use_global = True

    try:
        use_global = bool(self.global_warp_var.get())
    except Exception:
        use_global = True

    if not use_global:
        return _old_v46_get_audio(self, pair)

    pair = int(pair)
    block = self.block_by_pair[pair]

    if block.get("storage") != "index_only":
        return _old_v46_get_audio(self, pair)

    warp_info = v46_get_warped_loop(self)

    warped = warp_info["audio"]
    source_loop_start = int(warp_info["source_loop_start"])
    source_loop_len = max(1, int(warp_info["source_loop_len"]))
    warped_len = max(1, int(warp_info["warped_len"]))

    source_a = int(block.get("source_start_sample"))
    source_b = int(block.get("source_end_sample"))

    rel_a = max(0, source_a - source_loop_start)
    rel_b = max(rel_a + 1, source_b - source_loop_start)

    wa = int(round(rel_a / source_loop_len * warped_len))
    wb = int(round(rel_b / source_loop_len * warped_len))

    wa = max(0, min(len(warped) - 1, wa))
    wb = max(wa + 1, min(len(warped), wb))

    return warped[wa:wb].astype(np.float32)


_old_v46_build_ui = SliceIndexTracker.build_ui


def v46_build_ui(self):
    _old_v46_build_ui(self)

    try:
        self.global_warp_var = tk.BooleanVar(value=True)

        frame = tk.Frame(self.root, bg="#101820")
        frame.pack(fill="x", padx=14, pady=(0, 8))

        tk.Checkbutton(
            frame,
            text="M8 global warp 155 BPM",
            variable=self.global_warp_var,
            bg="#101820",
            fg="#77f5b5",
            selectcolor="#30283f",
            activebackground="#101820",
            activeforeground="#77f5b5",
        ).pack(side="left", padx=8)

        tk.Label(
            frame,
            text="source unique + markers | warp de la boucle entière | slices extraites après warp",
            bg="#101820",
            fg="#b9acc8",
        ).pack(side="left", padx=8)

        if hasattr(self, "set_status"):
            self.set_status("v46 : M8 markers + global warp 155 BPM ON.")
        else:
            self.output_label.config(text="v46 : M8 markers + global warp 155 BPM ON.")

    except Exception as exc:
        print(f"[v140] UI extension impossible : {exc}")


_old_v46_switch_break = SliceIndexTracker.switch_break


def v46_switch_break(self, event=None):
    result = _old_v46_switch_break(self, event)

    self.step_count = LOCKED_GRID_STEPS
    self.canvas_width = self.left_width + self.step_width * self.step_count

    try:
        self.canvas.configure(width=self.canvas_width)
    except Exception:
        pass

    self._v46_global_warp_cache = {}

    try:
        self.step_ms_var.set(str(round(LOCKED_STEP_MS, 4)))
    except Exception:
        pass

    return result


# Patch final.
SliceIndexTracker.guess_step_ms = v46_guess_step_ms
SliceIndexTracker.build_initial_pattern = v46_build_initial_pattern
SliceIndexTracker.get_audio = v46_get_audio
SliceIndexTracker.build_ui = v46_build_ui
SliceIndexTracker.switch_break = v46_switch_break



# ---------------------------------------------------------------------
# v47 EXTENSION : anti-son fantôme
# ---------------------------------------------------------------------

def v47_audio_peak(audio):
    try:
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) == 0:
            return 0.0
        return float(np.max(np.abs(audio)))
    except Exception:
        return 0.0


def v47_is_silent(audio, threshold=1e-6):
    return v47_audio_peak(audio) <= float(threshold)


def v47_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


_old_v47_play_audio_array = SliceIndexTracker.play_audio_array


def v47_play_audio_array(self, audio, label="audio"):
    """
    Sécurité principale :
    si le rendu est vide/silencieux, on ne lance PAS pw-play.
    Ça évite le petit 'poinnn' résiduel quand il n'y a aucune case.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) == 0 or v47_is_silent(audio):
        try:
            self.stop_audio()
        except Exception:
            pass

        v47_status(self, f"Aucun son à jouer : {label} est vide.")
        print(f"[v140] lecture annulée : audio silencieux ({label})")
        return 0

    return _old_v47_play_audio_array(self, audio, label=label)


_old_v47_toggle_loop = SliceIndexTracker.toggle_loop


def v47_toggle_loop(self):
    """
    Si aucune case n'est posée, Space ne doit rien lancer.
    """
    if not getattr(self, "pattern", None):
        self.looping = False

        try:
            if self.loop_after_id:
                self.root.after_cancel(self.loop_after_id)
                self.loop_after_id = None
        except Exception:
            pass

        try:
            self.stop_playhead()
        except Exception:
            pass

        try:
            self.stop_audio()
        except Exception:
            pass

        v47_status(self, "Pattern vide : aucun son lancé.")
        print("[v140] loop annulée : pattern vide")
        return "break"

    return _old_v47_toggle_loop(self)


_old_v47_delete_selected = SliceIndexTracker.delete_selected


def v47_delete_selected(self):
    result = _old_v47_delete_selected(self)

    if not getattr(self, "pattern", None):
        self.looping = False

        try:
            if self.loop_after_id:
                self.root.after_cancel(self.loop_after_id)
                self.loop_after_id = None
        except Exception:
            pass

        try:
            self.stop_playhead()
        except Exception:
            pass

        try:
            self.stop_audio()
        except Exception:
            pass

        v47_status(self, "Dernière case supprimée : audio coupé, pattern vide.")
        print("[v140] pattern vide après suppression : audio coupé")

    return result


_old_v47_render_audio_with_timeline = SliceIndexTracker.render_audio_with_timeline


def v47_render_audio_with_timeline(self):
    """
    Si pattern vide : vrai silence, timeline vide.
    """
    if not getattr(self, "pattern", None):
        loop_samples = self.get_loop_samples()
        return np.zeros(loop_samples, dtype=np.float32), []

    audio, timeline = _old_v47_render_audio_with_timeline(self)

    if v47_is_silent(audio):
        return np.zeros(len(audio), dtype=np.float32), timeline

    return audio, timeline


_old_v47_build_gapless_loop_buffer = SliceIndexTracker.build_gapless_loop_buffer


def v47_build_gapless_loop_buffer(self):
    if not getattr(self, "pattern", None):
        loop_samples = self.get_loop_samples()
        return np.zeros(loop_samples, dtype=np.float32), 1, 0, 0.0

    audio, repeats, one_loop_ms, one_loop_sec = _old_v47_build_gapless_loop_buffer(self)

    if v47_is_silent(audio):
        return np.zeros(len(audio), dtype=np.float32), 1, 0, 0.0

    return audio, repeats, one_loop_ms, one_loop_sec


_old_v47_build_ui = SliceIndexTracker.build_ui


def v47_build_ui(self):
    _old_v47_build_ui(self)
    v47_status(self, "v47 : anti-son fantôme ON | pattern vide = silence total.")


SliceIndexTracker.play_audio_array = v47_play_audio_array
SliceIndexTracker.toggle_loop = v47_toggle_loop
SliceIndexTracker.delete_selected = v47_delete_selected
SliceIndexTracker.render_audio_with_timeline = v47_render_audio_with_timeline
SliceIndexTracker.build_gapless_loop_buffer = v47_build_gapless_loop_buffer
SliceIndexTracker.build_ui = v47_build_ui



# ---------------------------------------------------------------------
# v48 EXTENSION : vrai pattern vide + debug timeline
# ---------------------------------------------------------------------

_old_v48_build_initial_pattern = SliceIndexTracker.build_initial_pattern


def v48_build_initial_pattern(self, randomize=False):
    """
    v48 :
    - ouverture de l'app = pattern vraiment vide
    - randomize peut encore remplir
    """
    if randomize:
        return _old_v48_build_initial_pattern(self, randomize=True)

    print("[v140] pattern initial forcé VIDE : aucun son ne doit jouer.")
    return []


def v48_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v48_load_first_slices(self):
    """
    Remplit volontairement la grille avec les premières slices.
    Ça remplace le pré-remplissage automatique.
    """
    pattern = _old_v48_build_initial_pattern(self, randomize=False)

    self.pattern = pattern
    self.selected_id = self.pattern[0]["id"] if self.pattern else None

    try:
        self.draw()
    except Exception:
        pass

    try:
        self.refresh_panel()
    except Exception:
        pass

    try:
        self.write_latest_pattern(reason="v48_load_first_slices")
    except Exception:
        pass

    v48_status(self, f"Load 16 slices : {len(self.pattern)} cases posées.")
    print(f"[v140] Load 16 slices : {len(self.pattern)} items")


def v48_debug_pattern(self, label="render"):
    print("")
    print(f"[v140] DEBUG PATTERN avant {label}")
    print(f"[v140] items = {len(getattr(self, 'pattern', []) or [])}")

    if not getattr(self, "pattern", None):
        print("[v140] pattern vide confirmé")
        print("")
        return

    for item in self.pattern:
        x = int(item.get("x_step", -999))
        length = int(item.get("length", 0))
        pair = item.get("pair")
        lane = item.get("lane")
        item_id = item.get("id")

        end = x + length - 1

        flag = ""
        if x in (2, 20) or end in (3, 21) or (x <= 2 <= end) or (x <= 20 <= end):
            flag = "  <<< ICI possiblement ton 'poinnn'"

        print(
            f"[v140] id={item_id} | step {x:02d}/{end:02d} | "
            f"pair={pair} | lane={lane} | len={length}{flag}"
        )

    print("")


_old_v48_render_audio_with_timeline = SliceIndexTracker.render_audio_with_timeline


def v48_render_audio_with_timeline(self):
    v48_debug_pattern(self, label="render_audio_with_timeline")

    if not getattr(self, "pattern", None):
        loop_samples = self.get_loop_samples()
        print(f"[v140] rendu silence total : pattern vide | samples={loop_samples}")
        return np.zeros(loop_samples, dtype=np.float32), []

    audio, timeline = _old_v48_render_audio_with_timeline(self)

    peak = 0.0
    try:
        if len(audio):
            peak = float(np.max(np.abs(audio)))
    except Exception:
        peak = 0.0

    print(f"[v140] rendu audio peak={peak:.8f} | timeline_events={len(timeline) if timeline is not None else 'None'}")

    if timeline:
        print("[v140] timeline events:")
        for ev in timeline:
            print(f"[v140]   {ev}")

    return audio, timeline


_old_v48_toggle_loop = SliceIndexTracker.toggle_loop


def v48_toggle_loop(self):
    if not getattr(self, "pattern", None):
        try:
            self.stop_playhead()
        except Exception:
            pass

        try:
            self.stop_audio()
        except Exception:
            pass

        v48_status(self, "Pattern vide : Space ne lance aucun son.")
        print("[v140] Space ignoré : pattern vide.")
        return "break"

    return _old_v48_toggle_loop(self)


_old_v48_reset = SliceIndexTracker.reset


def v48_reset(self):
    """
    Reset = vide vraiment la grille, au lieu de régénérer un pattern.
    """
    try:
        self.stop_playhead()
    except Exception:
        pass

    try:
        self.stop_audio()
    except Exception:
        pass

    self.looping = False
    self.pattern = []
    self.selected_id = None

    try:
        self.draw()
    except Exception:
        pass

    try:
        self.refresh_panel()
    except Exception:
        pass

    try:
        self.write_latest_pattern(reason="v48_reset_empty")
    except Exception:
        pass

    v48_status(self, "Reset : pattern vide, silence total.")
    print("[v140] reset => pattern vide")
    return "break"


_old_v48_build_ui = SliceIndexTracker.build_ui


def v48_build_ui(self):
    _old_v48_build_ui(self)

    try:
        frame = tk.Frame(self.root, bg="#101820")
        frame.pack(fill="x", padx=14, pady=(0, 8))

        tk.Button(
            frame,
            text="Load 16 slices",
            command=self.load_first_slices,
            bg="#30513f",
            fg="#f5eefe",
        ).pack(side="left", padx=8)

        tk.Label(
            frame,
            text="v48 : ouverture vide. Si tu entends quelque chose sans case, c'est qu'il reste un event externe.",
            bg="#101820",
            fg="#b9acc8",
        ).pack(side="left", padx=8)

    except Exception as exc:
        print(f"[v140] ajout UI impossible : {exc}")

    v48_status(self, "v48 : pattern vide au lancement | Load 16 slices pour remplir.")


SliceIndexTracker.build_initial_pattern = v48_build_initial_pattern
SliceIndexTracker.load_first_slices = v48_load_first_slices
SliceIndexTracker.debug_pattern = v48_debug_pattern
SliceIndexTracker.render_audio_with_timeline = v48_render_audio_with_timeline
SliceIndexTracker.toggle_loop = v48_toggle_loop
SliceIndexTracker.reset = v48_reset
SliceIndexTracker.build_ui = v48_build_ui



# ---------------------------------------------------------------------
# v49 EXTENSION : clic delete + Suppr + Ctrl+D
# ---------------------------------------------------------------------

def v49_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v49_canvas_xy(self, event):
    try:
        x = float(self.canvas.canvasx(event.x))
        y = float(self.canvas.canvasy(event.y))
    except Exception:
        x = float(event.x)
        y = float(event.y)

    return x, y


def v49_find_item_at_xy(self, x, y):
    """
    Trouve une note existante sous la souris.
    """
    if not getattr(self, "pattern", None):
        return None

    for item in reversed(self.pattern):
        try:
            x_step = int(item.get("x_step", 0))
            length = int(item.get("length", HIT_LENGTH_STEPS))
            lane = int(item.get("lane", 0))

            x0 = float(self.left_width + x_step * self.step_width)
            x1 = float(self.left_width + (x_step + length) * self.step_width)
            y0 = float(lane * self.row_height)
            y1 = float((lane + 1) * self.row_height)

            if x0 <= x <= x1 and y0 <= y <= y1:
                return item
        except Exception:
            continue

    return None


def v49_next_item_id(self):
    ids = []

    for item in getattr(self, "pattern", []) or []:
        try:
            ids.append(int(item.get("id", -1)))
        except Exception:
            pass

    return (max(ids) + 1) if ids else 0


def v49_clamp_x_step(self, x_step):
    try:
        max_x = int(self.step_count) - int(HIT_LENGTH_STEPS)
    except Exception:
        max_x = 30

    x_step = max(0, min(int(max_x), int(x_step)))

    # Snap sur la grille de hits : 0,2,4,6...
    x_step = int(round(x_step / float(HIT_SPACING_STEPS))) * int(HIT_SPACING_STEPS)
    x_step = max(0, min(int(max_x), int(x_step)))

    return x_step


def v49_make_item(self, x_step, lane, pair):
    x_step = v49_clamp_x_step(self, x_step)
    lane = max(0, min(len(self.pair_values) - 1, int(lane)))
    pair = int(pair)

    return {
        "id": v49_next_item_id(self),
        "x_step": int(x_step),
        "lane": int(lane),
        "pair": int(pair),
        "length": int(HIT_LENGTH_STEPS),
        "variation_bar": int(x_step) // 8,
        "variation_pos": int(x_step) % 8,
        "hit_slot": int(x_step) // int(HIT_SPACING_STEPS),
        "randomized": False,
    }


def v49_refresh_after_edit(self, reason):
    try:
        self.draw()
    except Exception:
        pass

    try:
        self.refresh_panel()
    except Exception:
        pass

    try:
        self.write_latest_pattern(reason=reason)
    except Exception:
        pass


def v49_delete_item(self, item, reason="delete_note"):
    if item is None:
        v49_status(self, "Aucune note à supprimer.")
        return "break"

    before = dict(item)
    item_id = int(item.get("id"))

    self.pattern = [
        it for it in getattr(self, "pattern", []) or []
        if int(it.get("id", -999999)) != item_id
    ]

    if self.pattern:
        self.selected_id = int(self.pattern[min(len(self.pattern) - 1, 0)]["id"])
    else:
        self.selected_id = None

    try:
        self.record_correction(reason, before=before, after=None)
    except Exception as exc:
        print(f"[v140] record_correction delete impossible : {exc}")

    v49_refresh_after_edit(self, reason)
    v49_status(self, f"Note supprimée : slice {before.get('pair')} | step {before.get('x_step')}")

    print(
        f"[v140] delete | id={before.get('id')} | "
        f"step={before.get('x_step')} | pair={before.get('pair')}"
    )

    return "break"


def v49_add_item_at_xy(self, x, y):
    if x < self.left_width:
        return "break"

    raw_step = int(round((x - self.left_width) / float(self.step_width)))
    x_step = v49_clamp_x_step(self, raw_step)

    lane = int(y // float(self.row_height))
    lane = max(0, min(len(self.pair_values) - 1, lane))

    pair = int(self.pair_values[lane])

    item = v49_make_item(self, x_step=x_step, lane=lane, pair=pair)

    self.pattern.append(item)
    self.selected_id = int(item["id"])

    try:
        self.record_correction("add_note_click", before=None, after=dict(item))
    except Exception as exc:
        print(f"[v140] record_correction add impossible : {exc}")

    v49_refresh_after_edit(self, "add_note_click")
    v49_status(self, f"Note ajoutée : slice {pair} | step {x_step}")

    try:
        self.audition_selected_case()
    except Exception as exc:
        print(f"[v140] audition après ajout impossible : {exc}")

    print(f"[v140] add | id={item['id']} | step={x_step} | pair={pair} | lane={lane}")

    return "break"


def v49_canvas_click(self, event):
    """
    Nouveau comportement :
    - clic sur une note existante = suppression
    - clic sur case vide = ajout
    """
    x, y = v49_canvas_xy(self, event)
    item = v49_find_item_at_xy(self, x, y)

    if item is not None:
        self.selected_id = int(item.get("id"))
        return v49_delete_item(self, item, reason="delete_note_click")

    return v49_add_item_at_xy(self, x, y)


def v49_get_selected_item(self):
    selected_id = getattr(self, "selected_id", None)

    if selected_id is None:
        return None

    for item in getattr(self, "pattern", []) or []:
        try:
            if int(item.get("id")) == int(selected_id):
                return item
        except Exception:
            continue

    return None


def v49_delete_selected(self, event=None):
    item = v49_get_selected_item(self)
    return v49_delete_item(self, item, reason="delete_note_key")


def v49_has_collision(self, x_step, lane, ignore_id=None):
    for item in getattr(self, "pattern", []) or []:
        try:
            if ignore_id is not None and int(item.get("id")) == int(ignore_id):
                continue

            if int(item.get("x_step")) == int(x_step) and int(item.get("lane")) == int(lane):
                return True
        except Exception:
            pass

    return False


def v49_duplicate_selected_forward(self, event=None):
    item = v49_get_selected_item(self)

    if item is None:
        v49_status(self, "Ctrl+D : aucune note sélectionnée.")
        print("[v140] duplicate annulé : aucune note sélectionnée")
        return "break"

    before = dict(item)

    old_x = int(item.get("x_step", 0))
    lane = int(item.get("lane", 0))
    pair = int(item.get("pair", 0))

    try:
        max_x = int(self.step_count) - int(HIT_LENGTH_STEPS)
    except Exception:
        max_x = 30

    new_x = old_x + int(HIT_SPACING_STEPS)

    # Cherche la prochaine position libre sur la même ligne.
    while new_x <= max_x and v49_has_collision(self, new_x, lane, ignore_id=item.get("id")):
        new_x += int(HIT_SPACING_STEPS)

    if new_x > max_x:
        v49_status(self, "Ctrl+D : impossible, fin de grille.")
        print("[v140] duplicate annulé : fin de grille")
        return "break"

    new_item = v49_make_item(self, x_step=new_x, lane=lane, pair=pair)
    new_item["duplicated_from"] = int(item.get("id"))

    self.pattern.append(new_item)
    self.selected_id = int(new_item["id"])

    try:
        self.record_correction("duplicate_note_forward", before=before, after=dict(new_item))
    except Exception as exc:
        print(f"[v140] record_correction duplicate impossible : {exc}")

    v49_refresh_after_edit(self, "duplicate_note_forward")
    v49_status(self, f"Dupliquée : slice {pair} | step {old_x} -> {new_x}")

    try:
        self.audition_selected_case()
    except Exception as exc:
        print(f"[v140] audition après duplicate impossible : {exc}")

    print(
        f"[v140] duplicate | old_id={item.get('id')} | new_id={new_item.get('id')} | "
        f"pair={pair} | {old_x} -> {new_x}"
    )

    return "break"


_old_v49_build_ui = SliceIndexTracker.build_ui


def v49_build_ui(self):
    _old_v49_build_ui(self)

    try:
        self.canvas.bind("<Button-1>", self.v49_canvas_click)
    except Exception as exc:
        print(f"[v140] bind canvas click impossible : {exc}")

    try:
        self.root.bind("<Delete>", self.delete_selected)
        self.root.bind("<BackSpace>", self.delete_selected)
        self.root.bind("<Control-d>", self.duplicate_selected_forward)
        self.root.bind("<Control-D>", self.duplicate_selected_forward)

        self.canvas.bind("<Delete>", self.delete_selected)
        self.canvas.bind("<BackSpace>", self.delete_selected)
        self.canvas.bind("<Control-d>", self.duplicate_selected_forward)
        self.canvas.bind("<Control-D>", self.duplicate_selected_forward)

        self.root.bind_all("<Control-d>", self.duplicate_selected_forward)
        self.root.bind_all("<Control-D>", self.duplicate_selected_forward)
    except Exception as exc:
        print(f"[v140] bind keys impossible : {exc}")

    v49_status(
        self,
        "v49 : clic note=suppr | clic vide=ajout | Suppr=suppr sélection | Ctrl+D=duplique vers l’avant."
    )


SliceIndexTracker.v49_canvas_click = v49_canvas_click
SliceIndexTracker.duplicate_selected_forward = v49_duplicate_selected_forward
SliceIndexTracker.delete_selected = v49_delete_selected
SliceIndexTracker.build_ui = v49_build_ui



# ---------------------------------------------------------------------
# v50 EXTENSION : génération IA basée sur les beats sauvegardés
# ---------------------------------------------------------------------

STYLE_MODEL_PATH = Path("dataset/learning/beat_style_model_v01.json")


def v50_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v50_load_style_model():
    if not STYLE_MODEL_PATH.exists():
        return None

    try:
        return json.loads(STYLE_MODEL_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[v140] impossible de charger le modèle IA : {exc}")
        return None


def v50_float_count(d, key, default=0.0):
    if not isinstance(d, dict):
        return default

    return float(d.get(str(key), d.get(int(key), default)))


def v50_nested_count(d, outer, inner, default=0.0):
    if not isinstance(d, dict):
        return default

    bucket = d.get(str(outer), d.get(int(outer), {}))

    if not isinstance(bucket, dict):
        return default

    return float(bucket.get(str(inner), bucket.get(int(inner), default)))


def v50_weighted_choice(rng, values, weights):
    weights = np.asarray(weights, dtype=np.float64)

    if len(values) == 0:
        return None

    if not np.all(np.isfinite(weights)) or float(np.sum(weights)) <= 0.0:
        return int(rng.choice(values))

    weights = weights / float(np.sum(weights))
    idx = int(rng.choice(np.arange(len(values)), p=weights))

    return int(values[idx])


def v50_generate_ai_pattern(self, event=None):
    """
    Génère un beat avec la mémoire entraînée par 04_train_beat_style_v01.py.

    Principe :
    - slot counts : ce qui marche à chaque position
    - transition counts : ce qui marche après la slice précédente
    - pair counts : favoris globaux
    - exploration : petite part de hasard
    """
    model = v50_load_style_model()

    if model is None:
        v50_status(self, "Aucun modèle IA. Lance : python pipeline/04_train_beat_style_v01.py")
        print("[v140] modèle absent :", STYLE_MODEL_PATH)
        return "break"

    safe = self.project.get("safe", "")
    break_model = model.get("breaks", {}).get(safe, {})
    global_model = model.get("global", {})

    slot_pair_counts_break = break_model.get("slot_pair_counts", {})
    pair_counts_break = break_model.get("pair_counts", {})
    transition_counts_break = break_model.get("transition_counts", {})

    slot_pair_counts_global = global_model.get("slot_pair_counts", {})
    pair_counts_global = global_model.get("pair_counts", {})
    transition_counts_global = global_model.get("transition_counts", {})

    rng = np.random.default_rng(int(time.time_ns() % (2**32)))

    pair_values = [int(p) for p in self.pair_values]
    max_slots = min(16, max(1, int(getattr(self, "step_count", 32)) // int(HIT_SPACING_STEPS)))

    pattern = []
    prev_pair = None

    temperature = 0.85
    exploration = 0.58

    print("")
    print(f"[v140] GENERATE IA | safe={safe} | pairs={pair_values} | slots={max_slots}")

    for slot in range(max_slots):
        x_step = slot * int(HIT_SPACING_STEPS)

        weights = []

        for pair in pair_values:
            w = 0.55

            # Apprentissage spécifique au break.
            w += 4.00 * v50_nested_count(slot_pair_counts_break, slot, pair)
            w += 1.60 * v50_float_count(pair_counts_break, pair)

            # Apprentissage global si peu de données sur ce break.
            w += 1.50 * v50_nested_count(slot_pair_counts_global, slot, pair)
            w += 0.60 * v50_float_count(pair_counts_global, pair)

            # Transitions.
            if prev_pair is not None:
                w += 3.20 * v50_nested_count(transition_counts_break, prev_pair, pair)
                w += 1.30 * v50_nested_count(transition_counts_global, prev_pair, pair)

                # Évite un peu les répétitions mécaniques, mais sans les interdire.
                if int(pair) == int(prev_pair):
                    w *= 0.62

            # Exploration contrôlée.
            w = max(0.0001, w)
            w = math.pow(w, 1.0 / max(0.05, temperature))
            w += float(rng.random()) * exploration

            weights.append(w)

        chosen_pair = v50_weighted_choice(rng, pair_values, weights)

        if chosen_pair is None:
            continue

        lane = self.pair_to_lane.get(int(chosen_pair), 0)

        item = {
            "id": len(pattern),
            "x_step": int(x_step),
            "lane": int(lane),
            "pair": int(chosen_pair),
            "length": int(HIT_LENGTH_STEPS),
            "variation_bar": int(x_step) // 8,
            "variation_pos": int(x_step) % 8,
            "hit_slot": int(slot),
            "randomized": False,
            "ai_generated": True,
            "ai_model": "beat_style_model_v01",
        }

        pattern.append(item)
        prev_pair = int(chosen_pair)

        print(
            f"[v140] slot {slot:02d} | step {x_step:02d} | "
            f"pair={chosen_pair:02d} | lane={lane}"
        )

    self.stop_playhead()
    self.stop_audio()
    self.looping = False

    self.pattern = pattern
    self.selected_id = self.pattern[0]["id"] if self.pattern else None

    try:
        self.draw()
    except Exception:
        pass

    try:
        self.refresh_panel()
    except Exception:
        pass

    try:
        self.write_latest_pattern(reason="v50_ai_generate")
    except Exception as exc:
        print(f"[v140] write_latest_pattern impossible : {exc}")

    v50_status(
        self,
        f"Generate IA OK : {len(pattern)} notes | modèle appris depuis {model.get('patterns_count', 0)} patterns."
    )

    return "break"


_old_v50_build_ui = SliceIndexTracker.build_ui


def v50_build_ui(self):
    _old_v50_build_ui(self)

    try:
        frame = tk.Frame(self.root, bg="#101820")
        frame.pack(fill="x", padx=14, pady=(0, 8))

        tk.Button(
            frame,
            text="Generate IA",
            command=self.generate_ai_pattern,
            bg="#52306f",
            fg="#f5eefe",
        ).pack(side="left", padx=8)

        tk.Button(
            frame,
            text="Train IA reminder",
            command=lambda: v50_status(self, "Terminal : python pipeline/04_train_beat_style_v01.py puis relance Generate IA"),
            bg="#30283f",
            fg="#f5eefe",
        ).pack(side="left", padx=8)

        tk.Label(
            frame,
            text="v50 : apprend tes beats sauvegardés → génère des variations.",
            bg="#101820",
            fg="#b9acc8",
        ).pack(side="left", padx=8)

    except Exception as exc:
        print(f"[v140] UI IA impossible : {exc}")

    v50_status(self, "v50 : lance Train IA dans le terminal, puis Generate IA.")


SliceIndexTracker.generate_ai_pattern = v50_generate_ai_pattern
SliceIndexTracker.build_ui = v50_build_ui



# ---------------------------------------------------------------------
# v51 EXTENSION : clic court = delete, clic-glissé = déplacement
# ---------------------------------------------------------------------

import math as _v51_math

V51_DRAG_THRESHOLD_PX = 5


def v51_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v51_canvas_xy(self, event):
    try:
        x = float(self.canvas.canvasx(event.x))
        y = float(self.canvas.canvasy(event.y))
    except Exception:
        x = float(event.x)
        y = float(event.y)

    return x, y


def v51_find_item_at_xy(self, x, y):
    if not getattr(self, "pattern", None):
        return None

    for item in reversed(self.pattern):
        try:
            x_step = int(item.get("x_step", 0))
            length = int(item.get("length", HIT_LENGTH_STEPS))
            lane = int(item.get("lane", 0))

            x0 = float(self.left_width + x_step * self.step_width)
            x1 = float(self.left_width + (x_step + length) * self.step_width)
            y0 = float(lane * self.row_height)
            y1 = float((lane + 1) * self.row_height)

            if x0 <= x <= x1 and y0 <= y <= y1:
                return item
        except Exception:
            continue

    return None


def v51_get_item_by_id(self, item_id):
    if item_id is None:
        return None

    for item in getattr(self, "pattern", []) or []:
        try:
            if int(item.get("id")) == int(item_id):
                return item
        except Exception:
            pass

    return None


def v51_next_item_id(self):
    ids = []

    for item in getattr(self, "pattern", []) or []:
        try:
            ids.append(int(item.get("id", -1)))
        except Exception:
            pass

    return max(ids) + 1 if ids else 0


def v51_clamp_x_step(self, x_step):
    try:
        max_x = int(self.step_count) - int(HIT_LENGTH_STEPS)
    except Exception:
        max_x = 30

    x_step = max(0, min(int(max_x), int(x_step)))
    x_step = int(round(x_step / float(HIT_SPACING_STEPS))) * int(HIT_SPACING_STEPS)
    x_step = max(0, min(int(max_x), int(x_step)))

    return x_step


def v51_xy_to_grid(self, x, y):
    raw_step = int(round((x - self.left_width) / float(self.step_width)))
    x_step = v51_clamp_x_step(self, raw_step)

    lane = int(y // float(self.row_height))
    lane = max(0, min(len(self.pair_values) - 1, lane))

    pair = int(self.pair_values[lane])

    return x_step, lane, pair


def v51_refresh_after_edit(self, reason):
    try:
        self.draw()
    except Exception:
        pass

    try:
        self.refresh_panel()
    except Exception:
        pass

    try:
        self.write_latest_pattern(reason=reason)
    except Exception:
        pass


def v51_delete_item(self, item, reason="delete_note"):
    if item is None:
        v51_status(self, "Aucune note à supprimer.")
        return "break"

    before = dict(item)
    item_id = int(item.get("id"))

    self.pattern = [
        it for it in getattr(self, "pattern", []) or []
        if int(it.get("id", -999999)) != item_id
    ]

    self.selected_id = self.pattern[0]["id"] if self.pattern else None

    try:
        self.record_correction(reason, before=before, after=None)
    except Exception as exc:
        print(f"[v140] record_correction delete impossible : {exc}")

    v51_refresh_after_edit(self, reason)

    v51_status(
        self,
        f"Note supprimée : slice {before.get('pair')} | step {before.get('x_step')}"
    )

    print(
        f"[v140] delete | id={before.get('id')} | "
        f"step={before.get('x_step')} | pair={before.get('pair')}"
    )

    return "break"


def v51_add_item_at_xy(self, x, y):
    if x < self.left_width:
        return "break"

    x_step, lane, pair = v51_xy_to_grid(self, x, y)

    item = {
        "id": v51_next_item_id(self),
        "x_step": int(x_step),
        "lane": int(lane),
        "pair": int(pair),
        "length": int(HIT_LENGTH_STEPS),
        "variation_bar": int(x_step) // 8,
        "variation_pos": int(x_step) % 8,
        "hit_slot": int(x_step) // int(HIT_SPACING_STEPS),
        "randomized": False,
    }

    self.pattern.append(item)
    self.selected_id = int(item["id"])

    try:
        self.record_correction("add_note_click", before=None, after=dict(item))
    except Exception as exc:
        print(f"[v140] record_correction add impossible : {exc}")

    v51_refresh_after_edit(self, "add_note_click")

    v51_status(self, f"Note ajoutée : slice {pair} | step {x_step}")

    try:
        self.audition_selected_case()
    except Exception as exc:
        print(f"[v140] audition après ajout impossible : {exc}")

    print(f"[v140] add | id={item['id']} | step={x_step} | pair={pair} | lane={lane}")

    return "break"


def v51_mouse_down(self, event):
    """
    Ne supprime PAS immédiatement.
    On attend le release pour savoir si c'était un clic ou un drag.
    """
    try:
        self.canvas.focus_set()
    except Exception:
        pass

    x, y = v51_canvas_xy(self, event)
    item = v51_find_item_at_xy(self, x, y)

    state = {
        "press_x": x,
        "press_y": y,
        "last_x": x,
        "last_y": y,
        "moved": False,
        "item_id": None,
        "before": None,
        "empty": item is None,
    }

    if item is not None:
        self.selected_id = int(item.get("id"))
        state["item_id"] = int(item.get("id"))
        state["before"] = dict(item)

        try:
            self.draw()
            self.refresh_panel()
        except Exception:
            pass

        print(
            f"[v140] mouse down note | id={item.get('id')} | "
            f"step={item.get('x_step')} | pair={item.get('pair')}"
        )
    else:
        print("[v140] mouse down empty")

    self._v51_mouse_state = state

    return "break"


def v51_mouse_motion(self, event):
    state = getattr(self, "_v51_mouse_state", None)

    if not state:
        return "break"

    x, y = v51_canvas_xy(self, event)

    dx = abs(x - float(state["press_x"]))
    dy = abs(y - float(state["press_y"]))

    if dx > V51_DRAG_THRESHOLD_PX or dy > V51_DRAG_THRESHOLD_PX:
        state["moved"] = True

    state["last_x"] = x
    state["last_y"] = y

    # Si on a cliqué sur du vide puis glissé : on ne fait rien.
    if state.get("empty"):
        return "break"

    # Drag d'une note existante.
    if not state.get("moved"):
        return "break"

    item = v51_get_item_by_id(self, state.get("item_id"))

    if item is None:
        return "break"

    x_step, lane, pair = v51_xy_to_grid(self, x, y)

    changed = (
        int(item.get("x_step", -999)) != int(x_step)
        or int(item.get("lane", -999)) != int(lane)
        or int(item.get("pair", -999)) != int(pair)
    )

    if changed:
        item["x_step"] = int(x_step)
        item["lane"] = int(lane)
        item["pair"] = int(pair)
        item["variation_bar"] = int(x_step) // 8
        item["variation_pos"] = int(x_step) % 8
        item["hit_slot"] = int(x_step) // int(HIT_SPACING_STEPS)

        self.selected_id = int(item.get("id"))

        try:
            self.draw()
        except Exception:
            pass

        try:
            self.refresh_panel()
        except Exception:
            pass

    return "break"


def v51_mouse_up(self, event):
    state = getattr(self, "_v51_mouse_state", None)
    self._v51_mouse_state = None

    if not state:
        return "break"

    x, y = v51_canvas_xy(self, event)

    # Clic sur vide sans drag = ajoute une note.
    if state.get("empty"):
        if not state.get("moved"):
            return v51_add_item_at_xy(self, x, y)

        print("[v140] drag depuis vide ignoré")
        return "break"

    item = v51_get_item_by_id(self, state.get("item_id"))

    if item is None:
        return "break"

    # Clic court sur une note = suppression.
    if not state.get("moved"):
        return v51_delete_item(self, item, reason="delete_note_click")

    # Drag d'une note = sauvegarde mouvement, pas suppression.
    before = state.get("before") or {}
    after = dict(item)

    if before != after:
        try:
            self.record_correction("move_note_drag", before=before, after=after)
        except Exception as exc:
            print(f"[v140] record_correction drag impossible : {exc}")

        v51_refresh_after_edit(self, "move_note_drag")

        v51_status(
            self,
            f"Note déplacée : slice {after.get('pair')} | "
            f"step {before.get('x_step')} -> {after.get('x_step')}"
        )

        print(
            f"[v140] drag move | id={after.get('id')} | "
            f"step {before.get('x_step')} -> {after.get('x_step')} | "
            f"pair {before.get('pair')} -> {after.get('pair')}"
        )

        try:
            self.audition_selected_case()
        except Exception as exc:
            print(f"[v140] audition après drag impossible : {exc}")

    return "break"


def v51_get_selected_item(self):
    selected_id = getattr(self, "selected_id", None)
    return v51_get_item_by_id(self, selected_id)


def v51_delete_selected(self, event=None):
    item = v51_get_selected_item(self)
    return v51_delete_item(self, item, reason="delete_note_key")


_old_v51_build_ui = SliceIndexTracker.build_ui


def v51_build_ui(self):
    _old_v51_build_ui(self)

    try:
        self.canvas.bind("<Button-1>", self.v51_mouse_down)
        self.canvas.bind("<B1-Motion>", self.v51_mouse_motion)
        self.canvas.bind("<ButtonRelease-1>", self.v51_mouse_up)
    except Exception as exc:
        print(f"[v140] bind souris impossible : {exc}")

    try:
        self.root.bind("<Delete>", self.delete_selected)
        self.root.bind("<BackSpace>", self.delete_selected)
        self.canvas.bind("<Delete>", self.delete_selected)
        self.canvas.bind("<BackSpace>", self.delete_selected)
    except Exception as exc:
        print(f"[v140] bind Suppr impossible : {exc}")

    v51_status(
        self,
        "v51 : clic court note=suppr | clic-glissé=déplace | clic vide=ajout | Suppr=suppr | Ctrl+D=duplique."
    )


SliceIndexTracker.v51_mouse_down = v51_mouse_down
SliceIndexTracker.v51_mouse_motion = v51_mouse_motion
SliceIndexTracker.v51_mouse_up = v51_mouse_up
SliceIndexTracker.delete_selected = v51_delete_selected
SliceIndexTracker.build_ui = v51_build_ui



# ---------------------------------------------------------------------
# v52 EXTENSION : restaurer un beat sauvegardé
# ---------------------------------------------------------------------

from pathlib import Path as _V52Path
import json as _v52_json


V52_SAVE_KEYS = ["pattern", "items", "notes", "hits", "events", "blocks"]


def v52_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v52_looks_like_note(x):
    if not isinstance(x, dict):
        return False

    if "pair" not in x or "x_step" not in x:
        return False

    try:
        int(float(x.get("pair")))
        int(float(x.get("x_step")))
        return True
    except Exception:
        return False


def v52_extract_notes_from_data(data):
    patterns = []

    if isinstance(data, list):
        notes = [x for x in data if v52_looks_like_note(x)]
        if notes:
            patterns.append(("root_list", notes))
        return patterns

    if not isinstance(data, dict):
        return patterns

    for key in V52_SAVE_KEYS:
        value = data.get(key)

        if not isinstance(value, list):
            continue

        # Important : un pair_blocks_v02 contient aussi "blocks",
        # mais ce sont des slices source_start_sample, pas un beat.
        if key == "blocks":
            if any(isinstance(x, dict) and "source_start_sample" in x for x in value):
                continue

        notes = [x for x in value if v52_looks_like_note(x)]

        if notes:
            patterns.append((key, notes))

    return patterns


def v52_find_saved_beats(self):
    """
    Cherche les beats sauvegardés dans dataset/.
    Priorité :
    - fichiers du break courant
    - fichiers les plus récents
    """
    dataset = _V52Path("dataset")
    safe = str(self.project.get("safe", "")).lower()

    candidates = []

    if not dataset.exists():
        return []

    for path in dataset.rglob("*.json"):
        name = path.name.lower()
        full = str(path).lower()

        if "pair_blocks" in name:
            continue

        if "beat_style_model" in name:
            continue

        if "_slice_indexes_" in full:
            continue

        if "debug_slices" in full:
            continue

        try:
            data = _v52_json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        patterns = v52_extract_notes_from_data(data)

        if not patterns:
            continue

        file_safe = ""

        if isinstance(data, dict):
            file_safe = str(
                data.get("safe")
                or data.get("break")
                or data.get("source")
                or data.get("project", {}).get("safe", "")
                if isinstance(data.get("project"), dict)
                else ""
            ).lower()

        safe_bonus = 0

        if safe and safe in full:
            safe_bonus += 1000000000

        if safe and file_safe and safe in file_safe:
            safe_bonus += 1000000000

        mtime = path.stat().st_mtime

        for key, notes in patterns:
            candidates.append({
                "path": path,
                "key": key,
                "notes": notes,
                "mtime": mtime,
                "score": safe_bonus + mtime,
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def v52_clamp_x_step(self, x_step):
    try:
        max_x = int(self.step_count) - int(HIT_LENGTH_STEPS)
    except Exception:
        max_x = 30

    x_step = int(round(float(x_step)))
    x_step = max(0, min(max_x, x_step))

    try:
        x_step = int(round(x_step / float(HIT_SPACING_STEPS))) * int(HIT_SPACING_STEPS)
    except Exception:
        pass

    return max(0, min(max_x, x_step))


def v52_normalize_loaded_notes(self, notes):
    valid_pairs = {int(p) for p in self.pair_values}
    new_pattern = []

    for raw in sorted(notes, key=lambda n: (int(float(n.get("x_step", 0))), int(float(n.get("pair", 0))))):
        try:
            pair = int(float(raw.get("pair")))
            x_step = v52_clamp_x_step(self, raw.get("x_step", 0))
        except Exception:
            continue

        # Si une slice a été mise en quarantaine et n'existe plus, on la saute.
        if pair not in valid_pairs:
            print(f"[v140] slice ignorée car absente du break actuel : pair={pair}")
            continue

        lane = self.pair_to_lane.get(pair, int(raw.get("lane", 0)))

        try:
            lane = int(lane)
        except Exception:
            lane = self.pair_to_lane.get(pair, 0)

        item = dict(raw)
        item["id"] = len(new_pattern)
        item["x_step"] = int(x_step)
        item["pair"] = int(pair)
        item["lane"] = int(lane)
        item["length"] = int(raw.get("length", HIT_LENGTH_STEPS) or HIT_LENGTH_STEPS)
        item["variation_bar"] = int(x_step) // 8
        item["variation_pos"] = int(x_step) % 8
        item["hit_slot"] = int(x_step) // int(HIT_SPACING_STEPS)
        item["restored_from_save"] = True

        new_pattern.append(item)

    return new_pattern


def v52_load_saved_beat(self, event=None):
    saves = v52_find_saved_beats(self)

    if not saves:
        v52_status(self, "Aucune sauvegarde de beat trouvée dans dataset/.")
        print("[v140] aucune sauvegarde trouvée")
        return "break"

    chosen = saves[0]
    pattern = v52_normalize_loaded_notes(self, chosen["notes"])

    if not pattern:
        v52_status(self, "Sauvegarde trouvée, mais aucune note compatible avec les slices actuelles.")
        print("[v140] sauvegarde incompatible :", chosen["path"])
        return "break"

    try:
        self.stop_playhead()
        self.stop_audio()
    except Exception:
        pass

    self.looping = False
    self.pattern = pattern
    self.selected_id = self.pattern[0]["id"] if self.pattern else None

    try:
        self.draw()
    except Exception:
        pass

    try:
        self.refresh_panel()
    except Exception:
        pass

    try:
        self.write_latest_pattern(reason="v52_load_saved_beat")
    except Exception as exc:
        print(f"[v140] write_latest_pattern impossible : {exc}")

    v52_status(
        self,
        f"Beat restauré : {len(pattern)} notes | {chosen['path'].name}"
    )

    print("")
    print("[v140] BEAT RESTAURÉ")
    print("[v140] fichier :", chosen["path"])
    print("[v140] clé :", chosen["key"])
    print("[v140] notes :", len(pattern))
    print("")

    return "break"


def v52_list_saved_beats(self, event=None):
    saves = v52_find_saved_beats(self)

    print("")
    print("[v140] SAUVEGARDES TROUVÉES")
    for i, save in enumerate(saves[:30]):
        print(
            f"[v140] {i:02d} | notes={len(save['notes']):02d} | "
            f"key={save['key']} | {save['path']}"
        )
    print("")

    if saves:
        v52_status(self, f"{len(saves)} sauvegardes trouvées. La plus récente sera chargée avec Load saved beat.")
    else:
        v52_status(self, "Aucune sauvegarde trouvée.")


_old_v52_build_ui = SliceIndexTracker.build_ui


def v52_build_ui(self):
    _old_v52_build_ui(self)

    try:
        frame = tk.Frame(self.root, bg="#101820")
        frame.pack(fill="x", padx=14, pady=(0, 8))

        tk.Button(
            frame,
            text="Load saved beat",
            command=self.load_saved_beat,
            bg="#30513f",
            fg="#f5eefe",
        ).pack(side="left", padx=8)

        tk.Button(
            frame,
            text="List saved beats",
            command=self.list_saved_beats,
            bg="#30283f",
            fg="#f5eefe",
        ).pack(side="left", padx=8)

        tk.Label(
            frame,
            text="v52 : recharge le beat sauvegardé le plus récent compatible avec ce break.",
            bg="#101820",
            fg="#b9acc8",
        ).pack(side="left", padx=8)

    except Exception as exc:
        print(f"[v140] UI load saved impossible : {exc}")

    v52_status(self, "v52 : clique Load saved beat pour restaurer ton dernier beat sauvegardé.")


SliceIndexTracker.load_saved_beat = v52_load_saved_beat
SliceIndexTracker.list_saved_beats = v52_list_saved_beats
SliceIndexTracker.build_ui = v52_build_ui



# ---------------------------------------------------------------------
# v53 EXTENSION : chaque Save entraîne le modèle IA
# ---------------------------------------------------------------------

import subprocess as _v53_subprocess
import sys as _v53_sys


def v53_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v53_run_training(self):
    trainer = Path("pipeline/04_train_beat_style_v02.py")

    if not trainer.exists():
        v53_status(self, "Trainer IA introuvable : pipeline/04_train_beat_style_v02.py")
        print("[v140] trainer introuvable :", trainer)
        return False

    print("")
    print("[v140] AUTO-TRAIN IA après Save...")
    print("")

    try:
        result = _v53_subprocess.run(
            [_v53_sys.executable, str(trainer)],
            cwd=str(Path(".").resolve()),
            text=True,
            capture_output=True,
            check=False,
        )

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:
            v53_status(self, "Save OK, mais entraînement IA échoué. Regarde le terminal.")
            print("[v140] auto-train échec code :", result.returncode)
            return False

        v53_status(self, "Save OK + IA entraînée automatiquement.")
        print("[v140] AUTO-TRAIN OK")
        return True

    except Exception as exc:
        v53_status(self, f"Save OK, mais auto-train impossible : {exc}")
        print("[v140] exception auto-train :", exc)
        return False


_old_v53_save = SliceIndexTracker.save


def v53_save(self):
    """
    Save validation normal, puis entraînement automatique.
    """
    result = _old_v53_save(self)

    try:
        v53_run_training(self)
    except Exception as exc:
        print(f"[v140] auto-train après save impossible : {exc}")

    return result


_old_v53_build_ui = SliceIndexTracker.build_ui


def v53_build_ui(self):
    _old_v53_build_ui(self)

    try:
        frame = tk.Frame(self.root, bg="#101820")
        frame.pack(fill="x", padx=14, pady=(0, 8))

        tk.Label(
            frame,
            text="v53 : Save validation = sauvegarde + entraînement IA automatique.",
            bg="#101820",
            fg="#77f5b5",
        ).pack(side="left", padx=8)

    except Exception as exc:
        print(f"[v140] UI auto-train impossible : {exc}")

    v53_status(self, "v53 : chaque Save validation entraîne l’IA automatiquement.")


SliceIndexTracker.save = v53_save
SliceIndexTracker.build_ui = v53_build_ui



# ---------------------------------------------------------------------
# v55 EXTENSION : clic = play/select, pas delete
# ---------------------------------------------------------------------

V55_DRAG_THRESHOLD_PX = 5


def v55_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v55_canvas_xy(self, event):
    try:
        x = float(self.canvas.canvasx(event.x))
        y = float(self.canvas.canvasy(event.y))
    except Exception:
        x = float(event.x)
        y = float(event.y)

    return x, y


def v55_find_item_at_xy(self, x, y):
    if not getattr(self, "pattern", None):
        return None

    for item in reversed(self.pattern):
        try:
            x_step = int(item.get("x_step", 0))
            length = int(item.get("length", HIT_LENGTH_STEPS))
            lane = int(item.get("lane", 0))

            x0 = float(self.left_width + x_step * self.step_width)
            x1 = float(self.left_width + (x_step + length) * self.step_width)
            y0 = float(lane * self.row_height)
            y1 = float((lane + 1) * self.row_height)

            if x0 <= x <= x1 and y0 <= y <= y1:
                return item
        except Exception:
            continue

    return None


def v55_get_item_by_id(self, item_id):
    if item_id is None:
        return None

    for item in getattr(self, "pattern", []) or []:
        try:
            if int(item.get("id")) == int(item_id):
                return item
        except Exception:
            pass

    return None


def v55_next_item_id(self):
    ids = []

    for item in getattr(self, "pattern", []) or []:
        try:
            ids.append(int(item.get("id", -1)))
        except Exception:
            pass

    return max(ids) + 1 if ids else 0


def v55_clamp_x_step(self, x_step):
    try:
        max_x = int(self.step_count) - int(HIT_LENGTH_STEPS)
    except Exception:
        max_x = 30

    x_step = int(round(float(x_step)))
    x_step = max(0, min(max_x, x_step))
    x_step = int(round(x_step / float(HIT_SPACING_STEPS))) * int(HIT_SPACING_STEPS)
    x_step = max(0, min(max_x, x_step))

    return x_step


def v55_xy_to_grid(self, x, y):
    raw_step = int(round((x - self.left_width) / float(self.step_width)))
    x_step = v55_clamp_x_step(self, raw_step)

    lane = int(y // float(self.row_height))
    lane = max(0, min(len(self.pair_values) - 1, lane))

    pair = int(self.pair_values[lane])

    return x_step, lane, pair


def v55_refresh_after_edit(self, reason):
    try:
        self.draw()
    except Exception:
        pass

    try:
        self.refresh_panel()
    except Exception:
        pass

    try:
        self.write_latest_pattern(reason=reason)
    except Exception:
        pass


def v55_delete_item(self, item, reason="delete_note"):
    if item is None:
        v55_status(self, "Aucune note à supprimer.")
        return "break"

    before = dict(item)
    item_id = int(item.get("id"))

    self.pattern = [
        it for it in getattr(self, "pattern", []) or []
        if int(it.get("id", -999999)) != item_id
    ]

    self.selected_id = self.pattern[0]["id"] if self.pattern else None

    try:
        self.record_correction(reason, before=before, after=None)
    except Exception as exc:
        print(f"[v140] record_correction delete impossible : {exc}")

    v55_refresh_after_edit(self, reason)

    v55_status(
        self,
        f"Note supprimée : slice {before.get('pair')} | step {before.get('x_step')}"
    )

    print(
        f"[v140] delete | id={before.get('id')} | "
        f"step={before.get('x_step')} | pair={before.get('pair')}"
    )

    return "break"


def v55_add_item_at_xy(self, x, y):
    if x < self.left_width:
        return "break"

    x_step, lane, pair = v55_xy_to_grid(self, x, y)

    item = {
        "id": v55_next_item_id(self),
        "x_step": int(x_step),
        "lane": int(lane),
        "pair": int(pair),
        "length": int(HIT_LENGTH_STEPS),
        "variation_bar": int(x_step) // 8,
        "variation_pos": int(x_step) % 8,
        "hit_slot": int(x_step) // int(HIT_SPACING_STEPS),
        "randomized": False,
    }

    self.pattern.append(item)
    self.selected_id = int(item["id"])

    try:
        self.record_correction("add_note_click", before=None, after=dict(item))
    except Exception as exc:
        print(f"[v140] record_correction add impossible : {exc}")

    v55_refresh_after_edit(self, "add_note_click")

    v55_status(self, f"Note ajoutée : slice {pair} | step {x_step}")

    try:
        self.audition_selected_case()
    except Exception as exc:
        print(f"[v140] audition après ajout impossible : {exc}")

    print(f"[v140] add | id={item['id']} | step={x_step} | pair={pair} | lane={lane}")

    return "break"


def v55_mouse_down(self, event):
    try:
        self.canvas.focus_set()
    except Exception:
        pass

    x, y = v55_canvas_xy(self, event)
    item = v55_find_item_at_xy(self, x, y)

    state = {
        "press_x": x,
        "press_y": y,
        "last_x": x,
        "last_y": y,
        "moved": False,
        "item_id": None,
        "before": None,
        "empty": item is None,
    }

    if item is not None:
        self.selected_id = int(item.get("id"))
        state["item_id"] = int(item.get("id"))
        state["before"] = dict(item)

        try:
            self.draw()
            self.refresh_panel()
        except Exception:
            pass

        print(
            f"[v140] mouse down note | id={item.get('id')} | "
            f"step={item.get('x_step')} | pair={item.get('pair')}"
        )
    else:
        print("[v140] mouse down empty")

    self._v55_mouse_state = state

    return "break"


def v55_mouse_motion(self, event):
    state = getattr(self, "_v55_mouse_state", None)

    if not state:
        return "break"

    x, y = v55_canvas_xy(self, event)

    dx = abs(x - float(state["press_x"]))
    dy = abs(y - float(state["press_y"]))

    if dx > V55_DRAG_THRESHOLD_PX or dy > V55_DRAG_THRESHOLD_PX:
        state["moved"] = True

    state["last_x"] = x
    state["last_y"] = y

    if state.get("empty"):
        return "break"

    if not state.get("moved"):
        return "break"

    item = v55_get_item_by_id(self, state.get("item_id"))

    if item is None:
        return "break"

    x_step, lane, pair = v55_xy_to_grid(self, x, y)

    changed = (
        int(item.get("x_step", -999)) != int(x_step)
        or int(item.get("lane", -999)) != int(lane)
        or int(item.get("pair", -999)) != int(pair)
    )

    if changed:
        item["x_step"] = int(x_step)
        item["lane"] = int(lane)
        item["pair"] = int(pair)
        item["variation_bar"] = int(x_step) // 8
        item["variation_pos"] = int(x_step) % 8
        item["hit_slot"] = int(x_step) // int(HIT_SPACING_STEPS)

        self.selected_id = int(item.get("id"))

        try:
            self.draw()
            self.refresh_panel()
        except Exception:
            pass

    return "break"


def v55_mouse_up(self, event):
    state = getattr(self, "_v55_mouse_state", None)
    self._v55_mouse_state = None

    if not state:
        return "break"

    x, y = v55_canvas_xy(self, event)

    # Clic sur vide sans drag = ajoute une note.
    if state.get("empty"):
        if not state.get("moved"):
            return v55_add_item_at_xy(self, x, y)

        print("[v140] drag depuis vide ignoré")
        return "break"

    item = v55_get_item_by_id(self, state.get("item_id"))

    if item is None:
        return "break"

    # NOUVEAU :
    # Clic court sur une note = sélection + play.
    # Ne supprime plus jamais au clic gauche.
    if not state.get("moved"):
        self.selected_id = int(item.get("id"))

        try:
            self.draw()
            self.refresh_panel()
        except Exception:
            pass

        try:
            self.audition_selected_case()
        except Exception as exc:
            print(f"[v140] audition clic note impossible : {exc}")

        v55_status(
            self,
            f"Slice {item.get('pair')} jouée | step {item.get('x_step')} | Suppr pour supprimer."
        )

        print(
            f"[v140] click play/select | id={item.get('id')} | "
            f"step={item.get('x_step')} | pair={item.get('pair')}"
        )

        return "break"

    # Drag d'une note = déplacement sauvegardé.
    before = state.get("before") or {}
    after = dict(item)

    if before != after:
        try:
            self.record_correction("move_note_drag", before=before, after=after)
        except Exception as exc:
            print(f"[v140] record_correction drag impossible : {exc}")

        v55_refresh_after_edit(self, "move_note_drag")

        v55_status(
            self,
            f"Note déplacée : slice {after.get('pair')} | "
            f"step {before.get('x_step')} -> {after.get('x_step')}"
        )

        print(
            f"[v140] drag move | id={after.get('id')} | "
            f"step {before.get('x_step')} -> {after.get('x_step')} | "
            f"pair {before.get('pair')} -> {after.get('pair')}"
        )

        try:
            self.audition_selected_case()
        except Exception as exc:
            print(f"[v140] audition après drag impossible : {exc}")

    return "break"


def v55_right_click_delete(self, event):
    x, y = v55_canvas_xy(self, event)
    item = v55_find_item_at_xy(self, x, y)

    if item is None:
        v55_status(self, "Clic droit : aucune note sous la souris.")
        return "break"

    self.selected_id = int(item.get("id"))
    return v55_delete_item(self, item, reason="delete_note_right_click")


def v55_get_selected_item(self):
    selected_id = getattr(self, "selected_id", None)
    return v55_get_item_by_id(self, selected_id)


def v55_delete_selected(self, event=None):
    item = v55_get_selected_item(self)
    return v55_delete_item(self, item, reason="delete_note_key")


_old_v55_build_ui = SliceIndexTracker.build_ui


def v55_build_ui(self):
    _old_v55_build_ui(self)

    # Rebind complet pour écraser les anciens comportements v49/v51.
    try:
        self.canvas.bind("<Button-1>", self.v55_mouse_down)
        self.canvas.bind("<B1-Motion>", self.v55_mouse_motion)
        self.canvas.bind("<ButtonRelease-1>", self.v55_mouse_up)

        # Clic droit = suppression rapide.
        self.canvas.bind("<Button-3>", self.v55_right_click_delete)
    except Exception as exc:
        print(f"[v140] bind souris impossible : {exc}")

    try:
        self.root.bind("<Delete>", self.delete_selected)
        self.root.bind("<BackSpace>", self.delete_selected)
        self.canvas.bind("<Delete>", self.delete_selected)
        self.canvas.bind("<BackSpace>", self.delete_selected)
    except Exception as exc:
        print(f"[v140] bind Suppr impossible : {exc}")

    v55_status(
        self,
        "v55 : clic note=play/select | clic vide=ajout | drag=déplace | Suppr ou clic droit=supprime | Ctrl+D=duplique."
    )


SliceIndexTracker.v55_mouse_down = v55_mouse_down
SliceIndexTracker.v55_mouse_motion = v55_mouse_motion
SliceIndexTracker.v55_mouse_up = v55_mouse_up
SliceIndexTracker.v55_right_click_delete = v55_right_click_delete
SliceIndexTracker.delete_selected = v55_delete_selected
SliceIndexTracker.build_ui = v55_build_ui



# ---------------------------------------------------------------------
# v58 EXTENSION : édition fine sur demi-grille
# ---------------------------------------------------------------------

V58_FINE_GRID_DEFAULT = True
V58_ONE_CASE_DEFAULT = False


def v58_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v58_fine_enabled(self):
    try:
        return bool(self.v58_fine_grid_var.get())
    except Exception:
        return V58_FINE_GRID_DEFAULT


def v58_one_case_enabled(self):
    try:
        return bool(self.v58_one_case_var.get())
    except Exception:
        return V58_ONE_CASE_DEFAULT


def v58_note_length_for_new_note(self):
    if v58_one_case_enabled(self):
        return 1
    return int(HIT_LENGTH_STEPS)


def v58_clamp_x_step(self, x_step, length=None):
    """
    Mode normal :
        snap 0,2,4,6...
    Mode demi-grille :
        snap 0,1,2,3...
    """
    if length is None:
        length = int(HIT_LENGTH_STEPS)

    try:
        max_x = int(self.step_count) - int(length)
    except Exception:
        max_x = 32 - int(length)

    x_step = int(round(float(x_step)))
    x_step = max(0, min(max_x, x_step))

    if not v58_fine_enabled(self):
        x_step = int(round(x_step / float(HIT_SPACING_STEPS))) * int(HIT_SPACING_STEPS)
        x_step = max(0, min(max_x, x_step))

    return int(x_step)


def v58_canvas_xy(self, event):
    try:
        x = float(self.canvas.canvasx(event.x))
        y = float(self.canvas.canvasy(event.y))
    except Exception:
        x = float(event.x)
        y = float(event.y)

    return x, y


def v58_xy_to_grid(self, x, y, length=None):
    if length is None:
        length = v58_note_length_for_new_note(self)

    raw_step = int(round((x - self.left_width) / float(self.step_width)))
    x_step = v58_clamp_x_step(self, raw_step, length=length)

    lane = int(y // float(self.row_height))
    lane = max(0, min(len(self.pair_values) - 1, lane))

    pair = int(self.pair_values[lane])

    return x_step, lane, pair


def v58_find_item_at_xy(self, x, y):
    if not getattr(self, "pattern", None):
        return None

    for item in reversed(self.pattern):
        try:
            x_step = int(item.get("x_step", 0))
            length = int(item.get("length", HIT_LENGTH_STEPS))
            lane = int(item.get("lane", 0))

            x0 = float(self.left_width + x_step * self.step_width)
            x1 = float(self.left_width + (x_step + length) * self.step_width)
            y0 = float(lane * self.row_height)
            y1 = float((lane + 1) * self.row_height)

            if x0 <= x <= x1 and y0 <= y <= y1:
                return item
        except Exception:
            continue

    return None


def v58_get_item_by_id(self, item_id):
    if item_id is None:
        return None

    for item in getattr(self, "pattern", []) or []:
        try:
            if int(item.get("id")) == int(item_id):
                return item
        except Exception:
            pass

    return None


def v58_next_item_id(self):
    ids = []

    for item in getattr(self, "pattern", []) or []:
        try:
            ids.append(int(item.get("id", -1)))
        except Exception:
            pass

    return max(ids) + 1 if ids else 0


def v58_refresh_after_edit(self, reason):
    try:
        self.draw()
    except Exception:
        pass

    try:
        self.refresh_panel()
    except Exception:
        pass

    try:
        self.write_latest_pattern(reason=reason)
    except Exception:
        pass


def v58_add_item_at_xy(self, x, y):
    if x < self.left_width:
        return "break"

    length = v58_note_length_for_new_note(self)
    x_step, lane, pair = v58_xy_to_grid(self, x, y, length=length)

    item = {
        "id": v58_next_item_id(self),
        "x_step": int(x_step),
        "lane": int(lane),
        "pair": int(pair),
        "length": int(length),
        "variation_bar": int(x_step) // 8,
        "variation_pos": int(x_step) % 8,
        "hit_slot": int(x_step) // int(HIT_SPACING_STEPS),
        "randomized": False,
        "fine_grid": bool(v58_fine_enabled(self)),
    }

    self.pattern.append(item)
    self.selected_id = int(item["id"])

    try:
        self.record_correction("add_note_fine_grid", before=None, after=dict(item))
    except Exception as exc:
        print(f"[v140] record_correction add impossible : {exc}")

    v58_refresh_after_edit(self, "add_note_fine_grid")

    v58_status(
        self,
        f"Note ajoutée : slice {pair} | case {x_step}"
        + (f"/{x_step + length - 1}" if length > 1 else "")
    )

    try:
        self.audition_selected_case()
    except Exception as exc:
        print(f"[v140] audition après ajout impossible : {exc}")

    print(
        f"[v140] add | id={item['id']} | step={x_step} | "
        f"len={length} | pair={pair} | lane={lane}"
    )

    return "break"


def v58_mouse_down(self, event):
    try:
        self.canvas.focus_set()
    except Exception:
        pass

    x, y = v58_canvas_xy(self, event)
    item = v58_find_item_at_xy(self, x, y)

    state = {
        "press_x": x,
        "press_y": y,
        "last_x": x,
        "last_y": y,
        "moved": False,
        "item_id": None,
        "before": None,
        "empty": item is None,
    }

    if item is not None:
        self.selected_id = int(item.get("id"))
        state["item_id"] = int(item.get("id"))
        state["before"] = dict(item)

        try:
            self.draw()
            self.refresh_panel()
        except Exception:
            pass

        print(
            f"[v140] mouse down note | id={item.get('id')} | "
            f"step={item.get('x_step')} | len={item.get('length')} | pair={item.get('pair')}"
        )
    else:
        print("[v140] mouse down empty")

    self._v58_mouse_state = state

    return "break"


def v58_mouse_motion(self, event):
    state = getattr(self, "_v58_mouse_state", None)

    if not state:
        return "break"

    x, y = v58_canvas_xy(self, event)

    dx = abs(x - float(state["press_x"]))
    dy = abs(y - float(state["press_y"]))

    if dx > 5 or dy > 5:
        state["moved"] = True

    state["last_x"] = x
    state["last_y"] = y

    if state.get("empty"):
        return "break"

    if not state.get("moved"):
        return "break"

    item = v58_get_item_by_id(self, state.get("item_id"))

    if item is None:
        return "break"

    length = int(item.get("length", HIT_LENGTH_STEPS))
    x_step, lane, pair = v58_xy_to_grid(self, x, y, length=length)

    changed = (
        int(item.get("x_step", -999)) != int(x_step)
        or int(item.get("lane", -999)) != int(lane)
        or int(item.get("pair", -999)) != int(pair)
    )

    if changed:
        item["x_step"] = int(x_step)
        item["lane"] = int(lane)
        item["pair"] = int(pair)
        item["variation_bar"] = int(x_step) // 8
        item["variation_pos"] = int(x_step) % 8
        item["hit_slot"] = int(x_step) // int(HIT_SPACING_STEPS)
        item["fine_grid"] = bool(v58_fine_enabled(self))

        self.selected_id = int(item.get("id"))

        try:
            self.draw()
            self.refresh_panel()
        except Exception:
            pass

    return "break"


def v58_mouse_up(self, event):
    state = getattr(self, "_v58_mouse_state", None)
    self._v58_mouse_state = None

    if not state:
        return "break"

    x, y = v58_canvas_xy(self, event)

    if state.get("empty"):
        if not state.get("moved"):
            return v58_add_item_at_xy(self, x, y)

        print("[v140] drag depuis vide ignoré")
        return "break"

    item = v58_get_item_by_id(self, state.get("item_id"))

    if item is None:
        return "break"

    if not state.get("moved"):
        self.selected_id = int(item.get("id"))

        try:
            self.draw()
            self.refresh_panel()
        except Exception:
            pass

        try:
            self.audition_selected_case()
        except Exception as exc:
            print(f"[v140] audition clic note impossible : {exc}")

        v58_status(
            self,
            f"Slice {item.get('pair')} jouée | case {item.get('x_step')} | Suppr/clic droit pour supprimer."
        )

        print(
            f"[v140] click play/select | id={item.get('id')} | "
            f"step={item.get('x_step')} | len={item.get('length')} | pair={item.get('pair')}"
        )

        return "break"

    before = state.get("before") or {}
    after = dict(item)

    if before != after:
        try:
            self.record_correction("move_note_fine_grid", before=before, after=after)
        except Exception as exc:
            print(f"[v140] record_correction drag impossible : {exc}")

        v58_refresh_after_edit(self, "move_note_fine_grid")

        v58_status(
            self,
            f"Note déplacée : slice {after.get('pair')} | "
            f"case {before.get('x_step')} -> {after.get('x_step')}"
        )

        print(
            f"[v140] drag move | id={after.get('id')} | "
            f"step {before.get('x_step')} -> {after.get('x_step')} | "
            f"pair {before.get('pair')} -> {after.get('pair')}"
        )

        try:
            self.audition_selected_case()
        except Exception as exc:
            print(f"[v140] audition après drag impossible : {exc}")

    return "break"


def v58_right_click_delete(self, event):
    x, y = v58_canvas_xy(self, event)
    item = v58_find_item_at_xy(self, x, y)

    if item is None:
        v58_status(self, "Clic droit : aucune note sous la souris.")
        return "break"

    self.selected_id = int(item.get("id"))

    try:
        return v55_delete_item(self, item, reason="delete_note_right_click")
    except Exception:
        try:
            return v51_delete_item(self, item, reason="delete_note_right_click")
        except Exception:
            # fallback minimal
            item_id = int(item.get("id"))
            self.pattern = [
                it for it in getattr(self, "pattern", []) or []
                if int(it.get("id", -999999)) != item_id
            ]
            self.selected_id = self.pattern[0]["id"] if self.pattern else None
            v58_refresh_after_edit(self, "delete_note_right_click")
            return "break"


def v58_get_selected_item(self):
    return v58_get_item_by_id(self, getattr(self, "selected_id", None))


def v58_toggle_selected_length(self, event=None):
    item = v58_get_selected_item(self)

    if item is None:
        v58_status(self, "Aucune note sélectionnée pour changer la longueur.")
        return "break"

    before = dict(item)

    old_len = int(item.get("length", HIT_LENGTH_STEPS))
    new_len = 1 if old_len != 1 else int(HIT_LENGTH_STEPS)

    item["length"] = int(new_len)
    item["x_step"] = v58_clamp_x_step(self, item.get("x_step", 0), length=new_len)

    after = dict(item)

    try:
        self.record_correction("toggle_note_length", before=before, after=after)
    except Exception as exc:
        print(f"[v140] record_correction length impossible : {exc}")

    v58_refresh_after_edit(self, "toggle_note_length")

    v58_status(self, f"Longueur note : {old_len} -> {new_len} case(s).")
    print(f"[v140] length toggle | id={item.get('id')} | {old_len} -> {new_len}")

    return "break"


def v58_duplicate_selected_forward(self, event=None):
    item = v58_get_selected_item(self)

    if item is None:
        v58_status(self, "Ctrl+D : aucune note sélectionnée.")
        return "break"

    step_delta = 1 if v58_fine_enabled(self) else int(HIT_SPACING_STEPS)

    before = dict(item)
    length = int(item.get("length", HIT_LENGTH_STEPS))

    new_x = v58_clamp_x_step(self, int(item.get("x_step", 0)) + step_delta, length=length)

    if new_x == int(item.get("x_step", 0)):
        v58_status(self, "Ctrl+D : impossible, fin de grille.")
        return "break"

    new_item = dict(item)
    new_item["id"] = v58_next_item_id(self)
    new_item["x_step"] = int(new_x)
    new_item["variation_bar"] = int(new_x) // 8
    new_item["variation_pos"] = int(new_x) % 8
    new_item["hit_slot"] = int(new_x) // int(HIT_SPACING_STEPS)
    new_item["duplicated_from"] = int(item.get("id"))
    new_item["fine_grid"] = bool(v58_fine_enabled(self))

    self.pattern.append(new_item)
    self.selected_id = int(new_item["id"])

    try:
        self.record_correction("duplicate_note_fine_grid", before=before, after=dict(new_item))
    except Exception as exc:
        print(f"[v140] record_correction duplicate impossible : {exc}")

    v58_refresh_after_edit(self, "duplicate_note_fine_grid")

    v58_status(
        self,
        f"Dupliquée : case {item.get('x_step')} -> {new_x}"
        + (" | demi-grille" if v58_fine_enabled(self) else "")
    )

    try:
        self.audition_selected_case()
    except Exception as exc:
        print(f"[v140] audition duplicate impossible : {exc}")

    print(
        f"[v140] duplicate | old_id={item.get('id')} | new_id={new_item.get('id')} | "
        f"step {item.get('x_step')} -> {new_x} | len={length}"
    )

    return "break"


_old_v58_build_ui = SliceIndexTracker.build_ui


def v58_build_ui(self):
    _old_v58_build_ui(self)

    try:
        frame = tk.Frame(self.root, bg="#101820")
        frame.pack(fill="x", padx=14, pady=(0, 8))

        self.v58_fine_grid_var = tk.BooleanVar(value=V58_FINE_GRID_DEFAULT)
        self.v58_one_case_var = tk.BooleanVar(value=V58_ONE_CASE_DEFAULT)

        tk.Checkbutton(
            frame,
            text="édition demi-grille",
            variable=self.v58_fine_grid_var,
            bg="#101820",
            fg="#77f5b5",
            selectcolor="#30283f",
            activebackground="#101820",
            activeforeground="#77f5b5",
        ).pack(side="left", padx=8)

        tk.Checkbutton(
            frame,
            text="nouvelle note = 1 case",
            variable=self.v58_one_case_var,
            bg="#101820",
            fg="#f5d67b",
            selectcolor="#30283f",
            activebackground="#101820",
            activeforeground="#f5d67b",
        ).pack(side="left", padx=8)

        tk.Label(
            frame,
            text="v58 : demi-grille = placement sur 0,1,2,3… | L = longueur 1/2 cases.",
            bg="#101820",
            fg="#b9acc8",
        ).pack(side="left", padx=8)

    except Exception as exc:
        print(f"[v140] UI demi-grille impossible : {exc}")

    try:
        self.canvas.bind("<Button-1>", self.v58_mouse_down)
        self.canvas.bind("<B1-Motion>", self.v58_mouse_motion)
        self.canvas.bind("<ButtonRelease-1>", self.v58_mouse_up)
        self.canvas.bind("<Button-3>", self.v58_right_click_delete)

        self.root.bind("<Control-d>", self.duplicate_selected_forward)
        self.root.bind("<Control-D>", self.duplicate_selected_forward)
        self.root.bind_all("<Control-d>", self.duplicate_selected_forward)
        self.root.bind_all("<Control-D>", self.duplicate_selected_forward)

        self.root.bind("l", self.toggle_selected_length)
        self.root.bind("L", self.toggle_selected_length)
        self.canvas.bind("l", self.toggle_selected_length)
        self.canvas.bind("L", self.toggle_selected_length)
    except Exception as exc:
        print(f"[v140] bind demi-grille impossible : {exc}")

    v58_status(
        self,
        "v58 : édition demi-grille ON | clic note=play | drag=déplace case par case | L=longueur 1/2."
    )


SliceIndexTracker.v58_mouse_down = v58_mouse_down
SliceIndexTracker.v58_mouse_motion = v58_mouse_motion
SliceIndexTracker.v58_mouse_up = v58_mouse_up
SliceIndexTracker.v58_right_click_delete = v58_right_click_delete
SliceIndexTracker.toggle_selected_length = v58_toggle_selected_length
SliceIndexTracker.duplicate_selected_forward = v58_duplicate_selected_forward
SliceIndexTracker.build_ui = v58_build_ui

# Compatibilité :
# les handlers v55/v51 appellent des fonctions globales v55_xy_to_grid/v51_xy_to_grid.
# On les redirige vers la logique demi-grille au cas où un ancien bind reste actif.
try:
    v55_xy_to_grid = v58_xy_to_grid
    v55_clamp_x_step = v58_clamp_x_step
except Exception:
    pass

try:
    v51_xy_to_grid = v58_xy_to_grid
    v51_clamp_x_step = v58_clamp_x_step
except Exception:
    pass




# v60 import safety
try:
    Counter
except NameError:
    from collections import Counter

try:
    subprocess
except NameError:
    import subprocess

try:
    sys
except NameError:
    import sys

try:
    math
except NameError:
    import math

# ---------------------------------------------------------------------
# v59 EXTENSION : IA role-aware
# ---------------------------------------------------------------------

ROLE_MODEL_PATH = Path("dataset/learning/beat_role_model_v01.json")
ROLE_TRAINER_PATH = Path("pipeline/04_train_beat_roles_v03.py")

V59_FALLBACK_SNARE_STEPS = [6, 14, 22, 30]
V59_FALLBACK_KICK_STEPS = [0, 8, 16, 24]
V59_FALLBACK_HAT_STEPS = [2, 4, 10, 12, 18, 20, 26, 28]


def v59_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v59_band_energy(y, low, high):
    y = np.asarray(y, dtype=np.float32)

    if len(y) < 256:
        y = np.pad(y, (0, 256 - len(y)))

    n = min(len(y), 4096)
    chunk = y[:n] * np.hanning(n).astype(np.float32)

    mag = np.abs(np.fft.rfft(chunk)).astype(np.float32)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    mask = (freqs >= low) & (freqs <= high)

    if not np.any(mask):
        return 0.0

    return float(np.sum(mag[mask] ** 2))


def v59_classify_audio_role(y):
    y = np.asarray(y, dtype=np.float32)

    if len(y) == 0:
        return "other", {"kick": 0.0, "snare": 0.0, "hat": 0.0, "other": 1.0}

    attack = y[:min(len(y), int(SR * 0.500))]
    full = y[:min(len(y), int(SR * 0.500))]

    sub = v59_band_energy(attack, 35, 90)
    low = v59_band_energy(attack, 90, 250)
    lowmid = v59_band_energy(attack, 250, 700)
    mid = v59_band_energy(attack, 700, 2800)
    high = v59_band_energy(attack, 2800, 9000)
    air = v59_band_energy(attack, 9000, 16000)

    total = sub + low + lowmid + mid + high + air + 1e-9

    sub_r = sub / total
    low_r = low / total
    lowmid_r = lowmid / total
    mid_r = mid / total
    high_r = high / total
    air_r = air / total

    rms = float(np.sqrt(np.mean(full * full) + 1e-12))

    if len(attack) > 3:
        zcr = float(np.mean(np.abs(np.diff(np.signbit(attack).astype(np.float32)))))
    else:
        zcr = 0.0

    tail_start = min(len(y), int(SR * 0.520))
    tail_end = min(len(y), int(SR * 0.450))

    if tail_end > tail_start:
        tail = y[tail_start:tail_end]
        tail_rms = float(np.sqrt(np.mean(tail * tail) + 1e-12))
    else:
        tail_rms = 0.0

    tail_ratio = tail_rms / (rms + 1e-9)

    kick_score = (
        3.2 * sub_r
        + 2.5 * low_r
        + 0.7 * lowmid_r
        + 0.25 * rms
        - 1.0 * high_r
        - 0.9 * air_r
    )

    snare_score = (
        1.1 * lowmid_r
        + 2.1 * mid_r
        + 1.4 * high_r
        + 0.8 * air_r
        + 0.9 * zcr
        - 1.1 * sub_r
        - 0.7 * low_r
    )

    hat_score = (
        2.6 * high_r
        + 2.3 * air_r
        + 1.1 * zcr
        - 1.4 * sub_r
        - 1.0 * low_r
        - 0.35 * mid_r
        - 0.45 * tail_ratio
    )

    other_score = 0.5 * tail_ratio + 0.2 * rms

    scores = {
        "kick": float(kick_score),
        "snare": float(snare_score),
        "hat": float(hat_score),
        "other": float(other_score),
    }

    role = max(scores, key=scores.get)

    if scores[role] < 0.52:
        role = "other"

    return role, scores


def v59_rank_current_slices(self):
    """
    Analyse les slices du break courant et les range par rôle.
    """
    ranks = {
        "kick": [],
        "snare": [],
        "hat": [],
        "other": [],
        "all": [],
    }

    for pair in self.pair_values:
        pair = int(pair)

        try:
            y = self.get_audio(pair)
            role, scores = v59_classify_audio_role(y)
        except Exception as exc:
            print(f"[v140] analyse pair {pair} impossible : {exc}")
            role = "other"
            scores = {"kick": 0.0, "snare": 0.0, "hat": 0.0, "other": 1.0}

        info = {
            "pair": pair,
            "role": role,
            "scores": scores,
            "lane": self.pair_to_lane.get(pair, 0),
        }

        ranks["all"].append(info)

        for r in ("kick", "snare", "hat", "other"):
            role_info = dict(info)
            role_info["score"] = float(scores.get(r, 0.0))
            ranks[r].append(role_info)

    for r in ("kick", "snare", "hat", "other"):
        ranks[r] = sorted(ranks[r], key=lambda x: x["score"], reverse=True)

    print("")
    print("[v140] TOP ROLES CURRENT BREAK")
    for r in ("kick", "snare", "hat"):
        print(f"[v140] {r:5s}:", [(x["pair"], round(x["score"], 3)) for x in ranks[r][:6]])
    print("")

    return ranks


def v59_pick_pair_for_role(ranks, role, index=0, avoid=None):
    avoid = set(int(x) for x in (avoid or []))
    role = str(role)

    candidates = ranks.get(role) or ranks.get("other") or ranks.get("all") or []

    for item in candidates:
        pair = int(item["pair"])
        if pair in avoid:
            continue

        if index <= 0:
            return pair

        index -= 1

    if candidates:
        return int(candidates[0]["pair"])

    all_items = ranks.get("all") or []
    if all_items:
        return int(all_items[0]["pair"])

    return 0


def v59_load_role_model():
    if not ROLE_MODEL_PATH.exists():
        return None

    try:
        return json.loads(ROLE_MODEL_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[v140] lecture modèle rôle impossible : {exc}")
        return None


def v61_flexible_get(d, key, default=None):
    """
    Lecture robuste des clés JSON :
    - JSON stocke souvent les clés en string
    - certaines clés sont numériques : "6"
    - certaines clés sont textuelles : "kick"
    On ne tente int(key) QUE si c'est vraiment numérique.
    """
    if not isinstance(d, dict):
        return default

    s = str(key)

    if s in d:
        return d[s]

    if key in d:
        return d[key]

    try:
        i = int(s)
    except Exception:
        return default

    if i in d:
        return d[i]

    si = str(i)
    if si in d:
        return d[si]

    return default


def v59_count(bucket, key, default=0.0):
    value = v61_flexible_get(bucket, key, default)

    try:
        return float(value)
    except Exception:
        return float(default)


def v59_nested_count(bucket, a, b, default=0.0):
    sub = v61_flexible_get(bucket, a, {})

    if not isinstance(sub, dict):
        return float(default)

    value = v61_flexible_get(sub, b, default)

    try:
        return float(value)
    except Exception:
        return float(default)


def v59_best_length_for_step_role(model, safe, step, role):
    """
    Trouve la longueur la plus apprise pour ce rôle à cette case.
    """
    candidates = []

    break_bucket = model.get("breaks", {}).get(safe, {})
    global_bucket = model.get("global", {})

    for weight_mul, bucket in ((2.5, break_bucket), (1.0, global_bucket)):
        step_len = bucket.get("step_role_length_counts", {})
        step_data = step_len.get(str(step), {})
        role_data = step_data.get(str(role), {})

        if isinstance(role_data, dict):
            for length, count in role_data.items():
                try:
                    candidates.append((int(length), float(count) * weight_mul))
                except Exception:
                    pass

    if not candidates:
        return int(HIT_LENGTH_STEPS)

    counts = Counter()
    for length, count in candidates:
        counts[length] += count

    length = max(counts.items(), key=lambda kv: kv[1])[0]
    return max(1, min(8, int(length)))


def v59_role_plan_from_model(self, model):
    """
    Construit un plan :
        step -> role

    Priorité :
    - modèle du break courant
    - modèle global
    - fallback snare/kick/hat si peu de données
    """
    safe = str(self.project.get("safe", ""))

    break_bucket = model.get("breaks", {}).get(safe, {})
    global_bucket = model.get("global", {})

    break_steps = break_bucket.get("step_role_counts", {})
    global_steps = global_bucket.get("step_role_counts", {})

    plan = {}

    all_steps = set()
    all_steps.update(int(s) for s in break_steps.keys() if str(s).lstrip("-").isdigit())
    all_steps.update(int(s) for s in global_steps.keys() if str(s).lstrip("-").isdigit())

    for step in sorted(all_steps):
        if step < 0 or step > 31:
            continue

        role_scores = Counter()

        for role in ("kick", "snare", "hat", "other"):
            role_scores[role] += 3.0 * v59_nested_count(break_steps, step, role)
            role_scores[role] += 1.0 * v59_nested_count(global_steps, step, role)

        role, score = max(role_scores.items(), key=lambda kv: kv[1])

        if score <= 0:
            continue

        # On ignore les "other" faibles, mais on garde si vraiment appris.
        if role == "other" and score < 6.0:
            continue

        length = v59_best_length_for_step_role(model, safe, step, role)

        plan[int(step)] = {
            "role": role,
            "weight": float(score),
            "length": int(length),
            "source": "role_model",
        }

    # Si modèle trop pauvre, fallback musical.
    if len(plan) < 4:
        for step in V59_FALLBACK_SNARE_STEPS:
            plan[step] = {"role": "snare", "weight": 100.0, "length": int(HIT_LENGTH_STEPS), "source": "fallback_snare6"}

        for step in V59_FALLBACK_KICK_STEPS:
            plan.setdefault(step, {"role": "kick", "weight": 80.0, "length": int(HIT_LENGTH_STEPS), "source": "fallback_kick"})

        for step in V59_FALLBACK_HAT_STEPS:
            plan.setdefault(step, {"role": "hat", "weight": 40.0, "length": int(HIT_LENGTH_STEPS), "source": "fallback_hat"})

    return plan


def v59_make_role_note(self, step, role, pair, length, source):
    step = int(step)
    length = int(length)

    try:
        max_step = int(self.step_count) - length
    except Exception:
        max_step = 32 - length

    step = max(0, min(max_step, step))

    lane = self.pair_to_lane.get(int(pair), 0)

    return {
        "id": 0,
        "x_step": int(step),
        "lane": int(lane),
        "pair": int(pair),
        "length": int(length),
        "variation_bar": int(step) // 8,
        "variation_pos": int(step) % 8,
        "hit_slot": int(step) // int(HIT_SPACING_STEPS),
        "randomized": False,
        "ai_generated": True,
        "ai_model": "v59_role_aware",
        "learned_role": str(role),
        "role_plan_source": str(source),
    }


def v59_generate_role_aware(self, event=None):
    model = v59_load_role_model()

    if model is None:
        v59_status(self, "Aucun modèle rôle. Lance : python pipeline/04_train_beat_roles_v03.py")
        print("[v140] modèle rôle absent, fallback musical snare 6.")
        model = {
            "breaks": {},
            "global": {},
            "patterns_used": 0,
        }

    ranks = v59_rank_current_slices(self)
    plan = v59_role_plan_from_model(self, model)

    pattern = []
    role_use_index = Counter()

    # Les snares/kicks appris gagnent contre les hats en cas de collision.
    role_priority = {
        "snare": 100,
        "kick": 80,
        "hat": 50,
        "other": 20,
    }

    # Résolution collisions par case.
    by_step = {}

    for step, info in sorted(plan.items(), key=lambda kv: int(kv[0])):
        role = str(info["role"])
        length = int(info.get("length", HIT_LENGTH_STEPS))
        source = str(info.get("source", "role_model"))

        # Variation légère : alterne entre top 1 / top 2 du rôle.
        idx = role_use_index[role] % 2
        pair = v59_pick_pair_for_role(ranks, role, index=idx)
        role_use_index[role] += 1

        item = v59_make_role_note(self, step, role, pair, length, source)

        old = by_step.get(int(step))
        if old is None:
            by_step[int(step)] = item
            continue

        if role_priority.get(role, 0) > role_priority.get(old.get("learned_role", "other"), 0):
            by_step[int(step)] = item

    pattern = [by_step[k] for k in sorted(by_step.keys())]

    for i, item in enumerate(pattern):
        item["id"] = i

    try:
        self.stop_playhead()
        self.stop_audio()
    except Exception:
        pass

    self.looping = False
    self.pattern = pattern
    self.selected_id = self.pattern[0]["id"] if self.pattern else None

    try:
        self.draw()
        self.refresh_panel()
    except Exception:
        pass

    try:
        self.write_latest_pattern(reason="v59_generate_role_aware")
    except Exception as exc:
        print(f"[v140] write_latest_pattern impossible : {exc}")

    print("")
    print("[v140] ROLE-AWARE GENERATION")
    print("[v140] model patterns_used:", model.get("patterns_used", 0))
    for item in pattern:
        print(
            f"[v140] step {item['x_step']:02d}/{item['x_step'] + item['length'] - 1:02d} | "
            f"role={item['learned_role']:6s} | pair={item['pair']:02d} | {item['role_plan_source']}"
        )
    print("")

    v59_status(
        self,
        f"Generate IA Roles OK : {len(pattern)} notes | positions apprises par rôle."
    )

    return "break"


def v59_run_role_training(self, event=None):
    if not ROLE_TRAINER_PATH.exists():
        v59_status(self, "Trainer rôle introuvable : pipeline/04_train_beat_roles_v03.py")
        return "break"

    print("")
    print("[v140] TRAIN ROLE MODEL...")
    print("")

    try:
        result = subprocess.run(
            [sys.executable, str(ROLE_TRAINER_PATH)],
            cwd=str(Path(".").resolve()),
            text=True,
            capture_output=True,
            check=False,
        )

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:
            v59_status(self, "Training rôles échoué. Regarde le terminal.")
        else:
            v59_status(self, "Training rôles OK : beat_role_model_v01.json mis à jour.")

    except Exception as exc:
        print(f"[v140] training rôle impossible : {exc}")
        v59_status(self, f"Training rôles impossible : {exc}")

    return "break"


_old_v59_save = SliceIndexTracker.save


def v59_save(self):
    """
    Save normal + training du modèle rôle.
    """
    result = _old_v59_save(self)

    try:
        v59_run_role_training(self)
    except Exception as exc:
        print(f"[v140] auto-training rôle après save impossible : {exc}")

    return result


_old_v59_build_ui = SliceIndexTracker.build_ui


def v59_build_ui(self):
    _old_v59_build_ui(self)

    try:
        frame = tk.Frame(self.root, bg="#101820")
        frame.pack(fill="x", padx=14, pady=(0, 8))

        tk.Button(
            frame,
            text="Generate IA Roles",
            command=self.generate_role_aware,
            bg="#5a365f",
            fg="#f5eefe",
        ).pack(side="left", padx=8)

        tk.Button(
            frame,
            text="Train Roles",
            command=self.run_role_training,
            bg="#30513f",
            fg="#f5eefe",
        ).pack(side="left", padx=8)

        tk.Label(
            frame,
            text="v59 : apprend case→rôle musical, puis rôle→slice du break courant.",
            bg="#101820",
            fg="#b9acc8",
        ).pack(side="left", padx=8)

    except Exception as exc:
        print(f"[v140] UI role-aware impossible : {exc}")

    v59_status(self, "v59 : Generate IA Roles = positions de kick/snare/hat apprises, pas numéros de slices.")


SliceIndexTracker.generate_role_aware = v59_generate_role_aware
SliceIndexTracker.generate_ai_pattern = v59_generate_role_aware
SliceIndexTracker.run_role_training = v59_run_role_training
SliceIndexTracker.save = v59_save
SliceIndexTracker.build_ui = v59_build_ui



# ---------------------------------------------------------------------
# v62 EXTENSION : drag bord droit = resize note
# ---------------------------------------------------------------------

V62_DRAG_THRESHOLD_PX = 5
V62_RESIZE_EDGE_PX = 10
V62_MIN_NOTE_LENGTH = 1
V62_MAX_NOTE_LENGTH = 8


def v62_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v62_fine_enabled(self):
    try:
        return bool(self.v58_fine_grid_var.get())
    except Exception:
        return True


def v62_one_case_enabled(self):
    try:
        return bool(self.v58_one_case_var.get())
    except Exception:
        return False


def v62_canvas_xy(self, event):
    try:
        x = float(self.canvas.canvasx(event.x))
        y = float(self.canvas.canvasy(event.y))
    except Exception:
        x = float(event.x)
        y = float(event.y)

    return x, y


def v62_get_item_by_id(self, item_id):
    if item_id is None:
        return None

    for item in getattr(self, "pattern", []) or []:
        try:
            if int(item.get("id")) == int(item_id):
                return item
        except Exception:
            pass

    return None


def v62_next_item_id(self):
    ids = []

    for item in getattr(self, "pattern", []) or []:
        try:
            ids.append(int(item.get("id", -1)))
        except Exception:
            pass

    return max(ids) + 1 if ids else 0


def v62_note_rect(self, item):
    x_step = int(item.get("x_step", 0))
    length = int(item.get("length", HIT_LENGTH_STEPS))
    lane = int(item.get("lane", 0))

    x0 = float(self.left_width + x_step * self.step_width)
    x1 = float(self.left_width + (x_step + length) * self.step_width)
    y0 = float(lane * self.row_height)
    y1 = float((lane + 1) * self.row_height)

    return x0, y0, x1, y1


def v62_find_item_at_xy(self, x, y):
    """
    Retourne (item, zone)
    zone = "resize_right" si souris proche du bord droit
    zone = "body" sinon
    """
    if not getattr(self, "pattern", None):
        return None, None

    for item in reversed(self.pattern):
        try:
            x0, y0, x1, y1 = v62_note_rect(self, item)

            if y0 <= y <= y1 and x0 <= x <= x1:
                if abs(x - x1) <= V62_RESIZE_EDGE_PX:
                    return item, "resize_right"

                return item, "body"

            # Tolérance un peu à droite du bord pour attraper la poignée.
            if y0 <= y <= y1 and x1 < x <= x1 + V62_RESIZE_EDGE_PX:
                return item, "resize_right"

        except Exception:
            continue

    return None, None


def v62_clamp_length(self, x_step, length):
    try:
        step_count = int(self.step_count)
    except Exception:
        step_count = 32

    x_step = int(x_step)
    length = int(round(float(length)))

    max_len = max(V62_MIN_NOTE_LENGTH, min(V62_MAX_NOTE_LENGTH, step_count - x_step))
    length = max(V62_MIN_NOTE_LENGTH, min(max_len, length))

    return int(length)


def v62_clamp_x_step(self, x_step, length=None):
    if length is None:
        length = int(HIT_LENGTH_STEPS)

    try:
        max_x = int(self.step_count) - int(length)
    except Exception:
        max_x = 32 - int(length)

    x_step = int(round(float(x_step)))
    x_step = max(0, min(max_x, x_step))

    # Si demi-grille OFF, snap 0,2,4,6...
    if not v62_fine_enabled(self):
        x_step = int(round(x_step / float(HIT_SPACING_STEPS))) * int(HIT_SPACING_STEPS)
        x_step = max(0, min(max_x, x_step))

    return int(x_step)


def v62_xy_to_grid(self, x, y, length=None):
    if length is None:
        length = int(HIT_LENGTH_STEPS)

    raw_step = int(round((x - self.left_width) / float(self.step_width)))
    x_step = v62_clamp_x_step(self, raw_step, length=length)

    lane = int(y // float(self.row_height))
    lane = max(0, min(len(self.pair_values) - 1, lane))

    pair = int(self.pair_values[lane])

    return x_step, lane, pair


def v62_refresh_after_edit(self, reason):
    try:
        self.draw()
    except Exception:
        pass

    try:
        self.refresh_panel()
    except Exception:
        pass

    try:
        self.write_latest_pattern(reason=reason)
    except Exception:
        pass


def v62_add_item_at_xy(self, x, y):
    if x < self.left_width:
        return "break"

    length = 1 if v62_one_case_enabled(self) else int(HIT_LENGTH_STEPS)
    x_step, lane, pair = v62_xy_to_grid(self, x, y, length=length)

    item = {
        "id": v62_next_item_id(self),
        "x_step": int(x_step),
        "lane": int(lane),
        "pair": int(pair),
        "length": int(length),
        "variation_bar": int(x_step) // 8,
        "variation_pos": int(x_step) % 8,
        "hit_slot": int(x_step) // int(HIT_SPACING_STEPS),
        "randomized": False,
        "fine_grid": bool(v62_fine_enabled(self)),
    }

    self.pattern.append(item)
    self.selected_id = int(item["id"])

    try:
        self.record_correction("add_note_v62", before=None, after=dict(item))
    except Exception as exc:
        print(f"[v140] record_correction add impossible : {exc}")

    v62_refresh_after_edit(self, "add_note_v62")

    try:
        self.audition_selected_case()
    except Exception as exc:
        print(f"[v140] audition après ajout impossible : {exc}")

    v62_status(
        self,
        f"Note ajoutée : slice {pair} | case {x_step}"
        + (f"/{x_step + length - 1}" if length > 1 else "")
    )

    print(f"[v140] add | id={item['id']} | step={x_step} | len={length} | pair={pair}")

    return "break"


def v62_resize_item_to_x(self, item, x):
    """
    Calcule la longueur depuis le bord droit glissé.
    """
    x_step = int(item.get("x_step", 0))

    raw_right_step = int(round((x - self.left_width) / float(self.step_width)))
    new_len = raw_right_step - x_step

    # Si la souris est dans la première case, on force 1.
    new_len = max(1, new_len)

    new_len = v62_clamp_length(self, x_step, new_len)

    return new_len


def v62_mouse_down(self, event):
    try:
        self.canvas.focus_set()
    except Exception:
        pass

    x, y = v62_canvas_xy(self, event)
    item, zone = v62_find_item_at_xy(self, x, y)

    state = {
        "press_x": x,
        "press_y": y,
        "last_x": x,
        "last_y": y,
        "moved": False,
        "mode": "empty",
        "item_id": None,
        "before": None,
    }

    if item is not None:
        self.selected_id = int(item.get("id"))
        state["item_id"] = int(item.get("id"))
        state["before"] = dict(item)
        state["mode"] = "resize" if zone == "resize_right" else "move"

        try:
            self.draw()
            self.refresh_panel()
        except Exception:
            pass

        print(
            f"[v140] mouse down {state['mode']} | id={item.get('id')} | "
            f"step={item.get('x_step')} | len={item.get('length')} | pair={item.get('pair')}"
        )
    else:
        print("[v140] mouse down empty")

    self._v62_mouse_state = state

    return "break"


def v62_mouse_motion(self, event):
    state = getattr(self, "_v62_mouse_state", None)

    if not state:
        return "break"

    x, y = v62_canvas_xy(self, event)

    dx = abs(x - float(state["press_x"]))
    dy = abs(y - float(state["press_y"]))

    if dx > V62_DRAG_THRESHOLD_PX or dy > V62_DRAG_THRESHOLD_PX:
        state["moved"] = True

    state["last_x"] = x
    state["last_y"] = y

    mode = state.get("mode")

    if mode == "empty":
        return "break"

    item = v62_get_item_by_id(self, state.get("item_id"))

    if item is None:
        return "break"

    if mode == "resize":
        if not state.get("moved"):
            return "break"

        old_len = int(item.get("length", HIT_LENGTH_STEPS))
        new_len = v62_resize_item_to_x(self, item, x)

        if new_len != old_len:
            item["length"] = int(new_len)

            try:
                self.draw()
                self.refresh_panel()
            except Exception:
                pass

        return "break"

    if mode == "move":
        if not state.get("moved"):
            return "break"

        length = int(item.get("length", HIT_LENGTH_STEPS))
        x_step, lane, pair = v62_xy_to_grid(self, x, y, length=length)

        changed = (
            int(item.get("x_step", -999)) != int(x_step)
            or int(item.get("lane", -999)) != int(lane)
            or int(item.get("pair", -999)) != int(pair)
        )

        if changed:
            item["x_step"] = int(x_step)
            item["lane"] = int(lane)
            item["pair"] = int(pair)
            item["variation_bar"] = int(x_step) // 8
            item["variation_pos"] = int(x_step) % 8
            item["hit_slot"] = int(x_step) // int(HIT_SPACING_STEPS)
            item["fine_grid"] = bool(v62_fine_enabled(self))

            self.selected_id = int(item.get("id"))

            try:
                self.draw()
                self.refresh_panel()
            except Exception:
                pass

        return "break"

    return "break"


def v62_mouse_up(self, event):
    state = getattr(self, "_v62_mouse_state", None)
    self._v62_mouse_state = None

    if not state:
        return "break"

    x, y = v62_canvas_xy(self, event)
    mode = state.get("mode")

    if mode == "empty":
        if not state.get("moved"):
            return v62_add_item_at_xy(self, x, y)

        print("[v140] drag depuis vide ignoré")
        return "break"

    item = v62_get_item_by_id(self, state.get("item_id"))

    if item is None:
        return "break"

    before = state.get("before") or {}
    after = dict(item)

    if mode == "resize":
        if not state.get("moved"):
            self.selected_id = int(item.get("id"))

            try:
                self.audition_selected_case()
            except Exception as exc:
                print(f"[v140] audition clic resize impossible : {exc}")

            v62_status(
                self,
                f"Slice {item.get('pair')} jouée | bord droit = resize | longueur {item.get('length')} case(s)."
            )
            return "break"

        if before != after:
            try:
                self.record_correction("resize_note_edge", before=before, after=after)
            except Exception as exc:
                print(f"[v140] record_correction resize impossible : {exc}")

            v62_refresh_after_edit(self, "resize_note_edge")

            v62_status(
                self,
                f"Note redimensionnée : {before.get('length')} → {after.get('length')} case(s)."
            )

            print(
                f"[v140] resize | id={after.get('id')} | "
                f"step={after.get('x_step')} | len {before.get('length')} -> {after.get('length')}"
            )

            try:
                self.audition_selected_case()
            except Exception as exc:
                print(f"[v140] audition après resize impossible : {exc}")

        return "break"

    if mode == "move":
        if not state.get("moved"):
            self.selected_id = int(item.get("id"))

            try:
                self.draw()
                self.refresh_panel()
            except Exception:
                pass

            try:
                self.audition_selected_case()
            except Exception as exc:
                print(f"[v140] audition clic note impossible : {exc}")

            v62_status(
                self,
                f"Slice {item.get('pair')} jouée | case {item.get('x_step')} | bord droit pour resize."
            )
            return "break"

        if before != after:
            try:
                self.record_correction("move_note_v62", before=before, after=after)
            except Exception as exc:
                print(f"[v140] record_correction move impossible : {exc}")

            v62_refresh_after_edit(self, "move_note_v62")

            v62_status(
                self,
                f"Note déplacée : case {before.get('x_step')} → {after.get('x_step')}"
            )

            print(
                f"[v140] move | id={after.get('id')} | "
                f"step {before.get('x_step')} -> {after.get('x_step')} | "
                f"pair {before.get('pair')} -> {after.get('pair')}"
            )

            try:
                self.audition_selected_case()
            except Exception as exc:
                print(f"[v140] audition après move impossible : {exc}")

        return "break"

    return "break"


def v62_right_click_delete(self, event):
    x, y = v62_canvas_xy(self, event)
    item, zone = v62_find_item_at_xy(self, x, y)

    if item is None:
        v62_status(self, "Clic droit : aucune note sous la souris.")
        return "break"

    self.selected_id = int(item.get("id"))

    # Réutilise les delete existants si disponibles.
    for fn_name in ("v55_delete_item", "v51_delete_item", "v49_delete_item"):
        fn = globals().get(fn_name)
        if callable(fn):
            return fn(self, item, reason="delete_note_right_click")

    before = dict(item)
    item_id = int(item.get("id"))

    self.pattern = [
        it for it in getattr(self, "pattern", []) or []
        if int(it.get("id", -999999)) != item_id
    ]
    self.selected_id = self.pattern[0]["id"] if self.pattern else None

    try:
        self.record_correction("delete_note_right_click", before=before, after=None)
    except Exception:
        pass

    v62_refresh_after_edit(self, "delete_note_right_click")
    v62_status(self, f"Note supprimée : slice {before.get('pair')}")

    return "break"


def v62_toggle_selected_length(self, event=None):
    item = v62_get_item_by_id(self, getattr(self, "selected_id", None))

    if item is None:
        v62_status(self, "Aucune note sélectionnée.")
        return "break"

    before = dict(item)
    old_len = int(item.get("length", HIT_LENGTH_STEPS))
    new_len = 1 if old_len != 1 else int(HIT_LENGTH_STEPS)
    new_len = v62_clamp_length(self, item.get("x_step", 0), new_len)

    item["length"] = int(new_len)
    after = dict(item)

    try:
        self.record_correction("toggle_note_length_v62", before=before, after=after)
    except Exception as exc:
        print(f"[v140] record_correction toggle length impossible : {exc}")

    v62_refresh_after_edit(self, "toggle_note_length_v62")
    v62_status(self, f"Longueur note : {old_len} → {new_len} case(s).")

    print(f"[v140] toggle length | id={item.get('id')} | {old_len} -> {new_len}")

    return "break"


_old_v62_draw = SliceIndexTracker.draw


def v62_draw(self):
    """
    Ajoute une petite poignée visible sur le bord droit des notes.
    """
    result = _old_v62_draw(self)

    try:
        for item in getattr(self, "pattern", []) or []:
            x0, y0, x1, y1 = v62_note_rect(self, item)

            handle_w = 5
            pad_y = 7

            self.canvas.create_rectangle(
                x1 - handle_w,
                y0 + pad_y,
                x1,
                y1 - pad_y,
                fill="#f5d67b",
                outline="",
                tags=("v62_resize_handle",),
            )
    except Exception as exc:
        print(f"[v140] draw handles impossible : {exc}")

    return result


_old_v62_build_ui = SliceIndexTracker.build_ui


def v62_build_ui(self):
    _old_v62_build_ui(self)

    try:
        self.canvas.bind("<Button-1>", self.v62_mouse_down)
        self.canvas.bind("<B1-Motion>", self.v62_mouse_motion)
        self.canvas.bind("<ButtonRelease-1>", self.v62_mouse_up)
        self.canvas.bind("<Button-3>", self.v62_right_click_delete)

        self.root.bind("l", self.toggle_selected_length)
        self.root.bind("L", self.toggle_selected_length)
        self.canvas.bind("l", self.toggle_selected_length)
        self.canvas.bind("L", self.toggle_selected_length)
    except Exception as exc:
        print(f"[v140] bind resize impossible : {exc}")

    try:
        frame = tk.Frame(self.root, bg="#101820")
        frame.pack(fill="x", padx=14, pady=(0, 8))

        tk.Label(
            frame,
            text="v62 : bord droit jaune = resize | glisse vers la gauche pour 2→1 case | L alterne 1/2.",
            bg="#101820",
            fg="#f5d67b",
        ).pack(side="left", padx=8)
    except Exception as exc:
        print(f"[v140] UI resize impossible : {exc}")

    v62_status(
        self,
        "v62 : drag centre=déplacer | drag bord droit jaune=rétrécir/rallonger | L=1/2 cases."
    )


SliceIndexTracker.v62_mouse_down = v62_mouse_down
SliceIndexTracker.v62_mouse_motion = v62_mouse_motion
SliceIndexTracker.v62_mouse_up = v62_mouse_up
SliceIndexTracker.v62_right_click_delete = v62_right_click_delete
SliceIndexTracker.toggle_selected_length = v62_toggle_selected_length
SliceIndexTracker.draw = v62_draw
SliceIndexTracker.build_ui = v62_build_ui



# ---------------------------------------------------------------------
# v63 EXTENSION : Generate IA safe, pas n'importe quoi
# ---------------------------------------------------------------------

ROLE_TRAINER_PATH = Path("pipeline/04_train_beat_roles_v04_strict.py")
ROLE_MODEL_PATH = Path("dataset/learning/beat_role_model_v01.json")

V63_SNARE_STEPS = [6, 14, 22, 30]
V63_KICK_STEPS = [0, 8, 16, 24]
V63_HAT_STEPS = [2, 4, 10, 12, 18, 20, 26, 28]
V63_GHOST_STEPS = [5, 13, 21, 29]


def v63_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v63_load_model():
    if not ROLE_MODEL_PATH.exists():
        return None

    try:
        return json.loads(ROLE_MODEL_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[v140] lecture modèle strict impossible : {exc}")
        return None


def v63_rank_slices(self):
    if "v59_rank_current_slices" in globals():
        return v59_rank_current_slices(self)

    if "v56_rank_slices" in globals():
        old = v56_rank_slices(self)
        return {
            "kick": old.get("kick", []),
            "snare": old.get("snare", []),
            "hat": old.get("hat", []),
            "other": old.get("all", []),
            "all": old.get("all", []),
        }

    raise RuntimeError("Aucune fonction de ranking slices disponible.")


def v63_pick_pair(ranks, role, index=0, avoid=None):
    avoid = set(int(x) for x in (avoid or []))
    role = str(role)

    candidates = ranks.get(role) or ranks.get("all") or []

    for item in candidates:
        pair = int(item.get("pair", 0))

        if pair in avoid:
            continue

        if index <= 0:
            return pair

        index -= 1

    if candidates:
        return int(candidates[0].get("pair", 0))

    all_items = ranks.get("all") or []
    if all_items:
        return int(all_items[0].get("pair", 0))

    return 0


def v63_make_note(self, step, pair, length, role, label):
    step = int(step)
    length = int(length)

    try:
        max_step = int(self.step_count) - length
    except Exception:
        max_step = 32 - length

    step = max(0, min(max_step, step))

    lane = self.pair_to_lane.get(int(pair), 0)

    return {
        "id": 0,
        "x_step": int(step),
        "lane": int(lane),
        "pair": int(pair),
        "length": int(length),
        "variation_bar": int(step) // 8,
        "variation_pos": int(step) % 8,
        "hit_slot": int(step) // int(HIT_SPACING_STEPS),
        "randomized": False,
        "ai_generated": True,
        "ai_model": "v63_safe_role_ai",
        "learned_role": str(role),
        "prior_label": str(label),
    }


def v63_add_or_replace(by_step, item):
    priority = {
        "snare_anchor": 100,
        "kick_anchor": 80,
        "ghost_snare": 55,
        "hat_fill": 40,
        "other": 10,
    }

    step = int(item["x_step"])
    old = by_step.get(step)

    if old is None:
        by_step[step] = item
        return

    old_p = priority.get(old.get("prior_label", "other"), 0)
    new_p = priority.get(item.get("prior_label", "other"), 0)

    if new_p >= old_p:
        by_step[step] = item


def v63_generate_safe_role_ai(self, event=None):
    """
    Generate IA safe :
    - ne dépend pas d'un modèle sale
    - impose une grammaire breakbeat propre
    - utilise les rôles pour choisir les bonnes slices du break courant
    """
    try:
        self.stop_playhead()
        self.stop_audio()
    except Exception:
        pass

    model = v63_load_model()
    ranks = v63_rank_slices(self)

    # Picks robustes.
    main_snare = v63_pick_pair(ranks, "snare", 0)
    alt_snare = v63_pick_pair(ranks, "snare", 1, avoid=[main_snare])

    main_kick = v63_pick_pair(ranks, "kick", 0, avoid=[main_snare, alt_snare])
    alt_kick = v63_pick_pair(ranks, "kick", 1, avoid=[main_snare, alt_snare, main_kick])

    main_hat = v63_pick_pair(ranks, "hat", 0, avoid=[main_snare, alt_snare, main_kick, alt_kick])
    alt_hat = v63_pick_pair(ranks, "hat", 1, avoid=[main_snare, alt_snare, main_kick, alt_kick, main_hat])

    by_step = {}

    # 1. Snares fixes : le cœur du break.
    for i, step in enumerate(V63_SNARE_STEPS):
        pair = main_snare if i % 2 == 0 else alt_snare
        item = v63_make_note(self, step, pair, 2, "snare", "snare_anchor")
        v63_add_or_replace(by_step, item)

    # 2. Kicks fixes : départs / relances.
    for i, step in enumerate(V63_KICK_STEPS):
        pair = main_kick if i % 2 == 0 else alt_kick
        item = v63_make_note(self, step, pair, 2, "kick", "kick_anchor")
        v63_add_or_replace(by_step, item)

    # 3. Hats propres dans les trous.
    for i, step in enumerate(V63_HAT_STEPS):
        if step in by_step:
            continue

        pair = main_hat if i % 2 == 0 else alt_hat
        item = v63_make_note(self, step, pair, 1, "hat", "hat_fill")
        v63_add_or_replace(by_step, item)

    # 4. Ghost snares courtes juste avant les snares.
    # On les met courtes pour éviter le bazar.
    for i, step in enumerate(V63_GHOST_STEPS):
        if step in by_step:
            continue

        pair = alt_snare if i % 2 == 0 else main_snare
        item = v63_make_note(self, step, pair, 1, "snare", "ghost_snare")
        v63_add_or_replace(by_step, item)

    pattern = [by_step[k] for k in sorted(by_step.keys())]

    for i, item in enumerate(pattern):
        item["id"] = i

    self.looping = False
    self.pattern = pattern
    self.selected_id = self.pattern[0]["id"] if self.pattern else None

    try:
        self.draw()
        self.refresh_panel()
    except Exception:
        pass

    try:
        self.write_latest_pattern(reason="v63_generate_safe_role_ai")
    except Exception as exc:
        print(f"[v140] write_latest_pattern impossible : {exc}")

    print("")
    print("[v140] SAFE ROLE AI GENERATED")
    print("[v140] model strict:", "yes" if model else "no")
    print("[v140] snare main/alt:", main_snare, alt_snare)
    print("[v140] kick  main/alt:", main_kick, alt_kick)
    print("[v140] hat   main/alt:", main_hat, alt_hat)
    print("[v140] pattern:")
    for item in pattern:
        print(
            f"[v140] step {item['x_step']:02d}/{item['x_step'] + item['length'] - 1:02d} | "
            f"role={item['learned_role']:6s} | pair={item['pair']:02d} | {item['prior_label']}"
        )
    print("")

    v63_status(
        self,
        "Generate IA Safe OK : snares/kicks/hats placés proprement. Corrige puis Save validation."
    )

    return "break"


def v63_run_strict_training(self, event=None):
    if not ROLE_TRAINER_PATH.exists():
        v63_status(self, "Trainer strict introuvable : pipeline/04_train_beat_roles_v04_strict.py")
        return "break"

    print("")
    print("[v140] TRAIN STRICT ROLE MODEL...")
    print("")

    try:
        result = subprocess.run(
            [sys.executable, str(ROLE_TRAINER_PATH)],
            cwd=str(Path(".").resolve()),
            text=True,
            capture_output=True,
            check=False,
        )

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:
            v63_status(self, "Training strict échoué. Regarde le terminal.")
        else:
            v63_status(self, "Training strict OK : modèle rôle propre mis à jour.")

    except Exception as exc:
        print(f"[v140] training strict impossible : {exc}")
        v63_status(self, f"Training strict impossible : {exc}")

    return "break"


# Important :
# L'ancien v59_save appelle le nom global v59_run_role_training.
# On le redirige vers le trainer strict v04.
def v59_run_role_training(self, event=None):
    return v63_run_strict_training(self, event=event)


_old_v63_build_ui = SliceIndexTracker.build_ui


def v63_build_ui(self):
    _old_v63_build_ui(self)

    try:
        frame = tk.Frame(self.root, bg="#101820")
        frame.pack(fill="x", padx=14, pady=(0, 8))

        tk.Button(
            frame,
            text="Generate IA Safe",
            command=self.generate_safe_role_ai,
            bg="#6b3d67",
            fg="#f5eefe",
        ).pack(side="left", padx=8)

        tk.Button(
            frame,
            text="Train Strict",
            command=self.run_role_training,
            bg="#30513f",
            fg="#f5eefe",
        ).pack(side="left", padx=8)

        tk.Label(
            frame,
            text="v63 : Generate IA n'apprend plus ses brouillons. Structure safe kick/snare/hat.",
            bg="#101820",
            fg="#f5d67b",
        ).pack(side="left", padx=8)

    except Exception as exc:
        print(f"[v140] UI safe impossible : {exc}")

    v63_status(
        self,
        "v63 : Generate IA Safe = base propre. Save validation = training strict."
    )


SliceIndexTracker.generate_safe_role_ai = v63_generate_safe_role_ai
SliceIndexTracker.generate_role_aware = v63_generate_safe_role_ai
SliceIndexTracker.generate_ai_pattern = v63_generate_safe_role_ai
SliceIndexTracker.run_role_training = v63_run_strict_training
SliceIndexTracker.build_ui = v63_build_ui



# ---------------------------------------------------------------------
# v65 EXTENSION : grille de rôles cross-song
# ---------------------------------------------------------------------

V65_TEMPLATE_PATH = Path("dataset/learning/role_grid_template_v01.json")


def v65_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v65_band_energy(y, low, high):
    y = np.asarray(y, dtype=np.float32)

    if len(y) < 256:
        y = np.pad(y, (0, 256 - len(y)))

    n = min(len(y), 4096)
    chunk = y[:n] * np.hanning(n).astype(np.float32)

    mag = np.abs(np.fft.rfft(chunk)).astype(np.float32)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)

    mask = (freqs >= low) & (freqs <= high)

    if not np.any(mask):
        return 0.0

    return float(np.sum(mag[mask] ** 2))


def v65_classify_slice(y):
    y = np.asarray(y, dtype=np.float32)

    if len(y) == 0:
        return {
            "kick": 0.0,
            "snare": 0.0,
            "hat": 0.0,
            "ghost_snare": 0.0,
            "other": 1.0,
        }

    attack = y[:min(len(y), int(SR * 0.500))]
    full = y[:min(len(y), int(SR * 0.500))]

    sub = v65_band_energy(attack, 35, 90)
    low = v65_band_energy(attack, 90, 250)
    lowmid = v65_band_energy(attack, 250, 700)
    mid = v65_band_energy(attack, 700, 2800)
    high = v65_band_energy(attack, 2800, 9000)
    air = v65_band_energy(attack, 9000, 16000)

    total = sub + low + lowmid + mid + high + air + 1e-9

    sub_r = sub / total
    low_r = low / total
    lowmid_r = lowmid / total
    mid_r = mid / total
    high_r = high / total
    air_r = air / total

    rms = float(np.sqrt(np.mean(full * full) + 1e-12))

    if len(attack) > 3:
        zcr = float(np.mean(np.abs(np.diff(np.signbit(attack).astype(np.float32)))))
    else:
        zcr = 0.0

    tail_start = min(len(y), int(SR * 0.520))
    tail_end = min(len(y), int(SR * 0.450))

    if tail_end > tail_start:
        tail = y[tail_start:tail_end]
        tail_rms = float(np.sqrt(np.mean(tail * tail) + 1e-12))
    else:
        tail_rms = 0.0

    tail_ratio = tail_rms / (rms + 1e-9)

    kick_score = (
        3.4 * sub_r
        + 2.6 * low_r
        + 0.6 * lowmid_r
        + 0.25 * rms
        - 1.0 * high_r
        - 0.9 * air_r
    )

    snare_score = (
        1.2 * lowmid_r
        + 2.2 * mid_r
        + 1.5 * high_r
        + 0.8 * air_r
        + 0.9 * zcr
        - 1.1 * sub_r
        - 0.7 * low_r
    )

    hat_score = (
        2.7 * high_r
        + 2.4 * air_r
        + 1.1 * zcr
        - 1.4 * sub_r
        - 1.0 * low_r
        - 0.35 * mid_r
        - 0.45 * tail_ratio
    )

    ghost_score = (
        0.85 * snare_score
        + 0.35 * high_r
        - 0.25 * rms
    )

    other_score = 0.4 * tail_ratio + 0.2 * rms

    return {
        "kick": float(kick_score),
        "snare": float(snare_score),
        "hat": float(hat_score),
        "ghost_snare": float(ghost_score),
        "other": float(other_score),
    }


def v65_rank_slices(self):
    ranks = {
        "kick": [],
        "snare": [],
        "hat": [],
        "ghost_snare": [],
        "other": [],
        "all": [],
    }

    for pair in self.pair_values:
        pair = int(pair)

        try:
            y = self.get_audio(pair)
            scores = v65_classify_slice(y)
        except Exception as exc:
            print(f"[v140] analyse slice {pair} impossible : {exc}")
            scores = {
                "kick": 0.0,
                "snare": 0.0,
                "hat": 0.0,
                "ghost_snare": 0.0,
                "other": 1.0,
            }

        best_role = max(scores, key=scores.get)

        item = {
            "pair": pair,
            "scores": scores,
            "role": best_role,
            "lane": self.pair_to_lane.get(pair, 0),
        }

        ranks["all"].append(item)

        for role in ("kick", "snare", "hat", "ghost_snare", "other"):
            role_item = dict(item)
            role_item["score"] = float(scores.get(role, 0.0))
            ranks[role].append(role_item)

    for role in ("kick", "snare", "hat", "ghost_snare", "other"):
        ranks[role] = sorted(ranks[role], key=lambda x: x["score"], reverse=True)

    print("")
    print("[v140] TOP SLICES PAR ROLE — break courant")
    for role in ("kick", "snare", "hat", "ghost_snare"):
        print(f"[v140] {role:12s}:", [(x["pair"], round(x["score"], 3)) for x in ranks[role][:8]])
    print("")

    return ranks


def v65_load_template():
    if not V65_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template introuvable : {V65_TEMPLATE_PATH}")

    return json.loads(V65_TEMPLATE_PATH.read_text(encoding="utf-8"))


def v65_weighted_pick(rng, candidates, top_n=6, temperature=0.70, avoid=None):
    avoid = set(int(x) for x in (avoid or []))
    pool = []

    for item in candidates:
        pair = int(item.get("pair", 0))

        if pair in avoid:
            continue

        pool.append(item)

        if len(pool) >= top_n:
            break

    if not pool:
        pool = list(candidates[:top_n])

    if not pool:
        return 0

    scores = np.array([float(x.get("score", 0.0)) for x in pool], dtype=np.float64)
    scores = scores - scores.min()
    scores = scores + 0.05

    temperature = max(0.05, float(temperature))
    weights = np.power(scores, 1.0 / temperature)
    weights = weights / weights.sum()

    idx = int(rng.choice(np.arange(len(pool)), p=weights))

    return int(pool[idx].get("pair", 0))


def v65_make_note(self, step, pair, length, role, strength):
    step = int(step)
    length = int(length)

    try:
        step_count = int(self.step_count)
    except Exception:
        step_count = 32

    max_step = step_count - length
    step = max(0, min(max_step, step))

    lane = self.pair_to_lane.get(int(pair), 0)

    return {
        "id": 0,
        "x_step": int(step),
        "lane": int(lane),
        "pair": int(pair),
        "length": int(length),
        "variation_bar": int(step) // 8,
        "variation_pos": int(step) % 8,
        "hit_slot": int(step) // int(HIT_SPACING_STEPS),
        "randomized": True,
        "ai_generated": True,
        "ai_model": "v65_cross_song_role_template",
        "learned_role": str(role),
        "role_template_step": int(step),
        "role_template_strength": str(strength),
        "cross_song_role": True
    }


def v65_add_note(by_step, item):
    priority = {
        "anchor": 100,
        "ghost": 55,
        "fill": 40,
        "other": 10,
    }

    step = int(item["x_step"])
    old = by_step.get(step)

    if old is None:
        by_step[step] = item
        return

    old_p = priority.get(old.get("role_template_strength", "other"), 0)
    new_p = priority.get(item.get("role_template_strength", "other"), 0)

    if new_p >= old_p:
        by_step[step] = item


def v65_generate_from_role_template(self, event=None):
    """
    Génère depuis une grille universelle :
        step -> role

    Puis mappe :
        role -> meilleure slice du break courant
    """
    try:
        self.stop_playhead()
        self.stop_audio()
    except Exception:
        pass

    try:
        template = v65_load_template()
    except Exception as exc:
        v65_status(self, f"Template rôle introuvable : {exc}")
        print(f"[v140] template introuvable : {exc}")
        return "break"

    rng = np.random.default_rng(int(time.time_ns() % (2**32)))
    ranks = v65_rank_slices(self)

    try:
        temperature = float(self.v65_experiment_var.get())
    except Exception:
        temperature = 0.70

    try:
        hat_density = float(self.v65_hat_density_var.get())
    except Exception:
        hat_density = 0.75

    try:
        ghost_density = float(self.v65_ghost_density_var.get())
    except Exception:
        ghost_density = 0.55

    by_step = {}
    recent_pairs = {
        "kick": [],
        "snare": [],
        "hat": [],
        "ghost_snare": [],
    }

    roles = template.get("roles", {})

    for role, events in roles.items():
        role = str(role)

        if not isinstance(events, list):
            continue

        for event in events:
            step = int(event.get("step", 0))
            length = int(event.get("length", 1))
            strength = str(event.get("strength", "fill"))

            if role == "hat" and strength == "fill":
                if rng.random() > hat_density:
                    continue

            if role == "ghost_snare":
                if rng.random() > ghost_density:
                    continue

            pick_role = role
            if role == "ghost_snare":
                pick_role = "ghost_snare"

            avoid = recent_pairs.get(pick_role, [])[-1:]
            pair = v65_weighted_pick(
                rng,
                ranks.get(pick_role, ranks.get("snare", [])),
                top_n=8,
                temperature=temperature,
                avoid=avoid,
            )

            recent_pairs.setdefault(pick_role, []).append(pair)

            item = v65_make_note(
                self,
                step=step,
                pair=pair,
                length=length,
                role=role,
                strength=strength,
            )

            v65_add_note(by_step, item)

    pattern = [by_step[k] for k in sorted(by_step.keys())]

    for i, item in enumerate(pattern):
        item["id"] = i

    self.looping = False
    self.pattern = pattern
    self.selected_id = self.pattern[0]["id"] if self.pattern else None

    try:
        self.draw()
        self.refresh_panel()
    except Exception:
        pass

    try:
        self.write_latest_pattern(reason="v65_generate_cross_song_role_template")
    except Exception as exc:
        print(f"[v140] write_latest_pattern impossible : {exc}")

    print("")
    print("[v140] CROSS-SONG ROLE TEMPLATE GENERATED")
    print("[v140] template:", V65_TEMPLATE_PATH)
    print("[v140] temperature:", temperature, "| hats:", hat_density, "| ghosts:", ghost_density)
    print("[v140] pattern:")
    for item in pattern:
        end = int(item["x_step"]) + int(item["length"]) - 1
        print(
            f"[v140] step {item['x_step']:02d}/{end:02d} | "
            f"role={item['learned_role']:12s} | pair={item['pair']:02d} | "
            f"{item['role_template_strength']}"
        )
    print("")

    v65_status(
        self,
        "v65 : grille cross-song générée. Kick=7/15/23/31, snare=6/14/22/30."
    )

    return "break"


_old_v65_build_ui = SliceIndexTracker.build_ui


def v65_build_ui(self):
    _old_v65_build_ui(self)

    try:
        frame = tk.Frame(self.root, bg="#101820")
        frame.pack(fill="x", padx=14, pady=(0, 8))

        self.v65_experiment_var = tk.DoubleVar(value=0.70)
        self.v65_hat_density_var = tk.DoubleVar(value=0.75)
        self.v65_ghost_density_var = tk.DoubleVar(value=0.55)

        tk.Button(
            frame,
            text="Generate Cross-Song",
            command=self.generate_cross_song_template,
            bg="#6b3d67",
            fg="#f5eefe",
        ).pack(side="left", padx=8)

        tk.Label(
            frame,
            text="v65 : snare=6/14/22/30 | kick=7/15/23/31 | apprend les rôles cross-sons.",
            bg="#101820",
            fg="#f5d67b",
        ).pack(side="left", padx=8)

        tk.Label(
            frame,
            text="exp",
            bg="#101820",
            fg="#b9acc8",
        ).pack(side="left", padx=(12, 2))

        tk.Scale(
            frame,
            from_=0.25,
            to=1.25,
            resolution=0.05,
            orient="horizontal",
            variable=self.v65_experiment_var,
            length=90,
            bg="#101820",
            fg="#b9acc8",
            troughcolor="#30283f",
            highlightthickness=0,
        ).pack(side="left", padx=2)

        tk.Label(
            frame,
            text="hats",
            bg="#101820",
            fg="#b9acc8",
        ).pack(side="left", padx=(12, 2))

        tk.Scale(
            frame,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient="horizontal",
            variable=self.v65_hat_density_var,
            length=80,
            bg="#101820",
            fg="#b9acc8",
            troughcolor="#30283f",
            highlightthickness=0,
        ).pack(side="left", padx=2)

        tk.Label(
            frame,
            text="ghosts",
            bg="#101820",
            fg="#b9acc8",
        ).pack(side="left", padx=(12, 2))

        tk.Scale(
            frame,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient="horizontal",
            variable=self.v65_ghost_density_var,
            length=80,
            bg="#101820",
            fg="#b9acc8",
            troughcolor="#30283f",
            highlightthickness=0,
        ).pack(side="left", padx=2)

    except Exception as exc:
        print(f"[v140] UI impossible : {exc}")

    v65_status(
        self,
        "v65 : Generate Cross-Song = rôles fixes cross-break, slices choisies selon le break courant."
    )


SliceIndexTracker.generate_cross_song_template = v65_generate_from_role_template
SliceIndexTracker.generate_safe_experiment = v65_generate_from_role_template
SliceIndexTracker.generate_safe_role_ai = v65_generate_from_role_template
SliceIndexTracker.generate_role_aware = v65_generate_from_role_template
SliceIndexTracker.generate_ai_pattern = v65_generate_from_role_template
SliceIndexTracker.build_ui = v65_build_ui



# ---------------------------------------------------------------------
# v67 EXTENSION : feedback humain good/bad
# ---------------------------------------------------------------------

V67_FEEDBACK_PATH = Path("dataset/learning/role_feedback_v01.jsonl")
V67_VALIDATED_DIR = Path("dataset/learning/validated_patterns")
V67_REJECTED_DIR = Path("dataset/learning/rejected_patterns")


def v67_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v67_safe(self):
    try:
        safe = self.project.get("safe")
        if safe:
            return str(safe)
    except Exception:
        pass

    try:
        return str(self.safe)
    except Exception:
        return "unknown_break"


def v67_clone_pattern(pattern):
    try:
        return json.loads(json.dumps(pattern, ensure_ascii=False))
    except Exception:
        return [dict(x) for x in pattern]


def v67_current_pattern(self):
    return v67_clone_pattern(getattr(self, "pattern", []) or [])


def v67_write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def v67_append_feedback(data):
    V67_FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

    with V67_FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def v67_timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def v67_train_strict_if_available(self):
    candidates = [
        Path("pipeline/04_train_beat_roles_v04_strict.py"),
        Path("pipeline/04_train_beat_roles_v03.py"),
        Path("pipeline/04_train_beat_style_v02.py"),
    ]

    trainer = None

    for p in candidates:
        if p.exists():
            trainer = p
            break

    if trainer is None:
        print("[v140] aucun trainer trouvé.")
        return False

    print("")
    print("[v140] TRAIN après validation :", trainer)
    print("")

    try:
        result = subprocess.run(
            [sys.executable, str(trainer)],
            cwd=str(Path(".").resolve()),
            text=True,
            capture_output=True,
            check=False,
        )

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr)

        return result.returncode == 0

    except Exception as exc:
        print(f"[v140] training impossible : {exc}")
        return False


_old_v67_generate = (
    getattr(SliceIndexTracker, "generate_locked_roles", None)
    or getattr(SliceIndexTracker, "generate_cross_song_template", None)
    or getattr(SliceIndexTracker, "generate_safe_experiment", None)
    or getattr(SliceIndexTracker, "generate_ai_pattern", None)
)


def v67_generate_candidate(self, event=None):
    """
    Génère une proposition et garde un snapshot AVANT tes corrections.
    """
    if _old_v67_generate is None:
        v67_status(self, "Aucun générateur trouvé dans cette app.")
        return "break"

    result = _old_v67_generate(self, event)

    candidate = v67_current_pattern(self)

    self.v67_candidate_before_edit = candidate
    self.v67_candidate_time = time.strftime("%Y-%m-%d %H:%M:%S")
    self.v67_candidate_safe = v67_safe(self)

    print("")
    print("[v140] CANDIDATE GENERATED")
    print("[v140] break:", self.v67_candidate_safe)
    print("[v140] notes:", len(candidate))
    print("[v140] Maintenant : corrige puis clique Good après modifs, ou Reject.")
    print("")

    v67_status(self, "Candidate générée. Corrige les notes puis Good après modifs, ou Reject.")

    return result


def v67_reject_candidate(self, event=None):
    """
    Marque la proposition comme mauvaise.
    Ne l'utilise PAS comme exemple positif.
    """
    safe = v67_safe(self)
    now = v67_timestamp()

    candidate = getattr(self, "v67_candidate_before_edit", None)

    if candidate is None:
        candidate = v67_current_pattern(self)

    current = v67_current_pattern(self)

    record = {
        "version": "v67_feedback",
        "rating": "bad",
        "break": safe,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_before_edit": candidate,
        "current_pattern_when_rejected": current,
        "notes": "Human rejected this generation. Do not use as positive training.",
    }

    rejected_path = V67_REJECTED_DIR / f"{safe}_rejected_v67_{now}.json"

    v67_write_json(rejected_path, record)
    v67_append_feedback(record)

    print("")
    print("[v140] REJECTED")
    print("[v140] saved:", rejected_path)
    print("")

    v67_status(self, "Noté mauvais. Cette génération ne sera pas apprise comme bonne.")

    return "break"


def v67_accept_corrected(self, event=None):
    """
    Valide la version actuelle comme bonne après tes corrections.
    Sauvegarde un exemple humain :
      before = génération IA
      after = pattern corrigé
    """
    safe = v67_safe(self)
    now = v67_timestamp()

    candidate = getattr(self, "v67_candidate_before_edit", None)

    if candidate is None:
        candidate = []

    corrected = v67_current_pattern(self)

    if not corrected:
        v67_status(self, "Impossible de valider : pattern vide.")
        return "break"

    record = {
        "version": "v67_feedback",
        "rating": "good_after_human_edit",
        "break": safe,
        "safe": safe,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": "human_validated_good_after_modifications",
        "candidate_before_edit": candidate,
        "pattern": corrected,
        "notes_count": len(corrected),
    }

    validated_path = V67_VALIDATED_DIR / f"{safe}_human_validated_v67_{now}.json"

    v67_write_json(validated_path, record)
    v67_append_feedback(record)

    # Écrit aussi le latest pattern si la fonction existe.
    try:
        self.write_latest_pattern(reason="v67_human_validated_good")
    except Exception:
        pass

    trained = v67_train_strict_if_available(self)

    print("")
    print("[v140] ACCEPTED GOOD AFTER EDIT")
    print("[v140] saved:", validated_path)
    print("[v140] trained:", trained)
    print("")

    if trained:
        v67_status(self, "Bon après modifs : exemple validé + training relancé.")
    else:
        v67_status(self, "Bon après modifs : exemple validé. Training non relancé.")

    return "break"


def v67_accept_without_training(self, event=None):
    safe = v67_safe(self)
    now = v67_timestamp()

    candidate = getattr(self, "v67_candidate_before_edit", None) or []
    corrected = v67_current_pattern(self)

    if not corrected:
        v67_status(self, "Impossible de valider : pattern vide.")
        return "break"

    record = {
        "version": "v67_feedback",
        "rating": "good_after_human_edit",
        "break": safe,
        "safe": safe,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": "human_validated_good_after_modifications_no_auto_train",
        "candidate_before_edit": candidate,
        "pattern": corrected,
        "notes_count": len(corrected),
    }

    validated_path = V67_VALIDATED_DIR / f"{safe}_human_validated_v67_{now}.json"

    v67_write_json(validated_path, record)
    v67_append_feedback(record)

    v67_status(self, "Exemple validé sans training auto.")
    print("[v140] accepted no-train:", validated_path)

    return "break"


_old_v67_build_ui = SliceIndexTracker.build_ui


def v67_build_ui(self):
    _old_v67_build_ui(self)

    try:
        frame = tk.Frame(self.root, bg="#101820")
        frame.pack(fill="x", padx=14, pady=(0, 8))

        tk.Button(
            frame,
            text="Generate Candidate",
            command=self.generate_candidate,
            bg="#6b3d67",
            fg="#f5eefe",
        ).pack(side="left", padx=6)

        tk.Button(
            frame,
            text="Good après modifs",
            command=self.accept_corrected_candidate,
            bg="#30513f",
            fg="#f5eefe",
        ).pack(side="left", padx=6)

        tk.Button(
            frame,
            text="Good sans train",
            command=self.accept_corrected_no_train,
            bg="#3c4b35",
            fg="#f5eefe",
        ).pack(side="left", padx=6)

        tk.Button(
            frame,
            text="Reject / Bad",
            command=self.reject_candidate,
            bg="#5a2430",
            fg="#f5eefe",
        ).pack(side="left", padx=6)

        tk.Label(
            frame,
            text="v67 : génère → corrige → Good. Si nul : Reject.",
            bg="#101820",
            fg="#f5d67b",
        ).pack(side="left", padx=8)

    except Exception as exc:
        print(f"[v140] UI feedback impossible : {exc}")

    v67_status(self, "v67 : Generate Candidate → corrige → Good après modifs, ou Reject.")


SliceIndexTracker.generate_candidate = v67_generate_candidate
SliceIndexTracker.accept_corrected_candidate = v67_accept_corrected
SliceIndexTracker.accept_corrected_no_train = v67_accept_without_training
SliceIndexTracker.reject_candidate = v67_reject_candidate

# Les anciens boutons IA pointent vers le mode candidate.
SliceIndexTracker.generate_locked_roles = v67_generate_candidate
SliceIndexTracker.generate_cross_song_template = v67_generate_candidate
SliceIndexTracker.generate_safe_experiment = v67_generate_candidate
SliceIndexTracker.generate_safe_role_ai = v67_generate_candidate
SliceIndexTracker.generate_role_aware = v67_generate_candidate
SliceIndexTracker.generate_ai_pattern = v67_generate_candidate

SliceIndexTracker.build_ui = v67_build_ui



# ---------------------------------------------------------------------
# v68 EXTENSION : notes 2 cases par défaut, 1 case rare
# ---------------------------------------------------------------------

V68_ONE_CASE_CHANCE_HAT = 0.50
V68_ONE_CASE_CHANCE_GHOST = 0.25
V68_ONE_CASE_CHANCE_ANCHOR = 0.00


def v68_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v68_normalize_lengths(self):
    """
    Après génération :
    - kick/snare anchors = 2 cases
    - hats = 2 cases presque toujours
    - ghosts = parfois 1 case
    - si une note de 2 cases dépasse la grille, on la décale à 30/31
    """
    changed = False

    try:
        step_count = int(self.step_count)
    except Exception:
        step_count = 32

    for item in getattr(self, "pattern", []) or []:
        before = dict(item)

        role = str(
            item.get("learned_role")
            or item.get("cross_song_role")
            or item.get("prior_label")
            or ""
        ).lower()

        label = str(
            item.get("prior_label")
            or item.get("role_template_strength")
            or item.get("role_source")
            or ""
        ).lower()

        is_ghost = "ghost" in role or "ghost" in label
        is_hat = "hat" in role or "hat" in label
        is_anchor = "kick" in role or "snare" in role or "anchor" in label

        if is_ghost:
            length = 1 if random.random() < V68_ONE_CASE_CHANCE_GHOST else 2
        elif is_hat:
            length = 1 if random.random() < V68_ONE_CASE_CHANCE_HAT else 2
        elif is_anchor:
            length = 1 if random.random() < V68_ONE_CASE_CHANCE_ANCHOR else 2
        else:
            length = 2

        x = int(item.get("x_step", 0))

        if x + length > step_count:
            x = max(0, step_count - length)

        item["x_step"] = int(x)
        item["length"] = int(length)
        item["variation_bar"] = int(x) // 8
        item["variation_pos"] = int(x) % 8
        item["hit_slot"] = int(x) // int(HIT_SPACING_STEPS)
        item["v68_length_policy"] = "two_case_default_one_case_rare"

        if before != item:
            changed = True

    if changed:
        try:
            self.draw()
            self.refresh_panel()
            self.write_latest_pattern(reason="v68_normalize_two_case_default")
        except Exception:
            pass

    return changed


_old_v68_generate_candidate = getattr(SliceIndexTracker, "generate_candidate", None)
_old_v68_generate_ai = getattr(SliceIndexTracker, "generate_ai_pattern", None)


def v68_generate_candidate(self, event=None):
    gen = _old_v68_generate_candidate or _old_v68_generate_ai

    if gen is None:
        v68_status(self, "Aucun générateur trouvé.")
        return "break"

    result = gen(self, event)

    v68_normalize_lengths(self)

    # Met à jour le snapshot v67 après correction automatique des longueurs.
    try:
        self.v67_candidate_before_edit = json.loads(json.dumps(self.pattern, ensure_ascii=False))
    except Exception:
        pass

    v68_status(self, "Candidate générée : 2 cases par défaut, 1 case seulement rare.")

    print("[v140] Length policy appliquée : 2 cases par défaut, 1 case rare.")

    return result


_old_v68_build_ui = SliceIndexTracker.build_ui


def v68_build_ui(self):
    _old_v68_build_ui(self)

    try:
        frame = tk.Frame(self.root, bg="#101820")
        frame.pack(fill="x", padx=14, pady=(0, 8))

        tk.Label(
            frame,
            text="v68 : génération = notes 2 cases par défaut ; 1 case seulement rare.",
            bg="#101820",
            fg="#f5d67b",
        ).pack(side="left", padx=8)
    except Exception:
        pass

    v68_status(self, "v68 : 2 cases par défaut, 1 case rare.")


SliceIndexTracker.generate_candidate = v68_generate_candidate
SliceIndexTracker.generate_ai_pattern = v68_generate_candidate
SliceIndexTracker.generate_cross_song_template = v68_generate_candidate
SliceIndexTracker.generate_safe_experiment = v68_generate_candidate
SliceIndexTracker.generate_locked_roles = v68_generate_candidate
SliceIndexTracker.build_ui = v68_build_ui



# ---------------------------------------------------------------------
# v69 EXTENSION : génération verrouillée sur grille 2 cases
# ---------------------------------------------------------------------

V69_NOTE_LENGTH = 2
V69_GRID_STEP = 2


def v69_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v69_snap_even_start(self, x_step, length=V69_NOTE_LENGTH):
    """
    Force les départs de notes sur :
        0,2,4,6,8...
    Jamais :
        1,3,5,7...
    """
    try:
        step_count = int(self.step_count)
    except Exception:
        step_count = 32

    length = int(length)
    max_step = max(0, step_count - length)

    x = int(round(float(x_step)))

    # Snap vers la grille paire la plus proche.
    # 7 devient 8, 5 devient 4 selon l'arrondi Python.
    # Puis sécurité anti-fin-de-grille.
    x = int(round(x / float(V69_GRID_STEP))) * V69_GRID_STEP

    x = max(0, min(max_step, x))

    # Si max_step vaut 30, on garantit encore un départ pair.
    if x % 2 != 0:
        x -= 1

    x = max(0, min(max_step, x))

    return int(x)


def v69_force_pattern_hard_2_grid(self, reason="v69_force_hard_2_grid"):
    """
    Appliqué après chaque génération :
    - toutes les notes commencent sur une case paire
    - toutes les notes font 2 cases
    - collisions nettoyées
    """
    pattern = getattr(self, "pattern", []) or []

    if not pattern:
        return False

    changed = False
    by_step_role = {}

    priority = {
        "snare": 100,
        "kick": 90,
        "ghost_snare": 70,
        "hat": 50,
        "other": 10,
    }

    fixed = []

    for item in pattern:
        before = dict(item)

        role = str(
            item.get("learned_role")
            or item.get("cross_song_role")
            or item.get("prior_label")
            or item.get("role_template_strength")
            or ""
        ).lower()

        if "ghost" in role:
            role_key = "ghost_snare"
        elif "snare" in role:
            role_key = "snare"
        elif "kick" in role:
            role_key = "kick"
        elif "hat" in role:
            role_key = "hat"
        else:
            role_key = "other"

        x = v69_snap_even_start(self, item.get("x_step", 0), length=V69_NOTE_LENGTH)

        item["x_step"] = int(x)
        item["length"] = int(V69_NOTE_LENGTH)
        item["variation_bar"] = int(x) // 8
        item["variation_pos"] = int(x) % 8
        item["hit_slot"] = int(x) // int(HIT_SPACING_STEPS)
        item["v69_grid_policy"] = "hard_even_start_length_2"
        item["v69_role_key"] = role_key

        lane_pair = int(item.get("pair", 0))
        try:
            item["lane"] = int(self.pair_to_lane.get(lane_pair, item.get("lane", 0)))
        except Exception:
            pass

        key = (int(item["x_step"]), role_key)
        old = by_step_role.get(key)

        if old is None:
            by_step_role[key] = item
        else:
            # Si même rôle au même départ, garde celui qui était le plus prioritaire/ancien.
            old_p = priority.get(str(old.get("v69_role_key", "other")), 0)
            new_p = priority.get(role_key, 0)
            if new_p >= old_p:
                by_step_role[key] = item

        if before != item:
            changed = True

    # Nettoie aussi les collisions exactes de step :
    # une snare/kick gagne contre hats/ghosts.
    by_step = {}

    for item in by_step_role.values():
        step = int(item["x_step"])
        role_key = str(item.get("v69_role_key", "other"))
        old = by_step.get(step)

        if old is None:
            by_step[step] = item
            continue

        old_role = str(old.get("v69_role_key", "other"))
        if priority.get(role_key, 0) > priority.get(old_role, 0):
            by_step[step] = item

    fixed = [by_step[k] for k in sorted(by_step.keys())]

    for i, item in enumerate(fixed):
        item["id"] = i

    old_serialized = json.dumps(pattern, sort_keys=True, ensure_ascii=False)
    new_serialized = json.dumps(fixed, sort_keys=True, ensure_ascii=False)

    if old_serialized != new_serialized:
        changed = True

    self.pattern = fixed
    self.selected_id = self.pattern[0]["id"] if self.pattern else None

    if changed:
        try:
            self.draw()
            self.refresh_panel()
        except Exception:
            pass

        try:
            self.write_latest_pattern(reason=reason)
        except Exception:
            pass

    print("")
    print("[v140] HARD 2 GRID APPLIQUÉ")
    for item in self.pattern:
        end = int(item["x_step"]) + int(item["length"]) - 1
        print(
            f"[v140] step {item['x_step']:02d}/{end:02d} | "
            f"len={item['length']} | pair={item.get('pair')} | role={item.get('v69_role_key')}"
        )
    print("")

    return changed


_old_v69_generate_candidate = getattr(SliceIndexTracker, "generate_candidate", None)
_old_v69_generate_ai = getattr(SliceIndexTracker, "generate_ai_pattern", None)
_old_v69_generate_locked = getattr(SliceIndexTracker, "generate_locked_roles", None)
_old_v69_generate_cross = getattr(SliceIndexTracker, "generate_cross_song_template", None)


def v69_generate_candidate(self, event=None):
    """
    Génère avec l'ancien générateur, puis verrouille le résultat :
    0/1, 2/3, 4/5, 6/7...
    """
    gen = (
        _old_v69_generate_candidate
        or _old_v69_generate_locked
        or _old_v69_generate_cross
        or _old_v69_generate_ai
    )

    if gen is None:
        v69_status(self, "Aucun générateur trouvé.")
        return "break"

    result = gen(self, event)

    v69_force_pattern_hard_2_grid(self, reason="v69_generate_candidate_hard_2_grid")

    # Si v67 existe, le snapshot candidat doit être la version après snap.
    try:
        self.v67_candidate_before_edit = json.loads(json.dumps(self.pattern, ensure_ascii=False))
        self.v67_candidate_time = time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

    v69_status(self, "Candidate générée : départs pairs uniquement + longueur 2 cases.")

    return result


def v69_hard_fix_current(self, event=None):
    v69_force_pattern_hard_2_grid(self, reason="v69_manual_fix_current_pattern")
    v69_status(self, "Pattern corrigé : toutes les notes sont sur la grille 2 cases.")
    return "break"


_old_v69_build_ui = SliceIndexTracker.build_ui


def v69_build_ui(self):
    _old_v69_build_ui(self)

    try:
        frame = tk.Frame(self.root, bg="#101820")
        frame.pack(fill="x", padx=14, pady=(0, 8))

        tk.Button(
            frame,
            text="Fix 2-Grid",
            command=self.fix_hard_2_grid,
            bg="#304a6b",
            fg="#f5eefe",
        ).pack(side="left", padx=6)

        tk.Label(
            frame,
            text="v69 : génération verrouillée en 0/1, 2/3, 4/5… aucune case impaire en départ.",
            bg="#101820",
            fg="#f5d67b",
        ).pack(side="left", padx=8)

    except Exception as exc:
        print(f"[v140] UI impossible : {exc}")

    v69_status(self, "v69 : Generate Candidate = grille 2 cases stricte.")


SliceIndexTracker.generate_candidate = v69_generate_candidate
SliceIndexTracker.generate_ai_pattern = v69_generate_candidate
SliceIndexTracker.generate_cross_song_template = v69_generate_candidate
SliceIndexTracker.generate_safe_experiment = v69_generate_candidate
SliceIndexTracker.generate_locked_roles = v69_generate_candidate
SliceIndexTracker.generate_role_aware = v69_generate_candidate
SliceIndexTracker.generate_safe_role_ai = v69_generate_candidate
SliceIndexTracker.fix_hard_2_grid = v69_hard_fix_current
SliceIndexTracker.build_ui = v69_build_ui



# ---------------------------------------------------------------------
# v70 EXTENSION : Ctrl+D décale de 2 cases
# ---------------------------------------------------------------------

V70_DUPLICATE_STEP_DELTA = 2


def v70_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v70_get_selected_item(self):
    selected_id = getattr(self, "selected_id", None)

    if selected_id is None:
        return None

    for item in getattr(self, "pattern", []) or []:
        try:
            if int(item.get("id")) == int(selected_id):
                return item
        except Exception:
            pass

    return None


def v70_next_item_id(self):
    ids = []

    for item in getattr(self, "pattern", []) or []:
        try:
            ids.append(int(item.get("id", -1)))
        except Exception:
            pass

    return max(ids) + 1 if ids else 0


def v70_snap_even_start(self, x_step, length=2):
    try:
        step_count = int(self.step_count)
    except Exception:
        step_count = 32

    length = int(length)
    max_step = max(0, step_count - length)

    x = int(round(float(x_step)))

    # Ctrl+D doit rester sur 0,2,4,6...
    x = int(round(x / 2.0)) * 2
    x = max(0, min(max_step, x))

    if x % 2 != 0:
        x -= 1

    x = max(0, min(max_step, x))

    return int(x)


def v70_refresh_after_edit(self, reason):
    try:
        self.draw()
    except Exception:
        pass

    try:
        self.refresh_panel()
    except Exception:
        pass

    try:
        self.write_latest_pattern(reason=reason)
    except Exception:
        pass


def v70_duplicate_selected_forward(self, event=None):
    item = v70_get_selected_item(self)

    if item is None:
        v70_status(self, "Ctrl+D : aucune note sélectionnée.")
        return "break"

    before = dict(item)

    old_x = int(item.get("x_step", 0))

    # On garde la longueur de la note si elle a été resize à la main.
    # Mais si elle est absente/invalide, on part sur 2.
    try:
        length = int(item.get("length", 2))
    except Exception:
        length = 2

    length = max(1, min(8, length))

    # Le point important : +2, jamais +1.
    new_x = v70_snap_even_start(
        self,
        old_x + V70_DUPLICATE_STEP_DELTA,
        length=length,
    )

    if new_x == old_x:
        v70_status(self, "Ctrl+D : impossible, fin de grille.")
        return "break"

    new_item = dict(item)
    new_item["id"] = v70_next_item_id(self)
    new_item["x_step"] = int(new_x)
    new_item["length"] = int(length)
    new_item["variation_bar"] = int(new_x) // 8
    new_item["variation_pos"] = int(new_x) % 8
    new_item["hit_slot"] = int(new_x) // int(HIT_SPACING_STEPS)
    new_item["duplicated_from"] = int(item.get("id"))
    new_item["v70_duplicate_policy"] = "ctrl_d_plus_2_cases"

    # Réaligne la lane selon la pair.
    try:
        pair = int(new_item.get("pair", 0))
        new_item["lane"] = int(self.pair_to_lane.get(pair, new_item.get("lane", 0)))
    except Exception:
        pass

    self.pattern.append(new_item)
    self.selected_id = int(new_item["id"])

    try:
        self.record_correction(
            "duplicate_note_ctrl_d_plus_2",
            before=before,
            after=dict(new_item),
        )
    except Exception as exc:
        print(f"[v140] record_correction duplicate impossible : {exc}")

    v70_refresh_after_edit(self, "duplicate_note_ctrl_d_plus_2")

    try:
        self.audition_selected_case()
    except Exception as exc:
        print(f"[v140] audition duplicate impossible : {exc}")

    v70_status(
        self,
        f"Ctrl+D : note dupliquée de +2 cases | {old_x}/{old_x + length - 1} → {new_x}/{new_x + length - 1}"
    )

    print("")
    print("[v140] CTRL+D DUPLICATE +2")
    print(f"[v140] old_id={item.get('id')} new_id={new_item.get('id')}")
    print(f"[v140] step {old_x}/{old_x + length - 1} -> {new_x}/{new_x + length - 1}")
    print(f"[v140] pair={new_item.get('pair')} len={length}")
    print("")

    return "break"


_old_v70_build_ui = SliceIndexTracker.build_ui


def v70_build_ui(self):
    _old_v70_build_ui(self)

    # Rebind fort : ça écrase le Ctrl+D hérité de v58 qui faisait +1.
    try:
        self.root.bind("<Control-d>", self.duplicate_selected_forward)
        self.root.bind("<Control-D>", self.duplicate_selected_forward)
        self.root.bind_all("<Control-d>", self.duplicate_selected_forward)
        self.root.bind_all("<Control-D>", self.duplicate_selected_forward)

        self.canvas.bind("<Control-d>", self.duplicate_selected_forward)
        self.canvas.bind("<Control-D>", self.duplicate_selected_forward)
    except Exception as exc:
        print(f"[v140] bind Ctrl+D impossible : {exc}")

    try:
        frame = tk.Frame(self.root, bg="#101820")
        frame.pack(fill="x", padx=14, pady=(0, 8))

        tk.Label(
            frame,
            text="v70 : Ctrl+D = duplication +2 cases, jamais +1.",
            bg="#101820",
            fg="#f5d67b",
        ).pack(side="left", padx=8)
    except Exception:
        pass

    v70_status(self, "v70 : Ctrl+D décale maintenant de 2 cases.")


SliceIndexTracker.duplicate_selected_forward = v70_duplicate_selected_forward
SliceIndexTracker.build_ui = v70_build_ui



# ---------------------------------------------------------------------
# v71 EXTENSION : full candidate + pré-learning kick/snare/hat
# ---------------------------------------------------------------------

V71_ROLE_OVERRIDES_PATH = Path("dataset/learning/break_role_overrides_v01.json")

# Génération 16 slots : tous les départs sont pairs, toutes les notes font 2 cases.
V71_FULL_TEMPLATE = [
    (0,  "kick",        "anchor"),
    (2,  "hat",         "fill"),
    (4,  "ghost_snare", "ghost"),
    (6,  "snare",       "anchor"),

    (8,  "kick",        "anchor"),
    (10, "hat",         "fill"),
    (12, "ghost_snare", "ghost"),
    (14, "snare",       "anchor"),

    (16, "kick",        "anchor"),
    (18, "hat",         "fill"),
    (20, "ghost_snare", "ghost"),
    (22, "snare",       "anchor"),

    (24, "kick",        "anchor"),
    (26, "hat",         "fill"),
    (28, "ghost_snare", "ghost"),
    (30, "snare",       "anchor"),
]

# Comme tu as repéré le kick sur le 7 :
# en grille 2 cases, "sur le 7" = note 6/7.
# On peut donc ajouter un kick layer sur les steps de snare.
V71_KICK_LAYER_ON_7_STEPS = [6, 14, 22, 30]


def v71_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v71_safe(self):
    try:
        safe = self.project.get("safe")
        if safe:
            return str(safe)
    except Exception:
        pass

    try:
        return str(self.safe)
    except Exception:
        return "unknown_break"


def v71_load_overrides():
    if not V71_ROLE_OVERRIDES_PATH.exists():
        return {"version": "break_role_overrides_v01", "breaks": {}}

    try:
        data = json.loads(V71_ROLE_OVERRIDES_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = {"version": "break_role_overrides_v01", "breaks": {}}

    if "breaks" not in data or not isinstance(data["breaks"], dict):
        data["breaks"] = {}

    return data


def v71_save_overrides(data):
    V71_ROLE_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    V71_ROLE_OVERRIDES_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def v71_get_roles(self):
    data = v71_load_overrides()
    safe = v71_safe(self)

    br = data["breaks"].setdefault(safe, {})
    roles = br.setdefault("roles", {})

    for role in ("kick", "snare", "hat", "ghost_snare", "bad"):
        roles.setdefault(role, [])

    return data, safe, roles


def v71_band_energy(y, low, high):
    y = np.asarray(y, dtype=np.float32)

    if len(y) < 256:
        y = np.pad(y, (0, 256 - len(y)))

    n = min(len(y), 4096)
    chunk = y[:n] * np.hanning(n).astype(np.float32)

    mag = np.abs(np.fft.rfft(chunk)).astype(np.float32)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)

    mask = (freqs >= low) & (freqs <= high)

    if not np.any(mask):
        return 0.0

    return float(np.sum(mag[mask] ** 2))


def v71_classify_slice(y):
    y = np.asarray(y, dtype=np.float32)

    if len(y) == 0:
        return {
            "kick": 0.0,
            "snare": 0.0,
            "hat": 0.0,
            "ghost_snare": 0.0,
            "other": 1.0,
        }

    attack = y[:min(len(y), int(SR * 0.500))]
    full = y[:min(len(y), int(SR * 0.500))]

    sub = v71_band_energy(attack, 35, 90)
    low = v71_band_energy(attack, 90, 250)
    lowmid = v71_band_energy(attack, 250, 700)
    mid = v71_band_energy(attack, 700, 2800)
    high = v71_band_energy(attack, 2800, 9000)
    air = v71_band_energy(attack, 9000, 16000)

    total = sub + low + lowmid + mid + high + air + 1e-9

    sub_r = sub / total
    low_r = low / total
    lowmid_r = lowmid / total
    mid_r = mid / total
    high_r = high / total
    air_r = air / total

    rms = float(np.sqrt(np.mean(full * full) + 1e-12))

    if len(attack) > 3:
        zcr = float(np.mean(np.abs(np.diff(np.signbit(attack).astype(np.float32)))))
    else:
        zcr = 0.0

    tail_start = min(len(y), int(SR * 0.520))
    tail_end = min(len(y), int(SR * 0.450))

    if tail_end > tail_start:
        tail = y[tail_start:tail_end]
        tail_rms = float(np.sqrt(np.mean(tail * tail) + 1e-12))
    else:
        tail_rms = 0.0

    tail_ratio = tail_rms / (rms + 1e-9)

    kick_score = (
        3.4 * sub_r
        + 2.7 * low_r
        + 0.6 * lowmid_r
        + 0.25 * rms
        - 1.0 * high_r
        - 0.9 * air_r
    )

    snare_score = (
        1.2 * lowmid_r
        + 2.2 * mid_r
        + 1.5 * high_r
        + 0.8 * air_r
        + 0.9 * zcr
        - 1.1 * sub_r
        - 0.7 * low_r
    )

    hat_score = (
        2.7 * high_r
        + 2.4 * air_r
        + 1.1 * zcr
        - 1.4 * sub_r
        - 1.0 * low_r
        - 0.35 * mid_r
        - 0.45 * tail_ratio
    )

    ghost_score = (
        0.82 * snare_score
        + 0.32 * high_r
        - 0.28 * rms
    )

    other_score = 0.4 * tail_ratio + 0.2 * rms

    return {
        "kick": float(kick_score),
        "snare": float(snare_score),
        "hat": float(hat_score),
        "ghost_snare": float(ghost_score),
        "other": float(other_score),
    }


def v71_rank_slices(self):
    ranks = {
        "kick": [],
        "snare": [],
        "hat": [],
        "ghost_snare": [],
        "other": [],
        "all": [],
    }

    for pair in self.pair_values:
        pair = int(pair)

        try:
            y = self.get_audio(pair)
            scores = v71_classify_slice(y)
        except Exception as exc:
            print(f"[v140] analyse slice {pair} impossible : {exc}")
            scores = {
                "kick": 0.0,
                "snare": 0.0,
                "hat": 0.0,
                "ghost_snare": 0.0,
                "other": 1.0,
            }

        item = {
            "pair": pair,
            "scores": scores,
            "role": max(scores, key=scores.get),
            "lane": self.pair_to_lane.get(pair, 0),
        }

        ranks["all"].append(item)

        for role in ("kick", "snare", "hat", "ghost_snare", "other"):
            role_item = dict(item)
            role_item["score"] = float(scores.get(role, 0.0))
            ranks[role].append(role_item)

    for role in ("kick", "snare", "hat", "ghost_snare", "other"):
        ranks[role] = sorted(ranks[role], key=lambda x: x["score"], reverse=True)

    print("")
    print("[v140] PRE-LEARNING TOP SLICES")
    for role in ("kick", "snare", "hat", "ghost_snare"):
        print(f"[v140] {role:12s}:", [(x["pair"], round(x["score"], 3)) for x in ranks[role][:8]])
    print("")

    return ranks


def v71_take_top_pairs(ranks, role, n=4, bad=None, avoid=None):
    bad = set(int(x) for x in (bad or []))
    avoid = set(int(x) for x in (avoid or []))

    out = []

    for item in ranks.get(role, []):
        pair = int(item["pair"])

        if pair in bad:
            continue

        if pair in avoid:
            continue

        if pair not in out:
            out.append(pair)

        if len(out) >= n:
            break

    return out


def v71_prelearn_roles(self, force=False):
    """
    Pré-remplit les labels kick/snare/hat/ghost_snare pour le break courant.

    - Si tu as déjà marqué une snare/kick/hat à la main, on ne l'écrase pas.
    - Si un rôle est vide, on remplit avec les meilleures slices détectées.
    - Les slices Mark Bad restent exclues.
    """
    data, safe, roles = v71_get_roles(self)
    ranks = v71_rank_slices(self)

    bad = set(int(x) for x in roles.get("bad", []))

    changed = False

    existing_kick = set(int(x) for x in roles.get("kick", []))
    existing_snare = set(int(x) for x in roles.get("snare", []))
    existing_hat = set(int(x) for x in roles.get("hat", []))
    existing_ghost = set(int(x) for x in roles.get("ghost_snare", []))

    if force or not existing_kick:
        roles["kick"] = v71_take_top_pairs(
            ranks,
            "kick",
            n=4,
            bad=bad,
            avoid=existing_snare | existing_hat,
        )
        changed = True

    if force or not existing_snare:
        roles["snare"] = v71_take_top_pairs(
            ranks,
            "snare",
            n=4,
            bad=bad,
            avoid=set(roles.get("kick", [])) | existing_hat,
        )
        changed = True

    if force or not existing_hat:
        roles["hat"] = v71_take_top_pairs(
            ranks,
            "hat",
            n=6,
            bad=bad,
            avoid=set(roles.get("kick", [])) | set(roles.get("snare", [])),
        )
        changed = True

    if force or not existing_ghost:
        roles["ghost_snare"] = v71_take_top_pairs(
            ranks,
            "ghost_snare",
            n=4,
            bad=bad,
            avoid=set(roles.get("kick", [])),
        )
        changed = True

    for role in ("kick", "snare", "hat", "ghost_snare", "bad"):
        roles[role] = sorted(set(int(x) for x in roles.get(role, [])))

    data["breaks"][safe]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    data["breaks"][safe]["prelearned_v71"] = True

    v71_save_overrides(data)

    print("")
    print("[v140] PRELEARN ROLES POUR", safe)
    for role in ("kick", "snare", "hat", "ghost_snare", "bad"):
        print(f"[v140] {role:12s}:", roles.get(role, []))
    print("")

    v71_status(
        self,
        f"Prelearn OK : kick={roles.get('kick', [])} snare={roles.get('snare', [])} hat={roles.get('hat', [])}"
    )

    return roles


def v71_prelearn_button(self):
    v71_prelearn_roles(self, force=False)
    return "break"


def v71_reprelearn_button(self):
    v71_prelearn_roles(self, force=True)
    return "break"


def v71_pick_pair(roles, role, index=0):
    bad = set(int(x) for x in roles.get("bad", []))
    candidates = [int(x) for x in roles.get(role, []) if int(x) not in bad]

    if not candidates:
        return None

    return candidates[index % len(candidates)]


def v71_make_note(self, step, pair, role, label):
    step = int(step)

    # Hard 2-grid : départ pair uniquement.
    step = int(round(step / 2.0)) * 2
    step = max(0, min(62, step))

    length = 2
    lane = self.pair_to_lane.get(int(pair), 0)

    return {
        "id": 0,
        "x_step": int(step),
        "lane": int(lane),
        "pair": int(pair),
        "length": int(length),
        "variation_bar": int(step) // 8,
        "variation_pos": int(step) % 8,
        "hit_slot": int(step) // int(HIT_SPACING_STEPS),
        "randomized": False,
        "ai_generated": True,
        "ai_model": "v71_full_candidate_prelearn",
        "learned_role": str(role),
        "prior_label": str(label),
        "v71_full_slot": True,
        "v71_grid_policy": "all_slots_filled_even_start_len_2",
    }


def v71_generate_full_candidate(self, event=None):
    """
    Génération propre :
    - pré-learning automatique kick/snare/hat si labels absents
    - 16 slots remplis
    - toutes les notes en 2 cases
    - départs toujours pairs
    """
    try:
        self.stop_playhead()
        self.stop_audio()
    except Exception:
        pass

    roles = v71_prelearn_roles(self, force=False)

    pattern = []

    role_index = {
        "kick": 0,
        "snare": 0,
        "hat": 0,
        "ghost_snare": 0,
    }

    # 1 note par slot pair.
    for step, role, label in V71_FULL_TEMPLATE:
        pair = v71_pick_pair(roles, role, index=role_index.get(role, 0))

        # Si pas de ghost_snare, on recycle snare.
        if pair is None and role == "ghost_snare":
            pair = v71_pick_pair(roles, "snare", index=role_index.get("snare", 0))

        # Si pas de hat, on recycle ghost/snare.
        if pair is None and role == "hat":
            pair = v71_pick_pair(roles, "ghost_snare", index=role_index.get("ghost_snare", 0))
            if pair is None:
                pair = v71_pick_pair(roles, "snare", index=role_index.get("snare", 0))

        # Si rôle vide, fallback très simple sur première slice.
        if pair is None:
            try:
                pair = int(self.pair_values[0])
            except Exception:
                pair = 0

        pattern.append(v71_make_note(self, step, pair, role, label))
        role_index[role] = role_index.get(role, 0) + 1

    # Option : kick layer sur le “7” = 6/7, 14/15, etc.
    kick_layer = True
    try:
        kick_layer = bool(self.v71_kick_layer_var.get())
    except Exception:
        pass

    if kick_layer:
        for i, step in enumerate(V71_KICK_LAYER_ON_7_STEPS):
            pair = v71_pick_pair(roles, "kick", index=i)

            if pair is None:
                continue

            note = v71_make_note(self, step, pair, "kick", "kick_layer_on_7")
            note["v71_layer"] = True
            pattern.append(note)

    # Supprime les slices bad.
    bad = set(int(x) for x in roles.get("bad", []))
    pattern = [p for p in pattern if int(p.get("pair", -999)) not in bad]

    # Important : on autorise kick+snare au même step si lanes différentes.
    pattern = sorted(pattern, key=lambda x: (int(x["x_step"]), str(x["learned_role"]), int(x["pair"])))

    for i, item in enumerate(pattern):
        item["id"] = i

    self.looping = False
    self.pattern = pattern
    self.selected_id = self.pattern[0]["id"] if self.pattern else None

    try:
        self.draw()
        self.refresh_panel()
    except Exception:
        pass

    try:
        self.write_latest_pattern(reason="v71_generate_full_candidate_prelearn")
    except Exception as exc:
        print(f"[v140] write_latest_pattern impossible : {exc}")

    # Snapshot pour le workflow v67 : Good / Reject.
    try:
        self.v67_candidate_before_edit = json.loads(json.dumps(self.pattern, ensure_ascii=False))
        self.v67_candidate_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.v67_candidate_safe = v71_safe(self)
    except Exception:
        pass

    print("")
    print("[v140] FULL CANDIDATE GENERATED")
    print("[v140] break:", v71_safe(self))
    print("[v140] roles:", roles)
    print("[v140] pattern:")
    for item in pattern:
        end = int(item["x_step"]) + int(item["length"]) - 1
        print(
            f"[v140] step {item['x_step']:02d}/{end:02d} | "
            f"role={item['learned_role']:12s} | pair={item['pair']:02d} | {item.get('prior_label')}"
        )
    print("")

    v71_status(
        self,
        "Generate Full Candidate OK : 16 slots remplis + pré-learning kick/snare/hat."
    )

    return "break"


_old_v71_build_ui = SliceIndexTracker.build_ui


def v71_build_ui(self):
    _old_v71_build_ui(self)

    try:
        frame = tk.Frame(self.root, bg="#101820")
        frame.pack(fill="x", padx=14, pady=(0, 8))

        self.v71_kick_layer_var = tk.BooleanVar(value=True)

        tk.Button(
            frame,
            text="Generate Full Candidate",
            command=self.generate_full_candidate,
            bg="#6b3d67",
            fg="#f5eefe",
        ).pack(side="left", padx=6)

        tk.Button(
            frame,
            text="Prelearn Roles",
            command=self.prelearn_roles,
            bg="#30513f",
            fg="#f5eefe",
        ).pack(side="left", padx=6)

        tk.Button(
            frame,
            text="Re-Prelearn",
            command=self.reprelearn_roles,
            bg="#3e5f4a",
            fg="#f5eefe",
        ).pack(side="left", padx=6)

        tk.Checkbutton(
            frame,
            text="kick layer sur 7",
            variable=self.v71_kick_layer_var,
            bg="#101820",
            fg="#f5d67b",
            selectcolor="#30283f",
            activebackground="#101820",
            activeforeground="#f5d67b",
        ).pack(side="left", padx=8)

        tk.Label(
            frame,
            text="v71 : 16 slots remplis, départs pairs, longueur 2, prélabels kick/snare/hat.",
            bg="#101820",
            fg="#b9acc8",
        ).pack(side="left", padx=8)

    except Exception as exc:
        print(f"[v140] UI impossible : {exc}")

    v71_status(
        self,
        "v71 : Generate Full Candidate remplit tout + pré-learning kick/snare/hat."
    )


SliceIndexTracker.prelearn_roles = v71_prelearn_button
SliceIndexTracker.reprelearn_roles = v71_reprelearn_button
SliceIndexTracker.generate_full_candidate = v71_generate_full_candidate

# Tous les anciens boutons de génération pointent vers la version complète.
SliceIndexTracker.generate_candidate = v71_generate_full_candidate
SliceIndexTracker.generate_ai_pattern = v71_generate_full_candidate
SliceIndexTracker.generate_cross_song_template = v71_generate_full_candidate
SliceIndexTracker.generate_safe_experiment = v71_generate_full_candidate
SliceIndexTracker.generate_locked_roles = v71_generate_full_candidate
SliceIndexTracker.generate_role_aware = v71_generate_full_candidate
SliceIndexTracker.generate_safe_role_ai = v71_generate_full_candidate

SliceIndexTracker.build_ui = v71_build_ui



# ---------------------------------------------------------------------
# v72 EXTENSION : sélecteur de breaks dans l'app
# ---------------------------------------------------------------------

V72_PAIR_DIR = Path("dataset/pair_blocks_v02")


def v72_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v72_current_safe(self):
    try:
        safe = self.project.get("safe")
        if safe:
            return str(safe)
    except Exception:
        pass

    try:
        return str(self.safe)
    except Exception:
        return ""


def v72_read_pair_block_safe(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        safe = data.get("safe")
        if safe:
            return str(safe)
    except Exception:
        pass

    name = path.name

    if name.endswith("_pair_blocks_v02.json"):
        name = name[:-len("_pair_blocks_v02.json")]

    return name


def v72_list_breaks():
    if not V72_PAIR_DIR.exists():
        return []

    breaks = []

    for path in sorted(V72_PAIR_DIR.glob("*_pair_blocks_v02.json")):
        safe = v72_read_pair_block_safe(path)
        if safe and safe not in breaks:
            breaks.append(safe)

    return sorted(breaks, key=lambda x: x.lower())


def v72_open_break(self, safe=None):
    safe = safe or ""

    try:
        if hasattr(self, "v72_break_var"):
            safe = self.v72_break_var.get()
    except Exception:
        pass

    safe = str(safe).strip()

    if not safe:
        v72_status(self, "Aucun break sélectionné.")
        return "break"

    current = v72_current_safe(self)

    if safe == current:
        v72_status(self, f"Déjà sur {safe}.")
        return "break"

    try:
        self.stop_playhead()
        self.stop_audio()
    except Exception:
        pass

    app_path = Path(__file__).resolve()

    cmd = [
        sys.executable,
        str(app_path),
        "--source",
        safe,
    ]

    print("")
    print("[v140] OPEN BREAK")
    print("[v140] current:", current)
    print("[v140] next   :", safe)
    print("[v140] cmd    :", " ".join(cmd))
    print("")

    subprocess.Popen(cmd, cwd=str(Path(".").resolve()))

    try:
        self.root.after(250, self.root.destroy)
    except Exception:
        pass

    return "break"


def v72_open_next_break(self):
    breaks = v72_list_breaks()

    if not breaks:
        v72_status(self, "Aucun pair_block trouvé.")
        return "break"

    current = v72_current_safe(self)

    if current in breaks:
        idx = breaks.index(current)
        next_safe = breaks[(idx + 1) % len(breaks)]
    else:
        next_safe = breaks[0]

    try:
        self.v72_break_var.set(next_safe)
    except Exception:
        pass

    return v72_open_break(self, next_safe)


def v72_open_prev_break(self):
    breaks = v72_list_breaks()

    if not breaks:
        v72_status(self, "Aucun pair_block trouvé.")
        return "break"

    current = v72_current_safe(self)

    if current in breaks:
        idx = breaks.index(current)
        next_safe = breaks[(idx - 1) % len(breaks)]
    else:
        next_safe = breaks[0]

    try:
        self.v72_break_var.set(next_safe)
    except Exception:
        pass

    return v72_open_break(self, next_safe)


def v72_open_random_break(self):
    breaks = v72_list_breaks()

    if not breaks:
        v72_status(self, "Aucun pair_block trouvé.")
        return "break"

    current = v72_current_safe(self)
    choices = [b for b in breaks if b != current] or breaks
    next_safe = random.choice(choices)

    try:
        self.v72_break_var.set(next_safe)
    except Exception:
        pass

    return v72_open_break(self, next_safe)


def v72_print_breaks(self):
    breaks = v72_list_breaks()

    print("")
    print("[v140] BREAKS DISPONIBLES")
    for i, safe in enumerate(breaks, start=1):
        marker = " <==" if safe == v72_current_safe(self) else ""
        print(f"[v140] {i:03d}. {safe}{marker}")
    print("")

    v72_status(self, f"{len(breaks)} breaks disponibles. Regarde le terminal.")

    return "break"


_old_v72_build_ui = SliceIndexTracker.build_ui


def v72_build_ui(self):
    _old_v72_build_ui(self)

    breaks = v72_list_breaks()
    current = v72_current_safe(self)

    if not breaks:
        breaks = [current or "aucun_break"]

    default = current if current in breaks else breaks[0]

    try:
        frame = tk.Frame(self.root, bg="#101820")
        frame.pack(fill="x", padx=14, pady=(0, 8))

        tk.Label(
            frame,
            text="Break:",
            bg="#101820",
            fg="#f5d67b",
        ).pack(side="left", padx=(6, 4))

        self.v72_break_var = tk.StringVar(value=default)

        menu = tk.OptionMenu(frame, self.v72_break_var, *breaks)
        menu.config(
            bg="#30283f",
            fg="#f5eefe",
            activebackground="#5a365f",
            activeforeground="#ffffff",
            highlightthickness=0,
            width=28,
        )
        menu.pack(side="left", padx=4)

        tk.Button(
            frame,
            text="Open Break",
            command=self.open_selected_break,
            bg="#6b3d67",
            fg="#f5eefe",
        ).pack(side="left", padx=4)

        tk.Button(
            frame,
            text="Prev",
            command=self.open_prev_break,
            bg="#30384a",
            fg="#f5eefe",
        ).pack(side="left", padx=3)

        tk.Button(
            frame,
            text="Next",
            command=self.open_next_break,
            bg="#30384a",
            fg="#f5eefe",
        ).pack(side="left", padx=3)

        tk.Button(
            frame,
            text="Random",
            command=self.open_random_break,
            bg="#304a3a",
            fg="#f5eefe",
        ).pack(side="left", padx=3)

        tk.Button(
            frame,
            text="List",
            command=self.print_breaks,
            bg="#303030",
            fg="#eeeeee",
        ).pack(side="left", padx=3)

        tk.Label(
            frame,
            text="v72 : change de break puis Generate Candidate / Full Candidate.",
            bg="#101820",
            fg="#b9acc8",
        ).pack(side="left", padx=8)

    except Exception as exc:
        print(f"[v140] UI break switcher impossible : {exc}")

    v72_status(
        self,
        f"v72 : break courant = {current}. Choisis un autre break puis Open Break."
    )


SliceIndexTracker.open_selected_break = v72_open_break
SliceIndexTracker.open_next_break = v72_open_next_break
SliceIndexTracker.open_prev_break = v72_open_prev_break
SliceIndexTracker.open_random_break = v72_open_random_break
SliceIndexTracker.print_breaks = v72_print_breaks
SliceIndexTracker.build_ui = v72_build_ui



# ---------------------------------------------------------------------
# v73 EXTENSION : tracker strict, aucune superposition
# ---------------------------------------------------------------------

V73_NOTE_LENGTH = 2
V73_GRID_STEP = 2

V73_ROLE_PRIORITY = {
    "snare": 100,
    "kick": 90,
    "ghost_snare": 60,
    "hat": 40,
    "other": 10,
}


def v73_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v73_role_key(item):
    raw = " ".join([
        str(item.get("learned_role", "")),
        str(item.get("prior_label", "")),
        str(item.get("role_template_strength", "")),
        str(item.get("v69_role_key", "")),
        str(item.get("role_source", "")),
    ]).lower()

    if "ghost" in raw:
        return "ghost_snare"
    if "snare" in raw:
        return "snare"
    if "kick" in raw:
        return "kick"
    if "hat" in raw:
        return "hat"
    return "other"


def v73_snap_even_start(self, x_step, length=V73_NOTE_LENGTH):
    try:
        step_count = int(self.step_count)
    except Exception:
        step_count = 32

    length = int(length)
    max_step = max(0, step_count - length)

    x = int(round(float(x_step)))
    x = int(round(x / float(V73_GRID_STEP))) * V73_GRID_STEP
    x = max(0, min(max_step, x))

    if x % 2 != 0:
        x -= 1

    return max(0, min(max_step, int(x)))


def v73_intervals_overlap(a_start, a_len, b_start, b_len):
    a_end = int(a_start) + int(a_len)
    b_end = int(b_start) + int(b_len)
    return int(a_start) < b_end and int(b_start) < a_end


def v73_sanitize_tracker_pattern(self, reason="v73_tracker_strict_no_overlap"):
    """
    Nettoie un pattern généré :
    - longueur 2
    - départ pair
    - aucune collision temporelle
    - une seule note par slot
    """
    original = getattr(self, "pattern", []) or []

    if not original:
        return False

    prepared = []

    for idx, item in enumerate(original):
        item = dict(item)

        role = v73_role_key(item)
        x = v73_snap_even_start(self, item.get("x_step", 0), length=V73_NOTE_LENGTH)

        item["x_step"] = int(x)
        item["length"] = int(V73_NOTE_LENGTH)
        item["variation_bar"] = int(x) // 8
        item["variation_pos"] = int(x) % 8
        item["hit_slot"] = int(x) // int(HIT_SPACING_STEPS)
        item["v73_role_key"] = role
        item["v73_tracker_policy"] = "no_overlap_even_start_len_2"

        try:
            pair = int(item.get("pair", 0))
            item["lane"] = int(self.pair_to_lane.get(pair, item.get("lane", 0)))
        except Exception:
            pass

        priority = V73_ROLE_PRIORITY.get(role, 0)

        prepared.append((int(x), -priority, idx, item))

    # Trie par step, puis priorité forte.
    prepared.sort(key=lambda t: (t[0], t[1], t[2]))

    accepted = []
    rejected = []

    for x, neg_prio, idx, item in prepared:
        role = item.get("v73_role_key", "other")
        length = int(item.get("length", V73_NOTE_LENGTH))
        priority = V73_ROLE_PRIORITY.get(role, 0)

        collision_index = None

        for j, old in enumerate(accepted):
            if v73_intervals_overlap(
                item["x_step"],
                length,
                old["x_step"],
                old.get("length", V73_NOTE_LENGTH),
            ):
                collision_index = j
                break

        if collision_index is None:
            accepted.append(item)
            continue

        old = accepted[collision_index]
        old_role = old.get("v73_role_key", "other")
        old_priority = V73_ROLE_PRIORITY.get(old_role, 0)

        if priority > old_priority:
            rejected.append(old)
            accepted[collision_index] = item
        else:
            rejected.append(item)

    accepted.sort(key=lambda it: int(it["x_step"]))

    for i, item in enumerate(accepted):
        item["id"] = i

    old_serialized = json.dumps(original, sort_keys=True, ensure_ascii=False)
    new_serialized = json.dumps(accepted, sort_keys=True, ensure_ascii=False)
    changed = old_serialized != new_serialized

    self.pattern = accepted
    self.selected_id = self.pattern[0]["id"] if self.pattern else None

    if changed:
        try:
            self.draw()
            self.refresh_panel()
        except Exception:
            pass

        try:
            self.write_latest_pattern(reason=reason)
        except Exception:
            pass

    print("")
    print("[v140] TRACKER STRICT NO OVERLAP")
    print("[v140] kept:", len(accepted), "| rejected overlaps:", len(rejected))

    for item in accepted:
        end = int(item["x_step"]) + int(item["length"]) - 1
        print(
            f"[v140] KEEP step {item['x_step']:02d}/{end:02d} | "
            f"role={item.get('v73_role_key')} | pair={item.get('pair')}"
        )

    if rejected:
        print("[v140] rejected:")
        for item in rejected:
            end = int(item["x_step"]) + int(item.get("length", V73_NOTE_LENGTH)) - 1
            print(
                f"[v140] DROP step {item['x_step']:02d}/{end:02d} | "
                f"role={item.get('v73_role_key')} | pair={item.get('pair')}"
            )

    print("")

    return changed


_old_v73_generate_candidate = getattr(SliceIndexTracker, "generate_candidate", None)
_old_v73_generate_full = getattr(SliceIndexTracker, "generate_full_candidate", None)
_old_v73_generate_ai = getattr(SliceIndexTracker, "generate_ai_pattern", None)
_old_v73_generate_locked = getattr(SliceIndexTracker, "generate_locked_roles", None)


def v73_generate_tracker_strict(self, event=None):
    gen = (
        _old_v73_generate_full
        or _old_v73_generate_candidate
        or _old_v73_generate_locked
        or _old_v73_generate_ai
    )

    if gen is None:
        v73_status(self, "Aucun générateur trouvé.")
        return "break"

    result = gen(self, event)

    v73_sanitize_tracker_pattern(self, reason="v73_generate_tracker_strict")

    try:
        self.v67_candidate_before_edit = json.loads(json.dumps(self.pattern, ensure_ascii=False))
        self.v67_candidate_time = time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

    v73_status(self, "Generate Candidate : tracker strict, aucune note superposée.")

    return result


def v73_fix_current_pattern(self, event=None):
    v73_sanitize_tracker_pattern(self, reason="v73_fix_current_pattern")
    v73_status(self, "Pattern nettoyé : aucune superposition.")
    return "break"


_old_v73_build_ui = SliceIndexTracker.build_ui


def v73_build_ui(self):
    _old_v73_build_ui(self)

    try:
        frame = tk.Frame(self.root, bg="#101820")
        frame.pack(fill="x", padx=14, pady=(0, 8))

        tk.Button(
            frame,
            text="Fix No Overlap",
            command=self.fix_no_overlap,
            bg="#304a6b",
            fg="#f5eefe",
        ).pack(side="left", padx=6)

        tk.Label(
            frame,
            text="v73 : tracker strict = une seule note par slot, aucune superposition.",
            bg="#101820",
            fg="#f5d67b",
        ).pack(side="left", padx=8)

    except Exception as exc:
        print(f"[v140] UI impossible : {exc}")

    v73_status(self, "v73 : Generate Candidate nettoie toute superposition.")


SliceIndexTracker.generate_candidate = v73_generate_tracker_strict
SliceIndexTracker.generate_full_candidate = v73_generate_tracker_strict
SliceIndexTracker.generate_ai_pattern = v73_generate_tracker_strict
SliceIndexTracker.generate_cross_song_template = v73_generate_tracker_strict
SliceIndexTracker.generate_safe_experiment = v73_generate_tracker_strict
SliceIndexTracker.generate_locked_roles = v73_generate_tracker_strict
SliceIndexTracker.generate_role_aware = v73_generate_tracker_strict
SliceIndexTracker.generate_safe_role_ai = v73_generate_tracker_strict
SliceIndexTracker.fix_no_overlap = v73_fix_current_pattern
SliceIndexTracker.build_ui = v73_build_ui



# ---------------------------------------------------------------------
# v81 EXTENSION : mêmes boutons training sur TOUS les breaks chargés
# ---------------------------------------------------------------------

V81_PAIR_DIR = Path("dataset/pair_blocks_v02")
V81_LEARNING_DIR = Path("dataset/learning")
V81_VALIDATED_DIR = V81_LEARNING_DIR / "validated_patterns"
V81_REJECTED_DIR = V81_LEARNING_DIR / "rejected_patterns"
V81_FEEDBACK_PATH = V81_LEARNING_DIR / "role_feedback_v01.jsonl"


def v81_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v81_current_break(self):
    try:
        safe = self.project.get("safe")
        if safe:
            return str(safe)
    except Exception:
        pass

    try:
        return str(self.safe)
    except Exception:
        return "unknown_break"


def v81_slug(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown_break"


def v81_norm(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def v81_pairblock_sources():
    if not V81_PAIR_DIR.exists():
        return []

    out = []

    for path in sorted(V81_PAIR_DIR.glob("*_pair_blocks_v02.json")):
        name = path.name
        if name.endswith("_pair_blocks_v02.json"):
            name = name[:-len("_pair_blocks_v02.json")]

        if name and name not in out:
            out.append(name)

    return sorted(out, key=lambda x: x.lower())


def v81_all_widgets(widget):
    out = [widget]

    try:
        children = widget.winfo_children()
    except Exception:
        children = []

    for child in children:
        out.extend(v81_all_widgets(child))

    return out


def v81_selected_break_from_ui(self):
    """
    Lit le menu déjà existant en haut :
    Break [ ... ] [Charger break]
    """
    breaks = v81_pairblock_sources()
    norm_to_real = {v81_norm(b): b for b in breaks}

    candidates = []

    # D'abord les variables connues.
    for attr in [
        "break_var",
        "source_var",
        "selected_break_var",
        "break_choice_var",
        "combo_var",
        "v72_break_var",
        "v74_break_var",
        "v75_break_var",
        "v76_break_var",
        "v78_break_var",
        "v79_break_var",
    ]:
        try:
            value = getattr(self, attr).get()
            if value:
                candidates.append(str(value).strip())
        except Exception:
            pass

    # Puis tous les widgets qui ont .get(), dont le combobox du haut.
    try:
        for widget in v81_all_widgets(self.root):
            try:
                value = widget.get()
            except Exception:
                continue

            value = str(value).strip()

            if not value:
                continue

            if value.isdigit():
                continue

            candidates.append(value)
    except Exception:
        pass

    for value in candidates:
        n = v81_norm(value)

        if n in norm_to_real:
            return norm_to_real[n]

        # Exemple : "camo" retrouve "Camo_Break_-_3A"
        matches = [b for b in breaks if n and n in v81_norm(b)]
        if len(matches) == 1:
            return matches[0]

    return v81_current_break(self)


def v81_launch_break(self, source):
    """
    Le bouton Charger break relance cette même app v81 sur le break choisi.
    Comme ça, les boutons training existent aussi sur ce break.
    """
    source = str(source).strip()

    if not source:
        v81_status(self, "Aucun break sélectionné.")
        return "break"

    current = v81_current_break(self)

    if v81_norm(source) == v81_norm(current):
        v81_status(self, f"Déjà sur {current}.")
        return "break"

    try:
        self.stop_playhead()
        self.stop_audio()
    except Exception:
        pass

    app_path = Path(__file__).resolve()

    cmd = [
        sys.executable,
        str(app_path),
        "--source",
        source,
    ]

    print("")
    print("[v140] CHARGER BREAK")
    print("[v140] courant :", current)
    print("[v140] nouveau :", source)
    print("[v140] cmd     :", " ".join(cmd))
    print("")

    subprocess.Popen(
        cmd,
        cwd=str(Path(".").resolve()),
        env=dict(os.environ),
    )

    try:
        self.root.after(250, self.root.destroy)
    except Exception:
        pass

    return "break"


def v81_load_selected_break(self, event=None):
    selected = v81_selected_break_from_ui(self)
    return v81_launch_break(self, selected)


def v81_patch_charger_break_buttons(self):
    """
    Remplace l'action du bouton 'Charger break' existant.
    """
    patched = 0

    for widget in v81_all_widgets(self.root):
        try:
            text = str(widget.cget("text")).strip().lower()
        except Exception:
            continue

        if text in [
            "charger break",
            "charger le break",
            "load break",
            "open break",
        ]:
            try:
                widget.configure(command=self.v81_load_selected_break)
                patched += 1
            except Exception:
                pass

    print(f"[v140] boutons Charger break patchés : {patched}")


def v81_real_generator():
    """
    Vrai générateur musical du break chargé.
    On ne passe PAS par un menu Camo / cible.
    """
    for name in [
        "v73_generate_tracker_strict",
        "v71_generate_full_candidate",
        "v66_generate_locked_roles",
        "v65_generate_from_role_template",
        "v64_generate_safe_experiment",
        "v63_generate_safe_role_ai",
    ]:
        fn = globals().get(name)
        if callable(fn):
            return fn

    return None


def v81_generate_candidate(self, event=None):
    """
    Generate Candidate sur le break actuellement chargé.
    """
    current = v81_current_break(self)
    fn = v81_real_generator()

    print("")
    print("[v140] GENERATE CANDIDATE")
    print("[v140] break chargé :", current)
    print("[v140] générateur   :", getattr(fn, "__name__", None))
    print("")

    if fn is None:
        v81_status(self, "Aucun générateur musical trouvé.")
        return "break"

    result = fn(self, event)

    try:
        self.v67_candidate_before_edit = json.loads(json.dumps(self.pattern, ensure_ascii=False))
        self.v67_candidate_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.v67_candidate_safe = current
    except Exception:
        pass

    v81_status(self, f"Generate Candidate OK sur {current}.")
    return result


def v81_feedback_payload(self, verdict, train_now):
    current = v81_current_break(self)

    try:
        candidate_before = self.v67_candidate_before_edit
    except Exception:
        candidate_before = None

    return {
        "version": "v81_training_buttons_all_breaks",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "break": current,
        "verdict": verdict,
        "train_now": bool(train_now),
        "pattern": getattr(self, "pattern", []) or [],
        "candidate_before_edit": candidate_before,
        "note": "Boutons training permanents, appliqués au break actuellement chargé."
    }


def v81_run_training(self):
    candidates = [
        Path("pipeline/04_train_beat_roles_v04_strict.py"),
        Path("pipeline/04_train_beat_roles_v03.py"),
    ]

    script = None

    for p in candidates:
        if p.exists():
            script = p
            break

    if script is None:
        print("[v140] Aucun script training trouvé.")
        v81_status(self, "Good sauvegardé, mais aucun script training trouvé.")
        return

    cmd = [
        sys.executable,
        str(script),
    ]

    print("[v140] TRAINING:", " ".join(cmd))

    try:
        subprocess.Popen(cmd, cwd=str(Path(".").resolve()))
    except Exception as exc:
        print(f"[v140] training impossible : {exc}")
        v81_status(self, f"Training impossible : {exc}")


def v81_save_feedback(self, verdict, train_now=False):
    current = v81_current_break(self)
    pattern = getattr(self, "pattern", []) or []

    if verdict != "reject_bad" and not pattern:
        v81_status(self, "Pattern vide : rien à sauvegarder.")
        return "break"

    V81_VALIDATED_DIR.mkdir(parents=True, exist_ok=True)
    V81_REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    V81_FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = v81_feedback_payload(self, verdict, train_now)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe = v81_slug(current)

    if verdict == "reject_bad":
        out_path = V81_REJECTED_DIR / f"{stamp}_{safe}_reject_bad.json"
    elif train_now:
        out_path = V81_VALIDATED_DIR / f"{stamp}_{safe}_good_train.json"
    else:
        out_path = V81_VALIDATED_DIR / f"{stamp}_{safe}_good_save_only.json"

    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with V81_FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print("")
    print("[v140] FEEDBACK SAUVÉ")
    print("[v140] break  :", current)
    print("[v140] verdict:", verdict)
    print("[v140] train  :", train_now)
    print("[v140] fichier:", out_path)
    print("")

    if train_now:
        v81_run_training(self)

    if verdict == "reject_bad":
        v81_status(self, f"Reject / Bad sauvegardé pour {current}.")
    elif train_now:
        v81_status(self, f"Good après modifs + training pour {current}.")
    else:
        v81_status(self, f"Good sans train sauvegardé pour {current}.")

    return "break"


def v81_good_after_modifs(self, event=None):
    return v81_save_feedback(self, "good_after_modifs", train_now=True)


def v81_good_no_train(self, event=None):
    return v81_save_feedback(self, "good_no_train", train_now=False)


def v81_reject_bad(self, event=None):
    return v81_save_feedback(self, "reject_bad", train_now=False)


def v81_install_training_toolbar(self):
    """
    Barre fixe, toujours visible, pas en bas de l'UI.
    Elle reste au-dessus et elle agit sur le break chargé.
    """
    try:
        if hasattr(self, "v81_training_panel") and self.v81_training_panel.winfo_exists():
            return
    except Exception:
        pass

    panel = tk.Frame(
        self.root,
        bg="#160b1e",
        bd=2,
        relief="ridge",
    )

    # En haut à droite, par-dessus l'ancien label d'import.
    panel.place(relx=1.0, x=-8, y=8, anchor="ne")
    panel.lift()

    self.v81_training_panel = panel
    self.v81_training_label_var = tk.StringVar(
        value=f"Training : {v81_current_break(self)}"
    )

    tk.Label(
        panel,
        textvariable=self.v81_training_label_var,
        bg="#160b1e",
        fg="#7dffd1",
        font=("Sans", 9, "bold"),
    ).pack(side="left", padx=(8, 6), pady=5)

    tk.Button(
        panel,
        text="Generate Candidate",
        command=self.generate_candidate,
        bg="#7a3f7a",
        fg="#ffffff",
    ).pack(side="left", padx=3, pady=5)

    tk.Button(
        panel,
        text="Good après modifs",
        command=self.good_after_modifs_v81,
        bg="#3f5f48",
        fg="#ffffff",
    ).pack(side="left", padx=3, pady=5)

    tk.Button(
        panel,
        text="Good sans train",
        command=self.good_no_train_v81,
        bg="#506050",
        fg="#ffffff",
    ).pack(side="left", padx=3, pady=5)

    tk.Button(
        panel,
        text="Reject / Bad",
        command=self.reject_bad_v81,
        bg="#743747",
        fg="#ffffff",
    ).pack(side="left", padx=(3, 8), pady=5)

    def keep_visible():
        try:
            self.v81_training_label_var.set(f"Training : {v81_current_break(self)}")
            self.v81_training_panel.lift()
        except Exception:
            pass

        try:
            self.root.after(500, keep_visible)
        except Exception:
            pass

    keep_visible()


_old_v81_build_ui = SliceIndexTracker.build_ui


def v81_build_ui(self):
    _old_v81_build_ui(self)

    try:
        self.root.title(f"BreakbeatAI v140 — Princess clean toolbar")
    except Exception:
        pass

    try:
        v81_patch_charger_break_buttons(self)
    except Exception as exc:
        print(f"[v140] patch Charger break impossible : {exc}")

    try:
        v81_install_training_toolbar(self)
    except Exception as exc:
        print(f"[v140] toolbar impossible : {exc}")

    v81_status(
        self,
        f"v81 : boutons training permanents pour le break chargé = {v81_current_break(self)}."
    )


SliceIndexTracker.v81_load_selected_break = v81_load_selected_break
SliceIndexTracker.good_after_modifs_v81 = v81_good_after_modifs
SliceIndexTracker.good_no_train_v81 = v81_good_no_train
SliceIndexTracker.reject_bad_v81 = v81_reject_bad

# Tous les anciens boutons Generate Candidate utilisent maintenant le break actuellement chargé.
SliceIndexTracker.generate_candidate = v81_generate_candidate
SliceIndexTracker.generate_full_candidate = v81_generate_candidate
SliceIndexTracker.generate_ai_pattern = v81_generate_candidate
SliceIndexTracker.generate_cross_song_template = v81_generate_candidate
SliceIndexTracker.generate_safe_experiment = v81_generate_candidate
SliceIndexTracker.generate_locked_roles = v81_generate_candidate
SliceIndexTracker.generate_role_aware = v81_generate_candidate
SliceIndexTracker.generate_safe_role_ai = v81_generate_candidate

SliceIndexTracker.build_ui = v81_build_ui



# ---------------------------------------------------------------------
# v82 EXTENSION : nouvelle candidate à chaque clic + mémoire des rejects
# ---------------------------------------------------------------------

V82_LEARNING_DIR = Path("dataset/learning")
V82_VALIDATED_DIR = V82_LEARNING_DIR / "validated_patterns"
V82_REJECTED_DIR = V82_LEARNING_DIR / "rejected_patterns"
V82_FEEDBACK_PATH = V82_LEARNING_DIR / "role_feedback_v01.jsonl"
V82_REJECT_FP_PATH = V82_LEARNING_DIR / "rejected_fingerprints_v82.json"
V82_ROLE_OVERRIDES_PATH = V82_LEARNING_DIR / "break_role_overrides_v01.json"

V82_STEPS = list(range(0, 32, 2))

V82_TEMPLATES = [
    [
        "kick", "hat", "ghost_snare", "snare",
        "kick", "hat", "ghost_snare", "snare",
        "kick", "hat", "ghost_snare", "snare",
        "kick", "hat", "ghost_snare", "snare",
    ],
    [
        "kick", "hat", "kick", "snare",
        "ghost_snare", "hat", "kick", "snare",
        "kick", "hat", "ghost_snare", "snare",
        "kick", "hat", "ghost_snare", "snare",
    ],
    [
        "kick", "ghost_snare", "hat", "snare",
        "kick", "hat", "ghost_snare", "snare",
        "kick", "ghost_snare", "hat", "snare",
        "hat", "kick", "ghost_snare", "snare",
    ],
    [
        "kick", "hat", "ghost_snare", "snare",
        "hat", "kick", "ghost_snare", "snare",
        "kick", "hat", "kick", "snare",
        "ghost_snare", "hat", "kick", "snare",
    ],
    [
        "kick", "hat", "hat", "snare",
        "kick", "ghost_snare", "hat", "snare",
        "kick", "hat", "ghost_snare", "snare",
        "kick", "hat", "kick", "snare",
    ],
]


def v82_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        else:
            self.output_label.config(text=text)
    except Exception:
        pass


def v82_current_break(self):
    try:
        safe = self.project.get("safe")
        if safe:
            return str(safe)
    except Exception:
        pass

    try:
        return str(self.safe)
    except Exception:
        return "unknown_break"


def v82_slug(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown_break"


def v82_hit_spacing():
    try:
        return int(globals().get("HIT_SPACING_STEPS", 2))
    except Exception:
        return 2


def v82_sr(self):
    try:
        return int(getattr(self, "sr"))
    except Exception:
        pass

    try:
        return int(globals().get("SR", 44100))
    except Exception:
        return 44100


def v82_pattern_fingerprint(pattern):
    compact = []

    for item in pattern or []:
        compact.append([
            int(item.get("x_step", 0)),
            int(item.get("pair", -1)),
            str(item.get("learned_role", item.get("prior_label", ""))),
        ])

    compact = sorted(compact, key=lambda x: (x[0], x[1], x[2]))
    return json.dumps(compact, ensure_ascii=False, sort_keys=True)


def v82_load_rejected_fps():
    if not V82_REJECT_FP_PATH.exists():
        return {}

    try:
        data = json.loads(V82_REJECT_FP_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = {}

    if not isinstance(data, dict):
        data = {}

    return data


def v82_save_rejected_fp(break_name, pattern):
    data = v82_load_rejected_fps()
    key = v82_slug(break_name)
    data.setdefault(key, [])

    fp = v82_pattern_fingerprint(pattern)

    if fp not in data[key]:
        data[key].append(fp)

    # On garde les 300 derniers rejects par break.
    data[key] = data[key][-300:]

    V82_REJECT_FP_PATH.parent.mkdir(parents=True, exist_ok=True)
    V82_REJECT_FP_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return fp


def v82_get_rejected_set(break_name):
    data = v82_load_rejected_fps()
    return set(data.get(v82_slug(break_name), []))


def v82_load_roles_from_overrides(self):
    break_name = v82_current_break(self)

    roles = {
        "kick": [],
        "snare": [],
        "hat": [],
        "ghost_snare": [],
        "bad": [],
    }

    if not V82_ROLE_OVERRIDES_PATH.exists():
        return roles

    try:
        data = json.loads(V82_ROLE_OVERRIDES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return roles

    br = data.get("breaks", {}).get(break_name)

    if br is None:
        # fallback nom slug
        for k, v in data.get("breaks", {}).items():
            if v82_slug(k).lower() == v82_slug(break_name).lower():
                br = v
                break

    if not isinstance(br, dict):
        return roles

    raw_roles = br.get("roles", {})

    if not isinstance(raw_roles, dict):
        return roles

    for role in roles:
        values = raw_roles.get(role, [])
        out = []

        for x in values:
            try:
                out.append(int(x))
            except Exception:
                pass

        roles[role] = sorted(set(out))

    return roles


def v82_band_energy(y, sr, low, high):
    y = np.asarray(y, dtype=np.float32)

    if len(y) < 256:
        y = np.pad(y, (0, 256 - len(y)))

    n = min(len(y), 4096)

    if n <= 0:
        return 0.0

    chunk = y[:n] * np.hanning(n).astype(np.float32)
    mag = np.abs(np.fft.rfft(chunk)).astype(np.float32)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))

    mask = (freqs >= low) & (freqs <= high)

    if not np.any(mask):
        return 0.0

    return float(np.sum(mag[mask] ** 2))


def v82_score_pair(self, pair):
    sr = v82_sr(self)

    try:
        y = self.get_audio(int(pair))
    except Exception:
        y = np.zeros(512, dtype=np.float32)

    y = np.asarray(y, dtype=np.float32)

    if len(y) == 0:
        y = np.zeros(512, dtype=np.float32)

    attack = y[:min(len(y), int(sr * 0.52))]

    sub = v82_band_energy(attack, sr, 35, 100)
    low = v82_band_energy(attack, sr, 100, 250)
    mid = v82_band_energy(attack, sr, 250, 2500)
    high = v82_band_energy(attack, sr, 2500, 9000)
    air = v82_band_energy(attack, sr, 9000, 16000)

    total = sub + low + mid + high + air + 1e-9

    sub_r = sub / total
    low_r = low / total
    mid_r = mid / total
    high_r = high / total
    air_r = air / total

    rms = float(np.sqrt(np.mean(attack * attack) + 1e-12))

    if len(attack) > 3:
        zcr = float(np.mean(np.abs(np.diff(np.signbit(attack).astype(np.float32)))))
    else:
        zcr = 0.0

    kick = 3.2 * sub_r + 2.0 * low_r - 1.2 * high_r - 1.0 * air_r + 0.2 * rms
    snare = 1.3 * mid_r + 1.6 * high_r + 0.8 * air_r + 0.6 * zcr - 0.8 * sub_r
    hat = 2.3 * high_r + 2.0 * air_r + 0.8 * zcr - 1.0 * sub_r - 0.7 * low_r
    ghost = 0.7 * snare + 0.4 * high_r - 0.55 * rms

    return {
        "kick": float(kick),
        "snare": float(snare),
        "hat": float(hat),
        "ghost_snare": float(ghost),
    }


def v82_auto_role_pools(self):
    roles = v82_load_roles_from_overrides(self)

    bad = set(int(x) for x in roles.get("bad", []))

    # Si tu as déjà marqué à la main, on respecte.
    has_manual = any(roles.get(r) for r in ["kick", "snare", "hat", "ghost_snare"])

    if has_manual:
        for r in ["kick", "snare", "hat", "ghost_snare"]:
            roles[r] = [int(x) for x in roles.get(r, []) if int(x) not in bad]

        return roles

    pairs = []

    try:
        pairs = [int(x) for x in self.pair_values]
    except Exception:
        pairs = list(range(8))

    pairs = [p for p in pairs if p not in bad]

    scored = []

    for p in pairs:
        scores = v82_score_pair(self, p)
        scored.append((p, scores))

    def top(role, n=6, avoid=None):
        avoid = set(avoid or [])
        arr = []

        for p, scores in scored:
            if p in avoid:
                continue

            arr.append((float(scores.get(role, 0.0)), p))

        arr.sort(reverse=True)
        return [p for score, p in arr[:n]]

    kick = top("kick", 6)
    snare = top("snare", 6, avoid=kick[:2])
    hat = top("hat", 8, avoid=set(kick[:2]) | set(snare[:2]))
    ghost = top("ghost_snare", 6, avoid=kick[:2])

    roles["kick"] = kick or pairs[:1]
    roles["snare"] = snare or pairs[:1]
    roles["hat"] = hat or pairs[:1]
    roles["ghost_snare"] = ghost or roles["snare"] or pairs[:1]
    roles["bad"] = sorted(bad)

    print("")
    print("[v140] ROLE POOLS")
    for r in ["kick", "snare", "hat", "ghost_snare", "bad"]:
        print(f"[v140] {r:12s}:", roles.get(r, []))
    print("")

    return roles


def v82_pick_pair(rng, pool, fallback):
    pool = [int(x) for x in pool if x is not None]

    if not pool:
        pool = [int(x) for x in fallback if x is not None]

    if not pool:
        return 0

    # Tirage biaisé vers les meilleurs, mais pas toujours le premier.
    limit = min(len(pool), rng.choice([2, 3, 4, 6, len(pool)]))
    limit = max(1, limit)

    return int(rng.choice(pool[:limit]))


def v82_make_note(self, step, pair, role, nonce):
    step = int(step)

    if step % 2 != 0:
        step -= 1

    step = max(0, min(62, step))

    length = 2

    try:
        lane = int(self.pair_to_lane.get(int(pair), 0))
    except Exception:
        lane = 0

    return {
        "id": 0,
        "x_step": int(step),
        "lane": lane,
        "pair": int(pair),
        "length": int(length),
        "variation_bar": int(step) // 8,
        "variation_pos": int(step) % 8,
        "hit_slot": int(step) // v82_hit_spacing(),
        "ai_generated": True,
        "ai_model": "v82_new_candidate_each_click",
        "learned_role": str(role),
        "prior_label": str(role),
        "v82_nonce": int(nonce),
        "v82_no_overlap": True,
        "v82_new_each_click": True,
    }


def v82_build_candidate_once(self, rng, nonce):
    roles = v82_auto_role_pools(self)

    fallback = []

    try:
        fallback = [int(x) for x in self.pair_values]
    except Exception:
        fallback = list(range(8))

    bad = set(int(x) for x in roles.get("bad", []))
    fallback = [p for p in fallback if p not in bad] or fallback

    template = list(rng.choice(V82_TEMPLATES))

    # Micro variations de grille : pas de superposition, juste rôles différents.
    if rng.random() < 0.35:
        idx = rng.choice([1, 2, 5, 6, 9, 10, 13, 14])
        template[idx] = rng.choice(["hat", "ghost_snare", "kick"])

    if rng.random() < 0.25:
        idx = rng.choice([2, 6, 10, 14])
        template[idx] = rng.choice(["hat", "ghost_snare"])

    if rng.random() < 0.20:
        idx = rng.choice([4, 8, 12])
        template[idx] = rng.choice(["kick", "hat"])

    pattern = []
    previous_pair = None

    for step, role in zip(V82_STEPS, template):
        pool = roles.get(role, []) or fallback

        pair = v82_pick_pair(rng, pool, fallback)

        # Évite de répéter exactement la même slice deux slots de suite si possible.
        if previous_pair is not None and pair == previous_pair and len(pool) > 1:
            for _ in range(6):
                alt = v82_pick_pair(rng, pool, fallback)
                if alt != previous_pair:
                    pair = alt
                    break

        previous_pair = pair

        pattern.append(v82_make_note(self, step, pair, role, nonce))

    for i, item in enumerate(pattern):
        item["id"] = i

    return pattern


def v82_generate_candidate(self, event=None):
    """
    Nouvelle candidate à chaque clic.
    Évite aussi les fingerprints rejetés.
    """
    current = v82_current_break(self)

    try:
        self.stop_playhead()
        self.stop_audio()
    except Exception:
        pass

    nonce = int(getattr(self, "v82_generation_nonce", 0)) + 1
    self.v82_generation_nonce = nonce

    rejected = v82_get_rejected_set(current)

    recent = getattr(self, "v82_recent_fingerprints", [])
    recent_set = set(recent)

    seed_base = time.time_ns() + nonce * 1000003 + random.randint(0, 99999999)

    chosen = None
    chosen_fp = None

    for attempt in range(80):
        rng = random.Random(seed_base + attempt * 7919)
        pattern = v82_build_candidate_once(self, rng, nonce + attempt)
        fp = v82_pattern_fingerprint(pattern)

        if fp not in rejected and fp not in recent_set:
            chosen = pattern
            chosen_fp = fp
            break

        chosen = pattern
        chosen_fp = fp

    self.pattern = chosen or []
    self.selected_id = self.pattern[0]["id"] if self.pattern else None

    recent.append(chosen_fp)
    self.v82_recent_fingerprints = recent[-80:]

    try:
        self.draw()
        self.refresh_panel()
    except Exception:
        pass

    try:
        self.write_latest_pattern(reason="v82_generate_new_candidate_each_click")
    except Exception as exc:
        print(f"[v140] write_latest_pattern impossible : {exc}")

    try:
        self.v67_candidate_before_edit = json.loads(json.dumps(self.pattern, ensure_ascii=False))
        self.v67_candidate_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.v67_candidate_safe = current
    except Exception:
        pass

    print("")
    print("[v140] NEW CANDIDATE")
    print("[v140] break :", current)
    print("[v140] nonce :", nonce)
    print("[v140] rejected known:", len(rejected))
    print("[v140] fingerprint rejeté ?: ", chosen_fp in rejected)
    for item in self.pattern:
        end = int(item["x_step"]) + int(item["length"]) - 1
        print(
            f"[v140] step {item['x_step']:02d}/{end:02d} "
            f"role={item.get('learned_role'):12s} pair={item.get('pair')}"
        )
    print("")

    v82_status(self, f"Nouvelle candidate v82 générée sur {current}.")

    return "break"


def v82_feedback_payload(self, verdict, train_now):
    current = v82_current_break(self)

    try:
        candidate_before = self.v67_candidate_before_edit
    except Exception:
        candidate_before = None

    return {
        "version": "v82_new_candidate_each_click",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "break": current,
        "verdict": verdict,
        "train_now": bool(train_now),
        "pattern": getattr(self, "pattern", []) or [],
        "candidate_before_edit": candidate_before,
        "fingerprint": v82_pattern_fingerprint(getattr(self, "pattern", []) or []),
    }


def v82_run_training(self):
    candidates = [
        Path("pipeline/04_train_beat_roles_v04_strict.py"),
        Path("pipeline/04_train_beat_roles_v03.py"),
    ]

    script = None

    for p in candidates:
        if p.exists():
            script = p
            break

    if script is None:
        v82_status(self, "Feedback sauvegardé, mais aucun script training trouvé.")
        return

    cmd = [sys.executable, str(script)]
    print("[v140] TRAINING:", " ".join(cmd))

    try:
        subprocess.Popen(cmd, cwd=str(Path(".").resolve()))
    except Exception as exc:
        print(f"[v140] training impossible : {exc}")


def v82_save_feedback(self, verdict, train_now=False, auto_new_after_reject=False):
    current = v82_current_break(self)
    pattern = getattr(self, "pattern", []) or []

    if verdict != "reject_bad" and not pattern:
        v82_status(self, "Pattern vide : rien à sauvegarder.")
        return "break"

    V82_VALIDATED_DIR.mkdir(parents=True, exist_ok=True)
    V82_REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    V82_FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = v82_feedback_payload(self, verdict, train_now)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe = v82_slug(current)

    if verdict == "reject_bad":
        out_path = V82_REJECTED_DIR / f"{stamp}_{safe}_reject_bad.json"
        rejected_fp = v82_save_rejected_fp(current, pattern)
        payload["rejected_fingerprint_saved"] = rejected_fp
    elif train_now:
        out_path = V82_VALIDATED_DIR / f"{stamp}_{safe}_good_train.json"
    else:
        out_path = V82_VALIDATED_DIR / f"{stamp}_{safe}_good_save_only.json"

    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with V82_FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print("")
    print("[v140] FEEDBACK SAUVÉ")
    print("[v140] break  :", current)
    print("[v140] verdict:", verdict)
    print("[v140] train  :", train_now)
    print("[v140] fichier:", out_path)
    print("")

    if train_now:
        v82_run_training(self)

    if verdict == "reject_bad":
        v82_status(self, f"Reject sauvegardé pour {current}. Nouvelle candidate...")
        if auto_new_after_reject:
            return v82_generate_candidate(self)
    elif train_now:
        v82_status(self, f"Good après modifs + training pour {current}.")
    else:
        v82_status(self, f"Good sans train sauvegardé pour {current}.")

    return "break"


def v82_good_after_modifs(self, event=None):
    return v82_save_feedback(self, "good_after_modifs", train_now=True)


def v82_good_no_train(self, event=None):
    return v82_save_feedback(self, "good_no_train", train_now=False)


def v82_reject_bad(self, event=None):
    # Important : Reject régénère directement une nouvelle candidate.
    return v82_save_feedback(self, "reject_bad", train_now=False, auto_new_after_reject=True)


def v82_all_widgets(widget):
    out = [widget]

    try:
        children = widget.winfo_children()
    except Exception:
        children = []

    for child in children:
        out.extend(v82_all_widgets(child))

    return out


def v82_patch_training_buttons(self):
    patched = 0

    for widget in v82_all_widgets(self.root):
        try:
            text = str(widget.cget("text")).strip().lower()
        except Exception:
            continue

        try:
            if text == "generate candidate":
                widget.configure(command=self.generate_candidate)
                patched += 1
            elif text == "good après modifs":
                widget.configure(command=self.good_after_modifs_v82)
                patched += 1
            elif text == "good sans train":
                widget.configure(command=self.good_no_train_v82)
                patched += 1
            elif text in ["reject / bad", "reject/bad", "bad"]:
                widget.configure(command=self.reject_bad_v82)
                patched += 1
        except Exception:
            pass

    print(f"[v140] boutons training patchés : {patched}")


_old_v82_build_ui = SliceIndexTracker.build_ui


def v82_build_ui(self):
    _old_v82_build_ui(self)

    try:
        self.root.title(f"BreakbeatAI v140 — Princess clean toolbar")
    except Exception:
        pass

    try:
        v82_patch_training_buttons(self)
    except Exception as exc:
        print(f"[v140] patch boutons impossible : {exc}")

    v82_status(
        self,
        f"v82 : Generate Candidate = nouvelle variation à chaque clic. Break = {v82_current_break(self)}."
    )


SliceIndexTracker.good_after_modifs_v82 = v82_good_after_modifs
SliceIndexTracker.good_no_train_v82 = v82_good_no_train
SliceIndexTracker.reject_bad_v82 = v82_reject_bad

SliceIndexTracker.generate_candidate = v82_generate_candidate
SliceIndexTracker.generate_full_candidate = v82_generate_candidate
SliceIndexTracker.generate_ai_pattern = v82_generate_candidate
SliceIndexTracker.generate_cross_song_template = v82_generate_candidate
SliceIndexTracker.generate_safe_experiment = v82_generate_candidate
SliceIndexTracker.generate_locked_roles = v82_generate_candidate
SliceIndexTracker.generate_role_aware = v82_generate_candidate
SliceIndexTracker.generate_safe_role_ai = v82_generate_candidate

SliceIndexTracker.build_ui = v82_build_ui









# ---------------------------------------------------------------------
# v110 EXTENSION PROPRE — PAS DE CALLBACKS FANTÔMES
# ---------------------------------------------------------------------
# Un seul bloc final :
# - loop 64
# - variations du premier 32 sur Reject / Bad
# - Reject / Bad change aussi les lignes kick/snare/hat si possible
# - Export WAV dans la barre du haut
# - Ctrl+E exporte
# - Espace = Loop / Space
# - pas de .after() récursif, pas de bouton flottant, pas de destroy tardif
# ---------------------------------------------------------------------

V110_OVERRIDES_PATH = Path("dataset/learning/break_role_overrides_v01.json")
V110_LOCKS_PATH = Path("dataset/learning/main_role_locks_v87.json")
V110_STATE_PATH = Path("dataset/learning/first32_variants_v110.json")
V110_EXPORT_DIR = Path("exports")
V110_PAIR_DIR = Path("dataset/pair_blocks_v02")

V110_FIRST_32_VARIANTS = [
    {
        "id": "A_original_first32",
        "grid": [
            (0,  "kick"), (2,  "hat"), (4,  "snare"), (6,  "hat"),
            (8,  "hat"),  (10, "kick"), (12, "snare"), (14, "hat"),
            (16, "kick"), (18, "hat"), (20, "snare"), (22, "hat"),
            (24, "hat"),  (26, "kick"), (28, "snare"), (30, "hat"),
        ],
    },
    {
        "id": "B_first32_kick_forward",
        "grid": [
            (0,  "kick"), (2,  "hat"), (4,  "snare"), (6,  "hat"),
            (8,  "kick"), (10, "hat"), (12, "snare"), (14, "hat"),
            (16, "kick"), (18, "hat"), (20, "snare"), (22, "hat"),
            (24, "hat"),  (26, "kick"), (28, "snare"), (30, "hat"),
        ],
    },
    {
        "id": "C_first32_kick_late",
        "grid": [
            (0,  "kick"), (2,  "hat"),  (4,  "snare"), (6,  "hat"),
            (8,  "hat"),  (10, "kick"), (12, "snare"), (14, "hat"),
            (16, "hat"),  (18, "kick"), (20, "snare"), (22, "hat"),
            (24, "hat"),  (26, "kick"), (28, "snare"), (30, "hat"),
        ],
    },
    {
        "id": "D_first32_last_push",
        "grid": [
            (0,  "kick"), (2,  "hat"), (4,  "snare"), (6,  "hat"),
            (8,  "hat"),  (10, "kick"), (12, "snare"), (14, "hat"),
            (16, "kick"), (18, "hat"), (20, "snare"), (22, "hat"),
            (24, "kick"), (26, "hat"), (28, "snare"), (30, "hat"),
        ],
    },
    {
        "id": "E_first32_hat_answer",
        "grid": [
            (0,  "kick"), (2,  "hat"), (4,  "snare"), (6,  "hat"),
            (8,  "hat"),  (10, "hat"), (12, "snare"), (14, "kick"),
            (16, "kick"), (18, "hat"), (20, "snare"), (22, "hat"),
            (24, "hat"),  (26, "kick"), (28, "snare"), (30, "hat"),
        ],
    },
    {
        "id": "F_first32_extra_answer_kick",
        "grid": [
            (0,  "kick"), (2,  "hat"), (4,  "snare"), (6,  "hat"),
            (8,  "hat"),  (10, "kick"), (12, "snare"), (14, "hat"),
            (16, "kick"), (18, "hat"), (20, "snare"), (22, "kick"),
            (24, "hat"),  (26, "hat"), (28, "snare"), (30, "hat"),
        ],
    },
]

V110_SECOND_32_RESPONSES = ["strict_mirror_first32_no_sample_variation"]

V110_SNARE_STEPS = [4, 12, 20, 28, 36, 44, 52, 60]
_V110_AUDIO_CACHE = {}


def v110_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        elif hasattr(self, "output_label"):
            self.output_label.config(text=text)
    except Exception:
        pass


def v110_norm(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def v110_slug(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "unknown_break"


def v110_load_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def v110_save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def v110_all_widgets(widget):
    out = [widget]
    try:
        children = widget.winfo_children()
    except Exception:
        children = []
    for child in children:
        out.extend(v110_all_widgets(child))
    return out


def v110_widget_text(widget):
    try:
        return str(widget.cget("text")).strip()
    except Exception:
        return ""


def v110_pairblock_stems():
    out = {}
    for p in sorted(V110_PAIR_DIR.glob("*_pair_blocks_v02.json")):
        stem = p.name.replace("_pair_blocks_v02.json", "")
        out[v110_norm(stem)] = stem
    return out


def v110_current_break(self):
    stems = v110_pairblock_stems()

    for widget in v110_all_widgets(self.root):
        try:
            value = str(widget.get()).strip()
        except Exception:
            continue
        n = v110_norm(value)
        if n in stems:
            return stems[n]

    for attr in [
        "current_source", "source", "selected_source", "active_source",
        "loaded_source", "current_safe", "safe", "break_name",
        "current_break", "selected_break", "active_break",
    ]:
        try:
            value = str(getattr(self, attr)).strip()
        except Exception:
            continue
        n = v110_norm(value)
        if n in stems:
            return stems[n]

    try:
        project = getattr(self, "project", None)
        if isinstance(project, dict):
            for key in ["current_source", "source", "safe", "name", "break", "break_name"]:
                value = str(project.get(key, "")).strip()
                n = v110_norm(value)
                if n in stems:
                    return stems[n]
    except Exception:
        pass

    return "Camo_Break_-_3A"


def v110_force_64(self):
    for attr in [
        "total_steps", "loop_steps", "grid_steps", "pattern_steps",
        "num_steps", "step_count", "n_steps", "steps",
        "cols", "columns", "grid_cols", "loop_len_steps",
        "loop_length_steps",
    ]:
        try:
            if hasattr(self, attr):
                setattr(self, attr, 64)
        except Exception:
            pass

    try:
        if hasattr(self, "canvas"):
            self.canvas.configure(scrollregion=(0, 0, 4096, 2400))
    except Exception:
        pass


def v110_all_pairs(self):
    try:
        pairs = [int(x) for x in self.pair_values]
    except Exception:
        pairs = [0]
    return pairs or [0]


def v110_find_record(data, current):
    br = data.get("breaks", {}).get(current)
    if br is not None:
        return br
    for name, obj in data.get("breaks", {}).items():
        if v110_norm(name) == v110_norm(current):
            return obj
    return None


def v110_load_roles(self):
    current = v110_current_break(self)
    roles = {"kick": [], "snare": [], "hat": [], "ghost_snare": [], "bad": []}

    data = v110_load_json(V110_OVERRIDES_PATH, {"breaks": {}})
    br = v110_find_record(data, current)

    if isinstance(br, dict):
        raw = br.get("roles", {})
        if isinstance(raw, dict):
            for role in roles:
                clean = []
                for x in raw.get(role, []):
                    try:
                        clean.append(int(x))
                    except Exception:
                        pass
                roles[role] = sorted(set(clean))

    bad = set(roles.get("bad", []))
    for role in ["kick", "snare", "hat", "ghost_snare"]:
        roles[role] = [int(p) for p in roles.get(role, []) if int(p) not in bad]
    return roles


def v110_load_locks(self):
    current = v110_current_break(self)
    data = v110_load_json(V110_LOCKS_PATH, {"breaks": {}})
    br = v110_find_record(data, current)
    if not isinstance(br, dict):
        return {}

    raw = br.get("main_roles", {})
    if not isinstance(raw, dict):
        return {}

    locks = {}
    for role in ["kick", "snare", "hat"]:
        try:
            locks[role] = int(raw[role])
        except Exception:
            pass
    return locks


def v110_candidates(self, roles, locks, role):
    out = []
    if role in locks:
        out.append((int(locks[role]), "manual_lock"))
    for p in roles.get(role, []):
        out.append((int(p), "role_pool"))
    for p in v110_all_pairs(self):
        out.append((int(p), "fallback_distinct"))

    clean = []
    seen = set()
    for pair, source in out:
        if pair in seen:
            continue
        seen.add(pair)
        clean.append((pair, source))
    return clean


def v110_choose_main_pairs(self, roles, locks, role_offset=0):
    chosen = {}
    used = set()

    for role in ["snare", "kick", "hat"]:
        candidates = v110_candidates(self, roles, locks, role)
        if not candidates:
            chosen[role] = {"pair": 0, "source": "fallback_zero"}
            used.add(0)
            continue

        picked = None
        n = len(candidates)
        for delta in range(n):
            idx = (role_offset + delta) % n
            pair, source = candidates[idx]
            if pair not in used:
                picked = (pair, source)
                break

        if picked is None:
            picked = candidates[role_offset % n]

        pair, source = picked
        chosen[role] = {"pair": int(pair), "source": f"{source}_offset_{role_offset}"}
        used.add(int(pair))
    return chosen


def v110_load_state():
    data = v110_load_json(
        V110_STATE_PATH,
        {"version": "first32_variants_v110", "updated_at": None, "breaks": {}},
    )
    data.setdefault("breaks", {})
    return data


def v110_save_state(data):
    data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    v110_save_json(V110_STATE_PATH, data)


def v110_next_first32_index(self, reject_mode=False):
    current = v110_current_break(self)
    data = v110_load_state()
    br = data["breaks"].setdefault(current, {})
    br.setdefault("first32_index", -1)
    br.setdefault("reject_count", 0)

    if reject_mode:
        br["reject_count"] = int(br.get("reject_count", 0)) + 1

    last = int(br.get("first32_index", -1))
    idx = (last + 1) % len(V110_FIRST_32_VARIANTS)

    br["first32_index"] = idx
    br["last_variant"] = V110_FIRST_32_VARIANTS[idx]["id"]
    br["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    v110_save_state(data)

    role_offset = int(br.get("reject_count", 0))
    return idx, role_offset




def v110_make_second32_from_first32(first_grid, response_id):
    """
    v124 :
    Le deuxième 32 est une copie stricte du premier 32.

    - mêmes positions
    - mêmes rôles
    - décalage +32 cases uniquement
    """
    return [(int(step) + 32, role) for step, role in first_grid]



def v110_alt_pair_for_second32(self, roles, chosen, role, occurrence, role_offset):
    """
    v124 :
    ZÉRO variation de sample dans le deuxième 32.

    Le deuxième 32 reprend le même pair que le premier 32 :
    - même kick
    - même snare
    - même hat
    """
    main_pair = int(chosen[role]["pair"])
    return main_pair, f"v124_strict_mirror_same_{role}"


def v110_make_note(self, step, pair, role, source, first32_id, response_id, role_offset):
    step = int(step)
    if step % 2 != 0:
        step -= 1
    step = max(0, min(62, step))

    try:
        lane = int(self.pair_to_lane.get(int(pair), 0))
    except Exception:
        lane = 0

    return {
        "id": 0,
        "x_step": int(step),
        "lane": lane,
        "pair": int(pair),
        "length": 2,
        "variation_bar": int(step) // 8,
        "variation_pos": int(step) % 8,
        "hit_slot": int(step) // 2,
        "ai_generated": True,
        "ai_model": "v116_export_case_length_fix",
        "learned_role": str(role),
        "prior_label": str(role),
        "main_role_source": str(source),
        "v110_loop64": True,
        "v110_first32_variant": str(first32_id),
        "v110_second32_response": str(response_id),
        "v110_role_offset": int(role_offset),
    }


def v110_validate_pattern(pattern):
    if len(pattern) != 32:
        raise RuntimeError(f"BUG v110 : pattern devrait avoir 32 notes, obtenu {len(pattern)}")

    occupied = set()
    for item in pattern:
        x = int(item["x_step"])
        role = str(item["learned_role"])

        if x % 2 != 0:
            raise RuntimeError(f"BUG v110 : départ impair {x}")

        if x < 0 or x > 62:
            raise RuntimeError(f"BUG v110 : step hors 64 : {x}")

        if x in occupied or (x + 1) in occupied:
            raise RuntimeError(f"BUG v110 : collision à {x}/{x+1}")

        occupied.add(x)
        occupied.add(x + 1)

        if role == "snare" and x not in V110_SNARE_STEPS:
            raise RuntimeError(f"BUG v110 : snare à {x}, interdit")


def v110_generate(self, event=None, reject_mode=False):
    v110_force_64(self)

    current = v110_current_break(self)
    roles = v110_load_roles(self)
    locks = v110_load_locks(self)

    first_idx, role_offset = v110_next_first32_index(self, reject_mode=reject_mode)
    first_variant = V110_FIRST_32_VARIANTS[first_idx]
    first32_id = first_variant["id"]
    first_grid = first_variant["grid"]

    response_id = V110_SECOND_32_RESPONSES[first_idx % len(V110_SECOND_32_RESPONSES)]
    second_grid = v110_make_second32_from_first32(first_grid, response_id)

    chosen = v110_choose_main_pairs(self, roles, locks, role_offset=role_offset)

    self.v110_current_first32_id = str(first32_id)
    self.v110_current_response_id = str(response_id)
    self.v110_role_offset = int(role_offset)

    full_grid = list(first_grid) + list(second_grid)

    print("")
    if reject_mode:
        print("[v140] REJECT/BAD -> variation premier 32 + autres lignes")
    else:
        print("[v140] GENERATE 64")
    print("[v140] break:", current)
    print("[v140] first32:", first32_id)
    print("[v140] response:", response_id)
    print("[v140] role_offset:", role_offset)
    for role in ["kick", "snare", "hat"]:
        lane = "?"
        try:
            lane = self.pair_to_lane.get(chosen[role]["pair"], "?")
        except Exception:
            pass
        print(f"[v140] {role:6s}: pair={chosen[role]['pair']} lane={lane} source={chosen[role]['source']}")
    print("")

    try:
        self.stop_playhead()
        self.stop_audio()
    except Exception:
        pass

    pattern = []
    second_occ = {"kick": 0, "snare": 0, "hat": 0}

    for step, role in full_grid:
        if step < 32:
            pair = chosen[role]["pair"]
            source = chosen[role]["source"]
        else:
            occ = second_occ.get(role, 0)
            second_occ[role] = occ + 1
            pair, source = v110_alt_pair_for_second32(self, roles, chosen, role, occ, role_offset)

        pattern.append(v110_make_note(self, step, pair, role, source, first32_id, response_id, role_offset))

    v110_validate_pattern(pattern)

    pattern = sorted(pattern, key=lambda item: int(item["x_step"]))
    for i, item in enumerate(pattern):
        item["id"] = i

    self.pattern = pattern
    self.selected_id = self.pattern[0]["id"] if self.pattern else None

    try:
        self.draw()
    except Exception as exc:
        print(f"[v140] draw impossible : {exc}")

    v110_force_64(self)

    try:
        self.refresh_panel()
    except Exception as exc:
        print(f"[v140] refresh_panel impossible : {exc}")

    try:
        self.write_latest_pattern(reason="v116_export_case_length_fix")
    except Exception as exc:
        print(f"[v140] write_latest_pattern impossible : {exc}")

    v110_status(self, f"v110 : {first32_id}. Espace = Loop / Space.")
    return "break"


def v110_reject_bad(self, event=None):
    return v110_generate(self, event=event, reject_mode=True)


# -----------------------------
# Export WAV
# -----------------------------

def v110_read_wav(path):
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sw == 1:
        y = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        y = (y - 128.0) / 128.0
    elif sw == 2:
        y = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        vals = b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)
        vals = np.where(vals & 0x800000, vals - 0x1000000, vals)
        y = vals.astype(np.float32) / 8388608.0
    elif sw == 4:
        y = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError("format WAV non supporté")

    if ch > 1:
        y = y.reshape(-1, ch).mean(axis=1)
    return y.astype(np.float32), int(sr)


def v110_read_soundfile(path):
    import soundfile as sf
    y, sr = sf.read(str(path), always_2d=False, dtype="float32")
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y, int(sr)


def v110_read_ffmpeg(path):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg introuvable")
    sr = 44100
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "pipe:1"]
    raw = subprocess.check_output(cmd)
    y = np.frombuffer(raw, dtype=np.float32).copy()
    return y.astype(np.float32), sr


def v110_read_audio(path):
    path = Path(path).resolve()
    if path in _V110_AUDIO_CACHE:
        return _V110_AUDIO_CACHE[path]

    loaders = []
    try:
        import soundfile  # noqa
        loaders.append(v110_read_soundfile)
    except Exception:
        pass
    loaders.append(v110_read_wav)
    loaders.append(v110_read_ffmpeg)

    last = None
    for loader in loaders:
        try:
            result = loader(path)
            _V110_AUDIO_CACHE[path] = result
            return result
        except Exception as exc:
            last = exc
    raise RuntimeError(f"audio illisible {path}: {last}")


def v110_find_pairblock_file(current):
    exact = V110_PAIR_DIR / f"{current}_pair_blocks_v02.json"
    if exact.exists():
        return exact

    n_current = v110_norm(current)
    for p in sorted(V110_PAIR_DIR.glob("*_pair_blocks_v02.json")):
        stem = p.name.replace("_pair_blocks_v02.json", "")
        if v110_norm(stem) == n_current:
            return p

    for p in sorted(V110_PAIR_DIR.glob("*_pair_blocks_v02.json")):
        stem = p.name.replace("_pair_blocks_v02.json", "")
        if n_current in v110_norm(stem) or v110_norm(stem) in n_current:
            return p
    return None


def v110_extract_blocks(data):
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in ["pair_blocks", "blocks", "pairs", "slices", "items", "data"]:
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


def v110_find_existing_path(value):
    if not value:
        return None
    p = Path(str(value)).expanduser()
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path(".") / p)
        candidates.append(Path("breaks") / p)
        candidates.append(Path("breaks") / p.name)
    for c in candidates:
        if c.exists():
            return c
    return None


def v110_resolve_audio_path(block, root):
    for obj in [block, root]:
        if not isinstance(obj, dict):
            continue
        for key in ["source_audio", "audio_path", "source_path", "file", "path", "filename", "wav", "audio"]:
            p = v110_find_existing_path(obj.get(key))
            if p:
                return p
    return None


def v110_num(obj, keys, default=None):
    if not isinstance(obj, dict):
        return default
    for key in keys:
        if key not in obj:
            continue
        try:
            return float(obj[key])
        except Exception:
            pass
    return default


def v110_pair_id(block, fallback):
    try:
        return int(v110_num(block, ["pair", "slice", "slice_index", "index", "id"], fallback))
    except Exception:
        return int(fallback)


def v110_build_slice_map(current):
    pairblock_file = v110_find_pairblock_file(current)
    if pairblock_file is None:
        raise RuntimeError(f"pair_blocks introuvable pour {current}")

    root = v110_load_json(pairblock_file, None)
    blocks = v110_extract_blocks(root)
    if not blocks:
        raise RuntimeError(f"aucun block dans {pairblock_file}")

    slice_map = {}
    source_audio_path = None

    for i, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue

        pair = v110_pair_id(block, i)
        audio_path = v110_resolve_audio_path(block, root)
        if audio_path is None:
            continue

        source_audio_path = audio_path
        y, sr = v110_read_audio(audio_path)

        start = v110_num(block, ["source_start_sample", "start_sample", "start", "start_frame"], None)
        end = v110_num(block, ["source_end_sample", "end_sample", "end", "end_frame"], None)

        if start is None:
            start_ms = v110_num(block, ["source_start_ms", "start_ms"], None)
            if start_ms is not None:
                start = int(start_ms * sr / 1000.0)

        if end is None:
            end_ms = v110_num(block, ["source_end_ms", "end_ms"], None)
            if end_ms is not None:
                end = int(end_ms * sr / 1000.0)

        if start is None:
            start = 0

        if end is None:
            dur_ms = v110_num(block, ["duration_ms", "dur_ms"], 250.0)
            end = int(start) + int(dur_ms * sr / 1000.0)

        start = int(max(0, min(len(y), int(start))))
        end = int(max(start + 1, min(len(y), int(end))))
        seg = y[start:end].astype(np.float32)

        if len(seg) > 0:
            fade = min(int(sr * 0.004), max(1, len(seg) // 8))
            if fade > 1:
                seg[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
                seg[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

        slice_map[pair] = {"audio": seg, "sr": sr, "source_audio_path": str(audio_path), "pairblock_file": str(pairblock_file)}

    if not slice_map:
        raise RuntimeError(f"aucune slice audio exportable dans {pairblock_file}")

    return slice_map, source_audio_path, pairblock_file



def v110_infer_case_seconds(self, source_audio_path):
    """
    v116 FIX :
    Les pair_blocks viennent d'un break 32 cases.
    Même quand le pattern généré fait 64 cases, la durée d'UNE case doit rester :

        durée du break source / 32

    Avant, certaines versions utilisaient source_duration / 64,
    ce qui rendait l'export deux fois trop court et coupait les notes.
    """

    # 1) Si l'app expose step_ms, on respecte cette valeur.
    for attr in ["step_ms", "step_ms_var"]:
        try:
            value = getattr(self, attr)
            if hasattr(value, "get"):
                value = value.get()
            value = float(value)
            if value > 1:
                return value / 1000.0, f"{attr}_{value:g}ms"
        except Exception:
            pass

    # 2) Si BPM disponible :
    # une case = double croche dans notre grille 32/64 actuelle.
    bpm = None

    for attr in ["bpm", "tempo", "project_bpm", "current_bpm"]:
        try:
            value = getattr(self, attr)
            if value:
                bpm = float(value)
                break
        except Exception:
            pass

    try:
        if bpm is None and isinstance(getattr(self, "project", None), dict):
            for key in ["bpm", "tempo"]:
                if self.project.get(key):
                    bpm = float(self.project[key])
                    break
    except Exception:
        pass

    if bpm is not None and bpm > 20:
        return 60.0 / bpm / 4.0, f"bpm_{bpm:g}"

    # 3) Source audio :
    # IMPORTANT : source / 32, pas source / 64.
    try:
        if source_audio_path:
            y, sr = v110_read_audio(source_audio_path)
            dur = len(y) / float(sr)

            if dur > 0.2:
                return dur / 32.0, "source_duration_div_32_v116"
    except Exception:
        pass

    # 4) Fallback.
    return 60.0 / 155.0 / 4.0, "default_155bpm_v116"


def v110_render_pattern_to_audio(self):
    current = v110_current_break(self)
    pattern = list(getattr(self, "pattern", []) or [])
    if not pattern:
        raise RuntimeError("pattern vide : clique Generate Candidate avant Export WAV")

    slice_map, source_audio_path, pairblock_file = v110_build_slice_map(current)
    case_seconds, timing_source = v110_infer_case_seconds(self, source_audio_path)

    sr = 44100
    try:
        first = next(iter(slice_map.values()))
        sr = int(first["sr"])
    except Exception:
        pass

    total_cases = 64
    total_samples = int(total_cases * case_seconds * sr)
    out = np.zeros(total_samples, dtype=np.float32)

    rendered = 0
    skipped = 0

    for item in pattern:
        try:
            step = int(item.get("x_step", 0))
            pair = int(item.get("pair", 0))
            length = int(item.get("length", 2) or 2)
        except Exception:
            skipped += 1
            continue

        if pair not in slice_map:
            skipped += 1
            print(f"[v140] skip pair introuvable : {pair}")
            continue

        seg = slice_map[pair]["audio"].astype(np.float32)
        seg_sr = int(slice_map[pair]["sr"])

        if seg_sr != sr:
            ratio = sr / float(seg_sr)
            new_len = max(1, int(len(seg) * ratio))
            old_x = np.linspace(0.0, 1.0, len(seg), endpoint=False)
            new_x = np.linspace(0.0, 1.0, new_len, endpoint=False)
            seg = np.interp(new_x, old_x, seg).astype(np.float32)

        max_len = int(max(1, length) * case_seconds * sr)
        if max_len > 0 and len(seg) > max_len:
            seg = seg[:max_len].copy()
            fade = min(int(sr * 0.006), max(1, len(seg) // 6))
            if fade > 1:
                seg[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

        pos = int(step * case_seconds * sr)
        end = min(len(out), pos + len(seg))
        if end <= pos:
            skipped += 1
            continue

        out[pos:end] += seg[:end - pos]
        rendered += 1

    peak = float(np.max(np.abs(out)) + 1e-12)
    if peak > 0.98:
        out = out * (0.98 / peak)

    info = {
        "break": current,
        "pairblock_file": str(pairblock_file),
        "source_audio_path": str(source_audio_path),
        "case_seconds": case_seconds,
        "timing_source": timing_source,
        "sample_rate": sr,
        "rendered_notes": rendered,
        "skipped_notes": skipped,
        "duration_seconds": len(out) / float(sr),
        "exporter": "v116_export_case_length_fix",
    }

    return out.astype(np.float32), sr, info


def v110_write_wav(path, audio, sr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype("<i2")

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())


def v110_export_wav_current(self, event=None):
    try:
        current = v110_current_break(self)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        variant = getattr(self, "v110_current_first32_id", None) or "pattern"

        audio, sr, info = v110_render_pattern_to_audio(self)

        out_name = f"{v110_slug(current)}_{v110_slug(variant)}_v110_{stamp}.wav"
        out_path = V110_EXPORT_DIR / out_name

        v110_write_wav(out_path, audio, sr)

        info_path = out_path.with_suffix(".json")
        info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

        print("")
        print("[v140] EXPORT WAV OK")
        print("[v140] break:", current)
        print("[v140] wav  :", out_path)
        print("[v140] json :", info_path)
        print("[v140] duration:", round(info["duration_seconds"], 3), "s")
        print("[v140] rendered:", info["rendered_notes"], "skipped:", info["skipped_notes"])
        print("")

        v110_status(self, f"Export WAV OK : {out_path}")

    except Exception as exc:
        print("")
        print("[v140] EXPORT WAV ERREUR")
        print("[v140]", exc)
        print(traceback.format_exc())
        print("")
        v110_status(self, f"Export WAV erreur : {exc}")

    return "break"


# -----------------------------
# Espace solide
# -----------------------------

def v110_find_loop_button(self):
    for widget in v110_all_widgets(self.root):
        if v110_widget_text(widget).lower() == "loop / space":
            return widget
    return None


def v110_space_play(self, event=None):
    if getattr(self, "_v110_space_busy", False):
        return "break"

    self._v110_space_busy = True

    try:
        btn = v110_find_loop_button(self)
        if btn is not None:
            print("[v140] SPACE -> Loop / Space")
            try:
                btn.invoke()
            except Exception as exc:
                print("[v140] invoke Loop / Space impossible :", exc)
            return "break"

        print("[v140] SPACE : bouton Loop / Space introuvable.")
        v110_status(self, "Espace : bouton Loop / Space introuvable.")
        return "break"
    finally:
        try:
            self.root.after(80, lambda: setattr(self, "_v110_space_busy", False))
        except Exception:
            self._v110_space_busy = False


def v110_install_space_bindings(self):
    try:
        self.root.bind_all("<space>", self.space_play_v110)
        self.root.bind_all("<KeyPress-space>", self.space_play_v110)
    except Exception:
        pass

    for widget in v110_all_widgets(self.root):
        try:
            tags = list(widget.bindtags())
            if "V110Space" not in tags:
                widget.bindtags(("V110Space",) + tuple(tags))
        except Exception:
            pass

    try:
        self.root.bind_class("V110Space", "<space>", self.space_play_v110)
        self.root.bind_class("V110Space", "<KeyPress-space>", self.space_play_v110)
    except Exception:
        pass

    print("[v140] espace installé.")


# -----------------------------
# Bouton export en haut
# -----------------------------

def v110_find_top_toolbar(self):
    wanted = {"generate candidate", "good après modifs", "good sans train", "reject / bad"}
    groups = {}

    for widget in v110_all_widgets(self.root):
        text = v110_widget_text(widget).lower()
        if text not in wanted:
            continue

        try:
            parent = widget.master
        except Exception:
            continue

        groups.setdefault(parent, {"widgets": {}, "count": 0, "y": 999999})
        groups[parent]["widgets"][text] = widget
        groups[parent]["count"] = len(groups[parent]["widgets"])
        try:
            groups[parent]["y"] = min(groups[parent]["y"], int(parent.winfo_rooty()))
        except Exception:
            pass

    if not groups:
        return None, {}

    best_parent = None
    best_score = None
    for parent, info in groups.items():
        score = (info["y"], -info["count"])
        if best_score is None or score < best_score:
            best_score = score
            best_parent = parent

    return best_parent, groups[best_parent]["widgets"]


def v110_add_export_button(self):
    if getattr(self, "_v110_export_button_added", False):
        return

    parent, widgets = v110_find_top_toolbar(self)
    if parent is None:
        parent = self.root
        widgets = {}

    reject_widget = widgets.get("reject / bad")

    btn = tk.Button(
        parent,
        text="Export WAV",
        command=self.export_wav_current_v110,
        bg="#2d7a4b",
        fg="#ffffff",
        activebackground="#39a96b",
        activeforeground="#ffffff",
        font=("Sans", 9, "bold"),
        padx=8,
        pady=3,
    )

    placed = False
    manager = ""
    try:
        if reject_widget is not None:
            manager = reject_widget.winfo_manager()
    except Exception:
        pass

    if manager == "pack" and reject_widget is not None:
        try:
            btn.pack(side="left", padx=4, pady=3, after=reject_widget)
            placed = True
        except Exception:
            pass

    if not placed and manager == "grid" and reject_widget is not None:
        try:
            info = reject_widget.grid_info()
            row = int(info.get("row", 0))
            col = int(info.get("column", 0)) + 1
            btn.grid(row=row, column=col, padx=4, pady=3, sticky="w")
            placed = True
        except Exception:
            pass

    if not placed:
        try:
            btn.pack(side="left", padx=4, pady=3)
            placed = True
        except Exception:
            pass

    self.export_wav_button_v110 = btn
    self._v110_export_button_added = True
    print("[v140] Export WAV ajouté.")


def v110_patch_buttons(self):
    gen_patched = 0
    bad_patched = 0

    for widget in v110_all_widgets(self.root):
        text = v110_widget_text(widget).lower()

        try:
            if text == "generate candidate":
                widget.configure(command=self.generate_candidate)
                gen_patched += 1
            elif text in ["reject / bad", "reject", "bad"]:
                widget.configure(command=self.reject_bad_v110)
                bad_patched += 1
        except Exception:
            pass

    print(f"[v140] boutons patchés : Generate={gen_patched}, Reject/Bad={bad_patched}")


_old_v110_draw = getattr(SliceIndexTracker, "draw", None)


def v110_draw(self, *args, **kwargs):
    v110_force_64(self)

    if callable(_old_v110_draw):
        result = _old_v110_draw(self, *args, **kwargs)
    else:
        result = None

    v110_force_64(self)
    return result


_old_v110_build_ui = SliceIndexTracker.build_ui


def v110_build_ui(self):
    _old_v110_build_ui(self)

    v110_force_64(self)

    try:
        self.root.title(f"BreakbeatAI v110 clean — {v110_current_break(self)}")
    except Exception:
        pass

    try:
        v110_patch_buttons(self)
    except Exception as exc:
        print("[v140] patch boutons impossible :", exc)

    try:
        v110_add_export_button(self)
    except Exception as exc:
        print("[v140] ajout Export WAV impossible :", exc)

    try:
        v110_install_space_bindings(self)
    except Exception as exc:
        print("[v140] bind espace impossible :", exc)

    try:
        self.root.bind("<Control-e>", self.export_wav_current_v110)
        self.root.bind("<F12>", self.generate_candidate)
        self.root.bind("<F9>", self.reject_bad_v110)
    except Exception:
        pass

    v110_status(self, "v110 clean : Espace réparé, Export WAV en haut, pas de callbacks fantômes.")


SliceIndexTracker.generate_candidate = v110_generate
SliceIndexTracker.generate_full_candidate = v110_generate
SliceIndexTracker.generate_ai_pattern = v110_generate
SliceIndexTracker.generate_cross_song_template = v110_generate
SliceIndexTracker.generate_safe_experiment = v110_generate
SliceIndexTracker.generate_locked_roles = v110_generate
SliceIndexTracker.generate_role_aware = v110_generate
SliceIndexTracker.generate_safe_role_ai = v110_generate

SliceIndexTracker.reject_bad_v110 = v110_reject_bad
SliceIndexTracker.reject_bad = v110_reject_bad
SliceIndexTracker.bad_candidate = v110_reject_bad
SliceIndexTracker.reject_candidate = v110_reject_bad
SliceIndexTracker.reject_pattern = v110_reject_bad

SliceIndexTracker.export_wav_current_v110 = v110_export_wav_current
SliceIndexTracker.space_play_v110 = v110_space_play

if callable(_old_v110_draw):
    SliceIndexTracker.draw = v110_draw

SliceIndexTracker.build_ui = v110_build_ui



# ---------------------------------------------------------------------
# v113 : bouton Export WAV visible, sans casser Espace
# ---------------------------------------------------------------------
# Base : v110 clean.
# Ne supprime aucun widget.
# Ne touche pas aux bindings espace.
# Ajoute seulement un bouton Export WAV visible près de Generate Candidate.
# ---------------------------------------------------------------------


def v113_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        elif hasattr(self, "output_label"):
            self.output_label.config(text=text)
    except Exception:
        pass


def v113_export(self, event=None):
    if hasattr(self, "export_wav_current_v110"):
        return self.export_wav_current_v110(event)

    print("[v140] ERREUR : export_wav_current_v110 introuvable.")
    v113_status(self, "Export WAV introuvable.")
    return "break"


def v113_all_widgets(widget):
    out = [widget]
    try:
        children = widget.winfo_children()
    except Exception:
        children = []

    for child in children:
        out.extend(v113_all_widgets(child))

    return out


def v113_widget_text(widget):
    try:
        return str(widget.cget("text")).strip()
    except Exception:
        return ""


def v113_find_button(self, label):
    target = label.strip().lower()

    for widget in v113_all_widgets(self.root):
        if v113_widget_text(widget).lower() == target:
            return widget

    return None


def v113_place_export_button(self):
    if getattr(self, "_v113_export_button_added", False):
        try:
            self.export_button_v113.lift()
        except Exception:
            pass
        return

    btn = tk.Button(
        self.root,
        text="Export WAV",
        command=self.export_wav_v113,
        bg="#2d7a4b",
        fg="#ffffff",
        activebackground="#39a96b",
        activeforeground="#ffffff",
        font=("Sans", 9, "bold"),
        padx=8,
        pady=3,
    )

    try:
        self.root.update_idletasks()

        generate = v113_find_button(self, "Generate Candidate")

        if generate is not None:
            root_x = int(self.root.winfo_rootx())
            root_y = int(self.root.winfo_rooty())

            gx = int(generate.winfo_rootx()) - root_x
            gy = int(generate.winfo_rooty()) - root_y
            gh = int(generate.winfo_height())

            btn.update_idletasks()
            bw = int(btn.winfo_reqwidth())

            # Même ligne que Generate Candidate, à gauche.
            # On garde une position visible même si la barre est serrée.
            x = max(690, gx - bw - 10)
            y = max(4, gy)

            btn.place(x=x, y=y, height=max(gh, 30))
        else:
            # Fallback visible en haut si Generate Candidate n'est pas encore trouvé.
            btn.place(x=760, y=32, height=30)

        btn.lift()
        self.export_button_v113 = btn
        self._v113_export_button_added = True

        print("[v140] bouton Export WAV visible ajouté.")
        v113_status(self, "v113 : Export WAV visible. Espace reste Loop / Space.")

    except Exception as exc:
        print("[v140] placement bouton Export WAV impossible :", exc)
        try:
            btn.place(x=760, y=32, height=30)
            btn.lift()
            self.export_button_v113 = btn
            self._v113_export_button_added = True
        except Exception:
            pass


_old_v113_build_ui = SliceIndexTracker.build_ui


def v113_build_ui(self):
    _old_v113_build_ui(self)

    try:
        self.root.title("BreakbeatAI v113 clean — export fixed visible")
    except Exception:
        pass

    try:
        self.root.bind("<Control-e>", self.export_wav_v113)
    except Exception:
        pass

    # Plusieurs tentatives non destructives, parce que la barre du haut finit
    # parfois de se construire après build_ui.
    try:
        self.root.after(200, lambda: v113_place_export_button(self))
        self.root.after(700, lambda: v113_place_export_button(self))
        self.root.after(1200, lambda: v113_place_export_button(self))
    except Exception:
        v113_place_export_button(self)


SliceIndexTracker.export_wav_v113 = v113_export
SliceIndexTracker.build_ui = v113_build_ui



# ---------------------------------------------------------------------
# v115 : cases / colonnes à -3.5px
# ---------------------------------------------------------------------
# Base : v113.
# Ne touche pas à Export WAV.
# Ne touche pas à Espace.
# Applique seulement -3.5px aux largeurs de grille détectées, une seule fois.
# ---------------------------------------------------------------------

V115_CELL_MINUS = 3.5

V115_WIDTH_GLOBAL_NAMES = [
    "STEP_W",
    "CELL_W",
    "COL_W",
    "COLUMN_W",
    "GRID_W",
    "GRID_STEP_W",
    "STEP_WIDTH",
    "CELL_WIDTH",
    "COL_WIDTH",
    "NOTE_WIDTH",
]

V115_WIDTH_ATTR_NAMES = [
    "step_w",
    "cell_w",
    "col_w",
    "column_w",
    "grid_step_w",
    "step_width",
    "cell_width",
    "col_width",
    "note_width",
    "beat_w",
    "x_step_w",
    "x_cell_w",
    "tracker_step_w",
]


def v115_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        elif hasattr(self, "output_label"):
            self.output_label.config(text=text)
    except Exception:
        pass


def v115_apply_cells_minus_25(self):
    if getattr(self, "_v116_export_case_length_fix_done", False):
        return

    self._v116_export_case_length_fix_done = True

    print("")
    print("[v140] APPLICATION CASES -3.5px")

    for name in V115_WIDTH_GLOBAL_NAMES:
        try:
            value = globals().get(name, None)
            if isinstance(value, (int, float)) and value > 6:
                globals()[name] = float(value) - V115_CELL_MINUS
                print(f"[v140] global {name}: {value} -> {globals()[name]}")
        except Exception:
            pass

    for attr in V115_WIDTH_ATTR_NAMES:
        try:
            if hasattr(self, attr):
                value = getattr(self, attr)
                if isinstance(value, (int, float)) and value > 6:
                    setattr(self, attr, float(value) - V115_CELL_MINUS)
                    print(f"[v140] attr {attr}: {value} -> {getattr(self, attr)}")
        except Exception:
            pass

    # garde le scroll large pour le 64 steps
    try:
        if hasattr(self, "canvas"):
            self.canvas.configure(scrollregion=(0, 0, 4096, 2400))
    except Exception:
        pass

    print("")


_old_v115_draw = getattr(SliceIndexTracker, "draw", None)


def v115_draw(self, *args, **kwargs):
    v115_apply_cells_minus_25(self)

    if callable(_old_v115_draw):
        result = _old_v115_draw(self, *args, **kwargs)
    else:
        result = None

    try:
        if hasattr(self, "canvas"):
            self.canvas.configure(scrollregion=(0, 0, 4096, 2400))
    except Exception:
        pass

    return result


_old_v115_build_ui = SliceIndexTracker.build_ui


def v115_build_ui(self):
    _old_v115_build_ui(self)

    try:
        self.root.title("BreakbeatAI v140 — Princess clean toolbar")
    except Exception:
        pass

    try:
        v115_apply_cells_minus_25(self)
    except Exception as exc:
        print("[v140] impossible d'appliquer -3.5px :", exc)

    v115_status(self, "v115 : cases à -3.5px. Export WAV et Espace conservés.")


if callable(_old_v115_draw):
    SliceIndexTracker.draw = v115_draw

SliceIndexTracker.build_ui = v115_build_ui



# ---------------------------------------------------------------------
# v117 EXPORT FIX : longueur exacte des cases
# ---------------------------------------------------------------------
# Le problème :
# l'export posait des slices de durées réelles différentes dans des cases fixes.
# Donc certains hits étaient trop courts -> trous -> rendu saccadé.
#
# v117 :
# - calcule case_samples depuis les pair_blocks eux-mêmes
# - 1 pair_block = 2 cases
# - chaque note est resamplée exactement à length * case_samples
# - export final = exactement 64 cases
# ---------------------------------------------------------------------


def v117_get_block_sample_bounds(block, sr, audio_len):
    start = v110_num(block, ["source_start_sample", "start_sample", "start", "start_frame"], None)
    end = v110_num(block, ["source_end_sample", "end_sample", "end", "end_frame"], None)

    if start is None:
        start_ms = v110_num(block, ["source_start_ms", "start_ms"], None)
        if start_ms is not None:
            start = int(float(start_ms) * sr / 1000.0)

    if end is None:
        end_ms = v110_num(block, ["source_end_ms", "end_ms"], None)
        if end_ms is not None:
            end = int(float(end_ms) * sr / 1000.0)

    if start is None:
        start = 0

    if end is None:
        dur_ms = v110_num(block, ["duration_ms", "dur_ms"], None)
        if dur_ms is not None:
            end = int(start) + int(float(dur_ms) * sr / 1000.0)

    if end is None:
        end = int(start) + int(2 * 60.0 / 155.0 / 4.0 * sr)

    start = int(max(0, min(audio_len, int(start))))
    end = int(max(start + 1, min(audio_len, int(end))))

    return start, end


def v117_resample_exact(seg, target_len):
    seg = np.asarray(seg, dtype=np.float32)
    target_len = int(max(1, target_len))

    if len(seg) == target_len:
        return seg.copy()

    if len(seg) <= 1:
        return np.zeros(target_len, dtype=np.float32)

    old_x = np.linspace(0.0, 1.0, len(seg), endpoint=False)
    new_x = np.linspace(0.0, 1.0, target_len, endpoint=False)
    out = np.interp(new_x, old_x, seg).astype(np.float32)

    return out


def v117_build_slice_map_exact(current):
    pairblock_file = v110_find_pairblock_file(current)

    if pairblock_file is None:
        raise RuntimeError(f"pair_blocks introuvable pour {current}")

    root = v110_load_json(pairblock_file, None)
    blocks = v110_extract_blocks(root)

    if not blocks:
        raise RuntimeError(f"aucun block dans {pairblock_file}")

    slice_map = {}
    source_audio_path = None
    durations = []
    sr_ref = None

    for i, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue

        pair = v110_pair_id(block, i)
        audio_path = v110_resolve_audio_path(block, root)

        if audio_path is None:
            continue

        source_audio_path = audio_path
        y, sr = v110_read_audio(audio_path)

        if sr_ref is None:
            sr_ref = int(sr)

        start, end = v117_get_block_sample_bounds(block, sr, len(y))
        seg = y[start:end].astype(np.float32)

        if len(seg) <= 1:
            continue

        durations.append(int(end - start))

        slice_map[pair] = {
            "audio": seg,
            "sr": int(sr),
            "source_audio_path": str(audio_path),
            "pairblock_file": str(pairblock_file),
            "source_start_sample": int(start),
            "source_end_sample": int(end),
            "source_duration_samples": int(end - start),
        }

    if not slice_map:
        raise RuntimeError(f"aucune slice audio exportable dans {pairblock_file}")

    if not durations:
        raise RuntimeError(f"durées pair_blocks introuvables dans {pairblock_file}")

    # Le point important :
    # un pair_block représente une cellule de tracker longue de 2 cases.
    # Donc la durée d'une case = médiane(pair_block_duration) / 2.
    median_pair_samples = int(np.median(np.asarray(durations, dtype=np.float64)))
    case_samples = int(round(median_pair_samples / 2.0))
    case_samples = max(1, case_samples)

    return slice_map, source_audio_path, pairblock_file, int(sr_ref or 44100), int(case_samples), int(median_pair_samples)


def v117_render_pattern_to_audio(self):
    current = v110_current_break(self)
    pattern = list(getattr(self, "pattern", []) or [])

    if not pattern:
        raise RuntimeError("pattern vide : clique Generate Candidate avant Export WAV")

    slice_map, source_audio_path, pairblock_file, sr, case_samples, median_pair_samples = v117_build_slice_map_exact(current)

    total_cases = 64
    total_samples = int(total_cases * case_samples)
    out = np.zeros(total_samples, dtype=np.float32)

    rendered = 0
    skipped = 0

    print("")
    print("[v140] EXPORT EXACT CELLS")
    print("[v140] break:", current)
    print("[v140] pairblock:", pairblock_file)
    print("[v140] sr:", sr)
    print("[v140] median_pair_samples:", median_pair_samples)
    print("[v140] case_samples:", case_samples)
    print("[v140] export_samples:", total_samples)
    print("[v140] export_seconds:", round(total_samples / float(sr), 6))
    print("")

    for item in pattern:
        try:
            step = int(item.get("x_step", 0))
            pair = int(item.get("pair", 0))
            length = int(item.get("length", 2) or 2)
        except Exception:
            skipped += 1
            continue

        if pair not in slice_map:
            skipped += 1
            print(f"[v140] skip pair introuvable : {pair}")
            continue

        raw_seg = slice_map[pair]["audio"].astype(np.float32)
        seg_sr = int(slice_map[pair]["sr"])

        # Resample sample-rate si nécessaire.
        if seg_sr != sr:
            ratio = sr / float(seg_sr)
            new_len = max(1, int(round(len(raw_seg) * ratio)))
            raw_seg = v117_resample_exact(raw_seg, new_len)

        target_len = int(max(1, length) * case_samples)

        # FIX PRINCIPAL :
        # On force chaque note à remplir exactement ses cases.
        seg = v117_resample_exact(raw_seg, target_len)

        # Mini fade uniquement à la fin pour éviter les clicks,
        # pas au début sinon ça bouffe les attaques kick/snare.
        fade = min(int(sr * 0.0015), max(1, len(seg) // 16))
        if fade > 1:
            seg[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

        pos = int(step * case_samples)
        end = min(len(out), pos + len(seg))

        if end <= pos:
            skipped += 1
            continue

        out[pos:end] += seg[:end - pos]
        rendered += 1

    peak = float(np.max(np.abs(out)) + 1e-12)
    if peak > 0.98:
        out *= 0.98 / peak

    info = {
        "break": current,
        "pairblock_file": str(pairblock_file),
        "source_audio_path": str(source_audio_path),
        "sample_rate": int(sr),
        "total_cases": int(total_cases),
        "case_samples": int(case_samples),
        "case_seconds": float(case_samples / float(sr)),
        "median_pair_samples": int(median_pair_samples),
        "rendered_notes": int(rendered),
        "skipped_notes": int(skipped),
        "duration_samples": int(len(out)),
        "duration_seconds": float(len(out) / float(sr)),
        "timing_source": "pairblock_median_duration_div_2_v117",
        "exporter": "v117_export_exact_cells",
    }

    return out.astype(np.float32), int(sr), info


def v117_export_wav_current(self, event=None):
    try:
        current = v110_current_break(self)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        variant = (
            getattr(self, "v110_current_first32_id", None)
            or getattr(self, "v115_current_first32_id", None)
            or "pattern"
        )

        audio, sr, info = v117_render_pattern_to_audio(self)

        out_name = f"{v110_slug(current)}_{v110_slug(variant)}_v117_exact_cells_{stamp}.wav"
        out_path = V110_EXPORT_DIR / out_name

        v110_write_wav(out_path, audio, sr)

        info_path = out_path.with_suffix(".json")
        info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

        print("")
        print("[v140] EXPORT WAV OK")
        print("[v140] wav  :", out_path)
        print("[v140] json :", info_path)
        print("[v140] duration:", round(info["duration_seconds"], 6), "s")
        print("[v140] case_seconds:", round(info["case_seconds"], 6))
        print("[v140] rendered:", info["rendered_notes"], "skipped:", info["skipped_notes"])
        print("")

        v110_status(self, f"Export WAV exact cells OK : {out_path}")

    except Exception as exc:
        print("")
        print("[v140] EXPORT WAV ERREUR")
        print("[v140]", exc)
        print(traceback.format_exc())
        print("")
        v110_status(self, f"Export WAV erreur : {exc}")

    return "break"


# On force tous les boutons/raccourcis Export WAV vers le moteur exact v117.
SliceIndexTracker.export_wav_current_v110 = v117_export_wav_current
SliceIndexTracker.export_wav_v113 = v117_export_wav_current
SliceIndexTracker.export_wav_current_v117 = v117_export_wav_current
SliceIndexTracker.render_pattern_to_audio_v117 = v117_render_pattern_to_audio



# ---------------------------------------------------------------------
# v118 EXPORT FIX : copie le WAV réel du playback
# ---------------------------------------------------------------------
# On arrête de reconstruire l'audio depuis les pair_blocks.
# Si Loop / Space sonne bien, Export WAV doit copier le même rendu WAV.
# ---------------------------------------------------------------------

V118_EXPORT_DIR = Path("exports")


def v118_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        elif hasattr(self, "output_label"):
            self.output_label.config(text=text)
    except Exception:
        pass


def v118_norm(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def v118_slug(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "unknown_break"


def v118_all_widgets(widget):
    out = [widget]
    try:
        children = widget.winfo_children()
    except Exception:
        children = []
    for child in children:
        out.extend(v118_all_widgets(child))
    return out


def v118_widget_text(widget):
    try:
        return str(widget.cget("text")).strip()
    except Exception:
        return ""


def v118_pairblock_stems():
    out = {}
    pair_dir = Path("dataset/pair_blocks_v02")
    for p in sorted(pair_dir.glob("*_pair_blocks_v02.json")):
        stem = p.name.replace("_pair_blocks_v02.json", "")
        out[v118_norm(stem)] = stem
    return out


def v118_current_break(self):
    stems = v118_pairblock_stems()
    candidates = []

    def add(value, source, score):
        try:
            value = str(value).strip()
        except Exception:
            return

        if not value:
            return

        for prefix in ["Break chargé :", "Training :", "Break:", "source:"]:
            if value.lower().startswith(prefix.lower()):
                value = value[len(prefix):].strip()

        n = v118_norm(value)

        if n in stems:
            candidates.append((score, source, stems[n], value))
            return

        for nn, stem in stems.items():
            if n and (n in nn or nn in n):
                candidates.append((score - 5, source, stem, value))
                return

    try:
        self.root.update_idletasks()
    except Exception:
        pass

    for widget in v118_all_widgets(self.root):
        cls = ""
        try:
            cls = str(widget.winfo_class())
        except Exception:
            pass

        try:
            value = widget.get()
        except Exception:
            value = None

        if value is not None:
            score = 40
            if "combo" in cls.lower():
                score = 200
            elif "entry" in cls.lower():
                score = 160
            add(value, f"widget.get:{cls}", score)

        try:
            text = widget.cget("text")
            if isinstance(text, str) and ("Training" in text or "Break chargé" in text):
                add(text, f"label:{cls}", 120)
        except Exception:
            pass

    for attr in [
        "current_source", "source", "selected_source", "active_source",
        "loaded_source", "current_safe", "safe", "break_name",
        "current_break", "selected_break", "active_break",
    ]:
        try:
            add(getattr(self, attr), f"attr:{attr}", 60)
        except Exception:
            pass

    if candidates:
        candidates.sort(reverse=True, key=lambda row: row[0])
        score, source, stem, raw = candidates[0]
        print("[v140] break courant:", stem, "| source:", source, "| raw:", raw)
        return stem

    print("[v140] WARNING : break courant introuvable, fallback Camo.")
    return "Camo_Break_-_3A"


def v118_find_loop_button(self):
    for widget in v118_all_widgets(self.root):
        if v118_widget_text(widget).lower() == "loop / space":
            return widget
    return None


def v118_score_wav(path, since):
    path = Path(path)
    name = path.name.lower()
    parts = [p.lower() for p in path.parts]

    try:
        mtime = path.stat().st_mtime
        size = path.stat().st_size
    except Exception:
        return None

    if size < 2048:
        return None

    score = 0

    # On évite les exports déjà faits et les samples source.
    if "exports" in parts:
        score -= 10000
    if "dataset" in parts:
        score -= 3000
    if "breaks" in parts:
        score -= 2500
    if "raw" in parts:
        score -= 1000

    # On cherche les fichiers que l'app génère pour écouter.
    for word in ["preview", "live", "audition", "tracker_app", "current", "render", "loop"]:
        if word in name:
            score += 2000

    # Très récent = très probable que ce soit le playback.
    if mtime >= since:
        score += 5000

    # Les WAV modifiés récemment passent devant.
    score += int(mtime % 100000)

    return score, mtime, size


def v118_find_best_playback_wav(since):
    roots = [
        Path("."),
        Path("pipeline"),
        Path("tmp"),
        Path("/tmp"),
    ]

    candidates = []

    for root in roots:
        if not root.exists():
            continue

        try:
            wavs = list(root.rglob("*.wav")) if root != Path("/tmp") else list(root.glob("*.wav"))
        except Exception:
            continue

        for wav in wavs:
            scored = v118_score_wav(wav, since)
            if scored is None:
                continue

            score, mtime, size = scored
            candidates.append((score, mtime, size, wav))

    candidates.sort(reverse=True, key=lambda row: row[0])

    print("")
    print("[v140] WAV candidats playback :")
    for score, mtime, size, wav in candidates[:12]:
        age = time.time() - mtime
        print(f"[v140] score={score:8d} age={age:7.2f}s size={size:10d} path={wav}")
    print("")

    if not candidates:
        return None

    return candidates[0][3]


def v118_copy_playback_wav(self, since):
    current = v118_current_break(self)
    best = v118_find_best_playback_wav(since)

    if best is None:
        v118_status(self, "Export WAV : aucun WAV playback trouvé. Lance Loop / Space puis réessaie.")
        print("[v140] aucun WAV playback trouvé.")
        return "break"

    V118_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = V118_EXPORT_DIR / f"{v118_slug(current)}_PLAYBACK_EXACT_v118_{stamp}.wav"

    shutil.copy2(best, out)

    info = {
        "exporter": "v118_copy_playback_wav",
        "break": current,
        "copied_from": str(best),
        "export_path": str(out),
        "note": "Copie du WAV généré par le moteur playback de l'app, pas reconstruction pair_blocks.",
    }

    info_path = out.with_suffix(".json")
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    print("")
    print("[v140] EXPORT PLAYBACK OK")
    print("[v140] source :", best)
    print("[v140] export :", out)
    print("[v140] json   :", info_path)
    print("")

    v118_status(self, f"Export playback OK : {out}")
    return "break"


def v118_export_playback_wav(self, event=None):
    """
    Export en 2 temps :
    1. On déclenche Loop / Space pour forcer l'app à générer son vrai WAV de playback.
    2. Après un court délai, on copie le WAV playback le plus récent.
    """
    since = time.time() - 0.25

    try:
        btn = v118_find_loop_button(self)
        if btn is not None:
            print("[v140] déclenche Loop / Space pour générer le playback WAV...")
            btn.invoke()
        else:
            print("[v140] bouton Loop / Space introuvable, copie du dernier WAV playback existant.")
    except Exception as exc:
        print("[v140] invoke Loop / Space impossible :", exc)

    try:
        self.root.after(900, lambda: v118_copy_playback_wav(self, since))
    except Exception:
        time.sleep(0.9)
        v118_copy_playback_wav(self, since)

    v118_status(self, "Export WAV : rendu playback en cours, copie dans 1 seconde...")
    return "break"


# Force tous les chemins d'export vers v118.
SliceIndexTracker.export_wav_current_v110 = v118_export_playback_wav
SliceIndexTracker.export_wav_v113 = v118_export_playback_wav
SliceIndexTracker.export_wav_current_v117 = v118_export_playback_wav
SliceIndexTracker.export_wav_current_v118 = v118_export_playback_wav



# ---------------------------------------------------------------------
# v121 EXPORT : reprend le WAV long propre, puis le coupe à 64 cases
# ---------------------------------------------------------------------
# Objectif :
# - utiliser le rendu qui sonnait bien mais faisait ~3 min
# - NE PAS reconstruire slice par slice
# - couper le fichier à la vraie durée de la grille 64 cases
# ---------------------------------------------------------------------

V121_EXPORT_DIR = Path("exports")
V121_PAIR_DIR = Path("dataset/pair_blocks_v02")


def v121_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        elif hasattr(self, "output_label"):
            self.output_label.config(text=text)
    except Exception:
        pass


def v121_norm(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def v121_slug(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "unknown_break"


def v121_all_widgets(widget):
    out = [widget]
    try:
        children = widget.winfo_children()
    except Exception:
        children = []
    for child in children:
        out.extend(v121_all_widgets(child))
    return out


def v121_widget_text(widget):
    try:
        return str(widget.cget("text")).strip()
    except Exception:
        return ""


def v121_pairblock_stems():
    out = {}
    for p in sorted(V121_PAIR_DIR.glob("*_pair_blocks_v02.json")):
        stem = p.name.replace("_pair_blocks_v02.json", "")
        out[v121_norm(stem)] = stem
    return out


def v121_current_break(self):
    # Essaie d'abord les fonctions existantes propres.
    for fn_name in ["v120_current_break", "v118_current_break", "v110_current_break"]:
        fn = globals().get(fn_name)
        if callable(fn):
            try:
                return fn(self)
            except Exception:
                pass

    stems = v121_pairblock_stems()
    candidates = []

    def add(value, source, score):
        try:
            value = str(value).strip()
        except Exception:
            return
        if not value:
            return

        for prefix in ["Break chargé :", "Training :", "Break:", "source:"]:
            if value.lower().startswith(prefix.lower()):
                value = value[len(prefix):].strip()

        n = v121_norm(value)

        if n in stems:
            candidates.append((score, source, stems[n], value))
            return

        for nn, stem in stems.items():
            if n and (n in nn or nn in n):
                candidates.append((score - 5, source, stem, value))
                return

    for widget in v121_all_widgets(self.root):
        cls = ""
        try:
            cls = str(widget.winfo_class())
        except Exception:
            pass

        try:
            value = widget.get()
            add(value, f"widget.get:{cls}", 200 if "combo" in cls.lower() else 80)
        except Exception:
            pass

        try:
            text = widget.cget("text")
            if isinstance(text, str) and ("Training" in text or "Break chargé" in text):
                add(text, f"label:{cls}", 120)
        except Exception:
            pass

    if candidates:
        candidates.sort(reverse=True, key=lambda x: x[0])
        return candidates[0][2]

    return "Camo_Break_-_3A"


def v121_find_loop_button(self):
    for widget in v121_all_widgets(self.root):
        if v121_widget_text(widget).lower() == "loop / space":
            return widget
    return None


def v121_load_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def v121_find_pairblock_file(current):
    exact = V121_PAIR_DIR / f"{current}_pair_blocks_v02.json"
    if exact.exists():
        return exact

    n_current = v121_norm(current)

    for p in sorted(V121_PAIR_DIR.glob("*_pair_blocks_v02.json")):
        stem = p.name.replace("_pair_blocks_v02.json", "")
        if v121_norm(stem) == n_current:
            return p

    for p in sorted(V121_PAIR_DIR.glob("*_pair_blocks_v02.json")):
        stem = p.name.replace("_pair_blocks_v02.json", "")
        n = v121_norm(stem)
        if n_current in n or n in n_current:
            return p

    return None


def v121_extract_blocks(data):
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        return []

    for key in ["pair_blocks", "blocks", "pairs", "slices", "items", "data"]:
        value = data.get(key)
        if isinstance(value, list):
            return value

    return []


def v121_num(obj, keys, default=None):
    if not isinstance(obj, dict):
        return default

    for key in keys:
        if key not in obj:
            continue
        try:
            return float(obj[key])
        except Exception:
            pass

    return default


def v121_get_sr(path):
    try:
        with wave.open(str(path), "rb") as wf:
            return int(wf.getframerate())
    except Exception:
        pass

    try:
        import soundfile as sf
        info = sf.info(str(path))
        return int(info.samplerate)
    except Exception:
        pass

    return 44100


def v121_wav_duration(path):
    try:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            frames = wf.getnframes()
            if sr > 0:
                return frames / float(sr)
    except Exception:
        pass

    try:
        import soundfile as sf
        info = sf.info(str(path))
        if info.samplerate > 0:
            return info.frames / float(info.samplerate)
    except Exception:
        pass

    try:
        raw = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ], stderr=subprocess.DEVNULL).decode().strip()
        return float(raw)
    except Exception:
        pass

    return None


def v121_find_existing_path(value):
    if not value:
        return None

    p = Path(str(value)).expanduser()
    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            Path(".") / p,
            Path("breaks") / p,
            Path("breaks") / p.name,
            Path("dataset") / p,
            Path("dataset/audio") / p.name,
        ])

    for c in candidates:
        if c.exists():
            return c

    return None


def v121_resolve_audio_path(block, root):
    for obj in [block, root]:
        if not isinstance(obj, dict):
            continue

        for key in [
            "source_audio", "audio_path", "source_path",
            "file", "path", "filename", "wav", "audio",
        ]:
            p = v121_find_existing_path(obj.get(key))
            if p:
                return p

    return None


def v121_case_seconds_from_pairblocks(current):
    """
    Calcule la durée d'une case depuis les pair_blocks.
    1 pair_block = note length 2 = 2 cases.
    Donc case = median(pair_duration) / 2.
    """
    pairblock_file = v121_find_pairblock_file(current)

    if pairblock_file is None:
        return None, {"reason": "pairblock_file introuvable"}

    root = v121_load_json(pairblock_file, None)
    blocks = v121_extract_blocks(root)

    if not blocks:
        return None, {"reason": "blocks introuvables", "pairblock_file": str(pairblock_file)}

    durations = []
    audio_path_found = None
    sr_found = 44100

    for i, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue

        audio_path = v121_resolve_audio_path(block, root)
        if audio_path is not None:
            audio_path_found = audio_path
            sr_found = v121_get_sr(audio_path)

        start = v121_num(block, ["source_start_sample", "start_sample"], None)
        end = v121_num(block, ["source_end_sample", "end_sample"], None)

        if start is not None and end is not None and end > start:
            durations.append((end - start) / float(sr_found))
            continue

        start_ms = v121_num(block, ["source_start_ms", "start_ms"], None)
        end_ms = v121_num(block, ["source_end_ms", "end_ms"], None)

        if start_ms is not None and end_ms is not None and end_ms > start_ms:
            durations.append((end_ms - start_ms) / 1000.0)
            continue

        start_s = v121_num(block, ["source_start_sec", "start_sec", "start_s"], None)
        end_s = v121_num(block, ["source_end_sec", "end_sec", "end_s"], None)

        if start_s is not None and end_s is not None and end_s > start_s:
            durations.append(end_s - start_s)
            continue

        dur_ms = v121_num(block, ["duration_ms", "dur_ms"], None)
        if dur_ms is not None and dur_ms > 0:
            durations.append(dur_ms / 1000.0)
            continue

        dur_s = v121_num(block, ["duration_sec", "dur_sec", "duration_s", "dur_s"], None)
        if dur_s is not None and dur_s > 0:
            durations.append(dur_s)
            continue

    if not durations:
        return None, {
            "reason": "durations introuvables",
            "pairblock_file": str(pairblock_file),
            "audio_path": str(audio_path_found),
        }

    import numpy as np
    median_pair_seconds = float(np.median(np.asarray(durations, dtype=np.float64)))
    case_seconds = median_pair_seconds / 2.0

    return case_seconds, {
        "pairblock_file": str(pairblock_file),
        "audio_path": str(audio_path_found),
        "sr": sr_found,
        "median_pair_seconds": median_pair_seconds,
        "case_seconds": case_seconds,
        "durations_min": float(min(durations)),
        "durations_max": float(max(durations)),
        "durations_count": len(durations),
    }


def v121_target_seconds(self, current):
    # 1) Si l'app expose step_ms, c'est prioritaire.
    for attr in ["step_ms", "step_ms_var"]:
        try:
            value = getattr(self, attr)
            if hasattr(value, "get"):
                value = value.get()
            value = float(value)
            if value > 1:
                case_seconds = value / 1000.0
                return 64 * case_seconds, {
                    "source": attr,
                    "case_seconds": case_seconds,
                    "target_seconds": 64 * case_seconds,
                }
        except Exception:
            pass

    # 2) Pairblocks.
    case_seconds, info = v121_case_seconds_from_pairblocks(current)
    if case_seconds is not None and 0.01 <= case_seconds <= 1.5:
        info = dict(info)
        info["source"] = "pairblock_median_div_2"
        info["target_seconds"] = 64 * case_seconds
        return 64 * case_seconds, info

    # 3) Fallback jungle 155 bpm : case = double croche.
    case_seconds = 60.0 / 155.0 / 4.0
    return 64 * case_seconds, {
        "source": "fallback_155bpm",
        "case_seconds": case_seconds,
        "target_seconds": 64 * case_seconds,
        "pairblock_info": info,
    }


def v121_score_long_wav(path, since):
    """
    Reprend l'esprit v118 :
    on accepte les WAV longs, parce que c'est celui qui sonnait propre.
    Mais on évite exports/dataset si possible.
    """
    path = Path(path)
    name = path.name.lower()
    parts = [p.lower() for p in path.parts]

    try:
        stat = path.stat()
        size = stat.st_size
        mtime = stat.st_mtime
    except Exception:
        return None

    if size < 2048:
        return None

    dur = v121_wav_duration(path)
    if dur is None or dur < 0.5:
        return None

    score = 0

    # Comme v118 : on cherche le rendu playback / preview / live.
    for word in ["preview", "live", "audition", "tracker_app", "current", "render", "loop", "playback"]:
        if word in name:
            score += 4000

    if mtime >= since:
        score += 8000

    age = time.time() - mtime
    if age < 10:
        score += 4000
    elif age < 60:
        score += 2000

    # On ne veut pas reprendre un ancien export.
    if "exports" in parts:
        score -= 20000

    # Mais contrairement à v120, on n'exclut pas les longs.
    # On pénalise dataset/breaks sans les interdire : v118 prenait peut-être ce fichier.
    if "dataset" in parts:
        score -= 2500
    if "breaks" in parts:
        score -= 2500
    if "raw" in parts:
        score -= 1500

    # Le long propre était souvent grand : on ne le rejette pas.
    if dur > 45:
        score += 500

    return {
        "score": score,
        "path": path,
        "duration": dur,
        "mtime": mtime,
        "size": size,
    }


def v121_iter_wavs():
    roots = [Path("."), Path("/tmp")]

    skip_dirs = {".git", ".venv", "venv", "__pycache__", "exports"}

    for root in roots:
        if not root.exists():
            continue

        if root == Path("/tmp"):
            for p in root.glob("*.wav"):
                yield p
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
            for filename in filenames:
                if filename.lower().endswith(".wav"):
                    yield Path(dirpath) / filename


def v121_find_best_long_wav(since):
    candidates = []

    for wav in v121_iter_wavs():
        scored = v121_score_long_wav(wav, since)
        if scored is not None:
            candidates.append(scored)

    candidates.sort(key=lambda row: row["score"], reverse=True)

    print("")
    print("[v140] WAV candidats longs/propre à raccourcir :")
    for row in candidates[:20]:
        age = time.time() - row["mtime"]
        print(
            f"[v140] score={row['score']:7d} "
            f"dur={row['duration']:9.3f}s "
            f"age={age:7.2f}s "
            f"size={row['size']:10d} "
            f"path={row['path']}"
        )
    print("")

    if not candidates:
        return None

    return candidates[0]


def v121_trim_wav_ffmpeg(src, dst, target_seconds):
    if shutil.which("ffmpeg") is None:
        return False

    cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",
        "-i", str(src),
        "-t", f"{float(target_seconds):.9f}",
        "-map", "0:a:0",
        "-acodec", "pcm_s16le",
        str(dst),
    ]

    subprocess.check_call(cmd)
    return True


def v121_trim_wav_python(src, dst, target_seconds):
    with wave.open(str(src), "rb") as r:
        params = r.getparams()
        sr = r.getframerate()
        target_frames = int(round(float(target_seconds) * sr))
        target_frames = max(1, min(target_frames, r.getnframes()))
        audio = r.readframes(target_frames)

    with wave.open(str(dst), "wb") as w:
        w.setparams(params)
        w.writeframes(audio)


def v121_export_trimmed_long(self, event=None):
    since = time.time() - 2.0

    # On déclenche le playback comme v118 pour générer le WAV propre.
    try:
        btn = v121_find_loop_button(self)
        if btn is not None:
            print("[v140] déclenche Loop / Space pour générer le WAV long/propre...")
            btn.invoke()
    except Exception as exc:
        print("[v140] Loop / Space impossible :", exc)

    def do_export():
        try:
            current = v121_current_break(self)
            target_seconds, timing_info = v121_target_seconds(self, current)
            best = v121_find_best_long_wav(since)

            if best is None:
                raise RuntimeError("aucun WAV long/propre trouvé à raccourcir")

            src_wav = Path(best["path"])
            src_dur = float(best["duration"])

            if target_seconds >= src_dur:
                # Sécurité : si le target dépasse la source, on ne rallonge pas.
                target_seconds = src_dur

            V121_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

            stamp = time.strftime("%Y%m%d_%H%M%S")
            out = V121_EXPORT_DIR / f"{v121_slug(current)}_TRIMMED_64cases_v121_{stamp}.wav"

            try:
                ok = v121_trim_wav_ffmpeg(src_wav, out, target_seconds)
            except Exception as exc:
                print("[v140] ffmpeg trim impossible, fallback wave :", exc)
                ok = False

            if not ok:
                v121_trim_wav_python(src_wav, out, target_seconds)

            out_dur = v121_wav_duration(out)

            info = {
                "exporter": "v121_trim_long_clean_wav",
                "break": current,
                "source_wav": str(src_wav),
                "source_duration_seconds": src_dur,
                "target_duration_seconds": target_seconds,
                "output_duration_seconds": out_dur,
                "timing_info": timing_info,
                "note": "Reprend le WAV long/propre type v118 puis le raccourcit à 64 cases.",
            }

            info_path = out.with_suffix(".json")
            info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

            print("")
            print("[v140] EXPORT TRIM OK")
            print("[v140] source :", src_wav)
            print("[v140] source durée :", round(src_dur, 6), "s")
            print("[v140] target durée :", round(target_seconds, 6), "s")
            print("[v140] export :", out)
            print("[v140] export durée :", round(float(out_dur or 0), 6), "s")
            print("[v140] json :", info_path)
            print("")

            v121_status(self, f"Export v121 OK : {out}")

        except Exception as exc:
            print("")
            print("[v140] EXPORT ERREUR")
            print("[v140]", exc)
            try:
                print(traceback.format_exc())
            except Exception:
                pass
            print("")
            v121_status(self, f"Export v121 erreur : {exc}")

    try:
        self.root.after(900, do_export)
    except Exception:
        time.sleep(0.9)
        do_export()

    v121_status(self, "Export v121 : génération du WAV long puis trim 64 cases...")
    return "break"


# Tous les boutons export pointent vers le trim v121.
SliceIndexTracker.export_wav_current_v110 = v121_export_trimmed_long
SliceIndexTracker.export_wav_v113 = v121_export_trimmed_long
SliceIndexTracker.export_wav_current_v117 = v121_export_trimmed_long
SliceIndexTracker.export_wav_current_v118 = v121_export_trimmed_long
SliceIndexTracker.export_wav_current_v119 = v121_export_trimmed_long
SliceIndexTracker.export_wav_current_v120 = v121_export_trimmed_long
SliceIndexTracker.export_wav_current_v121 = v121_export_trimmed_long



# ---------------------------------------------------------------------
# v124 : export WAV seulement, plus aucun JSON dans exports/
# ---------------------------------------------------------------------
# Base : v121 qui sonnait bien.
# Changement unique :
# - Export WAV ne crée plus le fichier .json à côté.
# ---------------------------------------------------------------------


def v124_export_trimmed_long_no_json(self, event=None):
    since = time.time() - 2.0

    try:
        btn = v121_find_loop_button(self)
        if btn is not None:
            print("[v140] déclenche Loop / Space pour générer le WAV long/propre...")
            btn.invoke()
    except Exception as exc:
        print("[v140] Loop / Space impossible :", exc)

    def do_export():
        try:
            current = v121_current_break(self)
            target_seconds, timing_info = v121_target_seconds(self, current)
            best = v121_find_best_long_wav(since)

            if best is None:
                raise RuntimeError("aucun WAV long/propre trouvé à raccourcir")

            src_wav = Path(best["path"])
            src_dur = float(best["duration"])

            if target_seconds >= src_dur:
                target_seconds = src_dur

            V121_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

            stamp = time.strftime("%Y%m%d_%H%M%S")
            out = V121_EXPORT_DIR / f"{v121_slug(current)}_TRIMMED_64cases_v124_{stamp}.wav"

            try:
                ok = v121_trim_wav_ffmpeg(src_wav, out, target_seconds)
            except Exception as exc:
                print("[v140] ffmpeg trim impossible, fallback wave :", exc)
                ok = False

            if not ok:
                v121_trim_wav_python(src_wav, out, target_seconds)

            out_dur = v121_wav_duration(out)

            print("")
            print("[v140] EXPORT WAV ONLY OK")
            print("[v140] source :", src_wav)
            print("[v140] source durée :", round(src_dur, 6), "s")
            print("[v140] target durée :", round(target_seconds, 6), "s")
            print("[v140] export :", out)
            print("[v140] export durée :", round(float(out_dur or 0), 6), "s")
            print("[v140] aucun JSON exporté")
            print("")

            v121_status(self, f"Export WAV OK : {out}")

        except Exception as exc:
            print("")
            print("[v140] EXPORT ERREUR")
            print("[v140]", exc)
            try:
                print(traceback.format_exc())
            except Exception:
                pass
            print("")
            v121_status(self, f"Export v124 erreur : {exc}")

    try:
        self.root.after(900, do_export)
    except Exception:
        time.sleep(0.9)
        do_export()

    v121_status(self, "Export v124 : WAV seulement, second 32 miroir strict...")
    return "break"


SliceIndexTracker.export_wav_current_v110 = v124_export_trimmed_long_no_json
SliceIndexTracker.export_wav_v113 = v124_export_trimmed_long_no_json
SliceIndexTracker.export_wav_current_v117 = v124_export_trimmed_long_no_json
SliceIndexTracker.export_wav_current_v118 = v124_export_trimmed_long_no_json
SliceIndexTracker.export_wav_current_v119 = v124_export_trimmed_long_no_json
SliceIndexTracker.export_wav_current_v120 = v124_export_trimmed_long_no_json
SliceIndexTracker.export_wav_current_v121 = v124_export_trimmed_long_no_json
SliceIndexTracker.export_wav_current_v124 = v124_export_trimmed_long_no_json



# ---------------------------------------------------------------------
# v125 : multisélection par ligne Slice
# ---------------------------------------------------------------------
# Clic sur la colonne de gauche "Slice 1 / Slice 2 / Slice 3..."
# => sélectionne toutes les notes de cette ligne.
#
# Flèches :
# - gauche/droite : bouge toute la ligne sélectionnée dans le temps
# - haut/bas      : change toute la ligne sélectionnée de slice
#
# Clic ailleurs :
# - annule la multisélection v125
# - laisse le comportement normal de l'app
# ---------------------------------------------------------------------


def v125_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        elif hasattr(self, "output_label"):
            self.output_label.config(text=text)
    except Exception:
        pass


def v125_get_canvas(self):
    try:
        return self.canvas
    except Exception:
        return None


def v125_parse_slice_text(text):
    m = re.search(r"\bslice\s*([0-9]+)\b", str(text), flags=re.I)
    if not m:
        return None
    return max(0, int(m.group(1)) - 1)


def v125_slice_labels(self):
    canvas = v125_get_canvas(self)
    if canvas is None:
        return []

    labels = []

    try:
        for item in canvas.find_all():
            try:
                if canvas.type(item) != "text":
                    continue
                text = canvas.itemcget(item, "text")
                lane = v125_parse_slice_text(text)
                if lane is None:
                    continue
                bbox = canvas.bbox(item)
                if not bbox:
                    continue
                x1, y1, x2, y2 = bbox
                labels.append({
                    "item": item,
                    "text": text,
                    "lane": lane,
                    "bbox": bbox,
                    "cx": (x1 + x2) / 2.0,
                    "cy": (y1 + y2) / 2.0,
                })
            except Exception:
                pass
    except Exception:
        pass

    labels.sort(key=lambda row: row["lane"])
    return labels


def v125_lane_from_header_click(self, event):
    canvas = v125_get_canvas(self)
    if canvas is None:
        return None

    labels = v125_slice_labels(self)
    if not labels:
        return None

    try:
        cx = float(canvas.canvasx(event.x))
        cy = float(canvas.canvasy(event.y))
    except Exception:
        cx = float(getattr(event, "x", 0))
        cy = float(getattr(event, "y", 0))

    # Si on clique directement sur un texte Slice.
    try:
        hits = canvas.find_overlapping(cx - 6, cy - 6, cx + 6, cy + 6)
        for hit in hits:
            if canvas.type(hit) == "text":
                text = canvas.itemcget(hit, "text")
                lane = v125_parse_slice_text(text)
                if lane is not None:
                    return lane
    except Exception:
        pass

    # Sinon, clic dans la colonne de gauche, on prend la ligne Slice la plus proche.
    header_limit = max(row["bbox"][2] for row in labels) + 40

    if cx > header_limit:
        return None

    closest = min(labels, key=lambda row: abs(row["cy"] - cy))
    return int(closest["lane"])


def v125_select_lane(self, lane):
    lane = int(lane)
    pattern = list(getattr(self, "pattern", []) or [])

    ids = []

    for item in pattern:
        try:
            if int(item.get("lane", -999)) == lane:
                ids.append(int(item.get("id")))
        except Exception:
            pass

    ids = sorted(set(ids))

    self.selected_ids_v125 = set(ids)
    self.selected_lane_v125 = lane

    if ids:
        self.selected_id = ids[0]

    print("")
    print(f"[v140] Slice {lane + 1} sélectionnée")
    print(f"[v140] notes sélectionnées : {ids}")
    print("")

    v125_status(self, f"v125 : Slice {lane + 1} sélectionnée — {len(ids)} notes. Flèches = déplacer toute la ligne.")

    try:
        self.draw()
    except Exception:
        pass

    return "break"


def v125_clear_multiselect(self):
    if getattr(self, "selected_ids_v125", None):
        self.selected_ids_v125 = set()
        self.selected_lane_v125 = None


def v125_canvas_click(self, event=None):
    lane = v125_lane_from_header_click(self, event)

    if lane is not None:
        return v125_select_lane(self, lane)

    # Clic ailleurs : comportement normal de l'app.
    v125_clear_multiselect(self)
    return None


def v125_pair_for_lane(self, lane, fallback_pair=None):
    lane = int(lane)

    try:
        if hasattr(self, "lane_to_pair") and isinstance(self.lane_to_pair, dict):
            if lane in self.lane_to_pair:
                return int(self.lane_to_pair[lane])
    except Exception:
        pass

    try:
        if hasattr(self, "pair_to_lane") and isinstance(self.pair_to_lane, dict):
            for pair, ln in self.pair_to_lane.items():
                try:
                    if int(ln) == lane:
                        return int(pair)
                except Exception:
                    pass
    except Exception:
        pass

    try:
        vals = list(getattr(self, "pair_values"))
        if 0 <= lane < len(vals):
            return int(vals[lane])
    except Exception:
        pass

    try:
        pairs = sorted(int(item.get("pair")) for item in getattr(self, "pattern", []) if int(item.get("lane", -1)) == lane)
        if pairs:
            return pairs[0]
    except Exception:
        pass

    return int(fallback_pair or 0)


def v125_max_lane(self):
    max_lane = 0

    try:
        if hasattr(self, "pair_values"):
            max_lane = max(max_lane, len(list(self.pair_values)) - 1)
    except Exception:
        pass

    try:
        if hasattr(self, "pair_to_lane") and isinstance(self.pair_to_lane, dict):
            vals = [int(v) for v in self.pair_to_lane.values()]
            if vals:
                max_lane = max(max_lane, max(vals))
    except Exception:
        pass

    try:
        lanes = [int(item.get("lane", 0)) for item in getattr(self, "pattern", [])]
        if lanes:
            max_lane = max(max_lane, max(lanes))
    except Exception:
        pass

    return max(0, int(max_lane))


def v125_selected_items(self):
    ids = set(getattr(self, "selected_ids_v125", set()) or set())
    if not ids:
        return []

    out = []

    for item in getattr(self, "pattern", []) or []:
        try:
            if int(item.get("id")) in ids:
                out.append(item)
        except Exception:
            pass

    return out


def v125_redraw_after_move(self):
    try:
        self.draw()
    except Exception as exc:
        print("[v140] draw impossible :", exc)

    try:
        self.refresh_panel()
    except Exception:
        pass

    try:
        self.write_latest_pattern(reason="v125_slice_row_multiselect")
    except Exception:
        pass


def v125_move_multiselection(self, dx_steps=0, dy_lanes=0):
    selected = v125_selected_items(self)

    if not selected:
        return None

    if dx_steps:
        new_steps = []

        for item in selected:
            try:
                length = int(item.get("length", 2) or 2)
                x = int(item.get("x_step", 0))
                nx = x + int(dx_steps)
                nx = max(0, min(64 - length, nx))

                # garde les notes sur les cases paires, comme le tracker strict.
                if nx % 2 != 0:
                    nx -= 1

                new_steps.append(nx)
            except Exception:
                pass

        if len(new_steps) != len(selected):
            return "break"

        for item, nx in zip(selected, new_steps):
            item["x_step"] = int(nx)

    if dy_lanes:
        max_lane = v125_max_lane(self)

        for item in selected:
            try:
                old_lane = int(item.get("lane", 0))
                new_lane = max(0, min(max_lane, old_lane + int(dy_lanes)))
                old_pair = int(item.get("pair", 0))

                item["lane"] = int(new_lane)
                item["pair"] = int(v125_pair_for_lane(self, new_lane, old_pair))
            except Exception:
                pass

        try:
            self.selected_lane_v125 = int(selected[0].get("lane", 0))
        except Exception:
            pass

    self.selected_ids_v125 = set(int(item.get("id")) for item in selected)

    v125_redraw_after_move(self)

    lane_txt = ""
    try:
        lane_txt = f" — Slice {int(self.selected_lane_v125) + 1}"
    except Exception:
        pass

    v125_status(self, f"v125 : multisélection déplacée{lane_txt}.")
    return "break"


def v125_key_left(self, event=None):
    if v125_selected_items(self):
        return v125_move_multiselection(self, dx_steps=-2)
    return None


def v125_key_right(self, event=None):
    if v125_selected_items(self):
        return v125_move_multiselection(self, dx_steps=2)
    return None


def v125_key_up(self, event=None):
    if v125_selected_items(self):
        return v125_move_multiselection(self, dy_lanes=-1)
    return None


def v125_key_down(self, event=None):
    if v125_selected_items(self):
        return v125_move_multiselection(self, dy_lanes=1)
    return None


def v125_grid_geometry(self):
    canvas = v125_get_canvas(self)
    labels = v125_slice_labels(self)

    if canvas is None or not labels:
        return None

    try:
        header_limit = max(row["bbox"][2] for row in labels) + 20
    except Exception:
        header_limit = 90

    xs = []

    try:
        for item in canvas.find_all():
            if canvas.type(item) != "line":
                continue
            coords = canvas.coords(item)
            if len(coords) >= 4:
                x1, y1, x2, y2 = coords[:4]
                if abs(x1 - x2) < 0.01 and x1 > header_limit:
                    xs.append(float(x1))
    except Exception:
        pass

    xs = sorted(set(round(x, 3) for x in xs))

    if len(xs) >= 4:
        diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1) if xs[i + 1] - xs[i] > 1]
        if diffs:
            cell_w = sorted(diffs)[len(diffs) // 2]
            grid_x0 = xs[0]
        else:
            return None
    else:
        return None

    ys = sorted(row["cy"] for row in labels)
    if len(ys) >= 2:
        diffs_y = [ys[i + 1] - ys[i] for i in range(len(ys) - 1) if ys[i + 1] - ys[i] > 1]
        row_h = sorted(diffs_y)[len(diffs_y) // 2] if diffs_y else 22
    else:
        row_h = 22

    lane_y = {int(row["lane"]): float(row["cy"]) for row in labels}

    return {
        "grid_x0": float(grid_x0),
        "cell_w": float(cell_w),
        "row_h": float(row_h),
        "lane_y": lane_y,
    }


_old_v125_draw = getattr(SliceIndexTracker, "draw", None)


def v125_draw(self, *args, **kwargs):
    result = None

    if callable(_old_v125_draw):
        result = _old_v125_draw(self, *args, **kwargs)

    selected = v125_selected_items(self)

    if not selected:
        return result

    canvas = v125_get_canvas(self)
    geom = v125_grid_geometry(self)

    if canvas is None or geom is None:
        return result

    # Overlays roses de sélection multiple.
    try:
        canvas.delete("v125_multi_select_overlay")
    except Exception:
        pass

    for item in selected:
        try:
            step = int(item.get("x_step", 0))
            length = int(item.get("length", 2) or 2)
            lane = int(item.get("lane", 0))

            x1 = geom["grid_x0"] + step * geom["cell_w"]
            x2 = x1 + length * geom["cell_w"]
            cy = geom["lane_y"].get(lane)

            if cy is None:
                continue

            y1 = cy - geom["row_h"] * 0.42
            y2 = cy + geom["row_h"] * 0.42

            canvas.create_rectangle(
                x1 + 1,
                y1 + 1,
                x2 - 1,
                y2 - 1,
                outline="#ff5fd7",
                width=3,
                tags=("v125_multi_select_overlay",),
            )
        except Exception:
            pass

    try:
        canvas.tag_raise("v125_multi_select_overlay")
    except Exception:
        pass

    return result


_old_v125_build_ui = SliceIndexTracker.build_ui


def v125_build_ui(self):
    _old_v125_build_ui(self)

    self.selected_ids_v125 = set()
    self.selected_lane_v125 = None

    try:
        self.root.title("BreakbeatAI v140 — Princess clean toolbar")
    except Exception:
        pass

    canvas = v125_get_canvas(self)

    if canvas is not None:
        try:
            tags = list(canvas.bindtags())
            if "V125CanvasClick" not in tags:
                canvas.bindtags(("V125CanvasClick",) + tuple(tags))
            self.root.bind_class("V125CanvasClick", "<Button-1>", self.v125_canvas_click)
        except Exception as exc:
            print("[v140] bind canvas click impossible :", exc)

    # Bindtags prioritaires pour intercepter les flèches seulement si multisélection active.
    try:
        for widget in v121_all_widgets(self.root) if "v121_all_widgets" in globals() else [self.root, canvas]:
            if widget is None:
                continue
            tags = list(widget.bindtags())
            if "V125Keys" not in tags:
                widget.bindtags(("V125Keys",) + tuple(tags))
    except Exception:
        pass

    try:
        self.root.bind_class("V125Keys", "<Left>", self.v125_key_left)
        self.root.bind_class("V125Keys", "<Right>", self.v125_key_right)
        self.root.bind_class("V125Keys", "<Up>", self.v125_key_up)
        self.root.bind_class("V125Keys", "<Down>", self.v125_key_down)
    except Exception as exc:
        print("[v140] bind flèches impossible :", exc)

    v125_status(self, "v125 : clic sur Slice 1/Slice 2... = sélection de toute la ligne.")


SliceIndexTracker.v125_canvas_click = v125_canvas_click
SliceIndexTracker.v125_key_left = v125_key_left
SliceIndexTracker.v125_key_right = v125_key_right
SliceIndexTracker.v125_key_up = v125_key_up
SliceIndexTracker.v125_key_down = v125_key_down

if callable(_old_v125_draw):
    SliceIndexTracker.draw = v125_draw

SliceIndexTracker.build_ui = v125_build_ui



# ---------------------------------------------------------------------
# v126 : vraie multisélection par Slice / Pair
# ---------------------------------------------------------------------
# Correction v125 :
# v125 sélectionnait par lane, mais les notes du tracker sont surtout reliées
# au champ pair/slice audio.
#
# v126 :
# - clic sur "Slice N" récupère N
# - trouve le pair correspondant à cette ligne
# - sélectionne TOUTES les notes du pattern qui utilisent ce pair/slice
# - fallback lane si besoin
# ---------------------------------------------------------------------


def v126_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        elif hasattr(self, "output_label"):
            self.output_label.config(text=text)
    except Exception:
        pass


def v126_canvas(self):
    return getattr(self, "canvas", None)


def v126_all_canvas_texts(self):
    canvas = v126_canvas(self)
    out = []

    if canvas is None:
        return out

    try:
        for item in canvas.find_all():
            try:
                if canvas.type(item) != "text":
                    continue

                text = canvas.itemcget(item, "text")
                m = re.search(r"\bslice\s*([0-9]+)\b", str(text), flags=re.I)

                if not m:
                    continue

                num = int(m.group(1))
                bbox = canvas.bbox(item)

                if not bbox:
                    continue

                x1, y1, x2, y2 = bbox

                out.append({
                    "canvas_item": item,
                    "text": text,
                    "slice_number": num,
                    "zero_index": num - 1,
                    "bbox": bbox,
                    "cx": (x1 + x2) / 2,
                    "cy": (y1 + y2) / 2,
                })
            except Exception:
                pass
    except Exception:
        pass

    out.sort(key=lambda row: row["zero_index"])
    return out


def v126_clicked_slice_row(self, event):
    canvas = v126_canvas(self)

    if canvas is None:
        return None

    rows = v126_all_canvas_texts(self)

    if not rows:
        return None

    try:
        cx = float(canvas.canvasx(event.x))
        cy = float(canvas.canvasy(event.y))
    except Exception:
        cx = float(getattr(event, "x", 0))
        cy = float(getattr(event, "y", 0))

    # 1) clic direct sur le texte Slice N.
    try:
        hits = canvas.find_overlapping(cx - 8, cy - 8, cx + 8, cy + 8)
        for hit in hits:
            if canvas.type(hit) != "text":
                continue

            text = canvas.itemcget(hit, "text")
            m = re.search(r"\bslice\s*([0-9]+)\b", str(text), flags=re.I)

            if m:
                num = int(m.group(1))
                return {
                    "slice_number": num,
                    "zero_index": num - 1,
                    "text": text,
                    "reason": "direct_text_hit",
                }
    except Exception:
        pass

    # 2) clic dans la colonne gauche, on prend la ligne Slice la plus proche.
    try:
        left_limit = max(row["bbox"][2] for row in rows) + 50
    except Exception:
        left_limit = 160

    if cx > left_limit:
        return None

    closest = min(rows, key=lambda row: abs(float(row["cy"]) - cy))

    return {
        "slice_number": int(closest["slice_number"]),
        "zero_index": int(closest["zero_index"]),
        "text": closest["text"],
        "reason": "closest_left_header",
    }


def v126_pair_candidates_for_slice(self, zero_index, slice_number):
    """
    Retourne plusieurs candidats possibles car selon les versions :
    - pair peut être 0-indexé
    - pair peut être 1-indexé
    - lane_to_pair peut exister
    - pair_to_lane peut exister
    """
    candidates = set()

    zero_index = int(zero_index)
    slice_number = int(slice_number)

    candidates.add(zero_index)
    candidates.add(slice_number)

    # pair_values : souvent liste des vrais pairs dans l'ordre des lignes.
    try:
        vals = list(getattr(self, "pair_values"))
        if 0 <= zero_index < len(vals):
            candidates.add(int(vals[zero_index]))
        if 0 <= slice_number < len(vals):
            candidates.add(int(vals[slice_number]))
    except Exception:
        pass

    # lane_to_pair.
    try:
        lane_to_pair = getattr(self, "lane_to_pair", None)
        if isinstance(lane_to_pair, dict):
            for key in [zero_index, slice_number]:
                if key in lane_to_pair:
                    candidates.add(int(lane_to_pair[key]))
                if str(key) in lane_to_pair:
                    candidates.add(int(lane_to_pair[str(key)]))
    except Exception:
        pass

    # pair_to_lane inverse : tous les pairs qui pointent vers cette ligne.
    try:
        pair_to_lane = getattr(self, "pair_to_lane", None)
        if isinstance(pair_to_lane, dict):
            for pair, lane in pair_to_lane.items():
                try:
                    if int(lane) in [zero_index, slice_number]:
                        candidates.add(int(pair))
                except Exception:
                    pass
    except Exception:
        pass

    return sorted(candidates)


def v126_note_matches_slice(self, note, zero_index, slice_number, pair_candidates):
    pair_candidates = set(int(x) for x in pair_candidates)

    try:
        note_pair = int(note.get("pair", -999999))
    except Exception:
        note_pair = -999999

    try:
        note_lane = int(note.get("lane", -999999))
    except Exception:
        note_lane = -999999

    # Match principal : pair/slice audio.
    if note_pair in pair_candidates:
        return True

    # Fallback : lane visuelle.
    if note_lane in [int(zero_index), int(slice_number)]:
        return True

    # Fallback via pair_to_lane.
    try:
        pair_to_lane = getattr(self, "pair_to_lane", None)
        if isinstance(pair_to_lane, dict):
            lane = pair_to_lane.get(note_pair, pair_to_lane.get(str(note_pair), None))
            if lane is not None and int(lane) in [int(zero_index), int(slice_number)]:
                return True
    except Exception:
        pass

    return False


def v126_select_slice(self, zero_index, slice_number, reason=""):
    pattern = list(getattr(self, "pattern", []) or [])

    pair_candidates = v126_pair_candidates_for_slice(self, zero_index, slice_number)

    ids = []
    debug_rows = []

    for note in pattern:
        if v126_note_matches_slice(self, note, zero_index, slice_number, pair_candidates):
            try:
                nid = int(note.get("id"))
                ids.append(nid)
                debug_rows.append({
                    "id": nid,
                    "step": note.get("x_step"),
                    "lane": note.get("lane"),
                    "pair": note.get("pair"),
                    "role": note.get("learned_role", note.get("prior_label", "")),
                })
            except Exception:
                pass

    ids = sorted(set(ids))

    self.selected_ids_v125 = set(ids)
    self.selected_ids_v126 = set(ids)
    self.selected_lane_v125 = int(zero_index)
    self.selected_slice_number_v126 = int(slice_number)
    self.selected_pair_candidates_v126 = set(pair_candidates)

    if ids:
        self.selected_id = ids[0]

    print("")
    print("[v140] SÉLECTION SLICE")
    print("[v140] reason:", reason)
    print("[v140] Slice affichée:", slice_number)
    print("[v140] zero_index:", zero_index)
    print("[v140] pair_candidates:", pair_candidates)
    print("[v140] ids sélectionnés:", ids)
    for row in debug_rows:
        print("[v140] note:", row)
    print("")

    try:
        self.draw()
    except Exception as exc:
        print("[v140] draw impossible :", exc)

    if ids:
        v126_status(self, f"v126 : Slice {slice_number} sélectionnée — {len(ids)} notes.")
    else:
        v126_status(self, f"v126 : Slice {slice_number} — aucune note trouvée. Voir terminal pair_candidates.")

    return "break"


def v126_canvas_click(self, event=None):
    row = v126_clicked_slice_row(self, event)

    if row is not None:
        return v126_select_slice(
            self,
            zero_index=row["zero_index"],
            slice_number=row["slice_number"],
            reason=row["reason"],
        )

    # Clic ailleurs : on nettoie seulement la sélection v126/v125.
    try:
        self.selected_ids_v126 = set()
        self.selected_ids_v125 = set()
        self.selected_pair_candidates_v126 = set()
        self.selected_slice_number_v126 = None
    except Exception:
        pass

    return None


# On remplace le handler canvas de v125 par celui de v126.
SliceIndexTracker.v125_canvas_click = v126_canvas_click
SliceIndexTracker.v126_canvas_click = v126_canvas_click
SliceIndexTracker.v126_select_slice = v126_select_slice

_old_v126_build_ui = SliceIndexTracker.build_ui


def v126_build_ui(self):
    _old_v126_build_ui(self)

    try:
        self.root.title("BreakbeatAI v140 — Princess clean toolbar")
    except Exception:
        pass

    canvas = v126_canvas(self)

    if canvas is not None:
        try:
            tags = list(canvas.bindtags())

            if "V126CanvasClick" not in tags:
                canvas.bindtags(("V126CanvasClick",) + tuple(tags))

            self.root.bind_class("V126CanvasClick", "<Button-1>", self.v126_canvas_click)
        except Exception as exc:
            print("[v140] bind canvas impossible :", exc)

    v126_status(self, "v126 : clic Slice N = toutes les notes utilisant cette slice/pair.")


SliceIndexTracker.build_ui = v126_build_ui



# ---------------------------------------------------------------------
# v127 : correction contour rose décalé
# ---------------------------------------------------------------------
# Bug v126 :
# l'overlay de multisélection utilisait un parsing type slice N -> N-1.
# Mais l'UI affiche déjà slice 0, slice 1, slice 2...
# Résultat : le contour rose apparaissait une ligne en dessous.
#
# v127 :
# - détecte si les labels sont zéro-indexés
# - aligne le contour rose sur la vraie ligne
# - garde la multisélection par slice/pair
# - garde les flèches
# ---------------------------------------------------------------------


def v127_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        elif hasattr(self, "output_label"):
            self.output_label.config(text=text)
    except Exception:
        pass


def v127_canvas(self):
    return getattr(self, "canvas", None)


def v127_slice_label_numbers(self):
    canvas = v127_canvas(self)
    nums = []

    if canvas is None:
        return nums

    try:
        for item in canvas.find_all():
            try:
                if canvas.type(item) != "text":
                    continue

                text = canvas.itemcget(item, "text")
                m = re.search(r"\bslice\s*([0-9]+)\b", str(text), flags=re.I)

                if m:
                    nums.append(int(m.group(1)))
            except Exception:
                pass
    except Exception:
        pass

    return sorted(set(nums))


def v127_labels_are_zero_based(self):
    nums = v127_slice_label_numbers(self)
    return 0 in nums


def v127_display_to_lane(self, displayed_number):
    displayed_number = int(displayed_number)

    # Ton UI affiche slice 0, slice 1...
    # Donc displayed_number == lane.
    if v127_labels_are_zero_based(self):
        return displayed_number

    # Fallback pour une autre UI éventuelle qui afficherait slice 1, slice 2...
    return max(0, displayed_number - 1)


def v127_parse_slice_text(text):
    m = re.search(r"\bslice\s*([0-9]+)\b", str(text), flags=re.I)
    if not m:
        return None
    return int(m.group(1))


def v127_all_slice_rows(self):
    canvas = v127_canvas(self)
    out = []

    if canvas is None:
        return out

    zero_based = v127_labels_are_zero_based(self)

    try:
        for item in canvas.find_all():
            try:
                if canvas.type(item) != "text":
                    continue

                text = canvas.itemcget(item, "text")
                displayed = v127_parse_slice_text(text)

                if displayed is None:
                    continue

                lane = int(displayed) if zero_based else max(0, int(displayed) - 1)

                bbox = canvas.bbox(item)
                if not bbox:
                    continue

                x1, y1, x2, y2 = bbox

                out.append({
                    "canvas_item": item,
                    "text": text,
                    "displayed": int(displayed),
                    "lane": int(lane),
                    "bbox": bbox,
                    "cx": (x1 + x2) / 2.0,
                    "cy": (y1 + y2) / 2.0,
                })
            except Exception:
                pass
    except Exception:
        pass

    out.sort(key=lambda row: row["lane"])
    return out


def v127_clicked_slice_row(self, event):
    canvas = v127_canvas(self)

    if canvas is None:
        return None

    rows = v127_all_slice_rows(self)

    if not rows:
        return None

    try:
        cx = float(canvas.canvasx(event.x))
        cy = float(canvas.canvasy(event.y))
    except Exception:
        cx = float(getattr(event, "x", 0))
        cy = float(getattr(event, "y", 0))

    # Clic direct sur le texte slice N.
    try:
        hits = canvas.find_overlapping(cx - 8, cy - 8, cx + 8, cy + 8)

        for hit in hits:
            if canvas.type(hit) != "text":
                continue

            text = canvas.itemcget(hit, "text")
            displayed = v127_parse_slice_text(text)

            if displayed is not None:
                lane = v127_display_to_lane(self, displayed)

                return {
                    "displayed": int(displayed),
                    "lane": int(lane),
                    "text": text,
                    "reason": "direct_text_hit_v127",
                }
    except Exception:
        pass

    # Clic dans la colonne de gauche.
    try:
        left_limit = max(row["bbox"][2] for row in rows) + 50
    except Exception:
        left_limit = 160

    if cx > left_limit:
        return None

    closest = min(rows, key=lambda row: abs(float(row["cy"]) - cy))

    return {
        "displayed": int(closest["displayed"]),
        "lane": int(closest["lane"]),
        "text": closest["text"],
        "reason": "closest_left_header_v127",
    }


def v127_pair_candidates_for_lane(self, lane, displayed):
    candidates = set()

    lane = int(lane)
    displayed = int(displayed)

    candidates.add(lane)
    candidates.add(displayed)

    try:
        vals = list(getattr(self, "pair_values"))
        if 0 <= lane < len(vals):
            candidates.add(int(vals[lane]))
        if 0 <= displayed < len(vals):
            candidates.add(int(vals[displayed]))
    except Exception:
        pass

    try:
        lane_to_pair = getattr(self, "lane_to_pair", None)
        if isinstance(lane_to_pair, dict):
            for key in [lane, displayed]:
                if key in lane_to_pair:
                    candidates.add(int(lane_to_pair[key]))
                if str(key) in lane_to_pair:
                    candidates.add(int(lane_to_pair[str(key)]))
    except Exception:
        pass

    try:
        pair_to_lane = getattr(self, "pair_to_lane", None)
        if isinstance(pair_to_lane, dict):
            for pair, ln in pair_to_lane.items():
                try:
                    if int(ln) == lane:
                        candidates.add(int(pair))
                except Exception:
                    pass
    except Exception:
        pass

    return sorted(candidates)


def v127_note_matches(self, note, lane, displayed, pair_candidates):
    pair_candidates = set(int(x) for x in pair_candidates)

    try:
        note_pair = int(note.get("pair", -999999))
    except Exception:
        note_pair = -999999

    try:
        note_lane = int(note.get("lane", -999999))
    except Exception:
        note_lane = -999999

    if note_pair in pair_candidates:
        return True

    if note_lane == int(lane):
        return True

    try:
        pair_to_lane = getattr(self, "pair_to_lane", None)
        if isinstance(pair_to_lane, dict):
            ln = pair_to_lane.get(note_pair, pair_to_lane.get(str(note_pair), None))
            if ln is not None and int(ln) == int(lane):
                return True
    except Exception:
        pass

    return False


def v127_select_slice(self, lane, displayed, reason=""):
    pattern = list(getattr(self, "pattern", []) or [])
    pair_candidates = v127_pair_candidates_for_lane(self, lane, displayed)

    ids = []
    debug_rows = []

    for note in pattern:
        if v127_note_matches(self, note, lane, displayed, pair_candidates):
            try:
                nid = int(note.get("id"))
                ids.append(nid)
                debug_rows.append({
                    "id": nid,
                    "step": note.get("x_step"),
                    "lane": note.get("lane"),
                    "pair": note.get("pair"),
                    "role": note.get("learned_role", note.get("prior_label", "")),
                })
            except Exception:
                pass

    ids = sorted(set(ids))

    self.selected_ids_v125 = set(ids)
    self.selected_ids_v126 = set(ids)
    self.selected_ids_v127 = set(ids)

    self.selected_lane_v125 = int(lane)
    self.selected_lane_v127 = int(lane)
    self.selected_slice_displayed_v127 = int(displayed)
    self.selected_pair_candidates_v127 = set(pair_candidates)

    if ids:
        self.selected_id = ids[0]

    print("")
    print("[v140] SÉLECTION SLICE ALIGNÉE")
    print("[v140] reason:", reason)
    print("[v140] label affiché:", displayed)
    print("[v140] lane réelle:", lane)
    print("[v140] labels zero-based:", v127_labels_are_zero_based(self))
    print("[v140] pair_candidates:", pair_candidates)
    print("[v140] ids sélectionnés:", ids)
    for row in debug_rows:
        print("[v140] note:", row)
    print("")

    try:
        self.draw()
    except Exception as exc:
        print("[v140] draw impossible :", exc)

    if ids:
        v127_status(self, f"v127 : slice {displayed} sélectionnée — {len(ids)} notes.")
    else:
        v127_status(self, f"v127 : slice {displayed} — aucune note trouvée.")

    return "break"


def v127_canvas_click(self, event=None):
    row = v127_clicked_slice_row(self, event)

    if row is not None:
        return v127_select_slice(
            self,
            lane=row["lane"],
            displayed=row["displayed"],
            reason=row["reason"],
        )

    try:
        self.selected_ids_v127 = set()
        self.selected_ids_v126 = set()
        self.selected_ids_v125 = set()
        self.selected_pair_candidates_v127 = set()
    except Exception:
        pass

    return None


def v127_grid_geometry(self):
    canvas = v127_canvas(self)
    rows = v127_all_slice_rows(self)

    if canvas is None or not rows:
        return None

    try:
        header_limit = max(row["bbox"][2] for row in rows) + 20
    except Exception:
        header_limit = 90

    xs = []

    try:
        for item in canvas.find_all():
            if canvas.type(item) != "line":
                continue

            coords = canvas.coords(item)

            if len(coords) >= 4:
                x1, y1, x2, y2 = coords[:4]

                if abs(float(x1) - float(x2)) < 0.01 and float(x1) > header_limit:
                    xs.append(float(x1))
    except Exception:
        pass

    xs = sorted(set(round(x, 3) for x in xs))

    if len(xs) < 4:
        return None

    diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1) if xs[i + 1] - xs[i] > 1]

    if not diffs:
        return None

    cell_w = sorted(diffs)[len(diffs) // 2]
    grid_x0 = xs[0]

    ys = sorted(row["cy"] for row in rows)
    diffs_y = [ys[i + 1] - ys[i] for i in range(len(ys) - 1) if ys[i + 1] - ys[i] > 1]
    row_h = sorted(diffs_y)[len(diffs_y) // 2] if diffs_y else 22

    lane_y = {int(row["lane"]): float(row["cy"]) for row in rows}

    return {
        "grid_x0": float(grid_x0),
        "cell_w": float(cell_w),
        "row_h": float(row_h),
        "lane_y": lane_y,
    }


def v127_selected_items(self):
    ids = set(getattr(self, "selected_ids_v127", set()) or set())

    if not ids:
        ids = set(getattr(self, "selected_ids_v126", set()) or set())

    if not ids:
        ids = set(getattr(self, "selected_ids_v125", set()) or set())

    if not ids:
        return []

    out = []

    for item in getattr(self, "pattern", []) or []:
        try:
            if int(item.get("id")) in ids:
                out.append(item)
        except Exception:
            pass

    return out


_old_v127_draw = getattr(SliceIndexTracker, "draw", None)


def v127_draw(self, *args, **kwargs):
    result = None

    if callable(_old_v127_draw):
        result = _old_v127_draw(self, *args, **kwargs)

    selected = v127_selected_items(self)

    if not selected:
        return result

    canvas = v127_canvas(self)
    geom = v127_grid_geometry(self)

    if canvas is None or geom is None:
        return result

    try:
        canvas.delete("v127_multi_select_overlay")
        canvas.delete("v126_multi_select_overlay")
        canvas.delete("v125_multi_select_overlay")
    except Exception:
        pass

    for item in selected:
        try:
            step = int(item.get("x_step", 0))
            length = int(item.get("length", 2) or 2)
            lane = int(item.get("lane", 0))

            x1 = geom["grid_x0"] + step * geom["cell_w"]
            x2 = x1 + length * geom["cell_w"]

            cy = geom["lane_y"].get(lane)

            # Fallback : si lane_y manque, utiliser la lane sélectionnée.
            if cy is None:
                selected_lane = getattr(self, "selected_lane_v127", None)
                if selected_lane is not None:
                    cy = geom["lane_y"].get(int(selected_lane))

            if cy is None:
                continue

            y1 = cy - geom["row_h"] * 0.42
            y2 = cy + geom["row_h"] * 0.42

            canvas.create_rectangle(
                x1 + 1,
                y1 + 1,
                x2 - 1,
                y2 - 1,
                outline="#ff5fd7",
                width=3,
                tags=("v127_multi_select_overlay",),
            )
        except Exception:
            pass

    try:
        canvas.tag_raise("v127_multi_select_overlay")
    except Exception:
        pass

    return result


_old_v127_build_ui = SliceIndexTracker.build_ui


def v127_build_ui(self):
    _old_v127_build_ui(self)

    try:
        self.root.title("BreakbeatAI v140 — Princess clean toolbar")
    except Exception:
        pass

    canvas = v127_canvas(self)

    if canvas is not None:
        try:
            tags = list(canvas.bindtags())

            if "V127CanvasClick" not in tags:
                canvas.bindtags(("V127CanvasClick",) + tuple(tags))

            self.root.bind_class("V127CanvasClick", "<Button-1>", self.v127_canvas_click)
        except Exception as exc:
            print("[v140] bind canvas impossible :", exc)

    v127_status(self, "v127 : contour rose aligné sur la vraie slice.")


SliceIndexTracker.v127_canvas_click = v127_canvas_click
SliceIndexTracker.v126_canvas_click = v127_canvas_click
SliceIndexTracker.v125_canvas_click = v127_canvas_click

SliceIndexTracker.v127_select_slice = v127_select_slice

if callable(_old_v127_draw):
    SliceIndexTracker.draw = v127_draw

SliceIndexTracker.build_ui = v127_build_ui



# ---------------------------------------------------------------------
# v128 : multisélection native sans contour calculé
# ---------------------------------------------------------------------
# Correction :
# - plus de contour rose calculé à la main
# - donc plus de décalage d'une case à droite
# - clic sur "slice N" = sélection logique de toutes les notes de cette ligne
# - flèches = bougent toute la sélection comme une note normale
# ---------------------------------------------------------------------


def v128_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        elif hasattr(self, "output_label"):
            self.output_label.config(text=text)
    except Exception:
        pass


def v128_canvas(self):
    return getattr(self, "canvas", None)


def v128_all_slice_rows(self):
    canvas = v128_canvas(self)
    rows = []

    if canvas is None:
        return rows

    try:
        for item in canvas.find_all():
            try:
                if canvas.type(item) != "text":
                    continue

                text = canvas.itemcget(item, "text")
                m = re.search(r"\bslice\s*([0-9]+)\b", str(text), flags=re.I)

                if not m:
                    continue

                number = int(m.group(1))
                bbox = canvas.bbox(item)

                if not bbox:
                    continue

                x1, y1, x2, y2 = bbox

                rows.append({
                    "number": number,
                    "lane": number,
                    "text": text,
                    "bbox": bbox,
                    "cx": (x1 + x2) / 2.0,
                    "cy": (y1 + y2) / 2.0,
                })
            except Exception:
                pass
    except Exception:
        pass

    rows.sort(key=lambda r: r["lane"])
    return rows


def v128_clicked_slice_lane(self, event):
    canvas = v128_canvas(self)

    if canvas is None:
        return None

    rows = v128_all_slice_rows(self)

    if not rows:
        return None

    try:
        x = float(canvas.canvasx(event.x))
        y = float(canvas.canvasy(event.y))
    except Exception:
        x = float(getattr(event, "x", 0))
        y = float(getattr(event, "y", 0))

    # Clic direct sur le texte slice N.
    try:
        hits = canvas.find_overlapping(x - 8, y - 8, x + 8, y + 8)

        for hit in hits:
            if canvas.type(hit) != "text":
                continue

            text = canvas.itemcget(hit, "text")
            m = re.search(r"\bslice\s*([0-9]+)\b", str(text), flags=re.I)

            if m:
                return int(m.group(1))
    except Exception:
        pass

    # Clic dans la colonne de gauche.
    try:
        left_limit = max(row["bbox"][2] for row in rows) + 45
    except Exception:
        left_limit = 150

    if x > left_limit:
        return None

    closest = min(rows, key=lambda row: abs(row["cy"] - y))
    return int(closest["lane"])


def v128_note_visual_lane(self, note):
    """
    Ligne réellement affichée.
    On préfère pair_to_lane, car le dessin peut utiliser pair -> ligne.
    Fallback : note['lane'].
    """
    try:
        pair = int(note.get("pair", -999999))
    except Exception:
        pair = -999999

    try:
        pair_to_lane = getattr(self, "pair_to_lane", None)
        if isinstance(pair_to_lane, dict):
            if pair in pair_to_lane:
                return int(pair_to_lane[pair])
            if str(pair) in pair_to_lane:
                return int(pair_to_lane[str(pair)])
    except Exception:
        pass

    try:
        return int(note.get("lane", 0))
    except Exception:
        return 0


def v128_pair_for_lane(self, lane, fallback_pair=None):
    lane = int(lane)

    try:
        pair_to_lane = getattr(self, "pair_to_lane", None)
        if isinstance(pair_to_lane, dict):
            for pair, ln in pair_to_lane.items():
                try:
                    if int(ln) == lane:
                        return int(pair)
                except Exception:
                    pass
    except Exception:
        pass

    try:
        lane_to_pair = getattr(self, "lane_to_pair", None)
        if isinstance(lane_to_pair, dict):
            if lane in lane_to_pair:
                return int(lane_to_pair[lane])
            if str(lane) in lane_to_pair:
                return int(lane_to_pair[str(lane)])
    except Exception:
        pass

    try:
        vals = list(getattr(self, "pair_values"))
        if 0 <= lane < len(vals):
            return int(vals[lane])
    except Exception:
        pass

    return int(fallback_pair if fallback_pair is not None else lane)


def v128_max_lane(self):
    max_lane = 0

    try:
        rows = v128_all_slice_rows(self)
        if rows:
            max_lane = max(max_lane, max(int(row["lane"]) for row in rows))
    except Exception:
        pass

    try:
        vals = list(getattr(self, "pair_values"))
        if vals:
            max_lane = max(max_lane, len(vals) - 1)
    except Exception:
        pass

    try:
        pair_to_lane = getattr(self, "pair_to_lane", None)
        if isinstance(pair_to_lane, dict):
            lanes = [int(v) for v in pair_to_lane.values()]
            if lanes:
                max_lane = max(max_lane, max(lanes))
    except Exception:
        pass

    return int(max_lane)


def v128_select_lane(self, lane):
    lane = int(lane)
    ids = []
    debug = []

    for note in getattr(self, "pattern", []) or []:
        try:
            note_lane = v128_note_visual_lane(self, note)

            if note_lane == lane:
                nid = int(note.get("id"))
                ids.append(nid)
                debug.append({
                    "id": nid,
                    "step": note.get("x_step"),
                    "lane": note.get("lane"),
                    "visual_lane": note_lane,
                    "pair": note.get("pair"),
                    "role": note.get("learned_role", note.get("prior_label", "")),
                })
        except Exception:
            pass

    ids = sorted(set(ids))

    self.selected_ids_v128 = set(ids)
    self.selected_lane_v128 = lane

    # Compat avec les anciennes versions, mais sans overlay.
    self.selected_ids_v127 = set()
    self.selected_ids_v126 = set()
    self.selected_ids_v125 = set()

    if ids:
        self.selected_id = ids[0]

    print("")
    print("[v140] MULTI-SÉLECTION LIGNE")
    print("[v140] slice/lane:", lane)
    print("[v140] ids:", ids)
    for row in debug:
        print("[v140] note:", row)
    print("")

    try:
        self.draw()
    except Exception:
        pass

    v128_status(self, f"v128 : slice {lane} sélectionnée — {len(ids)} notes. Flèches = déplacer.")
    return "break"


def v128_clear_selection(self):
    self.selected_ids_v128 = set()
    self.selected_lane_v128 = None

    # On nettoie aussi les anciens overlays.
    self.selected_ids_v127 = set()
    self.selected_ids_v126 = set()
    self.selected_ids_v125 = set()

    canvas = v128_canvas(self)
    if canvas is not None:
        try:
            canvas.delete("v127_multi_select_overlay")
            canvas.delete("v126_multi_select_overlay")
            canvas.delete("v125_multi_select_overlay")
        except Exception:
            pass


def v128_canvas_click(self, event=None):
    lane = v128_clicked_slice_lane(self, event)

    if lane is not None:
        return v128_select_lane(self, lane)

    v128_clear_selection(self)
    return None


def v128_selected_notes(self):
    ids = set(getattr(self, "selected_ids_v128", set()) or set())

    if not ids:
        return []

    out = []

    for note in getattr(self, "pattern", []) or []:
        try:
            if int(note.get("id")) in ids:
                out.append(note)
        except Exception:
            pass

    return out


def v128_after_move(self):
    try:
        self.draw()
    except Exception as exc:
        print("[v140] draw impossible :", exc)

    try:
        self.refresh_panel()
    except Exception:
        pass

    try:
        self.write_latest_pattern(reason="v128_slice_multiselect_native_move")
    except Exception:
        pass


def v128_move_selection(self, dx=0, dy=0):
    notes = v128_selected_notes(self)

    if not notes:
        return None

    if dx:
        for note in notes:
            try:
                length = int(note.get("length", 2) or 2)
                x = int(note.get("x_step", 0))
                nx = x + int(dx)
                nx = max(0, min(64 - length, nx))
                note["x_step"] = int(nx)
            except Exception:
                pass

    if dy:
        max_lane = v128_max_lane(self)

        for note in notes:
            try:
                old_pair = int(note.get("pair", 0))
                old_lane = v128_note_visual_lane(self, note)
                new_lane = max(0, min(max_lane, old_lane + int(dy)))
                new_pair = v128_pair_for_lane(self, new_lane, old_pair)

                note["lane"] = int(new_lane)
                note["pair"] = int(new_pair)
            except Exception:
                pass

        try:
            self.selected_lane_v128 = v128_note_visual_lane(self, notes[0])
        except Exception:
            pass

    v128_after_move(self)

    try:
        lane = int(getattr(self, "selected_lane_v128", -1))
        v128_status(self, f"v128 : {len(notes)} notes déplacées — slice {lane}.")
    except Exception:
        v128_status(self, f"v128 : {len(notes)} notes déplacées.")

    return "break"


def v128_key_left(self, event=None):
    if v128_selected_notes(self):
        return v128_move_selection(self, dx=-1)
    return None


def v128_key_right(self, event=None):
    if v128_selected_notes(self):
        return v128_move_selection(self, dx=1)
    return None


def v128_key_up(self, event=None):
    if v128_selected_notes(self):
        return v128_move_selection(self, dy=-1)
    return None


def v128_key_down(self, event=None):
    if v128_selected_notes(self):
        return v128_move_selection(self, dy=1)
    return None


_old_v128_draw = getattr(SliceIndexTracker, "draw", None)


def v128_draw(self, *args, **kwargs):
    result = None

    if callable(_old_v128_draw):
        result = _old_v128_draw(self, *args, **kwargs)

    # Important : on supprime les anciens contours roses calculés.
    canvas = v128_canvas(self)
    if canvas is not None:
        try:
            canvas.delete("v127_multi_select_overlay")
            canvas.delete("v126_multi_select_overlay")
            canvas.delete("v125_multi_select_overlay")
        except Exception:
            pass

    return result


_old_v128_build_ui = SliceIndexTracker.build_ui


def v128_build_ui(self):
    _old_v128_build_ui(self)

    self.selected_ids_v128 = set()
    self.selected_lane_v128 = None

    try:
        self.root.title("BreakbeatAI v140 — Princess clean toolbar")
    except Exception:
        pass

    canvas = v128_canvas(self)

    if canvas is not None:
        try:
            tags = list(canvas.bindtags())

            # On met v128 AVANT les anciens handlers.
            new_tags = []
            for tag in ["V128CanvasClick", "V128Keys"]:
                if tag not in tags:
                    new_tags.append(tag)

            canvas.bindtags(tuple(new_tags) + tuple(tags))

            self.root.bind_class("V128CanvasClick", "<Button-1>", self.v128_canvas_click)
        except Exception as exc:
            print("[v140] bind canvas impossible :", exc)

    try:
        for widget in [self.root, canvas]:
            if widget is None:
                continue

            tags = list(widget.bindtags())
            if "V128Keys" not in tags:
                widget.bindtags(("V128Keys",) + tuple(tags))
    except Exception:
        pass

    try:
        self.root.bind_class("V128Keys", "<Left>", self.v128_key_left)
        self.root.bind_class("V128Keys", "<Right>", self.v128_key_right)
        self.root.bind_class("V128Keys", "<Up>", self.v128_key_up)
        self.root.bind_class("V128Keys", "<Down>", self.v128_key_down)
    except Exception as exc:
        print("[v140] bind flèches impossible :", exc)

    v128_status(self, "v128 : clic sur slice N = sélection de toute la ligne, sans contour décalé.")


SliceIndexTracker.v128_canvas_click = v128_canvas_click
SliceIndexTracker.v128_key_left = v128_key_left
SliceIndexTracker.v128_key_right = v128_key_right
SliceIndexTracker.v128_key_up = v128_key_up
SliceIndexTracker.v128_key_down = v128_key_down

# Remplace aussi les anciens handlers pour éviter qu'ils reprennent la main.
SliceIndexTracker.v127_canvas_click = v128_canvas_click
SliceIndexTracker.v126_canvas_click = v128_canvas_click
SliceIndexTracker.v125_canvas_click = v128_canvas_click

if callable(_old_v128_draw):
    SliceIndexTracker.draw = v128_draw

SliceIndexTracker.build_ui = v128_build_ui



# ---------------------------------------------------------------------
# v129 : contour vert sur la multisélection
# ---------------------------------------------------------------------
# Base : v128.
# Ajoute seulement :
# - contour vert autour de toutes les notes multisélectionnées
# - pas de calcul case -> pixel pour éviter le décalage horizontal
# - on entoure les rectangles déjà dessinés par le tracker
# ---------------------------------------------------------------------


def v129_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        elif hasattr(self, "output_label"):
            self.output_label.config(text=text)
    except Exception:
        pass


def v129_canvas(self):
    return getattr(self, "canvas", None)


def v129_selected_notes(self):
    ids = set(getattr(self, "selected_ids_v128", set()) or set())

    if not ids:
        return []

    out = []

    for note in getattr(self, "pattern", []) or []:
        try:
            if int(note.get("id")) in ids:
                out.append(note)
        except Exception:
            pass

    return out


def v129_note_visual_lane(self, note):
    try:
        return v128_note_visual_lane(self, note)
    except Exception:
        pass

    try:
        return int(note.get("lane", 0))
    except Exception:
        return 0


def v129_slice_rows(self):
    canvas = v129_canvas(self)
    rows = {}

    if canvas is None:
        return rows

    try:
        for item in canvas.find_all():
            try:
                if canvas.type(item) != "text":
                    continue

                text = canvas.itemcget(item, "text")
                m = re.search(r"\bslice\s*([0-9]+)\b", str(text), flags=re.I)

                if not m:
                    continue

                lane = int(m.group(1))
                bbox = canvas.bbox(item)

                if not bbox:
                    continue

                x1, y1, x2, y2 = bbox

                rows[lane] = {
                    "lane": lane,
                    "bbox": bbox,
                    "cy": (y1 + y2) / 2.0,
                    "right": x2,
                }
            except Exception:
                pass
    except Exception:
        pass

    return rows


def v129_header_right(rows):
    if not rows:
        return 120

    try:
        return max(row["right"] for row in rows.values()) + 20
    except Exception:
        return 120


def v129_find_rect_around_text(canvas, text_bbox):
    tx1, ty1, tx2, ty2 = text_bbox
    cx = (tx1 + tx2) / 2.0
    cy = (ty1 + ty2) / 2.0

    try:
        hits = canvas.find_overlapping(tx1 - 12, ty1 - 10, tx2 + 12, ty2 + 10)
    except Exception:
        return None

    best = None
    best_area = None

    for hit in hits:
        try:
            if canvas.type(hit) != "rectangle":
                continue

            bbox = canvas.bbox(hit)
            if not bbox:
                continue

            x1, y1, x2, y2 = bbox

            # Le rectangle de la note contient le texte.
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue

            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            area = w * h

            # On évite les énormes rectangles de fond.
            if area > 20000:
                continue

            if best is None or area < best_area:
                best = bbox
                best_area = area
        except Exception:
            pass

    return best


def v129_draw_green_multiselect(self):
    canvas = v129_canvas(self)

    if canvas is None:
        return

    try:
        canvas.delete("v129_multi_select_outline")
        canvas.delete("v127_multi_select_overlay")
        canvas.delete("v126_multi_select_overlay")
        canvas.delete("v125_multi_select_overlay")
    except Exception:
        pass

    notes = v129_selected_notes(self)

    if not notes:
        return

    rows = v129_slice_rows(self)
    header_right = v129_header_right(rows)

    wanted = set()

    for note in notes:
        try:
            pair = int(note.get("pair", -999999))
            lane = int(v129_note_visual_lane(self, note))
            wanted.add((pair, lane))
        except Exception:
            pass

    drawn = 0
    seen_rects = set()

    try:
        all_items = list(canvas.find_all())
    except Exception:
        all_items = []

    for item in all_items:
        try:
            if canvas.type(item) != "text":
                continue

            text = str(canvas.itemcget(item, "text")).strip()

            if not re.fullmatch(r"-?[0-9]+", text):
                continue

            pair = int(text)
            text_bbox = canvas.bbox(item)

            if not text_bbox:
                continue

            tx1, ty1, tx2, ty2 = text_bbox
            tcx = (tx1 + tx2) / 2.0
            tcy = (ty1 + ty2) / 2.0

            # On ignore les numéros dans la colonne gauche.
            if tcx <= header_right:
                continue

            # Trouve la ligne slice la plus proche.
            if rows:
                lane = min(rows.keys(), key=lambda ln: abs(rows[ln]["cy"] - tcy))
            else:
                lane = None

            if lane is None:
                continue

            if (pair, int(lane)) not in wanted:
                continue

            rect_bbox = v129_find_rect_around_text(canvas, text_bbox)

            if rect_bbox is None:
                # Fallback autour du texte, au cas où le rectangle n'est pas retrouvé.
                x1, y1, x2, y2 = tx1 - 14, ty1 - 8, tx2 + 14, ty2 + 8
            else:
                x1, y1, x2, y2 = rect_bbox

            key = (round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2))

            if key in seen_rects:
                continue

            seen_rects.add(key)

            canvas.create_rectangle(
                x1 + 1,
                y1 + 1,
                x2 - 1,
                y2 - 1,
                outline="#00ff88",
                width=3,
                tags=("v129_multi_select_outline",),
            )

            drawn += 1

        except Exception:
            pass

    try:
        canvas.tag_raise("v129_multi_select_outline")
    except Exception:
        pass

    print(f"[v140] contours verts dessinés : {drawn}")


_old_v129_draw = getattr(SliceIndexTracker, "draw", None)


def v129_draw(self, *args, **kwargs):
    result = None

    if callable(_old_v129_draw):
        result = _old_v129_draw(self, *args, **kwargs)

    try:
        v129_draw_green_multiselect(self)
    except Exception as exc:
        print("[v140] contour vert impossible :", exc)

    return result


_old_v129_build_ui = SliceIndexTracker.build_ui


def v129_build_ui(self):
    _old_v129_build_ui(self)

    try:
        self.root.title("BreakbeatAI v140 — Princess clean toolbar")
    except Exception:
        pass

    v129_status(self, "v129 : multisélection avec contour vert, flèches conservées.")


if callable(_old_v129_draw):
    SliceIndexTracker.draw = v129_draw

SliceIndexTracker.build_ui = v129_build_ui



# ---------------------------------------------------------------------
# v130 : audition quand on bouge une multisélection
# ---------------------------------------------------------------------
# Base : v129.
# Ajoute :
# - flèches sur multisélection = déplacement
# - puis lecture/audition de la note/slice d'arrivée
# - comme quand on déplace une seule note
# ---------------------------------------------------------------------


def v130_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        elif hasattr(self, "output_label"):
            self.output_label.config(text=text)
    except Exception:
        pass


def v130_canvas(self):
    return getattr(self, "canvas", None)


def v130_find_button(self, label):
    target = str(label).strip().lower()

    def walk(widget):
        out = [widget]
        try:
            for child in widget.winfo_children():
                out.extend(walk(child))
        except Exception:
            pass
        return out

    try:
        widgets = walk(self.root)
    except Exception:
        widgets = []

    for widget in widgets:
        try:
            text = str(widget.cget("text")).strip().lower()
        except Exception:
            continue

        if text == target:
            return widget

    return None


def v130_selected_notes(self):
    # Utilise la sélection logique de v128/v129.
    ids = set(getattr(self, "selected_ids_v128", set()) or set())

    if not ids:
        return []

    out = []

    for note in getattr(self, "pattern", []) or []:
        try:
            if int(note.get("id")) in ids:
                out.append(note)
        except Exception:
            pass

    return out


def v130_representative_note(notes):
    """
    Pour ne pas jouer 12 sons en même temps,
    on auditionne la note sélectionnée la plus à gauche.
    """
    if not notes:
        return None

    try:
        return sorted(notes, key=lambda n: (int(n.get("x_step", 0)), int(n.get("lane", 0))))[0]
    except Exception:
        return notes[0]


def v130_audition_after_move(self, notes):
    note = v130_representative_note(notes)

    if note is None:
        return

    # Petit debounce pour éviter les doubles sons Tk.
    if getattr(self, "_v130_audition_busy", False):
        return

    self._v130_audition_busy = True

    try:
        nid = int(note.get("id"))
        self.selected_id = nid
    except Exception:
        pass

    try:
        self.refresh_panel()
    except Exception:
        pass

    def do_play():
        played = False

        # Méthodes possibles selon les versions du tracker.
        method_names = [
            "play_slice",
            "play_current_slice",
            "audition_slice",
            "audition_current_slice",
            "play_selected_slice",
            "preview_selected_slice",
            "audition_selected",
            "play_selected",
            "play_note",
            "audition_note",
        ]

        for name in method_names:
            fn = getattr(self, name, None)

            if not callable(fn):
                continue

            try:
                fn()
                print(f"[v140] audition via {name}()")
                played = True
                break
            except TypeError:
                try:
                    fn(None)
                    print(f"[v140] audition via {name}(None)")
                    played = True
                    break
                except TypeError:
                    try:
                        fn(note)
                        print(f"[v140] audition via {name}(note)")
                        played = True
                        break
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                pass

        # Fallback le plus fiable : bouton visible "Play slice".
        if not played:
            btn = v130_find_button(self, "Play slice")
            if btn is not None:
                try:
                    btn.invoke()
                    print("[v140] audition via bouton Play slice")
                    played = True
                except Exception as exc:
                    print("[v140] Play slice invoke impossible :", exc)

        if not played:
            print("[v140] audition introuvable : aucune méthode Play slice trouvée")

        try:
            self.root.after(90, lambda: setattr(self, "_v130_audition_busy", False))
        except Exception:
            self._v130_audition_busy = False

    try:
        self.root.after(20, do_play)
    except Exception:
        do_play()


def v128_move_selection(self, dx=0, dy=0):
    """
    v130 remplace le move de v128 :
    même déplacement, mais audition après le déplacement.
    """
    notes = v130_selected_notes(self)

    if not notes:
        return None

    if dx:
        for note in notes:
            try:
                length = int(note.get("length", 2) or 2)
                x = int(note.get("x_step", 0))
                nx = x + int(dx)
                nx = max(0, min(64 - length, nx))
                note["x_step"] = int(nx)
            except Exception:
                pass

    if dy:
        try:
            max_lane = v128_max_lane(self)
        except Exception:
            max_lane = 15

        for note in notes:
            try:
                old_pair = int(note.get("pair", 0))

                try:
                    old_lane = v128_note_visual_lane(self, note)
                except Exception:
                    old_lane = int(note.get("lane", 0))

                new_lane = max(0, min(max_lane, old_lane + int(dy)))

                try:
                    new_pair = v128_pair_for_lane(self, new_lane, old_pair)
                except Exception:
                    new_pair = old_pair

                note["lane"] = int(new_lane)
                note["pair"] = int(new_pair)
            except Exception:
                pass

        try:
            if notes:
                try:
                    self.selected_lane_v128 = v128_note_visual_lane(self, notes[0])
                except Exception:
                    self.selected_lane_v128 = int(notes[0].get("lane", 0))
        except Exception:
            pass

    try:
        v128_after_move(self)
    except Exception:
        try:
            self.draw()
        except Exception:
            pass
        try:
            self.refresh_panel()
        except Exception:
            pass

    # FIX v130 : jouer la note/slice d'arrivée.
    v130_audition_after_move(self, notes)

    try:
        lane = int(getattr(self, "selected_lane_v128", -1))
        v130_status(self, f"v130 : {len(notes)} notes déplacées — slice {lane}. Audition.")
    except Exception:
        v130_status(self, f"v130 : {len(notes)} notes déplacées. Audition.")

    return "break"


def v130_key_left(self, event=None):
    if v130_selected_notes(self):
        return v128_move_selection(self, dx=-1)
    return None


def v130_key_right(self, event=None):
    if v130_selected_notes(self):
        return v128_move_selection(self, dx=1)
    return None


def v130_key_up(self, event=None):
    if v130_selected_notes(self):
        return v128_move_selection(self, dy=-1)
    return None


def v130_key_down(self, event=None):
    if v130_selected_notes(self):
        return v128_move_selection(self, dy=1)
    return None


_old_v130_build_ui = SliceIndexTracker.build_ui


def v130_build_ui(self):
    _old_v130_build_ui(self)

    try:
        self.root.title("BreakbeatAI v140 — Princess clean toolbar")
    except Exception:
        pass

    canvas = v130_canvas(self)

    try:
        for widget in [self.root, canvas]:
            if widget is None:
                continue

            tags = list(widget.bindtags())
            if "V130Keys" not in tags:
                widget.bindtags(("V130Keys",) + tuple(tags))
    except Exception:
        pass

    try:
        self.root.bind_class("V130Keys", "<Left>", self.v130_key_left)
        self.root.bind_class("V130Keys", "<Right>", self.v130_key_right)
        self.root.bind_class("V130Keys", "<Up>", self.v130_key_up)
        self.root.bind_class("V130Keys", "<Down>", self.v130_key_down)

        # On repointe aussi les anciens tags vers les handlers v130.
        self.root.bind_class("V128Keys", "<Left>", self.v130_key_left)
        self.root.bind_class("V128Keys", "<Right>", self.v130_key_right)
        self.root.bind_class("V128Keys", "<Up>", self.v130_key_up)
        self.root.bind_class("V128Keys", "<Down>", self.v130_key_down)
    except Exception as exc:
        print("[v140] bind flèches impossible :", exc)

    v130_status(self, "v130 : déplacer une multisélection auditionne la note d'arrivée.")


SliceIndexTracker.v130_key_left = v130_key_left
SliceIndexTracker.v130_key_right = v130_key_right
SliceIndexTracker.v130_key_up = v130_key_up
SliceIndexTracker.v130_key_down = v130_key_down

# Les anciens noms appellent maintenant la version avec audition.
SliceIndexTracker.v128_key_left = v130_key_left
SliceIndexTracker.v128_key_right = v130_key_right
SliceIndexTracker.v128_key_up = v130_key_up
SliceIndexTracker.v128_key_down = v130_key_down

SliceIndexTracker.build_ui = v130_build_ui



# ---------------------------------------------------------------------
# v131 : restauration stricte du placement kick/hat/snare
# ---------------------------------------------------------------------
# Generate Candidate ne doit plus improviser la structure.
#
# Pattern forcé :
# 00 kick | 02 hat | 04 snare | 06 hat
# 08 hat  | 10 kick| 12 snare | 14 hat
# répété sur 16-31, puis miroir strict sur 32-63.
#
# Garde :
# - Export WAV only v122/v121
# - cases -3.5px
# - second 32 miroir strict
# - multisélection slice
# - audition au déplacement
# ---------------------------------------------------------------------

V131_ROLE_SEQUENCE_16 = [
    (0,  "kick"),
    (2,  "hat"),
    (4,  "snare"),
    (6,  "hat"),
    (8,  "hat"),
    (10, "kick"),
    (12, "snare"),
    (14, "hat"),
]


def v131_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        elif hasattr(self, "output_label"):
            self.output_label.config(text=text)
    except Exception:
        pass


def v131_current_break(self):
    for name in ["v121_current_break", "v120_current_break", "v118_current_break", "v110_current_break"]:
        fn = globals().get(name)
        if callable(fn):
            try:
                return fn(self)
            except Exception:
                pass

    try:
        return str(getattr(self, "safe"))
    except Exception:
        return "Camo_Break_-_3A"


def v131_force_64(self):
    try:
        v110_force_64(self)
        return
    except Exception:
        pass

    for attr in [
        "total_steps", "loop_steps", "grid_steps", "pattern_steps",
        "num_steps", "step_count", "n_steps", "steps",
        "cols", "columns", "grid_cols", "loop_len_steps",
        "loop_length_steps",
    ]:
        try:
            if hasattr(self, attr):
                setattr(self, attr, 64)
        except Exception:
            pass


def v131_all_pairs(self):
    try:
        pairs = [int(x) for x in self.pair_values]
        if pairs:
            return pairs
    except Exception:
        pass

    try:
        pairs = sorted(int(x) for x in self.pair_to_lane.keys())
        if pairs:
            return pairs
    except Exception:
        pass

    return list(range(16))


def v131_load_roles_safe(self):
    try:
        return v110_load_roles(self)
    except Exception:
        return {"kick": [], "snare": [], "hat": [], "ghost_snare": [], "bad": []}


def v131_load_locks_safe(self):
    try:
        return v110_load_locks(self)
    except Exception:
        return {}


def v131_pick_role_pairs(self):
    roles = v131_load_roles_safe(self)
    locks = v131_load_locks_safe(self)
    all_pairs = v131_all_pairs(self)

    chosen = {}
    used = set()

    for role in ["kick", "snare", "hat"]:
        candidates = []

        try:
            if role in locks:
                candidates.append((int(locks[role]), "manual_lock"))
        except Exception:
            pass

        for p in roles.get(role, []) or []:
            try:
                candidates.append((int(p), "role_pool"))
            except Exception:
                pass

        for p in all_pairs:
            candidates.append((int(p), "fallback"))

        picked = None

        for pair, source in candidates:
            if pair not in used:
                picked = (pair, source)
                break

        if picked is None:
            picked = candidates[0] if candidates else (0, "zero")

        pair, source = picked
        chosen[role] = {"pair": int(pair), "source": f"v131_{source}"}
        used.add(int(pair))

    return chosen


def v131_role_grid_64():
    grid = []

    for base in [0, 16, 32, 48]:
        for step, role in V131_ROLE_SEQUENCE_16:
            grid.append((base + step, role))

    return grid


def v131_make_note(self, idx, step, role, pair, source):
    try:
        lane = int(self.pair_to_lane.get(int(pair), 0))
    except Exception:
        try:
            lane = int(pair)
        except Exception:
            lane = 0

    return {
        "id": int(idx),
        "x_step": int(step),
        "lane": int(lane),
        "pair": int(pair),
        "length": 2,
        "variation_bar": int(step) // 8,
        "variation_pos": int(step) % 8,
        "hit_slot": int(step) // 2,
        "ai_generated": True,
        "ai_model": "v131_restore_kick_hat_snare",
        "learned_role": str(role),
        "prior_label": str(role),
        "main_role_source": str(source),
        "v131_strict_role_sequence": True,
        "v131_sequence": "kick hat snare hat hat kick snare hat",
    }


def v131_validate(pattern):
    expected = []

    for base in [0, 16, 32, 48]:
        expected.extend([
            (base + 0,  "kick"),
            (base + 2,  "hat"),
            (base + 4,  "snare"),
            (base + 6,  "hat"),
            (base + 8,  "hat"),
            (base + 10, "kick"),
            (base + 12, "snare"),
            (base + 14, "hat"),
        ])

    got = [(int(n["x_step"]), str(n["learned_role"])) for n in sorted(pattern, key=lambda x: int(x["x_step"]))]

    if got != expected:
        raise RuntimeError(f"v131 sequence cassée : {got}")


def v131_generate_candidate(self, event=None, reject_mode=False):
    v131_force_64(self)

    current = v131_current_break(self)
    chosen = v131_pick_role_pairs(self)

    grid = v131_role_grid_64()
    pattern = []

    for idx, (step, role) in enumerate(grid):
        pair = chosen[role]["pair"]
        source = chosen[role]["source"]
        pattern.append(v131_make_note(self, idx, step, role, pair, source))

    v131_validate(pattern)

    self.pattern = pattern
    self.selected_id = 0 if self.pattern else None
    self.v131_current_role_sequence = "kick hat snare hat hat kick snare hat"

    print("")
    print("[v140] GENERATE CANDIDATE STRICT")
    print("[v140] break:", current)
    print("[v140] sequence: kick hat snare hat | hat kick snare hat")
    print("[v140] répété sur 64 cases")
    for role in ["kick", "snare", "hat"]:
        print(f"[v140] {role:6s}: pair={chosen[role]['pair']} source={chosen[role]['source']}")
    print("")

    try:
        self.stop_playhead()
        self.stop_audio()
    except Exception:
        pass

    try:
        self.draw()
    except Exception as exc:
        print("[v140] draw impossible :", exc)

    try:
        self.refresh_panel()
    except Exception:
        pass

    try:
        self.write_latest_pattern(reason="v131_restore_kick_hat_snare")
    except Exception:
        pass

    v131_status(self, "v131 : Generate Candidate = kick hat snare hat | hat kick snare hat.")
    return "break"


def v131_reject_bad(self, event=None):
    # Même grille stricte : Reject ne casse plus le placement.
    return v131_generate_candidate(self, event=event, reject_mode=True)


# Force tous les boutons generate/reject vers le générateur strict.
SliceIndexTracker.generate_candidate = v131_generate_candidate
SliceIndexTracker.generate_full_candidate = v131_generate_candidate
SliceIndexTracker.generate_ai_pattern = v131_generate_candidate
SliceIndexTracker.generate_cross_song_template = v131_generate_candidate
SliceIndexTracker.generate_safe_experiment = v131_generate_candidate
SliceIndexTracker.generate_locked_roles = v131_generate_candidate
SliceIndexTracker.generate_role_aware = v131_generate_candidate
SliceIndexTracker.generate_safe_role_ai = v131_generate_candidate

SliceIndexTracker.reject_bad_v110 = v131_reject_bad
SliceIndexTracker.reject_bad = v131_reject_bad
SliceIndexTracker.bad_candidate = v131_reject_bad
SliceIndexTracker.reject_candidate = v131_reject_bad
SliceIndexTracker.reject_pattern = v131_reject_bad

_old_v131_build_ui = SliceIndexTracker.build_ui


def v131_build_ui(self):
    _old_v131_build_ui(self)

    try:
        self.root.title("BreakbeatAI v140 — Princess clean toolbar")
    except Exception:
        pass

    # Repointe les boutons déjà créés.
    try:
        for widget in self.root.winfo_children():
            pass
    except Exception:
        pass

    def walk(widget):
        out = [widget]
        try:
            for child in widget.winfo_children():
                out.extend(walk(child))
        except Exception:
            pass
        return out

    try:
        for widget in walk(self.root):
            try:
                text = str(widget.cget("text")).strip().lower()
            except Exception:
                continue

            if text == "generate candidate":
                widget.configure(command=self.generate_candidate)
            elif text in ["reject / bad", "reject", "bad"]:
                widget.configure(command=self.reject_bad)
    except Exception as exc:
        print("[v140] repatch boutons impossible :", exc)

    try:
        self.root.bind("<F12>", self.generate_candidate)
        self.root.bind("<F9>", self.reject_bad)
    except Exception:
        pass

    v131_status(self, "v131 : placement restauré — kick hat snare hat | hat kick snare hat.")


SliceIndexTracker.build_ui = v131_build_ui



# ---------------------------------------------------------------------
# v140 : Princess clean toolbar fiable
# ---------------------------------------------------------------------
# Repart de v131 propre.
# Garde :
# - grille intacte
# - kick/hat/snare strict
# - multisélection + flèches + audition héritées
# - export WAV only hérité
#
# Fix :
# - bouton generate visible
# - bouton export visible
# - section du bas masquée
# - ancienne IA/train masquée
# ---------------------------------------------------------------------

P140_BG = "#0b0610"
P140_PANEL = "#1b1023"
P140_PINK = "#ff8be8"
P140_PINK_DARK = "#7a2f68"
P140_PURPLE = "#30223e"
P140_BLUE = "#27304d"
P140_GREEN = "#226b45"
P140_TEXT = "#fff4fb"
P140_MUTED = "#c8a7d0"


def v140_walk(widget):
    out = [widget]
    try:
        for child in widget.winfo_children():
            out.extend(v140_walk(child))
    except Exception:
        pass
    return out


def v140_text(widget):
    try:
        return str(widget.cget("text")).strip()
    except Exception:
        return ""


def v140_hide(widget):
    try:
        manager = widget.winfo_manager()
    except Exception:
        manager = ""

    try:
        if manager == "pack":
            widget.pack_forget()
        elif manager == "grid":
            widget.grid_forget()
        elif manager == "place":
            widget.place_forget()
        else:
            try:
                widget.pack_forget()
            except Exception:
                try:
                    widget.grid_forget()
                except Exception:
                    try:
                        widget.place_forget()
                    except Exception:
                        pass
    except Exception:
        pass


def v140_contains(parent, child):
    cur = child
    while cur is not None:
        if cur == parent:
            return True
        try:
            cur = cur.master
        except Exception:
            cur = None
    return False


def v140_canvas(self):
    canvas = getattr(self, "canvas", None)
    if canvas is not None:
        return canvas

    best = None
    best_area = 0

    for widget in v140_walk(self.root):
        try:
            if isinstance(widget, tk.Canvas):
                area = widget.winfo_width() * widget.winfo_height()
                if area > best_area:
                    best = widget
                    best_area = area
        except Exception:
            pass

    return best


def v140_status(self, text):
    try:
        if hasattr(self, "set_status"):
            self.set_status(text)
        elif hasattr(self, "output_label"):
            self.output_label.config(text=text)
    except Exception:
        pass


def v140_capture_old_commands(self):
    self._v140_old_commands = {}

    mapping = {
        "generate candidate": "generate",
        "loop / space": "loop",
        "export wav": "export",
        "export to wav": "export",
    }

    for widget in v140_walk(self.root):
        low = v140_text(widget).lower()
        if low not in mapping:
            continue

        try:
            cmd = widget.cget("command")
        except Exception:
            cmd = ""

        if cmd:
            self._v140_old_commands[mapping[low]] = cmd
            print(f"[v140] commande capturée : {low} -> {mapping[low]}")


def v140_call_old_command(self, key):
    try:
        cmd = getattr(self, "_v140_old_commands", {}).get(key)
        if cmd:
            self.root.tk.call(cmd)
            return True
    except Exception as exc:
        print(f"[v140] ancienne commande {key} impossible :", exc)

    return False


def v140_new_groove(self, event=None):
    fn = globals().get("v131_generate_candidate")
    if callable(fn):
        return fn(self, event)

    fn = getattr(self, "generate_candidate", None)
    if callable(fn):
        return fn(event)

    if v140_call_old_command(self, "generate"):
        return "break"

    v140_status(self, "Nouveau groove introuvable.")
    return "break"


def v140_export(self, event=None):
    if v140_call_old_command(self, "export"):
        return "break"

    for name in [
        "export_wav_current_v122",
        "export_wav_current_v121",
        "export_wav_current_v120",
        "export_wav_current_v118",
        "export_wav_v113",
        "export_wav_current_v110",
    ]:
        fn = getattr(self, name, None)
        if callable(fn):
            try:
                print(f"[v140] export via méthode {name}")
                return fn(event)
            except TypeError:
                try:
                    return fn()
                except Exception:
                    pass
            except Exception as exc:
                print(f"[v140] {name} erreur :", exc)

    v140_status(self, "Export introuvable.")
    print("[v140] ERREUR : aucun moteur export trouvé.")
    return "break"


def v140_loop(self, event=None):
    if v140_call_old_command(self, "loop"):
        return "break"

    for name in ["space_play_v110", "space_play_v109", "space_play_v108"]:
        fn = getattr(self, name, None)
        if callable(fn):
            try:
                return fn(event)
            except Exception:
                pass

    v140_status(self, "Loop introuvable.")
    return "break"


def v140_mirror_32(self, event=None):
    pattern = list(getattr(self, "pattern", []) or [])
    first = []

    for note in pattern:
        try:
            if int(note.get("x_step", 0)) < 32:
                first.append(dict(note))
        except Exception:
            pass

    if not first:
        v140_status(self, "Aucun premier 32 à dupliquer.")
        return "break"

    new_pattern = []

    for note in first:
        new_pattern.append(dict(note))

    for note in first:
        copy = dict(note)
        copy["x_step"] = int(copy.get("x_step", 0)) + 32
        copy["v140_mirror_32"] = True
        new_pattern.append(copy)

    new_pattern = sorted(new_pattern, key=lambda n: (int(n.get("x_step", 0)), int(n.get("lane", 0))))

    for i, note in enumerate(new_pattern):
        note["id"] = i

    self.pattern = new_pattern
    self.selected_id = 0 if self.pattern else None

    try:
        self.draw()
    except Exception:
        pass

    try:
        self.refresh_panel()
    except Exception:
        pass

    try:
        self.write_latest_pattern(reason="v140_mirror_32")
    except Exception:
        pass

    v140_status(self, "Miroir 32 appliqué.")
    return "break"


def v140_button(parent, text, command, bg):
    return tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=P140_TEXT,
        activebackground=bg,
        activeforeground="#ffffff",
        relief="flat",
        bd=0,
        padx=12,
        pady=6,
        font=("Sans", 9, "bold"),
        cursor="hand2",
        highlightthickness=1,
        highlightbackground="#9d5aa8",
    )


def v140_install_toolbar(self):
    old = getattr(self, "v140_toolbar", None)

    try:
        if old is not None and old.winfo_exists():
            old.lift()
            return
    except Exception:
        pass

    toolbar = tk.Frame(
        self.root,
        bg=P140_PANEL,
        bd=0,
        highlightthickness=1,
        highlightbackground="#704078",
    )

    titlebox = tk.Frame(toolbar, bg=P140_PANEL)
    titlebox.pack(side="left", padx=10, pady=5)

    tk.Label(
        titlebox,
        text="♡ Princess Breakbeat",
        bg=P140_PANEL,
        fg=P140_PINK,
        font=("Sans", 11, "bold"),
    ).pack(side="top", anchor="w")

    tk.Label(
        titlebox,
        text="tracker clean",
        bg=P140_PANEL,
        fg=P140_MUTED,
        font=("Sans", 8),
    ).pack(side="top", anchor="w")

    buttons = tk.Frame(toolbar, bg=P140_PANEL)
    buttons.pack(side="left", padx=6, pady=6)

    v140_button(buttons, "Nouveau groove", self.v140_new_groove, P140_PINK_DARK).pack(side="left", padx=4)
    v140_button(buttons, "Miroir 32", self.v140_mirror_32, P140_PURPLE).pack(side="left", padx=4)
    v140_button(buttons, "Loop / Space", self.v140_loop, P140_BLUE).pack(side="left", padx=4)

    # Pas le texte "Export WAV", pour éviter tout filtre ancien.
    v140_button(buttons, "Exporter .wav", self.v140_export, P140_GREEN).pack(side="left", padx=4)

    # Placement très visible : en haut à droite, au-dessus de la grille.
    toolbar.place(relx=1.0, x=-14, y=28, anchor="ne")
    toolbar.lift()

    self.v140_toolbar = toolbar

    print("[v140] toolbar visible installée.")
    v140_status(self, "v140 : Nouveau groove + Exporter .wav visibles.")


def v140_hide_old_ui(self):
    try:
        self.root.update_idletasks()
    except Exception:
        pass

    canvas = v140_canvas(self)
    if canvas is None:
        return

    try:
        canvas_bottom = canvas.winfo_rooty() + canvas.winfo_height()
    except Exception:
        canvas_bottom = 0

    old_exact = {
        "generate candidate",
        "good après modifs",
        "good sans train",
        "reject / bad",
        "generate ia",
        "train ia reminder",
        "load saved beat",
        "list saved beats",
        "generate ia roles",
        "train roles",
        "generate ia safe",
        "train strict",
        "generate cross-song",
        "save validation",
        "delete",
        "reset",
        "render preview",
        "play slice",
        "export wav",
    }

    old_contains = [
        "training :",
        "training:",
        "v48 :",
        "v50 :",
        "v52 :",
        "v53 :",
        "v58 :",
        "v59 :",
        "v62 :",
        "v63 :",
        "v65 :",
        "v70 :",
        "v71 :",
        "v72 :",
        "v73 :",
        "apprend",
        "entrainement",
        "entraînement",
        "cross-sons",
        "ghosts",
        "exp",
    ]

    hidden = 0

    for widget in v140_walk(self.root):
        if widget == self.root or widget == canvas:
            continue

        try:
            if hasattr(self, "v140_toolbar") and v140_contains(self.v140_toolbar, widget):
                continue
        except Exception:
            pass

        low = v140_text(widget).lower()

        should_hide = False

        if low in old_exact:
            should_hide = True

        if any(part in low for part in old_contains):
            should_hide = True

        try:
            y = widget.winfo_rooty()
            w = widget.winfo_width()
            h = widget.winfo_height()
            if w > 1 and h > 1 and y >= canvas_bottom + 2:
                should_hide = True
        except Exception:
            pass

        if should_hide:
            v140_hide(widget)
            hidden += 1

    try:
        self.root.update_idletasks()
        root_w = self.root.winfo_width()
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        new_h = max(520, int(canvas.winfo_y() + canvas.winfo_height() + 20))
        self.root.geometry(f"{root_w}x{new_h}+{root_x}+{root_y}")
    except Exception:
        pass

    try:
        self.v140_toolbar.lift()
    except Exception:
        pass

    print(f"[v140] vieux widgets cachés : {hidden}")


def v140_focus_grid(self, event=None):
    canvas = v140_canvas(self)

    try:
        if canvas is not None:
            canvas.focus_set()
            canvas.focus_force()
        else:
            self.root.focus_force()
    except Exception:
        pass

    return None


def v140_bind_focus_fix(self):
    for widget in v140_walk(self.root):
        cls = ""
        try:
            cls = widget.winfo_class().lower()
        except Exception:
            pass

        if "combo" not in cls:
            continue

        try:
            widget.bind("<<ComboboxSelected>>", lambda e: self.root.after(80, lambda: v140_focus_grid(self)), add="+")
            widget.bind("<Escape>", lambda e: v140_focus_grid(self), add="+")
        except Exception:
            pass


_old_v140_build_ui = SliceIndexTracker.build_ui


def v140_build_ui(self):
    _old_v140_build_ui(self)

    try:
        self.root.title("BreakbeatAI v140 — Princess clean toolbar")
        self.root.configure(bg=P140_BG)
    except Exception:
        pass

    v140_capture_old_commands(self)

    try:
        v140_install_toolbar(self)
    except Exception as exc:
        print("[v140] toolbar impossible :", exc)

    try:
        v140_hide_old_ui(self)
    except Exception as exc:
        print("[v140] hide old UI impossible :", exc)

    try:
        v140_bind_focus_fix(self)
    except Exception:
        pass

    for delay in [250, 700, 1200, 2000]:
        try:
            self.root.after(delay, lambda: v140_install_toolbar(self))
            self.root.after(delay + 80, lambda: v140_hide_old_ui(self))
        except Exception:
            pass

    try:
        self.root.bind("<F12>", self.v140_new_groove)
        self.root.bind("<Control-e>", self.v140_export)
    except Exception:
        pass


SliceIndexTracker.v140_new_groove = v140_new_groove
SliceIndexTracker.v140_export = v140_export
SliceIndexTracker.v140_loop = v140_loop
SliceIndexTracker.v140_mirror_32 = v140_mirror_32
SliceIndexTracker.export_wav_current_v140 = v140_export

SliceIndexTracker.build_ui = v140_build_ui


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="amen")
    args = parser.parse_args()

    pair_json = find_pair_json(args.source)

    root = tk.Tk()
    SliceIndexTracker(root, pair_json)
    root.mainloop()


if __name__ == "__main__":
    main()
