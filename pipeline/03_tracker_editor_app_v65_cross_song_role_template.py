import random
from collections import Counter
import math
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BreakbeatAI Tracker Editor v65 — slice index

- Un seul fichier audio source par break.
- Les slices sont des index start/end dans le JSON.
- Pas de WAV généré par slice.
- Affichage simple : slice 0, slice 1, slice 2...
- 32 positions.
- Randomize loop32.
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
LOCKED_GRID_STEPS = 32
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
            print(f"[v65] rubberband échec, fallback ffmpeg/no-speed : {exc}")

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
            print(f"[v65] ffmpeg atempo échec, fallback no-speed : {exc}")

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

        self.root.title("BreakbeatAI Tracker Editor v65 — slice index")
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
                print(f"[v65] Backend audio : {name} -> {path}")
                return name

        print("[v65] Aucun backend système trouvé, fallback sounddevice.")
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
            f"[v65] Génération pattern | break={self.project['safe']} | "
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

            print(f"[v65] slot {slot:02d} | step {x_step:02d} -> slice {pair:02d}")

        return pattern

    def build_ui(self):
        main = tk.Frame(self.root, bg="#111018")
        main.pack(fill="both", expand=True, padx=14, pady=14)

        title = tk.Label(
            main,
            text="BreakbeatAI v65 — cross song role template",
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
        tk.Button(break_frame, text="Randomize loop32", command=self.randomize_pattern, bg="#50315f", fg="#f5eefe").pack(side="left", padx=4)
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

        print(f"[v65] correction sauvegardée : {event_type}")

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
        self.output_label.config(text="Reset : loop32 slice-index regénérée.")

    def randomize_pattern(self):
        self.stop_playhead()
        self.stop_audio()
        self.looping = False

        self.pattern = self.build_initial_pattern(randomize=True)
        self.selected_id = self.pattern[0]["id"] if self.pattern else None

        self.draw()
        self.refresh_panel()

        latest = self.write_latest_pattern(reason="randomize_loop32")
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
        audition_wav = OUT_DIR / f"{self.project['safe']}_v65_audition.wav"
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
                print(f"[v65] Erreur audition {self.external_player} : {exc}")

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
        print("[v65] Fermeture : arrêt audio + destruction fenêtre")

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
        live_wav = OUT_DIR / f"{self.project['safe']}_v65_live.wav"
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
                print(f"[v65] Erreur backend {self.external_player} : {exc}")

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
        wav = OUT_DIR / f"{self.project['safe']}_tracker_app_v65_preview.wav"
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
        print("[v65] Espace détecté")
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

        path = LATEST_DIR / f"{self.project['safe']}_latest_slice_index_v65.json"

        data = {
            "version": "breakbeatai_latest_slice_index_v65",
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

        tracker_path = OUT_DIR / f"{self.project['safe']}_tracker_app_edit_v65.json"

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
        validated_path = VALIDATED_DIR / f"{self.project['safe']}_validated_slice_index_loop32_v65_{stamp}.json"

        validated = {
            "version": "breakbeatai_validated_slice_index_loop32_v65",
            "purpose": "human validated loop32 using source audio + slice indexes",
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

    print(f"[v65] correction sauvegardée : {event_type}")
    print(f"[v65] correction path : {corrections_path}")
    print(f"[v65] latest path : {latest_path}")


def v44_randomize_pattern(self):
    self.stop_playhead()
    self.stop_audio()
    self.looping = False

    self.pattern = self.build_initial_pattern(randomize=True)
    self.selected_id = self.pattern[0]["id"] if self.pattern else None

    self.draw()
    self.refresh_panel()

    self.write_latest_pattern(reason="randomize_loop32")
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
            print(f"[v65] audition après flèche haut/bas impossible : {exc}")

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

LOCKED_GRID_STEPS = 32
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
            print(f"[v65] rubberband global warp échec : {exc}")

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
            print(f"[v65] ffmpeg global warp échec : {exc}")

    print("[v65] Aucun warp pitch-preserve dispo, fallback fit no-speed.")
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
        f"[v65] M8 pattern | break={self.project['safe']} | "
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

        print(f"[v65] slot {slot:02d} | step {x_step:02d} -> slice {pair:02d}")

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
        f"[v65] global warp OK | source_len={loop_end-loop_start} | "
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
        print(f"[v65] UI extension impossible : {exc}")


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
        print(f"[v65] lecture annulée : audio silencieux ({label})")
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
        print("[v65] loop annulée : pattern vide")
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
        print("[v65] pattern vide après suppression : audio coupé")

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

    print("[v65] pattern initial forcé VIDE : aucun son ne doit jouer.")
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
    print(f"[v65] Load 16 slices : {len(self.pattern)} items")


def v48_debug_pattern(self, label="render"):
    print("")
    print(f"[v65] DEBUG PATTERN avant {label}")
    print(f"[v65] items = {len(getattr(self, 'pattern', []) or [])}")

    if not getattr(self, "pattern", None):
        print("[v65] pattern vide confirmé")
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
            f"[v65] id={item_id} | step {x:02d}/{end:02d} | "
            f"pair={pair} | lane={lane} | len={length}{flag}"
        )

    print("")


_old_v48_render_audio_with_timeline = SliceIndexTracker.render_audio_with_timeline


def v48_render_audio_with_timeline(self):
    v48_debug_pattern(self, label="render_audio_with_timeline")

    if not getattr(self, "pattern", None):
        loop_samples = self.get_loop_samples()
        print(f"[v65] rendu silence total : pattern vide | samples={loop_samples}")
        return np.zeros(loop_samples, dtype=np.float32), []

    audio, timeline = _old_v48_render_audio_with_timeline(self)

    peak = 0.0
    try:
        if len(audio):
            peak = float(np.max(np.abs(audio)))
    except Exception:
        peak = 0.0

    print(f"[v65] rendu audio peak={peak:.8f} | timeline_events={len(timeline) if timeline is not None else 'None'}")

    if timeline:
        print("[v65] timeline events:")
        for ev in timeline:
            print(f"[v65]   {ev}")

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
        print("[v65] Space ignoré : pattern vide.")
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
    print("[v65] reset => pattern vide")
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
        print(f"[v65] ajout UI impossible : {exc}")

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
        print(f"[v65] record_correction delete impossible : {exc}")

    v49_refresh_after_edit(self, reason)
    v49_status(self, f"Note supprimée : slice {before.get('pair')} | step {before.get('x_step')}")

    print(
        f"[v65] delete | id={before.get('id')} | "
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
        print(f"[v65] record_correction add impossible : {exc}")

    v49_refresh_after_edit(self, "add_note_click")
    v49_status(self, f"Note ajoutée : slice {pair} | step {x_step}")

    try:
        self.audition_selected_case()
    except Exception as exc:
        print(f"[v65] audition après ajout impossible : {exc}")

    print(f"[v65] add | id={item['id']} | step={x_step} | pair={pair} | lane={lane}")

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
        print("[v65] duplicate annulé : aucune note sélectionnée")
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
        print("[v65] duplicate annulé : fin de grille")
        return "break"

    new_item = v49_make_item(self, x_step=new_x, lane=lane, pair=pair)
    new_item["duplicated_from"] = int(item.get("id"))

    self.pattern.append(new_item)
    self.selected_id = int(new_item["id"])

    try:
        self.record_correction("duplicate_note_forward", before=before, after=dict(new_item))
    except Exception as exc:
        print(f"[v65] record_correction duplicate impossible : {exc}")

    v49_refresh_after_edit(self, "duplicate_note_forward")
    v49_status(self, f"Dupliquée : slice {pair} | step {old_x} -> {new_x}")

    try:
        self.audition_selected_case()
    except Exception as exc:
        print(f"[v65] audition après duplicate impossible : {exc}")

    print(
        f"[v65] duplicate | old_id={item.get('id')} | new_id={new_item.get('id')} | "
        f"pair={pair} | {old_x} -> {new_x}"
    )

    return "break"


_old_v49_build_ui = SliceIndexTracker.build_ui


def v49_build_ui(self):
    _old_v49_build_ui(self)

    try:
        self.canvas.bind("<Button-1>", self.v49_canvas_click)
    except Exception as exc:
        print(f"[v65] bind canvas click impossible : {exc}")

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
        print(f"[v65] bind keys impossible : {exc}")

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
        print(f"[v65] impossible de charger le modèle IA : {exc}")
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
        print("[v65] modèle absent :", STYLE_MODEL_PATH)
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
    exploration = 0.18

    print("")
    print(f"[v65] GENERATE IA | safe={safe} | pairs={pair_values} | slots={max_slots}")

    for slot in range(max_slots):
        x_step = slot * int(HIT_SPACING_STEPS)

        weights = []

        for pair in pair_values:
            w = 0.15

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
            f"[v65] slot {slot:02d} | step {x_step:02d} | "
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
        print(f"[v65] write_latest_pattern impossible : {exc}")

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
        print(f"[v65] UI IA impossible : {exc}")

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
        print(f"[v65] record_correction delete impossible : {exc}")

    v51_refresh_after_edit(self, reason)

    v51_status(
        self,
        f"Note supprimée : slice {before.get('pair')} | step {before.get('x_step')}"
    )

    print(
        f"[v65] delete | id={before.get('id')} | "
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
        print(f"[v65] record_correction add impossible : {exc}")

    v51_refresh_after_edit(self, "add_note_click")

    v51_status(self, f"Note ajoutée : slice {pair} | step {x_step}")

    try:
        self.audition_selected_case()
    except Exception as exc:
        print(f"[v65] audition après ajout impossible : {exc}")

    print(f"[v65] add | id={item['id']} | step={x_step} | pair={pair} | lane={lane}")

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
            f"[v65] mouse down note | id={item.get('id')} | "
            f"step={item.get('x_step')} | pair={item.get('pair')}"
        )
    else:
        print("[v65] mouse down empty")

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

        print("[v65] drag depuis vide ignoré")
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
            print(f"[v65] record_correction drag impossible : {exc}")

        v51_refresh_after_edit(self, "move_note_drag")

        v51_status(
            self,
            f"Note déplacée : slice {after.get('pair')} | "
            f"step {before.get('x_step')} -> {after.get('x_step')}"
        )

        print(
            f"[v65] drag move | id={after.get('id')} | "
            f"step {before.get('x_step')} -> {after.get('x_step')} | "
            f"pair {before.get('pair')} -> {after.get('pair')}"
        )

        try:
            self.audition_selected_case()
        except Exception as exc:
            print(f"[v65] audition après drag impossible : {exc}")

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
        print(f"[v65] bind souris impossible : {exc}")

    try:
        self.root.bind("<Delete>", self.delete_selected)
        self.root.bind("<BackSpace>", self.delete_selected)
        self.canvas.bind("<Delete>", self.delete_selected)
        self.canvas.bind("<BackSpace>", self.delete_selected)
    except Exception as exc:
        print(f"[v65] bind Suppr impossible : {exc}")

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
            print(f"[v65] slice ignorée car absente du break actuel : pair={pair}")
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
        print("[v65] aucune sauvegarde trouvée")
        return "break"

    chosen = saves[0]
    pattern = v52_normalize_loaded_notes(self, chosen["notes"])

    if not pattern:
        v52_status(self, "Sauvegarde trouvée, mais aucune note compatible avec les slices actuelles.")
        print("[v65] sauvegarde incompatible :", chosen["path"])
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
        print(f"[v65] write_latest_pattern impossible : {exc}")

    v52_status(
        self,
        f"Beat restauré : {len(pattern)} notes | {chosen['path'].name}"
    )

    print("")
    print("[v65] BEAT RESTAURÉ")
    print("[v65] fichier :", chosen["path"])
    print("[v65] clé :", chosen["key"])
    print("[v65] notes :", len(pattern))
    print("")

    return "break"


def v52_list_saved_beats(self, event=None):
    saves = v52_find_saved_beats(self)

    print("")
    print("[v65] SAUVEGARDES TROUVÉES")
    for i, save in enumerate(saves[:30]):
        print(
            f"[v65] {i:02d} | notes={len(save['notes']):02d} | "
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
        print(f"[v65] UI load saved impossible : {exc}")

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
        print("[v65] trainer introuvable :", trainer)
        return False

    print("")
    print("[v65] AUTO-TRAIN IA après Save...")
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
            print("[v65] auto-train échec code :", result.returncode)
            return False

        v53_status(self, "Save OK + IA entraînée automatiquement.")
        print("[v65] AUTO-TRAIN OK")
        return True

    except Exception as exc:
        v53_status(self, f"Save OK, mais auto-train impossible : {exc}")
        print("[v65] exception auto-train :", exc)
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
        print(f"[v65] auto-train après save impossible : {exc}")

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
        print(f"[v65] UI auto-train impossible : {exc}")

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
        print(f"[v65] record_correction delete impossible : {exc}")

    v55_refresh_after_edit(self, reason)

    v55_status(
        self,
        f"Note supprimée : slice {before.get('pair')} | step {before.get('x_step')}"
    )

    print(
        f"[v65] delete | id={before.get('id')} | "
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
        print(f"[v65] record_correction add impossible : {exc}")

    v55_refresh_after_edit(self, "add_note_click")

    v55_status(self, f"Note ajoutée : slice {pair} | step {x_step}")

    try:
        self.audition_selected_case()
    except Exception as exc:
        print(f"[v65] audition après ajout impossible : {exc}")

    print(f"[v65] add | id={item['id']} | step={x_step} | pair={pair} | lane={lane}")

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
            f"[v65] mouse down note | id={item.get('id')} | "
            f"step={item.get('x_step')} | pair={item.get('pair')}"
        )
    else:
        print("[v65] mouse down empty")

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

        print("[v65] drag depuis vide ignoré")
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
            print(f"[v65] audition clic note impossible : {exc}")

        v55_status(
            self,
            f"Slice {item.get('pair')} jouée | step {item.get('x_step')} | Suppr pour supprimer."
        )

        print(
            f"[v65] click play/select | id={item.get('id')} | "
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
            print(f"[v65] record_correction drag impossible : {exc}")

        v55_refresh_after_edit(self, "move_note_drag")

        v55_status(
            self,
            f"Note déplacée : slice {after.get('pair')} | "
            f"step {before.get('x_step')} -> {after.get('x_step')}"
        )

        print(
            f"[v65] drag move | id={after.get('id')} | "
            f"step {before.get('x_step')} -> {after.get('x_step')} | "
            f"pair {before.get('pair')} -> {after.get('pair')}"
        )

        try:
            self.audition_selected_case()
        except Exception as exc:
            print(f"[v65] audition après drag impossible : {exc}")

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
        print(f"[v65] bind souris impossible : {exc}")

    try:
        self.root.bind("<Delete>", self.delete_selected)
        self.root.bind("<BackSpace>", self.delete_selected)
        self.canvas.bind("<Delete>", self.delete_selected)
        self.canvas.bind("<BackSpace>", self.delete_selected)
    except Exception as exc:
        print(f"[v65] bind Suppr impossible : {exc}")

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
        print(f"[v65] record_correction add impossible : {exc}")

    v58_refresh_after_edit(self, "add_note_fine_grid")

    v58_status(
        self,
        f"Note ajoutée : slice {pair} | case {x_step}"
        + (f"/{x_step + length - 1}" if length > 1 else "")
    )

    try:
        self.audition_selected_case()
    except Exception as exc:
        print(f"[v65] audition après ajout impossible : {exc}")

    print(
        f"[v65] add | id={item['id']} | step={x_step} | "
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
            f"[v65] mouse down note | id={item.get('id')} | "
            f"step={item.get('x_step')} | len={item.get('length')} | pair={item.get('pair')}"
        )
    else:
        print("[v65] mouse down empty")

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

        print("[v65] drag depuis vide ignoré")
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
            print(f"[v65] audition clic note impossible : {exc}")

        v58_status(
            self,
            f"Slice {item.get('pair')} jouée | case {item.get('x_step')} | Suppr/clic droit pour supprimer."
        )

        print(
            f"[v65] click play/select | id={item.get('id')} | "
            f"step={item.get('x_step')} | len={item.get('length')} | pair={item.get('pair')}"
        )

        return "break"

    before = state.get("before") or {}
    after = dict(item)

    if before != after:
        try:
            self.record_correction("move_note_fine_grid", before=before, after=after)
        except Exception as exc:
            print(f"[v65] record_correction drag impossible : {exc}")

        v58_refresh_after_edit(self, "move_note_fine_grid")

        v58_status(
            self,
            f"Note déplacée : slice {after.get('pair')} | "
            f"case {before.get('x_step')} -> {after.get('x_step')}"
        )

        print(
            f"[v65] drag move | id={after.get('id')} | "
            f"step {before.get('x_step')} -> {after.get('x_step')} | "
            f"pair {before.get('pair')} -> {after.get('pair')}"
        )

        try:
            self.audition_selected_case()
        except Exception as exc:
            print(f"[v65] audition après drag impossible : {exc}")

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
        print(f"[v65] record_correction length impossible : {exc}")

    v58_refresh_after_edit(self, "toggle_note_length")

    v58_status(self, f"Longueur note : {old_len} -> {new_len} case(s).")
    print(f"[v65] length toggle | id={item.get('id')} | {old_len} -> {new_len}")

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
        print(f"[v65] record_correction duplicate impossible : {exc}")

    v58_refresh_after_edit(self, "duplicate_note_fine_grid")

    v58_status(
        self,
        f"Dupliquée : case {item.get('x_step')} -> {new_x}"
        + (" | demi-grille" if v58_fine_enabled(self) else "")
    )

    try:
        self.audition_selected_case()
    except Exception as exc:
        print(f"[v65] audition duplicate impossible : {exc}")

    print(
        f"[v65] duplicate | old_id={item.get('id')} | new_id={new_item.get('id')} | "
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
        print(f"[v65] UI demi-grille impossible : {exc}")

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
        print(f"[v65] bind demi-grille impossible : {exc}")

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

    attack = y[:min(len(y), int(SR * 0.100))]
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

    tail_start = min(len(y), int(SR * 0.120))
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

    if scores[role] < 0.12:
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
            print(f"[v65] analyse pair {pair} impossible : {exc}")
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
    print("[v65] TOP ROLES CURRENT BREAK")
    for r in ("kick", "snare", "hat"):
        print(f"[v65] {r:5s}:", [(x["pair"], round(x["score"], 3)) for x in ranks[r][:6]])
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
        print(f"[v65] lecture modèle rôle impossible : {exc}")
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
        print("[v65] modèle rôle absent, fallback musical snare 6.")
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
        print(f"[v65] write_latest_pattern impossible : {exc}")

    print("")
    print("[v65] ROLE-AWARE GENERATION")
    print("[v65] model patterns_used:", model.get("patterns_used", 0))
    for item in pattern:
        print(
            f"[v65] step {item['x_step']:02d}/{item['x_step'] + item['length'] - 1:02d} | "
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
    print("[v65] TRAIN ROLE MODEL...")
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
        print(f"[v65] training rôle impossible : {exc}")
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
        print(f"[v65] auto-training rôle après save impossible : {exc}")

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
        print(f"[v65] UI role-aware impossible : {exc}")

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
        print(f"[v65] record_correction add impossible : {exc}")

    v62_refresh_after_edit(self, "add_note_v62")

    try:
        self.audition_selected_case()
    except Exception as exc:
        print(f"[v65] audition après ajout impossible : {exc}")

    v62_status(
        self,
        f"Note ajoutée : slice {pair} | case {x_step}"
        + (f"/{x_step + length - 1}" if length > 1 else "")
    )

    print(f"[v65] add | id={item['id']} | step={x_step} | len={length} | pair={pair}")

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
            f"[v65] mouse down {state['mode']} | id={item.get('id')} | "
            f"step={item.get('x_step')} | len={item.get('length')} | pair={item.get('pair')}"
        )
    else:
        print("[v65] mouse down empty")

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

        print("[v65] drag depuis vide ignoré")
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
                print(f"[v65] audition clic resize impossible : {exc}")

            v62_status(
                self,
                f"Slice {item.get('pair')} jouée | bord droit = resize | longueur {item.get('length')} case(s)."
            )
            return "break"

        if before != after:
            try:
                self.record_correction("resize_note_edge", before=before, after=after)
            except Exception as exc:
                print(f"[v65] record_correction resize impossible : {exc}")

            v62_refresh_after_edit(self, "resize_note_edge")

            v62_status(
                self,
                f"Note redimensionnée : {before.get('length')} → {after.get('length')} case(s)."
            )

            print(
                f"[v65] resize | id={after.get('id')} | "
                f"step={after.get('x_step')} | len {before.get('length')} -> {after.get('length')}"
            )

            try:
                self.audition_selected_case()
            except Exception as exc:
                print(f"[v65] audition après resize impossible : {exc}")

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
                print(f"[v65] audition clic note impossible : {exc}")

            v62_status(
                self,
                f"Slice {item.get('pair')} jouée | case {item.get('x_step')} | bord droit pour resize."
            )
            return "break"

        if before != after:
            try:
                self.record_correction("move_note_v62", before=before, after=after)
            except Exception as exc:
                print(f"[v65] record_correction move impossible : {exc}")

            v62_refresh_after_edit(self, "move_note_v62")

            v62_status(
                self,
                f"Note déplacée : case {before.get('x_step')} → {after.get('x_step')}"
            )

            print(
                f"[v65] move | id={after.get('id')} | "
                f"step {before.get('x_step')} -> {after.get('x_step')} | "
                f"pair {before.get('pair')} -> {after.get('pair')}"
            )

            try:
                self.audition_selected_case()
            except Exception as exc:
                print(f"[v65] audition après move impossible : {exc}")

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
        print(f"[v65] record_correction toggle length impossible : {exc}")

    v62_refresh_after_edit(self, "toggle_note_length_v62")
    v62_status(self, f"Longueur note : {old_len} → {new_len} case(s).")

    print(f"[v65] toggle length | id={item.get('id')} | {old_len} -> {new_len}")

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
        print(f"[v65] draw handles impossible : {exc}")

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
        print(f"[v65] bind resize impossible : {exc}")

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
        print(f"[v65] UI resize impossible : {exc}")

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
        print(f"[v65] lecture modèle strict impossible : {exc}")
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
        print(f"[v65] write_latest_pattern impossible : {exc}")

    print("")
    print("[v65] SAFE ROLE AI GENERATED")
    print("[v65] model strict:", "yes" if model else "no")
    print("[v65] snare main/alt:", main_snare, alt_snare)
    print("[v65] kick  main/alt:", main_kick, alt_kick)
    print("[v65] hat   main/alt:", main_hat, alt_hat)
    print("[v65] pattern:")
    for item in pattern:
        print(
            f"[v65] step {item['x_step']:02d}/{item['x_step'] + item['length'] - 1:02d} | "
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
    print("[v65] TRAIN STRICT ROLE MODEL...")
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
        print(f"[v65] training strict impossible : {exc}")
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
        print(f"[v65] UI safe impossible : {exc}")

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

    attack = y[:min(len(y), int(SR * 0.100))]
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

    tail_start = min(len(y), int(SR * 0.120))
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
            print(f"[v65] analyse slice {pair} impossible : {exc}")
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
    print("[v65] TOP SLICES PAR ROLE — break courant")
    for role in ("kick", "snare", "hat", "ghost_snare"):
        print(f"[v65] {role:12s}:", [(x["pair"], round(x["score"], 3)) for x in ranks[role][:8]])
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
        print(f"[v65] template introuvable : {exc}")
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
        print(f"[v65] write_latest_pattern impossible : {exc}")

    print("")
    print("[v65] CROSS-SONG ROLE TEMPLATE GENERATED")
    print("[v65] template:", V65_TEMPLATE_PATH)
    print("[v65] temperature:", temperature, "| hats:", hat_density, "| ghosts:", ghost_density)
    print("[v65] pattern:")
    for item in pattern:
        end = int(item["x_step"]) + int(item["length"]) - 1
        print(
            f"[v65] step {item['x_step']:02d}/{end:02d} | "
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
        print(f"[v65] UI impossible : {exc}")

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
