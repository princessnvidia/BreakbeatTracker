#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BreakbeatAI Tracker Editor v41 — slice index

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

# v41 :
# 1 hit audio = 2 cases visuelles.
# On place donc les hits sur 0, 2, 4, 6...
HIT_LENGTH_STEPS = 2
HIT_SPACING_STEPS = 2
HIT_SLOTS = 16

# v41 : tempo cible.
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


def warp_audio_to_length(audio, target_len):
    """
    v41 : warp exact façon Ableton simplifié.

    Le début de slice vient du slicer/transient,
    mais la durée finale est forcée à target_len samples.

    Ça permet :
    - 1 hit = exactement 2 cases
    - pas de décalage temporel
    - slices trop longues compressées
    - slices trop courtes étirées
    """
    audio = np.asarray(audio, dtype=np.float32)
    target_len = int(target_len)

    if target_len <= 1:
        return audio[:1].copy()

    if len(audio) <= 1:
        return np.zeros(target_len, dtype=np.float32)

    if len(audio) == target_len:
        return audio.copy()

    old_x = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    new_x = np.linspace(0.0, 1.0, target_len, endpoint=False)

    warped = np.interp(new_x, old_x, audio).astype(np.float32)

    # mini fade anti-clic, très court
    fade_len = min(64, target_len // 8)

    if fade_len > 2:
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        warped[:fade_len] *= fade_in
        warped[-fade_len:] *= fade_out

    return warped.astype(np.float32)


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

        self.step_count = 32
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

        self.root.title("BreakbeatAI Tracker Editor v41 — slice index")
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
                print(f"[v41] Backend audio : {name} -> {path}")
                return name

        print("[v41] Aucun backend système trouvé, fallback sounddevice.")
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
        v41 : tempo fixe à 155 BPM.
        4 cases = 1 temps, donc step_ms = 60000 / (155 * 4).
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

        print(
            f"[v41] Génération loop32 | break={self.project['safe']} | "
            f"storage={self.project.get('storage')} | method={self.project.get('slice_method')} | "
            f"randomize={randomize} | hit_length={HIT_LENGTH_STEPS}"
        )

        # v41 : 16 hits, chacun occupe 2 cases.
        # Donc positions : 0, 2, 4, 6, ... 30.
        for slot in range(HIT_SLOTS):
            x_step = slot * HIT_SPACING_STEPS

            pair = self.pick_pair_for_pos(slot, randomize=randomize, rng=rng)
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

            print(f"[v41] slot {slot:02d} | step {x_step:02d} -> slice {pair:02d}")

        return pattern

    def build_ui(self):
        main = tk.Frame(self.root, bg="#111018")
        main.pack(fill="both", expand=True, padx=14, pady=14)

        title = tk.Label(
            main,
            text="BreakbeatAI v41 — 155 BPM",
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
        tk.Checkbutton(panel, text="warp exact", variable=self.warp_var, bg="#1b1824", fg="#f5eefe", selectcolor="#30283f", activebackground="#1b1824", activeforeground="#f5eefe").grid(row=1, column=7, padx=4)

        tk.Button(panel, text="Play slice", command=self.play_selected_pair, bg="#30283f", fg="#f5eefe").grid(row=1, column=8, padx=4)
        tk.Button(panel, text="Loop / Space", command=self.toggle_loop, bg="#30283f", fg="#f5eefe").grid(row=1, column=9, padx=4)
        tk.Button(panel, text="Render preview", command=self.render_preview_only, bg="#30283f", fg="#f5eefe").grid(row=1, column=10, padx=4)
        tk.Button(panel, text="Save validation", command=self.save, bg="#30513f", fg="#f5eefe").grid(row=1, column=11, padx=4)
        tk.Button(panel, text="Delete", command=self.delete_selected, bg="#4a2630", fg="#f5eefe").grid(row=1, column=12, padx=4)
        tk.Button(panel, text="Reset", command=self.reset, bg="#30283f", fg="#f5eefe").grid(row=1, column=13, padx=4)

        self.output_label = tk.Label(
            panel,
            text="v41 : tempo par défaut 155 BPM ; clic case = son immédiat ; fermeture = stop audio.",
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
        v41 : les hits font 2 cases.
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
            "version": "breakbeatai_slice_index_correction_v41",
            "time": now_stamp(),
            "safe": safe,
            "event_type": event_type,
            "before": before,
            "after": after,
            "step_ms": self.get_step_ms(),
            "target_bpm": TARGET_BPM,
            "target_step_ms": TARGET_STEP_MS,
        }

        corrections_path = CORRECTIONS_DIR / f"{safe}_slice_index_corrections_v41.jsonl"

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

        print(f"[v41] correction sauvegardée : {event_type}")

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
        v41 : stoppe seulement le petit player d'écoute au clic,
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
        v41 : lecture courte au clic.
        On écrit juste un petit WAV temporaire de preview, puis pw-play le lit.
        Ça ne crée pas de slices permanentes : les slices restent index-only.
        """
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        audio = normalize(audio)
        audition_wav = OUT_DIR / f"{self.project['safe']}_v41_audition.wav"
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
                print(f"[v41] Erreur audition {self.external_player} : {exc}")

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
        v41 : clic sur une case = on joue la slice de cette case.
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
            audio = warp_audio_to_length(audio, target_len)

        self.play_audition_array(audio, label=f"Case {item['id']} / slice {pair}")

    def shutdown_audio(self):
        """
        v41 : arrêt dur de l'audio.
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
        v41 : quand tu fermes la fenêtre, on coupe la loop, le playhead
        et le player externe avant de détruire Tk.
        """
        print("[v41] Fermeture : arrêt audio + destruction fenêtre")

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
        live_wav = OUT_DIR / f"{self.project['safe']}_v41_live.wav"
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
                print(f"[v41] Erreur backend {self.external_player} : {exc}")

        try:
            import sounddevice as sd
            sd.play(audio, SR)
            return duration_ms
        except Exception as exc:
            messagebox.showwarning("Audio", f"Erreur audio : {exc}")
            return 0

    def play_selected_pair(self):
        item = self.selected()

        if item is None:
            return

        audio = self.get_audio(item["pair"])

        if hasattr(self, "warp_var") and self.warp_var.get():
            target_len = max(1, int(item.get("length", HIT_LENGTH_STEPS)) * self.get_step_samples())
            audio = warp_audio_to_length(audio, target_len)

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

            # v41 :
            # warp exact = la slice est stretch/shrink pour remplir exactement la case.
            # clip = simple coupe sans stretch.
            if hasattr(self, "warp_var") and self.warp_var.get():
                audio = warp_audio_to_length(audio, max_len)
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
        wav = OUT_DIR / f"{self.project['safe']}_tracker_app_v41_preview.wav"
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
        print("[v41] Espace détecté")
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

        path = LATEST_DIR / f"{self.project['safe']}_latest_slice_index_v41.json"

        data = {
            "version": "breakbeatai_latest_slice_index_v41",
            "reason": reason,
            "updated_at": now_stamp(),
            "safe": self.project["safe"],
            "source_pair_json": self.project.get("source_pair_json"),
            "source_audio": self.project.get("source_audio"),
            "step_count": self.step_count,
            "hit_length_steps": HIT_LENGTH_STEPS,
            "hit_spacing_steps": HIT_SPACING_STEPS,
            "hit_slots": HIT_SLOTS,
            "step_ms": self.get_step_ms(),
            "target_bpm": TARGET_BPM,
            "target_step_ms": TARGET_STEP_MS,
            "storage": "index_only_preferred",
            "warp_exact": bool(getattr(self, "warp_var", tk.BooleanVar(value=True)).get()),
            "warp_mode": "linear_resample_to_case_length_v41",
            "pattern": self.clean_pattern(),
        }

        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def save(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        VALIDATED_DIR.mkdir(parents=True, exist_ok=True)

        wav = self.render_preview_file()

        tracker_path = OUT_DIR / f"{self.project['safe']}_tracker_app_edit_v41.json"

        data = {
            "version": "tracker_app_edit_v41_slice_index",
            "storage": "index_only_single_source_audio",
            "warp_exact": bool(getattr(self, "warp_var", tk.BooleanVar(value=True)).get()),
            "warp_mode": "linear_resample_to_case_length_v41",
            "learning_memory": str(MEMORY_PATH),
            "step_ms": self.get_step_ms(),
            "target_bpm": TARGET_BPM,
            "target_step_ms": TARGET_STEP_MS,
            "step_count": self.step_count,
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
        validated_path = VALIDATED_DIR / f"{self.project['safe']}_validated_slice_index_loop32_v41_{stamp}.json"

        validated = {
            "version": "breakbeatai_validated_slice_index_loop32_v41",
            "purpose": "human validated loop32 using source audio + slice indexes",
            "source_tracker_json": str(tracker_path),
            "source_audio": self.project["source_audio"],
            "safe": self.project["safe"],
            "step_count": self.step_count,
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
# v41 EXTENSION : statut compact, sans casser la v39
# ---------------------------------------------------------------------

def v41_set_status(self, text):
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


_old_v41_build_ui = SliceIndexTracker.build_ui


def v41_build_ui(self):
    _old_v41_build_ui(self)

    try:
        self.output_label.config(
            text="v41 : warp exact ON | 1 hit = 2 cases | tempo 155 BPM.",
            width=130,
            height=2,
            wraplength=1180,
            anchor="w",
            justify="left",
        )
    except Exception:
        pass


def v41_record_correction(self, event_type, before=None, after=None):
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
        "version": "breakbeatai_slice_index_correction_v41",
        "time": now_stamp(),
        "safe": safe,
        "event_type": event_type,
        "before": before,
        "after": after,
        "step_ms": self.get_step_ms(),
    }

    corrections_path = CORRECTIONS_DIR / f"{safe}_slice_index_corrections_v41.jsonl"

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

    print(f"[v41] correction sauvegardée : {event_type}")
    print(f"[v41] correction path : {corrections_path}")
    print(f"[v41] latest path : {latest_path}")


def v41_randomize_pattern(self):
    self.stop_playhead()
    self.stop_audio()
    self.looping = False

    self.pattern = self.build_initial_pattern(randomize=True)
    self.selected_id = self.pattern[0]["id"] if self.pattern else None

    self.draw()
    self.refresh_panel()

    self.write_latest_pattern(reason="randomize_loop32")
    self.set_status("Randomize OK | mémoire appliquée | latest sauvegardé")


def v41_render_preview_only(self):
    wav = self.render_preview_file()
    self.set_status(f"Preview rendue : {wav.name}")


def v41_switch_break(self, event=None):
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


def v41_refresh_breaks(self):
    self.break_jsons = list_break_jsons()
    values = list(self.break_jsons.keys())
    self.break_box.configure(values=values)

    if values and self.break_var.get() not in values:
        self.break_var.set(values[0])

    self.set_status(f"Breaks détectés : {len(values)}")


# Monkey patch final.
SliceIndexTracker.set_status = v41_set_status
SliceIndexTracker.build_ui = v41_build_ui
SliceIndexTracker.record_correction = v41_record_correction
SliceIndexTracker.randomize_pattern = v41_randomize_pattern
SliceIndexTracker.render_preview_only = v41_render_preview_only
SliceIndexTracker.switch_break = v41_switch_break
SliceIndexTracker.refresh_breaks = v41_refresh_breaks


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
