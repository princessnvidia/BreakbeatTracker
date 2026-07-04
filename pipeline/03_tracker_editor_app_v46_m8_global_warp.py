#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BreakbeatAI Tracker Editor v46 — slice index

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
            print(f"[v46] rubberband échec, fallback ffmpeg/no-speed : {exc}")

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
            print(f"[v46] ffmpeg atempo échec, fallback no-speed : {exc}")

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

        self.root.title("BreakbeatAI Tracker Editor v46 — slice index")
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
                print(f"[v46] Backend audio : {name} -> {path}")
                return name

        print("[v46] Aucun backend système trouvé, fallback sounddevice.")
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
            f"[v46] Génération pattern | break={self.project['safe']} | "
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

            print(f"[v46] slot {slot:02d} | step {x_step:02d} -> slice {pair:02d}")

        return pattern

    def build_ui(self):
        main = tk.Frame(self.root, bg="#111018")
        main.pack(fill="both", expand=True, padx=14, pady=14)

        title = tk.Label(
            main,
            text="BreakbeatAI v46 — M8 global warp",
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

        print(f"[v46] correction sauvegardée : {event_type}")

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
        audition_wav = OUT_DIR / f"{self.project['safe']}_v44_audition.wav"
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
                print(f"[v46] Erreur audition {self.external_player} : {exc}")

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
        print("[v46] Fermeture : arrêt audio + destruction fenêtre")

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
        live_wav = OUT_DIR / f"{self.project['safe']}_v46_live.wav"
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
                print(f"[v46] Erreur backend {self.external_player} : {exc}")

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
        wav = OUT_DIR / f"{self.project['safe']}_tracker_app_v46_preview.wav"
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
        print("[v46] Espace détecté")
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

        path = LATEST_DIR / f"{self.project['safe']}_latest_slice_index_v46.json"

        data = {
            "version": "breakbeatai_latest_slice_index_v46",
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

        tracker_path = OUT_DIR / f"{self.project['safe']}_tracker_app_edit_v46.json"

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
        validated_path = VALIDATED_DIR / f"{self.project['safe']}_validated_slice_index_loop32_v46_{stamp}.json"

        validated = {
            "version": "breakbeatai_validated_slice_index_loop32_v46",
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

    print(f"[v46] correction sauvegardée : {event_type}")
    print(f"[v46] correction path : {corrections_path}")
    print(f"[v46] latest path : {latest_path}")


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
            print(f"[v46] audition après flèche haut/bas impossible : {exc}")

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
            print(f"[v46] rubberband global warp échec : {exc}")

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
            print(f"[v46] ffmpeg global warp échec : {exc}")

    print("[v46] Aucun warp pitch-preserve dispo, fallback fit no-speed.")
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
        f"[v46] M8 pattern | break={self.project['safe']} | "
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

        print(f"[v46] slot {slot:02d} | step {x_step:02d} -> slice {pair:02d}")

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
        f"[v46] global warp OK | source_len={loop_end-loop_start} | "
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
        print(f"[v46] UI extension impossible : {exc}")


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
