#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_tracker_editor_app_v12.py

BreakbeatAI Tracker Editor v12

Changements :
- Garde la loop sans latence de v09.
- Garde les 2 lignes par famille :
    HI-HAT A / HI-HAT B
    KICK A   / KICK B
    SNARE A  / SNARE B
- Ajoute une correction manuelle des rôles :
    Pair = HAT
    Pair = KICK
    Pair = SNARE
- Les corrections sont sauvegardées ici :
    dataset/tracker_edits/<safe>_pair_role_overrides.json
- Si une pair est mal classée, par exemple pair 1 entendue comme kick :
    sélectionner une case qui utilise pair 1
    cliquer "Pair = KICK"
"""

from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import soundfile as sf


PAIR_BLOCKS_DIR = Path("dataset/pair_blocks_v02")
OUT_DIR = Path("dataset/tracker_edits")
SR = 44100

LANES = [
    {"key": "hat_a", "role": "hat", "label": "HI-HAT A", "color": "#ffd37a"},
    {"key": "hat_b", "role": "hat", "label": "HI-HAT B", "color": "#ffd37a"},
    {"key": "kick_a", "role": "kick", "label": "KICK A", "color": "#ee8fa7"},
    {"key": "kick_b", "role": "kick", "label": "KICK B", "color": "#ee8fa7"},
    {"key": "snare_a", "role": "snare", "label": "SNARE A", "color": "#8bbcff"},
    {"key": "snare_b", "role": "snare", "label": "SNARE B", "color": "#8bbcff"},
]

ROLE_TO_FIRST_LANE = {
    "hat": 0,
    "kick": 2,
    "snare": 4,
}

LANE_CHOICES = [f"{i}: {lane['label']}" for i, lane in enumerate(LANES)]

DEFAULT_PATTERN = [
    {"id": 0, "x_step": 0,  "lane": 2, "role": "kick",  "length": 2, "pair": 0},
    {"id": 1, "x_step": 2,  "lane": 0, "role": "hat",   "length": 2, "pair": 1},
    {"id": 2, "x_step": 4,  "lane": 4, "role": "snare", "length": 2, "pair": 2},
    {"id": 3, "x_step": 6,  "lane": 1, "role": "hat",   "length": 2, "pair": 3},
    {"id": 4, "x_step": 8,  "lane": 0, "role": "hat",   "length": 2, "pair": 4},
    {"id": 5, "x_step": 10, "lane": 3, "role": "kick",  "length": 2, "pair": 5},
    {"id": 6, "x_step": 12, "lane": 5, "role": "snare", "length": 2, "pair": 6},
    {"id": 7, "x_step": 14, "lane": 1, "role": "hat",   "length": 2, "pair": 7},
]


def find_pair_json(source_query):
    files = sorted(PAIR_BLOCKS_DIR.glob("*_pair_blocks_v02.json"))
    matches = [p for p in files if source_query.lower() in p.name.lower()]

    if not matches:
        print(f"Aucun pair_blocks_v02 JSON trouvé pour : {source_query}")
        print(f'Essaie : python pipeline/01_find_pair_blocks_v03.py --source "{source_query}" --compat-v02')
        sys.exit(1)

    return matches[0]


def safe_name(pair_json):
    return pair_json.stem.replace("_pair_blocks_v02", "")


def normalize(y, peak=0.95):
    if len(y) == 0:
        return y

    m = np.max(np.abs(y))
    if m <= 1e-9:
        return y

    return (y / m * peak).astype(np.float32)


def fade(y, ms=2):
    if len(y) < 16:
        return y.astype(np.float32)

    n = min(int(SR * ms / 1000), len(y) // 4)
    if n <= 1:
        return y.astype(np.float32)

    y = y.astype(np.float32).copy()
    ramp = np.linspace(0, 1, n, dtype=np.float32)
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def load_wav(path):
    audio, sr = sf.read(path, always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    if sr != SR:
        raise RuntimeError(f"Sample rate inattendu {sr} pour {path}, attendu {SR}")

    return fade(normalize(audio), ms=2)


def lane_info(lane_index):
    lane_index = max(0, min(len(LANES) - 1, int(lane_index)))
    return LANES[lane_index]


def clean_role(role):
    role = str(role).lower().strip()
    if role in ("hihat", "hi-hat", "hi_hat", "hat", "hh"):
        return "hat"
    if role in ("kick", "bd", "bassdrum", "bass_drum"):
        return "kick"
    if role in ("snare", "sd", "rim", "clap"):
        return "snare"
    return None


class TrackerEditorApp:
    def __init__(self, root, pair_json):
        self.root = root
        self.pair_json = pair_json
        self.project = self.load_project(pair_json)

        self.blocks = self.project["blocks"]
        self.block_by_pair = {int(b["pair"]): b for b in self.blocks}
        self.pair_values = [int(b["pair"]) for b in self.blocks]

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.role_override_path = OUT_DIR / f"{self.project['safe']}_pair_role_overrides.json"
        self.pair_role_overrides = self.load_pair_role_overrides()

        # v12 : ne plus utiliser le vieux pattern hardcodé qui mettait 1/3/4/7 en HI-HAT.
        # On construit le pattern depuis les rôles réels des pairs.
        self.pattern = self.build_initial_pattern_from_pairs()
        self.sanitize_pattern(apply_pair_roles=True)

        self.selected_id = self.pattern[0]["id"] if self.pattern else None

        self.step_count = 64
        self.left_width = 170
        self.step_width = 18
        self.row_height = 38
        self.case_height = 31
        self.case_y_padding = 3
        self.max_case_length = 8

        self.canvas_width = self.left_width + self.step_width * self.step_count
        self.canvas_height = self.row_height * len(LANES)

        self.drag_mode = None
        self.drag_start_x = 0
        self.drag_start_step = 0
        self.drag_start_len = 0

        self.audio_cache = {}
        self.looping = False
        self.loop_after_id = None
        self.play_process = None
        self.external_player = self.detect_external_player()

        self.loop_target_sec = 180.0
        self.loop_max_repeats = 256

        # v12 : playhead visuelle synchronisée à la boucle.
        self.playhead_after_id = None
        self.loop_started_at = None
        self.current_loop_sec = 0.0
        self.current_timeline = []

        self.root.title("BreakbeatAI Tracker Editor v12")
        self.root.geometry("1500x820")
        self.root.configure(bg="#111018")

        self.build_ui()
        self.draw()
        self.refresh_panel()
        self.bind_keys()

        self.root.after(200, self.force_keyboard_focus)

    def detect_external_player(self):
        for name in ["pw-play", "paplay", "aplay", "ffplay"]:
            path = shutil.which(name)
            if path:
                print(f"[v12] Backend audio système trouvé : {name} -> {path}")
                return name

        print("[v12] Aucun backend système trouvé, fallback sounddevice.")
        return None

    def load_project(self, pair_json):
        meta = json.loads(pair_json.read_text(encoding="utf-8"))

        blocks = []
        for block in meta["blocks"]:
            pair = int(block["pair"])
            role_guess = clean_role(block.get("manual_role") or block.get("role_guess") or "")
            blocks.append({
                "pair": pair,
                "name": f"pair {pair:02d}",
                "audio_path": str(Path(block["audio_path"])),
                "duration_ms": float(block.get("duration_ms", 0.0)),
                "role_guess": role_guess,
                "role_confidence": float(block.get("role_confidence", 0.0) or 0.0),
            })

        blocks = sorted(blocks, key=lambda b: b["pair"])

        return {
            "safe": safe_name(pair_json),
            "source_audio": meta.get("source_audio") or meta.get("source"),
            "source_pair_json": str(pair_json),
            "blocks": blocks,
        }

    def load_pair_role_overrides(self):
        if not self.role_override_path.exists():
            return {}

        try:
            data = json.loads(self.role_override_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        out = {}
        for k, v in data.items():
            role = clean_role(v)
            if role is not None:
                out[str(int(k))] = role
        return out

    def save_pair_role_overrides(self):
        self.role_override_path.write_text(
            json.dumps(self.pair_role_overrides, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def get_pair_role(self, pair):
        pair = int(pair)
        override = self.pair_role_overrides.get(str(pair))
        if override:
            return override

        block = self.block_by_pair.get(pair)
        if block and block.get("role_guess"):
            return block["role_guess"]

        return None

    def role_to_lane(self, role, pair=None):
        role = clean_role(role)
        if role is None:
            return 0

        base = ROLE_TO_FIRST_LANE[role]

        if pair is None:
            return base

        # Alterne A/B pour éviter que toutes les variantes s'empilent sur la même sous-ligne.
        return base + (int(pair) % 2)

    def sanitize_pattern(self, apply_pair_roles=False):
        valid_pairs = {int(b["pair"]) for b in self.blocks}
        fallback_pair = self.pair_values[0] if self.pair_values else 0

        for item in self.pattern:
            item["id"] = int(item.get("id", 0))
            item["x_step"] = int(item.get("x_step", item.get("step", 0)))
            item["length"] = int(item.get("length", 2))
            item["pair"] = int(item.get("pair", fallback_pair))

            if item["pair"] not in valid_pairs:
                item["pair"] = fallback_pair

            if "lane" not in item:
                role = clean_role(item.get("role", "hat")) or "hat"
                item["lane"] = ROLE_TO_FIRST_LANE.get(role, 0)

            item["lane"] = max(0, min(len(LANES) - 1, int(item["lane"])))

            if apply_pair_roles:
                pair_role = self.get_pair_role(item["pair"])
                if pair_role is not None:
                    item["lane"] = self.role_to_lane(pair_role, item["pair"])

            item["role"] = lane_info(item["lane"])["role"]

    def guess_pair_role_from_audio(self, pair):
        """Fallback simple si le slicer n'a pas écrit role_guess."""
        try:
            block = self.block_by_pair.get(int(pair))
            if not block:
                return "hat"

            y = load_wav(block["audio_path"])
            n = min(len(y), int(0.18 * SR))

            if n < 64:
                return "hat"

            seg = y[:n] * np.hanning(n).astype(np.float32)
            mag = np.abs(np.fft.rfft(seg)).astype(np.float32)
            freqs = np.fft.rfftfreq(n, 1 / SR)

            total = float(mag.sum() + 1e-9)
            low = float(mag[freqs < 180].sum() / total)
            high = float(mag[freqs >= 4500].sum() / total)
            centroid = float((freqs * mag).sum() / total)

            if low > 0.30 and centroid < 2600:
                return "kick"

            if high > 0.33 and centroid > 3600:
                return "hat"

            return "snare"

        except Exception:
            return "hat"

    def build_initial_pattern_from_pairs(self):
        """
        Crée le pattern de départ depuis les pairs audio.
        Avant v12, 1/3/4/7 étaient forcés en HI-HAT.
        Maintenant :
          1) override manuel
          2) role_guess du slicer
          3) analyse audio fallback
        """
        pattern = []
        role_counts = {
            "hat": 0,
            "kick": 0,
            "snare": 0,
        }

        # 64 steps / longueur 2 = 32 cases max visibles au départ.
        pairs = list(self.pair_values)[:32]

        for idx, pair in enumerate(pairs):
            pair = int(pair)

            role = self.get_pair_role(pair)
            if role is None:
                role = self.guess_pair_role_from_audio(pair)

            role = clean_role(role) or "hat"

            base_lane = ROLE_TO_FIRST_LANE[role]
            sub_lane = role_counts[role] % 2
            lane = base_lane + sub_lane
            role_counts[role] += 1

            pattern.append({
                "id": idx,
                "x_step": idx * 2,
                "lane": lane,
                "role": role,
                "length": 2,
                "pair": pair,
            })

        if not pattern:
            return json.loads(json.dumps(DEFAULT_PATTERN))

        return pattern

    def apply_pair_roles_to_pattern(self):
        for item in self.pattern:
            pair_role = self.get_pair_role(item["pair"])
            if pair_role is not None:
                item["lane"] = self.role_to_lane(pair_role, item["pair"])
                item["role"] = pair_role

        self.draw()
        self.refresh_panel()

    def build_ui(self):
        main = tk.Frame(self.root, bg="#111018")
        main.pack(fill="both", expand=True, padx=14, pady=14)

        title = tk.Label(
            main,
            text="BreakbeatAI Tracker Editor v12 — auto-rangement des pairs + playhead",
            bg="#111018",
            fg="#ff7acc",
            font=("Sans", 20, "bold"),
        )
        title.pack(anchor="w")

        backend = self.external_player if self.external_player else "sounddevice fallback"
        subtitle = tk.Label(
            main,
            text=(
                f"Backend audio : {backend} | Loop sans latence | "
                "Le pattern est reconstruit depuis les rôles réels des pairs, plus depuis 1/3/4/7 en HI-HAT."
            ),
            bg="#111018",
            fg="#b9acc8",
        )
        subtitle.pack(anchor="w", pady=(0, 10))

        self.canvas = tk.Canvas(
            main,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#202020",
            highlightthickness=1,
            highlightbackground="#41334f",
            takefocus=True,
        )
        self.canvas.pack(fill="x")

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        panel = tk.Frame(main, bg="#1b1824")
        panel.pack(fill="x", pady=(12, 0))

        self.info_label = tk.Label(panel, text="", bg="#1b1824", fg="#f5eefe")
        self.info_label.grid(row=0, column=0, columnspan=14, sticky="w", padx=10, pady=8)

        tk.Label(panel, text="Pair audio", bg="#1b1824", fg="#b9acc8").grid(row=1, column=0, padx=5)

        self.pair_var = tk.IntVar(value=self.pair_values[0] if self.pair_values else 0)
        self.pair_box = ttk.Combobox(
            panel,
            textvariable=self.pair_var,
            values=self.pair_values,
            width=8,
            state="readonly",
        )
        self.pair_box.grid(row=1, column=1, padx=5)
        self.pair_box.bind("<<ComboboxSelected>>", lambda e: self.set_pair(int(self.pair_var.get())))

        tk.Label(panel, text="Ligne", bg="#1b1824", fg="#b9acc8").grid(row=1, column=2, padx=5)

        self.lane_var = tk.StringVar(value=LANE_CHOICES[2])
        self.lane_box = ttk.Combobox(
            panel,
            textvariable=self.lane_var,
            values=LANE_CHOICES,
            width=14,
            state="readonly",
        )
        self.lane_box.grid(row=1, column=3, padx=5)
        self.lane_box.bind("<<ComboboxSelected>>", self.set_lane_from_choice)

        tk.Button(panel, text="Pair = HAT", command=lambda: self.set_selected_pair_role("hat"), bg="#4b3d23", fg="#f5eefe").grid(row=1, column=4, padx=5)
        tk.Button(panel, text="Pair = KICK", command=lambda: self.set_selected_pair_role("kick"), bg="#4a2630", fg="#f5eefe").grid(row=1, column=5, padx=5)
        tk.Button(panel, text="Pair = SNARE", command=lambda: self.set_selected_pair_role("snare"), bg="#263a56", fg="#f5eefe").grid(row=1, column=6, padx=5)

        tk.Button(panel, text="Audio test", command=self.play_test_tone, bg="#4d3d23", fg="#f5eefe").grid(row=1, column=7, padx=5)
        tk.Button(panel, text="Play pair", command=self.play_selected_pair, bg="#30283f", fg="#f5eefe").grid(row=1, column=8, padx=5)
        tk.Button(panel, text="Play Loop / Space", command=self.toggle_loop, bg="#30283f", fg="#f5eefe").grid(row=1, column=9, padx=5)
        tk.Button(panel, text="Render preview", command=self.render_preview_only, bg="#30283f", fg="#f5eefe").grid(row=1, column=10, padx=5)
        tk.Button(panel, text="Save data", command=self.save, bg="#30513f", fg="#f5eefe").grid(row=1, column=11, padx=5)
        tk.Button(panel, text="Delete", command=self.delete_selected, bg="#4a2630", fg="#f5eefe").grid(row=1, column=12, padx=5)
        tk.Button(panel, text="Reset", command=self.reset, bg="#30283f", fg="#f5eefe").grid(row=1, column=13, padx=5)

        self.output_label = tk.Label(
            panel,
            text="Pair 1 pré-corrigée en KICK si override présent. Les overrides restent sauvegardés.",
            bg="#1b1824",
            fg="#77f5b5",
            justify="left",
        )
        self.output_label.grid(row=2, column=0, columnspan=14, sticky="w", padx=10, pady=8)

    def bind_keys(self):
        self.root.bind_all("<space>", self.toggle_loop_event)
        self.root.bind_all("<KeyPress-space>", self.toggle_loop_event)
        self.root.bind_all("<Delete>", lambda e: self.delete_selected())
        self.root.bind_all("<Left>", lambda e: self.move_selected(-1, 0))
        self.root.bind_all("<Right>", lambda e: self.move_selected(1, 0))
        self.root.bind_all("<Up>", lambda e: self.move_selected(0, -1))
        self.root.bind_all("<Down>", lambda e: self.move_selected(0, 1))
        self.root.bind_all("h", lambda e: self.set_selected_pair_role("hat"))
        self.root.bind_all("k", lambda e: self.set_selected_pair_role("kick"))
        self.root.bind_all("s", lambda e: self.set_selected_pair_role("snare"))
        print("[v12] Raccourcis : Espace loop | h/k/s = classer pair sélectionnée")

    def force_keyboard_focus(self):
        try:
            self.root.focus_force()
            self.canvas.focus_set()
        except Exception:
            pass

    def get_audio(self, pair):
        pair = int(pair)

        if pair not in self.audio_cache:
            if pair not in self.block_by_pair:
                raise RuntimeError(f"Pair audio introuvable : {pair}")

            self.audio_cache[pair] = load_wav(self.block_by_pair[pair]["audio_path"])

        return self.audio_cache[pair]

    def stop_audio(self):
        if self.play_process is not None:
            try:
                if self.play_process.poll() is None:
                    self.play_process.terminate()
            except Exception:
                pass
            self.play_process = None

        try:
            import sounddevice as sd
            sd.stop()
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
        live_wav = OUT_DIR / f"{self.project['safe']}_v12_live.wav"
        sf.write(live_wav, audio, SR)

        self.stop_audio()

        duration_ms = int(len(audio) / SR * 1000)

        if self.external_player:
            cmd = self.external_command(live_wav)

            try:
                self.play_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.output_label.config(
                    text=f"{label} lancé avec {self.external_player} — {duration_ms} ms"
                )
                print(f"[v12] {label} lancé avec {self.external_player} : {live_wav}")
                return duration_ms
            except Exception as exc:
                print(f"[v12] Erreur backend système {self.external_player} : {exc}")

        try:
            import sounddevice as sd
            sd.play(audio, SR)
            self.output_label.config(
                text=f"{label} lancé avec sounddevice — {duration_ms} ms"
            )
            print(f"[v12] {label} lancé avec sounddevice")
            return duration_ms
        except Exception as exc:
            self.output_label.config(text=f"Erreur audio : {exc}")
            messagebox.showwarning("Audio", f"Erreur audio : {exc}")
            return 0

    def step_to_x(self, step):
        return self.left_width + step * self.step_width

    def x_to_step(self, x):
        if x < self.left_width:
            return 0

        return max(0, min(self.step_count - 1, int((x - self.left_width) // self.step_width)))

    def y_to_lane(self, y):
        return max(0, min(len(LANES) - 1, int(y // self.row_height)))

    def selected(self):
        for item in self.pattern:
            if item["id"] == self.selected_id:
                return item

        return None

    def new_id(self):
        return max([i["id"] for i in self.pattern], default=-1) + 1

    def draw(self):
        self.canvas.delete("all")

        for lane_index, lane in enumerate(LANES):
            y0 = lane_index * self.row_height
            y1 = y0 + self.row_height
            row_fill = "#252525" if lane_index % 2 == 0 else "#202020"

            self.canvas.create_rectangle(
                0,
                y0,
                self.canvas_width,
                y1,
                fill=row_fill,
                outline="#343434",
            )

            self.canvas.create_rectangle(
                0,
                y0,
                self.left_width,
                y1,
                fill="#17131f",
                outline="#343044",
            )

            self.canvas.create_text(
                12,
                y0 + self.row_height / 2,
                text=lane["label"],
                fill=lane["color"],
                anchor="w",
                font=("Sans", 11, "bold"),
            )

            if lane_index in [1, 3]:
                self.canvas.create_line(
                    0,
                    y1,
                    self.canvas_width,
                    y1,
                    fill="#5b4a6d",
                    width=2,
                )

        for step in range(self.step_count + 1):
            x = self.step_to_x(step)

            if step % 16 == 0:
                color = "#8a8a8a"
                width = 2
            elif step % 8 == 0:
                color = "#6c6c6c"
                width = 1
            elif step % 4 == 0:
                color = "#555555"
                width = 1
            else:
                color = "#393939"
                width = 1

            self.canvas.create_line(x, 0, x, self.canvas_height, fill=color, width=width)

        self.canvas.create_line(
            self.left_width,
            0,
            self.left_width,
            self.canvas_height,
            fill="#888888",
            width=2,
        )

        for item in sorted(self.pattern, key=lambda e: (int(e["x_step"]), int(e["lane"]), int(e["id"]))):
            self.draw_case(item)

    def draw_case(self, item):
        lane_index = max(0, min(len(LANES) - 1, int(item.get("lane", 0))))
        lane = lane_info(lane_index)

        x0 = self.step_to_x(int(item["x_step"]))
        x1 = self.step_to_x(int(item["x_step"]) + int(item["length"]))
        y0 = lane_index * self.row_height + self.case_y_padding
        y1 = y0 + self.case_height

        outline = "#77f5b5" if item["id"] == self.selected_id else "#ffc0cf"
        width = 3 if item["id"] == self.selected_id else 1
        tags = ("case", f"id_{item['id']}")

        self.canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            fill=lane["color"],
            outline=outline,
            width=width,
            tags=tags,
        )

        pair = int(item["pair"])
        role = self.get_pair_role(pair)
        override_mark = "*" if str(pair) in self.pair_role_overrides else ""

        label = f"{pair}{override_mark}"
        if int(item["length"]) >= 3:
            label = f"{lane['label']} {pair}{override_mark}"

        if int(item["length"]) >= 2:
            self.canvas.create_text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                text=label,
                fill="#1a0d14",
                font=("Sans", 8, "bold"),
                tags=tags,
            )

        if item["id"] == self.selected_id:
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

        item_id = self.get_item_at(event.x, event.y)

        if item_id is None:
            if event.x >= self.left_width:
                step = self.x_to_step(event.x)
                lane_index = self.y_to_lane(event.y)
                lane = lane_info(lane_index)
                selected = self.selected()

                if selected is not None:
                    pair = int(selected["pair"])
                else:
                    pair = self.pair_values[0] if self.pair_values else 0

                pair_role = self.get_pair_role(pair)
                if pair_role is not None:
                    lane_index = self.role_to_lane(pair_role, pair)
                    lane = lane_info(lane_index)

                new_item = {
                    "id": self.new_id(),
                    "x_step": step,
                    "lane": lane_index,
                    "role": lane["role"],
                    "length": 2,
                    "pair": pair,
                }

                self.pattern.append(new_item)
                self.selected_id = new_item["id"]

                self.drag_mode = "move"
                self.drag_start_x = event.x
                self.drag_start_step = step
                self.drag_start_len = 2

                self.draw()
                self.refresh_panel()

            return

        self.selected_id = item_id
        item = self.selected()

        x0 = self.step_to_x(item["x_step"])
        x1 = self.step_to_x(item["x_step"] + item["length"])

        if abs(event.x - x0) <= 8:
            self.drag_mode = "resize_left"
        elif abs(event.x - x1) <= 8:
            self.drag_mode = "resize_right"
        else:
            self.drag_mode = "move"

        self.drag_start_x = event.x
        self.drag_start_step = item["x_step"]
        self.drag_start_len = item["length"]

        self.draw()
        self.refresh_panel()

    def on_drag(self, event):
        item = self.selected()

        if item is None or self.drag_mode is None:
            return

        delta_steps = round((event.x - self.drag_start_x) / self.step_width)

        if self.drag_mode == "move":
            item["x_step"] = max(
                0,
                min(self.step_count - item["length"], self.drag_start_step + delta_steps),
            )

            item["lane"] = self.y_to_lane(event.y)
            item["role"] = lane_info(item["lane"])["role"]

        elif self.drag_mode == "resize_right":
            new_len = self.drag_start_len + delta_steps
            new_len = max(1, min(self.max_case_length, new_len))
            item["length"] = min(self.step_count - item["x_step"], new_len)

        elif self.drag_mode == "resize_left":
            old_end = self.drag_start_step + self.drag_start_len
            new_start = self.drag_start_step + delta_steps
            new_start = max(0, min(old_end - 1, new_start))
            new_len = old_end - new_start

            if new_len > self.max_case_length:
                new_start = old_end - self.max_case_length
                new_len = self.max_case_length

            item["x_step"] = new_start
            item["length"] = max(1, new_len)

        self.draw()
        self.refresh_panel()

    def on_release(self, event):
        self.drag_mode = None

    def refresh_panel(self):
        item = self.selected()

        if item is None:
            self.info_label.config(text="Aucun bloc sélectionné")
            return

        lane_index = max(0, min(len(LANES) - 1, int(item.get("lane", 0))))
        lane = lane_info(lane_index)
        item["lane"] = lane_index
        item["role"] = lane["role"]

        pair = int(item["pair"])
        block = self.block_by_pair.get(pair, {})
        guessed = block.get("role_guess") or "?"
        override = self.pair_role_overrides.get(str(pair))
        final_role = self.get_pair_role(pair) or "manuel-ligne"

        override_text = f" | override {override}" if override else ""

        self.info_label.config(
            text=(
                f"id {item['id']} | step {item['x_step']} | length {item['length']} | "
                f"ligne {lane['label']} | pair {pair} | guess {guessed} | rôle final {final_role}{override_text}"
            )
        )

        self.pair_var.set(pair)
        self.lane_var.set(LANE_CHOICES[lane_index])

    def set_pair(self, pair):
        item = self.selected()

        if item is None:
            return

        item["pair"] = int(pair)

        pair_role = self.get_pair_role(pair)
        if pair_role is not None:
            item["lane"] = self.role_to_lane(pair_role, pair)
            item["role"] = pair_role

        self.draw()
        self.refresh_panel()
        self.force_keyboard_focus()

    def set_lane_from_choice(self, event=None):
        choice = self.lane_var.get()
        try:
            lane_index = int(choice.split(":", 1)[0])
        except Exception:
            lane_index = 0
        self.set_lane(lane_index)

    def set_lane(self, lane_index):
        item = self.selected()

        if item is None:
            return

        lane_index = max(0, min(len(LANES) - 1, int(lane_index)))
        item["lane"] = lane_index
        item["role"] = lane_info(lane_index)["role"]

        self.draw()
        self.refresh_panel()
        self.force_keyboard_focus()

    def set_selected_pair_role(self, role):
        role = clean_role(role)
        if role is None:
            return

        item = self.selected()
        if item is None:
            return

        pair = int(item["pair"])
        self.pair_role_overrides[str(pair)] = role
        self.save_pair_role_overrides()

        target_lane = self.role_to_lane(role, pair)

        moved = 0
        for case in self.pattern:
            if int(case["pair"]) == pair:
                case["lane"] = target_lane
                case["role"] = role
                moved += 1

        self.draw()
        self.refresh_panel()
        self.output_label.config(
            text=(
                f"Correction sauvegardée : pair {pair} = {role.upper()} "
                f"→ {lane_info(target_lane)['label']} | {moved} case(s) déplacée(s)."
            )
        )
        print(f"[v12] override pair {pair} = {role} -> {self.role_override_path}")

    def move_selected(self, dx, dy):
        item = self.selected()

        if item is None:
            return

        if dx:
            item["x_step"] = max(
                0,
                min(self.step_count - item["length"], item["x_step"] + dx),
            )

        if dy:
            lane_index = max(0, min(len(LANES) - 1, int(item.get("lane", 0)) + dy))
            item["lane"] = lane_index
            item["role"] = lane_info(lane_index)["role"]

        self.draw()
        self.refresh_panel()

    def delete_selected(self):
        if self.selected_id is None:
            return

        self.pattern = [i for i in self.pattern if i["id"] != self.selected_id]
        self.selected_id = self.pattern[0]["id"] if self.pattern else None

        self.draw()
        self.refresh_panel()

    def reset(self):
        # v12 : ne plus utiliser le vieux pattern hardcodé qui mettait 1/3/4/7 en HI-HAT.
        # On construit le pattern depuis les rôles réels des pairs.
        self.pattern = self.build_initial_pattern_from_pairs()
        self.sanitize_pattern(apply_pair_roles=True)
        self.selected_id = self.pattern[0]["id"] if self.pattern else None

        self.draw()
        self.refresh_panel()

    def play_test_tone(self):
        t = np.linspace(0, 0.45, int(SR * 0.45), endpoint=False)
        audio = 0.35 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        self.play_audio_array(audio, label="Audio test 440Hz")

    def play_selected_pair(self):
        item = self.selected()

        if item is None:
            return

        audio = self.get_audio(item["pair"])
        self.play_audio_array(audio, label=f"Pair {item['pair']}")

    def render_audio_with_timeline(self):
        ordered = sorted(self.pattern, key=lambda e: (int(e["x_step"]), int(e["id"])))

        chunks = []
        timeline = []
        cursor = 0

        for item in ordered:
            audio = self.get_audio(item["pair"])
            chunks.append(audio)

            start_sample = cursor
            end_sample = cursor + len(audio)
            cursor = end_sample

            timeline.append({
                "id": int(item["id"]),
                "pair": int(item["pair"]),
                "x_step": int(item["x_step"]),
                "length": int(item["length"]),
                "lane": int(item["lane"]),
                "start_sec": start_sample / SR,
                "end_sec": end_sample / SR,
            })

        if not chunks:
            return np.zeros(1, dtype=np.float32), []

        return normalize(np.concatenate(chunks)), timeline

    def render_audio(self):
        audio, _timeline = self.render_audio_with_timeline()
        return audio

    def render_preview_file(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        wav = OUT_DIR / f"{self.project['safe']}_tracker_app_v12_preview.wav"
        sf.write(wav, self.render_audio(), SR)
        return wav

    def render_preview_only(self):
        wav = self.render_preview_file()
        self.output_label.config(text=f"Preview : {wav}")

    def playhead_x_for_time(self, t_sec):
        if not self.current_timeline:
            return self.left_width

        # La playhead suit la vraie timeline audio :
        # les cases sont jouées par ordre x_step, et chaque slice a sa propre durée.
        for ev in self.current_timeline:
            start = float(ev["start_sec"])
            end = float(ev["end_sec"])

            if start <= t_sec < end:
                dur = max(1e-9, end - start)
                frac = (t_sec - start) / dur
                visual_step = float(ev["x_step"]) + frac * max(1, int(ev["length"]))
                return self.step_to_x(visual_step)

        last = self.current_timeline[-1]
        return self.step_to_x(int(last["x_step"]) + int(last["length"]))

    def draw_playhead(self, x):
        self.canvas.delete("playhead")

        x = max(self.left_width, min(self.canvas_width, x))

        self.canvas.create_line(
            x,
            0,
            x,
            self.canvas_height,
            fill="#77f5b5",
            width=3,
            tags=("playhead",),
        )

        self.canvas.create_polygon(
            x - 7, 0,
            x + 7, 0,
            x, 13,
            fill="#77f5b5",
            outline="",
            tags=("playhead",),
        )

    def update_playhead(self):
        if not self.looping:
            self.canvas.delete("playhead")
            return

        if self.loop_started_at is None or self.current_loop_sec <= 0:
            return

        elapsed = (time.monotonic() - self.loop_started_at) % self.current_loop_sec
        x = self.playhead_x_for_time(elapsed)
        self.draw_playhead(x)

        self.playhead_after_id = self.root.after(33, self.update_playhead)

    def start_playhead(self, loop_sec, timeline):
        self.stop_playhead(clear=False)

        self.current_loop_sec = float(loop_sec)
        self.current_timeline = list(timeline)
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

    def build_gapless_loop_buffer(self):
        one_loop, timeline = self.render_audio_with_timeline()

        if len(one_loop) <= 1:
            return one_loop, 1, 0, [], 0.0

        one_loop_sec = len(one_loop) / SR
        if one_loop_sec <= 0:
            return one_loop, 1, 0, [], 0.0

        repeats = int(np.ceil(self.loop_target_sec / one_loop_sec))
        repeats = max(2, min(self.loop_max_repeats, repeats))

        long_loop = np.tile(one_loop, repeats).astype(np.float32)
        one_loop_ms = int(one_loop_sec * 1000)

        return long_loop, repeats, one_loop_ms, timeline, one_loop_sec

    def loop_tick(self):
        if not self.looping:
            return

        audio, repeats, one_loop_ms, timeline, one_loop_sec = self.build_gapless_loop_buffer()

        duration_ms = self.play_audio_array(
            audio,
            label=f"Loop gapless x{repeats}"
        )

        if duration_ms <= 0:
            self.looping = False
            self.stop_playhead()
            return

        self.start_playhead(one_loop_sec, timeline)

        self.output_label.config(
            text=(
                f"Loop sans latence + playhead : motif {one_loop_ms} ms répété {repeats}x "
                f"({duration_ms / 1000:.1f}s). Stop/Start pour recharger les changements."
            )
        )

        # Redémarrage rare seulement à la fin du long buffer.
        self.loop_after_id = self.root.after(max(1000, duration_ms + 20), self.loop_tick)

    def toggle_loop_event(self, event=None):
        print("[v12] Espace détecté")
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
            self.output_label.config(text="Loop arrêtée — Espace pour relancer.")
            print("[v12] Loop arrêtée")
            return

        self.stop_playhead()
        print("[v12] Lancement loop...")
        self.looping = True
        self.output_label.config(text="Lancement loop...")
        self.loop_tick()

    def clean_pattern(self):
        out = []

        for item in sorted(self.pattern, key=lambda e: (int(e["x_step"]), int(e["id"]))):
            lane_index = max(0, min(len(LANES) - 1, int(item.get("lane", 0))))
            lane = lane_info(lane_index)
            pair = int(item["pair"])

            out.append({
                "id": int(item["id"]),
                "x_step": int(item["x_step"]),
                "lane": int(lane_index),
                "lane_key": lane["key"],
                "lane_label": lane["label"],
                "role": lane["role"],
                "pair_role": self.get_pair_role(pair),
                "length": int(item["length"]),
                "pair": pair,
            })

        return out

    def save(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        wav = self.render_preview_file()

        data = {
            "version": "tracker_app_edit_v12_auto_pattern_from_pair_roles",
            "audio_rule": "render sorted by x_step only; lane/role/length are visual annotations and do not affect audio",
            "audio_backend": self.external_player if self.external_player else "sounddevice",
            "source_pair_json": self.project["source_pair_json"],
            "source_audio": self.project["source_audio"],
            "safe": self.project["safe"],
            "pair_role_overrides_path": str(self.role_override_path),
            "pair_role_overrides": self.pair_role_overrides,
            "grid": {
                "steps": self.step_count,
                "lanes": [
                    {
                        "lane": i,
                        "key": lane["key"],
                        "role": lane["role"],
                        "name": lane["label"],
                        "color": lane["color"],
                    }
                    for i, lane in enumerate(LANES)
                ],
                "double_lane_per_role": True,
                "same_color_for_role_sublanes": True,
                "manual_pair_role_overrides": True,
                "default_case_length": 2,
                "allow_case_length": [1, 2, 3, 4, 5, 6, 7, 8],
                "pair_is_audio_block_not_visual_row": True,
                "audio_blocks": [
                    {
                        "pair": b["pair"],
                        "name": b["name"],
                        "audio_path": b["audio_path"],
                        "duration_ms": b["duration_ms"],
                        "role_guess": b.get("role_guess"),
                        "role_confidence": b.get("role_confidence"),
                        "role_override": self.pair_role_overrides.get(str(b["pair"])),
                    }
                    for b in self.blocks
                ],
            },
            "pattern": self.clean_pattern(),
            "preview_wav": str(wav),
        }

        path = OUT_DIR / f"{self.project['safe']}_tracker_app_edit_v12.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        self.output_label.config(text=f"OK\nJSON : {path}\nPreview : {wav}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Amen")
    args = parser.parse_args()

    pair_json = find_pair_json(args.source)

    root = tk.Tk()
    TrackerEditorApp(root, pair_json)
    root.mainloop()


if __name__ == "__main__":
    main()
