#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_tracker_editor_app_v21.py

BreakbeatAI Tracker Editor v21

Correction majeure :
- Une ligne = un sample / pair audio.
- Le rendu n'est PLUS une concaténation de samples.
- Le rendu utilise une vraie timeline fixe :
    start_sample = x_step * step_ms
- Changer une case de ligne change le sample joué,
  mais NE décale plus les autres cases.
- Les samples trop longs / mal découpés sont coupés à la longueur visuelle de la case.
- Numéro dans la ligne = sample/pair.
- Numéro dans la case = sample/pair joué.
- Loop gapless + playhead verte.
- Backend audio robuste : pw-play, paplay, aplay, ffplay, puis sounddevice.

Usage :
    cd ~/Applications/BreakbeatAI
    python pipeline/03_tracker_editor_app_v21.py --source "amen"
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

ROLE_COLORS = {
    "hat": "#ffd37a",
    "kick": "#ee8fa7",
    "snare": "#8bbcff",
    "unknown": "#8f8f9d",
}

ROLE_ORDER = {
    "hat": 0,
    "kick": 1,
    "snare": 2,
    "unknown": 3,
    "placer": 3,
    None: 3,
}

PALETTE = [
    "#ffd37a",
    "#ee8fa7",
    "#8bbcff",
    "#77f5b5",
    "#c69cff",
    "#ffb86c",
    "#8be9fd",
    "#f1fa8c",
    "#ff79c6",
    "#bd93f9",
]


def clean_role(role):
    role = str(role).strip().lower()

    aliases = {
        "hihat": "hat",
        "hi-hat": "hat",
        "hi_hat": "hat",
        "hh": "hat",
        "bd": "kick",
        "bassdrum": "kick",
        "bass_drum": "kick",
        "sd": "snare",
        "rim": "snare",
        "clap": "snare",
        "a_classer": "unknown",
        "à classer": "unknown",
        "placer": "unknown",
        "?": "unknown",
        "none": "unknown",
        "": "unknown",
    }

    role = aliases.get(role, role)

    if role not in ("hat", "kick", "snare", "unknown"):
        return "unknown"

    return role


def block_role(block):
    """
    Priorité :
    1. formal_role créé par 02_formalize_break8_roles_v01.py
    2. manual_role
    3. role_guess
    4. unknown
    """
    return clean_role(
        block.get("formal_role")
        or block.get("manual_role")
        or block.get("role_guess")
        or "unknown"
    )


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
    y = np.asarray(y, dtype=np.float32)

    if len(y) == 0:
        return y

    m = float(np.max(np.abs(y)))
    if m <= 1e-9:
        return y

    return (y / m * peak).astype(np.float32)


def fade(y, ms=2):
    y = np.asarray(y, dtype=np.float32)

    if len(y) < 16:
        return y

    n = min(int(SR * ms / 1000), len(y) // 4)
    if n <= 1:
        return y

    out = y.copy()
    ramp = np.linspace(0, 1, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def gate_to_length(y, max_len):
    """
    Coupe un sample à la longueur de la case.
    Ça évite qu'une mauvaise découpe déborde et dégomme le placement.
    """
    y = np.asarray(y, dtype=np.float32)

    if max_len <= 0:
        return np.zeros(1, dtype=np.float32)

    if len(y) <= max_len:
        return y

    out = y[:max_len].copy()

    fade_len = min(int(SR * 0.004), len(out) // 4)
    if fade_len > 2:
        ramp = np.linspace(1, 0, fade_len, dtype=np.float32)
        out[-fade_len:] *= ramp

    return out


def load_wav(path):
    audio, sr = sf.read(path, always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    if sr != SR:
        raise RuntimeError(f"Sample rate inattendu {sr} pour {path}, attendu {SR}")

    return fade(normalize(audio), ms=2)


class TrackerEditorApp:
    def __init__(self, root, pair_json):
        self.root = root
        self.pair_json = pair_json
        self.project = self.load_project(pair_json)

        self.blocks = self.project["blocks"]
        self.block_by_pair = {int(b["pair"]): b for b in self.blocks}
        self.pair_values = [int(b["pair"]) for b in self.blocks]
        self.pair_to_lane = {pair: i for i, pair in enumerate(self.pair_values)}

        self.step_count = 64
        self.left_width = 92
        self.step_width = 18
        self.row_height = 40
        self.case_height = 31
        self.case_y_padding = 4
        self.max_case_length = 8

        self.default_step_ms = self.guess_step_ms()
        self.clip_to_cell = True

        self.visible_canvas_height = min(
            620,
            max(220, self.row_height * max(1, len(self.pair_values))),
        )
        self.canvas_width = self.left_width + self.step_width * self.step_count
        self.canvas_height_total = self.row_height * max(1, len(self.pair_values))

        self.audio_cache = {}
        self.pattern = self.build_initial_pattern()

        self.selected_id = self.pattern[0]["id"] if self.pattern else None

        self.drag_mode = None
        self.drag_start_x = 0
        self.drag_start_step = 0
        self.drag_start_len = 0

        self.looping = False
        self.loop_after_id = None
        self.play_process = None
        self.external_player = self.detect_external_player()

        self.loop_target_sec = 180.0
        self.loop_max_repeats = 256

        self.playhead_after_id = None
        self.loop_started_at = None
        self.current_loop_sec = 0.0

        self.root.title("BreakbeatAI Tracker Editor v21 — rôles groupés + positions formelles")
        self.root.geometry("1480x880")
        self.root.configure(bg="#111018")

        self.build_ui()
        self.draw()
        self.refresh_panel()
        self.bind_keys()

        self.root.after(200, self.force_keyboard_focus)

    def load_project(self, pair_json):
        meta = json.loads(pair_json.read_text(encoding="utf-8"))

        blocks = []
        for block in meta.get("blocks", []):
            pair = int(block["pair"])
            blocks.append({
                "pair": pair,
                "name": f"sample {pair}",
                "audio_path": str(Path(block["audio_path"])),
                "duration_ms": float(block.get("duration_ms", 0.0)),
                "role": block_role(block),
                "formal_role": block.get("formal_role"),
                "formal_position": block.get("formal_position"),
                "formal_position_in_cycle": block.get("formal_position_in_cycle"),
                "role_guess": block.get("role_guess"),
                "manual_role": block.get("manual_role"),
                "role_confidence": block.get("role_confidence"),
            })

        # v21 : ordre des lignes = hats en haut, kicks au milieu, snares en bas.
        blocks = sorted(blocks, key=lambda b: (ROLE_ORDER.get(b.get("role"), 3), int(b["pair"])))

        if not blocks:
            raise RuntimeError(f"Aucun block dans {pair_json}")

        return {
            "safe": safe_name(pair_json),
            "source_audio": meta.get("source_audio") or meta.get("source"),
            "source_pair_json": str(pair_json),
            "blocks": blocks,
        }

    def guess_step_ms(self):
        durations = [
            float(b.get("duration_ms", 0.0))
            for b in self.project["blocks"]
            if float(b.get("duration_ms", 0.0)) > 10
        ]

        if not durations:
            return 90.0

        # Les cases font length=2 au départ, donc un sample médian ≈ 2 steps.
        med = float(np.median(durations))
        step = med / 2.0

        # Bornes raisonnables pour du breakbeat/jungle.
        step = max(45.0, min(180.0, step))
        return round(step, 2)

    def detect_external_player(self):
        for name in ["pw-play", "paplay", "aplay", "ffplay"]:
            path = shutil.which(name)
            if path:
                print(f"[v21] Backend audio système trouvé : {name} -> {path}")
                return name

        print("[v21] Aucun backend système trouvé, fallback sounddevice.")
        return None

    def lane_color(self, lane_index):
        pair = self.lane_to_pair(lane_index)
        role = self.block_by_pair.get(int(pair), {}).get("role", "unknown")
        return ROLE_COLORS.get(role, PALETTE[int(lane_index) % len(PALETTE)])

    def lane_to_pair(self, lane_index):
        lane_index = max(0, min(len(self.pair_values) - 1, int(lane_index)))
        return int(self.pair_values[lane_index])

    def pair_to_color(self, pair):
        lane = self.pair_to_lane.get(int(pair), 0)
        role = self.block_by_pair.get(int(pair), {}).get("role", "unknown")
        return ROLE_COLORS.get(role, self.lane_color(lane))

    def get_step_ms(self):
        try:
            value = float(self.step_ms_var.get())
        except Exception:
            value = self.default_step_ms

        return max(10.0, min(500.0, value))

    def get_step_samples(self):
        return max(1, int(SR * self.get_step_ms() / 1000.0))

    def get_loop_samples(self):
        return int(self.step_count * self.get_step_samples())

    def get_loop_sec(self):
        return self.get_loop_samples() / SR

    def get_pair_formal_position(self, pair):
        """
        Position musicale originale du sample dans le break.
        Elle vient de 02_formalize_break8_roles_v01.py.

        Exemple pattern 8 :
            0 K, 1 H, 2 S, 3 H, 4 H, 5 K, 6 S, 7 H

        Important :
        - les lignes peuvent être triées H/K/S
        - mais x_step doit rester basé sur cette position originale
        """
        pair = int(pair)
        block = self.block_by_pair.get(pair, {})

        for key in ("formal_position", "formal_position_in_cycle"):
            value = block.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                pass

        # Fallback si la formalisation n'a pas encore été appliquée.
        try:
            return int(pair)
        except Exception:
            return 0

    def build_initial_pattern(self):
        """
        v21 :
        - vertical = rôle groupé : hats haut, kicks milieu, snares bas
        - horizontal = position formelle originale du break
        Donc plus d'escalier créé par le tri des lignes.
        """
        pattern = []

        for pair in self.pair_values:
            pair = int(pair)

            lane = self.pair_to_lane.get(pair, 0)
            formal_pos = self.get_pair_formal_position(pair)

            # Chaque position formelle avance de 2 steps :
            # 0->0, 1->2, 2->4, etc.
            x_step = (formal_pos * 2) % self.step_count

            pattern.append({
                "id": len(pattern),
                "x_step": x_step,
                "lane": lane,
                "pair": pair,
                "length": 2,
                "formal_position": formal_pos,
            })

        return pattern

    def build_ui(self):
        main = tk.Frame(self.root, bg="#111018")
        main.pack(fill="both", expand=True, padx=14, pady=14)

        title = tk.Label(
            main,
            text="BreakbeatAI Tracker Editor v21 — HI-HAT haut / KICK milieu / SNARE bas, sans escalier",
            bg="#111018",
            fg="#ff7acc",
            font=("Sans", 20, "bold"),
        )
        title.pack(anchor="w")

        backend = self.external_player if self.external_player else "sounddevice fallback"
        subtitle = tk.Label(
            main,
            text=(
                f"Backend audio : {backend} | Timeline fixe. "
                "Lignes triées par rôle, cases placées selon la position formelle du break."
            ),
            bg="#111018",
            fg="#b9acc8",
        )
        subtitle.pack(anchor="w", pady=(0, 10))

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

        tk.Label(panel, text="Sample / pair", bg="#1b1824", fg="#b9acc8").grid(row=1, column=0, padx=5)

        self.pair_var = tk.IntVar(value=self.pair_values[0])
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

        self.lane_var = tk.IntVar(value=1)
        self.lane_box = ttk.Combobox(
            panel,
            textvariable=self.lane_var,
            values=[i + 1 for i in range(len(self.pair_values))],
            width=6,
            state="readonly",
        )
        self.lane_box.grid(row=1, column=3, padx=5)
        self.lane_box.bind("<<ComboboxSelected>>", self.set_lane_from_choice)

        tk.Label(panel, text="Step ms", bg="#1b1824", fg="#b9acc8").grid(row=1, column=4, padx=5)

        self.step_ms_var = tk.StringVar(value=str(self.default_step_ms))
        self.step_ms_spin = tk.Spinbox(
            panel,
            from_=10,
            to=500,
            increment=1,
            textvariable=self.step_ms_var,
            width=7,
            bg="#30283f",
            fg="#f5eefe",
            insertbackground="#f5eefe",
            command=self.on_step_ms_changed,
        )
        self.step_ms_spin.grid(row=1, column=5, padx=4)
        self.step_ms_spin.bind("<Return>", lambda e: self.on_step_ms_changed())
        self.step_ms_spin.bind("<FocusOut>", lambda e: self.on_step_ms_changed())

        self.clip_var = tk.BooleanVar(value=True)
        self.clip_check = tk.Checkbutton(
            panel,
            text="couper à la case",
            variable=self.clip_var,
            bg="#1b1824",
            fg="#f5eefe",
            selectcolor="#30283f",
            activebackground="#1b1824",
            activeforeground="#f5eefe",
        )
        self.clip_check.grid(row=1, column=6, padx=4)

        tk.Button(panel, text="Play sample", command=self.play_selected_pair, bg="#30283f", fg="#f5eefe").grid(row=1, column=7, padx=4)
        tk.Button(panel, text="Loop / Space", command=self.toggle_loop, bg="#30283f", fg="#f5eefe").grid(row=1, column=8, padx=4)
        tk.Button(panel, text="Render preview", command=self.render_preview_only, bg="#30283f", fg="#f5eefe").grid(row=1, column=9, padx=4)
        tk.Button(panel, text="Save", command=self.save, bg="#30513f", fg="#f5eefe").grid(row=1, column=10, padx=4)
        tk.Button(panel, text="Delete", command=self.delete_selected, bg="#4a2630", fg="#f5eefe").grid(row=1, column=11, padx=4)
        tk.Button(panel, text="Reset", command=self.reset, bg="#30283f", fg="#f5eefe").grid(row=1, column=12, padx=4)

        self.output_label = tk.Label(
            panel,
            text=(
                "v21 : hats en haut, kicks au milieu, snares en bas. "
                "Timeline fixe : changer de sample ne décale pas le pattern."
            ),
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

        print("[v21] Raccourcis : Espace loop | flèches = déplacer | Delete = supprimer")

    def force_keyboard_focus(self):
        try:
            self.root.focus_force()
            self.canvas.focus_set()
        except Exception:
            pass

    def on_step_ms_changed(self):
        self.output_label.config(
            text=(
                f"Step = {self.get_step_ms():.2f} ms. "
                "Stop/Start la loop pour entendre le nouveau timing."
            )
        )

    def on_mousewheel(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(3, "units")
        else:
            direction = -1 if event.delta > 0 else 1
            self.canvas.yview_scroll(direction * 3, "units")

        return "break"

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
        live_wav = OUT_DIR / f"{self.project['safe']}_v21_live.wav"
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
                self.output_label.config(text=f"{label} lancé avec {self.external_player} — {duration_ms} ms")
                print(f"[v21] {label} lancé avec {self.external_player} : {live_wav}")
                return duration_ms
            except Exception as exc:
                print(f"[v21] Erreur backend système {self.external_player} : {exc}")

        try:
            import sounddevice as sd
            sd.play(audio, SR)
            self.output_label.config(text=f"{label} lancé avec sounddevice — {duration_ms} ms")
            print(f"[v21] {label} lancé avec sounddevice")
            return duration_ms
        except Exception as exc:
            self.output_label.config(text=f"Erreur audio : {exc}")
            messagebox.showwarning("Audio", f"Erreur audio : {exc}")
            return 0

    def step_to_x(self, step):
        return self.left_width + float(step) * self.step_width

    def x_to_step(self, x):
        if x < self.left_width:
            return 0

        return max(0, min(self.step_count - 1, int((x - self.left_width) // self.step_width)))

    def y_to_lane(self, y):
        return max(0, min(len(self.pair_values) - 1, int(y // self.row_height)))

    def selected(self):
        for item in self.pattern:
            if int(item["id"]) == int(self.selected_id):
                return item

        return None

    def new_id(self):
        return max([int(i["id"]) for i in self.pattern], default=-1) + 1

    def draw(self):
        self.canvas.delete("all")
        self.canvas.configure(scrollregion=(0, 0, self.canvas_width, self.canvas_height_total))

        for lane_index, pair in enumerate(self.pair_values):
            y0 = lane_index * self.row_height
            y1 = y0 + self.row_height
            color = self.lane_color(lane_index)
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

            box_size = 30
            box_x0 = 16
            box_y0 = y0 + (self.row_height - box_size) / 2
            box_x1 = box_x0 + box_size
            box_y1 = box_y0 + box_size

            self.canvas.create_rectangle(
                box_x0,
                box_y0,
                box_x1,
                box_y1,
                fill=color,
                outline="#f5eefe",
                width=1,
            )

            self.canvas.create_text(
                (box_x0 + box_x1) / 2,
                (box_y0 + box_y1) / 2,
                text=str(pair),
                fill="#1a0d14",
                font=("Sans", 11, "bold"),
            )

            role = self.block_by_pair.get(int(pair), {}).get("role", "unknown")
            role_letter = {"hat": "H", "kick": "K", "snare": "S", "unknown": "?"}.get(role, "?")
            self.canvas.create_text(
                box_x1 + 18,
                (box_y0 + box_y1) / 2,
                text=role_letter,
                fill="#f5eefe",
                font=("Sans", 10, "bold"),
            )

        # Séparateurs de groupes : hat / kick / snare / unknown.
        previous_role = None
        for lane_index, pair in enumerate(self.pair_values):
            role = self.block_by_pair.get(int(pair), {}).get("role", "unknown")
            if previous_role is not None and role != previous_role:
                y = lane_index * self.row_height
                self.canvas.create_line(
                    0,
                    y,
                    self.canvas_width,
                    y,
                    fill="#ff7acc",
                    width=3,
                )
            previous_role = role

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

            self.canvas.create_line(x, 0, x, self.canvas_height_total, fill=color, width=width)

        self.canvas.create_line(
            self.left_width,
            0,
            self.left_width,
            self.canvas_height_total,
            fill="#888888",
            width=2,
        )

        for item in sorted(self.pattern, key=lambda e: (int(e["x_step"]), int(e["lane"]), int(e["id"]))):
            self.draw_case(item)

    def draw_case(self, item):
        lane_index = max(0, min(len(self.pair_values) - 1, int(item.get("lane", 0))))
        pair = self.lane_to_pair(lane_index)
        color = self.lane_color(lane_index)

        item["pair"] = pair

        x0 = self.step_to_x(int(item["x_step"]))
        x1 = self.step_to_x(int(item["x_step"]) + int(item["length"]))
        y0 = lane_index * self.row_height + self.case_y_padding
        y1 = y0 + self.case_height

        outline = "#77f5b5" if int(item["id"]) == int(self.selected_id) else "#ffc0cf"
        width = 3 if int(item["id"]) == int(self.selected_id) else 1
        tags = ("case", f"id_{item['id']}")

        self.canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            fill=color,
            outline=outline,
            width=width,
            tags=tags,
        )

        self.canvas.create_text(
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            text=str(pair),
            fill="#1a0d14",
            font=("Sans", 9, "bold"),
            tags=tags,
        )

        if int(item["id"]) == int(self.selected_id):
            self.canvas.create_rectangle(x0, y0, x0 + 5, y1, fill="#77f5b5", outline="", tags=tags)
            self.canvas.create_rectangle(x1 - 5, y0, x1, y1, fill="#77f5b5", outline="", tags=tags)

    def lane_to_pair(self, lane_index):
        lane_index = max(0, min(len(self.pair_values) - 1, int(lane_index)))
        return int(self.pair_values[lane_index])

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
                    "length": 2,
                }

                self.pattern.append(new_item)
                self.selected_id = new_item["id"]

                self.drag_mode = "move"
                self.drag_start_x = x
                self.drag_start_step = step
                self.drag_start_len = 2

                self.draw()
                self.refresh_panel()

            return

        self.selected_id = item_id
        item = self.selected()

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

    def on_drag(self, event):
        item = self.selected()

        if item is None or self.drag_mode is None:
            return

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        delta_steps = round((x - self.drag_start_x) / self.step_width)

        if self.drag_mode == "move":
            item["x_step"] = max(
                0,
                min(self.step_count - item["length"], self.drag_start_step + delta_steps),
            )

            lane_index = self.y_to_lane(y)
            item["lane"] = lane_index
            item["pair"] = self.lane_to_pair(lane_index)

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
            self.info_label.config(text="Aucune case sélectionnée")
            return

        lane_index = max(0, min(len(self.pair_values) - 1, int(item.get("lane", 0))))
        pair = self.lane_to_pair(lane_index)
        item["lane"] = lane_index
        item["pair"] = pair

        block = self.block_by_pair.get(pair, {})
        duration = block.get("duration_ms", 0.0)
        cell_ms = int(item["length"]) * self.get_step_ms()

        role = block.get("role", "unknown")
        formal_pos = self.get_pair_formal_position(pair)
        self.info_label.config(
            text=(
                f"case {item['id']} | sample/pair {pair} | rôle {role} | formal_pos {formal_pos} | "
                f"ligne {lane_index + 1}/{len(self.pair_values)} | "
                f"step {item['x_step']} | length {item['length']} | cellule {cell_ms:.1f} ms | "
                f"sample source {duration:.1f} ms"
            )
        )

        self.pair_var.set(pair)
        self.lane_var.set(lane_index + 1)

    def set_pair(self, pair):
        item = self.selected()

        if item is None:
            return

        pair = int(pair)
        lane = self.pair_to_lane.get(pair, 0)

        item["pair"] = pair
        item["lane"] = lane

        self.draw()
        self.refresh_panel()
        self.force_keyboard_focus()

    def set_lane_from_choice(self, event=None):
        item = self.selected()

        if item is None:
            return

        try:
            lane_index = int(self.lane_var.get()) - 1
        except Exception:
            lane_index = 0

        lane_index = max(0, min(len(self.pair_values) - 1, lane_index))
        item["lane"] = lane_index
        item["pair"] = self.lane_to_pair(lane_index)

        self.draw()
        self.refresh_panel()
        self.force_keyboard_focus()

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
            lane_index = max(0, min(len(self.pair_values) - 1, int(item.get("lane", 0)) + dy))
            item["lane"] = lane_index
            item["pair"] = self.lane_to_pair(lane_index)

        self.draw()
        self.refresh_panel()

    def delete_selected(self):
        if self.selected_id is None:
            return

        self.pattern = [i for i in self.pattern if int(i["id"]) != int(self.selected_id)]
        self.selected_id = self.pattern[0]["id"] if self.pattern else None

        self.draw()
        self.refresh_panel()

    def reset(self):
        self.stop_playhead()
        self.stop_audio()
        self.looping = False

        self.pattern = self.build_initial_pattern()
        self.selected_id = self.pattern[0]["id"] if self.pattern else None

        self.draw()
        self.refresh_panel()
        self.output_label.config(text="Reset : une case par sample, chaque case sur sa propre ligne.")

    def play_selected_pair(self):
        item = self.selected()

        if item is None:
            return

        audio = self.get_audio(item["pair"])
        self.play_audio_array(audio, label=f"Sample {item['pair']}")

    def render_audio_with_timeline(self):
        """
        v21 : rendu timeline fixe.

        Avant :
            audio = sample_a + sample_b + sample_c
            donc les longueurs des samples déplaçaient tout.

        Maintenant :
            start = x_step * step_samples
            audio[start:end] += sample
            donc changer de sample ne change jamais la position des autres cases.
        """
        step_samples = self.get_step_samples()
        loop_samples = self.get_loop_samples()

        out = np.zeros(loop_samples, dtype=np.float32)
        timeline = []

        ordered = sorted(self.pattern, key=lambda e: (int(e["x_step"]), int(e["id"])))

        for item in ordered:
            lane_index = max(0, min(len(self.pair_values) - 1, int(item.get("lane", 0))))
            pair = self.lane_to_pair(lane_index)
            item["pair"] = pair

            start = int(item["x_step"]) * step_samples
            cell_len = max(1, int(item["length"]) * step_samples)

            if start >= loop_samples:
                continue

            audio = self.get_audio(pair)

            if self.clip_var.get():
                audio = gate_to_length(audio, cell_len)

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
                "cell_end_sec": (start + cell_len) / SR,
            })

        return normalize(out), timeline

    def render_audio(self):
        audio, _timeline = self.render_audio_with_timeline()
        return audio

    def render_preview_file(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        wav = OUT_DIR / f"{self.project['safe']}_tracker_app_v21_preview.wav"
        sf.write(wav, self.render_audio(), SR)
        return wav

    def render_preview_only(self):
        wav = self.render_preview_file()
        self.output_label.config(text=f"Preview : {wav}")

    def draw_playhead(self, x):
        self.canvas.delete("playhead")

        x = max(self.left_width, min(self.canvas_width, x))

        self.canvas.create_line(
            x,
            0,
            x,
            self.canvas_height_total,
            fill="#77f5b5",
            width=3,
            tags=("playhead",),
        )

        self.canvas.create_polygon(
            x - 7,
            0,
            x + 7,
            0,
            x,
            13,
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

    def build_gapless_loop_buffer(self):
        one_loop, _timeline = self.render_audio_with_timeline()

        if len(one_loop) <= 1:
            return one_loop, 1, 0, 0.0

        one_loop_sec = len(one_loop) / SR
        if one_loop_sec <= 0:
            return one_loop, 1, 0, 0.0

        repeats = int(np.ceil(self.loop_target_sec / one_loop_sec))
        repeats = max(2, min(self.loop_max_repeats, repeats))

        long_loop = np.tile(one_loop, repeats).astype(np.float32)
        one_loop_ms = int(one_loop_sec * 1000)

        return long_loop, repeats, one_loop_ms, one_loop_sec

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
            text=(
                f"Loop timeline fixe : motif {one_loop_ms} ms répété {repeats}x. "
                f"Step={self.get_step_ms():.2f} ms. Changer de sample ne décale plus."
            )
        )
        self.loop_after_id = self.root.after(max(1000, duration_ms + 20), self.loop_tick)

    def toggle_loop_event(self, event=None):
        print("[v21] Espace détecté")
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
            print("[v21] Loop arrêtée")
            return

        self.stop_playhead()
        print("[v21] Lancement loop...")
        self.looping = True
        self.output_label.config(text="Lancement loop...")
        self.loop_tick()

    def clean_pattern(self):
        out = []

        for item in sorted(self.pattern, key=lambda e: (int(e["x_step"]), int(e["id"]))):
            lane_index = max(0, min(len(self.pair_values) - 1, int(item.get("lane", 0))))
            pair = self.lane_to_pair(lane_index)

            out.append({
                "id": int(item["id"]),
                "x_step": int(item["x_step"]),
                "lane": int(lane_index),
                "line_display": str(pair),
                "pair": int(pair),
                "role": self.block_by_pair[pair].get("role", "unknown"),
                "formal_position": self.get_pair_formal_position(pair),
                "length": int(item["length"]),
                "audio_path": self.block_by_pair[pair]["audio_path"],
                "timeline_start_sec": round(int(item["x_step"]) * self.get_step_samples() / SR, 6),
                "cell_duration_sec": round(int(item["length"]) * self.get_step_samples() / SR, 6),
            })

        return out

    def save(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        wav = self.render_preview_file()

        data = {
            "version": "tracker_app_edit_v21_fixed_timeline_one_line_per_sample",
            "audio_rule": "render by fixed timeline: start = x_step * step_ms; samples are mixed into the timeline, not concatenated",
            "ui_rule": "one row per audio sample/pair, grouped by formal role: hats top, kicks middle, snares bottom",
            "placement_rule": "changing a case sample/lane never changes other cases positions",
            "lane_sort_rule": "sorted by role order: hat, kick, snare, unknown; then pair number",
            "initial_placement_rule": "x_step = formal_position * 2; vertical lane = grouped role",
            "clip_to_cell": bool(self.clip_var.get()),
            "step_ms": self.get_step_ms(),
            "loop_duration_sec": round(self.get_loop_sec(), 6),
            "audio_backend": self.external_player if self.external_player else "sounddevice",
            "source_pair_json": self.project["source_pair_json"],
            "source_audio": self.project["source_audio"],
            "safe": self.project["safe"],
            "grid": {
                "steps": self.step_count,
                "one_line_per_sample": True,
                "fixed_timeline": True,
                "visible_box_numbers": True,
                "lanes": [
                    {
                        "lane": i,
                        "pair": int(pair),
                        "line_display": str(pair),
                        "role": self.block_by_pair[int(pair)].get("role", "unknown"),
                        "audio_path": self.block_by_pair[int(pair)]["audio_path"],
                        "duration_ms": self.block_by_pair[int(pair)]["duration_ms"],
                        "color": self.lane_color(i),
                    }
                    for i, pair in enumerate(self.pair_values)
                ],
                "default_case_length": 2,
                "allow_case_length": [1, 2, 3, 4, 5, 6, 7, 8],
            },
            "pattern": self.clean_pattern(),
            "preview_wav": str(wav),
        }

        path = OUT_DIR / f"{self.project['safe']}_tracker_app_edit_v21.json"
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
