#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_tracker_editor_app_v03.py

Tracker editor natif BreakbeatAI.

Corrections v03 :
- grille 32 cases
- blocs moins hauts
- espaces vides possibles
- longueur de bloc : 1, 2, 3 ou 4 cases
- défaut : 2 cases
- Espace = render + play loop
- Entrée = sauvegarder + render
- flèches = déplacer
- +/- = changer longueur
- 1..8 = changer pair audio
- A = ajouter bloc
- Suppr = supprimer bloc
- P = play pair sélectionnée

Usage :
    python pipeline/03_tracker_editor_app_v03.py --source "Amen"

Avec image :
    python pipeline/03_tracker_editor_app_v03.py --source "Amen" --image ~/Téléchargements/'break(1).png'
"""

from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import soundfile as sf

try:
    from PIL import Image
except Exception:
    Image = None


PAIR_BLOCKS_DIR = Path("dataset/pair_blocks_v02")
OUT_DIR = Path("dataset/tracker_edits")
SR = 44100

STEPS = 32
LANES = ["hat", "kick", "snare"]
LANE_LABELS = {"hat": "HIHAT", "kick": "KICK", "snare": "SNARE"}
ROLE_TO_LANE = {"hat": 0, "kick": 1, "snare": 2}
LANE_TO_ROLE = {0: "hat", 1: "kick", 2: "snare"}

DEFAULT_PATTERN = [
    {"id": 0, "step": 0,  "lane": 1, "role": "kick",  "pair": 0, "length": 2},
    {"id": 1, "step": 2,  "lane": 0, "role": "hat",   "pair": 1, "length": 2},
    {"id": 2, "step": 4,  "lane": 2, "role": "snare", "pair": 2, "length": 2},
    {"id": 3, "step": 6,  "lane": 0, "role": "hat",   "pair": 3, "length": 2},
    {"id": 4, "step": 8,  "lane": 0, "role": "hat",   "pair": 4, "length": 2},
    {"id": 5, "step": 10, "lane": 1, "role": "kick",  "pair": 5, "length": 2},
    {"id": 6, "step": 12, "lane": 2, "role": "snare", "pair": 6, "length": 2},
    {"id": 7, "step": 14, "lane": 0, "role": "hat",   "pair": 7, "length": 2},
    {"id": 8, "step": 16, "lane": 1, "role": "kick",  "pair": 0, "length": 2},
    {"id": 9, "step": 18, "lane": 0, "role": "hat",   "pair": 1, "length": 2},
    {"id": 10, "step": 20, "lane": 2, "role": "snare", "pair": 2, "length": 2},
    {"id": 11, "step": 22, "lane": 0, "role": "hat",   "pair": 3, "length": 2},
    {"id": 12, "step": 24, "lane": 0, "role": "hat",   "pair": 4, "length": 2},
    {"id": 13, "step": 26, "lane": 1, "role": "kick",  "pair": 5, "length": 2},
    {"id": 14, "step": 28, "lane": 2, "role": "snare", "pair": 6, "length": 2},
    {"id": 15, "step": 30, "lane": 0, "role": "hat",   "pair": 7, "length": 2},
]


def find_pair_json(source_query):
    files = sorted(PAIR_BLOCKS_DIR.glob("*_pair_blocks_v02.json"))
    matches = [p for p in files if source_query.lower() in p.name.lower()]
    if not matches:
        print(f"Aucun pair_blocks_v02 JSON trouvé pour : {source_query}")
        print(f'  python pipeline/01_find_pair_blocks_v02.py --source "{source_query}"')
        sys.exit(1)
    return matches[0]


def safe_name(pair_json):
    return pair_json.stem.replace("_pair_blocks_v02", "")


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


def load_wav(path):
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != SR:
        raise RuntimeError(f"Sample rate inattendu {sr} pour {path}, attendu {SR}")
    return fade(normalize(audio), ms=2)


def play_audio_file(path):
    path = str(path)
    for cmd in [
        ["paplay", path],
        ["aplay", path],
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
        ["mpv", "--no-video", "--really-quiet", path],
    ]:
        if shutil.which(cmd[0]):
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception:
                pass
    return False


def stretch_or_crop(audio, target_len):
    if target_len <= 0:
        return np.zeros(1, dtype=np.float32)
    if len(audio) == target_len:
        return audio.copy()
    if len(audio) > target_len:
        return fade(audio[:target_len].copy(), ms=2)
    out = np.zeros(target_len, dtype=np.float32)
    out[:len(audio)] = audio
    return fade(out, ms=2)


def pink_mask_from_image(image_path):
    if Image is None:
        return None, None
    img = Image.open(Path(image_path).expanduser())
    arr = np.array(img.convert("RGB"))
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    mask = (
        (r > 170)
        & (g > 55)
        & (g < 205)
        & (b > 75)
        & (b < 235)
        & (r > g + 20)
    )
    return mask, img.size


def group_contiguous(indices, min_len=1):
    if len(indices) == 0:
        return []
    groups = []
    start = indices[0]
    prev = indices[0]
    for x in indices[1:]:
        if x == prev + 1:
            prev = x
        else:
            if prev - start + 1 >= min_len:
                groups.append((start, prev))
            start = x
            prev = x
    if prev - start + 1 >= min_len:
        groups.append((start, prev))
    return groups


def detect_lanes(mask):
    row_strength = mask.sum(axis=1)
    ys = np.where(row_strength > 5)[0]
    if len(ys) == 0:
        return None
    bands = group_contiguous(list(ys), min_len=3)
    scored = []
    for y1, y2 in bands:
        scored.append((int(row_strength[y1:y2 + 1].sum()), y1, y2))
    scored.sort(reverse=True)
    selected = sorted([(y1, y2) for _, y1, y2 in scored[:3]], key=lambda x: x[0])
    if len(selected) < 3:
        return None
    return {"hat": selected[0], "kick": selected[1], "snare": selected[2]}


def pattern_from_image(image_path):
    mask, image_size = pink_mask_from_image(image_path)
    if mask is None:
        return None
    lanes = detect_lanes(mask)
    if lanes is None:
        return None

    events = []
    for role, (y1, y2) in lanes.items():
        lane_mask = mask[y1:y2 + 1, :]
        col_strength = lane_mask.sum(axis=0)
        xs = np.where(col_strength > 2)[0]
        intervals = group_contiguous(list(xs), min_len=3)
        for x1, x2 in intervals:
            events.append({
                "role": role,
                "lane": ROLE_TO_LANE[role],
                "x1": int(x1),
                "x2": int(x2),
            })

    if not events:
        return None

    events.sort(key=lambda e: (e["x1"], e["x2"]))
    x_min = min(e["x1"] for e in events)
    x_max = max(e["x2"] for e in events)
    span = max(1, x_max - x_min)

    pattern = []
    for i, e in enumerate(events):
        start_norm = (e["x1"] - x_min) / span
        end_norm = (e["x2"] - x_min) / span

        step = int(round(start_norm * (STEPS - 1)))
        step = max(0, min(STEPS - 1, step))

        width_steps = max(1, int(round((end_norm - start_norm) * STEPS)))
        length = max(1, min(4, width_steps))
        if step + length > STEPS:
            step = STEPS - length

        pattern.append({
            "id": i,
            "step": int(step),
            "lane": int(e["lane"]),
            "role": e["role"],
            "pair": int(i % 8),
            "length": int(length),
        })

    return {
        "pattern": pattern,
        "image_meta": {
            "image": str(image_path),
            "image_size": list(image_size),
            "event_count": len(pattern),
            "lanes": {k: [int(v[0]), int(v[1])] for k, v in lanes.items()},
            "x_min": int(x_min),
            "x_max": int(x_max),
        },
    }


class TrackerEditorApp:
    def __init__(self, root, pair_json, image_path=None):
        self.root = root
        self.pair_json = pair_json
        self.project = self.load_project(pair_json, image_path)
        self.blocks = self.project["blocks"]
        self.audio_cache = {}
        self.pattern = json.loads(json.dumps(self.project["pattern"]))
        self.selected_id = self.pattern[0]["id"] if self.pattern else None
        self.drag_item = None
        self.drag_offset_steps = 0

        self.canvas_width = 1184
        self.canvas_height = 255
        self.lane_height = self.canvas_height / 3
        self.step_width = self.canvas_width / STEPS
        self.block_h = 22

        self.root.title("BreakbeatAI Tracker Editor v03")
        self.root.geometry("1400x840")

        self.setup_style()
        self.build_ui()
        self.draw_tracker()
        self.refresh_selection_panel()

    def setup_style(self):
        self.root.configure(bg="#111018")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#111018")
        style.configure("Panel.TFrame", background="#1b1824")
        style.configure("TButton", background="#30283f", foreground="#f5eefe")
        style.configure("TCombobox", fieldbackground="#130f1c", foreground="#f5eefe")

    def load_project(self, pair_json, image_path=None):
        meta = json.loads(pair_json.read_text(encoding="utf-8"))
        blocks = []
        for block in meta["blocks"]:
            blocks.append({
                "pair": int(block["pair"]),
                "audio_path": str(Path(block["audio_path"])),
                "start_ms": float(block["start_ms"]),
                "end_ms": float(block["end_ms"]),
                "duration_ms": float(block["duration_ms"]),
                "rms": float(block.get("rms", 0.0)),
            })
        blocks = sorted(blocks, key=lambda b: b["pair"])

        image_result = None
        if image_path is not None:
            try:
                image_result = pattern_from_image(image_path)
            except Exception as exc:
                print("Image fallback :", exc)

        if image_result:
            pattern = image_result["pattern"]
            image_meta = image_result["image_meta"]
        else:
            pattern = json.loads(json.dumps(DEFAULT_PATTERN))
            image_meta = None

        return {
            "source_pair_json": str(pair_json),
            "source_audio": meta.get("source"),
            "safe": safe_name(pair_json),
            "blocks": blocks,
            "pattern": pattern,
            "image_meta": image_meta,
        }

    def build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=16, pady=16)

        tk.Label(main, text="BreakbeatAI Tracker Editor v03 — grid 32", bg="#111018", fg="#ff7acc", font=("Sans", 20, "bold")).pack(anchor="w")
        tk.Label(
            main,
            text="Espace = render + play | +/- = longueur | flèches = déplacer | 1..8 = pair | A = ajouter | P = play pair",
            bg="#111018",
            fg="#b9acc8",
        ).pack(anchor="w", pady=(0, 10))

        self.canvas = tk.Canvas(
            main,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#181321",
            highlightthickness=1,
            highlightbackground="#41334f",
        )
        self.canvas.pack(fill="x", pady=(0, 12))

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        body = ttk.Frame(main)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, style="Panel.TFrame")
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        right = ttk.Frame(body, style="Panel.TFrame")
        right.pack(side="right", fill="y", padx=(8, 0))

        self.build_blocks_panel(left)
        self.build_edit_panel(right)

        self.root.bind("<space>", lambda e: self.render_and_play())
        self.root.bind("<Return>", lambda e: self.save())
        self.root.bind("<Up>", lambda e: self.move_lane(-1))
        self.root.bind("<Down>", lambda e: self.move_lane(1))
        self.root.bind("<Left>", lambda e: self.move_step(-1))
        self.root.bind("<Right>", lambda e: self.move_step(1))
        self.root.bind("<Delete>", lambda e: self.delete_selected())
        self.root.bind("a", lambda e: self.add_block())
        self.root.bind("p", lambda e: self.play_selected())
        self.root.bind("+", lambda e: self.change_length(1))
        self.root.bind("-", lambda e: self.change_length(-1))
        for n in range(1, 9):
            self.root.bind(str(n), lambda e, num=n: self.assign_pair(num - 1))

    def build_blocks_panel(self, parent):
        tk.Label(parent, text="Pair blocks disponibles", bg="#1b1824", fg="#77f5b5", font=("Sans", 14, "bold")).pack(anchor="w", padx=12, pady=(12, 6))
        grid = ttk.Frame(parent, style="Panel.TFrame")
        grid.pack(fill="both", expand=True, padx=12, pady=12)
        for i, block in enumerate(self.blocks):
            frame = tk.Frame(grid, bg="#241f31", bd=1, relief="solid")
            frame.grid(row=i // 4, column=i % 4, sticky="nsew", padx=6, pady=6)
            tk.Label(frame, text=f"{i + 1} — pair {block['pair']:02d}", bg="#241f31", fg="#ffe08a", font=("Sans", 11, "bold")).pack(anchor="w", padx=8, pady=(8, 2))
            tk.Label(frame, text=f"{block['duration_ms']:.1f} ms", bg="#241f31", fg="#b9acc8").pack(anchor="w", padx=8)
            btns = tk.Frame(frame, bg="#241f31")
            btns.pack(fill="x", padx=8, pady=8)
            tk.Button(btns, text="Play", command=lambda b=block: self.play_block(b["pair"]), bg="#30283f", fg="#f5eefe").pack(side="left", padx=(0, 4))
            tk.Button(btns, text="Assigner", command=lambda b=block: self.assign_pair(b["pair"]), bg="#30283f", fg="#f5eefe").pack(side="left")
        for c in range(4):
            grid.columnconfigure(c, weight=1)

    def build_edit_panel(self, parent):
        tk.Label(parent, text="Bloc sélectionné", bg="#1b1824", fg="#77f5b5", font=("Sans", 14, "bold")).pack(anchor="w", padx=12, pady=(12, 8))
        self.selected_label = tk.Label(parent, text="Aucun", bg="#1b1824", fg="#f5eefe", justify="left")
        self.selected_label.pack(anchor="w", padx=12, pady=(0, 8))

        tk.Label(parent, text="Rôle / ligne", bg="#1b1824", fg="#b9acc8").pack(anchor="w", padx=12)
        self.role_var = tk.StringVar(value="hat")
        self.role_box = ttk.Combobox(parent, textvariable=self.role_var, values=["hat", "kick", "snare"], state="readonly")
        self.role_box.pack(fill="x", padx=12, pady=(0, 8))
        self.role_box.bind("<<ComboboxSelected>>", lambda e: self.set_selected_role(self.role_var.get()))

        tk.Label(parent, text="Pair audio", bg="#1b1824", fg="#b9acc8").pack(anchor="w", padx=12)
        self.pair_var = tk.IntVar(value=0)
        self.pair_box = ttk.Combobox(parent, textvariable=self.pair_var, values=[b["pair"] for b in self.blocks], state="readonly")
        self.pair_box.pack(fill="x", padx=12, pady=(0, 8))
        self.pair_box.bind("<<ComboboxSelected>>", lambda e: self.assign_pair(int(self.pair_var.get())))

        tk.Label(parent, text="Longueur cases", bg="#1b1824", fg="#b9acc8").pack(anchor="w", padx=12)
        self.length_var = tk.IntVar(value=2)
        self.length_box = ttk.Combobox(parent, textvariable=self.length_var, values=[1, 2, 3, 4], state="readonly")
        self.length_box.pack(fill="x", padx=12, pady=(0, 8))
        self.length_box.bind("<<ComboboxSelected>>", lambda e: self.set_length(int(self.length_var.get())))

        button_grid = tk.Frame(parent, bg="#1b1824")
        button_grid.pack(fill="x", padx=12, pady=8)

        buttons = [
            ("Monter ↑", lambda: self.move_lane(-1)),
            ("Descendre ↓", lambda: self.move_lane(1)),
            ("Gauche ←", lambda: self.move_step(-1)),
            ("Droite →", lambda: self.move_step(1)),
            ("Longueur +", lambda: self.change_length(1)),
            ("Longueur -", lambda: self.change_length(-1)),
            ("Play pair", self.play_selected),
            ("Supprimer", self.delete_selected),
            ("Ajouter A", self.add_block),
            ("Espace: boucle", self.render_and_play),
        ]
        for idx, (txt, cmd) in enumerate(buttons):
            tk.Button(button_grid, text=txt, command=cmd, bg="#30283f", fg="#f5eefe").grid(row=idx // 2, column=idx % 2, sticky="ew", padx=3, pady=3)
        button_grid.columnconfigure(0, weight=1)
        button_grid.columnconfigure(1, weight=1)

        tk.Label(parent, text="Preview loops", bg="#1b1824", fg="#b9acc8").pack(anchor="w", padx=12, pady=(12, 0))
        self.loops_var = tk.IntVar(value=1)
        ttk.Spinbox(parent, from_=1, to=32, textvariable=self.loops_var).pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(parent, text="Notes", bg="#1b1824", fg="#b9acc8").pack(anchor="w", padx=12)
        self.notes_text = tk.Text(parent, height=5, bg="#130f1c", fg="#f5eefe", insertbackground="#f5eefe")
        self.notes_text.pack(fill="x", padx=12, pady=(0, 8))

        tk.Button(parent, text="Sauvegarder + render", command=self.save, bg="#30513f", fg="#f5eefe").pack(fill="x", padx=12, pady=4)
        tk.Button(parent, text="Render + play", command=self.render_and_play, bg="#30283f", fg="#f5eefe").pack(fill="x", padx=12, pady=4)
        tk.Button(parent, text="Trier", command=self.sort_pattern, bg="#30283f", fg="#f5eefe").pack(fill="x", padx=12, pady=4)
        tk.Button(parent, text="Reset", command=self.reset_pattern, bg="#30283f", fg="#f5eefe").pack(fill="x", padx=12, pady=4)

        self.output_label = tk.Label(parent, text="En attente.", bg="#09070e", fg="#77f5b5", justify="left", wraplength=330, anchor="nw", padx=8, pady=8)
        self.output_label.pack(fill="both", expand=True, padx=12, pady=12)

    def draw_tracker(self):
        self.canvas.delete("all")
        lane_colors = ["#211a2d", "#1d2131", "#231a24"]

        for lane in range(3):
            y0 = lane * self.lane_height
            y1 = y0 + self.lane_height
            self.canvas.create_rectangle(0, y0, self.canvas_width, y1, fill=lane_colors[lane], outline="#343044")
            self.canvas.create_text(10, y0 + 15, text=LANE_LABELS[LANE_TO_ROLE[lane]], fill="#b9acc8", anchor="w", font=("Sans", 9, "bold"))

        for i in range(STEPS + 1):
            x = i * self.step_width
            color = "#6a5c7c" if i % 8 == 0 else ("#51435f" if i % 4 == 0 else "#30283f")
            width = 2 if i % 4 == 0 else 1
            self.canvas.create_line(x, 0, x, self.canvas_height, fill=color, width=width)

        for i in range(0, STEPS, 4):
            x = i * self.step_width + 3
            self.canvas.create_text(x, self.canvas_height - 9, text=str(i + 1), fill="#786a89", anchor="w", font=("Sans", 8))

        for item in self.pattern:
            self.draw_block(item)

    def draw_block(self, item):
        y = item["lane"] * self.lane_height + self.lane_height / 2 - self.block_h / 2
        x = item["step"] * self.step_width + 3
        w = item["length"] * self.step_width - 6
        outline = "#77f5b5" if item["id"] == self.selected_id else "#ffc0cf"
        width = 3 if item["id"] == self.selected_id else 1
        tags = ("block", f"id_{item['id']}")
        self.canvas.create_rectangle(x, y, x + w, y + self.block_h, fill="#ee8fa7", outline=outline, width=width, tags=tags)
        self.canvas.create_text(x + w / 2, y + self.block_h / 2, text=str(item["pair"]), fill="#1a0d14", font=("Sans", 9, "bold"), tags=tags)

    def get_item_at(self, x, y):
        found = self.canvas.find_overlapping(x, y, x, y)
        for obj in reversed(found):
            for tag in self.canvas.gettags(obj):
                if tag.startswith("id_"):
                    return int(tag.replace("id_", ""))
        return None

    def get_selected(self):
        for item in self.pattern:
            if item["id"] == self.selected_id:
                return item
        return None

    def on_canvas_click(self, event):
        item_id = self.get_item_at(event.x, event.y)
        if item_id is not None:
            self.selected_id = item_id
            selected = self.get_selected()
            self.drag_item = item_id
            self.drag_offset_steps = int(round(event.x / self.step_width)) - selected["step"]
        else:
            self.drag_item = None
        self.draw_tracker()
        self.refresh_selection_panel()

    def on_canvas_drag(self, event):
        if self.drag_item is None:
            return
        item = self.get_selected()
        if item is None:
            return
        raw_step = int(round(event.x / self.step_width)) - self.drag_offset_steps
        item["step"] = int(max(0, min(STEPS - item["length"], raw_step)))
        lane = int(event.y // self.lane_height)
        item["lane"] = int(max(0, min(2, lane)))
        item["role"] = LANE_TO_ROLE[item["lane"]]
        self.draw_tracker()
        self.refresh_selection_panel()

    def on_canvas_release(self, event):
        self.drag_item = None

    def refresh_selection_panel(self):
        item = self.get_selected()
        if item is None:
            self.selected_label.config(text="Aucun")
            return
        self.selected_label.config(text=f"id {item['id']}\nrole {item['role']}\npair {item['pair']}\nstep {item['step'] + 1}/32\nlength {item['length']} case(s)")
        self.role_var.set(item["role"])
        self.pair_var.set(item["pair"])
        self.length_var.set(item["length"])

    def set_selected_role(self, role):
        item = self.get_selected()
        if item is None:
            return
        item["role"] = role
        item["lane"] = ROLE_TO_LANE[role]
        self.draw_tracker()
        self.refresh_selection_panel()

    def assign_pair(self, pair):
        item = self.get_selected()
        if item is None:
            return
        item["pair"] = int(pair)
        self.draw_tracker()
        self.refresh_selection_panel()

    def set_length(self, length):
        item = self.get_selected()
        if item is None:
            return
        item["length"] = int(max(1, min(4, length)))
        item["step"] = int(max(0, min(STEPS - item["length"], item["step"])))
        self.draw_tracker()
        self.refresh_selection_panel()

    def change_length(self, delta):
        item = self.get_selected()
        if item is None:
            return
        self.set_length(item["length"] + delta)

    def move_lane(self, delta):
        item = self.get_selected()
        if item is None:
            return
        item["lane"] = int(max(0, min(2, item["lane"] + delta)))
        item["role"] = LANE_TO_ROLE[item["lane"]]
        self.draw_tracker()
        self.refresh_selection_panel()

    def move_step(self, delta):
        item = self.get_selected()
        if item is None:
            return
        item["step"] = int(max(0, min(STEPS - item["length"], item["step"] + delta)))
        self.draw_tracker()
        self.refresh_selection_panel()

    def delete_selected(self):
        if self.selected_id is None:
            return
        self.pattern = [i for i in self.pattern if i["id"] != self.selected_id]
        self.selected_id = self.pattern[0]["id"] if self.pattern else None
        self.draw_tracker()
        self.refresh_selection_panel()

    def add_block(self):
        new_id = max([i["id"] for i in self.pattern], default=-1) + 1
        self.pattern.append({"id": new_id, "step": 0, "lane": 0, "role": "hat", "pair": 1, "length": 2})
        self.selected_id = new_id
        self.draw_tracker()
        self.refresh_selection_panel()

    def sort_pattern(self):
        self.pattern.sort(key=lambda i: (i["step"], i["lane"]))
        self.draw_tracker()

    def reset_pattern(self):
        self.pattern = json.loads(json.dumps(self.project["pattern"]))
        self.selected_id = self.pattern[0]["id"] if self.pattern else None
        self.draw_tracker()
        self.refresh_selection_panel()

    def play_block(self, pair):
        ok = play_audio_file(self.blocks[int(pair)]["audio_path"])
        if not ok:
            messagebox.showwarning("Lecture audio", "Installe paplay, aplay, ffplay ou mpv.")

    def play_selected(self):
        item = self.get_selected()
        if item is not None:
            self.play_block(item["pair"])

    def load_audio_for_pair(self, pair):
        if pair not in self.audio_cache:
            self.audio_cache[pair] = load_wav(self.blocks[int(pair)]["audio_path"])
        return self.audio_cache[pair]

    def render_preview(self, pattern, loops):
        base_block = self.load_audio_for_pair(0)
        one_case_samples = max(1, len(base_block) // 4)  # pair block ~= 4 cases in grid32
        total_samples = STEPS * one_case_samples
        loop_audio = np.zeros(total_samples, dtype=np.float32)

        for item in sorted(pattern, key=lambda e: (e["step"], e["lane"])):
            audio = self.load_audio_for_pair(int(item["pair"]))
            target_len = one_case_samples * int(item["length"])
            audio = stretch_or_crop(audio, target_len)
            start = int(item["step"]) * one_case_samples
            end = min(total_samples, start + len(audio))
            loop_audio[start:end] += audio[:end - start]

        loop_audio = normalize(loop_audio)
        out = normalize(np.concatenate([loop_audio for _ in range(loops)]))

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        wav = OUT_DIR / f"{self.project['safe']}_tracker_app_v03_preview.wav"
        sf.write(wav, out, SR)
        return str(wav)

    def clean_pattern_for_save(self):
        out = []
        for item in sorted(self.pattern, key=lambda e: (e["step"], e["lane"])):
            out.append({
                "id": int(item["id"]),
                "step": int(item["step"]),
                "lane": int(item["lane"]),
                "role": LANE_TO_ROLE[int(item["lane"])],
                "pair": int(item["pair"]),
                "length": int(item["length"]),
            })
        return out

    def save(self):
        pattern = self.clean_pattern_for_save()
        loops = int(self.loops_var.get())
        preview = self.render_preview(pattern, loops)

        data = {
            "version": "tracker_app_edit_v03_grid32",
            "source_pair_json": self.project["source_pair_json"],
            "source_audio": self.project["source_audio"],
            "safe": self.project["safe"],
            "image_meta": self.project["image_meta"],
            "grid": "32 steps, 3 lanes hat/kick/snare, event length 1-4 cases",
            "loops_preview": loops,
            "notes": self.notes_text.get("1.0", "end").strip(),
            "pattern": pattern,
            "preview_wav": preview,
            "blocks": self.blocks,
        }

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        json_path = OUT_DIR / f"{self.project['safe']}_tracker_app_edit_v03.json"
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.output_label.config(text=f"OK\nJSON : {json_path}\nPreview : {preview}")

    def render_and_play(self):
        pattern = self.clean_pattern_for_save()
        loops = int(self.loops_var.get())
        preview = self.render_preview(pattern, loops)
        self.output_label.config(text=f"Preview rendue :\n{preview}")
        ok = play_audio_file(preview)
        if not ok:
            messagebox.showwarning("Lecture audio", "Installe paplay, aplay, ffplay ou mpv.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Amen")
    parser.add_argument("--image", default=None)
    args = parser.parse_args()

    pair_json = find_pair_json(args.source)
    root = tk.Tk()
    TrackerEditorApp(root, pair_json, image_path=args.image)
    root.mainloop()


if __name__ == "__main__":
    main()
