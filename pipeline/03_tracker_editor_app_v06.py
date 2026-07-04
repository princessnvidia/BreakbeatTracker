#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_tracker_editor_app_v06.py

BreakbeatAI Tracker Editor v06

Changements v06 :
- Corrige Espace qui ne lance rien.
- Bind clavier global avec bind_all.
- Le canvas reprend le focus automatiquement.
- Affiche clairement les erreurs audio.
- Garde la correction v05 :
    lane visuelle = HI-HAT / KICK / SNARE
    pair audio = bloc sonore joué
- Audio inchangé : rendu par ordre horizontal x_step.

Usage :
    cd ~/Applications/BreakbeatAI
    python pipeline/03_tracker_editor_app_v06.py --source "Amen"
"""

from pathlib import Path
import argparse
import json
import sys
import traceback
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import soundfile as sf


PAIR_BLOCKS_DIR = Path("dataset/pair_blocks_v02")
OUT_DIR = Path("dataset/tracker_edits")
SR = 44100

LANES = ["hat", "kick", "snare"]

LANE_LABELS = {
    "hat": "HI-HAT",
    "kick": "KICK",
    "snare": "SNARE",
}

LANE_COLORS = {
    "hat": "#ffd37a",
    "kick": "#ee8fa7",
    "snare": "#8bbcff",
}

LANE_TO_ROLE = {
    0: "hat",
    1: "kick",
    2: "snare",
}

ROLE_TO_LANE = {
    "hat": 0,
    "kick": 1,
    "snare": 2,
}

DEFAULT_PATTERN = [
    {"id": 0, "x_step": 0,  "lane": 1, "role": "kick",  "length": 2, "pair": 0},
    {"id": 1, "x_step": 2,  "lane": 0, "role": "hat",   "length": 2, "pair": 1},
    {"id": 2, "x_step": 4,  "lane": 2, "role": "snare", "length": 2, "pair": 2},
    {"id": 3, "x_step": 6,  "lane": 0, "role": "hat",   "length": 2, "pair": 3},
    {"id": 4, "x_step": 8,  "lane": 0, "role": "hat",   "length": 2, "pair": 4},
    {"id": 5, "x_step": 10, "lane": 1, "role": "kick",  "length": 2, "pair": 5},
    {"id": 6, "x_step": 12, "lane": 2, "role": "snare", "length": 2, "pair": 6},
    {"id": 7, "x_step": 14, "lane": 0, "role": "hat",   "length": 2, "pair": 7},
]


def find_pair_json(source_query):
    files = sorted(PAIR_BLOCKS_DIR.glob("*_pair_blocks_v02.json"))
    matches = [p for p in files if source_query.lower() in p.name.lower()]

    if not matches:
        print(f"Aucun pair_blocks_v02 JSON trouvé pour : {source_query}")
        print(f'Essaie d’abord : python pipeline/01_find_pair_blocks_v02.py --source "{source_query}"')
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

    return y / m * peak


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


class TrackerEditorApp:
    def __init__(self, root, pair_json):
        self.root = root
        self.pair_json = pair_json
        self.project = self.load_project(pair_json)

        self.blocks = self.project["blocks"]
        self.block_by_pair = {int(b["pair"]): b for b in self.blocks}
        self.pair_values = [int(b["pair"]) for b in self.blocks]

        self.pattern = json.loads(json.dumps(DEFAULT_PATTERN))
        self.sanitize_pattern()

        self.selected_id = self.pattern[0]["id"] if self.pattern else None

        self.step_count = 64
        self.left_width = 170
        self.step_width = 18
        self.row_height = 42
        self.case_height = 36
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

        self.root.title("BreakbeatAI Tracker Editor v06")
        self.root.geometry("1450x720")
        self.root.configure(bg="#111018")

        self.build_ui()
        self.draw()
        self.refresh_panel()

        self.bind_keys()

        self.root.after(200, self.force_keyboard_focus)

    def load_project(self, pair_json):
        meta = json.loads(pair_json.read_text(encoding="utf-8"))

        blocks = []
        for block in meta["blocks"]:
            pair = int(block["pair"])
            blocks.append({
                "pair": pair,
                "name": f"pair {pair:02d}",
                "audio_path": str(Path(block["audio_path"])),
                "duration_ms": float(block.get("duration_ms", 0.0)),
            })

        blocks = sorted(blocks, key=lambda b: b["pair"])

        return {
            "safe": safe_name(pair_json),
            "source_audio": meta.get("source"),
            "source_pair_json": str(pair_json),
            "blocks": blocks,
        }

    def sanitize_pattern(self):
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
                role = str(item.get("role", "hat"))
                item["lane"] = ROLE_TO_LANE.get(role, 0)

            item["lane"] = max(0, min(len(LANES) - 1, int(item["lane"])))
            item["role"] = LANE_TO_ROLE[item["lane"]]

    def bind_keys(self):
        # Bind global : Espace marche même si un bouton ou une combobox a le focus.
        self.root.bind_all("<space>", self.toggle_loop_event)
        self.root.bind_all("<KeyPress-space>", self.toggle_loop_event)

        self.root.bind_all("<Delete>", lambda e: self.delete_selected())
        self.root.bind_all("<Left>", lambda e: self.move_selected(-1, 0))
        self.root.bind_all("<Right>", lambda e: self.move_selected(1, 0))
        self.root.bind_all("<Up>", lambda e: self.move_selected(0, -1))
        self.root.bind_all("<Down>", lambda e: self.move_selected(0, 1))

        print("[v06] Raccourcis clavier actifs : Espace = Play/Stop loop")

    def force_keyboard_focus(self):
        try:
            self.root.focus_force()
            self.canvas.focus_set()
            self.output_label.config(
                text="Clavier actif : appuie sur Espace pour lancer/arrêter la loop."
            )
        except Exception:
            pass

    def build_ui(self):
        main = tk.Frame(self.root, bg="#111018")
        main.pack(fill="both", expand=True, padx=14, pady=14)

        title = tk.Label(
            main,
            text="BreakbeatAI Tracker Editor v06 — Espace réparé + HI-HAT/KICK/SNARE alignés",
            bg="#111018",
            fg="#ff7acc",
            font=("Sans", 20, "bold"),
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            main,
            text="Audio inchangé : le rendu lit les pairs dans l'ordre horizontal x_step. Les lignes servent à organiser hat/kick/snare.",
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
        self.canvas.bind("<FocusIn>", lambda e: print("[v06] Canvas focus OK"))

        panel = tk.Frame(main, bg="#1b1824")
        panel.pack(fill="x", pady=(12, 0))

        self.info_label = tk.Label(panel, text="", bg="#1b1824", fg="#f5eefe")
        self.info_label.grid(row=0, column=0, columnspan=10, sticky="w", padx=10, pady=8)

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

        tk.Label(panel, text="Lane", bg="#1b1824", fg="#b9acc8").grid(row=1, column=2, padx=5)

        self.lane_var = tk.StringVar(value="kick")
        self.lane_box = ttk.Combobox(
            panel,
            textvariable=self.lane_var,
            values=LANES,
            width=8,
            state="readonly",
        )
        self.lane_box.grid(row=1, column=3, padx=5)
        self.lane_box.bind("<<ComboboxSelected>>", lambda e: self.set_lane(ROLE_TO_LANE.get(self.lane_var.get(), 0)))

        tk.Button(panel, text="Play pair", command=self.play_selected_pair, bg="#30283f", fg="#f5eefe").grid(row=1, column=4, padx=5)
        tk.Button(panel, text="Play Loop / Space", command=self.toggle_loop, bg="#30283f", fg="#f5eefe").grid(row=1, column=5, padx=5)
        tk.Button(panel, text="Render preview", command=self.render_preview_only, bg="#30283f", fg="#f5eefe").grid(row=1, column=6, padx=5)
        tk.Button(panel, text="Save data", command=self.save, bg="#30513f", fg="#f5eefe").grid(row=1, column=7, padx=5)
        tk.Button(panel, text="Delete", command=self.delete_selected, bg="#4a2630", fg="#f5eefe").grid(row=1, column=8, padx=5)
        tk.Button(panel, text="Reset", command=self.reset, bg="#30283f", fg="#f5eefe").grid(row=1, column=9, padx=5)

        self.output_label = tk.Label(
            panel,
            text="Clic vide = nouvelle case. Drag vertical = change HI-HAT/KICK/SNARE. Espace = Play/Stop.",
            bg="#1b1824",
            fg="#77f5b5",
            justify="left",
        )
        self.output_label.grid(row=2, column=0, columnspan=10, sticky="w", padx=10, pady=8)

    def get_audio(self, pair):
        pair = int(pair)

        if pair not in self.audio_cache:
            if pair not in self.block_by_pair:
                raise RuntimeError(f"Pair audio introuvable : {pair}")

            self.audio_cache[pair] = load_wav(self.block_by_pair[pair]["audio_path"])

        return self.audio_cache[pair]

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

        for lane_index, role in enumerate(LANES):
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
                text=LANE_LABELS[role],
                fill=LANE_COLORS[role],
                anchor="w",
                font=("Sans", 11, "bold"),
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
        lane = max(0, min(len(LANES) - 1, int(item.get("lane", 0))))
        role = LANE_TO_ROLE[lane]

        x0 = self.step_to_x(int(item["x_step"]))
        x1 = self.step_to_x(int(item["x_step"]) + int(item["length"]))
        y0 = lane * self.row_height + self.case_y_padding
        y1 = y0 + self.case_height

        outline = "#77f5b5" if item["id"] == self.selected_id else "#ffc0cf"
        width = 3 if item["id"] == self.selected_id else 1
        tags = ("case", f"id_{item['id']}")

        self.canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            fill=LANE_COLORS[role],
            outline=outline,
            width=width,
            tags=tags,
        )

        label = f"{item['pair']}"
        if int(item["length"]) >= 3:
            label = f"{LANE_LABELS[role]} {item['pair']}"

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
                lane = self.y_to_lane(event.y)
                role = LANE_TO_ROLE[lane]
                selected = self.selected()

                if selected is not None:
                    pair = int(selected["pair"])
                else:
                    pair = self.pair_values[0] if self.pair_values else 0

                new_item = {
                    "id": self.new_id(),
                    "x_step": step,
                    "lane": lane,
                    "role": role,
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
            item["role"] = LANE_TO_ROLE[int(item["lane"])]

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

        lane = int(item.get("lane", 0))
        role = LANE_TO_ROLE[lane]
        item["role"] = role

        self.info_label.config(
            text=(
                f"id {item['id']} | step {item['x_step']} | length {item['length']} | "
                f"lane {LANE_LABELS[role]} | pair audio {item['pair']}"
            )
        )

        self.pair_var.set(int(item["pair"]))
        self.lane_var.set(role)

    def set_pair(self, pair):
        item = self.selected()

        if item is None:
            return

        item["pair"] = int(pair)
        self.draw()
        self.refresh_panel()
        self.force_keyboard_focus()

    def set_lane(self, lane):
        item = self.selected()

        if item is None:
            return

        lane = max(0, min(len(LANES) - 1, int(lane)))
        item["lane"] = lane
        item["role"] = LANE_TO_ROLE[lane]

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
            lane = max(0, min(len(LANES) - 1, int(item.get("lane", 0)) + dy))
            item["lane"] = lane
            item["role"] = LANE_TO_ROLE[lane]

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
        self.pattern = json.loads(json.dumps(DEFAULT_PATTERN))
        self.sanitize_pattern()
        self.selected_id = self.pattern[0]["id"] if self.pattern else None

        self.draw()
        self.refresh_panel()

    def play_selected_pair(self):
        item = self.selected()

        if item is None:
            return

        try:
            import sounddevice as sd
            audio = self.get_audio(item["pair"])
            sd.stop()
            sd.play(audio, SR)
            self.output_label.config(text=f"Play pair {item['pair']} — {len(audio) / SR:.2f}s")
        except Exception as exc:
            traceback.print_exc()
            self.output_label.config(text=f"Erreur audio Play pair : {exc}")
            messagebox.showwarning("Audio", f"Erreur audio : {exc}")

    def render_audio(self):
        ordered = sorted(self.pattern, key=lambda e: (int(e["x_step"]), int(e["id"])))
        chunks = [self.get_audio(i["pair"]) for i in ordered]

        if not chunks:
            return np.zeros(1, dtype=np.float32)

        return normalize(np.concatenate(chunks))

    def render_preview_file(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        wav = OUT_DIR / f"{self.project['safe']}_tracker_app_v06_preview.wav"
        sf.write(wav, self.render_audio(), SR)
        return wav

    def render_preview_only(self):
        wav = self.render_preview_file()
        self.output_label.config(text=f"Preview : {wav}")

    def loop_tick(self):
        if not self.looping:
            return

        try:
            import sounddevice as sd

            audio = self.render_audio()
            sd.stop()
            sd.play(audio, SR)

            duration_ms = int(len(audio) / SR * 1000)
            self.output_label.config(
                text=f"Loop en cours — durée {duration_ms} ms — Espace pour arrêter."
            )
            print(f"[v06] Loop audio lancée — {duration_ms} ms")

            self.loop_after_id = self.root.after(max(80, duration_ms), self.loop_tick)

        except Exception as exc:
            traceback.print_exc()
            self.looping = False
            self.output_label.config(text=f"Erreur audio loop : {exc}")
            messagebox.showwarning("Audio", f"Erreur audio loop : {exc}")

    def toggle_loop_event(self, event=None):
        print("[v06] Espace détecté")
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

            try:
                import sounddevice as sd
                sd.stop()
            except Exception:
                pass

            self.output_label.config(text="Loop arrêtée — Espace pour relancer.")
            print("[v06] Loop arrêtée")
            return

        print("[v06] Lancement loop...")
        self.looping = True
        self.output_label.config(text="Lancement loop...")
        self.loop_tick()

    def clean_pattern(self):
        out = []

        for item in sorted(self.pattern, key=lambda e: (int(e["x_step"]), int(e["id"]))):
            lane = max(0, min(len(LANES) - 1, int(item.get("lane", 0))))
            role = LANE_TO_ROLE[lane]

            out.append({
                "id": int(item["id"]),
                "x_step": int(item["x_step"]),
                "lane": int(lane),
                "role": role,
                "length": int(item["length"]),
                "pair": int(item["pair"]),
            })

        return out

    def save(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        wav = self.render_preview_file()

        data = {
            "version": "tracker_app_edit_v06_space_bind_all_three_lanes",
            "audio_rule": "render sorted by x_step only; lane/role/length are visual annotations and do not affect audio",
            "source_pair_json": self.project["source_pair_json"],
            "source_audio": self.project["source_audio"],
            "safe": self.project["safe"],
            "grid": {
                "steps": self.step_count,
                "lanes": [
                    {"lane": 0, "role": "hat", "name": "HI-HAT"},
                    {"lane": 1, "role": "kick", "name": "KICK"},
                    {"lane": 2, "role": "snare", "name": "SNARE"},
                ],
                "default_case_length": 2,
                "allow_case_length": [1, 2, 3, 4, 5, 6, 7, 8],
                "pair_is_audio_block_not_visual_row": True,
                "audio_blocks": [
                    {
                        "pair": b["pair"],
                        "name": b["name"],
                        "audio_path": b["audio_path"],
                        "duration_ms": b["duration_ms"],
                    }
                    for b in self.blocks
                ],
            },
            "pattern": self.clean_pattern(),
            "preview_wav": str(wav),
        }

        path = OUT_DIR / f"{self.project['safe']}_tracker_app_edit_v06.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

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
