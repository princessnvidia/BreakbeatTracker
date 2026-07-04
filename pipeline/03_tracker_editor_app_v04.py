#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_tracker_editor_app_v04.py

Objectif :
- Garder l'audio comme avant : rendu par ordre horizontal des blocs, sans changer la logique sonore.
- Améliorer seulement l'affichage tracker.
- Chaque sample/pair a sa propre ligne.
- Les noms sont dans une colonne à gauche séparée.
- Les cases sont collées aux lignes, sans marge verticale.
- Les cases peuvent faire 1 ou 2 steps visuellement.
- Clic vide = nouvelle case.
- Drag centre = déplacer.
- Drag bord gauche/droit = rétrécir/agrandir.
- Espace = play/stop boucle si sounddevice est installé.

Usage :
    python pipeline/03_tracker_editor_app_v04.py --source "Amen"

Option :
    python -m pip install sounddevice
"""

from pathlib import Path
import argparse
import json
import sys
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import soundfile as sf


PAIR_BLOCKS_DIR = Path("dataset/pair_blocks_v02")
OUT_DIR = Path("dataset/tracker_edits")
SR = 44100

DEFAULT_PATTERN = [
    {"id": 0, "x_step": 0,  "length": 2, "pair": 0},
    {"id": 1, "x_step": 2,  "length": 2, "pair": 1},
    {"id": 2, "x_step": 4,  "length": 2, "pair": 2},
    {"id": 3, "x_step": 6,  "length": 2, "pair": 3},
    {"id": 4, "x_step": 8,  "length": 2, "pair": 4},
    {"id": 5, "x_step": 10, "length": 2, "pair": 5},
    {"id": 6, "x_step": 12, "length": 2, "pair": 6},
    {"id": 7, "x_step": 14, "length": 2, "pair": 7},
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


class TrackerEditorApp:
    def __init__(self, root, pair_json):
        self.root = root
        self.pair_json = pair_json
        self.project = self.load_project(pair_json)

        self.blocks = self.project["blocks"]
        self.pattern = json.loads(json.dumps(DEFAULT_PATTERN))
        self.selected_id = self.pattern[0]["id"]

        self.step_count = 64
        self.left_width = 170
        self.step_width = 18
        self.row_height = 24
        self.case_height = 24
        self.canvas_width = self.left_width + self.step_width * self.step_count
        self.canvas_height = self.row_height * len(self.blocks)

        self.drag_mode = None
        self.drag_start_x = 0
        self.drag_start_step = 0
        self.drag_start_len = 0

        self.audio_cache = {}
        self.looping = False
        self.loop_after_id = None

        self.root.title("BreakbeatAI Tracker Editor v04")
        self.root.geometry("1450x760")
        self.root.configure(bg="#111018")

        self.build_ui()
        self.draw()
        self.refresh_panel()

        self.root.bind("<space>", self.toggle_loop_event)
        self.root.bind("<Delete>", lambda e: self.delete_selected())
        self.root.bind("<Left>", lambda e: self.move_selected(-1, 0))
        self.root.bind("<Right>", lambda e: self.move_selected(1, 0))
        self.root.bind("<Up>", lambda e: self.move_selected(0, -1))
        self.root.bind("<Down>", lambda e: self.move_selected(0, 1))

    def load_project(self, pair_json):
        meta = json.loads(pair_json.read_text(encoding="utf-8"))
        blocks = []
        for block in meta["blocks"]:
            pair = int(block["pair"])
            blocks.append({
                "pair": pair,
                "name": f"pair {pair:02d}",
                "audio_path": str(Path(block["audio_path"])),
                "duration_ms": float(block["duration_ms"]),
            })
        blocks = sorted(blocks, key=lambda b: b["pair"])
        return {
            "safe": safe_name(pair_json),
            "source_audio": meta.get("source"),
            "source_pair_json": str(pair_json),
            "blocks": blocks,
        }

    def build_ui(self):
        main = tk.Frame(self.root, bg="#111018")
        main.pack(fill="both", expand=True, padx=14, pady=14)

        title = tk.Label(
            main,
            text="BreakbeatAI Tracker Editor v04",
            bg="#111018",
            fg="#ff7acc",
            font=("Sans", 20, "bold"),
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            main,
            text="Audio inchangé : le rendu lit les blocs dans l'ordre horizontal. Les longueurs sont visuelles/data.",
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
        )
        self.canvas.pack(fill="x")

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        panel = tk.Frame(main, bg="#1b1824")
        panel.pack(fill="x", pady=(12, 0))

        self.info_label = tk.Label(panel, text="", bg="#1b1824", fg="#f5eefe")
        self.info_label.grid(row=0, column=0, columnspan=9, sticky="w", padx=10, pady=8)

        tk.Label(panel, text="Pair", bg="#1b1824", fg="#b9acc8").grid(row=1, column=0, padx=5)
        self.pair_var = tk.IntVar(value=0)
        self.pair_box = ttk.Combobox(
            panel,
            textvariable=self.pair_var,
            values=[b["pair"] for b in self.blocks],
            width=8,
            state="readonly",
        )
        self.pair_box.grid(row=1, column=1, padx=5)
        self.pair_box.bind("<<ComboboxSelected>>", lambda e: self.set_pair(int(self.pair_var.get())))

        tk.Button(panel, text="Play pair", command=self.play_selected_pair, bg="#30283f", fg="#f5eefe").grid(row=1, column=2, padx=5)
        tk.Button(panel, text="Play Loop / Space", command=self.toggle_loop, bg="#30283f", fg="#f5eefe").grid(row=1, column=3, padx=5)
        tk.Button(panel, text="Render preview", command=self.render_preview_only, bg="#30283f", fg="#f5eefe").grid(row=1, column=4, padx=5)
        tk.Button(panel, text="Save data", command=self.save, bg="#30513f", fg="#f5eefe").grid(row=1, column=5, padx=5)
        tk.Button(panel, text="Delete", command=self.delete_selected, bg="#4a2630", fg="#f5eefe").grid(row=1, column=6, padx=5)
        tk.Button(panel, text="Reset", command=self.reset, bg="#30283f", fg="#f5eefe").grid(row=1, column=7, padx=5)

        self.output_label = tk.Label(
            panel,
            text="Clic vide = nouvelle case. Bord = resize. Centre = move. Espace = boucle.",
            bg="#1b1824",
            fg="#77f5b5",
            justify="left",
        )
        self.output_label.grid(row=2, column=0, columnspan=9, sticky="w", padx=10, pady=8)

    def get_audio(self, pair):
        pair = int(pair)
        if pair not in self.audio_cache:
            self.audio_cache[pair] = load_wav(self.blocks[pair]["audio_path"])
        return self.audio_cache[pair]

    def step_to_x(self, step):
        return self.left_width + step * self.step_width

    def x_to_step(self, x):
        if x < self.left_width:
            return 0
        return max(0, min(self.step_count - 1, int((x - self.left_width) // self.step_width)))

    def y_to_pair(self, y):
        return max(0, min(len(self.blocks) - 1, int(y // self.row_height)))

    def selected(self):
        for item in self.pattern:
            if item["id"] == self.selected_id:
                return item
        return None

    def new_id(self):
        return max([i["id"] for i in self.pattern], default=-1) + 1

    def draw(self):
        self.canvas.delete("all")

        for row, block in enumerate(self.blocks):
            y0 = row * self.row_height
            y1 = y0 + self.row_height

            row_fill = "#252525" if row % 2 == 0 else "#202020"
            self.canvas.create_rectangle(0, y0, self.canvas_width, y1, fill=row_fill, outline="#343434")

            self.canvas.create_rectangle(0, y0, self.left_width, y1, fill="#17131f", outline="#343044")
            self.canvas.create_text(10, y0 + self.row_height / 2, text=block["name"], fill="#b9acc8", anchor="w", font=("Sans", 9, "bold"))
            self.canvas.create_text(86, y0 + self.row_height / 2, text=f"{block['duration_ms']:.0f}ms", fill="#7f728e", anchor="w", font=("Sans", 8))

        for step in range(self.step_count + 1):
            x = self.step_to_x(step)
            if step % 8 == 0:
                color = "#777777"
            elif step % 4 == 0:
                color = "#555555"
            else:
                color = "#393939"
            self.canvas.create_line(x, 0, x, self.canvas_height, fill=color)

        self.canvas.create_line(self.left_width, 0, self.left_width, self.canvas_height, fill="#888888", width=2)

        for item in self.pattern:
            self.draw_case(item)

    def draw_case(self, item):
        row = int(item["pair"])
        x0 = self.step_to_x(int(item["x_step"]))
        x1 = self.step_to_x(int(item["x_step"]) + int(item["length"]))
        y0 = row * self.row_height
        y1 = y0 + self.case_height

        outline = "#77f5b5" if item["id"] == self.selected_id else "#ffc0cf"
        width = 3 if item["id"] == self.selected_id else 1
        tags = ("case", f"id_{item['id']}")

        self.canvas.create_rectangle(x0, y0, x1, y1, fill="#ee8fa7", outline=outline, width=width, tags=tags)

        if int(item["length"]) >= 2:
            self.canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=str(item["pair"]), fill="#1a0d14", font=("Sans", 8, "bold"), tags=tags)

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
        item_id = self.get_item_at(event.x, event.y)

        if item_id is None:
            if event.x >= self.left_width:
                step = self.x_to_step(event.x)
                pair = self.y_to_pair(event.y)
                new_item = {"id": self.new_id(), "x_step": step, "length": 2, "pair": pair}
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
            item["x_step"] = max(0, min(self.step_count - item["length"], self.drag_start_step + delta_steps))
            item["pair"] = self.y_to_pair(event.y)

        elif self.drag_mode == "resize_right":
            item["length"] = max(1, min(self.step_count - item["x_step"], self.drag_start_len + delta_steps))

        elif self.drag_mode == "resize_left":
            old_end = self.drag_start_step + self.drag_start_len
            new_start = self.drag_start_step + delta_steps
            new_start = max(0, min(old_end - 1, new_start))
            item["x_step"] = new_start
            item["length"] = max(1, old_end - new_start)

        self.draw()
        self.refresh_panel()

    def on_release(self, event):
        self.drag_mode = None

    def refresh_panel(self):
        item = self.selected()
        if item is None:
            self.info_label.config(text="Aucun bloc sélectionné")
            return
        self.info_label.config(text=f"id {item['id']} | step {item['x_step']} | length {item['length']} | pair {item['pair']}")
        self.pair_var.set(item["pair"])

    def set_pair(self, pair):
        item = self.selected()
        if item is None:
            return
        item["pair"] = int(pair)
        self.draw()
        self.refresh_panel()

    def move_selected(self, dx, dy):
        item = self.selected()
        if item is None:
            return

        if dx:
            item["x_step"] = max(0, min(self.step_count - item["length"], item["x_step"] + dx))

        if dy:
            item["pair"] = max(0, min(len(self.blocks) - 1, item["pair"] + dy))

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
        self.selected_id = self.pattern[0]["id"]
        self.draw()
        self.refresh_panel()

    def play_selected_pair(self):
        item = self.selected()
        if item is None:
            return
        try:
            import sounddevice as sd
            sd.play(self.get_audio(item["pair"]), SR)
        except Exception:
            messagebox.showwarning("Audio", "Installe sounddevice : python -m pip install sounddevice")

    def render_audio(self):
        # IMPORTANT :
        # Audio comme avant. On ignore length/espacement.
        # On lit uniquement les blocs dans l'ordre horizontal.
        ordered = sorted(self.pattern, key=lambda e: (e["x_step"], e["id"]))
        chunks = [self.get_audio(i["pair"]) for i in ordered]
        return normalize(np.concatenate(chunks)) if chunks else np.zeros(1, dtype=np.float32)

    def render_preview_file(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        wav = OUT_DIR / f"{self.project['safe']}_tracker_app_preview.wav"
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
            sd.play(audio, SR)
            duration_ms = int(len(audio) / SR * 1000)
            self.loop_after_id = self.root.after(max(80, duration_ms), self.loop_tick)
        except Exception:
            self.looping = False
            self.output_label.config(text="Installe sounddevice : python -m pip install sounddevice")

    def toggle_loop_event(self, event=None):
        self.toggle_loop()
        return "break"

    def toggle_loop(self):
        if self.looping:
            self.looping = False
            if self.loop_after_id:
                self.root.after_cancel(self.loop_after_id)
                self.loop_after_id = None
            try:
                import sounddevice as sd
                sd.stop()
            except Exception:
                pass
            self.output_label.config(text="Loop arrêtée.")
            return

        self.looping = True
        self.output_label.config(text="Loop en cours.")
        self.loop_tick()

    def clean_pattern(self):
        return [
            {
                "id": int(i["id"]),
                "x_step": int(i["x_step"]),
                "length": int(i["length"]),
                "pair": int(i["pair"]),
            }
            for i in sorted(self.pattern, key=lambda e: (e["x_step"], e["id"]))
        ]

    def save(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        wav = self.render_preview_file()
        data = {
            "version": "tracker_app_edit_v04",
            "audio_rule": "render sorted by x_step only; length is visual annotation and does not affect audio",
            "source_pair_json": self.project["source_pair_json"],
            "source_audio": self.project["source_audio"],
            "safe": self.project["safe"],
            "grid": {
                "steps": self.step_count,
                "default_case_length": 2,
                "allow_case_length": [1, 2, 3, 4],
                "row_height_equals_case_height": True,
                "rows": [{"pair": b["pair"], "name": b["name"], "audio_path": b["audio_path"]} for b in self.blocks],
            },
            "pattern": self.clean_pattern(),
            "preview_wav": str(wav),
        }
        path = OUT_DIR / f"{self.project['safe']}_tracker_app_edit_v04.json"
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
