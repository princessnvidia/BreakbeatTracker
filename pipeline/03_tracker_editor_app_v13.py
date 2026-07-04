#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_tracker_editor_app_v13.py

BreakbeatAI Tracker Editor v13

But :
- Ne plus envoyer les pairs douteuses en HI-HAT par défaut.
- Ignorer les role_guess faibles du slicer.
- Reclasser les pairs avec une analyse audio stricte.
- Ajouter une ligne À CLASSER pour les sons ambigus.
- Garder :
    * audio backend robuste
    * loop gapless
    * playhead
    * overrides manuels

Usage :
    cd ~/Applications/BreakbeatAI
    python pipeline/03_tracker_editor_app_v13.py --source "amen"
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
    {"key": "unknown", "role": "unknown", "label": "À CLASSER", "color": "#8f8f9d"},
    {"key": "hat_a", "role": "hat", "label": "HI-HAT A", "color": "#ffd37a"},
    {"key": "hat_b", "role": "hat", "label": "HI-HAT B", "color": "#ffd37a"},
    {"key": "kick_a", "role": "kick", "label": "KICK A", "color": "#ee8fa7"},
    {"key": "kick_b", "role": "kick", "label": "KICK B", "color": "#ee8fa7"},
    {"key": "snare_a", "role": "snare", "label": "SNARE A", "color": "#8bbcff"},
    {"key": "snare_b", "role": "snare", "label": "SNARE B", "color": "#8bbcff"},
]

ROLE_TO_LANES = {
    "unknown": [0],
    "hat": [1, 2],
    "kick": [3, 4],
    "snare": [5, 6],
}

LANE_CHOICES = [f"{i}: {lane['label']}" for i, lane in enumerate(LANES)]

DEFAULT_PATTERN = [
    {"id": 0, "x_step": 0, "lane": 3, "role": "kick", "length": 2, "pair": 0},
    {"id": 1, "x_step": 2, "lane": 0, "role": "unknown", "length": 2, "pair": 1},
    {"id": 2, "x_step": 4, "lane": 5, "role": "snare", "length": 2, "pair": 2},
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


def load_wav(path):
    audio, sr = sf.read(path, always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    if sr != SR:
        raise RuntimeError(f"Sample rate inattendu {sr} pour {path}, attendu {SR}")

    return fade(normalize(audio), ms=2)


def clean_role(role):
    role = str(role).lower().strip()

    if role in ("unknown", "unk", "?", "a_classer", "à classer", "classer"):
        return "unknown"

    if role in ("hihat", "hi-hat", "hi_hat", "hat", "hh"):
        return "hat"

    if role in ("kick", "bd", "bassdrum", "bass_drum"):
        return "kick"

    if role in ("snare", "sd", "rim", "clap"):
        return "snare"

    return None


def lane_info(lane_index):
    lane_index = max(0, min(len(LANES) - 1, int(lane_index)))
    return LANES[lane_index]


def rms(y):
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0:
        return 0.0
    return float(np.sqrt(np.mean(y * y) + 1e-12))


def spectral_features(y):
    y = np.asarray(y, dtype=np.float32)
    n = min(len(y), int(0.24 * SR))

    if n < 128:
        return {
            "role": "unknown",
            "confidence": 0.0,
            "reason": "too_short",
            "centroid_hz": 0.0,
            "low_ratio": 0.0,
            "body_ratio": 0.0,
            "mid_ratio": 0.0,
            "high_ratio": 0.0,
        }

    seg = y[:n]
    seg = seg - float(np.mean(seg))
    win = np.hanning(n).astype(np.float32)
    mag = np.abs(np.fft.rfft(seg * win)).astype(np.float32)
    freqs = np.fft.rfftfreq(n, 1 / SR)

    total = float(mag.sum() + 1e-9)
    low = float(mag[freqs < 180].sum() / total)
    body = float(mag[(freqs >= 180) & (freqs < 900)].sum() / total)
    mid = float(mag[(freqs >= 900) & (freqs < 4500)].sum() / total)
    high = float(mag[freqs >= 4500].sum() / total)
    centroid = float((freqs * mag).sum() / total)

    attack_n = max(32, int(0.030 * SR))
    tail_a = min(len(seg), int(0.050 * SR))
    tail_b = min(len(seg), int(0.180 * SR))
    attack_rms = rms(seg[:attack_n])
    tail_rms = rms(seg[tail_a:tail_b]) if tail_b > tail_a else 0.0
    tail_ratio = tail_rms / max(attack_rms, 1e-9)

    role = "unknown"
    conf = 0.0
    reason = "ambiguous"

    # Très strict pour HI-HAT : on évite le faux rangement massif en hi-hat.
    if high > 0.55 and low < 0.10 and body < 0.20 and centroid > 5600:
        role = "hat"
        conf = min(1.0, 0.45 + high + max(0.0, (centroid - 5600) / 10000))
        reason = "strong_high_low_low"

    elif (low + body) > 0.42 and low > 0.16 and high < 0.33 and centroid < 3800:
        role = "kick"
        conf = min(1.0, 0.40 + low + body + max(0.0, (3800 - centroid) / 8000))
        reason = "low_body_dominant"

    elif (mid + high) > 0.48 and mid > 0.18 and high > 0.16 and low < 0.32:
        role = "snare"
        conf = min(1.0, 0.35 + mid + 0.6 * high)
        reason = "mid_high_broadband"

    elif high > 0.45 and low < 0.08 and centroid > 4800 and tail_ratio < 0.95:
        role = "hat"
        conf = min(0.82, 0.35 + high)
        reason = "likely_hat_but_strict"

    return {
        "role": role,
        "confidence": round(float(conf), 4),
        "reason": reason,
        "centroid_hz": round(float(centroid), 2),
        "low_ratio": round(low, 4),
        "body_ratio": round(body, 4),
        "mid_ratio": round(mid, 4),
        "high_ratio": round(high, 4),
        "tail_ratio": round(float(tail_ratio), 4),
    }


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
        self.seed_known_overrides()

        self.audio_cache = {}
        self.pair_role_analysis = {}
        self.reanalyse_pairs()

        self.pattern = self.build_initial_pattern_from_pairs()
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

        self.looping = False
        self.loop_after_id = None
        self.play_process = None
        self.external_player = self.detect_external_player()

        self.loop_target_sec = 180.0
        self.loop_max_repeats = 256

        self.playhead_after_id = None
        self.loop_started_at = None
        self.current_loop_sec = 0.0
        self.current_timeline = []

        self.root.title("BreakbeatAI Tracker Editor v13")
        self.root.geometry("1540x850")
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
            role_guess = clean_role(block.get("manual_role") or block.get("role_guess") or "")
            conf = float(block.get("role_confidence", 0.0) or 0.0)
            blocks.append({
                "pair": pair,
                "name": f"pair {pair:02d}",
                "audio_path": str(Path(block["audio_path"])),
                "duration_ms": float(block.get("duration_ms", 0.0)),
                "slicer_role_guess": role_guess,
                "slicer_role_confidence": conf,
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

    def seed_known_overrides(self):
        # Confirmé pendant le test : pair 1 est un kick.
        if 1 in self.pair_values and "1" not in self.pair_role_overrides:
            self.pair_role_overrides["1"] = "kick"
            self.save_pair_role_overrides()
            print("[v13] Override auto confirmé : pair 1 = kick")

    def save_pair_role_overrides(self):
        self.role_override_path.write_text(
            json.dumps(self.pair_role_overrides, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def detect_external_player(self):
        for name in ["pw-play", "paplay", "aplay", "ffplay"]:
            path = shutil.which(name)
            if path:
                print(f"[v13] Backend audio système trouvé : {name} -> {path}")
                return name

        print("[v13] Aucun backend système trouvé, fallback sounddevice.")
        return None

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
        self.root.bind_all("u", lambda e: self.set_selected_pair_role("unknown"))
        print("[v13] Raccourcis : Espace loop | h/k/s/u = classer pair sélectionnée")

    def force_keyboard_focus(self):
        try:
            self.root.focus_force()
            self.canvas.focus_set()
        except Exception:
            pass

    def build_ui(self):
        main = tk.Frame(self.root, bg="#111018")
        main.pack(fill="both", expand=True, padx=14, pady=14)

        title = tk.Label(
            main,
            text="BreakbeatAI Tracker Editor v13 — anti faux HI-HAT + ligne À CLASSER",
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
                "Les guesses faibles ne partent plus en HI-HAT : ils vont dans À CLASSER."
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
        self.info_label.grid(row=0, column=0, columnspan=15, sticky="w", padx=10, pady=8)

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

        self.lane_var = tk.StringVar(value=LANE_CHOICES[0])
        self.lane_box = ttk.Combobox(
            panel,
            textvariable=self.lane_var,
            values=LANE_CHOICES,
            width=16,
            state="readonly",
        )
        self.lane_box.grid(row=1, column=3, padx=5)
        self.lane_box.bind("<<ComboboxSelected>>", self.set_lane_from_choice)

        tk.Button(panel, text="Pair = UNKNOWN", command=lambda: self.set_selected_pair_role("unknown"), bg="#3b3b48", fg="#f5eefe").grid(row=1, column=4, padx=4)
        tk.Button(panel, text="Pair = HAT", command=lambda: self.set_selected_pair_role("hat"), bg="#4b3d23", fg="#f5eefe").grid(row=1, column=5, padx=4)
        tk.Button(panel, text="Pair = KICK", command=lambda: self.set_selected_pair_role("kick"), bg="#4a2630", fg="#f5eefe").grid(row=1, column=6, padx=4)
        tk.Button(panel, text="Pair = SNARE", command=lambda: self.set_selected_pair_role("snare"), bg="#263a56", fg="#f5eefe").grid(row=1, column=7, padx=4)
        tk.Button(panel, text="Clear override", command=self.clear_selected_pair_override, bg="#30283f", fg="#f5eefe").grid(row=1, column=8, padx=4)
        tk.Button(panel, text="Play pair", command=self.play_selected_pair, bg="#30283f", fg="#f5eefe").grid(row=1, column=9, padx=4)
        tk.Button(panel, text="Play Loop / Space", command=self.toggle_loop, bg="#30283f", fg="#f5eefe").grid(row=1, column=10, padx=4)
        tk.Button(panel, text="Auto reclasser", command=self.auto_reclassify_pattern, bg="#30513f", fg="#f5eefe").grid(row=1, column=11, padx=4)
        tk.Button(panel, text="Save", command=self.save, bg="#30513f", fg="#f5eefe").grid(row=1, column=12, padx=4)
        tk.Button(panel, text="Delete", command=self.delete_selected, bg="#4a2630", fg="#f5eefe").grid(row=1, column=13, padx=4)
        tk.Button(panel, text="Reset", command=self.reset, bg="#30283f", fg="#f5eefe").grid(row=1, column=14, padx=4)

        self.output_label = tk.Label(
            panel,
            text="v13 : les sons ambigus vont dans À CLASSER, pas en HI-HAT. Pair 1 reste override KICK.",
            bg="#1b1824",
            fg="#77f5b5",
            justify="left",
        )
        self.output_label.grid(row=2, column=0, columnspan=15, sticky="w", padx=10, pady=8)

    def get_audio(self, pair):
        pair = int(pair)

        if pair not in self.audio_cache:
            if pair not in self.block_by_pair:
                raise RuntimeError(f"Pair audio introuvable : {pair}")

            self.audio_cache[pair] = load_wav(self.block_by_pair[pair]["audio_path"])

        return self.audio_cache[pair]

    def reanalyse_pairs(self):
        self.pair_role_analysis = {}

        for pair in self.pair_values:
            try:
                audio = self.get_audio(pair)
                feat = spectral_features(audio)
            except Exception as exc:
                feat = {
                    "role": "unknown",
                    "confidence": 0.0,
                    "reason": f"error: {exc}",
                    "centroid_hz": 0.0,
                    "low_ratio": 0.0,
                    "body_ratio": 0.0,
                    "mid_ratio": 0.0,
                    "high_ratio": 0.0,
                }

            self.pair_role_analysis[int(pair)] = feat

        counts = {}
        for feat in self.pair_role_analysis.values():
            counts[feat["role"]] = counts.get(feat["role"], 0) + 1
        print("[v13] Analyse audio stricte :", counts)

    def get_pair_role(self, pair):
        pair = int(pair)

        override = self.pair_role_overrides.get(str(pair))
        if override:
            return override, "override", 1.0

        feat = self.pair_role_analysis.get(pair)
        if feat:
            role = clean_role(feat.get("role")) or "unknown"
            conf = float(feat.get("confidence", 0.0) or 0.0)

            if role != "unknown" and conf >= 0.62:
                return role, "audio_strict", conf

            return "unknown", "audio_uncertain", conf

        return "unknown", "missing", 0.0

    def role_to_lane(self, role, pair=None, occurrence=0):
        role = clean_role(role) or "unknown"
        lanes = ROLE_TO_LANES.get(role, [0])

        if len(lanes) == 1:
            return lanes[0]

        if pair is not None:
            return lanes[int(pair) % len(lanes)]

        return lanes[int(occurrence) % len(lanes)]

    def build_initial_pattern_from_pairs(self):
        if not self.pair_values:
            return json.loads(json.dumps(DEFAULT_PATTERN))

        pattern = []
        role_counts = {"unknown": 0, "hat": 0, "kick": 0, "snare": 0}

        for idx, pair in enumerate(self.pair_values[:32]):
            role, source, conf = self.get_pair_role(pair)
            role = clean_role(role) or "unknown"
            lane = self.role_to_lane(role, pair=pair, occurrence=role_counts.get(role, 0))
            role_counts[role] = role_counts.get(role, 0) + 1

            pattern.append({
                "id": idx,
                "x_step": idx * 2,
                "lane": lane,
                "role": role,
                "role_source": source,
                "role_confidence": conf,
                "length": 2,
                "pair": int(pair),
            })

        return pattern

    def auto_reclassify_pattern(self):
        self.reanalyse_pairs()

        moved = 0
        for item in self.pattern:
            pair = int(item["pair"])
            role, source, conf = self.get_pair_role(pair)
            role = clean_role(role) or "unknown"
            lane = self.role_to_lane(role, pair=pair)

            if int(item.get("lane", 0)) != lane:
                moved += 1

            item["lane"] = lane
            item["role"] = role
            item["role_source"] = source
            item["role_confidence"] = conf

        self.draw()
        self.refresh_panel()
        self.output_label.config(text=f"Auto reclassé : {moved} case(s) déplacée(s). Les incertaines vont dans À CLASSER.")

    def reset(self):
        self.reanalyse_pairs()
        self.pattern = self.build_initial_pattern_from_pairs()
        self.selected_id = self.pattern[0]["id"] if self.pattern else None
        self.draw()
        self.refresh_panel()

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
        live_wav = OUT_DIR / f"{self.project['safe']}_v13_live.wav"
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
                print(f"[v13] {label} lancé avec {self.external_player} : {live_wav}")
                return duration_ms
            except Exception as exc:
                print(f"[v13] Erreur backend système {self.external_player} : {exc}")

        try:
            import sounddevice as sd
            sd.play(audio, SR)
            self.output_label.config(text=f"{label} lancé avec sounddevice — {duration_ms} ms")
            print(f"[v13] {label} lancé avec sounddevice")
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
        return max(0, min(len(LANES) - 1, int(y // self.row_height)))

    def selected(self):
        for item in self.pattern:
            if int(item["id"]) == int(self.selected_id):
                return item
        return None

    def new_id(self):
        return max([int(i["id"]) for i in self.pattern], default=-1) + 1

    def draw(self):
        self.canvas.delete("all")

        for lane_index, lane in enumerate(LANES):
            y0 = lane_index * self.row_height
            y1 = y0 + self.row_height
            row_fill = "#252525" if lane_index % 2 == 0 else "#202020"

            self.canvas.create_rectangle(0, y0, self.canvas_width, y1, fill=row_fill, outline="#343434")
            self.canvas.create_rectangle(0, y0, self.left_width, y1, fill="#17131f", outline="#343044")
            self.canvas.create_text(
                12,
                y0 + self.row_height / 2,
                text=lane["label"],
                fill=lane["color"],
                anchor="w",
                font=("Sans", 11, "bold"),
            )

            if lane_index in [0, 2, 4]:
                self.canvas.create_line(0, y1, self.canvas_width, y1, fill="#5b4a6d", width=2)

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

        self.canvas.create_line(self.left_width, 0, self.left_width, self.canvas_height, fill="#888888", width=2)

        for item in sorted(self.pattern, key=lambda e: (int(e["x_step"]), int(e["lane"]), int(e["id"]))):
            self.draw_case(item)

    def draw_case(self, item):
        lane_index = max(0, min(len(LANES) - 1, int(item.get("lane", 0))))
        lane = lane_info(lane_index)

        x0 = self.step_to_x(int(item["x_step"]))
        x1 = self.step_to_x(int(item["x_step"]) + int(item["length"]))
        y0 = lane_index * self.row_height + self.case_y_padding
        y1 = y0 + self.case_height

        outline = "#77f5b5" if int(item["id"]) == int(self.selected_id) else "#ffc0cf"
        width = 3 if int(item["id"]) == int(self.selected_id) else 1
        tags = ("case", f"id_{item['id']}")

        self.canvas.create_rectangle(x0, y0, x1, y1, fill=lane["color"], outline=outline, width=width, tags=tags)

        pair = int(item["pair"])
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

        item_id = self.get_item_at(event.x, event.y)

        if item_id is None:
            if event.x >= self.left_width:
                step = self.x_to_step(event.x)
                lane_index = self.y_to_lane(event.y)
                lane = lane_info(lane_index)
                selected = self.selected()

                pair = int(selected["pair"]) if selected is not None else int(self.pair_values[0] if self.pair_values else 0)

                new_item = {
                    "id": self.new_id(),
                    "x_step": step,
                    "lane": lane_index,
                    "role": lane["role"],
                    "role_source": "manual_lane",
                    "role_confidence": 1.0,
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
            item["x_step"] = max(0, min(self.step_count - item["length"], self.drag_start_step + delta_steps))
            item["lane"] = self.y_to_lane(event.y)
            item["role"] = lane_info(item["lane"])["role"]
            item["role_source"] = "manual_lane"
            item["role_confidence"] = 1.0

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
        slicer_guess = block.get("slicer_role_guess") or "?"
        slicer_conf = block.get("slicer_role_confidence", 0.0)
        feat = self.pair_role_analysis.get(pair, {})
        final_role, source, final_conf = self.get_pair_role(pair)
        override = self.pair_role_overrides.get(str(pair))
        override_text = f" | override {override}" if override else ""

        self.info_label.config(
            text=(
                f"id {item['id']} | step {item['x_step']} | length {item['length']} | "
                f"ligne {lane['label']} | pair {pair} | final {final_role} ({source}, {final_conf:.2f}){override_text} | "
                f"slicer {slicer_guess}/{slicer_conf:.2f} | "
                f"audio {feat.get('role', '?')}/{feat.get('confidence', 0):.2f} {feat.get('reason', '')}"
            )
        )

        self.pair_var.set(pair)
        self.lane_var.set(LANE_CHOICES[lane_index])

    def set_pair(self, pair):
        item = self.selected()
        if item is None:
            return

        pair = int(pair)
        role, source, conf = self.get_pair_role(pair)
        lane = self.role_to_lane(role, pair=pair)

        item["pair"] = pair
        item["lane"] = lane
        item["role"] = role
        item["role_source"] = source
        item["role_confidence"] = conf

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
        item["role_source"] = "manual_lane"
        item["role_confidence"] = 1.0

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

        target_lane = self.role_to_lane(role, pair=pair)
        moved = 0

        for case in self.pattern:
            if int(case["pair"]) == pair:
                case["lane"] = target_lane
                case["role"] = role
                case["role_source"] = "override"
                case["role_confidence"] = 1.0
                moved += 1

        self.draw()
        self.refresh_panel()
        self.output_label.config(text=f"Correction sauvegardée : pair {pair} = {role.upper()} → {lane_info(target_lane)['label']} | {moved} case(s) déplacée(s).")
        print(f"[v13] override pair {pair} = {role} -> {self.role_override_path}")

    def clear_selected_pair_override(self):
        item = self.selected()
        if item is None:
            return

        pair = int(item["pair"])
        if str(pair) in self.pair_role_overrides:
            del self.pair_role_overrides[str(pair)]
            self.save_pair_role_overrides()

        role, source, conf = self.get_pair_role(pair)
        lane = self.role_to_lane(role, pair=pair)

        moved = 0
        for case in self.pattern:
            if int(case["pair"]) == pair:
                case["lane"] = lane
                case["role"] = role
                case["role_source"] = source
                case["role_confidence"] = conf
                moved += 1

        self.draw()
        self.refresh_panel()
        self.output_label.config(text=f"Override retiré pour pair {pair}. Nouveau rôle : {role} ({source}). {moved} case(s) mise(s) à jour.")

    def move_selected(self, dx, dy):
        item = self.selected()
        if item is None:
            return

        if dx:
            item["x_step"] = max(0, min(self.step_count - item["length"], item["x_step"] + dx))

        if dy:
            lane_index = max(0, min(len(LANES) - 1, int(item.get("lane", 0)) + dy))
            item["lane"] = lane_index
            item["role"] = lane_info(lane_index)["role"]
            item["role_source"] = "manual_lane"
            item["role_confidence"] = 1.0

        self.draw()
        self.refresh_panel()

    def delete_selected(self):
        if self.selected_id is None:
            return

        self.pattern = [i for i in self.pattern if int(i["id"]) != int(self.selected_id)]
        self.selected_id = self.pattern[0]["id"] if self.pattern else None

        self.draw()
        self.refresh_panel()

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
        wav = OUT_DIR / f"{self.project['safe']}_tracker_app_v13_preview.wav"
        sf.write(wav, self.render_audio(), SR)
        return wav

    def playhead_x_for_time(self, t_sec):
        if not self.current_timeline:
            return self.left_width

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
        self.canvas.create_line(x, 0, x, self.canvas_height, fill="#77f5b5", width=3, tags=("playhead",))
        self.canvas.create_polygon(x - 7, 0, x + 7, 0, x, 13, fill="#77f5b5", outline="", tags=("playhead",))

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
        duration_ms = self.play_audio_array(audio, label=f"Loop gapless x{repeats}")

        if duration_ms <= 0:
            self.looping = False
            self.stop_playhead()
            return

        self.start_playhead(one_loop_sec, timeline)
        self.output_label.config(text=f"Loop + playhead : motif {one_loop_ms} ms répété {repeats}x. Stop/Start pour recharger les changements.")
        self.loop_after_id = self.root.after(max(1000, duration_ms + 20), self.loop_tick)

    def toggle_loop_event(self, event=None):
        print("[v13] Espace détecté")
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
            print("[v13] Loop arrêtée")
            return

        self.stop_playhead()
        print("[v13] Lancement loop...")
        self.looping = True
        self.output_label.config(text="Lancement loop...")
        self.loop_tick()

    def clean_pattern(self):
        out = []

        for item in sorted(self.pattern, key=lambda e: (int(e["x_step"]), int(e["id"]))):
            lane_index = max(0, min(len(LANES) - 1, int(item.get("lane", 0))))
            lane = lane_info(lane_index)
            pair = int(item["pair"])
            final_role, source, conf = self.get_pair_role(pair)

            out.append({
                "id": int(item["id"]),
                "x_step": int(item["x_step"]),
                "lane": int(lane_index),
                "lane_key": lane["key"],
                "lane_label": lane["label"],
                "role": lane["role"],
                "pair_role": final_role,
                "pair_role_source": source,
                "pair_role_confidence": conf,
                "length": int(item["length"]),
                "pair": pair,
            })

        return out

    def save(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        wav = self.render_preview_file()

        data = {
            "version": "tracker_app_edit_v13_anti_false_hihat_unknown_lane",
            "audio_rule": "render sorted by x_step only; lane/role/length are visual annotations and do not affect audio",
            "audio_backend": self.external_player if self.external_player else "sounddevice",
            "source_pair_json": self.project["source_pair_json"],
            "source_audio": self.project["source_audio"],
            "safe": self.project["safe"],
            "pair_role_overrides_path": str(self.role_override_path),
            "pair_role_overrides": self.pair_role_overrides,
            "strict_audio_analysis": self.pair_role_analysis,
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
                "unknown_lane_for_uncertain_pairs": True,
                "strict_hat_classification": True,
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
                        "slicer_role_guess": b.get("slicer_role_guess"),
                        "slicer_role_confidence": b.get("slicer_role_confidence"),
                        "strict_audio_analysis": self.pair_role_analysis.get(int(b["pair"])),
                        "role_override": self.pair_role_overrides.get(str(b["pair"])),
                    }
                    for b in self.blocks
                ],
            },
            "pattern": self.clean_pattern(),
            "preview_wav": str(wav),
        }

        path = OUT_DIR / f"{self.project['safe']}_tracker_app_edit_v13.json"
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
