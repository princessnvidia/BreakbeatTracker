#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_tracker_editor_v01.py

Interface locale pour corriger le tracker à la main.

But :
- charger les 8 pair_blocks d'un break
- écouter chaque bloc
- modifier la grille tracker
- choisir quelle paire audio joue à chaque case
- sauvegarder une annotation propre JSON
- rendre un WAV preview depuis ta correction

Usage :
    python pipeline/03_tracker_editor_v01.py --source "Amen"

Puis ouvre :
    http://127.0.0.1:8765
"""

from pathlib import Path
import argparse
import json
import mimetypes
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import soundfile as sf


PAIR_BLOCKS_DIR = Path("dataset/pair_blocks_v02")
OUT_DIR = Path("dataset/tracker_edits")
SR = 44100

DEFAULT_PATTERN = [
    {"step": 0, "role": "kick", "pair": 0},
    {"step": 1, "role": "hat", "pair": 1},
    {"step": 2, "role": "snare", "pair": 2},
    {"step": 3, "role": "hat", "pair": 3},
    {"step": 4, "role": "hat", "pair": 4},
    {"step": 5, "role": "kick", "pair": 5},
    {"step": 6, "role": "snare", "pair": 6},
    {"step": 7, "role": "hat", "pair": 7},
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


def load_project(pair_json):
    meta = json.loads(pair_json.read_text(encoding="utf-8"))
    blocks = []

    for block in meta["blocks"]:
        audio_path = Path(block["audio_path"])
        blocks.append({
            "pair": int(block["pair"]),
            "audio_path": str(audio_path),
            "start_ms": float(block["start_ms"]),
            "end_ms": float(block["end_ms"]),
            "duration_ms": float(block["duration_ms"]),
            "rms": float(block.get("rms", 0.0)),
        })

    blocks = sorted(blocks, key=lambda b: b["pair"])

    return {
        "source_pair_json": str(pair_json),
        "source_audio": meta.get("source"),
        "safe": safe_name(pair_json),
        "blocks": blocks,
        "default_pattern": DEFAULT_PATTERN,
    }


def render_preview(project, pattern, loops=4):
    audio_by_pair = {}
    for block in project["blocks"]:
        pair = int(block["pair"])
        audio_by_pair[pair] = load_wav(block["audio_path"])

    chunks = []
    for _ in range(loops):
        for item in pattern:
            chunks.append(audio_by_pair[int(item["pair"])])

    out = normalize(np.concatenate(chunks)) if chunks else np.zeros(1, dtype=np.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wav = OUT_DIR / f"{project['safe']}_tracker_edit_preview.wav"
    sf.write(wav, out, SR)
    return str(wav)


def save_edit(project, payload):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pattern = payload.get("pattern", DEFAULT_PATTERN)
    notes = payload.get("notes", "")
    loops = int(payload.get("loops", 4))

    preview_wav = render_preview(project, pattern, loops=loops)

    data = {
        "version": "tracker_edit_v01",
        "source_pair_json": project["source_pair_json"],
        "source_audio": project["source_audio"],
        "safe": project["safe"],
        "grid": "8 steps, each step = pair block = 2 cases on grid16",
        "roles": ["kick", "hat", "snare", "hat", "hat", "kick", "snare", "hat"],
        "loops_preview": loops,
        "notes": notes,
        "pattern": pattern,
        "preview_wav": preview_wav,
        "blocks": project["blocks"],
    }

    json_path = OUT_DIR / f"{project['safe']}_tracker_edit_v01.json"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return {"ok": True, "json": str(json_path), "preview_wav": preview_wav}


HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>BreakbeatAI Tracker Editor</title>
<style>
:root {
  --bg: #111018;
  --panel: #1b1824;
  --panel2: #241f31;
  --text: #f5eefe;
  --muted: #b9acc8;
  --pink: #ff7acc;
  --green: #77f5b5;
  --blue: #8bbcff;
  --yellow: #ffe08a;
  --red: #ff8b8b;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main { max-width: 1200px; margin: 0 auto; padding: 24px; }
h1 { margin: 0 0 8px; color: var(--pink); }
h2 { margin-top: 28px; color: var(--green); }
small, .muted { color: var(--muted); }
.panel {
  background: var(--panel);
  border: 1px solid #342a44;
  border-radius: 16px;
  padding: 16px;
  margin: 16px 0;
}
.blocks { display: grid; grid-template-columns: repeat(8, 1fr); gap: 10px; }
.block, .step {
  background: var(--panel2);
  border: 1px solid #433652;
  border-radius: 14px;
  padding: 12px;
}
.block strong, .step strong { color: var(--yellow); }
button {
  background: #30283f;
  color: var(--text);
  border: 1px solid #5a496c;
  border-radius: 10px;
  padding: 8px 10px;
  cursor: pointer;
}
button:hover { border-color: var(--pink); color: var(--pink); }
.grid { display: grid; grid-template-columns: repeat(8, 1fr); gap: 10px; }
.step { min-height: 210px; }
.step.active { outline: 2px solid var(--pink); }
.role {
  display: inline-block;
  padding: 3px 7px;
  border-radius: 999px;
  margin: 4px 0;
  font-size: 12px;
}
.role.kick { background: #4b2c2c; color: var(--red); }
.role.hat { background: #454024; color: var(--yellow); }
.role.snare { background: #253e4f; color: var(--blue); }
select, textarea, input {
  width: 100%;
  background: #130f1c;
  color: var(--text);
  border: 1px solid #4b3d5f;
  border-radius: 10px;
  padding: 8px;
}
textarea { min-height: 90px; }
.controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
.out {
  white-space: pre-wrap;
  background: #09070e;
  border-radius: 12px;
  padding: 12px;
  color: var(--green);
}
audio { width: 100%; margin-top: 8px; }
@media (max-width: 900px) { .blocks, .grid { grid-template-columns: repeat(2, 1fr); } }
</style>
</head>
<body>
<main>
  <h1>BreakbeatAI Tracker Editor</h1>
  <div class="muted" id="source"></div>

  <section class="panel">
    <h2>1. Blocs audio disponibles</h2>
    <div class="blocks" id="blocks"></div>
  </section>

  <section class="panel">
    <h2>2. Tracker 8 steps</h2>
    <p class="muted">Chaque step = 2 cases sur une grille 16. Clique un step, change le rôle ou la paire audio.</p>
    <div class="grid" id="grid"></div>
  </section>

  <section class="panel">
    <h2>3. Sauvegarder comme data</h2>
    <div class="controls">
      <label>Loops preview
        <input id="loops" type="number" min="1" max="32" value="4">
      </label>
      <button onclick="saveEdit()">Sauvegarder + rendre preview WAV</button>
      <button onclick="resetPattern()">Reset pattern</button>
    </div>
    <p>Notes :</p>
    <textarea id="notes" placeholder="Ex: step 6 devrait répéter la snare du step 3..."></textarea>
    <h2>Résultat</h2>
    <div class="out" id="output">En attente.</div>
  </section>
</main>

<script>
let project = null;
let pattern = [];
let selectedStep = 0;

function roleClass(role) {
  if (role === "kick") return "kick";
  if (role === "snare") return "snare";
  return "hat";
}

async function loadProject() {
  const res = await fetch("/api/project");
  project = await res.json();
  pattern = JSON.parse(JSON.stringify(project.default_pattern));
  document.getElementById("source").textContent =
    "Source : " + project.source_audio + " | Pair JSON : " + project.source_pair_json;
  renderBlocks();
  renderGrid();
}

function renderBlocks() {
  const root = document.getElementById("blocks");
  root.innerHTML = "";

  project.blocks.forEach(block => {
    const div = document.createElement("div");
    div.className = "block";
    div.innerHTML = `
      <strong>pair ${String(block.pair).padStart(2, "0")}</strong>
      <div class="muted">${block.duration_ms.toFixed(1)} ms</div>
      <audio controls src="/audio?path=${encodeURIComponent(block.audio_path)}"></audio>
      <button onclick="assignPair(${block.pair})">Assigner au step sélectionné</button>
    `;
    root.appendChild(div);
  });
}

function renderGrid() {
  const root = document.getElementById("grid");
  root.innerHTML = "";

  pattern.forEach((item, idx) => {
    const div = document.createElement("div");
    div.className = "step" + (idx === selectedStep ? " active" : "");
    div.onclick = () => { selectedStep = idx; renderGrid(); };

    div.innerHTML = `
      <strong>Step ${idx + 1}</strong>
      <div><span class="role ${roleClass(item.role)}">${item.role}</span></div>
      <label>Rôle
        <select onchange="setRole(${idx}, this.value)">
          <option value="kick" ${item.role === "kick" ? "selected" : ""}>kick</option>
          <option value="hat" ${item.role === "hat" ? "selected" : ""}>hihat</option>
          <option value="snare" ${item.role === "snare" ? "selected" : ""}>snare</option>
        </select>
      </label>
      <label>Pair audio
        <select onchange="setPair(${idx}, this.value)">
          ${project.blocks.map(b => `<option value="${b.pair}" ${item.pair === b.pair ? "selected" : ""}>pair ${String(b.pair).padStart(2, "0")}</option>`).join("")}
        </select>
      </label>
      <audio controls src="/audio?path=${encodeURIComponent(project.blocks[item.pair].audio_path)}"></audio>
    `;
    root.appendChild(div);
  });
}

function setRole(idx, role) {
  pattern[idx].role = role;
  renderGrid();
}

function setPair(idx, pair) {
  pattern[idx].pair = parseInt(pair);
  renderGrid();
}

function assignPair(pair) {
  pattern[selectedStep].pair = pair;
  renderGrid();
}

function resetPattern() {
  pattern = JSON.parse(JSON.stringify(project.default_pattern));
  selectedStep = 0;
  renderGrid();
}

async function saveEdit() {
  const payload = {
    pattern,
    notes: document.getElementById("notes").value,
    loops: parseInt(document.getElementById("loops").value || "4")
  };

  const res = await fetch("/api/save", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });

  const data = await res.json();
  document.getElementById("output").textContent =
    "OK\\nJSON: " + data.json + "\\nPreview WAV: " + data.preview_wav;
}

loadProject();
</script>
</body>
</html>
"""


class TrackerHandler(BaseHTTPRequestHandler):
    project = None

    def send_json(self, data):
        raw = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            raw = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if parsed.path == "/api/project":
            self.send_json(self.project)
            return

        if parsed.path == "/audio":
            qs = parse_qs(parsed.query)
            path = Path(qs.get("path", [""])[0])

            if not path.exists():
                self.send_error(404, "audio not found")
                return

            raw = path.read_bytes()
            ctype = mimetypes.guess_type(str(path))[0] or "audio/wav"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/save":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            result = save_edit(self.project, payload)
            self.send_json(result)
            return

        self.send_error(404)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Amen")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    pair_json = find_pair_json(args.source)
    project = load_project(pair_json)

    TrackerHandler.project = project

    server = HTTPServer(("127.0.0.1", args.port), TrackerHandler)
    url = f"http://127.0.0.1:{args.port}"

    print("Tracker Editor lancé.")
    print("Source :", project["source_audio"])
    print("URL    :", url)
    print("")
    print("Ctrl+C pour arrêter.")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt.")


if __name__ == "__main__":
    main()
