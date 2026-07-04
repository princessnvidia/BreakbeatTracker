#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_tracker_editor_v02_image_grid.py

Éditeur tracker visuel basé sur une image de référence.

Nouveauté :
- le tracker reprend la logique de ton image :
    ligne haut   = hihat
    ligne milieu = kick
    ligne bas    = snare
- tu peux monter/descendre chaque bloc :
    hihat <-> kick <-> snare
- tu peux changer la paire audio jouée par chaque bloc
- sauvegarde une annotation JSON
- rend un WAV preview

Usage :
    python pipeline/03_tracker_editor_v02_image_grid.py --source "Amen" --image "break(1).png"

Si ton image est dans Téléchargements :
    python pipeline/03_tracker_editor_v02_image_grid.py --source "Amen" --image ~/Téléchargements/break.png

Puis ouvre :
    http://127.0.0.1:8766
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
from PIL import Image


PAIR_BLOCKS_DIR = Path("dataset/pair_blocks_v02")
OUT_DIR = Path("dataset/tracker_edits")
SR = 44100

ROLE_ORDER = ["hat", "kick", "snare"]
ROLE_TO_Y = {"hat": 0, "kick": 1, "snare": 2}
Y_TO_ROLE = {0: "hat", 1: "kick", 2: "snare"}


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


def pink_mask(img):
    arr = np.array(img.convert("RGB"))
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    return (
        (r > 170)
        & (g > 55)
        & (g < 205)
        & (b > 75)
        & (b < 235)
        & (r > g + 20)
    )


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
    img = Image.open(Path(image_path).expanduser())
    mask = pink_mask(img)
    lanes = detect_lanes(mask)

    if lanes is None:
        return default_visual_pattern(), {
            "image": str(image_path),
            "reason": "fallback_default_no_lanes",
        }

    events = []
    for role, (y1, y2) in lanes.items():
        lane_mask = mask[y1:y2 + 1, :]
        col_strength = lane_mask.sum(axis=0)
        xs = np.where(col_strength > 2)[0]
        intervals = group_contiguous(list(xs), min_len=3)

        for x1, x2 in intervals:
            events.append({
                "role": role,
                "lane": ROLE_TO_Y[role],
                "x1": int(x1),
                "x2": int(x2),
                "xc": float((x1 + x2) / 2),
            })

    events.sort(key=lambda e: (e["x1"], e["xc"]))

    if not events:
        return default_visual_pattern(), {
            "image": str(image_path),
            "reason": "fallback_default_no_events",
        }

    x_min = min(e["x1"] for e in events)
    x_max = max(e["x2"] for e in events)
    span = max(1, x_max - x_min)

    pattern = []
    for i, e in enumerate(events):
        x_norm = (e["xc"] - x_min) / span
        pair = i % 8
        pattern.append({
            "id": i,
            "x": float(x_norm),
            "lane": int(e["lane"]),
            "role": e["role"],
            "pair": int(pair),
            "width": float(max(0.018, (e["x2"] - e["x1"]) / span)),
        })

    return pattern, {
        "image": str(image_path),
        "reason": "ok",
        "image_size": list(img.size),
        "lanes": {k: [int(v[0]), int(v[1])] for k, v in lanes.items()},
        "event_count": len(pattern),
        "x_min": int(x_min),
        "x_max": int(x_max),
    }


def default_visual_pattern():
    roles = ["kick", "hat", "snare", "hat", "hat", "kick", "snare", "hat"]
    return [
        {
            "id": i,
            "x": i / 7,
            "lane": ROLE_TO_Y[role],
            "role": role,
            "pair": i,
            "width": 0.045,
        }
        for i, role in enumerate(roles)
    ]


def load_project(pair_json, image_path):
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
    pattern, image_meta = pattern_from_image(image_path)

    return {
        "source_pair_json": str(pair_json),
        "source_audio": meta.get("source"),
        "safe": safe_name(pair_json),
        "image": str(image_path),
        "image_meta": image_meta,
        "blocks": blocks,
        "pattern": pattern,
    }


def render_preview(project, pattern, loops=1):
    audio_by_pair = {}
    for block in project["blocks"]:
        audio_by_pair[int(block["pair"])] = load_wav(block["audio_path"])

    ordered = sorted(pattern, key=lambda e: e["x"])
    chunks = []

    for _ in range(loops):
        for item in ordered:
            chunks.append(audio_by_pair[int(item["pair"])])

    out = normalize(np.concatenate(chunks)) if chunks else np.zeros(1, dtype=np.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wav = OUT_DIR / f"{project['safe']}_visual_tracker_preview.wav"
    sf.write(wav, out, SR)
    return str(wav)


def save_edit(project, payload):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pattern = payload.get("pattern", project["pattern"])
    notes = payload.get("notes", "")
    loops = int(payload.get("loops", 1))

    for item in pattern:
        item["role"] = Y_TO_ROLE.get(int(item["lane"]), "hat")

    preview_wav = render_preview(project, pattern, loops=loops)

    data = {
        "version": "visual_tracker_edit_v02",
        "source_pair_json": project["source_pair_json"],
        "source_audio": project["source_audio"],
        "safe": project["safe"],
        "image": project["image"],
        "image_meta": project["image_meta"],
        "grid": "visual tracker from reference image, 3 lanes hat/kick/snare",
        "loops_preview": loops,
        "notes": notes,
        "pattern": sorted(pattern, key=lambda e: e["x"]),
        "preview_wav": preview_wav,
        "blocks": project["blocks"],
    }

    json_path = OUT_DIR / f"{project['safe']}_visual_tracker_edit_v02.json"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return {"ok": True, "json": str(json_path), "preview_wav": preview_wav}


HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>BreakbeatAI Visual Tracker</title>
<style>
:root {
  --bg: #111018;
  --panel: #1b1824;
  --text: #f5eefe;
  --muted: #b9acc8;
  --pink: #ff7acc;
  --green: #77f5b5;
  --blue: #8bbcff;
  --yellow: #ffe08a;
  --red: #ff8b8b;
  --grid: #343044;
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font-family:system-ui,sans-serif; }
main { max-width: 1300px; margin:0 auto; padding:24px; }
h1 { color:var(--pink); margin:0 0 8px; }
h2 { color:var(--green); }
.panel { background:var(--panel); border:1px solid #342a44; border-radius:16px; padding:16px; margin:16px 0; }
.muted { color:var(--muted); }
.tracker {
  position: relative;
  height: 250px;
  background:
    linear-gradient(to right, rgba(255,255,255,.08) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(255,255,255,.08) 1px, transparent 1px);
  background-size: calc(100% / 32) 100%, 100% calc(100% / 3);
  border: 1px solid var(--grid);
  border-radius: 14px;
  overflow: hidden;
}
.lane-label {
  position:absolute; left:8px; padding:3px 8px; border-radius:999px; font-size:12px;
  background:#211a2d; color:var(--muted); z-index:5;
}
.block {
  position:absolute;
  height: 30px;
  min-width: 18px;
  background: #ee8fa7;
  border:1px solid #ffc0cf;
  border-radius:3px;
  cursor: grab;
  display:flex;
  align-items:center;
  justify-content:center;
  color:#1a0d14;
  font-size:11px;
  font-weight:700;
  user-select:none;
}
.block.selected { outline:3px solid var(--green); z-index:10; }
.blocks {
  display:grid;
  grid-template-columns: repeat(8, 1fr);
  gap:10px;
}
.sample {
  background:#241f31;
  border:1px solid #433652;
  border-radius:14px;
  padding:10px;
}
button {
  background:#30283f; color:var(--text); border:1px solid #5a496c;
  border-radius:10px; padding:8px 10px; cursor:pointer;
}
button:hover { border-color:var(--pink); color:var(--pink); }
select, textarea, input {
  width:100%; background:#130f1c; color:var(--text);
  border:1px solid #4b3d5f; border-radius:10px; padding:8px;
}
textarea { min-height:80px; }
.controls { display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; align-items:end; }
.out { white-space:pre-wrap; background:#09070e; border-radius:12px; padding:12px; color:var(--green); }
audio { width:100%; margin-top:8px; }
@media(max-width:900px){ .blocks,.controls{grid-template-columns:1fr 1fr;} }
</style>
</head>
<body>
<main>
  <h1>BreakbeatAI Visual Tracker</h1>
  <div id="source" class="muted"></div>

  <section class="panel">
    <h2>1. Tracker visuel</h2>
    <p class="muted">Glisse les blocs horizontalement. Utilise ↑ / ↓ ou les boutons pour monter/descendre : hat / kick / snare.</p>
    <div id="tracker" class="tracker">
      <div class="lane-label" style="top:18px;">HIHAT</div>
      <div class="lane-label" style="top:100px;">KICK</div>
      <div class="lane-label" style="top:182px;">SNARE</div>
    </div>
  </section>

  <section class="panel">
    <h2>2. Bloc sélectionné</h2>
    <div class="controls">
      <button onclick="moveLane(-1)">Monter ↑</button>
      <button onclick="moveLane(1)">Descendre ↓</button>
      <label>Pair audio
        <select id="pairSelect" onchange="changePair(this.value)"></select>
      </label>
      <button onclick="playSelected()">Écouter pair</button>
    </div>
  </section>

  <section class="panel">
    <h2>3. Samples disponibles</h2>
    <div id="samples" class="blocks"></div>
  </section>

  <section class="panel">
    <h2>4. Sauvegarder annotation</h2>
    <div class="controls">
      <label>Loops preview
        <input id="loops" type="number" min="1" max="16" value="1">
      </label>
      <button onclick="saveEdit()">Sauvegarder + preview</button>
      <button onclick="sortPattern()">Trier horizontalement</button>
      <button onclick="resetPattern()">Reset image</button>
    </div>
    <p>Notes :</p>
    <textarea id="notes"></textarea>
    <h2>Résultat</h2>
    <div id="output" class="out">En attente.</div>
  </section>
</main>

<script>
let project = null;
let pattern = [];
let selectedId = null;
let drag = null;

function laneTop(lane) {
  const h = document.getElementById("tracker").clientHeight;
  const laneH = h / 3;
  return lane * laneH + laneH / 2 - 15;
}

function xLeft(x) {
  const w = document.getElementById("tracker").clientWidth;
  return Math.max(0, Math.min(w - 20, x * w));
}

async function loadProject() {
  const res = await fetch("/api/project");
  project = await res.json();
  pattern = JSON.parse(JSON.stringify(project.pattern));
  document.getElementById("source").textContent =
    "Source : " + project.source_audio + " | image : " + project.image;
  selectedId = pattern.length ? pattern[0].id : null;
  renderPairSelect();
  renderSamples();
  renderTracker();
}

function renderPairSelect() {
  const sel = document.getElementById("pairSelect");
  sel.innerHTML = project.blocks.map(b => `<option value="${b.pair}">pair ${String(b.pair).padStart(2,"0")}</option>`).join("");
}

function renderSamples() {
  const root = document.getElementById("samples");
  root.innerHTML = "";
  project.blocks.forEach(block => {
    const div = document.createElement("div");
    div.className = "sample";
    div.innerHTML = `
      <strong>pair ${String(block.pair).padStart(2,"0")}</strong>
      <div class="muted">${block.duration_ms.toFixed(1)} ms</div>
      <audio controls src="/audio?path=${encodeURIComponent(block.audio_path)}"></audio>
      <button onclick="assignPair(${block.pair})">Assigner au bloc sélectionné</button>
    `;
    root.appendChild(div);
  });
}

function renderTracker() {
  const tr = document.getElementById("tracker");
  [...tr.querySelectorAll(".block")].forEach(e => e.remove());

  pattern.forEach(item => {
    const b = document.createElement("div");
    b.className = "block" + (item.id === selectedId ? " selected" : "");
    b.style.left = xLeft(item.x) + "px";
    b.style.top = laneTop(item.lane) + "px";
    b.style.width = Math.max(18, item.width * tr.clientWidth) + "px";
    b.textContent = item.pair;
    b.onmousedown = ev => startDrag(ev, item.id);
    b.onclick = ev => { ev.stopPropagation(); selectedId = item.id; syncSelected(); renderTracker(); };
    tr.appendChild(b);
  });

  syncSelected();
}

function selected() {
  return pattern.find(x => x.id === selectedId);
}

function syncSelected() {
  const s = selected();
  if (!s) return;
  document.getElementById("pairSelect").value = s.pair;
}

function startDrag(ev, id) {
  selectedId = id;
  const s = selected();
  drag = { id, offsetX: ev.offsetX };
  ev.preventDefault();
  renderTracker();
}

document.addEventListener("mousemove", ev => {
  if (!drag) return;
  const tr = document.getElementById("tracker");
  const rect = tr.getBoundingClientRect();
  const s = selected();
  const x = (ev.clientX - rect.left - drag.offsetX) / rect.width;
  s.x = Math.max(0, Math.min(0.98, x));

  const lane = Math.floor((ev.clientY - rect.top) / (rect.height / 3));
  s.lane = Math.max(0, Math.min(2, lane));
  s.role = ["hat", "kick", "snare"][s.lane];
  renderTracker();
});

document.addEventListener("mouseup", () => { drag = null; });

document.addEventListener("keydown", ev => {
  if (ev.key === "ArrowUp") moveLane(-1);
  if (ev.key === "ArrowDown") moveLane(1);
});

function moveLane(delta) {
  const s = selected();
  if (!s) return;
  s.lane = Math.max(0, Math.min(2, s.lane + delta));
  s.role = ["hat", "kick", "snare"][s.lane];
  renderTracker();
}

function changePair(pair) {
  const s = selected();
  if (!s) return;
  s.pair = parseInt(pair);
  renderTracker();
}

function assignPair(pair) {
  const s = selected();
  if (!s) return;
  s.pair = pair;
  renderTracker();
}

function playSelected() {
  const s = selected();
  if (!s) return;
  const audio = new Audio("/audio?path=" + encodeURIComponent(project.blocks[s.pair].audio_path));
  audio.play();
}

function sortPattern() {
  pattern.sort((a,b) => a.x - b.x);
  renderTracker();
}

function resetPattern() {
  pattern = JSON.parse(JSON.stringify(project.pattern));
  selectedId = pattern.length ? pattern[0].id : null;
  renderTracker();
}

async function saveEdit() {
  pattern.forEach(item => item.role = ["hat", "kick", "snare"][item.lane]);
  const payload = {
    pattern,
    notes: document.getElementById("notes").value,
    loops: parseInt(document.getElementById("loops").value || "1")
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

window.addEventListener("resize", renderTracker);
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
    parser.add_argument("--image", required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    pair_json = find_pair_json(args.source)
    project = load_project(pair_json, args.image)
    TrackerHandler.project = project

    server = HTTPServer(("127.0.0.1", args.port), TrackerHandler)
    url = f"http://127.0.0.1:{args.port}"

    print("Visual Tracker Editor lancé.")
    print("Source :", project["source_audio"])
    print("Image  :", project["image"])
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
