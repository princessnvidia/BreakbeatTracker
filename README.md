# BreakbeatTracker 🥁

Open-source tracker for **analyzing, slicing and reinventing classic breakbeats**.

BreakbeatTracker combines automatic drum recognition, intelligent sample organization and tracker-style sequencing to accelerate Jungle, Drum & Bass and Breakcore production.

---

<p align="center">
  <img src="docs/demo.gif" alt="BreakbeatTracker Demo" width="100%">
</p>

---

# Features

## 🥁 Break Analysis

- Automatic break slicing
- Kick, snare, hat and cymbal recognition
- Intelligent slice classification
- Waveform visualization
- Automatic sample organization

## 🎛 Tracker Workflow

- Classic tracker interface
- Step sequencing
- Pattern editing
- Instant playback
- Keyboard-first workflow

## 🎲 Pattern Generation

- Rhythm variations
- Groove mutations
- Humanized timing
- Syncopated fills
- Creative randomization

## 🎧 Audio Engine

- Real-time playback
- Slice preview
- Tempo control
- Pitch shifting
- Volume control

## 📚 Sample Library

- Automatic slice browser
- Favorites
- Search
- Tags
- Drum categorization

## 🤖 Intelligent Assistance

- Drum recognition
- Automatic slice sorting
- Pattern suggestions
- Break reconstruction

---

# Tech Stack

- Python
- PySide6
- Qt6
- NumPy
- SciPy
- Librosa

---

# Audio Pipeline

```
Audio Loop
      │
      ▼
Slice Detection
      │
      ▼
Kick / Snare / Hat Recognition
      │
      ▼
Slice Library
      │
      ▼
Tracker Editor
      │
      ▼
Pattern Generator
      │
      ▼
Playback
```

---

# Roadmap

## Tracker

- [ ] Multi-track patterns
- [ ] Song mode
- [ ] Pattern chaining

## Export

- [ ] MIDI export
- [ ] WAV export
- [ ] Renoise export
- [ ] M8 Tracker export

## Intelligence

- [ ] AI groove generation
- [ ] Batch break analysis
- [ ] Smart sample recommendations

## Performance

- [ ] Live performance mode
- [ ] VST support

---

# Installation

```bash
git clone https://github.com/princessnvidia/BreakbeatTracker.git
cd BreakbeatTracker

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python main.py
```

---

# Philosophy

BreakbeatTracker is built around a simple idea:

Creating breakbeats should feel fast, playful and musical.

Instead of spending hours manually slicing loops and organizing drum hits, producers should be able to focus on experimentation. The software handles repetitive analysis while preserving the creative workflow that made classic trackers so enjoyable.

The long-term vision is to become an open-source tracker dedicated to sampled breakbeat music, inspired by decades of tracker culture while exploring modern intelligent audio tools.

---

# Inspiration

- Renoise
- Dirtywave M8
- OctaMED
- FastTracker II
- OpenMPT

---

# Status

🚧 Active Development

---

# License

MIT License
