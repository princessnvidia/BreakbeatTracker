#!/usr/bin/env bash
set -e

APP_DIR="$HOME/Applications/BreakbeatAI"
cd "$APP_DIR"

mkdir -p logs exports

if [ -f "$APP_DIR/.venv/bin/activate" ]; then
  source "$APP_DIR/.venv/bin/activate"
fi

SCRIPT="$APP_DIR/pipeline/03_tracker_editor_app_v154_random_ksh_full_slots.py"
SOURCE="${1:-Camo_Break_-_3A}"

python "$SCRIPT" --source "$SOURCE" >> "$APP_DIR/logs/breakbeatai.log" 2>&1
