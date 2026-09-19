#!/usr/bin/env bash
# Launch the resumable MIMII driver in the background.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .mimii_logs
LOG=".mimii_logs/driver_$(date +%Y%m%d_%H%M%S).log"
nohup .venv/bin/python -u scripts/mimii_driver.py > "$LOG" 2> "$LOG.err" &
PID=$!
echo "started pid=$PID log=$LOG"
echo "tail -f $LOG"
