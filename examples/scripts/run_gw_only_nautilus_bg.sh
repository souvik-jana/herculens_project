#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="$REPO_ROOT/examples/outputs/outputs_gw_only_nautilus/run.log"
PIDFILE="$REPO_ROOT/examples/outputs/outputs_gw_only_nautilus/run.pid"

mkdir -p "$(dirname "$LOG")"

if pgrep -f 'examples/scripts/gw_only_nautilus.py' >/dev/null 2>&1; then
    echo "Already running:"
    pgrep -fl 'examples/scripts/gw_only_nautilus.py'
    exit 1
fi

cd "$REPO_ROOT"
PYTHONUNBUFFERED=1 caffeinate -i nohup uv run python examples/scripts/gw_only_nautilus.py \
    > "$LOG" 2>&1 </dev/null &
PID=$!
disown
echo "$PID" > "$PIDFILE"

echo "Started PID $PID (caffeinate -i)"
echo "Log: $LOG"
echo "Monitor: tail -f $LOG"
