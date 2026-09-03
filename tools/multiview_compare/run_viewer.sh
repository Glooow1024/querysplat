#!/usr/bin/env bash
set -euo pipefail

base=${MULTIVIEW_COMPARE_ROOT:-/root/multiview_compare}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
viewer="$script_dir/viewer.py"
log="$base/logs/viewer.log"
port=${1:-18765}

mkdir -p "$base/logs"
old_pid=$(ss -ltnp | sed -n "s/.*:${port} .*pid=\([0-9]*\).*/\1/p")
if [ -n "$old_pid" ]; then
  kill "$old_pid"
  for _ in $(seq 1 20); do
    ss -ltnp | grep -q ":${port} " || break
    sleep 1
  done
fi
setsid -f python3 "$viewer" "$base/experiments" \
  --host 0.0.0.0 --port "$port" --no-browser >"$log" 2>&1
echo "Viewer started on port $port"
