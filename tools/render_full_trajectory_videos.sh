#!/usr/bin/env bash
set -euo pipefail
cd /root/zipsplat/ZipSplat
export TORCH_EXTENSIONS_DIR=/root/zipsplat/ZipSplat/.torch_extensions
for worker in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$worker .venv/bin/python /tmp/render_full_trajectory_videos.py \
    --worker "$worker" --num-workers 8 --overwrite --num-frames 240 --fps 24 \
    --reference-root /root/multiview_compare/experiments/random \
    --result-root /root/multiview_compare/experiments/random \
    --result-root /root/multiview_compare/experiments/short60 \
    >/tmp/full_trajectory_${worker}.log 2>&1 &
done
status=0
for pid in $(jobs -p); do wait "$pid" || status=1; done
for log in /tmp/full_trajectory_*.log; do tail -30 "$log"; done
exit "$status"
