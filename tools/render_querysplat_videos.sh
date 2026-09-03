#!/usr/bin/env bash
set -euo pipefail
cd /root/zipsplat/ZipSplat
export TORCH_EXTENSIONS_DIR=/root/zipsplat/ZipSplat/.torch_extensions
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c 'from gsplat.cuda._backend import _C; print("gsplat ready")'
pids=()
for worker in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES="$worker" .venv/bin/python /tmp/render_querysplat_videos.py \
    --worker "$worker" --num-workers 8 \
    --root /root/multiview_compare/experiments/random \
    --root /root/multiview_compare/experiments/short60 \
    --output-name video_wide.mp4 --sweep-deg 120 \
    >/tmp/querysplat_video_worker_${worker}.log 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
for f in /tmp/querysplat_video_worker_*.log; do tail -10 "$f"; done
exit "$status"
