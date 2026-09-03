#!/usr/bin/env bash
set -euo pipefail
cd /root/zipsplat/ZipSplat
export TORCH_EXTENSIONS_DIR=/root/zipsplat/ZipSplat/.torch_extensions
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c 'from gsplat.cuda._backend import _C; import scipy; print("renderer ready")'
pids=()
for worker in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$worker .venv/bin/python /tmp/render_gt_trajectory_videos.py \
    --worker "$worker" --num-workers 8 \
    --input-root /root/multiview_compare/inputs/random_gt \
    --result-root /root/multiview_compare/experiments/random \
    --result-root /root/multiview_compare/querysplat_gt_cameras \
    --result-root /root/multiview_compare/zipsplat_gt_cameras \
    --num-frames 240 --fps 24 --overwrite \
    >/tmp/gt_trajectory_video_${worker}.log 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
for f in /tmp/gt_trajectory_video_*.log; do tail -12 "$f"; done
exit "$status"
