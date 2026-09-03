#!/usr/bin/env bash
set -euo pipefail
cd /root/zipsplat/ZipSplat
export TORCH_EXTENSIONS_DIR=/root/zipsplat/ZipSplat/.torch_extensions
script=/root/querysplat_ws/querysplat/tools/multiview_compare/rendering/render_gt_aligned_views.py
for worker in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$worker .venv/bin/python "$script" \
    --worker "$worker" --num-workers 8 --overwrite \
    --input-root /root/multiview_compare/inputs/random_gt \
    --result-root /root/multiview_compare/experiments/random \
    >/tmp/gt_aligned_views_${worker}.log 2>&1 &
done
status=0
for pid in $(jobs -p); do wait "$pid" || status=1; done
for log in /tmp/gt_aligned_views_*.log; do tail -20 "$log"; done
exit "$status"
