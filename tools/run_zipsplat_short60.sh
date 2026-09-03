#!/usr/bin/env bash
set -euo pipefail
cd /root/zipsplat/ZipSplat
export TORCH_EXTENSIONS_DIR=/root/zipsplat/ZipSplat/.torch_extensions
root=/root/multiview_compare/experiments/short60
for worker in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$worker .venv/bin/python /tmp/run_zipsplat_dl3dv_gt_prior.py \
    --worker "$worker" --num-workers 8 \
    --input-root "$root" --output-root "$root" --input-method querysplat --output-method zipsplat \
    >/tmp/zipsplat_short60_${worker}.log 2>&1 &
done
status=0
for pid in $(jobs -p); do wait "$pid" || status=1; done
for log in /tmp/zipsplat_short60_*.log; do tail -12 "$log"; done
exit "$status"
