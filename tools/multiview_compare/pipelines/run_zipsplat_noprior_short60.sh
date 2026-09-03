#!/usr/bin/env bash
set -euo pipefail

cd /root/zipsplat/ZipSplat
export TORCH_EXTENSIONS_DIR=/root/zipsplat/ZipSplat/.torch_extensions
root=/root/multiview_compare/experiments/short60
script=/root/querysplat_ws/querysplat/tools/multiview_compare/pipelines/run_zipsplat.py

for worker in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$worker .venv/bin/python "$script" \
    --worker "$worker" --num-workers 8 \
    --input-root "$root" --output-root "$root" \
    --input-method querysplat --output-method zipsplat-noprior --no-priors \
    --pose-refinement-steps 30 --overwrite \
    >/root/multiview_compare/logs/zipsplat_noprior_short60_${worker}.log 2>&1 &
done

status=0
for pid in $(jobs -p); do wait "$pid" || status=1; done
for log in /root/multiview_compare/logs/zipsplat_noprior_short60_*.log; do tail -20 "$log"; done
exit "$status"
