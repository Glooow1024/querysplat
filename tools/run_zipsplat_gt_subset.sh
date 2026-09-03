#!/usr/bin/env bash
set -euo pipefail
cd /root/zipsplat/ZipSplat
export TORCH_EXTENSIONS_DIR=/root/zipsplat/ZipSplat/.torch_extensions
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c 'from gsplat.cuda._backend import _C; print("gsplat ready")'
root=/root/multiview_compare/zipsplat_gt_cameras
mkdir -p "$root/logs"
pids=()
for worker in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$worker .venv/bin/python /tmp/run_zipsplat_dl3dv_gt_prior.py \
    --worker "$worker" --num-workers 8 --external-priors \
    --input-root /root/multiview_compare/dl3dv_gt_subset_inputs --output-root "$root" \
    >"$root/logs/worker_${worker}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
for f in "$root"/logs/*.log; do tail -10 "$f"; done
exit "$status"
