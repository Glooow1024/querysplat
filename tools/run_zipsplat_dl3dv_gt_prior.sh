#!/usr/bin/env bash
set -euo pipefail
cd /root/zipsplat/ZipSplat
export TORCH_EXTENSIONS_DIR=/root/zipsplat/ZipSplat/.torch_extensions
root=/root/zipsplat/ZipSplat/outputs/dl3dv_random10_shared_query_cameras
mkdir -p "$root/logs"
# Build/load the gsplat extension once before parallel workers use it.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c 'from gsplat.cuda._backend import _C; print("gsplat ready")'
pids=()
for worker in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES="$worker" .venv/bin/python /tmp/run_zipsplat_dl3dv_gt_prior.py \
    --worker "$worker" --num-workers 8 \
    --input-root /root/querysplat_ws/random10_view_matrix --output-root "$root" \
    >"$root/logs/worker_${worker}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
for log in "$root"/logs/worker_*.log; do echo "===== $log ====="; tail -20 "$log"; done
exit "$status"
