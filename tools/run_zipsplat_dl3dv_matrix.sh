#!/usr/bin/env bash
set -euo pipefail

cd /root/zipsplat/ZipSplat
input_root=/root/querysplat_ws/random10_view_matrix
output_root=/root/zipsplat/ZipSplat/outputs/dl3dv_random10_view_matrix
num_workers=8
mkdir -p "$output_root/logs"

pids=()
for worker in $(seq 0 $((num_workers - 1))); do
  CUDA_VISIBLE_DEVICES="$worker" .venv/bin/python /tmp/run_zipsplat_dl3dv_matrix.py \
    --worker "$worker" \
    --num-workers "$num_workers" \
    --input-root "$input_root" \
    --output-root "$output_root" \
    >"$output_root/logs/worker_${worker}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

for log in "$output_root"/logs/worker_*.log; do
  echo "===== $log ====="
  tail -30 "$log"
done
exit "$status"
