#!/usr/bin/env bash
set -euo pipefail

cd /root/querysplat_ws/querysplat
source_root=/root/multiview_compare/experiments/short60
mapfile -t jobs < <(find "$source_root" -mindepth 2 -maxdepth 2 -type d -name '*views' | sort)

run_worker() {
  local worker=$1
  for ((i=worker; i<${#jobs[@]}; i+=8)); do
    local source_exp=${jobs[$i]}
    local rel=${source_exp#"$source_root"/}
    local input_exp="$source_exp/querysplat"
    local exp="$source_exp/querysplat-tto20"
    local out="$exp/output"
    mkdir -p "$out"
    if [ -f "$out/gaussians_opacity0.ply" ] \
       && [ -f "$out/predicted_input_cameras.json" ] \
       && [ -f "$out/tto_losses.json" ]; then
      echo "SKIP $rel"
      continue
    fi
    /root/querysplat_ws/.venv/bin/python -m scripts.infer \
      --config checkpoints/querysplat_vggto_1B_512_8192.yaml \
      --checkpoint checkpoints/querysplat_vggto_1B_512_8192.safetensors \
      --input_folder "$input_exp/input" \
      --output_dir "$out" \
      --use_tto --tto_n_steps 20 \
      --gaussian_save_opacity_threshold 0 0.05 \
      --save_predicted_input_cameras \
      >"$out/run.log" 2>&1
    echo "DONE $rel"
  done
}

pids=()
for worker in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$worker run_worker "$worker" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
