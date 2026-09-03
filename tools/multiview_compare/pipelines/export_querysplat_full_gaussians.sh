#!/usr/bin/env bash
set -euo pipefail
cd /root/querysplat_ws/querysplat
roots=(/root/multiview_compare/experiments/random /root/multiview_compare/experiments/short60)
jobs=()
for root in "${roots[@]}"; do
  while IFS= read -r exp; do jobs+=("$exp"); done < <(find "$root" -mindepth 3 -maxdepth 3 -type d -name querysplat -exec test -d '{}/input' ';' -print | sort)
done
run_worker() {
  local worker=$1
  for ((i=worker; i<${#jobs[@]}; i+=8)); do
    exp=${jobs[$i]}; out="$exp/output"
    if [ -f "$out/gaussians_opacity0.ply" ]; then echo "SKIP $exp"; continue; fi
    /root/querysplat_ws/.venv/bin/python -m scripts.infer \
      --config checkpoints/querysplat_vggto_1B_512_8192.yaml \
      --checkpoint checkpoints/querysplat_vggto_1B_512_8192.safetensors \
      --input_folder "$exp/input" --output_dir "$out" --save_predicted_input_cameras \
      --gaussian_save_opacity_threshold 0 0.05 >"$out/full_export.log" 2>&1
    echo "DONE $exp"
  done
}
pids=()
for worker in $(seq 0 7); do CUDA_VISIBLE_DEVICES=$worker run_worker "$worker" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
