#!/usr/bin/env bash
set -euo pipefail
cd /root/querysplat_ws/querysplat
root=/root/multiview_compare/experiments/short60
mapfile -t jobs < <(find "$root" -mindepth 2 -maxdepth 2 -type d -name '*views' | sort)
run_worker() {
  local worker=$1
  for ((i=worker; i<${#jobs[@]}; i+=8)); do
    exp=${jobs[$i]}
    method_exp="$exp/querysplat"
    out="$method_exp/output"
    mkdir -p "$out"
    if [ -f "$out/gaussians.ply" ] && [ -f "$out/predicted_input_cameras.json" ]; then continue; fi
    /root/querysplat_ws/.venv/bin/python -m scripts.infer \
      --config checkpoints/querysplat_vggto_1B_512_8192.yaml \
      --checkpoint checkpoints/querysplat_vggto_1B_512_8192.safetensors \
      --input_folder "$method_exp/input" --output_dir "$out" --save_predicted_input_cameras \
      >"$out/run.log" 2>&1
    echo "DONE ${exp#"$root"/}"
  done
}
pids=()
for worker in $(seq 0 7); do CUDA_VISIBLE_DEVICES=$worker run_worker "$worker" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
