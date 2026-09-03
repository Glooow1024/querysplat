#!/usr/bin/env bash
set -euo pipefail
cd /root/querysplat_ws/querysplat
root=/root/multiview_compare/dl3dv_gt_subset_inputs
output_root=/root/multiview_compare/querysplat_gt_cameras
mapfile -t jobs < <(find "$root" -mindepth 2 -maxdepth 2 -type d -name '*views' -exec test -f '{}/camera_priors.json' ';' -print | sort)
run_worker() {
  local worker=$1
  for ((i=worker; i<${#jobs[@]}; i+=8)); do
    exp=${jobs[$i]}
    rel=${exp#"$root"/}
    out="$output_root/$rel/output"
    mkdir -p "$out"
    [ -f "$out/gaussians.ply" ] && continue
    /root/querysplat_ws/.venv/bin/python -m scripts.infer \
      --config checkpoints/querysplat_vggto_1B_512_8192.yaml \
      --checkpoint checkpoints/querysplat_vggto_1B_512_8192.safetensors \
      --input_folder "$exp/input" --output_dir "$out" \
      --camera_priors_json "$exp/camera_priors.json" \
      --save_predicted_input_cameras \
      >"$out/run.log" 2>&1
    ln -sfn "$exp/transforms.json" "$(dirname "$out")/transforms.json"
    echo "DONE $rel"
  done
}
pids=()
for worker in $(seq 0 7); do CUDA_VISIBLE_DEVICES=$worker run_worker $worker & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
