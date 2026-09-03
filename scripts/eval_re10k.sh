#!/usr/bin/env bash
# Copyright (c) 2026 Inspatio. SPDX-License-Identifier: Apache-2.0
# QuerySplat RE10K evaluation (C3G-aligned). Mirrors C3G eval_re10k.sh options.
#
# Usage:
#   bash scripts/eval_re10k.sh                              # 12 views / 1000 scenes / no TTO
#   bash scripts/eval_re10k.sh --views 24 --tto on          # 24 views + TTO
#   bash scripts/eval_re10k.sh --views 2 --num_samples 100  # 2 views / 100 scenes
set -euo pipefail
cd "$(dirname "$0")/.."

VIEWS=12
NUM_SAMPLES=1000
TTO=off
ALIGN_POSE=on
SAVE_COMPARE_N=10
GPUS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --views) VIEWS="$2"; shift 2;;
    --num_samples) NUM_SAMPLES="$2"; shift 2;;
    --tto) TTO="$2"; shift 2;;
    --align_pose) ALIGN_POSE="$2"; shift 2;;
    --save_compare_n) SAVE_COMPARE_N="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

RUN_NAME=re10k_v${VIEWS}_n${NUM_SAMPLES}
if [ "$TTO" = on ]; then RUN_NAME="${RUN_NAME}_tto"; fi
if [ "$ALIGN_POSE" = off ]; then RUN_NAME="${RUN_NAME}_noalign"; fi
OUTPUT_DIR=outputs/${RUN_NAME}/$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p "$OUTPUT_DIR"

TTO_FLAG=""
if [ "$TTO" = on ]; then TTO_FLAG="--use_tto"; fi
ALIGN_FLAG=""
if [ "$ALIGN_POSE" = off ]; then ALIGN_FLAG="--no_align_pose"; fi

CUDA_VISIBLE_DEVICES=$GPUS python -m scripts.eval_re10k \
  --config checkpoints/querysplat_vggto_1B_512_8192.yaml \
  --checkpoint checkpoints/querysplat_vggto_1B_512_8192.safetensors \
  --re10k_root /root/C3G_ws/C3G/datasets/re10k \
  --index_path /root/C3G_ws/C3G/assets/evaluation_index_re10k.json \
  --num_context_views "$VIEWS" \
  --num_samples "$NUM_SAMPLES" \
  --save_compare_n "$SAVE_COMPARE_N" \
  --output_dir "$OUTPUT_DIR" \
  $TTO_FLAG \
  $ALIGN_FLAG
