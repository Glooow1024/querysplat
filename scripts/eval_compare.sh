#!/usr/bin/env bash
# Copyright (c) 2026 Inspatio. SPDX-License-Identifier: Apache-2.0
# Unified fair comparison of QuerySplat (pose-free) vs C3G (GT target poses) on
# DL3DV-Evaluation or RE10K. Both methods eval the SAME scenes/views (shared index),
# SAME metric resolution, SAME align_pose / TTO setting. Results + representative
# comparison images for both are collected into one obvious folder.
#
# Usage:
#   bash scripts/eval_compare.sh                              # DL3DV, all 55, 12-view, align_pose off, 256, 8 GPUs
#   bash scripts/eval_compare.sh --dataset re10k             # RE10K, first 100 scenes
#   bash scripts/eval_compare.sh --render_size 512           # compare at QuerySplat native 512
#   bash scripts/eval_compare.sh --align_pose on --tto on    # with pose alignment + TTO
#   bash scripts/eval_compare.sh --num_samples 10 --views 24
set -euo pipefail

QS_DIR=/root/querysplat_ws/querysplat
C3G_DIR=/root/C3G_ws/C3G
QS_PY=/root/querysplat_ws/.venv/bin/python
C3G_PY=/root/C3G_ws/.venv/bin/python
QS_CKPT=$QS_DIR/checkpoints/querysplat_vggto_1B_512_8192.safetensors
QS_CFG=$QS_DIR/checkpoints/querysplat_vggto_1B_512_8192.yaml
C3G_CKPT=/root/C3G_ws/checkpoints/gaussian_decoder_multiview.ckpt

DATASET=dl3dv
VIEWS=12
NUM_SAMPLES=auto        # auto: dl3dv=all 55, re10k=100
ALIGN_POSE=off
TTO=off
RENDER_SIZE=auto        # auto|256|512 (512 = QuerySplat native high-res, both run at 512)
SAVE_COMPARE_N=5
GPUS=8                  # C3G uses this many GPUs (devices="auto"); QuerySplat uses 1
QS_GPU=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --views) VIEWS="$2"; shift 2;;
    --num_samples) NUM_SAMPLES="$2"; shift 2;;
    --align_pose) ALIGN_POSE="$2"; shift 2;;
    --tto) TTO="$2"; shift 2;;
    --render_size) RENDER_SIZE="$2"; shift 2;;
    --save_compare_n) SAVE_COMPARE_N="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --qs_gpu) QS_GPU="$2"; shift 2;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

# ---- dataset-specific config ----
case "$DATASET" in
  dl3dv)
    CHUNKS=/data/datasets/3dvision/DL3DV-Evaluation/c3g_eval
    FULL_INDEX=/data/datasets/3dvision/DL3DV-Evaluation/eval_index_dl3dv.json
    C3G_EVAL_CFG=dl3dv_multiview; C3G_DS=dl3dv
    [ "$NUM_SAMPLES" = auto ] && NUM_SAMPLES=0
    ;;
  re10k)
    CHUNKS=/root/C3G_ws/C3G/datasets/re10k
    FULL_INDEX=/root/C3G_ws/C3G/assets/evaluation_index_re10k.json
    C3G_EVAL_CFG=re10k_multiview; C3G_DS=re10k
    [ "$NUM_SAMPLES" = auto ] && NUM_SAMPLES=100
    ;;
  *) echo "Unknown dataset: $DATASET (use dl3dv|re10k)"; exit 1;;
esac

# ---- build (possibly truncated) shared eval index ----
if [ "$NUM_SAMPLES" -gt 0 ]; then
  INDEX="${FULL_INDEX%.json}_first${NUM_SAMPLES}.json"
  if [ ! -f "$INDEX" ]; then
    $QS_PY -c "import json; d=json.load(open('$FULL_INDEX')); json.dump(dict(list(d.items())[:$NUM_SAMPLES]), open('$INDEX','w'))"
  fi
else
  INDEX=$FULL_INDEX
fi

# ---- render-size flags ----
QS_EVAL_SHAPE=0                                  # 0 = auto (224 for 2-view, 256 for multi-view)
C3G_SHAPE_FLAG=""
case "$RENDER_SIZE" in
  512) QS_EVAL_SHAPE=512; C3G_SHAPE_FLAG="dataset.$C3G_DS.input_image_shape=[512,512]";;
  256) QS_EVAL_SHAPE=256;;
  auto) ;;
  *) echo "Unknown render_size: $RENDER_SIZE (use auto|256|512)"; exit 1;;
esac

# ---- align_pose / tto flags ----
QS_ALIGN_FLAG="--no_align_pose"; C3G_ALIGN_FLAG="test.align_pose=false"
[ "$ALIGN_POSE" = on ] && { QS_ALIGN_FLAG=""; C3G_ALIGN_FLAG="test.align_pose=true"; }
QS_TTO_FLAG=""; C3G_TTO_FLAG="test.tto=false"
[ "$TTO" = on ] && { QS_TTO_FLAG="--use_tto"; C3G_TTO_FLAG="test.tto=true"; }

# ---- GPU env ----
C3G_GPUS=$(seq -s, 0 $((GPUS-1)))

TS=$(date +%Y-%m-%d_%H-%M-%S)
TAG=${DATASET}_v${VIEWS}_n${NUM_SAMPLES}_align${ALIGN_POSE}_tto${TTO}_r${RENDER_SIZE}
QS_OUT=$QS_DIR/outputs/compare_qs_$TAG/$TS
C3G_NAME=compare_c3g_$TAG
COMPARE_DIR=$QS_DIR/outputs/compare_$TAG/$TS

echo "=================================================="
echo "Compare QuerySplat vs C3G"
echo "  dataset=$DATASET  views=$VIEWS  samples=$NUM_SAMPLES"
echo "  align_pose=$ALIGN_POSE  tto=$TTO  render_size=$RENDER_SIZE"
echo "  C3G gpus=$GPUS  QuerySplat gpu=$QS_GPU"
echo "  shared index: $INDEX"
echo "  QuerySplat -> $QS_OUT"
echo "  C3G        -> $C3G_DIR/outputs/test/$C3G_NAME"
echo "  collected  -> $COMPARE_DIR"
echo "=================================================="

# ===================== QuerySplat (pose-free, 1 GPU) =====================
echo "[QuerySplat] running..."
mkdir -p "$QS_OUT"
cd "$QS_DIR"
CUDA_VISIBLE_DEVICES=$QS_GPU $QS_PY -m scripts.eval_re10k \
  --config "$QS_CFG" --checkpoint "$QS_CKPT" \
  --re10k_root "$CHUNKS" --index_path "$INDEX" \
  --num_context_views "$VIEWS" --num_samples 0 \
  --eval_shape "$QS_EVAL_SHAPE" \
  --save_compare_n "$SAVE_COMPARE_N" \
  --output_dir "$QS_OUT" \
  $QS_ALIGN_FLAG $QS_TTO_FLAG

# ===================== C3G (GT target poses, N GPUs) =====================
echo "[C3G] running..."
cd "$C3G_DIR"
CUDA_VISIBLE_DEVICES=$C3G_GPUS $C3G_PY -m src.main +evaluation=$C3G_EVAL_CFG mode=test \
  dataset/view_sampler@dataset.$C3G_DS.view_sampler=evaluation \
  dataset.$C3G_DS.view_sampler.index_path="$INDEX" \
  dataset.$C3G_DS.view_sampler.num_context_views="$VIEWS" \
  "dataset.$C3G_DS.roots=[$CHUNKS]" \
  $C3G_SHAPE_FLAG \
  test.save_compare=true test.save_compare_n=$SAVE_COMPARE_N test.save_image=false \
  $C3G_ALIGN_FLAG $C3G_TTO_FLAG \
  checkpointing.load=$C3G_CKPT \
  wandb.mode=offline wandb.name=$C3G_NAME

# ===================== collect both into one folder =====================
mkdir -p "$COMPARE_DIR/c3g_comparisons"
cp "$QS_OUT/results.txt" "$COMPARE_DIR/querysplat_results.txt"
cp "$QS_OUT/results.json" "$COMPARE_DIR/querysplat_results.json" 2>/dev/null || true
[ -d "$QS_OUT/comparisons" ] && cp -r "$QS_OUT/comparisons" "$COMPARE_DIR/querysplat_comparisons"
C3G_RES=$C3G_DIR/outputs/test/$C3G_NAME
cp "$C3G_RES/results.txt" "$COMPARE_DIR/c3g_results.txt"
cp "$C3G_RES/results.json" "$COMPARE_DIR/c3g_results.json" 2>/dev/null || true
# C3G comparison images are <scene>_<psnr>.png (exclude <scene>_projections.png)
for f in "$C3G_RES"/*_*.png; do
  case "$f" in *_projections.png) continue;; esac
  cp "$f" "$COMPARE_DIR/c3g_comparisons/" 2>/dev/null || true
done

echo "=================================================="
echo "RESULTS  (dataset=$DATASET views=$VIEWS align_pose=$ALIGN_POSE tto=$TTO render_size=$RENDER_SIZE)"
echo "-------------------------------------------------- QuerySplat"
cat "$COMPARE_DIR/querysplat_results.txt"
echo "-------------------------------------------------- C3G"
cat "$COMPARE_DIR/c3g_results.txt"
echo "=================================================="
echo "All results + comparison images saved to: $COMPARE_DIR"
