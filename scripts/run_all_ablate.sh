#!/usr/bin/env bash
set -euo pipefail

# Scenes to process
SCENES=(
  bicycle
  bonsai
  counter
  kitchen
  room
  stump
  garden
  treehill
  flowers
)

DATA_ROOT="/home/kemove/data/mipnerf360"
MODEL_ROOT="/home/kemove/data/3dgs-model"
OUT_ROOT="/home/kemove/efficient-output"

METHOD_ARGS=(
  --eval
  --compression_method zlib
  --compression_level 4
  --xyz_codec morton16
  --quality_target_diff 0.65
  --segments 100
  --use_importance_v4 True
  --topk_views_percent 0.1
  --view_importance_weight 2.0
  --ablation_setting "w/o early stopping"
)

for scene in "${SCENES[@]}"; do
  echo "============== ${scene} =============="
  python /home/kemove/github/FlexGaussian/compress_ablate.py \
    -s "${DATA_ROOT}/${scene}/" \
    -m "${MODEL_ROOT}/${scene}/" \
    --output_path "${OUT_ROOT}/" \
    "${METHOD_ARGS[@]}"
done
