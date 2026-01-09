#!/bin/bash

# 移除严格错误检查，以允许继续处理其他场景
SCENE_ARG=${1:-}

MODEL_ROOT="/home/kemove/data/3dgs-model"
OUTPUT_ROOT="/home/kemove/efficient-output"
REPO_ROOT="/home/kemove/github/FlexGaussian"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

# Simple per-scene run: one config, print concise summary per scene

declare -A DATASET_SOURCE_ROOT=(
  [mipnerf360]="/home/kemove/data/mipnerf360"
  [tandt]="/home/kemove/data/tandt"
  [db]="/home/kemove/data/db"
)

declare -A DATASET_SCENES
DATASET_SCENES[mipnerf360]="bicycle bonsai counter kitchen room stump garden treehill flowers"
DATASET_SCENES[tandt]="truck train"
DATASET_SCENES[db]="playroom drjohnson"

DATASET_KEYS=(mipnerf360 tandt db)

should_run_scene() {
  local dataset="$1"
  local scene="$2"
  if [ -z "$SCENE_ARG" ]; then
    return 0
  fi

  if [ "$dataset" == "$SCENE_ARG" ] || [ "$scene" == "$SCENE_ARG" ]; then
    return 0
  fi

  return 1
}

printf "\n[INFO] Logs: %s\n\n" "$LOG_DIR"

for DATASET in "${DATASET_KEYS[@]}"; do
  SOURCE_ROOT="${DATASET_SOURCE_ROOT[$DATASET]}"
  IFS=' ' read -r -a SCENE_LIST <<< "${DATASET_SCENES[$DATASET]}"

  for SCENE in "${SCENE_LIST[@]}"; do
    if ! should_run_scene "$DATASET" "$SCENE"; then
      continue
    fi

    echo "[INFO] Running dataset: $DATASET scene: $SCENE"
    SRC_DIR="$SOURCE_ROOT/$SCENE"
    MOD_DIR="$MODEL_ROOT/$SCENE"

    TS=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOG_DIR/${DATASET}_${SCENE}_$TS.log"

#   { /usr/bin/time -v \
#     python "$REPO_ROOT/compress.py" \
#       -s "$SRC_DIR" \
#       -m "$MOD_DIR" \
#       --output_path "$OUTPUT_ROOT" \
#       --quality_target_diff 1.0 \
#       --eval \
#       --xyz_codec morton16 \
#       --use_delta_encoding true; \
#   } &> "$LOG_FILE" || true

    { /usr/bin/time -v \
      python "$REPO_ROOT/compress.py" \
        -s "$SRC_DIR" \
        -m "$MOD_DIR" \
        --output_path "$OUTPUT_ROOT" \
        --quality_target_diff 0.55 \
        --eval \
        --compression_method zlib \
        --compression_level 4 \
        --xyz_codec morton16 \
        --segments 1000 \
        --use_importance_v4 True \
        --topk_views_percent 0.1 \
        --view_importance_weight 2.0 \
        --save_render True; \
    } &> "$LOG_FILE" || true


    # 检查日志文件中是否包含成功完成的标志
    if grep -q "\[INFO\] Best PSNR:" "$LOG_FILE"; then
      BEST_PSNR=$(grep -E "\[INFO\] Best PSNR:" "$LOG_FILE" | tail -n 1 | sed -E 's/.*Best PSNR:[[:space:]]*([0-9]+(\.[0-9]+)?)[[:space:]]*dB.*/\1/')
      PRUNE_RATE=$(grep -E "\[INFO\] Best config:" "$LOG_FILE" | tail -n 1 | grep -oE "pruning_rate = [0-9\.]+" | awk '{print $3}')
      SH_RATE=$(grep -E "\[INFO\] Best config:" "$LOG_FILE" | tail -n 1 | grep -oE "sh_rate = [0-9\.]+" | awk '{print $3}')
      MEM_MB=$(grep -E "\[INFO\] Output file size:" "$LOG_FILE" | tail -n 1 | sed -E 's/.*Output file size:[[:space:]]*([0-9]+(\.[0-9]+)?)[[:space:]]*MB.*/\1/')
    else
      echo "[WARNING] Scene $SCENE (dataset $DATASET) failed. Check log: $LOG_FILE"
      # 设置默认值或失败标记
      BEST_PSNR="FAILED"
      PRUNE_RATE="FAILED"
      SH_RATE="FAILED"
      MEM_MB="FAILED"
    fi

    TOTAL_PIPELINE_LINE=$(grep -n "\[STATS\] Pipeline Time Breakdown:" "$LOG_FILE" | tail -n 1 | cut -d: -f1)
    if [[ -n "$TOTAL_PIPELINE_LINE" ]]; then
      TIME_SEC=$(tail -n +$((TOTAL_PIPELINE_LINE)) "$LOG_FILE" | grep -E "Total time:" | head -n 1 | awk '{print $(NF-1)}')
    else
      # 如果找不到时间信息，设置为FAILED
      TIME_SEC="FAILED"
    fi

    SSIM_VALUE=""
    LPIPS_VALUE=""
    if grep -q "ssim:" "$LOG_FILE"; then
      SSIM_VALUE=$(grep -E "ssim:" "$LOG_FILE" | tail -n 1 | sed -E 's/.*ssim:[[:space:]]*([0-9]+(\.[0-9]+)?).*/\1/')
    fi
    if grep -q "lpips:" "$LOG_FILE"; then
      LPIPS_VALUE=$(grep -E "lpips:" "$LOG_FILE" | tail -n 1 | sed -E 's/.*lpips:[[:space:]]*([0-9]+(\.[0-9]+)?).*/\1/')
    fi

    SUMMARY="dataset=$DATASET scene=$SCENE psnr=$BEST_PSNR mem_mb=$MEM_MB time_sec=$TIME_SEC pruning_rate=$PRUNE_RATE sh_rate=$SH_RATE"
    if [[ -n "$SSIM_VALUE" && -n "$LPIPS_VALUE" ]]; then
      SUMMARY="$SUMMARY ssim=$SSIM_VALUE lpips=$LPIPS_VALUE"
    fi
    SUMMARY="$SUMMARY log=$LOG_FILE"

    echo "$SUMMARY"
  done
done

echo "\n[INFO] Done."
