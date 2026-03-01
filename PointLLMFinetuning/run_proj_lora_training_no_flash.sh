#!/bin/bash
set -e

CONDA_ENV_NAME="pointllm_finetune"

GPU_ID=${1:-""}       # ★追加：第1引数（物理GPU番号）。空なら全GPU
DATA_ROOT=${2:-""}    # 既存：第2引数（data root）

POINTLLM_DIR="PointLLM"
PROJECT_DIR=$(pwd)

echo "============================================================"
echo "PointLLM Projection + LoRA (No Install Method)"
echo "============================================================"

echo "[1/4] Setting up Conda environment..."
if ! conda env list | grep -q "^$CONDA_ENV_NAME "; then
    conda create -n "$CONDA_ENV_NAME" python=3.10 -y
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

echo "[4/4] Starting training..."

# ★ここが重要：GPU指定があれば1枚に絞る。なければ unset で全GPU
if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    # 1枚しか見せないのでプログラム側は常に cuda:0
    GPU_ARG=0
else
    unset CUDA_VISIBLE_DEVICES
    # 全GPU見えるなら物理と同じ番号でOK（ただし分散してない前提）
    GPU_ARG=0
fi

export PYTHONPATH="$PROJECT_DIR/$POINTLLM_DIR:$PYTHONPATH"

# ★デバッグ（次に詰まらないため）
echo "[DEBUG bash] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
python - <<'PY'
import os, torch, sys
print("[DEBUG py] exe =", sys.executable)
print("[DEBUG py] torch =", torch.__version__)
print("[DEBUG py] torch.version.cuda =", torch.version.cuda)
print("[DEBUG py] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("[DEBUG py] is_available =", torch.cuda.is_available())
print("[DEBUG py] device_count =", torch.cuda.device_count())
PY

CMD="python proj_tune_and_LoRA.py --gpu $GPU_ARG --pointllm-path $POINTLLM_DIR"
if [ -n "$DATA_ROOT" ]; then
    CMD="$CMD --data-root $DATA_ROOT"
fi

echo "Running: $CMD"
$CMD

echo "✅ Training completed!"

