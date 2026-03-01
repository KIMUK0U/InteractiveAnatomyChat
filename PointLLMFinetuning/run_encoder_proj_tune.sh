#!/bin/bash
set -e

CONDA_ENV_NAME="pointllm_finetune"

GPU_ID=${1:-""}       # 第1引数: 物理GPU番号（空なら全GPU）
DATA_ROOT=${2:-""}    # 第2引数: データルート

POINTLLM_DIR="PointLLM"
PROJECT_DIR=$(pwd)
RESUME_PATH=""  # チェックポイントから再学習する場合はここにパスを設定

echo "============================================================"
echo "PointLLM Encoder + Projector Fine-tuning"
echo "============================================================"

echo "[1/4] Setting up Conda environment..."
if ! conda env list | grep -q "^$CONDA_ENV_NAME "; then
    conda create -n "$CONDA_ENV_NAME" python=3.10 -y
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

echo "[2/4] Installing dependencies..."


echo "[3/4] Cloning PointLLM repository..."
if [ ! -d "$POINTLLM_DIR" ]; then
    git clone https://github.com/OpenRobotLab/PointLLM.git "$POINTLLM_DIR"
    cd "$POINTLLM_DIR"
    pip install -e .
    cd "$PROJECT_DIR"
else
    echo "PointLLM already exists, skipping clone"
fi

echo "[4/4] Starting training..."
# --- 追加: Blackwell/Hopper最適化のための環境変数 ---
export TORCH_CUDNN_V8_API_ENABLED=1
# ▼▼▼ 【追加】 コンパイル (Inductor) 用の最適化設定 ▼▼▼
export TORCH_COMPILE_DISABLE_VOLATILE_SERVER=1
# GPU指定: 指定があれば1枚に絞る。なければ全GPU
if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    GPU_ARG=0  # 1枚しか見せないのでプログラム側は常に cuda:0
else
    unset CUDA_VISIBLE_DEVICES
    GPU_ARG=0  # 全GPU見える場合も0番から
fi

export PYTHONPATH="$PROJECT_DIR/$POINTLLM_DIR:$PYTHONPATH"

# デバッグ情報
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

# コマンド構築
CMD="python encoder_proj_tune.py --gpu $GPU_ARG --pointllm-path $POINTLLM_DIR"
if [ -n "$DATA_ROOT" ]; then
    CMD="$CMD --data-root $DATA_ROOT"
fi

# ★修正: RESUME_PATHが空でない場合のみ引数を追加
if [ -n "$RESUME_PATH" ]; then
    CMD="$CMD --resume-path $RESUME_PATH"
fi

echo "Running: $CMD"
$CMD

echo "✅ Training completed!"