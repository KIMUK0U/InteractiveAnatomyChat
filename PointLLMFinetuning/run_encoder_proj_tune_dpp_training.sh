#!/bin/bash
set -e

CONDA_ENV_NAME="pointllm_finetune"

# 使用するGPU数（デフォルト: 2）
NUM_GPUS=${1:-2}
DATA_ROOT=${2:-""}
RESUME_PATH=${3:-""}

POINTLLM_DIR="PointLLM"
PROJECT_DIR=$(pwd)

echo "============================================================"
echo "PointLLM DDP Training (Multi-GPU)"
echo "============================================================"
echo "Number of GPUs: $NUM_GPUS"

echo "[1/3] Activating Conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

echo "[2/3] Setting up environment..."
# Blackwell/Hopper最適化
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_COMPILE_DISABLE_VOLATILE_SERVER=1

# NCCL設定（通信最適化）
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0  # InfiniBandがある場合は0、ない場合は1
export NCCL_P2P_DISABLE=0  # GPU間直接通信

export PYTHONPATH="$PROJECT_DIR/$POINTLLM_DIR:$PYTHONPATH"

echo "[3/3] Starting DDP training..."

# torchrunコマンド構築
CMD="torchrun --nproc_per_node=$NUM_GPUS encoder_proj_tune_ddp.py --pointllm-path $POINTLLM_DIR"

if [ -n "$DATA_ROOT" ]; then
    CMD="$CMD --data-root $DATA_ROOT"
fi

if [ -n "$RESUME_PATH" ]; then
    CMD="$CMD --resume-path $RESUME_PATH"
fi

echo "Running: $CMD"
$CMD

echo "✅ DDP Training completed!"