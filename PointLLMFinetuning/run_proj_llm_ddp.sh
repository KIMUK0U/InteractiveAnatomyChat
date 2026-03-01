#!/bin/bash
set -e

CONDA_ENV_NAME="pointllm_finetune"

# 引数パース
# 使い方: bash run_proj_llm_ddp.sh [NUM_GPUS] [PRETRAINED_PATH] [DATA_ROOT] [RESUME_PATH]
NUM_GPUS=${1:-2}
PRETRAINED_PATH=${2:-""}
DATA_ROOT=${3:-""}
RESUME_PATH=${4:-""}

POINTLLM_DIR="PointLLM"
PROJECT_DIR=$(pwd)

echo "============================================================"
echo "PointLLM DDP Training (Projector + LLM)"
echo "============================================================"
echo "Number of GPUs: $NUM_GPUS"
echo "Pretrained path: ${PRETRAINED_PATH:-<required>}"
echo "Resume path: ${RESUME_PATH:-<none - start from scratch>}"

if [ -z "$PRETRAINED_PATH" ]; then
    echo "❌ Error: Pretrained path required"
    echo ""
    echo "Usage:"
    echo "  New training:  bash run_proj_llm_ddp.sh NUM_GPUS PRETRAINED_PATH [DATA_ROOT]"
    echo "  Resume:        bash run_proj_llm_ddp.sh NUM_GPUS PRETRAINED_PATH DATA_ROOT RESUME_PATH"
    echo ""
    echo "Examples:"
    echo "  bash run_proj_llm_ddp.sh 2 /path/to/encoder_proj/best_model"
    echo "  bash run_proj_llm_ddp.sh 2 /path/to/encoder_proj/best_model '' ./outputs/checkpoints/checkpoint-epoch-3"
    exit 1
fi

if [ ! -d "$PRETRAINED_PATH" ]; then
    echo "❌ Error: Pretrained path does not exist: $PRETRAINED_PATH"
    exit 1
fi

# Resume pathのチェック
if [ -n "$RESUME_PATH" ]; then
    if [ ! -d "$RESUME_PATH" ]; then
        echo "❌ Error: Resume path does not exist: $RESUME_PATH"
        exit 1
    fi
    echo "🔄 Resuming from checkpoint: $RESUME_PATH"
fi

echo "[1/3] Activating Conda..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

echo "[2/3] Setting up environment..."
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_COMPILE_DISABLE_VOLATILE_SERVER=1

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

export PYTHONPATH="$PROJECT_DIR/$POINTLLM_DIR:$PYTHONPATH"

echo "[3/3] Starting DDP training..."

CMD="torchrun --nproc_per_node=$NUM_GPUS encoder_proj_llm_finetune_ddp.py --pointllm-path $POINTLLM_DIR --pretrained-path $PRETRAINED_PATH"

if [ -n "$DATA_ROOT" ]; then
    CMD="$CMD --data-root $DATA_ROOT"
fi

if [ -n "$RESUME_PATH" ]; then
    CMD="$CMD --resume-path $RESUME_PATH"
fi

echo "Running: $CMD"
$CMD

echo "✅ DDP Training completed!"