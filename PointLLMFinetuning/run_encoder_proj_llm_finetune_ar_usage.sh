#!/bin/bash
set -e

# CUDA デバイス順序を明示的に設定
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# GPU番号チェック
GPU_ID=${1}

if [ -z "$GPU_ID" ]; then
    echo "❌ エラー: GPU番号を指定してください。"
    echo "例: ./run_encoder_proj_llm_finetune.sh 1"
    exit 1
fi

CONDA_ENV_NAME="pointllm_finetune"
DATA_ROOT=${2:-""}    # 第2引数: データルート（オプション）

POINTLLM_DIR="PointLLM"
PROJECT_DIR=$(pwd)

# 事前学習済みEncoder+Projectorの重みパス
PRETRAINED_PATH="outputs/encoder_projector_finetune_v2/checkpoints/best_model"

echo "============================================================"
echo "PointLLM Encoder凍結 + Projector + LLM Full Fine-tuning"
echo "============================================================"
echo "GPU: $GPU_ID"
echo "Pretrained Path: $PRETRAINED_PATH"
echo "============================================================"

echo "[1/4] Setting up Conda environment..."
if ! conda env list | grep -q "^$CONDA_ENV_NAME "; then
    echo "Creating new conda environment: $CONDA_ENV_NAME"
    conda create -n "$CONDA_ENV_NAME" python=3.10 -y
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

echo "✅ Conda environment activated: $CONDA_ENV_NAME"

echo "[2/4] Installing dependencies..."
# 必要に応じて依存関係をインストール
# pip install -r requirements.txt

echo "[3/4] Checking PointLLM repository..."
if [ ! -d "$POINTLLM_DIR" ]; then
    echo "Cloning PointLLM repository..."
    git clone https://github.com/OpenRobotLab/PointLLM.git "$POINTLLM_DIR"
    cd "$POINTLLM_DIR"
    pip install -e .
    cd "$PROJECT_DIR"
    echo "✅ PointLLM installed"
else
    echo "✅ PointLLM already exists"
fi

echo "[4/4] Starting training..."

# 事前学習済みチェックポイントの存在確認
if [ ! -d "$PRETRAINED_PATH" ]; then
    echo "❌ Error: Pretrained checkpoint not found at $PRETRAINED_PATH"
    echo "Please run encoder_proj_tune.py first to train Encoder+Projector"
    exit 1
fi

echo "✅ Using pretrained weights from: $PRETRAINED_PATH"

# GPU設定（指定したGPUのみを使用）
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# PYTHONPATHの設定
export PYTHONPATH="$PROJECT_DIR/$POINTLLM_DIR:$PYTHONPATH"

# デバッグ情報
echo ""
echo "=== Debug Information ==="
echo "[DEBUG bash] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[DEBUG bash] PYTHONPATH=$PYTHONPATH"

python - <<'PY'
import os, torch, sys
print("[DEBUG py] Python executable:", sys.executable)
print("[DEBUG py] PyTorch version:", torch.__version__)
print("[DEBUG py] CUDA version:", torch.version.cuda)
print("[DEBUG py] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("[DEBUG py] CUDA available:", torch.cuda.is_available())
print("[DEBUG py] GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("[DEBUG py] Current device:", torch.cuda.current_device())
    print("[DEBUG py] Device name:", torch.cuda.get_device_name(0))
PY

echo "========================="
echo ""

# コマンド構築
CMD="python encoder_proj_llm_finetune_ar_usage.py --gpu 0 --pointllm-path $POINTLLM_DIR --pretrained-path $PRETRAINED_PATH"

if [ -n "$DATA_ROOT" ]; then
    CMD="$CMD --data-root $DATA_ROOT"
fi

echo "Running command: $CMD"
echo ""

# トレーニング実行
$CMD

echo ""
echo "============================================================"
echo "✅ Training completed successfully!"
echo "============================================================"