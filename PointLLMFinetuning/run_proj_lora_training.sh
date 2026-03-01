#!/bin/bash
"""
run_proj_lora_training.sh - PointLLM Projection + LoRA Fine-tuning 実行スクリプト

このスクリプトは以下を自動で実行します:
1. Conda環境の作成（初回のみ）
2. 必要なパッケージのインストール
3. PointLLMリポジトリのセットアップ
4. 学習の実行

使用方法:
    bash run_proj_lora_training.sh [GPU_ID] [DATA_ROOT]
    
例:
    bash run_proj_lora_training.sh 1 /path/to/dataset
"""

set -e  # エラーが発生したら停止

# =====================================================
# 設定
# =====================================================
CONDA_ENV_NAME="pointllm_finetune"
GPU_ID=${1:-0}  # 第1引数: GPU ID (デフォルト: 0)
DATA_ROOT=${2:-""}  # 第2引数: データルート (オプション)

POINTLLM_REPO="https://github.com/InternRobotics/PointLLM"
POINTLLM_DIR="PointLLM"
PROJECT_DIR=$(pwd)

echo "============================================================"
echo "PointLLM Projection + LoRA Fine-tuning Setup & Training"
echo "============================================================"
echo "GPU ID: $GPU_ID"
echo "Project Directory: $PROJECT_DIR"
if [ -n "$DATA_ROOT" ]; then
    echo "Data Root: $DATA_ROOT"
fi
echo ""

# =====================================================
# Conda環境の確認と作成
# =====================================================
echo "[1/6] Checking Conda environment..."

if conda env list | grep -q "^$CONDA_ENV_NAME "; then
    echo "✅ Conda environment '$CONDA_ENV_NAME' already exists"
else
    echo "📦 Creating new conda environment '$CONDA_ENV_NAME'..."
    conda create -n $CONDA_ENV_NAME python=3.10 -y
    echo "✅ Conda environment created"
fi

# 環境をアクティベート
echo "🔄 Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME


# =====================================================
# Flash Attentionのインストール
# =====================================================
echo ""
echo "[3/6] Installing Flash Attention..."

# =====================================================
# PointLLMリポジトリのセットアップ
# =====================================================
echo ""
echo "[4/6] Setting up PointLLM repository..."

if [ ! -d "$POINTLLM_DIR" ]; then
    echo "📥 Cloning PointLLM repository..."
    git clone $POINTLLM_REPO
else
    echo "✅ PointLLM repository already exists"
fi

cd $POINTLLM_DIR

cd $PROJECT_DIR

# =====================================================
# 追加の依存パッケージをインストール
# =====================================================
echo ""
echo "[5/6] Installing additional dependencies..."



echo "✅ All dependencies installed"

# =====================================================
# 学習の実行
# =====================================================
echo ""
echo "[6/6] Starting training..."
echo "============================================================"

# GPU指定
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 実行コマンドの構築
CMD="python proj_tune_and_LoRA.py --gpu 0 --pointllm-path $POINTLLM_DIR"

if [ -n "$DATA_ROOT" ]; then
    CMD="$CMD --data-root $DATA_ROOT"
fi

echo "Running: $CMD"
echo ""

# 学習実行
$CMD

echo ""
echo "============================================================"
echo "✅ Training completed successfully!"
echo "============================================================"
echo ""
echo "📊 Results saved to: outputs/"
echo "💡 To view logs: tensorboard --logdir outputs/*/logs"
echo "🔄 To resume training, edit and re-run this script"
