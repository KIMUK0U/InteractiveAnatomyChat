#!/bin/bash
"""
run_main_training.sh - PointLLM Projection Fine-tuning 実行スクリプト

このスクリプトは以下を自動で実行します:
1. Conda環境の作成（初回のみ）
2. 必要なパッケージのインストール
3. PointLLMリポジトリのセットアップ
4. 学習の実行

使用方法:
    bash run_main_training.sh [GPU_ID] [DATA_ROOT]
    
例:
    bash run_main_training.sh 1 /path/to/dataset
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
echo "PointLLM Projection Fine-tuning Setup & Training"
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
# PyTorchのインストール
# =====================================================
echo ""
echo "[2/6] Installing PyTorch 2.4.1 with CUDA 12.1..."

# 既にインストール済みかチェック
if python -c "import torch; assert torch.__version__ == '2.4.1'" 2>/dev/null; then
    echo "✅ PyTorch 2.4.1 already installed"
else
    echo "📦 Installing PyTorch..."
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121
    echo "✅ PyTorch installed"
fi

# =====================================================
# Flash Attentionのインストール
# =====================================================
echo ""
echo "[3/6] Installing Flash Attention..."

if python -c "import flash_attn" 2>/dev/null; then
    echo "✅ Flash Attention already installed"
else
    echo "📦 Installing Flash Attention (this may take a while)..."
    pip install ninja packaging
    pip install flash-attn==2.6.1 --no-build-isolation
    echo "✅ Flash Attention installed"
fi

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

# pyproject.tomlの修正
echo "📝 Updating pyproject.toml..."
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "pointllm"
version = "0.1.2"
description = "Empower large language models to understand point clouds."
readme = "README.md"
requires-python = ">=3.8"
dependencies = [
    "accelerate",
    "einops",
    "fastapi",
    "gradio",
    "markdown2[all]",
    "numpy",
    "requests",
    "sentencepiece",
    "tokenizers==0.19.1",
    "uvicorn",
    "wandb",
    "shortuuid",
    "deepspeed",
    "peft==0.12.0",
    "transformers==4.44.2",
    "openai",
    "tqdm",
    "easydict",
    "timm==0.4.12",
    "ftfy==6.0.1",
    "regex",
    "open3d",
    "h5py",
    "termcolor",
    "plyfile",
    "nltk",
    "rouge",
    "scikit-learn",
    "py-rouge",
]
EOF

# PointLLMのインストール
if python -c "import pointllm" 2>/dev/null; then
    echo "✅ PointLLM already installed"
else
    echo "📦 Installing PointLLM..."
    pip install -e .
    echo "✅ PointLLM installed"
fi

# PointNet++ 演算モジュールのコンパイル
if [ -d "pointnet2_ops_lib" ]; then
    echo "📦 Compiling PointNet++ ops..."
    cd pointnet2_ops_lib
    pip install .
    cd ..
    echo "✅ PointNet++ ops compiled"
fi

cd $PROJECT_DIR

# =====================================================
# 追加の依存パッケージをインストール
# =====================================================
echo ""
echo "[5/6] Installing additional dependencies..."

# requirements.txtがあればインストール
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# 必須パッケージを個別に確認
REQUIRED_PACKAGES=(
    "scipy"
    "matplotlib"
    "tensorboard"
)

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo "✅ $package already installed"
    else
        echo "📦 Installing $package..."
        pip install $package
    fi
done

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
CMD="python main_finetune.py --gpu 0 --pointllm-path $POINTLLM_DIR"

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
