#!/bin/bash
"""
run_main_training_no_flash.sh - Flash Attentionなしで実行

Flash Attentionのコンパイルをスキップし、より高速にセットアップします。
Flash Attentionなしでも動作しますが、学習速度が若干遅くなります。

使用方法:
    bash run_main_training_no_flash.sh [GPU_ID] [DATA_ROOT]
    
例:
    bash run_main_training_no_flash.sh 1 /path/to/dataset
"""

set -e  # エラーが発生したら停止

# =====================================================
# 設定
# =====================================================
CONDA_ENV_NAME="pointllm_finetune"
GPU_ID=${1:-0}
DATA_ROOT=${2:-""}

POINTLLM_REPO="https://github.com/InternRobotics/PointLLM"
POINTLLM_DIR="PointLLM"
PROJECT_DIR=$(pwd)

echo "============================================================"
echo "PointLLM Projection Fine-tuning (No Flash Attention)"
echo "============================================================"
echo "GPU ID: $GPU_ID"
echo "Project Directory: $PROJECT_DIR"
if [ -n "$DATA_ROOT" ]; then
    echo "Data Root: $DATA_ROOT"
fi
echo "⚠️ Flash Attention will be SKIPPED for faster setup"
echo ""

# =====================================================
# Conda環境の確認と作成
# =====================================================
echo "[1/5] Checking Conda environment..."

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
echo "[2/5] Installing PyTorch 2.4.1 with CUDA 12.1..."

if python -c "import torch; assert torch.__version__ == '2.4.1'" 2>/dev/null; then
    echo "✅ PyTorch 2.4.1 already installed"
else
    echo "📦 Installing PyTorch..."
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121
    echo "✅ PyTorch installed"
fi

# =====================================================
# PointLLMリポジトリのセットアップ
# =====================================================
echo ""
echo "[3/5] Setting up PointLLM repository..."

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
echo "[4/5] Installing additional dependencies..."

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
# config.pyの設定確認
# =====================================================
echo ""
echo "⚠️ IMPORTANT: Flash Attention is NOT installed"
echo ""
echo "Please ensure your config.py has:"
echo "  config.model.use_flash_attention = False"
echo ""
echo "Creating a config override file..."

# config_no_flash.pyを作成（既に存在しない場合のみ）
if [ ! -f "config_no_flash.py" ]; then
cat > config_no_flash.py << 'EOF'
"""
config_no_flash.py - Flash Attentionなしの設定

このファイルをインポートすることで、Flash Attentionを無効化した設定を使用できます。

使用方法:
    from config_no_flash import create_no_flash_config
    config = create_no_flash_config()
"""

from config import FullConfig, create_default_config

def create_no_flash_config() -> FullConfig:
    """Flash Attentionを無効化した設定を作成"""
    config = create_default_config()
    config.model.use_flash_attention = False
    return config

if __name__ == "__main__":
    config = create_no_flash_config()
    config.print_summary()
EOF
    echo "✅ Created config_no_flash.py"
else
    echo "✅ config_no_flash.py already exists"
fi

# =====================================================
# 学習の実行
# =====================================================
echo ""
echo "[5/5] Starting training..."
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
echo "⚠️ Note: Flash Attention is disabled"
echo "   Make sure config.model.use_flash_attention = False"
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
echo ""
echo "📝 Note: Training without Flash Attention is slightly slower"
echo "   but should work reliably on all GPUs."
