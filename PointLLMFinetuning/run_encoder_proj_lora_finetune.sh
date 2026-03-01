#!/bin/bash
set -e
# ★これを追加！ nvidia-smi の順番(上から0, 1)とプログラムの認識を確実に合わせる
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# 第1引数があればそれを使う。なければエラーにして止める（事故防止）
GPU_ID=${1}

if [ -z "$GPU_ID" ]; then
    echo "❌ エラー: GPU番号を指定してください。"
    echo "例: ./script.sh 1"
    exit 1
fi

CONDA_ENV_NAME="pointllm_finetune"

GPU_ID=${1:-""}       # 第1引数: 物理GPU番号（空なら全GPU）
DATA_ROOT=${2:-""}    # 第2引数: データルート

POINTLLM_DIR="PointLLM"
PROJECT_DIR=$(pwd)

# 事前学習済みEncoder+Projectorの重みパス
PRETRAINED_PATH="/home/yyamashita/Desktop/kkimu/test/ProjectionFinetuning/outputs/encoder_proj_tune_low_lr_v10_demo_ar_dataset_32_final/checkpoints/checkpoint-epoch-20"

echo "============================================================"
echo "PointLLM Encoder + Projector + LoRA Fine-tuning"
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

# 事前学習済みチェックポイントの存在確認
if [ ! -d "$PRETRAINED_PATH" ]; then
    echo "❌ Error: Pretrained checkpoint not found at $PRETRAINED_PATH"
    echo "Please run encoder_proj_tune.py first to train Encoder+Projector"
    exit 1
fi

echo "✅ Using pretrained weights from: $PRETRAINED_PATH"

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

# ... (前半はそのまま) ...

# [重要] exportは消して、コマンドラインに直接埋め込む方針に変更
# ここで export しても消えることがあるため、CMD作成時に埋め込みます

export PYTHONPATH="$PROJECT_DIR/$POINTLLM_DIR:$PYTHONPATH"

# コマンド構築：変数を python の直前に置くことで強制適用させる
# "env" コマンドを使うとより確実です
CMD="env CUDA_VISIBLE_DEVICES=$GPU_ID python encoder_proj_LoRA_finetune.py --gpu 0 --pointllm-path $POINTLLM_DIR --pretrained-path $PRETRAINED_PATH"

if [ -n "$DATA_ROOT" ]; then
    CMD="$CMD --data-root $DATA_ROOT"
fi

echo "Running with FORCE GPU ISOLATION: $CMD"
$CMD

echo "✅ Training completed!"