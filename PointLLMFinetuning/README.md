# PointLLM Point Projector Fine-tuning - ローカルGPU実行ガイド

このディレクトリには、PointLLMのPoint Projectorをローカル環境の複数GPUで実行するためのスクリプトが含まれています。

## ファイル構成

```
.
├── check_gpu.py                           # GPU確認スクリプト
├── proj_tune_and_LoRA.py                  # Projection + LoRA学習スクリプト
├── main_finetune.py                       # Projection学習スクリプト
├── run_proj_lora_training.sh              # Projection + LoRA実行（Flash Attention対応）
├── run_main_training.sh                   # Projection実行（Flash Attention対応）
├── run_proj_lora_training_no_flash.sh     # Projection + LoRA実行（Flash Attentionなし）⭐推奨
├── run_main_training_no_flash.sh          # Projection実行（Flash Attentionなし）⭐推奨
├── trainer.py                             # 標準トレーナー（改良版）
├── trainer_LoRA.py                        # LoRA用トレーナー（改良版）
├── config.py                              # 設定ファイル
├── dataset.py                             # データセット処理
├── model_utils.py (or model_utils_v1.py)  # モデルユーティリティ
└── evaluation.py                          # 評価ユーティリティ
```

## ⚠️ Flash Attentionについて

Flash Attentionはコンパイルに時間がかかり、環境によっては失敗することがあります。

### 推奨: Flash Attentionなしで実行

最も確実な方法は、Flash Attentionを使わないスクリプトを使用することです：

```bash
# Projection + LoRA学習（推奨）
bash run_proj_lora_training_no_flash.sh 1 /path/to/dataset

# Projection学習（推奨）
bash run_main_training_no_flash.sh 1 /path/to/dataset
```

**利点:**
- ✅ セットアップが高速（Flash Attentionのコンパイル不要）
- ✅ 確実に動作する
- ✅ ほとんどのGPUで問題なし

**欠点:**
- ⚠️ 学習速度が若干遅くなる（5-10%程度）

### Flash Attentionありで実行（上級者向け）

```bash
# 対話形式でFlash Attentionのインストール確認
bash run_proj_lora_training.sh 1 /path/to/dataset
bash run_main_training.sh 1 /path/to/dataset
```

インストール時に選択肢が表示されます：
- `y`: Flash Attentionをインストール（10-30分かかる可能性）
- `n/skip`: スキップして続行

## セットアップ手順

### 1. GPU確認

まず、使用可能なGPUを確認します：

```bash
python check_gpu.py
```

このスクリプトは以下を表示します：
- 利用可能なGPUの数と名前
- 各GPUのメモリ使用状況
- 使用推奨GPU（空きメモリが最大のGPU）

**出力例:**
```
============================================================
GPU Details
============================================================
Number of GPUs: 2

GPU 0:
  Name: NVIDIA GeForce RTX 4090
  Total Memory: 24.00 GB
  Memory Free: 23.50 GB

GPU 1:
  Name: NVIDIA GeForce RTX 4090
  Total Memory: 24.00 GB
  Memory Free: 10.20 GB

============================================================
Recommended GPU for Training
============================================================
✅ RECOMMENDED GPU 0: NVIDIA GeForce RTX 4090
     Free: 23.50 GB / 24.00 GB

💡 Use GPU 0 for training
```

### 2. データセットの準備

データセットを以下の構造で配置します：

```
your_dataset/
├── point_clouds/
│   ├── sample_001.npy
│   ├── sample_002.npy
│   └── ...
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json (optional)
```

詳細は元の `SETUP.md` を参照してください。

### 3. 学習の実行

#### 🌟 推奨方法: Flash Attentionなしで実行

```bash
# Projection + LoRA学習
bash run_proj_lora_training_no_flash.sh 1 /path/to/your/dataset

# Projection学習
bash run_main_training_no_flash.sh 1 /path/to/your/dataset
```

#### 代替方法: Flash Attentionありで実行

```bash
# 対話形式で確認しながら実行
bash run_proj_lora_training.sh 1 /path/to/your/dataset
bash run_main_training.sh 1 /path/to/your/dataset
```

## シェルスクリプトの動作

各シェルスクリプトは以下を自動的に実行します：

### 標準版（Flash Attention対応）
1. Conda環境の作成 (`pointllm_finetune`)
2. PyTorch 2.4.1 + CUDA 12.1のインストール
3. **Flash Attentionのインストール確認（対話形式）**
4. PointLLMのセットアップ
5. 依存パッケージのインストール
6. 学習の実行

### No Flash版（推奨）
1. Conda環境の作成 (`pointllm_finetune`)
2. PyTorch 2.4.1 + CUDA 12.1のインストール
3. **Flash Attentionをスキップ**
4. PointLLMのセットアップ
5. 依存パッケージのインストール
6. `config_no_flash.py` の作成
7. 学習の実行

## Python スクリプトの直接実行

シェルスクリプトを使わず、Pythonスクリプトを直接実行することもできます：

### Projection + LoRA学習

```bash
# Conda環境をアクティベート
conda activate pointllm_finetune

# Flash Attentionなしで実行する場合、config.pyを編集
# config.model.use_flash_attention = False

# GPU 1を使用して実行
python proj_tune_and_LoRA.py \
    --gpu 1 \
    --pointllm-path ./PointLLM \
    --data-root /path/to/dataset
```

### Projection学習

```bash
conda activate pointllm_finetune

python main_finetune.py \
    --gpu 1 \
    --pointllm-path ./PointLLM \
    --data-root /path/to/dataset
```

## 📊 学習曲線とロス監視

改良版のtrainerは以下の機能を持っています：

### リアルタイムロス曲線

学習中、以下が自動的に記録・可視化されます：
- **Train Loss**: エポックごとの学習ロス
- **Val Loss**: 検証ロス（eval_stepsごと）
- **Learning Rate**: 学習率の推移

### 保存されるファイル

```
outputs/
└── [experiment_name]/
    ├── logs/
    │   └── loss_curves.png  # 📈 学習曲線グラフ
    └── checkpoints/
        ├── checkpoint-500/
        │   ├── point_proj.pt
        │   ├── trainer_state.pt  # 学習履歴含む
        │   └── history.json      # 学習履歴（JSON）
        └── best_model/
```

### TensorBoardでの確認

```bash
tensorboard --logdir outputs/
# ブラウザで http://localhost:6006 を開く
```

### config.pyでの設定

```python
# 検証評価の頻度
config.training.eval_strategy = "steps"  # または "epoch"
config.training.eval_steps = 250  # 250ステップごとに検証

# ログ出力の頻度
config.training.logging_steps = 10
```

## トラブルシューティング

### 1. Flash Attention インストール失敗

**症状**: `subprocess-exited-with-error` エラー

**解決策**:
```bash
# Flash Attentionなしのスクリプトを使用（推奨）
bash run_proj_lora_training_no_flash.sh 1 /path/to/dataset

# または、config.pyで無効化
config.model.use_flash_attention = False
```

### 2. CUDA Out of Memory

**症状**: GPU メモリ不足エラー

**解決策**:
```python
# config.pyで調整
config.training.batch_size = 16  # デフォルト: 32
config.training.gradient_accumulation_steps = 8  # デフォルト: 4
```

### 3. ModuleNotFoundError: pointllm

**症状**: `pointllm` モジュールが見つからない

**解決策**:
```bash
# Conda環境を確認
conda activate pointllm_finetune

# PointLLMを再インストール
cd PointLLM
pip install -e .
```

### 4. transformers バージョン不一致

**症状**: 推論時に異常な出力

**解決策**:
```bash
# 正しいバージョンを強制インストール
pip install transformers==4.44.2 --force-reinstall
```

### 5. PointNet++ コンパイルエラー

**症状**: `pointnet2_ops_lib` のインストールに失敗

**解決策**:
```bash
# ninjaとpackagingを先にインストール
pip install ninja packaging

# 手動でコンパイル
cd PointLLM/pointnet2_ops_lib
pip install .
```

## コマンドライン引数

### proj_tune_and_LoRA.py / main_finetune.py

| 引数 | 説明 | デフォルト |
|------|------|-----------|
| `--gpu` | 使用するGPU ID | 0 |
| `--config` | 設定ファイルのパス (JSON) | なし |
| `--pointllm-path` | PointLLMリポジトリのパス | `/content/PointLLM` |
| `--project-path` | プロジェクトディレクトリのパス | 現在のディレクトリ |
| `--data-root` | データセットのルートパス | なし |
| `--skip-inference-test` | 学習後の推論テストをスキップ | False |

## 設定のカスタマイズ

### config.pyを直接編集

```python
from config import FullConfig, create_default_config

config = create_default_config()

# Flash Attentionの設定
config.model.use_flash_attention = False  # 推奨: False

# データセットのパス
config.data.data_root = "/your/dataset/path"

# 学習パラメータ
config.training.num_epochs = 5
config.training.batch_size = 32
config.training.learning_rate = 5e-4

# 検証評価の設定
config.training.eval_strategy = "steps"
config.training.eval_steps = 250

# 出力設定
config.output.experiment_name = "my_experiment"
config.output.use_wandb = False
```

### JSONファイルで設定を保存

```python
config.save("my_config.json")

# 読み込み
config = FullConfig.load("my_config.json")
```

## 学習の監視

### リアルタイム監視

学習中、以下が表示されます：
```
Epoch 1/5: 100%|██████████| 100/100 [05:23<00:00, loss=2.3456, lr=5.0e-04]

============================================================
🔍 Running Validation
============================================================
Evaluating: 100%|██████████| 20/20 [00:30<00:00]
📊 Validation Loss: 2.1234
============================================================

📈 Loss curves saved to: outputs/experiment_name/logs/loss_curves.png
```

### Weights & Biases（オプション）

```python
# config.pyで有効化
config.output.use_wandb = True
config.output.wandb_project = "pointllm-finetune"
```

## パフォーマンス比較

| 設定 | 学習速度 | セットアップ時間 | 安定性 |
|------|----------|-----------------|--------|
| Flash Attentionあり | 速い（100%） | 長い（20-40分） | 環境依存 ⚠️ |
| Flash Attentionなし | やや遅い（90-95%） | 短い（5-10分） | 安定 ✅ |

**推奨**: ほとんどの場合、Flash Attentionなしで十分です。

## 注意事項

1. **transformersバージョン**: 必ず `transformers==4.44.2` を使用
2. **点群の正規化**: 入力点群は `pc_normalize` 関数で正規化が必要
3. **点数の制限**: PointLLMは最大8192点まで対応
4. **メモリ要件**: 少なくとも16GB以上のGPUメモリを推奨
5. **Flash Attention**: 学習速度向上は5-10%程度なので、インストールに問題がある場合は無効化を推奨

## 参考資料

- [PointLLM GitHub](https://github.com/InternRobotics/PointLLM)
- [PointLLM論文](https://arxiv.org/abs/2308.16911)
- 元の `README.md`, `SETUP.md`, `MIGRATION_GUIDE.md` も参照してください
