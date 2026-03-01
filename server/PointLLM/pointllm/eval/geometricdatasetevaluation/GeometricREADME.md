# Geometric Dataset PointLLM Evaluation System

PointLLMを使用して幾何学構造データセットの推論と評価を行うシステムです。

## 機能

- **自動フォルダ探索**: `geometric_datasets`ディレクトリ構造を自動的に探索
- **バッチ推論**: 全てのnpyファイルに対してPointLLMで推論を実行
- **複数の評価指標**:
  - **完全一致 (Exact Match)**: 正規化後のテキストが完全に一致するか
  - **部分一致 (Partial Match)**: トークンレベルのF1スコア
  - **意味的類似度 (Semantic Similarity)**: GPT APIによる意味的評価（オプション）
- **再評価機能**: 推論結果のCSVファイルから再評価可能
- **詳細なレポート生成**: CSV、JSON、HTMLフォーマットでの出力

## ディレクトリ構造

```
geometric_dataset_evaluation/
├── geometric_dataset_inference.py  # メイン推論スクリプト
├── evaluation_metrics.py           # 評価メトリクスクラス
├── run_evaluation.py               # 再評価用スクリプト
├── requirements.txt                # 依存パッケージ
├── .env.template                   # 環境変数テンプレート
└── README.md                       # このファイル
```

## 期待されるデータ構造

```
geometric_datasets/
├── inclusion/
│   └── level_01/
│       ├── random_color/
│       │   ├── point_clouds_npy/    <- .npyファイル (x,y,z,r,g,b形式)
│       │   ├── metadata/            <- .jsonファイル (qa_pairs含む)
│       │   └── results/             <- 生成される結果フォルダ
│       └── similar_color/
│           └── ...
├── entanglement/
├── branching/
├── adjacency/
└── nested/
    └── inclusion_entanglement/
        └── ...
```

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. PointLLMの準備

PointLLMリポジトリをクローンまたはインストール:

```bash
git clone https://github.com/OpenRobotLab/PointLLM.git
```

### 3. 環境変数の設定（意味的評価を使用する場合）

```bash
cp .env.template .env
# .envファイルを編集してOPENAI_API_KEYを設定
```

## 使用方法

### 基本的な推論と評価

```bash
python geometric_dataset_inference.py \
    --base_dir /path/to/geometric_datasets \
    --model_path RunsenXu/PointLLM_7B_v1.2 \
    --device mps
```

### テスト実行（各フォルダ5シーンのみ）

```bash
python geometric_dataset_inference.py \
    --base_dir /path/to/geometric_datasets \
    --max_scenes 5
```

### 意味的評価を含む実行

```bash
python geometric_dataset_inference.py \
    --base_dir /path/to/geometric_datasets \
    --use_semantic \
    --openai_model gpt-4.1-nano
```

### 再評価のみ実行

推論結果が既に存在する場合、評価のみを再実行できます:

```bash
python run_evaluation.py \
    --base_dir /path/to/geometric_datasets
```

意味的評価を追加:

```bash
python run_evaluation.py \
    --base_dir /path/to/geometric_datasets \
    --use_semantic

python run_ollama_eval.py --base_dir /Users/user/Documents/kimura/study/1020PointNetLLM/ClaudeGenProg/geometric_datasets --use_semantic --use_ollama --ollama_model qwen3:8b
```

## コマンドライン引数

### geometric_dataset_inference.py

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--base_dir` | (必須) | geometric_datasetsのベースディレクトリ |
| `--model_path` | `RunsenXu/PointLLM_7B_v1.2` | PointLLMモデルパス |
| `--pointllm_path` | `PointLLM` | PointLLMリポジトリパス |
| `--device` | `mps` | 使用デバイス (cuda/cpu/mps) |
| `--use_semantic` | False | GPT APIによる意味的評価を使用 |
| `--openai_model` | `gpt-4.1-nano` | OpenAIモデル |
| `--max_scenes` | -1 | フォルダあたりの最大シーン数 (-1で全て) |
| `--no_visualizations` | False | 可視化画像の保存を無効化 |

### run_evaluation.py

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--base_dir` | (必須) | geometric_datasetsのベースディレクトリ |
| `--use_semantic` | False | GPT APIによる意味的評価を使用 |
| `--openai_model` | `gpt-4.1-nano` | OpenAIモデル |
| `--single_csv` | None | 単一のCSVファイルのみ評価 |

## 出力ファイル

### フォルダごとの出力 (`results/`)

- `inference_results.csv`: 推論結果
- `evaluation_metrics.json`: 評価メトリクス
- `visualizations/`: 点群の可視化画像

### 全体の出力 (`geometric_datasets/`)

- `evaluation_summary_YYYYMMDD_HHMMSS.csv`: フォルダごとの評価サマリー
- `evaluation_detailed_YYYYMMDD_HHMMSS.csv`: 全QAペアの詳細結果
- `evaluation_stats_YYYYMMDD_HHMMSS.json`: 集計統計
- `evaluation_report_YYYYMMDD_HHMMSS.html`: HTMLレポート

## 評価メトリクス

### 完全一致 (Exact Match)

正規化されたテキストが完全に一致する場合は1.0、そうでない場合は0.0

正規化処理:
- 小文字変換
- 余分な空白の除去
- 末尾のピリオド除去

### 部分一致 (Partial Match)

トークンレベルのF1スコア (0.0 〜 1.0)

```
F1 = 2 * precision * recall / (precision + recall)
```

### 意味的類似度 (Semantic Similarity)

GPT APIを使用して、予測と正解の意味的な類似度を0.0〜1.0で評価

- 1.0: 同じ意味
- 0.8-0.9: 軽微な違いはあるが本質的に正解
- 0.5-0.7: 部分的に正解
- 0.2-0.4: ほとんど不正解だが関連要素あり
- 0.0-0.1: 完全に不正解

## メタデータJSONフォーマット

```json
{
  "scene_id": 0,
  "scene_type": "inclusion",
  "level": 1,
  "color_mode": "random_color",
  "num_objects": 2,
  "objects": [...],
  "relations": [...],
  "qa_pairs": [
    {
      "question": "Count the number of distinct objects present.",
      "answer": "There are 2 distinct objects in the scene."
    },
    ...
  ]
}
```

## トラブルシューティング

### PointLLMのインポートエラー

```
[Error] Failed to import PointLLM modules
```

→ `--pointllm_path`でPointLLMリポジトリの正しいパスを指定してください。

### OpenAI APIエラー

```
OPENAI_API_KEY not found in environment variables
```

→ `.env`ファイルを作成し、`OPENAI_API_KEY`を設定してください。

### メモリ不足

GPUメモリが不足する場合は`--device cpu`を試してください。

## ライセンス

MIT License
