# PointLLM Fine-tuning - 特殊トークン修正ガイド

## 問題の概要

以前のコードでは、PointLLMの正しい特殊トークン処理を実装していませんでした：

### 誤った実装 (旧コード)
```python
# カスタムトークン: <point>
user_content = "<point>\n{question}"
```

### 正しい実装 (新コード)
```python
# PointLLM公式の特殊トークン:
# <point_start> + <point_patch> × 513 + <point_end>
point_tokens = "<point_start>" + ("<point_patch>" * 513) + "<point_end>"
user_content = f"{point_tokens}\n{question}"
```

## 主な変更点

### 1. build_dataset.py の修正

**変更ファイル**: `build_dataset_CORRECTED.py`

**主な変更**:
- ✅ `POINT_TOKEN_CONFIG` 設定を追加
  - `point_patch_token`: `<point_patch>`
  - `point_start_token`: `<point_start>`
  - `point_end_token`: `<point_end>`
  - `point_token_len`: 513
  - `mm_use_point_start_end`: True

- ✅ `format_point_token_sequence()` 関数を追加
  ```python
  def format_point_token_sequence() -> str:
      """
      Format: <point_start><point_patch>...(513回)...<point_end>
      """
      if mm_use_point_start_end:
          return point_start + (point_patch * 513) + point_end
      else:
          return point_patch * 513
  ```

- ✅ 会話サンプル生成を修正
  ```python
  conversations = [
      {
          "role": "user",
          "content": f"{point_tokens}\n{question}"  # 正しい特殊トークン列
      },
      {
          "role": "assistant",
          "content": response
      }
  ]
  ```

### 2. model_utils.py の修正

**変更ファイル**: `model_utils_CORRECTED.py`

**主な変更**:
- ✅ `initialize_pointllm_tokenizer()` 関数を追加
  - 3つの特殊トークンを正しく登録
  - 既存トークンの確認機能
  - トークンIDの検証

- ✅ `initialize_point_backbone_config()` 関数を追加
  - `point_token_len = 513` を設定
  - `mm_use_point_start_end = True` を設定
  - 特殊トークンIDを正しく設定
  - PointLLMの `initialize_tokenizer_point_backbone_config_wo_embedding()` を使用

- ✅ `prepare_model_for_training()` 関数の処理順序を修正
  1. 環境セットアップ
  2. モデル・トークナイザー読み込み
  3. **point_backbone_config初期化** ← 追加
  4. パラメータ凍結
  5. 勾配チェックポイント
  6. 検証

### 3. dataset.py の修正

**変更**: 不要

`dataset.py` は既にデータセットファイルから特殊トークンをそのまま読み込む設計なので、`build_dataset_CORRECTED.py` が正しいトークンを生成すれば自動的に対応します。

## マイグレーション手順

### ステップ1: 既存データセットの再構築

```bash
# 旧データセットをバックアップ
mv /path/to/datasets/annotations /path/to/datasets/annotations_backup

# 新しいデータセット構築スクリプトを使用
python build_dataset_CORRECTED.py \
    --point_cloud_dir "/path/to/point_clouds" \
    --hand_tracking_dir "/path/to/hand_tracking" \
    --output_dir "/path/to/datasets/annotations" \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

### ステップ2: model_utils.pyの置き換え

```bash
# 旧ファイルをバックアップ
cp model_utils.py model_utils_OLD.py

# 新しいファイルをコピー
cp model_utils_CORRECTED.py model_utils.py
```

### ステップ3: main_finetune.ipynb の修正

既存のノートブックでインポート文は変更不要ですが、以下を確認：

```python
# これらのインポートは変更なし
from model_utils import prepare_model_for_training
from dataset import PointLLMDataset, create_dataloaders

# モデル準備 - 内部で正しい特殊トークンが設定される
model, tokenizer = prepare_model_for_training(
    config=config.model,
    pointllm_path="/content/PointLLM",
    device="cuda"
)

# トークンIDの確認（デバッグ用）
print("Special Token IDs:")
print(f"  <point_patch>: {tokenizer.convert_tokens_to_ids('<point_patch>')}")
print(f"  <point_start>: {tokenizer.convert_tokens_to_ids('<point_start>')}")
print(f"  <point_end>: {tokenizer.convert_tokens_to_ids('<point_end>')}")
```

### ステップ4: 動作確認

```python
# サンプルデータで確認
sample_batch = next(iter(train_loader))
print("Input IDs shape:", sample_batch['input_ids'].shape)
print("Point clouds shape:", sample_batch['point_clouds'].shape)

# 特殊トークンが含まれているか確認
input_ids = sample_batch['input_ids'][0]
has_patch = (input_ids == tokenizer.convert_tokens_to_ids('<point_patch>')).any()
has_start = (input_ids == tokenizer.convert_tokens_to_ids('<point_start>')).any()
has_end = (input_ids == tokenizer.convert_tokens_to_ids('<point_end>')).any()

print(f"Contains <point_start>: {has_start}")
print(f"Contains <point_patch>: {has_patch}")
print(f"Contains <point_end>: {has_end}")

# <point_patch>の出現回数を確認（513回であるべき）
patch_count = (input_ids == tokenizer.convert_tokens_to_ids('<point_patch>')).sum().item()
print(f"Number of <point_patch> tokens: {patch_count}")
```

## 期待される動作

### データセット構築時
```
Point Token Configuration:
  Use start/end tokens: True
  Point token length: 513
  Token sequence format: <point_start><point_patch><point_patch>...<point_end>

Dataset Build Summary
Total samples: 150
  Train: 105
  Val: 22
  Test: 23
```

### モデル準備時
```
Token IDs:
  <point_patch>: 32000
  <point_start>: 32001
  <point_end>: 32002

Point Backbone Config:
  point_token_len: 513
  mm_use_point_start_end: True
  point_patch_token ID: 32000
  point_start_token ID: 32001
  point_end_token ID: 32002

✅ Point backbone config properly initialized
```

### 学習時
```
Batch contains:
  - Point clouds: torch.Size([4, 8192, 6])
  - Input IDs with proper special tokens
  - 513 <point_patch> tokens per sample
  - Correct <point_start> and <point_end> tokens
```

## トラブルシューティング

### 問題1: 特殊トークンが認識されない

**症状**: `token ID == unk_token_id`

**解決策**:
```python
# トークナイザーを確認
vocab = tokenizer.get_vocab()
print('<point_patch>' in vocab)  # True であるべき
print('<point_start>' in vocab)  # True であるべき
print('<point_end>' in vocab)    # True であるべき
```

### 問題2: point_backbone_config が None

**症状**: `AttributeError: 'PointLLMLlamaForCausalLM' object has no attribute 'point_backbone_config'`

**解決策**:
```python
# initialize_point_backbone_config() が呼ばれているか確認
# prepare_model_for_training() 内で自動的に呼ばれます
if not hasattr(model, 'point_backbone_config'):
    from model_utils import initialize_point_backbone_config
    initialize_point_backbone_config(model, tokenizer)
```

### 問題3: <point_patch> トークン数が513でない

**症状**: patch_count != 513

**原因**: データセット構築時の `point_token_len` 設定が間違っている

**解決策**:
```python
# build_dataset_CORRECTED.py で確認
POINT_TOKEN_CONFIG = {
    'point_token_len': 513,  # これが513であることを確認
    'mm_use_point_start_end': True,
}
```

## 参考: PointLLM原典の実装

```python
# pointllm/eval/eval_objaverse.py より
if mm_use_point_start_end:
    qs = default_point_start_token + \
         default_point_patch_token * point_token_len + \
         default_point_end_token + '\n' + qs
else:
    qs = default_point_patch_token * point_token_len + '\n' + qs
```

## チェックリスト

ファインチューニングを開始する前に、以下を確認してください：

- [ ] `build_dataset_CORRECTED.py` でデータセットを再構築した
- [ ] `model_utils_CORRECTED.py` を `model_utils.py` として使用している
- [ ] 特殊トークンがトークナイザーに正しく登録されている
- [ ] `point_backbone_config` が正しく初期化されている
- [ ] サンプルバッチに513個の `<point_patch>` トークンが含まれている
- [ ] `<point_start>` と `<point_end>` トークンが正しい位置にある

すべてのチェック項目が✅になったら、ファインチューニングを開始できます！

## まとめ

この修正により、PointLLMの原典実装と完全に互換性のある特殊トークン処理が実現されました。これにより：

1. ✅ Point Encoderが点群特徴を正しく抽出できる
2. ✅ Point Projectorが特徴を正しくLLM空間に射影できる
3. ✅ LLMが点群情報を適切に処理できる
4. ✅ ファインチューニングが期待通りに動作する

修正前の `<point>` トークンでは、モデルが点群情報を無視していた可能性が高いです。この修正により、正しく点群を処理できるようになります。
