# LoRAファイル名の修正について

## 問題点

誤って `adapter.pt` というファイル名を参照していましたが、実際にPEFTライブラリの`save_pretrained()`が生成するファイル名は：
- `adapter_model.bin` (PyTorch形式、デフォルト)
- `adapter_model.safetensors` (SafeTensors形式、オプション)

です。

## 修正内容

### 1. `pointllm_manager/model_manager.py`

**変更点:**
- `_validate_checkpoint_dir()`メソッドで、以下のファイルをチェックするように修正：
  - `adapter_config.json` (LoRA設定)
  - `adapter_model.bin` (LoRAの重み - PyTorch形式)
  - `adapter_model.safetensors` (LoRAの重み - SafeTensors形式)
  - `point_proj.pt` (Point Projectorの重み)

**コード変更:**
```python
# 修正前
adapter_path = checkpoint_dir / "adapter.pt"
if adapter_config_path.exists() or adapter_path.exists():
    files_found.append("LoRA adapter")

# 修正後
adapter_model_bin = checkpoint_dir / "adapter_model.bin"
adapter_model_safetensors = checkpoint_dir / "adapter_model.safetensors"

has_lora = False
if adapter_config_path.exists():
    files_found.append("adapter_config.json")
    has_lora = True

if adapter_model_bin.exists():
    files_found.append("adapter_model.bin")
    has_lora = True
elif adapter_model_safetensors.exists():
    files_found.append("adapter_model.safetensors")
    has_lora = True
```

### 2. `README.md`

**変更点:**
- チェックポイントディレクトリ構造の説明を修正
- セットアップ手順の確認コマンドを更新
- LoRA設定セクションの情報を修正

**主要な修正箇所:**
```markdown
# 修正前
20260108_checkpoint-epoch-5/
├── adapter_config.json
├── adapter.pt
└── point_proj.pt

# 修正後
20260108_checkpoint-epoch-5/
├── adapter_config.json
├── adapter_model.bin (または adapter_model.safetensors)
└── point_proj.pt
```

### 3. 新規作成: `validate_checkpoint.py`

チェックポイントディレクトリの内容を検証するスクリプトを作成しました。

**機能:**
- 必須ファイルの存在確認
- ファイルサイズの表示
- `adapter_config.json`の内容解析
- 総合的な検証レポート

**使用方法:**
```bash
python validate_checkpoint.py pointllm/outputs/projection_finetune_v1/checkpoints/20260108_checkpoint-epoch-5
```

**出力例:**
```
================================================================================
PointLLM Checkpoint Validation
================================================================================
Directory: /Users/user/Documents/kimura/study/PointLLM/pointllm/outputs/...

✅ Directory exists

--------------------------------------------------------------------------------
Required Files Check
--------------------------------------------------------------------------------
✅ adapter_config.json           512 B
✅ adapter_model.bin             45.23 MB
❌ adapter_model.safetensors     Not found
✅ point_proj.pt                 312.45 MB

--------------------------------------------------------------------------------
LoRA Adapter Check
--------------------------------------------------------------------------------
✅ LoRA adapter files found

Adapter Configuration:
  PEFT Type:       LORA
  Task Type:       CAUSAL_LM
  Rank (r):        8
  Alpha:           16
  Dropout:         0.05
  Target Modules:  q_proj, k_proj, v_proj, o_proj

--------------------------------------------------------------------------------
Point Projector Check
--------------------------------------------------------------------------------
✅ Point Projector file found
   Size: 312.45 MB

================================================================================
Summary
================================================================================
✅ All required files are present!
   This checkpoint can be used for LoRA inference.
================================================================================
```

## PEFTライブラリのファイル生成について

### `save_pretrained()`の動作

```python
# trainer_LoRA.py での保存処理
self.model.save_pretrained(checkpoint_path)
```

この呼び出しにより、以下のファイルが生成されます：

1. **`adapter_config.json`** - LoRAの設定
   - `peft_type`: "LORA"
   - `r`: LoRAのrank
   - `lora_alpha`: スケーリングファクター
   - `target_modules`: LoRAを適用するモジュールのリスト
   - `lora_dropout`: ドロップアウト率
   - など

2. **`adapter_model.bin`** (デフォルト) - LoRAの重み（PyTorch形式）
   - LoRAの学習可能パラメータ（A行列、B行列）
   - サイズ: rank、target_modulesの数に依存（通常10-100MB程度）

3. **`adapter_model.safetensors`** (オプション) - LoRAの重み（SafeTensors形式）
   - `safe_serialization=True`を指定した場合に生成
   - より安全で高速なフォーマット
   - サイズは`.bin`とほぼ同じ

### 推論時の読み込み

`chat_gradio_finetuning_LoRA.py`の`load_model_and_checkpoint()`では：

```python
if checkpoint_path and os.path.isdir(checkpoint_path):
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        print(f"\nLoading LoRA adapters from: {checkpoint_path}")
        model = PeftModel.from_pretrained(
            model,
            checkpoint_path,  # ディレクトリを指定
            torch_dtype=dtype
        )
```

`PeftModel.from_pretrained()`は自動的に：
1. `adapter_config.json`を読み込んで設定を取得
2. `adapter_model.bin`または`adapter_model.safetensors`を読み込んで重みを適用

## 実際のファイル確認手順

### 1. チェックポイントディレクトリの確認

```bash
cd /Users/user/Documents/kimura/study/PointLLM

# ファイルリストを表示
ls -lh pointllm/outputs/projection_finetune_v1/checkpoints/20260108_checkpoint-epoch-5/
```

**期待される出力:**
```
-rw-r--r--  1 user  staff   512B  Jan  8 10:00 adapter_config.json
-rw-r--r--  1 user  staff    45M  Jan  8 10:00 adapter_model.bin
-rw-r--r--  1 user  staff   312M  Jan  8 10:00 point_proj.pt
-rw-r--r--  1 user  staff   1.2K  Jan  8 10:00 trainer_state.pt
```

### 2. adapter_config.jsonの内容確認

```bash
cat pointllm/outputs/projection_finetune_v1/checkpoints/20260108_checkpoint-epoch-5/adapter_config.json
```

**期待される内容:**
```json
{
  "auto_mapping": null,
  "base_model_name_or_path": "RunsenXu/PointLLM_7B_v1.2",
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "layers_pattern": null,
  "layers_to_transform": null,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "modules_to_save": null,
  "peft_type": "LORA",
  "r": 8,
  "revision": null,
  "target_modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
  ],
  "task_type": "CAUSAL_LM"
}
```

### 3. 検証スクリプトの実行

```bash
python validate_checkpoint.py pointllm/outputs/projection_finetune_v1/checkpoints/20260108_checkpoint-epoch-5
```

## トラブルシューティング

### ファイルが見つからない場合

1. **チェックポイントディレクトリの確認**
   ```bash
   find pointllm/outputs -name "adapter_config.json"
   ```

2. **最新のチェックポイントを確認**
   ```bash
   ls -ltr pointllm/outputs/projection_finetune_v1/checkpoints/
   ```

3. **ファイルの再生成が必要な場合**
   - 学習スクリプト(`trainer_LoRA.py`)を再実行
   - または、既存のチェックポイントから必要なファイルをコピー

### adapter_model.binのサイズが異常に小さい/大きい場合

**正常なサイズの目安:**
- LoRA rank r=8, target_modules=4: 約40-60MB
- LoRA rank r=16, target_modules=4: 約80-120MB
- LoRA rank r=32, target_modules=4: 約160-240MB

**異常な場合:**
- 数KB以下: 保存が正常に完了していない可能性
- 数GB以上: 間違ってベースモデル全体を保存している可能性

## まとめ

- ✅ **正しいファイル名**: `adapter_model.bin` または `adapter_model.safetensors`
- ❌ **間違ったファイル名**: `adapter.pt`
- 📝 **必須ファイル**: `adapter_config.json` + (`adapter_model.bin` OR `adapter_model.safetensors`) + `point_proj.pt`
- 🔧 **検証ツール**: `validate_checkpoint.py`を使用して確認
