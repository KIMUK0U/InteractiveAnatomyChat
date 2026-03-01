"""
chat_gradio_finetuning_encoder_proj_lora.py

2段階学習モデルの推論スクリプト:
- Stage 1: Encoder + Projectorの学習済み重み
- Stage 2: Projector更新分 + LoRAアダプタ

使用方法:
    PYTHONPATH=$PWD python pointllm/eval/chat_gradio_finetuning_encoder_proj_lora.py \
        --dataset_path "/home/yyamashita/Desktop/kkimu/test/llm_dataset20260109" \
        --test_json "annotations/test.json" \
        --model_path "RunsenXu/PointLLM_7B_v1.2" \
        --encoder_checkpoint_path "/home/yyamashita/Desktop/kkimu/test/ProjectionFinetuning/outputs/encoder_projector_finetune_v1/checkpoints/best_model" \
        --lora_checkpoint_path "/home/yyamashita/Desktop/kkimu/test/ProjectionFinetuning/outputs/encoder_proj_lora_finetune_v1/checkpoints/best_model" \
        --output_file "results_encoder_proj_lora.xlsx" \
        --device "cuda"
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------------------------------------------------
# 1. Environment & Setup
# -----------------------------------------------------------------------------

def setup_pointllm_environment(pointllm_path: Optional[str] = None):
    """PointLLM環境をセットアップします"""
    if pointllm_path is None:
        possible_paths = [
            "PointLLM", "../PointLLM", "../../PointLLM",
            os.path.expanduser("~/PointLLM"), os.path.join(os.getcwd(), "PointLLM")
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pointllm_path = path
                break
    
    if pointllm_path:
        pointllm_path = os.path.abspath(pointllm_path)
        if pointllm_path not in sys.path:
            sys.path.insert(0, pointllm_path)
    
    try:
        from pointllm.model import PointLLMLlamaForCausalLM
        from pointllm.model.pointllm import PointLLMConfig
        from transformers import AutoConfig, AutoModelForCausalLM
        
        # 既存の登録を確認して重複登録を避ける
        try:
            AutoConfig.register("pointllm", PointLLMConfig)
            AutoModelForCausalLM.register(PointLLMConfig, PointLLMLlamaForCausalLM)
        except Exception:
            pass
            
        print("✅ PointLLM environment setup complete")
        return True
    except ImportError as e:
        print(f"❌ Failed to setup PointLLM: {e}")
        print("Please ensure PointLLM is installed or specify --pointllm_path")
        return False


def get_device(device_name: str = "auto", verbose: bool = False) -> torch.device:
    """デバイスを取得します"""
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        try:
            device = torch.device(device_name)
        except Exception:
            device = torch.device("cpu")
    
    if verbose:
        print(f"Using device: {device}")
        
    return device

# -----------------------------------------------------------------------------
# 2. Model Loading (Encoder + Projector + LoRA Support)
# -----------------------------------------------------------------------------

def validate_checkpoint_files(checkpoint_path: str, stage_name: str) -> Dict[str, Path]:
    """
    チェックポイントディレクトリ内のファイルを検証
    
    Args:
        checkpoint_path: チェックポイントディレクトリのパス
        stage_name: "Stage 1" or "Stage 2"
    
    Returns:
        検証済みファイルパスの辞書
    """
    checkpoint_dir = Path(checkpoint_path)
    files = {}
    
    print(f"\n[{stage_name}] Validating checkpoint: {checkpoint_dir}")
    
    if not checkpoint_dir.exists():
        print(f"   ⚠️ Directory not found")
        return files
    
    if stage_name == "Stage 1":
        # Encoder + Projectorの検証
        encoder_proj_path = checkpoint_dir / "encoder_projector.pt"
        if encoder_proj_path.exists():
            files['encoder_projector'] = encoder_proj_path
            print(f"   ✅ Found: encoder_projector.pt")
        else:
            print(f"   ⚠️ Missing: encoder_projector.pt")
    
    elif stage_name == "Stage 2":
        # Projector更新分の検証
        point_proj_path = checkpoint_dir / "point_proj.pt"
        if point_proj_path.exists():
            files['point_proj'] = point_proj_path
            print(f"   ✅ Found: point_proj.pt")
        else:
            print(f"   ⚠️ Missing: point_proj.pt")
        
        # LoRAアダプタの検証
        adapter_config_path = checkpoint_dir / "adapter_config.json"
        if adapter_config_path.exists():
            files['adapter_config'] = adapter_config_path
            print(f"   ✅ Found: adapter_config.json")
        else:
            print(f"   ⚠️ Missing: adapter_config.json")
        
        adapter_model_bin = checkpoint_dir / "adapter_model.bin"
        adapter_model_safetensors = checkpoint_dir / "adapter_model.safetensors"
        
        if adapter_model_bin.exists():
            files['adapter_model'] = adapter_model_bin
            print(f"   ✅ Found: adapter_model.bin")
        elif adapter_model_safetensors.exists():
            files['adapter_model'] = adapter_model_safetensors
            print(f"   ✅ Found: adapter_model.safetensors")
        else:
            print(f"   ⚠️ Missing: adapter_model.bin/safetensors")
    
    return files


def load_encoder_projector_weights(
    model,
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype
) -> int:
    """
    Stage 1のEncoder + Projector重みをロード
    
    Args:
        model: ベースモデル
        checkpoint_path: チェックポイントディレクトリのパス
        device: デバイス
        dtype: データ型
    
    Returns:
        ロードしたパラメータ数
    """
    files = validate_checkpoint_files(checkpoint_path, "Stage 1")
    
    if 'encoder_projector' not in files:
        print("\n[Stage 1] Skipping: No encoder_projector.pt found")
        return 0
    
    print(f"\n[Stage 1] Loading Encoder + Projector weights...")
    
    encoder_proj_path = files['encoder_projector']
    state_dict = torch.load(encoder_proj_path, map_location="cpu")
    
    # state_dictの展開
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    # モデルの現在のstate_dict
    model_state_dict = model.state_dict()
    
    # Encoder + Projectorの重みを更新
    updated_keys = []
    for key, value in state_dict.items():
        if "point_backbone" in key or "point_proj" in key:
            if key in model_state_dict:
                model_state_dict[key] = value.to(device=device, dtype=dtype)
                updated_keys.append(key)
    
    # state_dictをモデルにロード
    model.load_state_dict(model_state_dict, strict=False)
    
    # 統計情報
    encoder_keys = [k for k in updated_keys if "point_backbone" in k]
    projector_keys = [k for k in updated_keys if "point_proj" in k]
    
    print(f"   ✅ Loaded {len(updated_keys)} parameters:")
    print(f"      - Encoder: {len(encoder_keys)} parameters")
    print(f"      - Projector: {len(projector_keys)} parameters")
    
    return len(updated_keys)


def load_projector_update(
    model,
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype
) -> int:
    """
    Stage 2のProjector更新分をロード（LoRA適用後）
    
    Args:
        model: モデル（LoRAロード済み）
        checkpoint_path: チェックポイントディレクトリのパス
        device: デバイス
        dtype: データ型
    
    Returns:
        更新したパラメータ数
    """
    files = validate_checkpoint_files(checkpoint_path, "Stage 2")
    
    if 'point_proj' not in files:
        print("\n[Stage 2] Skipping Projector update: No point_proj.pt found")
        return 0
    
    print(f"\n[Stage 2] Loading Projector updates...")
    
    point_proj_path = files['point_proj']
    checkpoint = torch.load(point_proj_path, map_location="cpu")
    
    # state_dictの展開
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    print(f"\n🔍 point_proj.pt keys:")
    for i, (key, value) in enumerate(state_dict.items()):
        print(f"   {i+1}. {key} -> {value.shape}")
        if i >= 4:
            break
    
    # モデルの現在のstate_dict
    model_state_dict = model.state_dict()
    
    print(f"\n🔍 Model keys (point_proj, first 5):")
    proj_keys = [k for k in model_state_dict.keys() if 'point_proj' in k][:5]
    for key in proj_keys:
        print(f"   - {key}")
    
    # ★ 直接マッチング（LoRAロード後のキー名で完全一致するはず）
    updated_keys = []
    updated_param_count = 0
    
    for ckpt_key, ckpt_value in state_dict.items():
        # 完全一致を試す
        if ckpt_key in model_state_dict:
            model_state_dict[ckpt_key] = ckpt_value.to(device=device, dtype=dtype)
            updated_keys.append(ckpt_key)
            updated_param_count += ckpt_value.numel()
            print(f"   ✅ Matched: {ckpt_key}")
        else:
            print(f"   ⚠️ Could not match: {ckpt_key}")
            # デバッグ：類似キーを探す
            similar = [k for k in model_state_dict.keys() if ckpt_key.split('.')[-1] in k][:3]
            if similar:
                print(f"      Similar in model: {similar}")
    
    # 更新された状態をロード
    if len(updated_keys) > 0:
        model.load_state_dict(model_state_dict, strict=False)
        print(f"\n   ✅ Updated {len(updated_keys)} Projector parameters")
        print(f"   ✅ Total updated parameters: {updated_param_count:,}")
    else:
        print(f"\n   ⚠️ WARNING: No parameters were updated!")
    
    return updated_param_count


def load_lora_adapter(model, checkpoint_path: str):
    """
    LoRAアダプタをロード
    
    Args:
        model: モデル（Stage 1, 2の重みロード済み）
        checkpoint_path: チェックポイントディレクトリのパス
    
    Returns:
        LoRA適用後のモデル
    """
    files = validate_checkpoint_files(checkpoint_path, "Stage 2")
    
    if 'adapter_config' not in files or 'adapter_model' not in files:
        print("\n[Stage 2] Skipping LoRA: Adapter files not found")
        return model
    
    print(f"\n[Stage 2] Loading LoRA adapter...")
    
    try:
        from peft import PeftModel
        
        checkpoint_dir = Path(checkpoint_path)
        model = PeftModel.from_pretrained(
            model,
            str(checkpoint_dir),
            is_trainable=False  # 推論モード
        )
        
        print(f"   ✅ LoRA adapter loaded")
        print(f"   - Model type: {type(model).__name__}")
        
        return model
        
    except ImportError:
        print("   ⚠️ peft library not installed, skipping LoRA")
        return model
    except Exception as e:
        print(f"   ⚠️ Failed to load LoRA adapter: {e}")
        return model


def load_model_with_two_stage_checkpoints(
    model_path: str,
    encoder_checkpoint_path: Optional[str] = None,
    lora_checkpoint_path: Optional[str] = None,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float16
):
    """
    2段階学習の重みをロードしてモデルを構築
    
    Args:
        model_path: ベースモデルのパス
        encoder_checkpoint_path: Stage 1チェックポイント（Encoder + Projector）
        lora_checkpoint_path: Stage 2チェックポイント（Projector更新 + LoRA）
        device: 使用するデバイス
        dtype: モデルのデータ型
    
    Returns:
        (model, tokenizer) のタプル
    """
    print(f"\n{'='*80}")
    print("Loading Model (2-Stage: Encoder + Projector + LoRA)")
    print(f"{'='*80}")
    print(f"Base model: {model_path}")
    print(f"Stage 1 checkpoint: {encoder_checkpoint_path or 'None'}")
    print(f"Stage 2 checkpoint: {lora_checkpoint_path or 'None'}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    # Step 1: トークナイザーのロード
    print("\n[Base] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=False, 
        padding_side="left"  # ✅ decoder-onlyモデルではleft-paddingを使用
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("   ✅ Tokenizer loaded (padding_side='left')")
    
    # Step 2: ベースモデルのロード
    print("\n[Base] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device=device, dtype=dtype)
    print("   ✅ Base model loaded")
    
    # Step 3: Point Backbone Configの初期化
    if hasattr(model, "initialize_tokenizer_point_backbone_config_wo_embedding"):
        model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
        print("   ✅ Point backbone config initialized")
    elif hasattr(model, "get_model"):
        inner_model = model.get_model()
        if hasattr(inner_model, "initialize_tokenizer_point_backbone_config_wo_embedding"):
            inner_model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
            print("   ✅ Point backbone config initialized (via get_model)")
    
    # Step 4: Stage 1の重み（Encoder + Projector）をロード
    if encoder_checkpoint_path:
        encoder_params = load_encoder_projector_weights(
            model, encoder_checkpoint_path, device, dtype
        )
    else:
        print("\n[Stage 1] Skipped: No checkpoint provided")
        encoder_params = 0
    
    # ★★★ 順序変更：先にLoRAをロード ★★★
    # Step 5: LoRAアダプタをロード
    if lora_checkpoint_path:
        model = load_lora_adapter(model, lora_checkpoint_path)
    
    # Step 6: Stage 2の重み（Projector更新）をロード
    # ※LoRAロード後にキー名が変わるため、最後にロード
    if lora_checkpoint_path:
        proj_params = load_projector_update(
            model, lora_checkpoint_path, device, dtype
        )
    else:
        print("\n[Stage 2] Skipped: No checkpoint provided")
        proj_params = 0
    
    # 評価モードに設定
    model.eval()
    
    print(f"\n{'='*80}")
    print("Model Loading Summary")
    print(f"{'='*80}")
    print(f"Stage 1 parameters loaded: {encoder_params}")
    print(f"Stage 2 parameters updated: {proj_params}")
    print(f"LoRA enabled: {'Yes' if hasattr(model, 'peft_config') else 'No'}")
    print(f"Model type: {type(model).__name__}")
    print(f"{'='*80}\n")
    
    return model, tokenizer

# -----------------------------------------------------------------------------
# 3. Data Processing
# -----------------------------------------------------------------------------

def load_and_process_point_cloud(
    point_cloud_path: str, 
    num_points: int = 8192
) -> torch.Tensor:
    """
    点群を読み込んで前処理します
    
    Args:
        point_cloud_path: 点群ファイルのパス
        num_points: サンプリングする点数
    
    Returns:
        処理済み点群テンソル (1, num_points, 6)
    """
    try:
        point_cloud = np.load(point_cloud_path)
    except Exception as e:
        raise ValueError(f"Failed to load point cloud from {point_cloud_path}: {e}")
    
    # パディング (6次元未満の場合)
    if point_cloud.shape[1] < 6:
        padding = np.ones((point_cloud.shape[0], 6 - point_cloud.shape[1])) * 0.5
        point_cloud = np.concatenate([point_cloud, padding], axis=1)
    
    point_cloud = point_cloud[:, :6]
    
    # サンプリング
    n_points = point_cloud.shape[0]
    if n_points != num_points:
        indices = np.random.choice(n_points, num_points, replace=(n_points < num_points))
        point_cloud = point_cloud[indices]
    
    # 座標の正規化
    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:6]
    
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    if max_dist > 0:
        xyz = xyz / max_dist
    
    # RGBの正規化
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    
    # 結合
    point_cloud = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
    
    return torch.from_numpy(point_cloud).unsqueeze(0)


def load_test_dataset(dataset_path: str, test_json: str) -> Dict[str, Any]:
    """
    テストデータセットJSONを読み込みます
    
    Args:
        dataset_path: データセットのルートパス
        test_json: テストJSONファイルのパス
    
    Returns:
        データセット辞書
    """
    test_json_path = test_json
    if not os.path.exists(test_json_path):
        test_json_path = os.path.join(dataset_path, test_json)
    
    if not os.path.exists(test_json_path):
        # annotationsフォルダ内も探す
        test_json_path = os.path.join(dataset_path, "annotations", test_json)

    if not os.path.exists(test_json_path):
        raise FileNotFoundError(f"Test JSON not found: {test_json}")

    print(f"Loading dataset from: {test_json_path}")
    with open(test_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if isinstance(data, dict) and 'data' in data:
        return data
    elif isinstance(data, list):
        return {'data': data}
    else:
        return {'data': []}

# -----------------------------------------------------------------------------
# 4. Inference Logic
# -----------------------------------------------------------------------------

def generate_response_batch(
    model,
    tokenizer,
    point_clouds: torch.Tensor,  # (batch_size, num_points, 6)
    questions: List[str],
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> List[str]:
    """
    バッチ処理で複数の点群と質問からレスポンスを生成します
    
    Args:
        model: PointLLMモデル
        tokenizer: トークナイザー
        point_clouds: 点群テンソル (batch_size, num_points, 6)
        questions: 質問文のリスト
        device: デバイス
        dtype: データ型
        max_new_tokens: 生成する最大トークン数
        temperature: サンプリング温度
        top_p: nucleus sampling閾値
    
    Returns:
        生成されたレスポンスのリスト
    """
    batch_size = len(questions)
    
    # Point backbone configの取得
    if hasattr(model, "get_model"):
        point_config = model.get_model().point_backbone_config
    elif hasattr(model, "base_model"):  # PeftModelの場合
        point_config = model.base_model.get_model().point_backbone_config
    elif hasattr(model, "point_backbone_config"):
        point_config = model.point_backbone_config
    else:
        raise AttributeError("point_backbone_config not found in model")
    
    point_token_len = point_config['point_token_len']
    default_point_patch_token = point_config['default_point_patch_token']
    mm_use_point_start_end = point_config.get('mm_use_point_start_end', False)
    
    # プロンプトの構築（バッチ）
    prompts = []
    for question in questions:
        if mm_use_point_start_end:
            start = point_config['default_point_start_token']
            end = point_config['default_point_end_token']
            prompt = f"{start}{default_point_patch_token * point_token_len}{end}\n{question}"
        else:
            prompt = f"{default_point_patch_token * point_token_len}\n{question}"
        prompts.append(prompt)
    
    # Conversation Templateの適用（バッチ）
    try:
        from pointllm.conversation import conv_templates, SeparatorStyle
        full_prompts = []
        for prompt in prompts:
            conv = conv_templates["vicuna_v1_1"].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            full_prompts.append(conv.get_prompt())
        sep = conv.sep
        sep2 = conv.sep2 if conv.sep_style == SeparatorStyle.TWO else None
    except ImportError:
        full_prompts = [f"USER: {p}\nASSISTANT: " for p in prompts]
        sep = None
        sep2 = None
    
    # トークン化（バッチ、パディング）
    # ✅ max_lengthを4096に増やして余裕を持たせる
    inputs = tokenizer(
        full_prompts, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096  # 2048 → 4096
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    # ✅ 実際の入力長に基づいてmax_new_tokensを動的に調整
    max_input_len = (attention_mask.sum(dim=1).max().item())  # バッチ内の最大入力長
    
    # モデルの最大コンテキスト長（通常2048 or 4096）
    model_max_length = getattr(tokenizer, 'model_max_length', 2048)
    if model_max_length > 100000:  # デフォルト値の場合
        model_max_length = 2048
    
    # 生成可能な最大トークン数を計算
    available_tokens = model_max_length - max_input_len
    adjusted_max_new_tokens = min(max_new_tokens, max(available_tokens, 128))  # 最低128は確保
    
    if adjusted_max_new_tokens < max_new_tokens:
        print(f"\n⚠️ Adjusted max_new_tokens: {max_new_tokens} → {adjusted_max_new_tokens}")
        print(f"   (input_len={max_input_len}, model_max={model_max_length})")
    
    # ✅ デバッグ: パディングの確認
    if batch_size > 1:
        for i in range(min(2, batch_size)):  # 最初の2サンプルのみ表示
            actual_len = (attention_mask[i] == 1).sum().item()
            pad_len = attention_mask[i].shape[0] - actual_len
            if i == 0:  # 最初のサンプルのみ詳細表示
                print(f"[DEBUG] Batch input_len={max_input_len}, max_new={adjusted_max_new_tokens}, total_max={max_input_len + adjusted_max_new_tokens}")
    
    # 点群を適切な型とデバイスに変換
    point_clouds = point_clouds.to(device=device, dtype=dtype)
    
    # 生成
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            point_clouds=point_clouds,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=adjusted_max_new_tokens,  # ✅ 調整後の値を使用
            use_cache=True
        )
    
    # ✅ デバッグ: 出力の形状確認
    if batch_size == 1:
        print(f"[DEBUG] Generated output shape: {output_ids.shape}, input shape: {input_ids.shape}")
    
    # デコード（バッチ）
    responses = []
    for i in range(batch_size):
        try:
            # ✅ 修正: input_idsの実際の長さを使用（パディング含む）
            # output_idsはinput_idsと同じ長さから始まるため、input_idsの長さを使う
            input_len = input_ids.shape[1]  # バッチ全体で同じ長さ（パディング後）
            
            # 生成された出力の実際の長さを確認
            output_len = output_ids[i].shape[0]
            
            # デバッグ出力（最初のサンプルのみ）
            if i == 0 and batch_size == 1:
                print(f"[DEBUG] Decoding: input_len={input_len}, output_len={output_len}")
            
            # ✅ 安全なデコード
            if input_len <= output_len:
                # 通常のケース: 新しいトークンが生成された
                response = tokenizer.decode(
                    output_ids[i, input_len:], 
                    skip_special_tokens=True
                )
            else:
                # エッジケース: 出力が短い（通常発生しない）
                print(f"⚠️ Warning: output shorter than input for sample {i}")
                response = ""
            
            # 後処理
            if sep and response.endswith(sep):
                response = response[:-len(sep)]
            if sep2 and response.endswith(sep2):
                response = response[:-len(sep2)]
            
            responses.append(response.strip())
            
        except Exception as decode_error:
            print(f"\n⚠️ Decode error for sample {i}: {decode_error}")
            print(f"   input shape: {input_ids.shape}, output shape: {output_ids.shape}")
            # フォールバック: 空文字列を返す
            responses.append(f"ERROR: {str(decode_error)}")
    
    return responses

# -----------------------------------------------------------------------------
# 5. Main Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PointLLM 2-Stage (Encoder + Projector + LoRA) Batch Inference"
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=True, 
        help="Path to dataset root"
    )
    parser.add_argument(
        "--test_json", 
        type=str, 
        default="annotations/test.json", 
        help="Path to test json (relative to dataset_path or absolute)"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="RunsenXu/PointLLM_7B_v1.2", 
        help="Base model path"
    )
    parser.add_argument(
        "--encoder_checkpoint_path", 
        type=str, 
        default=None, 
        help="Stage 1: Encoder + Projector checkpoint directory path"
    )
    parser.add_argument(
        "--lora_checkpoint_path", 
        type=str, 
        default=None, 
        help="Stage 2: Projector update + LoRA checkpoint directory path"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="results_encoder_proj_lora.xlsx", 
        help="Output Excel file path"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto", 
        help="Device: cuda, mps, cpu, or auto"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=512,  # ✅ 1024 → 512 に変更（より安全）
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--num_points", 
        type=int, 
        default=8192,
        help="Number of points to sample from point cloud"
    )
    parser.add_argument(
        "--pointllm_path", 
        type=str, 
        default=None,
        help="Path to PointLLM repository"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    
    args = parser.parse_args()
    
    # PointLLM環境のセットアップ
    if not setup_pointllm_environment(args.pointllm_path):
        print("Failed to setup PointLLM environment. Exiting.")
        return
    
    # デバイスの取得
    device = get_device(args.device, verbose=True)
    
    # データ型の選択
    if device.type == "cuda":
        dtype = torch.float16  # CUDA では float16
    else:
        dtype = torch.float32  # CPU/MPS では float32
    
    print(f"Using dtype: {dtype}")
    
    # モデルのロード（2段階）
    model, tokenizer = load_model_with_two_stage_checkpoints(
        model_path=args.model_path,
        encoder_checkpoint_path=args.encoder_checkpoint_path,
        lora_checkpoint_path=args.lora_checkpoint_path,
        device=device,
        dtype=dtype
    )
    
    # ✅ モデルのコンテキスト長を確認
    model_max_length = getattr(tokenizer, 'model_max_length', 2048)
    if model_max_length > 100000:  # デフォルト値の場合
        model_max_length = 2048
        print(f"\n⚠️ Using default model_max_length: {model_max_length}")
    else:
        print(f"\n✅ Model context length: {model_max_length}")
    
    # 点群トークン数を確認
    try:
        if hasattr(model, "get_model"):
            point_config = model.get_model().point_backbone_config
        elif hasattr(model, "base_model"):
            point_config = model.base_model.get_model().point_backbone_config
        else:
            point_config = model.point_backbone_config
        
        point_token_len = point_config.get('point_token_len', 513)
        print(f"✅ Point token length: {point_token_len}")
        
        # 推定される入力長
        estimated_input_len = point_token_len + 100  # プロンプト分
        estimated_output_space = model_max_length - estimated_input_len
        
        if estimated_output_space < args.max_new_tokens:
            print(f"\n⚠️ WARNING: max_new_tokens ({args.max_new_tokens}) may be too large!")
            print(f"   Estimated available space: ~{estimated_output_space} tokens")
            print(f"   Consider using: --max_new_tokens {min(512, estimated_output_space)}")
    except Exception as e:
        print(f"⚠️ Could not determine point token length: {e}")
    
    print()  # 空行
    
    # データセットのロード
    dataset = load_test_dataset(args.dataset_path, args.test_json)
    samples = dataset['data']
    print(f"\nFound {len(samples)} test samples.\n")
    
    # バッチ推論
    results = []
    batch_size = args.batch_size
    
    # バッチに分割して処理
    for batch_start in tqdm(range(0, len(samples), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(samples))
        batch_samples = samples[batch_start:batch_end]
        current_batch_size = len(batch_samples)
        
        # バッチデータの準備
        batch_point_clouds = []
        batch_questions = []
        batch_metadata = []  # ID, path, ground_truth を保持
        
        for i, sample in enumerate(batch_samples):
            sample_id = sample.get('id', f'sample_{batch_start + i}')
            pc_rel_path = sample['point_cloud']
            pc_path = os.path.join(args.dataset_path, pc_rel_path)
            
            # 会話データから質問と正解を抽出
            conversations = sample.get('conversations', [])
            question = ""
            ground_truth = ""
            
            for turn in conversations:
                if turn['role'] == 'user':
                    content = turn['content']
                    lines = content.split('\n')
                    if len(lines) > 1 and 'point' in lines[0].lower():
                        question = '\n'.join(lines[1:])
                    else:
                        question = content
                elif turn['role'] == 'assistant':
                    ground_truth = turn['content']
            
            if not question:
                question = "Describe this 3D object."
            
            # 点群のロードと前処理
            try:
                pc_tensor = load_and_process_point_cloud(pc_path, args.num_points)
                batch_point_clouds.append(pc_tensor)
                batch_questions.append(question)
                batch_metadata.append({
                    'id': sample_id,
                    'point_cloud': pc_rel_path,
                    'question': question,
                    'ground_truth': ground_truth
                })
            except Exception as e:
                print(f"\n⚠️ Error loading {sample_id}: {e}")
                # エラーの場合は結果に記録してスキップ
                results.append({
                    "id": sample_id,
                    "point_cloud": pc_rel_path,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": f"ERROR (loading): {str(e)}"
                })
        
        # バッチが空の場合はスキップ
        if not batch_point_clouds:
            continue
        
        # バッチテンソルに変換
        batch_point_clouds_tensor = torch.cat(batch_point_clouds, dim=0)  # (batch_size, num_points, 6)
        
        # バッチ推論
        try:
            batch_responses = generate_response_batch(
                model=model,
                tokenizer=tokenizer,
                point_clouds=batch_point_clouds_tensor,
                questions=batch_questions,
                device=device,
                dtype=dtype,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            # 結果を記録
            for metadata, response in zip(batch_metadata, batch_responses):
                results.append({
                    "id": metadata['id'],
                    "point_cloud": metadata['point_cloud'],
                    "question": metadata['question'],
                    "ground_truth": metadata['ground_truth'],
                    "prediction": response
                })
                
        except Exception as e:
            print(f"\n⚠️ Error processing batch {batch_start}-{batch_end}: {e}")
            print("   Falling back to individual processing...")
            
            # ✅ フォールバック: 個別に処理
            for i, (pc_tensor, question, metadata) in enumerate(zip(batch_point_clouds, batch_questions, batch_metadata)):
                try:
                    # 1サンプルずつ処理
                    individual_response = generate_response_batch(
                        model=model,
                        tokenizer=tokenizer,
                        point_clouds=pc_tensor,  # (1, num_points, 6)
                        questions=[question],
                        device=device,
                        dtype=dtype,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p
                    )[0]
                    
                    results.append({
                        "id": metadata['id'],
                        "point_cloud": metadata['point_cloud'],
                        "question": metadata['question'],
                        "ground_truth": metadata['ground_truth'],
                        "prediction": individual_response
                    })
                    
                except Exception as e2:
                    print(f"   ⚠️ Error on sample {metadata['id']}: {e2}")
                    results.append({
                        "id": metadata['id'],
                        "point_cloud": metadata['point_cloud'],
                        "question": metadata['question'],
                        "ground_truth": metadata['ground_truth'],
                        "prediction": f"ERROR (individual): {str(e2)}"
                    })
        
        # 定期的に保存（バッチごと）
        if (batch_end) % (batch_size * 5) < batch_size:  # 5バッチごと
            df = pd.DataFrame(results)
            df.to_excel(args.output_file, index=False)
            print(f"\nIntermediate results saved ({len(results)} samples)")
    
    # 最終結果の保存
    df = pd.DataFrame(results)
    df.to_excel(args.output_file, index=False)
    
    print(f"\n{'='*80}")
    print("Inference Complete!")
    print(f"{'='*80}")
    print(f"Total samples processed: {len(results)}")
    print(f"Results saved to: {args.output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()