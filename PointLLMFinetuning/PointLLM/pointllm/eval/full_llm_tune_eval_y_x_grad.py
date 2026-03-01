"""
chat_gradio_full_llm_finetune.py

2段階学習(Phase 1: Encoder+Projector, Phase 2: Projector+LLM)で学習した重みを読み込み、
データセットに対してバッチ推論を実行してExcelファイルに出力します。

主な修正点:
- 2段階学習に対応（Phase 1とPhase 2のチェックポイントを別々にロード)
- LLM全体の重みロードに対応
- point_llm_weight.pt形式の読み込みに対応
- データセット内の誤った特殊トークン形式(<point_start><point_patch>*513<point_end>)に対応
- 型の統一(float16)
- 勾配ベースの色別重要度分析機能を追加

使用方法:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python pointllm/eval/full_llm_tune_eval_y_x_grad.py \
        --dataset_path "/home/yyamashita/Desktop/kkimu/test/demo_410_ar_instraction_dataset" \
        --test_json "/home/yyamashita/Desktop/kkimu/test/demo_410_ar_instraction_dataset/annotations/test.json" \
        --model_path "RunsenXu/PointLLM_7B_v1.2" \
        --phase1_checkpoint "/home/yyamashita/Desktop/kkimu/test/ProjectionFinetuning/outputs/encoder_proj_tune_ddp_demo_ar_410_v1/checkpoints/checkpoint-epoch-20" \
        --phase2_checkpoint "/home/yyamashita/Desktop/kkimu/test/ProjectionFinetuning/outputs/demo_410_ar_intration_encoder_proj_llm_ddp_finetune_LowLR5e-5_v4_EPLR_high_30epoch/checkpoints/best_model" \
        --output_file "delete_410_results_full_llm.xlsx" \
        --enable_gradient_analysis \
        --gradient_output_csv "410_gradient_analysis_by_color.csv" \
        --device "cuda"
"""

import argparse
import json
import os
import sys
import re
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Union, Tuple, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from collections import defaultdict

# -----------------------------------------------------------------------------
# 1. Environment & Setup
# -----------------------------------------------------------------------------

def setup_pointllm_environment(pointllm_path: Optional[str] = None):
    """PointLLM環境をセットアップします"""
    if pointllm_path is None:
        possible_paths = [
            "PointLLM", "../PointLLM", "../../PointLLM",
            os.path.expanduser("~/PointLLM"), os.path.join(os.getcwd(), "pointllm")
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
# 2. Helper Functions
# -----------------------------------------------------------------------------

def _find_matching_key(ckpt_key: str, model_state_dict: dict) -> Optional[str]:
    """
    チェックポイントのキーとモデルの状態辞書のキーをマッチングします
    
    Args:
        ckpt_key: チェックポイントのキー
        model_state_dict: モデルの状態辞書
        
    Returns:
        マッチしたキー、見つからない場合はNone
    """
    # 1. 完全一致
    if ckpt_key in model_state_dict:
        return ckpt_key
    
    # 2. プレフィックス削除/追加
    clean_key = ckpt_key[6:] if ckpt_key.startswith("model.") else ckpt_key
    
    candidates = [
        clean_key,
        f"model.{clean_key}",
        f"base_model.model.{clean_key}",
    ]
    
    for candidate in candidates:
        if candidate in model_state_dict:
            return candidate
    
    # 3. 部分一致(後方一致)
    for model_key in model_state_dict.keys():
        if model_key.endswith(clean_key):
            return model_key
        if clean_key in model_key:
            return model_key
    
    return None

# -----------------------------------------------------------------------------
# 3. Model Loading (Full LLM Fine-tuning Support)
# -----------------------------------------------------------------------------

def load_full_finetuned_model(
    model_path: str,
    phase1_checkpoint_path: Optional[str] = None,
    phase2_checkpoint_path: Optional[str] = None,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float16
):
    """
    2段階学習で学習したモデルの重みをロードします
    
    Args:
        model_path: ベースモデルのパス
        phase1_checkpoint_path: Phase 1 (Encoder + Projector学習) のチェックポイント
        phase2_checkpoint_path: Phase 2 (Projector + LLM学習) のチェックポイント
        device: デバイス
        dtype: データ型
    """
    print(f"\n{'='*60}")
    print("Loading Two-Phase Fine-tuned Model")
    print(f"{'='*60}")
    print(f"Base model: {model_path}")
    print(f"Phase 1 checkpoint: {phase1_checkpoint_path}")
    print(f"Phase 2 checkpoint: {phase2_checkpoint_path}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    # Tokenizerの準備
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # ベースモデルのロード
    print("\n[Step 1] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    
    model = model.to(device=device, dtype=dtype)
    
    # Point Backbone Configの初期化
    base_model_ref = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base_model_ref = model.base_model.model
    
    if hasattr(base_model_ref, "initialize_tokenizer_point_backbone_config_wo_embedding"):
        base_model_ref.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
        print("✅ Point backbone config initialized")
    
    # ============================================================
    # Phase 1: Encoder + Projector の学習済み重みをロード
    # ============================================================
    if phase1_checkpoint_path and os.path.isdir(phase1_checkpoint_path):
        print(f"\n[Phase 1] Loading Encoder + Projector weights...")
        print(f"   Path: {phase1_checkpoint_path}")
        
        encoder_proj_path = os.path.join(phase1_checkpoint_path, "encoder_projector.pt")
        if os.path.exists(encoder_proj_path):
            print(f"   Loading: encoder_projector.pt")
            load_encoder_projector_weights(model, encoder_proj_path, device, dtype)
        else:
            print(f"   ⚠️ Warning: encoder_projector.pt not found")
    else:
        print(f"\n⚠️ Phase 1 checkpoint not provided - skipping Encoder + Projector loading")
    
    # ============================================================
    # Phase 2: Projector + LLM の学習済み重みをロード
    # ============================================================
    if phase2_checkpoint_path and os.path.isdir(phase2_checkpoint_path):
        print(f"\n[Phase 2] Loading updated Projector + LLM weights...")
        print(f"   Path: {phase2_checkpoint_path}")
        
        # 2-1. Updated Projector weights (point_proj.pt)
        point_proj_path = os.path.join(phase2_checkpoint_path, "point_proj.pt")
        if os.path.exists(point_proj_path):
            print(f"   Loading: point_proj.pt (updated from Phase 2)")
            load_projector_weights(model, point_proj_path, device, dtype)
        else:
            print(f"   ⚠️ Warning: point_proj.pt not found")
        
        # 2-2. LLM weights - 複数のファイル形式に対応
        loaded_llm = False
        
        # 利用可能なファイルパターンのリスト（優先順位順）
        llm_weight_patterns = [
            ("proj_llm_weights.pt", "projection LLM weights"),
            ("point_llm_weight.pt", "legacy format"),
            ("llm_weights.pt", "LLM weights"),
            ("llm_model.pt", "LLM model"),
            ("model.pt", "model weights"),
            ("pytorch_model.bin", "PyTorch model binary"),
        ]
        
        # まず、ディレクトリ内のファイルを確認
        if os.path.isdir(phase2_checkpoint_path):
            available_files = os.listdir(phase2_checkpoint_path)
            print(f"   📁 Available files in checkpoint: {available_files}")
        
        # 各パターンを順番に試す
        for filename, description in llm_weight_patterns:
            llm_weight_path = os.path.join(phase2_checkpoint_path, filename)
            if os.path.exists(llm_weight_path):
                print(f"   Loading: {filename} ({description})")
                load_llm_weights(model, llm_weight_path, device, dtype)
                loaded_llm = True
                break
        
        # safetensors形式もチェック
        if not loaded_llm:
            safetensors_index = os.path.join(phase2_checkpoint_path, "model.safetensors.index.json")
            if os.path.exists(safetensors_index):
                print(f"   Loading: LLM weights from safetensors")
                try:
                    load_safetensors_weights(model, phase2_checkpoint_path, device, dtype)
                    loaded_llm = True
                except Exception as e:
                    print(f"   ⚠️ Error loading safetensors: {e}")
        
        if not loaded_llm:
            print(f"   ⚠️ Warning: No LLM weights found in Phase 2 checkpoint")
            print(f"   💡 Note: Only point_proj.pt will be loaded")
            print(f"   💡 Searched for: {[p[0] for p in llm_weight_patterns]}")
    else:
        print(f"\n⚠️ Phase 2 checkpoint not provided - skipping Projector + LLM loading")
    
    model.eval()
    print("\n✅ Model loaded successfully")
    return model, tokenizer


def load_encoder_projector_weights(model, checkpoint_path, device, dtype):
    """Encoder + Projectorの重みをロード"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    else:
        state_dict = checkpoint
    
    model_state_dict = model.state_dict()
    loaded_keys = []
    
    for ckpt_key, ckpt_value in state_dict.items():
        matched_key = _find_matching_key(ckpt_key, model_state_dict)
        
        if matched_key:
            # 形状チェック
            if model_state_dict[matched_key].shape != ckpt_value.shape:
                print(f"   ⚠️ Shape mismatch for {matched_key}: expected {model_state_dict[matched_key].shape}, got {ckpt_value.shape}")
                continue
            
            model_state_dict[matched_key] = ckpt_value.to(device=device, dtype=dtype)
            loaded_keys.append(matched_key)
    
    if len(loaded_keys) > 0:
        model.load_state_dict(model_state_dict, strict=False)
    
    encoder_params = sum(1 for k in loaded_keys if 'point_backbone' in k or 'encoder' in k)
    projector_params = sum(1 for k in loaded_keys if 'point_proj' in k or 'projector' in k)
    
    print(f"   ✅ Loaded {len(loaded_keys)} parameters")
    print(f"      - Encoder: {encoder_params}")
    print(f"      - Projector: {projector_params}")


def load_llm_weights(model, checkpoint_path, device, dtype):
    """
    LLM部分の重みをロード（point_llm_weight.pt、safetensors、pytorch形式対応）
    
    Args:
        model: ロード先のモデル
        checkpoint_path: チェックポイントファイルのパス
        device: デバイス
        dtype: データ型
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    else:
        state_dict = checkpoint
    
    model_state_dict = model.state_dict()
    loaded_keys = []
    total_params = 0
    
    for ckpt_key, ckpt_value in state_dict.items():
        matched_key = _find_matching_key(ckpt_key, model_state_dict)
        
        if matched_key:
            # 形状チェック
            if model_state_dict[matched_key].shape != ckpt_value.shape:
                print(f"   ⚠️ Shape mismatch for {matched_key}: expected {model_state_dict[matched_key].shape}, got {ckpt_value.shape}")
                continue
            
            model_state_dict[matched_key] = ckpt_value.to(device=device, dtype=dtype)
            loaded_keys.append(matched_key)
            total_params += ckpt_value.numel()
    
    if len(loaded_keys) > 0:
        model.load_state_dict(model_state_dict, strict=False)
        print(f"   ✅ Loaded {len(loaded_keys)} LLM parameters")
        
        # 統計情報
        llm_params = sum(1 for k in loaded_keys if 'llama_model' in k or 'lm_head' in k or 'embed' in k or 'norm' in k)
        print(f"      - LLM components: {llm_params}")
        print(f"      - Total parameters: {total_params:,}")
    else:
        print(f"   ⚠️ WARNING: No parameters were loaded from {checkpoint_path}!")


def load_safetensors_weights(model, checkpoint_dir, device, dtype):
    """
    safetensors形式のモデル重みをロード
    
    Args:
        model: ロード先のモデル
        checkpoint_dir: チェックポイントディレクトリ
        device: デバイス
        dtype: データ型
    """
    from safetensors.torch import load_file
    import json
    from pathlib import Path
    
    checkpoint_dir = Path(checkpoint_dir)
    index_file = checkpoint_dir / "model.safetensors.index.json"
    
    if not index_file.exists():
        print(f"   ⚠️ Warning: model.safetensors.index.json not found")
        return
    
    # インデックスファイルを読み込み
    with open(index_file, 'r') as f:
        index_data = json.load(f)
    
    weight_map = index_data.get("weight_map", {})
    
    # safetensorsファイルのリストを取得
    safetensor_files = set(weight_map.values())
    
    print(f"   📦 Found {len(safetensor_files)} safetensors files")
    
    # 各safetensorsファイルから重みをロード
    all_state_dict = {}
    for safetensor_file in safetensor_files:
        safetensor_path = checkpoint_dir / safetensor_file
        if safetensor_path.exists():
            state_dict = load_file(str(safetensor_path))
            all_state_dict.update(state_dict)
            print(f"      - Loaded: {safetensor_file}")
    
    # モデルに重みをロード
    model_state_dict = model.state_dict()
    loaded_keys = []
    missing_keys = []
    total_params = 0
    
    for ckpt_key in all_state_dict.keys():
        matched_key = _find_matching_key(ckpt_key, model_state_dict)
        
        if matched_key:
            # 形状チェック
            if model_state_dict[matched_key].shape != all_state_dict[ckpt_key].shape:
                print(f"   ⚠️ Shape mismatch for {matched_key}")
                continue
            
            model_state_dict[matched_key] = all_state_dict[ckpt_key].to(device=device, dtype=dtype)
            loaded_keys.append(matched_key)
            total_params += all_state_dict[ckpt_key].numel()
        else:
            missing_keys.append(ckpt_key)
    
    # モデルに適用
    model.load_state_dict(model_state_dict, strict=False)
    
    # 統計情報
    llm_params = sum(1 for k in loaded_keys if 'llama_model' in k or 'lm_head' in k or 'embed' in k or 'norm' in k)
    encoder_params = sum(1 for k in loaded_keys if 'point_backbone' in k)
    projector_params = sum(1 for k in loaded_keys if 'point_proj' in k)
    
    print(f"   ✅ Loaded {len(loaded_keys)} parameters from safetensors")
    print(f"      - LLM parameters: {llm_params}")
    print(f"      - Encoder parameters: {encoder_params}")
    print(f"      - Projector parameters: {projector_params}")
    print(f"      - Total parameters: {total_params:,}")
    print(f"      - Missing keys: {len(missing_keys)}")
    
    if len(missing_keys) > 0 and len(missing_keys) < 20:
        print(f"      - Sample missing keys: {missing_keys[:5]}")


def load_full_model_weights(model, checkpoint_path, device, dtype):
    """モデル全体の重みをロード (pytorch_model.bin等)"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    else:
        state_dict = checkpoint
    
    # 直接ロードを試みる
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"   ✅ Loaded model weights")
        print(f"      - Missing keys: {len(missing_keys)}")
        print(f"      - Unexpected keys: {len(unexpected_keys)}")
    except Exception as e:
        print(f"   ⚠️ Error loading full model weights: {e}")


def load_projector_weights(model, checkpoint_path, device, dtype):
    """Projectorの更新された重みをロード"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    else:
        state_dict = checkpoint
    
    model_state_dict = model.state_dict()
    updated_keys = []
    
    for ckpt_key, ckpt_value in state_dict.items():
        matched_key = _find_matching_key(ckpt_key, model_state_dict)
        
        if matched_key:
            # 形状チェック
            if model_state_dict[matched_key].shape != ckpt_value.shape:
                print(f"   ⚠️ Shape mismatch for {matched_key}")
                continue
            
            model_state_dict[matched_key] = ckpt_value.to(device=device, dtype=dtype)
            updated_keys.append(matched_key)
    
    if len(updated_keys) > 0:
        model.load_state_dict(model_state_dict, strict=False)
    
    print(f"   ✅ Updated {len(updated_keys)} Projector parameters")

# -----------------------------------------------------------------------------
# 4. Data Processing
# -----------------------------------------------------------------------------

def load_and_process_point_cloud(point_cloud_path: str, num_points: int = 8192) -> torch.Tensor:
    """点群を読み込んで前処理します"""
    try:
        point_cloud = np.load(point_cloud_path)
    except Exception as e:
        raise ValueError(f"Failed to load numpy array from {point_cloud_path}: {e}")
    
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
    
    # 正規化
    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:6]
    
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    if max_dist > 0:
        xyz = xyz / max_dist
        
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    
    point_cloud = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
    return torch.from_numpy(point_cloud).unsqueeze(0)


def clean_question_text(content: str, point_config: dict) -> str:
    """
    データセット内の誤った特殊トークン形式を修正します
    
    例: <point_start><point_patch>*513<point_end>質問文
    → 適切な形式に変換
    """
    # point_configから必要な情報を取得
    point_token_len = point_config['point_token_len']
    default_point_patch_token = point_config['default_point_patch_token']
    mm_use_point_start_end = point_config.get('mm_use_point_start_end', False)
    
    # 誤った形式のパターンを検出
    # Pattern 1: <point_start><point_patch>*N<point_end>
    pattern1 = r'<point_start><point_patch>\*(\d+)<point_end>'
    match1 = re.search(pattern1, content)
    
    if match1:
        # 誤った部分を削除
        content = re.sub(pattern1, '', content)
    
    # Pattern 2: <point><point_patch>*N
    pattern2 = r'<point><point_patch>\*(\d+)'
    match2 = re.search(pattern2, content)
    
    if match2:
        content = re.sub(pattern2, '', content)
    
    # Pattern 3: <point_patch>*N (スタンドアロン)
    pattern3 = r'<point_patch>\*(\d+)'
    match3 = re.search(pattern3, content)
    
    if match3:
        content = re.sub(pattern3, '', content)
    
    # 残った特殊トークンを削除
    content = content.replace('<point_start>', '')
    content = content.replace('<point_end>', '')
    content = content.replace('<point_patch>', '')
    content = content.replace('<point>', '')
    
    # 先頭の改行や空白を削除
    content = content.strip()
    
    return content


def load_test_dataset(dataset_path: str, test_json: str) -> Dict[str, Any]:
    """テストデータセットJSONを読み込みます"""
    test_json_path = test_json
    if not os.path.exists(test_json_path):
        test_json_path = os.path.join(dataset_path, test_json)
    
    if not os.path.exists(test_json_path):
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
# 5. Inference Logic
# -----------------------------------------------------------------------------

def generate_response(
    model, tokenizer, point_cloud, question, device, dtype,
    max_new_tokens=512, temperature=0.7, top_p=0.9
):
    """推論を実行して応答を生成します"""
    
    # モデル設定の取得
    if hasattr(model, "get_model"):
        point_config = model.get_model().point_backbone_config
    elif hasattr(model, "base_model") and hasattr(model.base_model, "get_model"):
        point_config = model.base_model.get_model().point_backbone_config
    else:
        point_config = getattr(model.config, "point_backbone_config", None)
        if point_config is None:
            point_config = model.base_model.model.point_backbone_config

    point_token_len = point_config['point_token_len']
    default_point_patch_token = point_config['default_point_patch_token']
    mm_use_point_start_end = point_config.get('mm_use_point_start_end', False)
    
    # 質問文のクリーニング
    question = clean_question_text(question, point_config)
    
    # プロンプトの構築
    if mm_use_point_start_end:
        start = point_config['default_point_start_token']
        end = point_config['default_point_end_token']
        prompt = f"{start}{default_point_patch_token * point_token_len}{end}\n{question}"
    else:
        prompt = f"{default_point_patch_token * point_token_len}\n{question}"
    
    # Conversation Template
    from pointllm.conversation import conv_templates, SeparatorStyle
    conv = conv_templates["vicuna_v1_1"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    
    inputs = tokenizer([full_prompt], return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    
    # 点群データの型変換
    point_cloud = point_cloud.to(device=device, dtype=dtype)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            point_clouds=point_cloud,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True
        )
        
    input_len = input_ids.shape[1]
    response = tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0]
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    if response.endswith(stop_str):
        response = response[:-len(stop_str)]
        
    return response.strip()

# -----------------------------------------------------------------------------
# 6. Gradient Attribution Module
# -----------------------------------------------------------------------------

class PointCloudGradientAttribution:
    """点群に対する勾配ベースの帰属分析クラス"""
    
    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ):
        """
        Args:
            model: PointLLMモデル
            tokenizer: トークナイザー
            device: 計算デバイス
            dtype: 計算精度
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        
        # モデルを評価モードに
        self.model.eval()
        
        # Point Backbone Configの取得
        self.point_config = self._get_point_config()
        
    def _get_point_config(self) -> dict:
        """モデルからpoint_backbone_configを取得"""
        if hasattr(self.model, "get_model"):
            return self.model.get_model().point_backbone_config
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "get_model"):
            return self.model.base_model.get_model().point_backbone_config
        else:
            point_config = getattr(self.model.config, "point_backbone_config", None)
            if point_config is None:
                point_config = self.model.base_model.model.point_backbone_config
            return point_config
    
    def _prepare_prompt(self, question: str) -> str:
        """プロンプトを構築"""
        from pointllm.conversation import conv_templates
        
        question = clean_question_text(question, self.point_config)
        
        point_token_len = self.point_config['point_token_len']
        default_point_patch_token = self.point_config['default_point_patch_token']
        mm_use_point_start_end = self.point_config.get('mm_use_point_start_end', False)
        
        # プロンプト構築
        if mm_use_point_start_end:
            start = self.point_config['default_point_start_token']
            end = self.point_config['default_point_end_token']
            prompt = f"{start}{default_point_patch_token * point_token_len}{end}\n{question}"
        else:
            prompt = f"{default_point_patch_token * point_token_len}\n{question}"
        
        # Conversation Template
        conv = conv_templates["vicuna_v1_1"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()
        
        return full_prompt
    
    def _tokenize_target(self, target_text: str) -> torch.Tensor:
        """正解テキストをトークン化"""
        tokens = self.tokenizer(
            target_text,
            return_tensors="pt",
            add_special_tokens=False
        )
        return tokens.input_ids.to(self.device)
    
    def compute_loss(
        self,
        point_cloud: torch.Tensor,
        question: str,
        target_text: str
    ) -> torch.Tensor:
        """
        モデルの損失を計算
        
        Args:
            point_cloud: 点群テンソル [1, N, 6]
            question: 質問文
            target_text: 正解テキスト
            
        Returns:
            loss: 損失値
        """
        # プロンプト準備
        full_prompt = self._prepare_prompt(question)
        
        # 入力のトークン化
        inputs = self.tokenizer([full_prompt], return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        
        # 正解のトークン化
        target_ids = self._tokenize_target(target_text)
        
        # ラベルの構築: 入力部分は-100(無視)、正解部分のみ学習対象
        labels = torch.full_like(input_ids, -100)
        labels = torch.cat([labels, target_ids], dim=1)
        
        # 入力IDも正解分を結合
        full_input_ids = torch.cat([input_ids, target_ids], dim=1)
        
        # モデルのフォワード
        outputs = self.model(
            input_ids=full_input_ids,
            point_clouds=point_cloud,
            labels=labels,
            return_dict=True
        )
        
        return outputs.loss
    
    def compute_gradients(
        self,
        point_cloud: torch.Tensor,
        question: str,
        target_text: str
    ) -> torch.Tensor:
        """
        点群の各要素に対する勾配を計算
        
        Args:
            point_cloud: 点群テンソル [1, N, 6]
            question: 質問文
            target_text: 正解テキスト
            
        Returns:
            gradients: 勾配テンソル [N, 6]
        """
        # 点群のコピーを作成し、勾配計算を有効化
        pc_copy = point_cloud.clone().detach().to(dtype=self.dtype)
        pc_copy.requires_grad = True
        
        # 損失計算
        loss = self.compute_loss(pc_copy, question, target_text)
        
        # 勾配計算
        loss.backward()
        
        # 勾配を取得 [1, N, 6] -> [N, 6]
        gradients = pc_copy.grad.squeeze(0).detach()
        
        return gradients
    
    def compute_importance_scores(
        self,
        gradients: torch.Tensor,
        point_cloud: torch.Tensor,
        aggregation: Literal["l2_norm", "abs_sum", "max_abs", "mean_abs"] = "l2_norm"
    ) -> np.ndarray:
        """
        勾配から各点の重要度スコアを計算
        
        Args:
            gradients: 勾配テンソル [N, 6]
            point_cloud: 元の点群 [1, N, 6]
            aggregation: 集約方法
                
        Returns:
            importance_scores: 各点の重要度 [N]
        """
        # 点群を [N, 6] に整形
        pc = point_cloud.squeeze(0)
        
        if aggregation == "l2_norm":
            # 各点の6次元ベクトルのL2ノルム
            scores = torch.norm(gradients, p=2, dim=1)
            
        elif aggregation == "abs_sum":
            # 絶対値の合計
            scores = torch.sum(torch.abs(gradients), dim=1)
            
        elif aggregation == "max_abs":
            # 絶対値の最大値
            scores = torch.max(torch.abs(gradients), dim=1)[0]
            
        elif aggregation == "mean_abs":
            # 絶対値の平均
            scores = torch.mean(torch.abs(gradients), dim=1)
            
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        return scores.cpu().numpy()


# -----------------------------------------------------------------------------
# 7. Color Detection and Segmentation
# -----------------------------------------------------------------------------

def detect_target_colors_in_question(question: str) -> List[str]:
    """
    質問文から対象となる色を検出
    
    Args:
        question: 質問文
        
    Returns:
        detected_colors: 検出された色のリスト (例: ['red', 'blue', 'dark_gray'])
    """
    question_lower = question.lower()
    
    # 検出する色のパターン
    color_patterns = {
        'red': ['red', '赤'],
        'blue': ['blue', '青'],
        'dark_gray': ['dark gray', 'dark grey', 'darkgray', 'darkgrey', 'black', '暗灰色']
    }
    
    detected = []
    for color_name, patterns in color_patterns.items():
        if any(pattern in question_lower for pattern in patterns):
            detected.append(color_name)
    
    return detected


def get_color_reference_rgb(color_name: str) -> np.ndarray:
    """
    色名から参照RGB値を取得
    
    Args:
        color_name: 色名 ('red', 'blue', 'dark_gray')
        
    Returns:
        rgb: RGB値 [3] (0-1範囲)
    """
    color_map = {
        'red': np.array([1.0, 0.0, 0.0]),
        'blue': np.array([0.0, 0.0, 1.0]),
        'dark_gray': np.array([0.0, 0.0, 0.0])  # 暗めのグレー
    }
    
    return color_map.get(color_name, np.array([0.5, 0.5, 0.5]))


def compute_color_membership(
    point_cloud: np.ndarray,
    target_color_rgb: np.ndarray,
    sigma: float = 0.2
) -> np.ndarray:
    """
    ガウス減衰関数による色の所属度を計算
    
    Args:
        point_cloud: 点群 [N, 6] (xyz + rgb)
        target_color_rgb: 目標色のRGB [3]
        sigma: ガウスカーネルの標準偏差
        
    Returns:
        membership: 各点の所属度 [N] (0-1範囲)
    """
    rgb = point_cloud[:, 3:6]
    
    # RGB空間でのユークリッド距離
    distances = np.linalg.norm(rgb - target_color_rgb, axis=1)
    
    # ガウス減衰
    membership = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    
    return membership


def segment_points_by_color(
    point_cloud: np.ndarray,
    detected_colors: List[str],
    threshold: float = 0.5,
    sigma: float = 0.2
) -> Dict[str, np.ndarray]:
    """
    色ごとに点群をセグメンテーション
    
    Args:
        point_cloud: 点群 [N, 6]
        detected_colors: 検出された色のリスト
        threshold: 所属度の閾値
        sigma: ガウスカーネルの標準偏差
        
    Returns:
        segmentation: 各色の所属マスク辞書 {color_name: mask[N]}
    """
    segmentation = {}
    
    for color_name in detected_colors:
        target_rgb = get_color_reference_rgb(color_name)
        membership = compute_color_membership(point_cloud, target_rgb, sigma)
        
        # 閾値処理
        mask = membership >= threshold
        segmentation[color_name] = mask
    
    # "other"カテゴリ: どの色にも属さない点
    all_color_masks = np.zeros(len(point_cloud), dtype=bool)
    for mask in segmentation.values():
        all_color_masks |= mask
    
    segmentation['other'] = ~all_color_masks
    
    return segmentation


# -----------------------------------------------------------------------------
# 8. Gradient Analysis with Color Segmentation
# -----------------------------------------------------------------------------

def analyze_gradients_by_color(
    model,
    tokenizer,
    point_cloud: torch.Tensor,
    question: str,
    target_text: str,
    device: torch.device,
    dtype: torch.dtype,
    aggregation: str = "l2_norm"
) -> Dict[str, Any]:
    """
    色別に勾配を分析
    
    Args:
        model: PointLLMモデル
        tokenizer: トークナイザー
        point_cloud: 点群テンソル [1, 8192, 6]
        question: 質問文
        target_text: 正解テキスト
        device: デバイス
        dtype: データ型
        aggregation: スコア集約方法
        
    Returns:
        analysis_result: 分析結果の辞書
    """
    # 色検出
    detected_colors = detect_target_colors_in_question(question)
    
    if not detected_colors:
        # 色が検出されない場合はスキップ
        return None
    
    # Attribution計算器の初期化
    attributor = PointCloudGradientAttribution(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype
    )
    
    # 点群をデバイスとdtypeに変換
    point_cloud_device = point_cloud.to(device=device, dtype=dtype)
    
    # 勾配計算
    gradients = attributor.compute_gradients(point_cloud_device, question, target_text)
    
    # 重要度スコア計算
    importance_scores = attributor.compute_importance_scores(
        gradients, 
        point_cloud_device, 
        aggregation=aggregation
    )
    
    # 点群をnumpy配列に変換
    pc_numpy = point_cloud.squeeze(0).cpu().numpy()
    
    # 色別セグメンテーション
    segmentation = segment_points_by_color(pc_numpy, detected_colors)
    
    # 各色カテゴリの平均勾配を計算
    color_gradient_means = {}
    color_point_counts = {}
    
    for color_name, mask in segmentation.items():
        if mask.sum() > 0:
            color_gradient_means[color_name] = importance_scores[mask].mean()
            color_point_counts[color_name] = mask.sum()
        else:
            color_gradient_means[color_name] = 0.0
            color_point_counts[color_name] = 0
    
    return {
        'detected_colors': detected_colors,
        'color_gradient_means': color_gradient_means,
        'color_point_counts': color_point_counts,
        'total_points': len(pc_numpy)
    }


# -----------------------------------------------------------------------------
# 9. Main Execution with Gradient Analysis
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PointLLM Two-Phase Fine-tuning Batch Inference with Gradient Analysis"
    )
    parser.add_argument("--dataset_path", type=str, required=True, 
                       help="Path to dataset root")
    parser.add_argument("--test_json", type=str, default="annotations/test.json", 
                       help="Path to test json")
    parser.add_argument("--model_path", type=str, default="RunsenXu/PointLLM_7B_v1.2", 
                       help="Base model path")
    parser.add_argument("--phase1_checkpoint", type=str, required=True,
                       help="Phase 1 checkpoint path (Encoder + Projector training)")
    parser.add_argument("--phase2_checkpoint", type=str, required=True,
                       help="Phase 2 checkpoint path (Projector + LLM training)")
    parser.add_argument("--output_file", type=str, default="results.xlsx", 
                       help="Output Excel file path")
    parser.add_argument("--device", type=str, default="auto", 
                       help="cuda, mps, cpu, or auto")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_points", type=int, default=8192)
    parser.add_argument("--pointllm_path", type=str, default=None)
    
    # 勾配分析用のオプション
    parser.add_argument("--enable_gradient_analysis", action="store_true",
                       help="Enable gradient-based color analysis")
    parser.add_argument("--gradient_output_csv", type=str, default="gradient_analysis_by_color.csv",
                       help="Output CSV file for gradient analysis")
    parser.add_argument("--gradient_aggregation", type=str, default="l2_norm",
                       choices=["l2_norm", "abs_sum", "max_abs", "mean_abs"],
                       help="Gradient aggregation method")
    
    args = parser.parse_args()
    
    # PointLLM環境のセットアップ
    setup_pointllm_environment(args.pointllm_path)
    
    # デバイス設定
    device = get_device(args.device, verbose=True)
    
    # データタイプの設定
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Using dtype: {dtype}")
    
    # モデルのロード
    model, tokenizer = load_full_finetuned_model(
        args.model_path, 
        phase1_checkpoint_path=args.phase1_checkpoint,
        phase2_checkpoint_path=args.phase2_checkpoint,
        device=device, 
        dtype=dtype
    )
    
    # データセットのロード
    dataset = load_test_dataset(args.dataset_path, args.test_json)
    samples = dataset['data']
    print(f"\nFound {len(samples)} samples for inference")
    
    results = []
    gradient_results = []
    
    # 色ごとの勾配を集計するための辞書
    color_gradient_accumulator = defaultdict(list)
    
    print(f"\n{'='*60}")
    print("Starting Batch Inference")
    if args.enable_gradient_analysis:
        print("Gradient Analysis: ENABLED")
        print(f"Gradient Output: {args.gradient_output_csv}")
    print(f"{'='*60}\n")
    
    for i, sample in enumerate(tqdm(samples, desc="Processing")):
        sample_id = sample.get('id', f'sample_{i}')
        pc_rel_path = sample['point_cloud']
        pc_path = os.path.join(args.dataset_path, pc_rel_path)
        
        # 会話データから質問と正解を抽出
        conversations = sample.get('conversations', [])
        question = ""
        ground_truth = ""
        
        for turn in conversations:
            if turn['role'] == 'user':
                content = turn['content']
                # 古い形式の場合の処理
                lines = content.split('\n')
                if len(lines) > 1 and any(token in lines[0] for token in ['<point', 'point>']):
                    question = '\n'.join(lines[1:])
                else:
                    question = content
            elif turn['role'] == 'assistant':
                ground_truth = turn['content']
        
        if not question:
            question = "Describe this 3D object."
            
        try:
            # 点群のロードと前処理
            pc_tensor = load_and_process_point_cloud(pc_path, args.num_points)
            
            # 推論実行
            response = generate_response(
                model, tokenizer, pc_tensor, question, device, dtype,
                max_new_tokens=args.max_new_tokens
            )
            
            # 勾配分析（オプション）
            if args.enable_gradient_analysis and ground_truth:
                try:
                    analysis = analyze_gradients_by_color(
                        model=model,
                        tokenizer=tokenizer,
                        point_cloud=pc_tensor,
                        question=question,
                        target_text=ground_truth,
                        device=device,
                        dtype=dtype,
                        aggregation=args.gradient_aggregation
                    )
                    
                    if analysis is not None:
                        # サンプルごとの結果を保存
                        gradient_row = {
                            'sample_id': sample_id,
                            'detected_colors': ','.join(analysis['detected_colors']),
                            'total_points': analysis['total_points']
                        }
                        
                        # 各色カテゴリの情報を追加
                        for color_name in ['red', 'blue', 'dark_gray', 'other']:
                            gradient_mean = analysis['color_gradient_means'].get(color_name, 0.0)
                            point_count = analysis['color_point_counts'].get(color_name, 0)
                            
                            gradient_row[f'{color_name}_gradient_mean'] = gradient_mean
                            gradient_row[f'{color_name}_point_count'] = point_count
                            
                            # 全体集計用に蓄積
                            if point_count > 0:
                                color_gradient_accumulator[color_name].append(gradient_mean)
                        
                        gradient_results.append(gradient_row)
                        
                except Exception as e:
                    print(f"\n⚠️ Gradient analysis failed for {sample_id}: {e}")
                    
        except Exception as e:
            print(f"\n❌ Error processing {sample_id}: {e}")
            response = f"ERROR: {str(e)}"
            
        results.append({
            "id": sample_id,
            "point_cloud": pc_rel_path,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": response
        })
        
        # 中間保存 (10サンプルごと)
        if (i + 1) % 10 == 0:
            pd.DataFrame(results).to_excel(args.output_file, index=False)
            if args.enable_gradient_analysis and gradient_results:
                pd.DataFrame(gradient_results).to_csv(args.gradient_output_csv, index=False)
            print(f"\n💾 Intermediate save: {i+1}/{len(samples)} samples processed")
            
    # 最終保存
    df = pd.DataFrame(results)
    df.to_excel(args.output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Inference completed successfully!")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output_file}")
    print(f"Total samples: {len(results)}")
    
    # 勾配分析結果の保存と集計
    if args.enable_gradient_analysis and gradient_results:
        # サンプルごとの結果を保存
        gradient_df = pd.DataFrame(gradient_results)
        gradient_df.to_csv(args.gradient_output_csv, index=False)
        
        # 全体統計を計算
        overall_stats = {}
        for color_name in ['red', 'blue', 'dark_gray', 'other']:
            if color_name in color_gradient_accumulator and color_gradient_accumulator[color_name]:
                overall_stats[f'{color_name}_mean_gradient'] = np.mean(color_gradient_accumulator[color_name])
                overall_stats[f'{color_name}_std_gradient'] = np.std(color_gradient_accumulator[color_name])
                overall_stats[f'{color_name}_sample_count'] = len(color_gradient_accumulator[color_name])
            else:
                overall_stats[f'{color_name}_mean_gradient'] = 0.0
                overall_stats[f'{color_name}_std_gradient'] = 0.0
                overall_stats[f'{color_name}_sample_count'] = 0
        
        # 統計情報を追加行として保存
        stats_df = pd.DataFrame([overall_stats])
        stats_df.insert(0, 'sample_id', 'OVERALL_STATISTICS')
        
        # 既存のCSVに統計行を追加
        combined_df = pd.concat([gradient_df, stats_df], ignore_index=True)
        combined_df.to_csv(args.gradient_output_csv, index=False)
        
        print(f"\n{'='*60}")
        print("Gradient Analysis Summary")
        print(f"{'='*60}")
        print(f"Gradient analysis saved to: {args.gradient_output_csv}")
        print(f"Analyzed samples: {len(gradient_results)}")
        print(f"\nOverall Statistics:")
        for color_name in ['red', 'blue', 'dark_gray', 'other']:
            mean_grad = overall_stats.get(f'{color_name}_mean_gradient', 0.0)
            std_grad = overall_stats.get(f'{color_name}_std_gradient', 0.0)
            count = overall_stats.get(f'{color_name}_sample_count', 0)
            print(f"  {color_name:12s}: mean={mean_grad:.6f}, std={std_grad:.6f}, n={count}")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()