
"""
chat_gradio_full_llm_finetune.py

2段階学習(Phase 1: Encoder+Projector, Phase 2: Projector+LLM)で学習した重みを読み込み、
データセットに対してバッチ推論を実行してExcelファイルに出力します。

主な修正点:
- 2段階学習に対応（Phase 1とPhase 2のチェックポイントを別々にロード）
- LLM全体の重みロードに対応
- データセット内の誤った特殊トークン形式(<point_start><point_patch>*513<point_end>)に対応
- 型の統一(float16)

使用方法:
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD python pointllm/eval/chat_gradio_full_llm_finetune.py \
        --dataset_path "/Users/user/Downloads/demo_20260120_ar_dataset_final" \
        --test_json "/Users/user/Downloads/demo_20260120_ar_dataset_final/annotations/test.json" \
        --model_path "RunsenXu/PointLLM_7B_v1.2" \
        --phase1_checkpoint "/Users/user/Documents/kimura/study/PointLLM/pointllm/outputs/encoder_proj_tune_v8_demo_dental_model_head_coord_410_final/checkpoints/best_model" \
        --phase2_checkpoint "/Users/user/Documents/kimura/study/PointLLM/pointllm/outputs/demo_dental_model_head_coord_410_final_encoder_proj_llm_finetune_v5/checkpoints/checkpoint-epoch-5" \
        --sample_indices 3
        --device mps
        
        (--output_file "demo_dental_model_410_results_full_llm.xlsx" \
        --device "cuda")
"""

import argparse
import json
import os
import sys
from pathlib import Path
import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from PIL import Image
import io

# -----------------------------------------------------------------------------
# 1. Environment & Setup
# -----------------------------------------------------------------------------
# --- Attentionをキャプチャするためのグローバル（またはクラス）変数 ---
attention_storage = {}
def save_attention_hook(module, input, output):
    # LlamaAttentionの出力は (hidden_states, self_attn_weights, present_key_value)
    # weightsが返ってくるように設定して呼び出す
    if len(output) > 1:
        attention_storage['weights'] = output[1].detach().cpu()

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
# 2. Model Loading (Full LLM Fine-tuning Support)
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
        
        # 2-2. LLM weights - safetensors形式を優先
        safetensors_index = os.path.join(phase2_checkpoint_path, "model.safetensors.index.json")
        llm_model_path = os.path.join(phase2_checkpoint_path, "llm_model.pt")
        
        if os.path.exists(safetensors_index):
            # safetensors形式でロード
            print(f"   Loading: LLM weights from safetensors")
            try:
                load_safetensors_weights(model, phase2_checkpoint_path, device, dtype)
            except Exception as e:
                print(f"   ⚠️ Error loading safetensors: {e}")
                
        elif os.path.exists(llm_model_path):
            # llm_model.pt形式でロード
            print(f"   Loading: llm_model.pt")
            load_llm_weights(model, llm_model_path, device, dtype)
        else:
            # Alternative: Try pytorch_model.bin
            alt_path = os.path.join(phase2_checkpoint_path, "pytorch_model.bin")
            if os.path.exists(alt_path):
                print(f"   Loading: pytorch_model.bin (alternative)")
                load_full_model_weights(model, alt_path, device, dtype)
            else:
                print(f"   ⚠️ Warning: No LLM weights found in Phase 2 checkpoint")
                print(f"   💡 Note: Only point_proj.pt will be loaded")
    else:
        print(f"\n⚠️ Phase 2 checkpoint not provided - skipping Projector + LLM loading")
    
    # --- デバッグ済みの最終パッチ ---
    # 1. ライブラリが地雷を踏まないよう、設定レベルでアテンション出力を完全に封印する
    model.config.output_attentions = False
    
    # 2. モンキーパッチが残っている場合でも、この属性を消せばライブラリは detach を呼ばなくなります
    if hasattr(model, 'orig_embeds_params'):
        del model.orig_embeds_params
    # ----------------------------
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
        # キーのプレフィックス調整
        clean_key = ckpt_key[6:] if ckpt_key.startswith("model.") else ckpt_key
        
        candidates = [
            ckpt_key,
            clean_key,
            f"model.{clean_key}",
        ]
        
        matched = False
        for candidate in candidates:
            if candidate in model_state_dict:
                model_state_dict[candidate] = ckpt_value.to(device=device, dtype=dtype)
                loaded_keys.append(candidate)
                matched = True
                break
        
        # 部分一致も試す
        if not matched:
            for model_key in model_state_dict.keys():
                if model_key.endswith(clean_key) or clean_key in model_key:
                    model_state_dict[model_key] = ckpt_value.to(device=device, dtype=dtype)
                    loaded_keys.append(model_key)
                    matched = True
                    break
    
    model.load_state_dict(model_state_dict, strict=False)
    
    encoder_params = sum(1 for k in loaded_keys if 'point_backbone' in k or 'encoder' in k)
    projector_params = sum(1 for k in loaded_keys if 'point_proj' in k or 'projector' in k)
    
    print(f"   ✅ Loaded {len(loaded_keys)} parameters")
    print(f"      - Encoder: {encoder_params}")
    print(f"      - Projector: {projector_params}")


def load_llm_weights(model, checkpoint_path, device, dtype):
    """LLM部分の重みをロード（safetensors/pytorch形式対応）"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    else:
        state_dict = checkpoint
    
    model_state_dict = model.state_dict()
    loaded_keys = []
    
    for ckpt_key, ckpt_value in state_dict.items():
        clean_key = ckpt_key[6:] if ckpt_key.startswith("model.") else ckpt_key
        
        candidates = [
            ckpt_key,
            clean_key,
            f"model.{clean_key}",
        ]
        
        matched = False
        for candidate in candidates:
            if candidate in model_state_dict:
                model_state_dict[candidate] = ckpt_value.to(device=device, dtype=dtype)
                loaded_keys.append(candidate)
                matched = True
                break
    
    model.load_state_dict(model_state_dict, strict=False)
    
    llm_params = sum(1 for k in loaded_keys if 'llama_model' in k or 'lm_head' in k)
    
    print(f"   ✅ Loaded {len(loaded_keys)} LLM parameters")
    print(f"      - LLM components: {llm_params}")


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
    
    for key in model_state_dict.keys():
        if key in all_state_dict:
            model_state_dict[key] = all_state_dict[key].to(device=device, dtype=dtype)
            loaded_keys.append(key)
        else:
            # プレフィックスを試す
            clean_key = key[6:] if key.startswith("model.") else key
            alt_key = f"model.{clean_key}" if not key.startswith("model.") else clean_key
            
            if clean_key in all_state_dict:
                model_state_dict[key] = all_state_dict[clean_key].to(device=device, dtype=dtype)
                loaded_keys.append(key)
            elif alt_key in all_state_dict:
                model_state_dict[key] = all_state_dict[alt_key].to(device=device, dtype=dtype)
                loaded_keys.append(key)
            else:
                missing_keys.append(key)
    
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
        clean_key = ckpt_key[6:] if ckpt_key.startswith("model.") else ckpt_key
        
        candidates = [
            ckpt_key,
            clean_key,
            f"model.{clean_key}",
        ]
        
        matched = False
        for candidate in candidates:
            if candidate in model_state_dict:
                model_state_dict[candidate] = ckpt_value.to(device=device, dtype=dtype)
                updated_keys.append(candidate)
                matched = True
                break
    
    model.load_state_dict(model_state_dict, strict=False)
    
    print(f"   ✅ Updated {len(updated_keys)} Projector parameters")

# -----------------------------------------------------------------------------
# 3. Data Processing
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
        print(f"   🔧 Cleaned malformed tokens: <point_start><point_patch>*{match1.group(1)}<point_end>")
    
    # Pattern 2: <point><point_patch>*N
    pattern2 = r'<point><point_patch>\*(\d+)'
    match2 = re.search(pattern2, content)
    
    if match2:
        content = re.sub(pattern2, '', content)
        print(f"   🔧 Cleaned malformed tokens: <point><point_patch>*{match2.group(1)}")
    
    # Pattern 3: <point_patch>*N (スタンドアロン)
    pattern3 = r'<point_patch>\*(\d+)'
    match3 = re.search(pattern3, content)
    
    if match3:
        content = re.sub(pattern3, '', content)
        print(f"   🔧 Cleaned malformed tokens: <point_patch>*{match3.group(1)}")
    
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
# 4. 可視化ロジック (新規追加)
# -----------------------------------------------------------------------------

def create_comparison_gif(pc_xyz, pc_rgb, attention_weights, question, prediction, gt, save_path):
    """左：元点群、右：AttentionヒートマップのGIFを生成"""
    fig = plt.figure(figsize=(14, 7))
    
    # 重みの正規化 (0-1)
    weights = attention_weights.numpy()
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    
    frames = []
    # 45度ずつ回転させて8枚のフレームを作成
    for angle in range(0, 360, 45):
        plt.clf()
        # 左：Original
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(pc_xyz[:, 0], pc_xyz[:, 2], pc_xyz[:, 1], c=pc_rgb, s=1, alpha=0.8)
        ax1.set_title("Original Point Cloud")
        ax1.view_init(elev=20, azim=angle)
        ax1.axis('off')

        # 右：Attention
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(pc_xyz[:, 0], pc_xyz[:, 2], pc_xyz[:, 1], c=weights, cmap='jet', s=3)
        ax2.set_title("Model Attention (Focus Area)")
        ax2.view_init(elev=20, azim=angle)
        ax2.axis('off')

        plt.figtext(0.5, 0.05, f"Q: {question}\nPred: {prediction}\nGT: {gt}", 
                    ha="center", fontsize=9, bbox={"facecolor":"white", "alpha":0.8, "pad":5})

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf))
    
    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=300, loop=0)
    plt.close(fig)

def generate_response_with_vis(
    model, tokenizer, point_cloud, question, device, dtype, 
    sample_id, ground_truth, visualize=True
):
    print(f"\n--- Processing {sample_id} (Final Logic) ---")
    
    # 1. プロンプト構築（PointLLM公式仕様: 513トークン）
    point_token_len = 513
    prompt = f"<point_start>{'<point_patch>' * point_token_len}<point_end>\n{question}"
    
    from pointllm.conversation import conv_templates
    conv = conv_templates["vicuna_v1_1"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    
    inputs = tokenizer([full_prompt], return_tensors="pt").to(device)
    pc_tensor = point_cloud.to(device=device, dtype=dtype)
    
    # 2. フックの設置（model.model.layers に直接アクセス）
    # pointllm.py 107行目の self.model = PointLLMLlamaModel(config) に対応
    global attention_storage
    attention_storage = {}
    
    try:
        # ForCausalLM -> PointLLMLlamaModel -> layers
        target_layer = model.model.layers[-1].self_attn
    except:
        # 直接 Model が渡されている場合
        target_layer = model.layers[-1].self_attn

    hook = target_layer.register_forward_hook(save_attention_hook)

    # 3. 推論実行（output_attentionsを使わず、通常推論中にフックで抜く）
    response = "GEN_FAILED"
    try:
        with torch.no_grad():
            # output_attentions=True を渡すと内部で detach エラーが出るため、
            # あえて指定せず、フックのみで内部データを盗み取ります
            output = model.generate(
                input_ids=inputs.input_ids,
                point_clouds=pc_tensor,
                max_new_tokens=30,
                output_attentions=False, 
                use_cache=False # 注目度を正しく取るため
            )
        response = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"Prediction: {response}")
    except Exception as e:
        print(f"❌ Generation failed: {e}")

    hook.remove()

    # 4. 可視化（フックが成功していれば実行）
    if visualize and 'weights' in attention_storage:
        try:
            # PointLLM v1.2 のアテンションマップから点群への注目を抽出
            # [Batch, Head, Seq, Seq]
            weights = attention_storage['weights'][0].mean(dim=0)
            
            # PointLLMの入力順序に基づきスライス
            # BOS(0), start(1), point_patch*513(2:515)
            token_attn = weights[-1, 2:515].float().cpu()
            
            pc_np = point_cloud[0].cpu().numpy()
            os.makedirs("vis_results", exist_ok=True)
            save_path = f"vis_results/{sample_id}.gif"
            
            create_comparison_gif(pc_np[:, :3], pc_np[:, 3:6], token_attn, question, response, ground_truth, save_path)
            print(f"✅ SUCCESS: {save_path}")
        except Exception as e:
            print(f"⚠️ Visualization skip: {e}")

    return response
# -----------------------------------------------------------------------------
# 6. Main (修正)
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # (既存の引数)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--test_json", type=str, default="annotations/test.json")
    parser.add_argument("--model_path", type=str, default="RunsenXu/PointLLM_7B_v1.2")
    parser.add_argument("--phase1_checkpoint", type=str, required=True)
    parser.add_argument("--phase2_checkpoint", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="results.xlsx")
    parser.add_argument("--device", type=str, default="mps")
    # --- 新規引数 ---
    parser.add_argument("--sample_indices", type=int, nargs="+", help="可視化したいインデックス (例: 0 5 10)")
    
    args = parser.parse_args()
    setup_pointllm_environment(str(Path(__file__).parent.parent.parent))
    device = get_device(args.device, verbose=True)
    dtype = torch.float16
    
    model, tokenizer = load_full_finetuned_model(args.model_path, args.phase1_checkpoint, args.phase2_checkpoint, device, dtype)
    dataset = load_test_dataset(args.dataset_path, args.test_json)
    
    samples = dataset['data']
    # 特定のインデックスが指定されている場合、そのサンプルのみ抽出
    if args.sample_indices:
        print(f"📊 Target indices: {args.sample_indices}")
        samples = [samples[i] for i in args.sample_indices if i < len(samples)]

    results = []
    for i, sample in enumerate(tqdm(samples)):
        sample_id = sample.get('id', f'sample_{i}')
        pc_path = os.path.join(args.dataset_path, sample['point_cloud'])
        
        # 会話からQとGTを抽出 (元のロジック)
        question = sample['conversations'][0]['content']
        ground_truth = sample['conversations'][1]['content']
        
        try:
            pc_tensor = load_and_process_point_cloud(pc_path, 8192)
            # 可視化付き推論を実行
            response = generate_response_with_vis(
                model, tokenizer, pc_tensor, question, device, dtype, 
                sample_id, ground_truth, visualize=True
            )
        except Exception as e:
            print(f"Error at {sample_id}: {e}")
            response = "ERROR"

        results.append({"id": sample_id, "prediction": response, "gt": ground_truth})

    pd.DataFrame(results).to_excel(args.output_file, index=False)
    print(f"Done. Check 'vis_results/' folder for GIFs.")

if __name__ == "__main__":
    main()