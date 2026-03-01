"""
chat_gradio_finetuning_LoRA.py

LoRAおよびProjection層の学習済み重みを読み込み、データセットに対してバッチ推論を実行して
結果をExcelファイルに出力します。

修正点: 
- モデルと入力データの型を float16 に統一して型不一致エラーを回避します。
- get_device関数を追加し、外部からのインポートに対応しました。

使用方法:
    python pointllm/eval/chat_gradio_finetuning_LoRA.py \
        --dataset_path "/path/to/dataset" \
        --test_json "annotations/test.json" \
        --model_path "RunsenXu/PointLLM_7B_v1.2" \
        --checkpoint_path "/path/to/lora_checkpoint/best_model" \
        --output_file "results.xlsx" \
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
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        
        # 既存の登録を確認して重複登録を避ける（警告抑制）
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
            # フォールバック
            device = torch.device("cpu")
    
    if verbose:
        print(f"Using device: {device}")
        
    return device

# -----------------------------------------------------------------------------
# 2. Model Loading (LoRA & Projection Support)
# -----------------------------------------------------------------------------

def load_model_and_checkpoint(
    model_path: str,
    checkpoint_path: Optional[str] = None,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float16
):
    """ベースモデルとLoRA/Projectionチェックポイントをロードします"""
    print(f"\n{'='*60}")
    print("Loading Model")
    print(f"{'='*60}")
    print(f"Base model: {model_path}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype} (Using float16 as requested)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # ベースモデルのロード
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    
    # LoRA (PEFT) のロード
    if checkpoint_path and os.path.isdir(checkpoint_path):
        adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            try:
                from peft import PeftModel
                print(f"Loading LoRA adapters from: {checkpoint_path}")
                model = PeftModel.from_pretrained(model, checkpoint_path, torch_dtype=dtype)
                print("✅ LoRA adapters loaded successfully")
            except ImportError:
                print("⚠️ 'peft' library not found. LoRA adapters could not be loaded.")
            except Exception as e:
                print(f"⚠️ Error loading LoRA adapters: {e}")
    
    # モデル全体をデバイスへ移動し、型を確実に変換
    model = model.to(device=device, dtype=dtype)
    
    # Point Backbone Configの初期化
    base_model_ref = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base_model_ref = model.base_model.model
    
    if hasattr(base_model_ref, "initialize_tokenizer_point_backbone_config_wo_embedding"):
        base_model_ref.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
        print("✅ Point backbone config initialized")
    
    # Projection Layerのロード (point_proj.pt または bin)
    if checkpoint_path:
        proj_candidates = [
            os.path.join(checkpoint_path, "point_proj.pt"),
            os.path.join(checkpoint_path, "point_proj.bin"),
            os.path.join(checkpoint_path, "non_lora_trainables.bin")
        ]
        
        proj_path = None
        for p in proj_candidates:
            if os.path.exists(p):
                proj_path = p
                break
        
        if proj_path:
            print(f"Loading Point Projector from: {proj_path}")
            checkpoint = torch.load(proj_path, map_location="cpu")
            
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
            else:
                state_dict = checkpoint
            
            model_state_dict = model.state_dict()
            updated_keys = []
            
            for key, value in state_dict.items():
                target_key = key
                if key not in model_state_dict:
                    # PEFTラップ時のプレフィックス対応
                    for mk in model_state_dict.keys():
                        if key in mk and mk.endswith(key):
                            target_key = mk
                            break
                
                if target_key in model_state_dict:
                    # 重みをロードし、指定の型とデバイスに変換
                    model_state_dict[target_key] = value.to(device=device, dtype=dtype)
                    updated_keys.append(target_key)
            
            model.load_state_dict(model_state_dict, strict=False)
            print(f"✅ Loaded {len(updated_keys)} Point Projector parameters")
        else:
            print("ℹ️ No specific point_proj file found. Using LoRA/Base weights.")

    model.eval()
    return model, tokenizer

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

def load_test_dataset(dataset_path: str, test_json: str) -> Dict[str, Any]:
    """テストデータセットJSONを読み込みます"""
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

def generate_response(
    model, tokenizer, point_cloud, question, device, dtype,
    max_new_tokens=512, temperature=0.7, top_p=0.9
):
    # モデル設定の取得
    model_ref = model
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
    
    # ★★★ デバッグログ追加 ★★★
    print(f"\n{'='*80}")
    print(f"🔍 DEBUG: Point Token Configuration")
    print(f"{'='*80}")
    print(f"📊 point_token_len: {point_token_len}")
    print(f"🎯 default_point_patch_token: '{default_point_patch_token}'")
    print(f"🔄 mm_use_point_start_end: {mm_use_point_start_end}")
    
    if mm_use_point_start_end:
        start = point_config['default_point_start_token']
        end = point_config['default_point_end_token']
        prompt = f"{start}{default_point_patch_token * point_token_len}{end}\n{question}"
        print(f"✅ Using point start/end tokens: '{start}', '{end}'")
    else:
        prompt = f"{default_point_patch_token * point_token_len}\n{question}"
        print("✅ Using default point patch tokens only.")
    
    # ★★★ トークン数の実際の確認 ★★★
    actual_patch_count = prompt.count('<point_patch>')
    print(f"🔢 Actual <point_patch> count in prompt: {actual_patch_count}")
    print(f"❓ Question: {question[:100]}...")
    print(f"📝 Prompt preview (first 200 chars): {prompt[:200]}...")
    print(f"{'='*80}\n")
    
    # 以降は既存のコードそのまま...
    # Conversation Template
    from pointllm.conversation import conv_templates, SeparatorStyle
    conv = conv_templates["vicuna_v1_1"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    
    inputs = tokenizer([full_prompt], return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    
    # -----------------------------------------------------------
    # CRITICAL FIX: 入力点群を明示的に float16 (dtype) に変換する
    # -----------------------------------------------------------
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
# 5. Main Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PointLLM LoRA Batch Inference (Float16)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--test_json", type=str, default="annotations/test.json", help="Path to test json")
    parser.add_argument("--model_path", type=str, default="RunsenXu/PointLLM_7B_v1.2", help="Base model path")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="LoRA/Projection checkpoint path")
    parser.add_argument("--output_file", type=str, default="results.xlsx", help="Output Excel file path")
    parser.add_argument("--device", type=str, default="auto", help="cuda, mps, cpu, or auto")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_points", type=int, default=8192)
    parser.add_argument("--pointllm_path", type=str, default=None)
    
    args = parser.parse_args()
    
    setup_pointllm_environment(args.pointllm_path)
    
    # Device setup (Using new get_device function)
    device = get_device(args.device)
    
    # -----------------------------------------------------------
    # Dtype Selection: cudaならfloat16を強制
    # -----------------------------------------------------------
    if device.type == "cuda":
        # 学習時がbf16でも推論時はfp16の方が汎用性が高いことが多いが、
        # 型不一致エラー解消のためユーザー指定に従い float16 とする
        dtype = torch.float16
    else:
        dtype = torch.float32  # CPU/MPSはfloat32が安全
        
    print(f"Using device: {device}, dtype: {dtype}")
    
    # Model Load
    model, tokenizer = load_model_and_checkpoint(
        args.model_path, args.checkpoint_path, device, dtype
    )
    
    # Dataset Load
    dataset = load_test_dataset(args.dataset_path, args.test_json)
    samples = dataset['data']
    print(f"Found {len(samples)} samples.")
    
    results = []
    
    for i, sample in enumerate(tqdm(samples)):
        sample_id = sample.get('id', f'sample_{i}')
        pc_rel_path = sample['point_cloud']
        pc_path = os.path.join(args.dataset_path, pc_rel_path)
        
        conversations = sample.get('conversations', [])
        question = ""
        ground_truth = ""
        
        for turn in conversations:
            if turn['role'] == 'user':
                content = turn['content']
                lines = content.split('\n')
                if len(lines) > 1 and 'point' in lines[0]:
                    question = '\n'.join(lines[1:])
                else:
                    question = content
            elif turn['role'] == 'assistant':
                ground_truth = turn['content']
        
        if not question:
            question = "Describe this 3D object."
            
        try:
            pc_tensor = load_and_process_point_cloud(pc_path, args.num_points)
            
            # 生成関数へdtypeも渡す（または内部でtensor.to(dtype)する）
            response = generate_response(
                model, tokenizer, pc_tensor, question, device, dtype,
                max_new_tokens=args.max_new_tokens
            )
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            response = f"ERROR: {str(e)}"
            
        results.append({
            "id": sample_id,
            "point_cloud": pc_rel_path,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": response
        })
        
        if (i + 1) % 10 == 0:
            pd.DataFrame(results).to_excel(args.output_file, index=False)
            
    # Final save
    df = pd.DataFrame(results)
    df.to_excel(args.output_file, index=False)
    print(f"Saved results to {args.output_file}")

if __name__ == "__main__":
    main()