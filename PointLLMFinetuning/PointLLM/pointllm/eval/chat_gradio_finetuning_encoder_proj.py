"""
chat_gradio_finetuning_encoder_proj.py

Point EncoderおよびProjection層の学習済み重みを読み込み、データセットに対してバッチ推論を実行して
結果をExcelファイルに出力します。

LoRA版との違い:
- LoRAアダプタは使用しない
- Point Encoder (point_backbone) と Projection層 (point_proj) の重みをロード
- LLMは完全に凍結状態（学習済みのまま）

使用方法:
    python pointllm/eval/chat_gradio_finetuning_encoder_proj.py \
        --dataset_path "/home/yyamashita/Desktop/kkimu/test/dental_model_dataset" \
        --test_json "/home/yyamashita/Desktop/kkimu/test/dental_model_dataset/annotations/test.json" \
        --model_path "RunsenXu/PointLLM_7B_v1.2" \
        --checkpoint_path "/home/yyamashita/Desktop/kkimu/test/ProjectionFinetuning/outputs/encoder_projector_finetune_v4_dental_model/checkpoints/best_model" \
        --output_file "/home/yyamashita/Desktop/kkimu/test/dental_model_dataset_results.xlsx" \
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
# 2. Model Loading (Encoder + Projector Support)
# -----------------------------------------------------------------------------

def load_model_and_checkpoint(
    model_path: str,
    checkpoint_path: Optional[str] = None,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float16
):
    """
    ベースモデルとEncoder + Projectorチェックポイントをロードします
    
    Args:
        model_path: ベースモデルのパス
        checkpoint_path: Encoder + Projectorチェックポイントのパス
        device: 使用するデバイス
        dtype: モデルのデータ型
    
    Returns:
        (model, tokenizer) のタプル
    """
    print(f"\n{'='*60}")
    print("Loading Model (Encoder + Projector)")
    print(f"{'='*60}")
    print(f"Base model: {model_path}")
    print(f"Checkpoint: {checkpoint_path or 'None (using base weights)'}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    # トークナイザーのロード
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=False, 
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # ベースモデルのロード
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    
    # モデルをデバイスへ移動
    model = model.to(device=device, dtype=dtype)
    
    # Point Backbone Configの初期化
    if hasattr(model, "initialize_tokenizer_point_backbone_config_wo_embedding"):
        model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
        print("✅ Point backbone config initialized")
    elif hasattr(model, "get_model"):
        inner_model = model.get_model()
        if hasattr(inner_model, "initialize_tokenizer_point_backbone_config_wo_embedding"):
            inner_model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
            print("✅ Point backbone config initialized (via get_model)")
    
    # Encoder + Projectorの重みをロード
    if checkpoint_path and os.path.isdir(checkpoint_path):
        # encoder_projector.pt ファイルを探す
        encoder_proj_path = os.path.join(checkpoint_path, "encoder_projector.pt")
        
        if os.path.exists(encoder_proj_path):
            print(f"\nLoading Encoder + Projector weights from: {encoder_proj_path}")
            
            try:
                # チェックポイントのロード
                checkpoint = torch.load(encoder_proj_path, map_location="cpu")
                
                # state_dictの取得
                if isinstance(checkpoint, dict):
                    state_dict = checkpoint.get("state_dict", checkpoint)
                else:
                    state_dict = checkpoint
                
                # モデルの現在のstate_dict
                model_state_dict = model.state_dict()
                
                # Encoder + Projectorの重みを更新
                updated_keys = []
                for key, value in state_dict.items():
                    # point_backbone または point_proj を含むキーのみ
                    if "point_backbone" in key or "point_proj" in key:
                        if key in model_state_dict:
                            # 重みをロードし、指定の型とデバイスに変換
                            model_state_dict[key] = value.to(device=device, dtype=dtype)
                            updated_keys.append(key)
                        else:
                            # キー名が異なる可能性を考慮
                            for model_key in model_state_dict.keys():
                                if key in model_key and model_key.endswith(key.split('.')[-1]):
                                    model_state_dict[model_key] = value.to(device=device, dtype=dtype)
                                    updated_keys.append(model_key)
                                    break
                
                # state_dictをモデルにロード
                model.load_state_dict(model_state_dict, strict=False)
                
                # 統計情報の表示
                encoder_keys = [k for k in updated_keys if "point_backbone" in k]
                projector_keys = [k for k in updated_keys if "point_proj" in k]
                
                print(f"✅ Loaded {len(updated_keys)} parameters:")
                print(f"   - Point Encoder: {len(encoder_keys)} parameters")
                print(f"   - Point Projector: {len(projector_keys)} parameters")
                
            except Exception as e:
                print(f"⚠️ Error loading Encoder + Projector weights: {e}")
                print("Using base model weights instead.")
        else:
            print(f"⚠️ encoder_projector.pt not found in {checkpoint_path}")
            print("Using base model weights instead.")
    elif checkpoint_path:
        print(f"⚠️ Checkpoint path is not a directory: {checkpoint_path}")
        print("Using base model weights instead.")
    
    # 評価モードに設定
    model.eval()
    
    print(f"\n{'='*60}")
    print("Model loaded successfully")
    print(f"{'='*60}\n")
    
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
    inputs = tokenizer(
        full_prompts, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
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
            max_new_tokens=max_new_tokens,
            use_cache=True
        )
    
    # デコード（バッチ）
    responses = []
    for i in range(batch_size):
        input_len = (attention_mask[i] == 1).sum().item()  # 各サンプルの実際の入力長
        response = tokenizer.decode(
            output_ids[i, input_len:], 
            skip_special_tokens=True
        )
        
        # 後処理
        if sep and response.endswith(sep):
            response = response[:-len(sep)]
        if sep2 and response.endswith(sep2):
            response = response[:-len(sep2)]
        
        responses.append(response.strip())
    
    return responses

def generate_response(
    model,
    tokenizer,
    point_cloud: torch.Tensor,
    question: str,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """
    点群と質問からレスポンスを生成します
    
    Args:
        model: PointLLMモデル
        tokenizer: トークナイザー
        point_cloud: 点群テンソル (1, num_points, 6)
        question: 質問文
        device: デバイス
        dtype: データ型
        max_new_tokens: 生成する最大トークン数
        temperature: サンプリング温度
        top_p: nucleus sampling閾値
    
    Returns:
        生成されたレスポンス
    """
    # Point backbone configの取得
    if hasattr(model, "get_model"):
        point_config = model.get_model().point_backbone_config
    elif hasattr(model, "point_backbone_config"):
        point_config = model.point_backbone_config
    else:
        raise AttributeError("point_backbone_config not found in model")
    
    point_token_len = point_config['point_token_len']
    default_point_patch_token = point_config['default_point_patch_token']
    mm_use_point_start_end = point_config.get('mm_use_point_start_end', False)
    
    # プロンプトの構築
    if mm_use_point_start_end:
        start = point_config['default_point_start_token']
        end = point_config['default_point_end_token']
        prompt = f"{start}{default_point_patch_token * point_token_len}{end}\n{question}"
    else:
        prompt = f"{default_point_patch_token * point_token_len}\n{question}"
    
    # Conversation Templateの適用
    try:
        from pointllm.conversation import conv_templates, SeparatorStyle
        conv = conv_templates["vicuna_v1_1"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()
    except ImportError:
        # フォールバック: シンプルなフォーマット
        full_prompt = f"USER: {prompt}\nASSISTANT: "
    
    # トークン化
    inputs = tokenizer([full_prompt], return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    
    # 点群を適切な型とデバイスに変換
    point_cloud = point_cloud.to(device=device, dtype=dtype)
    
    # 生成
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
    
    # デコード
    input_len = input_ids.shape[1]
    response = tokenizer.batch_decode(
        output_ids[:, input_len:], 
        skip_special_tokens=True
    )[0]
    
    # 後処理
    try:
        from pointllm.conversation import SeparatorStyle
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        if response.endswith(stop_str):
            response = response[:-len(stop_str)]
    except:
        pass
    
    return response.strip()

# -----------------------------------------------------------------------------
# 5. Main Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PointLLM Encoder + Projector Batch Inference"
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
        "--checkpoint_path", 
        type=str, 
        default=None, 
        help="Encoder + Projector checkpoint directory path"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="results_encoder_proj.xlsx", 
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
        default=1024,
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
        default=0.7,
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
    
    # モデルのロード
    model, tokenizer = load_model_and_checkpoint(
        args.model_path,
        args.checkpoint_path,
        device,
        dtype
    )
    
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
            # バッチ全体でエラーの場合は個別に処理
            for metadata in batch_metadata:
                results.append({
                    "id": metadata['id'],
                    "point_cloud": metadata['point_cloud'],
                    "question": metadata['question'],
                    "ground_truth": metadata['ground_truth'],
                    "prediction": f"ERROR (batch): {str(e)}"
                })
        
        # 定期的に保存（バッチごと）
        if (batch_end) % (batch_size * 5) < batch_size:  # 5バッチごと
            df = pd.DataFrame(results)
            df.to_excel(args.output_file, index=False)
            print(f"\nIntermediate results saved ({len(results)} samples)")
    
    # 最終結果の保存
    df = pd.DataFrame(results)
    df.to_excel(args.output_file, index=False)
    
    print(f"\n{'='*60}")
    print("Inference Complete!")
    print(f"{'='*60}")
    print(f"Total samples processed: {len(results)}")
    print(f"Results saved to: {args.output_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()