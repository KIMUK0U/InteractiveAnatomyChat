"""
inference_mps.py - PointLLM Inference on Apple Silicon (MPS)

ローカルMac（Apple Silicon）でファインチューニングしたPoint Projectorの重みをロードして推論を行います。

使用方法:
    python inference_mps.py \
        --model_path "RunsenXu/PointLLM_7B_v1.2" \
        --checkpoint_path "path/to/point_proj.pt" \
        --point_cloud_path "path/to/pointcloud.npy" \
        --question "この解剖学的構造は何ですか？"
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import AutoTokenizer


def get_device(preferred_device: Optional[str] = None, verbose: bool = True) -> torch.device:
    """
    利用可能なデバイスを検出して返します
    
    Args:
        preferred_device: 希望するデバイス ("cuda", "mps", "cpu", None)
        verbose: デバイス情報を表示するか
    
    Returns:
        torch.device: 使用するデバイス
    """
    if preferred_device:
        device = torch.device(preferred_device)
        if verbose:
            print(f"[INFO] Using specified device: {device}")
        return device
    
    # 自動検出
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"[INFO] CUDA available: {torch.cuda.get_device_name(1)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("[INFO] MPS (Apple Silicon) available")
    else:
        device = torch.device("cpu")
        if verbose:
            print("[INFO] Using CPU")
    
    if verbose:
        print(f"[INFO] Selected device: {device}")
    
    return device


def setup_pointllm_environment(pointllm_path: Optional[str] = None):
    """
    PointLLM環境をセットアップします
    
    Args:
        pointllm_path: PointLLMリポジトリのパス（Noneの場合は自動検出）
    """
    if pointllm_path is None:
        # 可能なパスを検索
        possible_paths = [
            "PointLLM",
            "../PointLLM",
            "../../PointLLM",
            os.path.expanduser("~/PointLLM"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                pointllm_path = path
                break
        
        if pointllm_path is None:
            raise RuntimeError(
                "PointLLM not found. Please specify --pointllm_path or install PointLLM"
            )
    
    pointllm_path = os.path.abspath(pointllm_path)
    
    if pointllm_path not in sys.path:
        sys.path.insert(0, pointllm_path)
    
    try:
        from pointllm.model import PointLLMLlamaForCausalLM
        from pointllm.model.pointllm import PointLLMConfig
        from transformers import AutoConfig, AutoModelForCausalLM
        
        AutoConfig.register("pointllm", PointLLMConfig)
        AutoModelForCausalLM.register(PointLLMConfig, PointLLMLlamaForCausalLM)
        
        print("✅ PointLLM environment setup complete")
        return True
    except ImportError as e:
        print(f"❌ Failed to setup PointLLM: {e}")
        return False


def load_model_and_checkpoint(
    model_path: str,
    checkpoint_path: Optional[str] = None,
    device: torch.device = torch.device("mps"),
    dtype: torch.dtype = torch.float32
) -> tuple:
    """
    ベースモデルとファインチューニングされたPoint Projectorをロードします
    
    Args:
        model_path: ベースモデルのパス
        checkpoint_path: Point Projectorのチェックポイントパス
        device: デバイス
        dtype: データ型
    
    Returns:
        (model, tokenizer): モデルとトークナイザー
    """
    print(f"\n{'='*60}")
    print("Loading Model")
    print(f"{'='*60}")
    print(f"Base model: {model_path}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    # トークナイザーの読み込み
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("✅ Tokenizer loaded")
    
    # モデルの読み込み
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    
    # デバイスに移動
    model = model.to(device)
    print(f"✅ Base model loaded and moved to {device}")
    
    # Point backbone configの初期化
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    print("✅ Point backbone config initialized")
    
    # チェックポイントの読み込み（指定されている場合）
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # state_dictの取得
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Point Projector関連のキーのみを更新
        model_state_dict = model.state_dict()
        updated_keys = []
        
        for key, value in state_dict.items():
            if "point_proj" in key:
                if key in model_state_dict:
                    model_state_dict[key] = value.to(device)
                    updated_keys.append(key)
                else:
                    # キー名が完全一致しない場合の処理
                    for model_key in model_state_dict.keys():
                        if "point_proj" in model_key and key.endswith(model_key.split(".")[-1]):
                            model_state_dict[model_key] = value.to(device)
                            updated_keys.append(model_key)
                            break
        
        model.load_state_dict(model_state_dict, strict=False)
        print(f"✅ Loaded {len(updated_keys)} Point Projector parameters")
        
        if len(updated_keys) > 0:
            print("\nLoaded parameters:")
            for key in updated_keys[:5]:  # 最初の5つのみ表示
                print(f"  - {key}")
            if len(updated_keys) > 5:
                print(f"  ... and {len(updated_keys) - 5} more")
    else:
        if checkpoint_path:
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
        print("Using base model weights only")
    
    # 評価モードに設定
    model.eval()
    
    # モデルの精度を統一
    model = model.to(dtype)
    
    print(f"\n{'='*60}")
    print("Model Ready")
    print(f"{'='*60}\n")
    
    return model, tokenizer


def load_and_process_point_cloud(
    point_cloud_path: str,
    num_points: int = 8192
) -> torch.Tensor:
    """
    点群ファイルを読み込んで前処理します
    
    Args:
        point_cloud_path: 点群ファイルのパス (.npy)
        num_points: ターゲット点数
    
    Returns:
        torch.Tensor: 処理済み点群 (1, num_points, 6)
    """
    print(f"\nLoading point cloud: {point_cloud_path}")
    
    # 点群の読み込み
    point_cloud = np.load(point_cloud_path)
    print(f"Original shape: {point_cloud.shape}")
    
    # 6次元未満の場合はパディング
    if point_cloud.shape[1] < 6:
        padding = np.ones((point_cloud.shape[0], 6 - point_cloud.shape[1])) * 0.5
        point_cloud = np.concatenate([point_cloud, padding], axis=1)
    
    point_cloud = point_cloud[:, :6]  # 最初の6次元のみ使用
    
    # 点数の調整
    n_points = point_cloud.shape[0]
    if n_points != num_points:
        if n_points > num_points:
            # ダウンサンプリング
            indices = np.random.choice(n_points, num_points, replace=False)
            point_cloud = point_cloud[indices]
        else:
            # アップサンプリング
            indices = np.random.choice(n_points, num_points, replace=True)
            point_cloud = point_cloud[indices]
    
    # 正規化
    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:6]
    
    # 座標の正規化
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
    
    print(f"Processed shape: {point_cloud.shape}")
    
    # Tensorに変換
    point_cloud_tensor = torch.from_numpy(point_cloud).unsqueeze(0)  # (1, N, 6)
    
    return point_cloud_tensor


def generate_response(
    model,
    tokenizer,
    point_cloud: torch.Tensor,
    question: str,
    device: torch.device,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50
) -> str:
    """
    点群と質問から回答を生成します
    
    Args:
        model: PointLLMモデル
        tokenizer: トークナイザー
        point_cloud: 点群テンソル (1, N, 6)
        question: 質問文
        device: デバイス
        max_new_tokens: 最大生成トークン数
        temperature: サンプリング温度
        top_p: nucleus sampling
        top_k: top-k sampling
    
    Returns:
        str: 生成された回答
    """
    # Point backbone configの取得
    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    mm_use_point_start_end = point_backbone_config.get('mm_use_point_start_end', False)
    
    # プロンプトの構築
    if mm_use_point_start_end:
        default_point_start_token = point_backbone_config['default_point_start_token']
        default_point_end_token = point_backbone_config['default_point_end_token']
        prompt = f"{default_point_start_token}{default_point_patch_token * point_token_len}{default_point_end_token}\n{question}"
    else:
        prompt = f"{default_point_patch_token * point_token_len}\n{question}"
    
    # 会話テンプレートの適用
    from pointllm.conversation import conv_templates
    conv = conv_templates["vicuna_v1_1"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    
    print(f"\n{'='*60}")
    print("Generating Response")
    print(f"{'='*60}")
    print(f"Question: {question}")
    print(f"Prompt length: {len(full_prompt)} characters")
    
    # トークン化
    inputs = tokenizer([full_prompt], return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    
    # 点群をデバイスに移動
    point_cloud = point_cloud.to(device)
    
    # 生成
    print("\nGenerating...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            point_clouds=point_cloud,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    
    # デコード
    input_token_len = input_ids.shape[1]
    response = tokenizer.batch_decode(
        output_ids[:, input_token_len:],
        skip_special_tokens=True
    )[0]
    
    # 後処理
    response = response.strip()
    
    # ストップシーケンスの除去
    from pointllm.conversation import SeparatorStyle
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    if response.endswith(stop_str):
        response = response[:-len(stop_str)]
    response = response.strip()
    
    print(f"\n{'='*60}")
    print("Response Generated")
    print(f"{'='*60}\n")
    
    return response


def main():
    parser = argparse.ArgumentParser(description="PointLLM Inference on MPS")
    
    # モデル設定
    parser.add_argument(
        "--model_path",
        type=str,
        default="RunsenXu/PointLLM_7B_v1.2",
        help="ベースモデルのパス"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="ファインチューニングされたPoint Projectorのチェックポイントパス"
    )
    parser.add_argument(
        "--pointllm_path",
        type=str,
        default=None,
        help="PointLLMリポジトリのパス（自動検出されない場合）"
    )
    
    # 推論設定
    parser.add_argument(
        "--point_cloud_path",
        type=str,
        required=True,
        help="点群ファイルのパス (.npy)"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="この3Dオブジェクトは何ですか？詳しく説明してください。",
        help="質問文"
    )
    
    # デバイス設定
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu", "auto"],
        default="auto",
        help="使用するデバイス"
    )
    
    # 生成設定
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_points", type=int, default=8192)
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("PointLLM Inference on MPS/CPU")
    print(f"{'='*60}\n")
    
    # デバイスの設定
    if args.device == "auto":
        device = get_device(verbose=True)
    else:
        device = torch.device(args.device)
        print(f"[INFO] Using specified device: {device}")
    
    # データ型の設定
    dtype = torch.float32  # MPSではfloat32を使用
    
    # PointLLM環境のセットアップ
    if not setup_pointllm_environment(args.pointllm_path):
        print("❌ Failed to setup PointLLM environment")
        return
    
    # モデルの読み込み
    model, tokenizer = load_model_and_checkpoint(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        device=device,
        dtype=dtype
    )
    
    # 点群の読み込みと前処理
    point_cloud = load_and_process_point_cloud(
        point_cloud_path=args.point_cloud_path,
        num_points=args.num_points
    )
    
    # 推論の実行
    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        point_cloud=point_cloud,
        question=args.question,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
    
    # 結果の表示
    print("Question:")
    print(f"  {args.question}")
    print("\nResponse:")
    print(f"  {response}")
    print()


if __name__ == "__main__":
    main()
