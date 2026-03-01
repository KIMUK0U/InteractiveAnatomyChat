"""
model_utils.py - PointLLM Model Utilities

このモジュールは、PointLLMモデルの読み込みと、
LLM/Encoderの凍結処理を担当します。

PointLLMのアーキテクチャ:
┌─────────────────────────────────────────────────────────┐
│  PointLLM Model                                          │
├─────────────────────────────────────────────────────────┤
│  point_backbone (PointTransformer)  ← 凍結              │
│    └─ 出力: (B, N_patches, 384)                         │
├─────────────────────────────────────────────────────────┤
│  point_proj (Point Projector)       ← 学習対象          │
│    ├─ Linear(384, 4096) + GELU                          │
│    └─ Linear(4096, 4096)                                │
│    └─ 出力: (B, N_patches, 4096)                        │
├─────────────────────────────────────────────────────────┤
│  model (LLaMA)                      ← 凍結              │
│    ├─ embed_tokens                                      │
│    ├─ layers (32 Transformer blocks)                    │
│    ├─ norm                                              │
│    └─ lm_head                                           │
└─────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import sys
import os

from config import ModelConfig


def setup_pointllm_environment(pointllm_path: str = "/content/PointLLM"):
    """
    PointLLMの環境をセットアップします
    
    PointLLMはHugging Faceの標準ライブラリではないため、
    カスタムクラスを登録する必要があります。
    
    Args:
        pointllm_path: PointLLMリポジトリのパス
    """
    # パスを追加
    if pointllm_path not in sys.path:
        sys.path.insert(0, pointllm_path)
    
    # PointLLMのモジュールをインポート
    try:
        from pointllm.model import PointLLMLlamaForCausalLM
        from pointllm.model.pointllm import PointLLMConfig
        
        # AutoConfigに登録
        AutoConfig.register("pointllm", PointLLMConfig)
        AutoModelForCausalLM.register(PointLLMConfig, PointLLMLlamaForCausalLM)
        
        print("✅ PointLLM environment setup complete")
        return True
    except ImportError as e:
        print(f"❌ Failed to setup PointLLM: {e}")
        print("Make sure PointLLM is installed correctly")
        return False


def load_model_and_tokenizer(
    config: ModelConfig,
    device: str = "cuda"
) -> Tuple[Any, Any]:
    """
    PointLLMモデルとトークナイザーを読み込みます
    
    Args:
        config: モデル設定
        device: デバイス ("cuda" or "cpu")
    
    Returns:
        (model, tokenizer) のタプル
    """
    print(f"Loading model: {config.model_name}")
    
    # データ型の設定
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)
    
    # モデルの読み込み設定
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    
    # Flash Attention 2を使用する場合
    if config.use_flash_attention:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using Flash Attention 2")
        except Exception as e:
            print(f"Flash Attention not available: {e}")
    
    # トークナイザーの読み込み
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=False,
        padding_side="right",
    )
    
    # パディングトークンの設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # モデルの読み込み
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs
    )
    
    # デバイスに移動
    model = model.to(device)
    
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"   Dtype: {torch_dtype}")
    print(f"   Device: {device}")
    
    return model, tokenizer


def freeze_model_components(
    model: nn.Module,
    config: ModelConfig
) -> Tuple[List[str], List[str]]:
    """
    モデルのLLMとEncoderを凍結し、Point Projectorのみを学習可能にします
    
    PointLLMの構造:
    - model.point_backbone: PointTransformer (凍結)
    - model.point_proj: Point Projector (学習対象)
    - model.model.*: LLaMA layers (凍結)
    - model.lm_head: LLaMA出力層 (凍結)
    
    Args:
        model: PointLLMモデル
        config: モデル設定
    
    Returns:
        (frozen_params, trainable_params) パラメータ名のリスト
    """
    frozen_params = []
    trainable_params = []
    
    # まず全パラメータを凍結
    for name, param in model.named_parameters():
        param.requires_grad = False
        frozen_params.append(name)
    
    # Point Projectorのパラメータのみを学習可能に
    trainable_module_name = config.trainable_module_name
    
    for name, param in model.named_parameters():
        if trainable_module_name in name:
            param.requires_grad = True
            frozen_params.remove(name)
            trainable_params.append(name)
    
    # 統計を表示
    total_params = sum(p.numel() for p in model.parameters())
    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_total = total_params - trainable_total
    
    print("\n" + "=" * 60)
    print("Parameter Freezing Summary")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Frozen parameters: {frozen_total:,} ({frozen_total/total_params*100:.2f}%)")
    print(f"Trainable parameters: {trainable_total:,} ({trainable_total/total_params*100:.2f}%)")
    print("-" * 60)
    print("Trainable modules:")
    for name in trainable_params:
        param = dict(model.named_parameters())[name]
        print(f"  {name}: {param.numel():,} params")
    print("=" * 60 + "\n")
    
    return frozen_params, trainable_params


def enable_gradient_checkpointing(model: nn.Module):
    """
    勾配チェックポイントを有効化してメモリ使用量を削減します
    
    勾配チェックポイントは、フォワードパス中にactivationを保存せず、
    バックワードパス中に再計算することでメモリを節約します。
    計算時間は約20-30%増加しますが、メモリ使用量を大幅に削減できます。
    
    Args:
        model: PointLLMモデル
    """
    # LLaMAバックボーンの勾配チェックポイント
    if hasattr(model, 'model') and hasattr(model.model, 'gradient_checkpointing_enable'):
        model.model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled for LLM backbone")
    
    # PointTransformerの勾配チェックポイント（サポートされている場合）
    if hasattr(model, 'point_backbone'):
        if hasattr(model.point_backbone, 'gradient_checkpointing_enable'):
            model.point_backbone.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled for Point backbone")


def get_point_projector_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Point Projectorの重みのみを抽出します
    
    学習後のチェックポイント保存時に使用します。
    フルモデルを保存するよりもストレージを大幅に節約できます。
    
    Args:
        model: PointLLMモデル
    
    Returns:
        Point Projectorの重み辞書
    """
    state_dict = {}
    
    for name, param in model.named_parameters():
        if "point_proj" in name:
            state_dict[name] = param.data.clone()
    
    return state_dict


def load_point_projector_weights(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = True
) -> nn.Module:
    """
    保存されたPoint Projectorの重みを読み込みます
    
    Args:
        model: PointLLMモデル
        checkpoint_path: チェックポイントファイルのパス
        strict: 厳密なキーマッチングを行うか
    
    Returns:
        重みを読み込んだモデル
    """
    print(f"Loading Point Projector weights from: {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # state_dictがネストされている場合の処理
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "point_proj" in state_dict:
        state_dict = state_dict["point_proj"]
    
    # モデルの現在のstate_dictを取得
    model_state_dict = model.state_dict()
    
    # Point Projector関連のキーのみを更新
    updated_keys = []
    for key, value in state_dict.items():
        if key in model_state_dict:
            model_state_dict[key] = value
            updated_keys.append(key)
        elif "point_proj" in key:
            # キー名の調整が必要な場合
            for model_key in model_state_dict.keys():
                if "point_proj" in model_key and key.split(".")[-1] == model_key.split(".")[-1]:
                    model_state_dict[model_key] = value
                    updated_keys.append(model_key)
                    break
    
    model.load_state_dict(model_state_dict, strict=False)
    print(f"✅ Loaded {len(updated_keys)} Point Projector parameters")
    
    return model


def print_model_architecture(model: nn.Module, max_depth: int = 2):
    """
    モデルのアーキテクチャを表示します（デバッグ用）
    
    Args:
        model: PointLLMモデル
        max_depth: 表示する最大の深さ
    """
    print("\n" + "=" * 60)
    print("Model Architecture")
    print("=" * 60)
    
    def print_module(module, prefix="", depth=0):
        if depth > max_depth:
            return
        
        for name, child in module.named_children():
            num_params = sum(p.numel() for p in child.parameters())
            trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
            
            status = "🔥" if trainable > 0 else "❄️"
            print(f"{prefix}{status} {name}: {type(child).__name__} ({num_params:,} params, {trainable:,} trainable)")
            
            print_module(child, prefix + "  ", depth + 1)
    
    print_module(model)
    print("=" * 60)
    print("Legend: 🔥 = Trainable, ❄️ = Frozen")
    print("=" * 60 + "\n")


def verify_model_setup(model: nn.Module, config: ModelConfig) -> bool:
    """
    モデルのセットアップが正しく行われたかを検証します
    
    Args:
        model: PointLLMモデル
        config: モデル設定
    
    Returns:
        検証が成功した場合True
    """
    print("\n" + "=" * 60)
    print("Verifying Model Setup")
    print("=" * 60)
    
    issues = []
    
    # 1. Point Projectorのパラメータが存在するか確認
    has_point_proj = any("point_proj" in name for name, _ in model.named_parameters())
    
    if not has_point_proj:
        issues.append("point_proj parameters not found")
    else:
        print("✅ Point Projector found")
    
    # 2. Point Projectorが学習可能か確認
    trainable_proj_params = 0
    for name, param in model.named_parameters():
        if "point_proj" in name and param.requires_grad:
            trainable_proj_params += param.numel()
    
    if trainable_proj_params == 0:
        issues.append("Point Projector has no trainable parameters")
    else:
        print(f"✅ Point Projector trainable params: {trainable_proj_params:,}")
    
    # 3. LLMが凍結されているか確認
    if config.freeze_llm:
        llm_trainable = 0
        for name, param in model.named_parameters():
            if "model." in name and "point" not in name.lower():
                if param.requires_grad:
                    llm_trainable += param.numel()
        
        if llm_trainable > 0:
            issues.append(f"LLM has {llm_trainable:,} trainable parameters (should be 0)")
        else:
            print("✅ LLM is frozen")
    
    # 4. Point Encoderが凍結されているか確認
    if config.freeze_point_encoder:
        encoder_trainable = 0
        for name, param in model.named_parameters():
            if "point_backbone" in name and param.requires_grad:
                encoder_trainable += param.numel()
        
        if encoder_trainable > 0:
            issues.append(f"Point Encoder has {encoder_trainable:,} trainable parameters (should be 0)")
        else:
            print("✅ Point Encoder is frozen")
    
    # 5. 結果の表示
    print("=" * 60)
    if issues:
        print("❌ Verification FAILED:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Verification PASSED")
        return True


def prepare_model_for_training(
    config: ModelConfig,
    pointllm_path: str = "/content/PointLLM",
    device: str = "cuda"
) -> Tuple[nn.Module, Any]:
    """
    学習用にモデルを準備するメイン関数
    
    この関数は以下の処理を行います:
    1. PointLLM環境のセットアップ
    2. モデルとトークナイザーの読み込み
    3. LLMとEncoderの凍結
    4. 勾配チェックポイントの有効化（オプション）
    5. セットアップの検証
    
    Args:
        config: モデル設定
        pointllm_path: PointLLMリポジトリのパス
        device: 使用するデバイス
    
    Returns:
        (model, tokenizer) のタプル
    """
    # 1. 環境セットアップ
    if not setup_pointllm_environment(pointllm_path):
        raise RuntimeError("Failed to setup PointLLM environment")
    
    # 2. モデルとトークナイザーの読み込み
    model, tokenizer = load_model_and_tokenizer(config, device)

    # 2.1. モデル全体を指定された精度に変換
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)
    model = model.to(dtype=torch_dtype)
    print(f"✅ Model converted to {torch_dtype}")
    
    # 2.5. point_backbone_configの設定（必須）
    # PointLLMモデルが期待する特殊トークンのIDを設定
    if not hasattr(model, 'point_backbone_config') or model.point_backbone_config is None:
        model.point_backbone_config = {}
    
    # 特殊トークンのIDを設定
    special_tokens = {
        '<point>': tokenizer.convert_tokens_to_ids('<point>'),
        '<point_start>': tokenizer.convert_tokens_to_ids('<point_start>'),
        '<point_end>': tokenizer.convert_tokens_to_ids('<point_end>'),
        '<point_patch>': tokenizer.convert_tokens_to_ids('<point_patch>'),
    }
    
    for token_name, token_id in special_tokens.items():
        if token_id == tokenizer.unk_token_id:
            # トークンが存在しない場合は追加
            tokenizer.add_tokens([token_name], special_tokens=True)
            token_id = tokenizer.convert_tokens_to_ids(token_name)
            print(f"⚠️ Added special token: {token_name} (ID: {token_id})")
    
    model.point_backbone_config['point_patch_token'] = tokenizer.convert_tokens_to_ids('<point_patch>')
    model.point_backbone_config['point_start_token'] = tokenizer.convert_tokens_to_ids('<point_start>')
    model.point_backbone_config['point_end_token'] = tokenizer.convert_tokens_to_ids('<point_end>')
    
    print(f"✅ Point backbone config initialized:")
    print(f"   point_patch_token: {model.point_backbone_config.get('point_patch_token')}")
    print(f"   point_start_token: {model.point_backbone_config.get('point_start_token')}")
    print(f"   point_end_token: {model.point_backbone_config.get('point_end_token')}")
    
    # 3. パラメータの凍結
    frozen, trainable = freeze_model_components(model, config)
    
    # 4. 勾配チェックポイント
    if config.use_gradient_checkpointing:
        enable_gradient_checkpointing(model)
    
    # 5. 検証
    if not verify_model_setup(model, config):
        raise RuntimeError("Model setup verification failed")

    # 5.5. Point backbone configの初期化（追加）
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    print("✅ Point backbone config initialized with special tokens")

    # 学習モードに設定
    model.train()
    
    return model, tokenizer


# エクスポート
__all__ = [
    "setup_pointllm_environment",
    "load_model_and_tokenizer",
    "freeze_model_components",
    "enable_gradient_checkpointing",
    "get_point_projector_state_dict",
    "load_point_projector_weights",
    "print_model_architecture",
    "verify_model_setup",
    "prepare_model_for_training",
]
