"""
model_utils_full.py - PointLLM Model Utilities for Full LLM Training

このモジュールは、PointLLMモデルの読み込みと、
Encoderの凍結、Projector + LLM全体の学習設定を担当します。

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
│  model (LLaMA)                      ← 学習対象           │
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
    Setup PointLLM environment and register custom model classes
    
    Args:
        pointllm_path: Path to PointLLM repository
    
    Returns:
        True if setup successful
    """
    if pointllm_path not in sys.path:
        sys.path.insert(0, pointllm_path)
    
    try:
        from pointllm.model import PointLLMLlamaForCausalLM
        from pointllm.model.pointllm import PointLLMConfig
        
        AutoConfig.register("pointllm", PointLLMConfig)
        AutoModelForCausalLM.register(PointLLMConfig, PointLLMLlamaForCausalLM)
        
        print("✅ PointLLM environment setup complete")
        return True
    except ImportError as e:
        print(f"❌ Failed to setup PointLLM: {e}")
        print("Make sure PointLLM is installed correctly")
        return False


def initialize_pointllm_tokenizer(
    model_name: str,
    config: ModelConfig
) -> AutoTokenizer:
    """
    Initialize tokenizer with proper PointLLM special tokens
    
    Args:
        model_name: Model name or path
        config: Model configuration
    
    Returns:
        Initialized tokenizer with special tokens
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    existing_tokens = set(tokenizer.get_vocab().keys())
    point_tokens = ['<point_patch>', '<point_start>', '<point_end>']
    
    tokens_to_add = [token for token in point_tokens if token not in existing_tokens]
    
    if tokens_to_add:
        print(f"⚠️ Adding special tokens: {tokens_to_add}")
        tokenizer.add_tokens(tokens_to_add, special_tokens=True)
    else:
        print("✅ Special tokens already present in tokenizer")
    
    point_patch_id = tokenizer.convert_tokens_to_ids('<point_patch>')
    point_start_id = tokenizer.convert_tokens_to_ids('<point_start>')
    point_end_id = tokenizer.convert_tokens_to_ids('<point_end>')
    
    print(f"Token IDs:")
    print(f"  <point_patch>: {point_patch_id}")
    print(f"  <point_start>: {point_start_id}")
    print(f"  <point_end>: {point_end_id}")
    
    return tokenizer


def load_model_and_tokenizer(
    config: ModelConfig,
    device: str = "cuda"
) -> Tuple[Any, Any]:
    print(f"Loading model: {config.model_name}")
    
    # Blackwellでは float16 よりも bfloat16 が圧倒的に推奨されます
    # 混合精度訓練での数値的安定性と速度が向上します
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    # Blackwell環境なら、設定が何であれ bfloat16 を優先的に検討してください
    torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16) 
    
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "device_map": device, # 明示的なデバイス割り当て
    }
    
    # Flash Attention 2 の設定
    if config.use_flash_attention:
        # Transformers 4.34以降、Blackwell + PyTorch 2.x 環境での標準的な指定方法
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("🚀 Flash Attention 2 enabled for Blackwell GPU")

    tokenizer = initialize_pointllm_tokenizer(config.model_name, config)
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs
    )
    
    # tokenizerのサイズに合わせてEmbedding層をリサイズ（Special Token追加対応）
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"   Dtype: {torch_dtype}")
    return model, tokenizer


def initialize_point_backbone_config(
    model: nn.Module,
    tokenizer: AutoTokenizer
):
    """
    Initialize point_backbone_config for PointLLM model
    
    Args:
        model: PointLLM model
        tokenizer: Tokenizer with special tokens
    """
    if hasattr(model, 'initialize_tokenizer_point_backbone_config_wo_embedding'):
        model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
        print("✅ Point backbone config initialized using built-in method")
        
        if hasattr(model, 'get_model'):
            inner_model = model.get_model()
            if hasattr(inner_model, 'point_backbone_config'):
                config = inner_model.point_backbone_config
                model.point_backbone_config = config
                
                print("\n✅ Point Backbone Config (from inner model):")
                print(f"  point_token_len: {config.get('point_token_len', 'NOT SET')}")
                print(f"  mm_use_point_start_end: {config.get('mm_use_point_start_end', 'NOT SET')}")
                print(f"  point_patch_token ID: {config.get('point_patch_token', 'NOT SET')}")
                print(f"  point_start_token ID: {config.get('point_start_token', 'NOT SET')}")
                print(f"  point_end_token ID: {config.get('point_end_token', 'NOT SET')}")
                return
    
    print("⚠️ Using manual point_backbone_config initialization")
    
    if hasattr(model, 'get_model'):
        inner_model = model.get_model()
        if not hasattr(inner_model, 'point_backbone_config'):
            inner_model.point_backbone_config = {}
        config = inner_model.point_backbone_config
    else:
        if not hasattr(model, 'point_backbone_config'):
            model.point_backbone_config = {}
        config = model.point_backbone_config
    
    config['point_token_len'] = 513
    config['mm_use_point_start_end'] = True
    config['default_point_patch_token'] = '<point_patch>'
    config['default_point_start_token'] = '<point_start>'
    config['default_point_end_token'] = '<point_end>'
    config['point_patch_token'] = tokenizer.convert_tokens_to_ids('<point_patch>')
    config['point_start_token'] = tokenizer.convert_tokens_to_ids('<point_start>')
    config['point_end_token'] = tokenizer.convert_tokens_to_ids('<point_end>')
    
    model.point_backbone_config = config
    
    print("\nPoint Backbone Config:")
    print(f"  point_token_len: {config.get('point_token_len')}")
    print(f"  mm_use_point_start_end: {config.get('mm_use_point_start_end')}")
    print(f"  point_patch_token ID: {config.get('point_patch_token')}")
    print(f"  point_start_token ID: {config.get('point_start_token')}")
    print(f"  point_end_token ID: {config.get('point_end_token')}")


def freeze_model_components(
    model: nn.Module,
    config: ModelConfig
) -> Tuple[List[str], List[str]]:
    """
    Freeze Encoder only, keep Projector + LLM trainable
    
    PointLLM structure:
    - model.point_backbone: PointTransformer (freeze)
    - model.point_proj: Point Projector (trainable)
    - model.model.*: LLaMA layers (trainable)
    - model.lm_head: LLaMA output layer (trainable)
    
    Args:
        model: PointLLM model
        config: Model configuration
    
    Returns:
        (frozen_params, trainable_params) lists of parameter names
    """
    frozen_params = []
    trainable_params = []
    
    # Freeze only the Point Encoder
    for name, param in model.named_parameters():
        if "point_backbone" in name and config.freeze_point_encoder:
            param.requires_grad = False
            frozen_params.append(name)
        else:
            # Keep Projector and LLM trainable
            param.requires_grad = True
            trainable_params.append(name)
    
    # Print statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_total = total_params - trainable_total
    
    encoder_params = sum(p.numel() for n, p in model.named_parameters() if "point_backbone" in n)
    proj_params = sum(p.numel() for n, p in model.named_parameters() if "point_proj" in n and p.requires_grad)
    llm_params = sum(p.numel() for n, p in model.named_parameters() 
                     if ("model." in n or "lm_head" in n) and p.requires_grad and "point" not in n.lower())
    
    print("\n" + "=" * 60)
    print("Parameter Freezing Summary (Full LLM Training)")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Frozen parameters (Encoder): {frozen_total:,} ({frozen_total/total_params*100:.2f}%)")
    print(f"Trainable parameters: {trainable_total:,} ({trainable_total/total_params*100:.2f}%)")
    print("-" * 60)
    print("Component breakdown:")
    print(f"  Encoder (frozen): {encoder_params:,}")
    print(f"  Projector (trainable): {proj_params:,}")
    print(f"  LLM (trainable): {llm_params:,}")
    print("=" * 60 + "\n")
    
    return frozen_params, trainable_params


def enable_gradient_checkpointing(model: nn.Module):
    """
    Enable gradient checkpointing to reduce memory usage
    
    Args:
        model: PointLLM model
    """
    if hasattr(model, 'model') and hasattr(model.model, 'gradient_checkpointing_enable'):
        model.model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled for LLM backbone")
    
    if hasattr(model, 'point_backbone'):
        if hasattr(model.point_backbone, 'gradient_checkpointing_enable'):
            model.point_backbone.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled for Point backbone")


def verify_model_setup(model: nn.Module, config: ModelConfig) -> bool:
    """
    Verify that model is properly configured for full LLM training
    
    Args:
        model: PointLLM model
        config: Model configuration
    
    Returns:
        True if verification passes
    """
    print("\n" + "=" * 60)
    print("Verifying Model Setup (Full LLM Training)")
    print("=" * 60)
    
    issues = []
    
    # 1. Check Point Projector exists and is trainable
    trainable_proj_params = 0
    for name, param in model.named_parameters():
        if "point_proj" in name and param.requires_grad:
            trainable_proj_params += param.numel()
    
    if trainable_proj_params == 0:
        issues.append("Point Projector has no trainable parameters")
    else:
        print(f"✅ Point Projector trainable params: {trainable_proj_params:,}")
    
    # 2. Check LLM is trainable
    llm_trainable = 0
    for name, param in model.named_parameters():
        if ("model." in name or "lm_head" in name) and "point" not in name.lower():
            if param.requires_grad:
                llm_trainable += param.numel()
    
    if llm_trainable == 0:
        issues.append("LLM has no trainable parameters (should be trainable)")
    else:
        print(f"✅ LLM trainable params: {llm_trainable:,}")
    
    # 3. Check Point Encoder is frozen
    if config.freeze_point_encoder:
        encoder_trainable = 0
        for name, param in model.named_parameters():
            if "point_backbone" in name and param.requires_grad:
                encoder_trainable += param.numel()
        
        if encoder_trainable > 0:
            issues.append(f"Point Encoder has {encoder_trainable:,} trainable parameters (should be 0)")
        else:
            print("✅ Point Encoder is frozen")
    
    # 4. Check point_backbone_config
    config_obj = None
    if hasattr(model, 'point_backbone_config'):
        config_obj = model.point_backbone_config
    elif hasattr(model, 'get_model'):
        inner_model = model.get_model()
        if hasattr(inner_model, 'point_backbone_config'):
            config_obj = inner_model.point_backbone_config
    
    if config_obj is None:
        issues.append("point_backbone_config not found")
    else:
        required_keys = ['point_token_len', 'point_patch_token', 'point_start_token', 'point_end_token']
        missing_keys = [key for key in required_keys if key not in config_obj]
        
        if missing_keys:
            issues.append(f"point_backbone_config missing keys: {missing_keys}")
        else:
            print("✅ point_backbone_config properly initialized")
            print(f"   point_token_len: {config_obj.get('point_token_len')}")
    
    # Print results
    print("=" * 60)
    if issues:
        print("❌ Verification FAILED:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Verification PASSED - Ready for Full LLM Training")
        return True


def prepare_model_for_training(
    config: ModelConfig,
    pointllm_path: str = "/content/PointLLM",
    device: str = "cuda"
) -> Tuple[nn.Module, Any]:
    """
    Main function to prepare model for full LLM training
    
    This function performs:
    1. Setup PointLLM environment
    2. Load model and tokenizer
    3. Initialize point_backbone_config with special tokens
    4. Freeze Encoder only (keep Projector + LLM trainable)
    5. Enable gradient checkpointing (optional)
    6. Verify setup
    
    Args:
        config: Model configuration
        pointllm_path: Path to PointLLM repository
        device: Device to use
    
    Returns:
        (model, tokenizer) tuple
    """
    if not setup_pointllm_environment(pointllm_path):
        raise RuntimeError("Failed to setup PointLLM environment")
    
    model, tokenizer = load_model_and_tokenizer(config, device)
    
    initialize_point_backbone_config(model, tokenizer)
    
    frozen, trainable = freeze_model_components(model, config)
    
    if config.use_gradient_checkpointing:
        enable_gradient_checkpointing(model)
    
    if not verify_model_setup(model, config):
        raise RuntimeError("Model setup verification failed")
    # prepare_model_for_training 内の最後の方に追加
    if hasattr(torch, "compile"):
        print("Compiling model for Blackwell architecture...")
        model = torch.compile(model)
    model.train()
    
    return model, tokenizer


__all__ = [
    "setup_pointllm_environment",
    "initialize_pointllm_tokenizer",
    "load_model_and_tokenizer",
    "initialize_point_backbone_config",
    "freeze_model_components",
    "enable_gradient_checkpointing",
    "verify_model_setup",
    "prepare_model_for_training",
]