"""
model_utils_proj_llm.py - Projector + LLM Training用モデルユーティリティ

Encoder凍結、Projector + LLMを学習
"""

import torch
import torch.nn as nn
from typing import Tuple, Any, List
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import sys

from config import ModelConfig


def setup_pointllm_environment(pointllm_path: str):
    """PointLLM環境セットアップ"""
    if pointllm_path not in sys.path:
        sys.path.insert(0, pointllm_path)
    
    try:
        from pointllm.model import PointLLMLlamaForCausalLM
        from pointllm.model.pointllm import PointLLMConfig
        
        AutoConfig.register("pointllm", PointLLMConfig)
        AutoModelForCausalLM.register(PointLLMConfig, PointLLMLlamaForCausalLM)
        
        print("✅ PointLLM environment setup")
        return True
    except ImportError as e:
        print(f"❌ Failed: {e}")
        return False


def initialize_pointllm_tokenizer(model_name: str, config: ModelConfig) -> AutoTokenizer:
    """トークナイザー初期化"""
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
    
    tokens_to_add = [t for t in point_tokens if t not in existing_tokens]
    
    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add, special_tokens=True)
    
    return tokenizer


def load_model_and_tokenizer(config: ModelConfig, device: str) -> Tuple[Any, Any]:
    """モデルとトークナイザーのロード"""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    
    # Flash Attention使用時はfloat32を避ける
    if config.use_flash_attention and config.torch_dtype == "float32":
        print("⚠️ FA2 requires fp16/bf16. Switching to bfloat16")
        config.torch_dtype = "bfloat16"
    
    torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)
    
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    
    if config.use_flash_attention:
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print(f"🚀 Flash Attention 2 (dtype: {torch_dtype})")
        except ImportError:
            print("⚠️ flash_attn not found")
    
    tokenizer = initialize_pointllm_tokenizer(config.model_name, config)
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs
    )
    
    model = model.to(device)
    
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"   Dtype: {torch_dtype}")
    
    return model, tokenizer


def initialize_point_backbone_config(model: nn.Module, tokenizer: AutoTokenizer):
    """point_backbone_config初期化"""
    if hasattr(model, 'initialize_tokenizer_point_backbone_config_wo_embedding'):
        model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
        print("✅ Point backbone config initialized")
        return
    
    print("⚠️ Manual initialization")
    
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
    config['point_patch_token'] = tokenizer.convert_tokens_to_ids('<point_patch>')
    config['point_start_token'] = tokenizer.convert_tokens_to_ids('<point_start>')
    config['point_end_token'] = tokenizer.convert_tokens_to_ids('<point_end>')
    
    model.point_backbone_config = config


def freeze_model_components(model: nn.Module, config: ModelConfig) -> Tuple[List[str], List[str]]:
    """Encoder凍結、Projector + LLMを学習可能に"""
    frozen_params = []
    trainable_params = []
    
    for name, param in model.named_parameters():
        if "point_backbone" in name and config.freeze_point_encoder:
            param.requires_grad = False
            frozen_params.append(name)
        else:
            param.requires_grad = True
            trainable_params.append(name)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_total = total_params - trainable_total
    
    print("\n" + "=" * 60)
    print("Parameter Freezing (Projector + LLM)")
    print("=" * 60)
    print(f"Total: {total_params:,}")
    print(f"Frozen (Encoder): {frozen_total:,} ({frozen_total/total_params*100:.2f}%)")
    print(f"Trainable: {trainable_total:,} ({trainable_total/total_params*100:.2f}%)")
    print("=" * 60 + "\n")
    
    return frozen_params, trainable_params


def enable_gradient_checkpointing(model: nn.Module):
    """勾配チェックポイント有効化"""
    if hasattr(model, 'model') and hasattr(model.model, 'gradient_checkpointing_enable'):
        model.model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing (LLM)")
    
    if hasattr(model, 'point_backbone'):
        if hasattr(model.point_backbone, 'gradient_checkpointing_enable'):
            model.point_backbone.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing (Encoder)")


def prepare_model_for_training(
    config: ModelConfig,
    pointllm_path: str,
    device: str
) -> Tuple[nn.Module, Any]:
    """モデル準備（Projector + LLM学習用）"""
    if not setup_pointllm_environment(pointllm_path):
        raise RuntimeError("Failed to setup PointLLM")
    
    model, tokenizer = load_model_and_tokenizer(config, device)
    
    initialize_point_backbone_config(model, tokenizer)
    
    freeze_model_components(model, config)
    
    if config.use_gradient_checkpointing:
        enable_gradient_checkpointing(model)
    
    model.train()
    
    return model, tokenizer


__all__ = [
    "setup_pointllm_environment",
    "initialize_pointllm_tokenizer",
    "load_model_and_tokenizer",
    "initialize_point_backbone_config",
    "freeze_model_components",
    "enable_gradient_checkpointing",
    "prepare_model_for_training",
]