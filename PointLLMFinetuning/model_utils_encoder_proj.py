"""
model_utils_encoder_proj.py - PointLLM Model Utilities for Encoder + Projector Training

このモジュールは、PointLLMモデルの読み込みと、
Point EncoderとProjection層のみを学習対象とする設定を行います。

PointLLMのアーキテクチャ:
┌─────────────────────────────────────────────────────────┐
│  PointLLM Model                                          │
├─────────────────────────────────────────────────────────┤
│  point_backbone (PointTransformer)  ← 学習対象 (NEW!)   │
│    └─ 出力: (B, N_patches, 384)                         │
├─────────────────────────────────────────────────────────┤
│  point_proj (Point Projector)       ← 学習対象          │
│    ├─ Linear(384, 4096) + GELU                          │
│    └─ Linear(4096, 4096)                                │
│    └─ 出力: (B, N_patches, 4096)                        │
├─────────────────────────────────────────────────────────┤
│  model (LLaMA)                      ← 凍結 (変更!)      │
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
    # Add to path
    if pointllm_path not in sys.path:
        sys.path.insert(0, pointllm_path)
    
    # Import PointLLM modules
    try:
        from pointllm.model import PointLLMLlamaForCausalLM
        from pointllm.model.pointllm import PointLLMConfig
        
        # Register with AutoConfig
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
    
    This function adds the three special tokens required by PointLLM:
    - <point_patch>: Repeated 513 times to represent point features
    - <point_start>: Marks beginning of point sequence
    - <point_end>: Marks end of point sequence
    
    Args:
        model_name: Model name or path
        config: Model configuration
    
    Returns:
        Initialized tokenizer with special tokens
    """
    # Load base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        padding_side="right",
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Check if special tokens already exist (for pre-trained PointLLM models)
    existing_tokens = set(tokenizer.get_vocab().keys())
    point_tokens = ['<point_patch>', '<point_start>', '<point_end>']
    
    tokens_to_add = [token for token in point_tokens if token not in existing_tokens]
    
    if tokens_to_add:
        print(f"⚠️ Adding special tokens: {tokens_to_add}")
        tokenizer.add_tokens(tokens_to_add, special_tokens=True)
    else:
        print("✅ Special tokens already present in tokenizer")
    
    # Verify token IDs
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
    """
    Load PointLLM model and tokenizer with proper initialization
    
    Args:
        config: Model configuration
        device: Device to load model on
    
    Returns:
        (model, tokenizer) tuple
    """
    print(f"Loading model: {config.model_name}")
    
    # Dtype mapping
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    # ▼▼▼ 【修正】 Flash Attention 使用時のデータ型強制 ▼▼▼
    if config.use_flash_attention:
        # FA2は float32 では動作しないため、bfloat16 (推奨) か float16 に強制
        if config.torch_dtype == "float32":
            print("⚠️ Flash Attention 2 requires float16 or bfloat16. Switching to bfloat16.")
            config.torch_dtype = "bfloat16"
    torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)
    
    # Model loading configuration
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    
    # Flash Attention 2 if requested
    if config.use_flash_attention:
        try:
            # flash_attn ライブラリの存在確認
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print(f"🚀 Using Flash Attention 2 (dtype: {torch_dtype})")
        except ImportError:
            print("⚠️ flash_attn package not found. Falling back to default attention.")
            config.use_flash_attention = False
        except Exception as e:
            print(f"⚠️ Flash Attention not available: {e}")
    
    # Initialize tokenizer with special tokens
    tokenizer = initialize_pointllm_tokenizer(config.model_name, config)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs
    )
    
    # Move to device
    model = model.to(device)
    
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"   Dtype: {torch_dtype}")
    print(f"   Device: {device}")
    
    return model, tokenizer


def initialize_point_backbone_config(
    model: nn.Module,
    tokenizer: AutoTokenizer
):
    """
    Initialize point_backbone_config for PointLLM model
    
    This function properly sets up the point token configuration that
    PointLLM uses during forward pass to identify and process point tokens.
    
    Args:
        model: PointLLM model
        tokenizer: Tokenizer with special tokens
    """
    # Use PointLLM's built-in initialization method
    if hasattr(model, 'initialize_tokenizer_point_backbone_config_wo_embedding'):
        model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
        print("✅ Point backbone config initialized using built-in method")
        
        # Access the config from the nested model structure
        if hasattr(model, 'get_model'):
            inner_model = model.get_model()
            if hasattr(inner_model, 'point_backbone_config'):
                config = inner_model.point_backbone_config
                
                # Store reference at top level for easier access
                model.point_backbone_config = config
                
                print("\n✅ Point Backbone Config (from inner model):")
                print(f"  point_token_len: {config.get('point_token_len', 'NOT SET')}")
                print(f"  mm_use_point_start_end: {config.get('mm_use_point_start_end', 'NOT SET')}")
                print(f"  point_patch_token ID: {config.get('point_patch_token', 'NOT SET')}")
                print(f"  point_start_token ID: {config.get('point_start_token', 'NOT SET')}")
                print(f"  point_end_token ID: {config.get('point_end_token', 'NOT SET')}")
                return
    
    # Fallback: Manual initialization
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
    
    # Set point token configuration
    config['point_token_len'] = 513
    config['mm_use_point_start_end'] = True
    
    # Set default token strings
    config['default_point_patch_token'] = '<point_patch>'
    config['default_point_start_token'] = '<point_start>'
    config['default_point_end_token'] = '<point_end>'
    
    # Set token IDs
    config['point_patch_token'] = tokenizer.convert_tokens_to_ids('<point_patch>')
    config['point_start_token'] = tokenizer.convert_tokens_to_ids('<point_start>')
    config['point_end_token'] = tokenizer.convert_tokens_to_ids('<point_end>')
    
    # Store reference at top level
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
    Freeze LLM, keep Point Encoder and Point Projector trainable
    
    PointLLM structure:
    - model.point_backbone: PointTransformer (trainable) ← NEW!
    - model.point_proj: Point Projector (trainable)
    - model.model.*: LLaMA layers (freeze) ← CHANGED!
    - model.lm_head: LLaMA output layer (freeze)
    
    Args:
        model: PointLLM model
        config: Model configuration
    
    Returns:
        (frozen_params, trainable_params) lists of parameter names
    """
    frozen_params = []
    trainable_params = []
    
    # First, freeze all parameters
    for name, param in model.named_parameters():
        param.requires_grad = False
        frozen_params.append(name)
    
    # Enable Point Encoder and Point Projector parameters
    trainable_modules = ['point_backbone', 'point_proj']
    
    for name, param in model.named_parameters():
        for module_name in trainable_modules:
            if module_name in name:
                param.requires_grad = True
                frozen_params.remove(name)
                trainable_params.append(name)
                break
    
    # Print statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_total = total_params - trainable_total
    
    print("\n" + "=" * 60)
    print("Parameter Freezing Summary (Encoder + Projector)")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Frozen parameters: {frozen_total:,} ({frozen_total/total_params*100:.2f}%)")
    print(f"Trainable parameters: {trainable_total:,} ({trainable_total/total_params*100:.2f}%)")
    print("-" * 60)
    print("Trainable modules:")
    
    # Count parameters by module
    encoder_params = sum(p.numel() for n, p in model.named_parameters() 
                        if 'point_backbone' in n and p.requires_grad)
    projector_params = sum(p.numel() for n, p in model.named_parameters() 
                          if 'point_proj' in n and p.requires_grad)
    
    print(f"  Point Encoder: {encoder_params:,} params")
    print(f"  Point Projector: {projector_params:,} params")
    print("=" * 60 + "\n")
    
    return frozen_params, trainable_params


def enable_gradient_checkpointing(model: nn.Module):
    """
    Enable gradient checkpointing to reduce memory usage
    
    Args:
        model: PointLLM model
    """
    # LLM backbone gradient checkpointing (凍結されているが、メモリ削減のため有効化)
    if hasattr(model, 'model') and hasattr(model.model, 'gradient_checkpointing_enable'):
        model.model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled for LLM backbone")
    
    # Point Transformer gradient checkpointing
    if hasattr(model, 'point_backbone'):
        if hasattr(model.point_backbone, 'gradient_checkpointing_enable'):
            model.point_backbone.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled for Point Encoder")


def get_encoder_projector_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract Point Encoder and Point Projector weights
    
    Args:
        model: PointLLM model
    
    Returns:
        Dictionary of Encoder + Projector weights
    """
    state_dict = {}
    
    for name, param in model.named_parameters():
        if "point_backbone" in name or "point_proj" in name:
            state_dict[name] = param.data.clone()
    
    return state_dict


def load_encoder_projector_weights(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False
) -> nn.Module:
    """
    Load saved Point Encoder and Point Projector weights
    
    Args:
        model: PointLLM model
        checkpoint_path: Path to checkpoint file
        strict: Whether to enforce strict key matching
    
    Returns:
        Model with loaded weights
    """
    print(f"Loading Encoder + Projector weights from: {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle nested state_dict
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    # Get model's current state_dict
    model_state_dict = model.state_dict()
    
    # Update only Encoder and Projector keys
    updated_keys = []
    for key, value in state_dict.items():
        if key in model_state_dict:
            if "point_backbone" in key or "point_proj" in key:
                model_state_dict[key] = value
                updated_keys.append(key)
    
    model.load_state_dict(model_state_dict, strict=strict)
    print(f"✅ Loaded {len(updated_keys)} parameters")
    
    return model


def verify_model_setup(model: nn.Module, config: ModelConfig) -> bool:
    """
    Verify that model is properly configured for training
    
    Args:
        model: PointLLM model
        config: Model configuration
    
    Returns:
        True if verification passes
    """
    print("\n" + "=" * 60)
    print("Verifying Model Setup (Encoder + Projector Training)")
    print("=" * 60)
    
    issues = []
    
    # 1. Check Point Encoder exists and is trainable
    encoder_trainable = 0
    for name, param in model.named_parameters():
        if "point_backbone" in name and param.requires_grad:
            encoder_trainable += param.numel()
    
    if encoder_trainable == 0:
        issues.append("Point Encoder has no trainable parameters")
    else:
        print(f"✅ Point Encoder trainable params: {encoder_trainable:,}")
    
    # 2. Check Point Projector is trainable
    trainable_proj_params = 0
    for name, param in model.named_parameters():
        if "point_proj" in name and param.requires_grad:
            trainable_proj_params += param.numel()
    
    if trainable_proj_params == 0:
        issues.append("Point Projector has no trainable parameters")
    else:
        print(f"✅ Point Projector trainable params: {trainable_proj_params:,}")
    
    # 3. Check LLM is frozen
    llm_trainable = 0
    for name, param in model.named_parameters():
        if "model." in name and "point" not in name.lower():
            if param.requires_grad:
                llm_trainable += param.numel()
    
    if llm_trainable > 0:
        issues.append(f"LLM has {llm_trainable:,} trainable parameters (should be 0)")
    else:
        print("✅ LLM is frozen")
    
    # 4. Check point_backbone_config is properly set
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
        print("✅ Verification PASSED")
        return True


def prepare_model_for_training(
    config: ModelConfig,
    pointllm_path: str = "/content/PointLLM",
    device: str = "cuda"
) -> Tuple[nn.Module, Any]:
    """
    Main function to prepare model for Encoder + Projector training
    
    This function performs:
    1. Setup PointLLM environment
    2. Load model and tokenizer
    3. Initialize point_backbone_config with special tokens
    4. Freeze LLM only (keep Encoder and Projector trainable)
    5. Enable gradient checkpointing (optional)
    6. Verify setup
    
    Args:
        config: Model configuration
        pointllm_path: Path to PointLLM repository
        device: Device to use
    
    Returns:
        (model, tokenizer) tuple
    """
    # 1. Environment setup
    if not setup_pointllm_environment(pointllm_path):
        raise RuntimeError("Failed to setup PointLLM environment")
    
    # 2. Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config, device)
    
    # 3. Initialize point_backbone_config
    initialize_point_backbone_config(model, tokenizer)
    
    # 4. Freeze parameters (LLM only)
    frozen, trainable = freeze_model_components(model, config)
    
    # 5. Gradient checkpointing
    if config.use_gradient_checkpointing:
        enable_gradient_checkpointing(model)
    
    # 6. Verify setup
    if not verify_model_setup(model, config):
        raise RuntimeError("Model setup verification failed")
    
    # Set to training mode
    model.train()
    
    return model, tokenizer


# Exports
__all__ = [
    "setup_pointllm_environment",
    "initialize_pointllm_tokenizer",
    "load_model_and_tokenizer",
    "initialize_point_backbone_config",
    "freeze_model_components",
    "enable_gradient_checkpointing",
    "get_encoder_projector_state_dict",
    "load_encoder_projector_weights",
    "verify_model_setup",
    "prepare_model_for_training",
]