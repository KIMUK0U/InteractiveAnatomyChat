#!/usr/bin/env python3
"""
encoder_proj_LoRA_finetune.py - PointLLM Encoder + Projector + LoRA Fine-tuning

このスクリプトは事前学習されたEncoder + Projectorの重みをロードし、
Encoderを凍結した上でProjection層とLoRAを使用したファインチューニングを実行します。

使用方法:
    python encoder_proj_LoRA_finetune.py --gpu 0 --pretrained-path /path/to/encoder_projector_weights
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np


def setup_paths(pointllm_path: str, project_path: str):
    """パスの設定とインポート準備"""
    for path in [pointllm_path, project_path]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    os.chdir(project_path)
    print(f"Working directory: {os.getcwd()}")


def check_gpu():
    """GPU環境の確認"""
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"CUDA memory: {mem_gb:.2f} GB")
    else:
        raise RuntimeError("CUDA is not available")


def load_config(config_path: str = None):
    """設定の読み込み"""
    from config import FullConfig, create_default_config
    
    if config_path and os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        config = FullConfig.load(config_path)
    else:
        print("Using default config for Projection + LoRA training")
        config = create_default_config()
        
        # Projection + LoRA学習用のデフォルト設定
        config.model.freeze_llm = False  # LoRAで学習するのでFalse
        config.model.freeze_point_encoder = True  # Encoderは凍結
        config.model.use_flash_attention = True
        config.training.num_epochs = 20
        config.training.batch_size = 32
        config.training.gradient_accumulation_steps = 4
        config.training.learning_rate = 1e-5
        config.output.use_wandb = False
        config.output.experiment_name = "encoder_proj_lora_finetune_v6_demo_ar_32_dental_model"
    
    config.print_summary()
    return config


def load_pretrained_weights(model, pretrained_path: str, device: str):
    """
    事前学習されたEncoder + Projectorの重みをロード
    
    Args:
        model: PointLLMモデル（LoRA適用前）
        pretrained_path: チェックポイントディレクトリのパス
        device: デバイス
    """
    print(f"\n{'='*60}")
    print("Loading Pretrained Encoder + Projector Weights")
    print(f"{'='*60}")
    print(f"From: {pretrained_path}")
    
    pretrained_path = Path(pretrained_path)
    
    # Encoder + Projectorの重みファイル
    model_ckpt_path = pretrained_path / "encoder_projector.pt"
    
    if not model_ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_ckpt_path}")
    
    print(f"Loading model weights from {model_ckpt_path}")
    state_dict = torch.load(model_ckpt_path, map_location=device)
    
    # モデルに重みをロード（strict=Falseでエラー回避）
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    print(f"✅ Pretrained weights loaded successfully")
    print(f"   Missing keys: {len(missing_keys)} (expected for LLM部分)")
    print(f"   Unexpected keys: {len(unexpected_keys)}")
    
    # Encoder部分が正しくロードされたか確認
    encoder_params_loaded = sum(1 for key in state_dict.keys() if 'point_backbone' in key)
    proj_params_loaded = sum(1 for key in state_dict.keys() if 'point_proj' in key)
    
    print(f"   Encoder parameters loaded: {encoder_params_loaded}")
    print(f"   Projector parameters loaded: {proj_params_loaded}")
    
    if encoder_params_loaded == 0:
        print("⚠️ Warning: No encoder parameters found in checkpoint")
    if proj_params_loaded == 0:
        print("⚠️ Warning: No projector parameters found in checkpoint")


def prepare_model_and_tokenizer(config, pointllm_path: str, device: str, pretrained_path: str = None):
    """モデルとトークナイザーの準備"""
    print("\n" + "="*60)
    print("Loading PointLLM Model with LoRA")
    print("="*60)
    
    from model_utils_proj_lora import prepare_model_for_training
    from peft import LoraConfig, get_peft_model, TaskType
    
    # 1. ベースモデルをロード（LoRA適用前）
    model, tokenizer = prepare_model_for_training(
        config=config.model,
        pointllm_path=pointllm_path,
        device=device
    )
    
    # 2. point_backbone_configの初期化
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    
    print("✅ Point backbone config initialized")
    print(f"   point_patch_token: {model.get_model().point_backbone_config.get('point_patch_token')}")
    print(f"   point_start_token: {model.get_model().point_backbone_config.get('point_start_token')}")
    print(f"   point_end_token: {model.get_model().point_backbone_config.get('point_end_token')}")
    
    # 3. 事前学習済みEncoder + Projectorの重みをロード
    if pretrained_path:
        load_pretrained_weights(model, pretrained_path, device)
    else:
        print("⚠️ No pretrained weights provided, starting from scratch")
    
    # 4. LoRAの設定（LLM部分の学習設定）
    lora_config = LoraConfig(
        r=8,                                # 低ランク行列のランク
        lora_alpha=32,                      # スケーリング係数
        target_modules=["q_proj", "v_proj"], # Vicuna(Llama)のSelf-Attention層を対象
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM        # 因果言語モデル用
    )
    
    # 5. モデルにLoRAを適用
    model = get_peft_model(model, lora_config)
    
    # 6. Encoderを凍結（事前学習済みの重みを保持）
    for name, param in model.named_parameters():
        if "point_backbone" in name:
            param.requires_grad = False
    
    # 7. Projection層を明示的に学習対象に設定
    for name, param in model.named_parameters():
        if "point_proj" in name:
            param.requires_grad = True
    
    # 学習可能なパラメータ数を確認
    model.print_trainable_parameters()
    
    # 詳細な内訳を表示
    print("\n" + "="*60)
    print("Trainable Parameters Breakdown")
    print("="*60)
    
    encoder_trainable = sum(p.numel() for n, p in model.named_parameters() 
                           if p.requires_grad and 'point_backbone' in n)
    proj_trainable = sum(p.numel() for n, p in model.named_parameters() 
                        if p.requires_grad and 'point_proj' in n)
    lora_trainable = sum(p.numel() for n, p in model.named_parameters() 
                        if p.requires_grad and 'lora' in n)
    
    print(f"Encoder (should be 0): {encoder_trainable:,}")
    print(f"Projector: {proj_trainable:,}")
    print(f"LoRA adapters: {lora_trainable:,}")
    print(f"Total trainable: {encoder_trainable + proj_trainable + lora_trainable:,}")
    print("="*60 + "\n")
    
    print("✅ LoRA adapters and Projection layer are now trainable")
    print("✅ Encoder is frozen with pretrained weights")
    
    return model, tokenizer


def create_data_loaders(config, tokenizer):
    """データローダーの作成"""
    print("\n" + "="*60)
    print("Creating Data Loaders")
    print("="*60)
    
    from dataset import create_dataloaders
    
    train_loader, val_loader = create_dataloaders(
        config=config,
        tokenizer=tokenizer,
        train_batch_size=config.training.batch_size,
        val_batch_size=config.training.batch_size
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def run_training(model, tokenizer, train_loader, val_loader, config):
    """学習の実行"""
    print("\n" + "="*60)
    print("Starting Training (Projector + LoRA)")
    print("="*60)
    
    from trainer_LoRA import Trainer
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    history = trainer.train()
    
    print("\n✅ Training completed successfully")
    return history


def test_inference(model, tokenizer, config):
    """推論のテスト"""
    print("\n" + "="*60)
    print("Testing Inference")
    print("="*60)
    
    from evaluation import quick_inference
    
    # サンプル点群を読み込み
    sample_path = Path(config.data.data_root) / "point_clouds"
    sample_files = list(sample_path.glob("*.npy"))
    
    if sample_files:
        point_cloud_path = str(sample_files[0])
        
        # 推論テスト
        question = "What is this? And, does this object have highlight region? Where?"
        response = quick_inference(
            model=model,
            tokenizer=tokenizer,
            point_cloud_path=point_cloud_path,
            question=question,
            config=config,
            device="cuda"
        )
        
        print(f"\nSample Point Cloud: {sample_files[0].name}")
        print(f"Question: {question}")
        print(f"Response: {response}")
    else:
        print("⚠️ No point cloud files found for testing")


def main():
    parser = argparse.ArgumentParser(
        description="PointLLM Encoder + Projector + LoRA Fine-tuning"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID to use (default: 0)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file (optional)"
    )
    parser.add_argument(
        "--pointllm-path",
        type=str,
        required=True,
        help="Path to PointLLM repository"
    )
    parser.add_argument(
        "--project-path",
        type=str,
        default=None,
        help="Path to project directory (default: current directory)"
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        required=True,
        help="Path to pretrained encoder+projector checkpoint directory (e.g., outputs/.../best_model)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root directory"
    )
    parser.add_argument(
        "--skip-inference-test",
        action="store_true",
        help="Skip inference testing after training"
    )
    
    args = parser.parse_args()
    
    # GPU設定
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = "cuda:0"  # CUDA_VISIBLE_DEVICESで絞った後なので常に0
    
    print("="*60)
    print("PointLLM Encoder + Projector + LoRA Fine-tuning")
    print("="*60)
    print(f"GPU Device: {args.gpu}")
    print(f"Config: {args.config or 'default'}")
    print(f"Pretrained Path: {args.pretrained_path}")
    
    # プロジェクトパスの設定
    project_path = args.project_path or os.getcwd()
    
    # パス設定
    setup_paths(args.pointllm_path, project_path)
    
    # GPU確認
    check_gpu()
    
    # 設定読み込み
    config = load_config(args.config)
    
    # データルートの上書き
    if args.data_root:
        config.data.data_root = args.data_root
        print(f"Data root overridden to: {args.data_root}")
    
    # モデルとトークナイザーの準備（事前学習済み重みのロードを含む）
    model, tokenizer = prepare_model_and_tokenizer(
        config, 
        args.pointllm_path, 
        device,
        pretrained_path=args.pretrained_path
    )
    
    # データローダーの作成
    train_loader, val_loader = create_data_loaders(config, tokenizer)
    
    # 学習実行
    history = run_training(
        model, 
        tokenizer, 
        train_loader, 
        val_loader,
        config
    )
    
    # 推論テスト
    if not args.skip_inference_test:
        test_inference(model, tokenizer, config)
    
    print("\n" + "="*60)
    print("✅ All processes completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()