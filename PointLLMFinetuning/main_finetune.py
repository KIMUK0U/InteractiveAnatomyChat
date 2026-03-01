#!/usr/bin/env python3
"""
main_finetune.py - PointLLM Point Projector Fine-tuning

このスクリプトは main_finetune.ipynb を Python スクリプト化したものです。
Point Projector のみをファインチューニングします。

使用方法:
    python main_finetune.py --gpu 0 --config config.json
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import json
from typing import Optional

# パス設定
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

def load_config(config_path: Optional[str] = None):
    """設定の読み込み"""
    from config import (
        FullConfig,
        create_default_config,
    )
    
    if config_path and os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        config = FullConfig.load(config_path)
    else:
        print("Using default config")
        config = create_default_config()
        
        # デフォルト設定のカスタマイズ
        config.model.use_flash_attention = False
        config.training.num_epochs = 6
        config.training.batch_size = 32
        config.training.gradient_accumulation_steps = 4
        config.training.learning_rate = 3e-3
        config.output.use_wandb = False
        config.output.experiment_name = "projection_finetune_v1"
    
    config.print_summary()
    return config

def prepare_model_and_tokenizer(config, pointllm_path: str, device: str):
    """モデルとトークナイザーの準備"""
    print("\n" + "="*60)
    print("Loading PointLLM Model")
    print("="*60)
    
    from model_utils_proj_only import prepare_model_for_training
    
    model, tokenizer = prepare_model_for_training(
        config=config.model,
        pointllm_path=pointllm_path,
        device=device
    )
    
    print("\n✅ Model and tokenizer prepared successfully")
    return model, tokenizer

def create_data_loaders(config, tokenizer):
    """データローダーの作成"""
    print("\n" + "="*60)
    print("Creating Data Loaders")
    print("="*60)
    
    from dataset import create_dataloaders
    
    train_loader, val_loader = create_dataloaders(
        config=config,
        tokenizer=tokenizer
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    return train_loader, val_loader

def setup_training_components(config, model, num_training_steps: int):
    """学習コンポーネントのセットアップ"""
    print("\n" + "="*60)
    print("Setting up Training Components")
    print("="*60)
    
    from trainer import setup_trainer_components
    
    optimizer, scheduler, scaler = setup_trainer_components(
        model=model,
        config=config.training,
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler, scaler

def run_training(model, tokenizer, train_loader, val_loader, optimizer, scheduler, scaler, config):
    """学習の実行"""
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    from trainer import train
    
    train(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config
    )
    
    print("\n✅ Training completed successfully")

def test_inference(model, tokenizer, config):
    """推論のテスト"""
    print("\n" + "="*60)
    print("Testing Inference")
    print("="*60)
    
    from evaluation import evaluate_model, generate_sample
    
    # サンプル点群を読み込み
    sample_path = Path(config.data.data_root) / "point_clouds"
    sample_files = list(sample_path.glob("*.npy"))
    
    if sample_files:
        point_cloud = np.load(sample_files[0])
        
        # 推論テスト
        question = "この3Dオブジェクトについて説明してください。"
        response = generate_sample(
            model=model,
            tokenizer=tokenizer,
            point_cloud=point_cloud,
            question=question,
            config=config.generation
        )
        
        print(f"\nSample Point Cloud: {sample_files[0].name}")
        print(f"Question: {question}")
        print(f"Response: {response}")
    else:
        print("⚠️ No point cloud files found for testing")

def main():
    parser = argparse.ArgumentParser(
        description="PointLLM Point Projector Fine-tuning"
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
        default="/content/PointLLM",
        help="Path to PointLLM repository"
    )
    parser.add_argument(
        "--project-path",
        type=str,
        default=None,
        help="Path to project directory (default: current directory)"
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
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = f"cuda:{args.gpu}"
    
    print("="*60)
    print("PointLLM Point Projector Fine-tuning")
    print("="*60)
    print(f"GPU Device: {args.gpu}")
    print(f"Config: {args.config or 'default'}")
    
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
    
    # モデルとトークナイザーの準備
    model, tokenizer = prepare_model_and_tokenizer(
        config, 
        args.pointllm_path, 
        device
    )
    
    # データローダーの作成
    train_loader, val_loader = create_data_loaders(config, tokenizer)
    
    # 総ステップ数計算
    num_training_steps = (
        len(train_loader) * config.training.num_epochs 
        // config.training.gradient_accumulation_steps
    )
    
    # 学習コンポーネントのセットアップ
    optimizer, scheduler, scaler = setup_training_components(
        config, 
        model,
        num_training_steps
    )
    
    # 学習実行
    run_training(
        model, 
        tokenizer, 
        train_loader, 
        val_loader,
        optimizer,
        scheduler,
        scaler,
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
