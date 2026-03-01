#!/usr/bin/env python3
"""
proj_tune_and_LoRA.py - PointLLM Point Projector + LoRA Fine-tuning

このスクリプトは proj_tune_and_LoRA.ipynb を Python スクリプト化したものです。
Point Projector と LoRA を使用したファインチューニングを実行します。

使用方法:
    python proj_tune_and_LoRA.py --gpu 0 --config config.json
"""
import os, torch
print("[DEBUG] CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("[DEBUG] torch.cuda.device_count()=", torch.cuda.device_count())
print("[DEBUG] current_device=", torch.cuda.current_device() if torch.cuda.is_available() else None)

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
        config.model.freeze_llm = False # LLMはフリーズしない
        config.model.freeze_point_encoder = True  # Encoderはフリーズ
        config.model.use_flash_attention = True
        config.training.num_epochs = 50
        config.training.batch_size = 48
        config.training.gradient_accumulation_steps = 4
        config.training.learning_rate = 3e-5
        config.data.num_workers = 8
        config.output.use_wandb = False
        config.output.experiment_name = "32_demo_ar_dataset_projection_lora_finetune_v3"
    
    config.print_summary()
    return config

def prepare_model_and_tokenizer(config, pointllm_path: str, device: str):
    """モデルとトークナイザーの準備"""
    print("\n" + "="*60)
    print("Loading PointLLM Model")
    print("="*60)
    
    from model_utils_proj_lora import prepare_model_for_training
    
    model, tokenizer = prepare_model_for_training(
        config=config.model,
        pointllm_path=pointllm_path,
        device=device
    )
    
    print("\n✅ Model and tokenizer prepared successfully")

    from model_utils_proj_lora import prepare_model_for_training, print_model_architecture
    from peft import LoraConfig, get_peft_model, TaskType

    # ✅ 追加: point_backbone_configの初期化
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    print("✅ Point backbone config initialized")
    print(f"   point_patch_token: {model.get_model().point_backbone_config.get('point_patch_token')}")
    print(f"   point_start_token: {model.get_model().point_backbone_config.get('point_start_token')}")
    print(f"   point_end_token: {model.get_model().point_backbone_config.get('point_end_token')}")

    # 1. LoRAの設定（LLM部分の学習設定）
    lora_config = LoraConfig(
        r=8,                                # 低ランク行列のランク
        lora_alpha=32,                      # スケーリング係数
        target_modules=["q_proj", "v_proj"], # Vicuna(Llama)のSelf-Attention層を対象
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM        # 因果言語モデル用
    )

    # 2. モデルにLoRAを適用
    # この操作により、LLM本体の重みが固定され、LoRAアダプタが挿入されます。
    # 同時に、既存のProjection層もデフォルトでフリーズされます。
    model = get_peft_model(model, lora_config)

    # 3. ✅ Projection層を明示的に学習対象（アンフリーズ）に設定
    # PointLLMの内部構造におけるProjector（point_projector）を走査して有効化します。
    for name, param in model.named_parameters():
        if "point_projector" in name:
            param.requires_grad = True

    # 学習可能なパラメータ数を確認
    model.print_trainable_parameters()
    print("✅ LoRA adapters and Projection layer are now trainable.")
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

def setup_training_components(config, model):
    """学習コンポーネントのセットアップ"""
    print("\n" + "="*60)
    print("Setting up Training Components")
    print("="*60)
    
    from trainer_LoRA import setup_trainer_components
    
    optimizer, scheduler, scaler = setup_trainer_components(
        model=model,
        config=config.training,
        num_training_steps=None  # trainer内で計算
    )
    
    return optimizer, scheduler, scaler

def run_training(model, tokenizer, train_loader, val_loader, optimizer, scheduler, scaler, config):
    """学習の実行"""
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    from trainer_LoRA import train
    
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
        description="PointLLM Point Projector + LoRA Fine-tuning"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
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
    print("PointLLM Point Projector + LoRA Fine-tuning")
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
    
    # 学習コンポーネントのセットアップ
    optimizer, scheduler, scaler = setup_training_components(config, model)
    
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
