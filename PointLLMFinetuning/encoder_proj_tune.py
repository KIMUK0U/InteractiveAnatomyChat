#!/usr/bin/env python3
"""
encoder_proj_tune.py - PointLLM Point Encoder + Projector Fine-tuning

このスクリプトはPoint EncoderとProjection層を学習するファインチューニングを実行します。
LLMは完全に凍結されます。

使用方法:
    python encoder_proj_tune.py --gpu 1 --config config.json --pointllm-path /path/to/PointLLM
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
        print("Using default config for Encoder + Projector training")
        config = create_default_config()
        
        # Encoder + Projector学習用のデフォルト設定
        config.model.freeze_llm = True
        config.model.freeze_point_encoder = False  # Encoderを学習
        config.model.use_flash_attention = True
        config.training.num_epochs = 20
        config.training.batch_size =32
        config.training.gradient_accumulation_steps = 4
        config.training.learning_rate = 5e-4  # Encoderも学習するので低めに
        config.output.use_wandb = False
        config.output.experiment_name = "encoder_proj_tune_v9_demo_dental_model_head_coord_32_final"
    
    config.print_summary()
    return config

def load_checkpoint(model, trainer, resume_path: str):
    """チェックポイントから重みと学習状態をロード"""
    print(f"\nLoading checkpoint from: {resume_path}")
    resume_path = Path(resume_path)
    
    # 1. モデルの重み (Encoder + Projector) のロード
    model_ckpt_path = resume_path / "encoder_projector.pt"
    if model_ckpt_path.exists():
        print(f"Loading model weights from {model_ckpt_path}")
        state_dict = torch.load(model_ckpt_path, map_location="cpu")
        
        # PointLLM全体に対してロードするため strict=False (LLM部分は含まれていないため)
        keys = model.load_state_dict(state_dict, strict=False)
        print(f"✅ Model weights loaded.")
        print(f"   Missing keys: {len(keys.missing_keys)} (expected for frozen LLM)")
        print(f"   Unexpected keys: {len(keys.unexpected_keys)}")
    else:
        print(f"⚠️ Model checkpoint not found at {model_ckpt_path}")

    # 2. トレーナー状態 (Optimizer, Scheduler, Epoch) のロード
    # Note: EncoderProjectorTrainerの実装依存ですが、一般的なPyTorch実装を想定しています
    trainer_ckpt_path = resume_path / "trainer_state.pt"
    start_epoch = 0
    
    if trainer_ckpt_path.exists():
        print(f"Loading trainer state from {trainer_ckpt_path}")
        state = torch.load(trainer_ckpt_path, map_location="cpu")
        
        # Optimizerのロード
        if "optimizer" in state and hasattr(trainer, "optimizer"):
            trainer.optimizer.load_state_dict(state["optimizer"])
            print("✅ Optimizer state loaded")
            
        # Schedulerのロード
        if "scheduler" in state and hasattr(trainer, "scheduler"):
            trainer.scheduler.load_state_dict(state["scheduler"])
            print("✅ Scheduler state loaded")
            
        # Epoch/Stepの復元
        if "epoch" in state:
            start_epoch = state["epoch"] + 1  # 次のエポックから開始
            print(f"✅ Resuming from epoch {start_epoch}")
            
        # 必要に応じて trainer 内部変数を更新
        if hasattr(trainer, "start_epoch"):
            trainer.start_epoch = start_epoch
            
    else:
        print(f"⚠️ Trainer state not found at {trainer_ckpt_path}")
        
    return start_epoch

def prepare_model_and_tokenizer(config, pointllm_path: str, device: str):
    """モデルとトークナイザーの準備"""
    print("\n" + "="*60)
    print("Loading PointLLM Model (Encoder + Projector Training)")
    print("="*60)
    
    from model_utils_encoder_proj import prepare_model_for_training
    
    model, tokenizer = prepare_model_for_training(
        config=config.model,
        pointllm_path=pointllm_path,
        device=device
    )
    
    # point_backbone_configの初期化を確認
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    
    print("✅ Point backbone config initialized")
    print(f"   point_patch_token: {model.get_model().point_backbone_config.get('point_patch_token')}")
    print(f"   point_start_token: {model.get_model().point_backbone_config.get('point_start_token')}")
    print(f"   point_end_token: {model.get_model().point_backbone_config.get('point_end_token')}")
    
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
        tokenizer=tokenizer,
        train_batch_size=config.training.batch_size,
        val_batch_size=config.training.batch_size
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def run_training(model, tokenizer, train_loader, val_loader, config, resume_path=None):
    """学習の実行（修正版）"""
    print("\n" + "="*60)
    print("Starting Training (Encoder + Projector)")
    print("="*60)
    
    from trainer_encoder_proj import EncoderProjectorTrainer
    
    trainer = EncoderProjectorTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # ★追加: チェックポイントのロード処理
    if resume_path:
        start_epoch = load_checkpoint(model, trainer, resume_path)
        
        # トレーナーが start_epoch 引数を受け取れない場合は、
        # trainer.train() の引数や内部変数を調整する必要があります。
        # ここでは一般的な実装として trainer.start_epoch を書き換えるか
        # train() メソッドに引数を渡すことを想定しています。
        
        # 例: もし trainer.train() が start_epoch を受け取るなら:
        # history = trainer.train(start_epoch=start_epoch)
        
        # trainer_encoder_proj.py の実装に合わせて調整してください。
        # 以下は trainer 内部の変数を書き換えたと仮定した場合:
        if hasattr(trainer, 'current_epoch'):
            trainer.current_epoch = start_epoch - 1
    
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
        description="PointLLM Point Encoder + Projector Fine-tuning"
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
    # ★修正: デフォルトを空文字列に変更
    parser.add_argument(
        "--resume-path",
        type=str,
        default="",
        help="Path to checkpoint directory to resume from (containing .pt files)"
    )
    
    args = parser.parse_args()
    
    # GPU設定
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = "cuda:0"  # CUDA_VISIBLE_DEVICESで絞った後なので常に0
    
    print("="*60)
    print("PointLLM Point Encoder + Projector Fine-tuning")
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
    
    # ★修正: チェックポイントの有無によるエラー処理を追加
    if args.resume_path:
        # チェックポイントの存在確認
        resume_path = Path(args.resume_path)
        if not resume_path.exists():
            print(f"\n⚠️ ERROR: Resume path does not exist: {args.resume_path}")
            print("Please provide a valid checkpoint directory or omit --resume-path for new training")
            sys.exit(1)
        
        # 必要なファイルの確認
        model_ckpt = resume_path / "encoder_projector.pt"
        trainer_ckpt = resume_path / "trainer_state.pt"
        
        if not model_ckpt.exists() and not trainer_ckpt.exists():
            print(f"\n⚠️ ERROR: No checkpoint files found in {args.resume_path}")
            print("Expected files: encoder_projector.pt and/or trainer_state.pt")
            print("Please check the checkpoint directory or omit --resume-path for new training")
            sys.exit(1)
        
        print(f"\n✅ Resuming from checkpoint: {args.resume_path}")
        history = run_training(
            model, 
            tokenizer, 
            train_loader, 
            val_loader,
            config,
            resume_path=args.resume_path
        )
    else:
        print("\n✅ Starting new fine-tuning (no checkpoint resume)")
        history = run_training(
            model, 
            tokenizer, 
            train_loader, 
            val_loader,
            config,
            resume_path=None
        )
    
    # 推論テスト
    if not args.skip_inference_test:
        test_inference(model, tokenizer, config)
    
    print("\n" + "="*60)
    print("✅ All processes completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()