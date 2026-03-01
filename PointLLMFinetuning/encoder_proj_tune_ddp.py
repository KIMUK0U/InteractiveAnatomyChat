#!/usr/bin/env python3
"""
encoder_proj_tune_ddp.py - Multi-GPU Training with DistributedDataParallel

使用方法:
    torchrun --nproc_per_node=2 encoder_proj_tune_ddp.py --pointllm-path /path/to/PointLLM
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    """分散学習環境のセットアップ"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed mode")
        return -1, 1, 0
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """分散学習環境のクリーンアップ"""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_paths(pointllm_path: str, project_path: str):
    """パスの設定"""
    for path in [pointllm_path, project_path]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    os.chdir(project_path)
    if dist.get_rank() == 0:
        print(f"Working directory: {os.getcwd()}")


def load_config(config_path: str = None, rank: int = 0):
    """設定の読み込み"""
    from config import FullConfig, create_default_config
    
    if config_path and os.path.exists(config_path):
        if rank == 0:
            print(f"Loading config from: {config_path}")
        config = FullConfig.load(config_path)
    else:
        if rank == 0:
            print("Using default config for Encoder + Projector training")
        config = create_default_config()
        
        # Encoder + Projector学習用のデフォルト設定
        config.model.freeze_llm = True
        config.model.freeze_point_encoder = False
        config.model.use_flash_attention = True
        config.training.num_epochs = 20
        config.training.batch_size = 32  # per GPU
        config.training.gradient_accumulation_steps = 8  # 実効: 16*2*4=128
        config.training.learning_rate = 5e-4
        config.output.use_wandb = False  # rank 0のみで有効化
        config.output.experiment_name = "encoder_proj_tune_ddp_100prompt_ar_32_highLR_5e-4"
    
    if rank == 0:
        config.print_summary()
    
    return config


def prepare_model_and_tokenizer(config, pointllm_path: str, local_rank: int, rank: int):
    """モデルとトークナイザーの準備（DDP対応）"""
    if rank == 0:
        print("\n" + "="*60)
        print("Loading PointLLM Model (DDP Mode)")
        print("="*60)
    
    from model_utils_encoder_proj import prepare_model_for_training
    
    # 各プロセスで独立にモデルをロード
    device = f"cuda:{local_rank}"
    model, tokenizer = prepare_model_for_training(
        config=config.model,
        pointllm_path=pointllm_path,
        device=device
    )
    
    # point_backbone_configの初期化
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    
    # DDPでラップ
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,  # 高速化のためFalse
        broadcast_buffers=False,  # BatchNormがない場合False
    )
    
    if rank == 0:
        print(f"\n✅ Model wrapped with DDP (world_size={dist.get_world_size()})")
    
    return model, tokenizer


def create_data_loaders(config, tokenizer, rank: int, world_size: int):
    """データローダーの作成（DDP対応）"""
    if rank == 0:
        print("\n" + "="*60)
        print("Creating Data Loaders (DDP)")
        print("="*60)
    
    from dataset import PointCloudProcessor, PointLLMDataset, DataCollator
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    
    # プロセッサー
    processor = PointCloudProcessor(config)
    
    # データセット
    train_dataset = PointLLMDataset(
        annotation_path=config.data.get_train_annotation_path(),
        data_root=config.data.data_root,
        processor=processor,
        tokenizer=tokenizer,
        training=True,
    )
    
    val_dataset = PointLLMDataset(
        annotation_path=config.data.get_val_annotation_path(),
        data_root=config.data.data_root,
        processor=processor,
        tokenizer=tokenizer,
        training=False,
    )
    
    # DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    
    # コレーター
    collator = DataCollator(
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else 0
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collator,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collator,
    )
    
    if rank == 0:
        print(f"Train batches per GPU: {len(train_loader)}")
        print(f"Val batches per GPU: {len(val_loader)}")
        print(f"Total train batches: {len(train_loader) * world_size}")
    
    return train_loader, val_loader, train_sampler


def run_training(model, tokenizer, train_loader, val_loader, train_sampler, config, rank, resume_path=None):
    """DDP対応の学習実行"""
    if rank == 0:
        print("\n" + "="*60)
        print("Starting Training (DDP - Encoder + Projector)")
        print("="*60)
    
    from trainer_encoder_proj_ddp import EncoderProjectorTrainerDDP
    
    trainer = EncoderProjectorTrainerDDP(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        config=config,
        rank=rank,
        world_size=dist.get_world_size(),
    )
    
    if resume_path:
        trainer.load_checkpoint(resume_path)
    
    history = trainer.train()
    
    if rank == 0:
        print("\n✅ Training completed successfully")
    
    return history


def main():
    parser = argparse.ArgumentParser(
        description="PointLLM Encoder + Projector Fine-tuning (DDP)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file"
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
        help="Path to project directory"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root directory"
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default="",
        help="Path to checkpoint directory to resume from"
    )
    
    args = parser.parse_args()
    
    # 分散環境のセットアップ
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print("="*60)
        print("PointLLM DDP Training")
        print("="*60)
        print(f"World size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Local rank: {local_rank}")
    
    # プロジェクトパスの設定
    project_path = args.project_path or os.getcwd()
    
    # パス設定
    setup_paths(args.pointllm_path, project_path)
    
    # 設定読み込み
    config = load_config(args.config, rank)
    
    # データルートの上書き
    if args.data_root:
        config.data.data_root = args.data_root
    
    # WandBはrank 0のみ
    if rank != 0:
        config.output.use_wandb = False
    
    # モデルとトークナイザーの準備
    model, tokenizer = prepare_model_and_tokenizer(
        config,
        args.pointllm_path,
        local_rank,
        rank
    )
    
    # データローダーの作成
    train_loader, val_loader, train_sampler = create_data_loaders(
        config,
        tokenizer,
        rank,
        world_size
    )
    
    # 学習実行
    resume_path = args.resume_path if args.resume_path else None
    if resume_path and not Path(resume_path).exists():
        if rank == 0:
            print(f"⚠️ Resume path does not exist: {resume_path}")
        resume_path = None
    
    history = run_training(
        model,
        tokenizer,
        train_loader,
        val_loader,
        train_sampler,
        config,
        rank,
        resume_path=resume_path
    )
    
    # クリーンアップ
    cleanup_distributed()
    
    if rank == 0:
        print("\n" + "="*60)
        print("✅ All processes completed successfully!")
        print("="*60)


if __name__ == "__main__":
    main()