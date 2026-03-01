#!/usr/bin/env python3
"""
encoder_proj_llm_finetune_ddp.py - Multi-GPU Projector + LLM Fine-tuning
高速化版
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Blackwell最適化
torch.set_float32_matmul_precision('high')

# Graph break対策
os.environ['TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS'] = '1'


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
            print("Using default config for Projector + LLM training")
        config = create_default_config()
        
        config.model.freeze_llm = False
        config.model.freeze_point_encoder = True
        config.model.use_flash_attention = True
        config.training.num_epochs = 100
        config.training.batch_size = 16
        config.training.gradient_accumulation_steps = 2
        config.training.learning_rate = 1e-5
        config.output.use_wandb = False
        config.output.experiment_name = "encoder_proj_llm_ddp_finetune_100prompt_32_ar_LowLR1e-5_v6_EPHighLR5e-4_100epoch"
    
    if rank == 0:
        config.print_summary()
    
    return config


def load_pretrained_weights(model, pretrained_path: str, device: str, rank: int):
    """事前学習済みEncoder + Projectorの重みをロード"""
    if rank == 0:
        print(f"\n{'='*60}")
        print("Loading Pretrained Encoder + Projector Weights")
        print(f"{'='*60}")
        print(f"From: {pretrained_path}")
    
    pretrained_path = Path(pretrained_path)
    model_ckpt_path = pretrained_path / "encoder_projector.pt"
    
    if not model_ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_ckpt_path}")
    
    if rank == 0:
        print(f"Loading weights from {model_ckpt_path}")
    
    state_dict = torch.load(model_ckpt_path, map_location=device)
    
    if isinstance(model, DDP):
        missing_keys, unexpected_keys = model.module.load_state_dict(state_dict, strict=False)
    else:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if rank == 0:
        print(f"✅ Pretrained weights loaded")
        print(f"   Missing keys: {len(missing_keys)}")
        print(f"   Unexpected keys: {len(unexpected_keys)}")
        
        encoder_params = sum(1 for key in state_dict.keys() if 'point_backbone' in key)
        proj_params = sum(1 for key in state_dict.keys() if 'point_proj' in key)
        
        print(f"   Encoder parameters loaded: {encoder_params}")
        print(f"   Projector parameters loaded: {proj_params}")


def prepare_model_and_tokenizer(config, pointllm_path: str, local_rank: int, rank: int, 
                                pretrained_path: str = None, enable_compile: bool = True):
    """モデルとトークナイザーの準備（DDP対応）"""
    if rank == 0:
        print("\n" + "="*60)
        print("Loading PointLLM Model (DDP - Projector + LLM Training)")
        print("="*60)
    
    from model_utils_proj_llm import prepare_model_for_training
    
    device = f"cuda:{local_rank}"
    model, tokenizer = prepare_model_for_training(
        config=config.model,
        pointllm_path=pointllm_path,
        device=device
    )
    
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    
    if pretrained_path:
        load_pretrained_weights(model, pretrained_path, device, rank)
    
    # Encoderを凍結
    for name, param in model.named_parameters():
        if "point_backbone" in name:
            param.requires_grad = False
    
    # Projector + LLMを学習可能に
    for name, param in model.named_parameters():
        if "point_proj" in name or "model." in name or "lm_head" in name:
            if "point_backbone" not in name:
                param.requires_grad = True
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        encoder_trainable = sum(p.numel() for n, p in model.named_parameters() 
                               if p.requires_grad and 'point_backbone' in n)
        proj_trainable = sum(p.numel() for n, p in model.named_parameters() 
                            if p.requires_grad and 'point_proj' in n)
        llm_trainable = sum(p.numel() for n, p in model.named_parameters() 
                           if p.requires_grad and ('model.' in n or 'lm_head' in n) and 'point' not in n.lower())
        
        print("\n" + "="*60)
        print("Trainable Parameters (DDP)")
        print("="*60)
        print(f"Encoder (frozen): {encoder_trainable:,}")
        print(f"Projector: {proj_trainable:,}")
        print(f"LLM: {llm_trainable:,}")
        print(f"Total trainable: {trainable_params:,}")
        print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        print("="*60 + "\n")
    
    # DDPでラップ
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,  # メモリ効率化
    )
    
    # torch.compile適用（オプション）
    if enable_compile and hasattr(torch, "compile"):
        if rank == 0:
            print("🚀 Applying torch.compile()...")
        try:
            # より緩いモード
            model = torch.compile(
                model, 
                backend="inductor",
                mode="reduce-overhead",  # max-autotune から変更
                fullgraph=False,  # graph breakを許容
            )
            if rank == 0:
                print("✅ Model compiled (reduce-overhead mode)")
        except Exception as e:
            if rank == 0:
                print(f"⚠️ torch.compile failed: {e}")
                print("Continuing without compilation...")
    
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
    
    processor = PointCloudProcessor(config)
    
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
    
    collator = DataCollator(
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else 0
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collator,
        persistent_workers=True if config.data.num_workers > 0 else False,  # 高速化
        prefetch_factor=2 if config.data.num_workers > 0 else None,  # 先読み
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collator,
        persistent_workers=True if config.data.num_workers > 0 else False,
        prefetch_factor=2 if config.data.num_workers > 0 else None,
    )
    
    if rank == 0:
        print(f"Train batches per GPU: {len(train_loader)}")
        print(f"Val batches per GPU: {len(val_loader)}")
    
    return train_loader, val_loader, train_sampler


def run_training(model, tokenizer, train_loader, val_loader, train_sampler, config, rank, resume_path=None):
    """DDP対応の学習実行"""
    if rank == 0:
        print("\n" + "="*60)
        print("Starting Training (DDP - Projector + LLM)")
        print("="*60)
    
    from trainer_proj_llm_ddp import ProjectorLLMTrainerDDP
    
    trainer = ProjectorLLMTrainerDDP(
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
        print("\n✅ Training completed")
    
    return history


def main():
    parser = argparse.ArgumentParser(
        description="PointLLM Projector + LLM Fine-tuning (DDP)"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--pointllm-path", type=str, required=True)
    parser.add_argument("--project-path", type=str, default=None)
    parser.add_argument("--pretrained-path", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--resume-path", type=str, default="")
    parser.add_argument("--no-compile", action="store_true",
                       help="Disable torch.compile (use if graph breaks are problematic)")
    
    args = parser.parse_args()
    
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print("="*60)
        print("PointLLM DDP Training (Projector + LLM)")
        print("="*60)
        print(f"World size: {world_size}")
        print(f"Pretrained path: {args.pretrained_path}")
        print(f"Compile enabled: {not args.no_compile}")
    
    project_path = args.project_path or os.getcwd()
    setup_paths(args.pointllm_path, project_path)
    
    config = load_config(args.config, rank)
    
    if args.data_root:
        config.data.data_root = args.data_root
    
    if rank != 0:
        config.output.use_wandb = False
    
    model, tokenizer = prepare_model_and_tokenizer(
        config,
        args.pointllm_path,
        local_rank,
        rank,
        pretrained_path=args.pretrained_path,
        enable_compile=not args.no_compile
    )
    
    train_loader, val_loader, train_sampler = create_data_loaders(
        config,
        tokenizer,
        rank,
        world_size
    )
    
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
    
    cleanup_distributed()
    
    if rank == 0:
        print("\n" + "="*60)
        print("✅ All processes completed!")
        print("="*60)


if __name__ == "__main__":
    main()