"""
trainer_proj_llm_ddp.py - DDP版 Projector + LLM トレーナー
チェックポイント自動削除・Resume機能付き
"""

import os
import time
import math
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from config import FullConfig


class TrainingState:
    """学習状態"""
    def __init__(self):
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0


class ProjectorLLMTrainerDDP:
    """DDP対応 Projector + LLM トレーナー"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_sampler: DistributedSampler,
        config: FullConfig,
        rank: int,
        world_size: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_sampler = train_sampler
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = f"cuda:{rank}"
        
        self.training_config = config.training
        self.output_config = config.output
        
        if self.rank == 0:
            self.output_dir = Path(config.output.get_output_path())
            self.checkpoint_dir = Path(config.output.get_checkpoint_dir())
            self.log_dir = Path(config.output.get_log_dir())
            
            for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
        
        self.total_steps = (
            len(train_loader)
            // self.training_config.gradient_accumulation_steps
            * self.training_config.num_epochs
        )
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler() if self.training_config.fp16 else None
        
        self.state = TrainingState()
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_steps': [],
            'val_steps': [],
            'learning_rates': []
        }
        
        self.wandb_run = None
        if self.rank == 0 and self.output_config.use_wandb:
            self._init_wandb()
        
        if self.rank == 0:
            print(f"\n{'=' * 60}")
            print("DDP Trainer Initialized (Projector + LLM)")
            print(f"{'=' * 60}")
            print(f"World size: {self.world_size}")
            print(f"Batch size per GPU: {self.training_config.batch_size}")
            print(f"Effective batch size: {self.training_config.batch_size * self.world_size * self.training_config.gradient_accumulation_steps}")
            print(f"{'=' * 60}\n")
    
    def _create_optimizer(self) -> AdamW:
        """オプティマイザ作成"""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        return AdamW(
            trainable_params,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _create_scheduler(self) -> Any:
        """スケジューラー作成"""
        warmup_steps = int(self.total_steps * self.training_config.warmup_ratio)
        
        if self.training_config.lr_scheduler_type == "cosine":
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, self.total_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
            return LambdaLR(self.optimizer, lr_lambda)
        
        elif self.training_config.lr_scheduler_type == "linear":
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(0.0, float(self.total_steps - current_step) / float(max(1, self.total_steps - warmup_steps)))
            
            return LambdaLR(self.optimizer, lr_lambda)
        
        else:
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0
            
            return LambdaLR(self.optimizer, lr_lambda)
    
    def _init_wandb(self):
        """WandB初期化"""
        try:
            import wandb
            
            # Resume時はWandBもresume
            resume_mode = "allow" if self.state.epoch > 0 else None
            
            self.wandb_run = wandb.init(
                project=self.output_config.wandb_project,
                entity=self.output_config.wandb_entity,
                name=self.output_config.experiment_name,
                resume=resume_mode,
                config={
                    "model": self.config.model.__dict__,
                    "training": self.config.training.__dict__,
                    "data": self.config.data.__dict__,
                    "world_size": self.world_size,
                }
            )
            print("✅ WandB initialized")
        except Exception as e:
            print(f"⚠️ WandB init failed: {e}")
            self.output_config.use_wandb = False
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """損失計算"""
        point_clouds = batch['point_clouds'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        use_amp = self.training_config.fp16 or self.training_config.bf16
        dtype = torch.float16 if self.training_config.fp16 else torch.bfloat16
        
        if use_amp:
            point_clouds = point_clouds.to(dtype)
        
        with autocast(enabled=use_amp, dtype=dtype if use_amp else torch.float32):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                point_clouds=point_clouds,
                use_cache=False,
            )
            loss = outputs.loss
        
        return loss
    
    def _cleanup_old_checkpoints(self, current_epoch: int):
        """古いチェックポイントを削除
        
        ルール:
        - 4の倍数のエポックは保持
        - それ以外は3エポック後に削除
        - best_modelは常に保持
        """
        if self.rank != 0:
            return
        
        try:
            # 削除対象のエポック = 現在のエポック - 3
            delete_epoch = current_epoch - 3

            # 削除対象が0以下、または10の倍数なら削除しない
            if delete_epoch <= 0 or delete_epoch % 10 == 0:
                return
            
            # 削除対象のチェックポイント名
            checkpoint_name = f"checkpoint-epoch-{delete_epoch}"
            checkpoint_path = self.checkpoint_dir / checkpoint_name
            
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
                print(f"🗑️  Deleted old checkpoint: {checkpoint_name}")
        
        except Exception as e:
            print(f"⚠️ Failed to delete checkpoint: {e}")
    
    def train_epoch(self, epoch: int) -> float:
        """1エポックの学習"""
        self.model.train()
        self.train_sampler.set_epoch(epoch)
        
        total_loss = 0.0
        num_batches = 0
        
        if self.rank == 0:
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.training_config.num_epochs}",
                leave=True
            )
        else:
            progress_bar = self.train_loader
        
        accumulation_steps = self.training_config.gradient_accumulation_steps
        
        for step, batch in enumerate(progress_bar):
            loss = self._compute_loss(batch)
            loss = loss / accumulation_steps
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * accumulation_steps
            num_batches += 1
            
            if (step + 1) % accumulation_steps == 0:
                if self.training_config.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.max_grad_norm
                    )
                
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.state.global_step += 1
                current_lr = self.scheduler.get_last_lr()[0]
                
                if self.rank == 0:
                    self.history['learning_rates'].append(current_lr)
                    
                    if isinstance(progress_bar, tqdm):
                        progress_bar.set_postfix({
                            'loss': f'{loss.item() * accumulation_steps:.4f}',
                            'lr': f'{current_lr:.2e}'
                        })
                    
                    if self.state.global_step % self.training_config.logging_steps == 0:
                        self._log_metrics({
                            'train/loss': loss.item() * accumulation_steps,
                            'train/learning_rate': current_lr,
                            'train/epoch': epoch + (step / len(self.train_loader)),
                        })
                    
                    if self.state.global_step % self.training_config.save_steps == 0:
                        self.save_checkpoint(f"checkpoint-{self.state.global_step}")
                
                if self.training_config.eval_strategy == "steps":
                    if self.state.global_step % self.training_config.eval_steps == 0:
                        val_loss = self.evaluate()
                        
                        if self.rank == 0:
                            self.history['val_loss'].append(val_loss)
                            self.history['val_steps'].append(self.state.global_step)
                            self.plot_loss_curves()
                            
                            if self._check_early_stopping(val_loss):
                                print("Early stopping!")
                                return total_loss / max(num_batches, 1)
                        
                        self.model.train()
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_loss_tensor = torch.tensor([avg_loss], device=self.device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()
        
        if self.rank == 0:
            self.history['train_loss'].append(avg_loss)
            self.history['train_steps'].append(self.state.global_step)
        
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """検証"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        if self.rank == 0:
            progress_bar = tqdm(self.val_loader, desc="Evaluating", leave=False)
        else:
            progress_bar = self.val_loader
        
        for batch in progress_bar:
            loss = self._compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_loss_tensor = torch.tensor([avg_loss], device=self.device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()
        
        if self.rank == 0:
            self._log_metrics({
                'eval/loss': avg_loss,
                'eval/step': self.state.global_step,
            })
            print(f"\n📊 Validation Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """早期停止チェック"""
        threshold = self.training_config.early_stopping_threshold
        
        if val_loss < self.state.best_val_loss - threshold:
            self.state.best_val_loss = val_loss
            self.state.early_stopping_counter = 0
            
            if self.output_config.save_best_only:
                self.save_checkpoint("best_model")
            
            return False
        else:
            self.state.early_stopping_counter += 1
            
            if self.state.early_stopping_counter >= self.training_config.early_stopping_patience:
                return True
            
            return False
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """メトリクスログ"""
        if self.output_config.use_wandb and self.wandb_run is not None:
            import wandb
            wandb.log(metrics, step=self.state.global_step)
    
    def plot_loss_curves(self):
        """学習曲線プロット"""
        try:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            if self.history['train_loss']:
                plt.plot(self.history['train_steps'], self.history['train_loss'],
                        'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
            if self.history['val_loss']:
                plt.plot(self.history['val_steps'], self.history['val_loss'],
                        'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            if self.history['learning_rates']:
                steps = range(len(self.history['learning_rates']))
                plt.plot(steps, self.history['learning_rates'], 'g-', linewidth=2)
            plt.xlabel('Steps')
            plt.ylabel('Learning Rate')
            plt.title('LR Schedule')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_path = self.log_dir / 'loss_curves.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            if self.output_config.use_wandb and self.wandb_run is not None:
                import wandb
                wandb.log({"loss_curves": wandb.Image(str(plot_path))},
                         step=self.state.global_step)
            
        except Exception as e:
            print(f"⚠️ Failed to plot: {e}")
    
    def save_checkpoint(self, name: str):
        """チェックポイント保存（rank 0のみ）- モデル重みのみ"""
        if self.rank != 0:
            return
        
        checkpoint_path = self.checkpoint_dir / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 学習可能なパラメータをすべて保存（Projector + LLM）
        trainable_state_dict = {}
        for n, p in self.model.module.named_parameters():
            if p.requires_grad:
                trainable_state_dict[n] = p.data.cpu().clone()
        
        # Projectorだけ別ファイルでも保存（Stage1との互換性）
        projector_state_dict = {n: p for n, p in trainable_state_dict.items() if 'point_proj' in n}
        torch.save(projector_state_dict, checkpoint_path / "point_proj.pt")
        
        # Projector + LLMすべて保存
        torch.save(trainable_state_dict, checkpoint_path / "proj_llm_weights.pt")
        
        # 2. 軽量な情報のみ保存
        with open(checkpoint_path / "training_info.json", 'w') as f:
            json.dump({
                'epoch': self.state.epoch,
                'global_step': self.state.global_step,
                'best_val_loss': self.state.best_val_loss,
            }, f, indent=2)
        
        # 3. 設定を保存
        self.config.save(str(checkpoint_path / "config.json"))
        
        # 4. 学習履歴を保存
        with open(checkpoint_path / "history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # ファイルサイズを表示
        proj_size = (checkpoint_path / "point_proj.pt").stat().st_size / 1024**2
        trainable_size = (checkpoint_path / "proj_llm_weights.pt").stat().st_size / 1024**2
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        print(f"   point_proj.pt: {proj_size:.1f} MB")
        print(f"   proj_llm_weights.pt: {trainable_size:.1f} MB")


    def load_checkpoint(self, checkpoint_path: str):
        """チェックポイント読み込み（モデル重みのみ）"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            return
        
        if self.rank == 0:
            print(f"\n{'='*60}")
            print("Loading Checkpoint")
            print(f"{'='*60}")
            print(f"Path: {checkpoint_path}")
        
        # モデル重み読み込み（全rank）
        weights_path = checkpoint_path / "proj_llm_weights.pt"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=self.device)
            
            model_state_dict = self.model.module.state_dict()
            loaded_keys = []
            for key, value in state_dict.items():
                if key in model_state_dict:
                    model_state_dict[key] = value
                    loaded_keys.append(key)
            
            self.model.module.load_state_dict(model_state_dict, strict=False)
            
            if self.rank == 0:
                print(f"✅ Model weights loaded: {len(loaded_keys)} parameters")
                print(f"Note: Training will start from scratch (optimizer state not saved)")
                print(f"{'='*60}\n")
        
        # 全rankで同期
        if dist.is_initialized():
            dist.barrier()

    def train(self) -> Dict[str, List[float]]:
        """学習メインループ"""
        if self.rank == 0:
            print("\n" + "=" * 60)
            if self.state.epoch > 0:
                print(f"🔄 Resuming DDP Training from Epoch {self.state.epoch + 1}")
            else:
                print("🚀 Starting DDP Training (Projector + LLM)")
            print("=" * 60)
            print("📋 Checkpoint retention policy:")
            print("   - Epochs divisible by 4: Kept permanently")
            print("   - Other epochs: Deleted after 3 epochs")
            print("   - best_model: Always kept")
            print("=" * 60 + "\n")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.state.epoch, self.training_config.num_epochs):
                self.state.epoch = epoch
                
                train_loss = self.train_epoch(epoch)
                
                if self.training_config.eval_strategy == "epoch":
                    val_loss = self.evaluate()
                    
                    if self.rank == 0:
                        self.history['val_loss'].append(val_loss)
                        self.history['val_steps'].append(self.state.global_step)
                        self.plot_loss_curves()
                        
                        if self._check_early_stopping(val_loss):
                            print("Early stopping!")
                            break
                
                if self.rank == 0:
                    # エポック終了時にチェックポイント保存
                    self.save_checkpoint(f"checkpoint-epoch-{epoch + 1}")
                    print(f"\n📈 Epoch {epoch + 1} done: train_loss={train_loss:.4f}")
                    
                    # 古いチェックポイントを削除
                    self._cleanup_old_checkpoints(epoch + 1)
                    print()
        
        except KeyboardInterrupt:
            if self.rank == 0:
                print("\n⚠️ Interrupted")
                self.save_checkpoint("interrupted")
        
        if self.rank == 0:
            elapsed_time = time.time() - start_time
            print("\n" + "=" * 60)
            print("✅ Training Completed!")
            print(f"   Time: {elapsed_time / 60:.2f} min")
            print(f"   Best val loss: {self.state.best_val_loss:.4f}")
            print("=" * 60 + "\n")
            
            self.plot_loss_curves()
            
            if self.wandb_run is not None:
                import wandb
                wandb.finish()
        
        return self.history


__all__ = ["ProjectorLLMTrainerDDP"]