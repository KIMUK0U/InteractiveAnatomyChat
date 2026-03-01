"""
trainer_encoder_proj_ddp.py - DDP版トレーナー

DistributedDataParallelに対応したトレーナークラス
"""

import os
import time
import math
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
from trainer_encoder_proj import TrainingState


class EncoderProjectorTrainerDDP:
    """DDP対応トレーナー"""
    
    def __init__(
        self,
        model: nn.Module,  # DDPでラップ済み
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
        
        # 出力ディレクトリ（rank 0のみ作成）
        if self.rank == 0:
            self.output_dir = Path(config.output.get_output_path())
            self.checkpoint_dir = Path(config.output.get_checkpoint_dir())
            self.log_dir = Path(config.output.get_log_dir())
            
            for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # 総ステップ数（全GPUでの合計を各GPUのステップ数に変換）
        self.total_steps = (
            len(train_loader)
            // self.training_config.gradient_accumulation_steps
            * self.training_config.num_epochs
        )
        
        # オプティマイザ（DDP内のモデルに対して）
        self.optimizer = self._create_optimizer()
        
        # スケジューラー
        self.scheduler = self._create_scheduler()
        
        # スケーラー
        self.scaler = GradScaler() if self.training_config.fp16 else None
        
        # 学習状態
        self.state = TrainingState()
        
        # 学習履歴
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_steps': [],
            'val_steps': [],
            'learning_rates': []
        }
        
        # WandB（rank 0のみ）
        self.wandb_run = None
        if self.rank == 0 and self.output_config.use_wandb:
            self._init_wandb()
        
        if self.rank == 0:
            print(f"\n{'=' * 60}")
            print("DDP Trainer Initialized")
            print(f"{'=' * 60}")
            print(f"World size: {self.world_size}")
            print(f"Total epochs: {self.training_config.num_epochs}")
            print(f"Batch size per GPU: {self.training_config.batch_size}")
            print(f"Effective batch size: {self.training_config.batch_size * self.world_size * self.training_config.gradient_accumulation_steps}")
            print(f"Total steps per GPU: {self.total_steps}")
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
            
            self.wandb_run = wandb.init(
                project=self.output_config.wandb_project,
                entity=self.output_config.wandb_entity,
                name=self.output_config.experiment_name,
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
    
    def train_epoch(self, epoch: int) -> float:
        """1エポックの学習"""
        self.model.train()
        
        # エポックごとにサンプラーのシャッフルをリセット
        self.train_sampler.set_epoch(epoch)
        
        total_loss = 0.0
        num_batches = 0
        
        # プログレスバーはrank 0のみ
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
                # 勾配クリッピング
                if self.training_config.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.max_grad_norm
                    )
                
                # パラメータ更新
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
                    
                    # ログ
                    if self.state.global_step % self.training_config.logging_steps == 0:
                        self._log_metrics({
                            'train/loss': loss.item() * accumulation_steps,
                            'train/learning_rate': current_lr,
                            'train/epoch': epoch + (step / len(self.train_loader)),
                        })
                    
                    # チェックポイント
                    if self.state.global_step % self.training_config.save_steps == 0:
                        self.save_checkpoint(f"checkpoint-{self.state.global_step}")
                
                # 評価（全ランクで同期して実行）
                if self.training_config.eval_strategy == "steps":
                    if self.state.global_step % self.training_config.eval_steps == 0:
                        val_loss = self.evaluate()
                        
                        if self.rank == 0:
                            self.history['val_loss'].append(val_loss)
                            self.history['val_steps'].append(self.state.global_step)
                            self.plot_loss_curves()
                            
                            if self._check_early_stopping(val_loss):
                                print("Early stopping triggered!")
                                return total_loss / max(num_batches, 1)
                        
                        self.model.train()
        
        # 全GPUの平均損失を計算
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
        
        # 全GPUの平均
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
            plt.xlabel('Global Steps', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            if self.history['learning_rates']:
                steps = range(len(self.history['learning_rates']))
                plt.plot(steps, self.history['learning_rates'],
                        'g-', label='Learning Rate', linewidth=2)
            plt.xlabel('Steps', fontsize=12)
            plt.ylabel('Learning Rate', fontsize=12)
            plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
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
        """チェックポイント保存（rank 0のみ）"""
        if self.rank != 0:
            return
        
        checkpoint_path = self.checkpoint_dir / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # DDP内のモデルにアクセス
        from model_utils_encoder_proj import get_encoder_projector_state_dict
        encoder_proj_state_dict = get_encoder_projector_state_dict(self.model.module)
        
        torch.save(
            encoder_proj_state_dict,
            checkpoint_path / "encoder_projector.pt"
        )
        
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if self.scaler else None,
            'state': self.state.__dict__,
            'history': self.history,
        }, checkpoint_path / "trainer_state.pt")
        
        self.config.save(str(checkpoint_path / "config.json"))
        
        with open(checkpoint_path / "history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """チェックポイント読み込み"""
        checkpoint_path = Path(checkpoint_path)
        
        # モデルの重み
        encoder_proj_path = checkpoint_path / "encoder_projector.pt"
        if encoder_proj_path.exists():
            state_dict = torch.load(encoder_proj_path, map_location=self.device)
            
            # DDP内のモデルにロード
            model_state_dict = self.model.module.state_dict()
            for key, value in state_dict.items():
                if key in model_state_dict:
                    model_state_dict[key] = value
            self.model.module.load_state_dict(model_state_dict, strict=False)
            
            if self.rank == 0:
                print("✅ Model weights loaded")
        
        # トレーナー状態（rank 0のみ）
        if self.rank == 0:
            trainer_state_path = checkpoint_path / "trainer_state.pt"
            if trainer_state_path.exists():
                state = torch.load(trainer_state_path, map_location=self.device)
                
                self.optimizer.load_state_dict(state['optimizer'])
                self.scheduler.load_state_dict(state['scheduler'])
                
                if self.scaler and state.get('scaler'):
                    self.scaler.load_state_dict(state['scaler'])
                
                for key, value in state['state'].items():
                    setattr(self.state, key, value)
                
                if 'history' in state:
                    self.history = state['history']
                
                print(f"✅ Loaded checkpoint: {checkpoint_path}")
    
    def train(self) -> Dict[str, List[float]]:
        """学習メインループ"""
        if self.rank == 0:
            print("\n" + "=" * 60)
            print("🚀 Starting DDP Training")
            print("=" * 60 + "\n")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.state.epoch, self.training_config.num_epochs):
                self.state.epoch = epoch
                
                train_loss = self.train_epoch(epoch)
                
                # エポック評価
                if self.training_config.eval_strategy == "epoch":
                    val_loss = self.evaluate()
                    
                    if self.rank == 0:
                        self.history['val_loss'].append(val_loss)
                        self.history['val_steps'].append(self.state.global_step)
                        self.plot_loss_curves()
                        
                        if self._check_early_stopping(val_loss):
                            print("Early stopping!")
                            break
                
                # エポック終了チェックポイント
                if self.rank == 0:
                    self.save_checkpoint(f"checkpoint-epoch-{epoch + 1}")
                    print(f"\n📈 Epoch {epoch + 1} completed: train_loss={train_loss:.4f}\n")
        
        except KeyboardInterrupt:
            if self.rank == 0:
                print("\n⚠️ Interrupted")
                self.save_checkpoint("interrupted")
        
        # 終了
        if self.rank == 0:
            elapsed_time = time.time() - start_time
            print("\n" + "=" * 60)
            print("✅ Training Completed!")
            print(f"   Total time: {elapsed_time / 60:.2f} minutes")
            print(f"   Best validation loss: {self.state.best_val_loss:.4f}")
            print("=" * 60 + "\n")
            
            self.plot_loss_curves()
            
            if self.wandb_run is not None:
                import wandb
                wandb.finish()
        
        return self.history


__all__ = ["EncoderProjectorTrainerDDP"]