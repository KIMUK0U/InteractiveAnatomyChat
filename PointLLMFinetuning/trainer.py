"""
trainer.py - PointLLM Point Projector Training Loop (Enhanced)

このモジュールは、Point Projectorのファインチューニングのための
学習ループを実装しています。

主な機能:
- 混合精度学習（FP16/BF16）
- 勾配累積
- 学習率スケジューラー
- チェックポイント保存
- WandBへのログ記録
- 早期停止
- リアルタイムな検証ロスの計算と可視化
- 学習曲線のプロット保存
"""

import os
import time
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from config import FullConfig, TrainingConfig, OutputConfig


@dataclass
class TrainingState:
    """
    学習状態を保持するクラス
    
    チェックポイントからの再開時に使用します。
    """
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    early_stopping_counter: int = 0


class Trainer:
    """
    Point Projectorのファインチューニングを行うトレーナークラス
    
    PointLLMのPoint Projector部分のみを学習し、
    LLM（LLaMA）とPoint Encoder（PointTransformer）は凍結されています。
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: FullConfig,
        device: str = "cuda"
    ):
        """
        トレーナーを初期化します
        
        Args:
            model: PointLLMモデル（Point Projectorのみ学習可能）
            tokenizer: トークナイザー
            train_loader: 学習用DataLoader
            val_loader: 検証用DataLoader
            config: 全体設定
            device: 使用するデバイス
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # 学習設定の展開
        self.training_config = config.training
        self.output_config = config.output
        
        # 出力ディレクトリの作成
        self.output_dir = Path(config.output.get_output_path())
        self.checkpoint_dir = Path(config.output.get_checkpoint_dir())
        self.log_dir = Path(config.output.get_log_dir())
        
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 総ステップ数の計算
        self.total_steps = (
            len(train_loader) 
            // self.training_config.gradient_accumulation_steps 
            * self.training_config.num_epochs
        )
        
        # オプティマイザの設定
        self.optimizer = self._create_optimizer()
        
        # 学習率スケジューラーの設定
        self.scheduler = self._create_scheduler()
        
        # 混合精度学習のスケーラー
        self.scaler = GradScaler() if self.training_config.fp16 else None
        
        # 学習状態
        self.state = TrainingState()
        
        # 学習履歴（ロス曲線用）
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_steps': [],
            'val_steps': [],
            'learning_rates': []
        }
        
        # WandBの初期化
        self.wandb_run = None
        if self.output_config.use_wandb:
            self._init_wandb()
        
        print(f"\n{'=' * 60}")
        print("Trainer Initialized")
        print(f"{'=' * 60}")
        print(f"Total epochs: {self.training_config.num_epochs}")
        print(f"Batch size: {self.training_config.batch_size}")
        print(f"Gradient accumulation: {self.training_config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.training_config.batch_size * self.training_config.gradient_accumulation_steps}")
        print(f"Total steps: {self.total_steps}")
        print(f"Learning rate: {self.training_config.learning_rate}")
        print(f"Eval steps: {self.training_config.eval_steps}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'=' * 60}\n")
    
    def _create_optimizer(self) -> AdamW:
        """オプティマイザを作成"""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = AdamW(
            trainable_params,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
    
    def _create_scheduler(self) -> Any:
        """学習率スケジューラーを作成"""
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
        """Weights & Biasesを初期化"""
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
                }
            )
            print("✅ WandB initialized")
        except ImportError:
            print("⚠️ WandB not installed, logging disabled")
            self.output_config.use_wandb = False
        except Exception as e:
            print(f"⚠️ WandB initialization failed: {e}")
            self.output_config.use_wandb = False
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """バッチから損失を計算"""
        point_clouds = batch['point_clouds'].to(self.device, dtype=torch.float16)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        use_amp = self.training_config.fp16 or self.training_config.bf16
        dtype = torch.float16 if self.training_config.fp16 else torch.bfloat16
        
        with autocast(enabled=use_amp, dtype=dtype):
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
        """1エポックの学習を実行"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.training_config.num_epochs}",
            ncols=100
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            try:
                loss = self._compute_loss(batch)
                
                # 勾配累積のためにlossをスケーリング
                loss = loss / self.training_config.gradient_accumulation_steps
                
                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_loss += loss.item() * self.training_config.gradient_accumulation_steps
                
                # 勾配累積の更新
                if (batch_idx + 1) % self.training_config.gradient_accumulation_steps == 0:
                    # 勾配クリッピング
                    if self.training_config.max_grad_norm is not None:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.training_config.max_grad_norm
                        )
                    
                    # オプティマイザのステップ
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.state.global_step += 1
                    
                    # 現在の学習率を記録
                    current_lr = self.scheduler.get_last_lr()[0]
                    
                    # ログ記録
                    if self.state.global_step % self.training_config.logging_steps == 0:
                        avg_loss = total_loss / (batch_idx + 1)
                        pbar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'lr': f'{current_lr:.2e}'
                        })
                        
                        self._log_metrics({
                            'train_loss': avg_loss,
                            'learning_rate': current_lr,
                            'epoch': epoch,
                        })
                        
                        # 学習率を記録
                        self.history['learning_rates'].append(current_lr)
                    
                    # ✅ ステップごとの検証評価
                    if (self.training_config.eval_strategy == "steps" and 
                        self.state.global_step % self.training_config.eval_steps == 0):
                        
                        val_loss = self.evaluate()
                        self.history['val_loss'].append(val_loss)
                        self.history['val_steps'].append(self.state.global_step)
                        
                        # ロス曲線を保存
                        self.plot_loss_curves()
                        
                        # 早期停止チェック
                        if self._check_early_stopping(val_loss):
                            print("\n⚠️ Early stopping triggered!")
                            return total_loss / num_batches
                        
                        # モデルを学習モードに戻す
                        self.model.train()
                    
                    # チェックポイント保存
                    if self.state.global_step % self.training_config.save_steps == 0:
                        self.save_checkpoint(f"checkpoint-{self.state.global_step}")
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️ GPU OOM at batch {batch_idx}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        avg_epoch_loss = total_loss / num_batches
        
        # エポック終了時の学習ロスを記録
        self.history['train_loss'].append(avg_epoch_loss)
        self.history['train_steps'].append(self.state.global_step)
        
        return avg_epoch_loss
    
    def evaluate(self) -> float:
        """検証データで評価を実行"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        print(f"\n{'=' * 60}")
        print("🔍 Running Validation")
        print(f"{'=' * 60}")
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", ncols=100):
                try:
                    loss = self._compute_loss(batch)
                    total_loss += loss.item()
                    num_batches += 1
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("\n⚠️ GPU OOM during validation")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # ログ記録
        self._log_metrics({
            'val_loss': avg_loss,
            'epoch': self.state.epoch,
        })
        
        print(f"📊 Validation Loss: {avg_loss:.4f}")
        print(f"{'=' * 60}\n")
        
        return avg_loss
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """早期停止の条件をチェック"""
        threshold = self.training_config.early_stopping_threshold
        
        if val_loss < self.state.best_val_loss - threshold:
            # 改善があった場合
            self.state.best_val_loss = val_loss
            self.state.early_stopping_counter = 0
            
            # ベストモデルを保存
            if self.output_config.save_best_only:
                self.save_checkpoint("best_model")
            
            return False
        else:
            # 改善がない場合
            self.state.early_stopping_counter += 1
            
            if self.state.early_stopping_counter >= self.training_config.early_stopping_patience:
                return True
            
            return False
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """メトリクスをログに記録"""
        if self.output_config.use_wandb and self.wandb_run is not None:
            import wandb
            wandb.log(metrics, step=self.state.global_step)
    
    def plot_loss_curves(self):
        """学習曲線をプロット"""
        try:
            plt.figure(figsize=(12, 5))
            
            # Train loss
            plt.subplot(1, 2, 1)
            if self.history['train_loss']:
                plt.plot(self.history['train_steps'], self.history['train_loss'], 
                        'b-', label='Train Loss', linewidth=2)
            if self.history['val_loss']:
                plt.plot(self.history['val_steps'], self.history['val_loss'], 
                        'r-', label='Val Loss', linewidth=2)
            plt.xlabel('Global Steps')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Learning rate
            plt.subplot(1, 2, 2)
            if self.history['learning_rates']:
                steps = range(len(self.history['learning_rates']))
                plt.plot(steps, self.history['learning_rates'], 
                        'g-', label='Learning Rate', linewidth=2)
            plt.xlabel('Steps')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存
            plot_path = self.log_dir / 'loss_curves.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # WandBにもアップロード
            if self.output_config.use_wandb and self.wandb_run is not None:
                import wandb
                wandb.log({"loss_curves": wandb.Image(str(plot_path))}, 
                         step=self.state.global_step)
            
        except Exception as e:
            print(f"⚠️ Failed to plot loss curves: {e}")
    
    def save_checkpoint(self, name: str):
        """チェックポイントを保存"""
        checkpoint_path = self.checkpoint_dir / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Point Projectorの重みを保存
        try:
            from model_utils_proj_only import get_point_projector_state_dict
            projector_state_dict = get_point_projector_state_dict(self.model)
            torch.save(
                projector_state_dict,
                checkpoint_path / "point_proj.pt"
            )
        except ImportError:
            # model_utils がない場合は point_proj を直接保存
            state_dict = {}
            for key, value in self.model.state_dict().items():
                if 'point_proj' in key:
                    state_dict[key] = value
            torch.save(state_dict, checkpoint_path / "point_proj.pt")
        
        # LoRAアダプタの保存（存在する場合）
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(checkpoint_path)
        
        # オプティマイザとスケジューラーの状態を保存
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if self.scaler else None,
            'state': self.state.__dict__,
            'history': self.history,  # ✅ 学習履歴も保存
        }, checkpoint_path / "trainer_state.pt")
        
        # 設定を保存
        self.config.save(str(checkpoint_path / "config.json"))
        
        # 学習履歴をJSON形式でも保存
        history_path = checkpoint_path / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """古いチェックポイントを削除"""
        if self.training_config.save_total_limit is None:
            return
        
        checkpoints = sorted(
            [d for d in self.checkpoint_dir.iterdir() 
             if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[-1])
        )
        
        while len(checkpoints) > self.training_config.save_total_limit:
            old_checkpoint = checkpoints.pop(0)
            import shutil
            shutil.rmtree(old_checkpoint)
            print(f"🗑️ Removed old checkpoint: {old_checkpoint}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """チェックポイントを読み込み"""
        checkpoint_path = Path(checkpoint_path)
        
        # LoRAアダプタの読み込み
        adapter_path = checkpoint_path / "adapter_model.bin"
        if not adapter_path.exists():
            adapter_path = checkpoint_path / "adapter_model.safetensors"
            
        if adapter_path.exists():
            from peft import set_peft_model_state_dict
            adapters_weights = torch.load(adapter_path, map_location=self.device)
            set_peft_model_state_dict(self.model, adapters_weights)
            print("✅ LoRA weights loaded")

        # Point Projectorの重みを読み込み
        projector_path = checkpoint_path / "point_proj.pt"
        if projector_path.exists():
            state_dict = torch.load(projector_path, map_location=self.device)
            model_state_dict = self.model.state_dict()
            for key, value in state_dict.items():
                if key in model_state_dict:
                    model_state_dict[key] = value
            self.model.load_state_dict(model_state_dict, strict=False)
            print("✅ Projector weights loaded")
        
        # トレーナーの状態を読み込み
        trainer_state_path = checkpoint_path / "trainer_state.pt"
        if trainer_state_path.exists():
            state = torch.load(trainer_state_path, map_location=self.device)
            
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
            
            if self.scaler and state.get('scaler'):
                self.scaler.load_state_dict(state['scaler'])
            
            # TrainingStateの復元
            for key, value in state['state'].items():
                setattr(self.state, key, value)
            
            # ✅ 学習履歴の復元
            if 'history' in state:
                self.history = state['history']
        
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
        print(f"   Resuming from epoch {self.state.epoch}, step {self.state.global_step}")
    
    def train(self) -> Dict[str, List[float]]:
        """学習のメインループを実行"""
        print("\n" + "=" * 60)
        print("🚀 Starting Training")
        print("=" * 60 + "\n")
        
        start_time = time.time()
        
        # チェックポイントからの再開
        if self.output_config.resume_from_checkpoint:
            self.load_checkpoint(self.output_config.resume_from_checkpoint)
        
        try:
            for epoch in range(self.state.epoch, self.training_config.num_epochs):
                self.state.epoch = epoch
                
                # 学習
                train_loss = self.train_epoch(epoch)
                
                # エポック終了時の評価
                if self.training_config.eval_strategy == "epoch":
                    val_loss = self.evaluate()
                    self.history['val_loss'].append(val_loss)
                    self.history['val_steps'].append(self.state.global_step)
                    
                    # ロス曲線を保存
                    self.plot_loss_curves()
                    
                    # 早期停止チェック
                    if self._check_early_stopping(val_loss):
                        print("⚠️ Early stopping triggered!")
                        break
                
                # エポック終了時のチェックポイント
                self.save_checkpoint(f"checkpoint-epoch-{epoch + 1}")
                
                print(f"\n📈 Epoch {epoch + 1} completed: train_loss={train_loss:.4f}\n")
        
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
            self.save_checkpoint("interrupted")
        
        # 学習終了
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("✅ Training Completed!")
        print(f"   Total time: {elapsed_time / 60:.2f} minutes")
        print(f"   Final step: {self.state.global_step}")
        print(f"   Best validation loss: {self.state.best_val_loss:.4f}")
        print("=" * 60 + "\n")
        
        # 最終的なロス曲線を保存
        self.plot_loss_curves()
        
        # WandBの終了
        if self.wandb_run is not None:
            import wandb
            wandb.finish()
        
        return self.history


def setup_trainer_components(
    model: nn.Module,
    config: FullConfig,
    num_training_steps: Optional[int] = None
) -> Tuple[Any, Any, Optional[GradScaler]]:
    """
    学習コンポーネント（optimizer, scheduler, scaler）をセットアップ
    
    これは後方互換性のために残されたヘルパー関数です。
    新しいコードではTrainerクラスを直接使用してください。
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = AdamW(
        trainable_params,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    # スケジューラー
    if num_training_steps is None:
        # デフォルトで適当な値を設定
        num_training_steps = 1000
    
    warmup_steps = int(num_training_steps * config.training.warmup_ratio)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Scaler
    scaler = GradScaler() if config.training.fp16 else None
    
    return optimizer, scheduler, scaler


def train(
    model: nn.Module,
    tokenizer: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Any,
    scheduler: Any,
    scaler: Optional[GradScaler],
    config: FullConfig,
    device: str = "cuda"
) -> Dict[str, List[float]]:
    """
    学習を実行するヘルパー関数（後方互換性用）
    
    新しいコードではTrainerクラスを直接使用してください。
    """
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    return trainer.train()


# エクスポート
__all__ = [
    "TrainingState",
    "Trainer",
    "setup_trainer_components",
    "train",
]
