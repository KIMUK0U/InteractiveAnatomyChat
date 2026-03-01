"""
trainer_LoRA.py - PointLLM Point Projector Training Loop

このモジュールは、Point Projectorのファインチューニングのための
学習ループを実装しています。

主な機能:
- 混合精度学習（FP16/BF16）
- 勾配累積
- 学習率スケジューラー
- チェックポイント保存
- WandBへのログ記録
- 早期停止
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
from model_utils_proj_lora import get_point_projector_state_dict


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
        
        # 総ステップ数の計算（スケジューラー作成前に必要）
        self.total_steps = (
            len(train_loader)
            // self.training_config.gradient_accumulation_steps
            * self.training_config.num_epochs
        )
        
        # オプティマイザの設定
        self.optimizer = self._create_optimizer()
        
        # 学習率スケジューラーの設定（total_stepsが必要）
        self.scheduler = self._create_scheduler()
        
        # 混合精度学習のスケーラー
        self.scaler = GradScaler() if self.training_config.fp16 else None
        
        # 学習状態
        self.state = TrainingState()
        
        # ✅ 学習履歴（ロス曲線用）を追加
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
        print(f"Output directory: {self.output_dir}")
        print(f"{'=' * 60}\n")
    
    def _create_optimizer(self) -> AdamW:
        """
        オプティマイザを作成します
        
        学習可能なパラメータ（Point Projector）のみを対象とします。
        """
        # 学習可能なパラメータのみを取得
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
        """
        学習率スケジューラーを作成します
        
        ウォームアップ期間を経て、指定されたスケジューリング方式で学習率を調整します。
        """
        warmup_steps = int(self.total_steps * self.training_config.warmup_ratio)
        
        if self.training_config.lr_scheduler_type == "cosine":
            # コサインアニーリング（ウォームアップ付き）
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    # 線形ウォームアップ
                    return float(current_step) / float(max(1, warmup_steps))
                # コサイン減衰
                progress = float(current_step - warmup_steps) / float(max(1, self.total_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
            return LambdaLR(self.optimizer, lr_lambda)
        
        elif self.training_config.lr_scheduler_type == "linear":
            # 線形減衰（ウォームアップ付き）
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(0.0, float(self.total_steps - current_step) / float(max(1, self.total_steps - warmup_steps)))
            
            return LambdaLR(self.optimizer, lr_lambda)
        
        else:
            # 定数（ウォームアップのみ）
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0
            
            return LambdaLR(self.optimizer, lr_lambda)
    
    def _init_wandb(self):
        """
        Weights & Biasesを初期化します
        """
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
        """
        バッチから損失を計算します
        
        PointLLMの内部で点群の処理が行われ、
        LLaMAの言語モデル損失が計算されます。
        
        Args:
            batch: 入力バッチ
        
        Returns:
            損失テンソル
        """
        # デバイスに移動
        point_clouds = batch['point_clouds'].to(self.device, dtype=torch.float16)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # 混合精度学習の適用
        use_amp = self.training_config.fp16 or self.training_config.bf16
        dtype = torch.float16 if self.training_config.fp16 else torch.bfloat16
        
        with autocast(enabled=use_amp, dtype=dtype):
            # PointLLMのフォワードパス
            # 点群は内部でエンコードされ、テキストトークンと結合されます
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                point_clouds=point_clouds,
                use_cache=False,  # 学習時はキャッシュを使用しない
            )
            
            loss = outputs.loss
        
        return loss
    
    def train_epoch(self, epoch: int) -> float:
        """
        1エポック分の学習を行います
        
        Args:
            epoch: 現在のエポック番号
        
        Returns:
            エポックの平均損失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # プログレスバー
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.training_config.num_epochs}",
            leave=True
        )
        
        # 勾配累積のカウンタ
        accumulation_steps = self.training_config.gradient_accumulation_steps
        
        for step, batch in enumerate(progress_bar):
            # 損失の計算
            loss = self._compute_loss(batch)
            
            # 勾配累積を考慮したスケーリング
            loss = loss / accumulation_steps
            
            # バックワードパス
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * accumulation_steps
            num_batches += 1
            
            # 勾配累積が完了したらパラメータ更新
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
                
                # 学習率更新
                self.scheduler.step()
                
                # 勾配をリセット
                self.optimizer.zero_grad()
                
                # グローバルステップを更新
                self.state.global_step += 1
                
                # 現在の学習率を取得
                current_lr = self.scheduler.get_last_lr()[0]
                
                # ✅ 学習率を記録
                self.history['learning_rates'].append(current_lr)
                
                # プログレスバーの更新
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * accumulation_steps:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                
                # ログ出力
                if self.state.global_step % self.training_config.logging_steps == 0:
                    self._log_metrics({
                        'train/loss': loss.item() * accumulation_steps,
                        'train/learning_rate': current_lr,
                        'train/epoch': epoch + (step / len(self.train_loader)),
                    })
                
                # チェックポイント保存
                if self.state.global_step % self.training_config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.state.global_step}")
                
                # 評価
                if self.training_config.eval_strategy == "steps":
                    if self.state.global_step % self.training_config.eval_steps == 0:
                        val_loss = self.evaluate()
                        
                        # ✅ 検証ロスを記録
                        self.history['val_loss'].append(val_loss)
                        self.history['val_steps'].append(self.state.global_step)
                        
                        # ✅ ロス曲線を保存
                        self.plot_loss_curves()
                        
                        self.model.train()  # 学習モードに戻す
                        
                        # 早期停止チェック
                        if self._check_early_stopping(val_loss):
                            print("Early stopping triggered!")
                            return total_loss / max(num_batches, 1)
        
        # ✅ エポック終了時に学習ロスを記録
        avg_loss = total_loss / max(num_batches, 1)
        self.history['train_loss'].append(avg_loss)
        self.history['train_steps'].append(self.state.global_step)
        
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """
        検証データで評価を行います
        
        Returns:
            検証損失の平均値
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.val_loader,
            desc="Evaluating",
            leave=False
        )
        
        for batch in progress_bar:
            loss = self._compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # ログ出力
        self._log_metrics({
            'eval/loss': avg_loss,
            'eval/step': self.state.global_step,
        })
        
        print(f"\n📊 Validation Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """
        早期停止の条件をチェックします
        
        Args:
            val_loss: 現在の検証損失
        
        Returns:
            早期停止する場合True
        """
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
        """
        メトリクスをログに記録します
        
        Args:
            metrics: メトリクスの辞書
        """
        if self.output_config.use_wandb and self.wandb_run is not None:
            import wandb
            wandb.log(metrics, step=self.state.global_step)
    
    def plot_loss_curves(self):
        """✅ 学習曲線をプロット"""
        try:
            plt.figure(figsize=(12, 5))
            
            # Train & Val loss
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
            
            # Learning rate
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
            
            # 保存
            plot_path = self.log_dir / 'loss_curves.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Loss curves saved to: {plot_path}")
            
            # WandBにもアップロード
            if self.output_config.use_wandb and self.wandb_run is not None:
                import wandb
                wandb.log({"loss_curves": wandb.Image(str(plot_path))},
                         step=self.state.global_step)
            
        except Exception as e:
            print(f"⚠️ Failed to plot loss curves: {e}")
    
    def save_checkpoint(self, name: str):
        """チェックポイント保存（rank 0のみ）- trainer_LoRA.pyと同じ形式"""
        if self.rank != 0:
            return
        
        checkpoint_path = self.checkpoint_dir / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # ✅ 1. Point Projectorの重みを保存（互換性維持）
        from model_utils_proj_lora import get_point_projector_state_dict
        projector_state_dict = get_point_projector_state_dict(self.model.module)
        torch.save(projector_state_dict, checkpoint_path / "point_proj.pt")
        
        # ✅ 2. model.save_pretrained()でLLM全体を保存（trainer_LoRA.pyと同じ）
        if hasattr(self.model.module, "save_pretrained"):
            self.model.module.save_pretrained(checkpoint_path)
            print("✅ Full model saved via save_pretrained()")
        
        # 3. トレーナー状態を保存
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if self.scaler else None,
            'state': self.state.__dict__,
            'history': self.history,
        }, checkpoint_path / "trainer_state.pt")
        
        # 4. 設定を保存
        self.config.save(str(checkpoint_path / "config.json"))
        
        # 5. 学習履歴を保存
        with open(checkpoint_path / "history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # ファイルサイズを表示
        proj_size = (checkpoint_path / "point_proj.pt").stat().st_size / 1024**2
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        print(f"   point_proj.pt: {proj_size:.1f} MB (Projector only)")
        
        # model.safetensorsのサイズも表示
        safetensors_path = checkpoint_path / "model.safetensors"
        if safetensors_path.exists():
            model_size = safetensors_path.stat().st_size / 1024**3
            print(f"   model.safetensors: {model_size:.2f} GB (Full model)")
    
    def _cleanup_old_checkpoints(self):
        """
        古いチェックポイントを削除します
        """
        if self.training_config.save_total_limit is None:
            return
        
        # チェックポイントディレクトリを取得
        checkpoints = sorted(
            [d for d in self.checkpoint_dir.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[-1])
        )
        
        # 制限を超える古いものを削除
        while len(checkpoints) > self.training_config.save_total_limit:
            old_checkpoint = checkpoints.pop(0)
            import shutil
            shutil.rmtree(old_checkpoint)
            print(f"🗑️ Removed old checkpoint: {old_checkpoint}")
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint_path = Path(checkpoint_path)
        
        # 1. LoRAアダプタの読み込み (追加)
        # ※ ipynb側で既に get_peft_model 済みのモデルを想定
        # adapter_path = checkpoint_path / "adapter_model.bin"
        # if not adapter_path.exists():
        #     adapter_path = checkpoint_path / "adapter_model.safetensors" # safetensorsの場合
            
        # if adapter_path.exists():
        #     from peft import set_peft_model_state_dict
        #     adapters_weights = torch.load(adapter_path, map_location=self.device)
        #     set_peft_model_state_dict(self.model, adapters_weights)
        #     print("✅ LoRA weights loaded")

        # 2. Point Projectorの重みを読み込み (既存の処理)
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
                print("✅ Training history restored")
        
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
        print(f"   Resuming from epoch {self.state.epoch}, step {self.state.global_step}")
    
    def train(self) -> Dict[str, List[float]]:
        """
        学習のメインループを実行します
        
        Returns:
            学習履歴（損失の推移など）
        """
        print("\n" + "=" * 60)
        print("🚀 Starting Training")
        print("=" * 60 + "\n")
        
        history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        start_time = time.time()
        
        # チェックポイントからの再開
        if self.output_config.resume_from_checkpoint:
            self.load_checkpoint(self.output_config.resume_from_checkpoint)
        
        try:
            for epoch in range(self.state.epoch, self.training_config.num_epochs):
                self.state.epoch = epoch
                
                # 学習
                train_loss = self.train_epoch(epoch)
                history['train_loss'].append(train_loss)
                
                # エポック終了時の評価
                if self.training_config.eval_strategy == "epoch":
                    val_loss = self.evaluate()
                    history['val_loss'].append(val_loss)
                    
                    # ✅ 検証ロスを記録
                    self.history['val_loss'].append(val_loss)
                    self.history['val_steps'].append(self.state.global_step)
                    
                    # ✅ ロス曲線を保存
                    self.plot_loss_curves()
                    
                    # 早期停止チェック
                    if self._check_early_stopping(val_loss):
                        print("Early stopping triggered!")
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
        
        # ✅ 最終的なロス曲線を保存
        self.plot_loss_curves()
        
        # WandBの終了
        if self.wandb_run is not None:
            import wandb
            wandb.finish()
        
        # ✅ self.historyを返す（より詳細な履歴）
        return self.history


def train_point_projector(
    model: nn.Module,
    tokenizer: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: FullConfig,
    device: str = "cuda"
) -> Dict[str, List[float]]:
    """
    Point Projectorのファインチューニングを実行するヘルパー関数
    
    Args:
        model: PointLLMモデル
        tokenizer: トークナイザー
        train_loader: 学習用DataLoader
        val_loader: 検証用DataLoader
        config: 全体設定
        device: 使用するデバイス
    
    Returns:
        学習履歴
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


def setup_trainer_components(
    model: nn.Module,
    config: TrainingConfig,
    num_training_steps: Optional[int] = None
) -> Tuple[AdamW, Any, Optional[GradScaler]]:
    """
    学習コンポーネント(optimizer, scheduler, scaler)をセットアップ
    
    proj_tune_and_LoRA.pyから呼び出される互換性関数
    
    Args:
        model: PointLLMモデル
        config: TrainingConfig
        num_training_steps: 総学習ステップ数（Noneの場合は後で計算）
    
    Returns:
        optimizer, scheduler, scaler のタプル
    """
    # 学習可能なパラメータのみを取得
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # オプティマイザ
    optimizer = AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # スケジューラ（num_training_stepsがNoneの場合は後で作成）
    scheduler = None
    if num_training_steps is not None:
        warmup_steps = int(num_training_steps * config.warmup_ratio)
        
        if config.lr_scheduler_type == "cosine":
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            scheduler = LambdaLR(optimizer, lr_lambda)
        
        elif config.lr_scheduler_type == "linear":
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps)))
            scheduler = LambdaLR(optimizer, lr_lambda)
        
        else:  # constant
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0
            scheduler = LambdaLR(optimizer, lr_lambda)
    
    # 混合精度学習のスケーラー
    scaler = GradScaler() if config.fp16 else None
    
    return optimizer, scheduler, scaler


def train(
    model: nn.Module,
    tokenizer: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: AdamW,
    scheduler: Any,
    scaler: Optional[GradScaler],
    config: FullConfig
) -> Dict[str, List[float]]:
    """
    学習を実行する関数（Trainerクラスのラッパー）
    
    proj_tune_and_LoRA.pyから呼び出される互換性関数
    
    Args:
        model: PointLLMモデル
        tokenizer: トークナイザー
        train_loader: 学習用DataLoader
        val_loader: 検証用DataLoader
        optimizer: オプティマイザ
        scheduler: 学習率スケジューラー
        scaler: 混合精度学習のスケーラー
        config: 全体設定
    
    Returns:
        学習履歴
    """
    # Trainerクラスを使って学習
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device="cuda"
    )
    
    # 外部から渡されたoptimizer, scheduler, scalerを使用
    trainer.optimizer = optimizer
    if scheduler is not None:
        trainer.scheduler = scheduler
    if scaler is not None:
        trainer.scaler = scaler
    
    return trainer.train()


# エクスポート
__all__ = [
    "TrainingState",
    "Trainer",
    "train_point_projector",
    "setup_trainer_components",
    "train",
]
