"""
config.py - PointLLM Point Projector Fine-tuning Configuration

このモジュールは学習に必要なすべての設定を一元管理します。
dataclassを使用して型安全性と可読性を確保しています。
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import json
import os


@dataclass
class ModelConfig:
    """
    モデル関連の設定を管理するクラス
    
    PointLLMのアーキテクチャ:
    - Point Encoder (PointTransformer): 点群→384次元特徴
    - Point Projector: 384次元→4096次元（学習対象）
    - LLM (LLaMA-7B): テキスト生成
    """
    
    # モデル名またはパス（Hugging Face形式）
    model_name: str = "RunsenXu/PointLLM_7B_v1.2"
    
    # 計算精度設定
    # "float16": 高速だがやや精度低下
    # "bfloat16": A100推奨、float16より安定
    # "float32": 最高精度だがメモリ消費大
    torch_dtype: str = "bfloat16"
    
    # 勾配チェックポイント: メモリ節約のためにactivationを再計算
    # True推奨: メモリ使用量を大幅に削減（速度は若干低下）
    use_gradient_checkpointing: bool = True
    
    # Flash Attention 2を使用するか
    # A100では大幅な高速化が期待できる
    use_flash_attention: bool = True
    
    """
    Finetuningごとに変更箇所
    """
    # 凍結する部分の制御
    freeze_llm: bool = True          # LLaMA部分を凍結
    freeze_point_encoder: bool = True  # PointTransformer部分を凍結
    
    # 学習対象のレイヤー名（Point Projector）
    # PointLLMでは "point_proj" という名前で保存されている
    trainable_module_name: str = "point_proj"


@dataclass
class DataConfig:
    """
    データセット関連の設定を管理するクラス
    
    データの形式:
    - 点群: .npy形式、形状(N, 6) = (点数, XYZ+RGB)
    - アノテーション: JSON形式（SETUP.md参照）
    """
    
    # データセットのルートディレクトリ
    # Google Driveのパスを想定
    data_root: str = "/home/yyamashita/Desktop/kkimu/test/llm_dataset_20260106"
    
    # アノテーションファイルのパス
    train_annotation: str = "annotations/train.json"
    val_annotation: str = "annotations/val.json"
    
    num_points: int = 8192  # PointLLMが受け付ける最大点数
    use_color: bool = True  # RGB色情報を使用するか
    
    # データ拡張設定
    use_augmentation: bool = True
    augmentation_rotation: bool = False     # ランダム回転
    augmentation_noise_std: float = 0.01   # ガウシアンノイズの標準偏差
    augmentation_scale_range: tuple = (0.9, 1.0)  # スケール変動範囲
    
    # データローダー設定
    num_workers: int = 4    # データ読み込みの並列数
    pin_memory: bool = True  # GPUへのデータ転送を高速化
    
    def get_train_annotation_path(self) -> str:
        """学習用アノテーションの完全パスを取得"""
        return os.path.join(self.data_root, self.train_annotation)
    
    def get_val_annotation_path(self) -> str:
        """検証用アノテーションの完全パスを取得"""
        return os.path.join(self.data_root, self.val_annotation)


@dataclass 
class TrainingConfig:
    """
    学習ハイパーパラメータを管理するクラス
    
    Point Projectorは比較的小さなモジュール（約18Mパラメータ）なので、
    一般的なファインチューニングより高めの学習率が使用可能です。
    """
    
    # 基本設定
    num_epochs: int = 5
    batch_size: int = 4           # A100 40GBで安全なサイズ
    gradient_accumulation_steps: int = 4  # 実効バッチサイズ = 4 * 4 = 16
    
    # 学習率設定
    learning_rate: float = 5e-4   # Point Projector用の学習率
    weight_decay: float = 0.01    # L2正則化
    warmup_ratio: float = 0.1     # ウォームアップに使用する割合
    
    # 学習率スケジューラー
    # "cosine": コサインアニーリング（推奨）
    # "linear": 線形減衰
    # "constant": 一定
    lr_scheduler_type: str = "cosine"
    
    # 勾配クリッピング
    # 勾配爆発を防ぐために重要
    max_grad_norm: float = 1.0
    
    # 混合精度学習
    # A100ではbf16が推奨だが、互換性のためfp16をデフォルトに
    fp16: bool = False
    bf16: bool = True  # A100で使用する場合はTrueに
    
    # チェックポイント保存
    save_steps: int = 2000         # N ステップごとに保存
    save_total_limit: int = 3     # 保持するチェックポイント数
    
    # 評価設定
    eval_steps: int = 110         # N ステップごとに評価
    eval_strategy: str = "epoch"  # "steps" または "epoch"
    
    # 早期停止
    early_stopping_patience: int = 10  # 改善がない場合の許容エポック数
    early_stopping_threshold: float = 0.0001  # 改善とみなす最小変化量
    
    # ログ設定
    logging_steps: int = 10       # N ステップごとにログ出力
    
    # 乱数シード（再現性のため）
    seed: int = 42


@dataclass
class OutputConfig:
    """
    出力関連の設定を管理するクラス
    """
    
    # 出力ディレクトリ（Google Drive上）
    output_dir: str = "/home/yyamashita/Desktop/kkimu/test/ProjectionFinetuning/outputs"
    
    """
    変更箇所2
    """
    # 実験名（サブディレクトリ名として使用）
    experiment_name: str = "projection_vicuna_finetune_v1"
    
    # ログ設定
    use_wandb: bool = True        # Weights & Biasesでログを取るか
    wandb_project: str = "pointllm-finetune"
    wandb_entity: Optional[str] = None  # W&Bのチーム名（個人は None）
    
    # チェックポイント設定
    save_best_only: bool = True   # 最良モデルのみ保存
    resume_from_checkpoint: Optional[str] = None  # 再開するチェックポイントのパス
    
    def get_output_path(self) -> str:
        """実験の出力パスを取得"""
        return os.path.join(self.output_dir, self.experiment_name)
    
    def get_checkpoint_dir(self) -> str:
        """チェックポイント保存ディレクトリを取得"""
        return os.path.join(self.get_output_path(), "checkpoints")
    
    def get_log_dir(self) -> str:
        """ログ保存ディレクトリを取得"""
        return os.path.join(self.get_output_path(), "logs")


@dataclass
class GenerationConfig:
    """
    推論・生成時の設定を管理するクラス
    
    これらのパラメータは学習後の評価や推論時に使用します。
    """
    
    # 生成の最大トークン数
    max_new_tokens: int = 512
    
    # サンプリング設定
    do_sample: bool = True        # True: サンプリング、False: greedy
    temperature: float = 0.7      # 高いほどランダム性が増加
    top_p: float = 0.9            # nucleus sampling
    top_k: int = 50               # top-k sampling
    
    # 繰り返し防止
    # この値が重要: 1.0より大きいと繰り返しを抑制
    repetition_penalty: float = 1.2
    
    # 長さペナルティ
    length_penalty: float = 1.0
    
    # 最小生成長
    min_new_tokens: int = 10


@dataclass
class FullConfig:
    """
    すべての設定を統合するクラス
    
    使用例:
        config = FullConfig()
        config.training.learning_rate = 1e-4  # 個別に変更可能
        config.save("config.json")  # 設定を保存
    """
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    def save(self, path: str):
        """設定をJSONファイルに保存"""
        config_dict = {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "output": self.output.__dict__,
            "generation": self.generation.__dict__,
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        print(f"Config saved to: {path}")
    
    @classmethod
    def load(cls, path: str) -> "FullConfig":
        """JSONファイルから設定を読み込み"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = cls()
        
        # 各サブ設定を更新
        for key, value in config_dict.get("model", {}).items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)
        
        for key, value in config_dict.get("data", {}).items():
            if hasattr(config.data, key):
                setattr(config.data, key, value)
        
        for key, value in config_dict.get("training", {}).items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)
        
        for key, value in config_dict.get("output", {}).items():
            if hasattr(config.output, key):
                setattr(config.output, key, value)
        
        for key, value in config_dict.get("generation", {}).items():
            if hasattr(config.generation, key):
                setattr(config.generation, key, value)
        
        print(f"Config loaded from: {path}")
        return config
    
    def print_summary(self):
        """設定のサマリーを表示"""
        print("=" * 60)
        print("Configuration Summary")
        print("=" * 60)
        
        print("\n[Model]")
        print(f"  Model: {self.model.model_name}")
        print(f"  Dtype: {self.model.torch_dtype}")
        print(f"  Freeze LLM: {self.model.freeze_llm}")
        print(f"  Freeze Encoder: {self.model.freeze_point_encoder}")
        
        print("\n[Data]")
        print(f"  Root: {self.data.data_root}")
        print(f"  Points: {self.data.num_points}")
        print(f"  Augmentation: {self.data.use_augmentation}")
        
        print("\n[Training]")
        print(f"  Epochs: {self.training.num_epochs}")
        print(f"  Batch size: {self.training.batch_size}")
        print(f"  Gradient accumulation: {self.training.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.training.batch_size * self.training.gradient_accumulation_steps}")
        print(f"  Learning rate: {self.training.learning_rate}")
        print(f"  LR scheduler: {self.training.lr_scheduler_type}")
        
        print("\n[Output]")
        print(f"  Output dir: {self.output.get_output_path()}")
        print(f"  WandB: {self.output.use_wandb}")
        
        print("=" * 60)


def create_default_config() -> FullConfig:
    """
    デフォルト設定を作成するヘルパー関数
    
    Google Colab A100での実行を想定した設定を返します。
    """
    return FullConfig()


def create_memory_efficient_config() -> FullConfig:
    """
    メモリ効率を重視した設定を作成
    
    メモリが不足する場合はこちらを使用してください。
    """
    config = FullConfig()
    
    # バッチサイズを削減
    config.training.batch_size = 2
    config.training.gradient_accumulation_steps = 8
    
    # 勾配チェックポイントを有効化
    config.model.use_gradient_checkpointing = True
    
    # 評価頻度を下げる
    config.training.eval_steps = 100
    
    return config


def create_quick_test_config() -> FullConfig:
    """
    動作確認用の軽量設定を作成
    
    データセットのセットアップ確認や、
    コードのデバッグに使用してください。
    """
    config = FullConfig()
    
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.training.gradient_accumulation_steps = 1
    config.training.save_steps = 50
    config.training.eval_steps = 25
    config.training.logging_steps = 5
    
    config.output.experiment_name = "quick_test"
    config.output.use_wandb = False
    
    return config


# 設定のエクスポート
__all__ = [
    "ModelConfig",
    "DataConfig", 
    "TrainingConfig",
    "OutputConfig",
    "GenerationConfig",
    "FullConfig",
    "create_default_config",
    "create_memory_efficient_config",
    "create_quick_test_config",
]
