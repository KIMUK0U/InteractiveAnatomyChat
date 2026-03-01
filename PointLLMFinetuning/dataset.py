"""
dataset.py - PointLLM Fine-tuning Dataset Module

このモジュールは、カスタムデータセットの読み込みと前処理を担当します。
PointLLMが期待する形式に点群データとテキストを変換します。

主な機能:
- JSONアノテーションファイルの読み込み
- 点群データ(.npy)の読み込みと正規化
- データ拡張（回転、ノイズ、スケール）
- 会話形式データの処理
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import random
import copy

from config import DataConfig


class PointCloudProcessor:
    """
    点群データの前処理を行うクラス
    
    PointLLMは正規化された点群（座標が[-1,1]、RGBが[0,1]）を期待します。
    このクラスでは、その正規化と各種データ拡張を行います。
    """
    
    def __init__(self, config: DataConfig):
        """
        初期化
        
        Args:
            config: データ設定オブジェクト
        """
        self.config = config
        self.num_points = 8192
        self.use_color = True
        self.use_augmentation = True
    
    def pc_normalize(self, points: np.ndarray) -> np.ndarray:
        """
        点群を正規化します（PointLLM公式の正規化方法）
        
        手順:
        1. 重心を原点に移動
        2. 最大距離で正規化（球内に収める）
        
        Args:
            points: 点群の座標部分 (N, 3)
        
        Returns:
            正規化された座標 (N, 3)
        """
        # 重心を計算して原点に移動
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # 最大距離で正規化
        max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if max_dist > 0:
            points = points / max_dist
        
        return points
    
    def normalize_color(self, colors: np.ndarray) -> np.ndarray:
        """
        RGB値を[0, 1]の範囲に正規化します
        
        Args:
            colors: RGB値 (N, 3)、0-255または0-1の範囲
        
        Returns:
            正規化されたRGB (N, 3)、0-1の範囲
        """
        # 値の範囲を判定
        if colors.max() > 1.0:
            # 0-255の範囲と仮定
            colors = colors / 255.0
        
        # [0, 1]にクリップ
        colors = np.clip(colors, 0.0, 1.0)
        
        return colors
    
    def sample_points(self, points: np.ndarray, n_points: int) -> np.ndarray:
        """
        点群を指定された点数にサンプリングします
        
        Args:
            points: 入力点群 (N, C)
            n_points: 目標点数
        
        Returns:
            サンプリングされた点群 (n_points, C)
        """
        n_current = points.shape[0]
        
        if n_current == n_points:
            return points
        elif n_current > n_points:
            # ダウンサンプリング（ランダム選択）
            indices = np.random.choice(n_current, n_points, replace=False)
            return points[indices]
        else:
            # アップサンプリング（繰り返し選択）
            indices = np.random.choice(n_current, n_points, replace=True)
            return points[indices]
    
    def augment_rotation(self, points: np.ndarray) -> np.ndarray:
        """
        点群にランダムな回転を適用します
        
        Z軸周りの回転のみを適用（上下方向は保持）
        
        Args:
            points: 座標 (N, 3)
        
        Returns:
            回転後の座標 (N, 3)
        """
        # Z軸周りのランダム角度
        theta = np.random.uniform(0, 2 * np.pi)
        
        # 回転行列
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
        
        return points @ rotation_matrix.T
    
    def augment_noise(self, points: np.ndarray, std: float) -> np.ndarray:
        """
        点群にガウシアンノイズを追加します
        
        Args:
            points: 座標 (N, 3)
            std: ノイズの標準偏差
        
        Returns:
            ノイズ付加後の座標 (N, 3)
        """
        noise = np.random.normal(0, std, points.shape)
        return points + noise
    
    def augment_scale(self, points: np.ndarray, scale_range: Tuple[float, float]) -> np.ndarray:
        """
        点群にランダムなスケーリングを適用します
        
        Args:
            points: 座標 (N, 3)
            scale_range: スケール範囲 (min, max)
        
        Returns:
            スケーリング後の座標 (N, 3)
        """
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return points * scale
    
    def process(self, point_cloud: np.ndarray, training: bool = True) -> np.ndarray:
        """
        点群データを処理します（メイン処理関数）
        
        処理の流れ:
        1. 座標とRGBを分離
        2. 点数をサンプリング
        3. 座標を正規化
        4. RGBを正規化
        5. データ拡張（学習時のみ）
        6. 結合して返す
        
        Args:
            point_cloud: 入力点群 (N, 6) = [X, Y, Z, R, G, B]
            training: 学習モードかどうか
        
        Returns:
            処理済み点群 (num_points, 6)
        """
        # 座標とRGBを分離
        xyz = point_cloud[:, :3].copy()
        rgb = point_cloud[:, 3:6].copy() if self.use_color else None
        
        # 点数を調整
        if point_cloud.shape[0] != self.num_points:
            indices = None
            if point_cloud.shape[0] > self.num_points:
                indices = np.random.choice(point_cloud.shape[0], self.num_points, replace=False)
            else:
                indices = np.random.choice(point_cloud.shape[0], self.num_points, replace=True)
            xyz = xyz[indices]
            if rgb is not None:
                rgb = rgb[indices]
        
        # 座標の正規化
        xyz = self.pc_normalize(xyz)
        
        # RGBの正規化
        if rgb is not None:
            rgb = self.normalize_color(rgb)
        else:
            # 色情報がない場合はグレーで埋める
            rgb = np.ones((self.num_points, 3)) * 0.5
        
        # データ拡張（学習時のみ）
        if training and self.use_augmentation:
            if self.config.data.augmentation_rotation:
                xyz = self.augment_rotation(xyz)
                print("[DEBUG] Applied rotation augmentation")
            
            if self.config.data.augmentation_noise_std > 0:
                xyz = self.augment_noise(xyz, self.config.data.augmentation_noise_std)
            
            if self.config.data.augmentation_scale_range is not None:
                xyz = self.augment_scale(xyz, self.config.data.augmentation_scale_range)
                # スケーリング後に再度正規化
                xyz = self.pc_normalize(xyz)
        
        # 結合
        processed = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
        
        return processed


class PointLLMDataset(Dataset):
    """
    PointLLM用のPyTorchデータセットクラス
    
    JSONアノテーションファイルから学習データを読み込み、
    点群とテキストのペアを返します。
    """
    
    # PointLLMの特殊トークン
    POINT_TOKEN = "<point>"
    POINT_START_TOKEN = "<point_start>"
    POINT_END_TOKEN = "<point_end>"
    POINT_PATCH_TOKEN = "<point_patch>"
    
    def __init__(
        self,
        annotation_path: str,
        data_root: str,
        processor: PointCloudProcessor,
        tokenizer: Any,
        training: bool = True,
        max_length: int = 2048
    ):
        """
        データセットを初期化します
        
        Args:
            annotation_path: アノテーションJSONファイルのパス
            data_root: データセットのルートディレクトリ
            processor: 点群前処理オブジェクト
            tokenizer: テキストトークナイザー
            training: 学習モードかどうか
            max_length: テキストの最大長
        """
        self.data_root = Path(data_root)
        self.processor = processor
        self.tokenizer = tokenizer
        self.training = training
        self.max_length = max_length
        
        # アノテーションを読み込み
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.samples = data.get('data', [])
        print(f"Loaded {len(self.samples)} samples from {annotation_path}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_point_cloud(self, relative_path: str) -> np.ndarray:
        """
        点群ファイルを読み込みます
        
        Args:
            relative_path: データルートからの相対パス
        
        Returns:
            点群データ (N, 6)
        """
        full_path = self.data_root / relative_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Point cloud not found: {full_path}")
        
        point_cloud = np.load(str(full_path))
        
        # 形状チェック
        if len(point_cloud.shape) != 2 or point_cloud.shape[1] < 3:
            raise ValueError(f"Invalid point cloud shape: {point_cloud.shape}")
        
        # 6次元未満の場合はパディング
        if point_cloud.shape[1] < 6:
            padding = np.ones((point_cloud.shape[0], 6 - point_cloud.shape[1])) * 0.5
            point_cloud = np.concatenate([point_cloud, padding], axis=1)
        
        return point_cloud[:, :6]  # 最初の6次元のみ使用
    
    def _format_conversation(self, conversations: List[Dict]) -> Tuple[str, str]:
        """
        会話データをプロンプトとターゲットに変換します
        
        PointLLMの入力形式:
        - プロンプト: ユーザーの質問（<point>トークンを含む）
        - ターゲット: アシスタントの回答
        
        Args:
            conversations: 会話ターンのリスト
        
        Returns:
            (prompt, target) のタプル
        """
        prompt_parts = []
        target_parts = []
        
        for i, turn in enumerate(conversations):
            role = turn['role']
            content = turn['content']
            
            if role == 'user':
                # ユーザーターンをプロンプトに追加
                if i == 0:
                    # 最初のターンはシステムプロンプトを含める
                    prompt_parts.append(f"User: {content}")
                else:
                    prompt_parts.append(f"\nUser: {content}")
            elif role == 'assistant':
                # アシスタントターンをターゲットに追加
                target_parts.append(content)
        
        prompt = "".join(prompt_parts) + "\nAssistant: "
        target = " ".join(target_parts)
        
        return prompt, target
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        インデックスに対応するサンプルを取得します
        
        Returns:
            {
                'point_cloud': 点群テンソル (num_points, 6),
                'input_ids': 入力トークンID,
                'attention_mask': アテンションマスク,
                'labels': ラベル(損失計算用)
            }
        """
        sample = self.samples[idx]
        
        # 点群の読み込みと処理
        point_cloud = self._load_point_cloud(sample['point_cloud'])
        point_cloud = self.processor.process(point_cloud, training=self.training)
        
        # 会話の処理
        conversations = sample.get('conversations', [])
        prompt, target = self._format_conversation(conversations)
        
        # ==========================================================
        # 特殊トークンの処理（二重展開を防止）
        # ==========================================================
        point_token_len = 513  # パッチ512 + グローバル1
        expanded_token = self.POINT_START_TOKEN + (self.POINT_PATCH_TOKEN * point_token_len) + self.POINT_END_TOKEN
        
        # ケース1: <point>タグがある場合 → 展開済みトークン列に置換
        if self.POINT_TOKEN in prompt:
            prompt = prompt.replace(self.POINT_TOKEN, expanded_token)
        
        # ケース2: 既に<point_start>と<point_end>がある場合 → そのまま（二重展開防止）
        elif self.POINT_START_TOKEN in prompt and self.POINT_END_TOKEN in prompt:
            pass  # 既に展開済みなので何もしない
        
        # ケース3: どちらもない場合 → プロンプトの先頭に展開済みトークン列を追加
        else:
            prompt = expanded_token + "\n" + prompt
        # ==========================================================

        # トークン化
        full_text = prompt + target + self.tokenizer.eos_token
        
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # ラベルの作成
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        prompt_length = prompt_encoding['input_ids'].shape[1]
        
        labels = input_ids.clone()
        labels[:prompt_length] = -100  # プロンプト部分をマスク
        
        # パディング部分もマスク
        labels[attention_mask == 0] = -100
        
        return {
            'point_cloud': torch.from_numpy(point_cloud),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

class DataCollator:
    """
    バッチ処理用のデータコレーター
    
    複数のサンプルをバッチにまとめる際の処理を行います。
    """
    
    def __init__(self, tokenizer: Any, pad_token_id: int = 0):
        """
        初期化
        
        Args:
            tokenizer: トークナイザー
            pad_token_id: パディングトークンID
        """
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        バッチをまとめます
        
        Args:
            batch: サンプルのリスト
        
        Returns:
            バッチ化されたテンソルの辞書
        """
        # 点群をスタック
        point_clouds = torch.stack([item['point_cloud'] for item in batch])
        
        # テキストデータをスタック
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'point_clouds': point_clouds,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def create_dataloaders(
    config: DataConfig,
    tokenizer: Any,
    train_batch_size: int,
    val_batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    学習用と検証用のDataLoaderを作成します
    
    Args:
        config: データ設定
        tokenizer: トークナイザー
        train_batch_size: 学習バッチサイズ
        val_batch_size: 検証バッチサイズ
    
    Returns:
        (train_loader, val_loader) のタプル
    """
    # 点群プロセッサーを作成
    processor = PointCloudProcessor(config)
    
    # データセットを作成
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
    
    # データコレーターを作成
    collator = DataCollator(
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else 0
    )
    
    # DataLoaderを作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collator,
        drop_last=True,  # 最後の不完全バッチを破棄
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collator,
        drop_last=False,
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    return train_loader, val_loader


# エクスポート
__all__ = [
    "PointCloudProcessor",
    "PointLLMDataset",
    "DataCollator",
    "create_dataloaders",
]
