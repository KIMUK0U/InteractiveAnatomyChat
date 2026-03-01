"""
点群ファイルの読み込みモジュール
"""

import numpy as np
from pathlib import Path
from typing import Optional


def find_point_cloud_file(
    model_filename: str,
    usdz_pc_dir: Path
) -> Optional[Path]:
    """
    USDZファイル名から対応する点群ファイル(.npy)を検索
    """
    # 拡張子を除去してベース名を取得
    base_name = Path(model_filename).stem
    
    # 対応する.npyファイルを検索
    candidates = [
        usdz_pc_dir / f"{base_name}.npy",
        usdz_pc_dir / f"{base_name}_pc.npy",
        usdz_pc_dir / f"{base_name}_pointcloud.npy",
        usdz_pc_dir / f"{base_name}_8192pts.npy",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            print(f"✅ Found point cloud: {candidate}")
            return candidate
    
    # ファイル名の一部マッチングも試す
    for npy_file in usdz_pc_dir.glob("*.npy"):
        if base_name.lower() in npy_file.stem.lower():
            print(f"✅ Found point cloud (partial match): {npy_file}")
            return npy_file
    
    print(f"⚠️ Point cloud not found for: {model_filename}")
    print(f"   Searched in: {usdz_pc_dir}")
    print(f"   Base name: {base_name}")
    
    return None


def load_point_cloud(
    file_path: Path,
    expected_shape: tuple = None
) -> np.ndarray:
    """
    点群ファイルを読み込む
    
    Args:
        file_path: 点群ファイルのパス
        expected_shape: 期待される形状（チェック用）
    
    Returns:
        np.ndarray: 点群データ (N, 3) or (N, 6)
        ※7チャンネル以上ある場合は先頭6チャンネルにトリミングされます
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")
    
    try:
        point_cloud = np.load(file_path)
        print(f"📊 Loaded point cloud: {point_cloud.shape}")
        
        # 形状チェック
        if point_cloud.ndim != 2:
            raise ValueError(f"Invalid point cloud shape: {point_cloud.shape}")
        
        # ------------------------------------------------------------
        # ★修正箇所: チャンネル数が6を超える場合は強制的に6にトリミング
        # ------------------------------------------------------------
        if point_cloud.shape[1] > 6:
            print(f"⚠️ Trimming extra channels: {point_cloud.shape[1]} -> 6 (XYZRGB)")
            # [x, y, z, r, g, b, class...] -> [x, y, z, r, g, b]
            point_cloud = point_cloud[:, :6]
        
        # チャンネル数チェック (3 or 6)
        if point_cloud.shape[1] not in [3, 6]:
            raise ValueError(
                f"Point cloud must have 3 (XYZ) or 6 (XYZ+RGB) channels, "
                f"got {point_cloud.shape[1]}"
            )
        
        if expected_shape and point_cloud.shape != expected_shape:
            print(f"⚠️ Warning: Expected shape {expected_shape}, got {point_cloud.shape}")
        
        return point_cloud
        
    except Exception as e:
        raise RuntimeError(f"Failed to load point cloud: {str(e)}")


def save_point_cloud(
    point_cloud: np.ndarray,
    file_path: Path
):
    """
    点群をファイルに保存
    """
    try:
        np.save(file_path, point_cloud)
        print(f"💾 Saved point cloud to: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save point cloud: {str(e)}")