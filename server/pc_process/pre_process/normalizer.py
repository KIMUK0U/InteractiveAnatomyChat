"""
normalizer.py - 点群データの正規化を行うモジュール
"""

import numpy as np

def normalize_point_cloud(point_cloud: np.ndarray) -> np.ndarray:
    """
    点群データを正規化する
    
    処理内容:
    1. XYZ座標: 中心化（重心を原点へ）し、最大距離でスケーリング（単位球内に収める）
    2. RGB色情報: 0〜1の範囲に正規化（もし0〜255なら変換）
    3. 形式: [x, y, z, r, g, b] の6次元に整形（クラスIDなどの余分な列は削除）
    
    Args:
        point_cloud: (N, M) の点群データ (M >= 6)
                     [x, y, z, r, g, b, ...]
    
    Returns:
        np.ndarray: (N, 6) の正規化済み点群データ [x, y, z, r, g, b]
    """
    if point_cloud.shape[1] < 6:
        raise ValueError(f"Input point cloud must have at least 6 channels (XYZRGB), got {point_cloud.shape[1]}")
    
    # データをコピーして操作
    pc_data = point_cloud.copy()
    
    # --- 1. XYZ座標の正規化 ---
    xyz = pc_data[:, :3]
    
    # 中心化 (Centering)
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    
    # スケーリング (Scaling into unit sphere)
    # 原点からの最大距離を求めて割る
    max_dist = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    
    if max_dist > 0:
        xyz = xyz / max_dist
        
    # --- 2. RGB色情報の正規化 ---
    rgb = pc_data[:, 3:6]
    
    # 値の範囲チェック: 1.0を超える値があれば0-255とみなして正規化
    if np.max(rgb) > 1.0:
        rgb = rgb / 255.0
    
    # 0-1にクリップ（念のため）
    rgb = np.clip(rgb, 0.0, 1.0)
    
    # --- 3. 結合と整形 ---
    # [x, y, z, r, g, b] の形にする
    normalized_pc = np.hstack([xyz, rgb])
    
    return normalized_pc.astype(np.float32)