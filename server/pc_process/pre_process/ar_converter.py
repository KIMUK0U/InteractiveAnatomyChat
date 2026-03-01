"""
ar_converter.py - 点群データをAR空間座標系に変換するモジュール
"""

import numpy as np
from typing import Dict, Optional, Union
from pathlib import Path
import sys

# usdz_utilsがルートまたは適切なパスにあると仮定
try:
    from pc_process.pre_process.ConvertPCFromARData.usdz_utils import USDZAnalyzer
except ImportError:
    # パスが通っていない場合のフォールバック（必要に応じて調整してください）
    sys.path.append(str(Path(__file__).parent.parent))
    from pc_process.pre_process.ConvertPCFromARData.usdz_utils import USDZAnalyzer

def quaternion_to_rotation_matrix(quat: Dict[str, float]) -> np.ndarray:
    """
    クォータニオン(w,x,y,z)を回転行列に変換
    """
    # Pydanticモデルや辞書に対応
    if not isinstance(quat, dict):
        # オブジェクトの場合は属性アクセスを試みる (Pydantic対応)
        w, x, y, z = quat.w, quat.x, quat.y, quat.z
    else:
        w, x, y, z = quat['w'], quat['x'], quat['y'], quat['z']
    
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    return R

def convert_pc_from_ar_data(
    point_cloud: np.ndarray,
    object_data: Union[dict, object],
    usdz_path: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """
    TrackingDataのオブジェクト情報に基づいて、点群をARワールド座標系に変換する
    
    Args:
        point_cloud: 入力点群 (N, 3) または (N, 6+) [x,y,z, r,g,b, ...]
                     ※ 元データはmm単位と仮定
        object_data: TrackingData内の 'objects' の1要素 (position, rotation, scaleを持つ)
                     dict または Pydanticモデル
        usdz_path: 対応するUSDZファイルのパス（重心補正用）
                   Noneの場合は点群の幾何学的中心を使用
    
    Returns:
        np.ndarray: 変換後の点群データ (m単位, ARワールド座標系)
    """
    # データのコピー（元の配列を変更しないため）
    pc_data = point_cloud.copy()
    
    # 座標(XYZ)とそれ以外(色など)を分離
    xyz = pc_data[:, :3]
    others = pc_data[:, 3:] if pc_data.shape[1] > 3 else None
    
    # 1. 変換済み　（mm -> m 変換 (DICOM座標系などは通常mm)）
    # xyz = xyz / 1000.0
    
    # 2. 重心補正 (Centroid Correction)
    # USDZファイルがある場合はVisual Centerを取得して補正に使用
    centroid = None
    if usdz_path and Path(usdz_path).exists():
        try:
            analyzer = USDZAnalyzer(str(usdz_path))
            centroid = analyzer.get_visual_center()
            # print(f"[AR Convert] Using USDZ visual center: {centroid}")
        except Exception as e:
            print(f"[AR Convert] Warning: Failed to load USDZ center: {e}")
    
    # USDZから取得できなかった場合は点群の幾何学的中心を使用
    if centroid is None:
        centroid = (xyz.min(axis=0) + xyz.max(axis=0)) / 2.0
        # print(f"[AR Convert] Using geometric center: {centroid}")
    
    # 重心分だけ移動（原点へ）
    xyz_centered = xyz - centroid
    
    # 3. AR変換パラメータの取得 (PydanticモデルとDict両対応)
    if isinstance(object_data, dict):
        pos = object_data['position']
        rot = object_data['rotation']
        scl = object_data['scale']
        # Dictの場合は値を取り出す(x,y,zキーが必要)
        t_vec = np.array([pos['x'], pos['y'], pos['z']])
        s_vec = np.array([scl['x'], scl['y'], scl['z']])
    else:
        # Pydanticモデルの場合
        pos = object_data.position
        rot = object_data.rotation
        scl = object_data.scale
        t_vec = np.array([pos.x, pos.y, pos.z])
        s_vec = np.array([scl.x, scl.y, scl.z])

    # 4. 変換行列の適用: v' = R * S * v + t
    
    # スケール行列
    S = np.diag(s_vec)
    
    # 回転行列
    R = quaternion_to_rotation_matrix(rot)
    
    # 変換実行
    # (N,3) @ (3,3) -> (N,3)
    # 行列演算: (R @ S @ v.T).T + t  <=>  v @ S.T @ R.T + t
    xyz_transformed = (xyz_centered @ S.T) @ R.T + t_vec
    
    # 5. データの再結合
    if others is not None:
        return np.hstack([xyz_transformed, others])
    else:
        return xyz_transformed