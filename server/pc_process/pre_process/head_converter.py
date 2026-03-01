"""
head_converter.py - ワールド座標系の点群を頭部座標系（Head Space）に変換するモジュール
"""

import numpy as np
from typing import Dict, Union, Any

def _get_quat_values(quat: Union[Dict, Any]) -> tuple:
    """クォータニオンの値(w,x,y,z)を取得 (Dict/Pydantic対応)"""
    if isinstance(quat, dict):
        return quat['w'], quat['x'], quat['y'], quat['z']
    else:
        # Pydanticモデルやオブジェクトの場合
        return quat.w, quat.x, quat.y, quat.z

def _quaternion_inverse_values(w, x, y, z) -> tuple:
    """クォータニオンの逆を計算して値を返す"""
    norm_sq = w**2 + x**2 + y**2 + z**2
    if norm_sq == 0:
        return 1.0, 0.0, 0.0, 0.0
    return w / norm_sq, -x / norm_sq, -y / norm_sq, -z / norm_sq

def _quaternion_to_rotation_matrix(w, x, y, z) -> np.ndarray:
    """クォータニオンの値から回転行列を生成"""
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    return R

def transform_pc_to_head_space(
    point_cloud: np.ndarray, 
    device_transform: Union[Dict, Any]
) -> np.ndarray:
    """
    点群データをワールド座標系から頭部座標系(Head Space)に変換する
    
    Args:
        point_cloud: (N, 3) または (N, 6+) の点群データ
                     [x, y, z, (r, g, b, ...)]
        device_transform: Vision Proのデバイス位置姿勢情報
                          Dict または Pydanticモデル (position, rotationを持つこと)
    
    Returns:
        np.ndarray: 変換後の点群データ (形状は入力と同じ)
    """
    if device_transform is None:
        print("⚠️ Warning: device_transform is None, skipping head space conversion.")
        return point_cloud

    # 1. 座標と属性データの分離
    xyz_world = point_cloud[:, :3]
    others = point_cloud[:, 3:] if point_cloud.shape[1] > 3 else None
    
    # 2. デバイス情報の抽出 (Dict/Pydantic対応)
    if isinstance(device_transform, dict):
        pos = device_transform['position']
        rot = device_transform['rotation']
        device_pos = np.array([pos['x'], pos['y'], pos['z']])
    else:
        # Pydanticモデル
        pos = device_transform.position
        rot = device_transform.rotation
        device_pos = np.array([pos.x, pos.y, pos.z])
    
    # 3. 変換行列の作成 (World -> Head)
    # Head座標系への変換は、デバイスの位置分引いて、デバイスの逆回転を掛ける
    
    # クォータニオン取得
    w, x, y, z = _get_quat_values(rot)
    
    # 逆クォータニオン計算
    inv_w, inv_x, inv_y, inv_z = _quaternion_inverse_values(w, x, y, z)
    
    # 回転行列（逆回転）生成
    R_inv = _quaternion_to_rotation_matrix(inv_w, inv_x, inv_y, inv_z)
    
    # 4. 座標変換実行
    # 平行移動 (原点をデバイス位置へ)
    centered = xyz_world - device_pos
    
    # 回転適用
    # (N, 3) @ (3, 3).T  <=>  v @ R.T
    xyz_head = (R_inv @ centered.T).T
    
    # 5. データの再結合
    if others is not None:
        return np.hstack([xyz_head, others])
    else:
        return xyz_head