import numpy as np
import json
from pathlib import Path
from scipy.spatial import cKDTree
from collections import Counter
from typing import Dict, List, Tuple, Union, Optional

# --- 定数定義 ---
# 距離閾値設定
DISTANCE_THRESHOLD = 0.05  # 50mm
MAX_NEIGHBORS = 32       # 統計用近傍点数
COLOR_NEIGHBORS = 410     # 着色用近傍点数

# 減衰パラメータ (σ ≈ 42.5mm)
import math
SIGMA = DISTANCE_THRESHOLD / math.sqrt(-2 * math.log(0.01))
DECAY_THRESHOLD = 0.1

# 手の骨格定義 (補間用)
HAND_SKELETON = {
    'thumb': {
        'bones': [('wrist', 'thumbKnuckle'), ('thumbKnuckle', 'thumbIntermediateBase'),
                  ('thumbIntermediateBase', 'thumbIntermediateTip'), ('thumbIntermediateTip', 'thumbTip')],
        'points_per_bone': [3, 4, 5, 6]
    },
    'index': {
        'bones': [('wrist', 'indexFingerMetacarpal'), ('indexFingerMetacarpal', 'indexFingerKnuckle'),
                  ('indexFingerKnuckle', 'indexFingerIntermediateBase'), 
                  ('indexFingerIntermediateBase', 'indexFingerIntermediateTip'),
                  ('indexFingerIntermediateTip', 'indexFingerTip')],
        'points_per_bone': [2, 3, 4, 5, 6]
    },
    'middle': {
        'bones': [('wrist', 'middleFingerMetacarpal'), ('middleFingerMetacarpal', 'middleFingerKnuckle'),
                  ('middleFingerKnuckle', 'middleFingerIntermediateBase'), 
                  ('middleFingerIntermediateBase', 'middleFingerIntermediateTip'),
                  ('middleFingerIntermediateTip', 'middleFingerTip')],
        'points_per_bone': [2, 3, 4, 5, 6]
    },
    'ring': {
        'bones': [('wrist', 'ringFingerMetacarpal'), ('ringFingerMetacarpal', 'ringFingerKnuckle'),
                  ('ringFingerKnuckle', 'ringFingerIntermediateBase'), 
                  ('ringFingerIntermediateBase', 'ringFingerIntermediateTip'),
                  ('ringFingerIntermediateTip', 'ringFingerTip')],
        'points_per_bone': [2, 3, 4, 5, 6]
    },
    'little': {
        'bones': [('wrist', 'littleFingerMetacarpal'), ('littleFingerMetacarpal', 'littleFingerKnuckle'),
                  ('littleFingerKnuckle', 'littleFingerIntermediateBase'), 
                  ('littleFingerIntermediateBase', 'littleFingerIntermediateTip'),
                  ('littleFingerIntermediateTip', 'littleFingerTip')],
        'points_per_bone': [2, 3, 4, 5, 6]
    },
    'palm': {
        'bones': [('wrist', 'forearmArm')],
        'points_per_bone': [2]
    }
}

def _extract_joint_pos(joint_data: Union[dict, object]) -> np.ndarray:
    """関節データから座標抽出 (Dict/Pydantic対応)"""
    if isinstance(joint_data, dict):
        pos = joint_data['position']
        return np.array([pos['x'], pos['y'], pos['z']])
    else:
        pos = joint_data.position
        return np.array([pos.x, pos.y, pos.z])

def _generate_hand_points(hand_data: Union[dict, object]) -> np.ndarray:
    """骨格情報に基づいて手の点群を補間生成"""
    if not hand_data:
        return np.array([]).reshape(0, 3)

    # Dict/Pydantic対応
    is_tracked = hand_data['isTracked'] if isinstance(hand_data, dict) else hand_data.isTracked
    joints = hand_data['joints'] if isinstance(hand_data, dict) else hand_data.joints

    if not is_tracked:
        return np.array([]).reshape(0, 3)

    joint_positions = {}
    for joint in joints:
        # Dict/Pydantic対応
        j_tracked = joint['isTracked'] if isinstance(joint, dict) else joint.isTracked
        j_name = joint['jointName'] if isinstance(joint, dict) else joint.jointName
        
        if j_tracked:
            joint_positions[j_name] = _extract_joint_pos(joint)

    all_points = []
    
    # 骨格補間
    for _, finger_data in HAND_SKELETON.items():
        bones = finger_data['bones']
        points_per_bone = finger_data['points_per_bone']
        for (j1, j2), num in zip(bones, points_per_bone):
            if j1 in joint_positions and j2 in joint_positions:
                t = np.linspace(0, 1, num)
                start = joint_positions[j1]
                end = joint_positions[j2]
                points = np.outer(1 - t, start) + np.outer(t, end)
                all_points.append(points)
    
    # 重要関節点の追加
    important_joints = ['wrist', 'thumbTip', 'indexFingerTip', 'middleFingerTip', 
                        'ringFingerTip', 'littleFingerTip']
    for j_name in important_joints:
        if j_name in joint_positions:
            all_points.append(joint_positions[j_name].reshape(1, 3))
            
    return np.vstack(all_points) if all_points else np.array([]).reshape(0, 3)

def _compute_subclass_statistics(pointcloud: np.ndarray, neighbor_indices: np.ndarray, class_info: dict) -> dict:
    """近傍点のクラス/サブクラス統計計算"""
    if len(neighbor_indices) == 0 or class_info is None:
        return {'total_neighbors': 0, 'statistics': []}
    
    # 7列目: class, 8列目: subclass (index 6, 7)
    if pointcloud.shape[1] < 8:
        return {'total_neighbors': len(neighbor_indices), 'statistics': [], 'error': 'No class data'}

    neighbor_subclasses = pointcloud[neighbor_indices, 7].astype(int)
    neighbor_classes = pointcloud[neighbor_indices, 6].astype(int)
    
    subclass_counter = Counter(neighbor_subclasses)
    total = len(neighbor_indices)
    sorted_subclasses = subclass_counter.most_common()
    
    stats = []
    if sorted_subclasses:
        # 1位
        top1_sub = sorted_subclasses[0][0]
        top1_count = sorted_subclasses[0][1]
        top1_ratio = top1_count / total
        
        # クラスID特定 (中央値)
        top1_cls = int(np.median(neighbor_classes[neighbor_subclasses == top1_sub]))
        
        # 名前解決
        sub_name = class_info.get('subclasses', {}).get(str(top1_sub), {}).get('name', f'Subclass_{top1_sub}')
        cls_name = class_info.get('classes', {}).get(str(top1_cls), {}).get('name', f'Class_{top1_cls}')
        
        stats.append({
            'rank': 1,
            'subclass_id': int(top1_sub),
            'subclass_name': sub_name,
            'class_id': int(top1_cls),
            'class_name': cls_name,
            'count': int(top1_count),
            'ratio': float(top1_ratio)
        })
        
        # 2位 (8割未満の場合)
        if top1_ratio < 0.8 and len(sorted_subclasses) > 1:
            top2_sub = sorted_subclasses[1][0]
            top2_count = sorted_subclasses[1][1]
            top2_ratio = top2_count / total
            top2_cls = int(np.median(neighbor_classes[neighbor_subclasses == top2_sub]))
            
            sub_name = class_info.get('subclasses', {}).get(str(top2_sub), {}).get('name', f'Subclass_{top2_sub}')
            cls_name = class_info.get('classes', {}).get(str(top2_cls), {}).get('name', f'Class_{top2_cls}')
            
            stats.append({
                'rank': 2,
                'subclass_id': int(top2_sub),
                'subclass_name': sub_name,
                'class_id': int(top2_cls),
                'class_name': cls_name,
                'count': int(top2_count),
                'ratio': float(top2_ratio)
            })
            
    return {'total_neighbors': total, 'statistics': stats}

def _apply_fixed_color(original_colors: np.ndarray, distances: np.ndarray, 
                      fixed_color: np.ndarray) -> np.ndarray:
    """ガウシアン減衰を用いた固定色への変色（デバッグ出力付き）"""
    
    # 距離の二乗
    dist_sq = distances**2
    sigma_sq_2 = 2 * SIGMA**2
    
    # 重み計算
    weights = np.exp(-dist_sq / sigma_sq_2)
    
    # マスク作成
    valid_mask = (distances <= DISTANCE_THRESHOLD) & (weights >= DECAY_THRESHOLD)
    
    # --- デバッグ出力開始 ---
    print(f"\n--- [DEBUG] _apply_fixed_color ---")
    print(f"  Input points count: {len(distances)}")
    print(f"  Target Color: {fixed_color}")
    
    if len(distances) > 0:
        min_dist = np.min(distances)
        max_weight = np.max(weights)
        print(f"  Nearest point distance: {min_dist:.6f} (Threshold: {DISTANCE_THRESHOLD})")
        print(f"  Max calculated weight:  {max_weight:.6f} (Decay Threshold: {DECAY_THRESHOLD})")
        print(f"  SIGMA: {SIGMA:.6f}")
        
        count_dist_ok = np.sum(distances <= DISTANCE_THRESHOLD)
        count_weight_ok = np.sum(weights >= DECAY_THRESHOLD)
        count_valid = np.sum(valid_mask)
        
        print(f"  Points within dist limit: {count_dist_ok}")
        print(f"  Points above weight limit: {count_weight_ok}")
        print(f"  FINAL VALID POINTS to color: {count_valid}")
        
        if count_valid == 0:
            print("  ⚠️ WARNING: No points matched the criteria. Color will NOT change.")
            if count_dist_ok > 0:
                print("  -> Distance is OK, but weight is too low. Check SIGMA or DECAY_THRESHOLD.")
            else:
                print("  -> Points are too far away.")
    else:
        print("  ⚠️ WARNING: No neighbor points provided.")
    # --- デバッグ出力終了 ---

    new_colors = original_colors.copy()
    if np.any(valid_mask):
        w = weights[valid_mask, np.newaxis]
        # 色の混合: 元の色 * (1-w) + 目標色 * w
        new_colors[valid_mask] = (
            original_colors[valid_mask] * (1 - w) +
            fixed_color * w
        )
    return new_colors

def process_and_analyze_interaction(
    point_cloud: np.ndarray,
    frame_data: Union[dict, object],
    target_color: str = 'blue',
    class_info: Optional[dict] = None
) -> Tuple[np.ndarray, dict]:
    # ... (前半部分は変更なし) ...
    
    # 1. 色の設定
    color_map = {
        'red': np.array([1.0, 0.0, 0.0]),
        'blue': np.array([0.0, 0.0, 1.0]),
        'black': np.array([0.0, 0.0, 0.0]),
        'white': np.array([1.0, 1.0, 1.0])
    }
    fixed_color = color_map.get(target_color.lower(), np.array([0.0, 0.0, 1.0]))
    print(f'fixed_color: {fixed_color}')
    # 2. 手の点群生成
    left_hand = frame_data['leftHand'] if isinstance(frame_data, dict) else frame_data.leftHand
    right_hand = frame_data['rightHand'] if isinstance(frame_data, dict) else frame_data.rightHand
    
    left_points = _generate_hand_points(left_hand)
    right_points = _generate_hand_points(right_hand)
    
    all_hand_points = []
    if len(left_points) > 0: all_hand_points.append(left_points)
    if len(right_points) > 0: all_hand_points.append(right_points)
    
    if not all_hand_points:
        return point_cloud, {"closest_interaction": None, "hand_point_analysis": [], "message": "No hand data"}

    all_hand_points = np.vstack(all_hand_points)
    
    # 3. KD-Tree構築
    pc_xyz = point_cloud[:, :3]
    tree_pc = cKDTree(pc_xyz)
    # tree_hand = cKDTree(all_hand_points) # 未使用ならコメントアウト可
    
    # --- デバッグ出力 ---
    print(f"\n--- [DEBUG] process_and_analyze_interaction ---")
    print(f"  PointCloud Range X: {np.min(pc_xyz[:,0]):.4f} ~ {np.max(pc_xyz[:,0]):.4f}")
    print(f"  HandPoints Range X: {np.min(all_hand_points[:,0]):.4f} ~ {np.max(all_hand_points[:,0]):.4f}")
    
    # 4. 分析 (各手の点 -> 点群)
    analysis_results = []
    distances, indices = tree_pc.query(all_hand_points, k=MAX_NEIGHBORS)
    
    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        valid_mask = dists <= DISTANCE_THRESHOLD
        if not np.any(valid_mask):
            continue
        valid_dists = dists[valid_mask]
        valid_idxs = idxs[valid_mask]
        if valid_dists[0] > DISTANCE_THRESHOLD:
            continue

        stats = _compute_subclass_statistics(point_cloud, valid_idxs, class_info)
        analysis_results.append({
            'hand_point_index': int(i),
            'distance_to_nearest': float(valid_dists[0]),
            'neighbor_statistics': stats
        })

    # 5. 最も近い相互作用点の特定
    min_dist_idx = np.argmin(distances[:, 0])
    closest_hand_point = all_hand_points[min_dist_idx]
    min_distance = distances[min_dist_idx, 0]
    nearest_pc_idx_global = indices[min_dist_idx, 0]
    
    print(f"  Closest Hand-PC Distance: {min_distance:.6f}")
    if min_distance > DISTANCE_THRESHOLD:
        print(f"  ⚠️ Hand is too far! ({min_distance:.4f} > {DISTANCE_THRESHOLD}) -> Colorization might fail.")

    # 6. 着色処理
    pc_colored = point_cloud.copy()
    
    color_dists, color_idxs = tree_pc.query(
        closest_hand_point.reshape(1, 3), 
        k=COLOR_NEIGHBORS
    )
    
    # query結果の整形
    color_dists = color_dists[0]
    color_idxs = color_idxs[0]
    
    new_rgb = _apply_fixed_color(
        pc_colored[color_idxs, 3:6], 
        color_dists,
        fixed_color
    )
    pc_colored[color_idxs, 3:6] = new_rgb

    # 7. 結果構築
    result_json = {
        "total_hand_points": len(all_hand_points),
        "total_pc_points": len(point_cloud),
        "target_color": target_color,
        "closest_interaction": {
            "pc_index": int(nearest_pc_idx_global),
            "distance_mm": float(min_distance * 1000),
            "hand_point": closest_hand_point.tolist(),
            "colorized_points": int(np.sum(color_dists <= DISTANCE_THRESHOLD))
        },
        "hand_point_analysis": analysis_results
    }
    
    return pc_colored, result_json

# def process_and_analyze_interaction(
#     point_cloud: np.ndarray,
#     frame_data: Union[dict, object],
#     target_color: str = 'blue',
#     class_info: Optional[dict] = None
# ) -> Tuple[np.ndarray, dict]:
#     """
#     点群とトラッキングデータを統合し、着色と詳細分析を行う
    
#     Args:
#         point_cloud: AR座標系に変換済みの点群 (N, 6) or (N, 8) [xyz, rgb, (class, subclass)]
#         frame_data: FrameDataオブジェクト または 辞書
#         target_color: 'red', 'blue', 'black' のいずれか
#         class_info: class_subclass_info.json の内容 (統計分析に必要)

#     Returns:
#         Tuple[np.ndarray, dict]: 
#             - 色相変化処理後の点群データ (N, 6+)
#             - 分析結果 (hand_analysis json互換の辞書)
#     """
    
#     # 1. 色の設定
#     color_map = {
#         'red': np.array([1.0, 0.0, 0.0]),
#         'blue': np.array([0.0, 0.0, 1.0]),
#         'black': np.array([0.0, 0.0, 0.0])
#     }
#     fixed_color = color_map.get(target_color.lower(), np.array([0.0, 0.0, 1.0])) # デフォルト青

#     # 2. 手の点群生成 (Left/Right統合)
#     left_hand = frame_data['leftHand'] if isinstance(frame_data, dict) else frame_data.leftHand
#     right_hand = frame_data['rightHand'] if isinstance(frame_data, dict) else frame_data.rightHand
    
#     left_points = _generate_hand_points(left_hand)
#     right_points = _generate_hand_points(right_hand)
    
#     all_hand_points = []
#     if len(left_points) > 0: all_hand_points.append(left_points)
#     if len(right_points) > 0: all_hand_points.append(right_points)
    
#     # 手が検出されていない場合
#     if not all_hand_points:
#         empty_analysis = {
#             "closest_interaction": None,
#             "hand_point_analysis": [],
#             "message": "No hand tracking data found"
#         }
#         return point_cloud, empty_analysis

#     all_hand_points = np.vstack(all_hand_points)
    
#     # 3. KD-Tree構築
#     pc_xyz = point_cloud[:, :3]
#     tree_pc = cKDTree(pc_xyz)
#     tree_hand = cKDTree(all_hand_points)
    
#     # 4. 分析 (各手の点 -> 点群)
#     analysis_results = []
    
#     # 統計計算用の近傍探索 (全手の点に対して行うと重い場合は間引くことも検討)
#     distances, indices = tree_pc.query(all_hand_points, k=MAX_NEIGHBORS)
    
#     for i, (dists, idxs) in enumerate(zip(distances, indices)):
#         # 閾値フィルタリング
#         valid_mask = dists <= DISTANCE_THRESHOLD
#         if not np.any(valid_mask):
#             continue
            
#         valid_dists = dists[valid_mask]
#         valid_idxs = idxs[valid_mask]
        
#         # 最近傍距離が閾値を超えていたらスキップ
#         if valid_dists[0] > DISTANCE_THRESHOLD:
#             continue

#         # 統計情報取得
#         stats = _compute_subclass_statistics(point_cloud, valid_idxs, class_info)
        
#         analysis_results.append({
#             'hand_point_index': int(i),
#             'distance_to_nearest': float(valid_dists[0]),
#             'neighbor_statistics': stats
#         })

#     # 5. 最も近い相互作用点 (点群 -> 手) の特定
#     # 点群全体から手の点への距離を計算するのは重いため、
#     # 逆（手から点群への最近傍の中で最小のもの）を利用して近似的に特定
    
#     # 最も点群に近い手の点を探す
#     min_dist_idx = np.argmin(distances[:, 0])
#     closest_hand_point = all_hand_points[min_dist_idx]
#     min_distance = distances[min_dist_idx, 0]
#     nearest_pc_idx_global = indices[min_dist_idx, 0]
    
#     # 6. 着色処理 (最も近い手の点周辺の点群を変色)
#     # 対象点: closest_hand_point から一定距離内の点群
#     pc_colored = point_cloud.copy()
    
#     # 変色対象の近傍点を取得 (ここだけは正確に変色するため、closest_hand_pointを中心に探索)
#     color_dists, color_idxs = tree_pc.query(
#         closest_hand_point.reshape(1, 3), 
#         k=COLOR_NEIGHBORS
#     )
    
#     # queryの結果は2次元配列で返るため整形
#     color_dists = color_dists[0]
#     color_idxs = color_idxs[0]
    
#     # 変色関数の適用
#     new_rgb = _apply_fixed_color(
#         pc_colored[color_idxs, 3:6], # RGB列
#         color_dists,
#         fixed_color
#     )
#     pc_colored[color_idxs, 3:6] = new_rgb

#     # 7. 結果JSONの構築
#     result_json = {
#         "total_hand_points": len(all_hand_points),
#         "total_pc_points": len(point_cloud),
#         "target_color": target_color,
#         "closest_interaction": {
#             "pc_index": int(nearest_pc_idx_global),
#             "distance_mm": float(min_distance * 1000),
#             "hand_point": closest_hand_point.tolist(),
#             "colorized_points": int(np.sum(color_dists <= DISTANCE_THRESHOLD))
#         },
#         "hand_point_analysis": analysis_results
#     }
    
#     return pc_colored, result_json