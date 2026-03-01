#!/usr/bin/env python3
"""
head_space_lps_analyzer.py - Head Space LPS Direction Analyzer

Head座標系における解剖学的方向(LPS)を特定するモジュール

重要な前提条件:
- 点群データは既に正規化済み (normalized)
- Objectの元の座標系はDICOM (LPS)
- 回転はUSDZファイルの3Dモデル重心で行う (点群重心ではない)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import sys

# Try importing usdz_utils
try:
    from .usdz_utils import USDZAnalyzer
except ImportError:
    print("Warning: usdz_utils not found. USDZ centroid auto-detection disabled.", file=sys.stderr)
    USDZAnalyzer = None


@dataclass
class LPSDirections:
    """Head座標系におけるLPS方向ベクトル"""
    left: np.ndarray      # +X in DICOM → ? in Head
    posterior: np.ndarray  # +Y in DICOM → ? in Head  
    superior: np.ndarray   # +Z in DICOM → ? in Head
    
    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            'left': {
                'x': float(self.left[0]),
                'y': float(self.left[1]), 
                'z': float(self.left[2])
            },
            'posterior': {
                'x': float(self.posterior[0]),
                'y': float(self.posterior[1]),
                'z': float(self.posterior[2])
            },
            'superior': {
                'x': float(self.superior[0]),
                'y': float(self.superior[1]),
                'z': float(self.superior[2])
            }
        }
    
    def get_dominant_axis(self, dicom_axis: str) -> Tuple[str, float]:
        """
        DICOM軸がHead座標系でどの軸に最も近いか判定
        
        Args:
            dicom_axis: 'L', 'P', 'S'
        
        Returns:
            (axis_name, direction): ('X'/'Y'/'Z', +1/-1)
        """
        if dicom_axis == 'L':
            vec = self.left
        elif dicom_axis == 'P':
            vec = self.posterior
        elif dicom_axis == 'S':
            vec = self.superior
        else:
            raise ValueError(f"Invalid DICOM axis: {dicom_axis}")
        
        abs_vec = np.abs(vec)
        max_idx = np.argmax(abs_vec)
        axis_names = ['X', 'Y', 'Z']
        direction = np.sign(vec[max_idx])
        
        return axis_names[max_idx], direction


def compute_lps_from_rotation(rotation_matrix: np.ndarray) -> LPSDirections:
    """
    回転行列からLPS方向をHead座標系で計算
    
    Args:
        rotation_matrix: 3x3回転行列
    
    Returns:
        LPSDirections object
    """
    # DICOM座標系の基底ベクトル
    # +X = Left, +Y = Posterior, +Z = Superior
    dicom_left = np.array([1, 0, 0])
    dicom_posterior = np.array([0, 1, 0])
    dicom_superior = np.array([0, 0, 1])
    
    # 回転行列で変換
    head_left = rotation_matrix @ dicom_left
    head_posterior = rotation_matrix @ dicom_posterior
    head_superior = rotation_matrix @ dicom_superior
    
    # 正規化
    head_left = head_left / np.linalg.norm(head_left)
    head_posterior = head_posterior / np.linalg.norm(head_posterior)
    head_superior = head_superior / np.linalg.norm(head_superior)
    
    return LPSDirections(
        left=head_left,
        posterior=head_posterior,
        superior=head_superior
    )


def create_system_context_from_rotation(
    rotation_matrix: np.ndarray,
    lang: str = "en",
    task_type: str = "color_identification"
) -> str:
    """
    回転行列からシステムコンテキストを生成
    
    Args:
        rotation_matrix: 3x3回転行列
        lang: 'en' or 'ja'
        task_type: 'color_identification', 'mask_identification', 'wisdom_teeth'
    
    Returns:
        システムコンテキスト文字列
    """
    lps = compute_lps_from_rotation(rotation_matrix)
    
    # 主軸を取得
    left_axis, left_sign = lps.get_dominant_axis('L')
    post_axis, post_sign = lps.get_dominant_axis('P')
    sup_axis, sup_sign = lps.get_dominant_axis('S')
    
    def signed_axis(axis: str, sign: float) -> str:
        return f"+{axis}" if sign > 0 else f"-{axis}"
    
    L_repr = signed_axis(left_axis, left_sign)
    R_repr = signed_axis(left_axis, -left_sign)
    P_repr = signed_axis(post_axis, post_sign)
    A_repr = signed_axis(post_axis, -post_sign)
    S_repr = signed_axis(sup_axis, sup_sign)
    I_repr = signed_axis(sup_axis, -sup_sign)
    
    if lang == "en":
        base_context = (
            "You are an AI assistant analyzing 3D dental anatomy in Head Space coordinate system. "
            "The data follows the LPS (Left Posterior Superior) coordinate system, defined relative to the patient:\n"
            f"1. {left_axis} axis: {L_repr} is the patient's Left (+L), {R_repr} is the patient's Right (-R).\n"
            f"2. {post_axis} axis: {P_repr} is Posterior (+P), {A_repr} is Anterior (-A).\n"
            f"3. {sup_axis} axis: {S_repr} is Superior (+S), {I_repr} is Inferior (-I).\n"
        )
        
        if task_type == "color_identification":
            base_context += "This point cloud contains a color-marked region at the user-pointed location.\nWhen identifying anatomical structures, provide responses in the following format:\n1. Anatomical location description (using LPS directional terms)\n2. FDI notation for specific teeth (if applicable)\n\nExample responses:\n- 'Right Posterior Superior of Maxilla and Upper Skull'\n- 'Upper left 4th tooth from the midline (1st premolar)'\n- 'Left Posterior Inferior Buccal of Mandible'\n- 'Lower left 5th tooth from the midline (2nd premolar)'\n- 'Upper right 2nd tooth from the midline (Lateral incisor)'\nUSER:"
        elif task_type == "mask_identification":
            base_context += "Please identify the missing anatomical structure, considering this spatial orientation."
        elif task_type == "wisdom_teeth":
            base_context += "Please evaluate the presence of wisdom teeth, considering this spatial orientation."
        else:
            base_context += "Please analyze the dental anatomy considering this spatial orientation."
    
    else:  # Japanese
        base_context = (
            "あなたはHead Space座標系における3D歯科解剖構造を分析するAIアシスタントです。\n"
            "データは患者を基準としたLPS(左・後・上)座標系に従っています:\n"
            f"1. {left_axis}軸:{L_repr}は患者の「左」(+L)、{R_repr}は患者の「右」(-R)\n"
            f"2. {post_axis}軸:{P_repr}は「後方」(+P)、{A_repr}は「前方」(-A)\n"
            f"3. {sup_axis}軸:{S_repr}は「上方」(+S)、{I_repr}は「下方」(-I)\n"
        )
        
        if task_type == "color_identification":
            base_context += "この空間的な方向定義に基づき、指定された色で強調表示されている解剖学的構造を特定してください。"
        elif task_type == "mask_identification":
            base_context += "この空間的な方向定義に基づき、欠損している解剖学的構造を特定してください。"
        elif task_type == "wisdom_teeth":
            base_context += "この空間的な方向定義に基づき、親知らず(第三大臼歯)の有無を評価してください。"
        else:
            base_context += "この空間的な方向定義に基づき、歯科解剖構造を分析してください。"
    
    return base_context


class HeadSpaceLPSAnalyzer:
    """Head座標系におけるLPS方向を解析"""
    
    def __init__(self, usdz_centroid: np.ndarray):
        """
        Args:
            usdz_centroid: USDZファイルの3Dモデル重心 (DICOM座標系, メートル単位)
        """
        self.usdz_centroid = np.array(usdz_centroid)
    
    @staticmethod
    def get_usdz_centroid(usdz_path: str) -> Optional[np.ndarray]:
        """
        USDZファイルから重心を自動取得
        
        Args:
            usdz_path: USDZファイルパス
        
        Returns:
            centroid (DICOM座標系, メートル) or None
        """
        if USDZAnalyzer is None:
            print("Error: usdz_utils.USDZAnalyzer not available", file=sys.stderr)
            return None
        
        try:
            analyzer = USDZAnalyzer(usdz_path)
            centroid = analyzer.get_visual_center()
            print(f"✓ USDZ centroid detected: [{centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}] m")
            return centroid
        except Exception as e:
            print(f"Error loading USDZ: {e}", file=sys.stderr)
            return None
    
    @staticmethod
    def quaternion_to_rotation_matrix(quat: Dict[str, float]) -> np.ndarray:
        """クォータニオンから回転行列を生成"""
        w, x, y, z = quat['w'], quat['x'], quat['y'], quat['z']
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        return R
    
    def analyze_lps_in_head_space(
        self,
        object_transform: Dict
    ) -> LPSDirections:
        """
        Head座標系におけるLPS方向を計算
        
        Args:
            object_transform: TrackingDataのobjectTransform
                {
                    'position': {'x', 'y', 'z'},
                    'rotation': {'w', 'x', 'y', 'z'},
                    'scale': {'x', 'y', 'z'}
                }
        
        Returns:
            LPSDirections: Head座標系におけるLPS方向ベクトル
        """
        # 回転行列を取得
        R = self.quaternion_to_rotation_matrix(object_transform['rotation'])
        
        # スケール行列
        S = np.diag([
            object_transform['scale']['x'],
            object_transform['scale']['y'],
            object_transform['scale']['z']
        ])
        
        # 結合変換行列: R @ S
        transform_matrix = R @ S
        
        # DICOM座標系の基底ベクトル
        # +X = Left, +Y = Posterior, +Z = Superior
        dicom_left = np.array([1, 0, 0])
        dicom_posterior = np.array([0, 1, 0])
        dicom_superior = np.array([0, 0, 1])
        
        # Head座標系における方向ベクトルを計算
        # 注意: 重心での回転なので、方向ベクトルのみ変換
        head_left = transform_matrix @ dicom_left
        head_posterior = transform_matrix @ dicom_posterior
        head_superior = transform_matrix @ dicom_superior
        
        # 正規化
        head_left = head_left / np.linalg.norm(head_left)
        head_posterior = head_posterior / np.linalg.norm(head_posterior)
        head_superior = head_superior / np.linalg.norm(head_superior)
        
        return LPSDirections(
            left=head_left,
            posterior=head_posterior,
            superior=head_superior
        )
    
    @staticmethod
    def load_tracking_data(tracking_json: str, object_id: str = "UserTargetModel") -> Optional[Dict]:
        """
        TrackingDataからobjectTransformを取得
        
        Args:
            tracking_json: TrackingData JSONファイルパス
            object_id: 対象オブジェクトID
        
        Returns:
            objectTransform or None
        """
        with open(tracking_json, 'r') as f:
            data = json.load(f)
        
        if not data.get('frames'):
            return None
        
        frame = data['frames'][0]
        
        if 'objects' not in frame:
            return None
        
        for obj in frame['objects']:
            # objectID or id をチェック
            obj_id = obj.get('objectID') or obj.get('id')
            
            if obj_id == object_id:
                # objectTransformがある場合はそれを返す
                if 'objectTransform' in obj:
                    return obj['objectTransform']
                
                # objectTransformがない場合、position/rotation/scaleを直接使用
                if 'position' in obj and 'rotation' in obj and 'scale' in obj:
                    return {
                        'position': obj['position'],
                        'rotation': obj['rotation'],
                        'scale': obj['scale']
                    }
        
        return None
    
    @staticmethod
    def from_tracking_data(
        tracking_json: str,
        usdz_centroid: np.ndarray,
        object_id: str = "UserTargetModel"
    ) -> Tuple[Optional['HeadSpaceLPSAnalyzer'], Optional[LPSDirections]]:
        """
        TrackingDataから直接LPS方向を解析
        
        Args:
            tracking_json: TrackingData JSONファイルパス
            usdz_centroid: USDZモデル重心 (DICOM座標系, メートル)
            object_id: オブジェクトID
        
        Returns:
            (analyzer, lps_directions) or (None, None)
        """
        object_transform = HeadSpaceLPSAnalyzer.load_tracking_data(tracking_json, object_id)
        
        if not object_transform:
            return None, None
        
        analyzer = HeadSpaceLPSAnalyzer(usdz_centroid)
        lps_directions = analyzer.analyze_lps_in_head_space(object_transform)
        
        return analyzer, lps_directions


def format_direction_string(vec: np.ndarray, threshold: float = 0.1) -> str:
    """
    方向ベクトルを文字列で表現
    
    Args:
        vec: 3D方向ベクトル
        threshold: この値以下の成分は無視
    
    Returns:
        例: "+X-Z" (主にX正方向、少しZ負方向)
    """
    axis_names = ['X', 'Y', 'Z']
    components = []
    
    for i, val in enumerate(vec):
        if abs(val) > threshold:
            sign = '+' if val > 0 else '-'
            components.append(f"{sign}{axis_names[i]}")
    
    return ''.join(components) if components else "~0"


def print_lps_analysis(lps_directions: LPSDirections):
    """LPS解析結果を見やすく表示"""
    print("\n" + "="*70)
    print("LPS DIRECTIONS IN HEAD SPACE")
    print("="*70)
    
    print("\nDICOM Axis → Head Space Direction:")
    print(f"  Left (+X):      {format_direction_string(lps_directions.left):>10}  "
          f"[{lps_directions.left[0]:+.4f}, {lps_directions.left[1]:+.4f}, {lps_directions.left[2]:+.4f}]")
    print(f"  Posterior (+Y): {format_direction_string(lps_directions.posterior):>10}  "
          f"[{lps_directions.posterior[0]:+.4f}, {lps_directions.posterior[1]:+.4f}, {lps_directions.posterior[2]:+.4f}]")
    print(f"  Superior (+Z):  {format_direction_string(lps_directions.superior):>10}  "
          f"[{lps_directions.superior[0]:+.4f}, {lps_directions.superior[1]:+.4f}, {lps_directions.superior[2]:+.4f}]")
    
    print("\nDominant Axes:")
    for dicom_axis, name in [('L', 'Left'), ('P', 'Posterior'), ('S', 'Superior')]:
        head_axis, direction = lps_directions.get_dominant_axis(dicom_axis)
        sign = '+' if direction > 0 else '-'
        print(f"  {name:10} → {sign}{head_axis}")
    
    print("="*70 + "\n")


def main():
    """使用例とテスト"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze LPS directions in Head Space"
    )
    parser.add_argument(
        "--tracking-json",
        required=True,
        help="TrackingData JSON file"
    )
    
    # USDZ centroid: ファイルまたは座標値
    centroid_group = parser.add_mutually_exclusive_group(required=True)
    centroid_group.add_argument(
        "--usdz-file",
        help="USDZ file path (centroid will be auto-detected)"
    )
    centroid_group.add_argument(
        "--usdz-centroid",
        nargs=3,
        type=float,
        help="USDZ model centroid in DICOM space (meters): x y z"
    )
    
    parser.add_argument(
        "--object-id",
        default="UserTargetModel",
        help="Object ID to analyze"
    )
    parser.add_argument(
        "--output-json",
        help="Output JSON file (optional)"
    )
    
    args = parser.parse_args()
    
    # USDZ重心を取得
    if args.usdz_file:
        usdz_centroid = HeadSpaceLPSAnalyzer.get_usdz_centroid(args.usdz_file)
        if usdz_centroid is None:
            print("✗ Failed to extract USDZ centroid")
            return 1
        print(f"Using USDZ file: {args.usdz_file}")
    else:
        usdz_centroid = np.array(args.usdz_centroid)
        print(f"Using manual centroid: [{usdz_centroid[0]:.6f}, {usdz_centroid[1]:.6f}, {usdz_centroid[2]:.6f}] m")
    
    print(f"\nAnalyzing: {args.tracking_json}")
    
    # LPS方向を解析
    analyzer, lps_directions = HeadSpaceLPSAnalyzer.from_tracking_data(
        args.tracking_json,
        usdz_centroid,
        args.object_id
    )
    
    if not lps_directions:
        print(f"✗ Failed to load object transform for '{args.object_id}'")
        return 1
    
    # 結果を表示
    print_lps_analysis(lps_directions)
    
    # JSON出力
    if args.output_json:
        output_data = {
            'tracking_source': args.tracking_json,
            'object_id': args.object_id,
            'usdz_centroid': {
                'x': float(usdz_centroid[0]),
                'y': float(usdz_centroid[1]),
                'z': float(usdz_centroid[2])
            },
            'lps_directions': lps_directions.to_dict()
        }
        
        if args.usdz_file:
            output_data['usdz_source'] = args.usdz_file
        
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())