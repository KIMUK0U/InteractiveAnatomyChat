import numpy as np
from pxr import Usd, UsdGeom, Gf

class USDZAnalyzer:
    """
    USDZファイルを解析し、再構成に必要な幾何情報を抽出するクラス。
    RealityKitの計算アルゴリズムに基づき、モデルの視覚的中心を算出します。
    """
    
    def __init__(self, usdz_path: str):
        """
        Args:
            usdz_path (str): 解析対象の.usdzファイルのパス
        """
        self.usdz_path = usdz_path
        # ステージを読み込み専用で開く
        self.stage = Usd.Stage.Open(usdz_path)
        if not self.stage:
            raise RuntimeError(f"USDZファイルの読み込みに失敗しました: {usdz_path}")

    def get_visual_center(self) -> np.ndarray:
        """
        モデル全体のAxis-Aligned Bounding Box (AABB) の中心座標を計算します。
        これはvisionOSの RealityKit における entity.visualBounds(relativeTo: nil).center に相当します。
        
        Returns:
            np.ndarray: [x, y, z] メートル単位の中心座標
        """
        # ステージ内の全オブジェクトを含むルートプリミティブを取得
        pseudo_root = self.stage.GetPseudoRoot()
        imageable = UsdGeom.Imageable(pseudo_root)
        
        # 単位スケール（metersPerUnit）を取得。通常USDZは1.0 = 1m。
        meters_per_unit = UsdGeom.GetStageMetersPerUnit(self.stage)
        
        # ステージのデフォルトタイムコードにおけるバウンディングボックスを計算
        time = Usd.TimeCode.Default()
        
        # ComputeLocalBoundは、各プリミティブのトランスフォームを考慮した
        # そのステージ内でのローカル境界を返します
        bounds = imageable.ComputeLocalBound(time, "default")
        
        # GfRange3dから最小・最大座標を取得
        bbox_range = bounds.GetRange()
        min_pt = bbox_range.GetMin()
        max_pt = bbox_range.GetMax()
        
        # 中心点の計算
        center_x = (min_pt[0] + max_pt[0]) / 2.0
        center_y = (min_pt[1] + max_pt[1]) / 2.0
        center_z = (min_pt[2] + max_pt[2]) / 2.0
        
        # 単位をメートルに変換してnumpy配列で返す
        center = np.array([center_x, center_y, center_z], dtype=np.float32) * meters_per_unit
        
        return center

    def get_metadata(self):
        """参考用のメタデータを取得"""
        return {
            "upAxis": UsdGeom.GetStageUpAxis(self.stage),
            "metersPerUnit": UsdGeom.GetStageMetersPerUnit(self.stage),
            "doc": self.stage.GetMetadata('comment')
        }

# 単体テスト用
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        analyzer = USDZAnalyzer(path)
        center = analyzer.get_visual_center()
        print(f"File: {path}")
        print(f"Visual Center (m): {center}")
        print(f"Metadata: {analyzer.get_metadata()}")