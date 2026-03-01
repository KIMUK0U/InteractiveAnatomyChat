# データセット準備ガイド (SETUP.md)

このドキュメントでは、PointLLM Point Projectorファインチューニング用のカスタムデータセットの準備方法を詳細に説明します。

## データセットの構成

### 必要なディレクトリ構造

```
your_dataset/
├── point_clouds/           # 点群ファイル（.npy形式）
│   ├── sample_001.npy
│   ├── sample_002.npy
│   └── ...
├── annotations/            # アノテーションファイル
│   ├── train.json          # 学習用アノテーション
│   ├── val.json            # 検証用アノテーション
│   └── test.json           # テスト用アノテーション（オプション）
└── metadata.json           # データセットメタ情報（オプション）
```

## 点群ファイル形式 (.npy)

### 基本仕様

点群データは NumPy 配列として保存します。各点は6次元ベクトル（XYZ座標 + RGB色）で表現されます。

```python
import numpy as np

# 点群の形状: (N, 6) where N <= 8192
# カラム: [X, Y, Z, R, G, B]
point_cloud = np.load("sample.npy")
print(point_cloud.shape)  # 例: (8192, 6)
```

### 座標系とスケール

PointLLMは正規化された点群を入力として受け取ります。以下の条件を満たす必要があります：

```
座標 (XYZ): 重心を原点に配置し、最大距離が1.0になるように正規化
           → 範囲: おおよそ [-1, 1]

色 (RGB):  0～1の範囲に正規化
           → 範囲: [0, 1]
```

### 正規化コード例

```python
import numpy as np

def prepare_point_cloud(points: np.ndarray) -> np.ndarray:
    """
    点群をPointLLM用に正規化します。
    
    Args:
        points: 形状 (N, 6) の点群 [X, Y, Z, R, G, B]
                RGBは0-255または0-1の範囲
    
    Returns:
        正規化された点群 (N, 6)
    """
    # 座標とRGBを分離
    xyz = points[:, :3].copy()
    rgb = points[:, 3:6].copy()
    
    # 座標の正規化（公式pc_normalize関数と同等）
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    if max_dist > 0:
        xyz = xyz / max_dist
    
    # RGBの正規化（0-1の範囲に）
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    
    return np.concatenate([xyz, rgb], axis=1).astype(np.float32)


def downsample_point_cloud(points: np.ndarray, target_points: int = 8192) -> np.ndarray:
    """
    点群を指定された点数にダウンサンプリングします。
    
    Args:
        points: 入力点群 (N, 6)
        target_points: 目標点数（デフォルト: 8192）
    
    Returns:
        ダウンサンプリングされた点群 (target_points, 6)
    """
    n_points = points.shape[0]
    
    if n_points == target_points:
        return points
    elif n_points > target_points:
        # ランダムサンプリング
        indices = np.random.choice(n_points, target_points, replace=False)
        return points[indices]
    else:
        # アップサンプリング（繰り返し）
        indices = np.random.choice(n_points, target_points, replace=True)
        return points[indices]
```

### ファイル保存例

```python
import numpy as np

# 例: メッシュから点群を生成して保存
import trimesh

def mesh_to_pointcloud_npy(mesh_path: str, output_path: str, n_points: int = 8192):
    """
    メッシュファイルから点群を生成し、.npy形式で保存します。
    """
    mesh = trimesh.load(mesh_path)
    
    # 表面から均等にサンプリング
    points, face_indices = trimesh.sample.sample_surface(mesh, n_points)
    
    # 頂点色または面色を取得
    if mesh.visual.kind == 'vertex':
        # 頂点色を補間
        colors = mesh.visual.vertex_colors[mesh.faces[face_indices]]
        colors = np.mean(colors, axis=1)[:, :3] / 255.0
    elif mesh.visual.kind == 'face':
        colors = mesh.visual.face_colors[face_indices][:, :3] / 255.0
    else:
        # デフォルトグレー
        colors = np.ones((n_points, 3)) * 0.5
    
    # 結合
    point_cloud = np.concatenate([points, colors], axis=1)
    
    # 正規化
    point_cloud = prepare_point_cloud(point_cloud)
    
    # 保存
    np.save(output_path, point_cloud.astype(np.float32))
    print(f"Saved: {output_path}, shape: {point_cloud.shape}")
```

## アノテーションファイル形式 (JSON)

### 基本構造

```json
{
  "version": "1.0",
  "description": "Custom dataset for PointLLM fine-tuning",
  "data": [
    {
      "id": "sample_001",
      "point_cloud": "point_clouds/sample_001.npy",
      "conversations": [
        {
          "role": "user",
          "content": "<point>\nこの3Dオブジェクトは何ですか？"
        },
        {
          "role": "assistant", 
          "content": "これは人間の手の3Dモデルです。5本の指が明確に見え、手のひらと甲の形状が正確に再現されています。"
        }
      ]
    },
    {
      "id": "sample_002",
      "point_cloud": "point_clouds/sample_002.npy",
      "conversations": [
        {
          "role": "user",
          "content": "<point>\nこのオブジェクトの特徴を詳しく説明してください。"
        },
        {
          "role": "assistant",
          "content": "このオブジェクトは下顎骨（かがくこつ）です。U字型の形状をしており、左右に顎関節に接続する部分があります。前方には歯槽骨があり、歯を支える構造になっています。"
        }
      ]
    }
  ]
}
```

### フィールド説明

| フィールド | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| `version` | string | オプション | データセットのバージョン |
| `description` | string | オプション | データセットの説明 |
| `data` | array | 必須 | サンプルの配列 |
| `data[].id` | string | 必須 | ユニークな識別子 |
| `data[].point_cloud` | string | 必須 | 点群ファイルへの相対パス |
| `data[].conversations` | array | 必須 | 会話ターンの配列 |
| `conversations[].role` | string | 必須 | "user" または "assistant" |
| `conversations[].content` | string | 必須 | 発話内容 |

### 特殊トークン `<point>`

ユーザーの質問内で `<point>` トークンを使用すると、その位置に点群の特徴が挿入されます。これはPointLLMの内部処理で以下のように展開されます：

```
<point> → <point_start><point_patch>×513<point_end>
```

したがって、アノテーションでは必ずユーザー発話に `<point>` を含めてください。

### マルチターン会話の例

```json
{
  "id": "sample_003",
  "point_cloud": "point_clouds/sample_003.npy",
  "conversations": [
    {
      "role": "user",
      "content": "<point>\nこれは何ですか？"
    },
    {
      "role": "assistant",
      "content": "これは人間の歯の3Dモデルです。具体的には上顎の大臼歯（奥歯）に見えます。"
    },
    {
      "role": "user",
      "content": "この歯の特徴をもっと詳しく教えてください。"
    },
    {
      "role": "assistant",
      "content": "この大臼歯には、咬合面（噛む面）に4つの咬頭が見られます。歯根は3本あり、頬側に2本、舌側に1本という典型的な上顎大臼歯の構造をしています。エナメル質の表面は滑らかで、健康な歯の状態を示しています。"
    }
  ]
}
```

## データセット分割の推奨

### 分割比率

| セット | 比率 | 用途 |
|--------|------|------|
| train | 80% | モデル学習 |
| val | 10% | ハイパーパラメータ調整、早期停止 |
| test | 10% | 最終評価（オプション） |

### 分割スクリプト例

```python
import json
import random
from pathlib import Path

def split_dataset(
    annotation_file: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
):
    """
    データセットを学習/検証/テストに分割します。
    """
    random.seed(seed)
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data['data']
    random.shuffle(samples)
    
    n_total = len(samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_data in [
        ('train', train_samples),
        ('val', val_samples),
        ('test', test_samples)
    ]:
        output_data = {
            'version': data.get('version', '1.0'),
            'description': f"{data.get('description', '')} - {split_name} split",
            'data': split_data
        }
        
        output_path = output_dir / f'{split_name}.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"{split_name}: {len(split_data)} samples → {output_path}")

# 使用例
split_dataset(
    'all_annotations.json',
    'annotations/',
    train_ratio=0.8,
    val_ratio=0.1
)
```

## データ品質チェックリスト

学習を開始する前に、以下の項目を確認してください：

### 点群ファイル

- [ ] すべての.npyファイルが `(N, 6)` の形状を持つ
- [ ] 点数Nが8192以下である
- [ ] XYZ座標が正規化されている（およそ[-1, 1]の範囲）
- [ ] RGB値が[0, 1]の範囲に収まっている
- [ ] ファイルが`float32`型で保存されている
- [ ] 欠損値（NaN, Inf）が含まれていない

### アノテーションファイル

- [ ] JSONが正しい形式である
- [ ] すべての`point_cloud`パスが存在するファイルを指している
- [ ] すべてのユーザー発話に`<point>`トークンが含まれている
- [ ] `role`が"user"または"assistant"のみである
- [ ] 各サンプルに少なくとも1ターンの会話がある

### 検証スクリプト

```python
import json
import numpy as np
from pathlib import Path

def validate_dataset(annotation_path: str, base_dir: str):
    """
    データセットの品質を検証します。
    """
    base_dir = Path(base_dir)
    
    with open(annotation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    errors = []
    warnings = []
    
    for i, sample in enumerate(data['data']):
        sample_id = sample.get('id', f'index_{i}')
        
        # 点群ファイルの検証
        pc_path = base_dir / sample['point_cloud']
        if not pc_path.exists():
            errors.append(f"[{sample_id}] Point cloud not found: {pc_path}")
            continue
        
        try:
            pc = np.load(pc_path)
            
            # 形状チェック
            if len(pc.shape) != 2 or pc.shape[1] != 6:
                errors.append(f"[{sample_id}] Invalid shape: {pc.shape}, expected (N, 6)")
            
            # 点数チェック
            if pc.shape[0] > 8192:
                errors.append(f"[{sample_id}] Too many points: {pc.shape[0]} > 8192")
            
            # 座標範囲チェック
            xyz = pc[:, :3]
            if np.abs(xyz).max() > 2.0:
                warnings.append(f"[{sample_id}] Coordinates may not be normalized: max={np.abs(xyz).max():.2f}")
            
            # RGB範囲チェック
            rgb = pc[:, 3:6]
            if rgb.max() > 1.0 or rgb.min() < 0.0:
                errors.append(f"[{sample_id}] RGB out of [0,1] range: min={rgb.min():.2f}, max={rgb.max():.2f}")
            
            # NaN/Infチェック
            if np.any(np.isnan(pc)) or np.any(np.isinf(pc)):
                errors.append(f"[{sample_id}] Contains NaN or Inf values")
            
            # データ型チェック
            if pc.dtype != np.float32:
                warnings.append(f"[{sample_id}] Data type is {pc.dtype}, recommended float32")
                
        except Exception as e:
            errors.append(f"[{sample_id}] Failed to load: {e}")
        
        # 会話の検証
        conversations = sample.get('conversations', [])
        if len(conversations) == 0:
            errors.append(f"[{sample_id}] No conversations")
        
        has_point_token = False
        for turn in conversations:
            if turn['role'] == 'user' and '<point>' in turn['content']:
                has_point_token = True
                break
        
        if not has_point_token:
            errors.append(f"[{sample_id}] Missing <point> token in user message")
    
    # 結果表示
    print(f"\n=== Validation Results ===")
    print(f"Total samples: {len(data['data'])}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    
    if errors:
        print("\n--- Errors ---")
        for e in errors[:20]:  # 最初の20件
            print(f"  ❌ {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
    
    if warnings:
        print("\n--- Warnings ---")
        for w in warnings[:10]:
            print(f"  ⚠️ {w}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")
    
    return len(errors) == 0

# 使用例
validate_dataset('annotations/train.json', 'your_dataset/')
```

## よくある質問

### Q: 点群の点数が8192未満でも大丈夫ですか？

A: はい、大丈夫です。ただし、すべての点群を同じ点数に揃えることを推奨します。dataset.pyでは自動的にパディングまたはサンプリングを行います。

### Q: RGB情報がない点群を使えますか？

A: はい、使用できます。RGBをダミー値（例：0.5, 0.5, 0.5のグレー）で埋めてください。

### Q: 日本語のアノテーションは使えますか？

A: はい、PointLLMのベースであるLLaMAは日本語をある程度理解できます。ただし、英語のほうが精度が高い傾向があります。

### Q: 複数の質問形式を混在させてもいいですか？

A: はい、むしろ推奨します。多様な質問形式（「これは何？」「説明して」「特徴は？」など）を含めることで、モデルの汎化性能が向上します。
