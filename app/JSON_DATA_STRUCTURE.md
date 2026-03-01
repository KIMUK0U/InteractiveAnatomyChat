# ARKit Data Provider - JSON保存データ一覧

## 概要
このアプリケーションは、ARKitのHand TrackingとVision Proのデバイストラッキング情報を0.1秒間隔で記録し、JSON形式で保存します。

## 保存されるJSONファイルの構造

### トップレベル構造
```json
{
  "sessionStartTime": "2025-10-23T10:30:00Z",
  "sessionEndTime": "2025-10-23T10:35:00Z",
  "metadata": { ... },
  "frames": [ ... ]
}
```

## 1. セッション情報

### sessionStartTime
- **説明**: 記録開始時刻
- **形式**: ISO8601形式の日時文字列

### sessionEndTime
- **説明**: 記録終了時刻
- **形式**: ISO8601形式の日時文字列

### metadata
記録セッションのメタデータ
```json
{
  "deviceModel": "Apple Vision Pro",
  "osVersion": "2.0",
  "appVersion": "1.0.0"
}
```

## 2. フレームデータ (frames配列)

各フレーム（0.1秒ごと）に以下の情報が記録されます：

### timestamp
- **説明**: セッション開始からの経過時間（秒）
- **型**: Float
- **例**: 0.1, 0.2, 0.3...

### frameNumber
- **説明**: フレーム番号（0から始まる連番）
- **型**: Integer

## 3. Vision Proの位置・姿勢情報 (deviceTransform)

```json
{
  "position": {
    "x": 0.0,
    "y": 1.6,
    "z": 0.0
  },
  "forward": {
    "x": 0.0,
    "y": 0.0,
    "z": -1.0
  },
  "up": {
    "x": 0.0,
    "y": 1.0,
    "z": 0.0
  },
  "right": {
    "x": 1.0,
    "y": 0.0,
    "z": 0.0
  },
  "rotation": {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "w": 1.0
  }
}
```

### position
- **説明**: ワールド座標系でのVision Proの位置
- **単位**: メートル

### forward, up, right
- **説明**: デバイスの3軸方向ベクトル（正規化済み）
- **forward**: Z軸（前方向）
- **up**: Y軸（上方向）
- **right**: X軸（右方向）

### rotation
- **説明**: デバイスの姿勢（クォータニオン表現）
- **形式**: {x, y, z, w}

## 4. 手のトラッキングデータ (leftHand, rightHand)

```json
{
  "isTracked": true,
  "chirality": "left",
  "handTransform": {
    "position": { "x": 0.2, "y": 1.2, "z": -0.5 },
    "rotation": { "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0 }
  },
  "joints": [ ... ]
}
```

### isTracked
- **説明**: 手が現在トラッキングされているか
- **型**: Boolean

### chirality
- **説明**: 左手または右手
- **値**: "left" または "right"

### handTransform
- **説明**: 手のアンカーのワールド座標での位置と姿勢

### joints配列
各手には27個の関節が記録されます。

## 5. 関節データ (joints)

```json
{
  "jointName": "indexFingerTip",
  "position": {
    "x": 0.25,
    "y": 1.35,
    "z": -0.48
  },
  "rotation": {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "w": 1.0
  },
  "isTracked": true,
  "localPosition": {
    "x": 0.05,
    "y": 0.15,
    "z": 0.02
  },
  "localRotation": {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "w": 1.0
  }
}
```

### 記録される関節一覧（27点）

#### 手首
- `wrist`: 手首

#### 親指 (Thumb) - 4点
- `thumbKnuckle`: 親指の付け根
- `thumbIntermediateBase`: 親指の中間関節（付け根側）
- `thumbIntermediateTip`: 親指の中間関節（先端側）
- `thumbTip`: 親指の先端

#### 人差し指 (Index) - 5点
- `indexFingerMetacarpal`: 人差し指の中手骨
- `indexFingerKnuckle`: 人差し指の付け根
- `indexFingerIntermediateBase`: 人差し指の第二関節
- `indexFingerIntermediateTip`: 人差し指の第一関節
- `indexFingerTip`: 人差し指の先端

#### 中指 (Middle) - 5点
- `middleFingerMetacarpal`: 中指の中手骨
- `middleFingerKnuckle`: 中指の付け根
- `middleFingerIntermediateBase`: 中指の第二関節
- `middleFingerIntermediateTip`: 中指の第一関節
- `middleFingerTip`: 中指の先端

#### 薬指 (Ring) - 5点
- `ringFingerMetacarpal`: 薬指の中手骨
- `ringFingerKnuckle`: 薬指の付け根
- `ringFingerIntermediateBase`: 薬指の第二関節
- `ringFingerIntermediateTip`: 薬指の第一関節
- `ringFingerTip`: 薬指の先端

#### 小指 (Little) - 5点
- `littleFingerMetacarpal`: 小指の中手骨
- `littleFingerKnuckle`: 小指の付け根
- `littleFingerIntermediateBase`: 小指の第二関節
- `littleFingerIntermediateTip`: 小指の第一関節
- `littleFingerTip`: 小指の先端

#### その他
- `forearmArm`: 前腕部

### position
- **説明**: ワールド座標系での関節の位置
- **単位**: メートル

### rotation
- **説明**: ワールド座標系での関節の回転（クォータニオン）

### isTracked
- **説明**: この関節が現在トラッキングされているか
- **型**: Boolean

### localPosition
- **説明**: 親関節からの相対位置（手のアンカーからのローカル座標）
- **単位**: メートル

### localRotation
- **説明**: 親関節からの相対回転

## 6. 3Dオブジェクトデータ (objects配列)

```json
{
  "objectID": "demo_sphere_001",
  "objectName": "デモ球体",
  "modelFileName": "sphere.usdz",
  "position": {
    "x": 0.0,
    "y": 1.5,
    "z": -2.0
  },
  "forward": {
    "x": 0.0,
    "y": 0.0,
    "z": -1.0
  },
  "up": {
    "x": 0.0,
    "y": 1.0,
    "z": 0.0
  },
  "right": {
    "x": 1.0,
    "y": 0.0,
    "z": 0.0
  },
  "rotation": {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "w": 1.0
  },
  "scale": {
    "x": 0.2,
    "y": 0.2,
    "z": 0.2
  }
}
```

### objectID
- **説明**: オブジェクトの一意識別子
- **型**: String

### objectName
- **説明**: オブジェクトの名前
- **型**: String

### modelFileName
- **説明**: USDZモデルファイル名
- **型**: String

### position
- **説明**: オブジェクトのローカル座標原点のワールド座標位置
- **単位**: メートル

### forward, up, right
- **説明**: オブジェクトのローカル座標系の3軸方向ベクトル
- **forward**: オブジェクトのZ軸（前方向）
- **up**: オブジェクトのY軸（上方向）
- **right**: オブジェクトのX軸（右方向）

### rotation
- **説明**: オブジェクトの回転（クォータニオン）

### scale
- **説明**: オブジェクトのスケール
- **型**: Vector3

## データ取得頻度
- **記録間隔**: 0.1秒（10Hz）
- **例**: 10秒間の記録で100フレーム

## ファイル命名規則
```
tracking_data_YYYYMMDD_HHmmss.json
```

例: `tracking_data_20251023_103000.json`

## 座標系について

### ワールド座標系
- **X軸**: 右方向（正の値）
- **Y軸**: 上方向（正の値）
- **Z軸**: 前方向（負の値 = ユーザーが見ている方向）
- **単位**: メートル

### クォータニオン
回転の表現に使用される4次元ベクトル {x, y, z, w}
- **w**: 実数部
- **x, y, z**: 虚数部

## 使用例

### Pythonでの読み込み
```python
import json

with open('tracking_data_20251023_103000.json', 'r') as f:
    data = json.load(f)

# フレーム数を取得
frame_count = len(data['frames'])

# 最初のフレームの左手の人差し指の先端位置を取得
first_frame = data['frames'][0]
if first_frame['leftHand']:
    for joint in first_frame['leftHand']['joints']:
        if joint['jointName'] == 'indexFingerTip':
            print(f"Position: {joint['position']}")
```

## 注意事項

1. **トラッキングの信頼性**: `isTracked`フラグがfalseの場合、そのデータの精度が低い可能性があります
2. **座標系**: すべての位置と方向はワールド座標系で記録されます
3. **null値**: 手がトラッキングされていない場合、`leftHand`または`rightHand`はnullになります
4. **ファイルサイズ**: 長時間の記録は大きなファイルサイズになる可能性があります（10秒 ≈ 数MB）
