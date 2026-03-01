# ARKit Data Provider for Vision Pro

Vision ProのARKitを使用して、Hand Trackingとデバイストラッキング情報を記録するアプリケーションです。

## 機能

### 1. Hand Tracking（手のトラッキング）
- 両手の27個の関節位置をリアルタイムでトラッキング
- ワールド座標とローカル座標の両方を記録
- 各関節の回転情報（クォータニオン）を記録

### 2. Device Tracking（デバイストラッキング）
- Vision Proの位置と姿勢をリアルタイムでトラッキング
- 3軸方向ベクトル（forward, up, right）を記録
- ワールド座標系での位置を記録

### 3. 3Dオブジェクトのトラッキング
- 配置された3Dオブジェクト（USDZ）の位置を記録
- オブジェクトのローカル座標原点のワールド座標を記録
- オブジェクトの向き（3軸ベクトル）とスケールを記録

### 4. データ記録
- 0.1秒間隔（10Hz）でデータを取得
- JSON形式でエクスポート
- タイムスタンプ付きファイル名で自動保存

## ファイル構成

### 新規作成ファイル

1. **TrackingDataModel.swift**
   - データ構造の定義
   - Recording Session、Frame Data、Hand Data、Object Dataなど
   - 27個の手関節の定義
   - Vector3、Quaternionなどの数学型

2. **TrackingSessionManager.swift**
   - ARKitセッションの管理
   - Hand TrackingとWorld Trackingの制御
   - 0.1秒ごとのデータキャプチャ
   - JSON保存機能

3. **RecordingView.swift**
   - 記録用のUI画面
   - 開始/停止ボタン
   - ステータス表示
   - 保存機能

4. **JSON_DATA_STRUCTURE.md**
   - JSONファイルの詳細な構造説明
   - 保存されるデータの一覧
   - 使用例

### 更新ファイル

1. **ContentView.swift**
   - 記録画面を開くボタンを追加

2. **DataProviderApp.swift**
   - RecordingViewのウィンドウを追加

3. **ImmersiveView.swift**
   - デモ用の球体を追加

4. **Info.plist**
   - ARKitの権限を追加
   - Hand Trackingの権限を追加

## セットアップ手順

### 1. プロジェクトへのファイル追加

Xcodeで以下の手順を実行：

1. 新規作成したSwiftファイルをプロジェクトに追加：
   - `TrackingDataModel.swift`
   - `TrackingSessionManager.swift`
   - `RecordingView.swift`

2. 既存ファイルを更新：
   - `ContentView.swift`
   - `DataProviderApp.swift`
   - `ImmersiveView.swift`
   - `Info.plist`

### 2. 必要な権限の設定

Info.plistに以下の権限が追加されています：

```xml
<key>NSCameraUsageDescription</key>
<string>このアプリは、ARKitを使用して手のトラッキングとデバイスの位置情報を記録します。</string>

<key>NSHandsTrackingUsageDescription</key>
<string>このアプリは、手の動きをトラッキングしてデータを記録します。</string>

<key>NSWorldSensingUsageDescription</key>
<string>このアプリは、周囲の環境とデバイスの位置をトラッキングしてデータを記録します。</string>
```

### 3. Required Capabilitiesの設定

Xcodeのプロジェクト設定で以下を有効化：

1. **Signing & Capabilities**タブを開く
2. **+ Capability**をクリック
3. 以下を追加：
   - "ARKit"
   - "Group Activities" (必要に応じて)

### 4. Info.plistの確認

**UIRequiredDeviceCapabilities**に以下が含まれていることを確認：
- `arkit`
- `world-facing-camera`

## 使用方法

### 1. アプリの起動

1. Vision Proでアプリケーションをビルドして実行
2. メインウィンドウが表示されます

### 2. データ記録の開始

1. 「データ記録画面を開く」ボタンをクリック
2. 記録用ウィンドウが表示されます
3. 「記録開始」ボタンをクリック
4. ARKitの権限許可を求められた場合は許可してください

### 3. データの記録中

- 赤い丸のインジケーターが表示されます
- 記録フレーム数がリアルタイムで更新されます
- 0.1秒ごとにデータが取得されます

### 4. 記録の停止

1. 「記録停止」ボタンをクリック
2. 記録されたフレーム数が表示されます

### 5. データの保存

1. 「JSONで保存」ボタンをクリック
2. ファイルがDocumentsフォルダに保存されます
3. ファイル名: `tracking_data_YYYYMMDD_HHmmss.json`

### 6. ファイルの取り出し

**方法1: Finder経由（Mac接続時）**
```
Finder > デバイス > Vision Pro > ファイル > DataProvider > Documents
```

**方法2: iTunes File Sharing**
1. MacでFinder/iTunesを開く
2. Vision Proを選択
3. ファイル共有セクションでDataProviderを選択
4. Documentsフォルダからファイルをドラッグ&ドロップ

**方法3: iCloud Drive（実装が必要）**
- iCloud Driveへのエクスポート機能を追加することで、他のデバイスからアクセス可能

## JSONファイルの構造

詳細は`JSON_DATA_STRUCTURE.md`を参照してください。

### 主要なデータ

```json
{
  "sessionStartTime": "記録開始時刻",
  "sessionEndTime": "記録終了時刻",
  "metadata": {
    "deviceModel": "Apple Vision Pro",
    "osVersion": "2.0",
    "appVersion": "1.0.0"
  },
  "frames": [
    {
      "timestamp": 0.1,
      "frameNumber": 1,
      "deviceTransform": { /* Vision Proの位置と姿勢 */ },
      "leftHand": { /* 左手の27個の関節データ */ },
      "rightHand": { /* 右手の27個の関節データ */ },
      "objects": [
        { /* 3Dオブジェクトの位置情報 */ }
      ]
    }
  ]
}
```

## 記録される手の関節一覧（各手27点）

### 手首
- wrist

### 親指（4点）
- thumbKnuckle
- thumbIntermediateBase
- thumbIntermediateTip
- thumbTip

### 人差し指（5点）
- indexFingerMetacarpal
- indexFingerKnuckle
- indexFingerIntermediateBase
- indexFingerIntermediateTip
- indexFingerTip

### 中指（5点）
- middleFingerMetacarpal
- middleFingerKnuckle
- middleFingerIntermediateBase
- middleFingerIntermediateTip
- middleFingerTip

### 薬指（5点）
- ringFingerMetacarpal
- ringFingerKnuckle
- ringFingerIntermediateBase
- ringFingerIntermediateTip
- ringFingerTip

### 小指（5点）
- littleFingerMetacarpal
- littleFingerKnuckle
- littleFingerIntermediateBase
- littleFingerIntermediateTip
- littleFingerTip

### その他
- forearmArm（前腕部）

## 3Dオブジェクトの追加方法

デフォルトではデモ用の球体が配置されていますが、USDZモデルを追加することができます。

### コードでの追加例

```swift
// ImmersiveView.swiftで
if let entity = try? await Entity(named: "YourModel", in: realityKitContentBundle) {
    entity.position = SIMD3<Float>(0, 1.5, -2)
    content.add(entity)
    
    // TrackingSessionManagerに登録（記録用）
    // ※この機能を使うには、TrackingSessionManagerへのアクセスを追加する必要があります
}
```

## データ分析の例

### Pythonでの読み込み

```python
import json
import numpy as np

# JSONファイルを読み込む
with open('tracking_data_20251023_103000.json', 'r') as f:
    data = json.load(f)

# 左手の人差し指の軌跡を取得
positions = []
for frame in data['frames']:
    if frame['leftHand']:
        for joint in frame['leftHand']['joints']:
            if joint['jointName'] == 'indexFingerTip':
                pos = joint['position']
                positions.append([pos['x'], pos['y'], pos['z']])

positions = np.array(positions)

# 軌跡の可視化やデータ分析が可能
```

## トラブルシューティング

### ARKitセッションが開始しない
- Info.plistの権限設定を確認
- Vision Proの設定でアプリの権限を確認
- デバイスの再起動を試す

### 手がトラッキングされない
- 手が視界内に入っているか確認
- 照明条件を確認
- 手を大きく動かしてみる

### JSONファイルが見つからない
- Finderの「最近使った項目」を確認
- Documentsフォルダのパスを確認
- アプリを再インストールしていないか確認

## 今後の拡張案

1. **リアルタイムビジュアライゼーション**
   - 手の関節を球体で表示
   - トラッキングの軌跡を可視化

2. **データフィルタリング**
   - 特定の関節のみを記録
   - 記録頻度の調整

3. **複数オブジェクトの対応**
   - 動的にオブジェクトを追加/削除
   - オブジェクトのインタラクション記録

4. **クラウド同期**
   - iCloud Driveへの自動アップロード
   - デバイス間でのデータ共有

5. **データ分析ツール**
   - アプリ内でのプレビュー機能
   - 統計情報の表示

## ライセンス

このプロジェクトは教育・研究目的で作成されています。

## 作成者

木村亘汰
2025年10月23日
