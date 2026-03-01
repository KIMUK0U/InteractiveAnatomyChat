//
//  TrackingSessionManager.swift
//  DataProvider
//
//  ARKit Hand Tracking and Device Tracking Session Manager (Snapshot Version)
//

import Foundation
import ARKit
import RealityKit
import UIKit

@MainActor
@Observable
class TrackingSessionManager {
    
    // MARK: - Properties
    
    private var arSession: ARKitSession?
    private var handTracking: HandTrackingProvider?
    private var worldTracking: WorldTrackingProvider?
    
    // 状態管理
    var isCapturing = false
    var captureStatusMessage = "待機中"
    
    // 最後に取得したデータ（保存用）
    var lastCapturedSessionData: RecordingSession?
    
    // MARK: - Initialization
    init() {
        // 初期化処理（必要に応じて）
    }
    
    // MARK: - Snapshot Capture
    
    /// 1フレームだけ取得して終了する
    /// - Parameter targetEntity: ImmersiveViewで操作中のEntity
    // MARK: - Snapshot Capture
    func captureSnapshot(targetEntity: Entity?, modelFileName: String, markerColor: String) async {
            guard !isCapturing else { return }
            isCapturing = true
            captureStatusMessage = "ARセッション初期化中..."
            
            do {
                // 1. セッション初期化
                arSession = ARKitSession()
                handTracking = HandTrackingProvider()
                worldTracking = WorldTrackingProvider()
                
                // 2. 権限リクエスト
                let authorizationResult = await arSession?.requestAuthorization(for: [.handTracking, .worldSensing])
                
                guard let authResult = authorizationResult,
                      authResult[.handTracking] == .allowed,
                      authResult[.worldSensing] == .allowed else {
                    captureStatusMessage = "エラー: AR権限がありません"
                    isCapturing = false
                    return
                }
                
                // 3. 実行
                try await arSession?.run([handTracking!, worldTracking!])
                captureStatusMessage = "手を認識中..."
                
                // 4. ウォームアップと手の検出待機 (ポーリング処理)
                // 最大2秒間、0.1秒ごとにチェックして、どちらかの手が isTracked になるのを待つ
                let timeout = Date().addingTimeInterval(2.0)
                var handsDetected = false
                
                while Date() < timeout {
                    if let handTracking = self.handTracking {
                        let anchors = handTracking.latestAnchors
                        let leftTracked = anchors.leftHand?.isTracked ?? false
                        let rightTracked = anchors.rightHand?.isTracked ?? false
                        
                        if leftTracked || rightTracked {
                            handsDetected = true
                            print("✅ 手を検出しました (Wait time: \(2.0 - timeout.timeIntervalSinceNow)s)")
                            break
                        }
                    }
                    // まだ見つからない場合は0.1秒待つ
                    try await Task.sleep(nanoseconds: 100_000_000)
                }
                
                if !handsDetected {
                    print("⚠️ 警告: 手が検出されないままタイムアウトしました")
                }
                
                captureStatusMessage = "データ保存処理中..."
                
                // 5. データ取得 (現在の瞬間をキャプチャ)
                let timestamp = Date()
                
                // 最新のデバイス位置
                let deviceAnchor = worldTracking?.queryDeviceAnchor(atTimestamp: CACurrentMediaTime())
                let deviceTransformData = deviceAnchor != nil ? DeviceTransform(from: deviceAnchor!.originFromAnchorTransform) : nil
                
                // 最新の手の情報
                let (leftHand, rightHand) = await getCurrentHandData()
                
                // デバッグ出力
                print("📊 Left Hand: \(leftHand != nil ? "あり" : "なし")")
                print("📊 Right Hand: \(rightHand != nil ? "あり" : "なし")")
                
                // 3Dモデルのワールド座標を取得
                var objectsData: [ObjectData] = []
                if let entity = targetEntity {
                    // ワールド座標系での行列を取得 (relativeTo: nil)
                    let worldMatrix = entity.transformMatrix(relativeTo: nil)
                    
                    let objData = ObjectData(
                        matrix: worldMatrix,
                        objectID: "UserTargetModel",
                        objectName: entity.name,
                        modelFileName: modelFileName
                    )
                    objectsData.append(objData)
                }
                
                // 6. フレームデータ作成
                let frame = FrameData(
                    timestamp: 0.0,
                    frameNumber: 1,
                    deviceTransform: deviceTransformData,
                    leftHand: leftHand,
                    rightHand: rightHand,
                    objects: objectsData
                )
                
                // 7. セッションデータ作成
                let sessionData = RecordingSession(
                    sessionStartTime: timestamp,
                    sessionEndTime: timestamp,
                    frames: [frame], // 1フレームのみ
                    metadata: SessionMetadata(
                        deviceModel: "Apple Vision Pro",
                        osVersion: UIDevice.current.systemVersion,
                        appVersion: "1.0.0",
                        markerColor: markerColor
                    )
                )
                
                self.lastCapturedSessionData = sessionData
                captureStatusMessage = "取得完了"
                
                // 8. 停止
                arSession?.stop()
                arSession = nil
                
            } catch {
                captureStatusMessage = "エラー: \(error.localizedDescription)"
                print("Error: \(error)")
            }
            
            isCapturing = false
        }
    
    // 手の最新データを取得するヘルパー（修正済み）
    private func getCurrentHandData() async -> (HandData?, HandData?) {
        guard let handTracking = handTracking else { return (nil, nil) }
        
        // ★ 修正箇所: タプルから直接プロパティへアクセス
        let latestAnchors = handTracking.latestAnchors
        let leftAnchor = latestAnchors.leftHand
        let rightAnchor = latestAnchors.rightHand
        
        let leftData = (leftAnchor != nil && leftAnchor!.isTracked) ? captureHandData(from: leftAnchor!, chirality: .left) : nil
        let rightData = (rightAnchor != nil && rightAnchor!.isTracked) ? captureHandData(from: rightAnchor!, chirality: .right) : nil
        
        return (leftData, rightData)
    }
    
    private func captureHandData(from anchor: HandAnchor, chirality: HandChirality) -> HandData {
        let handTransform = HandTransform(
            position: Vector3(simd: SIMD3<Float>(
                anchor.originFromAnchorTransform.columns.3.x,
                anchor.originFromAnchorTransform.columns.3.y,
                anchor.originFromAnchorTransform.columns.3.z
            )),
            rotation: Quaternion(simd: simd_quatf(anchor.originFromAnchorTransform))
        )
        
        // Capture all joints
        var joints: [JointData] = []
        
        for jointName in HandJointName.allCases {
            if let skeleton = anchor.handSkeleton,
               let joint = getHandSkeletonJoint(skeleton: skeleton, jointName: jointName) {
                
                // ワールド座標系での関節位置 = 手のアンカー位置 * 関節の相対位置
                let jointTransform = anchor.originFromAnchorTransform * joint.anchorFromJointTransform
                
                let jointData = JointData(
                    jointName: jointName,
                    position: Vector3(simd: SIMD3<Float>(
                        jointTransform.columns.3.x,
                        jointTransform.columns.3.y,
                        jointTransform.columns.3.z
                    )),
                    rotation: Quaternion(simd: simd_quatf(jointTransform)),
                    isTracked: joint.isTracked,
                    localPosition: Vector3(simd: SIMD3<Float>(
                        joint.anchorFromJointTransform.columns.3.x,
                        joint.anchorFromJointTransform.columns.3.y,
                        joint.anchorFromJointTransform.columns.3.z
                    )),
                    localRotation: Quaternion(simd: simd_quatf(joint.anchorFromJointTransform))
                )
                
                joints.append(jointData)
            }
        }
        
        return HandData(
            isTracked: anchor.isTracked,
            chirality: chirality,
            handTransform: handTransform,
            joints: joints
        )
    }
    
    private func getHandSkeletonJoint(skeleton: HandSkeleton, jointName: HandJointName) -> HandSkeleton.Joint? {
        let arkitJointName: HandSkeleton.JointName
        
        switch jointName {
        case .wrist: arkitJointName = .wrist
        case .thumbKnuckle: arkitJointName = .thumbKnuckle
        case .thumbIntermediateBase: arkitJointName = .thumbIntermediateBase
        case .thumbIntermediateTip: arkitJointName = .thumbIntermediateTip
        case .thumbTip: arkitJointName = .thumbTip
        case .indexFingerMetacarpal: arkitJointName = .indexFingerMetacarpal
        case .indexFingerKnuckle: arkitJointName = .indexFingerKnuckle
        case .indexFingerIntermediateBase: arkitJointName = .indexFingerIntermediateBase
        case .indexFingerIntermediateTip: arkitJointName = .indexFingerIntermediateTip
        case .indexFingerTip: arkitJointName = .indexFingerTip
        case .middleFingerMetacarpal: arkitJointName = .middleFingerMetacarpal
        case .middleFingerKnuckle: arkitJointName = .middleFingerKnuckle
        case .middleFingerIntermediateBase: arkitJointName = .middleFingerIntermediateBase
        case .middleFingerIntermediateTip: arkitJointName = .middleFingerIntermediateTip
        case .middleFingerTip: arkitJointName = .middleFingerTip
        case .ringFingerMetacarpal: arkitJointName = .ringFingerMetacarpal
        case .ringFingerKnuckle: arkitJointName = .ringFingerKnuckle
        case .ringFingerIntermediateBase: arkitJointName = .ringFingerIntermediateBase
        case .ringFingerIntermediateTip: arkitJointName = .ringFingerIntermediateTip
        case .ringFingerTip: arkitJointName = .ringFingerTip
        case .littleFingerMetacarpal: arkitJointName = .littleFingerMetacarpal
        case .littleFingerKnuckle: arkitJointName = .littleFingerKnuckle
        case .littleFingerIntermediateBase: arkitJointName = .littleFingerIntermediateBase
        case .littleFingerIntermediateTip: arkitJointName = .littleFingerIntermediateTip
        case .littleFingerTip: arkitJointName = .littleFingerTip
        case .forearmArm: arkitJointName = .forearmArm
        }
        
        return skeleton.joint(arkitJointName)
    }
}

// MARK: - Extensions for ObjectData

extension ObjectData {
    // ワールド行列からデータを生成するイニシャライザ
    init(matrix: simd_float4x4, objectID: String, objectName: String, modelFileName: String) {
        self.objectID = objectID
        self.objectName = objectName
        self.modelFileName = modelFileName
        
        // 位置
        self.position = Vector3(simd: SIMD3<Float>(matrix.columns.3.x, matrix.columns.3.y, matrix.columns.3.z))
        
        // 回転
        self.rotation = Quaternion(simd: simd_quatf(matrix))
        
        // スケール
        let scaleX = simd_length(SIMD3<Float>(matrix.columns.0.x, matrix.columns.0.y, matrix.columns.0.z))
        let scaleY = simd_length(SIMD3<Float>(matrix.columns.1.x, matrix.columns.1.y, matrix.columns.1.z))
        let scaleZ = simd_length(SIMD3<Float>(matrix.columns.2.x, matrix.columns.2.y, matrix.columns.2.z))
        self.scale = Vector3(x: scaleX, y: scaleY, z: scaleZ)
        
        // 方向ベクトル (正規化)
        if scaleX > 0 && scaleY > 0 && scaleZ > 0 {
            self.right = Vector3(simd: SIMD3<Float>(matrix.columns.0.x, matrix.columns.0.y, matrix.columns.0.z) / scaleX)
            self.up = Vector3(simd: SIMD3<Float>(matrix.columns.1.x, matrix.columns.1.y, matrix.columns.1.z) / scaleY)
            self.forward = Vector3(simd: SIMD3<Float>(matrix.columns.2.x, matrix.columns.2.y, matrix.columns.2.z) / scaleZ)
        } else {
            self.right = Vector3(x: 1, y: 0, z: 0)
            self.up = Vector3(x: 0, y: 1, z: 0)
            self.forward = Vector3(x: 0, y: 0, z: 1)
        }
    }
}
