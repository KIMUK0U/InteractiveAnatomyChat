//
//  ImmersiveView.swift
//  DataProvider
//
//  Hand Tracking Based 3D Object Manipulation
//

import SwiftUI
import RealityKit
import RealityKitContent
import ARKit
import UniformTypeIdentifiers

struct ImmersiveView: View {
    @Environment(AppModel.self) private var appModel
    @State private var loadedEntity: Entity?
    
    @State private var currentRotation: simd_quatf = simd_quatf(angle: 0, axis: [0, 1, 0])
    @State private var gestureRotation: simd_quatf = simd_quatf(angle: 0, axis: [0, 1, 0])
    
    // ARKit Session for Hand Tracking
    @State private var arSession: ARKitSession?
    @State private var handTracking: HandTrackingProvider?
    @State private var worldTracking: WorldTrackingProvider?
    
    // Session Manager for Data Capture
    @State private var sessionManager = TrackingSessionManager()
    
    // Device position for billboard effect
    @State private var devicePosition: SIMD3<Float> = SIMD3<Float>(0, 1.6, 0)
    
    // Hand tracking for object manipulation
    @State private var isGrabbing = false
    @State private var grabStartHandPosition: SIMD3<Float>?
    @State private var grabStartObjectPosition: SIMD3<Float>?
    @State private var currentHandPosition: SIMD3<Float>?
    
    // Background task for hand tracking
    @State private var handTrackingTask: Task<Void, Never>?
    
    // マーカー管理用（単一マーカーに変更）
    @State private var markerEntity: Entity?
    
    var body: some View {
        ZStack {
            RealityView { content, attachments in
                // ルートアンカー
                let anchorEntity = Entity()
                anchorEntity.name = "UserFileAnchor"
                anchorEntity.position = SIMD3<Float>(0, 1.0, -1)
                content.add(anchorEntity)
                self.loadedEntity = anchorEntity
                
                // 初回表示時、もし既にURLが選択されていれば読み込む
                if let url = appModel.selectedModelURL {
                    await loadUserSelectedModel(url: url, parent: anchorEntity)
                }
                
                // ARKitセッション開始
                await startTracking()
                
                // HandTrackingの継続的な監視を開始
                startHandTrackingTask()
                
            } update: { content, attachments in
                // モデルの回転を更新
                updateModelRotation(in: content)
                
                // オブジェクト位置を更新（@Stateは変更しない）
                if let currentHandPos = currentHandPosition,
                   let anchorEntity = content.entities.first,
                   isGrabbing,
                   let startHandPos = grabStartHandPosition,
                   let startObjPos = grabStartObjectPosition {
                    
                    let handDelta = currentHandPos - startHandPos
                    let newPosition = startObjPos + handDelta
                    anchorEntity.position = newPosition
                }
            } attachments: {
                // Attachmentなし（3Dボタンを削除）
            }
            .gesture(
                RotateGesture3D()
                    .targetedToAnyEntity()
                    .onChanged { value in
                        handleRotationChanged(value)
                    }
                    .onEnded { value in
                        handleRotationEnded(value)
                    }
            )
            
            // ステータス表示（3D空間上部に浮かぶUI）
            VStack {
                if isGrabbing {
                    Text("✊ つかんでいます")
                        .font(.title3)
                        .padding()
                        .background(.green.opacity(0.8))
                        .foregroundStyle(.white)
                        .cornerRadius(12)
                        .padding(.top, 50)
                }
                
                Spacer()
            }
        }
        .onChange(of: appModel.selectedModelURL) { _, newURL in
            guard let url = newURL, let parent = loadedEntity else { return }
            Task {
                parent.children.removeAll()
                currentRotation = simd_quatf(angle: 180, axis: [0, 1, 0])
                appModel.currentTrackedEntity = nil
                
                await loadUserSelectedModel(url: url, parent: parent)
            }
        }
        .onChange(of: appModel.captureTriggerId) { _, _ in
            Task {
                await updateMarkerPosition()
            }
        }
        .onChange(of: appModel.clearMarkersTrigger) { _, _ in
            clearMarker()
        }
        .onDisappear {
            appModel.currentTrackedEntity = nil
            handTrackingTask?.cancel()
            arSession?.stop()
        }
    }
    
    // MARK: - Tracking Setup
    
    private func startTracking() async {
        do {
            print("🔵 ARKitセッション初期化開始...")
            arSession = ARKitSession()
            handTracking = HandTrackingProvider()
            worldTracking = WorldTrackingProvider()
            
            print("🔵 権限リクエスト中...")
            let authorizationResult = await arSession?.requestAuthorization(for: [.handTracking, .worldSensing])
            
            guard let authResult = authorizationResult,
                  authResult[.handTracking] == .allowed,
                  authResult[.worldSensing] == .allowed else {
                print("⚠️ Tracking not authorized")
                return
            }
            
            print("🔵 Tracking実行中...")
            try await arSession?.run([handTracking!, worldTracking!])
            
            // セッションが完全に起動するまで少し待つ
            try await Task.sleep(nanoseconds: 500_000_000)
            
            // デバイス位置の継続的な更新を開始
            Task {
                await updateDevicePositionContinuously()
            }
            
            print("✅ Tracking started and ready")
            
        } catch {
            print("❌ Failed to start tracking: \(error)")
        }
    }
    
    // MARK: - Update Device Position
    
    private func updateDevicePositionContinuously() async {
        while !Task.isCancelled {
            if let worldTracking = worldTracking {
                let deviceAnchor = worldTracking.queryDeviceAnchor(atTimestamp: CACurrentMediaTime())
                if let anchor = deviceAnchor {
                    let position = SIMD3<Float>(
                        anchor.originFromAnchorTransform.columns.3.x,
                        anchor.originFromAnchorTransform.columns.3.y,
                        anchor.originFromAnchorTransform.columns.3.z
                    )
                    
                    await MainActor.run {
                        self.devicePosition = position
                    }
                }
            }
            
            try? await Task.sleep(nanoseconds: 16_670_000) // 60FPS
        }
    }
    
    // MARK: - Gesture Handlers
    
    private func handleRotationChanged(_ value: EntityTargetValue<RotateGesture3D.Value>) {
        let rotation = value.rotation
        var quat = simd_quatf(rotation)
        
        // X軸とZ軸の回転を反転
        quat = simd_quatf(
            ix: -quat.imag.x,  // X軸反転
            iy: quat.imag.y,   // Y軸はそのまま
            iz: -quat.imag.z,  // Z軸反転
            r: quat.real
        )
        
        gestureRotation = quat
    }
    
    private func handleRotationEnded(_ value: EntityTargetValue<RotateGesture3D.Value>) {
        let rotation = value.rotation
        var quat = simd_quatf(rotation)
        
        // X軸とZ軸の回転を反転
        quat = simd_quatf(
            ix: -quat.imag.x,  // X軸反転
            iy: quat.imag.y,   // Y軸はそのまま
            iz: -quat.imag.z,  // Z軸反転
            r: quat.real
        )
        
        currentRotation = quat * currentRotation
        gestureRotation = simd_quatf(angle: 0, axis: [0,1,0])
    }
    
    // MARK: - Update Model Transform
    
    private func updateModelRotation(in content: RealityViewContent) {
        guard let anchorEntity = content.entities.first else { return }
        guard let pivotEntity = anchorEntity.children.first(where: { $0.name == "RotationPivot" }) else { return }
        
        // 回転を適用
        let finalRotation = gestureRotation * currentRotation
        pivotEntity.transform.rotation = finalRotation
    }
    
    // MARK: - Hand Tracking Task
    
    private func startHandTrackingTask() {
        handTrackingTask?.cancel()
        
        handTrackingTask = Task { @MainActor in
            while !Task.isCancelled {
                guard let handTracking = handTracking,
                      let anchorEntity = loadedEntity else {
                    try? await Task.sleep(nanoseconds: 16_670_000)
                    continue
                }
                
                let latestAnchors = handTracking.latestAnchors
                let leftAnchor = latestAnchors.leftHand
                let rightAnchor = latestAnchors.rightHand
                
                // 両手が検出されているか確認
                let leftTracked = leftAnchor?.isTracked ?? false
                let rightTracked = rightAnchor?.isTracked ?? false
                
                if leftTracked && rightTracked,
                   let leftAnchor = leftAnchor,
                   let rightAnchor = rightAnchor,
                   let leftSkeleton = leftAnchor.handSkeleton,
                   let rightSkeleton = rightAnchor.handSkeleton {
                    
                    // ピンチ判定（親指と人差し指の距離）
                    let leftPinching = isPinching(skeleton: leftSkeleton, anchor: leftAnchor)
                    let rightPinching = isPinching(skeleton: rightSkeleton, anchor: rightAnchor)
                    
                    // 両手でピンチしている場合のみつかむ
                    if leftPinching && rightPinching {
                        // 両手の手首位置を取得
                        let leftWrist = leftSkeleton.joint(.wrist)
                        let rightWrist = rightSkeleton.joint(.wrist)
                        
                        let leftWristTransform = leftAnchor.originFromAnchorTransform * leftWrist.anchorFromJointTransform
                        let rightWristTransform = rightAnchor.originFromAnchorTransform * rightWrist.anchorFromJointTransform
                        
                        let leftWristPos = SIMD3<Float>(
                            leftWristTransform.columns.3.x,
                            leftWristTransform.columns.3.y,
                            leftWristTransform.columns.3.z
                        )
                        let rightWristPos = SIMD3<Float>(
                            rightWristTransform.columns.3.x,
                            rightWristTransform.columns.3.y,
                            rightWristTransform.columns.3.z
                        )
                        
                        // 両手の中間点を計算
                        let handPos = (leftWristPos + rightWristPos) / 2.0
                        currentHandPosition = handPos
                        
                        if !isGrabbing {
                            // つかみ開始
                            isGrabbing = true
                            grabStartHandPosition = handPos
                            grabStartObjectPosition = anchorEntity.position
                            print("✊ ピンチでつかみ開始: 手の位置 = \(handPos)")
                        }
                    } else {
                        // ピンチしていない場合、つかみ解除
                        if isGrabbing {
                            print("🖐 ピンチ解除")
                            isGrabbing = false
                            grabStartHandPosition = nil
                            grabStartObjectPosition = nil
                            currentHandPosition = nil
                        }
                    }
                    
                } else {
                    // 片手または両手とも検出されていない場合、つかみ解除
                    if isGrabbing {
                        print("🖐 手が見えなくなったのでつかみ解除")
                        isGrabbing = false
                        grabStartHandPosition = nil
                        grabStartObjectPosition = nil
                        currentHandPosition = nil
                    }
                }
                
                // 60FPSで更新
                try? await Task.sleep(nanoseconds: 16_670_000)
            }
        }
    }
    
    // MARK: - Pinch Detection
    
    private func isPinching(skeleton: HandSkeleton, anchor: HandAnchor) -> Bool {
        // 親指の先端
        let thumbTip = skeleton.joint(.thumbTip)
        let thumbTipTransform = anchor.originFromAnchorTransform * thumbTip.anchorFromJointTransform
        let thumbTipPos = SIMD3<Float>(
            thumbTipTransform.columns.3.x,
            thumbTipTransform.columns.3.y,
            thumbTipTransform.columns.3.z
        )
        
        // 人差し指の先端
        let indexTip = skeleton.joint(.indexFingerTip)
        let indexTipTransform = anchor.originFromAnchorTransform * indexTip.anchorFromJointTransform
        let indexTipPos = SIMD3<Float>(
            indexTipTransform.columns.3.x,
            indexTipTransform.columns.3.y,
            indexTipTransform.columns.3.z
        )
        
        // 2点間の距離を計算
        let distance = simd_distance(thumbTipPos, indexTipPos)
        
        // 閾値: 3cm以下ならピンチしていると判定
        let pinchThreshold: Float = 0.03
        
        return distance < pinchThreshold
    }
    
    // MARK: - Hand Data Capture Helper
    
    private func captureHandData(from anchor: HandAnchor, chirality: HandChirality) -> HandData {
        let handTransform = HandTransform(
            position: Vector3(simd: SIMD3<Float>(
                anchor.originFromAnchorTransform.columns.3.x,
                anchor.originFromAnchorTransform.columns.3.y,
                anchor.originFromAnchorTransform.columns.3.z
            )),
            rotation: Quaternion(simd: simd_quatf(anchor.originFromAnchorTransform))
        )
        
        var joints: [JointData] = []
        
        if let skeleton = anchor.handSkeleton {
            for jointName in HandJointName.allCases {
                if let joint = getHandSkeletonJoint(skeleton: skeleton, jointName: jointName) {
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
    
    // MARK: - Model Loading
    
    private func loadUserSelectedModel(url: URL, parent: Entity) async {
        do {
            let modelEntity = try await Entity(contentsOf: url)
            modelEntity.name = "ImportedModel"
            
            let bounds = modelEntity.visualBounds(relativeTo: nil)
            let center = bounds.center
            
            let pivotEntity = Entity()
            pivotEntity.name = "RotationPivot"
            
            modelEntity.position = -center
            
            pivotEntity.addChild(modelEntity)
            
            modelEntity.generateCollisionShapes(recursive: true)
            modelEntity.components.set(InputTargetComponent())
            
            parent.addChild(pivotEntity)
            
            appModel.currentTrackedEntity = pivotEntity
            
            print("✅ モデル読み込み成功: 重心補正完了 (Center: \(center))")
            
        } catch {
            print("❌ モデル読み込み失敗: \(error)")
        }
    }
    
    // MARK: - Marker Management (単一マーカー制御に変更)
    
    @MainActor
    private func updateMarkerPosition() async {
        guard let modelEntity = appModel.currentTrackedEntity,
              let handTracking = handTracking else { return }
        
        // 1. すべての手の関節点を収集
        var allJointPositions: [(SIMD3<Float>, String)] = []
        
        let latestAnchors = handTracking.latestAnchors
        
        // 左手の全関節を取得
        if let leftHand = latestAnchors.leftHand,
           leftHand.isTracked,
           let skeleton = leftHand.handSkeleton {
            for jointName in HandJointName.allCases {
                if let joint = getHandSkeletonJoint(skeleton: skeleton, jointName: jointName),
                   joint.isTracked {
                    let jointTransform = leftHand.originFromAnchorTransform * joint.anchorFromJointTransform
                    let worldPos = SIMD3<Float>(
                        jointTransform.columns.3.x,
                        jointTransform.columns.3.y,
                        jointTransform.columns.3.z
                    )
                    allJointPositions.append((worldPos, "Left_\(jointName)"))
                }
            }
        }
        
        // 右手の全関節を取得
        if let rightHand = latestAnchors.rightHand,
           rightHand.isTracked,
           let skeleton = rightHand.handSkeleton {
            for jointName in HandJointName.allCases {
                if let joint = getHandSkeletonJoint(skeleton: skeleton, jointName: jointName),
                   joint.isTracked {
                    let jointTransform = rightHand.originFromAnchorTransform * joint.anchorFromJointTransform
                    let worldPos = SIMD3<Float>(
                        jointTransform.columns.3.x,
                        jointTransform.columns.3.y,
                        jointTransform.columns.3.z
                    )
                    allJointPositions.append((worldPos, "Right_\(jointName)"))
                }
            }
        }
        
        guard !allJointPositions.isEmpty else {
            print("⚠️ 手の関節が検出されていません")
            return
        }
        
        // 2. オブジェクトの中心位置（ワールド座標）を取得
        let bounds = modelEntity.visualBounds(relativeTo: nil)
        let objectCenter = bounds.center
        
        // 3. オブジェクト中心に最も近い関節点を探す
        var closestJoint: (position: SIMD3<Float>, name: String, distance: Float)?
        
        for (jointPos, jointName) in allJointPositions {
            let dist = distance(jointPos, objectCenter)
            
            if closestJoint == nil || dist < closestJoint!.distance {
                closestJoint = (jointPos, jointName, dist)
            }
        }
        
        guard let closest = closestJoint else {
            print("⚠️ 最近傍関節が見つかりませんでした")
            return
        }
        
        // 4. マーカーを作成または更新
        if markerEntity == nil {
            // 初回作成
            let marker = await createMarkerEntity()
            markerEntity = marker
            modelEntity.addChild(marker)
            print("✅ マーカー初回作成")
        }
        
        // 5. マーカーの位置を更新
        if let marker = markerEntity {
            marker.setPosition(closest.position, relativeTo: nil)
            
            // マーカーの色を更新
            let color = appModel.selectedMarkerColor.uiColor
            let material = SimpleMaterial(color: color, isMetallic: false)
            applyMaterialRecursively(entity: marker, material: material)
            
            print("✅ マーカー位置更新 (\(appModel.selectedMarkerColor.label)): \(closest.name)")
        }
    }
    
    private func createMarkerEntity() async -> Entity {
        // マーカーの色を取得
        let color = appModel.selectedMarkerColor.uiColor
        let material = SimpleMaterial(color: color, isMetallic: false)
        
        // "star.usdz" をロード
        if let starEntity = try? await Entity(named: "star") {
            starEntity.transform.rotation = simd_quatf(angle: .pi, axis: SIMD3<Float>(0, 0, 1))
            applyMaterialRecursively(entity: starEntity, material: material)
            starEntity.name = "InteractionMarker"
            print("✅ star.usdzをロードしました")
            return starEntity
        } else {
            // フォールバック: 球体
            let radius: Float = 0.005
            let mesh = MeshResource.generateSphere(radius: radius)
            let sphereEntity = ModelEntity(mesh: mesh, materials: [material])
            sphereEntity.name = "InteractionMarker"
            print("✅ フォールバック球体を作成しました")
            return sphereEntity
        }
    }
    
    // エンティティとその子孫にマテリアルを再帰的に適用
    private func applyMaterialRecursively(entity: Entity, material: RealityKit.Material) {
        if let modelEntity = entity as? ModelEntity {
            modelEntity.model?.materials = [material]
        }
        for child in entity.children {
            applyMaterialRecursively(entity: child, material: material)
        }
    }
    
    @MainActor
    private func clearMarker() {
        markerEntity?.removeFromParent()
        markerEntity = nil
        print("🗑️ マーカーを削除しました")
    }
}

// MARK: - Gesture Extension

extension simd_quatf {
    init(_ rotation: Rotation3D) {
        let quat = rotation.quaternion
        self.init(ix: Float(quat.vector.x), iy: Float(quat.vector.y), iz: Float(quat.vector.z), r: Float(quat.vector.w))
    }
}
