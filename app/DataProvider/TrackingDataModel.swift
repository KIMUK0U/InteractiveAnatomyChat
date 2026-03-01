//
//  TrackingDataModel.swift
//  DataProvider
//
//  Created for ARKit Hand Tracking and Device Tracking
//

import Foundation
import ARKit
import RealityKit

// MARK: - Recording Session Data

struct RecordingSession: Codable {
    let sessionStartTime: Date
    let sessionEndTime: Date?
    let frames: [FrameData]
    let metadata: SessionMetadata
}

struct SessionMetadata: Codable {
    let deviceModel: String
    let osVersion: String
    let appVersion: String
    
    // ★ 追加: マーカーの色情報だけ保存
    let markerColor: String
}

// MARK: - Frame Data (0.1秒ごとのデータ)

struct FrameData: Codable {
    let timestamp: TimeInterval
    let frameNumber: Int
    
    // Vision Proの位置と姿勢
    let deviceTransform: DeviceTransform?
    
    // 手のトラッキングデータ
    let leftHand: HandData?
    let rightHand: HandData?
    
    // 3Dオブジェクトのデータ
    let objects: [ObjectData]
}

// MARK: - Device Transform (Vision Proの自己位置とベクトル)

struct DeviceTransform: Codable {
    // ワールド座標での位置
    let position: Vector3
    
    // 姿勢の3軸ベクトル
    let forward: Vector3  // Z軸（前方向）
    let up: Vector3       // Y軸（上方向）
    let right: Vector3    // X軸（右方向）
    
    // クォータニオン（代替表現）
    let rotation: Quaternion
}

// MARK: - Hand Data

struct HandData: Codable {
    let isTracked: Bool
    let chirality: HandChirality
    
    // 手のアンカーのワールド座標での位置と姿勢
    let handTransform: HandTransform?
    
    // 各関節の情報
    let joints: [JointData]
}

enum HandChirality: String, Codable {
    case left = "left"
    case right = "right"
}

struct HandTransform: Codable {
    let position: Vector3
    let rotation: Quaternion
}

// MARK: - Joint Data (手の関節データ)

struct JointData: Codable {
    let jointName: HandJointName
    let position: Vector3          // ワールド座標での位置
    let rotation: Quaternion       // ワールド座標での回転
    let isTracked: Bool
    
    // 親関節からの相対位置（オプション）
    let localPosition: Vector3?
    let localRotation: Quaternion?
}

// ARKitの全手関節名
enum HandJointName: String, Codable, CaseIterable {
    // 手首
    case wrist = "wrist"
    
    // 親指 (Thumb)
    case thumbKnuckle = "thumbKnuckle"           // 手首に近い関節
    case thumbIntermediateBase = "thumbIntermediateBase"
    case thumbIntermediateTip = "thumbIntermediateTip"
    case thumbTip = "thumbTip"
    
    // 人差し指 (Index)
    case indexFingerMetacarpal = "indexFingerMetacarpal"
    case indexFingerKnuckle = "indexFingerKnuckle"
    case indexFingerIntermediateBase = "indexFingerIntermediateBase"
    case indexFingerIntermediateTip = "indexFingerIntermediateTip"
    case indexFingerTip = "indexFingerTip"
    
    // 中指 (Middle)
    case middleFingerMetacarpal = "middleFingerMetacarpal"
    case middleFingerKnuckle = "middleFingerKnuckle"
    case middleFingerIntermediateBase = "middleFingerIntermediateBase"
    case middleFingerIntermediateTip = "middleFingerIntermediateTip"
    case middleFingerTip = "middleFingerTip"
    
    // 薬指 (Ring)
    case ringFingerMetacarpal = "ringFingerMetacarpal"
    case ringFingerKnuckle = "ringFingerKnuckle"
    case ringFingerIntermediateBase = "ringFingerIntermediateBase"
    case ringFingerIntermediateTip = "ringFingerIntermediateTip"
    case ringFingerTip = "ringFingerTip"
    
    // 小指 (Little)
    case littleFingerMetacarpal = "littleFingerMetacarpal"
    case littleFingerKnuckle = "littleFingerKnuckle"
    case littleFingerIntermediateBase = "littleFingerIntermediateBase"
    case littleFingerIntermediateTip = "littleFingerIntermediateTip"
    case littleFingerTip = "littleFingerTip"
    
    // 手のひら中央
    case forearmArm = "forearmArm"
}

// MARK: - Object Data (3Dオブジェクトの位置情報)

struct ObjectData: Codable {
    let objectID: String
    let objectName: String
    let modelFileName: String  // USDZファイル名
    
    // ワールド座標でのオブジェクトの原点位置
    let position: Vector3
    
    // オブジェクトの3軸ベクトル
    let forward: Vector3  // Z軸（前方向）
    let up: Vector3       // Y軸（上方向）
    let right: Vector3    // X軸（右方向）
    
    // 回転（クォータニオン）
    let rotation: Quaternion
    
    // スケール
    let scale: Vector3
}

// MARK: - Basic Math Types

struct Vector3: Codable {
    let x: Float
    let y: Float
    let z: Float
    
    init(x: Float, y: Float, z: Float) {
        self.x = x
        self.y = y
        self.z = z
    }
    
    init(simd: SIMD3<Float>) {
        self.x = simd.x
        self.y = simd.y
        self.z = simd.z
    }
}

struct Quaternion: Codable {
    let x: Float
    let y: Float
    let z: Float
    let w: Float
    
    init(x: Float, y: Float, z: Float, w: Float) {
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    }
    
    init(simd: simd_quatf) {
        self.x = simd.imag.x
        self.y = simd.imag.y
        self.z = simd.imag.z
        self.w = simd.real
    }
}

// MARK: - Extension for Transform Conversion

extension DeviceTransform {
    init(from transform: simd_float4x4) {
        // 位置を抽出
        self.position = Vector3(simd: SIMD3<Float>(transform.columns.3.x, 
                                                     transform.columns.3.y, 
                                                     transform.columns.3.z))
        
        // 3軸ベクトルを抽出
        self.right = Vector3(simd: SIMD3<Float>(transform.columns.0.x, 
                                                  transform.columns.0.y, 
                                                  transform.columns.0.z))
        self.up = Vector3(simd: SIMD3<Float>(transform.columns.1.x, 
                                               transform.columns.1.y, 
                                               transform.columns.1.z))
        self.forward = Vector3(simd: SIMD3<Float>(transform.columns.2.x, 
                                                    transform.columns.2.y, 
                                                    transform.columns.2.z))
        
        // クォータニオンに変換
        self.rotation = Quaternion(simd: simd_quatf(transform))
    }
}

extension ObjectData {
    init(entity: Entity, objectID: String, objectName: String, modelFileName: String) {
        self.objectID = objectID
        self.objectName = objectName
        self.modelFileName = modelFileName
        
        let transform = entity.transform.matrix
        
        // 位置を抽出
        self.position = Vector3(simd: SIMD3<Float>(transform.columns.3.x, 
                                                     transform.columns.3.y, 
                                                     transform.columns.3.z))
        
        // 3軸ベクトルを抽出
        self.right = Vector3(simd: SIMD3<Float>(transform.columns.0.x, 
                                                  transform.columns.0.y, 
                                                  transform.columns.0.z))
        self.up = Vector3(simd: SIMD3<Float>(transform.columns.1.x, 
                                               transform.columns.1.y, 
                                               transform.columns.1.z))
        self.forward = Vector3(simd: SIMD3<Float>(transform.columns.2.x, 
                                                    transform.columns.2.y, 
                                                    transform.columns.2.z))
        
        // 回転とスケールを抽出
        self.rotation = Quaternion(simd: simd_quatf(transform))
        self.scale = Vector3(simd: entity.scale)
    }
}
