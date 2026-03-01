// NetworkManager.swift
import Foundation

@MainActor
class NetworkManager: ObservableObject {
    @Published var serverIP: String = "10.1.199.116" // Mac miniのIPアドレス
    @Published var isConnected: Bool = false
    @Published var lastError: String?
    
    private let port = 8000
    
    // サーバーの疎通確認
    func checkConnection() async -> Bool {
        guard let url = URL(string: "http://\(serverIP):\(port)/api/health") else {
            return false
        }
        
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            let response = try JSONDecoder().decode(HealthResponse.self, from: data)
            isConnected = response.status == "ok"
            lastError = nil // 成功したらエラーをクリア
            return isConnected
        } catch {
            lastError = "接続エラー: \(error.localizedDescription)"
            isConnected = false
            return false
        }
    }
    
    // トラッキングデータの送信（修正版）
    func sendTrackingData(_ session: RecordingSession, question: String) async throws -> ProcessingResult {
        guard let url = URL(string: "http://\(serverIP):\(port)/api/process_tracking") else {
            throw NetworkError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 60 // PointLLM処理のため長めに設定
        
        // 1. ラッパー構造体を作成（これが422エラー対策の肝です）
        let trackingRequest = TrackingRequest(session: session, question: question)
        
        // 2. エンコーダー設定（日付形式の指定が必須）
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        
        do {
            request.httpBody = try encoder.encode(trackingRequest)
        } catch {
            throw NetworkError.encodingError
        }
        
        // 3. 送信とレスポンス受信
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw NetworkError.serverError
        }
        
        // エラーハンドリングの強化: 422などが返ってきた場合に詳細をログに出す
        if !(200...299).contains(httpResponse.statusCode) {
            if let errorString = String(data: data, encoding: .utf8) {
                print("❌ Server Error (\(httpResponse.statusCode)): \(errorString)")
            }
            throw NetworkError.serverError
        }
        
        // 4. 結果のデコード
        do {
            let result = try JSONDecoder().decode(ProcessingResult.self, from: data)
            return result
        } catch {
            print("❌ Decode Error: \(error)")
            // デバッグ用に生のレスポンスを表示
            if let str = String(data: data, encoding: .utf8) {
                print("Raw response: \(str)")
            }
            throw error
        }
    }
}

// MARK: - Data Structures

struct HealthResponse: Codable {
    let status: String
    let server: String
}

enum NetworkError: Error {
    case invalidURL
    case serverError
    case encodingError
}

// リクエスト用ラッパー（サーバーの期待する形）
struct TrackingRequest: Codable {
    let session: RecordingSession
    let question: String
}

// レスポンス型
struct ProcessingResult: Codable {
    let status: String
    let processed_at: String
    let result: PointLLMResult
}

// 分析結果詳細（サーバーの更新に合わせて拡張）
struct PointLLMResult: Codable {
    let detected_structure: String
    let question: String
    let point_cloud_file: String
    let num_points: Int
    let hand_tracking: HandTrackingInfo
    
    // ★追加: サーバー側で追加した分析結果を受け取る用
    let interaction_analysis: InteractionAnalysis?
}

struct HandTrackingInfo: Codable {
    let left_hand_tracked: Bool
    let right_hand_tracked: Bool
}

// MARK: - Interaction Analysis Structures
// サーバーの interaction_analyzer.py が返すJSONに対応

struct InteractionAnalysis: Codable {
    let total_hand_points: Int
    let total_pc_points: Int
    let target_color: String
    let closest_interaction: InteractionResult?
    let hand_point_analysis: [HandPointAnalysis]
}

struct InteractionResult: Codable {
    let pc_index: Int
    let distance_mm: Float
    let hand_point: [Float] // [x, y, z]
    let colorized_points: Int
}

struct HandPointAnalysis: Codable {
    let hand_point_index: Int
    let distance_to_nearest: Float
    let neighbor_statistics: NeighborStatistics
}

struct NeighborStatistics: Codable {
    let total_neighbors: Int
    let statistics: [SubclassStatistic]
}

struct SubclassStatistic: Codable {
    let rank: Int
    let subclass_id: Int
    let subclass_name: String
    let class_id: Int
    let class_name: String
    let count: Int
    let ratio: Float
}
