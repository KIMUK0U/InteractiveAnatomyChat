import SwiftUI
import Speech

struct RecordingView: View {
    @Environment(\.dismissWindow) private var dismissWindow
    @Environment(AppModel.self) private var appModel
    
    @State private var sessionManager = TrackingSessionManager()
    @State private var networkManager = NetworkManager()
    
    // 音声入力関連
    @State private var speechRecognizer = SpeechRecognitionManager()
    @State private var questionText: String = ""
    @State private var isListening: Bool = false
    
    // 読み上げマネージャー
    @State private var ttsManager = TextToSpeechManager()
    // File Exporter用
    @State private var isExporting = false
    @State private var documentToExport: TrackingJSONDocument?
    
    // 処理結果
    @State private var isProcessing = false
    @State private var processingResult: ProcessingResult?
    
    // 応答速度計測
    @State private var responseTimes: [Double] = []
    @State private var requestStartTime: Date?
    
    private var averageResponseTime: Double {
        guard !responseTimes.isEmpty else { return 0 }
        return responseTimes.reduce(0, +) / Double(responseTimes.count)
    }
    
    private var hasValidHandData: Bool {
        guard let session = sessionManager.lastCapturedSessionData,
              let frame = session.frames.first else {
            return false
        }
        return frame.leftHand != nil && frame.rightHand != nil
    }
    
    var body: some View {
        VStack(spacing: 20) {
            // タイトル
            Text("データ取得 & 分析")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            // サーバー接続状態
            serverConnectionSection
            
            // ステータス表示
            statusSection
            
            Divider()
            
            // メインコンテンツエリア（スクロール可能に）
            ScrollView {
                VStack(spacing: 20) {
                    if sessionManager.lastCapturedSessionData == nil {
                        // データ未取得時
                        captureSection
                            .padding(.top, 30)
                    } else {
                        // データ取得後
                        if hasValidHandData {
                            resultSection
                        } else {
                            errorSection
                        }
                    }
                }
                .padding()
            }
        }
        .padding(30)
        .frame(width: 700, height: 700)
        .onAppear {
            Task {
                await speechRecognizer.requestPermission()
            }
        }
        .fileExporter(
            isPresented: $isExporting,
            document: documentToExport,
            contentType: .json,
            defaultFilename: "TrackingData_\(Int(Date().timeIntervalSince1970))"
        ) { result in
            handleFileExportResult(result)
        }
    }
    
    // MARK: - サーバー接続セクション
    
    private var serverConnectionSection: some View {
        HStack(spacing: 15) {
            Text("Server:")
                .font(.headline)
            TextField("IP Address", text: $networkManager.serverIP)
                .textFieldStyle(.roundedBorder)
                .frame(width: 150)
            
            Button("接続確認") {
                Task { await networkManager.checkConnection() }
            }
            .buttonStyle(.bordered)
            
            Circle()
                .fill(networkManager.isConnected ? .green : .red)
                .frame(width: 12, height: 12)
            
            if let error = networkManager.lastError {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(1)
            }
        }
        .padding(10)
        .background(.regularMaterial)
        .cornerRadius(12)
    }
    
    // MARK: - ステータスセクション
    
    private var statusSection: some View {
        HStack {
            if sessionManager.isCapturing || isProcessing {
                ProgressView()
                    .padding(.trailing, 5)
            }
            Text(isProcessing ? "PointLLMで解析中..." : sessionManager.captureStatusMessage)
                .font(.headline)
                .foregroundStyle(isProcessing ? .blue : .primary)
        }
    }
    
    // MARK: - データ取得ボタン
    
    private var captureSection: some View {
        Button(action: {
            Task {
                let fileName = appModel.selectedModelURL?.lastPathComponent ?? "default_model.usdz"
                await sessionManager.captureSnapshot(targetEntity: appModel.currentTrackedEntity, modelFileName: fileName, markerColor: appModel.selectedMarkerColor.rawValue)
                
                appModel.captureTriggerId = UUID()
            }
        }) {
            VStack(spacing: 15) {
                Image(systemName: "camera.aperture")
                    .font(.system(size: 50))
                Text("現在の状態をキャプチャ")
                    .font(.title2)
                    .fontWeight(.semibold)
            }
            .frame(width: 280, height: 160)
            .background(sessionManager.isCapturing ? Color.gray : Color.blue)
            .foregroundStyle(.white)
            .cornerRadius(25)
            .shadow(radius: 5)
        }
        .disabled(sessionManager.isCapturing)
    }
    
    // MARK: - 結果操作セクション（データ取得成功時）
    
    private var resultSection: some View {
        VStack(spacing: 20) {
            // 1. 音声入力エリア
            voiceInputSection
            
            // 2. 解析実行ボタン
            if !questionText.isEmpty {
                Button(action: {
                    Task { await processWithPointLLM() }
                }) {
                    HStack {
                        Image(systemName: "sparkles")
                        Text("AIに質問する")
                    }
                    .font(.title3)
                    .fontWeight(.bold)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(isProcessing ? Color.gray : Color.indigo)
                    .foregroundStyle(.white)
                    .cornerRadius(15)
                }
                .disabled(isProcessing || !networkManager.isConnected)
            }
            
            // 3. 解析結果の表示
            if let result = processingResult {
                Divider()
                resultDisplaySection(result: result)
            }
            
            Divider()
            
            // 4. アクションボタン（保存・リセット）
            HStack(spacing: 20) {
                Button(action: {
                    if let session = sessionManager.lastCapturedSessionData {
                        documentToExport = TrackingJSONDocument(session: session)
                        isExporting = true
                    }
                }) {
                    Label("JSON保存", systemImage: "square.and.arrow.down")
                        .padding()
                        .background(Color.orange)
                        .foregroundStyle(.white)
                        .cornerRadius(10)
                }
                
                Button(action: { resetSession() }) {
                    Label("リセット", systemImage: "trash")
                        .padding()
                        .background(Color.gray.opacity(0.2))
                        .foregroundStyle(.primary)
                        .cornerRadius(10)
                }
            }
        }
    }
    
    // MARK: - 音声入力
    
    private var voiceInputSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("質問内容")
                .font(.caption)
                .foregroundStyle(.secondary)
            
            HStack {
                TextField("質問を入力...", text: $questionText)
                    .textFieldStyle(.roundedBorder)
                
                Button(action: {
                    Task {
                        isListening ? await stopVoiceInput() : await startVoiceInput()
                    }
                }) {
                    Image(systemName: isListening ? "stop.circle.fill" : "mic.circle.fill")
                        .font(.system(size: 30))
                        .foregroundStyle(isListening ? .red : .blue)
                }
                .buttonStyle(.plain)
            }
            
            if isListening {
                Text("聞き取っています...")
                    .font(.caption)
                    .foregroundStyle(.blue)
            }
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(10)
    }
    
    // MARK: - 結果詳細表示（応答速度計測特化版）
    
    private func resultDisplaySection(result: ProcessingResult) -> some View {
        VStack(alignment: .leading, spacing: 15) {
            
            // --- AI回答カード ---
            GroupBox {
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Image(systemName: "bubble.left.and.bubble.right.fill")
                            .foregroundStyle(.indigo)
                        Text("AI回答")
                            .font(.headline)
                        Spacer()
                        // 読み上げ/停止ボタン
                        Button(action: {
                            if ttsManager.isSpeaking {
                                ttsManager.stop()
                            } else {
                                ttsManager.speak(result.result.detected_structure)
                            }
                        }) {
                            Image(systemName: ttsManager.isSpeaking ? "stop.circle.fill" : "speaker.wave.2.circle.fill")
                                .font(.title2)
                                .foregroundStyle(ttsManager.isSpeaking ? .red : .blue)
                        }
                        .buttonStyle(.plain)
                    }
                    
                    // ここを修正: 不透明なグレー背景 + 黒文字で視認性向上
                    Text(result.result.detected_structure)
                                .font(.body)
                                .foregroundStyle(.black)
                                .padding(10)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color.white)
                                .cornerRadius(8)
                }
            } label: {
                // Empty label
            }
            // --- 応答速度表示カード（大きく目立つように）---
            GroupBox {
                VStack(spacing: 10) {
                    HStack {
                        Image(systemName: "timer")
                            .foregroundStyle(.green)
                        Text("応答速度")
                            .font(.headline)
                        Spacer()
                    }
                    
                    // 最新の応答時間
                    if let lastTime = responseTimes.last {
                        VStack(spacing: 5) {
                            Text("最新")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(String(format: "%.4f", lastTime))
                                .font(.system(size: 48, weight: .bold, design: .monospaced))
                                .foregroundStyle(.green)
                            Text("秒")
                                .font(.title3)
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green.opacity(0.1))
                        .cornerRadius(10)
                    }
                    
                    Divider()
                    
                    // 平均応答時間
                    HStack {
                        VStack(alignment: .leading, spacing: 3) {
                            Text("平均時間")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(String(format: "%.4f 秒", averageResponseTime))
                                .font(.system(size: 24, weight: .semibold, design: .monospaced))
                                .foregroundStyle(.blue)
                        }
                        
                        Spacer()
                        
                        VStack(alignment: .trailing, spacing: 3) {
                            Text("計測回数")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text("\(responseTimes.count) 回")
                                .font(.system(size: 24, weight: .semibold, design: .monospaced))
                                .foregroundStyle(.orange)
                        }
                    }
                    .padding(.horizontal, 5)
                    
                    // 履歴クリアボタン
                    if !responseTimes.isEmpty {
                        Button(action: {
                            responseTimes.removeAll()
                        }) {
                            HStack {
                                Image(systemName: "trash.circle.fill")
                                Text("履歴をクリア")
                            }
                            .font(.caption)
                            .foregroundStyle(.red)
                        }
                        .buttonStyle(.plain)
                    }
                }
            } label: {
                // Empty label
            }
            .backgroundStyle(Color.green.opacity(0.1))
            
            Divider()
            
            
            
            // --- メタデータ ---
            HStack {
                Text("点群数: \(result.result.num_points)")
                Spacer()
                if result.result.hand_tracking.right_hand_tracked {
                    Label("右手", systemImage: "hand.raised.fill").font(.caption).foregroundStyle(.green)
                }
                if result.result.hand_tracking.left_hand_tracked {
                    Label("左手", systemImage: "hand.raised.fill").font(.caption).foregroundStyle(.blue)
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }
    
    // MARK: - エラー画面
    
    private var errorSection: some View {
        VStack {
            Image(systemName: "exclamationmark.triangle")
                .font(.largeTitle)
                .foregroundStyle(.yellow)
            Text("手が検出されませんでした")
            Button("再試行") { resetSession() }
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(Color.red.opacity(0.1))
        .cornerRadius(10)
    }
    
    // MARK: - ロジック
    
    private func startVoiceInput() async {
        isListening = true
        await speechRecognizer.startRecording { text in
            questionText = text
        }
    }
    
    private func stopVoiceInput() async {
        await speechRecognizer.stopRecording()
        isListening = false
    }
    
    private func processWithPointLLM() async {
        guard let session = sessionManager.lastCapturedSessionData else { return }
        
        // 応答速度計測開始
        requestStartTime = Date()
        isProcessing = true
        
        do {
            let result = try await networkManager.sendTrackingData(session, question: questionText)
            
            // 応答速度計測終了
            if let startTime = requestStartTime {
                let responseTime = Date().timeIntervalSince(startTime)
                responseTimes.append(responseTime)
                print("Response Time: \(String(format: "%.4f", responseTime)) seconds")
            }
            
            withAnimation {
                processingResult = result
            }
            
            // 結果が返ってきたら自動で読み上げ開始
            ttsManager.speak(result.result.detected_structure)
            
        } catch {
            print("Processing Error: \(error)")
        }
        
        isProcessing = false
        requestStartTime = nil
    }
    
    private func resetSession() {
        sessionManager.lastCapturedSessionData = nil
        sessionManager.captureStatusMessage = "待機中"
        questionText = ""
        processingResult = nil
        // 応答速度履歴はクリアしない（累積）
    }
    
    private func handleFileExportResult(_ result: Result<URL, Error>) {
        // エクスポート結果のハンドリング（省略可）
    }
}

#Preview {
    RecordingView()
        .environment(AppModel())
}
