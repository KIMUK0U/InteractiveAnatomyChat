import Foundation
import Speech
import AVFoundation

@MainActor
@Observable
class SpeechRecognitionManager {
    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    
    var lastError: String?
    var isAuthorized: Bool = false
    
    init() {
        // 変更点: 英語(米国)の音声認識を使用するように設定
        // "en-US" はアメリカ英語です。イギリス英語なら "en-GB" などに変更可能です。
        speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    }
    
    /// 音声認識の権限をリクエスト
    func requestPermission() async {
        let status = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
        
        switch status {
        case .authorized:
            isAuthorized = true
            lastError = nil
        case .denied:
            isAuthorized = false
            lastError = "音声認識の権限が拒否されました"
        case .restricted:
            isAuthorized = false
            lastError = "音声認識が制限されています"
        case .notDetermined:
            isAuthorized = false
            lastError = "音声認識の権限が未決定です"
        @unknown default:
            isAuthorized = false
            lastError = "不明なエラー"
        }
    }
    
    /// 音声認識を開始
    func startRecording(onRecognition: @escaping (String) -> Void) async {
        // 既存のタスクをキャンセル
        if recognitionTask != nil {
            await stopRecording()
        }
        
        guard isAuthorized else {
            lastError = "音声認識の権限がありません"
            return
        }
        
        do {
            // オーディオセッションの設定
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
            
            // 認識リクエストの作成
            recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
            
            guard let recognitionRequest = recognitionRequest else {
                lastError = "認識リクエストの作成に失敗しました"
                return
            }
            
            recognitionRequest.shouldReportPartialResults = true
            
            // オーディオ入力の設定
            let inputNode = audioEngine.inputNode
            let recordingFormat = inputNode.outputFormat(forBus: 0)
            
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
                recognitionRequest.append(buffer)
            }
            
            // オーディオエンジンの起動
            audioEngine.prepare()
            try audioEngine.start()
            
            // 認識タスクの開始
            recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { result, error in
                if let result = result {
                    let recognizedText = result.bestTranscription.formattedString
                    Task { @MainActor in
                        onRecognition(recognizedText)
                    }
                }
                
                if error != nil || result?.isFinal == true {
                    Task { @MainActor in
                        await self.stopRecording()
                    }
                }
            }
            
            lastError = nil
            
        } catch {
            lastError = "音声認識の開始に失敗: \(error.localizedDescription)"
            await stopRecording()
        }
    }
    
    /// 音声認識を停止
    func stopRecording() async {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        
        recognitionTask?.cancel()
        recognitionTask = nil
    }
}
