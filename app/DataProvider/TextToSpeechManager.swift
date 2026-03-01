import Foundation
import AVFoundation

@MainActor
@Observable
class TextToSpeechManager {
    private let synthesizer = AVSpeechSynthesizer()
    
    // 読み上げ中かどうか
    var isSpeaking: Bool {
        synthesizer.isSpeaking
    }
    
    func speak(_ text: String, language: String = "en-US") {
        // 既に話している場合は止める
        stop()
        
        let utterance = AVSpeechUtterance(string: text)
        
        // PointLLMは英語で返すのでデフォルトは英語設定
        utterance.voice = AVSpeechSynthesisVoice(language: language)
        utterance.rate = 0.5 // 読み上げ速度 (0.0 - 1.0)
        utterance.pitchMultiplier = 1.0 // ピッチ
        
        synthesizer.speak(utterance)
    }
    
    func stop() {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
    }
}//
//  TextToSpeechManager.swift
//  DataProvider
//
//  Created by MAC mini on 2026/01/09.
//

