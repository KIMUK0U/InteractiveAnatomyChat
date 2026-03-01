import SwiftUI
import RealityKit

// マーカーの色定義用Enum
enum MarkerColorType: String, CaseIterable, Identifiable, Codable {
    case red
    case blue
    case black
    case white // Highlightとして使用
    
    var id: Self { self }
    
    var uiColor: UIColor {
        switch self {
        case .red: return .red
        case .blue: return .blue
        case .black: return .black
        case .white: return .white
        }
    }
    
    var label: String {
        switch self {
        case .red: return "Red"
        case .blue: return "Blue"
        case .black: return "Black"
        case .white: return "Highlight (White)"
        }
    }
}

@MainActor
@Observable
class AppModel {
    let immersiveSpaceID = "ImmersiveSpace"
    
    enum ImmersiveSpaceState {
        case closed
        case inTransition
        case open
    }
    
    var immersiveSpaceState = ImmersiveSpaceState.closed
    var selectedModelURL: URL?
    
    var currentTrackedEntity: Entity?
    
    // ★ 変更: 単なるUIColorではなく、Enumで管理
    var selectedMarkerColor: MarkerColorType = .red
    
    var captureTriggerId: UUID? = nil
    var clearMarkersTrigger: UUID? = nil
}
