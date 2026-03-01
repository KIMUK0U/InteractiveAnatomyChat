import SwiftUI
import RealityKit
import RealityKitContent
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(\.openWindow) private var openWindow
    @Environment(AppModel.self) private var appModel

    var body: some View {
        VStack(spacing: 30) {
            Text("ARKit Data Provider")
                .font(.largeTitle)
                .fontWeight(.bold)

            Text("Hand Tracking & Device Tracking")
                .font(.title3)
                .foregroundStyle(.secondary)
            
            // ★ 追加: マーカーの色選択
            VStack(alignment: .leading, spacing: 10) {
                Text("Marker Color")
                    .font(.headline)
                    .foregroundStyle(.secondary)
                
                Picker("Marker Color", selection: Bindable(appModel).selectedMarkerColor) {
                    ForEach(MarkerColorType.allCases) { colorType in
                        Text(colorType.label)
                            .tag(colorType)
                    }
                }
                .pickerStyle(.segmented)
            }
            .padding(.horizontal)
            .padding(.bottom, 10)
            
            VStack(spacing: 15) {
                Button(action: {
                    openWindow(id: "recording")
                }) {
                    Label("データ記録画面を開く", systemImage: "record.circle")
                        .font(.title3)
                        .fontWeight(.semibold)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .foregroundStyle(.white)
                        .cornerRadius(15)
                }
                
                FileImportButton()
                
                ToggleImmersiveSpaceButton()
            }
            .padding(.horizontal)
        }
        .padding(40)
    }
}

// (FileImportButton と #Preview は変更なしのため省略)
struct FileImportButton: View {
    @Environment(AppModel.self) private var appModel
    @State private var isImporting: Bool = false
    
    var body: some View {
        Button(action: {
            isImporting = true
        }) {
            Label(appModel.selectedModelURL == nil ? "USDZファイルを選択" : "Change Model",
                  systemImage: "folder.badge.plus")
            .font(.title3)
            .fontWeight(.semibold)
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color.orange)
            .foregroundStyle(.white)
            .cornerRadius(15)
        }
        .fileImporter(
            isPresented: $isImporting,
            allowedContentTypes: [UTType.usdz],
            allowsMultipleSelection: false
        ){ result in
            switch result {
            case .success(let urls):
                guard let url = urls.first else { return }
                if url.startAccessingSecurityScopedResource() {
                    Task {@MainActor in
                        appModel.selectedModelURL = url
                        print("📂 ファイルを選択しました: \(url.lastPathComponent)")
                    }
                }else {
                    print("❌ ファイルへのアクセス権限が取得できませんでした")
                }
                
            case .failure(let error):
                print("❌ ファイル選択エラー: \(error.localizedDescription)")
            }
        }
        // 選択されたファイル名の表示（オプション）
        if let url = appModel.selectedModelURL {
            Text("選択中: \(url.lastPathComponent)")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}
#Preview(windowStyle: .automatic) {
    ContentView()
        .environment(AppModel())
}
