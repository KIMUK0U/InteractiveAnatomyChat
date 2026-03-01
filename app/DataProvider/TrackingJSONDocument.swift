import SwiftUI
import UniformTypeIdentifiers

// JSON書き出し用のドキュメント定義
struct TrackingJSONDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.json] }
    
    var session: RecordingSession
    
    init(session: RecordingSession) {
        self.session = session
    }
    
    init(configuration: ReadConfiguration) throws {
        // 読み込みは今回は不要だがプロトコル要件のため実装
        let data = try configuration.file.regularFileContents
        self.session = try JSONDecoder().decode(RecordingSession.self, from: data!)
    }
    
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(session)
        return FileWrapper(regularFileWithContents: data)
    }
}
