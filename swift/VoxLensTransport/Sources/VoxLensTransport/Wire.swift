import Foundation

/// The message framing `voxlens-serve` speaks.
///
/// Four bytes of big-endian length, a UTF-8 JSON header, then the raw crops the
/// header's `bytes` field describes. The Python side is `voxlens/transport.py`,
/// and the two have to agree byte for byte — which is why the constants below
/// are restated here rather than assumed.
public enum Wire {
    public static let protocolVersion = 1

    /// The crop contract: 96x96 RGB, 25 fps. The recogniser centre-crops to 88
    /// itself, so 96 is what crosses the wire.
    public static let cropSize = 96
    public static let cropBytes = 96 * 96 * 3
    public static let targetFPS = 25.0

    public static let maxHeaderBytes = 64 * 1024
    public static let maxCropsPerMessage = 25 * 60

    public enum Mode: String {
        /// One Transcript per batch of crops.
        case clip
        /// Overlapping windows with a revision boundary (ADR-0012).
        case stream
    }
}

public enum WireError: Error, CustomStringConvertible {
    case malformed(String)
    case offContract(String)
    case server(code: String, message: String)
    case disconnected

    public var description: String {
        switch self {
        case .malformed(let detail): return "malformed message: \(detail)"
        case .offContract(let detail): return "off contract: \(detail)"
        case .server(let code, let message): return "server \(code): \(message)"
        case .disconnected: return "the server closed the connection"
        }
    }
}

/// One message, ready to be written to a socket.
public struct WireMessage {
    public let header: [String: Any]
    public let payload: Data

    public init(header: [String: Any], payload: Data = Data()) {
        self.header = header
        self.payload = payload
    }

    public var type: String { header["type"] as? String ?? "" }

    /// Encode header and payload as one buffer.
    ///
    /// One buffer rather than two writes: a header that reaches the far side
    /// ahead of its payload invites a reader to act on a message it has not
    /// fully received.
    public func encoded() throws -> Data {
        var header = self.header
        if !payload.isEmpty { header["bytes"] = payload.count }
        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        guard json.count <= Wire.maxHeaderBytes else {
            throw WireError.malformed("header of \(json.count) bytes exceeds the maximum")
        }
        var out = Data(capacity: 4 + json.count + payload.count)
        var length = UInt32(json.count).bigEndian
        withUnsafeBytes(of: &length) { out.append(contentsOf: $0) }
        out.append(json)
        out.append(payload)
        return out
    }

    /// How many bytes of payload the header says follow it.
    public static func payloadSize(of header: [String: Any]) throws -> Int {
        guard let size = header["bytes"] as? Int else { return 0 }
        guard size >= 0, size <= Wire.maxCropsPerMessage * Wire.cropBytes else {
            throw WireError.malformed("payload size \(size) is out of range")
        }
        return size
    }

    /// Decode a header from its bytes, refusing anything that is not one.
    public static func header(from data: Data) throws -> [String: Any] {
        guard
            let object = try? JSONSerialization.jsonObject(with: data),
            let header = object as? [String: Any],
            header["type"] is String
        else {
            throw WireError.malformed("every message needs a JSON object header with a type")
        }
        return header
    }

    /// Read a 4-byte big-endian length prefix.
    public static func length(from data: Data) throws -> Int {
        guard data.count == 4 else { throw WireError.malformed("a length prefix is four bytes") }
        let value = data.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).bigEndian }
        guard value > 0, value <= UInt32(Wire.maxHeaderBytes) else {
            throw WireError.malformed("header length \(value) is out of range")
        }
        return Int(value)
    }
}

/// Mouth Region crops, as the wire carries them.
public struct CropBatch {
    public let frames: Int
    public let bytes: Data

    /// - Parameter bytes: `frames` × 96 × 96 × 3 bytes of RGB, row-major.
    public init(frames: Int, bytes: Data) throws {
        guard frames > 0 else { throw WireError.offContract("no crops to send") }
        guard frames <= Wire.maxCropsPerMessage else {
            throw WireError.offContract(
                "\(frames) crops exceeds \(Wire.maxCropsPerMessage) in one message")
        }
        guard bytes.count == frames * Wire.cropBytes else {
            throw WireError.offContract(
                "\(frames) crops need \(frames * Wire.cropBytes) bytes; got \(bytes.count)")
        }
        self.frames = frames
        self.bytes = bytes
    }
}
