import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// The camera side of the crop transport: crops out, Transcripts back.
///
/// Deliberately a plain socket rather than `NWConnection`. This has to be
/// verifiable on a Mac with no Xcode and no device, which is the only place it
/// can be exercised until the app exists (#20); the app should reconsider,
/// because `NWConnection` is what knows about cellular, path changes and the
/// phone going to sleep — none of which a POSIX socket notices.
///
/// Blocking, and meant to be driven from a background queue. The camera thread
/// must never wait on the Mac.
public final class CropClient {
    public let session: String
    public let fps: Double
    public let mode: Wire.Mode

    private var descriptor: Int32 = -1
    private var incoming = Data()
    public private(set) var ready: [String: Any] = [:]

    public init(session: String = "swift-\(UUID().uuidString.prefix(8))",
                fps: Double = Wire.targetFPS,
                mode: Wire.Mode = .clip) {
        self.session = session
        self.fps = fps
        self.mode = mode
    }

    deinit { closeSocket() }

    /// Connect and say hello. Returns the server's ready message.
    @discardableResult
    public func open(host: String = "127.0.0.1", port: UInt16) throws -> [String: Any] {
        descriptor = socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { throw WireError.malformed("could not open a socket") }

        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = port.bigEndian
        guard inet_pton(AF_INET, host, &address.sin_addr) == 1 else {
            closeSocket()
            throw WireError.malformed("\(host) is not an address this client understands")
        }
        let connected = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard connected == 0 else {
            closeSocket()
            throw WireError.malformed("nothing answered at \(host):\(port)")
        }

        try write(WireMessage(header: [
            "type": "hello",
            "protocol": Wire.protocolVersion,
            "session": session,
            "fps": fps,
            "mode": mode.rawValue,
        ]))
        ready = try expect("ready")
        return ready
    }

    /// Send one batch of crops and wait for the reply.
    @discardableResult
    public func send(_ batch: CropBatch) throws -> [String: Any] {
        try write(WireMessage(header: ["type": "crops", "frames": batch.frames],
                              payload: batch.bytes))
        return try expect("transcript")
    }

    /// Report Frames the camera could not read, instead of sending noise.
    ///
    /// Whatever was in front of the lens would be read as mouth movement, and
    /// invented text is the one thing this project refuses to produce. In a
    /// Stream session this also cuts the decoding window.
    @discardableResult
    public func reportOcclusion(frames: Int) throws -> [String: Any] {
        guard frames > 0 else {
            throw WireError.offContract("an Occlusion covers at least one Frame")
        }
        try write(WireMessage(header: ["type": "occlusion", "frames": frames]))
        return try expect("transcript")
    }

    /// Say goodbye. A Stream session gets one last reading back: the tail no
    /// window had covered, and whatever was still provisional, settled.
    @discardableResult
    public func close() -> [String: Any]? {
        guard descriptor >= 0 else { return nil }
        var final: [String: Any]?
        try? write(WireMessage(header: ["type": "bye"]))
        if mode == .stream { final = try? expect("transcript") }
        closeSocket()
        return final
    }

    // MARK: - the socket

    private func closeSocket() {
        if descriptor >= 0 { Darwin.close(descriptor) }
        descriptor = -1
    }

    private func write(_ message: WireMessage) throws {
        let data = try message.encoded()
        try data.withUnsafeBytes { buffer in
            var sent = 0
            while sent < buffer.count {
                let wrote = Darwin.send(descriptor, buffer.baseAddress! + sent,
                                        buffer.count - sent, 0)
                guard wrote > 0 else { throw WireError.disconnected }
                sent += wrote
            }
        }
    }

    private func read(_ count: Int) throws -> Data {
        var chunk = [UInt8](repeating: 0, count: 64 * 1024)
        while incoming.count < count {
            let got = recv(descriptor, &chunk, chunk.count, 0)
            guard got > 0 else { throw WireError.disconnected }
            incoming.append(contentsOf: chunk[0..<got])
        }
        let taken = Data(incoming.prefix(count))
        incoming = Data(incoming.dropFirst(count))
        return taken
    }

    private func receive() throws -> (header: [String: Any], payload: Data) {
        let length = try WireMessage.length(from: try read(4))
        let header = try WireMessage.header(from: try read(length))
        let size = try WireMessage.payloadSize(of: header)
        return (header, size > 0 ? try read(size) : Data())
    }

    private func expect(_ kind: String) throws -> [String: Any] {
        let (header, _) = try receive()
        if header["type"] as? String == "error" {
            throw WireError.server(code: header["code"] as? String ?? "error",
                                   message: header["message"] as? String ?? "no detail")
        }
        guard header["type"] as? String == kind else {
            throw WireError.malformed("expected a \(kind), got \(header["type"] ?? "nothing")")
        }
        return header
    }
}
