import Foundation
import VoxLensTransport

/// The framing checks, as a runnable program rather than an XCTest target.
///
/// Whether the two implementations actually agree is not a question either
/// side's unit tests can answer. That is what running this against a live
/// `voxlens-serve` is for; these only cover the arithmetic of the frame.
enum Checks {
    static func run() -> Int32 {
        var failures: [String] = []

        func expect(_ condition: Bool, _ description: String) {
            if !condition { failures.append(description) }
        }
        func expectThrows(_ description: String, _ body: () throws -> Void) {
            do {
                try body()
                failures.append(description)
            } catch {}
        }

        do {
            let payload = Data(repeating: 7, count: 10)
            let encoded = try WireMessage(header: ["type": "crops", "frames": 1],
                                          payload: payload).encoded()
            let length = try WireMessage.length(from: encoded.prefix(4))
            let header = try WireMessage.header(from: encoded.dropFirst(4).prefix(length))

            expect(header["type"] as? String == "crops", "a message keeps its type")
            expect(try WireMessage.payloadSize(of: header) == 10,
                   "the header declares its payload size")
            expect(encoded.count == 4 + length + payload.count,
                   "a message is length, header, payload and nothing else")

            let bye = try WireMessage(header: ["type": "bye"]).encoded()
            let byeLength = try WireMessage.length(from: bye.prefix(4))
            let byeHeader = try WireMessage.header(from: bye.dropFirst(4).prefix(byeLength))
            expect(try WireMessage.payloadSize(of: byeHeader) == 0,
                   "a message without a payload declares none")

            let batch = try CropBatch(frames: 3,
                                      bytes: Data(repeating: 1, count: 3 * Wire.cropBytes))
            expect(batch.frames == 3, "a batch counts its frames")
            expect(Wire.cropBytes == 96 * 96 * 3, "the crop contract is 96x96 RGB")
        } catch {
            failures.append("the happy path threw: \(error)")
        }

        expectThrows("garbage is not read as a header") {
            _ = try WireMessage.header(from: Data("not json".utf8))
        }
        expectThrows("a header without a type is refused") {
            _ = try WireMessage.header(from: Data("{\"no\":\"type\"}".utf8))
        }
        expectThrows("a zero length prefix is refused") {
            _ = try WireMessage.length(from: Data([0, 0, 0, 0]))
        }
        // A phone shipping the wrong shape should hear about it where the bug is.
        expectThrows("crops that do not match their frame count are refused") {
            _ = try CropBatch(frames: 2, bytes: Data(repeating: 0, count: 100))
        }
        expectThrows("an empty batch is refused") {
            _ = try CropBatch(frames: 0, bytes: Data())
        }

        if failures.isEmpty {
            print("selftest: all checks passed")
            return 0
        }
        for failure in failures { print("selftest: FAILED — \(failure)") }
        return 1
    }
}
