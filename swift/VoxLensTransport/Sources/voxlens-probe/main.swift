import Foundation
import VoxLensTransport

// Drives a running voxlens-serve from Swift, so the phone half of the protocol
// can be exercised before there is a phone. Sends synthetic crops: this proves
// the wire, never the reading.
//
//   voxlens-probe <port> [clip|stream] [batches] [framesPerBatch]

let arguments = CommandLine.arguments
if arguments.count > 1, arguments[1] == "selftest" {
    exit(Checks.run())
}
guard arguments.count > 1, let port = UInt16(arguments[1]) else {
    let usage = "usage: voxlens-probe <port> [clip|stream] [batches] [frames]\n"
        + "       voxlens-probe selftest\n"
    FileHandle.standardError.write(Data(usage.utf8))
    exit(2)
}
let mode = Wire.Mode(rawValue: arguments.count > 2 ? arguments[2] : "clip") ?? .clip
let batches = arguments.count > 3 ? Int(arguments[3]) ?? 1 : 1
let frames = arguments.count > 4 ? Int(arguments[4]) ?? 25 : 25

func syntheticCrops(_ count: Int, seed: UInt8) -> Data {
    var data = Data(capacity: count * Wire.cropBytes)
    for frame in 0..<count {
        let value = UInt8((Int(seed) &+ frame * 7) % 251)
        data.append(Data(repeating: value, count: Wire.cropBytes))
    }
    return data
}

let client = CropClient(session: "swift-probe", mode: mode)
do {
    let ready = try client.open(port: port)
    let crop = ready["crop"] as? [String: Any] ?? [:]
    print("ready: mode=\(ready["mode"] ?? "?") crop=\(crop["size"] ?? "?") fps=\(ready["fps"] ?? "?")")

    for index in 0..<batches {
        let batch = try CropBatch(frames: frames, bytes: syntheticCrops(frames, seed: UInt8(index % 200)))
        let reply = try client.send(batch)
        if mode == .stream {
            let windows = (reply["windows"] as? [[String: Any]])?.count ?? 0
            print("batch \(index + 1): \(windows) window(s)  frozen=\"\(reply["frozen"] ?? "")\"  provisional=\"\(reply["provisional"] ?? "")\"")
        } else {
            print("batch \(index + 1): \(reply["transcript"] ?? "")")
        }
    }
    if let final = client.close() {
        print("final: \(final["text"] ?? "")")
    }
} catch {
    FileHandle.standardError.write(Data("voxlens-probe: \(error)\n".utf8))
    exit(1)
}
