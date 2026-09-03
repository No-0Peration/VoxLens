# VoxLensTransport

The half of the iOS capture app ([#20](https://github.com/No-0Peration/VoxLens/issues/20))
that does not need a camera: the wire it speaks to `voxlens-serve`.

The interesting problems in #20 are camera problems — holding a mouth in frame at
5× zoom while a hand shakes, tracking a face someone tapped, choosing a lens. None
of them can be built or verified without Xcode and a device. The protocol can, and
getting it wrong later would be an expensive way to discover a byte-order mistake,
so it is here, built and run against the real server.

## What it is

```swift
let client = CropClient(session: "seminar", mode: .stream)
try client.open(host: "192.168.1.20", port: 9601)

let batch = try CropBatch(frames: 25, bytes: crops)   // 96×96 RGB, 25 fps
let reply = try client.send(batch)
print(reply["frozen"] ?? "", reply["provisional"] ?? "")

try client.reportOcclusion(frames: 12)   // the mouth was lost; send no crops for it
let settled = client.close()
```

`Wire.swift` is the framing and the crop contract; `CropClient.swift` is the
session. Both mirror `src/voxlens/transport.py`, and the two have to agree byte
for byte.

## Running it

```bash
swift run voxlens-probe selftest          # the framing checks
swift run voxlens-probe 9601 clip 2 40    # against a running voxlens-serve
swift run voxlens-probe 9601 stream 6 25  # windowed decoding, ADR-0012
```

The probe sends synthetic crops. It proves the wire and never the reading.

Verified against a live `voxlens-serve` on the real checkpoint: both session
modes, hello through goodbye, including the Stream tail that arrives after the
goodbye.

## Two decisions the app should revisit

**A POSIX socket, not `NWConnection`.** This had to be verifiable on a Mac with
Command Line Tools and no Xcode, which a socket is and `NWConnection` mostly is
not. The app should reconsider: `NWConnection` is what knows about cellular, path
changes and a phone going to sleep, and a socket notices none of it.

**Checks in the probe, not XCTest.** XCTest needs a full Xcode. When the app
arrives and Xcode with it, `Checks.run()` belongs in a test target.
