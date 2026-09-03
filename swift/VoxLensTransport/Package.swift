// swift-tools-version: 5.9
import PackageDescription

// The camera half of live capture will be an iOS app (#20). This is the part
// of it that does not need a camera: the wire it speaks to voxlens-serve.
// Kept as a package rather than buried in an app target so it can be built and
// tested on a Mac — which is the only place it can be tested at all until
// there is an app.
let package = Package(
    name: "VoxLensTransport",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [
        .library(name: "VoxLensTransport", targets: ["VoxLensTransport"]),
        .executable(name: "voxlens-probe", targets: ["voxlens-probe"]),
    ],
    targets: [
        .target(name: "VoxLensTransport"),
        // The checks live in the probe rather than an XCTest target on
        // purpose: XCTest needs a full Xcode, and this package has to stay
        // verifiable on a machine that has only the Command Line Tools —
        // which is the machine it was written on. `swift run voxlens-probe
        // selftest` is the whole suite. When the app arrives and Xcode with
        // it, these belong in XCTest.
        .executableTarget(name: "voxlens-probe", dependencies: ["VoxLensTransport"]),
    ]
)
