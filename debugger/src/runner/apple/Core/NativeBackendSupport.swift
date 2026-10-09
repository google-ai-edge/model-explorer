// Copyright 2026 The AI Edge Model Explorer Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

import CryptoKit
import Darwin
import Foundation

struct BackendLibraryEvidence: Codable, Sendable {
  let name: String
  let path: String
  let binarySHA256: String
  let sourceSHA256: String?
  let loadMode: String
}

/// Library loading evidence is separate from evidence of actual model execution.
struct BackendEvidence: Codable, Sendable {
  let requestedBackend: String
  let libraries: [BackendLibraryEvidence]
  let runtimeCommit: String?
  let acceleratorResidencyVerified: Bool
  let observedDelegate: String?
  let observedAdapter: String?
  var effectiveBackend: String? = nil
  var engineInitialized = false
}

enum NativeBackendSupport {
  private static let gpuLibraries = [
    "libwebgpu_dawn.dylib", "libLiteRtWebGpuAccelerator.dylib", "libLiteRtTopKWebGpuSampler.dylib",
  ]
  private static let lock = NSLock()
  // The LiteRT registry and sampler retain plugin code beyond any one Engine.
  // Keep handles for the Runner process lifetime instead of dlclose on Run end.
  private static var loaded: [String: UnsafeMutableRawPointer] = [:]

  private static var defaultLibraryDirectory: URL? {
    if HostBundle.facts.isAppBundle { return HostBundle.facts.privateFrameworksURL }
    // The existing native CLI smoke contract explicitly supplies this path.
    if let directory = ProcessInfo.processInfo.environment["LITERT_LM_LIBS"], !directory.isEmpty {
      return URL(fileURLWithPath: directory, isDirectory: true)
    }
    return nil
  }

  static var availableBackends: [String] {
    #if os(macOS)
      if let directory = defaultLibraryDirectory,
        gpuLibraries.allSatisfy({
          FileManager.default.isReadableFile(atPath: directory.appendingPathComponent($0).path)
        })
      {
        return ["CPU", "GPU"]
      }
    #endif
    return ["CPU"]
  }

  static func prepare(backend: String) throws -> BackendEvidence {
    let requested = backend.uppercased()
    guard requested == "CPU" || requested == "GPU" else {
      throw RunnerError("Unsupported LiteRT-LM backend: \(backend).")
    }
    let directory = defaultLibraryDirectory
    let provenanceURL =
      HostBundle.facts.runtimeBuildURL
      ?? directory?.appendingPathComponent("runtime-build.json")
    let provenance = provenanceURL.flatMap { try? Data(contentsOf: $0) }
      .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
    let commit = provenance?["litertLMCommit"] as? String
    if requested == "CPU" {
      return BackendEvidence(
        requestedBackend: requested, libraries: [], runtimeCommit: commit,
        acceleratorResidencyVerified: false, observedDelegate: nil, observedAdapter: nil)
    }
    #if os(macOS)
      guard let directory else {
        throw RunnerError("This Runner has no packaged macOS GPU libraries.")
      }
      return try lock.withLock {
        let sourceHashes = provenance?["files"] as? [String: String] ?? [:]
        let embeddedHashes = provenance?["embeddedFiles"] as? [String: String] ?? [:]
        var evidence: [BackendLibraryEvidence] = []
        // Validate every required file before loading any new plugin.
        for name in gpuLibraries {
          let path = directory.appendingPathComponent(name).standardizedFileURL
            .resolvingSymlinksInPath()
          guard FileManager.default.isReadableFile(atPath: path.path) else {
            throw RunnerError(
              "Missing macOS GPU library: \(name). Rebuild this Runner with its GPU dependencies.")
          }
          let bytes = try Data(contentsOf: path, options: .mappedIfSafe)
          let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
          if let expected = embeddedHashes[name] ?? sourceHashes[name], expected != digest {
            throw RunnerError(
              "Packaged GPU library hash differs from runtime-build.json: \(name).")
          }
          evidence.append(
            BackendLibraryEvidence(
              name: name, path: path.path, binarySHA256: digest,
              sourceSHA256: sourceHashes[name], loadMode: "RTLD_NOW | RTLD_LOCAL"))
        }
        for library in evidence where loaded[library.path] == nil {
          guard let handle = dlopen(library.path, RTLD_NOW | RTLD_LOCAL) else {
            let detail = dlerror().map { String(cString: $0) } ?? "Unknown dynamic loader error"
            throw RunnerError("Could not load macOS GPU library \(library.name): \(detail)")
          }
          loaded[library.path] = handle
        }
        return BackendEvidence(
          requestedBackend: requested, libraries: evidence, runtimeCommit: commit,
          acceleratorResidencyVerified: false, observedDelegate: nil, observedAdapter: nil)
      }
    #else
      throw RunnerError("GPU execution is not supported by this iOS Runner build.")
    #endif
  }
}
