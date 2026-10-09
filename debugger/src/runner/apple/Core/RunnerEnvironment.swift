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

import Darwin
import Foundation
import os

/// Portable records. Values describe the Runner process/device, never the server.
struct RunnerDeviceEnvironment: Codable, Sendable {
  let capturedAt: String
  let platform: String
  let operatingSystem: String
  let architecture: String
  let modelIdentifier: String?
  let chip: String?
  let physicalMemoryBytes: UInt64
  let logicalProcessorCount: Int
  let appVersion: String?
  let appBuild: String?
  let runtimeCommit: String?
  let runtimeSourceDiffSHA256: String?
}

struct RunnerRuntimeState: Codable, Sendable {
  let capturedAt: String
  let thermalState: String?
  let lowPowerMode: Bool
  /// Process memory headroom, not system-wide free RAM; unavailable on macOS.
  let processMemoryHeadroomBytes: UInt64?
  let processResidentBytes: UInt64?
}

struct RunnerCapabilities: Encodable, Sendable {
  let runtime = "LiteRT-LM"
  let backends = NativeBackendSupport.availableBackends
  let modalities = ["text"]
  let persistentConversation = true
  let tensorCapture: Bool
  let contextLengths = CaptureJob.supportedContextLengths
  let maxOutputTokens = 32
  let maxCapturePoints = 16
}

struct RunnerSessionEnvironment: Encodable, Sendable {
  let modelSHA256: String
  var backend = "CPU"
  let contextLength: Int
  let capturePoints: Int
  let threadPolicy = "runtimeDefault"
  let sampler = ["topK": 1.0, "topP": 1.0, "temperature": 0.0, "seed": 0.0]
  let thinkingEnabled = false
  let speculativeDecodingEnabled = false
}

struct RunnerCaptureEnvironment: Encodable, Sendable {
  let version = 1
  let device: RunnerDeviceEnvironment
  let capabilities: RunnerCapabilities
  let session: RunnerSessionEnvironment
  let maxOutputTokens: Int
  let before: RunnerRuntimeState
  let after: RunnerRuntimeState
}

enum RunnerEnvironment {
  static func timestamp() -> String { ISO8601DateFormatter().string(from: Date()) }

  private static func string(_ key: String) -> String? {
    var size = 0
    guard sysctlbyname(key, nil, &size, nil, 0) == 0, size > 1, size < 4096 else { return nil }
    var buffer = [CChar](repeating: 0, count: size)
    guard sysctlbyname(key, &buffer, &size, nil, 0) == 0 else { return nil }
    return String(cString: buffer)
  }

  static let device: RunnerDeviceEnvironment = {
    #if targetEnvironment(simulator)
      let platform = "iOS Simulator"
    #elseif os(iOS)
      let platform = "iOS"
    #else
      let platform = "macOS"
    #endif
    #if arch(arm64)
      let architecture = "arm64"
    #elseif arch(x86_64)
      let architecture = "x86_64"
    #else
      let architecture = "unknown"
    #endif
    #if os(macOS)
      let model = string("hw.model")
      let chip = string("machdep.cpu.brand_string")
    #else
      let model = string("hw.machine")
      let chip: String? = nil  // No guessed chip-to-device lookup table.
    #endif
    let facts = HostBundle.facts
    let build = facts.runtimeBuildURL
      .flatMap { try? Data(contentsOf: $0) }
      .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
    return RunnerDeviceEnvironment(
      capturedAt: timestamp(), platform: platform,
      operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
      architecture: architecture, modelIdentifier: model, chip: chip,
      physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
      logicalProcessorCount: ProcessInfo.processInfo.processorCount,
      appVersion: facts.appVersion,
      appBuild: facts.appBuild,
      runtimeCommit: build?["litertLMCommit"] as? String,
      runtimeSourceDiffSHA256: build?["nativeSourceDiffSHA256"] as? String)
  }()

  static func sample() -> RunnerRuntimeState {
    let thermal: String?
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: thermal = "nominal"
    case .fair: thermal = "fair"
    case .serious: thermal = "serious"
    case .critical: thermal = "critical"
    @unknown default: thermal = nil
    }
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(
      MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
      }
    }
    #if os(iOS) && !targetEnvironment(simulator)
      let available = UInt64(os_proc_available_memory())
      let headroom: UInt64? = available > 0 ? available : nil
    #else
      let headroom: UInt64? = nil
    #endif
    return RunnerRuntimeState(
      capturedAt: timestamp(), thermalState: thermal,
      lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
      processMemoryHeadroomBytes: headroom,
      processResidentBytes: result == KERN_SUCCESS ? UInt64(info.resident_size) : nil)
  }
}
