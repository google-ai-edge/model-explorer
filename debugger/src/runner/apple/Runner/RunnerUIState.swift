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

import Foundation

/// Portable display data. Credentials, file contents and native handles stay in the host.
struct RunnerUIState: Encodable {
  let version = 1
  struct Device: Encodable {
    let name: String
    let platform: String
    let foregroundRequired: Bool
  }
  struct Connection: Encodable {
    let phase: String
    let detail: String
    let transport: String
    let address: String?
  }
  struct Activity: Encodable {
    let phase: String
    let detail: String
    let output: String
    let capturedTensors: Int?
    let transfer: Transfer?
  }
  typealias Transfer = RunnerTransferProgress
  typealias Session = RunnerSessionSummary
  struct Interface: Encodable {
    let address: String
    let label: String
  }
  struct Manual: Encodable {
    let hasJob: Bool
    let hasModel: Bool
    let prompt: String?
    let modelHash: String?
    let capturePoints: Int?
    let outputLimit: Int?
  }
  struct Lifecycle: Encodable { let phase: String }
  struct Build: Encodable {
    let id: String?
    let label: String
    let version: String?
    let cl: String?
  }
  let device: Device
  let connection: Connection
  let activity: Activity
  let sessions: [Session]
  let interfaces: [Interface]
  let manual: Manual
  /// Named actions expose actual host capabilities and current ownership restrictions.
  let actions: [String: Bool]
  var runtimes: [RunnerRuntimeCapability] = []
  let environment: RunnerDeviceEnvironment
  let environmentState: RunnerRuntimeState
  var owner: RunnerOwner? = nil
  var lifecycle: Lifecycle = .init(phase: "waiting")
  var build: Build = .init(id: nil, label: "Unknown build", version: nil, cl: nil)
  /// Execution slot stays ref/target on the wire; target is presented as Debug.
  var slot: String? = nil

  var windowTitle: String {
    let role = owner?.runID ?? slot
    let label = role == "ref" ? "Ref Runner" : role == "target" ? "Debug Runner" : "Runner"
    return "Model Debugger · " + label
  }
}
