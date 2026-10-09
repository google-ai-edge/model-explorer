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

#if os(iOS)
  import UIKit
#endif

// Display projection only; execution/ownership rules live in RunnerModel.
extension RunnerModel {
  var uiState: RunnerUIState {
    #if os(macOS)
      let device = RunnerUIState.Device(
        name: Host.current().localizedName ?? "This Mac", platform: "macOS",
        foregroundRequired: false)
      let transport = listenHost == "127.0.0.1" ? "This Mac" : "Thunderbolt"
      let address: String? = listenHost + ":" + String(RunnerPlatform.listenPort)
      let interfaces = listenAddresses.map { RunnerUIState.Interface(address: $0.0, label: $0.1) }
      let slot: String? = RunnerPlatform.runnerSlot
    #else
      let device = RunnerUIState.Device(
        name: UIDevice.current.name, platform: "iOS", foregroundRequired: true)
      let transport = "USB"
      let address: String? = nil
      let interfaces: [RunnerUIState.Interface] = []
      let slot: String? = nil
    #endif
    return RunnerUIState(
      device: device,
      connection: .init(
        phase: connectionPhase.rawValue, detail: connectionStatus, transport: transport,
        address: address),
      activity: .init(
        phase: activity.rawValue, detail: status, output: output, capturedTensors: captureCount,
        transfer: transfer),
      sessions: residentSummaries.sorted { $0.key < $1.key }, interfaces: interfaces,
      manual: .init(
        hasJob: job != nil, hasModel: modelURL != nil, prompt: job?.prompt,
        modelHash: job?.modelSHA256,
        capturePoints: job?.manifest.taps.count, outputLimit: job?.maxOutputTokens),
      actions: allowedActions,
      runtimes: runtimeCapabilities, environment: RunnerEnvironment.device,
      environmentState: environmentState,
      owner: owner, lifecycle: .init(phase: ownership.lifecycle.rawValue),
      build: build, slot: slot)
  }

}
