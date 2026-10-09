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

import SwiftUI

@main
struct RunnerApp: App {
  @StateObject private var model = RunnerModel()
  @Environment(\.scenePhase) private var scenePhase

  init() { HostBundle.install(HostBundleFacts(bundle: .main)) }

  var body: some Scene {
    WindowGroup(model.uiState.windowTitle) {
      RunnerView(model: model)
        .task {
          model.startTransport()
          #if DEBUG
            if ProcessInfo.processInfo.arguments.contains("--capture-smoke-test") {
              model.runCaptureFixture()
            }
          #endif
        }
        .onChange(of: scenePhase) { _, phase in
          #if os(iOS)
            if phase == .background {
              model.stop()
              model.transport.disconnect()
            }
          #endif
        }
    }
    #if os(macOS)
      .defaultSize(width: 980, height: 680)
    #endif
  }
}
