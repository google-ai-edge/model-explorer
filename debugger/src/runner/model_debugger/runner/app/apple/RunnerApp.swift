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

import SwiftUI

#if os(macOS)
  import AppKit
#else
  import UIKit
#endif

/// The Runner's App, on macOS and on iOS.
///
/// On macOS it has one window, and closing it quits the App. The Runner stops
/// when the App terminates; an iOS App the system kills while suspended
/// leaves its run directory behind, which the next Runner removes.
@main
struct RunnerApp: App {
  @State private var runner = RunnerModel()
  #if os(iOS)
    /// Whether the App is in front, where it keeps the screen on.
    @Environment(\.scenePhase) private var phase
  #endif

  var body: some Scene {
    #if os(macOS)
      Window("Model Debugger Runner", id: "runner") {
        page.frame(minWidth: 360, minHeight: 480)
      }
    #else
      WindowGroup { page }
    #endif
  }

  private var page: some View {
    RunnerView(state: runner.state, failure: runner.failure)
      .onReceive(NotificationCenter.default.publisher(for: Self.willTerminate)) { _ in
        runner.stop()
      }
      #if os(iOS)
        // The screen stays on while the page shows, as on Android: the system
        // suspends an App that is not in front, and a suspended Runner answers
        // nothing while its page still reads as serving. A Mac does not
        // suspend an App behind its screen saver.
        .onChange(of: phase, initial: true) { _, phase in
          UIApplication.shared.isIdleTimerDisabled = phase == .active
        }
      #endif
  }

  #if os(macOS)
    private static let willTerminate = NSApplication.willTerminateNotification
  #else
    private static let willTerminate = UIApplication.willTerminateNotification
  #endif
}
