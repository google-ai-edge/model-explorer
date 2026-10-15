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

import Foundation
import ModelDebuggerRunnerC
import Observation

/// The Runner behind the window.
///
/// It starts with the model, on `--listen HOST:PORT` from the App's arguments
/// or 127.0.0.1:8769, the port a USB bridge forwards, and is asked for its
/// state twice a second until stopped.
@MainActor
@Observable
final class RunnerModel {
  /// What the Runner last reported; nil until it reports.
  private(set) var state: RunnerState?
  /// Why the Runner did not start; nil if it did.
  private(set) var failure: String?
  private var runner: OpaquePointer?
  private var pollTask: Task<Void, Never>?

  /// Starts the Runner.
  init() {
    var error = [CChar](repeating: 0, count: 1024)
    runner = model_debugger_runner_start(Self.listen, &error, error.count)
    guard runner != nil else {
      failure = String(cString: error)
      return
    }
    pollTask = Task {
      while !Task.isCancelled, let runner {
        var buffer = [CChar](repeating: 0, count: 4096)
        var needed = model_debugger_runner_get_state(runner, &buffer, buffer.count)
        if needed >= buffer.count {
          buffer = [CChar](repeating: 0, count: needed + 1)
          needed = model_debugger_runner_get_state(runner, &buffer, buffer.count)
        }
        if needed >= 0 && needed < buffer.count {
          state = try? Self.decoder.decode(
            RunnerState.self, from: Data(String(cString: buffer).utf8))
        }
        do {
          try await Task.sleep(for: .milliseconds(500))
        } catch {
          break
        }
      }
    }
  }

  /// Stops the Runner: in-flight calls finish, and its run directory goes.
  ///
  /// The App calls it when it terminates, the last moment it may run code.
  func stop() {
    pollTask?.cancel()
    pollTask = nil
    model_debugger_runner_delete(runner)
    runner = nil
  }

  /// The state's JSON is in snake case.
  private static let decoder = {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return decoder
  }()

  private static var listen: String {
    let arguments = CommandLine.arguments
    guard let flag = arguments.firstIndex(of: "--listen"), flag + 1 < arguments.count else {
      return "127.0.0.1:8769"
    }
    return arguments[flag + 1]
  }
}
