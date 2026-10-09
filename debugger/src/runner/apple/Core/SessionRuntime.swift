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

struct RunnerPreflight: Encodable {
  let contextTokenCount: Int
  let inputTokenCount: Int
  let contextLimit: Int
  var accepted: Bool { contextTokenCount + inputTokenCount <= contextLimit }
}

/// One resident Conversation. Execution and close are serialized by the caller;
/// cancellation is safe from the host actor while native execution is in flight.
protocol SessionRuntime: AnyObject, Sendable {
  func execute(
    job: CaptureJob, model: URL, destination: URL,
    progress: @escaping (NativeProgress) -> Void, delta: @escaping (String) -> Void,
    initializeOnly: Bool, persistent: Bool
  ) throws -> RunnerResult
  func cancel()
  func preflight(prompt: String, contextLimit: Int) throws -> RunnerPreflight
  /// End only the current Chat, after execution/callbacks have finished.
  func closeChat()
  func close()
}

extension NativeRunner: SessionRuntime {}
