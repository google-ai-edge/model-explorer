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

/// What the Runner reports: `model_debugger_runner_get_state`, decoded from
/// snake case.
struct RunnerState: Decodable, Sendable {
  /// The platform the Runner runs on, such as "macOS" or "iOS".
  let platform: String
  /// The operating system's version, such as "27.2".
  let osVersion: String
  /// The chip, else the machine's model identifier.
  let device: String
  /// The host the Runner listens on.
  let address: String
  /// The port the Runner listens on.
  let port: Int
  /// The server's address as gRPC names it; empty until the server calls.
  let client: String
  /// The loaded model's SHA-256; empty until the server loads one.
  let modelSha256: String
  /// The turns the chat has completed.
  let turns: Int
  /// Whether a turn was cancelled or failed, so that nothing more is generated.
  let chatOver: Bool
}
