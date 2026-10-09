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

enum RunnerConnectionState: String { case unconfigured, waiting, connected, failed }

/// Server control transport. Connection identity belongs to an authenticated peer,
/// and changes across reconnects even when the same computer reconnects.
@MainActor protocol RunnerTransport: AnyObject {
  var isConnected: Bool { get }
  var connectionID: UUID? { get }
  var onMessage: (([String: Any]) -> Void)? { get set }
  var onDisconnect: (() -> Void)? { get set }
  var onStatus: ((RunnerConnectionState, String) -> Void)? { get set }
  var inspection: (() -> RunnerDescriptor?)? { get set }
  func start(documents: URL) throws
  func restart(documents: URL) throws
  func disconnect()
  func shutdown()
  func send(_ value: [String: Any])
  /// A binary frame: the JSON header, then raw bytes (protocol v5 file chunks).
  func send(_ header: [String: Any], payload: Data)
  func sendFinal(_ value: [String: Any], completion: @escaping () -> Void)
}
