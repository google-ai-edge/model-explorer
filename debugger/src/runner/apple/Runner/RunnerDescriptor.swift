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

/// A read-only observation. Public ownership is included; prompts and credentials are not.
struct RunnerDescriptor: Encodable {
  let version = 1
  let protocolVersion = RunnerCommandCodec.protocolVersion
  let runnerInstance: UUID
  let capabilities: RunnerCapabilities
  var runtimes: [RunnerRuntimeCapability] = []
  let environment: RunnerDeviceEnvironment
  let state: RunnerRuntimeState
  let controlConnected: Bool
  let busy: Bool
  let residentSessions: Int
  var owner: RunnerOwner? = nil
  var lifecycle: String = "waiting"
  var buildID: String? = nil
  var build: RunnerUIState.Build? = nil

  private enum CodingKeys: String, CodingKey {
    case version
    case protocolVersion
    case runnerInstance
    case capabilities
    case runtimes
    case environment
    case state
    case controlConnected
    case busy
    case residentSessions
    case owner
    case lifecycle
    case buildID = "buildId"
    case build
  }
}

struct RunnerRuntimeCapability: Encodable {
  let id: String
  let backends: [String]
  let transport: String
}
