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

/// One activation owns the entire Runner, including its one Ref or Target side.
/// Public identity is diagnostic data, never a credential.
struct RunnerOwner: Encodable, Equatable {
  let serverID: String
  let serverName: String
  let sessionID: String
  let sessionName: String
  let runID: String

  private enum CodingKeys: String, CodingKey {
    case serverID = "serverId"
    case serverName
    case sessionID = "sessionId"
    case sessionName
    case runID = "runId"
  }
}

struct RunnerOwnership {
  enum Lifecycle: String { case waiting, active, ending, ended }
  private(set) var owner: RunnerOwner?
  private(set) var connectionID: UUID?
  private(set) var lifecycle = Lifecycle.waiting
  private(set) var lastHeartbeat: TimeInterval?
  let heartbeatTimeout: TimeInterval

  init(heartbeatTimeout: TimeInterval = 30) { self.heartbeatTimeout = heartbeatTimeout }

  mutating func activate(
    _ value: RunnerOwner, connection: UUID, now: TimeInterval, allowRestart: Bool = false
  ) throws {
    guard lifecycle == .waiting || (allowRestart && lifecycle == .ended), owner == nil else {
      throw RunnerError("Runner already belongs to a Session.")
    }
    owner = value
    connectionID = connection
    lastHeartbeat = now
    lifecycle = .active
  }

  func authorizes(connection: UUID?, session: String? = nil, role: String? = nil) -> Bool {
    guard lifecycle == .active, connection != nil, connectionID == connection, let owner else {
      return false
    }
    let sessionMatches: Bool
    if let session {
      guard let requestedUUID = UUID(uuidString: session),
        let ownerUUID = UUID(uuidString: owner.sessionID)
      else {
        return false
      }
      sessionMatches = requestedUUID == ownerUUID
    } else {
      sessionMatches = true
    }
    return sessionMatches && (role == nil || role == owner.runID)
  }

  mutating func heartbeat(server: String, session: String, connection: UUID?, now: TimeInterval)
    throws
  {
    guard authorizes(connection: connection, session: session), server == owner?.serverID else {
      throw RunnerError("Heartbeat does not belong to this Runner's Session.")
    }
    lastHeartbeat = now
  }

  func expired(at now: TimeInterval) -> Bool {
    lifecycle == .active && lastHeartbeat.map { now - $0 >= heartbeatTimeout } == true
  }

  mutating func beginEnding() -> Bool {
    guard lifecycle == .active else { return false }
    lifecycle = .ending
    return true
  }

  mutating func finishEnding() {
    owner = nil
    connectionID = nil
    lastHeartbeat = nil
    lifecycle = .ended
  }
}

struct RunnerSessionKey: Hashable {
  enum Role: String { case ref, target }
  let id: UUID
  let role: Role
  var wireKey: String { id.uuidString.lowercased() + ":" + role.rawValue }
}

struct RunnerSessionSummary: Encodable {
  let key: String
  let id: String
  let role: String
  let modelHash: String
  let tokens: Int?
  let contextLength: Int?
  let capturePoints: Int?
  var runtime = "LiteRT-LM"
  var backend = "CPU"

  func withoutChat() -> RunnerSessionSummary {
    RunnerSessionSummary(
      key: key, id: id, role: role, modelHash: modelHash, tokens: nil,
      contextLength: contextLength, capturePoints: capturePoints, runtime: runtime, backend: backend
    )
  }
}

/// Runtime ownership and its last successful summary share one registry entry.
struct ResidentRunnerSession {
  let runtime: any SessionRuntime
  var summary: RunnerSessionSummary?
  var activeChatID: UUID?
}

struct RunnerTransferProgress: Encodable, Sendable {
  let received: UInt64
  let total: UInt64
}

enum RunnerActivity: String {
  case idle, importing, verifying, loading, generating, validating, stopping, releasing, failed
}
