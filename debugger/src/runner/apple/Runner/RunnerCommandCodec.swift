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

/// The control protocol on the wire. Field names and reply shapes live here so
/// the model reads typed values and the strings exist in one place.
/// v5 adds to v4: binary file frames, queued file commands and local file links.
/// `src/contracts` holds the schema and one fixture per message.
enum RunnerCommandCodec {
  static let protocolVersion = 5
  static let fileCommands: Set<String> = [
    "upload_begin", "upload_chunk", "upload_finish", "upload_abort", "model_link",
    "run_files", "file_chunk", "run_received",
  ]
  /// File commands that may only name paths on a filesystem the Server shares.
  static func asksForLink(_ message: [String: Any]) -> Bool {
    kind(message) == "model_link" || message["link"] as? Bool == true
  }

  /// A binary frame is a 4-byte big-endian header length, the JSON header, then raw bytes.
  static func encodeFrame(_ header: [String: Any], payload: Data) throws -> Data {
    let head = try JSONSerialization.data(withJSONObject: header)
    var frame = withUnsafeBytes(of: UInt32(head.count).bigEndian) { Data($0) }
    frame.append(head)
    frame.append(payload)
    return frame
  }
  static func decodeFrame(_ frame: Data) throws -> (header: [String: Any], payload: Data) {
    guard frame.count >= 4 else { throw RunnerError("Truncated binary frame.") }
    let size = frame.prefix(4).reduce(0) { $0 << 8 | Int($1) }
    guard frame.count >= 4 + size,
      let header = try JSONSerialization.jsonObject(
        with: frame.subdata(in: frame.startIndex + 4..<frame.startIndex + 4 + size))
        as? [String: Any]
    else {
      throw RunnerError("Invalid binary frame header.")
    }
    return (header, frame.subdata(in: frame.startIndex + 4 + size..<frame.endIndex))
  }

  struct Heartbeat {
    let server: String
    let session: String
    let sequence: Int
  }
  struct Activation {
    let server: String
    let serverName: String
    let sessionID: UUID
    let sessionName: String
    let role: String
    let buildID: String?
  }
  struct ExecutionTarget {
    let key: RunnerSessionKey
  }
  struct ExecutionJob {
    let job: CaptureJob
    let chatID: UUID
  }

  static func kind(_ message: [String: Any]) -> String { message["type"] as? String ?? "" }
  static func requestID(_ message: [String: Any]) -> String {
    message["requestId"] as? String ?? ""
  }
  static func sessionID(_ message: [String: Any]) -> String? { message["sessionId"] as? String }
  static func runID(_ message: [String: Any]) -> String? { message["runId"] as? String }
  static func isPython(_ message: [String: Any]) -> Bool {
    message["runtime"] as? String == "PyTorch"
  }

  static func heartbeat(_ message: [String: Any]) throws -> Heartbeat {
    guard let server = message["serverId"] as? String,
      let session = message["sessionId"] as? String,
      let sequence = message["sequence"] as? Int
    else { throw RunnerError("Invalid Session heartbeat.") }
    return Heartbeat(server: server, session: session, sequence: sequence)
  }

  static func activation(_ message: [String: Any]) throws -> Activation {
    guard let server = message["serverId"] as? String, !server.isEmpty, server.count <= 256,
      let session = message["sessionId"] as? String, let sessionID = UUID(uuidString: session),
      let role = message["runId"] as? String, RunnerSessionKey.Role(rawValue: role) != nil
    else {
      throw RunnerError(
        "Activation requires an idle Runner, Server identity, Session UUID and Ref/Target side.")
    }
    return Activation(
      server: server,
      serverName: String((message["serverName"] as? String ?? server).prefix(256)),
      sessionID: sessionID,
      sessionName: String((message["sessionName"] as? String ?? session).prefix(256)),
      role: role, buildID: message["buildId"] as? String)
  }

  static func modelHash(_ message: [String: Any]) throws -> String {
    guard let hash = message["modelSHA256"] as? String, hash.count == 64,
      hash.allSatisfy({ $0.isHexDigit })
    else {
      throw RunnerError("Invalid model hash.")
    }
    return hash
  }

  static func executionTarget(_ message: [String: Any]) throws -> ExecutionTarget {
    guard let session = message["sessionId"] as? String, let sessionID = UUID(uuidString: session),
      let run = message["runId"] as? String, let role = RunnerSessionKey.Role(rawValue: run)
    else {
      throw RunnerError("Session UUID and runId are required.")
    }
    return ExecutionTarget(key: RunnerSessionKey(id: sessionID, role: role))
  }

  static func executionJob(_ message: [String: Any], requestID: String) throws -> ExecutionJob {
    let data = try JSONSerialization.data(withJSONObject: message["job"] ?? [:])
    let job = try JSONDecoder().decode(CaptureJob.self, from: data)
    try job.validate()
    guard job.id == UUID(uuidString: requestID) else {
      throw RunnerError("Job and request IDs differ.")
    }
    guard let chat = message["chatId"] as? String, let chatID = UUID(uuidString: chat) else {
      throw RunnerError("A Chat UUID is required.")
    }
    return ExecutionJob(job: job, chatID: chatID)
  }

  // MARK: Replies

  static func error(for requestID: String, message: String, extra: [String: Any] = [:]) -> [String:
    Any]
  {
    var reply: [String: Any] = ["type": "error", "requestId": requestID, "error": message]
    for (key, value) in extra { reply[key] = value }
    return reply
  }
  static func heartbeatAck(_ beat: Heartbeat) -> [String: Any] {
    [
      "type": "heartbeat_ack", "serverId": beat.server, "sessionId": beat.session,
      "sequence": beat.sequence,
    ]
  }
  static func activated(
    requestID: String, owner: RunnerOwner, runnerInstance: UUID, buildID: String?
  ) throws -> [String: Any] {
    let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(owner))
    var reply: [String: Any] = [
      "type": "activated", "requestId": requestID, "owner": encoded,
      "runnerInstance": runnerInstance.uuidString.lowercased(),
    ]
    reply["buildId"] = buildID
    return reply
  }
  static func sessionAction(_ type: String, owner: RunnerOwner) -> [String: Any] {
    ["type": type, "serverId": owner.serverID, "sessionId": owner.sessionID, "runId": owner.runID]
  }
  static func modelInfo(requestID: String, available: Bool) -> [String: Any] {
    ["type": "model_info", "requestId": requestID, "available": available]
  }
  static func cancelling(requestID: String) -> [String: Any] {
    ["type": "cancelling", "requestId": requestID]
  }
  static func progress(requestID: String, message: String) -> [String: Any] {
    ["type": "progress", "requestId": requestID, "message": message]
  }
  static func delta(requestID: String, text: String) -> [String: Any] {
    ["type": "delta", "requestId": requestID, "text": text]
  }
  static func closed(requestID: String, sessionID: String) -> [String: Any] {
    ["type": "closed", "requestId": requestID, "sessionId": sessionID]
  }
  static func reset(requestID: String, sessionID: String) -> [String: Any] {
    ["type": "reset", "requestId": requestID, "sessionId": sessionID]
  }
  static func completed(requestID: String, jobID: UUID, result: RunnerResult) -> [String: Any] {
    let encoded = (try? JSONSerialization.jsonObject(with: JSONEncoder().encode(result))) ?? [:]
    return [
      "type": "completed", "requestId": requestID,
      "runDirectory": "Documents/Runs/" + jobID.uuidString,
      "output": result.output,
      "generationStatus": "succeeded", "dumpStatus": result.dumpStatus,
      "result": encoded,
    ]
  }
}
