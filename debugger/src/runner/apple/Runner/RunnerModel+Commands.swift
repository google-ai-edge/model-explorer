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

/// Dispatch of v4 commands from the Server, decoded by `RunnerCommandCodec`.
extension RunnerModel {
  func receiveCommand(_ message: [String: Any]) {
    let identity = RunnerCommandCodec.requestID(message)
    do {
      let kind = RunnerCommandCodec.kind(message)
      if kind == "heartbeat" {
        let beat = try RunnerCommandCodec.heartbeat(message)
        try ownership.heartbeat(
          server: beat.server, session: beat.session, connection: transport.connectionID,
          now: monotonicNow())
        transport.send(RunnerCommandCodec.heartbeatAck(beat))
        return
      }
      guard UUID(uuidString: identity) != nil else {
        throw RunnerError("A UUID requestId is required.")
      }
      if kind == "activate" {
        try activate(message, identity: identity)
        return
      }
      guard ownership.authorizes(connection: transport.connectionID) else {
        throw RunnerError("Activate this Runner for a Session before issuing commands.")
      }
      if let session = RunnerCommandCodec.sessionID(message),
        !ownership.authorizes(
          connection: transport.connectionID, session: session,
          role: RunnerCommandCodec.runID(message))
      {
        throw RunnerError("This Runner belongs to a different Session or execution side.")
      }
      if kind == "close" || kind == "reset" {
        guard let session = RunnerCommandCodec.sessionID(message),
          ownership.authorizes(connection: transport.connectionID, session: session)
        else {
          throw RunnerError("The owning Session UUID is required.")
        }
        if kind == "close" {
          endOwnedSession(reason: "Session ended", requestID: identity)
        } else {
          resetOwnedSession(requestID: identity)
        }
        return
      }
      guard cleanupTask == nil else { throw RunnerError("Wait for Runner cleanup to finish.") }
      if RunnerCommandCodec.fileCommands.contains(kind) {
        // File commands queue behind one another (the v5 send window); nothing else may be running.
        guard !busy || (fileTask != nil && queuedFileCommands < Self.fileQueueLimit) else {
          throw RunnerError("Wait for the active request before transferring files.")
        }
        if RunnerCommandCodec.asksForLink(message) {
          #if os(macOS)
            guard listenHost == "127.0.0.1" else {
              throw RunnerError("File links need a Runner on the Server's own Mac.")
            }
          #else
            throw RunnerError("File links need a Runner on the Server's own Mac.")
          #endif
        }
        try receiveFileCommand(message)
        return
      }
      guard transfer == nil else { throw RunnerError("Finish the model transfer first.") }
      #if os(macOS)
        if RunnerCommandCodec.isPython(message) {
          try pythonAdapter.begin(message, identity: identity)
          return
        }
        if kind == "cancel", try pythonAdapter.cancel(identity: identity) { return }
      #endif
      switch kind {
      case "model_info", "model_commit":
        let hash = try RunnerCommandCodec.modelHash(message)
        let models = documents.appendingPathComponent("Models")
        try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
        if kind == "model_commit" {
          guard !busy else {
            throw RunnerError("Cannot replace a model while capture is active.")
          }
          let pending = models.appendingPathComponent(hash + ".pending")
          let target = models.appendingPathComponent(hash + ".litertlm")
          guard FileManager.default.fileExists(atPath: pending.path) else {
            throw RunnerError("Model transfer is incomplete.")
          }
          if FileManager.default.fileExists(atPath: target.path) {
            _ = try FileManager.default.replaceItemAt(target, withItemAt: pending)
          } else {
            try FileManager.default.moveItem(at: pending, to: target)
          }
        }
        let available = FileManager.default.fileExists(
          atPath: models.appendingPathComponent(hash + ".litertlm").path)
        transport.send(RunnerCommandCodec.modelInfo(requestID: identity, available: available))
      case "cancel":
        guard let request = execution?.request, request.connectionID == transport.connectionID,
          UUID(uuidString: request.wireID) == UUID(uuidString: identity)
        else { throw RunnerError("No matching active request.") }
        stop()
        transport.send(RunnerCommandCodec.cancelling(requestID: identity))
      case "initialize", "generate", "preflight":
        guard !busy else { throw RunnerError("Another capture is active.") }
        guard !sessionPoisoned else {
          throw RunnerError("Model execution failed. End this Session before starting another.")
        }
        guard let connectionID = transport.connectionID else {
          throw RunnerError("Session UUID and runId are required.")
        }
        let key = try RunnerCommandCodec.executionTarget(message).key
        #if os(macOS)
          guard !pythonAdapter.hasSession(key) else {
            throw RunnerError("Close the Session before changing its runtime.")
          }
        #endif
        let initializing = kind == "initialize"
        if !initializing && sessions[key] == nil {
          throw RunnerError("Session is closed. Initialize again.")
        }
        let request = try RunnerCommandCodec.executionJob(message, requestID: identity)
        let value = request.job
        let chatID = request.chatID
        if !initializing {
          guard let activeChat = sessions[key]?.activeChatID, chatID == activeChat else {
            throw RunnerError("This Chat has ended. Create a new Chat before sending a message.")
          }
        }
        if kind == "preflight" {
          guard let worker = sessions[key]?.runtime else {
            throw RunnerError("Initialize a Chat first.")
          }
          beginPreflight(
            worker: worker, job: value, identity: identity, session: key, connection: connectionID)
          return
        }
        guard !executedRequestIDs.contains(value.id),
          !FileManager.default.fileExists(
            atPath: documents.appendingPathComponent("Runs/" + value.id.uuidString).path)
        else {
          throw RunnerError(
            "Request already executed; retrieve its existing run instead of repeating it.")
        }
        guard debuggerEnabled else { throw RunnerError("Tensor capture is unavailable.") }
        if initializing {
          guard (value.messages ?? []).isEmpty else {
            throw RunnerError("New Chat cannot replay ended Chat history.")
          }
          let nextChat = chatID
          guard !endedChatIDs.contains(nextChat), sessions[key]?.activeChatID != nextChat else {
            throw RunnerError("This Chat already exists or has ended.")
          }
          if let oldChat = sessions[key]?.activeChatID { endedChatIDs.insert(oldChat) }
          if sessions[key] == nil { sessions[key] = .init(runtime: makeRuntime()) }
          sessions[key]?.activeChatID = nextChat
        }
        executedRequestIDs.insert(value.id)
        begin(
          job: value,
          model: documents.appendingPathComponent("Models/" + value.modelSHA256 + ".litertlm"),
          request: RemoteRequest(
            wireID: identity, session: key, connectionID: connectionID, initializing: initializing))
      default: throw RunnerError("Unknown Runner command.")
      }
    } catch {
      if !busy { reportError(error) }
      transport.send(RunnerCommandCodec.error(for: identity, message: error.localizedDescription))
    }
  }

  func beginPreflight(
    worker: any SessionRuntime, job: CaptureJob, identity: String,
    session: RunnerSessionKey, connection: UUID
  ) {
    let operation = Execution(
      runtime: worker,
      request: RemoteRequest(
        wireID: identity, session: session,
        connectionID: connection, initializing: false))
    execution = operation
    busy = true
    nativeQueue.async {
      do {
        let checked = try worker.preflight(prompt: job.prompt, contextLimit: job.contextLength)
        Task { @MainActor in
          guard self.isCurrent(operation) else { return }
          self.transport.send([
            "type": "preflight", "requestId": identity, "accepted": checked.accepted,
            "contextTokenCount": checked.contextTokenCount,
            "inputTokenCount": checked.inputTokenCount,
            "contextLimit": checked.contextLimit,
            "reason": checked.accepted ? "" : "Input exceeds the Chat context capacity.",
          ])
          self.finish(operation)
        }
      } catch {
        Task { @MainActor in
          guard self.isCurrent(operation) else { return }
          self.sessionPoisoned = true
          self.reportError(error)
          self.transport.send(
            RunnerCommandCodec.error(
              for: identity, message: error.localizedDescription,
              extra: ["errorCode": "preflight_failed", "generationStatus": "failed"]))
          self.finish(operation)
        }
      }
    }
  }

  func receiveFileCommand(_ message: [String: Any]) throws {
    guard let owner = transport.connectionID else {
      throw RunnerError("Connect before transferring files.")
    }
    var command = message
    let payload = message["data"] as? Data
    if payload != nil { command.removeValue(forKey: "data") }
    let data = try JSONSerialization.data(withJSONObject: command)
    // The epoch fences a whole connection's queue: cleanup replaces it, queued commands share it.
    let epoch = fileEpoch
    let previous = fileTask
    busy = true
    queuedFileCommands += 1
    let kind = RunnerCommandCodec.kind(message)
    fileTask = Task {
      await previous?.value
      defer {
        // Cleanup owns `busy` and the queue once it has replaced the epoch.
        if fileEpoch == epoch {
          queuedFileCommands -= 1
          if queuedFileCommands == 0 {
            busy = false
            fileTask = nil
          }
        }
      }
      guard fileEpoch == epoch, transport.connectionID == owner else { return }
      do {
        let response = try await files.handle(data, payload: payload)
        guard fileEpoch == epoch, transport.connectionID == owner else { return }
        transfer = response.transfer
        if kind == "upload_begin" {
          activity = .idle
          status = "Receiving model from the server Mac"
        }
        if kind == "upload_finish" || kind == "model_link" {
          status = "Model received and verified"
        }
        if let reply = try JSONSerialization.jsonObject(with: response.data) as? [String: Any] {
          if let bytes = response.payload {
            transport.send(reply, payload: bytes)
          } else {
            transport.send(reply)
          }
        }
      } catch {
        guard fileEpoch == epoch, transport.connectionID == owner else { return }
        // Preserve acknowledged upload state on a rejected chunk.
        let remaining = await files.transferProgress
        guard fileEpoch == epoch, transport.connectionID == owner else { return }
        transfer = remaining
        reportError(error)
        transport.send(
          RunnerCommandCodec.error(
            for: RunnerCommandCodec.requestID(message), message: error.localizedDescription))
      }
    }
  }
}
