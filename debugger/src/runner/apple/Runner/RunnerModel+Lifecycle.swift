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

/// Session activation, heartbeat watchdog, ownership transitions and resource cleanup.
extension RunnerModel {
  func activate(_ message: [String: Any], identity: String) throws {
    guard ownership.lifecycle == .waiting || (reuseAfterEnd && ownership.lifecycle == .ended) else {
      throw RunnerError("Runner already belongs to a Session.")
    }
    let activation = try RunnerCommandCodec.activation(message)
    guard !busy, transfer == nil, cleanupTask == nil, sessions.isEmpty,
      let connection = transport.connectionID
    else {
      throw RunnerError(
        "Activation requires an idle Runner, Server identity, Session UUID and Ref/Target side.")
    }
    if let requested = activation.buildID, requested != buildID {
      throw RunnerError("The running Runner build does not match the selected build.")
    }
    let value = RunnerOwner(
      serverID: activation.server, serverName: activation.serverName,
      sessionID: activation.sessionID.uuidString.lowercased(),
      sessionName: activation.sessionName, runID: activation.role)
    try claimDevice(activation.server, value.sessionID, activation.role)
    do {
      try ownership.activate(
        value, connection: connection, now: monotonicNow(), allowRestart: reuseAfterEnd)
    } catch {
      releaseDevice()
      throw error
    }
    sessionPoisoned = false
    endedChatIDs.removeAll()
    executedRequestIDs.removeAll()
    startHeartbeatWatchdog()
    status = "Session activated · waiting for initialization"
    transport.send(
      try RunnerCommandCodec.activated(
        requestID: identity, owner: value,
        runnerInstance: runnerInstance, buildID: buildID))
    RunnerPlatform.keepAwake(true)
  }

  /// The watchdog runs on the main actor; native inference runs on nativeQueue.
  func checkHeartbeat() {
    if ownership.expired(at: monotonicNow()) {
      endOwnedSession(reason: "Server heartbeat timed out")
    }
  }

  func requestStopGeneration() {
    guard allowedActions["stopGeneration"] == true else { return }
    requestSessionAction("stop_requested")
    status = "Stop requested · waiting for Server"
  }

  func requestEndSession() {
    guard allowedActions["endSession"] == true else { return }
    requestSessionAction("session_end_requested")
    status = "End requested · waiting for Server"
  }

  private func requestSessionAction(_ type: String) {
    guard let owner else { return }
    transport.send(RunnerCommandCodec.sessionAction(type, owner: owner))
  }

  func resetOwnedSession(requestID: String) {
    guard cleanupTask == nil else {
      transport.send(
        RunnerCommandCodec.error(for: requestID, message: "Runner cleanup is already in progress."))
      return
    }
    cleanupOwnedResources(reason: "Ending Chat and releasing KV cache", resetRequestID: requestID)
  }

  func endOwnedSession(reason: String, requestID: String? = nil) {
    guard ownership.beginEnding() else { return }
    closingRequestID = requestID
    heartbeatTask?.cancel()
    heartbeatTask = nil
    if cleanupTask == nil { cleanupOwnedResources(reason: reason) }
  }

  func cleanupOwnedResources(reason: String, resetRequestID: String? = nil) {
    guard let originalOwner = owner else { return }
    let connection = ownership.connectionID
    busy = true
    activity = .releasing
    status = reason
    let workers = sessions.values.map(\.runtime)
    for value in sessions.values { if let chat = value.activeChatID { endedChatIDs.insert(chat) } }
    let active = execution
    execution = nil
    active?.runtime.cancel()
    for worker in workers { worker.cancel() }
    fileTask?.cancel()
    fileEpoch = UUID()
    queuedFileCommands = 0
    #if os(macOS)
      endedChatIDs.formUnion(pythonAdapter.endAllChats())
    #endif
    cleanupTask = Task {
      await files.abort()
      fileTask = nil
      transfer = nil
      var cleanupFailure: String?
      #if os(macOS)
        var pythonModelReleased = false
        if resetRequestID != nil && ownership.lifecycle == .active {
          do { try await pythonAdapter.resetAndWait() } catch {
            cleanupFailure = error.localizedDescription
            sessionPoisoned = true
          }
        } else {
          await pythonAdapter.shutdownAndWait()
          pythonModelReleased = true
        }
      #endif
      let releaseModel = resetRequestID == nil || ownership.lifecycle == .ending
      await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
        nativeQueue.async {
          for worker in workers { if releaseModel { worker.close() } else { worker.closeChat() } }
          done.resume()
        }
      }
      #if os(macOS)
        if ownership.lifecycle == .ending && !pythonModelReleased {
          await pythonAdapter.shutdownAndWait()
        }
      #endif
      if ownership.lifecycle == .ending && !releaseModel {
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
          nativeQueue.async {
            for worker in workers { worker.close() }
            done.resume()
          }
        }
      }
      if releaseModel || ownership.lifecycle == .ending {
        sessions.removeAll()
      } else {
        for key in Array(sessions.keys) {
          sessions[key]?.activeChatID = nil
          sessions[key]?.summary = sessions[key]?.summary?.withoutChat()
        }
        #if os(macOS)
          pythonAdapter.detachChats()
        #endif
      }
      output = ""
      captureCount = nil
      completedRun = nil
      busy = false
      activity = .idle
      cleanupTask = nil
      if ownership.lifecycle == .ending {
        let requestID = closingRequestID
        // Keep public ownership until all native, Python and file cleanup has finished.
        let finish = { [self] in
          // Activation remains fenced by `ending` until the old peer's
          // final response has been flushed and its channel is closed.
          if reuseAfterEnd { transport.disconnect() } else { transport.shutdown() }
          ownership.finishEnding()
          closingRequestID = nil
          releaseDevice()
          status =
            reuseAfterEnd
            ? "Session ended · ready for a new Session" : "Session ended · Runner stopped"
          connectionPhase = reuseAfterEnd ? .waiting : .unconfigured
          connectionStatus = reuseAfterEnd ? "Waiting for Session activation" : "Runner stopped"
          RunnerPlatform.keepAwake(false)
          if reuseAfterEnd { runnerInstance = UUID() } else { exitSessionRunner() }
        }
        if let requestID, connection == transport.connectionID {
          transport.sendFinal(
            RunnerCommandCodec.closed(requestID: requestID, sessionID: originalOwner.sessionID),
            completion: finish)
        } else {
          finish()
        }
      } else {
        status = cleanupFailure ?? "Chat ended · model retained for a new Chat"
        if let resetRequestID, connection == transport.connectionID {
          if let cleanupFailure {
            transport.send(
              RunnerCommandCodec.error(for: resetRequestID, message: cleanupFailure))
          } else {
            transport.send(
              RunnerCommandCodec.reset(
                requestID: resetRequestID, sessionID: originalOwner.sessionID))
          }
        }
      }
    }
  }

  func startHeartbeatWatchdog() {
    if heartbeatTask == nil {
      let startedAt = monotonicNow()
      heartbeatTask = Task { [weak self] in
        while !Task.isCancelled {
          try? await Task.sleep(for: .seconds(1))
          guard !Task.isCancelled, let self else { return }
          self.checkHeartbeat()
          if self.launchedForSession, self.ownership.lifecycle == .waiting,
            self.monotonicNow() - startedAt >= self.ownership.heartbeatTimeout
          {
            self.status = "Session activation timed out · Runner stopped"
            if self.reuseAfterEnd { self.transport.disconnect() } else { self.transport.shutdown() }
            self.ownership.finishEnding()
            self.heartbeatTask = nil
            if self.reuseAfterEnd { self.runnerInstance = UUID() } else { self.exitSessionRunner() }
            return
          }
        }
      }
    }
  }
}
