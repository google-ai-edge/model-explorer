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

#if os(macOS)
  import Foundation

  /// PyTorch requests forwarded to the local Python Runner. The adapter owns the
  /// request in flight and the Chat bookkeeping; the model keeps the published
  /// resident-session summaries and UI state.
  @MainActor
  final class PythonExecutionAdapter {
    private struct InFlightRequest {
      let id: String
      let owner: UUID
      let session: RunnerSessionKey
      let operation: String
      let hash: String
      let backend: String
    }

    private unowned let model: RunnerModel
    private let python = PythonRunner()
    private var request: InFlightRequest?
    private var activeChats: [RunnerSessionKey: UUID] = [:]
    private var requestedChat: UUID?

    init(model: RunnerModel) { self.model = model }

    /// Backends of the configured local Python Runner; nil when Python is not set up.
    var backends: [String]? { python.configuration?.backends }
    func hasSession(_ key: RunnerSessionKey) -> Bool { model.pythonSessions[key] != nil }

    func attach() {
      python.onMessage = { [weak self] in self?.receive($0) }
      python.onExit = { [weak self] in
        guard let self else { return }
        if let request = self.request, request.owner == self.model.transport.connectionID {
          self.model.transport.send(
            RunnerCommandCodec.error(
              for: request.id, message: "Python Runner exited. Initialize again."))
        }
        if self.request != nil {
          self.model.busy = false
          self.model.activity = .failed
        }
        self.request = nil
        self.model.pythonSessions.removeAll()
        self.activeChats.removeAll()
        self.requestedChat = nil
      }
    }

    func begin(_ message: [String: Any], identity: String) throws {
      let model = self.model
      guard !model.sessionPoisoned else {
        throw RunnerError("Model execution failed. End this Session before starting another.")
      }
      guard !model.busy, model.listenHost == "127.0.0.1", python.configuration != nil,
        let owner = model.transport.connectionID, let value = message["request"] as? [String: Any],
        let session = value["session_id"] as? String, let sessionID = UUID(uuidString: session),
        let run = value["run"] as? [String: Any], let roleText = run["id"] as? String,
        let role = RunnerSessionKey.Role(rawValue: roleText),
        run["runtime"] as? String == "PyTorch",
        let operation = message["type"] as? String,
        ["initialize", "generate", "preflight"].contains(operation),
        value["operation"] as? String == operation, let hash = value["model_sha256"] as? String
      else {
        throw RunnerError("PyTorch requires an idle, configured local Runner.")
      }
      guard model.ownership.authorizes(connection: owner, session: session, role: roleText) else {
        throw RunnerError("This Runner belongs to a different Session or execution side.")
      }
      let key = RunnerSessionKey(id: sessionID, role: role)
      guard let requestUUID = UUID(uuidString: identity) else {
        throw RunnerError("A UUID requestId is required.")
      }
      guard operation == "preflight" || !model.executedRequestIDs.contains(requestUUID) else {
        throw RunnerError("Request already executed; it cannot be repeated after dump deletion.")
      }
      guard operation == "initialize" || model.pythonSessions[key] != nil else {
        throw RunnerError("Initialize the Python Session first.")
      }
      guard model.sessions[key] == nil else {
        throw RunnerError("Close the Session before changing its runtime.")
      }
      guard let chat = value["chat_id"] as? String, let chatID = UUID(uuidString: chat) else {
        throw RunnerError("A Chat UUID is required.")
      }
      if operation == "initialize" {
        guard !model.endedChatIDs.contains(chatID), activeChats[key] != chatID else {
          throw RunnerError("This Chat already exists or has ended.")
        }
        if let previous = activeChats.removeValue(forKey: key) {
          model.endedChatIDs.insert(previous)
        }
      } else {
        guard activeChats[key] == chatID else {
          throw RunnerError("This Chat has ended. Create a new Chat.")
        }
      }
      requestedChat = chatID
      if operation != "preflight" { model.executedRequestIDs.insert(requestUUID) }
      request = InFlightRequest(
        id: identity, owner: owner, session: key, operation: operation, hash: hash,
        backend: run["backend"] as? String ?? "CPU")
      model.busy = true
      model.activity = operation == "initialize" ? .loading : .generating
      model.output = ""
      model.captureCount = nil
      model.completedRun = nil
      model.status = "PyTorch · " + (run["backend"] as? String ?? "CPU")
      RunnerPlatform.keepAwake(true)
      do { try python.send(["type": operation, "requestId": identity, "request": value]) } catch {
        request = nil
        model.busy = false
        throw error
      }
    }

    func receive(_ message: [String: Any]) {
      let model = self.model
      let identity = message["requestId"] as? String
      guard let request, identity == request.id, request.owner == model.transport.connectionID
      else { return }
      switch message["type"] as? String {
      case "event":
        if let event = message["event"] as? [String: Any] {
          if event["type"] as? String == "delta", let text = event["text"] as? String {
            model.output += text
          }
          model.transport.send(["type": "runtime_event", "requestId": request.id, "event": event])
        }
      case "completed":
        let summary = message["summary"] as? [String: Any] ?? [:]
        let effective = summary["effective"] as? [String: Any] ?? [:]
        model.pythonSessions[request.session] = .init(
          key: request.session.wireKey, id: request.session.id.uuidString.lowercased(),
          role: request.session.role.rawValue, modelHash: request.hash,
          tokens: summary["processed_token_count"] as? Int,
          contextLength: effective["contextLength"] as? Int, capturePoints: nil, runtime: "PyTorch",
          backend: request.backend)
        if let chat = requestedChat { activeChats[request.session] = chat }
        model.captureCount = summary["captured_tensors"] as? Int
        model.activity = .idle
        model.status = "Python Session retained · model and cache owned by Runner"
        var terminal = message
        terminal["runtime"] = "PyTorch"
        model.transport.send(terminal)
        self.request = nil
        model.busy = false
      case "preflight":
        model.transport.send(message)
        self.request = nil
        model.busy = false
        model.activity = .idle
      case "error":
        if message["errorCode"] as? String != "input_rejected" {
          if let chat = activeChats.removeValue(forKey: request.session) {
            model.endedChatIDs.insert(chat)
          }
        }
        if message["errorCode"] as? String != "input_rejected"
          && message["generationStatus"] as? String != "stopped"
          && request.operation != "initialize"
        {
          model.sessionPoisoned = true
        }
        model.activity = .failed
        model.status = message["error"] as? String ?? "Python runtime failed"
        model.transport.send(message)
        self.request = nil
        model.busy = false
      default: return
      }
      model.environmentState = RunnerEnvironment.sample()
      RunnerPlatform.keepAwake(model.transport.isConnected || model.busy)
    }

    /// A Server cancel for the Python request in flight; false when the request is someone else's.
    func cancel(identity: String) throws -> Bool {
      guard let request, request.id == identity, request.owner == model.transport.connectionID
      else { return false }
      try python.send(["type": "cancel", "requestId": identity])
      model.activity = .stopping
      model.transport.send(RunnerCommandCodec.cancelling(requestID: identity))
      return true
    }

    /// A local Stop while a Python request is running; false when nothing is running.
    func stop() -> Bool {
      guard let request else { return false }
      try? python.send(["type": "cancel", "requestId": request.id])
      model.activity = .stopping
      model.status = "Stopping Python runtime"
      return true
    }

    /// Chats end with their Session or Chat; returns the ended Chat IDs for the model's ledger.
    func endAllChats() -> Set<UUID> {
      let ended = Set(activeChats.values)
      activeChats.removeAll()
      requestedChat = nil
      request = nil
      return ended
    }

    func resetAndWait() async throws { try await python.resetAndWait() }

    func shutdownAndWait() async {
      await python.shutdownAndWait()
      model.pythonSessions.removeAll()
    }

    func detachChats() {
      for key in Array(model.pythonSessions.keys) {
        model.pythonSessions[key] = model.pythonSessions[key]?.withoutChat()
      }
    }
  }
#endif
