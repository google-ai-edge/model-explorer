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

import Darwin
import Foundation

/// Deterministic scheduling fixtures, not model-inference evidence.
private final class ControlledRuntime: SessionRuntime, @unchecked Sendable {
  let gate = DispatchSemaphore(value: 0)
  let closeGate = DispatchSemaphore(value: 0)
  let chatCloseGate = DispatchSemaphore(value: 0)
  let fail: Bool
  let holdClose: Bool
  let failGeneration: Bool
  let dumpUnavailable: Bool
  let holdChatClose: Bool
  private let lock = NSLock()
  private var started = false
  private var closing = false
  private var cancellations = 0
  private var chatsClosed = 0
  private var executions = 0
  private var lateCallback: (() -> Void)?
  var didStart: Bool { lock.withLock { started } }
  var didClose: Bool { lock.withLock { closing } }
  var cancelCount: Int { lock.withLock { cancellations } }
  var closeChatCount: Int { lock.withLock { chatsClosed } }
  var executeCount: Int { lock.withLock { executions } }
  init(
    fail: Bool = false, holdClose: Bool = false, failGeneration: Bool = false,
    dumpUnavailable: Bool = false, holdChatClose: Bool = false
  ) {
    self.fail = fail
    self.holdClose = holdClose
    self.failGeneration = failGeneration
    self.dumpUnavailable = dumpUnavailable
    self.holdChatClose = holdChatClose
  }
  func execute(
    job: CaptureJob, model: URL, destination: URL,
    progress: @escaping (NativeProgress) -> Void, delta: @escaping (String) -> Void,
    initializeOnly: Bool, persistent: Bool
  ) throws -> RunnerResult {
    try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
    lock.withLock {
      started = true
      executions += 1
      lateCallback = {
        progress(.generating)
        delta("stale callback")
      }
    }
    guard gate.wait(timeout: .now() + 5) == .success else {
      throw RunnerError("Fixture release timed out")
    }
    // A native callback can arrive after cancellation/disconnection.
    progress(.generating)
    delta("old connection output")
    if fail || (failGeneration && !initializeOnly) {
      try writeJSON(
        ["generationStatus": "failed", "error": "Synthetic model execution failure"],
        to: destination.appendingPathComponent("terminal.json"))
      throw RunnerError("Late native failure")
    }
    var result = RunnerResult(
      runtimeInstance: UUID(), tokenCountBefore: 0, tokenCount: 17,
      turnSequence: initializeOnly ? 0 : 1, jobID: job.id, modelSHA256: job.modelSHA256,
      platform: "Synthetic", operatingSystem: "Synthetic", hardwareIdentifier: "Synthetic",
      input: job.prompt, output: "old connection output", maxOutputTokens: job.maxOutputTokens,
      contextLength: job.contextLength, elapsedSecondsWithCapture: 0, tensors: [],
      uncapturedPoints: [])
    if dumpUnavailable {
      result.dumpStatus = "unavailable"
      result.dumpError = "Synthetic export failure"
    }
    return result
  }
  func cancel() { lock.withLock { cancellations += 1 } }
  func closeChat() {
    lock.withLock { chatsClosed += 1 }
    if holdChatClose { _ = chatCloseGate.wait(timeout: .now() + 5) }
  }
  func preflight(prompt: String, contextLimit: Int) throws -> RunnerPreflight {
    RunnerPreflight(
      contextTokenCount: 17, inputTokenCount: prompt == "overflow" ? 1024 : 3,
      contextLimit: contextLimit)
  }
  func replayLateCallback() { lock.withLock { lateCallback }?() }
  func close() {
    lock.withLock { closing = true }
    if holdClose { _ = closeGate.wait(timeout: .now() + 5) }
  }
}

@MainActor private final class MemoryTransport: RunnerTransport {
  var connectionID: UUID?
  var isConnected: Bool { connectionID != nil }
  var onMessage: (([String: Any]) -> Void)?
  var onDisconnect: (() -> Void)?
  var onStatus: ((RunnerConnectionState, String) -> Void)?
  var inspection: (() -> RunnerDescriptor?)?
  var sent: [(UUID, [String: Any])] = []
  var shutdownCount = 0
  var delayFinal = false
  var finalCompletion: (() -> Void)?
  func start(documents: URL) throws { onStatus?(.waiting, "Synthetic transport") }
  func restart(documents: URL) throws {
    disconnect()
    try start(documents: documents)
  }
  func connect() {
    connectionID = UUID()
    onStatus?(.connected, "Synthetic peer connected")
  }
  func disconnect() {
    connectionID = nil
    onDisconnect?()
    onStatus?(.waiting, "Synthetic peer disconnected")
  }
  func shutdown() {
    shutdownCount += 1
    disconnect()
  }
  func send(_ value: [String: Any]) { if let connectionID { sent.append((connectionID, value)) } }
  var binary: [([String: Any], Data)] = []
  func send(_ header: [String: Any], payload: Data) {
    if connectionID != nil { binary.append((header, payload)) }
  }
  func sendFinal(_ value: [String: Any], completion: @escaping () -> Void) {
    send(value)
    if delayFinal { finalCompletion = completion } else { completion() }
  }
  func command(_ value: [String: Any]) { onMessage?(value) }
}

@main struct RunnerLifecycleTests {
  @MainActor static func main() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let hash = String(repeating: "a", count: 64)
    let tap = TapPoint(
      signature: "decode", subgraph: 0, op: 10, output: 0, tensor: 354,
      tensorName: "synthetic", outputName: "tap", shape: [1, 2], tensorType: 0)
    func job(_ tappedHash: String? = nil) -> CaptureJob {
      CaptureJob(
        formatVersion: 1, id: UUID(), modelSHA256: hash, prompt: "Synthetic test",
        contextLength: 1024,
        maxOutputTokens: 2,
        manifest: .init(
          formatVersion: 1, sourceSHA256: hash,
          tappedSHA256: tappedHash ?? hash, taps: [tap]))
    }
    func initialize(_ value: CaptureJob, session: UUID, chat: UUID? = nil) throws -> [String: Any] {
      [
        "type": "initialize", "requestId": value.id.uuidString.lowercased(),
        "sessionId": session.uuidString.lowercased(),
        "chatId": (chat ?? session).uuidString.lowercased(),
        "runId": "target",
        "job": try JSONSerialization.jsonObject(with: JSONEncoder().encode(value)),
      ]
    }
    func activate(
      _ transport: MemoryTransport, session: UUID = UUID(), role: String = "target",
      server: String = "server-a"
    ) {
      transport.command([
        "type": "activate", "requestId": UUID().uuidString, "sessionId": session.uuidString,
        "runId": role, "serverId": server, "serverName": "Server A", "sessionName": "Test Session",
      ])
    }
    var failures = 0
    var checks = 0
    func expect(_ valid: Bool, _ label: String) {
      checks += 1
      if !valid { failures += 1 }
      print("\(valid ? "PASS" : "FAIL") \(label)")
    }
    func eventually(_ condition: @MainActor () -> Bool) async throws {
      for _ in 0..<1000 {
        if condition() { return }
        try await Task.sleep(nanoseconds: 5_000_000)
      }
      throw RunnerError("Asynchronous fixture timed out")
    }
    do {
      let transport = MemoryTransport()
      var now: TimeInterval = 0
      var exits = 0
      let model = RunnerModel(
        claimDevice: { _, _, _ in }, releaseDevice: {}, transport: transport,
        storage: root.appendingPathComponent(UUID().uuidString),
        monotonicNow: { now }, launchedForSession: true, exitSessionRunner: { exits += 1 })
      model.startTransport()
      expect(
        model.descriptor.protocolVersion == 5 && model.descriptor.owner == nil,
        "read-only inspection advertises v4 before activation")
      now = 30
      try await eventually { exits == 1 }
      expect(
        model.ownership.lifecycle == .ended,
        "a Session-launched Runner exits if activation never arrives")
    }
    do {
      let transport = MemoryTransport()
      let first = ControlledRuntime(holdClose: true)
      let second = ControlledRuntime()
      var created = 0
      var exits = 0
      var now: TimeInterval = 0
      let model = RunnerModel(
        claimDevice: { _, _, _ in }, releaseDevice: {}, transport: transport,
        storage: root.appendingPathComponent(UUID().uuidString),
        debuggerEnabled: true,
        makeRuntime: {
          created += 1
          return created == 1 ? first : second
        },
        monotonicNow: { now }, reuseAfterEnd: true, exitSessionRunner: { exits += 1 })
      model.startTransport()
      transport.connect()
      let originalSession = UUID()
      let nextSession = UUID()
      let oldJob = job()
      activate(transport, session: originalSession)
      let originalInstance = model.descriptor.runnerInstance
      transport.command(try initialize(oldJob, session: originalSession))
      first.gate.signal()
      try await eventually { !model.busy }
      transport.delayFinal = true
      transport.command([
        "type": "close", "requestId": UUID().uuidString, "sessionId": originalSession.uuidString,
      ])
      try await eventually { first.didClose }
      activate(transport, session: nextSession, server: "server-b")
      expect(
        model.owner?.sessionID == originalSession.uuidString.lowercased(),
        "iOS shell rejects reactivation during native cleanup")
      first.closeGate.signal()
      try await eventually { transport.finalCompletion != nil }
      activate(transport, session: nextSession, server: "server-b")
      expect(
        model.ownership.lifecycle == .ending
          && model.owner?.sessionID == originalSession.uuidString.lowercased(),
        "reactivation remains fenced until the old connection's final reply is flushed")
      transport.finalCompletion?()
      transport.finalCompletion = nil
      let ended = transport.inspection!()!
      expect(
        ended.lifecycle == "ended" && ended.owner == nil && !ended.controlConnected && !ended.busy
          && ended.residentSessions == 0,
        "ended iOS shell remains inspectable with no owner, model or KV")
      expect(
        transport.shutdownCount == 0 && exits == 0 && ended.runnerInstance != originalInstance,
        "iOS retains its listener and creates a fresh logical Runner identity")
      transport.connect()
      let nextPeer = transport.connectionID!
      activate(transport, session: nextSession, server: "server-b")
      expect(
        model.owner?.serverID == "server-b" && model.connectionPhase == .connected,
        "a new Server can activate the same iOS shell without replacing the app")
      transport.command(try initialize(job(), session: nextSession))
      second.gate.signal()
      try await eventually { !model.busy }
      model.output = ""
      first.replayLateCallback()
      try await Task.sleep(nanoseconds: 20_000_000)
      expect(
        created == 2 && model.sessions.count == 1 && model.output.isEmpty
          && !transport.sent.contains {
            $0.0 == nextPeer && $0.1["requestId"] as? String == oldJob.id.uuidString.lowercased()
          },
        "old runtime callbacks cannot alter a newly activated iOS Session")
      transport.delayFinal = false
      now = 30
      model.checkHeartbeat()
      try await eventually { model.ownership.lifecycle == .ended }
      expect(
        transport.shutdownCount == 0 && exits == 0,
        "the reactivated iOS Runner has a working heartbeat timeout and remains reusable")
    }
    do {
      let transport = MemoryTransport()
      let runtime = ControlledRuntime(holdClose: true)
      var now: TimeInterval = 100
      var exits = 0
      let model = RunnerModel(
        claimDevice: { _, _, _ in }, releaseDevice: {}, transport: transport,
        storage: root.appendingPathComponent(UUID().uuidString),
        debuggerEnabled: true, makeRuntime: { runtime }, monotonicNow: { now },
        exitSessionRunner: { exits += 1 })
      model.startTransport()
      transport.connect()
      let session = UUID()
      let value = job()
      transport.command(try initialize(value, session: session))
      expect(
        !runtime.didStart && transport.sent.last?.1["type"] as? String == "error",
        "execution requires explicit activation")
      activate(transport, session: session)
      expect(
        model.owner?.sessionID == session.uuidString.lowercased()
          && model.uiState.owner?.runID == "target",
        "activation publishes one Server, Session and side")
      activate(transport, session: UUID(), server: "server-b")
      expect(model.owner?.serverID == "server-a", "another Server cannot replace an active owner")
      var wrongSide = try initialize(job(), session: session)
      wrongSide["runId"] = "ref"
      transport.command(wrongSide)
      expect(!runtime.didStart, "the owner's other execution side cannot share this Runner")
      now = 125
      transport.command([
        "type": "heartbeat", "serverId": "server-a", "sessionId": session.uuidString, "sequence": 1,
      ])
      expect(
        transport.sent.last?.1["type"] as? String == "heartbeat_ack",
        "application heartbeat replies without a request UUID")
      now = 150
      model.checkHeartbeat()
      expect(
        model.ownership.lifecycle == .active,
        "valid heartbeat extends activation while Session is idle")
      transport.command(try initialize(value, session: session))
      try await eventually { runtime.didStart }
      now = 154
      transport.command([
        "type": "heartbeat", "serverId": "server-b", "sessionId": session.uuidString, "sequence": 2,
      ])
      now = 155
      model.checkHeartbeat()
      expect(
        model.ownership.lifecycle == .ending && runtime.cancelCount > 0,
        "foreign heartbeat cannot delay timeout and watchdog cancels blocked native execution")
      expect(
        model.owner != nil && exits == 0, "timeout retains device ownership during native cleanup")
      runtime.gate.signal()
      try await eventually { runtime.didClose }
      expect(model.owner != nil && exits == 0, "blocked native close still reserves the device")
      runtime.closeGate.signal()
      try await eventually { exits == 1 }
      expect(
        model.owner == nil && model.ownership.lifecycle == .ended && !transport.isConnected,
        "Runner exits only after cleanup and clears public ownership")
    }
    do {
      let transport = MemoryTransport()
      let runtime = ControlledRuntime()
      var exits = 0
      let model = RunnerModel(
        claimDevice: { _, _, _ in }, releaseDevice: {}, transport: transport,
        storage: root.appendingPathComponent(UUID().uuidString),
        debuggerEnabled: true, makeRuntime: { runtime }, exitSessionRunner: { exits += 1 })
      model.startTransport()
      transport.connect()
      let session = UUID()
      activate(transport, session: session)
      transport.command(try initialize(job(), session: session))
      runtime.gate.signal()
      try await eventually { !model.busy }
      var generating = try initialize(job(), session: session)
      generating["type"] = "generate"
      transport.command(generating)
      model.requestStopGeneration()
      expect(
        transport.sent.last?.1["type"] as? String == "stop_requested" && runtime.cancelCount == 0,
        "local Stop asks the owning Server to coordinate both sides")
      model.requestEndSession()
      expect(
        transport.sent.last?.1["type"] as? String == "session_end_requested" && model.owner != nil,
        "local End asks the owning Server to coordinate the whole Session")
      let resetID = UUID().uuidString
      transport.command(["type": "reset", "requestId": resetID, "sessionId": session.uuidString])
      runtime.gate.signal()
      try await eventually { transport.sent.contains { $0.1["type"] as? String == "reset" } }
      expect(
        model.owner != nil && model.ownership.lifecycle == .active && model.sessions.count == 1
          && runtime.closeChatCount == 1 && !runtime.didClose && exits == 0,
        "reset releases the Chat while retaining its model and active ownership")
      transport.command(try initialize(job(), session: session, chat: UUID()))
      runtime.gate.signal()
      try await eventually { !model.busy }
      expect(
        model.sessions.count == 1 && model.owner?.sessionID == session.uuidString.lowercased(),
        "reset Runner accepts a fresh Chat using its existing model")
      transport.command([
        "type": "close", "requestId": UUID().uuidString, "sessionId": session.uuidString,
      ])
      try await eventually { exits == 1 }
      expect(
        transport.sent.last?.1["type"] as? String == "closed",
        "close acknowledges cleanup before disconnect and exit")
    }
    do {
      let transport = MemoryTransport()
      let runtime = ControlledRuntime(dumpUnavailable: true)
      var created = 0
      let storage = root.appendingPathComponent(UUID().uuidString)
      let model = RunnerModel(
        claimDevice: { _, _, _ in }, releaseDevice: {}, transport: transport,
        storage: storage, debuggerEnabled: true,
        makeRuntime: {
          created += 1
          return runtime
        }, exitSessionRunner: {})
      model.startTransport()
      transport.connect()
      let session = UUID()
      let original = job()
      let chat = UUID()
      activate(transport, session: session)
      var initial = try initialize(original, session: session)
      initial["chatId"] = chat.uuidString
      transport.command(initial)
      runtime.gate.signal()
      try await eventually { !model.busy }
      var preflight = try initialize(job(), session: session)
      preflight["type"] = "preflight"
      preflight["chatId"] = chat.uuidString
      transport.command(preflight)
      try await eventually { !model.busy }
      expect(
        transport.sent.last?.1["type"] as? String == "preflight"
          && transport.sent.last?.1["accepted"] as? Bool == true && runtime.executeCount == 1,
        "capacity preflight returns before model execution without creating another runtime")
      var overflowing = try initialize(job(), session: session)
      overflowing["type"] = "preflight"
      overflowing["chatId"] = chat.uuidString
      var overflowJob = overflowing["job"] as! [String: Any]
      overflowJob["prompt"] = "overflow"
      overflowing["job"] = overflowJob
      transport.command(overflowing)
      try await eventually { !model.busy }
      expect(
        transport.sent.last?.1["type"] as? String == "preflight"
          && transport.sent.last?.1["accepted"] as? Bool == false && runtime.executeCount == 1,
        "input capacity rejection leaves the current Chat ready without executing either prompt")
      var generate = try initialize(job(), session: session)
      generate["type"] = "generate"
      generate["chatId"] = chat.uuidString
      transport.command(generate)
      runtime.gate.signal()
      try await eventually { !model.busy }
      let completed = transport.sent.last?.1
      expect(
        completed?["type"] as? String == "completed"
          && completed?["output"] as? String == "old connection output"
          && completed?["dumpStatus"] as? String == "unavailable"
          && completed?["result"] is [String: Any],
        "successful output and full result remain completed when dump is unavailable")
      let generatedID = UUID(uuidString: generate["requestId"] as! String)!
      try FileManager.default.removeItem(
        at: storage.appendingPathComponent("Runs/" + generatedID.uuidString))
      transport.command(generate)
      expect(
        transport.sent.last?.1["type"] as? String == "error" && runtime.executeCount == 2,
        "deleting an acknowledged dump does not permit executing its request ID again")
      transport.command([
        "type": "reset", "requestId": UUID().uuidString, "sessionId": session.uuidString,
      ])
      try await eventually { !model.busy }
      var old = try initialize(job(), session: session)
      old["chatId"] = chat.uuidString
      transport.command(old)
      expect(
        transport.sent.last?.1["type"] as? String == "error" && runtime.executeCount == 2,
        "ended Chat identity cannot be initialized again")
      var fresh = try initialize(job(), session: session)
      fresh["chatId"] = UUID().uuidString
      transport.command(fresh)
      runtime.gate.signal()
      try await eventually { !model.busy }
      expect(
        created == 1 && runtime.closeChatCount == 1 && !runtime.didClose,
        "fresh Chat reuses its retained runtime instead of loading a replacement model")
      transport.disconnect()
      try await eventually { !model.busy }
    }
    do {
      let transport = MemoryTransport()
      let runtime = ControlledRuntime(failGeneration: true)
      let model = RunnerModel(
        claimDevice: { _, _, _ in }, releaseDevice: {}, transport: transport,
        storage: root.appendingPathComponent(UUID().uuidString), debuggerEnabled: true,
        makeRuntime: { runtime }, exitSessionRunner: {})
      model.startTransport()
      transport.connect()
      let session = UUID()
      activate(transport, session: session)
      transport.command(try initialize(job(), session: session))
      runtime.gate.signal()
      try await eventually { !model.busy }
      var generate = try initialize(job(), session: session)
      generate["type"] = "generate"
      transport.command(generate)
      runtime.gate.signal()
      try await eventually { !model.busy }
      expect(
        transport.sent.last?.1["generationStatus"] as? String == "failed"
          && transport.sent.last?.1["runDirectory"] != nil && !runtime.didClose,
        "execution error exposes terminal artifact while retaining model until Session teardown")
      transport.command(try initialize(job(), session: session))
      expect(
        transport.sent.last?.1["type"] as? String == "error" && runtime.executeCount == 2,
        "model execution failure poisons this Session and blocks another Chat")
      transport.disconnect()
      try await eventually { !model.busy }
    }
    do {
      let transport = MemoryTransport()
      let runtime = ControlledRuntime(holdClose: true, holdChatClose: true)
      var exits = 0
      let model = RunnerModel(
        claimDevice: { _, _, _ in }, releaseDevice: {}, transport: transport,
        storage: root.appendingPathComponent(UUID().uuidString), debuggerEnabled: true,
        makeRuntime: { runtime }, exitSessionRunner: { exits += 1 })
      model.startTransport()
      transport.connect()
      let session = UUID()
      activate(transport, session: session)
      transport.command(try initialize(job(), session: session))
      runtime.gate.signal()
      try await eventually { !model.busy }
      transport.command([
        "type": "reset", "requestId": UUID().uuidString, "sessionId": session.uuidString,
      ])
      try await eventually { runtime.closeChatCount == 1 }
      transport.command([
        "type": "close", "requestId": UUID().uuidString, "sessionId": session.uuidString,
      ])
      runtime.chatCloseGate.signal()
      try await eventually { runtime.didClose }
      expect(
        model.ownership.lifecycle == .ending && model.owner != nil && exits == 0,
        "Session close during Chat reset upgrades cleanup to release the retained model")
      runtime.closeGate.signal()
      try await eventually { exits == 1 }
      expect(
        transport.sent.last?.1["type"] as? String == "closed" && model.sessions.isEmpty,
        "reset-to-close transition acknowledges only after full model teardown")
    }
    for fail in [false, true] {
      let transport = MemoryTransport()
      let runtime = ControlledRuntime(fail: fail)
      let model = RunnerModel(
        claimDevice: { _, _, _ in }, releaseDevice: {}, transport: transport,
        storage: root.appendingPathComponent(UUID().uuidString),
        debuggerEnabled: true, makeRuntime: { runtime }, exitSessionRunner: {})
      model.startTransport()
      transport.connect()
      let value = job()
      let session = UUID()
      activate(transport, session: session)
      transport.command(try initialize(value, session: session))
      try await eventually { runtime.didStart }
      transport.disconnect()
      transport.connect()
      let newPeer = transport.connectionID!
      runtime.gate.signal()
      try await eventually { !model.busy }
      let leaked = transport.sent.filter {
        $0.0 == newPeer && $0.1["requestId"] as? String == value.id.uuidString.lowercased()
      }
      expect(
        leaked.isEmpty,
        "late \(fail ? "failure" : "success") never reaches a replacement connection")
      expect(
        model.output.isEmpty && model.uiState.sessions.isEmpty && model.completedRun == nil,
        "disconnected request cannot publish retained state or output")
      transport.disconnect()
    }
    do {
      let transport = MemoryTransport()
      let runtime = ControlledRuntime()
      let model = RunnerModel(
        claimDevice: { _, _, _ in }, releaseDevice: {}, transport: transport,
        storage: root.appendingPathComponent(UUID().uuidString),
        debuggerEnabled: true, makeRuntime: { runtime }, exitSessionRunner: {})
      model.startTransport()
      model.job = job()
      model.modelURL = root.appendingPathComponent("synthetic.litertlm")
      model.start()
      try await eventually { runtime.didStart }
      transport.connect()
      transport.disconnect()
      expect(
        runtime.cancelCount == 0,
        "closing a server probe does not cancel a locally owned manual capture")
      runtime.gate.signal()
      try await eventually { !model.busy }
    }
    do {
      let transport = MemoryTransport()
      let runtime = ControlledRuntime(holdClose: true)
      var created = 0
      let model = RunnerModel(
        claimDevice: { _, _, _ in }, releaseDevice: {}, transport: transport,
        storage: root.appendingPathComponent(UUID().uuidString), debuggerEnabled: true,
        makeRuntime: {
          created += 1
          return runtime
        }, exitSessionRunner: {})
      model.startTransport()
      transport.connect()
      let session = UUID()
      activate(transport, session: session)
      transport.command(try initialize(job(), session: session))
      runtime.gate.signal()
      try await eventually { !model.busy }
      transport.command([
        "type": "close", "requestId": UUID().uuidString,
        "sessionId": session.uuidString.lowercased(),
      ])
      try await eventually { runtime.didClose }
      transport.command(try initialize(job(), session: UUID()))
      expect(created == 1, "close reserves execution until native release finishes")
      runtime.closeGate.signal()
      runtime.gate.signal()
      try await eventually { !model.busy }
      transport.disconnect()
    }
    do {
      let transport = MemoryTransport()
      let runtime = ControlledRuntime()
      let model = RunnerModel(
        claimDevice: { _, _, _ in }, releaseDevice: {}, transport: transport,
        storage: root.appendingPathComponent(UUID().uuidString),
        debuggerEnabled: true, makeRuntime: { runtime }, exitSessionRunner: {})
      model.startTransport()
      transport.connect()
      let session = UUID()
      activate(transport, session: session)
      transport.command(try initialize(job(), session: session))
      runtime.gate.signal()
      try await eventually { !model.busy }
      transport.command([
        "type": "close", "requestId": UUID().uuidString,
        "sessionId": session.uuidString.uppercased(),
      ])
      try await eventually { transport.sent.contains { $0.1["type"] as? String == "closed" } }
      expect(model.uiState.sessions.isEmpty, "Session UUID casing does not change close identity")
      transport.disconnect()
    }
    var accepted = true
    do { try job(String(repeating: "b", count: 64)).validate() } catch { accepted = false }
    expect(accepted, "container and embedded TFLite section have distinct valid hashes")
    do {
      let transport = MemoryTransport()
      let runtime = ControlledRuntime()
      let model = RunnerModel(
        claimDevice: { _, _, _ in }, releaseDevice: {}, transport: transport,
        storage: root.appendingPathComponent(UUID().uuidString),
        debuggerEnabled: true, makeRuntime: { runtime }, exitSessionRunner: {})
      model.startTransport()
      model.job = job()
      model.modelURL = root.appendingPathComponent("synthetic.litertlm")
      model.start()
      try await eventually { runtime.didStart }
      let first = transport.inspection?()
      let second = transport.inspection?()
      expect(
        first?.runnerInstance == second?.runnerInstance && first?.busy == true
          && transport.connectionID == nil && runtime.cancelCount == 0,
        "read-only descriptor sampling does not acquire control or cancel manual execution")
      runtime.gate.signal()
      try await eventually { !model.busy }
    }
    #if os(macOS)
      do {
        let directory = root.appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
        let files = RunnerFiles(root: directory)
        let first = UUID()
        let other = UUID()
        for (run, status) in [(first, "stopped"), (other, "failed")] {
          let path = directory.appendingPathComponent("Runs/" + run.uuidString)
          try FileManager.default.createDirectory(
            at: path.appendingPathComponent("raw"), withIntermediateDirectories: true)
          try writeJSON(
            ["generationStatus": status], to: path.appendingPathComponent("terminal.json"))
          try Data("synthetic partial raw bytes".utf8).write(
            to: path.appendingPathComponent("raw/tensor.safetensors"))
        }
        func handle(_ value: [String: Any]) async throws -> [String: Any] {
          let result = try await files.handle(JSONSerialization.data(withJSONObject: value))
          return try JSONSerialization.jsonObject(with: result.data) as! [String: Any]
        }
        let manifest = try await handle(["type": "run_files", "run": first.uuidString])
        let repeatManifest = try await handle(["type": "run_files", "run": first.uuidString])
        let manifestID = manifest["manifestId"] as! String
        expect(
          manifest["generationStatus"] as? String == "stopped"
            && (manifest["files"] as? [Any])?.count == 2
            && repeatManifest["manifestId"] as? String == manifestID,
          "stopped raw export has a stable immutable manifest")
        var wrongReceipt = false
        var unlisted = false
        do {
          _ = try await handle([
            "type": "run_received", "run": other.uuidString, "manifestId": manifestID,
          ])
        } catch { wrongReceipt = true }
        do {
          _ = try await handle([
            "type": "file_chunk", "run": first.uuidString, "manifestId": manifestID,
            "path": "job.json", "offset": 0,
          ])
        } catch { unlisted = true }
        expect(
          wrongReceipt && unlisted,
          "dump receipt and file reads are bound to the exact run manifest")
        let chunk = try await handle([
          "type": "file_chunk", "run": first.uuidString, "manifestId": manifestID,
          "path": "raw/tensor.safetensors", "offset": 0,
        ])
        expect(
          chunk["data"] as? String
            == Data("synthetic partial raw bytes".utf8).base64EncodedString(),
          "stopped raw bytes are readable without valid tensor metadata")
        let received = try await handle([
          "type": "run_received", "run": first.uuidString, "manifestId": manifestID,
        ])
        expect(
          received["deleted"] as? Bool == true
            && !FileManager.default.fileExists(
              atPath: directory.appendingPathComponent("Runs/" + first.uuidString).path)
            && FileManager.default.fileExists(
              atPath: directory.appendingPathComponent("Runs/" + other.uuidString).path),
          "verified receipt promptly deletes only its matching local dump")
        let failed = try await handle(["type": "run_files", "run": other.uuidString])
        try Data("changed".utf8).write(
          to: directory.appendingPathComponent(
            "Runs/" + other.uuidString + "/raw/tensor.safetensors"))
        var changedRejected = false
        do {
          _ = try await handle([
            "type": "file_chunk", "run": other.uuidString, "manifestId": failed["manifestId"]!,
            "path": "raw/tensor.safetensors", "offset": 0,
          ])
        } catch { changedRejected = true }
        expect(
          changedRejected, "a changed sealed dump is rejected rather than served under stale hashes"
        )
      }
      do {
        let directory = root.appendingPathComponent(UUID().uuidString)
        let source = root.appendingPathComponent("file-fixture")
        let payload = Data("Synthetic file-transfer fixture".utf8)
        try payload.write(to: source)
        let digest = try sha256(of: source)
        let files = RunnerFiles(root: directory)
        func command(_ value: [String: Any]) throws -> Data {
          try JSONSerialization.data(withJSONObject: value)
        }
        _ = try await files.handle(
          command([
            "type": "upload_begin", "requestId": UUID().uuidString, "modelSHA256": digest,
            "size": payload.count,
          ]))
        var rejectedOffset = false
        do {
          _ = try await files.handle(
            command(["type": "upload_chunk", "offset": 1, "data": payload.base64EncodedString()]))
        } catch { rejectedOffset = true }
        let progress = await files.transferProgress
        expect(
          rejectedOffset && progress?.received == 0,
          "file actor rejects an out-of-order chunk without advancing acknowledged bytes")
        _ = try await files.handle(
          command(["type": "upload_chunk", "offset": 0, "data": payload.base64EncodedString()]))
        _ = try await files.handle(command(["type": "upload_finish"]))
        let cached = directory.appendingPathComponent("Models/" + digest + ".litertlm")
        expect(
          try Data(contentsOf: cached) == payload,
          "file actor atomically commits only verified model bytes")
        _ = try await files.handle(
          command(["type": "upload_begin", "modelSHA256": hash, "size": payload.count]))
        _ = try await files.handle(
          command(["type": "upload_chunk", "offset": 0, "data": payload.base64EncodedString()]))
        var rejectedHash = false
        do { _ = try await files.handle(command(["type": "upload_finish"])) } catch {
          rejectedHash = true
        }
        let uploading = await files.uploading
        expect(
          rejectedHash && !uploading
            && !FileManager.default.fileExists(
              atPath: directory.appendingPathComponent("Models/" + hash + ".pending").path),
          "checksum failure removes incomplete upload")
        let data = try command(["type": "upload_begin", "modelSHA256": hash, "size": payload.count])
        let cancelled = Task { try await files.handle(data) }
        cancelled.cancel()
        var cancelledBeforeIO = false
        do { _ = try await cancelled.value } catch is CancellationError { cancelledBeforeIO = true }
        expect(cancelledBeforeIO, "cancelled file request stops before disk work")
      }
      do {
        // Protocol v5 file plane: binary frames, file links and queued file commands.
        let frame = try RunnerCommandCodec.encodeFrame(
          ["type": "upload_chunk", "offset": 7], payload: Data([0, 1, 2, 255]))
        let decoded = try RunnerCommandCodec.decodeFrame(frame)
        var truncated = false
        do { _ = try RunnerCommandCodec.decodeFrame(frame.prefix(6)) } catch { truncated = true }
        expect(
          decoded.header["offset"] as? Int == 7 && decoded.payload == Data([0, 1, 2, 255])
            && truncated,
          "binary frame carries a JSON header and raw bytes; a truncated frame is rejected")

        let directory = root.appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
        let files = RunnerFiles(root: directory)
        func handle(_ value: [String: Any], payload: Data? = nil) async throws -> (
          reply: [String: Any], payload: Data?
        ) {
          let result = try await files.handle(
            JSONSerialization.data(withJSONObject: value), payload: payload)
          return (
            try JSONSerialization.jsonObject(with: result.data) as! [String: Any], result.payload
          )
        }
        let source = root.appendingPathComponent("link-fixture")
        let bytes = Data("Synthetic linked model".utf8)
        try bytes.write(to: source)
        let digest = try sha256(of: source)
        var wrongHashRejected = false
        do {
          _ = try await handle(["type": "model_link", "modelSHA256": hash, "path": source.path])
        } catch { wrongHashRejected = true }
        let models = directory.appendingPathComponent("Models")
        expect(
          wrongHashRejected
            && ((try? FileManager.default.contentsOfDirectory(atPath: models.path)) ?? []).isEmpty,
          "a linked file with another hash is rejected and leaves nothing behind")
        let linked = try await handle([
          "type": "model_link", "modelSHA256": digest, "path": source.path,
        ])
        expect(
          linked.reply["available"] as? Bool == true
            && FileManager.default.fileExists(atPath: source.path)
            && (try? Data(contentsOf: models.appendingPathComponent(digest + ".litertlm")))
              == bytes,
          "a verified link commits the model and leaves the Server's file in place")

        _ = try await handle(["type": "upload_begin", "modelSHA256": digest, "size": bytes.count])
        let chunk = try await handle(["type": "upload_chunk", "offset": 0], payload: bytes)
        let finished = try await handle(["type": "upload_finish"])
        expect(
          chunk.reply["offset"] as? Int == bytes.count
            && finished.reply["available"] as? Bool == true,
          "a binary upload chunk is written without base64")

        let run = UUID()
        let path = directory.appendingPathComponent("Runs/" + run.uuidString)
        try FileManager.default.createDirectory(
          at: path.appendingPathComponent("raw"), withIntermediateDirectories: true)
        try writeJSON(
          ["generationStatus": "succeeded"], to: path.appendingPathComponent("terminal.json"))
        try bytes.write(to: path.appendingPathComponent("raw/tensor.safetensors"))
        let plain = try await handle(["type": "run_files", "run": run.uuidString])
        let shared = try await handle(["type": "run_files", "run": run.uuidString, "link": true])
        expect(
          plain.reply["root"] == nil && shared.reply["root"] as? String == path.path,
          "the export directory is named only when the Server asks for a link")
        let manifestID = shared.reply["manifestId"] as! String
        let raw = try await handle([
          "type": "file_chunk", "run": run.uuidString, "manifestId": manifestID,
          "path": "raw/tensor.safetensors", "offset": 0, "encoding": "binary",
        ])
        expect(
          raw.payload == bytes && raw.reply["data"] == nil && raw.reply["offset"] as? Int == 0,
          "a binary file chunk is returned as raw bytes beside its JSON header")
      }
      do {
        let directory = root.appendingPathComponent(UUID().uuidString)
        let transport = MemoryTransport()
        let model = RunnerModel(
          claimDevice: { _, _, _ in }, releaseDevice: {}, transport: transport, storage: directory,
          debuggerEnabled: true, exitSessionRunner: {})
        model.startTransport()
        transport.connect()
        activate(transport)
        let bytes = Data((0..<40).map { UInt8($0) })
        let source = root.appendingPathComponent("queue-fixture")
        try bytes.write(to: source)
        let digest = try sha256(of: source)
        let before = transport.sent.count
        // The whole transfer is on the wire before the first reply, as a v5 Server sends it.
        transport.command([
          "type": "upload_begin", "requestId": UUID().uuidString, "modelSHA256": digest,
          "size": bytes.count,
        ])
        for offset in stride(from: 0, to: bytes.count, by: 10) {
          transport.command([
            "type": "upload_chunk", "requestId": UUID().uuidString, "offset": offset,
            "data": bytes.subdata(in: offset..<offset + 10),
          ])
        }
        transport.command(["type": "upload_finish", "requestId": UUID().uuidString])
        try await eventually { !model.busy && transport.sent.count == before + 6 }
        let replies = transport.sent[before...].map(\.1)
        expect(
          replies.map { $0["type"] as? String } == ["upload_begin"]
            + Array(repeating: "upload_chunk", count: 4) + ["upload_finish"]
            && replies[1...4].map { $0["offset"] as? Int } == [10, 20, 30, 40]
            && replies.last?["available"] as? Bool == true,
          "queued file commands run in arrival order and each is answered once")
        transport.command([
          "type": "model_info", "requestId": UUID().uuidString, "modelSHA256": digest,
        ])
        expect(
          transport.sent.last?.1["available"] as? Bool == true,
          "the queue releases the Runner for other commands when it drains")
        transport.disconnect()
      }
      do {
        let directory = root.appendingPathComponent(UUID().uuidString)
        let transport = MemoryTransport()
        let model = RunnerModel(
          claimDevice: { _, _, _ in }, releaseDevice: {}, transport: transport, storage: directory,
          debuggerEnabled: true, exitSessionRunner: {})
        model.startTransport()
        transport.connect()
        activate(transport)
        transport.command([
          "type": "upload_begin", "requestId": UUID().uuidString, "modelSHA256": hash, "size": 20,
        ])
        try await eventually { model.transfer != nil && !model.busy }
        transport.disconnect()
        transport.connect()
        try await eventually { !model.busy && model.transfer == nil }
        expect(
          !FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("Models/" + hash + ".pending").path),
          "disconnect awaits file actor cleanup before accepting another transfer")
        transport.disconnect()
      }
    #endif
    // The wire fixtures (src/contracts/fixtures/wire) are the contract both sides are tested against.
    do {
      let wire = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("../../../contracts/fixtures/wire").standardizedFileURL
      func fixture(_ name: String) throws -> [String: Any] {
        try JSONSerialization.jsonObject(
          with: Data(contentsOf: wire.appendingPathComponent(name + ".json"))) as! [String: Any]
      }
      func sameMembers(_ reply: [String: Any], _ name: String) throws -> Bool {
        try Set(reply.keys) == Set(fixture("runner/" + name).keys)
          && reply["type"] as? String == fixture("runner/" + name)["type"] as? String
      }
      let activation = try RunnerCommandCodec.activation(fixture("server/activate"))
      let beat = try RunnerCommandCodec.heartbeat(fixture("server/heartbeat"))
      let generate = try fixture("server/generate")
      let target = try RunnerCommandCodec.executionTarget(generate)
      let request = try RunnerCommandCodec.executionJob(
        generate, requestID: RunnerCommandCodec.requestID(generate))
      let link = try fixture("server/model_link")
      let linkHash = try RunnerCommandCodec.modelHash(link)
      let linkedRun = try fixture("server/run_files.link")
      let plainRun = try fixture("server/run_files")
      let python = try fixture("server/generate.pytorch")
      expect(
        activation.role == "ref" && activation.buildID != nil && beat.sequence == 12
          && target.key.role == .ref && request.job.manifest.taps.count == 1 && linkHash.count == 64
          && RunnerCommandCodec.asksForLink(link) && RunnerCommandCodec.asksForLink(linkedRun)
          && !RunnerCommandCodec.asksForLink(plainRun) && RunnerCommandCodec.isPython(python),
        "every Server fixture decodes through the command codec")
      for name in [
        "upload_begin", "upload_chunk.v4", "upload_chunk.v5-header", "upload_finish",
        "upload_abort", "model_link",
        "run_files", "file_chunk.v4", "file_chunk.v5", "run_received",
      ] {
        let command = try fixture("server/" + name)
        expect(
          RunnerCommandCodec.fileCommands.contains(RunnerCommandCodec.kind(command)),
          "server/\(name) is a file command")
      }
      let owner = RunnerOwner(
        serverID: activation.server, serverName: activation.serverName,
        sessionID: activation.sessionID.uuidString.lowercased(),
        sessionName: activation.sessionName, runID: activation.role)
      let id = UUID().uuidString
      let replies: [(String, [String: Any])] = [
        ("heartbeat_ack", RunnerCommandCodec.heartbeatAck(beat)),
        (
          "activated",
          try RunnerCommandCodec.activated(
            requestID: id, owner: owner, runnerInstance: UUID(), buildID: "1@build")
        ),
        ("model_info", RunnerCommandCodec.modelInfo(requestID: id, available: false)),
        ("cancelling", RunnerCommandCodec.cancelling(requestID: id)),
        ("progress", RunnerCommandCodec.progress(requestID: id, message: "Loading model")),
        ("delta", RunnerCommandCodec.delta(requestID: id, text: " body")),
        ("closed", RunnerCommandCodec.closed(requestID: id, sessionID: owner.sessionID)),
        ("reset", RunnerCommandCodec.reset(requestID: id, sessionID: owner.sessionID)),
        ("stop_requested", RunnerCommandCodec.sessionAction("stop_requested", owner: owner)),
        (
          "session_end_requested",
          RunnerCommandCodec.sessionAction("session_end_requested", owner: owner)
        ),
        ("error.unaddressed", ["type": "error", "error": "Expected a JSON command."]),
      ]
      for (name, reply) in replies {
        let same = try sameMembers(reply, name)
        expect(same, "runner/\(name) has exactly the members the codec sends")
      }
      let errorFixture = try fixture("runner/error")
      let error = RunnerCommandCodec.error(
        for: id, message: "Native operation cancelled",
        extra: [
          "generationStatus": "stopped", "errorCode": "stopped",
          "runDirectory": "Documents/Runs/" + id,
        ])
      expect(
        Set(error.keys) == Set(errorFixture.keys),
        "runner/error carries the terminal members of a stopped generation")

      let directory = root.appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
      let files = RunnerFiles(root: directory)
      let bytes = Data("Synthetic wire fixture".utf8)
      let source = root.appendingPathComponent("wire-fixture")
      try bytes.write(to: source)
      let digest = try sha256(of: source)
      let run = UUID()
      let path = directory.appendingPathComponent("Runs/" + run.uuidString)
      try FileManager.default.createDirectory(
        at: path.appendingPathComponent("raw"), withIntermediateDirectories: true)
      try writeJSON(
        ["generationStatus": "succeeded"], to: path.appendingPathComponent("terminal.json"))
      try bytes.write(to: path.appendingPathComponent("raw/tensor.safetensors"))
      func reply(_ value: [String: Any], payload: Data? = nil) async throws -> [String: Any] {
        var command = value
        command["requestId"] = id
        let result = try await files.handle(
          JSONSerialization.data(withJSONObject: command), payload: payload)
        return try JSONSerialization.jsonObject(with: result.data) as! [String: Any]
      }
      var fileReplies: [(String, [String: Any])] = []
      fileReplies.append(
        (
          "model_link",
          try await reply(["type": "model_link", "modelSHA256": digest, "path": source.path])
        ))
      fileReplies.append(
        (
          "upload_begin",
          try await reply(["type": "upload_begin", "modelSHA256": digest, "size": bytes.count])
        ))
      fileReplies.append(
        ("upload_chunk", try await reply(["type": "upload_chunk", "offset": 0], payload: bytes)))
      fileReplies.append(("upload_finish", try await reply(["type": "upload_finish"])))
      fileReplies.append(("upload_abort", try await reply(["type": "upload_abort"])))
      fileReplies.append(
        ("run_files", try await reply(["type": "run_files", "run": run.uuidString])))
      let manifest = try await reply(["type": "run_files", "run": run.uuidString, "link": true])
      fileReplies.append(("run_files.link", manifest))
      let chunk: [String: Any] = [
        "type": "file_chunk", "run": run.uuidString, "manifestId": manifest["manifestId"]!,
        "path": "raw/tensor.safetensors", "offset": 0,
      ]
      fileReplies.append(("file_chunk.v4", try await reply(chunk)))
      fileReplies.append(
        ("file_chunk.v5-header", try await reply(chunk.merging(["encoding": "binary"]) { $1 })))
      fileReplies.append(
        (
          "run_received",
          try await reply([
            "type": "run_received", "run": run.uuidString, "manifestId": manifest["manifestId"]!,
          ])
        ))
      for (name, value) in fileReplies {
        let same = try sameMembers(value, name)
        expect(same, "runner/\(name) has exactly the members the file actor sends")
      }
    }
    // The shared contract fixture (src/contracts/fixtures) decodes and validates unchanged.
    let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .appendingPathComponent("../../../contracts/fixtures/capture-job.json").standardizedFileURL
    let sharedJob = try JSONDecoder().decode(CaptureJob.self, from: Data(contentsOf: fixture))
    try sharedJob.validate()
    expect(
      sharedJob.manifest.taps.count == 1 && sharedJob.messages?.count == 2
        && sharedJob.backend == "CPU",
      "the shared CaptureJob fixture decodes with every field the Runner reads")
    let environment = RunnerEnvironment.sample()
    let encodedEnvironment =
      try JSONSerialization.jsonObject(with: JSONEncoder().encode(environment)) as! [String: Any]
    #if os(macOS)
      expect(
        encodedEnvironment["processMemoryHeadroomBytes"] == nil,
        "unavailable process memory headroom stays omitted on macOS")
    #endif
    expect(
      encodedEnvironment["capturedAt"] != nil && RunnerEnvironment.device.physicalMemoryBytes > 0,
      "environment records real device memory and a sample timestamp")
    print(
      "\(checks) lifecycle checks, \(failures) failures. All runtime/transport fixtures are synthetic."
    )
    if failures != 0 { exit(1) }
  }
}
