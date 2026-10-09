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

import SwiftUI

#if os(iOS)
  import UIKit
#else
  import AppKit
#endif

/// Runner state and wiring. Behaviour is split by concern:
/// - `RunnerModel+Lifecycle`: activation, heartbeat, ownership and cleanup;
/// - `RunnerModel+Commands`: v4 command dispatch through `RunnerCommandCodec`;
/// - `RunnerModel+ManualJobs`: locally imported jobs and native execution;
/// - `PythonExecutionAdapter`: macOS PyTorch requests.
/// Members are module-internal so those files can reach them; nothing outside the
/// App target links this module.
@MainActor
final class RunnerModel: ObservableObject {
  @Published var job: CaptureJob?
  @Published var modelURL: URL?
  @Published var status = "Create or open a Session in Model Debugger on your server Mac."
  @Published var output = ""
  @Published var busy = false
  @Published var completedRun: URL?
  @Published var captureCount: Int?
  @Published var activity = RunnerActivity.idle
  @Published var connectionPhase = RunnerConnectionState.waiting
  @Published var sessions: [RunnerSessionKey: ResidentRunnerSession] = [:]
  @Published var transfer: RunnerTransferProgress?
  @Published var environmentState = RunnerEnvironment.sample()
  var runnerInstance = UUID()
  @Published var ownership: RunnerOwnership
  var heartbeatTask: Task<Void, Never>?
  var cleanupTask: Task<Void, Never>?
  var closingRequestID: String?
  var endedChatIDs = Set<UUID>()
  var executedRequestIDs = Set<UUID>()
  var sessionPoisoned = false
  let monotonicNow: () -> TimeInterval
  let exitSessionRunner: () -> Void
  let launchedForSession: Bool
  let reuseAfterEnd: Bool
  let claimDevice: @MainActor (String, String, String) throws -> Void
  let releaseDevice: @MainActor () -> Void
  var build: RunnerUIState.Build {
    let environment = RunnerEnvironment.device
    let facts = HostBundle.facts
    let provenance = environment.runtimeCommit.map { commit in
      (environment.appBuild ?? "unknown") + "@" + commit
        + (environment.runtimeSourceDiffSHA256.map { "+" + $0 } ?? "")
    }
    let identity = facts.buildID ?? provenance ?? environment.appBuild
    return .init(
      id: identity,
      label: facts.buildLabel ?? facts.buildCL.map { "CL " + $0 } ?? environment.appBuild.map {
        "Build " + $0
      } ?? "Unknown build",
      version: environment.appVersion, cl: facts.buildCL)
  }
  var buildID: String? { build.id }
  var owner: RunnerOwner? { ownership.owner }
  var descriptor: RunnerDescriptor {
    .init(
      runnerInstance: runnerInstance, capabilities: .init(tensorCapture: debuggerEnabled),
      runtimes: runtimeCapabilities, environment: RunnerEnvironment.device,
      state: RunnerEnvironment.sample(),
      controlConnected: transport.isConnected, busy: busy || transfer != nil,
      residentSessions: residentSummaries.count, owner: owner,
      lifecycle: ownership.lifecycle.rawValue, buildID: buildID, build: build)
  }
  var runtimeCapabilities: [RunnerRuntimeCapability] {
    var values = [
      RunnerRuntimeCapability(
        id: "LiteRT-LM", backends: NativeBackendSupport.availableBackends, transport: "native")
    ]
    #if os(macOS)
      if listenHost == "127.0.0.1", let backends = pythonAdapter.backends {
        values.append(.init(id: "PyTorch", backends: backends, transport: "localFiles"))
      }
    #endif
    return values
  }
  var residentSummaries: [RunnerSessionSummary] {
    var values = sessions.values.compactMap(\.summary)
    #if os(macOS)
      values += pythonSessions.values
    #endif
    return values
  }
  func refreshEnvironment() { environmentState = RunnerEnvironment.sample() }

  struct RemoteRequest {
    let wireID: String
    let session: RunnerSessionKey
    let connectionID: UUID
    let initializing: Bool
  }
  struct Execution {
    let id = UUID()
    let runtime: any SessionRuntime
    let request: RemoteRequest?
  }
  var execution: Execution?
  let makeRuntime: () -> any SessionRuntime
  let storage: URL
  let nativeQueue = DispatchQueue(label: "dev.modeldebugger.native", qos: .userInitiated)
  lazy var files = RunnerFiles(root: documents)
  var fileTask: Task<Void, Never>?
  var fileEpoch = UUID()
  var queuedFileCommands = 0
  static let fileQueueLimit = 16

  let transport: any RunnerTransport
  @Published var connectionStatus = "Waiting for a connection"
  #if os(macOS)
    private(set) lazy var pythonAdapter = PythonExecutionAdapter(model: self)
    @Published var pythonSessions: [RunnerSessionKey: RunnerSessionSummary] = [:]
    @Published var listenHost = "127.0.0.1"
    @Published var listenAddresses = RunnerPlatform.addresses()
    func changeAddress() {
      do {
        var config = try RunnerPlatform.connectionConfig()
        config["host"] = listenHost
        try RunnerPlatform.saveConnection(config)
        try transport.restart(documents: documents)
      } catch {
        connectionPhase = .failed
        connectionStatus = error.localizedDescription
      }
    }
    func exportConnection() {
      do { try RunnerPlatform.exportConnection() } catch { reportError(error) }
    }
  #endif
  func startTransport() {
    guard ownership.lifecycle != .ended || reuseAfterEnd else { return }
    #if os(macOS)
      pythonAdapter.attach()
    #endif
    transport.inspection = { [weak self] in self?.descriptor }
    transport.onStatus = { [weak self] phase, detail in
      guard let self else { return }
      let stopped = self.ownership.lifecycle == .ended && !self.reuseAfterEnd
      self.connectionPhase = stopped ? .unconfigured : phase
      self.connectionStatus = stopped ? "Runner stopped" : detail
      RunnerPlatform.keepAwake(self.transport.isConnected || self.busy)
    }
    transport.onDisconnect = { [weak self] in
      guard let self else { return }
      if self.owner != nil { self.endOwnedSession(reason: "Server disconnected") }
    }
    transport.onMessage = { [weak self] in self?.receiveCommand($0) }
    startHeartbeatWatchdog()
    do {
      try transport.start(documents: documents)
      #if os(macOS)
        listenHost = try RunnerPlatform.connectionConfig()["host"] as? String ?? "127.0.0.1"
      #endif
    } catch {
      connectionPhase = .failed
      connectionStatus = error.localizedDescription
    }
  }
  var canStop: Bool { busy && execution != nil }

  let debuggerEnabled: Bool

  init(
    claimDevice: @escaping @MainActor (String, String, String) throws -> Void = {
      try RunnerPlatform.claimDevice(serverID: $0, sessionID: $1, role: $2)
    },
    releaseDevice: @escaping @MainActor () -> Void = { RunnerPlatform.releaseDevice() },
    transport: (any RunnerTransport)? = nil, storage: URL? = nil,
    debuggerEnabled: Bool = NativeRunner.debuggerEnabled,
    makeRuntime: @escaping () -> any SessionRuntime = { NativeRunner() },
    heartbeatTimeout: TimeInterval = 30,
    monotonicNow: @escaping () -> TimeInterval = { ProcessInfo.processInfo.systemUptime },
    launchedForSession: Bool = ProcessInfo.processInfo.arguments.contains("--runner-session"),
    reuseAfterEnd: Bool? = nil,
    exitSessionRunner: (() -> Void)? = nil
  ) {
    self.transport = transport ?? AppleRunnerTransport()
    self.storage = storage ?? RunnerPlatform.storage
    self.debuggerEnabled = debuggerEnabled
    self.makeRuntime = makeRuntime
    self.claimDevice = claimDevice
    self.releaseDevice = releaseDevice
    self.ownership = RunnerOwnership(heartbeatTimeout: heartbeatTimeout)
    self.monotonicNow = monotonicNow
    self.launchedForSession = launchedForSession
    #if os(iOS)
      self.reuseAfterEnd = reuseAfterEnd ?? true
    #else
      self.reuseAfterEnd = reuseAfterEnd ?? false
    #endif
    self.exitSessionRunner =
      exitSessionRunner ?? {
        #if os(macOS)
          NSApplication.shared.terminate(nil)
        #endif
      }
  }

  func reportError(_ error: Error) {
    activity = .failed
    status = error.localizedDescription
  }

  var allowedActions: [String: Bool] {
    let owned =
      owner != nil || transport.isConnected || (ownership.lifecycle == .ended && !reuseAfterEnd)
    let idle = !busy && transfer == nil && activity != .releasing
    var values = [
      "refreshEnvironment": true, "importJob": idle && !owned,
      "importModel": idle && !owned && job != nil,
      "run": idle && !owned && job != nil && modelURL != nil && debuggerEnabled,
      "stop": canStop && execution?.request == nil,
      "stopGeneration": owner != nil && activity == .generating && ownership.lifecycle == .active,
      "endSession": owner != nil && ownership.lifecycle == .active,
      "exportCapture": idle && !owned && completedRun != nil,
    ]
    #if os(macOS)
      values["setAddress"] = idle && !owned
      values["refreshInterfaces"] = idle && !owned
      values["exportConnection"] = true
    #endif
    return values
  }

  var documents: URL {
    storage
  }
}
