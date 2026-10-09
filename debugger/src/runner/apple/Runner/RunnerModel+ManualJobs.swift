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

/// Locally imported jobs and native execution shared by remote and manual runs.
extension RunnerModel {
  #if DEBUG
    // Exercise the real C API on a simulator or connected device using files
    // staged in the app container. Release builds omit this launch fixture.
    func runCaptureFixture() {
      do {
        let data = try Data(contentsOf: documents.appendingPathComponent("smoke-job.json"))
        let value = try JSONDecoder().decode(CaptureJob.self, from: data)
        try value.validate()
        job = value
        modelURL = documents.appendingPathComponent("smoke-model.litertlm")
        start()
      } catch { reportError(error) }
    }
  #endif

  func importFile(_ url: URL, isJob: Bool) {
    guard allowedActions[isJob ? "importJob" : "importModel"] == true else { return }
    let scoped = url.startAccessingSecurityScopedResource()
    let root = documents
    let expectedHash = job?.modelSHA256
    busy = true
    activity = .importing
    completedRun = nil
    captureCount = nil
    status = isJob ? "Reading job" : "Importing and verifying model"
    DispatchQueue.global(qos: .userInitiated).async {
      defer { if scoped { url.stopAccessingSecurityScopedResource() } }
      do {
        if isJob {
          let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? Int.max
          guard size <= 1024 * 1024 else { throw RunnerError("Job JSON is too large.") }
          let job = try JSONDecoder().decode(CaptureJob.self, from: Data(contentsOf: url))
          try job.validate()
          let cached = root.appendingPathComponent("Models/\(job.modelSHA256).litertlm")
          let exists = FileManager.default.fileExists(atPath: cached.path)
          Task { @MainActor in
            self.job = job
            self.modelURL = exists ? cached : nil
            self.busy = false
            self.activity = .idle
            self.output = ""
            self.status =
              exists ? "Model cached · ready to run" : "Import the matching .litertlm model."
          }
        } else {
          guard let expectedHash else { throw RunnerError("Import the job JSON first.") }
          guard url.pathExtension.lowercased() == "litertlm" else {
            throw RunnerError("Choose a .litertlm model.")
          }
          let directory = root.appendingPathComponent("Models", isDirectory: true)
          try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
          let pending = directory.appendingPathComponent(UUID().uuidString + ".pending")
          defer { try? FileManager.default.removeItem(at: pending) }
          try FileManager.default.copyItem(at: url, to: pending)
          let hash = try sha256(of: pending)
          guard hash == expectedHash else {
            throw RunnerError(
              "This model does not match the job. Import the prepared, tapped model.")
          }
          let destination = directory.appendingPathComponent(hash + ".litertlm")
          if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
          }
          try FileManager.default.moveItem(at: pending, to: destination)
          Task { @MainActor in
            self.modelURL = destination
            self.busy = false
            self.activity = .idle
            self.status = "Model verified · ready to run"
          }
        }
      } catch {
        Task { @MainActor in
          self.busy = false
          self.reportError(error)
        }
      }
    }
  }

  func start() {
    guard allowedActions["run"] == true, let job, let modelURL else { return }
    begin(job: job, model: modelURL, request: nil)
  }

  func begin(job: CaptureJob, model: URL, request: RemoteRequest?) {
    let worker = request.flatMap { sessions[$0.session]?.runtime } ?? makeRuntime()
    let operation = Execution(runtime: worker, request: request)
    execution = operation
    self.job = job
    modelURL = model
    busy = true
    let initializing = request?.initializing == true
    activity = initializing || request == nil ? .verifying : .generating
    status = activity == .verifying ? "Verifying model" : "Running · " + job.backend
    completedRun = nil
    captureCount = nil
    output = ""
    RunnerPlatform.keepAwake(true)
    let destination = documents.appendingPathComponent(
      "Runs/\(request == nil ? UUID().uuidString : job.id.uuidString)", isDirectory: true)
    let provenance = HostBundle.facts.runtimeBuildURL
    nativeQueue.async {
      do {
        var result = try worker.execute(
          job: job, model: model, destination: destination,
          progress: { value in
            DispatchQueue.main.async {
              MainActor.assumeIsolated {
                guard self.isCurrent(operation) else { return }
                if self.activity != .stopping {
                  self.activity = RunnerActivity(rawValue: value.rawValue) ?? .idle
                  self.status = value.description
                }
                if let request {
                  self.transport.send(
                    RunnerCommandCodec.progress(
                      requestID: request.wireID, message: value.description)
                  )
                }
              }
            }
          },
          delta: { value in
            DispatchQueue.main.async {
              MainActor.assumeIsolated {
                guard self.isCurrent(operation) else { return }
                self.output += value
                if let request {
                  self.transport.send(
                    RunnerCommandCodec.delta(requestID: request.wireID, text: value))
                }
              }
            }
          }, initializeOnly: initializing, persistent: request != nil)
        if let provenance {
          do {
            try FileManager.default.copyItem(
              at: provenance, to: destination.appendingPathComponent("runtime-build.json"))
          } catch {
            result.dumpStatus = "unavailable"
            result.dumpError = error.localizedDescription
          }
        }
        let completed = result
        DispatchQueue.main.async {
          MainActor.assumeIsolated {
            guard self.isCurrent(operation) else {
              self.finishDisconnected(operation)
              return
            }
            self.output = completed.output
            self.captureCount = completed.tensors.count
            self.environmentState = completed.environment?.after ?? RunnerEnvironment.sample()
            self.completedRun = destination
            self.activity = .idle
            if let request, self.sessions[request.session]?.runtime === worker {
              let key = request.session
              self.sessions[key]?.summary = .init(
                key: key.wireKey, id: key.id.uuidString.lowercased(), role: key.role.rawValue,
                modelHash: job.modelSHA256, tokens: completed.tokenCount,
                contextLength: job.contextLength,
                capturePoints: job.manifest.taps.count, backend: completed.backend)
            }
            self.status =
              initializing
              ? "Session loaded · Conversation and KV cache retained"
              : "Captured \(completed.tensors.count) tensors · \(completed.uncapturedPoints.count) points not executed"
            if let request {
              self.transport.send(
                RunnerCommandCodec.completed(
                  requestID: request.wireID, jobID: job.id, result: completed))
            }
            self.finish(operation)
          }
        }
      } catch {
        DispatchQueue.main.async {
          MainActor.assumeIsolated {
            guard self.isCurrent(operation) else {
              self.finishDisconnected(operation)
              return
            }
            let terminal =
              (try? Data(contentsOf: destination.appendingPathComponent("terminal.json")))
              .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? [:]
            let generationStatus = terminal["generationStatus"] as? String ?? "failed"
            if let request, self.sessions[request.session]?.runtime === worker {
              if let chat = self.sessions[request.session]?.activeChatID {
                self.endedChatIDs.insert(chat)
              }
              self.sessions[request.session]?.activeChatID = nil
              if !request.initializing && generationStatus != "stopped" {
                self.sessionPoisoned = true
              }
              self.nativeQueue.async { worker.closeChat() }
            }
            self.reportError(error)
            if let request {
              var extra: [String: Any] = [
                "generationStatus": generationStatus,
                "errorCode": generationStatus == "stopped" ? "stopped" : "failed",
              ]
              if !terminal.isEmpty { extra["runDirectory"] = "Documents/Runs/" + job.id.uuidString }
              self.transport.send(
                RunnerCommandCodec.error(
                  for: request.wireID, message: error.localizedDescription, extra: extra))
            }
            self.finish(operation)
          }
        }
      }
    }
  }

  /// Native events are scoped to both an execution and its original peer.
  func isCurrent(_ operation: Execution) -> Bool {
    guard execution?.id == operation.id else { return false }
    return operation.request.map { $0.connectionID == transport.connectionID } ?? true
  }

  func finishDisconnected(_ operation: Execution) {
    guard execution?.id == operation.id else { return }
    output = ""
    captureCount = nil
    completedRun = nil
    activity = .idle
    status = "Connection lost · initialize the Session again"
    finish(operation)
  }

  func stop() {
    #if os(macOS)
      if pythonAdapter.stop() { return }
    #endif
    guard let execution else { return }
    execution.runtime.cancel()
    activity = .stopping
    status = "Stopping · waiting for the native runtime"
  }

  func finish(_ operation: Execution) {
    guard execution?.id == operation.id else { return }
    busy = false
    execution = nil
    RunnerPlatform.keepAwake(transport.isConnected)
  }
}
