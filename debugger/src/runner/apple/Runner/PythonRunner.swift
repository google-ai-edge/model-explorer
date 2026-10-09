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

  /// App-owned supervisor. Configuration is local; peers cannot choose executables.
  @MainActor
  final class PythonRunner {
    struct Configuration: Decodable {
      let version: Int
      let executable: String
      let packageRoot: String
      let contractPackageRoot: String
      let module: String
      let runtimeRoot: String
      let backends: [String]

      var supervisorModule: String { module }

      var pythonPath: String {
        [packageRoot, contractPackageRoot, runtimeRoot].joined(separator: ":")
      }

      var isSupported: Bool {
        let files = FileManager.default
        guard version == 2, module == "model_debugger_runner.python_runner_host",
          files.isExecutableFile(atPath: executable), !backends.isEmpty,
          backends.allSatisfy({ ["CPU", "MPS", "CUDA"].contains($0) })
        else { return false }
        return files.fileExists(
          atPath: packageRoot + "/model_debugger_runner/python_runner_host.py")
          && files.fileExists(atPath: contractPackageRoot + "/model_debugger_contracts/__init__.py")
      }
    }
    let configuration: Configuration?
    private var process: Process?
    private var input: FileHandle?
    private var generation = UUID()
    private var shutdownTask: Task<Void, Never>?
    private var resets: [String: CheckedContinuation<Void, Error>] = [:]
    var onMessage: (([String: Any]) -> Void)?
    var onExit: (() -> Void)?

    init() {
      let path = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(
          "Library/Application Support/Model Debugger Runner/python-runner.json")
      let config = (try? Data(contentsOf: path)).flatMap {
        try? JSONDecoder().decode(Configuration.self, from: $0)
      }
      if let config, config.isSupported {
        configuration = config
      } else {
        configuration = nil
      }
    }

    func send(_ value: [String: Any]) throws {
      if process == nil { try start() }
      guard let input else { throw RunnerError("Python Runner is unavailable.") }
      var data = try JSONSerialization.data(withJSONObject: value)
      data.append(10)
      try input.write(contentsOf: data)
    }

    private func start() throws {
      guard let config = configuration else {
        throw RunnerError("Configure the Runner's local PyTorch environment first.")
      }
      let task = Process()
      let incoming = Pipe()
      let outgoing = Pipe()
      task.executableURL = URL(fileURLWithPath: config.executable)
      task.arguments = ["-m", config.supervisorModule, "--root", config.runtimeRoot]
      task.environment = ProcessInfo.processInfo.environment.merging([
        "PYTHONPATH": config.pythonPath,
        "PYTHONUNBUFFERED": "1",
      ]) { _, new in new }
      task.standardInput = incoming
      task.standardOutput = outgoing
      task.standardError = FileHandle.nullDevice
      let epoch = UUID()
      generation = epoch
      try task.run()
      process = task
      input = incoming.fileHandleForWriting
      let (lines, continuation) = AsyncStream<Data>.makeStream()
      DispatchQueue.global(qos: .userInitiated).async {
        defer { continuation.finish() }
        var pending = Data()
        while true {
          let data = outgoing.fileHandleForReading.availableData
          if data.isEmpty { break }
          pending.append(data)
          if pending.count > 16 * 1024 * 1024 { break }
          while let end = pending.firstIndex(of: 10) {
            let line = Data(pending[..<end])
            pending.removeSubrange(...end)
            continuation.yield(line)
          }
        }
      }
      Task { @MainActor [weak self] in
        for await line in lines {
          guard let self, self.generation == epoch else { return }
          guard
            let message = (try? JSONSerialization.jsonObject(with: line)) as? [String: Any]
          else { break }
          if let identity = message["requestId"] as? String,
            let pending = self.resets.removeValue(forKey: identity)
          {
            if message["type"] as? String == "reset" {
              pending.resume()
            } else {
              pending.resume(
                throwing: RunnerError(
                  message["error"] as? String ?? "Python Chat reset failed."))
            }
          } else {
            self.onMessage?(message)
          }
        }
        guard let self, self.generation == epoch else { return }
        self.shutdown()
        self.onExit?()
      }
    }

    func shutdown() {
      beginShutdown()
    }

    /// End the Chat while keeping the supervisor and loaded model alive.
    func resetAndWait() async throws {
      guard process != nil else { return }
      let identity = UUID().uuidString.lowercased()
      try await withCheckedThrowingContinuation { (done: CheckedContinuation<Void, Error>) in
        resets[identity] = done
        do { try send(["type": "reset", "requestId": identity]) } catch {
          resets.removeValue(forKey: identity)?.resume(throwing: error)
        }
        Task { @MainActor [weak self] in
          try? await Task.sleep(for: .seconds(30))
          self?.resets.removeValue(forKey: identity)?.resume(
            throwing: RunnerError("Python Chat reset timed out."))
        }
      }
    }

    /// Await child reaping before advertising the device as released.
    func shutdownAndWait() async {
      beginShutdown()
      await shutdownTask?.value
    }

    private func beginShutdown() {
      generation = UUID()
      let pending = resets.values
      resets.removeAll()
      for reset in pending {
        reset.resume(throwing: RunnerError("Python Runner disconnected during Chat reset."))
      }
      // EOF asks the supervisor to cancel, join and reap all model workers.
      try? input?.close()
      input = nil
      guard let task = process else { return }
      process = nil
      let previous = shutdownTask
      shutdownTask = Task {
        await previous?.value
        let deadline = Date().addingTimeInterval(12)
        while task.isRunning && Date() < deadline {
          try? await Task.sleep(for: .milliseconds(100))
        }
        if task.isRunning { task.terminate() }
        let terminated = Date().addingTimeInterval(3)
        while task.isRunning && Date() < terminated {
          try? await Task.sleep(for: .milliseconds(100))
        }
        if task.isRunning { kill(task.processIdentifier, SIGKILL) }
        while task.isRunning {
          try? await Task.sleep(for: .milliseconds(10))
        }
        task.waitUntilExit()
      }
    }
  }
#endif
