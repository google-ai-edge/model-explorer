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

/// Real pinned C API evidence. Requires an already prepared local model.
@main struct ChatLifecycleSmoke {
  static func main() throws {
    let args = CommandLine.arguments
    guard args.count == 4 else {
      throw RunnerError(
        "Usage: chat-lifecycle-smoke job.json model.litertlm NEW_OUTPUT_DIRECTORY")
    }
    let fixture = try JSONDecoder().decode(
      CaptureJob.self, from: Data(contentsOf: URL(fileURLWithPath: args[1])))
    let model = URL(fileURLWithPath: args[2])
    let root = URL(fileURLWithPath: args[3])
    let worker = NativeRunner(captureBenchmarks: true)
    defer { worker.close() }
    let lock = NSLock()
    var loads = 0
    func job(_ prompt: String, limit: Int = 2) -> CaptureJob {
      CaptureJob(
        formatVersion: 1, id: UUID(), modelSHA256: fixture.modelSHA256, prompt: prompt,
        contextLength: 1024, maxOutputTokens: limit, manifest: fixture.manifest)
    }
    func execute(
      _ value: CaptureJob, name: String, initialize: Bool = false, cancelOnDelta: Bool = false
    ) throws -> RunnerResult {
      try worker.execute(
        job: value, model: model, destination: root.appendingPathComponent(name),
        progress: { phase in
          if phase == .loading { lock.withLock { loads += 1 } }
          print("\(name): \(phase)")
        }, delta: { _ in if cancelOnDelta { worker.cancel() } }, initializeOnly: initialize,
        persistent: true)
    }
    let first = try execute(job("Initialize."), name: "initialize", initialize: true)
    let prompt = "Reply with the word hello."
    let check = try worker.preflight(prompt: prompt, contextLimit: 1024)
    let repeated = try worker.preflight(prompt: prompt, contextLimit: 1024)
    guard check.contextTokenCount == repeated.contextTokenCount,
      check.inputTokenCount == repeated.inputTokenCount
    else {
      throw RunnerError("Preflight mutated the Conversation.")
    }
    let generated = try execute(job(prompt), name: "successful")
    guard generated.tokenCountBefore == check.contextTokenCount,
      generated.prefillTokenCount == check.inputTokenCount
    else {
      throw RunnerError(
        "Preflight count \(check.inputTokenCount) differs from actual prefill \(String(describing: generated.prefillTokenCount))."
      )
    }
    print(
      "PASS exact initial input capacity: \(check.contextTokenCount) + \(check.inputTokenCount)")
    let nextPrompt = "Please write a long detailed explanation of how computers work."
    let next = try worker.preflight(prompt: nextPrompt, contextLimit: 1024)
    let nextAgain = try worker.preflight(prompt: nextPrompt, contextLimit: 1024)
    guard next.contextTokenCount == generated.tokenCount,
      next.inputTokenCount == nextAgain.inputTokenCount
    else {
      throw RunnerError("Repeated preflight changed resident history.")
    }
    let second = try execute(job(nextPrompt), name: "second-turn")
    guard second.prefillTokenCount == next.inputTokenCount else {
      throw RunnerError(
        "Later-turn preflight \(next.inputTokenCount) differs from actual \(String(describing: second.prefillTokenCount))."
      )
    }
    print(
      "PASS exact later-turn input capacity: \(next.contextTokenCount) + \(next.inputTokenCount)")
    do {
      _ = try execute(job(nextPrompt, limit: 32), name: "stopped", cancelOnDelta: true)
      throw RunnerError("Cancellation did not stop the generation.")
    } catch {
      let terminal =
        try JSONSerialization.jsonObject(
          with: Data(contentsOf: root.appendingPathComponent("stopped/terminal.json")))
        as? [String: Any]
      let files = try FileManager.default.contentsOfDirectory(
        atPath: root.appendingPathComponent("stopped/raw").path)
      guard terminal?["generationStatus"] as? String == "stopped", !files.isEmpty else {
        throw error
      }
      print("PASS stopped raw files sealed before Chat release: \(files.count)")
    }
    worker.closeChat()
    let fresh = try execute(job("Initialize."), name: "new-chat", initialize: true)
    guard fresh.runtimeInstance == first.runtimeInstance, loads == 1, fresh.turnSequence == 0 else {
      throw RunnerError("New Chat reloaded model or reused old Chat state.")
    }
    let freshCheck = try worker.preflight(prompt: prompt, contextLimit: 1024)
    guard freshCheck.contextTokenCount == check.contextTokenCount,
      freshCheck.inputTokenCount == check.inputTokenCount
    else {
      throw RunnerError("New Chat retained old context.")
    }
    let final = try execute(job(prompt), name: "new-chat-success")
    guard final.prefillTokenCount == freshCheck.inputTokenCount,
      final.runtimeInstance == first.runtimeInstance
    else {
      throw RunnerError("New Chat preflight or model identity changed.")
    }
    print("PASS Stop -> closeChat -> initialize keeps one Engine and clears Conversation/KV")
    print(
      "PASS REAL NATIVE CHAT LIFECYCLE: one model load, exact tokenizer preflight, stopped raw export, fresh Chat"
    )
  }
}
