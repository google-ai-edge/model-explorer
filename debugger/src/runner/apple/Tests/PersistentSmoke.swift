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

@main
struct PersistentSmoke {
  static func main() {
    do { try run() } catch {
      fputs(error.localizedDescription + "\n", stderr)
      exit(1)
    }
  }
  static func run() throws {
    let args = CommandLine.arguments
    guard args.count == 4 else {
      throw RunnerError("Usage: persistent-smoke job.json model.litertlm NEW_OUTPUT_DIRECTORY")
    }
    let fixture = try JSONDecoder().decode(
      CaptureJob.self, from: Data(contentsOf: URL(fileURLWithPath: args[1])))
    let model = URL(fileURLWithPath: args[2])
    let root = URL(fileURLWithPath: args[3])
    let worker = NativeRunner()
    defer { worker.close() }
    var previous: RunnerResult?
    for (index, prompt) in [
      "Initialize.", "Reply with the word hello.",
      "What word did I ask you to reply with? Reply with that word only.",
    ].enumerated() {
      var job = CaptureJob(
        formatVersion: 1, id: UUID(), modelSHA256: fixture.modelSHA256,
        prompt: prompt, contextLength: 1024, maxOutputTokens: 8, manifest: fixture.manifest)
      // What the Server sends from the second message on: where the previous result left the Conversation.
      if index > 0, let previous {
        job.expect = .init(
          runtimeInstance: previous.runtimeInstance.uuidString.lowercased(),
          turnSequence: previous.turnSequence, tokenCount: previous.tokenCount)
      }
      let result = try worker.execute(
        job: job, model: model, destination: root.appendingPathComponent(String(index)),
        progress: { print($0) }, delta: { print($0) }, initializeOnly: index == 0, persistent: true)
      if let previous {
        guard result.runtimeInstance == previous.runtimeInstance,
          result.tokenCountBefore == previous.tokenCount,
          result.turnSequence == index,
          result.output.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "hello",
          !result.tensors.isEmpty
        else { throw RunnerError("Persistent Conversation proof failed.") }
      }
      print(
        "TURN \(index) instance=\(result.runtimeInstance) tokens=\(result.tokenCountBefore)->\(result.tokenCount) captures=\(result.tensors.count)"
      )
      previous = result
    }
    if let previous {
      var stale = CaptureJob(
        formatVersion: 1, id: UUID(), modelSHA256: fixture.modelSHA256,
        prompt: "Must be refused", contextLength: 1024, maxOutputTokens: 8,
        manifest: fixture.manifest)
      stale.expect = .init(
        turnSequence: previous.turnSequence - 1, tokenCount: previous.tokenCountBefore)
      do {
        _ = try worker.execute(
          job: stale, model: model, destination: root.appendingPathComponent("stale"),
          progress: { _ in }, delta: { _ in fatalError("A refused message produced text") },
          persistent: true)
        throw RunnerError("A stale Conversation expectation was accepted")
      } catch let error as RunnerError {
        guard error.message.contains("Nothing was generated") else { throw error }
      }
      print("PASS: a message expecting an earlier Conversation state is refused before generation")
    }
    worker.close()
    let job = CaptureJob(
      formatVersion: 1, id: UUID(), modelSHA256: fixture.modelSHA256,
      prompt: "Must fail", contextLength: 1024, maxOutputTokens: 8, manifest: fixture.manifest)
    do {
      _ = try worker.execute(
        job: job, model: model, destination: root.appendingPathComponent("after-close"),
        progress: { _ in }, delta: { _ in }, persistent: true)
      throw RunnerError("Closed Session accepted generation")
    } catch let error as RunnerError {
      guard error.message.contains("Initialize before") else { throw error }
    }
    print("PASS: close releases Session; generate cannot silently reopen it")
  }
}
