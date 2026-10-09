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
struct ValidationTests {
  static func main() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let point = TapPoint(
      signature: "decode", subgraph: 0, op: 10, output: 0, tensor: 354,
      tensorName: "test", outputName: "tap", shape: [1, 2], tensorType: 0)
    func makeFile(step: String = "7", shape: [Int] = [1, 2], dtype: String = "F32", bytes: Int = 8)
      throws -> URL
    {
      let header: [String: Any] = [
        "__metadata__": ["signature": "decode", "step": step],
        "post_tap": ["shape": shape, "dtype": dtype, "data_offsets": [0, 8]],
      ]
      let json = try JSONSerialization.data(withJSONObject: header)
      var length = UInt64(json.count).littleEndian
      var data = withUnsafeBytes(of: &length) { Data($0) }
      data.append(json)
      data.append(Data(repeating: 0, count: bytes))
      let file = directory.appendingPathComponent(UUID().uuidString + ".safetensors")
      try data.write(to: file)
      return file
    }
    var count = 0
    func rejects(_ label: String, _ operation: () throws -> Void) throws {
      do { try operation() } catch {
        count += 1
        print("PASS \(label)")
        return
      }
      throw RunnerError("Expected rejection: \(label)")
    }
    let valid = try inspectCapture(makeFile(), taps: [point])
    guard valid?.shape == [1, 2], valid?.step == 7, valid?.dtype == "F32" else {
      throw RunnerError("Valid synthetic tensor was not parsed.")
    }
    count += 1
    try rejects("negative step") { _ = try inspectCapture(makeFile(step: "-1"), taps: [point]) }
    try rejects("shape mismatch") {
      _ = try inspectCapture(makeFile(shape: [2, 1]), taps: [point])
    }
    try rejects("dtype mismatch") { _ = try inspectCapture(makeFile(dtype: "I32"), taps: [point]) }
    try rejects("truncated payload") { _ = try inspectCapture(makeFile(bytes: 4), taps: [point]) }
    let invalid = directory.appendingPathComponent("invalid.safetensors")
    try Data(repeating: 255, count: 8).write(to: invalid)
    try rejects("oversized header") { _ = try inspectCapture(invalid, taps: [point]) }
    let unrelated = try inspectCapture(makeFile(), taps: [])
    guard unrelated == nil else { throw RunnerError("Unselected tensor was exported.") }
    count += 1
    let hash = String(repeating: "a", count: 64)
    let manifest = TapManifest(
      formatVersion: 1, sourceSHA256: hash, tappedSHA256: hash, taps: [point, point])
    let job = CaptureJob(
      formatVersion: 1, id: UUID(), modelSHA256: hash, prompt: "hello", contextLength: 1024,
      maxOutputTokens: 2, manifest: manifest)
    try rejects("duplicate capture identities") { try job.validate() }
    let singleManifest = TapManifest(
      formatVersion: 1, sourceSHA256: hash, tappedSHA256: hash, taps: [point])
    let baseJob = CaptureJob(
      formatVersion: 1, id: UUID(), modelSHA256: hash, prompt: "hello", contextLength: 1024,
      maxOutputTokens: 2, manifest: singleManifest)
    var wire =
      try JSONSerialization.jsonObject(with: JSONEncoder().encode(baseJob)) as! [String: Any]
    wire["backend"] = "GPU"
    let gpuJob = try JSONDecoder().decode(
      CaptureJob.self, from: JSONSerialization.data(withJSONObject: wire))
    try gpuJob.validate()
    let encodedGPU =
      try JSONSerialization.jsonObject(with: JSONEncoder().encode(gpuJob)) as! [String: Any]
    guard encodedGPU["backend"] as? String == "GPU" else {
      throw RunnerError("GPU backend was lost across the native job wire contract.")
    }
    count += 1
    wire["contextLength"] = 4096
    wire["prompt"] = String(repeating: "evaluation passage ", count: 600)
    let longJob = try JSONDecoder().decode(
      CaptureJob.self, from: JSONSerialization.data(withJSONObject: wire))
    try longJob.validate()
    count += 1
    wire["backend"] = "TPU"
    try rejects("unsupported backend") {
      try JSONDecoder().decode(CaptureJob.self, from: JSONSerialization.data(withJSONObject: wire))
        .validate()
    }
    wire.removeValue(forKey: "backend")
    let legacyJob = try JSONDecoder().decode(
      CaptureJob.self, from: JSONSerialization.data(withJSONObject: wire))
    let encodedLegacy =
      try JSONSerialization.jsonObject(with: JSONEncoder().encode(legacyJob)) as! [String: Any]
    guard encodedLegacy["backend"] as? String == "CPU" else {
      throw RunnerError("Legacy native job must default to CPU.")
    }
    count += 1
    wire["contextLength"] = 8192
    try rejects("unverified context capacity") {
      try JSONDecoder().decode(CaptureJob.self, from: JSONSerialization.data(withJSONObject: wire))
        .validate()
    }
    let text = try conversationText(
      from: "{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"hello\"}]}")
    guard text == "hello" else {
      throw RunnerError("Conversation JSON was treated as plain text.")
    }
    count += 1
    try rejects("malformed stream JSON") { _ = try conversationText(from: "hello") }
    let worker = NativeRunner()
    worker.cancel()
    try rejects("cancel before initialization") { try worker.checkCancellation() }
    print("\(count) validation checks passed. Tensor fixtures are synthetic.")
  }
}
