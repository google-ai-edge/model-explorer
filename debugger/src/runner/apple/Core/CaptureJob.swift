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

import CryptoKit
import Foundation

struct RunnerError: LocalizedError {
  let message: String
  init(_ message: String) { self.message = message }
  var errorDescription: String? { message }
}

struct CaptureJob {
  let formatVersion: Int
  let id: UUID
  let modelSHA256: String  // Entire .litertlm container, not its embedded TFLite section.
  let prompt: String
  let contextLength: Int
  let maxOutputTokens: Int
  let manifest: TapManifest
  var messages: [HistoryMessage]?
  var backend = "CPU"
  /// Expected pre-turn Conversation state (`runtimeInstance`, `turnSequence`, `tokenCount`)
  /// recorded by the Server after the previous turn. Verified before the incoming prompt
  /// touches the native KV cache so out-of-sync or restarted Conversations fail cleanly.
  var expect: Expectation?

  /// Snapshot of the resident Conversation state expected before executing a turn.
  struct Expectation: Codable {
    var runtimeInstance: String?
    let turnSequence: Int
    let tokenCount: Int
  }

  static let supportedContextLengths = [1024, 4096]
  static let maxPromptBytes = 65536

  func validate() throws {
    guard formatVersion == 1,
      modelSHA256.count == 64, modelSHA256.allSatisfy({ $0.isHexDigit }),
      !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      prompt.utf8.count <= Self.maxPromptBytes,
      Self.supportedContextLengths.contains(contextLength),
      ["CPU", "GPU"].contains(backend),
      (1...32).contains(maxOutputTokens), maxOutputTokens < contextLength,
      (1...16).contains(manifest.taps.count),
      manifest.formatVersion == 1,
      manifest.tappedSHA256.count == 64, manifest.tappedSHA256.allSatisfy({ $0.isHexDigit })
    else {
      throw RunnerError(
        "Invalid job. Use text, CPU or GPU, a supported context length and up to 16 capture points."
      )
    }
    if let messages {
      guard messages.count <= 128,
        messages.enumerated().allSatisfy({ index, message in
          message.role == (index % 2 == 0 ? "user" : "assistant")
            && message.content.allSatisfy { $0.type == "text" }
            && message.content.reduce(0, { $0 + $1.text.utf8.count }) <= 16384
        }), messages.count % 2 == 0
      else { throw RunnerError("Invalid conversation history.") }
    }
    let identities = manifest.taps.map { $0.signature + ":" + $0.outputName }
    guard Set(identities).count == identities.count else {
      throw RunnerError("Duplicate capture points in job.")
    }
    for tap in manifest.taps {
      guard !tap.signature.isEmpty, !tap.outputName.isEmpty,
        tap.subgraph >= 0, tap.op >= 0, tap.output >= 0, tap.tensor >= 0,
        !tap.shape.isEmpty, tap.shape.allSatisfy({ $0 > 0 }),
        tensorTypes[tap.tensorType] != nil
      else {
        throw RunnerError("Unsupported capture point in job.")
      }
    }
  }
}

extension CaptureJob: Codable {
  private enum CodingKeys: String, CodingKey {
    case formatVersion
    case id
    case modelSHA256
    case prompt
    case contextLength
    case maxOutputTokens
    case manifest
    case messages
    case backend
    case expect
  }

  init(from decoder: Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    formatVersion = try values.decode(Int.self, forKey: .formatVersion)
    id = try values.decode(UUID.self, forKey: .id)
    modelSHA256 = try values.decode(String.self, forKey: .modelSHA256)
    prompt = try values.decode(String.self, forKey: .prompt)
    contextLength = try values.decode(Int.self, forKey: .contextLength)
    maxOutputTokens = try values.decode(Int.self, forKey: .maxOutputTokens)
    manifest = try values.decode(TapManifest.self, forKey: .manifest)
    messages = try values.decodeIfPresent([HistoryMessage].self, forKey: .messages)
    backend = try values.decodeIfPresent(String.self, forKey: .backend) ?? "CPU"
    expect = try values.decodeIfPresent(Expectation.self, forKey: .expect)
  }
}

struct HistoryMessage: Codable {
  let role: String
  let content: [HistoryText]
}

struct HistoryText: Codable {
  let type: String
  let text: String
}

struct TapManifest: Codable {
  let formatVersion: Int
  let sourceSHA256: String
  // Prepared TFLite section; mapping is verified by the Mac preparation/import tools.
  let tappedSHA256: String
  let taps: [TapPoint]

  enum CodingKeys: String, CodingKey {
    case formatVersion = "format_version"
    case sourceSHA256 = "source_sha256"
    case tappedSHA256 = "tapped_sha256"
    case taps
  }
}

struct TapPoint: Codable {
  let signature: String
  let subgraph: Int
  let op: Int
  let output: Int
  let tensor: Int
  let tensorName: String
  let outputName: String
  let shape: [Int]
  let tensorType: Int

  enum CodingKeys: String, CodingKey {
    case signature
    case subgraph
    case op
    case output
    case tensor
    case tensorName = "tensor_name"
    case outputName = "output_name"
    case shape
    case tensorType = "tensor_type"
  }
}

// TFLite schema enum -> Safetensors dtype and element width.
private let tensorTypes: [Int: (String, Int)] = [
  0: ("F32", 4), 1: ("F16", 2), 2: ("I32", 4), 3: ("U8", 1),
  4: ("I64", 8), 6: ("BOOL", 1), 9: ("I8", 1),
]

func sha256(of url: URL, checkCancellation: () throws -> Void = {}) throws -> String {
  let handle = try FileHandle(forReadingFrom: url)
  defer { try? handle.close() }
  var hash = SHA256()
  while let block = try handle.read(upToCount: 4 * 1024 * 1024), !block.isEmpty {
    try checkCancellation()
    hash.update(data: block)
  }
  return hash.finalize().map { String(format: "%02x", $0) }.joined()
}

func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
  try encoder.encode(value).write(to: url, options: .atomic)
}

struct CapturedTensor: Codable {
  let path: String
  let sha256: String
  let signature: String
  let step: Int
  let key: String
  let shape: [Int]
  let dtype: String
}

// Read only the header, never load a large tensor payload into Swift memory.
func inspectCapture(_ url: URL, taps: [TapPoint]) throws -> CapturedTensor? {
  let file = try FileHandle(forReadingFrom: url)
  defer { try? file.close() }
  guard let prefix = try file.read(upToCount: 8), prefix.count == 8 else {
    throw RunnerError("Truncated Safetensors header: \(url.lastPathComponent)")
  }
  let length = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << ($1.offset * 8) }
  guard length > 0, length <= 1024 * 1024,
    let data = try file.read(upToCount: Int(length)), data.count == Int(length),
    let header = try JSONSerialization.jsonObject(with: data) as? [String: Any],
    let metadata = header["__metadata__"] as? [String: String],
    let signature = metadata["signature"]
  else {
    throw RunnerError("Invalid Safetensors metadata: \(url.lastPathComponent)")
  }
  let selected = taps.filter {
    $0.signature == signature && header["post_" + $0.outputName] != nil
  }
  guard !selected.isEmpty else { return nil }
  guard selected.count == 1, let tap = selected.first,
    let step = metadata["step"].flatMap(Int.init), step >= 0,
    let tensor = header["post_" + tap.outputName] as? [String: Any],
    let shape = tensor["shape"] as? [Int], shape == tap.shape,
    let tensorTypeEntry = tensorTypes[tap.tensorType]
  else {
    throw RunnerError("Capture does not match selected tensor: \(url.lastPathComponent)")
  }
  let (expectedDtype, elementBytes) = tensorTypeEntry
  guard
    let dtype = tensor["dtype"] as? String, dtype == expectedDtype,
    let offsets = tensor["data_offsets"] as? [Int], offsets.count == 2, offsets[0] == 0
  else {
    throw RunnerError("Capture does not match selected tensor: \(url.lastPathComponent)")
  }
  var bytes = elementBytes
  for dimension in shape {
    let product = bytes.multipliedReportingOverflow(by: dimension)
    guard !product.overflow else { throw RunnerError("Tensor size overflow.") }
    bytes = product.partialValue
  }
  let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize
  guard offsets[1] == bytes, size == 8 + Int(length) + bytes else {
    throw RunnerError("Truncated or inconsistent tensor payload: \(url.lastPathComponent)")
  }
  return CapturedTensor(
    path: "tensors/" + url.lastPathComponent, sha256: try sha256(of: url),
    signature: signature, step: step, key: "post_" + tap.outputName,
    shape: shape, dtype: dtype)
}
