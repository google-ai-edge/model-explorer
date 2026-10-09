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

/// Bounded, acknowledged chunks keep large models out of WebSocket buffers.
/// Paths are generated locally; the peer can only read completed run exports.
struct RunnerFileResponse: Sendable {
  let data: Data
  /// Raw bytes for a binary reply frame (a v5 `file_chunk` asked for with `encoding: binary`).
  let payload: Data?
  let transfer: RunnerTransferProgress?
}

actor RunnerFiles {
  static let chunkSize = 512 * 1024
  private struct ActiveUpload {
    let hash: String
    let size: UInt64
    var offset: UInt64
    let file: FileHandle
    let url: URL
  }
  private let root: URL
  private var upload: ActiveUpload?
  private var hasher = SHA256()
  private struct ExportFile {
    let path: String
    let size: Int
    let sha256: String
    let modified: Date?
    var wire: [String: Any] { ["path": path, "size": size, "sha256": sha256] }
  }
  private struct Export {
    let id: String
    let status: String
    let files: [ExportFile]
  }
  private var exports: [UUID: Export] = [:]
  var uploading: Bool { upload != nil }
  var transferProgress: RunnerTransferProgress? {
    upload.map { .init(received: $0.offset, total: $0.size) }
  }
  init(root: URL) { self.root = root }

  func abort() {
    if let value = upload {
      try? value.file.close()
      try? FileManager.default.removeItem(at: value.url)
    }
    upload = nil
    hasher = SHA256()
  }

  /// `payload` carries the bytes of a binary request frame (a v5 `upload_chunk`).
  func handle(_ data: Data, payload: Data? = nil) throws -> RunnerFileResponse {
    try Task.checkCancellation()
    var outgoing: Data?
    guard let message = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      let reply = try handleMessage(message, payload: payload, outgoing: &outgoing)
    else { throw RunnerError("Invalid file command.") }
    return RunnerFileResponse(
      data: try JSONSerialization.data(withJSONObject: reply), payload: outgoing,
      transfer: transferProgress)
  }

  private func handleMessage(_ message: [String: Any], payload: Data?, outgoing: inout Data?) throws
    -> [String: Any]?
  {
    let kind = message["type"] as? String ?? ""
    guard RunnerCommandCodec.fileCommands.contains(kind) else { return nil }
    var reply: [String: Any] = ["type": kind, "requestId": message["requestId"] ?? ""]
    switch kind {
    case "upload_begin":
      guard upload == nil, let hash = message["modelSHA256"] as? String,
        hash.count == 64, hash.allSatisfy({ $0.isHexDigit }),
        let size = message["size"] as? UInt64, size > 0, size <= 32 * 1024 * 1024 * 1024
      else {
        throw RunnerError("Invalid model upload or upload already active.")
      }
      let models = root.appendingPathComponent("Models", isDirectory: true)
      try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
      let free =
        try FileManager.default.attributesOfFileSystem(forPath: models.path)[.systemFreeSize]
        as? NSNumber
      guard (free?.uint64Value ?? 0) > size + 512 * 1024 * 1024 else {
        throw RunnerError("Insufficient space for model transfer.")
      }
      let pending = models.appendingPathComponent(hash + ".pending")
      FileManager.default.createFile(atPath: pending.path, contents: nil)
      upload = ActiveUpload(
        hash: hash, size: size, offset: 0, file: try FileHandle(forWritingTo: pending), url: pending
      )
      hasher = SHA256()
      reply["offset"] = 0
    case "upload_chunk":
      guard var value = upload, let offset = message["offset"] as? UInt64, offset == value.offset,
        let data = payload ?? (message["data"] as? String).flatMap({ Data(base64Encoded: $0) }),
        !data.isEmpty, data.count <= Self.chunkSize, value.offset + UInt64(data.count) <= value.size
      else {
        throw RunnerError("Invalid upload chunk or offset.")
      }
      try value.file.write(contentsOf: data)
      hasher.update(data: data)
      value.offset += UInt64(data.count)
      upload = value
      reply["offset"] = value.offset
    case "upload_finish":
      guard let value = upload, value.offset == value.size else {
        throw RunnerError("Model upload is incomplete.")
      }
      let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
      guard digest == value.hash else {
        abort()
        throw RunnerError("Model transfer checksum mismatch.")
      }
      try value.file.synchronize()
      try value.file.close()
      let destination = root.appendingPathComponent("Models/" + value.hash + ".litertlm")
      if FileManager.default.fileExists(atPath: destination.path) {
        _ = try FileManager.default.replaceItemAt(destination, withItemAt: value.url)
      } else {
        try FileManager.default.moveItem(at: value.url, to: destination)
      }
      upload = nil
      reply["available"] = true
    case "upload_abort": abort()
    case "model_link":
      // The Server and this App see one filesystem: take the file itself (an APFS clone on
      // a shared volume) and trust it only once its hash is the one asked for.
      guard upload == nil, let hash = message["modelSHA256"] as? String,
        hash.count == 64, hash.allSatisfy({ $0.isHexDigit }),
        let path = message["path"] as? String, path.hasPrefix("/")
      else {
        throw RunnerError("Invalid model link or upload already active.")
      }
      let fileManager = FileManager.default
      let models = root.appendingPathComponent("Models", isDirectory: true)
      try fileManager.createDirectory(at: models, withIntermediateDirectories: true)
      let pending = models.appendingPathComponent(hash + ".pending")
      try? fileManager.removeItem(at: pending)
      do {
        try fileManager.copyItem(at: URL(fileURLWithPath: path), to: pending)
        guard try sha256(of: pending, checkCancellation: { try Task.checkCancellation() }) == hash
        else {
          throw RunnerError("Linked model checksum mismatch.")
        }
        let destination = models.appendingPathComponent(hash + ".litertlm")
        if fileManager.fileExists(atPath: destination.path) {
          _ = try fileManager.replaceItemAt(destination, withItemAt: pending)
        } else {
          try fileManager.moveItem(at: pending, to: destination)
        }
      } catch {
        try? fileManager.removeItem(at: pending)
        throw error
      }
      reply["available"] = true
    case "run_files":
      let directory = try runDirectory(message)
      let run = try runID(message)
      if exports[run] == nil {
        let fileManager = FileManager.default
        let terminal = (try? Data(contentsOf: directory.appendingPathComponent("terminal.json")))
          .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        let status = terminal?["generationStatus"] as? String ?? "succeeded"
        guard ["succeeded", "stopped", "failed"].contains(status) else {
          throw RunnerError("Run has no terminal status.")
        }
        var paths = [
          "job.json", "result.json", "terminal.json", "failure.json", "runtime-build.json",
        ].filter {
          fileManager.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
        for folder in ["tensors", "raw"] {
          let nested = directory.appendingPathComponent(folder)
          if fileManager.fileExists(atPath: nested.path) {
            paths += try fileManager.contentsOfDirectory(
              at: nested, includingPropertiesForKeys: nil
            ).map {
              folder + "/" + $0.lastPathComponent
            }
          }
        }
        let files = try paths.sorted().map { path -> ExportFile in
          try Task.checkCancellation()
          let url = try safeFile(path, directory: directory)
          let values = try url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
          return ExportFile(
            path: path, size: values.fileSize ?? 0,
            sha256: try sha256(of: url, checkCancellation: { try Task.checkCancellation() }),
            modified: values.contentModificationDate)
        }
        exports[run] = Export(id: UUID().uuidString.lowercased(), status: status, files: files)
      }
      let export = exports[run]!
      reply["manifestId"] = export.id
      reply["generationStatus"] = export.status
      reply["files"] = export.files.map(\.wire)
      if message["link"] as? Bool == true { reply["root"] = directory.path }
    case "file_chunk":
      let directory = try runDirectory(message)
      guard let path = message["path"] as? String, let offset = message["offset"] as? UInt64 else {
        throw RunnerError("Invalid file request.")
      }
      let export = try matchingExport(message)
      guard let record = export.files.first(where: { $0.path == path }) else {
        throw RunnerError("File is not part of the sealed dump manifest.")
      }
      let url = try safeFile(path, directory: directory)
      try checkUnchanged(record, url: url)
      let file = try FileHandle(forReadingFrom: url)
      defer { try? file.close() }
      let size = try file.seekToEnd()
      guard offset <= size else { throw RunnerError("Invalid file offset.") }
      try file.seek(toOffset: offset)
      let bytes = try file.read(upToCount: Self.chunkSize) ?? Data()
      if message["encoding"] as? String == "binary" {
        outgoing = bytes
      } else {
        reply["data"] = bytes.base64EncodedString()
      }
      reply["offset"] = offset
      reply["manifestId"] = export.id
      try checkUnchanged(record, url: url)
    case "run_received":
      let run = try runID(message)
      let export = try matchingExport(message)
      let directory = try runDirectory(message)
      for record in export.files {
        try checkUnchanged(record, url: safeFile(record.path, directory: directory))
      }
      try FileManager.default.removeItem(at: directory)
      exports.removeValue(forKey: run)
      reply["manifestId"] = export.id
      reply["deleted"] = true
    default: break
    }
    return reply
  }

  private func runDirectory(_ message: [String: Any]) throws -> URL {
    let id = try runID(message)
    let directory = root.appendingPathComponent("Runs/" + id.uuidString)
    guard directory.resolvingSymlinksInPath().path == directory.path,
      FileManager.default.fileExists(atPath: directory.appendingPathComponent("terminal.json").path)
    else {
      throw RunnerError("Run is incomplete.")
    }
    return directory
  }

  private func runID(_ message: [String: Any]) throws -> UUID {
    guard let run = message["run"] as? String, let id = UUID(uuidString: run) else {
      throw RunnerError("A run UUID is required.")
    }
    return id
  }

  private func matchingExport(_ message: [String: Any]) throws -> Export {
    guard let export = exports[try runID(message)], message["manifestId"] as? String == export.id
    else {
      throw RunnerError("Receipt or file request does not match this run's sealed manifest.")
    }
    return export
  }

  private func checkUnchanged(_ record: ExportFile, url: URL) throws {
    let value = try url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
    guard value.fileSize == record.size, value.contentModificationDate == record.modified else {
      throw RunnerError("Sealed dump changed after its manifest was created.")
    }
  }

  private func safeFile(_ path: String, directory: URL) throws -> URL {
    let parts = path.split(separator: "/", omittingEmptySubsequences: false)
    let metadata = [
      "job.json", "result.json", "terminal.json", "failure.json", "runtime-build.json",
    ].contains(path)
    let tensor =
      parts.count == 2 && ["tensors", "raw"].contains(String(parts[0]))
      && (parts[1].hasSuffix(".safetensors")
        || (parts[0] == "raw"
          && ["generated_tokens.jsonl", "runtime_trace.jsonl"].contains(parts[1])))
      && !parts[1].hasPrefix(".")
    let url = directory.appendingPathComponent(path)
    guard metadata || tensor, url.resolvingSymlinksInPath().path == url.path,
      try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true
    else {
      throw RunnerError("Invalid capture path.")
    }
    return url
  }
}
