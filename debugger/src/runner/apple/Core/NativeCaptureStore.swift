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

struct NativeCaptureSeal {
  let rawDirectory: URL?
  let recoveryDirectory: URL?
  let error: String?
}

/// Owns the disk evidence after native callbacks stop. Model/KV ownership stays
/// with NativeRunner; capture export failures do not change generation success.
struct NativeCaptureStore {
  let recoveryDirectory: URL

  func prepare(capture: URL) throws {
    let fileManager = FileManager.default
    if fileManager.fileExists(atPath: capture.path) {
      for file in try fileManager.contentsOfDirectory(
        at: capture, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey])
      {
        let values = try file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        // Registering a fresh native session creates an empty trace.
        // Every other file belongs to evidence from an earlier attempt.
        guard ["runtime_trace.jsonl", "generated_tokens.jsonl"].contains(file.lastPathComponent),
          values.isRegularFile == true, values.fileSize == 0
        else {
          throw CaptureStoreError(
            "Unsealed raw evidence remains at \(capture.path). Recover it before another message; ending the Session still releases the model."
          )
        }
      }
    }
    try fileManager.createDirectory(at: capture, withIntermediateDirectories: true)
  }

  func removeCacheIfSafe(_ cache: URL) throws {
    let fileManager = FileManager.default
    guard fileManager.fileExists(atPath: cache.path) else { return }
    var scanError: Error?
    guard
      let files = fileManager.enumerator(
        at: cache, includingPropertiesForKeys: [.fileSizeKey],
        errorHandler: { _, error in
          scanError = error
          return false
        })
    else { throw CaptureStoreError("Could not inspect native cache; retained at \(cache.path).") }
    for case let file as URL in files {
      let isTensor = file.pathExtension == "safetensors"
      let isLog = ["runtime_trace.jsonl", "generated_tokens.jsonl"].contains(file.lastPathComponent)
      guard isTensor || isLog else { continue }
      let size = try file.resourceValues(forKeys: [.fileSizeKey]).fileSize
      if isTensor || size != 0 {
        throw CaptureStoreError(
          "Unsealed raw evidence retained at \(file.deletingLastPathComponent().path); native cache was not deleted."
        )
      }
    }
    if let scanError {
      throw CaptureStoreError(
        "Could not inspect native cache; retained at \(cache.path): \(scanError.localizedDescription)"
      )
    }
    try fileManager.removeItem(at: cache)
  }

  func seal(capture: URL, to destination: URL) -> NativeCaptureSeal {
    let raw = destination.appendingPathComponent("raw", isDirectory: true)
    do {
      let source = try capture.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
      guard source.isDirectory == true, source.isSymbolicLink != true else {
        throw CaptureStoreError("Source is not a capture directory.")
      }
    } catch {
      return NativeCaptureSeal(
        rawDirectory: nil, recoveryDirectory: nil,
        error:
          "Raw capture directory is unavailable at \(capture.path); destination \(raw.path): \(error.localizedDescription)"
      )
    }
    do {
      try moveDirectory(capture, to: raw)
      removeRecoveryRecord(for: capture)
      return NativeCaptureSeal(rawDirectory: raw, recoveryDirectory: nil, error: nil)
    } catch {
      var detail = "Could not seal raw capture at \(raw.path): \(error.localizedDescription)"
      let pending = recoveryDirectory.appendingPathComponent(
        destination.lastPathComponent + "-" + UUID().uuidString, isDirectory: true)
      var retained = capture
      do {
        try FileManager.default.createDirectory(at: pending, withIntermediateDirectories: true)
        let moved = pending.appendingPathComponent("raw", isDirectory: true)
        try moveDirectory(capture, to: moved)
        retained = moved
        removeRecoveryRecord(for: capture)
        let record = ["destination": raw.path, "rawDirectory": retained.path, "error": detail]
        let data = try JSONSerialization.data(
          withJSONObject: record, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: pending.appendingPathComponent("recovery.json"), options: .atomic)
      } catch {
        detail += "; recovery preparation: \(error.localizedDescription)"
        if retained == capture { pending.withUnsafeFileSystemRepresentation { _ = rmdir($0) } }
      }
      detail += ". Raw evidence retained at \(retained.path)."
      return NativeCaptureSeal(rawDirectory: nil, recoveryDirectory: retained, error: detail)
    }
  }

  private func removeRecoveryRecord(for source: URL) {
    guard
      source.standardizedFileURL.path.hasPrefix(recoveryDirectory.standardizedFileURL.path + "/")
    else { return }
    let parent = source.deletingLastPathComponent()
    let record = parent.appendingPathComponent("recovery.json")
    guard let data = try? Data(contentsOf: record),
      let metadata = try? JSONSerialization.jsonObject(with: data) as? [String: String],
      metadata["rawDirectory"] == source.path
    else { return }
    try? FileManager.default.removeItem(at: record)
    // rmdir cannot remove anything added concurrently to this directory.
    parent.withUnsafeFileSystemRepresentation { _ = rmdir($0) }
  }

  private func moveDirectory(_ source: URL, to destination: URL) throws {
    // Foundation's move may fall back to copying across volumes. Require
    // an atomic same-volume move, with no replacement even for empty targets.
    let result = source.withUnsafeFileSystemRepresentation { from in
      destination.withUnsafeFileSystemRepresentation { to in
        renameatx_np(AT_FDCWD, from, AT_FDCWD, to, UInt32(RENAME_EXCL))
      }
    }
    guard result == 0 else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
  }

  private struct CaptureStoreError: Error, LocalizedError {
    let errorDescription: String?
    init(_ message: String) { errorDescription = message }
  }
}
