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

private struct CaptureTestError: Error, LocalizedError {
  let errorDescription: String?
  init(_ message: String) { errorDescription = message }
}

@main
struct NativeCaptureStoreTests {
  static func main() {
    do { try run() } catch {
      FileHandle.standardError.write(Data("FAIL \(error.localizedDescription)\n".utf8))
      exit(1)
    }
  }

  static func run() throws {
    let fm = FileManager.default
    let root = fm.temporaryDirectory.appendingPathComponent(
      "capture-store-tests-" + UUID().uuidString)
    try fm.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? fm.removeItem(at: root) }
    func check(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
      if try !condition() { throw CaptureTestError(message) }
    }
    func directory(_ path: String) throws -> URL {
      let url = root.appendingPathComponent(path, isDirectory: true)
      try fm.createDirectory(at: url, withIntermediateDirectories: true)
      return url
    }
    let store = NativeCaptureStore(recoveryDirectory: root.appendingPathComponent("recovery"))
    let capture = try directory("cache/litert_lm_debugger/0")
    let complete = Data("complete tensor payload".utf8)
    try complete.write(to: capture.appendingPathComponent("output.safetensors"))
    try Data("{\"event\":\"sample\",\"token_id\":106}\n".utf8).write(
      to: capture.appendingPathComponent("runtime_trace.jsonl"))
    let destination = try directory("runs/conflict")
    let existing = try directory("runs/conflict/raw")
    let partial = Data("complete".utf8)
    try partial.write(to: existing.appendingPathComponent("output.safetensors"))
    let result = store.seal(capture: capture, to: destination)
    try check(
      result.error != nil && result.rawDirectory == nil,
      "An existing partial raw target must not be reported as a complete capture.")
    try check(
      try Data(contentsOf: existing.appendingPathComponent("output.safetensors")) == partial,
      "A conflict must not overwrite the existing destination.")
    guard let recovery = result.recoveryDirectory else {
      throw CaptureTestError("Failure did not expose the source recovery path.")
    }
    try check(
      try Data(contentsOf: recovery.appendingPathComponent("output.safetensors")) == complete,
      "The full original tensor must survive a partial destination conflict.")
    print("PASS partial target preserves the full source and reports recovery")
    try check(
      recovery.path.hasPrefix(store.recoveryDirectory.path + "/")
        && !fm.fileExists(atPath: capture.path),
      "Failed raw must leave the live capture path so a later turn cannot overwrite it.")
    let recoveryRecord = recovery.deletingLastPathComponent().appendingPathComponent(
      "recovery.json")
    let record =
      try JSONSerialization.jsonObject(with: Data(contentsOf: recoveryRecord)) as? [String: String]
    try check(
      record?["destination"] == existing.path && record?["rawDirectory"] == recovery.path,
      "Recovery must record the intended destination and actual retained source.")
    print("PASS failed seal moves into a documented recovery directory")

    let blockedCapture = try directory("blocked-cache/litert_lm_debugger/0")
    let blockedCache = root.appendingPathComponent("blocked-cache")
    try complete.write(to: blockedCapture.appendingPathComponent("output.safetensors"))
    let blockedDestination = root.appendingPathComponent("blocked-run")
    let blockedRecovery = root.appendingPathComponent("blocked-recovery")
    try Data("not a directory".utf8).write(to: blockedDestination)
    try Data("not a directory".utf8).write(to: blockedRecovery)
    let blockedStore = NativeCaptureStore(recoveryDirectory: blockedRecovery)
    let blocked = blockedStore.seal(capture: blockedCapture, to: blockedDestination)
    try check(
      blocked.error?.contains(blockedCapture.path) == true
        && blocked.recoveryDirectory == blockedCapture,
      "When both destinations fail, the exact surviving source path must be exposed.")
    var preparationRejected = false
    do { try blockedStore.prepare(capture: blockedCapture) } catch { preparationRejected = true }
    try check(
      preparationRejected, "A new turn must reject a live directory containing unsealed evidence.")
    try check(
      try Data(contentsOf: blockedCapture.appendingPathComponent("output.safetensors")) == complete,
      "Preparing another turn must not delete or overwrite unsealed evidence.")
    print("PASS failed recovery prevents the next turn from overwriting the source")
    var cleanupRejected = false
    do { try blockedStore.removeCacheIfSafe(blockedCache) } catch { cleanupRejected = true }
    try check(
      cleanupRejected,
      "Closing a model must preserve a cache that still contains the only raw evidence.")
    try check(
      try Data(contentsOf: blockedCapture.appendingPathComponent("output.safetensors")) == complete,
      "Cache cleanup must not delete the retained source after recovery fails.")
    print("PASS close preserves the only source when sealing and recovery both fail")

    // Repair the real filesystem obstruction and retry the same original.
    let originalIdentity =
      try fm.attributesOfItem(
        atPath: blockedCapture.appendingPathComponent("output.safetensors").path)[.systemFileNumber]
      as? NSNumber
    try fm.removeItem(at: blockedDestination)
    try fm.createDirectory(at: blockedDestination, withIntermediateDirectories: true)
    let retried = blockedStore.seal(capture: blockedCapture, to: blockedDestination)
    guard let sealed = retried.rawDirectory else {
      throw CaptureTestError(retried.error ?? "Retry did not seal the original.")
    }
    try check(
      retried.error == nil && !fm.fileExists(atPath: blockedCapture.path),
      "Successful retry must release the live capture path.")
    try check(
      try Data(contentsOf: sealed.appendingPathComponent("output.safetensors")) == complete,
      "Retry must retain the complete original payload.")
    let sealedIdentity =
      try fm.attributesOfItem(atPath: sealed.appendingPathComponent("output.safetensors").path)[
        .systemFileNumber] as? NSNumber
    try check(
      originalIdentity != nil && originalIdentity == sealedIdentity,
      "Same-volume sealing must move the original inode without copying tensor payloads.")
    print("PASS retry after a forced filesystem failure seals the original without copying")

    try blockedStore.prepare(capture: blockedCapture)
    try Data().write(to: blockedCapture.appendingPathComponent("runtime_trace.jsonl"))
    try blockedStore.prepare(capture: blockedCapture)
    let nextTrace = Data("{\"event\":\"sample\",\"sample_id\":2}\n".utf8)
    try nextTrace.write(to: blockedCapture.appendingPathComponent("runtime_trace.jsonl"))
    let nextRun = try directory("runs/next-turn")
    let next = blockedStore.seal(capture: blockedCapture, to: nextRun)
    guard let nextRaw = next.rawDirectory else {
      throw CaptureTestError(next.error ?? "The next turn was not sealed.")
    }
    try check(
      try Data(contentsOf: nextRaw.appendingPathComponent("runtime_trace.jsonl")) == nextTrace,
      "A new trace must be written and sealed at the same recreated native path.")
    try check(
      !fm.fileExists(atPath: sealed.appendingPathComponent("runtime_trace.jsonl").path),
      "Reusing the native path must not modify the previous turn's sealed files.")
    try blockedStore.removeCacheIfSafe(blockedCache)
    try check(
      !fm.fileExists(atPath: blockedCache.path) && fm.fileExists(atPath: sealed.path),
      "Successful cache cleanup must leave sealed evidence intact.")
    try check(
      fm.fileExists(atPath: recovery.appendingPathComponent("output.safetensors").path),
      "Cleaning a different runtime cache must leave pending recovery evidence intact.")
    print("PASS the same native path supports a second turn without changing earlier exports")

    try fm.removeItem(at: existing)
    let recovered = store.seal(capture: recovery, to: destination)
    try check(
      recovered.error == nil && recovered.rawDirectory == existing,
      "A retained recovery directory must be sealable after the conflict is repaired.")
    try check(
      try Data(contentsOf: existing.appendingPathComponent("output.safetensors")) == complete,
      "Recovery retry must replace no data with a former partial copy.")
    try check(
      !fm.fileExists(atPath: recoveryRecord.path),
      "A completed recovery must not leave a stale pending record.")
    print("PASS pending evidence can be retried and its recovery record is retired")

    let interrupted = try directory("interrupted-cache/litert_lm_debugger/0")
    let interruptedCache = root.appendingPathComponent("interrupted-cache")
    let partialTrace = Data("{\"event\":\"graph_pre\"}\n".utf8)
    try partialTrace.write(to: interrupted.appendingPathComponent("runtime_trace.jsonl"))
    var tracePreserved = false
    do { try store.removeCacheIfSafe(interruptedCache) } catch { tracePreserved = true }
    try check(
      tracePreserved && fm.fileExists(atPath: interrupted.path),
      "A stop/error trace without any tensors must survive close.")
    let interruptedRun = try directory("runs/interrupted")
    let stopped = store.seal(capture: interrupted, to: interruptedRun)
    guard let stoppedRaw = stopped.rawDirectory else {
      throw CaptureTestError(stopped.error ?? "Stopped trace was not sealed.")
    }
    try check(
      try Data(contentsOf: stoppedRaw.appendingPathComponent("runtime_trace.jsonl"))
        == partialTrace, "Sealing must preserve an incomplete native trace byte-for-byte.")
    try store.removeCacheIfSafe(interruptedCache)
    print("PASS stopped/failed trace-only captures survive until explicitly sealed")
    let missing = root.appendingPathComponent("missing-source")
    let missingRun = try directory("runs/missing-source")
    let absent = store.seal(capture: missing, to: missingRun)
    try check(
      absent.error != nil && absent.rawDirectory == nil && absent.recoveryDirectory == nil,
      "A missing source must not be advertised as retained recovery evidence.")
    print("PASS a missing source reports no recoverable evidence")

    let emptyCapture = try directory("empty-target-cache/litert_lm_debugger/0")
    try complete.write(to: emptyCapture.appendingPathComponent("output.safetensors"))
    let emptyRun = try directory("runs/empty-target")
    let emptyRaw = try directory("runs/empty-target/raw")
    let emptyConflict = store.seal(capture: emptyCapture, to: emptyRun)
    try check(
      emptyConflict.error != nil && emptyConflict.rawDirectory == nil,
      "An existing empty raw directory is still a conflict, not permission to replace it.")
    try check(
      try fm.contentsOfDirectory(atPath: emptyRaw.path).isEmpty,
      "An empty destination conflict must remain untouched.")
    print("PASS an empty destination is never overwritten or assumed complete")
    print("10 native capture store filesystem checks passed; no inference ran.")
  }
}
