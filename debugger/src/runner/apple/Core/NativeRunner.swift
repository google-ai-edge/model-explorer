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

import CLiteRTLM
import Darwin
import Foundation
import os

enum NativeProgress: String, CustomStringConvertible {
  case verifying, loading, generating, validating
  var description: String {
    switch self {
    case .verifying: return "Verifying model"
    case .loading: return "Loading model"
    case .generating: return "Running"
    case .validating: return "Validating capture"
    }
  }
}

struct RunnerResult: Encodable {
  let formatVersion = 1
  let runtimeInstance: UUID
  let modelLoadCount = 1
  let tokenCountBefore: Int
  let tokenCount: Int
  let turnSequence: Int
  let jobID: UUID
  let modelSHA256: String
  let platform: String
  let operatingSystem: String
  let hardwareIdentifier: String
  var backend = "CPU"
  let debuggerEnabled = true
  let sampler = ["topK": 1.0, "topP": 1.0, "temperature": 0.0, "seed": 0.0]
  let thinkingEnabled = false
  let speculativeDecodingEnabled = false
  let input: String
  let output: String
  let maxOutputTokens: Int
  let contextLength: Int
  let elapsedSecondsWithCapture: Double
  let tensors: [CapturedTensor]
  let uncapturedPoints: [String]
  var environment: RunnerCaptureEnvironment?
  var generationStatus = "succeeded"
  var dumpStatus = "ready"
  var dumpError: String?
  var prefillTokenCount: Int?
  var backendEvidence: BackendEvidence?
  var runtimeTracePath: String?
  /// The rendered template text the runtime admitted for this message and its
  /// engine-tokenized IDs (BOS tokens the pinned runtime adds are listed first).
  var renderedInput: String?
  var inputTokens: [RunnerInputToken]?
  /// The engine's configured stop tokens; a sampled stop token is never released as text.
  var stopTokens: [RunnerStopToken]?
}

struct RunnerInputToken: Encodable {
  let id: Int32
  let text: String?
}

struct RunnerStopToken: Encodable {
  let ids: [Int32]?
  let text: String?
}

/// One message as the runtime will admit it: rendered text, engine token IDs and the
/// BOS tokens the pinned runtime adds by itself (they are not part of `ids`).
private struct RenderedInput {
  let text: String
  let ids: [Int32]
  let bosID: Int32?
  let bosCount: Int
  var tokenCount: Int { ids.count + bosCount }
}

private final class StreamState {
  let done = DispatchSemaphore(value: 0)
  let lock = NSLock()
  var output = ""
  var failure: String?
  var finished = false
  let delta: (String) -> Void
  init(delta: @escaping (String) -> Void) { self.delta = delta }

  func receive(_ chunk: OpaquePointer) {
    lock.withLock {
      guard !finished else { return }
      if let error = litert_lm_stream_chunk_get_error(chunk) { failure = String(cString: error) }
      if let text = litert_lm_stream_chunk_get_text(chunk) {
        // Conversation C API chunks are JSON messages, unlike the lower
        // level Session API's plain text chunks.
        do {
          let value = try conversationText(from: String(cString: text))
          output += value
          if !value.isEmpty { delta(value) }
        } catch { failure = error.localizedDescription }
      }
      if litert_lm_stream_chunk_is_final(chunk) {
        finished = true
        done.signal()
      }
    }
  }
}

func conversationText(from json: String) throws -> String {
  guard let message = try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]
  else {
    throw RunnerError("Invalid JSON conversation chunk.")
  }
  if let content = message["content"] as? String { return content }
  return (message["content"] as? [[String: Any]] ?? [])
    .compactMap { $0["type"] as? String == "text" ? $0["text"] as? String : nil }
    .joined()
}

/// One instance per session run. execute()/close() are serialized; cancel() may
/// be called from the UI. The lock prevents cancellation racing native teardown.
final class NativeRunner: @unchecked Sendable {
  private static let logger = Logger(
    subsystem: "com.google.ai.edge.debugger.runner", category: "NativeRunner")
  private let lock = NSLock()
  private var cancelled = false
  private var closed = false
  private var conversation: OpaquePointer?
  private var engine: OpaquePointer?
  private var cache: URL?
  private var configuration: Data?
  private var backendEvidence: BackendEvidence?
  private var captureDirectory: URL?
  private var captureStore: NativeCaptureStore?
  private var captureExports: [URL: URL] = [:]
  private var poisoned = false
  private let operationLock = NSLock()
  private let runtimeInstance = UUID()
  private var turnSequence = 0
  private let captureBenchmarks: Bool

  init(captureBenchmarks: Bool = false) { self.captureBenchmarks = captureBenchmarks }

  deinit { release() }

  // Call on a background queue. Never delete a conversation with callbacks active.
  func close() {
    operationLock.withLock {
      closed = true
      release()
    }
  }

  func closeChat() {
    operationLock.withLock {
      releaseConversation()
    }
  }

  func preflight(prompt: String, contextLimit: Int) throws -> RunnerPreflight {
    try operationLock.withLock {
      guard !closed, !poisoned, let conversation, let engine else {
        throw RunnerError("Initialize a new Chat before sending a message.")
      }
      let context = Int(litert_lm_conversation_get_token_count(conversation))
      guard context >= 0 else {
        throw RunnerError("Runtime cannot inspect Chat input capacity without changing its state.")
      }
      let input = try renderInput(
        prompt: prompt, conversation: conversation, engine: engine, contextTokenCount: context)
      return RunnerPreflight(
        contextTokenCount: context, inputTokenCount: input.tokenCount, contextLimit: contextLimit)
    }
  }

  /// Render one user message on a cloned Conversation and tokenize it with the engine's
  /// own tokenizer; the live Conversation is never touched.
  private func renderInput(
    prompt: String, conversation: OpaquePointer, engine: OpaquePointer,
    contextTokenCount context: Int
  ) throws -> RenderedInput {
    guard let preview = litert_lm_conversation_clone(conversation) else {
      throw RunnerError("Runtime cannot inspect Chat input capacity without changing its state.")
    }
    defer { litert_lm_conversation_delete(preview) }
    let message = try JSONSerialization.data(withJSONObject: [
      "role": "user", "content": [["type": "text", "text": prompt]],
    ])
    guard
      let rendered = litert_lm_conversation_render_message_to_string(
        preview, String(decoding: message, as: UTF8.self))
    else {
      throw RunnerError("Runtime could not render the input template.")
    }
    let full = String(cString: rendered)
    var text = full
    var bosCount = 0
    var bosID: Int32?
    if let start = litert_lm_engine_get_start_token(engine) {
      defer { litert_lm_token_union_delete(start) }
      var ids: UnsafePointer<Int32>?
      var count = 0
      guard litert_lm_token_union_get_ids(start, &ids, &count) == 0, let ids, count == 1,
        let decoded = litert_lm_engine_detokenize(engine, ids, count)
      else {
        throw RunnerError("Runtime start-token metadata is unavailable.")
      }
      defer { litert_lm_detokenize_result_delete(decoded) }
      bosID = ids[0]
      let bos = litert_lm_detokenize_result_get_string(decoded).map { String(cString: $0) } ?? ""
      // Pinned Session::ApplyPromptTemplates adds a separate BOS on the
      // first prefill. StringToProcessedInputText preserves a leading BOS
      // as its configured ID instead of passing that spelling to tokenize.
      if context == 0 && !bos.isEmpty { bosCount += 1 }
      if !bos.isEmpty && text.hasPrefix(bos) {
        text.removeFirst(bos.count)
        bosCount += 1
      }
    }
    guard let tokens = litert_lm_engine_tokenize(engine, text) else {
      throw RunnerError("Runtime could not tokenize the rendered input.")
    }
    defer { litert_lm_tokenize_result_delete(tokens) }
    let count = Int(litert_lm_tokenize_result_get_num_tokens(tokens))
    var ids: [Int32] = []
    if count > 0, let buffer = litert_lm_tokenize_result_get_tokens(tokens) {
      ids = Array(UnsafeBufferPointer(start: buffer, count: count))
    }
    return RenderedInput(text: full, ids: ids, bosID: bosID, bosCount: bosCount)
  }

  /// The engine's configured stop tokens as IDs (string forms are tokenized) and text.
  private func stopTokenRecords(engine: OpaquePointer) -> [RunnerStopToken] {
    guard let unions = litert_lm_engine_get_stop_tokens(engine) else { return [] }
    defer { litert_lm_token_unions_delete(unions) }
    var records: [RunnerStopToken] = []
    for index in 0..<Int(litert_lm_token_unions_get_num_tokens(unions)) {
      guard let union = litert_lm_token_unions_get_token_at(unions, index) else { continue }
      defer { litert_lm_token_union_delete(union) }
      if litert_lm_token_union_get_type(union) == kLiteRtLmTokenUnionTypeIds {
        var ids: UnsafePointer<Int32>?
        var count = 0
        guard litert_lm_token_union_get_ids(union, &ids, &count) == 0, let ids, count > 0 else {
          continue
        }
        var text: String?
        if let decoded = litert_lm_engine_detokenize(engine, ids, count) {
          defer { litert_lm_detokenize_result_delete(decoded) }
          text = litert_lm_detokenize_result_get_string(decoded).map { String(cString: $0) }
        }
        records.append(
          RunnerStopToken(
            ids: Array(UnsafeBufferPointer(start: ids, count: count)),
            text: text.flatMap { $0.isEmpty ? nil : $0 }))
      } else if let raw = litert_lm_token_union_get_string(union) {
        let text = String(cString: raw)
        var values: [Int32]?
        if let tokens = litert_lm_engine_tokenize(engine, text) {
          defer { litert_lm_tokenize_result_delete(tokens) }
          let count = Int(litert_lm_tokenize_result_get_num_tokens(tokens))
          if count > 0, let buffer = litert_lm_tokenize_result_get_tokens(tokens) {
            values = Array(UnsafeBufferPointer(start: buffer, count: count))
          }
        }
        records.append(RunnerStopToken(ids: values, text: text))
      }
    }
    return records
  }

  /// Each admitted ID with the engine's own text for it; control tokens decode to nil.
  private func inputTokenRecords(_ input: RenderedInput, engine: OpaquePointer)
    -> [RunnerInputToken]
  {
    var records: [RunnerInputToken] = []
    if let bosID = input.bosID {
      records.append(
        contentsOf: Array(repeating: RunnerInputToken(id: bosID, text: nil), count: input.bosCount))
    }
    for id in input.ids {
      var value = id
      var text: String?
      if let decoded = litert_lm_engine_detokenize(engine, &value, 1) {
        defer { litert_lm_detokenize_result_delete(decoded) }
        text = litert_lm_detokenize_result_get_string(decoded).map { String(cString: $0) }
      }
      records.append(RunnerInputToken(id: id, text: text.flatMap { $0.isEmpty ? nil : $0 }))
    }
    return records
  }

  private func releaseConversation() {
    // Teardown never owns evidence deletion. Retry each export against its
    // original job, then retain any unresolved source for recovery.
    for source in Array(captureExports.keys) {
      if let error = sealRawFiles(capture: source) {
        Self.logger.error("\(error, privacy: .public)")
      }
    }
    lock.withLock {
      if let conversation { litert_lm_conversation_delete(conversation) }
      conversation = nil
      cancelled = false
    }
    captureDirectory = nil
    turnSequence = 0
  }

  private func release() {
    releaseConversation()
    lock.withLock {
      if let engine { litert_lm_engine_delete(engine) }
      engine = nil
    }
    if let cache, let captureStore {
      do { try captureStore.removeCacheIfSafe(cache) } catch {
        Self.logger.error("\(error.localizedDescription, privacy: .public)")
      }
    }
    cache = nil
    configuration = nil
    backendEvidence = nil
  }

  static var debuggerEnabled: Bool { litert_lm_experimental_is_debugger_enabled() == 1 }

  func cancel() {
    lock.withLock {
      cancelled = true
      if let conversation { litert_lm_conversation_cancel_process(conversation) }
    }
  }

  func checkCancellation() throws {
    let isCancelled = lock.withLock { cancelled }
    if isCancelled { throw RunnerError("Run cancelled. Partial data is not a completed capture.") }
  }

  func execute(
    job: CaptureJob, model: URL, destination: URL,
    progress: @escaping (NativeProgress) -> Void, delta: @escaping (String) -> Void,
    initializeOnly: Bool = false, persistent: Bool = false
  ) throws -> RunnerResult {
    try operationLock.withLock {
      defer { if !persistent { release() } }
      guard !closed else {
        throw RunnerError("Session is closed. Initialize before sending a message.")
      }
      guard !poisoned else {
        throw RunnerError(
          "Model execution failed. End this Session; its Runner cannot start another Chat.")
      }
      try job.validate()
      if persistent && !initializeOnly && conversation == nil {
        throw RunnerError("Session is closed. Initialize before sending a message.")
      }
      if persistent && !(job.messages ?? []).isEmpty {
        throw RunnerError("A new Chat starts empty; ended Chat history cannot be resumed.")
      }
      let encoder = JSONEncoder()
      encoder.outputFormatting = .sortedKeys
      let manifest = try encoder.encode(job.manifest)
      let key =
        Data((job.modelSHA256 + ":" + String(job.contextLength) + ":" + job.backend).utf8)
        + manifest
      if let configuration, configuration != key {
        throw RunnerError("Session configuration changed. Close and initialize again.")
      }
      configuration = key
      if initializeOnly { releaseConversation() }
      guard Self.debuggerEnabled else {
        throw RunnerError(
          "This runtime has no Tensor capture support. Build with LITERT_LM_DEBUGGER_ENABLED=1.")
      }
      let fileManager = FileManager.default
      guard !fileManager.fileExists(atPath: destination.path) else {
        throw RunnerError("Run directory already exists.")
      }
      try fileManager.createDirectory(at: destination, withIntermediateDirectories: true)
      try writeJSON(job, to: destination.appendingPathComponent("job.json"))
      // A cache retained after export failure must never be reopened by a new
      // Engine, whose session registration would truncate its old trace.
      let cache =
        self.cache
        ?? fileManager.temporaryDirectory.appendingPathComponent(
          "session-" + runtimeInstance.uuidString + "-" + UUID().uuidString, isDirectory: true)
      if captureStore == nil {
        captureStore = NativeCaptureStore(
          recoveryDirectory: destination.deletingLastPathComponent()
            .appendingPathComponent("NativeCaptureRecovery", isDirectory: true)
            .appendingPathComponent(runtimeInstance.uuidString, isDirectory: true))
      }
      self.cache = cache
      try fileManager.createDirectory(at: cache, withIntermediateDirectories: true)
      do {
        let result = try run(
          job: job, model: model, destination: destination, cache: cache,
          progress: progress, delta: delta, initializeOnly: initializeOnly)
        var completed = result
        do {
          try writeJSON(completed, to: destination.appendingPathComponent("result.json"))
          try writeJSON(
            [
              "generationStatus": "succeeded", "jobID": job.id.uuidString.lowercased(),
              "output": completed.output,
              "dumpStatus": completed.dumpStatus,
            ], to: destination.appendingPathComponent("terminal.json"))
        } catch {
          completed.dumpStatus = "unavailable"
          completed.dumpError = error.localizedDescription
        }
        return completed
      } catch {
        let stopped = lock.withLock { cancelled }
        let terminal = stopped ? "stopped" : "failed"
        // Native callbacks have ended before run() throws. Preserve their raw
        // files before closing the Conversation or releasing any cache.
        let ownsCapture = captureDirectory.flatMap { captureExports[$0] } == destination
        let dumpError =
          ownsCapture
          ? sealRawFiles(capture: captureDirectory)
          : "No native raw capture was produced for this job."
        try? writeJSON(
          ["status": terminal, "error": error.localizedDescription],
          to: destination.appendingPathComponent("failure.json"))
        var terminalRecord = [
          "generationStatus": terminal, "jobID": job.id.uuidString.lowercased(),
          "error": error.localizedDescription,
          "dumpStatus": dumpError == nil ? "ready" : "unavailable",
        ]
        terminalRecord["dumpError"] = dumpError
        try? writeJSON(terminalRecord, to: destination.appendingPathComponent("terminal.json"))
        if !initializeOnly && !stopped { poisoned = true }
        releaseConversation()
        throw error
      }
    }
  }

  private func sealRawFiles(capture: URL?) -> String? {
    guard let capture, let destination = captureExports[capture], let captureStore else {
      return nil
    }
    let result = captureStore.seal(capture: capture, to: destination)
    if result.recoveryDirectory != capture { captureExports.removeValue(forKey: capture) }
    return result.error
  }

  private func run(
    job: CaptureJob, model: URL, destination: URL, cache: URL,
    progress: @escaping (NativeProgress) -> Void, delta: @escaping (String) -> Void,
    initializeOnly: Bool = false
  ) throws -> RunnerResult {
    let fileManager = FileManager.default
    let started = Date()
    let before = RunnerEnvironment.sample()
    if engine == nil {
      progress(.verifying)
      guard try sha256(of: model, checkCancellation: checkCancellation) == job.modelSHA256 else {
        throw RunnerError(
          "Model hash differs from the prepared job. Import its matching .litertlm file.")
      }
      let free =
        try fileManager.attributesOfFileSystem(forPath: destination.path)[.systemFreeSize]
        as? NSNumber
      guard let free, free.int64Value >= 2 * 1024 * 1024 * 1024 else {
        throw RunnerError("At least 2 GB of free space is required for the native capture cache.")
      }
      progress(.loading)
      backendEvidence = try NativeBackendSupport.prepare(backend: job.backend)
      litert_lm_set_min_log_level(
        job.backend == "GPU" ? kLiteRtLmLogSeverityInfo : kLiteRtLmLogSeverityWarning)
      guard
        let settings = litert_lm_engine_settings_create(
          model.path, job.backend.lowercased(), nil, nil)
      else {
        throw RunnerError("Could not create engine settings.")
      }
      defer { litert_lm_engine_settings_delete(settings) }
      litert_lm_engine_settings_set_cache_dir(settings, cache.path)
      litert_lm_engine_settings_set_max_num_tokens(settings, Int32(job.contextLength))
      litert_lm_engine_settings_set_enable_speculative_decoding(settings, false)
      if captureBenchmarks { litert_lm_engine_settings_enable_benchmark(settings) }
      guard let engine = litert_lm_engine_create(settings) else {
        throw RunnerError("Model initialization failed. Check device memory and native logs.")
      }
      self.engine = engine
      backendEvidence?.effectiveBackend = job.backend
      backendEvidence?.engineInitialized = true
      try checkCancellation()
    }
    if conversation == nil {
      guard let engine else { throw RunnerError("Model is unavailable.") }
      guard let sessionConfig = litert_lm_session_config_create() else {
        throw RunnerError("Session config failed.")
      }
      defer { litert_lm_session_config_delete(sessionConfig) }
      litert_lm_session_config_set_max_output_tokens(sessionConfig, Int32(job.maxOutputTokens))
      litert_lm_session_config_set_enable_speculative_decoding(sessionConfig, false)
      guard let sampler = litert_lm_sampler_params_create(kLiteRtLmSamplerTypeTopP) else {
        throw RunnerError("Sampler config failed.")
      }
      defer { litert_lm_sampler_params_delete(sampler) }
      litert_lm_sampler_params_set_top_k(sampler, 1)
      litert_lm_sampler_params_set_top_p(sampler, 1)
      litert_lm_sampler_params_set_temperature(sampler, 0)
      litert_lm_sampler_params_set_seed(sampler, 0)
      litert_lm_session_config_set_sampler_params(sessionConfig, sampler)
      guard let config = litert_lm_conversation_config_create() else {
        throw RunnerError("Conversation config failed.")
      }
      defer { litert_lm_conversation_config_delete(config) }
      litert_lm_conversation_config_set_session_config(config, sessionConfig)
      guard let thinking = litert_lm_thinking_config_create() else {
        throw RunnerError("Thinking config failed.")
      }
      defer { litert_lm_thinking_config_delete(thinking) }
      litert_lm_thinking_config_set_enable_thinking(thinking, false)
      litert_lm_conversation_config_set_thinking_config(config, thinking)
      if let messages = job.messages, !messages.isEmpty {
        let history = try JSONEncoder().encode(messages)
        litert_lm_conversation_config_set_messages(config, String(decoding: history, as: UTF8.self))
      }
      guard let handle = litert_lm_conversation_create(engine, config) else {
        throw RunnerError("Conversation initialization failed.")
      }
      lock.withLock {
        conversation = handle
      }
    }
    guard let handle = conversation else { throw RunnerError("Session is closed.") }
    try checkCancellation()
    guard let debug = litert_lm_experimental_conversation_get_session_debug_info(handle) else {
      throw RunnerError("Runtime reports no debug session for this model/backend.")
    }
    defer { litert_lm_experimental_session_debug_info_delete(debug) }
    guard let nativePath = litert_lm_experimental_session_debug_info_get_capture_dir(debug) else {
      throw RunnerError("Runtime returned no capture directory.")
    }
    let relative = String(cString: nativePath)
    let capture = cache.appendingPathComponent(relative).standardizedFileURL
      .resolvingSymlinksInPath()
    guard !relative.hasPrefix("/"),
      capture.path.hasPrefix(cache.resolvingSymlinksInPath().path + "/")
    else {
      throw RunnerError("Capture directory is outside this run.")
    }
    captureDirectory = capture
    let tokenCountBefore = Int(litert_lm_conversation_get_token_count(handle))
    if !initializeOnly, let expected = job.expect {
      let sameRuntime =
        expected.runtimeInstance.map { UUID(uuidString: $0) == runtimeInstance } ?? true
      guard sameRuntime, expected.turnSequence == turnSequence,
        expected.tokenCount == tokenCountBefore
      else {
        throw RunnerError(
          "This Conversation is not where the Server left it (turn \(turnSequence), \(tokenCountBefore) tokens; "
            + "expected turn \(expected.turnSequence), \(expected.tokenCount) tokens). Nothing was generated."
        )
      }
    }
    var output = ""
    var admittedInput: RenderedInput?
    if !initializeOnly {
      if let engine = self.engine {
        admittedInput = try renderInput(
          prompt: job.prompt, conversation: handle, engine: engine,
          contextTokenCount: tokenCountBefore)
      }
      if let error = sealRawFiles(capture: capture), captureExports[capture] != nil {
        throw RunnerError(error)
      }
      guard let captureStore else { throw RunnerError("Native capture storage is unavailable.") }
      try captureStore.prepare(capture: capture)
      captureExports[capture] = destination
      guard let options = litert_lm_conversation_optional_args_create() else {
        throw RunnerError("Message options failed.")
      }
      defer { litert_lm_conversation_optional_args_delete(options) }
      litert_lm_conversation_optional_args_set_max_output_tokens(
        options, Int32(job.maxOutputTokens))
      let message = try JSONSerialization.data(withJSONObject: [
        "role": "user", "content": [["type": "text", "text": job.prompt]],
      ])
      let state = StreamState(delta: delta)
      progress(.generating)
      let status = litert_lm_conversation_send_message_stream(
        handle, String(decoding: message, as: UTF8.self), nil, options,
        { pointer, chunk in
          guard let pointer, let chunk else { return }
          Unmanaged<StreamState>.fromOpaque(pointer).takeUnretainedValue().receive(chunk)
        }, Unmanaged.passUnretained(state).toOpaque())
      guard status == 0 else {
        throw RunnerError("Could not start native inference (\(status)).")
      }
      // Keep callback state and the native conversation alive until the final
      // callback. A timeout requests cancellation; it never frees live pointers.
      while state.done.wait(timeout: .now() + 1) == .timedOut {
        if Date().timeIntervalSince(started) > 600 { cancel() }
        if let free = try? fileManager.attributesOfFileSystem(forPath: destination.path)[
          .systemFreeSize]
          as? NSNumber,
          free.int64Value < 512 * 1024 * 1024
        {
          cancel()
        }
      }
      let failure: String? = state.lock.withLock {
        output = state.output
        return state.failure
      }
      try checkCancellation()
      if let failure { throw RunnerError(failure) }
    }
    progress(.validating)
    var records: [CapturedTensor] = []
    var dumpFailure: String?
    do {
      let tensorsDir = destination.appendingPathComponent("tensors", isDirectory: true)
      try fileManager.createDirectory(at: tensorsDir, withIntermediateDirectories: true)
      var identities = Set<String>()
      for file
        in (initializeOnly
        ? []
        : try fileManager.contentsOfDirectory(
          at: capture, includingPropertiesForKeys: [.isRegularFileKey]))
        .sorted(by: { $0.path < $1.path })
      {
        guard file.pathExtension == "safetensors",
          try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true
        else { continue }
        if let record = try inspectCapture(file, taps: job.manifest.taps) {
          let key = "\(record.signature):\(record.key):\(record.step)"
          guard identities.insert(key).inserted else {
            throw RunnerError("Duplicate captured tensor.")
          }
          try fileManager.copyItem(at: file, to: destination.appendingPathComponent(record.path))
          records.append(record)
        }
      }
      guard initializeOnly || !records.isEmpty else {
        throw RunnerError("Inference completed, but none of the selected points was captured.")
      }
      let groups = Dictionary(grouping: records) { "\($0.signature):\($0.step)" }
      for group in groups.values {
        let signature = group[0].signature
        let required = Set(
          job.manifest.taps.compactMap {
            $0.signature == signature ? "post_" + $0.outputName : nil
          })
        guard Set(group.map(\.key)) == required else {
          throw RunnerError("Incomplete selected outputs at \(signature), step \(group[0].step).")
        }
      }
    } catch {
      // Generation already succeeded. A capture failure must not erase its
      // complete output or turn it into a model execution failure.
      dumpFailure = error.localizedDescription
    }
    // Selection reads the live files first. Move the complete directory
    // once afterwards, including on validation failure, without copying raw
    // tensor payloads. Server receipt owns deletion of the sealed export.
    if !initializeOnly, let sealFailure = sealRawFiles(capture: capture) {
      dumpFailure = [dumpFailure, sealFailure].compactMap { $0 }.joined(separator: "; ")
    }
    let observed = Set(records.map { $0.signature + ":" + String($0.key.dropFirst(5)) })
    let missing = job.manifest.taps.map { $0.signature + ":" + $0.outputName }.filter {
      !observed.contains($0)
    }
    #if targetEnvironment(simulator)
      let platform = "iOS Simulator"
    #elseif os(iOS)
      let platform = "iOS"
    #else
      let platform = "macOS"
    #endif
    var info = utsname()
    uname(&info)
    let machineSize = MemoryLayout.size(ofValue: info.machine)
    let hardware = withUnsafePointer(to: &info.machine) {
      $0.withMemoryRebound(to: CChar.self, capacity: machineSize) { String(cString: $0) }
    }
    if !initializeOnly { turnSequence += 1 }
    var result = RunnerResult(
      runtimeInstance: runtimeInstance, tokenCountBefore: tokenCountBefore,
      tokenCount: Int(litert_lm_conversation_get_token_count(handle)), turnSequence: turnSequence,
      jobID: job.id, modelSHA256: job.modelSHA256, platform: platform,
      operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
      hardwareIdentifier: hardware,
      input: job.prompt, output: output, maxOutputTokens: job.maxOutputTokens,
      contextLength: job.contextLength,
      elapsedSecondsWithCapture: Date().timeIntervalSince(started),
      tensors: records, uncapturedPoints: missing,
      environment: RunnerCaptureEnvironment(
        device: RunnerEnvironment.device,
        capabilities: .init(tensorCapture: Self.debuggerEnabled),
        session: .init(
          modelSHA256: job.modelSHA256, backend: job.backend, contextLength: job.contextLength,
          capturePoints: job.manifest.taps.count),
        maxOutputTokens: job.maxOutputTokens, before: before, after: RunnerEnvironment.sample()))
    if let dumpFailure {
      result.dumpStatus = "unavailable"
      result.dumpError = dumpFailure
    }
    result.backend = job.backend
    result.backendEvidence = backendEvidence
    if !initializeOnly,
      fileManager.fileExists(
        atPath: destination.appendingPathComponent("raw/runtime_trace.jsonl").path)
    {
      result.runtimeTracePath = "raw/runtime_trace.jsonl"
    }
    if let admittedInput, let engine = self.engine {
      result.renderedInput = admittedInput.text
      result.inputTokens = inputTokenRecords(admittedInput, engine: engine)
      result.stopTokens = stopTokenRecords(engine: engine)
    }
    if captureBenchmarks && !initializeOnly,
      let benchmark = litert_lm_conversation_get_benchmark_info(handle)
    {
      defer { litert_lm_benchmark_info_delete(benchmark) }
      let count = litert_lm_benchmark_info_get_num_prefill_turns(benchmark)
      if count > 0 {
        result.prefillTokenCount = Int(
          litert_lm_benchmark_info_get_prefill_token_count_at(benchmark, count - 1))
      }
    }
    return result
  }
}
