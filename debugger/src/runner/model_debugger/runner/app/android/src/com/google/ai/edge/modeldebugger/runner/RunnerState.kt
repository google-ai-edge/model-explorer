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

package com.google.ai.edge.modeldebugger.runner

import org.json.JSONObject

/** What the Runner reports: `model_debugger_runner_get_state`, decoded from snake case. */
data class RunnerState(
  /** The platform the Runner runs on, such as "Android". */
  val platform: String,
  /** The operating system's version, such as "15". */
  val osVersion: String,
  /** The chip, else the device's manufacturer and model. */
  val device: String,
  /** The host the Runner listens on. */
  val address: String,
  /** The port the Runner listens on. */
  val port: Int,
  /** The server's address as gRPC names it; empty until the server calls. */
  val client: String,
  /** The loaded model's SHA-256; empty until the server loads one. */
  val modelSha256: String,
  /** The turns the chat has completed. */
  val turns: Int,
  /** Whether a turn was cancelled or failed, so that nothing more is generated. */
  val chatOver: Boolean,
) {
  companion object {
    /** Decodes [json], as `model_debugger_runner_get_state` writes it. */
    fun parse(json: String): RunnerState {
      val jsonObj = JSONObject(json)
      return RunnerState(
        platform = jsonObj.getString("platform"),
        osVersion = jsonObj.getString("os_version"),
        device = jsonObj.getString("device"),
        address = jsonObj.getString("address"),
        port = jsonObj.getInt("port"),
        client = jsonObj.getString("client"),
        modelSha256 = jsonObj.getString("model_sha256"),
        turns = jsonObj.getInt("turns"),
        chatOver = jsonObj.getBoolean("chat_over"),
      )
    }
  }
}
