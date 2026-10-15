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

import android.system.Os
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import java.io.File
import kotlin.time.Duration.Companion.milliseconds
import kotlinx.coroutines.Job
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * The Runner behind the page, one for the process.
 *
 * It starts on `listen`, HOST:PORT from the App's launch, or 127.0.0.1:8769, the port adb forwards,
 * and is asked for its state twice a second until stopped. It keeps everything in the App's cache
 * directory, which it finds through TMPDIR.
 */
object RunnerModel {
  /** What the Runner last reported; null until it reports. */
  var state by mutableStateOf<RunnerState?>(null)
    private set

  /** Why the Runner did not start; null if it did. */
  var failure by mutableStateOf<String?>(null)
    private set

  private var runner = 0L

  private val scope = MainScope()

  /** What asks the running Runner for its state; null while none runs. */
  private var polling: Job? = null

  private val loopbackHosts = setOf("127.0.0.1", "localhost", "[::1]", "::1")

  init {
    // librunner.so, the C facade; the runtime and its GPU plugins are libraries beside it.
    System.loadLibrary("runner")
  }

  /**
   * Starts the Runner on [listen], or 127.0.0.1:8769 without one, once until [stop]: an Activity
   * recreated when the screen turns calls it again, and a Runner that failed to start is not tried
   * again until then.
   */
  fun start(cacheDir: File, listen: String?) {
    if (runner != 0L || failure != null) return
    val address = listen ?: "127.0.0.1:8769"
    val host = address.substringBeforeLast(":", "").lowercase()
    if (host !in loopbackHosts) {
      failure = "The Runner on Android only listens on a loopback address."
      return
    }
    Os.setenv("TMPDIR", cacheDir.path, true)
    try {
      runner = runnerStart(address)
    } catch (e: IllegalStateException) {
      failure = e.message
      return
    }
    polling = scope.launch {
      while (isActive) {
        state = RunnerState.parse(runnerGetState(runner))
        delay(500.milliseconds)
      }
    }
  }

  /**
   * Stops the Runner: in-flight calls finish, and its run directory goes. What it reported, or why
   * it failed, goes too, so that a later [start] in the same process starts afresh.
   */
  fun stop() {
    polling?.cancel()
    polling = null
    if (runner != 0L) runnerDelete(runner)
    runner = 0L
    state = null
    failure = null
  }

  // The C facade (runner_c.h), through runner_jni.cc. runnerStart throws
  // IllegalStateException with why the Runner cannot start.
  @JvmStatic private external fun runnerStart(listen: String): Long

  @JvmStatic private external fun runnerGetState(runner: Long): String

  @JvmStatic private external fun runnerDelete(runner: Long)
}
