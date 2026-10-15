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

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent

/**
 * The Runner's App on Android: one page over the process's Runner, which stops when the App is
 * closed. A process the system kills leaves its run directory behind, which the next Runner
 * removes.
 */
class RunnerActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    // The screen stays on while the page shows, as on iOS: the system freezes an App that is not
    // in front, and a frozen Runner answers nothing while its page still reads as serving.
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    // The loopback address to listen on, if the launch gives one:
    // `am start ... --es listen 127.0.0.1:PORT`.
    RunnerModel.start(cacheDir, intent.getStringExtra("listen"))
    setContent { RunnerView(RunnerModel.state, RunnerModel.failure) }
  }

  override fun onDestroy() {
    // Only when the App is closed, not when the screen turns.
    if (isFinishing) RunnerModel.stop()
    super.onDestroy()
  }
}
