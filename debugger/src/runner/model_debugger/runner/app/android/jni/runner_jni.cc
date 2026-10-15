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

// The C facade (runner_c.h) for Kotlin:
// com.google.ai.edge.modeldebugger.runner.RunnerModel's three native functions,
// one per facade function.

#include <jni.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "model_debugger/runner/runner_c.h"

extern "C" {

// Starts a Runner on `listen` and returns it as a handle; throws
// IllegalStateException with the reason when it cannot start.
JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_modeldebugger_runner_RunnerModel_runnerStart(
    JNIEnv* env, jclass /*runner_model*/, jstring listen) {
  const char* address = env->GetStringUTFChars(listen, nullptr);
  char error[1024] = "";
  ModelDebuggerRunner* runner =
      model_debugger_runner_start(address, error, sizeof(error));
  env->ReleaseStringUTFChars(listen, address);
  if (runner == nullptr) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error);
    return 0;
  }
  return static_cast<jlong>(reinterpret_cast<intptr_t>(runner));
}

// The Runner's state, as model_debugger_runner_get_state writes it, whole:
// the state can grow between two reads, so a buffer it did not fit in is
// grown and the state read again.
JNIEXPORT jstring JNICALL
Java_com_google_ai_edge_modeldebugger_runner_RunnerModel_runnerGetState(
    JNIEnv* env, jclass /*runner_model*/, jlong handle) {
  auto* runner = reinterpret_cast<ModelDebuggerRunner*>(handle);
  std::string state(4096, '\0');
  while (true) {
    const size_t length =
        model_debugger_runner_get_state(runner, state.data(), state.size());
    if (length < state.size()) break;
    state.resize(length + 1);
  }
  return env->NewStringUTF(state.c_str());
}

// Stops the Runner and frees it.
JNIEXPORT void JNICALL
Java_com_google_ai_edge_modeldebugger_runner_RunnerModel_runnerDelete(
    JNIEnv* /*env*/, jclass /*runner_model*/, jlong handle) {
  model_debugger_runner_delete(reinterpret_cast<ModelDebuggerRunner*>(handle));
}

}  // extern "C"
