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

// A Runner for the Apps: runner.h as a C API, one function per member of
// Runner the Apps use (Start, GetState, the destructor), so that the App on
// macOS and iOS needs no C++ interop and the Android App reaches it through
// JNI (app/android/jni/runner_jni.cc). Nothing is allocated for the caller:
// text goes into the caller's buffer.

#ifndef MODEL_DEBUGGER_RUNNER_RUNNER_C_H_
#define MODEL_DEBUGGER_RUNNER_RUNNER_C_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// A running Runner.
typedef struct ModelDebuggerRunner ModelDebuggerRunner;

// Starts a Runner listening on HOST:PORT (port 0 picks a free one). Returns
// NULL when it cannot start, with why in `error` (NUL-terminated, cut to
// `error_size`).
ModelDebuggerRunner* model_debugger_runner_start(const char* listen,
                                                 char* error,
                                                 size_t error_size);

// Writes runner.h's State as a JSON object into `buffer` (NUL-terminated):
//   {"platform": "macOS", "os_version": "27.2", "device": "Apple M2 Max",
//    "address": "127.0.0.1", "port": 8769, "client": "", "model_sha256": "",
//    "turns": 0, "chat_over": false}
// Returns the length it needs; more than `size` - 1 means it was cut.
// Callable from any thread.
size_t model_debugger_runner_get_state(ModelDebuggerRunner* runner,
                                       char* buffer, size_t size);

// Runner's destructor: shuts it down (in-flight calls finish), removes its
// run directory and frees it. The App calls it when the application ends; a
// Runner left to the end of the process is stopped under its feet. NULL is
// allowed.
void model_debugger_runner_delete(ModelDebuggerRunner* runner);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MODEL_DEBUGGER_RUNNER_RUNNER_C_H_
