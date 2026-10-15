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

#ifndef MODEL_DEBUGGER_RUNNER_LLM_LITERT_LM_GPU_PLUGINS_H_
#define MODEL_DEBUGGER_RUNNER_LLM_LITERT_LM_GPU_PLUGINS_H_

#include <string_view>

#include "absl/status/status.h"

namespace model_debugger::runner::llm {

// Loads the GPU plugins found in `directory` into the process.
//
// The runtime finds them by name among what is loaded. They are never
// unloaded, because LiteRT keeps their code beyond the life of any engine, so
// calling this again is harmless.
//
// Returns FailedPrecondition if a plugin is missing: this build cannot run on
// the GPU. Returns Internal with the loader's reason if one does not load.
absl::Status LoadLiteRtGpuPlugins(std::string_view directory);

// Returns whether every GPU plugin is in `directory`: what
// LoadLiteRtGpuPlugins needs of it.
bool HasLiteRtGpuPlugins(std::string_view directory);

}  // namespace model_debugger::runner::llm

#endif  // MODEL_DEBUGGER_RUNNER_LLM_LITERT_LM_GPU_PLUGINS_H_
