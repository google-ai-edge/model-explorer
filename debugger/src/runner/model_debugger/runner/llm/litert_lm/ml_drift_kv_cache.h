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

#ifndef MODEL_DEBUGGER_RUNNER_LLM_LITERT_LM_ML_DRIFT_KV_CACHE_H_
#define MODEL_DEBUGGER_RUNNER_LLM_LITERT_LM_ML_DRIFT_KV_CACHE_H_

#include <filesystem>  // NOLINT(build/c++17)
#include <set>
#include <string>

#include "absl/status/status.h"

namespace model_debugger::runner::llm {

// Rewrites `file`, a K or V cache tensor that ML Drift's cache kernel wrote on
// the GPU and LiteRT-LM captured, into the layout and dtype its safetensors
// header declares.
//
// The kernel keeps an int8 cache as uint8 (q + 128) in a layout of its own
// (add_values_to_cache_kernel.cc), and the capture reads it back as plain
// PHWC4 int8 (buffer_handler_utils.cc). Checked against CPU captures on
// macOS: of Gemma 4 E2B and E4B (1 and 2 KV heads) with LiteRT-LM a0f1e97,
// and of E2B with 61dcfc7. Remove once LiteRT reads the cache back as
// declared.
//
// Returns InvalidArgument if `file` is not a KV cache whose last two
// dimensions are multiples of 4, and Internal if it cannot be read or
// written; `file` is then unchanged. A write its disk has no room for ends the
// Runner instead (ExitIfOutOfRoom).
absl::Status UnpackMlDriftKvCache(const std::filesystem::path& file);

// Unpacks with UnpackMlDriftKvCache each K and V cache file of `names` in
// `dir`, if ML Drift's cache kernel wrote them: on the GPU, LiteRT-LM gives
// it the cache of a model with a param tensor (llm_executor_settings_utils.cc,
// single_kv_cache_buffer), which the capture shows as param_tensor files
// among `names`. `names` are files as the runtime last wrote them, each
// unpacked as often as it is passed: a file the runtime writes again under
// a name it wrote before is packed again.
//
// Returns UnpackMlDriftKvCache's error for the first file it refuses, the
// files before it unpacked.
absl::Status UnpackMlDriftKvCaches(const std::filesystem::path& dir,
                                   const std::set<std::string>& names);

}  // namespace model_debugger::runner::llm

#endif  // MODEL_DEBUGGER_RUNNER_LLM_LITERT_LM_ML_DRIFT_KV_CACHE_H_
