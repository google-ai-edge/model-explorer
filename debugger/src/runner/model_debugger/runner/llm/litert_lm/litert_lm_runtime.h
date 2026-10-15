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

#ifndef MODEL_DEBUGGER_RUNNER_LLM_LITERT_LM_LITERT_LM_RUNTIME_H_
#define MODEL_DEBUGGER_RUNNER_LLM_LITERT_LM_LITERT_LM_RUNTIME_H_

#include <filesystem>
#include <memory>

#include "absl/status/statusor.h"
#include "model_debugger/runner/llm/runtime.h"
#include "model_debugger/v1/runner.pb.h"

namespace model_debugger::runner::llm {

// LiteRT-LM as a Runtime, on its C API. The GPU plugins (gpu_plugins.h) are
// loaded from beside the library, where whoever built it put them.
//
// The runtime options of Load, and of a chat, are the ones Info() declares in
// its two schemas: engine settings by the names of their setters in
// third_party/LiteRT-LM/c/engine.h, in camelCase; session and conversation
// settings in OpenAI's style, in snake_case. Sampling maps to LiteRT-LM's
// top-p sampler, the one it implements; greedy is a top_k of 1. A chat that
// sets some of top_k, top_p, temperature and seed gets the model's own
// sampling for the rest, or LiteRT-LM's defaults where the model declares
// none; an unset seed is 0, since the C API does not say a model's.
class LiteRtLmRuntime final : public Runtime {
 public:
  LiteRtLmRuntime();

  const v1::RuntimeInfo& Info() const override { return info_; }

  // Whether the GPU plugins of this platform are beside the LiteRT-LM
  // library, where Load looks for them.
  bool CanRunOnGpu() const override;

  absl::StatusOr<std::unique_ptr<Model>> Load(
      const std::filesystem::path& model, const std::filesystem::path& data_dir,
      const v1::LoadRequest& request) override;

 private:
  const v1::RuntimeInfo info_;
};

}  // namespace model_debugger::runner::llm

#endif  // MODEL_DEBUGGER_RUNNER_LLM_LITERT_LM_LITERT_LM_RUNTIME_H_
