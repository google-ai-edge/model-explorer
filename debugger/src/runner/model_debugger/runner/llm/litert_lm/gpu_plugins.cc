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

#include "model_debugger/runner/llm/litert_lm/gpu_plugins.h"

#include <dlfcn.h>

#include <filesystem>
#include <string_view>
#include <system_error>

#include "absl/strings/str_cat.h"
#include "litert_lm/plugins.h"

namespace model_debugger::runner::llm {

absl::Status LoadLiteRtGpuPlugins(std::string_view directory) {
  std::error_code error;  // The overload without throw; nothing catches.
  // In the order the platform's LiteRT-LM names them: a plugin's own
  // dependencies first.
  for (const char* name : litert_lm::kGpuPlugins) {
    const std::filesystem::path file = std::filesystem::path(directory) / name;
    if (!std::filesystem::exists(file, error)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Missing GPU library: ", file.string(), "."));
    }
    if (dlopen(file.c_str(), RTLD_NOW | RTLD_LOCAL) == nullptr) {
      const char* detail = dlerror();
      return absl::InternalError(
          absl::StrCat("Could not load GPU library ", name, ": ",
                       detail != nullptr ? detail : "unknown error", "."));
    }
  }
  return absl::OkStatus();
}

bool HasLiteRtGpuPlugins(std::string_view directory) {
  std::error_code error;  // The overload without throw; nothing catches.
  for (const char* name : litert_lm::kGpuPlugins) {
    if (!std::filesystem::exists(std::filesystem::path(directory) / name,
                                 error)) {
      return false;
    }
  }
  return true;
}

}  // namespace model_debugger::runner::llm
