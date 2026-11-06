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

#ifndef MODEL_DEBUGGER_RUNNER_LLM_LITERT_LM_CAPTURE_FILES_H_
#define MODEL_DEBUGGER_RUNNER_LLM_LITERT_LM_CAPTURE_FILES_H_

#include <filesystem>  // NOLINT(build/c++17)
#include <set>
#include <string>

#include "absl/status/status.h"

namespace model_debugger::runner::llm {

// Checks that each safetensors file of `names` in `dir` holds every byte its
// header gives. LiteRT-LM's RuntimeDebugger does not check its writes
// (runtime_debugger.cc): one its disk has no room for leaves the file cut
// short, and the turn goes on. A file cut short by more bytes than its disk
// has free ends the Runner, saying so (ExitIfOutOfRoom). The other names, the
// tokens file among them, are not looked at.
//
// Returns Internal for the first file cut short otherwise, or that cannot be
// read or has no safetensors header.
absl::Status CheckWholeFiles(const std::filesystem::path& dir,
                             const std::set<std::string>& names);

}  // namespace model_debugger::runner::llm

#endif  // MODEL_DEBUGGER_RUNNER_LLM_LITERT_LM_CAPTURE_FILES_H_
