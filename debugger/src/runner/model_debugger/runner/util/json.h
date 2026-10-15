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

// JSON text, written and read without exceptions.
//
// JSON is a format of the edges of the Runner: layers hand each other objects.
// This header is included by a .cc that converts, never by another header:
// a rule for review, not for a tool.

#ifndef MODEL_DEBUGGER_RUNNER_UTIL_JSON_H_
#define MODEL_DEBUGGER_RUNNER_UTIL_JSON_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "nlohmann/json.hpp"

namespace model_debugger::runner {

// The JSON library of the Runner. It throws on a member that is absent or of
// another type, and nothing in the Runner catches: read a member only after
// checking it is there.
using Json = nlohmann::json;

// Returns the value that `text` holds.
//
// Returns InvalidArgument if `text` is not JSON, or is not UTF-8.
absl::StatusOr<Json> ParseJson(absl::string_view text);

// Returns `value` as compact JSON text.
//
// Returns InvalidArgument if a string in `value` is not UTF-8. The text is the
// input of a model, so it is written as given or not at all. A number that is
// NaN or infinite is written as null, which is not reported.
absl::StatusOr<std::string> Dump(const Json& value);

}  // namespace model_debugger::runner

#endif  // MODEL_DEBUGGER_RUNNER_UTIL_JSON_H_
