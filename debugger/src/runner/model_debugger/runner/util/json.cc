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

#include "model_debugger/runner/util/json.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace model_debugger::runner {
namespace {

// Returns `value` as compact text, treating bytes that are not UTF-8 as
// `handler` says.
std::string Write(const Json& value, Json::error_handler_t handler) {
  return value.dump(/*indent=*/-1, /*indent_char=*/' ', /*ensure_ascii=*/false,
                    handler);
}

}  // namespace

absl::StatusOr<Json> ParseJson(absl::string_view text) {
  Json value = Json::parse(text, /*cb=*/nullptr, /*allow_exceptions=*/false);
  if (value.is_discarded()) {
    return absl::InvalidArgumentError("Expected a JSON value.");
  }
  return value;
}

absl::StatusOr<std::string> Dump(const Json& value) {
  // The handler that refuses bytes that are not UTF-8 does so by throwing. Of
  // the other two, one replaces such bytes and one drops them, so their texts
  // differ exactly when there are any.
  std::string replaced = Write(value, Json::error_handler_t::replace);
  if (replaced != Write(value, Json::error_handler_t::ignore)) {
    return absl::InvalidArgumentError("Expected text in UTF-8.");
  }
  return replaced;
}

}  // namespace model_debugger::runner
