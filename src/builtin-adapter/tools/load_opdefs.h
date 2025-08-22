// Copyright 2024 The AI Edge Model Explorer Authors.
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
// =============================================================================

#ifndef TOOLS_LOAD_OPDEFS_H_
#define TOOLS_LOAD_OPDEFS_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace tooling {
namespace visualization_client {

struct OpMetadata {
  std::vector<std::string> arguments;
  std::vector<std::string> results;

  // Move Constructor
  OpMetadata(std::vector<std::string>&& arguments,
             std::vector<std::string>&& results)
      : arguments(std::move(arguments)), results(std::move(results)) {}
};

absl::flat_hash_map<std::string, OpMetadata> LoadTfliteOpdefs();

}  // namespace visualization_client
}  // namespace tooling

#endif  // TOOLS_LOAD_OPDEFS_H_
