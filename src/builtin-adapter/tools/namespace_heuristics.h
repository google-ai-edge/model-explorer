// Copyright 2024 The AI Edge Model Explorer Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TOOLS_NAMESPACE_HEURISTICS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TOOLS_NAMESPACE_HEURISTICS_H_

#include <string>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
namespace tooling {
namespace visualization_client {

// Obtains the best matching namespace for the TFLite node based on the provided
// node label and candidate names.
//
// The candidate names are obtained from the tensor names. The node namespace is
// obtained by the following steps:
// 1. If there are no candidate names, returns an empty string.
// 2. If there is only one candidate name, returns the candidate name.
// 3. If there are multiple candidate names, iterates backwards and returns the
// candidate name with the minimum edit distance to the node label.
std::string TfliteNodeNamespaceHeuristic(
    absl::string_view node_label,
    absl::Span<const std::string> candidate_names);

}  // namespace visualization_client
}  // namespace tooling

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TOOLS_NAMESPACE_HEURISTICS_H_
