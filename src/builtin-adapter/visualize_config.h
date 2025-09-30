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

#ifndef VISUALIZE_CONFIG_H_
#define VISUALIZE_CONFIG_H_

namespace tooling {
namespace visualization_client {

struct VisualizeConfig {
  VisualizeConfig() = default;
  explicit VisualizeConfig(const int const_element_count_limit)
      : const_element_count_limit(const_element_count_limit) {}

  // The maximum number of constant elements to be displayed. If the number
  // exceeds this threshold, the rest of data will be elided. The default
  // threshold is set to 16 (use -1 to print all).
  int const_element_count_limit = 16;

  // If true, adds the `tensor_name` meta attribute to the node from the MLIR
  // location. This attribute is always added for the tfl dialect.
  bool add_tensor_name_attribute = false;
};

}  // namespace visualization_client
}  // namespace tooling

#endif  // VISUALIZE_CONFIG_H_
