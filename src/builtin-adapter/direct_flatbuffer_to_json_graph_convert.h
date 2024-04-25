/* Copyright 2024 The Model Explorer Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_DIRECT_FLATBUFFER_TO_JSON_GRAPH_CONVERT_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_DIRECT_FLATBUFFER_TO_JSON_GRAPH_CONVERT_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "visualize_config.h"

namespace tooling {
namespace visualization_client {

// Converts a Flatbuffer to visualizer JSON string. This process entails neither
// converting to MLIR nor preparing the model for execution.
absl::StatusOr<std::string> ConvertFlatbufferDirectlyToJson(
    const VisualizeConfig& config, absl::string_view model_path);

}  // namespace visualization_client
}  // namespace tooling

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_DIRECT_FLATBUFFER_TO_JSON_GRAPH_CONVERT_H_
