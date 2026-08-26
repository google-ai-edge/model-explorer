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

#ifndef THIRD_PARTY_ADAPTERS_MLIR_MODEL_JSON_GRAPH_CONVERT_H_
#define THIRD_PARTY_ADAPTERS_MLIR_MODEL_JSON_GRAPH_CONVERT_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "common/visualize_config.h"

namespace model_explorer {
namespace adapter {

// Converts a SavedModel to visualizer JSON string through tf dialect MLIR.
absl::StatusOr<std::string> ConvertSavedModelToJson(
    const VisualizeConfig& config, absl::string_view model_path);

// Converts a Flatbuffer to visualizer JSON string through tfl dialect MLIR.
absl::StatusOr<std::string> ConvertFlatbufferToJson(
    const VisualizeConfig& config, absl::string_view model_path_or_buffer,
    bool is_modelpath);

// Converts a MLIR textual/bytecode file to visualizer JSON string.
// Note: now only supports tf, tfl, stablehlo dialects inside the file.
absl::StatusOr<std::string> ConvertMlirToJson(const VisualizeConfig& config,
                                              absl::string_view model_path);

}  // namespace adapter
}  // namespace model_explorer
#endif  // THIRD_PARTY_ADAPTERS_MLIR_MODEL_JSON_GRAPH_CONVERT_H_
