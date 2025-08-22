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

#ifndef MODELS_TO_JSON_LIB_H_
#define MODELS_TO_JSON_LIB_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "visualize_config.h"

namespace tooling {
namespace visualization_client {

// Converts a model to model explorer JSON string.
//
// The model can be a TFLite Flatbuffer, a TF SavedModel or GraphDef, or a
// StableHLO module represented using MLIR textual or bytecode format.
//
// If `disable_mlir` is true, the model will be converted to JSON directly
// without going through MLIR. Currently, this only applies to TFLite and TF
// adapters. For TFLite, it's preferred to set `disable_mlir` to true. For TF
// SavedModel, it's preferred to set `disable_mlir` to false.
absl::StatusOr<std::string> ConvertModelToJson(const VisualizeConfig& config,
                                               absl::string_view input_file,
                                               bool disable_mlir);

}  // namespace visualization_client
}  // namespace tooling

#endif  // MODELS_TO_JSON_LIB_H_
