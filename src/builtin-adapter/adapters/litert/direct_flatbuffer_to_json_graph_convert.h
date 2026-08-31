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

#ifndef MODEL_EXPLORER_BACKEND_ADAPTERS_LITERT_DIRECT_FLATBUFFER_TO_JSON_GRAPH_CONVERT_H_
#define MODEL_EXPLORER_BACKEND_ADAPTERS_LITERT_DIRECT_FLATBUFFER_TO_JSON_GRAPH_CONVERT_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "common/status_reporter.h"
#include "common/visualize_config.h"

namespace model_explorer {
namespace adapter {

// Converts a Flatbuffer to visualizer JSON string. This process entails neither
// converting to MLIR nor preparing the model for execution.
absl::StatusOr<std::string> ConvertFlatbufferDirectlyToJson(
    const VisualizeConfig& config, absl::string_view model_path,
    StatusReporter* reporter = nullptr);

// Converts custom options to attributes.
// Logic referred from `CustomOptionsToAttributes` in
// tensorflow/compiler/mlir/lite/flatbuffer_operator.cc.
void CustomOptionsToAttributes(
    const std::vector<uint8_t>& custom_options, mlir::Builder mlir_builder,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes);

}  // namespace adapter
}  // namespace model_explorer
#endif  // MODEL_EXPLORER_BACKEND_ADAPTERS_LITERT_DIRECT_FLATBUFFER_TO_JSON_GRAPH_CONVERT_H_
