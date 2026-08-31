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

#ifndef MODEL_EXPLORER_BACKEND_ADAPTERS_MLIR_TRANSLATE_HELPERS_H_
#define MODEL_EXPLORER_BACKEND_ADAPTERS_MLIR_TRANSLATE_HELPERS_H_

#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "common/schema_structs.h"
#include "common/visualize_config.h"

namespace model_explorer {
namespace adapter {

// Converts an MLIR function op to a subgraph.
absl::StatusOr<Subgraph> FuncOpToSubgraph(const VisualizeConfig& config,
                                          mlir::func::FuncOp& fop);

// Converts a diact-agnostic MLIR module to a JSON graph.
absl::StatusOr<Graph> MlirToGraph(const VisualizeConfig& config,
                                  mlir::Operation* module);

}  // namespace adapter
}  // namespace model_explorer
#endif  // MODEL_EXPLORER_BACKEND_ADAPTERS_MLIR_TRANSLATE_HELPERS_H_
