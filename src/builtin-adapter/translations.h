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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TRANSLATIONS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TRANSLATIONS_H_

#include <utility>

#include "absl/status/statusor.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "formats/schema_structs.h"
#include "translate_helpers.h"
#include "visualize_config.h"

namespace tooling {
namespace visualization_client {

static mlir::LogicalResult TfMlirToJsonTranslateImpl(
    const VisualizeConfig& config, mlir::Operation* op,
    llvm::raw_ostream& output) {
  absl::StatusOr<Graph> result = TfMlirToGraph(config, op);
  if (!result.ok()) {
    return mlir::LogicalResult::failure();
  }

  GraphCollection collection;
  collection.graphs.push_back(std::move(*result));
  llvm::json::Value json_result(collection.Json());
  output << llvm::formatv("{0:2}", json_result);
  return mlir::LogicalResult::success();
}

static mlir::LogicalResult TfliteMlirToJsonTranslateImpl(
    const VisualizeConfig& config, mlir::Operation* op,
    llvm::raw_ostream& output) {
  absl::StatusOr<Graph> result = TfliteMlirToGraph(config, op);
  if (!result.ok()) {
    return mlir::LogicalResult::failure();
  }

  GraphCollection collection;
  collection.graphs.push_back(std::move(*result));
  llvm::json::Value json_result(collection.Json());
  output << llvm::formatv("{0:2}", json_result);
  return mlir::LogicalResult::success();
}

static mlir::LogicalResult JaxConvertedMlirToJsonTranslateImpl(
    const VisualizeConfig& config, mlir::Operation* op,
    llvm::raw_ostream& output) {
  absl::StatusOr<Graph> result = JaxConvertedMlirToGraph(config, op);
  if (!result.ok()) {
    return mlir::LogicalResult::failure();
  }

  GraphCollection collection;
  collection.graphs.push_back(std::move(*result));
  llvm::json::Value json_result(collection.Json());
  output << llvm::formatv("{0:2}", json_result);
  return mlir::LogicalResult::success();
}

// NOLINTNEXTLINE
static mlir::LogicalResult TfMlirToJsonTranslate(mlir::Operation* op,
                                                 llvm::raw_ostream& output) {
  // When translating MLIR dump file to json graph, we assume users need all
  // element data. Users need to manage the desired element data in dump file.
  return TfMlirToJsonTranslateImpl(
      VisualizeConfig(/*const_element_count_limit=*/-1), op, output);
}

// NOLINTNEXTLINE
static mlir::LogicalResult TfliteMlirToJsonTranslate(
    mlir::Operation* op, llvm::raw_ostream& output) {
  // When translating MLIR dump file to json graph, we assume users need all
  // element data. Users need to manage the desired element data in dump file.
  return TfliteMlirToJsonTranslateImpl(
      VisualizeConfig(/*const_element_count_limit=*/-1), op, output);
}

// NOLINTNEXTLINE
static mlir::LogicalResult JaxConvertedMlirToJsonTranslate(
    mlir::Operation* op, llvm::raw_ostream& output) {
  // When translating MLIR dump file to json graph, we assume users need all
  // element data. Users need to manage the desired element data in dump file.
  return JaxConvertedMlirToJsonTranslateImpl(
      VisualizeConfig(/*const_element_count_limit=*/-1), op, output);
}

}  // namespace visualization_client
}  // namespace tooling

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TRANSLATIONS_H_
