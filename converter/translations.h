#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TRANSLATIONS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TRANSLATIONS_H_

#include "third_party/absl/status/statusor.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/FormatVariadic.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/JSON.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/raw_ostream.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Operation.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Support/LogicalResult.h"
#include "third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling/formats/schema_structs.h"
#include "third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling/translate_helpers.h"
#include "third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling/visualize_config.h"

namespace tooling {
namespace visualization_client {

static mlir::LogicalResult TfMlirToJsonTranslateImpl(
    const VisualizeConfig& config, mlir::Operation* op,
    llvm::raw_ostream& output) {
  absl::StatusOr<Graph> result = TfMlirToGraph(config, op);
  if (!result.ok()) {
    return mlir::LogicalResult::failure();
  }
  // TODO(@lukeboyer) Make json formatting opts.
  llvm::json::Value json_result = llvm::json::Value(result->Json());
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
  // TODO(@lukeboyer) Make json formatting opts.
  llvm::json::Value json_result = llvm::json::Value(result->Json());
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
  llvm::json::Value json_result = llvm::json::Value(result->Json());
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
