#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_MODEL_JSON_GRAPH_CONVERT_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_MODEL_JSON_GRAPH_CONVERT_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "visualize_config.h"

namespace tooling {
namespace visualization_client {

// Converts a SavedModel v1 to visualizer JSON string.
absl::StatusOr<std::string> ConvertSavedModelV1ToJson(
    const VisualizeConfig& config, absl::string_view model_path);

// Converts a SavedModel v2 to visualizer JSON string.
absl::StatusOr<std::string> ConvertSavedModelV2ToJson(
    const VisualizeConfig& config, absl::string_view model_path);

// Converts a Flatbuffer to visualizer JSON string.
absl::StatusOr<std::string> ConvertFlatbufferToJson(
    const VisualizeConfig& config, absl::string_view model_path_or_buffer,
    bool is_modelpath);

// Converts a MLIR textual/bytecode file to visualizer JSON string.
// Note: this expects StableHLO inside the bytecode file.
absl::StatusOr<std::string> ConvertStablehloMlirToJson(
    const VisualizeConfig& config, absl::string_view model_path);

}  // namespace visualization_client
}  // namespace tooling

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_MODEL_JSON_GRAPH_CONVERT_H_
