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
