#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_DIRECT_SAVED_MODEL_TO_JSON_GRAPH_CONVERT_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_DIRECT_SAVED_MODEL_TO_JSON_GRAPH_CONVERT_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "visualize_config.h"
#include "tensorflow/core/framework/graph.pb.h"

namespace tooling {
namespace visualization_client {

// Converts a GraphDef to visualizer JSON string. This process entails neither
// converting to MLIR nor preparing the model for execution, thus avoiding
// dependencies (such as those on custom ops) and saving time.
absl::StatusOr<std::string> ConvertGraphDefDirectlyToJsonImpl(
    const VisualizeConfig& config, const tensorflow::GraphDef& graph_def);

// Reads the a GraphDef from `.pb`, `.pbtxt` or `.graphdef` file and converts to
// JSON string without going through MLIR or execution.
absl::StatusOr<std::string> ConvertGraphDefDirectlyToJson(
    const VisualizeConfig& config, absl::string_view model_path);

// An alternative signature that takes the path to a SavedModel Proto
// `saved_model.pb`. Again, no conversion to MLIR nor preparing model for
// execution takes place.
absl::StatusOr<std::string> ConvertSavedModelDirectlyToJson(
    const VisualizeConfig& config, absl::string_view saved_model_pb_path);

}  // namespace visualization_client
}  // namespace tooling

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_DIRECT_SAVED_MODEL_TO_JSON_GRAPH_CONVERT_H_
