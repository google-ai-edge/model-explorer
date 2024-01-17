#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TRANSLATE_HELPERS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TRANSLATE_HELPERS_H_

#include "absl/status/statusor.h"
#include "mlir/IR/Value.h"
#include "formats/schema_structs.h"
#include "visualize_config.h"

namespace tooling {
namespace visualization_client {

// Converts a tf dialect MLIR module to a JSON graph.
absl::StatusOr<Graph> TfMlirToGraph(const VisualizeConfig& config,
                                    mlir::Operation* module);

// Converts a tfl dialect MLIR module to a JSON graph.
absl::StatusOr<Graph> TfliteMlirToGraph(const VisualizeConfig& config,
                                        mlir::Operation* module);

// Converts a JAX-converted tf & stablehlo MLIR module to a JSON graph.
absl::StatusOr<Graph> JaxConvertedMlirToGraph(const VisualizeConfig& config,
                                              mlir::Operation* module);

}  // namespace visualization_client
}  // namespace tooling

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TRANSLATE_HELPERS_H_
