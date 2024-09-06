#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TOOLS_NAMESPACE_HEURISTICS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TOOLS_NAMESPACE_HEURISTICS_H_

#include <string>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
namespace tooling {
namespace visualization_client {

// Obtains the best matching namespace for the TFLite node based on the provided
// node label and candidate names.
//
// The candidate names are obtained from the tensor names. The node namespace is
// obtained by the following steps:
// 1. If there are no candidate names, returns an empty string.
// 2. If there is only one candidate name, returns the candidate name.
// 3. If there are multiple candidate names, iterates backwards and returns the
// candidate name with the minimum edit distance to the node label.
std::string TfliteNodeNamespaceHeuristic(
    absl::string_view node_label,
    absl::Span<const std::string> candidate_names);

}  // namespace visualization_client
}  // namespace tooling

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TOOLS_NAMESPACE_HEURISTICS_H_
