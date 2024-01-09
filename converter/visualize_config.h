#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_VISUALIZE_CONFIG_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_VISUALIZE_CONFIG_H_

namespace tooling {
namespace visualization_client {

struct VisualizeConfig {
  VisualizeConfig() = default;
  explicit VisualizeConfig(const int const_element_count_limit)
      : const_element_count_limit(const_element_count_limit) {}

  // The maximum number of constant elements to be displayed. If the number
  // exceeds this threshold, the rest of data will be elided. The default
  // threshold is set to 16 (use -1 to print all).
  int const_element_count_limit = 16;
};

}  // namespace visualization_client
}  // namespace tooling

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_VISUALIZE_CONFIG_H_
