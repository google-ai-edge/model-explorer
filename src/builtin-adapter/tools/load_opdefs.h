#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TOOLS_LOAD_OPDEFS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TOOLS_LOAD_OPDEFS_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace tooling {
namespace visualization_client {

struct OpMetadata {
  std::vector<std::string> arguments;
  std::vector<std::string> results;
};

absl::flat_hash_map<std::string, OpMetadata> LoadTfliteOpdefs();

absl::flat_hash_map<std::string, OpMetadata> LoadTfOpdefs();

}  // namespace visualization_client
}  // namespace tooling

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TOOLS_LOAD_OPDEFS_H_
