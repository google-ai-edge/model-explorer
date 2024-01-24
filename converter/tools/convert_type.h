#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TOOLS_CONVERT_TYPE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TOOLS_CONVERT_TYPE_H_

#include <string>

#include "tensorflow/lite/schema/schema_generated.h"

namespace tooling {
namespace visualization_client {

std::string TensorTypeToString(tflite::TensorType type);

}  // namespace visualization_client
}  // namespace tooling

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TOOLS_CONVERT_TYPE_H_
