// Copyright 2024 The AI Edge Model Explorer Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "tools/convert_type.h"

#include <string>

#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

namespace tooling {
namespace visualization_client {

std::string TensorTypeToString(tflite::TensorType type) {
  switch (type) {
    case tflite::TensorType_FLOAT32:
      return "float32";
    case tflite::TensorType_BFLOAT16:
      return "bf16";
    case tflite::TensorType_FLOAT16:
      return "float16";
    case tflite::TensorType_INT32:
      return "int32";
    case tflite::TensorType_UINT8:
      return "uint8";
    case tflite::TensorType_INT64:
      return "int64";
    case tflite::TensorType_STRING:
      return "string";
    case tflite::TensorType_BOOL:
      return "bool";
    case tflite::TensorType_INT16:
      return "int16";
    case tflite::TensorType_COMPLEX64:
      return "complex64";
    case tflite::TensorType_INT8:
      return "int8";
    case tflite::TensorType_FLOAT64:
      return "float64";
    case tflite::TensorType_COMPLEX128:
      return "complex128";
    case tflite::TensorType_UINT64:
      return "uint64";
    case tflite::TensorType_RESOURCE:
      return "resource";
    case tflite::TensorType_VARIANT:
      return "variant";
    case tflite::TensorType_UINT32:
      return "uint32";
    case tflite::TensorType_UINT16:
      return "uint16";
    case tflite::TensorType_INT4:
      return "int4";
    default:
      return "unknown";
  }
}

}  // namespace visualization_client
}  // namespace tooling
