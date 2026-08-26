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

#ifndef THIRD_PARTY_UTILS_CONVERT_TYPE_H_
#define THIRD_PARTY_UTILS_CONVERT_TYPE_H_

#include <string>

#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

namespace model_explorer {
namespace adapter {

std::string TensorTypeToString(tflite::TensorType type);

}  // namespace adapter
}  // namespace model_explorer
#endif  // THIRD_PARTY_UTILS_CONVERT_TYPE_H_
