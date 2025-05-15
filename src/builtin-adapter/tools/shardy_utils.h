// Copyright 2025 The AI Edge Model Explorer Authors.
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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TOOLS_SHARDY_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TOOLS_SHARDY_UTILS_H_

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace tooling {
namespace visualization_client {

// Returns true if the attribute is from the Shardy dialect.
inline bool IsShardyDialect(mlir::Attribute attr) {
  return llvm::isa<mlir::sdy::SdyDialect>(attr.getDialect());
}

// Prints Shardy attributes to stream (using simplified pretty printing for
// select Shardy attributes).
void PrintShardyAttribute(mlir::Attribute attr, llvm::raw_string_ostream& os);

}  // namespace visualization_client
}  // namespace tooling

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_TOOLS_SHARDY_UTILS_H_
