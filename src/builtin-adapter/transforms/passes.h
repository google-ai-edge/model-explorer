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

#ifndef TRANSFORMS_PASSES_H_
#define TRANSFORMS_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace tooling {
namespace visualization_client {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "transforms/passes.h.inc"

// Creates a pass to assign unique names to operations.
std::unique_ptr<mlir::Pass> CreateUniqueOpNamesPass();

}  // namespace visualization_client
}  // namespace tooling

#endif  // TRANSFORMS_PASSES_H_
