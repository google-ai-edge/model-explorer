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

include "mlir/Pass/PassBase.td"

#ifndef THIRD_PARTY_TRANSFORMS_PASSES_TD_
#define THIRD_PARTY_TRANSFORMS_PASSES_TD_


def UniqueOpNames : Pass<"unique-op-names"> {
  let summary = "Assigns unique names to operations based on their location.";
  let description = [{
    This pass walks the module and ensures that each operation with a `NameLoc`
    has a unique name. If a duplicate name is found, it is uniquified by
    appending a numeric suffix. This is useful for visualization tools where
    duplicate names can cause rendering issues.
  }];
}

#endif  // THIRD_PARTY_TRANSFORMS_PASSES_TD_
