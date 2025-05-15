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

#include "tools/shardy_utils.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

using ::mlir::sdy::strippedAttrString;

namespace tooling {
namespace visualization_client {
namespace {

// Prints a simplified version of manual axes, which is easier for users to
// parse. For example it displays the default printing of
// #sdy<manual_axes{"a","b"}> as {"a", "b"}
void PrettyPrint(mlir::sdy::ManualAxesAttr attr, llvm::raw_string_ostream& os) {
  os << strippedAttrString(attr, /*strip_menemonic=*/true);
};

// Prints a simplified version of the sharding attribute, which is easier
// for users to parse. For example, it displays a sharding attr like
// #sdy.sharding<@mesh, [{}, {"c"}]> as <@mesh, [{}, {"c"}]>
void PrettyPrint(mlir::sdy::TensorShardingAttr attr, llvm::raw_ostream& os) {
  os << strippedAttrString(attr, /*strip_menemonic=*/true);
}

// Prints a simplified version of the sharding per value attribute, which is
// easier for users to parse. For example, it displays a sharding attr like
// #sdy.sharding_per_value<[<@meshaA, [{}, {"a"}]>, <@meshB, [{"b"}]>]> as
// [
//     0: <@meshaA, [{}, {'a'}]>,
//     1: <@meshB, [{'b'}]>
// ]
void PrettyPrint(mlir::sdy::TensorShardingPerValueAttr attr,
                 llvm::raw_ostream& os) {
  auto shardings = attr.getShardings();
  if (shardings.empty()) {
    return;
  }
  if (shardings.size() == 1) {
    PrettyPrint(shardings[0], os);
    return;
  }

  os << "[\n";
  int index = 0;
  for (mlir::sdy::TensorShardingAttr sharding : shardings) {
    os << "\t" << index++ << ": ";
    PrettyPrint(sharding, os);
    if (sharding != shardings.back()) {
      os << ",\n";
    }
  }
  os << "\n]";
}

}  // namespace

void PrintShardyAttribute(mlir::Attribute attr, llvm::raw_string_ostream& os) {
  return llvm::TypeSwitch<mlir::Attribute>(attr)
      .Case<mlir::sdy::TensorShardingAttr>(
          [&os](mlir::sdy::TensorShardingAttr sharding_attr) {
            PrettyPrint(sharding_attr, os);
          })
      .Case<mlir::sdy::TensorShardingPerValueAttr>(
          [&os](mlir::sdy::TensorShardingPerValueAttr sharding_per_value_attr) {
            PrettyPrint(sharding_per_value_attr, os);
          })
      .Case<mlir::sdy::ManualAxesAttr>(
          [&os](mlir::sdy::ManualAxesAttr manual_axes_attr) {
            PrettyPrint(manual_axes_attr, os);
          })
      .Default(
          [&](mlir::Attribute attr) { attr.print(os, /*elideType=*/true); });
}

}  // namespace visualization_client
}  // namespace tooling
