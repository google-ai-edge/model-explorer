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

#include <algorithm>
#include <string>

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace tooling {
namespace visualization_client {
namespace {

// Prints a simplified version of manual axes, which is easier for users to
// parse. For example it displays the default printing of
// #sdy<manual_axes{a,b}> as {'a', 'b'}
void PrettyPrint(const mlir::sdy::ManualAxesAttr& attr,
                 llvm::raw_string_ostream& os) {
  os << "{";
  auto axes = attr.getValue();
  int index = 0;
  for (auto& axis : axes) {
    os << "'" << axis.str() << "'";
    if (index != axes.size() - 1) {
      os << ", ";
    }
    index++;
  }
  os << "}";
};

// Prints a simplified version of the dimension sharding attribute, which is
// easier for users to parse. This would display a dimension sharding attr like
// {x, y:(2)2, ?} as {'x', 'y':(2)2, ?}
void PrettyPrint(const mlir::sdy::DimensionShardingAttr& attr,
                 llvm::raw_ostream& os) {
  auto axes = attr.getAxes();
  bool open = !attr.getIsClosed();

  os << "{";
  int index = 0;
  for (auto axis : axes) {
    std::string axis_str = axis.toString();
    std::replace(axis_str.begin(), axis_str.end(), '"', '\'');
    os << axis_str;
    if (index != axes.size() - 1 || open) {
      os << ", ";
    }
    index++;
  }
  if (open) {
    os << "?";
  }
  os << "}";
}

// Prints a simplified version of the sharding attribute, which is easier
// for users to parse. For example, it displays a sharding attr like
// #sdy.sharding<@mesh, [{}, {c}]> as <@mesh, [{}, {‘c’}]>
void PrettyPrint(const mlir::sdy::TensorShardingAttr& attr,
                 llvm::raw_ostream& os) {
  os << "<";
  attr.getMeshOrRef().print(os);
  os << ", [";
  int index = 0;
  for (mlir::sdy::DimensionShardingAttr dimSharding : attr.getDimShardings()) {
    PrettyPrint(dimSharding, os);
    if (index != attr.getDimShardings().size() - 1) {
      os << ", ";
    }
    index++;
  }
  os << "]>";
}

// Prints a simplified version of the sharding per value attribute, which is
// easier for users to parse. For example, it displays a sharding attr like
// #sdy.sharding_per_value<[<@meshaA, [{}, {a}]>, <@meshB, [{b}]>]> as
// [<@meshaA, [{}, {'a'}]>, <@meshB, [{'b'}]>]
void PrettyPrint(const mlir::sdy::TensorShardingPerValueAttr& attr,
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
    os << "\t" << index << ": ";
    PrettyPrint(sharding, os);
    if (sharding != shardings.back()) {
      os << ",\n";
    }
    index++;
  }
  os << "\n]";
}

}  // namespace

void PrintShardyAttribute(const mlir::Attribute& attr,
                          llvm::raw_string_ostream& os) {
  if (const auto& sharding_attr =
          llvm::dyn_cast_or_null<mlir::sdy::TensorShardingAttr>(attr)) {
    PrettyPrint(sharding_attr, os);
    return;
  }
  if (const auto& sharding_per_value_attr =
          llvm::dyn_cast_or_null<mlir::sdy::TensorShardingPerValueAttr>(attr)) {
    PrettyPrint(sharding_per_value_attr, os);
    return;
  }
  if (const auto& manual_axes_attr =
          llvm::dyn_cast_or_null<mlir::sdy::ManualAxesAttr>(attr)) {
    PrettyPrint(manual_axes_attr, os);
    return;
  }
  attr.print(os, /*elideType=*/true);
}

}  // namespace visualization_client
}  // namespace tooling
