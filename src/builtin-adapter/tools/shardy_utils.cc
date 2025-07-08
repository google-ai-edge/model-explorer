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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace tooling {
namespace visualization_client {
namespace {

using ::mlir::Attribute;
using ::mlir::Operation;
using ::mlir::sdy::strippedAttrString;

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

void PrettyPrint(mlir::sdy::PropagationEdgesAttr attr, llvm::raw_ostream& os) {
  for (mlir::sdy::PropagationOneStepAttr edge : attr) {
    os << "{ step-" << edge.getStepIndex() << " = [\n";
    ::llvm::ArrayRef<mlir::sdy::AxisToPropagationDetailsAttr> axis_entries =
        edge.getAxisEntries();
    for (mlir::sdy::AxisToPropagationDetailsAttr axis_entry : axis_entries) {
      os << '\t' << strippedAttrString(axis_entry, /*strip_menemonic=*/true);
      if (&axis_entry != &axis_entries.back()) {
        os << ",";
      }
      os << "\n";
    }
    os << "]}";
    if (&edge != &attr.back()) {
      os << ",\n";
    }
  }
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
      .Case<mlir::sdy::PropagationEdgesAttr>(
          [&os](mlir::sdy::PropagationEdgesAttr propagation_edges_attr) {
            PrettyPrint(propagation_edges_attr, os);
          })
      .Default(
          [&](mlir::Attribute attr) { attr.print(os, /*elideType=*/true); });
}

void AddReferencedMesh(
    Attribute attr,
    llvm::SmallDenseMap<llvm::StringRef, mlir::sdy::MeshAttr>& sdy_meshes,
    Operation& operation) {
  return llvm::TypeSwitch<mlir::Attribute>(attr)
      .Case<mlir::sdy::TensorShardingAttr>(
          [&sdy_meshes,
           &operation](mlir::sdy::TensorShardingAttr sharding_attr) {
            mlir::sdy::MeshAttr mesh_attr = sharding_attr.getMesh(&operation);
            llvm::StringRef mesh_name = sharding_attr.getMeshName();
            if (!mesh_attr.empty() && !sdy_meshes.contains(mesh_name)) {
              sdy_meshes[mesh_name] = mesh_attr;
            }
          })
      .Case<mlir::sdy::TensorShardingPerValueAttr>(
          [&sdy_meshes, &operation](
              mlir::sdy::TensorShardingPerValueAttr sharding_per_value_attr) {
            for (const auto& sharding :
                 sharding_per_value_attr.getShardings()) {
              mlir::sdy::MeshAttr mesh_attr = sharding.getMesh(&operation);
              llvm::StringRef mesh_name = sharding.getMeshName();
              if (!mesh_attr.empty() && !sdy_meshes.contains(mesh_name)) {
                sdy_meshes[mesh_name] = mesh_attr;
              }
            }
          })
      .Default([&](mlir::Attribute attr) {});
}

}  // namespace visualization_client
}  // namespace tooling
