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
#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace tooling {
namespace visualization_client {
namespace {

using ::mlir::Attribute;
using ::mlir::Operation;
using mlir::sdy::kBlockArgPropagationEdgesAttr;
using mlir::sdy::kPropagationEdgesAttr;
using mlir::sdy::kResultPropagationEdgesAttr;
using ::mlir::sdy::strippedAttrString;
using OpToNodeIdMap = absl::flat_hash_map<Operation*, std::string>;
using InputValueToNodeIdMap = llvm::SmallDenseMap<mlir::Value, std::string>;

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

bool HasShardyPropagationEdges(
    mlir::Operation& operation,
    llvm::SmallVector<mlir::sdy::PropagationEdgesAttr>& edges) {
  bool has_propagation_edges = false;
  auto propagation_edges =
      operation.getAttrOfType<mlir::sdy::PropagationEdgesAttr>(
          kPropagationEdgesAttr);
  if (propagation_edges) {
    edges.push_back(propagation_edges);
    has_propagation_edges = true;
  }

  for (const llvm::StringRef& edge_array_attr :
       {kBlockArgPropagationEdgesAttr, kResultPropagationEdgesAttr}) {
    if (operation.hasAttrOfType<mlir::ArrayAttr>(edge_array_attr)) {
      auto array_of_prop_edges =
          operation.getAttrOfType<mlir::ArrayAttr>(edge_array_attr);
      for (const auto& prop_edges : array_of_prop_edges) {
        edges.push_back(dyn_cast<mlir::sdy::PropagationEdgesAttr>(prop_edges));
        has_propagation_edges = true;
      }
    }
  }

  return has_propagation_edges;
}

absl::StatusOr<std::string> ResolveShardyOpFromValueRef(
    Operation* op, mlir::sdy::ValueRefAttr value_ref,
    const OpToNodeIdMap& op_to_id, const InputValueToNodeIdMap& input_nodes) {
  if (value_ref.getType() == mlir::sdy::EdgeNodeType::OPERAND) {
    if (value_ref.getIndex() >= op->getNumOperands()) {
      return absl::OutOfRangeError(
          "Value ref index is larger than number of operands.");
    }

    // If the operand is a block argument, we should resolve to the
    // corresponding node in the graph inputs.
    mlir::Value operand = op->getOperand(value_ref.getIndex());
    if (llvm::isa<mlir::BlockArgument>(operand)) {
      if (input_nodes.contains(operand)) {
        return input_nodes.at(operand);
      }
      return absl::NotFoundError("Value not found in input block args map.");
    }

    // Otherwise try to resolve to the defining op and return its node id.
    Operation* defining_op = operand.getDefiningOp();
    if (op_to_id.contains(defining_op)) {
      return op_to_id.at(defining_op);
    }
    return absl::NotFoundError("Op not found in op to node id map.");
  } else if (value_ref.getType() == mlir::sdy::EdgeNodeType::RESULT) {
    // When a vertex of a propagation edge is of type RESULT, then we should
    // resolve to the op itself.
    if (op_to_id.contains(op)) {
      return op_to_id.at(op);
    }
    return absl::NotFoundError("Op not found in op to node id map.");
  }
  return absl::InvalidArgumentError("Unsupported value ref type.");
}

std::string Color::toHexString() const {
  return absl::StrFormat("#%02x%02x%02x", r_, g_, b_);
}

Color Color::jitter() const {
  // Generate random offsets for r, g, and b channels.
  absl::BitGen bitgen;
  const int offset = 100;
  int offset_r = absl::Uniform(absl::IntervalClosed, bitgen, -offset, offset);
  int offset_g = absl::Uniform(absl::IntervalClosed, bitgen, -offset, offset);
  int offset_b = absl::Uniform(absl::IntervalClosed, bitgen, -offset, offset);

  // Add offsets to the channels (int channels promote uint8_t member variables
  // so overflow is avoided), and then clamp to the valid range.
  uint8_t new_r = static_cast<uint8_t>(std::clamp(r_ + offset_r, 0, 255));
  uint8_t new_g = static_cast<uint8_t>(std::clamp(g_ + offset_g, 0, 255));
  uint8_t new_b = static_cast<uint8_t>(std::clamp(b_ + offset_b, 0, 255));
  return Color(new_r, new_g, new_b);
}

const std::vector<Color>* ColorMapper::edge_colors_ = new std::vector<Color>(
    {Color(0xEA46C6), Color(0x5AE878), Color(0xDE6C1E), Color(0x56A9E5),
     Color(0x8A49E7), Color(0x4BE0CC), Color(0x4E6BE2), Color(0xE6C75E),
     Color(0xF1584A), Color(0xB4E554)});

absl::StatusOr<std::string> ColorMapper::getColor(mlir::sdy::AxisRefAttr axis) {
  if (sub_axis_color_map_.contains(axis)) {
    return sub_axis_color_map_.at(axis).toHexString();
  }

  if (axis_color_map_.contains(axis.getName())) {
    if (axis.getSubAxisInfo() != mlir::sdy::SubAxisInfoAttr()) {
      sub_axis_color_map_[axis] = axis_color_map_.at(axis.getName());
      return sub_axis_color_map_.at(axis).toHexString();
    }
    Color sub_axis_color = axis_color_map_.at(axis.getName()).jitter();
    sub_axis_color_map_[axis] = sub_axis_color;
    return sub_axis_color.toHexString();
  }

  if (next_color_index_ >= edge_colors_->size()) {
    return absl::OutOfRangeError(
        "Total number of axis exceeds the dedicated number of SDY edge overlay "
        "colors.");
  }

  Color next_color = edge_colors_->at(next_color_index_);
  next_color_index_++;
  axis_color_map_[axis.getName()] = next_color;
  if (axis.getSubAxisInfo() == mlir::sdy::SubAxisInfoAttr()) {
    sub_axis_color_map_[axis] = next_color;
  } else {
    Color jittered_subaxis_color = next_color.jitter();
    sub_axis_color_map_[axis] = jittered_subaxis_color;
    return jittered_subaxis_color.toHexString();
  }
  return next_color.toHexString();
}

}  // namespace visualization_client
}  // namespace tooling
