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
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
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
#include "mlir/Support/WalkResult.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "formats/schema_structs.h"
#include "status_macros.h"

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
  os << strippedAttrString(attr, /*stripMnemonic=*/true);
};

// Prints a simplified version of the sharding attribute, which is easier
// for users to parse. For example, it displays a sharding attr like
// #sdy.sharding<@mesh, [{}, {"c"}]> as <@mesh, [{}, {"c"}]>
void PrettyPrint(mlir::sdy::TensorShardingAttr attr, llvm::raw_ostream& os) {
  os << strippedAttrString(attr, /*stripMnemonic=*/true);
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
      os << '\t' << strippedAttrString(axis_entry, /*stripMnemonic=*/true);
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

// Prints a simplified version of AxisRefList, which is easier for users to
// parse. For example it displays the default printing of
// #sdy<axis_ref_list{z}> as {"z"}
void PrettyPrint(mlir::sdy::AxisRefListAttr attr, llvm::raw_ostream& os) {
  os << strippedAttrString(attr, /*stripMnemonic=*/true);
}

// Prints a simplified version of ListOfAxisRefLists, which is easier for users
// to parse. For example it displays the default printing of
// #sdy<list_of_axis_ref_lists[{}, {x}]> as [{}, {x}]
void PrettyPrint(mlir::sdy::ListOfAxisRefListsAttr attr,
                 llvm::raw_ostream& os) {
  os << strippedAttrString(attr, /*stripMnemonic=*/true);
}

// Prints a simplified version of AllToAllParamList, which is easier for users
// to parse. For example it displays the default printing of
// #sdy<all_to_all_param_list[{x}: 0->2, {y}: 1->3]> as
// [{"x"}: 0->2, {"y"}: 1->3]
void PrettyPrint(mlir::sdy::AllToAllParamListAttr attr, llvm::raw_ostream& os) {
  os << strippedAttrString(attr, /*stripMnemonic=*/true);
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
      .Case<mlir::sdy::AxisRefListAttr>(
          [&os](mlir::sdy::AxisRefListAttr axis_ref_list_attr) {
            PrettyPrint(axis_ref_list_attr, os);
          })
      .Case<mlir::sdy::ListOfAxisRefListsAttr>(
          [&os](mlir::sdy::ListOfAxisRefListsAttr list_of_axis_ref_lists_attr) {
            PrettyPrint(list_of_axis_ref_lists_attr, os);
          })
      .Case<mlir::sdy::AllToAllParamListAttr>(
          [&os](mlir::sdy::AllToAllParamListAttr all_to_all_param_list_attr) {
            PrettyPrint(all_to_all_param_list_attr, os);
          })
      .Default(
          [&](mlir::Attribute attr) { attr.print(os, /*elideType=*/true); });
}

void AddReferencedMesh(
    mlir::Attribute attr,
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
  bool has_edges = false;
  auto propagation_edges =
      operation.getAttrOfType<mlir::sdy::PropagationEdgesAttr>(
          kPropagationEdgesAttr);
  if (propagation_edges) {
    edges.push_back(propagation_edges);
    has_edges = true;
  }

  for (const llvm::StringRef& edge_array_attr :
       {kBlockArgPropagationEdgesAttr, kResultPropagationEdgesAttr}) {
    if (auto array_of_prop_edges =
            operation.getAttrOfType<mlir::ArrayAttr>(edge_array_attr)) {
      for (const auto& prop_edges : array_of_prop_edges) {
        edges.push_back(
            llvm::dyn_cast<mlir::sdy::PropagationEdgesAttr>(prop_edges));
        has_edges = true;
      }
    }
  }

  return has_edges;
}

absl::StatusOr<std::string> GetNodeIdFromEdgeValueRef(
    Operation* op, mlir::sdy::EdgeValueRefAttr edge_value_ref,
    const OpToNodeIdMap& op_to_id, const InputValueToNodeIdMap& input_nodes) {
  if (edge_value_ref.getType() == mlir::sdy::EdgeNodeType::OPERAND) {
    if (edge_value_ref.getIndex() >= op->getNumOperands()) {
      return absl::OutOfRangeError(
          "Index reference in the Sdy EdgeValueRefAttr is larger than number "
          "of operands for the operation.");
    }

    // If the operand is a block argument, we should resolve to the
    // corresponding node in the graph inputs.
    mlir::Value operand = op->getOperand(edge_value_ref.getIndex());
    if (llvm::isa<mlir::BlockArgument>(operand)) {
      if (auto it = input_nodes.find(operand); it != input_nodes.end()) {
        return it->second;
      }
      return absl::NotFoundError("Value not found in input block args map.");
    }

    // Otherwise try to resolve to the defining op and return its node id.
    Operation* defining_op = operand.getDefiningOp();
    if (auto it = op_to_id.find(defining_op); it != op_to_id.end()) {
      return it->second;
    }
    return absl::NotFoundError("Op not found in op to node id map.");
  } else if (edge_value_ref.getType() == mlir::sdy::EdgeNodeType::RESULT) {
    // When a vertex of a propagation edge is of type RESULT, then we should
    // resolve to the op itself.
    if (auto it = op_to_id.find(op); it != op_to_id.end()) {
      return it->second;
    }
    return absl::NotFoundError("Op not found in op to node id map.");
  }
  return absl::InvalidArgumentError("Unsupported value ref type.");
}

void CollectEdgesForOperation(
    Operation* op, const OpToNodeIdMap& op_to_id,
    const InputValueToNodeIdMap& input_nodes,
    llvm::DenseMap<mlir::sdy::AxisRefAttr, std::vector<Edge>>& axis_to_edges) {
  llvm::SmallVector<mlir::sdy::PropagationEdgesAttr> all_edges;
  if (!HasShardyPropagationEdges(*op, all_edges)) {
    return;
  }
  for (const mlir::sdy::PropagationEdgesAttr& edges : all_edges) {
    for (const mlir::sdy::PropagationOneStepAttr& propagation_step : edges) {
      const int64_t step_index = propagation_step.getStepIndex();
      for (const mlir::sdy::AxisToPropagationDetailsAttr& axis_to_details :
           propagation_step.getAxisEntries()) {
        mlir::sdy::AxisRefAttr axis = axis_to_details.getAxisName();
        mlir::sdy::EdgeValueRefAttr source = axis_to_details.getSource();
        for (const mlir::sdy::EdgeValueRefAttr& target :
             axis_to_details.getTargets()) {
          absl::StatusOr<std::string> source_op_id =
              GetNodeIdFromEdgeValueRef(op, source, op_to_id, input_nodes);
          absl::StatusOr<std::string> target_op_id =
              GetNodeIdFromEdgeValueRef(op, target, op_to_id, input_nodes);

          if (!source_op_id.ok()) {
            LOG(ERROR) << "Failed to get source node id: "
                       << source_op_id.status();
            continue;
          }
          if (!target_op_id.ok()) {
            LOG(ERROR) << "Failed to get target node id: "
                       << target_op_id.status();
            continue;
          }
          axis_to_edges[axis].emplace_back(Edge{
              .source_node_id = *source_op_id,
              .target_node_id = *target_op_id,
              .label = absl::StrFormat("%d: %s", step_index, axis.toString())});
        }
      }
    }
  }
}

absl::StatusOr<EdgeOverlaysData> ExtractShardyPropagationEdges(
    Operation* root, const OpToNodeIdMap& op_to_id,
    const InputValueToNodeIdMap& input_nodes) {
  llvm::DenseMap<mlir::sdy::AxisRefAttr, std::vector<Edge>> axis_to_edges;

  root->walk([&](Operation* op) -> mlir::WalkResult {
    CollectEdgesForOperation(op, op_to_id, input_nodes, axis_to_edges);
    return mlir::WalkResult::advance();
  });

  // TODO(varcho): update color mapper to map from a pair <MeshAttr, AxisRef> to
  // color instead of just AxisRef since axis names can be the same across
  // different meshes.
  ColorMapper color_mapper;
  EdgeOverlaysData layer;

  // Sort the axes to ensure a deterministic order of the overlays.
  std::vector<mlir::sdy::AxisRefAttr> sorted_axes;
  for (const auto& [axis, edges] : axis_to_edges) {
    sorted_axes.push_back(axis);
  }
  std::sort(sorted_axes.begin(), sorted_axes.end(),
            [](mlir::sdy::AxisRefAttr a, mlir::sdy::AxisRefAttr b) {
              return a.toString() < b.toString();
            });

  for (const auto& axis : sorted_axes) {
    const auto& edges = axis_to_edges[axis];
    ASSIGN_OR_RETURN(const std::string color, color_mapper.GetColor(axis));
    EdgeOverlay overlay = EdgeOverlayBuilder()
                              .WithName(axis.toString())
                              .WithEdges(edges)
                              .WithEdgeColor(color)
                              .WithEdgeWidth(3.0)
                              .WithEdgeLabelFontSize(7.0)
                              .WithShowEdgesConnectedToSelectedNodeOnly(true)
                              .BuildEdgeOverlay();
    layer.overlays.push_back(overlay);
  }

  return layer;
}

const std::array<Color, 10> ColorMapper::kEdgeColors = {
    Color(0xEA46C6),  // Pink
    Color(0x5AE878),  // Light Green
    Color(0xDE6C1E),  // Orange
    Color(0x56A9E5),  // Light Blue
    Color(0x8A49E7),  // Purple
    Color(0x4BE0CC),  // Cyan
    Color(0x4E6BE2),  // Dark Blue
    Color(0xE6C75E),  // Gold
    Color(0xF1584A),  // Salmon
    Color(0xB4E554)   // Lime Green
};

std::string Color::ToHexString() const {
  return absl::StrFormat("#%02x%02x%02x", r_, g_, b_);
}

Color Color::Jitter() const {
  // Generate random offsets for r, g, and b channels.
  absl::BitGen bitgen;
  const int offset = kColorJitterOffset;
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

absl::StatusOr<std::string> ColorMapper::GetColor(mlir::sdy::AxisRefAttr axis) {
  if (auto it = sub_axis_color_map_.find(axis);
      it != sub_axis_color_map_.end()) {
    return it->second.ToHexString();
  }

  auto axis_it = axis_color_map_.find(axis.getName());
  if (axis_it != axis_color_map_.end()) {
    const Color& base_color = axis_it->second;
    if (axis.getSubAxisInfo() != mlir::sdy::SubAxisInfoAttr()) {
      Color jittered_color = base_color.Jitter();
      sub_axis_color_map_[axis] = jittered_color;
      return jittered_color.ToHexString();
    }
    sub_axis_color_map_[axis] = base_color;
    return base_color.ToHexString();
  }

  if (next_color_index_ >= kEdgeColors.size()) {
    return absl::OutOfRangeError(
        "Total number of axis exceeds the dedicated number of SDY edge overlay "
        "colors.");
  }

  Color next_color = kEdgeColors.at(next_color_index_);
  next_color_index_++;
  axis_color_map_[axis.getName()] = next_color;
  if (axis.getSubAxisInfo() == mlir::sdy::SubAxisInfoAttr()) {
    sub_axis_color_map_[axis] = next_color;
  } else {
    Color jittered_subaxis_color = next_color.Jitter();
    sub_axis_color_map_[axis] = jittered_subaxis_color;
    return jittered_subaxis_color.ToHexString();
  }
  return next_color.ToHexString();
}

}  // namespace visualization_client
}  // namespace tooling
