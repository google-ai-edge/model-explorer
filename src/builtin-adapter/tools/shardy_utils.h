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

#ifndef TOOLS_SHARDY_UTILS_H_
#define TOOLS_SHARDY_UTILS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "formats/schema_structs.h"

namespace tooling {
namespace visualization_client {

// Returns true if the attribute is from the Shardy dialect.
inline bool IsShardyDialect(mlir::Attribute attr) {
  return llvm::isa<mlir::sdy::SdyDialect>(attr.getDialect());
}

// Prints Shardy attributes to stream (using simplified pretty printing for
// select Shardy attributes).
void PrintShardyAttribute(mlir::Attribute attr, llvm::raw_string_ostream& os);

// Adds referenced meshes from Shardy attributes to the set. This is used to
// collect all referenced meshes for a given op.
void AddReferencedMesh(
    mlir::Attribute attr,
    llvm::SmallDenseMap<llvm::StringRef, mlir::sdy::MeshAttr>& sdy_meshes,
    mlir::Operation& operation);

// Extracts all possible SDY propagation edges from the given operation and adds
// them to the `edges` vector. If none are present, returns false and does not
// modify the `edges` vector.
bool HasShardyPropagationEdges(
    mlir::Operation& operation,
    llvm::SmallVector<mlir::sdy::PropagationEdgesAttr>& edges);

// Returns the Model Explorer node ID for the operation referenced by the given
// EdgeValueRefAttr.
absl::StatusOr<std::string> GetNodeIdFromEdgeValueRef(
    mlir::Operation* op, mlir::sdy::EdgeValueRefAttr edge_value_ref,
    const absl::flat_hash_map<mlir::Operation*, std::string>& op_to_id,
    const llvm::SmallDenseMap<mlir::Value, std::string>& input_nodes);

// Extracts Model Explorer edge overlays from Shardy propagation edge attributes
// if they are present.
absl::StatusOr<tooling::visualization_client::EdgeOverlaysData>
ExtractShardyPropagationEdges(
    mlir::Operation* root,
    const absl::flat_hash_map<mlir::Operation*, std::string>& op_to_id,
    const llvm::SmallDenseMap<mlir::Value, std::string>& input_nodes);

class Color {
 public:
  Color() = default;
  Color(uint8_t r, uint8_t g, uint8_t b) : r_(r), g_(g), b_(b) {}
  explicit Color(uint32_t rgb)
      : r_(rgb >> 16 & 0xFF), g_(rgb >> 8 & 0xFF), b_(rgb & 0xFF) {}

  std::string ToHexString() const;
  Color Jitter() const;

 private:
  // The maximum per-channel RGB offset to use when jittering the color.
  static constexpr int kColorJitterOffset = 100;
  uint8_t r_, g_, b_;
};

// Class for setting and retrieving colors for Shardy propagation edge overlays.
class ColorMapper {
 public:
  ColorMapper() = default;
  // Returns the assigned color for the given axis. If the axis is not found in
  // the color map, a new color is assigned and returned.
  absl::StatusOr<std::string> GetColor(mlir::sdy::AxisRefAttr axis);

 private:
  static const std::array<Color, 10> kEdgeColors;
  llvm::DenseMap<mlir::sdy::AxisRefAttr, Color> sub_axis_color_map_;
  llvm::DenseMap<llvm::StringRef, Color> axis_color_map_;
  size_t next_color_index_ = 0;
};

// Builder class for easing EdgeOverlay creation.
class EdgeOverlayBuilder {
 public:
  EdgeOverlayBuilder() = default;

  EdgeOverlay BuildEdgeOverlay() { return edge_overlay_; }

  EdgeOverlayBuilder& WithName(std::string name) {
    edge_overlay_.name = std::move(name);
    return *this;
  }

  EdgeOverlayBuilder& WithEdges(std::vector<Edge> edges) {
    edge_overlay_.edges = std::move(edges);
    return *this;
  }

  EdgeOverlayBuilder& WithEdgeColor(std::string edge_color) {
    edge_overlay_.edge_color = std::move(edge_color);
    return *this;
  }

  EdgeOverlayBuilder& WithEdgeWidth(std::optional<float> edge_width) {
    edge_overlay_.edge_width = edge_width;
    return *this;
  }

  EdgeOverlayBuilder& WithEdgeLabelFontSize(
      std::optional<float> edge_label_font_size) {
    edge_overlay_.edge_label_font_size = edge_label_font_size;
    return *this;
  }

  EdgeOverlayBuilder& WithShowEdgesConnectedToSelectedNodeOnly(
      bool show_edges_connected_to_selected_node_only) {
    edge_overlay_.show_edges_connected_to_selected_node_only =
        show_edges_connected_to_selected_node_only;
    return *this;
  }

  EdgeOverlayBuilder& AddEdge(std::string source_node_id,
                              std::string target_node_id,
                              std::optional<std::string> label = std::nullopt) {
    edge_overlay_.edges.emplace_back(
        Edge{.source_node_id = std::move(source_node_id),
             .target_node_id = std::move(target_node_id),
             .label = label});
    return *this;
  }

 private:
  EdgeOverlay edge_overlay_;
};

}  // namespace visualization_client
}  // namespace tooling

#endif  // TOOLS_SHARDY_UTILS_H_
