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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_GRAPHNODE_BUILDER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_GRAPHNODE_BUILDER_H_

#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "formats/schema_structs.h"

namespace tooling {
namespace visualization_client {

enum class EdgeType { kInput, kOutput };

class GraphNodeBuilder {
 public:
  GraphNodeBuilder() = default;

  void SetNodeId(absl::string_view node_id_str);

  std::string GetNodeId();

  void SetNodeLabel(absl::string_view node_label);

  std::string GetNodeLabel();

  void SetNodeName(absl::string_view node_name);

  std::string GetNodeName();

  // Sets the node id, label and name of GraphNode.
  void SetNodeInfo(absl::string_view node_id_str, absl::string_view node_label,
                   absl::string_view node_name);

  // Populates the edge info to GraphEdge and appends the edge to
  // `incoming_edges` in GraphNode.
  // @param  source_node_id_str:  The id of the source node that outputs this
  // edge.
  // @param  source_node_output_id_str:  The index of this edge in the source
  //         node outputs.
  // @param  target_node_input_id_str:  The index of this edge in the target
  //         node inputs.
  void AppendEdgeInfo(absl::string_view source_node_id_str,
                      absl::string_view source_node_output_id_str,
                      absl::string_view target_node_input_id_str);

  // Appends the subgraph id to the `subgraph_ids` in GraphNode.
  void AppendSubgraphId(absl::string_view subgraph_id_str);

  // Appends the attribute key and value to `node_attrs` in GraphNode.
  void AppendNodeAttribute(absl::string_view key, absl::string_view value);

  // Appends the attribute to the input or output metadata list. If the metadata
  // already exists, we append the attribute to that metadata. If it doesn't
  // exist, we create a new metadata and add it to the list, then append the
  // attribute to the metadata.
  // @param  edge_type:  Distinguishes whether this is input or output metadata.
  // @param  metadata_id:  The index of the metadata in the list.
  // @param  attr_key:  The key of the attribute to be appended.
  // @param  attr_value:  The value of the attribute to be appended.
  void AppendAttrToMetadata(EdgeType edge_type, int metadata_id,
                            absl::string_view attr_key,
                            absl::string_view attr_value);

  // Sets the node to be pinned to the top of the group it belongs to.
  // User needs to ensure the node's namespace is indeed a layer for the pinning
  // to work.
  void SetPinToGroupTop(bool pin_to_group_top);

  // Returns the node that has been created by this class.
  GraphNode Build() && { return std::move(node_); }

 private:
  GraphNode node_;
};

}  // namespace visualization_client
}  // namespace tooling

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_GRAPHNODE_BUILDER_H_
