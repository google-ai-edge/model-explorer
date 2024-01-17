#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_GRAPHNODE_BUILDER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_GRAPHNODE_BUILDER_H_

#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "formats/schema_structs.h"

namespace tooling {
namespace visualization_client {

enum class EdgeType { kInput, kOutput };

class GraphNodeBuilder {
 public:
  GraphNodeBuilder() = default;

  void SetNodeId(absl::string_view node_id_str);

  void SetNodeLabel(absl::string_view node_label);

  void SetNodeName(absl::string_view node_name);

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

  // Appends the attribute to the input or output metadata list. The assumption
  // of the list is that the metadata ids are in arithmetic sequence which
  // starts from 0 and increments by 1. If the metadata already exists, we
  // append the attribute to that metadata. If it doesn't exist, we create a new
  // metadata and add it to the list, then append the attribute to the metadata.
  // @param  edge_type:  Distinguishes whether this is input or output metadata.
  // @param  metadata_id:  The index of the metadata in the list.
  // @param  attr_key:  The key of the attribute to be appended.
  // @param  attr_value:  The value of the attribute to be appended.
  absl::Status AppendAttrToMetadata(EdgeType edge_type, int metadata_id,
                                    absl::string_view attr_key,
                                    absl::string_view attr_value);

  // Returns the node that has been created by this class.
  GraphNode Build() && { return std::move(node_); }

 private:
  GraphNode node_;
};

}  // namespace visualization_client
}  // namespace tooling

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_GRAPHNODE_BUILDER_H_
