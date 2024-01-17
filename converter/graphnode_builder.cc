#include "converter/graphnode_builder.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "converter/formats/schema_structs.h"
#include "converter/status_macros.h"

namespace tooling {
namespace visualization_client {
namespace {

absl::Status AppendAttrToMetadataImpl(const int metadata_id,
                                      absl::string_view attr_key,
                                      absl::string_view attr_value,
                                      std::vector<Metadata>& metadata_list) {
  Attribute attr((std::string(attr_key)), std::string(attr_value));
  if (metadata_id < metadata_list.size()) {
    metadata_list.at(metadata_id).attrs.push_back(attr);
  } else if (metadata_id == metadata_list.size()) {
    Metadata metadata;
    metadata.id = absl::StrCat(metadata_id);
    metadata.attrs.push_back(attr);
    metadata_list.push_back(metadata);
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "metadata id ", metadata_id, " is larger than metadata list size ",
        metadata_list.size()));
  }
  return absl::OkStatus();
}

}  // namespace

void GraphNodeBuilder::SetNodeId(absl::string_view node_id_str) {
  node_.node_id = node_id_str;
}

void GraphNodeBuilder::SetNodeLabel(absl::string_view node_label) {
  node_.node_label = node_label;
}

void GraphNodeBuilder::SetNodeName(absl::string_view node_name) {
  node_.node_name = node_name;
}

void GraphNodeBuilder::SetNodeInfo(absl::string_view node_id_str,
                                   absl::string_view node_label,
                                   absl::string_view node_name) {
  node_.node_id = node_id_str;
  node_.node_label = node_label;
  node_.node_name = node_name;
}

void GraphNodeBuilder::AppendEdgeInfo(
    absl::string_view source_node_id_str,
    absl::string_view source_node_output_id_str,
    absl::string_view target_node_input_id_str) {
  GraphEdge edge;
  edge.source_node_id = source_node_id_str;
  edge.source_node_output_id = source_node_output_id_str;
  edge.target_node_input_id = target_node_input_id_str;
  node_.incoming_edges.push_back(edge);
}

void GraphNodeBuilder::AppendSubgraphId(absl::string_view subgraph_id_str) {
  node_.subgraph_ids.push_back(std::string(subgraph_id_str));
}

void GraphNodeBuilder::AppendNodeAttribute(absl::string_view key,
                                           absl::string_view value) {
  node_.node_attrs.push_back(Attribute(std::string(key), std::string(value)));
}

absl::Status GraphNodeBuilder::AppendAttrToMetadata(
    const EdgeType edge_type, const int metadata_id, absl::string_view attr_key,
    absl::string_view attr_value) {
  switch (edge_type) {
    case EdgeType::kInput: {
      RETURN_IF_ERROR(AppendAttrToMetadataImpl(
          metadata_id, attr_key, attr_value, node_.inputs_metadata));
      break;
    }
    case EdgeType::kOutput: {
      RETURN_IF_ERROR(AppendAttrToMetadataImpl(
          metadata_id, attr_key, attr_value, node_.outputs_metadata));
      break;
    }
    default:
      return absl::InvalidArgumentError("Unknown edge type.");
  }
  return absl::OkStatus();
}

}  // namespace visualization_client
}  // namespace tooling
