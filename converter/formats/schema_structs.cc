#include "formats/schema_structs.h"

#include <string>

#include "llvm/Support/JSON.h"

namespace tooling {
namespace visualization_client {

const char Attribute::kKey[] = "key";
const char Attribute::kValue[] = "value";

llvm::json::Object Attribute::Json() {
  llvm::json::Object json_attr;
  json_attr[kKey] = key;
  json_attr[kValue] = value;
  return json_attr;
}

const char Metadata::kId[] = "id";
const char Metadata::kAttrs[] = "attrs";

llvm::json::Object Metadata::Json() {
  llvm::json::Object json_metadata;
  json_metadata[kId] = id;
  json_metadata[kAttrs] = llvm::json::Array();
  llvm::json::Array* json_attrs = json_metadata[kAttrs].getAsArray();
  for (auto attr : attrs) {
    json_attrs->push_back(attr.Json());
  }
  return json_metadata;
}

const char GraphEdge::kSourceNodeId[] = "sourceNodeId";
const char GraphEdge::kSourceNodeOutputId[] = "sourceNodeOutputId";
const char GraphEdge::kTargetNodeInputId[] = "targetNodeInputId";
const char GraphEdge::kEdgeMetadata[] = "edgeMetadata";

llvm::json::Object GraphEdge::Json() {
  llvm::json::Object json_edge;
  json_edge[kSourceNodeId] = source_node_id;
  json_edge[kSourceNodeOutputId] = source_node_output_id;
  json_edge[kTargetNodeInputId] = target_node_input_id;
  json_edge[kEdgeMetadata] = llvm::json::Array();
  llvm::json::Array* json_attrs = json_edge[kEdgeMetadata].getAsArray();
  for (auto attr : edge_metadata) {
    json_attrs->push_back(attr.Json());
  }
  return json_edge;
}

const char GraphNode::kNodeId[] = "id";
const char GraphNode::kNodeLabel[] = "label";
const char GraphNode::kNodeName[] = "namespace";
const char GraphNode::kSubgraphIds[] = "subgraphIds";
const char GraphNode::kNodeAttrs[] = "attrs";
const char GraphNode::kIncomingEdges[] = "incomingEdges";
const char GraphNode::kInputsMetadata[] = "inputsMetadata";
const char GraphNode::kOutputsMetadata[] = "outputsMetadata";

llvm::json::Object GraphNode::Json() {
  llvm::json::Object json_node;
  json_node[kNodeId] = node_id;
  json_node[kNodeLabel] = node_label;
  json_node[kNodeName] = node_name;
  json_node[kSubgraphIds] = subgraph_ids;

  json_node[kNodeAttrs] = llvm::json::Array();
  llvm::json::Array* json_attrs = json_node[kNodeAttrs].getAsArray();
  for (auto attr : node_attrs) {
    json_attrs->push_back(attr.Json());
  }

  json_node[kIncomingEdges] = llvm::json::Array();
  llvm::json::Array* json_edges = json_node[kIncomingEdges].getAsArray();
  for (auto edge : incoming_edges) {
    json_edges->push_back(edge.Json());
  }

  json_node[kInputsMetadata] = llvm::json::Array();
  llvm::json::Array* json_inputs_metadata =
      json_node[kInputsMetadata].getAsArray();
  for (auto attr : inputs_metadata) {
    json_inputs_metadata->push_back(attr.Json());
  }

  json_node[kOutputsMetadata] = llvm::json::Array();
  llvm::json::Array* json_outputs_metadata =
      json_node[kOutputsMetadata].getAsArray();
  for (auto attr : outputs_metadata) {
    json_outputs_metadata->push_back(attr.Json());
  }
  return json_node;
}

const char Subgraph::kSubgraphId[] = "id";
const char Subgraph::kNodes[] = "nodes";

llvm::json::Object Subgraph::Json() {
  llvm::json::Object json_subgraph;
  json_subgraph[kSubgraphId] = subgraph_id;
  json_subgraph[kNodes] = llvm::json::Array();
  llvm::json::Array* json_nodes = json_subgraph[kNodes].getAsArray();
  for (auto node : nodes) {
    json_nodes->push_back(node.Json());
  }
  return json_subgraph;
}

const char Graph::kSubgraphs[] = "subgraphs";

llvm::json::Array Graph::Json() {
  llvm::json::Array json_graphs;
  for (auto subgraph : subgraphs) {
    json_graphs.push_back(subgraph.Json());
  }
  return json_graphs;
}

}  // namespace visualization_client
}  // namespace tooling
