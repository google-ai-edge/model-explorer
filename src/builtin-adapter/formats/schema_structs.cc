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
  for (Attribute& attr : attrs) {
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
  for (Attribute& attr : edge_metadata) {
    json_attrs->push_back(attr.Json());
  }
  return json_edge;
}

const char GraphNodeConfig::kPinToGroupTop[] = "pinToGroupTop";

llvm::json::Object GraphNodeConfig::Json() {
  llvm::json::Object json_config;
  json_config[kPinToGroupTop] = pin_to_group_top;
  return json_config;
}

const char GraphNode::kNodeId[] = "id";
const char GraphNode::kNodeLabel[] = "label";
const char GraphNode::kNodeName[] = "namespace";
const char GraphNode::kSubgraphIds[] = "subgraphIds";
const char GraphNode::kNodeAttrs[] = "attrs";
const char GraphNode::kIncomingEdges[] = "incomingEdges";
const char GraphNode::kInputsMetadata[] = "inputsMetadata";
const char GraphNode::kOutputsMetadata[] = "outputsMetadata";
const char GraphNode::kConfig[] = "config";

llvm::json::Object GraphNode::Json() {
  llvm::json::Object json_node;
  json_node[kNodeId] = node_id;
  json_node[kNodeLabel] = node_label;
  json_node[kNodeName] = node_name;
  json_node[kSubgraphIds] = subgraph_ids;

  json_node[kNodeAttrs] = llvm::json::Array();
  llvm::json::Array* json_attrs = json_node[kNodeAttrs].getAsArray();
  for (Attribute& attr : node_attrs) {
    json_attrs->push_back(attr.Json());
  }

  json_node[kIncomingEdges] = llvm::json::Array();
  llvm::json::Array* json_edges = json_node[kIncomingEdges].getAsArray();
  for (GraphEdge& edge : incoming_edges) {
    json_edges->push_back(edge.Json());
  }

  json_node[kInputsMetadata] = llvm::json::Array();
  llvm::json::Array* json_inputs_metadata =
      json_node[kInputsMetadata].getAsArray();
  for (Metadata& metadata : inputs_metadata) {
    json_inputs_metadata->push_back(metadata.Json());
  }

  json_node[kOutputsMetadata] = llvm::json::Array();
  llvm::json::Array* json_outputs_metadata =
      json_node[kOutputsMetadata].getAsArray();
  for (Metadata& metadata : outputs_metadata) {
    json_outputs_metadata->push_back(metadata.Json());
  }

  if (config.has_value()) {  // Only add config if it exists
    json_node[kConfig] = config->Json();
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
  for (GraphNode& node : nodes) {
    json_nodes->push_back(node.Json());
  }
  return json_subgraph;
}

const char Graph::kLabel[] = "label";
const char Graph::kSubgraphs[] = "subgraphs";

llvm::json::Object Graph::Json() {
  llvm::json::Object json_graph;
  json_graph[kLabel] = label;
  json_graph[kSubgraphs] = llvm::json::Array();
  llvm::json::Array* json_subgraphs = json_graph[kSubgraphs].getAsArray();
  for (Subgraph& subgraph : subgraphs) {
    json_subgraphs->push_back(subgraph.Json());
  }
  return json_graph;
}

const char GraphCollection::kGraphs[] = "graphs";

llvm::json::Array GraphCollection::Json() {
  llvm::json::Array json_graphs;
  for (Graph& graph : graphs) {
    json_graphs.push_back(graph.Json());
  }
  return json_graphs;
}

}  // namespace visualization_client
}  // namespace tooling
