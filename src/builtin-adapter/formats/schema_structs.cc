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

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/Support/JSON.h"

namespace tooling {
namespace visualization_client {

namespace {

template <typename T>
void AddDataListToJson(const char* key,
                       const std::optional<std::vector<T>>& data_list,
                       llvm::json::Object& json_object) {
  if (!data_list.has_value()) {
    return;
  }
  llvm::json::Array json_array;
  for (const T& data : data_list.value()) {
    json_array.push_back(data.Json());
  }
  json_object[key] = std::move(json_array);
}

}  // namespace

const char Attribute::kKey[] = "key";
const char Attribute::kValue[] = "value";

llvm::json::Object Attribute::Json() const {
  llvm::json::Object json_attr;
  json_attr[kKey] = key;
  json_attr[kValue] = value;
  return json_attr;
}

const char Metadata::kId[] = "id";
const char Metadata::kAttrs[] = "attrs";

llvm::json::Object Metadata::Json() const {
  llvm::json::Object json_metadata;
  json_metadata[kId] = id;
  json_metadata[kAttrs] = llvm::json::Array();
  llvm::json::Array* json_attrs = json_metadata[kAttrs].getAsArray();
  for (const Attribute& attr : attrs) {
    json_attrs->push_back(attr.Json());
  }
  return json_metadata;
}

const char GraphEdge::kSourceNodeId[] = "sourceNodeId";
const char GraphEdge::kSourceNodeOutputId[] = "sourceNodeOutputId";
const char GraphEdge::kTargetNodeInputId[] = "targetNodeInputId";
const char GraphEdge::kEdgeMetadata[] = "edgeMetadata";

llvm::json::Object GraphEdge::Json() const {
  llvm::json::Object json_edge;
  json_edge[kSourceNodeId] = source_node_id;
  json_edge[kSourceNodeOutputId] = source_node_output_id;
  json_edge[kTargetNodeInputId] = target_node_input_id;
  json_edge[kEdgeMetadata] = llvm::json::Array();
  llvm::json::Array* json_attrs = json_edge[kEdgeMetadata].getAsArray();
  for (const Attribute& attr : edge_metadata) {
    json_attrs->push_back(attr.Json());
  }
  return json_edge;
}

const char GraphNodeConfig::kPinToGroupTop[] = "pinToGroupTop";

llvm::json::Object GraphNodeConfig::Json() const {
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

const char Edge::kSourceNodeId[] = "sourceNodeId";
const char Edge::kTargetNodeId[] = "targetNodeId";
const char Edge::kLabel[] = "label";
llvm::json::Object Edge::Json() const {
  llvm::json::Object json_edge;
  json_edge[kSourceNodeId] = source_node_id;
  json_edge[kTargetNodeId] = target_node_id;
  if (label.has_value()) {
    json_edge[kLabel] = label.value();
  }
  return json_edge;
}

const char EdgeOverlay::kName[] = "name";
const char EdgeOverlay::kEdges[] = "edges";
const char EdgeOverlay::kEdgeColor[] = "edgeColor";
const char EdgeOverlay::kEdgeWidth[] = "edgeWidth";
const char EdgeOverlay::kEdgeLabelFontSize[] = "edgeLabelFontSize";
llvm::json::Object EdgeOverlay::Json() const {
  llvm::json::Object json_overlay;
  json_overlay[kName] = name;
  json_overlay[kEdges] = llvm::json::Array();
  llvm::json::Array* json_edges = json_overlay[kEdges].getAsArray();
  for (const Edge& edge : edges) {
    json_edges->push_back(edge.Json());
  }
  json_overlay[kEdgeColor] = edge_color;
  if (edge_width.has_value()) {
    json_overlay[kEdgeWidth] = edge_width.value();
  }
  if (edge_label_font_size.has_value()) {
    json_overlay[kEdgeLabelFontSize] = edge_label_font_size.value();
  }
  return json_overlay;
}

const char EdgeOverlaysData::kType[] = "type";
const char EdgeOverlaysData::kName[] = "name";
const char EdgeOverlaysData::kOverlays[] = "overlays";
llvm::json::Object EdgeOverlaysData::Json() const {
  llvm::json::Object json_edge_overlay;
  json_edge_overlay[kType] = type;
  json_edge_overlay[kName] = name;
  json_edge_overlay[kOverlays] = llvm::json::Array();
  llvm::json::Array* json_overlays = json_edge_overlay[kOverlays].getAsArray();
  for (const EdgeOverlay& overlay : overlays) {
    json_overlays->push_back(overlay.Json());
  }
  return json_edge_overlay;
}

const char TasksData::kEdgeOverlaysDataListLeftPane[] =
    "edgeOverlaysDataListLeftPane";
const char TasksData::kEdgeOverlaysDataListRightPane[] =
    "edgeOverlaysDataListRightPane";
llvm::json::Object TasksData::Json() const {
  llvm::json::Object json_tasks_data;
  AddDataListToJson<EdgeOverlaysData>(kEdgeOverlaysDataListLeftPane,
                                      edge_overlays_data_list_left_pane,
                                      json_tasks_data);
  AddDataListToJson<EdgeOverlaysData>(kEdgeOverlaysDataListRightPane,
                                      edge_overlays_data_list_right_pane,
                                      json_tasks_data);
  return json_tasks_data;
}

llvm::json::Object GraphNode::Json() const {
  llvm::json::Object json_node;
  json_node[kNodeId] = node_id;
  json_node[kNodeLabel] = node_label;
  json_node[kNodeName] = node_name;
  json_node[kSubgraphIds] = subgraph_ids;

  json_node[kNodeAttrs] = llvm::json::Array();
  llvm::json::Array* json_attrs = json_node[kNodeAttrs].getAsArray();
  for (const Attribute& attr : node_attrs) {
    json_attrs->push_back(attr.Json());
  }

  json_node[kIncomingEdges] = llvm::json::Array();
  llvm::json::Array* json_edges = json_node[kIncomingEdges].getAsArray();
  for (const GraphEdge& edge : incoming_edges) {
    json_edges->push_back(edge.Json());
  }

  json_node[kInputsMetadata] = llvm::json::Array();
  llvm::json::Array* json_inputs_metadata =
      json_node[kInputsMetadata].getAsArray();
  for (const Metadata& metadata : inputs_metadata) {
    json_inputs_metadata->push_back(metadata.Json());
  }

  json_node[kOutputsMetadata] = llvm::json::Array();
  llvm::json::Array* json_outputs_metadata =
      json_node[kOutputsMetadata].getAsArray();
  for (const Metadata& metadata : outputs_metadata) {
    json_outputs_metadata->push_back(metadata.Json());
  }

  if (config.has_value()) {  // Only add config if it exists
    json_node[kConfig] = config->Json();
  }

  return json_node;
}

const char Subgraph::kSubgraphId[] = "id";
const char Subgraph::kNodes[] = "nodes";
const char Subgraph::kTasksData[] = "tasksData";

llvm::json::Object Subgraph::Json() const {
  llvm::json::Object json_subgraph;
  json_subgraph[kSubgraphId] = subgraph_id;
  json_subgraph[kNodes] = llvm::json::Array();
  llvm::json::Array* json_nodes = json_subgraph[kNodes].getAsArray();
  for (const GraphNode& node : nodes) {
    json_nodes->push_back(node.Json());
  }
  if (tasks_data.has_value()) {
    json_subgraph[kTasksData] = tasks_data->Json();
  }
  return json_subgraph;
}

const char Graph::kLabel[] = "label";
const char Graph::kSubgraphs[] = "subgraphs";

llvm::json::Object Graph::Json() const {
  llvm::json::Object json_graph;
  json_graph[kLabel] = label;
  json_graph[kSubgraphs] = llvm::json::Array();
  llvm::json::Array* json_subgraphs = json_graph[kSubgraphs].getAsArray();
  for (const Subgraph& subgraph : subgraphs) {
    json_subgraphs->push_back(subgraph.Json());
  }
  return json_graph;
}

const char GraphCollection::kGraphs[] = "graphs";

llvm::json::Array GraphCollection::Json() const {
  llvm::json::Array json_graphs;
  for (const Graph& graph : graphs) {
    json_graphs.push_back(graph.Json());
  }
  return json_graphs;
}

}  // namespace visualization_client
}  // namespace tooling
