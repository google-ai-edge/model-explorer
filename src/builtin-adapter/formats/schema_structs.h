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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_FORMATS_SCHEMA_STRUCTS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_FORMATS_SCHEMA_STRUCTS_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/Support/JSON.h"

namespace tooling {
namespace visualization_client {

struct Attribute {
  Attribute(std::string key, std::string value)
      : key(std::move(key)), value(std::move(value)) {}
  std::string key;
  std::string value;

  llvm::json::Object Json();

 private:
  static const char kKey[];
  static const char kValue[];
};

struct Metadata {
  std::string id;
  std::vector<Attribute> attrs;

  llvm::json::Object Json();

 private:
  static const char kId[];
  static const char kAttrs[];
};

struct GraphEdge {
  std::string source_node_id;
  std::string source_node_output_id;
  std::string target_node_input_id;
  std::vector<Attribute> edge_metadata;

  llvm::json::Object Json();

 private:
  static const char kSourceNodeId[];
  static const char kSourceNodeOutputId[];
  static const char kTargetNodeInputId[];
  static const char kEdgeMetadata[];
};

// Configuration for a graph node.
struct GraphNodeConfig {
  // Whether to pin the node to the top of the group it belongs to.
  bool pin_to_group_top = false;

  llvm::json::Object Json();

 private:
  static const char kPinToGroupTop[];
};

struct GraphNode {
  std::string node_id;
  std::string node_label;
  std::string node_name;
  std::vector<std::string> subgraph_ids;
  std::vector<Attribute> node_attrs;
  std::vector<GraphEdge> incoming_edges;
  std::vector<Metadata> inputs_metadata;
  std::vector<Metadata> outputs_metadata;
  std::optional<GraphNodeConfig> config;

  llvm::json::Object Json();

 private:
  static const char kNodeId[];
  static const char kNodeLabel[];
  static const char kNodeName[];
  static const char kSubgraphIds[];
  static const char kNodeAttrs[];
  static const char kIncomingEdges[];
  static const char kInputsMetadata[];
  static const char kOutputsMetadata[];
  static const char kConfig[];
};

struct Subgraph {
  explicit Subgraph(std::string subgraph_id)
      : subgraph_id(std::move(subgraph_id)) {}
  std::string subgraph_id;
  std::vector<GraphNode> nodes;

  llvm::json::Object Json();

 private:
  static const char kSubgraphId[];
  static const char kNodes[];
};

struct Graph {
  std::string label;
  std::vector<Subgraph> subgraphs;

  llvm::json::Object Json();

 private:
  static const char kLabel[];
  static const char kSubgraphs[];
};

struct GraphCollection {
  std::vector<Graph> graphs;

  llvm::json::Array Json();

 private:
  static const char kGraphs[];
};

}  // namespace visualization_client
}  // namespace tooling

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_GOOGLE_TOOLING_FORMATS_SCHEMA_STRUCTS_H_
