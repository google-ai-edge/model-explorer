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

#include "direct_saved_model_to_json_graph_convert.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "formats/schema_structs.h"
#include "graphnode_builder.h"
#include "status_macros.h"
#include "visualize_config.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tsl/platform/env.h"

namespace tooling {
namespace visualization_client {
namespace {

using ::tensorflow::protobuf::RepeatedPtrField;

constexpr char kPathSeparator = '/';
constexpr absl::string_view kGraphInputs = "GraphInputs";
constexpr absl::string_view kGraphOutputs = "GraphOutputs";
constexpr absl::string_view kTensorName = "tensor_name";
constexpr absl::string_view kTensorShape = "tensor_shape";
constexpr absl::string_view kControlInput = "control_input";
constexpr absl::string_view kConfigProto = "config_proto";
constexpr absl::string_view kValue = "__value";

struct EdgeInfo {
  std::string source_node_id;
  std::string source_node_output_id;
};

absl::Status ReadGraphDef(absl::string_view model_path,
                          tensorflow::GraphDef& graph_def) {
  if (!ReadBinaryProto(tsl::Env::Default(), std::string(model_path), &graph_def)
           .ok()) {
    return ReadTextProto(tsl::Env::Default(), std::string(model_path),
                         &graph_def);
  }
  return absl::OkStatus();
}

// Skip serializing attributes that match the given name. This is a hard-code to
// avoid adding binary string attribute into JSON. This disallow list might
// expand as more unsupported attribute found.
inline bool SkipAttr(absl::string_view name) { return name == kConfigProto; }

// Returns true if the string is printable.
bool IsPrintable(absl::string_view str) {
  for (const char& c : str) {
    unsigned char uc = static_cast<unsigned char>(c);
    if ((uc < 0x20) || (uc > 0x7E)) {
      return false;
    }
  }
  return true;
}

std::string FormatJson(const llvm::json::Value& value) {
  return llvm::formatv("{0:2}", value);
}

std::string FormatJsonPrimitive(const llvm::json::Value& v) {
  return std::string(llvm::formatv("{0}", v));
}

template <typename T>
std::string FormatRepeatedPrimitive(
    const tensorflow::protobuf::RepeatedField<T>& field) {
  return FormatJsonPrimitive(llvm::json::Array(field));
}

std::string StringifyDataType(const tensorflow::DataType data_type) {
  return tensorflow::DataTypeString(data_type);
}

std::string StringifyListOfDataTypes(
    const tensorflow::AttrValue::ListValue& list_value) {
  llvm::json::Array jsonifid_list;
  for (int i = 0; i < list_value.type_size(); ++i) {
    jsonifid_list.push_back(StringifyDataType(list_value.type(i)));
  }
  return FormatJsonPrimitive(llvm::json::Array(jsonifid_list));
}

llvm::json::Value JsonifyTensorDim(
    const tensorflow::TensorShapeProto::Dim& dim) {
  if (dim.name().empty()) {
    // It is more straightforward to simply note the size. Don't create object.
    return dim.size();
  }

  // Creating an object is necessary: We must inform the name of the dimension.
  return llvm::json::Object{{"name", dim.name()}, {"size", dim.size()}};
}

llvm::json::Value JsonifyTensorShape(
    const tensorflow::TensorShapeProto& shape) {
  if (shape.unknown_rank()) {
    return "unknown rank";
  }

  llvm::json::Array dims;
  for (const tensorflow::TensorShapeProto::Dim& dim : shape.dim()) {
    dims.push_back(JsonifyTensorDim(dim));
  }
  return llvm::json::Array(dims);
}

absl::StatusOr<std::string> StringifyTensor(const tensorflow::Tensor& tensor,
                                            const VisualizeConfig& config) {
  const std::string tensor_string =
      tensor.SummarizeValue(config.const_element_count_limit);
  // Special handling for invalid string value tensor.
  if (tensor.dtype() == tensorflow::DataType::DT_STRING) {
    if (!IsPrintable(tensor_string)) {
      return "<const bytes>";
    }
  }
  return tensor_string;
}

absl::Status AddTensorInfo(absl::string_view attr_name,
                           const tensorflow::TensorProto& tensor_proto,
                           const VisualizeConfig& config,
                           const int relative_idx, GraphNodeBuilder& builder) {
  tensorflow::Tensor tensor;
  if (!tensor.FromProto(tensor_proto)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to parse tensor stored in attribute %s", attr_name));
  }
  ASSIGN_OR_RETURN(const std::string tensor_string,
                   StringifyTensor(tensor, config));
  std::string attr_key = (relative_idx == 0)
                             ? std::string(kValue)
                             : absl::StrCat(kValue, "_", relative_idx);
  builder.AppendNodeAttribute(attr_key, tensor_string);
  const std::string data_type = tensorflow::DataTypeString(tensor.dtype());
  const std::string shape =
      absl::StrCat(data_type, tensor.shape().DebugString());
  builder.AppendAttrToMetadata(EdgeType::kOutput, relative_idx, kTensorShape,
                               shape);
  return absl::OkStatus();
}

std::string StringifyTensorShape(const tensorflow::TensorShapeProto& shape) {
  if (shape.unknown_rank()) {
    // Repeat this check here (in addition to the branch in the
    // `JsonifyTensorShape` method). Otherwise, the JSON logic applies another
    // level of double quotes.
    return "unknown rank";
  }
  return FormatJsonPrimitive(JsonifyTensorShape(shape));
}

std::string StringifyListOfTensorShapes(
    const RepeatedPtrField<tensorflow::TensorShapeProto>& shapes) {
  llvm::json::Array jsonified;
  for (const tensorflow::TensorShapeProto& shape : shapes) {
    jsonified.push_back(JsonifyTensorShape(shape));
  }
  return FormatJsonPrimitive(llvm::json::Array(jsonified));
}

absl::Status AddListOfTensorInfo(
    absl::string_view attr_name,
    const RepeatedPtrField<tensorflow::TensorProto>& tensors,
    const VisualizeConfig& config, GraphNodeBuilder& builder) {
  for (int i = 0; i < tensors.size(); ++i) {
    const tensorflow::TensorProto& tensor = tensors[i];
    RETURN_IF_ERROR(AddTensorInfo(attr_name, tensor, config, i, builder));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> GetFunctionName(
    absl::string_view attr_name, const tensorflow::NameAttrList& func,
    const absl::flat_hash_set<std::string>& functions_set) {
  if (func.name().empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Attribute %s lacks a function name.", attr_name));
  }

  if (!functions_set.contains(func.name())) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Attribute %s references non-existent function %s",
                        attr_name, func.name()));
  }

  return func.name();
}

absl::Status AddFunctionAttribute(
    absl::string_view attr_name, const tensorflow::NameAttrList& func,
    const absl::flat_hash_set<std::string>& functions_set,
    GraphNodeBuilder& builder) {
  ASSIGN_OR_RETURN(const std::string function_name,
                   GetFunctionName(attr_name, func, functions_set));
  builder.AppendNodeAttribute(attr_name, function_name);
  builder.AppendSubgraphId(function_name);
  return absl::OkStatus();
}

absl::Status ProcessListOfFunctionsAttribute(
    const std::string& attr_name,
    const absl::flat_hash_set<std::string>& functions_set,
    const RepeatedPtrField<tensorflow::NameAttrList>& funcs,
    GraphNodeBuilder& builder) {
  llvm::json::Array jsonified;
  for (const tensorflow::NameAttrList& func : funcs) {
    ASSIGN_OR_RETURN(const std::string function_name,
                     GetFunctionName(attr_name, func, functions_set));
    jsonified.push_back(function_name);
    builder.AppendSubgraphId(function_name);
  }
  builder.AppendNodeAttribute(
      attr_name, FormatJsonPrimitive(llvm::json::Array(jsonified)));
  return absl::OkStatus();
}

absl::Status AddListAttribute(
    const absl::string_view attr_str_view,
    const absl::flat_hash_set<std::string>& functions_set,
    const VisualizeConfig& config, const tensorflow::AttrValue::ListValue& list,
    GraphNodeBuilder& builder) {
  const std::string attr_name = std::string(attr_str_view);
  if (!list.s().empty()) {
    builder.AppendNodeAttribute(
        attr_name, FormatJsonPrimitive(llvm::json::Array(list.s())));
  } else if (!list.i().empty()) {
    builder.AppendNodeAttribute(attr_name, FormatRepeatedPrimitive(list.i()));
  } else if (!list.f().empty()) {
    builder.AppendNodeAttribute(attr_name, FormatRepeatedPrimitive(list.f()));
  } else if (!list.b().empty()) {
    builder.AppendNodeAttribute(attr_name, FormatRepeatedPrimitive(list.b()));
  } else if (!list.type().empty()) {
    builder.AppendNodeAttribute(attr_name, StringifyListOfDataTypes(list));
  } else if (!list.shape().empty()) {
    builder.AppendNodeAttribute(attr_name,
                                StringifyListOfTensorShapes(list.shape()));
  } else if (!list.tensor().empty()) {
    RETURN_IF_ERROR(
        AddListOfTensorInfo(attr_name, list.tensor(), config, builder));
  } else if (!list.func().empty()) {
    RETURN_IF_ERROR(ProcessListOfFunctionsAttribute(attr_name, functions_set,
                                                    list.func(), builder));
  }
  return absl::OkStatus();
}

absl::Status AddAttributeInformation(
    const tensorflow::NodeDef& node_def, const VisualizeConfig& config,
    absl::string_view immediate_name,
    const absl::flat_hash_set<std::string>& functions_set,
    GraphNodeBuilder& builder) {
  builder.AppendNodeAttribute("op_immediate_name", immediate_name);
  if (!node_def.device().empty()) {
    builder.AppendNodeAttribute("device", node_def.device());
  }
  for (auto it = node_def.attr().begin(); it != node_def.attr().end(); ++it) {
    absl::string_view attr_name = it->first;
    const tensorflow::AttrValue& attr_value = it->second;
    if (SkipAttr(attr_name)) continue;
    if (attr_value.has_s()) {
      builder.AppendNodeAttribute(attr_name, attr_value.s());
    } else if (attr_value.has_i()) {
      builder.AppendNodeAttribute(attr_name, absl::StrCat(attr_value.i()));
    } else if (attr_value.has_f()) {
      builder.AppendNodeAttribute(attr_name,
                                  FormatJsonPrimitive(attr_value.f()));
    } else if (attr_value.has_b()) {
      builder.AppendNodeAttribute(attr_name,
                                  FormatJsonPrimitive(attr_value.b()));
    } else if (attr_value.has_type()) {
      builder.AppendNodeAttribute(attr_name,
                                  StringifyDataType(attr_value.type()));
    } else if (attr_value.has_shape()) {
      builder.AppendNodeAttribute(attr_name,
                                  StringifyTensorShape(attr_value.shape()));
    } else if (attr_value.has_tensor()) {
      RETURN_IF_ERROR(AddTensorInfo(attr_name, attr_value.tensor(), config,
                                    /*relative_idx=*/0, builder));
    } else if (attr_value.has_list()) {
      RETURN_IF_ERROR(AddListAttribute(attr_name, functions_set, config,
                                       attr_value.list(), builder));
    } else if (attr_value.has_func()) {
      RETURN_IF_ERROR(AddFunctionAttribute(attr_name, attr_value.func(),
                                           functions_set, builder));
    }
  }

  return absl::OkStatus();
}

// Parses the immediate name of the node (the last part) from the full path.
absl::StatusOr<std::string> ParseImmediateNodeName(absl::string_view path) {
  const std::vector<std::string> path_parts =
      absl::StrSplit(path, kPathSeparator);
  if (path_parts.empty()) {
    return absl::InvalidArgumentError(
        "The path of the NodeDef cannot be empty.");
  }
  absl::string_view name = path_parts[path_parts.size() - 1];
  if (name.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Could not determine name of node with path %s", path));
  }
  return std::string(name);
}

std::string MaybeAdjustForControlInput(absl::string_view source_node_name,
                                       GraphNodeBuilder& builder) {
  if (absl::StartsWith(source_node_name, "^")) {
    // Adds a node attribute for the control input.
    const std::string control_input = std::string(source_node_name).substr(1);
    builder.AppendNodeAttribute(kControlInput, control_input);
    // Per GraphDef proto docs, this is a control input. For now, we don't
    // differentiate between control and normal inputs: Remove the first char.
    return control_input;
  }
  return std::string(source_node_name);
}

absl::StatusOr<EdgeInfo> ComputeEdgeInfoFromInput(
    absl::string_view input,
    const absl::flat_hash_map<std::string, int>& input_to_idx,
    const absl::flat_hash_map<std::string, std::string>& node_to_id,
    GraphNodeBuilder& builder) {
  EdgeInfo edge_info;
  std::string source_node_name;
  std::vector<std::string> input_parts = absl::StrSplit(input, ':');
  if (input_parts.size() == 1) {
    // No `source_node_output_id` specified. Just use the whole path as the name
    // of the node.
    source_node_name = MaybeAdjustForControlInput(input, builder);

    // When no `source_node_output_id` is specified, the default output is 0.
    edge_info.source_node_output_id = "0";
  } else {
    // When `input` looks like `RestoreV2/shape_and_slices:output:0`, we assign
    // the part before the first colon to `source_node_name` and the last part
    // to `edgeinfo.source_node_output_id`.
    source_node_name = MaybeAdjustForControlInput(input_parts.front(), builder);
    edge_info.source_node_output_id = input_parts.back();
  }

  // Populates `source_node_id` to EdgeInfo. If the name doesn't come from a
  // node, then it has to come from the artificial node "GraphInputs".
  auto source_id_it = node_to_id.find(source_node_name);
  if (source_id_it != node_to_id.end()) {
    edge_info.source_node_id = absl::StrCat(source_id_it->second);
  } else {
    auto output_id_it = input_to_idx.find(source_node_name);
    if (output_id_it != input_to_idx.end()) {
      // `source_node_id` for "GraphInputs" node is 0 by default.
      edge_info.source_node_id = "0";
      edge_info.source_node_output_id = absl::StrCat(output_id_it->second);
    } else {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Could not find source node name \"%s\"", source_node_name));
    }
  }
  return edge_info;
}

void AddGraphInputsNode(
    const tensorflow::OpDef& signature,
    absl::flat_hash_map<std::string, int>& input_to_idx,
    absl::flat_hash_map<std::string, std::string>& node_to_id,
    size_t* next_node_id, Subgraph& subgraph) {
  const std::string node_id = absl::StrCat((*next_node_id)++);
  GraphNodeBuilder builder;
  builder.SetNodeInfo(node_id, /*node_label=*/kGraphInputs,
                      /*node_name=*/kGraphInputs);
  node_to_id.emplace(kGraphInputs, node_id);
  for (int i = 0; i < signature.input_arg().size(); ++i) {
    const tensorflow::OpDef::ArgDef& arg_def = signature.input_arg(i);
    // TODO: b/322649392 - Add tensor index and tensor shape.
    input_to_idx.emplace(arg_def.name(), i);
    builder.AppendAttrToMetadata(EdgeType::kOutput, i, kTensorName,
                                 arg_def.name());
  }
  subgraph.nodes.push_back(std::move(builder).Build());
}

absl::Status AddGraphOutputsNode(
    const tensorflow::FunctionDef& func_def,
    const absl::flat_hash_map<std::string, int>& input_to_idx,
    absl::flat_hash_map<std::string, std::string>& node_to_id,
    size_t* next_node_id, Subgraph& subgraph) {
  const tensorflow::OpDef& signature = func_def.signature();
  const std::string node_id = absl::StrCat((*next_node_id)++);
  GraphNodeBuilder builder;
  builder.SetNodeInfo(node_id, /*node_label=*/kGraphOutputs,
                      /*node_name=*/kGraphOutputs);
  node_to_id.emplace(kGraphOutputs, node_id);
  for (int i = 0; i < signature.output_arg().size(); ++i) {
    std::string output_arg_name = signature.output_arg()[i].name();
    auto node_name_it = func_def.ret().find(output_arg_name);
    if (node_name_it == func_def.ret().end()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Could not find output arg name \"%s\"", output_arg_name));
    }
    ASSIGN_OR_RETURN(
        const EdgeInfo edge_info,
        ComputeEdgeInfoFromInput(node_name_it->second, input_to_idx, node_to_id,
                                 builder));
    builder.AppendEdgeInfo(edge_info.source_node_id,
                           edge_info.source_node_output_id,
                           /*target_node_input_id_str=*/absl::StrCat(i));
  }
  subgraph.nodes.push_back(std::move(builder).Build());
  return absl::OkStatus();
}

absl::Status AddSubgraph(
    absl::string_view subgraph_name, const VisualizeConfig& config,
    const RepeatedPtrField<tensorflow::NodeDef>& node_defs,
    const absl::flat_hash_set<std::string>& functions_set,
    const std::optional<const tensorflow::FunctionDef>& func_def,
    Graph& graph) {
  Subgraph subgraph((std::string(subgraph_name)));

  // A map from the graph input name to its relative index.
  absl::flat_hash_map<std::string, int> input_to_idx;
  // A map from the graph node name to its node id.
  absl::flat_hash_map<std::string, std::string> node_to_id;
  size_t next_node_id = 0;
  if (func_def.has_value()) {
    AddGraphInputsNode(func_def->signature(), input_to_idx, node_to_id,
                       &next_node_id, subgraph);
  }
  // Compute integer IDs for all nodes.
  for (const tensorflow::NodeDef& node_def : node_defs) {
    node_to_id.emplace(node_def.name(), absl::StrCat(next_node_id++));
  }

  // Create nodes and add edges.
  for (const tensorflow::NodeDef& node_def : node_defs) {
    ASSIGN_OR_RETURN(const std::string immediate_name,
                     ParseImmediateNodeName(node_def.name()));
    // Make and add the node.
    GraphNodeBuilder builder;
    // Label with the op or the name of the node if the op is not set.
    const std::string node_label =
        node_def.op().empty() ? immediate_name : node_def.op();
    builder.SetNodeInfo(node_to_id[node_def.name()], node_label,
                        node_def.name());
    RETURN_IF_ERROR(AddAttributeInformation(node_def, config, immediate_name,
                                            functions_set, builder));

    // Add edges entering that node.
    for (size_t j = 0; j < node_def.input_size(); ++j) {
      const std::string& input = node_def.input(j);
      // TODO: b/322649392 - Add tensor index and tensor shape.
      ASSIGN_OR_RETURN(
          const EdgeInfo edge_info,
          ComputeEdgeInfoFromInput(input, input_to_idx, node_to_id, builder));

      // The input index represents the relative rank of the given input
      // compared to other inputs. It does NOT represent the output index of the
      // previous node.
      builder.AppendEdgeInfo(edge_info.source_node_id,
                             edge_info.source_node_output_id,
                             /*target_node_input_id_str=*/absl::StrCat(j));
    }

    subgraph.nodes.push_back(std::move(builder).Build());
  }

  if (func_def.has_value()) {
    RETURN_IF_ERROR(AddGraphOutputsNode(*func_def, input_to_idx, node_to_id,
                                        &next_node_id, subgraph));
  }

  graph.subgraphs.push_back(std::move(subgraph));
  return absl::OkStatus();
}

absl::StatusOr<absl::flat_hash_set<std::string>> GetNamesOfAllFunctions(
    const tensorflow::FunctionDefLibrary& library) {
  absl::flat_hash_set<std::string> names;
  for (const tensorflow::FunctionDef& function_def : library.function()) {
    if (function_def.signature().name().empty()) {
      return absl::InvalidArgumentError(
          "Every TensorFlow function must have a name.");
    }
    names.insert(function_def.signature().name());
  }
  return names;
}

absl::Status AddSubgraphsForFunctions(
    const VisualizeConfig& config,
    const tensorflow::FunctionDefLibrary& library,
    const absl::flat_hash_set<std::string>& functions_set, Graph& graph) {
  for (const tensorflow::FunctionDef& function_def : library.function()) {
    RETURN_IF_ERROR(AddSubgraph(function_def.signature().name(), config,
                                function_def.node_def(), functions_set,
                                function_def, graph));
  }
  return absl::OkStatus();
}

// A comma-separated tag string that's used to identify a MetaGraphDef.
std::string GetTagSetString(const tensorflow::MetaGraphDef& meta_graph) {
  const tensorflow::MetaGraphDef::MetaInfoDef& meta_info =
      meta_graph.meta_info_def();
  return absl::StrJoin(meta_info.tags(), ",");
}

absl::Status AddAllSubgraphsFromGraphDef(const VisualizeConfig& config,
                                         const tensorflow::GraphDef& graph_def,
                                         Graph& graph) {
  ASSIGN_OR_RETURN(const absl::flat_hash_set<std::string> functions_set,
                   GetNamesOfAllFunctions(graph_def.library()));

  RETURN_IF_ERROR(AddSubgraph("main", config, graph_def.node(), functions_set,
                              /*func_def=*/std::nullopt, graph));

  RETURN_IF_ERROR(AddSubgraphsForFunctions(config, graph_def.library(),
                                           functions_set, graph));

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::string> ConvertGraphDefDirectlyToJsonImpl(
    const VisualizeConfig& config, const tensorflow::GraphDef& graph_def) {
  Graph graph;
  RETURN_IF_ERROR(AddAllSubgraphsFromGraphDef(config, graph_def, graph));

  GraphCollection collection;
  collection.graphs.push_back(std::move(graph));
  llvm::json::Value json_result(collection.Json());
  return FormatJson(json_result);
}

absl::StatusOr<std::string> ConvertGraphDefDirectlyToJson(
    const VisualizeConfig& config, absl::string_view model_path) {
  tensorflow::GraphDef graph_def;
  RETURN_IF_ERROR(ReadGraphDef(model_path, graph_def));

  return ConvertGraphDefDirectlyToJsonImpl(config, graph_def);
}

absl::StatusOr<std::string> ConvertSavedModelDirectlyToJson(
    const VisualizeConfig& config, absl::string_view model_path) {
  tensorflow::SavedModel saved_model;
  RETURN_IF_ERROR(tensorflow::ReadSavedModel(model_path, &saved_model));

  GraphCollection collection;
  for (const tensorflow::MetaGraphDef& meta_graph : saved_model.meta_graphs()) {
    Graph graph;
    graph.label = GetTagSetString(meta_graph);
    RETURN_IF_ERROR(
        AddAllSubgraphsFromGraphDef(config, meta_graph.graph_def(), graph));
    collection.graphs.push_back(std::move(graph));
  }

  llvm::json::Value json_result(collection.Json());
  return FormatJson(json_result);
}

}  // namespace visualization_client
}  // namespace tooling
