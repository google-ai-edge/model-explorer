/* Copyright 2024 The Model Explorer Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "direct_flatbuffer_to_json_graph_convert.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/flexbuffers.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#include "formats/schema_structs.h"
#include "graphnode_builder.h"
#include "status_macros.h"
#include "tools/attribute_printer.h"
#include "tools/convert_type.h"
#include "tools/load_opdefs.h"
#include "tools/namespace_heuristics.h"
#include "visualize_config.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_operator.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/offset_buffer.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/const_tensor_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tsl/platform/env.h"

namespace tooling {
namespace visualization_client {
namespace {

using ::tflite::BuiltinOperator;
using ::tflite::FlatBufferModel;
using ::tflite::ModelT;
using ::tflite::OperatorCodeT;
using ::tflite::OperatorT;
using ::tflite::SubGraphT;
using ::tflite::TensorT;
using ::tooling::visualization_client::OpMetadata;

constexpr absl::string_view kGraphInputs = "GraphInputs";
constexpr absl::string_view kGraphOutputs = "GraphOutputs";
constexpr absl::string_view kPseudoConst = "pseudo_const";
constexpr absl::string_view kTensorIndex = "tensor_index";
constexpr absl::string_view kTensorName = "tensor_name";
constexpr absl::string_view kTensorShape = "tensor_shape";
constexpr absl::string_view kTensorTag = "__tensor_tag";
constexpr absl::string_view kValue = "__value";
constexpr absl::string_view kQuantization = "quantization";
constexpr absl::string_view kSignatureName = "signature_name";

struct EdgeInfo {
  std::string source_node_id = "";
  std::string source_node_output_id = "";
  std::string target_node_input_id = "";
};

enum NodeType {
  kInputNode,
  kOutputNode,
  kConstNode,
};

// A map from tensor index to EdgeInfo.
using EdgeMap = absl::flat_hash_map<int, EdgeInfo>;
// Maps from the tensor index to input/output signature name.
using SignatureNameMap = absl::flat_hash_map<int, std::string>;
// Maps from the subgraph index to its SignatureNameMap.
using SignatureMap = absl::flat_hash_map<int, SignatureNameMap>;
using Tensors = std::vector<std::unique_ptr<TensorT>>;
using OperatorCodes = std::vector<std::unique_ptr<OperatorCodeT>>;
using Buffers = std::vector<std::unique_ptr<tflite::BufferT>>;
// Maps from op name to op metadata.
using OpdefsMap = absl::flat_hash_map<std::string, OpMetadata>;

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

std::string EdgeInfoDebugString(const EdgeInfo& edge_info) {
  return absl::StrCat("sourceNodeId: ", edge_info.source_node_id,
                      ", sourceNodeOutputId: ", edge_info.source_node_output_id,
                      ", targetNodeInputId: ", edge_info.target_node_input_id);
}

std::string GetOpNameFromOpCode(const OperatorCodeT& op_code) {
  BuiltinOperator builtin_code = tflite::GetBuiltinCode(&op_code);
  return absl::AsciiStrToLower(tflite::EnumNameBuiltinOperator(builtin_code));
}

// Returns a list of op names from the given op codes.
// The index in the list should guarantee to match opcode_index in Operator.
std::vector<std::string> GetOpNames(const OperatorCodes& op_codes) {
  std::vector<std::string> op_names;
  op_names.reserve(op_codes.size());
  for (auto& op : op_codes) {
    op_names.push_back(GetOpNameFromOpCode(*op));
  }
  return op_names;
}

// Populates edge info to EdgeMap or inserts the edge info if it doesn't exist
// in EdgeMap.
void PopulateEdgeInfo(const int tensor_index, const EdgeInfo& edge_info,
                      EdgeMap& edge_map) {
  if (edge_map.contains(tensor_index)) {
    if (!edge_info.source_node_id.empty()) {
      edge_map.at(tensor_index).source_node_id = edge_info.source_node_id;
    }
    if (!edge_info.source_node_output_id.empty()) {
      edge_map.at(tensor_index).source_node_output_id =
          edge_info.source_node_output_id;
    }
    if (!edge_info.target_node_input_id.empty()) {
      edge_map.at(tensor_index).target_node_input_id =
          edge_info.target_node_input_id;
    }
  } else {
    edge_map.emplace(tensor_index, edge_info);
  }
}

bool EdgeInfoIncomplete(const EdgeInfo& edge_info) {
  return edge_info.source_node_id.empty() ||
         edge_info.source_node_output_id.empty() ||
         edge_info.target_node_input_id.empty();
}

void AppendIncomingEdge(const EdgeInfo& edge_info, GraphNodeBuilder& builder) {
  builder.AppendEdgeInfo(edge_info.source_node_id,
                         edge_info.source_node_output_id,
                         edge_info.target_node_input_id);
}

// Returns a string representation of the tensor shape, eg. "float32[3,2,5]".
// Unknown dimensions are represented with -1.
std::string StringifyTensorShape(const TensorT& tensor) {
  std::string shape_str;
  if (!tensor.shape_signature.empty()) {
    shape_str = absl::StrJoin(tensor.shape_signature, ",");
  } else {
    shape_str = absl::StrJoin(tensor.shape, ",");
  }
  if (shape_str.empty()) {
    return TensorTypeToString(tensor.type);
  }
  return absl::StrCat(TensorTypeToString(tensor.type), "[", shape_str, "]");
}

// Obtains the node namespace based on the node label and related tensor names.
std::string GenerateNodeName(absl::string_view node_label,
                             const std::vector<int>& tensor_indices,
                             const Tensors& tensors) {
  if (tensor_indices.empty()) return "";
  std::vector<std::string> candidate_names;
  for (const int index : tensor_indices) {
    // Skips the optional inputs which are indicated by -1.
    if (index < 0) {
      continue;
    }
    std::string tensor_name = tensors[index]->name;
    std::vector<std::string> tmp_names =
        absl::StrSplit(tensor_name, ';', absl::SkipEmpty());
    for (absl::string_view name : tmp_names) {
      candidate_names.push_back(std::string(name));
    }
  }
  return TfliteNodeNamespaceHeuristic(node_label, candidate_names);
}

void AppendMetadata(
    const EdgeType edge_type, const int metadata_id, const int tensor_index,
    const Tensors& tensors,
    const std::optional<const SignatureNameMap>& signature_name_map,
    GraphNodeBuilder& builder) {
  // Appends tensor index.
  builder.AppendAttrToMetadata(edge_type, metadata_id, kTensorIndex,
                               absl::StrCat(tensor_index));
  // Appends tensor name.
  builder.AppendAttrToMetadata(edge_type, metadata_id, kTensorName,
                               tensors[tensor_index]->name);
  // Appends tensor shape.
  builder.AppendAttrToMetadata(edge_type, metadata_id, kTensorShape,
                               StringifyTensorShape(*tensors[tensor_index]));
  // Appends tensor signature name.
  if (signature_name_map.has_value()) {
    auto name_it = signature_name_map->find(tensor_index);
    if (name_it != signature_name_map->end()) {
      builder.AppendAttrToMetadata(edge_type, metadata_id, kSignatureName,
                                   name_it->second);
    }
  }
}

// Converts the buffer data to an ElementsAttr. Logic is referred from
// `BuildConstOp` in tensorflow/compiler/mlir/lite/flatbuffer_import.cc.
absl::StatusOr<mlir::ElementsAttr> ConvertBufferToAttr(
    const TensorT& tensor, const std::vector<uint8_t>& buffer,
    mlir::Builder mlir_builder) {
  // TODO: b/319741948 - Support buffer for SparseConstOp and VariableOp
  if (tensor.sparsity != nullptr) {
    return absl::UnimplementedError("Sparse const op is not supported yet.");
  }

  if (tensor.is_variable) {
    return absl::UnimplementedError("Variable op is not supported yet.");
  }

  ASSIGN_OR_RETURN(mlir::TensorType type,
                   mlir::TFL::GetTensorType(tensor, mlir_builder,
                                            /*is_constant=*/true,
                                            /*is_intermediate=*/false,
                                            /*get_storage=*/true));
  const auto shaped_type = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (shaped_type == nullptr) {
    return absl::InternalError("Constant doesn't have a shape");
  }

  mlir::ElementsAttr value;
  if (mlir::TFL::IsQuantized(tensor)) {
    const bool truncate =
        shaped_type.getElementType().getIntOrFloatBitWidth() == 64;
    ASSIGN_OR_RETURN(
        value, mlir::TFL::ConvertIntBuffer(shaped_type, buffer, truncate));
    return value;
  }

  const mlir::Type elem_type = shaped_type.getElementType();
  if (const auto float_type = mlir::dyn_cast<mlir::FloatType>(elem_type)) {
    ASSIGN_OR_RETURN(value, mlir::TFL::ConvertFloatBuffer(shaped_type, buffer));
  } else if (mlir::isa<mlir::IntegerType>(elem_type)) {
    ASSIGN_OR_RETURN(value, mlir::TFL::ConvertIntBuffer(shaped_type, buffer));
  } else if (mlir::isa<mlir::TF::StringType>(elem_type)) {
    tensorflow::TensorProto repr =
        mlir::TFL::ConvertTfliteConstTensor(tensor, buffer);
    std::vector<llvm::StringRef> refs;
    refs.reserve(repr.string_val_size());

    for (absl::string_view ref : repr.string_val()) {
      refs.push_back({ref.data(), ref.size()});
    }

    value = mlir::DenseStringElementsAttr::get(shaped_type, refs);
  } else {
    return absl::UnimplementedError("Constant of unsupported type.");
  }

  return value;
}

absl::StatusOr<std::vector<uint8_t>> GetBuffer(
    const TensorT& tensor, const Buffers& buffers,
    const std::unique_ptr<FlatBufferModel>& model_ptr) {
  const uint64_t buffer_offset = buffers[tensor.buffer]->offset;
  const uint64_t buffer_size = buffers[tensor.buffer]->size;
  // Check if constant tensor is stored outside of the flatbuffers.
  if (tflite::IsValidBufferOffset(buffer_offset)) {
    if (!buffers[tensor.buffer]->data.empty()) {
      return absl::InvalidArgumentError(
          "Buffer data and offset cannot be set at the same time.");
    }
    const uint8_t* file_begin_ptr =
        reinterpret_cast<const uint8_t*>(model_ptr->allocation()->base());
    if (buffer_offset + buffer_size > model_ptr->allocation()->bytes()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Constant buffer of tensor \"", tensor.name,
                       "\" specified an out of range offset."));
    }
    return std::vector<uint8_t>(file_begin_ptr + buffer_offset,
                                file_begin_ptr + buffer_offset + buffer_size);
  }
  return buffers[tensor.buffer]->data;
}

absl::Status AddConstantToNodeAttr(
    const TensorT& tensor, const Buffers& buffers,
    const int const_element_count_limit,
    const std::unique_ptr<FlatBufferModel>& model_ptr,
    mlir::Builder mlir_builder, GraphNodeBuilder& builder) {
  ASSIGN_OR_RETURN(std::vector<uint8_t> buffer,
                   GetBuffer(tensor, buffers, model_ptr));
  if (buffer.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Buffer data for tensor \"", tensor.name, "\" is empty."));
  }
  ASSIGN_OR_RETURN(mlir::ElementsAttr elem_attr,
                   ConvertBufferToAttr(tensor, buffer, mlir_builder));
  std::string value;
  llvm::raw_string_ostream sstream(value);
  PrintAttribute(elem_attr, const_element_count_limit, sstream);
  builder.AppendNodeAttribute(/*key=*/kValue, /*value=*/value);
  return absl::OkStatus();
}

// Creates and adds a GraphInputs, GraphOutputs or const node into Subgraph.
absl::Status AddAuxiliaryNode(
    const NodeType node_type, const std::vector<int>& tensor_indices,
    const Tensors& tensors, const Buffers& buffers,
    const std::optional<const SignatureNameMap>& signature_name_map,
    const std::unique_ptr<FlatBufferModel>& model_ptr,
    const int const_element_count_limit, std::vector<std::string>& node_ids,
    EdgeMap& edge_map, mlir::Builder mlir_builder, Subgraph& subgraph) {
  // Skips adding auxiliary node if `tensor_indices` is empty.
  // This is to avoid adding GraphInputs/GraphOutputs node when there is no
  // input/output tensor.
  if (tensor_indices.empty()) {
    return absl::OkStatus();
  }
  EdgeType edge_type;
  std::string node_label, node_name;
  const std::string node_id_str = absl::StrCat(node_ids.size());
  switch (node_type) {
    case NodeType::kInputNode: {
      edge_type = EdgeType::kOutput;
      node_label = kGraphInputs;
      node_name = kGraphInputs;
      break;
    }
    case NodeType::kOutputNode: {
      edge_type = EdgeType::kInput;
      node_label = kGraphOutputs;
      node_name = kGraphOutputs;
      break;
    }
    case NodeType::kConstNode: {
      edge_type = EdgeType::kOutput;
      node_label = kPseudoConst;
      node_name = GenerateNodeName(node_label, tensor_indices, tensors);
      break;
    }
    default: {
      return absl::InvalidArgumentError("Invalid node type.");
    }
  }
  node_ids.push_back(node_id_str);
  GraphNodeBuilder builder;
  builder.SetNodeInfo(node_id_str, node_label, node_name);

  if (node_type == NodeType::kConstNode) {
    const TensorT& tensor = *tensors[tensor_indices[0]];
    absl::Status status =
        AddConstantToNodeAttr(tensor, buffers, const_element_count_limit,
                              model_ptr, mlir_builder, builder);
    // Logs the error and continues to add the node to the graph.
    if (!status.ok()) {
      LOG(ERROR) << status.message();
    }
  }

  for (int i = 0; i < tensor_indices.size(); ++i) {
    const int tensor_index = tensor_indices[i];

    // Skips the optional inputs which are indicated by -1.
    if (tensor_index < 0) {
      continue;
    }

    if (edge_type == EdgeType::kInput) {
      PopulateEdgeInfo(tensor_index, {.target_node_input_id = absl::StrCat(i)},
                       edge_map);
      if (EdgeInfoIncomplete(edge_map.at(tensor_index))) {
        // Adds the const node to subgraph and populates the rest of the
        // tensor's edge info to edge_map. Since kConstNode links to
        // EdgeType::kOutput, it always goes to else branch and should never run
        // into infinite loop.
        RETURN_IF_ERROR(AddAuxiliaryNode(
            NodeType::kConstNode, std::vector<int>{tensor_index}, tensors,
            buffers, signature_name_map, model_ptr, const_element_count_limit,
            node_ids, edge_map, mlir_builder, subgraph));
      }
      AppendIncomingEdge(edge_map.at(tensor_index), builder);
    } else {
      AppendMetadata(EdgeType::kOutput, i, tensor_index, tensors,
                     signature_name_map, builder);

      PopulateEdgeInfo(tensor_index,
                       {.source_node_id = node_id_str,
                        .source_node_output_id = absl::StrCat(i)},
                       edge_map);
    }
  }
  subgraph.nodes.push_back(std::move(builder).Build());
  return absl::OkStatus();
}

// Logic referred from `CustomOptionsToAttributes` in
// tensorflow/compiler/mlir/lite/flatbuffer_operator.cc.
void CustomOptionsToAttributes(
    const std::vector<uint8_t>& custom_options, mlir::Builder mlir_builder,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes) {
  const flexbuffers::Map& map = flexbuffers::GetRoot(custom_options).AsMap();
  const flexbuffers::TypedVector& keys = map.Keys();
  for (size_t i = 0; i < keys.size(); ++i) {
    const char* key = keys[i].AsKey();
    const flexbuffers::Reference& value = map[key];
    attributes.emplace_back(mlir_builder.getNamedAttr(
        key, mlir_builder.getStringAttr(value.ToString())));
  }
}

absl::Status SubgraphIdxToAttributes(
    const tflite::OperatorT& op, const std::vector<std::string>& func_names,
    mlir::Builder mlir_builder,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes) {
  if (auto* opts = op.builtin_options.AsCallOnceOptions()) {
    const uint32_t init_idx = opts->init_subgraph_index;
    if (init_idx >= func_names.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("subgraph with index not found: ", init_idx));
    }
    const auto init_attr = mlir_builder.getStringAttr(func_names[init_idx]);
    attributes.emplace_back(
        mlir_builder.getNamedAttr("session_init_function", init_attr));
  } else if (auto* opts = op.builtin_options.AsIfOptions()) {
    const uint32_t then_idx = opts->then_subgraph_index;
    if (then_idx >= func_names.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("subgraph with index not found: ", then_idx));
    }
    const auto then_attr = mlir::SymbolRefAttr::get(mlir_builder.getContext(),
                                                    func_names[then_idx]);
    const uint32_t else_idx = opts->else_subgraph_index;
    if (else_idx >= func_names.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("subgraph with index not found: ", else_idx));
    }
    const auto else_attr = mlir::SymbolRefAttr::get(mlir_builder.getContext(),
                                                    func_names[else_idx]);
    attributes.emplace_back(
        mlir_builder.getNamedAttr("then_branch", then_attr));
    attributes.emplace_back(
        mlir_builder.getNamedAttr("else_branch", else_attr));
    attributes.emplace_back(mlir_builder.getNamedAttr(
        "is_stateless", mlir_builder.getBoolAttr(false)));
  } else if (auto* opts = op.builtin_options.AsWhileOptions()) {
    const uint32_t cond_idx = opts->cond_subgraph_index;
    if (cond_idx >= func_names.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("subgraph with index not found: ", cond_idx));
    }
    const auto cond_attr = mlir::SymbolRefAttr::get(mlir_builder.getContext(),
                                                    func_names[cond_idx]);
    const uint32_t body_idx = opts->body_subgraph_index;
    if (body_idx >= func_names.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("subgraph with index not found: ", body_idx));
    }
    const auto body_attr = mlir::SymbolRefAttr::get(mlir_builder.getContext(),
                                                    func_names[body_idx]);
    attributes.emplace_back(mlir_builder.getNamedAttr("cond", cond_attr));
    attributes.emplace_back(mlir_builder.getNamedAttr("body", body_attr));
  } else if (auto* opts = op.builtin_options_2.AsStablehloReduceOptions()) {
    const int32_t body_idx = opts->body_subgraph_index;
    if (body_idx >= func_names.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("subgraph with index not found: ", body_idx));
    }
    const auto body_attr = mlir::SymbolRefAttr::get(mlir_builder.getContext(),
                                                    func_names[body_idx]);
    attributes.emplace_back(mlir_builder.getNamedAttr("body", body_attr));
  } else if (auto* opts =
                 op.builtin_options_2.AsStablehloReduceWindowOptions()) {
    const int32_t body_idx = opts->body_subgraph_index;
    if (body_idx >= func_names.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("subgraph with index not found: ", body_idx));
    }
    const auto body_attr = mlir::SymbolRefAttr::get(mlir_builder.getContext(),
                                                    func_names[body_idx]);
    attributes.emplace_back(mlir_builder.getNamedAttr("body", body_attr));
  } else if (auto* opts = op.builtin_options_2.AsStablehloSortOptions()) {
    const int32_t comparator_idx = opts->comparator_subgraph_index;
    if (comparator_idx >= func_names.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("subgraph with index not found: ", comparator_idx));
    }
    const auto comparator_attr = mlir::SymbolRefAttr::get(
        mlir_builder.getContext(), func_names[comparator_idx]);
    attributes.emplace_back(
        mlir_builder.getNamedAttr("comparator", comparator_attr));
  } else if (auto* opts = op.builtin_options_2.AsStablehloWhileOptions()) {
    const int32_t body_idx = opts->body_subgraph_index;
    const int32_t cond_idx = opts->cond_subgraph_index;
    if (body_idx >= func_names.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("subgraph with index not found: ", body_idx));
    }
    if (cond_idx >= func_names.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("subgraph with index not found: ", cond_idx));
    }
    const auto body_attr = mlir::SymbolRefAttr::get(mlir_builder.getContext(),
                                                    func_names[body_idx]);
    const auto cond_attr = mlir::SymbolRefAttr::get(mlir_builder.getContext(),
                                                    func_names[cond_idx]);
    attributes.emplace_back(mlir_builder.getNamedAttr("body", body_attr));
    attributes.emplace_back(mlir_builder.getNamedAttr("cond", cond_attr));
  } else if (auto* opts = op.builtin_options_2.AsStablehloScatterOptions()) {
    const uint32_t subgraph_idx = opts->update_computation_subgraph_index;

    if (subgraph_idx >= func_names.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("subgraph with index not found: ", subgraph_idx));
    }
    mlir::FlatSymbolRefAttr subgraph_attr = mlir::SymbolRefAttr::get(
        mlir_builder.getContext(), func_names[subgraph_idx]);
    attributes.emplace_back(mlir_builder.getNamedAttr(
        "update_computation_func_name", subgraph_attr));
  }
  return absl::OkStatus();
}

// Adds builtin and custom options to node attribute. Logic is referred from
// `ConvertOp` in tensorflow/compiler/mlir/lite/flatbuffer_import.cc.
absl::Status AddOptionsToNodeAttribute(
    const OperatorT& op, const OperatorCodes& op_codes,
    const std::vector<std::string>& func_names,
    const std::unique_ptr<FlatBufferModel>& model_ptr,
    mlir::Builder mlir_builder, GraphNodeBuilder& builder) {
  llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
  const OperatorCodeT& op_code = *op_codes.at(op.opcode_index);
  const BuiltinOperator builtin_code = tflite::GetBuiltinCode(&op_code);
  if (builtin_code == tflite::BuiltinOperator_CUSTOM) {
    // Overwrites the node label with the custom op code.
    absl::string_view custom_code = op_code.custom_code;
    builder.SetNodeLabel(custom_code);
    if (op.custom_options_format != tflite::CustomOptionsFormat_FLEXBUFFERS) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported custom options format: ",
          tflite::EnumNameCustomOptionsFormat(op.custom_options_format)));
    }
    std::vector<uint8_t> custom_options;
    if (tflite::IsValidBufferOffset(op.large_custom_options_offset)) {
      custom_options.resize(op.large_custom_options_size);
      memcpy(custom_options.data(),
             reinterpret_cast<const uint8_t*>(model_ptr->allocation()->base()) +
                 op.large_custom_options_offset,
             op.large_custom_options_size);
    } else {
      custom_options = op.custom_options;
    }
    CustomOptionsToAttributes(custom_options, mlir_builder, attrs);
  } else {
    mlir::BuiltinOptionsToAttributes(op.builtin_options, mlir_builder, attrs);
    mlir::BuiltinOptions2ToAttributes(op.builtin_options_2, mlir_builder,
                                      attrs);
  }
  absl::Status status =
      SubgraphIdxToAttributes(op, func_names, mlir_builder, attrs);
  if (!status.ok()) {
    LOG(ERROR) << status.message();
  }
  std::string value;
  llvm::raw_string_ostream sstream(value);
  for (const mlir::NamedAttribute& attr : attrs) {
    const mlir::Attribute& attr_val = attr.getValue();

    // FlatSymbolRefAttr represents the reference to a function call in several
    // tf ops (eg. tf.PartitionedCall, tf.While, tf.If etc). In another word,
    // its value refers to the child subgraph of this parent node.
    if (const auto flat_symbol_attr =
            llvm::dyn_cast_or_null<::mlir::FlatSymbolRefAttr>(attr_val);
        flat_symbol_attr != nullptr) {
      llvm::StringRef subgraph_id = flat_symbol_attr.getValue();
      builder.AppendSubgraphId(subgraph_id);
    }

    // Sets the limit to -1 since BuiltinOptions don't involve constant weights.
    PrintAttribute(attr_val, /*size_limit=*/-1, sstream);
    builder.AppendNodeAttribute(attr.getName().str(), value);
    value.clear();
  }
  return absl::OkStatus();
}

// Optionally adds tensor tag info to the node metadata.
absl::Status AddTensorTags(const OperatorT& op, absl::string_view op_label,
                           const OpdefsMap& op_defs,
                           GraphNodeBuilder& builder) {
  if (!op_defs.contains(op_label)) {
    return absl::InvalidArgumentError(
        absl::StrCat("No op def found for op: ", op_label));
  }
  const OpMetadata& op_metadata = op_defs.at(op_label);
  if (op_metadata.arguments.size() <= op.inputs.size()) {
    for (int i = 0; i < op_metadata.arguments.size(); ++i) {
      builder.AppendAttrToMetadata(EdgeType::kInput, i, kTensorTag,
                                   op_metadata.arguments[i]);
    }
  }
  if (op_metadata.results.size() <= op.outputs.size()) {
    for (int i = 0; i < op_metadata.results.size(); ++i) {
      builder.AppendAttrToMetadata(EdgeType::kOutput, i, kTensorTag,
                                   op_metadata.results[i]);
    }
  }
  return absl::OkStatus();
}

void AddQuantizationParameters(const std::unique_ptr<TensorT>& tensor,
                               const size_t size_limit,
                               const EdgeType edge_type, const int rel_idx,
                               GraphNodeBuilder& builder) {
  if (tensor->quantization == nullptr) return;
  const std::unique_ptr<tflite::QuantizationParametersT>& quant =
      tensor->quantization;
  if (quant->scale.size() != quant->zero_point.size()) {
    LOG(ERROR) << absl::StrCat(
        "Quantization parameters must have the same size: scale(",
        quant->scale.size(), ") != zero point(", quant->zero_point.size(), ")");
    return;
  }
  if (quant->scale.empty()) return;

  const unsigned num_params = (size_limit < 0)
                                  ? quant->scale.size()
                                  : std::min(quant->scale.size(), size_limit);
  if (num_params == 0) return;
  std::vector<std::string> parameters;
  parameters.reserve(num_params);
  for (int i = 0; i < num_params; ++i) {
    // Parameters will be shown as "[scale] * (q + [zero_point])"
    parameters.push_back(
        absl::StrCat(quant->scale[i], " * (q + ", quant->zero_point[i], ")"));
  }
  const std::string quant_str = absl::StrJoin(parameters, ",");
  builder.AppendAttrToMetadata(edge_type, rel_idx, kQuantization, quant_str);
}

// Adds a node to Subgraph.
absl::Status AddNode(
    const int node_index, const OperatorT& op, const OperatorCodes& op_codes,
    const std::vector<std::string>& op_names, const Tensors& tensors,
    const Buffers& buffers, const std::vector<std::string>& func_names,
    const std::optional<const SignatureNameMap>& signature_name_map,
    const OpdefsMap& op_defs, const std::unique_ptr<FlatBufferModel>& model_ptr,
    const VisualizeConfig& config, std::vector<std::string>& node_ids,
    EdgeMap& edge_map, mlir::Builder mlir_builder, Subgraph& subgraph) {
  if (op.opcode_index >= op_names.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Opcode index ", op.opcode_index,
                     " matches no op name for node ", node_index));
  }
  const std::string node_id_str = node_ids[node_index];
  absl::string_view node_label = op_names[op.opcode_index];
  const std::string node_name =
      GenerateNodeName(node_label, op.outputs, tensors);
  GraphNodeBuilder builder;
  builder.SetNodeInfo(node_id_str, node_label, node_name);
  // Logs the error and continues to add the node to the graph.
  absl::Status status = AddOptionsToNodeAttribute(
      op, op_codes, func_names, model_ptr, mlir_builder, builder);
  if (!status.ok()) {
    LOG(ERROR) << status.message();
  }

  for (int i = 0; i < op.inputs.size(); ++i) {
    const int tensor_index = op.inputs[i];

    // Skips the optional inputs which are indicated by -1.
    if (tensor_index < 0) {
      continue;
    }
    PopulateEdgeInfo(tensor_index, {.target_node_input_id = absl::StrCat(i)},
                     edge_map);
    // Since nodes are stored in execution order, EdgeInfo is incomplete only
    // when the input tensor is constant and not an output of a node. Thus we
    // create an auxiliary constant node to align with graph structure.
    if (EdgeInfoIncomplete(edge_map.at(tensor_index))) {
      RETURN_IF_ERROR(
          AddAuxiliaryNode(NodeType::kConstNode, std::vector<int>{tensor_index},
                           tensors, buffers, signature_name_map, model_ptr,
                           config.const_element_count_limit, node_ids, edge_map,
                           mlir_builder, subgraph));
    }
    AppendIncomingEdge(edge_map.at(tensor_index), builder);
    AddQuantizationParameters(tensors[tensor_index],
                              config.quant_params_count_limit, EdgeType::kInput,
                              i, builder);
  }

  for (int i = 0; i < op.outputs.size(); ++i) {
    const int tensor_index = op.outputs[i];
    AppendMetadata(EdgeType::kOutput, i, tensor_index, tensors,
                   signature_name_map, builder);
    PopulateEdgeInfo(tensor_index,
                     {.source_node_id = node_id_str,
                      .source_node_output_id = absl::StrCat(i)},
                     edge_map);

    AddQuantizationParameters(tensors[tensor_index],
                              config.quant_params_count_limit,
                              EdgeType::kOutput, i, builder);
  }

  status = AddTensorTags(op, node_label, op_defs, builder);
  if (!status.ok()) {
    LOG(ERROR) << status.message();
  }

  subgraph.nodes.push_back(std::move(builder).Build());
  return absl::OkStatus();
}

void ValidateSubgraph(absl::string_view subgraph_name,
                      const std::vector<std::string>& node_ids,
                      const EdgeMap& edge_map) {
  absl::flat_hash_set<std::string> node_ids_set(node_ids.begin(),
                                                node_ids.end());
  if (node_ids_set.size() != node_ids.size()) {
    LOG(INFO) << "Node ids: " << absl::StrJoin(node_ids, ",");
    LOG(ERROR) << "Node ids are not unique in " << subgraph_name;
  }

  bool has_incomplete_edges = false;
  for (const auto& edge : edge_map) {
    const int tensor_index = edge.first;
    const EdgeInfo& edge_info = edge.second;
    if (EdgeInfoIncomplete(edge_info)) {
      has_incomplete_edges = true;
      LOG(INFO) << "tensor index: " << tensor_index << ", "
                << EdgeInfoDebugString(edge_info);
    }
  }
  if (has_incomplete_edges) {
    LOG(ERROR) << "EdgeMap has incomplete EdgeInfo in " << subgraph_name;
  }
}

// Adds a subgraph to Graph.
absl::Status AddSubgraph(
    const VisualizeConfig& config, absl::string_view subgraph_name,
    const SubGraphT& subgraph_t, const std::vector<std::string>& op_names,
    const std::optional<const SignatureNameMap>& signature_name_map,
    const std::vector<std::string>& func_names, const OpdefsMap& op_defs,
    const std::unique_ptr<ModelT>& model,
    const std::unique_ptr<FlatBufferModel>& model_ptr,
    mlir::Builder mlir_builder, Graph& graph) {
  // Creates a Model Explorer subgraph.
  Subgraph subgraph((std::string(subgraph_name)));
  EdgeMap edge_map;
  std::vector<std::string> node_ids;
  const Buffers& buffers = model->buffers;
  const OperatorCodes& op_codes = model->operator_codes;
  // Reserves the node ids for original nodes stored in Flatbuffer.
  node_ids.reserve(subgraph_t.operators.size());
  for (int i = 0; i < subgraph_t.operators.size(); ++i) {
    node_ids.push_back(absl::StrCat(i));
  }

  // Adds GraphInputs node to the subgraph.
  RETURN_IF_ERROR(AddAuxiliaryNode(
      NodeType::kInputNode, subgraph_t.inputs, subgraph_t.tensors, buffers,
      signature_name_map, model_ptr, config.const_element_count_limit, node_ids,
      edge_map, mlir_builder, subgraph));

  for (int i = 0; i < subgraph_t.operators.size(); ++i) {
    auto& op = subgraph_t.operators[i];
    const Tensors& tensors = subgraph_t.tensors;
    RETURN_IF_ERROR(AddNode(i, *op, op_codes, op_names, tensors, buffers,
                            func_names, signature_name_map, op_defs, model_ptr,
                            config, node_ids, edge_map, mlir_builder,
                            subgraph));
  }

  // Adds GraphOutputs node to the subgraph.
  RETURN_IF_ERROR(AddAuxiliaryNode(
      NodeType::kOutputNode, subgraph_t.outputs, subgraph_t.tensors, buffers,
      signature_name_map, model_ptr, config.const_element_count_limit, node_ids,
      edge_map, mlir_builder, subgraph));

  ValidateSubgraph(subgraph_name, node_ids, edge_map);
  graph.subgraphs.push_back(std::move(subgraph));
  return absl::OkStatus();
}

std::string GetSubgraphName(int subgraph_index, const SubGraphT& subgraph_t) {
  if (!subgraph_t.name.empty()) {
    return subgraph_t.name;
  }
  return (subgraph_index == 0) ? "main"
                               : absl::StrCat("subgraph_", subgraph_index);
}

}  // namespace

absl::StatusOr<std::string> ConvertFlatbufferDirectlyToJson(
    const VisualizeConfig& config, absl::string_view model_path) {
  std::string model_content;
  RETURN_IF_ERROR(tsl::ReadFileToString(
      tsl::Env::Default(), std::string(model_path), &model_content));

  std::unique_ptr<FlatBufferModel> model_ptr =
      FlatBufferModel::VerifyAndBuildFromBuffer(model_content.data(),
                                                model_content.length());

  mlir::MLIRContext mlir_context;
  mlir::DialectRegistry registry;
  registry.insert<mlir::TFL::TensorFlowLiteDialect, mlir::TF::TensorFlowDialect,
                  mlir::stablehlo::StablehloDialect, mlir::vhlo::VhloDialect>();
  mlir::func::registerAllExtensions(registry);
  mlir_context.appendDialectRegistry(registry);
  mlir_context.loadDialect<
      mlir::TFL::TensorFlowLiteDialect, mlir::TF::TensorFlowDialect,
      mlir::stablehlo::StablehloDialect, mlir::vhlo::VhloDialect>();
  mlir::Builder mlir_builder(&mlir_context);

  Graph graph;
  std::unique_ptr<ModelT> model(model_ptr->GetModel()->UnPack());
  const std::vector<std::string> op_names = GetOpNames(model->operator_codes);

  SignatureMap signature_map;
  for (const auto& signature_def : model->signature_defs) {
    SignatureNameMap signature_name_map;
    for (const auto& tensormap : signature_def->inputs) {
      signature_name_map.emplace(tensormap->tensor_index, tensormap->name);
    }
    for (const auto& tensormap : signature_def->outputs) {
      signature_name_map.emplace(tensormap->tensor_index, tensormap->name);
    }
    signature_map.emplace(signature_def->subgraph_index, signature_name_map);
  }

  std::vector<std::string> func_names;
  func_names.reserve(model->subgraphs.size());
  for (const auto& subgraph : model->subgraphs) {
    func_names.push_back(subgraph->name);
  }

  const OpdefsMap op_defs = LoadTfliteOpdefs();

  for (int i = 0; i < model->subgraphs.size(); ++i) {
    const auto& subgraph = model->subgraphs[i];
    const std::string subgraph_name = GetSubgraphName(i, *subgraph);
    auto signature_name_it = signature_map.find(i);
    if (signature_name_it != signature_map.end()) {
      RETURN_IF_ERROR(AddSubgraph(
          config, subgraph_name, *subgraph, op_names, signature_name_it->second,
          func_names, op_defs, model, model_ptr, mlir_builder, graph));
    } else {
      RETURN_IF_ERROR(AddSubgraph(config, subgraph_name, *subgraph, op_names,
                                  /*signature_name_map=*/std::nullopt,
                                  func_names, op_defs, model, model_ptr,
                                  mlir_builder, graph));
    }
  }

  GraphCollection collection;
  collection.graphs.push_back(std::move(graph));
  llvm::json::Value json_result(collection.Json());
  return llvm::formatv("{0:2}", json_result);
}

}  // namespace visualization_client
}  // namespace tooling
