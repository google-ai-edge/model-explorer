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

#include "direct_flatbuffer_to_json_graph_convert.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
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
#include "tensorflow/compiler/mlir/lite/core/absl_error_model_builder.h"
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
#include "xla/tsl/platform/env.h"
#include "tensorflow/core/framework/tensor.pb.h"

namespace tooling {
namespace visualization_client {
namespace {

using ::mlir::TFL::FlatBufferModelAbslError;
using ::tflite::BuiltinOperator;
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
constexpr absl::string_view kQuantizedDimension = "quantized_dimension";
constexpr absl::string_view kSignatureName = "signature_name";

struct EdgeInfo {
  std::string source_node_id = "";
  std::string source_node_output_id = "";
  std::string target_node_input_id = "";
};

using EdgeMap = absl::flat_hash_map<int, EdgeInfo>;
using SignatureNameMap = absl::flat_hash_map<int, std::string>;
using SignatureMap = absl::flat_hash_map<int, SignatureNameMap>;
using Tensors = std::vector<std::unique_ptr<TensorT>>;
using OperatorCodes = std::vector<std::unique_ptr<OperatorCodeT>>;
using Buffers = std::vector<std::unique_ptr<tflite::BufferT>>;
using SignatureDefs = std::vector<std::unique_ptr<tflite::SignatureDefT>>;
using OpdefsMap = absl::flat_hash_map<std::string, OpMetadata>;

enum NodeType {
  kInputNode,
  kOutputNode,
  kConstNode,
};

// A struct to hold the context for building a subgraph.
struct SubgraphBuildContext {
  SubgraphBuildContext(const SubGraphT& subgraph_t,
                       const SignatureNameMap& signature_name_map,
                       Subgraph& subgraph)
      : subgraph_t(subgraph_t),
        signature_name_map(signature_name_map),
        subgraph(subgraph) {
    // Reserves the node ids for original nodes stored in Flatbuffer.
    node_ids.reserve(subgraph_t.operators.size());
    for (int i = 0; i < subgraph_t.operators.size(); ++i) {
      node_ids.push_back(absl::StrCat(i));
    }
  }

  // The subgraph data from the flatbuffer model.
  const SubGraphT& subgraph_t;
  // Map from the tensor index to input/output signature name.
  const SignatureNameMap& signature_name_map;

  // -- Mutable state while building the subgraph. --
  // Map from tensor index to EdgeInfo.
  EdgeMap edge_map;
  // The node ids for the subgraph.
  std::vector<std::string> node_ids;
  // The Model Explorer subgraph to be built.
  Subgraph& subgraph;
};

// A helper class to hold the TFLite model data and convert it to Model Explorer
// JSON graph.
class FlatbufferToJsonConverter {
 public:
  FlatbufferToJsonConverter(
      const VisualizeConfig& config,
      std::unique_ptr<FlatBufferModelAbslError> model_ptr);

  // Disables copy/move for this class.
  FlatbufferToJsonConverter(const FlatbufferToJsonConverter&) = delete;
  FlatbufferToJsonConverter& operator=(const FlatbufferToJsonConverter&) =
      delete;

  // Converts the Flatbuffer model content to JSON string.
  absl::StatusOr<std::string> Convert();

 private:
  // Gets the buffer data of the given tensor.
  absl::StatusOr<std::vector<uint8_t>> GetBuffer(const TensorT& tensor,
                                                 SubgraphBuildContext& context);

  // Adds the constant value to the node attribute.
  absl::Status AddConstantToNodeAttr(const TensorT& tensor,
                                     SubgraphBuildContext& context,
                                     GraphNodeBuilder& builder);

  // Adds an auxiliary node (GraphInputs, GraphOutputs or const node) to the
  // subgraph.
  absl::Status AddAuxiliaryNode(NodeType node_type,
                                const std::vector<int>& tensor_indices,
                                SubgraphBuildContext& context);

  // Converts subgraph index to attributes for certain ops (e.g. If, While).
  absl::Status SubgraphIdxToAttributes(
      const tflite::OperatorT& op, SubgraphBuildContext& context,
      llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes);

  // Adds the options of the given op to the node attribute. Logic is referred
  // from `ConvertOp` in tensorflow/compiler/mlir/lite/flatbuffer_import.cc.
  absl::Status AddOptionsToNodeAttribute(const OperatorT& op,
                                         SubgraphBuildContext& context,
                                         GraphNodeBuilder& builder);

  // Optionally adds tensor tags from tflite op defs to the node metadata.
  absl::Status AddTensorTags(const OperatorT& op, SubgraphBuildContext& context,
                             GraphNodeBuilder& builder);

  // Adds a node to the Model Explorer subgraph.
  absl::Status AddNode(int node_index, SubgraphBuildContext& context);

  // Adds a subgraph to the Model Explorer graph.
  absl::Status AddSubgraph(int subgraph_index, Graph& graph);

  // -- Initialized in constructor --
  // Config for visualization.
  const VisualizeConfig& config_;
  // Map from op name to op metadata.
  const OpdefsMap op_defs_;
  // Pointer to the flatbuffer model.
  std::unique_ptr<FlatBufferModelAbslError> model_ptr_;
  // MLIR context and builder for creating MLIR attributes.
  mlir::MLIRContext mlir_context_;  // Owns the mlir context.
  mlir::Builder mlir_builder_;

  // Parsed model from the flatbuffer.
  std::unique_ptr<ModelT> model_;
  // List of op names.
  std::vector<std::string> op_names_;
  // List of function names.
  std::vector<std::string> func_names_;
  // Map from subgraph index to signature name map.
  SignatureMap signature_map_;
};

std::string EdgeInfoDebugString(const EdgeInfo& edge_info) {
  return absl::StrCat("sourceNodeId: ", edge_info.source_node_id,
                      ", sourceNodeOutputId: ", edge_info.source_node_output_id,
                      ", targetNodeInputId: ", edge_info.target_node_input_id);
}

// Returns the lowercase op name string from the given op code.
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

// Appends the incoming edge to the graph node builder.
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
                             SubgraphBuildContext& context) {
  const Tensors& tensors = context.subgraph_t.tensors;
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

// Appends the metadata to the graph node builder.
void AppendMetadata(EdgeType edge_type, int metadata_id, int tensor_index,
                    SubgraphBuildContext& context, GraphNodeBuilder& builder) {
  const Tensors& tensors = context.subgraph_t.tensors;
  const SignatureNameMap& signature_name_map = context.signature_name_map;
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
  auto name_it = signature_name_map.find(tensor_index);
  if (name_it != signature_name_map.end()) {
    builder.AppendAttrToMetadata(edge_type, metadata_id, kSignatureName,
                                 name_it->second);
  }
}

// Adds quantization parameters to the graph node builder.
void AddQuantizationParameters(const std::unique_ptr<TensorT>& tensor,
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

  std::vector<std::string> parameters;
  parameters.reserve(quant->scale.size());
  for (int i = 0; i < quant->scale.size(); ++i) {
    // Parameters will be shown as "[scale] * (q - [zero_point])"
    const float scale = quant->scale[i];
    const int64_t zp = quant->zero_point[i];
    const char zp_sign = zp < 0 ? '+' : '-';
    const int64_t abs_zp = std::abs(zp);
    parameters.push_back(abs_zp == 0 ? absl::StrFormat("%f * q", scale)
                                     : absl::StrFormat("%f * (q %c %d)", scale,
                                                       zp_sign, abs_zp));
  }
  const std::string quant_str = absl::StrJoin(parameters, ",");
  builder.AppendAttrToMetadata(edge_type, rel_idx, kQuantization, quant_str);

  // Adds the quantized dimension.
  builder.AppendAttrToMetadata(edge_type, rel_idx, kQuantizedDimension,
                               absl::StrCat(quant->quantized_dimension));
}

// Validates whether the subgraph is complete with all nodes and edges.
void ValidateSubgraph(absl::string_view subgraph_name,
                      SubgraphBuildContext& context) {
  const std::vector<std::string>& node_ids = context.node_ids;
  const EdgeMap& edge_map = context.edge_map;

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

// Returns true if the node name is in the root space, i.e. it doesn't contain
// any slashes.
bool IsInRootSpace(absl::string_view node_name) {
  // If the node name doesn't contain '/', it's in the root namespace.
  return !absl::StrContains(node_name, '/');
}

// Returns the shared namespace of the given node namesapces.
// For example, if the node namespaces are ["a/b/c/d", "a/b/c", "a/b/f"], the
// shared namespace is "a/b".
std::string GetSharedNamespace(const std::vector<std::string>& node_names) {
  bool initialized = false;
  std::vector<std::string> namespace_parts;
  int shared_prefix_length = 0;  // Keep track of the shared prefix length.

  for (absl::string_view node_name : node_names) {
    // We skip the root namespace when finding the shared namespace.
    if (IsInRootSpace(node_name)) {
      continue;
    }
    if (!initialized) {
      namespace_parts = absl::StrSplit(node_name, '/');
      shared_prefix_length = namespace_parts.size();
      initialized = true;
      continue;
    }
    // Compares the node name, only keep the shared namespace parts.
    std::vector<std::string> compared_parts = absl::StrSplit(node_name, '/');
    int min_size = std::min(shared_prefix_length, (int)compared_parts.size());
    int current_match_length = 0;
    for (int i = 0; i < min_size; ++i) {
      if (namespace_parts[i] == compared_parts[i]) {
        current_match_length++;
      } else {
        break;  // Exit inner loop on mismatch.
      }
    }

    shared_prefix_length = current_match_length;
    if (shared_prefix_length == 0) {
      return "";
    }
  }
  namespace_parts.resize(shared_prefix_length);
  return absl::StrJoin(namespace_parts, "/");
}

// Post-processes the subgraph, performing operations after the subgraph is
// built.
void PostProcessSubgraph(Subgraph& subgraph) {
  // Creates a map from node ID to the node's index in the `subgraph.nodes`
  // vector. This allows for efficient lookup of nodes by ID.
  absl::flat_hash_map<std::string, int> node_id_to_index;
  for (int i = 0; i < subgraph.nodes.size(); ++i) {
    node_id_to_index[subgraph.nodes[i].node_id] = i;
  }

  // Find the "GraphOutputs" node and get the IDs of its input nodes.
  // The "GraphOutputs" node is assumed to be closer to the end of the nodes
  // vector, so the iteration starts from the end for efficiency.
  std::vector<std::string> input_node_ids;
  for (int i = subgraph.nodes.size() - 1; i >= 0; --i) {
    const GraphNode& node = subgraph.nodes[i];
    if (node.node_label == kGraphOutputs) {
      // Collect the source node IDs of all incoming edges to the
      // "GraphOutputs" node. These are the input nodes to "GraphOutputs".
      for (const GraphEdge& edge : node.incoming_edges) {
        input_node_ids.push_back(edge.source_node_id);
      }
      break;
    }
  }

  for (absl::string_view node_id : input_node_ids) {
    GraphNode& node = subgraph.nodes[node_id_to_index[node_id]];
    // Only process nodes that are in the root space, so we don't mess up the
    // namespace of nodes that are already in a nested namespace.
    if (!IsInRootSpace(node.node_name)) {
      continue;
    }
    std::vector<std::string> parent_node_names;
    for (const GraphEdge& edge : node.incoming_edges) {
      const GraphNode& parent_node =
          subgraph.nodes[node_id_to_index[edge.source_node_id]];
      parent_node_names.push_back(parent_node.node_name);
    }
    const std::string shared_namespace = GetSharedNamespace(parent_node_names);
    // If a shared namespace exists, prepend it to the current node's name.
    // This effectively moves the node into the shared namespace.
    if (!shared_namespace.empty()) {
      node.node_name = absl::StrCat(shared_namespace, "/", node.node_name);
    }
  }
}

// Returns the subgraph name for the given subgraph index.
// If the subgraph name is empty, use the signature key if it exists. If the
// signature key is also empty, use the default name.
std::string GetSubgraphName(int subgraph_index, const SubGraphT& subgraph_t,
                            const SignatureDefs& signature_defs) {
  if (!subgraph_t.name.empty()) {
    return subgraph_t.name;
  }

  // If the subgraph name is empty, use the signature key if it exists.
  // TODO(yijieyang): We should add this signature key to graph level info
  // regardless.
  for (const auto& signature_def : signature_defs) {
    if (signature_def->subgraph_index == subgraph_index) {
      if (!signature_def->signature_key.empty()) {
        return signature_def->signature_key;
      }
      break;
    }
  }

  return (subgraph_index == 0) ? "main"
                               : absl::StrCat("subgraph_", subgraph_index);
}

absl::StatusOr<std::vector<uint8_t>> FlatbufferToJsonConverter::GetBuffer(
    const TensorT& tensor, SubgraphBuildContext& context) {
  const uint64_t buffer_offset = model_->buffers[tensor.buffer]->offset;
  const uint64_t buffer_size = model_->buffers[tensor.buffer]->size;
  // Check if constant tensor is stored outside of the flatbuffers.
  if (tflite::IsValidBufferOffset(buffer_offset)) {
    if (!model_->buffers[tensor.buffer]->data.empty()) {
      return absl::InvalidArgumentError(
          "Buffer data and offset cannot be set at the same time.");
    }
    const uint8_t* file_begin_ptr =
        reinterpret_cast<const uint8_t*>(model_ptr_->allocation()->base());
    if (buffer_offset + buffer_size > model_ptr_->allocation()->bytes()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Constant buffer of tensor \"", tensor.name,
                       "\" specified an out of range offset."));
    }
    return std::vector<uint8_t>(file_begin_ptr + buffer_offset,
                                file_begin_ptr + buffer_offset + buffer_size);
  }
  return model_->buffers[tensor.buffer]->data;
}

absl::Status FlatbufferToJsonConverter::AddConstantToNodeAttr(
    const TensorT& tensor, SubgraphBuildContext& context,
    GraphNodeBuilder& builder) {
  ASSIGN_OR_RETURN(std::vector<uint8_t> buffer, GetBuffer(tensor, context));
  if (buffer.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Buffer data for tensor \"", tensor.name, "\" is empty."));
  }
  ASSIGN_OR_RETURN(mlir::ElementsAttr elem_attr,
                   ConvertBufferToAttr(tensor, buffer, mlir_builder_));
  std::string value;
  llvm::raw_string_ostream sstream(value);
  PrintAttribute(elem_attr, config_.const_element_count_limit, sstream);
  builder.AppendNodeAttribute(/*key=*/kValue, /*value=*/value);
  return absl::OkStatus();
}

absl::Status FlatbufferToJsonConverter::AddAuxiliaryNode(
    NodeType node_type, const std::vector<int>& tensor_indices,
    SubgraphBuildContext& context) {
  // Skips adding auxiliary node if `tensor_indices` is empty.
  // This is to avoid adding GraphInputs/GraphOutputs node when there is no
  // input/output tensor.
  if (tensor_indices.empty()) {
    return absl::OkStatus();
  }
  const Tensors& tensors = context.subgraph_t.tensors;
  EdgeMap& edge_map = context.edge_map;

  EdgeType edge_type;
  std::string node_label, node_name;
  std::vector<std::string>& node_ids = context.node_ids;
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
      node_name = GenerateNodeName(node_label, tensor_indices, context);
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
    absl::Status status = AddConstantToNodeAttr(tensor, context, builder);
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
            NodeType::kConstNode, std::vector<int>{tensor_index}, context));
      }
      AppendIncomingEdge(edge_map.at(tensor_index), builder);
    } else {
      AppendMetadata(EdgeType::kOutput, i, tensor_index, context, builder);

      PopulateEdgeInfo(tensor_index,
                       {.source_node_id = node_id_str,
                        .source_node_output_id = absl::StrCat(i)},
                       edge_map);
    }
  }
  context.subgraph.nodes.push_back(std::move(builder).Build());
  return absl::OkStatus();
}

absl::Status FlatbufferToJsonConverter::SubgraphIdxToAttributes(
    const tflite::OperatorT& op, SubgraphBuildContext& context,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes) {
  // Helper lambda to validate a subgraph index and add it as a SymbolRef
  // attribute. This handles the common case for control flow and other ops.
  auto add_symbol_ref_attr = [&](int32_t index,
                                 llvm::StringRef attr_name) -> absl::Status {
    if (index < 0 || index >= func_names_.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("subgraph with index not found: ", index));
    }
    const auto attr = mlir::SymbolRefAttr::get(mlir_builder_.getContext(),
                                               func_names_[index]);
    attributes.emplace_back(mlir_builder_.getNamedAttr(attr_name, attr));
    return absl::OkStatus();
  };

  if (auto* opts = op.builtin_options.AsCallOnceOptions()) {
    // Special handling for CallOnceOptions as it uses a StringAttr instead of
    // SymbolRefAttr.
    const uint32_t init_idx = opts->init_subgraph_index;
    if (init_idx >= func_names_.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("subgraph with index not found: ", init_idx));
    }
    const auto init_attr = mlir_builder_.getStringAttr(func_names_[init_idx]);
    attributes.emplace_back(
        mlir_builder_.getNamedAttr("session_init_function", init_attr));
  } else if (auto* opts = op.builtin_options.AsIfOptions()) {
    RETURN_IF_ERROR(
        add_symbol_ref_attr(opts->then_subgraph_index, "then_branch"));
    RETURN_IF_ERROR(
        add_symbol_ref_attr(opts->else_subgraph_index, "else_branch"));
    attributes.emplace_back(mlir_builder_.getNamedAttr(
        "is_stateless", mlir_builder_.getBoolAttr(false)));
  } else if (auto* opts = op.builtin_options.AsWhileOptions()) {
    RETURN_IF_ERROR(add_symbol_ref_attr(opts->cond_subgraph_index, "cond"));
    RETURN_IF_ERROR(add_symbol_ref_attr(opts->body_subgraph_index, "body"));
  } else if (auto* opts = op.builtin_options_2.AsStablehloReduceOptions()) {
    RETURN_IF_ERROR(add_symbol_ref_attr(opts->body_subgraph_index, "body"));
  } else if (auto* opts =
                 op.builtin_options_2.AsStablehloReduceWindowOptions()) {
    RETURN_IF_ERROR(add_symbol_ref_attr(opts->body_subgraph_index, "body"));
  } else if (auto* opts = op.builtin_options_2.AsStablehloSortOptions()) {
    RETURN_IF_ERROR(
        add_symbol_ref_attr(opts->comparator_subgraph_index, "comparator"));
  } else if (auto* opts = op.builtin_options_2.AsStablehloWhileOptions()) {
    RETURN_IF_ERROR(add_symbol_ref_attr(opts->cond_subgraph_index, "cond"));
    RETURN_IF_ERROR(add_symbol_ref_attr(opts->body_subgraph_index, "body"));
  } else if (auto* opts = op.builtin_options_2.AsStablehloScatterOptions()) {
    RETURN_IF_ERROR(add_symbol_ref_attr(opts->update_computation_subgraph_index,
                                        "update_computation_func_name"));
  } else if (auto* opts = op.builtin_options_2.AsStableHLOCompositeOptions()) {
    RETURN_IF_ERROR(add_symbol_ref_attr(opts->decomposition_subgraph_index,
                                        "decomposition"));
  }

  return absl::OkStatus();
}

absl::Status FlatbufferToJsonConverter::AddOptionsToNodeAttribute(
    const OperatorT& op, SubgraphBuildContext& context,
    GraphNodeBuilder& builder) {
  llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
  const OperatorCodeT& op_code = *model_->operator_codes.at(op.opcode_index);
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
      memcpy(
          custom_options.data(),
          reinterpret_cast<const uint8_t*>(model_ptr_->allocation()->base()) +
              op.large_custom_options_offset,
          op.large_custom_options_size);
    } else {
      custom_options = op.custom_options;
    }
    CustomOptionsToAttributes(custom_options, mlir_builder_, attrs);
  } else {
    mlir::BuiltinOptionsToAttributes(op.builtin_options, mlir_builder_, attrs);
    mlir::BuiltinOptions2ToAttributes(op.builtin_options_2, mlir_builder_,
                                      attrs);
  }
  absl::Status status = SubgraphIdxToAttributes(op, context, attrs);
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

absl::Status FlatbufferToJsonConverter::AddTensorTags(
    const OperatorT& op, SubgraphBuildContext& context,
    GraphNodeBuilder& builder) {
  const std::string op_label = builder.GetNodeLabel();
  if (!op_defs_.contains(op_label)) {
    return absl::InvalidArgumentError(
        absl::StrCat("No op def found for op: ", op_label));
  }
  const OpMetadata& op_metadata = op_defs_.at(op_label);
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

absl::Status FlatbufferToJsonConverter::AddNode(const int node_index,
                                                SubgraphBuildContext& context) {
  const OperatorT& op = *context.subgraph_t.operators[node_index];
  std::vector<std::string>& node_ids = context.node_ids;
  EdgeMap& edge_map = context.edge_map;
  const Tensors& tensors = context.subgraph_t.tensors;

  if (op.opcode_index >= op_names_.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Opcode index ", op.opcode_index,
                     " matches no op name for node ", node_index));
  }
  const std::string node_id_str = node_ids[node_index];
  absl::string_view node_label = op_names_[op.opcode_index];
  const std::string node_name =
      GenerateNodeName(node_label, op.outputs, context);
  GraphNodeBuilder builder;
  builder.SetNodeInfo(node_id_str, node_label, node_name);
  // Logs the error and continues to add the node to the graph.
  absl::Status status = AddOptionsToNodeAttribute(op, context, builder);
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
      RETURN_IF_ERROR(AddAuxiliaryNode(
          NodeType::kConstNode, std::vector<int>{tensor_index}, context));
    }
    AppendIncomingEdge(edge_map.at(tensor_index), builder);
    AddQuantizationParameters(tensors[tensor_index], EdgeType::kInput, i,
                              builder);
  }

  for (int i = 0; i < op.outputs.size(); ++i) {
    const int tensor_index = op.outputs[i];
    AppendMetadata(EdgeType::kOutput, i, tensor_index, context, builder);
    PopulateEdgeInfo(tensor_index,
                     {.source_node_id = node_id_str,
                      .source_node_output_id = absl::StrCat(i)},
                     edge_map);

    AddQuantizationParameters(tensors[tensor_index], EdgeType::kOutput, i,
                              builder);
  }

  status = AddTensorTags(op, context, builder);
  if (!status.ok()) {
    LOG(ERROR) << status.message();
  }

  context.subgraph.nodes.push_back(std::move(builder).Build());
  return absl::OkStatus();
}

absl::Status FlatbufferToJsonConverter::AddSubgraph(const int subgraph_index,
                                                    Graph& graph) {
  // Gets the required subgraph and signature data from the model.
  const SubGraphT& subgraph_t = *model_->subgraphs[subgraph_index];
  const std::string subgraph_name =
      GetSubgraphName(subgraph_index, subgraph_t, model_->signature_defs);
  SignatureNameMap signature_name_map;
  if (signature_map_.contains(subgraph_index)) {
    signature_name_map = signature_map_.at(subgraph_index);
  }

  // Creates a Model Explorer subgraph and its context.
  Subgraph subgraph(subgraph_name);
  SubgraphBuildContext context(subgraph_t, signature_name_map, subgraph);

  // Adds GraphInputs node to the subgraph.
  RETURN_IF_ERROR(
      AddAuxiliaryNode(NodeType::kInputNode, subgraph_t.inputs, context));

  for (int i = 0; i < subgraph_t.operators.size(); ++i) {
    RETURN_IF_ERROR(AddNode(i, context));
  }

  // Adds GraphOutputs node to the subgraph.
  RETURN_IF_ERROR(
      AddAuxiliaryNode(NodeType::kOutputNode, subgraph_t.outputs, context));

  ValidateSubgraph(subgraph_name, context);
  PostProcessSubgraph(subgraph);
  graph.subgraphs.push_back(std::move(subgraph));
  return absl::OkStatus();
}

FlatbufferToJsonConverter::FlatbufferToJsonConverter(
    const VisualizeConfig& config,
    std::unique_ptr<FlatBufferModelAbslError> model_ptr)
    : config_(config),
      op_defs_(LoadTfliteOpdefs()),
      model_ptr_(std::move(model_ptr)),
      mlir_context_(mlir::MLIRContext()),
      mlir_builder_(&mlir_context_) {
  model_ = std::unique_ptr<ModelT>(model_ptr_->GetModel()->UnPack());
  op_names_ = GetOpNames(model_->operator_codes);

  // Registers mlir dialects.
  mlir::DialectRegistry registry;
  registry.insert<mlir::TFL::TensorFlowLiteDialect, mlir::TF::TensorFlowDialect,
                  mlir::stablehlo::StablehloDialect, mlir::vhlo::VhloDialect>();
  mlir::func::registerAllExtensions(registry);
  mlir_context_.appendDialectRegistry(registry);
  mlir_context_.loadDialect<
      mlir::TFL::TensorFlowLiteDialect, mlir::TF::TensorFlowDialect,
      mlir::stablehlo::StablehloDialect, mlir::vhlo::VhloDialect>();

  // Initializes function names.
  func_names_.reserve(model_->subgraphs.size());
  for (const auto& subgraph : model_->subgraphs) {
    func_names_.push_back(subgraph->name);
  }

  // Initializes signature map.
  for (const auto& signature_def : model_->signature_defs) {
    SignatureNameMap signature_name_map;
    for (const auto& tensormap : signature_def->inputs) {
      signature_name_map.emplace(tensormap->tensor_index, tensormap->name);
    }
    for (const auto& tensormap : signature_def->outputs) {
      signature_name_map.emplace(tensormap->tensor_index, tensormap->name);
    }
    signature_map_.emplace(signature_def->subgraph_index,
                           std::move(signature_name_map));
  }
}

absl::StatusOr<std::string> FlatbufferToJsonConverter::Convert() {
  Graph graph;
  for (int index = 0; index < model_->subgraphs.size(); ++index) {
    RETURN_IF_ERROR(AddSubgraph(index, graph));
  }

  GraphCollection collection;
  collection.graphs.push_back(std::move(graph));
  llvm::json::Value json_result(collection.Json());
  return llvm::formatv("{0:2}", json_result);
}

}  // namespace

// Logic referred from `CustomOptionsToAttributes` in
// tensorflow/compiler/mlir/lite/flatbuffer_operator.cc.
void CustomOptionsToAttributes(
    const std::vector<uint8_t>& custom_options, mlir::Builder mlir_builder,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes) {
  if (custom_options.empty()) {
    // Avoid calling flexbuffers::GetRoot() with empty data. Otherwise it will
    // crash.
    //
    // TODO(yijieyang): We should use a default value for input custom_options
    // that is not empty to avoid this check.
    return;
  }
  const flexbuffers::Map& map = flexbuffers::GetRoot(custom_options).AsMap();
  if (map.IsTheEmptyMap()) {
    // The custom_options is not empty but not a valid flex buffer map.
    attributes.emplace_back(mlir_builder.getNamedAttr(
        "custom_options", mlir_builder.getStringAttr("<non-deserializable>")));
    return;
  }
  const flexbuffers::TypedVector& keys = map.Keys();
  for (size_t i = 0; i < keys.size(); ++i) {
    const char* key = keys[i].AsKey();
    const flexbuffers::Reference& value = map[key];
    attributes.emplace_back(mlir_builder.getNamedAttr(
        key, mlir_builder.getStringAttr(value.ToString())));
  }
}

absl::StatusOr<std::string> ConvertFlatbufferDirectlyToJson(
    const VisualizeConfig& config, absl::string_view model_path) {
  std::string model_content;
  RETURN_IF_ERROR(tsl::ReadFileToString(
      tsl::Env::Default(), std::string(model_path), &model_content));

  std::unique_ptr<FlatBufferModelAbslError> model_ptr =
      FlatBufferModelAbslError::VerifyAndBuildFromBuffer(
          model_content.data(), model_content.length());
  if (model_ptr == nullptr) {
    return absl::InvalidArgumentError("Failed to build model from buffer.");
  }

  FlatbufferToJsonConverter converter(config, std::move(model_ptr));
  return converter.Convert();
}

}  // namespace visualization_client
}  // namespace tooling
