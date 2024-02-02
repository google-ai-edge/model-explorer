#include "direct_flatbuffer_to_json_graph_convert.h"

#include <cstdint>
#include <cstring>
#include <iterator>
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
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "formats/schema_structs.h"
#include "graphnode_builder.h"
#include "status_macros.h"
#include "tools/attribute_printer.h"
#include "tools/convert_type.h"
#include "visualize_config.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_operator.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/offset_buffer.h"
#include "tensorflow/compiler/mlir/lite/utils/const_tensor_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
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

constexpr absl::string_view kGraphInputs = "GraphInputs";
constexpr absl::string_view kGraphOutputs = "GraphOutputs";
constexpr absl::string_view kPseudoConst = "pseudo_const";
constexpr absl::string_view kTensorIndex = "tensor_index";
constexpr absl::string_view kTensorName = "tensor_name";
constexpr absl::string_view kTensorShape = "tensor_shape";
constexpr absl::string_view kValue = "value";
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

// Generates the node name based on the provided tensor indices.
//
// In TFLite, a single tensor name could still contain several hierarchical info
// concatenated together with semicolons. In this case, we will find the last
// candidate node name that contains this node label. If no match is found, we
// will return the last candidate node name by default. This method also echos
// the MLIR-based conversion for Flatbuffer.
absl::StatusOr<std::string> GenerateNodeName(
    absl::string_view node_id_str, absl::string_view node_label,
    const std::vector<int>& tensor_indices, const Tensors& tensors) {
  if (tensor_indices.empty()) {
    return absl::InvalidArgumentError("Tensor indices cannot be empty.");
  }
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
  if (candidate_names.empty()) return "";
  if (candidate_names.size() == 1) {
    return candidate_names[0];
  }

  // Removes any underscores in `node_label`.
  const std::string node_label_substr =
      absl::StrReplaceAll(node_label, {{"_", ""}});

  // Iterates backwards to find if the last chunk of candidate_name contains the
  // node label in the end hierarchy.
  for (auto name_it = std::rbegin(candidate_names);
       name_it != std::rend(candidate_names); ++name_it) {
    const auto start_pos = name_it->find_last_of('/');
    std::string last_substr;
    if (start_pos != std::string::npos) {
      last_substr = name_it->substr(start_pos, name_it->size());
    } else {
      last_substr = *name_it;
    }
    if (absl::AsciiStrToLower(last_substr).find(node_label_substr) !=
        std::string::npos) {
      return *name_it;
    }
  }

  // If there is no match in `candidate_names` vector, we return the last
  // candidate name by default. Skipping "pseudo_const" node to reduce
  // verbosity.
  if (node_label != kPseudoConst) {
    LOG(WARNING) << "No matched name for node \"" << node_label << "\" at "
                 << node_id_str << ", using the last node name by default.";
  }
  return candidate_names.back();
}

absl::Status AppendMetadata(
    const EdgeType edge_type, const int metadata_id, const int tensor_index,
    const Tensors& tensors,
    const std::optional<const SignatureNameMap>& signature_name_map,
    GraphNodeBuilder& builder) {
  // Appends tensor index.
  RETURN_IF_ERROR(builder.AppendAttrToMetadata(
      edge_type, metadata_id, kTensorIndex, absl::StrCat(tensor_index)));
  // Appends tensor name.
  RETURN_IF_ERROR(builder.AppendAttrToMetadata(
      edge_type, metadata_id, kTensorName, tensors[tensor_index]->name));
  // Appends tensor shape.
  RETURN_IF_ERROR(builder.AppendAttrToMetadata(
      edge_type, metadata_id, kTensorShape,
      StringifyTensorShape(*tensors[tensor_index])));
  // Appends tensor signature name.
  if (signature_name_map.has_value()) {
    auto name_it = signature_name_map->find(tensor_index);
    if (name_it != signature_name_map->end()) {
      RETURN_IF_ERROR(builder.AppendAttrToMetadata(
          edge_type, metadata_id, kSignatureName, name_it->second));
    }
  }
  return absl::OkStatus();
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
  const auto shaped_type = type.dyn_cast<mlir::RankedTensorType>();
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
  if (const auto float_type = elem_type.dyn_cast<mlir::FloatType>()) {
    ASSIGN_OR_RETURN(value, mlir::TFL::ConvertFloatBuffer(shaped_type, buffer));
  } else if (elem_type.isa<mlir::IntegerType>()) {
    ASSIGN_OR_RETURN(value, mlir::TFL::ConvertIntBuffer(shaped_type, buffer));
  } else if (elem_type.isa<mlir::TF::StringType>()) {
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

absl::Status AddConstantToNodeAttr(const TensorT& tensor,
                                   const std::vector<uint8_t>& buffer,
                                   const int const_element_count_limit,
                                   mlir::Builder mlir_builder,
                                   GraphNodeBuilder& builder) {
  ASSIGN_OR_RETURN(mlir::ElementsAttr elem_attr,
                   ConvertBufferToAttr(tensor, buffer, mlir_builder));
  std::string value;
  llvm::raw_string_ostream sstream(value);
  PrintAttribute(elem_attr, const_element_count_limit, sstream);
  builder.AppendNodeAttribute(/*key=*/kValue, /*value=*/value);
  return absl::OkStatus();
}

std::vector<uint8_t> GetBuffer(
    const TensorT& tensor, const Buffers& buffers,
    const std::unique_ptr<FlatBufferModel>& model_ptr) {
  // Check if constant tensor is stored outside of the flatbuffers.
  if (tflite::IsValidBufferOffset(buffers[tensor.buffer]->offset)) {
    const uint8_t* file_begin_ptr =
        reinterpret_cast<const uint8_t*>(model_ptr->allocation()->base());
    return std::vector<uint8_t>(file_begin_ptr + buffers[tensor.buffer]->offset,
                                file_begin_ptr +
                                    buffers[tensor.buffer]->offset +
                                    buffers[tensor.buffer]->size);
  }
  return buffers[tensor.buffer]->data;
}

// Creates and adds a GraphInputs, GraphOutputs or const node into Subgraph.
absl::Status AddAuxiliaryNode(
    const NodeType node_type, const std::vector<int>& tensor_indices,
    const Tensors& tensors, const Buffers& buffers,
    const std::optional<const SignatureNameMap>& signature_name_map,
    const std::unique_ptr<FlatBufferModel>& model_ptr,
    const int const_element_count_limit, std::vector<std::string>& node_ids,
    EdgeMap& edge_map, mlir::Builder mlir_builder, Subgraph& subgraph) {
  if (tensor_indices.empty()) {
    return absl::InvalidArgumentError("Tensor indices cannot be empty.");
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
      ASSIGN_OR_RETURN(node_name, GenerateNodeName(node_id_str, node_label,
                                                   tensor_indices, tensors));
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
    std::vector<uint8_t> buffer = GetBuffer(tensor, buffers, model_ptr);
    absl::Status status = AddConstantToNodeAttr(
        tensor, buffer, const_element_count_limit, mlir_builder, builder);
    // Logs the error and continues to add the node to the graph.
    if (!status.ok()) {
      LOG(ERROR) << status;
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
      RETURN_IF_ERROR(AppendMetadata(EdgeType::kOutput, i, tensor_index,
                                     tensors, signature_name_map, builder));

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
    const std::string& custom_code, const std::vector<uint8_t>& custom_options,
    mlir::Builder mlir_builder,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes) {
  attributes.emplace_back(mlir_builder.getNamedAttr(
      "custom_code", mlir_builder.getStringAttr(custom_code)));
  std::string content;
  content.assign(reinterpret_cast<const char*>(custom_options.data()),
                 custom_options.size());
  attributes.emplace_back(mlir_builder.getNamedAttr(
      "custom_option",
      mlir::TFL::ConstBytesAttr::get(mlir_builder.getContext(), content)));
}

// Adds builtin and custom options to node attribute. Logic is referred from
// `ConvertOp` in tensorflow/compiler/mlir/lite/flatbuffer_import.cc.
absl::Status AddOptionsToNodeAttribute(
    const OperatorT& op, const OperatorCodes& op_codes,
    const std::unique_ptr<FlatBufferModel>& model_ptr,
    mlir::Builder mlir_builder, GraphNodeBuilder& builder) {
  llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
  const OperatorCodeT& op_code = *op_codes.at(op.opcode_index);
  const BuiltinOperator builtin_code = tflite::GetBuiltinCode(&op_code);
  if (builtin_code == tflite::BuiltinOperator_CUSTOM) {
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
    CustomOptionsToAttributes(op_code.custom_code, custom_options, mlir_builder,
                              attrs);
  } else {
    mlir::BuiltinOptionsToAttributes(op.builtin_options, mlir_builder, attrs);
    mlir::BuiltinOptions2ToAttributes(op.builtin_options_2, mlir_builder,
                                      attrs);
  }
  std::string value;
  llvm::raw_string_ostream sstream(value);
  for (const auto& attr : attrs) {
    // Sets the limit to -1 since BuiltinOptions don't involve constant weights.
    PrintAttribute(attr.getValue(), /*size_limit=*/-1, sstream);
    builder.AppendNodeAttribute(attr.getName().str(), value);
    value.clear();
  }
  return absl::OkStatus();
}

// Adds a node to Subgraph.
absl::Status AddNode(
    const int node_index, const OperatorT& op, const OperatorCodes& op_codes,
    const std::vector<std::string>& op_names, const Tensors& tensors,
    const Buffers& buffers,
    const std::optional<const SignatureNameMap>& signature_name_map,
    const std::unique_ptr<FlatBufferModel>& model_ptr,
    const int const_element_count_limit, std::vector<std::string>& node_ids,
    EdgeMap& edge_map, mlir::Builder mlir_builder, Subgraph& subgraph) {
  if (op.opcode_index >= op_names.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Opcode index ", op.opcode_index,
                     " matches no op name for node ", node_index));
  }
  const std::string node_id_str = node_ids[node_index];
  absl::string_view node_label = op_names[op.opcode_index];
  ASSIGN_OR_RETURN(
      const std::string node_name,
      GenerateNodeName(node_id_str, node_label, op.outputs, tensors));
  GraphNodeBuilder builder;
  builder.SetNodeInfo(node_id_str, node_label, node_name);
  // Logs the error and continues to add the node to the graph.
  absl::Status status =
      AddOptionsToNodeAttribute(op, op_codes, model_ptr, mlir_builder, builder);
  if (!status.ok()) {
    LOG(ERROR) << status;
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
          NodeType::kConstNode, std::vector<int>{tensor_index}, tensors,
          buffers, signature_name_map, model_ptr, const_element_count_limit,
          node_ids, edge_map, mlir_builder, subgraph));
    }
    AppendIncomingEdge(edge_map.at(tensor_index), builder);
  }

  for (int i = 0; i < op.outputs.size(); ++i) {
    const int tensor_index = op.outputs[i];
    RETURN_IF_ERROR(AppendMetadata(EdgeType::kOutput, i, tensor_index, tensors,
                                   signature_name_map, builder));
    PopulateEdgeInfo(tensor_index,
                     {.source_node_id = node_id_str,
                      .source_node_output_id = absl::StrCat(i)},
                     edge_map);
  }

  subgraph.nodes.push_back(std::move(builder).Build());
  return absl::OkStatus();
}

void ValidateSubgraph(const std::vector<std::string>& node_ids,
                      const EdgeMap& edge_map) {
  absl::flat_hash_set<std::string> node_ids_set(node_ids.begin(),
                                                node_ids.end());
  if (node_ids_set.size() != node_ids.size()) {
    LOG(INFO) << "Node ids: " << absl::StrJoin(node_ids, ",");
    LOG(ERROR) << "Node ids are not unique.";
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
    LOG(ERROR) << "EdgeMap has incomplete EdgeInfo.";
  }
}

// Adds a subgraph to Graph.
absl::Status AddSubgraph(
    const VisualizeConfig& config, const SubGraphT& subgraph_t,
    const std::vector<std::string>& op_names,
    const std::optional<const SignatureNameMap>& signature_name_map,
    const std::unique_ptr<ModelT>& model,
    const std::unique_ptr<FlatBufferModel>& model_ptr,
    mlir::Builder mlir_builder, Graph& graph) {
  // Creates a Model Explorer subgraph.
  Subgraph subgraph(subgraph_t.name);
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
                            signature_name_map, model_ptr,
                            config.const_element_count_limit, node_ids,
                            edge_map, mlir_builder, subgraph));
  }

  // Adds GraphOutputs node to the subgraph.
  RETURN_IF_ERROR(AddAuxiliaryNode(
      NodeType::kOutputNode, subgraph_t.outputs, subgraph_t.tensors, buffers,
      signature_name_map, model_ptr, config.const_element_count_limit, node_ids,
      edge_map, mlir_builder, subgraph));

  ValidateSubgraph(node_ids, edge_map);
  graph.subgraphs.push_back(subgraph);
  return absl::OkStatus();
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

  for (int i = 0; i < model->subgraphs.size(); ++i) {
    const auto& subgraph = model->subgraphs[i];
    auto signature_name_it = signature_map.find(i);
    if (signature_name_it != signature_map.end()) {
      RETURN_IF_ERROR(AddSubgraph(config, *subgraph, op_names,
                                  signature_name_it->second, model, model_ptr,
                                  mlir_builder, graph));
    } else {
      RETURN_IF_ERROR(AddSubgraph(config, *subgraph, op_names,
                                  /*signature_name_map=*/std::nullopt, model,
                                  model_ptr, mlir_builder, graph));
    }
  }

  llvm::json::Value json_result = llvm::json::Value(graph.Json());
  return llvm::formatv("{0:2}", json_result);
}

}  // namespace visualization_client
}  // namespace tooling
