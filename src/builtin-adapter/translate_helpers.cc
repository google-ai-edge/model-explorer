// Copyright 2024 The AI Edge Model Explorer Authors. All Rights Reserved.
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

#include "translate_helpers.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "formats/schema_structs.h"
#include "graphnode_builder.h"
#include "status_macros.h"
#include "tools/attribute_printer.h"
#include "tools/load_opdefs.h"
#include "tools/namespace_heuristics.h"
#include "visualize_config.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

namespace tooling {
namespace visualization_client {

namespace {
using ::mlir::Operation;
using ::mlir::func::FuncOp;
using ::tooling::visualization_client::OpMetadata;

inline constexpr llvm::StringLiteral kEmptyString("");
inline constexpr llvm::StringLiteral kSemicolonSeparator(";");
inline constexpr llvm::StringLiteral kConfigProto("config_proto");
inline constexpr llvm::StringLiteral kGraphInputs("GraphInputs");
inline constexpr llvm::StringLiteral kGraphOutputs("GraphOutputs");
inline constexpr llvm::StringLiteral kTensorName("tensor_name");
inline constexpr llvm::StringLiteral kTensorIndex("tensor_index");
inline constexpr llvm::StringLiteral kTensorShape("tensor_shape");
inline constexpr llvm::StringLiteral kPseudoConst("pseudo_const");
inline constexpr llvm::StringLiteral kTensorTag("__tensor_tag");
inline constexpr llvm::StringLiteral kValue("__value");

using OpdefsMap = absl::flat_hash_map<std::string, OpMetadata>;

class Counter {
 public:
  int increment() { return count_++; }
  int getValue() const { return count_; }

 private:
  int count_ = 0;
};

// Skip serializing attributes that match the given name. This is a hard-code to
// avoid adding binary string attribute into JSON. This disallow list might
// expand as more unsupported attribute found.
inline bool SkipAttr(llvm::StringRef name) { return name == kConfigProto; }

inline std::string GetTypeString(const mlir::Type& t) {
  std::string result;
  llvm::raw_string_ostream ost(result);
  t.print(ost);
  return result;
}

void AppendNodeAttrs(const int const_element_count_limit, Operation& operation,
                     GraphNodeBuilder& builder) {
  std::string value;
  llvm::raw_string_ostream sstream(value);
  for (const mlir::NamedAttribute& attr : operation.getAttrs()) {
    const llvm::StringRef name = attr.getName();
    const mlir::Attribute attr_val = attr.getValue();
    if (SkipAttr(name)) {
      continue;
    }

    // FlatSymbolRefAttr represents the reference to a function call in several
    // tf ops (eg. tf.PartitionedCall, tf.While, tf.If etc). In another word,
    // its value refers to the child subgraph of this parent node.
    if (const auto flat_symbol_attr =
            llvm::dyn_cast_or_null<::mlir::FlatSymbolRefAttr>(attr_val);
        flat_symbol_attr != nullptr) {
      llvm::StringRef subgraph_id = flat_symbol_attr.getValue();
      builder.AppendSubgraphId(subgraph_id);
    }
    PrintAttribute(attr_val, const_element_count_limit, sstream);
    if (name == "value") {
      // Special handles `value` attribute to represent the tensor data.
      builder.AppendNodeAttribute(kValue, value);
    } else {
      builder.AppendNodeAttribute(name, value);
    }
    value.clear();
  }
}

// Gets the node name (hierarchical info) from a JAX operation and stores the
// JAX op attributes if there exists.
void AddJaxNodeNameAndAttribute(Operation& operation,
                                GraphNodeBuilder& builder) {
  auto name_loc = operation.getLoc()->findInstanceOf<mlir::NameLoc>();
  if (name_loc == nullptr) {
    builder.SetNodeName(/*node_name=*/kEmptyString);
    return;
  }

  // Stablehlo MLIR stores ML-framework-related data into debugInfo.
  // For example, all JAX op attributes are stored in debugInfo in a format
  // similar to `jax/function/call/op[attr_a=(1, 1, 1, 4) attr_b=None]`, where
  // all info outside of `[]` is the node name or namespace, and inside bracket
  // are JAX op attributes.
  llvm::SmallVector<llvm::StringRef, 2> loc_vec;
  llvm::StringRef loc_info = name_loc.getName();

  // Splits the loc string into two, first part is op function call, second
  // half is node attribute. If JAX op doesn't have attribute, then there will
  // be only function call (hierarchical info).
  loc_info.split(loc_vec, '[', /*MaxSplit=*/1, /*KeepEmpty=*/false);
  builder.SetNodeName(/*node_name=*/loc_vec[0]);

  if (loc_vec.size() > 1) {
    // Removes the last char ']' from op attribute string.
    llvm::StringRef attr_value = loc_vec[1].substr(0, loc_vec[1].size() - 1);
    builder.AppendNodeAttribute(/*key=*/"jax_op_attr", attr_value);
  }
}

absl::Status TfliteMaybeAppendSubgraphs(Operation& operation,
                                        GraphNodeBuilder& builder) {
  if (operation.getNumRegions() == 0) {
    return absl::OkStatus();
  }

  if (auto while_op = llvm::dyn_cast_or_null<mlir::TFL::WhileOp>(operation)) {
    // TODO(b/311011560): Current only tfl.while op is supported. Support more
    // ops that have nested regions.
    for (auto& region : while_op->getRegions()) {
      for (auto& nested_op : region.getOps()) {
        if (auto call = llvm::dyn_cast_or_null<mlir::func::CallOp>(nested_op)) {
          builder.AppendSubgraphId(call.getCallee());
          break;
        }
      }
    }
  } else {
    std::string err_msg = absl::StrCat(
        "This operation's nested regions are currently not implemented and "
        "won't be serialized to JSON graph yet: ",
        operation.getName().getStringRef().str());
    return absl::UnimplementedError(err_msg);
  }

  return absl::OkStatus();
}

absl::Status StablehloMaybeAppendSubgraphs(Operation& operation) {
  if (operation.getNumRegions() == 0) {
    return absl::OkStatus();
  }

  // TODO(b/309554379): Explore Stablehlo ops that have nested regions. Some ops
  // don't have subgraph and directly include ops within their sub-regions. We
  // need to figure out how to handle those cases.
  std::string err_msg = absl::StrCat(
      "This operation's nested regions are currently not implemented and "
      "won't be serialized to JSON graph yet: ",
      operation.getName().getStringRef().str());
  return absl::UnimplementedError(err_msg);
}

// Adds a GraphInputs node to the subgraph, and adds input names if they exist.
// Returns the GraphInputs node.
absl::Status AddGraphInputsNode(mlir::func::FuncOp& fop, Counter& index_counter,
                                Subgraph& subgraph) {
  GraphNodeBuilder builder;
  builder.SetNodeId(kGraphInputs);
  builder.SetNodeLabel(kGraphInputs);
  llvm::SmallVector<llvm::StringRef, 2> input_names;
  auto dict_attr =
      fop->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
  if (dict_attr != nullptr) {
    auto inputs_str =
        mlir::dyn_cast_or_null<mlir::StringAttr>(dict_attr.get("inputs"));
    if (inputs_str != nullptr) {
      inputs_str.getValue().split(input_names, ',', /*MaxSplit=*/-1,
                                  /*KeepEmpty=*/false);
      if (input_names.size() != fop.getNumArguments()) {
        llvm::errs()
            << "WARNING: number of input names (" << input_names.size()
            << ") != number of arguments (" << fop.getNumArguments()
            << "). Input tensor names are not guaranteed to store in the "
               "correct edge.\n";
      }
    }
  }
  // Iterates over block arguments of this function op and adds tensor types and
  // shapes. If there are names of model inputs, we also add them to metadata.
  for (const auto& it : llvm::enumerate(fop.getArgumentTypes())) {
    builder.AppendAttrToMetadata(EdgeType::kOutput, it.index(), kTensorIndex,
                                 absl::StrCat(index_counter.increment()));
    if (it.index() < input_names.size()) {
      builder.AppendAttrToMetadata(EdgeType::kOutput, it.index(), kTensorName,
                                   input_names[it.index()]);
    }
    builder.AppendAttrToMetadata(EdgeType::kOutput, it.index(), kTensorShape,
                                 GetTypeString(it.value()));
  }
  subgraph.nodes.push_back(std::move(builder).Build());
  return absl::OkStatus();
}

// Gets the node name (the hierarchical information of the node) from a tf
// dialect operation.
llvm::StringRef GetTfNodeName(Operation& operation) {
  // Initializes `node_name` as an empty string literal.
  llvm::StringRef node_name = kEmptyString;
  auto fusedLoc = operation.getLoc()->findInstanceOf<mlir::FusedLoc>();

  // TF always generates FusedLoc for debug info, and the last element would
  // either look like "node_name@function_name" or simply "node_name"
  // when the op is in main graph where the function_name would be empty.
  if (fusedLoc == nullptr) {
    return node_name;
  }
  llvm::StringRef loc_info =
      llvm::dyn_cast<mlir::NameLoc>(fusedLoc.getLocations().back()).getName();
  auto end_pos = loc_info.find('@');
  node_name = loc_info.substr(0, end_pos);

  return node_name;
}

// Generates the node name (the hierarchical information of the node) from a tfl
// dialect operation.
std::string GenerateTfliteNodeName(llvm::StringRef node_label,
                                   Operation& operation) {
  auto fusedLoc = operation.getLoc()->findInstanceOf<mlir::FusedLoc>();
  auto nameLoc = operation.getLoc()->findInstanceOf<mlir::NameLoc>();
  if (fusedLoc == nullptr && nameLoc == nullptr) {
    return "";
  }
  // In TFLite, we store op's output tensor names in location attribute. So it
  // could be either a simple NameLoc of the original node_name; or a special
  // case when an op has multiple output tensors, it creates a FusedLoc to
  // store each tensor names.
  llvm::SmallVector<llvm::StringRef, 2> tensor_names;
  if (nameLoc != nullptr) {
    tensor_names.push_back(nameLoc.getName());
  } else {
    for (const mlir::Location& loc : fusedLoc.getLocations()) {
      tensor_names.push_back(llvm::dyn_cast<mlir::NameLoc>(loc).getName());
    }
  }
  // Some TFLite has fused op names with several hierarchical information
  // concatenated together with semicolons. In this case, we will find the last
  // single node name that contains this node label. If no matching found, we
  // will return the first single node name by default.
  std::vector<std::string> candidate_names;
  for (absl::string_view tensor_name : tensor_names) {
    std::vector<std::string> tmp_names =
        absl::StrSplit(tensor_name, ';', absl::SkipEmpty());
    for (absl::string_view name : tmp_names) {
      candidate_names.push_back(std::string(name));
    }
  }
  return TfliteNodeNamespaceHeuristic(node_label, candidate_names);
}

// Gets a list of output tensor name(s) of an TFLite operation. Returns empty
// list if there are errors or the operation has no output tensors.
llvm::SmallVector<llvm::StringRef, 2> GetTfliteTensorNames(
    Operation& operation) {
  llvm::SmallVector<llvm::StringRef, 2> tensor_names;
  const int num_tensors = operation.getNumResults();
  if (num_tensors == 0) {
    return tensor_names;
  }
  llvm::StringRef op_name = operation.getName().getStringRef();
  auto fusedLoc = operation.getLoc()->findInstanceOf<mlir::FusedLoc>();
  auto nameLoc = operation.getLoc()->findInstanceOf<mlir::NameLoc>();
  if (nameLoc != nullptr) {
    if (num_tensors == 1) {
      tensor_names.push_back(nameLoc.getName());
    } else {
      llvm::errs() << absl::StrCat(
          "ERROR: ", num_tensors,
          " output tensors are expected for operation: ", op_name.str(),
          ", but only 1 is found.\n");
    }
  } else if (fusedLoc != nullptr) {
    int num_locs = fusedLoc.getLocations().size();
    if (num_tensors == num_locs) {
      for (int i = 0; i < num_locs; ++i) {
        tensor_names.push_back(
            llvm::dyn_cast<mlir::NameLoc>(fusedLoc.getLocations()[i])
                .getName());
      }

    } else {
      llvm::errs() << absl::StrCat(
          "ERROR: ", num_tensors,
          " output tensors are expected for operation: ", op_name.str(),
          ", but ", num_locs, " are found.\n");
    }
  } else {
    llvm::errs() << "ERROR: No tensor names are found for operation: "
                 << op_name.str() << "\n";
  }
  return tensor_names;
}

// Populates the input edge information for a given graph node.
absl::Status PopulateInputEdgeInfo(
    const mlir::Value& val,
    const absl::flat_hash_map<Operation*, std::string>& seen_ops,
    int input_index, absl::string_view node_id, Counter& tensor_idx_counter,
    GraphNodeBuilder& builder) {
  // In general, there are two types of edges:
  // 1. An edge that comes from a source node
  // 2. An edge that comes from model inputs (or block arguments)
  // For the case one, we look up the source node to populate the edge
  // information. If it's the latter case, we populate the edge with block
  // arguments data (arg info also stored in dummy node `GraphInputs`).
  std::string source_node_id = "", source_node_output_idx_str = "";
  Operation* source_node = val.getDefiningOp();
  if (source_node != nullptr) {
    source_node_id = seen_ops.find(source_node)->second;
    source_node_output_idx_str =
        absl::StrCat(llvm::dyn_cast<mlir::OpResult>(val).getResultNumber());
  } else {
    auto block_arg = mlir::dyn_cast_or_null<mlir::BlockArgument>(val);
    if (block_arg != nullptr) {
      source_node_id = kGraphInputs.str();
      source_node_output_idx_str = absl::StrCat(block_arg.getArgNumber());
    } else {
      return absl::InvalidArgumentError(absl::StrCat(
          "Value at index ", input_index, " of node ", node_id,
          " is neither a block argument nor output edge of a node."));
    }
  }
  builder.AppendEdgeInfo(source_node_id, source_node_output_idx_str,
                         absl::StrCat(input_index));
  return absl::OkStatus();
}

// Adds tensor tags to the graph node.
// If the op name is not found in the opdefs map, we will return an error.
// Likely it means the opdefs map is not up to date, and we need to update it.
absl::Status AddTensorTags(absl::string_view op_label, const OpdefsMap& op_defs,
                           Operation& op, GraphNodeBuilder& builder) {
  if (!op_defs.contains(op_label)) {
    return absl::InvalidArgumentError(
        absl::StrCat("No op def found for op: ", op_label));
  }
  const OpMetadata& op_metadata = op_defs.at(op_label);
  if (op_metadata.arguments.size() <= op.getNumOperands()) {
    for (int i = 0; i < op_metadata.arguments.size(); ++i) {
      builder.AppendAttrToMetadata(EdgeType::kInput, i, kTensorTag,
                                   op_metadata.arguments[i]);
    }
  }
  if (op_metadata.results.size() <= op.getNumResults()) {
    for (int i = 0; i < op_metadata.results.size(); ++i) {
      builder.AppendAttrToMetadata(EdgeType::kOutput, i, kTensorTag,
                                   op_metadata.results[i]);
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<Subgraph> TfFunctionToSubgraph(const VisualizeConfig& config,
                                              FuncOp& fop) {
  Subgraph subgraph(fop.getSymName().str());
  absl::flat_hash_map<Operation*, std::string> seen_ops;
  Counter tensor_idx_counter;
  RETURN_IF_ERROR(AddGraphInputsNode(fop, tensor_idx_counter, subgraph));

  // Iterate in order across the `Operation`s in the first block of this
  // function (we assume there is only one block within each function). Since we
  // are checking incoming edges for each node, the forward order would
  // guarantee each Operand is processed before processing the Operation itself.
  mlir::Block& block = fop.getBody().front();
  for (Operation& operation : block) {
    std::string node_id, node_label;
    // A terminator operation is the function return value, or GraphOutputs.
    bool is_terminator = operation.hasTrait<mlir::OpTrait::IsTerminator>();
    if (!is_terminator) {
      node_id = absl::StrCat(seen_ops.size());
      node_label = operation.getName().stripDialect();
    } else {
      node_id = kGraphOutputs.str();
      node_label = kGraphOutputs.str();
    }
    seen_ops.insert({&operation, node_id});
    llvm::StringRef node_name = GetTfNodeName(operation);
    GraphNodeBuilder builder;
    builder.SetNodeInfo(node_id, node_label, node_name);
    AppendNodeAttrs(config.const_element_count_limit, operation, builder);
    for (int input_index = 0; input_index < operation.getNumOperands();
         ++input_index) {
      mlir::Value val = operation.getOperand(input_index);
      RETURN_IF_ERROR(PopulateInputEdgeInfo(val, seen_ops, input_index, node_id,
                                            tensor_idx_counter, builder));
    }
    // TODO: b/319035310 - Graph output names are not stored in TF converted
    // JSON graph.
    for (int output_index = 0; output_index < operation.getNumResults();
         ++output_index) {
      builder.AppendAttrToMetadata(
          EdgeType::kOutput, output_index, kTensorIndex,
          absl::StrCat(tensor_idx_counter.increment()));
      mlir::Value val = operation.getResult(output_index);
      builder.AppendAttrToMetadata(EdgeType::kOutput, output_index,
                                   kTensorShape, GetTypeString(val.getType()));
    }
    subgraph.nodes.push_back(std::move(builder).Build());
  }
  return subgraph;
}

absl::StatusOr<Subgraph> TfliteFunctionToSubgraph(const VisualizeConfig& config,
                                                  FuncOp& fop) {
  Subgraph subgraph(fop.getSymName().str());
  absl::flat_hash_map<Operation*, std::string> seen_ops;
  Counter tensor_idx_counter;
  OpdefsMap op_defs = LoadTfliteOpdefs();

  RETURN_IF_ERROR(AddGraphInputsNode(fop, tensor_idx_counter, subgraph));

  // Iterate in order across the `Operation`s in the first block of this
  // function (we assume there is only one block within each function). Since we
  // are checking incoming edges for each node, the forward order would
  // guarantee each Operand is processed before processing the Operation itself.
  mlir::Block& block = fop.getBody().front();
  for (Operation& operation : block) {
    std::string node_id, node_label;
    // A terminator operation is the function return value, or GraphOutputs.
    bool is_terminator = operation.hasTrait<mlir::OpTrait::IsTerminator>();
    if (!is_terminator) {
      node_id = absl::StrCat(seen_ops.size());
      node_label = operation.getName().stripDialect();
    } else {
      node_id = kGraphOutputs.str();
      node_label = kGraphOutputs.str();
    }
    seen_ops.insert({&operation, node_id});
    std::string node_name = GenerateTfliteNodeName(node_label, operation);
    GraphNodeBuilder builder;
    builder.SetNodeInfo(node_id, node_label, node_name);
    AppendNodeAttrs(config.const_element_count_limit, operation, builder);
    absl::Status append_subgraph_status =
        TfliteMaybeAppendSubgraphs(operation, builder);
    if (!append_subgraph_status.ok()) {
      llvm::errs() << append_subgraph_status.message() << "\n";
    }
    for (int input_index = 0; input_index < operation.getNumOperands();
         ++input_index) {
      mlir::Value val = operation.getOperand(input_index);
      // We make the (tflite) assumption that
      // functions bodies are single block, and that the only
      // operations with nested regions are control flow ops. We ignore
      // serializing the nested regions of these ops (they are usually
      // just a func call and yield anyways).
      // It is also worth noting that we must represent subgraphs
      // as being from a single Block for the visualizer to be a sensical
      // experience.
      if (val.isUsedOutsideOfBlock(&block)) continue;

      RETURN_IF_ERROR(PopulateInputEdgeInfo(val, seen_ops, input_index, node_id,
                                            tensor_idx_counter, builder));
    }
    llvm::SmallVector<llvm::StringRef, 2> tensor_names =
        GetTfliteTensorNames(operation);
    for (int output_index = 0; output_index < operation.getNumResults();
         ++output_index) {
      builder.AppendAttrToMetadata(
          EdgeType::kOutput, output_index, kTensorIndex,
          absl::StrCat(tensor_idx_counter.increment()));
      mlir::Value val = operation.getResult(output_index);
      if (output_index < tensor_names.size()) {
        builder.AppendAttrToMetadata(EdgeType::kOutput, output_index,
                                     kTensorName, tensor_names[output_index]);
      }
      builder.AppendAttrToMetadata(EdgeType::kOutput, output_index,
                                   kTensorShape, GetTypeString(val.getType()));
      // TODO(b/293348398): Tensor indices are not matched to indices in
      // Flatbuffer. Further investigation is needed.
    }
    // GraphOutputs node is an auxiliary node and doesn't have tensor tags.
    if (node_label != kGraphOutputs) {
      absl::Status status =
          AddTensorTags(node_label, op_defs, operation, builder);
      if (!status.ok()) {
        llvm::errs() << status.message() << "\n";
      }
    }
    subgraph.nodes.push_back(std::move(builder).Build());
  }
  return subgraph;
}

absl::StatusOr<Subgraph> StablehloFunctionToSubgraph(
    const VisualizeConfig& config, FuncOp& fop) {
  Subgraph subgraph(fop.getSymName().str());
  absl::flat_hash_map<Operation*, std::string> seen_ops;
  Counter tensor_idx_counter;
  RETURN_IF_ERROR(AddGraphInputsNode(fop, tensor_idx_counter, subgraph));

  // Iterate in order across the `Operation`s in the first block of this
  // function (we assume there is only one block within each function). Since we
  // are checking incoming edges for each node, the forward order would
  // guarantee each Operand is processed before processing the Operation itself.
  mlir::Block& block = fop.getBody().front();
  for (Operation& operation : block) {
    std::string node_id, node_label;
    // A terminator operation is the function return value, or GraphOutputs.
    bool is_terminator = operation.hasTrait<mlir::OpTrait::IsTerminator>();
    if (!is_terminator) {
      node_id = absl::StrCat(seen_ops.size());
      // We don't strip the "stablehlo" prefix from the op name to distinguish
      // it from tf dialect.
      node_label = operation.getName().getStringRef();
    } else {
      node_id = kGraphOutputs.str();
      node_label = kGraphOutputs.str();
    }
    seen_ops.insert({&operation, node_id});
    GraphNodeBuilder builder;
    builder.SetNodeId(node_id);
    builder.SetNodeLabel(node_label);
    AddJaxNodeNameAndAttribute(operation, builder);
    AppendNodeAttrs(config.const_element_count_limit, operation, builder);
    absl::Status append_subgraph_status =
        StablehloMaybeAppendSubgraphs(operation);
    if (!append_subgraph_status.ok()) {
      llvm::errs() << append_subgraph_status.message() << "\n";
    }
    for (int input_index = 0; input_index < operation.getNumOperands();
         ++input_index) {
      mlir::Value val = operation.getOperand(input_index);
      // Similar to tflite, we assume stablehlo function bodies are single block
      // and that the only operations with nested regions are control flow ops.
      // We ignore serializing the nested regions of these ops.
      if (val.isUsedOutsideOfBlock(&block)) continue;
      RETURN_IF_ERROR(PopulateInputEdgeInfo(val, seen_ops, input_index, node_id,
                                            tensor_idx_counter, builder));
    }
    // TODO: b/319035310 - Graph output names are not stored in JAX converted
    // JSON graph.
    for (int output_index = 0; output_index < operation.getNumResults();
         ++output_index) {
      builder.AppendAttrToMetadata(
          EdgeType::kOutput, output_index, kTensorIndex,
          absl::StrCat(tensor_idx_counter.increment()));
      mlir::Value val = operation.getResult(output_index);
      builder.AppendAttrToMetadata(EdgeType::kOutput, output_index,
                                   kTensorShape, GetTypeString(val.getType()));
    }
    subgraph.nodes.push_back(std::move(builder).Build());
  }
  return subgraph;
}

absl::StatusOr<Graph> TfMlirToGraph(const VisualizeConfig& config,
                                    Operation* module) {
  if (!llvm::isa<mlir::ModuleOp>(module)) {
    return absl::InvalidArgumentError("Given module is not valid.");
  }
  Graph result;
  auto walk_result = module->walk([&](FuncOp fop) -> mlir::WalkResult {
    llvm::StringRef func_name = fop.getSymName();
    // Avoids serializing "NoOp" function to JSON graph.
    if (func_name == "NoOp") {
      return mlir::WalkResult::skip();
    }
    absl::StatusOr<Subgraph> subgraph = TfFunctionToSubgraph(config, fop);
    if (!subgraph.ok()) {
      llvm::errs() << subgraph.status().message() << "\n";
      return mlir::WalkResult::interrupt();
    }
    result.subgraphs.push_back(*std::move(subgraph));
    return mlir::WalkResult::advance();
  });

  if (walk_result.wasInterrupted()) {
    return absl::InternalError("Module walk interrupted.");
  }
  return result;
}

absl::StatusOr<Graph> TfliteMlirToGraph(const VisualizeConfig& config,
                                        Operation* module) {
  mlir::ModuleOp module_op = llvm::dyn_cast_or_null<mlir::ModuleOp>(module);
  if (!module_op) {
    return absl::InvalidArgumentError("Given module is not valid.");
  }
  Graph result;
  auto walk_result = module->walk([&](FuncOp fop) -> mlir::WalkResult {
    absl::StatusOr<Subgraph> subgraph = TfliteFunctionToSubgraph(config, fop);
    if (!subgraph.ok()) {
      llvm::errs() << subgraph.status().message() << "\n";
      return mlir::WalkResult::interrupt();
    }
    result.subgraphs.push_back(*subgraph);
    return mlir::WalkResult::advance();
  });

  if (walk_result.wasInterrupted()) {
    return absl::InternalError("Module walk interrupted.");
  }
  return result;
}

absl::StatusOr<Graph> JaxConvertedMlirToGraph(const VisualizeConfig& config,
                                              Operation* module) {
  if (!llvm::isa<mlir::ModuleOp>(module)) {
    return absl::InvalidArgumentError("Given module is not valid.");
  }
  Graph result;
  auto walk_result = module->walk([&](FuncOp fop) -> mlir::WalkResult {
    llvm::StringRef func_name = fop.getSymName();
    // Avoids serializing "NoOp" function to JSON graph.
    if (func_name == "NoOp") {
      return mlir::WalkResult::skip();
    }

    // Since the ops in the each function can either be tf or stablehlo dialect,
    // we check the first operation in the function to decide which dialect this
    // function is.
    mlir::Block& block = fop.getBody().front();
    mlir::Operation& first_op = block.front();
    absl::StatusOr<Subgraph> subgraph;
    if (llvm::isa<mlir::TF::TensorFlowDialect>(first_op.getDialect())) {
      subgraph = TfFunctionToSubgraph(config, fop);
      if (!subgraph.ok()) {
        llvm::errs() << subgraph.status().message() << "\n";
        return mlir::WalkResult::interrupt();
      }
    } else if (llvm::isa<mlir::stablehlo::StablehloDialect>(
                   first_op.getDialect())) {
      subgraph = StablehloFunctionToSubgraph(config, fop);
      if (!subgraph.ok()) {
        llvm::errs() << subgraph.status().message() << "\n";
        return mlir::WalkResult::interrupt();
      }
    } else {
      llvm::errs() << "Unknown dialect: "
                   << first_op.getDialect()->getNamespace()
                   << " in function: " << func_name
                   << ", we skip serializing this function.\n";
      return mlir::WalkResult::skip();
    }
    result.subgraphs.push_back(*subgraph);
    return mlir::WalkResult::advance();
  });

  if (walk_result.wasInterrupted()) {
    return absl::InternalError("Module walk interrupted.");
  }
  return result;
}

}  // namespace visualization_client
}  // namespace tooling
