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

#include "translate_helpers.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/flexbuffers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
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
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "formats/schema_structs.h"
#include "graphnode_builder.h"
#include "status_macros.h"
#include "tools/attribute_printer.h"
#include "tools/load_opdefs.h"
#include "tools/namespace_heuristics.h"
#include "tools/shardy_utils.h"
#include "visualize_config.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

namespace tooling {
namespace visualization_client {

namespace {
using ::mlir::Operation;
using ::mlir::func::FuncOp;
using ::mlir::TFL::ConstBytesAttr;

using OpToNodeIdMap = absl::flat_hash_map<Operation*, std::string>;
using InputValueToNodeIdMap = llvm::SmallDenseMap<mlir::Value, std::string>;
using OpdefsMap = absl::flat_hash_map<std::string, OpMetadata>;

inline constexpr llvm::StringLiteral kEmptyString("");
inline constexpr llvm::StringLiteral kConfigProto("config_proto");
inline constexpr llvm::StringLiteral kGraphInputs("Inputs");
inline constexpr llvm::StringLiteral kGraphOutputs("Outputs");
inline constexpr llvm::StringLiteral kTensorName("tensor_name");
inline constexpr llvm::StringLiteral kTensorIndex("tensor_index");
inline constexpr llvm::StringLiteral kTensorShape("tensor_shape");
inline constexpr llvm::StringLiteral kTensorTag("__tensor_tag");
inline constexpr llvm::StringLiteral kValue("__value");

class Counter {
 public:
  int increment() { return count_++; }
  int getValue() const { return count_; }

 private:
  int count_ = 0;
};

// The context maintained during the graph building process.
struct GraphBuildContext {
  GraphBuildContext(const OpdefsMap& op_defs, const OpToNodeIdMap& seen_ops,
                    const InputValueToNodeIdMap& inputs, Counter node_counter,
                    Counter tensor_counter, bool has_debug_info)
      : op_defs(op_defs),
        seen_ops(seen_ops),
        inputs(inputs),
        node_counter(node_counter),
        tensor_counter(tensor_counter),
        has_debug_info(has_debug_info) {}

  GraphBuildContext() = default;

  // A map from Model Explorer node label to OpMetadata.
  OpdefsMap op_defs;
  // A map from MLIR Operation to Model Explorer node ID.
  OpToNodeIdMap seen_ops;
  // A map from a function input (MLIR) value to Model Explorer node ID.
  InputValueToNodeIdMap inputs;
  Counter node_counter;
  Counter tensor_counter;
  bool has_debug_info;
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

inline bool IsTfliteDialect(Operation& operation) {
  return llvm::isa<mlir::TFL::TensorFlowLiteDialect>(operation.getDialect());
}

inline bool IsTfDialect(Operation& operation) {
  return llvm::isa<mlir::TF::TensorFlowDialect>(operation.getDialect());
}

inline bool IsStablehloDialect(Operation& operation) {
  return llvm::isa<mlir::stablehlo::StablehloDialect>(operation.getDialect());
}

inline bool IsShardyDialect(Operation& operation) {
  return llvm::isa<mlir::sdy::SdyDialect>(operation.getDialect());
}

inline bool IsTosaDialect(Operation& operation) {
  return llvm::isa<mlir::tosa::TosaDialect>(operation.getDialect());
}

// Sets the contract for generating the node id for a block argument.
// If `parent_node_id` is empty, it means the block argument is in the graph
// level. Otherwise, it means the block argument is in a nested region, we add
// the parent node id as a prefix to distinguish it from the graph level block
// argument.
std::string GetArgNodeId(int arg_index, absl::string_view parent_node_id) {
  if (parent_node_id.empty()) {
    return absl::StrFormat("arg%d", arg_index);
  }
  return absl::StrFormat("pid%s_arg%d", parent_node_id, arg_index);
}

void AddCustomOptions(const ConstBytesAttr& const_bytes_attr,
                      GraphNodeBuilder& builder) {
  llvm::StringRef bytes = const_bytes_attr.getValue();
  std::vector<uint8_t> custom_options;
  custom_options.assign(bytes.begin(), bytes.end());
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
    builder.AppendNodeAttribute("custom_options", "<non-deserializable>");
    return;
  }
  const flexbuffers::TypedVector& keys = map.Keys();
  for (size_t i = 0; i < keys.size(); ++i) {
    const char* key = keys[i].AsKey();
    const flexbuffers::Reference& value = map[key];
    builder.AppendNodeAttribute(key, value.ToString());
  }
}

// Adds attributes from block arguments to the graph node.
void AddBlockArgAttrs(const VisualizeConfig& config,
                      const mlir::DictionaryAttr& arg_attr_dict,
                      GraphNodeBuilder& builder) {
  if (arg_attr_dict == nullptr) {
    return;
  }
  for (const mlir::NamedAttribute& attr : arg_attr_dict) {
    std::string value;
    llvm::raw_string_ostream sstream(value);
    const llvm::StringRef name = attr.getName();
    const mlir::Attribute attr_val = attr.getValue();
    if (SkipAttr(name)) {
      continue;
    }
    PrintAttribute(attr_val, config.const_element_count_limit, sstream);
    builder.AppendNodeAttribute(name, value);
  }
}

// Appends node attributes to the graph node builder.
void AppendNodeAttrs(const VisualizeConfig& config, Operation& operation,
                     GraphNodeBuilder& builder) {
  std::string value;
  llvm::raw_string_ostream sstream(value);
  llvm::SmallDenseMap<llvm::StringRef, mlir::sdy::MeshAttr> sdy_meshes;
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
    } else if (const auto const_bytes_attr =
                   llvm::dyn_cast_or_null<ConstBytesAttr>(attr_val);
               const_bytes_attr != nullptr) {
      AddCustomOptions(const_bytes_attr, builder);
      // Skips adding the const bytes attribute to the graph.
      continue;
    }
    PrintAttribute(attr_val, config.const_element_count_limit, sstream);
    if (tooling::visualization_client::IsShardyDialect(attr_val)) {
      AddReferencedMesh(attr_val, sdy_meshes, operation);
    }
    if (name == "value") {
      // Special handles `value` attribute to represent the tensor data.
      builder.AppendNodeAttribute(kValue, value);
    } else {
      builder.AppendNodeAttribute(name, value);
    }
    value.clear();
  }
  if (!sdy_meshes.empty()) {
    std::string mesh_attr;
    llvm::raw_string_ostream attr_sstream(mesh_attr);
    llvm::interleave(
        sdy_meshes, attr_sstream,
        [&](const std::pair<llvm::StringRef, mlir::sdy::MeshAttr>& mesh_entry) {
          attr_sstream << mesh_entry.first << ": ";
          mesh_entry.second.printStripped(attr_sstream);
        },
        "\n");
    builder.AppendNodeAttribute("sdy.meshes", mesh_attr);
  }
}

// Gets the node name (hierarchical info) from a JAX operation and stores the
// JAX op attributes if there exists.
void AddJaxNodeNameAndAttribute(Operation& operation,
                                GraphNodeBuilder& builder) {
  mlir::Location loc = operation.getLoc();

  // If the location is invalid or if it's an UnknownLoc, there's no name to
  // find, we exit early.
  if (llvm::isa<mlir::UnknownLoc>(loc)) {
    builder.SetNodeName(/*node_name=*/kEmptyString);
    return;
  }

  // Now that we know 'loc' is valid, find the NameLoc. If it doesn't exist,
  // we exit early.
  auto name_loc = loc->findInstanceOf<mlir::NameLoc>();
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

void TfliteMaybeAppendSubgraphs(Operation& operation,
                                GraphNodeBuilder& builder) {
  if (operation.getNumRegions() == 0) {
    return;
  }

  if (auto while_op = llvm::dyn_cast<mlir::TFL::WhileOp>(operation)) {
    // TODO(b/311011560): Current only tfl.while op is supported. Support more
    // ops that have nested regions.
    for (auto& region : while_op->getRegions()) {
      for (auto& nested_op : region.getOps()) {
        if (auto call = llvm::dyn_cast<mlir::func::CallOp>(nested_op)) {
          builder.AppendSubgraphId(call.getCallee());
          break;
        }
      }
    }
  }
  llvm::errs() << absl::StrCat(
      "Operation: ", operation.getName().getStringRef().str(),
      " has unsupported nested regions.");
}

// Adds the block arguments of a function op to the subgraph.
// Each block argument is represented as a node under the namespace of
// `GraphInputs`.
void AddGraphInputs(const VisualizeConfig& config, mlir::func::FuncOp& fop,
                    GraphBuildContext& context, Subgraph& subgraph) {
  Counter& tensor_counter = context.tensor_counter;

  // Gets the input names from the `tf.entry_function` attribute.
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
    GraphNodeBuilder builder;
    const std::string node_id = GetArgNodeId(it.index(), /*parent_node_id=*/"");
    context.inputs[fop.getArgument(it.index())] = node_id;
    const std::string node_label = absl::StrFormat("input_%d", it.index());
    builder.SetNodeInfo(/*node_id_str=*/node_id, /*node_label=*/node_label,
                        /*node_name=*/kGraphInputs);
    // Since each block argument is represented as a node, we only need to add
    // to the first output metadata.
    builder.AppendAttrToMetadata(EdgeType::kOutput, /*metadata_id=*/0,
                                 kTensorIndex,
                                 absl::StrCat(tensor_counter.increment()));
    if (it.index() < input_names.size()) {
      builder.AppendAttrToMetadata(EdgeType::kOutput, /*metadata_id=*/0,
                                   kTensorName, input_names[it.index()]);
    }
    builder.AppendAttrToMetadata(EdgeType::kOutput, /*metadata_id=*/0,
                                 kTensorShape, GetTypeString(it.value()));

    AddBlockArgAttrs(config, fop.getArgAttrDict(it.index()), builder);
    subgraph.nodes.push_back(std::move(builder).Build());
  }
}

// Gets the node name (the hierarchical information of the node) from a tf
// dialect operation.
llvm::StringRef GetTfNodeName(Operation& operation) {
  mlir::Location loc = operation.getLoc();
  // TF op stores the node name in a FusedLoc.
  auto fused_loc = loc->findInstanceOf<mlir::FusedLoc>();
  if (fused_loc == nullptr) {
    return kEmptyString;
  }

  const llvm::ArrayRef<mlir::Location> locations = fused_loc.getLocations();
  if (locations.empty()) {
    return kEmptyString;
  }

  // The last location in the FusedLoc is the node name.
  auto name_loc = llvm::dyn_cast<mlir::NameLoc>(locations.back());
  if (name_loc == nullptr) {
    return kEmptyString;
  }

  // TF always generates FusedLoc for debug info, and the last element would
  // either look like "node_name@function_name" or simply "node_name"
  // when the op is in main graph where the function_name would be empty.
  llvm::StringRef loc_info = name_loc.getName();
  auto end_pos = loc_info.find('@');
  return loc_info.substr(0, end_pos);
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

// Generates the node name (the hierarchical information of the node) from a sdy
// dialect operation.
void GenerateShardyNodeName(Operation& operation, GraphNodeBuilder& builder) {
  // Use the default JAX logic for setting hierarchical node name information.
  AddJaxNodeNameAndAttribute(operation, builder);
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
absl::Status PopulateInputEdgeInfo(const mlir::Value& val, int input_index,
                                   GraphBuildContext& context,
                                   GraphNodeBuilder& builder) {
  const OpToNodeIdMap& seen_ops = context.seen_ops;
  // In general, there are two types of edges:
  // 1. An edge that comes from a source node
  // 2. An edge that comes from model inputs (or block arguments)
  // For the case one, we look up the source node to populate the edge
  // information. If it's the latter case, we populate the edge with block
  // arguments data.
  std::string source_node_id_str, source_node_output_idx_str;
  Operation* source_node = val.getDefiningOp();
  if (source_node != nullptr) {
    source_node_id_str = seen_ops.find(source_node)->second;
    source_node_output_idx_str =
        absl::StrCat(llvm::dyn_cast<mlir::OpResult>(val).getResultNumber());
  } else {
    auto block_arg = mlir::dyn_cast_or_null<mlir::BlockArgument>(val);
    if (block_arg != nullptr) {
      Operation* parent_op = block_arg.getOwner()->getParentOp();
      std::string parent_node_id;
      if (seen_ops.contains(parent_op)) {
        parent_node_id = seen_ops.find(parent_op)->second;
      }
      source_node_id_str =
          GetArgNodeId(block_arg.getArgNumber(), parent_node_id);
      // Since each block argument is represented as a node, we can hardcode the
      // output index to be 0.
      source_node_output_idx_str = "0";
    } else {
      return absl::InvalidArgumentError(absl::StrCat(
          "Value at index ", input_index, " of node ", builder.GetNodeId(),
          " is neither a block argument nor output edge of a node."));
    }
  }
  builder.AppendEdgeInfo(source_node_id_str, source_node_output_idx_str,
                         absl::StrCat(input_index));
  return absl::OkStatus();
}

// Adds tensor tags to the graph node.
// If the op name is not found in the op_defs map, we will return an error.
// Likely it means the op_defs map is not up to date, and we need to update it.
void AddTensorTags(const OpdefsMap& op_defs, Operation& op,
                   GraphNodeBuilder& builder) {
  const std::string op_label = builder.GetNodeLabel();
  if (!op_defs.contains(op_label)) {
    // Some ops are not in the op_defs map, we will skip adding tensor tags for
    // them.
    return;
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
}

// Adds the basic node information to the graph node, including node id, label,
// and name.
void AddNodeInfo(Operation& operation, GraphBuildContext& context,
                 GraphNodeBuilder& builder) {
  Counter& node_counter = context.node_counter;
  const std::string node_id_str = absl::StrCat(node_counter.increment());
  // A terminator operation is the function return value, or GraphOutputs.
  if (operation.hasTrait<mlir::OpTrait::IsTerminator>()) {
    builder.SetNodeInfo(/*node_id_str=*/node_id_str,
                        /*node_label=*/kGraphOutputs,
                        /*node_name=*/kGraphOutputs);
    return;
  }
  // Node label is the op name without the dialect prefix.
  absl::string_view node_label = operation.getName().stripDialect();

  // Node name is retrieved from the operation, and is varied by the dialect.
  std::string node_name;
  if (IsTfDialect(operation)) {
    node_name = GetTfNodeName(operation);
    builder.SetNodeInfo(node_id_str, node_label, node_name);
    return;
  }
  if (IsTfliteDialect(operation)) {
    node_name = GenerateTfliteNodeName(node_label, operation);
    builder.SetNodeInfo(node_id_str, node_label, node_name);
    return;
  }
  if (IsShardyDialect(operation)) {
    GenerateShardyNodeName(operation, builder);
    builder.SetNodeId(node_id_str);
    builder.SetNodeLabel(operation.getName().getStringRef());
    return;
  }
  // For any other dialects, We apply the parsing logic as stablehlo ops by
  // default. New dialects should add their own parsing logic above.
  // `node_name` will be added within `AddJaxNodeNameAndAttribute` along with
  // the `jax_op_attr` attribute.
  AddJaxNodeNameAndAttribute(operation, builder);

  // We keep the op dialect prefix in the node label.
  builder.SetNodeId(node_id_str);
  node_label = operation.getName().getStringRef();
  builder.SetNodeLabel(node_label);
}

// Iterates through all operand values of an operation and adds the incoming
// edges to the graph node.
absl::Status AddIncomingEdges(Operation& operation, GraphBuildContext& context,
                              GraphNodeBuilder& builder) {
  for (int input_index = 0, e = operation.getNumOperands(); input_index < e;
       ++input_index) {
    mlir::Value val = operation.getOperand(input_index);
    RETURN_IF_ERROR(PopulateInputEdgeInfo(val, input_index, context, builder));
  }
  return absl::OkStatus();
}

// Iterates through all result values of an operation and adds the output
// metadata to the graph node.
void AddOutputsMetadata(const VisualizeConfig& config, Operation& operation,
                        GraphBuildContext& context, GraphNodeBuilder& builder) {
  Counter& tensor_counter = context.tensor_counter;
  llvm::SmallVector<llvm::StringRef, 2> tensor_names;
  if (IsTfliteDialect(operation) || config.add_tensor_name_attribute) {
    tensor_names = GetTfliteTensorNames(operation);
  }
  for (int output_index = 0, e = operation.getNumResults(); output_index < e;
       ++output_index) {
    builder.AppendAttrToMetadata(EdgeType::kOutput, output_index, kTensorIndex,
                                 absl::StrCat(tensor_counter.increment()));
    mlir::Value val = operation.getResult(output_index);
    builder.AppendAttrToMetadata(EdgeType::kOutput, output_index, kTensorShape,
                                 GetTypeString(val.getType()));
    if (output_index < tensor_names.size()) {
      builder.AppendAttrToMetadata(EdgeType::kOutput, output_index, kTensorName,
                                   tensor_names[output_index]);
    }
  }
}

// Forward declaration of the nested region function.
absl::Status MaybeAddNestedRegion(const VisualizeConfig& config,
                                  Operation& operation,
                                  GraphBuildContext& context,
                                  GraphNodeBuilder& cur_node,
                                  Subgraph& subgraph);

// Adds a node of a nested region to the graph.
// The root_name is used to create the corresponding namespace for the node.
absl::Status AddNestedRegionNode(const VisualizeConfig& config,
                                 absl::string_view root_name,
                                 Operation& operation,
                                 GraphBuildContext& context,
                                 Subgraph& subgraph) {
  GraphNodeBuilder builder;
  OpToNodeIdMap& seen_ops = context.seen_ops;
  AddNodeInfo(operation, context, builder);
  // Updates the node's namespace if the node name doesn't contain "/" to put it
  // under the corresponding region.
  if (!absl::StrContains(builder.GetNodeName(), '/')) {
    builder.SetNodeName(root_name);
  }
  seen_ops.insert({&operation, builder.GetNodeId()});
  AppendNodeAttrs(config, operation, builder);
  RETURN_IF_ERROR(AddIncomingEdges(operation, context, builder));
  if (IsShardyDialect(operation)) {
    // Add multi-layered nested regions for SDY ops if they exist.
    RETURN_IF_ERROR(
        MaybeAddNestedRegion(config, operation, context, builder, subgraph));
  }
  AddOutputsMetadata(config, operation, context, builder);
  subgraph.nodes.push_back(std::move(builder).Build());
  return absl::OkStatus();
}

// Determines the base namespace for a nested region group.
//
// Ops with nested regions are grouped within their own namespace, and the op is
// pinned to the top of this group. A unique namespace is generated using the
// node's label and ID. If the op is already within a parent namespace, the
// generated namespace is appended to the parent's namespace. This generated
// namespace is also used when no debug info is available.
std::string DetermineBaseNamespace(const GraphBuildContext& context,
                                   const GraphNodeBuilder& cur_node) {
  const std::string generated_namespace_part =
      absl::StrFormat("(%s_%s)", cur_node.GetNodeLabel(), cur_node.GetNodeId());

  if (cur_node.GetNodeName().empty()) {
    // If the current node doesn't already have a name (e.g., from debug info),
    // use the generated part as its namespace. This applies whether debug info
    // is present or not, as there's no existing name to take precedence.
    return generated_namespace_part;
  }
  // The current node already has a name (likely from parent scope or debug
  // info).
  if (context.has_debug_info) {
    // If debug information is present, it dictates the namespace and the
    // existing node name is used directly.
    // TODO(b/421175189): Note that for nested region ops, this may not always
    // group ops as intended if the debug info doesn't reflect the nested
    // hierarchy.
    return cur_node.GetNodeName();
  }
  // No debug information, but there's an existing node name (parent scope).
  // This is where the special handling for nested SDY ops applies.
  // Create a new, deeper namespace by appending the generated part to the
  // existing node name. This ensures the op is grouped within its own
  // sub-namespace.
  return absl::StrFormat("%s/%s", cur_node.GetNodeName(),
                         generated_namespace_part);
}

// Adds graph nodes for all operands of an operation entering a nested region.
void AddNestedRegionInputs(absl::string_view base_namespace,
                           Operation& operation, GraphBuildContext& context,
                           GraphNodeBuilder& cur_node, Subgraph& subgraph) {
  Counter& tensor_counter = context.tensor_counter;

  // Since this node has nested regions, we pin it to the top of the group.
  cur_node.SetNodeName(base_namespace);
  cur_node.SetPinToGroupTop(true);

  // Adds the input nodes of the nested region.
  for (int input_index = 0, e = operation.getNumOperands(); input_index < e;
       ++input_index) {
    GraphNodeBuilder builder;
    const std::string node_label = absl::StrFormat("input_%d", input_index);
    const std::string node_id = GetArgNodeId(input_index, cur_node.GetNodeId());
    const std::string node_name =
        absl::StrFormat("%s/%s", base_namespace, kGraphInputs);
    builder.SetNodeInfo(/*node_id_str=*/node_id, /*node_label=*/node_label,
                        /*node_name=*/node_name);
    // Since each input is represented as a node, we can hardcode the output
    // index to be 0.
    builder.AppendAttrToMetadata(EdgeType::kOutput, /*metadata_id=*/0,
                                 kTensorIndex,
                                 absl::StrCat(tensor_counter.increment()));
    builder.AppendAttrToMetadata(
        EdgeType::kOutput, /*metadata_id=*/0, kTensorShape,
        GetTypeString(operation.getOperand(input_index).getType()));
    subgraph.nodes.push_back(std::move(builder).Build());
  }
}

// Processes all generic or unknown regions of an operation.
absl::Status ProcessGenericRegions(
    const std::function<absl::Status(absl::string_view, mlir::Region&)>&
        process_region,
    Operation& op) {
  for (int i = 0, e = op.getNumRegions(); i < e; ++i) {
    const std::string generic_name = absl::StrCat("(region_", i, ")");
    RETURN_IF_ERROR(process_region(generic_name, op.getRegion(i)));
  }
  return absl::OkStatus();
}

// Processes the nested regions of a Stablehlo operation. Returns true if the
// operation's regions were processed, false otherwise.
absl::StatusOr<bool> ProcessStablehloRegions(
    const std::function<absl::Status(absl::string_view, mlir::Region&)>
        process_region,
    Operation& op) {
  return llvm::TypeSwitch<Operation*, absl::StatusOr<bool>>(&op)
      .Case<mlir::stablehlo::WhileOp>(
          [&](mlir::stablehlo::WhileOp op) -> absl::StatusOr<bool> {
            RETURN_IF_ERROR(process_region("cond", op.getCond()));
            RETURN_IF_ERROR(process_region("body", op.getBody()));
            return true;
          })
      .Case<mlir::stablehlo::IfOp>(
          [&](mlir::stablehlo::IfOp op) -> absl::StatusOr<bool> {
            RETURN_IF_ERROR(process_region("true_branch", op.getTrueBranch()));
            RETURN_IF_ERROR(
                process_region("false_branch", op.getFalseBranch()));
            return true;
          })
      .Case<mlir::stablehlo::AllReduceOp>(
          [&](mlir::stablehlo::AllReduceOp op) -> absl::StatusOr<bool> {
            RETURN_IF_ERROR(process_region("computation", op.getComputation()));
            return true;
          })
      .Case<mlir::stablehlo::ReduceScatterOp>(
          [&](mlir::stablehlo::ReduceScatterOp op) -> absl::StatusOr<bool> {
            RETURN_IF_ERROR(process_region("computation", op.getComputation()));
            return true;
          })
      .Case<mlir::stablehlo::ReduceOp>(
          [&](mlir::stablehlo::ReduceOp op) -> absl::StatusOr<bool> {
            RETURN_IF_ERROR(process_region("body", op.getBody()));
            return true;
          })
      .Case<mlir::stablehlo::MapOp>(
          [&](mlir::stablehlo::MapOp op) -> absl::StatusOr<bool> {
            RETURN_IF_ERROR(process_region("computation", op.getComputation()));
            return true;
          })
      .Case<mlir::stablehlo::ScatterOp>(
          [&](mlir::stablehlo::ScatterOp op) -> absl::StatusOr<bool> {
            RETURN_IF_ERROR(process_region("update_computation",
                                           op.getUpdateComputation()));
            return true;
          })
      .Case<mlir::stablehlo::SelectAndScatterOp>(
          [&](mlir::stablehlo::SelectAndScatterOp op) -> absl::StatusOr<bool> {
            RETURN_IF_ERROR(process_region("select", op.getSelect()));
            RETURN_IF_ERROR(process_region("scatter", op.getScatter()));
            return true;
          })
      .Case<mlir::stablehlo::SortOp>(
          [&](mlir::stablehlo::SortOp op) -> absl::StatusOr<bool> {
            RETURN_IF_ERROR(process_region("comparator", op.getComparator()));
            return true;
          })
      .Case<mlir::stablehlo::ReduceWindowOp>(
          [&](mlir::stablehlo::ReduceWindowOp op) -> absl::StatusOr<bool> {
            RETURN_IF_ERROR(process_region("body", op.getBody()));
            return true;
          })
      .Default([](Operation* op) -> absl::StatusOr<bool> {
        // If the op is not found, we return false to indicate it's not
        // processed.
        return false;
      });
}

// Processes the nested regions of a Shardy operation. Returns true if the
// operation's regions were processed, false otherwise.
absl::StatusOr<bool> ProcessShardyRegions(
    const std::function<absl::Status(absl::string_view, mlir::Region&)>
        process_region,
    Operation& op) {
  return llvm::TypeSwitch<Operation*, absl::StatusOr<bool>>(&op)
      .Case<mlir::sdy::ManualComputationOp>(
          [&](mlir::sdy::ManualComputationOp op) -> absl::StatusOr<bool> {
            RETURN_IF_ERROR(process_region("body", op.getBody()));
            return true;
          })
      .Case<mlir::sdy::NamedComputationOp>(
          [&](mlir::sdy::NamedComputationOp op) -> absl::StatusOr<bool> {
            RETURN_IF_ERROR(process_region("body", op.getBody()));
            return true;
          })
      .Default([](Operation* op) -> absl::StatusOr<bool> {
        // If the op is not found, we return false to indicate it's not
        // processed.
        return false;
      });
}

// Processes the nested regions of a Tosa operation. Returns true if the
// operation's regions were processed, false otherwise.
absl::StatusOr<bool> ProcessTosaRegions(
    const std::function<absl::Status(absl::string_view, mlir::Region&)>
        process_region,
    Operation& op) {
  return llvm::TypeSwitch<Operation*, absl::StatusOr<bool>>(&op)
      .Case<mlir::tosa::IfOp>([&](mlir::tosa::IfOp op) -> absl::StatusOr<bool> {
        RETURN_IF_ERROR(process_region("then_graph", op.getThenGraph()));
        RETURN_IF_ERROR(process_region("else_graph", op.getElseGraph()));
        return true;
      })
      .Case<mlir::tosa::WhileOp>(
          [&](mlir::tosa::WhileOp op) -> absl::StatusOr<bool> {
            RETURN_IF_ERROR(process_region("cond_graph", op.getCondGraph()));
            RETURN_IF_ERROR(process_region("body_graph", op.getBodyGraph()));
            return true;
          })
      .Default([](Operation* op) -> absl::StatusOr<bool> {
        // If the op is not found, we return false to indicate it's not
        // processed.
        return false;
      });
}

// Adds the nested region to the graph.
absl::Status MaybeAddNestedRegion(const VisualizeConfig& config,
                                  Operation& operation,
                                  GraphBuildContext& context,
                                  GraphNodeBuilder& cur_node,
                                  Subgraph& subgraph) {
  if (operation.getNumRegions() == 0) {
    return absl::OkStatus();
  }

  const std::string base_namespace = DetermineBaseNamespace(context, cur_node);
  AddNestedRegionInputs(base_namespace, operation, context, cur_node, subgraph);

  // Processes all ops within the region.
  auto process_region = [&](absl::string_view region_name,
                            mlir::Region& region) -> absl::Status {
    const std::string root_name =
        absl::StrCat(base_namespace, "/", region_name);
    for (auto& op : region.getOps()) {
      RETURN_IF_ERROR(
          AddNestedRegionNode(config, root_name, op, context, subgraph));
    }
    return absl::OkStatus();
  };

  bool region_processed = false;
  if (IsStablehloDialect(operation)) {
    ASSIGN_OR_RETURN(region_processed,
                     ProcessStablehloRegions(process_region, operation));
  } else if (IsShardyDialect(operation)) {
    ASSIGN_OR_RETURN(region_processed,
                     ProcessShardyRegions(process_region, operation));
  } else if (IsTosaDialect(operation)) {
    ASSIGN_OR_RETURN(region_processed,
                     ProcessTosaRegions(process_region, operation));
  }
  if (!region_processed) {
    RETURN_IF_ERROR(ProcessGenericRegions(process_region, operation));
  }
  return absl::OkStatus();
}

// Adds a node to the graph.
// This is the core logic for converting an MLIR operation to a graph node.
absl::Status AddNode(const VisualizeConfig& config, Operation& operation,
                     GraphBuildContext& context, Subgraph& subgraph) {
  GraphNodeBuilder builder;
  const OpdefsMap& op_defs = context.op_defs;
  OpToNodeIdMap& seen_ops = context.seen_ops;
  // Sets the node_id, node_label, and node_name according to the dialect.
  AddNodeInfo(operation, context, builder);
  seen_ops.insert({&operation, builder.GetNodeId()});
  AppendNodeAttrs(config, operation, builder);
  RETURN_IF_ERROR(AddIncomingEdges(operation, context, builder));
  llvm::SmallVector<llvm::StringRef, 2> tensor_names;
  if (IsTfliteDialect(operation)) {
    TfliteMaybeAppendSubgraphs(operation, builder);
    AddTensorTags(op_defs, operation, builder);
  }
  RETURN_IF_ERROR(
      MaybeAddNestedRegion(config, operation, context, builder, subgraph));
  AddOutputsMetadata(config, operation, context, builder);
  subgraph.nodes.push_back(std::move(builder).Build());
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<Subgraph> FuncOpToSubgraph(const VisualizeConfig& config,
                                          FuncOp& fop, bool has_debug_info) {
  Subgraph subgraph(fop.getSymName().str());
  GraphBuildContext context;
  context.has_debug_info = has_debug_info;
  AddGraphInputs(config, fop, context, subgraph);
  mlir::Block& block = fop.getBody().front();
  Operation* first_op = &block.front();
  if (IsTfliteDialect(*first_op)) {
    context.op_defs = LoadTfliteOpdefs();
  }
  for (Operation& operation : block) {
    RETURN_IF_ERROR(AddNode(config, operation, context, subgraph));
  }

  // Extract shardy propagation edges as an edge overlay if present and add
  // to the task data to auto-load in the UI.
  absl::StatusOr<EdgeOverlaysData> propagation_edges =
      ExtractShardyPropagationEdges(fop, context.seen_ops, context.inputs);
  if (!propagation_edges.ok()) {
    LOG(ERROR) << "Failed to extract shardy propagation edges: "
               << propagation_edges.status();
    return subgraph;
  }
  if (!propagation_edges->overlays.empty()) {
    // Set the name of the edge overlay to match the subgraph from which it
    // was extracted, and move it to the subgraph's task data.
    propagation_edges->name = subgraph.subgraph_id;
    TasksData edge_overlays_data;
    edge_overlays_data.edge_overlays_data_list_left_pane.emplace();
    edge_overlays_data.edge_overlays_data_list_left_pane->push_back(
        *std::move(propagation_edges));
    subgraph.tasks_data = std::move(edge_overlays_data);
  }
  return subgraph;
}

bool ModuleHasDebugInfo(Operation* module) {
  bool has_debug_info = false;
  // Determine if module has debug location information by checking for
  // the presence of NameLoc attr on any op.
  module->walk([&](Operation* op) -> mlir::WalkResult {
    if (op->getLoc()->findInstanceOf<mlir::NameLoc>() != nullptr) {
      has_debug_info = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return has_debug_info;
}

absl::StatusOr<Graph> MlirToGraph(const VisualizeConfig& config,
                                  Operation* module) {
  if (!llvm::isa<mlir::ModuleOp>(module)) {
    return absl::InvalidArgumentError("Given module is not valid.");
  }
  Graph graph;
  std::string log_msg;
  const bool has_debug_info = ModuleHasDebugInfo(module);
  auto walk_result = module->walk([&](FuncOp fop) -> mlir::WalkResult {
    llvm::StringRef func_name = fop.getSymName();
    // Avoids serializing "NoOp" function to JSON graph.
    if (func_name == "NoOp") {
      return mlir::WalkResult::skip();
    }

    absl::StatusOr<Subgraph> subgraph =
        FuncOpToSubgraph(config, fop, has_debug_info);

    if (!subgraph.ok()) {
      log_msg = subgraph.status().message();
      return mlir::WalkResult::interrupt();
    }

    graph.subgraphs.push_back(*std::move(subgraph));
    return mlir::WalkResult::advance();
  });

  if (walk_result.wasInterrupted()) {
    return absl::InternalError(log_msg);
  }
  return graph;
}

}  // namespace visualization_client
}  // namespace tooling
