#include "translate_helpers.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
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
#include "stablehlo/dialect/StablehloOps.h"
#include "formats/schema_structs.h"
#include "graphnode_builder.h"
#include "status_macros.h"
#include "tools/attribute_printer.h"
#include "visualize_config.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

namespace tooling {
namespace visualization_client {

namespace {
using ::mlir::Operation;
using ::mlir::func::FuncOp;

inline constexpr llvm::StringLiteral kEmptyString("");
inline constexpr llvm::StringLiteral kSemicolonSeparator(";");
inline constexpr llvm::StringLiteral kConfigProto("config_proto");
inline constexpr llvm::StringLiteral kGraphInputs("GraphInputs");
inline constexpr llvm::StringLiteral kGraphOutputs("GraphOutputs");
inline constexpr llvm::StringLiteral kTensorName("tensor_name");
inline constexpr llvm::StringLiteral kTensorIndex("tensor_index");
inline constexpr llvm::StringLiteral kTensorShape("tensor_shape");
inline constexpr llvm::StringLiteral kPseudoConst("pseudo_const");

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
    builder.AppendNodeAttribute(name, value);
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

  if (loc_vec.size() > 1 && !loc_vec[1].empty()) {
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
        dict_attr.get("inputs").dyn_cast_or_null<mlir::StringAttr>();
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
    RETURN_IF_ERROR(builder.AppendAttrToMetadata(
        EdgeType::kOutput, it.index(), kTensorIndex,
        absl::StrCat(index_counter.increment())));
    if (it.index() < input_names.size()) {
      RETURN_IF_ERROR(builder.AppendAttrToMetadata(
          EdgeType::kOutput, it.index(), kTensorName, input_names[it.index()]));
    }
    RETURN_IF_ERROR(builder.AppendAttrToMetadata(EdgeType::kOutput, it.index(),
                                                 kTensorShape,
                                                 GetTypeString(it.value())));
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

// Gets the node name (the hierarchical information of the node) from a tfl
// dialect operation.
llvm::StringRef GetTfliteNodeName(llvm::StringRef node_id_str,
                                  llvm::StringRef node_label,
                                  Operation& operation) {
  // Initializes `node_name` as an empty string literal.
  llvm::StringRef node_name = kEmptyString;
  auto fusedLoc = operation.getLoc()->findInstanceOf<mlir::FusedLoc>();
  auto nameLoc = operation.getLoc()->findInstanceOf<mlir::NameLoc>();
  if (fusedLoc == nullptr && nameLoc == nullptr) {
    return node_name;
  }
  // In TFLite, we store op's output tensor names in location attribute. So it
  // could be either a simple NameLoc of the original node_name; or a special
  // case when an op has multiple output tensors, it creates a FusedLoc to
  // store each tensor names. The naming of multiple tensors is by appending
  // incremental digits after the 1st tensor_name.
  if (fusedLoc) {
    node_name = llvm::dyn_cast<mlir::NameLoc>(fusedLoc.getLocations().front())
                    .getName();
  } else {
    node_name = nameLoc.getName();
  }
  // Some TFLite has fused op names with several hierarchical information
  // concatenated together with semicolons. In this case, we will find the last
  // single node name that contains this node label. If no matching found, we
  // will return the last single node name by default.
  const size_t num_substrs = node_name.count(kSemicolonSeparator) + 1;
  if (num_substrs > 1) {
    // Removes any underscores in `node_label`.
    const std::string node_label_substr =
        absl::StrReplaceAll(node_label, {{"_", ""}});
    llvm::SmallVector<llvm::StringRef, 4> single_names;
    single_names.reserve(num_substrs);
    node_name.split(single_names, kSemicolonSeparator, /*KeepEmpty=*/false);
    // We iterate backwards to find if a single node name contains the node
    // label in the end hierarchy.
    for (auto it = single_names.rbegin(); it != single_names.rend(); ++it) {
      llvm::StringRef name = *it;
      llvm::StringRef last_substr = name;
      const size_t start_pos = name.find_last_of('/');
      if (start_pos != std::string::npos) {
        last_substr = name.substr(start_pos);
      }
      if (last_substr.contains_insensitive(node_label_substr)) {
        return name;
      }
    }
    // if there is no match in `single_names` vector, we return the last node
    // name in that fused op names by default. Skipping "pseudo_const" node to
    // reduce verbosity.
    node_name = single_names.back();
    if (node_label != kPseudoConst) {
      llvm::errs() << "WARNING: No matched name for node \"" << node_label
                   << "\" at " << node_id_str
                   << ", using the last node name by default.\n";
    }
  }
  return node_name;
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
    auto block_arg = val.dyn_cast_or_null<mlir::BlockArgument>();
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

// Converts a tf dialect function op to a subgraph.
absl::Status TfFunctionToSubgraph(const VisualizeConfig& config, FuncOp& fop,
                                  Subgraph& subgraph) {
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
      RETURN_IF_ERROR(builder.AppendAttrToMetadata(
          EdgeType::kOutput, output_index, kTensorIndex,
          absl::StrCat(tensor_idx_counter.increment())));
      mlir::Value val = operation.getResult(output_index);
      RETURN_IF_ERROR(builder.AppendAttrToMetadata(
          EdgeType::kOutput, output_index, kTensorShape,
          GetTypeString(val.getType())));
    }
    subgraph.nodes.push_back(std::move(builder).Build());
  }
  return absl::OkStatus();
}

// Converts a tfl dialect function op to a subgraph.
absl::Status TfliteFunctionToSubgraph(const VisualizeConfig& config,
                                      FuncOp& fop, Subgraph& subgraph) {
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
    llvm::StringRef node_name =
        GetTfliteNodeName(node_id, node_label, operation);
    GraphNodeBuilder builder;
    builder.SetNodeInfo(node_id, node_label, node_name);
    AppendNodeAttrs(config.const_element_count_limit, operation, builder);
    absl::Status append_subgraph_status =
        TfliteMaybeAppendSubgraphs(operation, builder);
    if (!append_subgraph_status.ok()) {
      llvm::errs() << append_subgraph_status.ToString() << "\n";
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
      RETURN_IF_ERROR(builder.AppendAttrToMetadata(
          EdgeType::kOutput, output_index, kTensorIndex,
          absl::StrCat(tensor_idx_counter.increment())));
      mlir::Value val = operation.getResult(output_index);
      if (output_index < tensor_names.size()) {
        RETURN_IF_ERROR(builder.AppendAttrToMetadata(
            EdgeType::kOutput, output_index, kTensorName,
            tensor_names[output_index]));
      }
      RETURN_IF_ERROR(builder.AppendAttrToMetadata(
          EdgeType::kOutput, output_index, kTensorShape,
          GetTypeString(val.getType())));
      // TODO(b/293348398): Tensor indices are not matched to indices in
      // Flatbuffer. Further investigation is needed.
    }
    subgraph.nodes.push_back(std::move(builder).Build());
  }
  return absl::OkStatus();
}

// Converts a stablehlo dialect of JAX function op to a subgraph.
absl::Status StablehloJaxFunctionToSubgraph(const VisualizeConfig& config,
                                            FuncOp& fop, Subgraph& subgraph) {
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
      llvm::errs() << append_subgraph_status.ToString() << "\n";
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
      RETURN_IF_ERROR(builder.AppendAttrToMetadata(
          EdgeType::kOutput, output_index, kTensorIndex,
          absl::StrCat(tensor_idx_counter.increment())));
      mlir::Value val = operation.getResult(output_index);
      RETURN_IF_ERROR(builder.AppendAttrToMetadata(
          EdgeType::kOutput, output_index, kTensorShape,
          GetTypeString(val.getType())));
    }
    subgraph.nodes.push_back(std::move(builder).Build());
  }
  return absl::OkStatus();
}

}  // namespace

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
    Subgraph subgraph(func_name.str());
    absl::Status status = TfFunctionToSubgraph(config, fop, subgraph);
    if (!status.ok()) {
      llvm::errs() << status.ToString() << "\n";
      return mlir::WalkResult::interrupt();
    }
    result.subgraphs.push_back(subgraph);
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
  // Entry functions for signature defs.
  std::vector<FuncOp> entry_functions;
  std::vector<FuncOp> non_entry_functions;
  FuncOp main_fn = module_op.lookupSymbol<FuncOp>("main");
  if (main_fn != nullptr) {
    // Treat the main function as a signature def when the given main function
    // contains on the tf.entry_function attribute.
    auto attrs =
        main_fn->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
    if (attrs && !attrs.empty()) {
      entry_functions.push_back(main_fn);
    } else {
      non_entry_functions.push_back(main_fn);
    }
  }

  // Walk over the module collection ops with functions and while ops.
  module_op.walk([&](FuncOp fn) {
    if (main_fn == fn) return mlir::WalkResult::advance();
    auto attrs = fn->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
    if (attrs && !attrs.empty()) {
      entry_functions.push_back(fn);
    } else {
      non_entry_functions.push_back(fn);
    }
    return mlir::WalkResult::advance();
  });

  // We intentionally process the entry functions first and then the rest to
  // match the order of tensor index in TFLite converter.
  for (FuncOp fop : entry_functions) {
    Subgraph subgraph(fop.getSymName().str());
    RETURN_IF_ERROR(TfliteFunctionToSubgraph(config, fop, subgraph));
    result.subgraphs.push_back(subgraph);
  }
  for (FuncOp fop : non_entry_functions) {
    Subgraph subgraph(fop.getSymName().str());
    RETURN_IF_ERROR(TfliteFunctionToSubgraph(config, fop, subgraph));
    result.subgraphs.push_back(subgraph);
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
    Subgraph subgraph(func_name.str());

    // Since the ops in the each function can either be tf or stablehlo dialect,
    // we check the first operation in the function to decide which dialect this
    // function is.
    mlir::Block& block = fop.getBody().front();
    mlir::Operation& first_op = block.front();
    if (llvm::isa<mlir::TF::TensorFlowDialect>(first_op.getDialect())) {
      absl::Status status = TfFunctionToSubgraph(config, fop, subgraph);
      if (!status.ok()) {
        llvm::errs() << status.ToString() << "\n";
        return mlir::WalkResult::interrupt();
      }
    } else if (llvm::isa<mlir::stablehlo::StablehloDialect>(
                   first_op.getDialect())) {
      absl::Status status =
          StablehloJaxFunctionToSubgraph(config, fop, subgraph);
      if (!status.ok()) {
        llvm::errs() << status.ToString() << "\n";
        return mlir::WalkResult::interrupt();
      }
    } else {
      llvm::errs() << "Unknown dialect: "
                   << first_op.getDialect()->getNamespace()
                   << " in function: " << func_name
                   << ", we skip serializing this function.\n";
      return mlir::WalkResult::skip();
    }
    result.subgraphs.push_back(subgraph);
    return mlir::WalkResult::advance();
  });

  if (walk_result.wasInterrupted()) {
    return absl::InternalError("Module walk interrupted.");
  }
  return result;
}

}  // namespace visualization_client
}  // namespace tooling
