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

#include "hlo_adapter/direct_hlo_to_json_graph_convert.h"

#include <sys/types.h>

#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "formats/schema_structs.h"
#include "graphnode_builder.h"
#include "status_macros.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "tensorflow/core/profiler/utils/hlo_module_utils.h"

namespace tooling {
namespace visualization_client {
namespace {

constexpr absl::string_view kShapeWithLayout = "shape_with_layout";
constexpr absl::string_view kOpName = "op_name";
constexpr absl::string_view kOpType = "op_type";
constexpr absl::string_view kSourceFile = "source_file";
constexpr absl::string_view kSourceLine = "source_line";
constexpr absl::string_view kSourceStack = "source_stack";
constexpr absl::string_view kOpcode = "opcode";
constexpr absl::string_view kGetTupleElementIndex = "get_tuple_element_index";
constexpr absl::string_view kUsers = "users";
constexpr absl::string_view kOperands = "operands";
constexpr absl::string_view kLiteral = "literal";
constexpr absl::string_view kHide = "hide_node";
constexpr absl::string_view kAcfComputationName = "async_collective_fusion";
constexpr absl::string_view kAcsInstructionName = "AsyncCollectiveStart";
constexpr absl::string_view kAcdInstructionName = "AsyncCollectiveDone";

constexpr int kMaxUsersToRender = 16;

// OutputEdges is a map from source instruction id to a list of its users.
using OutputEdges =
    absl::flat_hash_map<std::string, std::vector<const xla::HloInstruction*>>;

// TODO(b/402148725) Move utility functions to a separate file.
// Detect if an instruction is an AsyncCollectiveFusion parameter that is
// implementation details.
bool IsAcfPrameter(const xla::HloInstruction* instruction) {
  // Parameter is fused
  if (instruction->opcode() != xla::HloOpcode::kParameter ||
      !instruction->IsFused())
    return false;

  // Parameter piped through and is only consumed by 1 user
  // Parameter 0 consumed by both root and all-gather will always persist.
  if (instruction->user_count() != 1) return false;

  const xla::HloComputation* parent_computation = instruction->parent();
  int64_t parameter_number = instruction->parameter_number();
  xla::HloInstruction* fusion_instruction =
      parent_computation->FusionInstruction();
  const xla::HloInstruction* parameterOperand =
      fusion_instruction->operand(parameter_number);
  // Operand is get-tuple-element
  if (parameterOperand->opcode() != xla::HloOpcode::kGetTupleElement) {
    return false;
  }

  const xla::HloInstruction* gteOperand = parameterOperand->operand(0);
  if (gteOperand->opcode() != xla::HloOpcode::kFusion) {
    return false;
  }
  auto src_instruction =
      gteOperand->fused_instructions_computation()->root_instruction();
  // (1) Parameter is fused into AsyncCollectiveFusion, operand is gte from
  // AsyncCollectiveStart custom call and user is the root node of ACF
  // (2) Parameter is mapped from Params in AsyncCollectiveFusion - operand is
  // gte from ACF, and user is AsyncCollectiveDone custom call
  return (absl::StartsWith(parent_computation->name(), kAcfComputationName) &&
          src_instruction->IsCustomCall(kAcsInstructionName) &&
          instruction->users()[0] == parent_computation->root_instruction()) ||
         (instruction->users()[0]->IsCustomCall(kAcdInstructionName) &&
          absl::StartsWith(gteOperand->fused_instructions_computation()->name(),
                           kAcfComputationName));
}

// Recursively include all instructions in the nested computations.
void RecursiveIncludeNestedComputations(
    const xla::HloInstruction* instruction,
    absl::flat_hash_set<const xla::HloInstruction*>& included_nodes) {
  for (const xla::HloComputation* subcomputation :
       instruction->called_computations()) {
    for (const xla::HloInstruction* subcomputation_instruction :
         subcomputation->instructions()) {
      included_nodes.insert(subcomputation_instruction);
      RecursiveIncludeNestedComputations(subcomputation_instruction,
                                         included_nodes);
    }
  }
}

// Gets a NodeFilter that includes roughly all instructions whose distance from
// root is <= radius. The fusion instruction (and its fused computation) is
// treated as a single entity. The scope will not go beyond instruction's parent
// computation.
NodeFilter MakeInstructionRadiusAroundFilter(const xla::HloInstruction* root,
                                             const int radius) {
  absl::flat_hash_set<const xla::HloInstruction*> included_nodes;
  std::deque<std::pair<const xla::HloInstruction*, int>> worklist;
  worklist.push_back({root, 0});

  while (!worklist.empty()) {
    const auto [instruction, depth] = worklist.front();
    worklist.pop_front();
    if (depth > radius) {
      continue;
    }
    included_nodes.insert(instruction);

    // Traverse instruction's operands.
    // Don't traverse into tuples' operands unless the tuple is the root.
    // Usually a tuple is the bottommost node in the graph, and so its operands
    // are not interesting to the graph at hand.
    if (instruction == root ||
        instruction->opcode() != xla::HloOpcode::kTuple) {
      for (const xla::HloInstruction* operand : instruction->operands()) {
        if (!included_nodes.contains(operand)) {
          worklist.push_back({operand, depth + 1});
        }
      }
    }

    // Recursively include all the instructions in nested computations.
    RecursiveIncludeNestedComputations(instruction, included_nodes);

    // Traverse instruction's users, omit users if there are too many.
    // Consider provide more context in filter lambda return value, so we can
    // provide omitted information.
    if (instruction->user_count() > kMaxUsersToRender) {
      included_nodes.insert(instruction);
      continue;
    }
    for (const xla::HloInstruction* user : instruction->users()) {
      if (!included_nodes.contains(user)) {
        worklist.push_back({user, depth + 1});
      }
    }
  }

  return [=](const xla::HloInstruction* instruction) {
    return included_nodes.contains(instruction);
  };
}

// Gets the computation hierarchy, split by "/", e.g., "computation_0/fusion_1".
std::string GetComputationHierarchy(
    const std::vector<std::string>& computation_stack) {
  return absl::StrJoin(computation_stack, "/");
}

bool IsGetTupleElement(const HloAdapterOption& options,
                       const xla::HloInstruction* instruction) {
  return options.get_tuple_element_folding &&
         instruction->opcode() == xla::HloOpcode::kGetTupleElement;
}

absl::Status AddHloInstructionIncomingEdges(
    const xla::HloInstruction* instruction, const HloAdapterOption& options,
    GraphNodeBuilder& builder, OutputEdges& output_edges,
    const ComputationExpand& computation_expand) {
  if (instruction->opcode() == xla::HloOpcode::kFusion &&
      computation_expand(instruction,
                         instruction->fused_instructions_computation())) {
    // If the instruction is an expanded fusion, don't connect it to anything
    // because the operands will be connected to the parameters.
    return absl::OkStatus();
  }
  std::vector<const xla::HloInstruction*> operands;
  // If the instruction is a Parameter within a Fusion computation, we connect
  // the operands of the fusion computation to the parameters.
  if (instruction->opcode() == xla::HloOpcode::kParameter &&
      instruction->IsFused()) {
    const xla::HloInstruction* fusion_instruction =
        instruction->parent()->FusionInstruction();
    if (fusion_instruction == nullptr) {
      return absl::InternalError("Fusion instruction not found");
    }
    operands.push_back(
        fusion_instruction->operand(instruction->parameter_number()));
  } else {
    operands.insert(operands.end(), instruction->operands().begin(),
                    instruction->operands().end());
  }

  int input_id = 0;
  for (const xla::HloInstruction* operand : operands) {
    std::string src_instruction_id;
    if (IsGetTupleElement(options, operand)) {
      // Skip the GTE operand, and connect the user to the tuple directly.
      operand = operand->operand(0);
    }

    if (operand->opcode() == xla::HloOpcode::kFusion &&
        computation_expand(operand,
                           operand->fused_instructions_computation())) {
      // If the operand is a fusion, we connect the user to the root of the
      // fusion computation.
      src_instruction_id = GetInstructionId(
          operand->fused_instructions_computation()->root_instruction());
    } else {
      src_instruction_id = GetInstructionId(operand);
    }

    const std::string output_id =
        absl::StrCat(output_edges[src_instruction_id].size());
    builder.AppendEdgeInfo(
        /*source_node_id_str=*/src_instruction_id,
        output_id, /*target_node_input_id_str=*/
        absl::StrCat(input_id++));
    output_edges[src_instruction_id].push_back(instruction);
  }
  return absl::OkStatus();
}

void SetInstructionNodeLabel(const xla::HloInstruction* instruction,
                             GraphNodeBuilder& builder) {
  // Instruction label.
  std::string instruction_label;
  if (instruction->opcode() == xla::HloOpcode::kParameter) {
    instruction_label =
        absl::StrFormat("Parameter %d", instruction->parameter_number());
  } else if (instruction->opcode() == xla::HloOpcode::kConstant) {
    instruction_label = "Constant";
  } else {
    instruction_label = instruction->name();
  }

  // Set the text inside the instruction node.
  builder.SetNodeLabel(instruction_label);
}

void SetInstructionNodeAttributes(const xla::HloInstruction* instruction,
                                  const HloAdapterOption& options,
                                  GraphNodeBuilder& builder) {
  // Instruction opcode.
  std::string opcode = absl::StrCat(HloOpcodeString(instruction->opcode()));
  builder.AppendNodeAttribute(kOpcode, opcode);

  // Instruction shape with layout.
  std::string instruction_shape =
      xla::ShapeUtil::HumanStringWithLayout(instruction->shape());
  // Truncate the shape if it's too long.
  constexpr int kMaxShapeLen = 64;
  if (instruction_shape.size() > kMaxShapeLen) {
    instruction_shape = instruction_shape.substr(0, kMaxShapeLen) + "...";
  }
  builder.AppendNodeAttribute(kShapeWithLayout, instruction_shape);

  // Add instruction users if the users are omitted with max threshold.
  // If within threshold, users are the same as inputs shown in the graph.
  if (instruction->user_count() > kMaxUsersToRender) {
    std::vector<std::string> users_str;
    for (const xla::HloInstruction* user : instruction->users()) {
      users_str.push_back(absl::StrCat(
          user->name(), "=", xla::ShapeUtil::HumanString(user->shape())));
    }
    builder.AppendNodeAttribute(kUsers, absl::StrJoin(users_str, "\n"));
  }

  // Add operands info if it is a tuple instruction.
  // Because operands for non-root tuple are emitted.
  if (instruction->opcode() == xla::HloOpcode::kTuple) {
    std::vector<std::string> operands_str;
    for (const xla::HloInstruction* operand : instruction->operands()) {
      operands_str.push_back(absl::StrCat(
          operand->name(), "=", xla::ShapeUtil::HumanString(operand->shape())));
    }
    builder.AppendNodeAttribute(kOperands, absl::StrJoin((operands_str), "\n"));
  }

  // Instruction metadata.
  if (!instruction->metadata().op_name().empty()) {
    builder.AppendNodeAttribute(kOpName, instruction->metadata().op_name());
  }
  if (!instruction->metadata().op_type().empty()) {
    builder.AppendNodeAttribute(kOpType, instruction->metadata().op_type());
  }
  if (!instruction->metadata().source_file().empty()) {
    builder.AppendNodeAttribute(kSourceFile,
                                instruction->metadata().source_file());
  }
  if (instruction->metadata().source_line() != 0) {
    builder.AppendNodeAttribute(
        kSourceLine, absl::StrCat(instruction->metadata().source_line()));
  }
  if (instruction->metadata().stack_frame_id() != 0) {
    builder.AppendNodeAttribute(
        kSourceStack,
        tensorflow::profiler::GetSourceInfo(*instruction).stack_frame);
  }

  // Attach get-tuple-element index if its define is a GTE and folded.
  if (options.get_tuple_element_folding) {
    absl::flat_hash_map<std::string, std::vector<std::string>>
        tuple_indexes_by_operand;
    for (const xla::HloInstruction* operand : instruction->operands()) {
      if (IsGetTupleElement(options, operand)) {
        tuple_indexes_by_operand[operand->operand(0)->name()].push_back(
            absl::StrCat(operand->tuple_index()));
      }
    }
    std::string tuple_indexes_string;
    for (const auto& [operand_name, tuple_indexes] : tuple_indexes_by_operand) {
      if (!tuple_indexes_string.empty()) {
        absl::StrAppend(&tuple_indexes_string, ";");
      }
      absl::StrAppend(&tuple_indexes_string, absl::StrJoin(tuple_indexes, ","),
                      " of ", operand_name);
    }
    if (!tuple_indexes_string.empty()) {
      builder.AppendNodeAttribute(kGetTupleElementIndex, tuple_indexes_string);
    }
  }

  // Constant literal.
  if (instruction->IsConstant() &&
      xla::Cast<xla::HloConstantInstruction>(instruction)->HasLiteral()) {
    builder.AppendNodeAttribute(kLiteral, instruction->literal().ToString());
  }

  if (options.hide_async_collective_fusion_parameter) {
    if (IsAcfPrameter(instruction)) {
      builder.AppendNodeAttribute(kHide, "true");
    }
  }
}

absl::Status BuildHloInstructionNode(
    const xla::HloInstruction* instruction, const HloAdapterOption& options,
    std::vector<std::string>& computation_stack, GraphNodeBuilder& builder,
    OutputEdges& output_edges, const ComputationExpand& computation_expand) {
  builder.SetNodeId(GetInstructionId(instruction));

  // Set namespace.
  builder.SetNodeName(GetComputationHierarchy(computation_stack));

  // Set node label.
  SetInstructionNodeLabel(instruction, builder);

  // Add incoming edges.
  RETURN_IF_ERROR(AddHloInstructionIncomingEdges(
      instruction, options, builder, output_edges, computation_expand));

  // Set node attributes.
  SetInstructionNodeAttributes(instruction, options, builder);

  return absl::OkStatus();
}

// Populates the outputs metadata for the given node.
void PopulateOutputsMetadata(
    GraphNodeBuilder& builder,
    const std::vector<const xla::HloInstruction*>& output_nodes) {
  for (int i = 0; i < output_nodes.size(); ++i) {
    builder.AppendAttrToMetadata(EdgeType::kOutput, i, kShapeWithLayout,
                                 builder.GetNodeAttribute(kShapeWithLayout));
  }
}

// Recursively builds a list of GraphNodeBuilders from a HLO computation and its
// subcomputations. The subcomputations, including the fusion computations, are
// represented with "namespace" feature in the Model Explorer.
//
// `instruction_node_builders`: the object that holds all GraphNodeBuilder.
//
// `computation`: the computation that is being built.
//
// `built_computations`: the computations that have been built.
//
// `computation_stack`: track current computation hierarchy.
//
// `node_filter`: decide which instruction to include.
//
// `computation_expand`: decide which computation to expand.
//
// `output_edges`: record all the edges from the instruction to its users.
absl::Status HloComputationToGraphImpl(
    const xla::HloComputation& computation, const NodeFilter& node_filter,
    const ComputationExpand& computation_expand,
    const HloAdapterOption& options,
    absl::flat_hash_set<const xla::HloComputation*>& built_computations,
    std::vector<std::string>& computation_stack,
    std::vector<GraphNodeBuilder>& instruction_node_builders,
    OutputEdges& output_edges) {
  if (built_computations.contains(&computation)) {
    return absl::OkStatus();
  }
  built_computations.insert(&computation);

  // Create a pinned node for the computation layer.
  GraphNodeBuilder builder;
  // Fusion computation is merged with its caller fusion instruction, make the
  // pinned node represents caller instruction instead of the computation.
  if (computation.FusionInstruction() != nullptr) {
    RETURN_IF_ERROR(BuildHloInstructionNode(computation.FusionInstruction(),
                                            options, computation_stack, builder,
                                            output_edges, computation_expand));
    builder.SetNodeLabel(GetInstructionId(computation.FusionInstruction()));
    builder.AppendNodeAttribute("Fusion Computation", computation.name());
  } else {
    // Build the pinned node representing the computation.
    builder.SetNodeId(GetComputationId(&computation));
    builder.SetNodeName(GetComputationHierarchy(computation_stack));
    builder.SetNodeLabel(absl::StrCat("Computation \n", computation.name()));
  }
  builder.SetPinToGroupTop(true);
  instruction_node_builders.push_back(builder);

  for (const xla::HloInstruction* instruction :
       computation.MakeInstructionPostOrder()) {
    if (!node_filter(instruction)) {
      continue;
    }

    if (instruction->opcode() == xla::HloOpcode::kFusion &&
        computation_expand(instruction,
                           instruction->fused_instructions_computation())) {
      // We do not construct a dedicated node for the variable assigned by
      // fusion op. Instead, we (1) build the fusion computation, (2) connect
      // the operands of the fusion computation to its parameters, and (3)
      // connect the ROOT of the fusion computation to its users.
      // (1) is done at this scope, (2) and (3) is done at
      // `AddHloInstructionIncomingEdges`.
      computation_stack.push_back(std::string(instruction->name()));
      RETURN_IF_ERROR(HloComputationToGraphImpl(
          *(instruction->fused_instructions_computation()), node_filter,
          computation_expand, options, built_computations, computation_stack,
          instruction_node_builders, output_edges));
      computation_stack.pop_back();
    } else if (IsGetTupleElement(options, instruction)) {
      continue;
    } else {
      // Build the hlo instruction node.
      GraphNodeBuilder builder;
      RETURN_IF_ERROR(
          BuildHloInstructionNode(instruction, options, computation_stack,
                                  builder, output_edges, computation_expand));

      // Convert subcomputations within the instruction to subgraphs.
      for (const xla::HloComputation* subcomputation :
           instruction->called_computations()) {
        if (!computation_expand(instruction, subcomputation)) {
          continue;
        }
        computation_stack.push_back(std::string(subcomputation->name()));
        int prev_node_count = instruction_node_builders.size();
        RETURN_IF_ERROR(HloComputationToGraphImpl(
            *subcomputation, node_filter, computation_expand, options,
            built_computations, computation_stack, instruction_node_builders,
            output_edges));
        int cur_node_count = instruction_node_builders.size();
        computation_stack.pop_back();

        // If some of the nodes from subcomputation are included, connect the
        // root of subcomputation to the caller instruction. Since the
        // subcomputation is traversed in post order, the last instruction is
        // the root of the subcomputation, and must be included.
        if (cur_node_count > prev_node_count) {
          const std::string last_node_id =
              instruction_node_builders.back().GetNodeId();
          builder.AppendEdgeInfo(last_node_id, "0", "0");
          output_edges[last_node_id].push_back(instruction);
        }
      }

      instruction_node_builders.push_back(builder);
    }
  }

  return absl::OkStatus();
}

// Convert to json.
std::string GraphCollectionToJson(GraphCollection& collection) {
  llvm::json::Value json_result(collection.Json());
  std::string json_output;
  llvm::raw_string_ostream json_ost(json_output);
  json_ost << llvm::formatv("{0:2}", json_result);
  return json_output;
}

}  // namespace

std::string GetInstructionId(const xla::HloInstruction* instruction) {
  return absl::StrCat(instruction->name());
}

std::string GetComputationId(const xla::HloComputation* computation) {
  return absl::StrCat(computation->name());
}

absl::StatusOr<GraphCollection> HloToGraph(
    const xla::HloComputation& computation, const NodeFilter& node_filter,
    const ComputationExpand& computation_expand,
    const HloAdapterOption& options) {
  Graph graph;
  Subgraph subgraph(std::string(computation.name()));
  graph.subgraphs.push_back(std::move(subgraph));
  absl::flat_hash_set<const xla::HloComputation*> built_computations;
  std::vector<std::string> computation_stack;
  computation_stack.push_back(std::string(computation.name()));
  OutputEdges output_edges;
  std::vector<GraphNodeBuilder> instruction_node_builders;

  RETURN_IF_ERROR(HloComputationToGraphImpl(
      computation, node_filter, computation_expand, options, built_computations,
      computation_stack, instruction_node_builders, output_edges));

  for (GraphNodeBuilder& builder : instruction_node_builders) {
    if (const auto& it = output_edges.find(builder.GetNodeId());
        it != output_edges.end()) {
      PopulateOutputsMetadata(builder, it->second);
    }
    graph.subgraphs.back().nodes.push_back(std::move(builder).Build());
  }

  GraphCollection collection;
  collection.graphs.push_back(std::move(graph));
  return collection;
}

absl::StatusOr<std::string> HloGraphAdapter(
    const xla::HloComputation& computation, const HloAdapterOption& options) {
  const NodeFilter node_filter = [&](const xla::HloInstruction* instruction) {
    return true;
  };

  ASSIGN_OR_RETURN(GraphCollection graph_collection,
                   HloToGraph(computation, node_filter, options));

  return GraphCollectionToJson(graph_collection);
}

absl::StatusOr<std::string> HloGraphAdapter(
    const xla::HloInstruction& instruction, const int radius,
    const HloAdapterOption& options) {
  const NodeFilter node_filter =
      MakeInstructionRadiusAroundFilter(&instruction, radius);

  ASSIGN_OR_RETURN(GraphCollection graph_collection,
                   HloToGraph(*instruction.parent(), node_filter, options));

  return GraphCollectionToJson(graph_collection);
}

}  // namespace visualization_client
}  // namespace tooling
