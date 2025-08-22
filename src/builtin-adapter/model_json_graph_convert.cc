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

#include "model_json_graph_convert.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/tests/CheckOps.h"
#include "stablehlo/transforms/Passes.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "formats/schema_structs.h"
#include "status_macros.h"
#include "transforms/conversion.h"
#include "translate_helpers.h"
#include "visualize_config.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_tf_xla_call_module_to_stablehlo_pass.h"
#include "tensorflow/compiler/mlir/stablehlo/transforms/rename_entrypoint_to_main.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_import_options.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/tsl/platform/env.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/protobuf/trackable_object_graph.pb.h"

namespace tooling {
namespace visualization_client {
namespace {

enum class MlirDialect {
  kTf,
  kTflite,
  kStablehlo,
};

struct TfMetadata {
  int tf_version;
  std::string saved_model_tags;
  std::vector<std::string> exported_names_vector;
};

// Referred logic from lite/python/flatbuffer_to_mlir.cc.
static mlir::OwningOpRef<mlir::ModuleOp> FlatBufferFileToMlirTranslation(
    llvm::SourceMgr* source_mgr, mlir::MLIRContext* context) {
  const llvm::MemoryBuffer* input =
      source_mgr->getMemoryBuffer(source_mgr->getMainFileID());
  std::string error;
  auto loc = mlir::FileLineColLoc::get(context, input->getBufferIdentifier(),
                                       /*line=*/0, /*column=*/0);
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  return tflite::FlatBufferToMlir(
      absl::string_view(input->getBufferStart(), input->getBufferSize()),
      context, loc, /*use_external_constant=*/false, inputs, outputs,
      /*experimental_prune_unreachable_nodes_unconditionally=*/false);
}

// TF2 SavedModel must have SavedObjectGraph and require 1:1 correspondence
// between tf.function <=> SavedFunction <=> SavedConcreteFunction <=>
// FunctionDef. Otherwise we assume it's a TF1 SavedModel.
static int GetTfVersion(const tensorflow::SavedModel& saved_model) {
  if (!saved_model.meta_graphs(0).has_object_graph_def()) {
    return 1;
  }
  const tensorflow::SavedObjectGraph& object_graph_def =
      saved_model.meta_graphs()[0].object_graph_def();
  const auto& saved_object_nodes = object_graph_def.nodes();
  for (const tensorflow::SavedObject& object : saved_object_nodes) {
    if (object.kind_case() == tensorflow::SavedObject::kFunction) {
      if (object.function().concrete_functions_size() != 1) {
        return 1;
      }
    }
  }
  return 2;
}

// Obtains TF metadata from the given SavedModel path.
// The metadata contains the tf version, saved model tags and exported names.
absl::StatusOr<TfMetadata> ObtainTfMetadata(absl::string_view model_path) {
  TfMetadata tf_metadata;
  tensorflow::SavedModel saved_model;
  RETURN_IF_ERROR(tensorflow::ReadSavedModel(model_path, &saved_model));
  if (saved_model.meta_graphs_size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Only `SavedModel`s with 1 MetaGraph are supported. Instead, it has ",
        saved_model.meta_graphs_size()));
  }
  tf_metadata.tf_version = GetTfVersion(saved_model);
  const tensorflow::MetaGraphDef::MetaInfoDef& meta_info_def =
      saved_model.meta_graphs()[0].meta_info_def();
  tf_metadata.saved_model_tags = absl::StrJoin(meta_info_def.tags(), ",");

  // Only TF2 model needs it, TF1 model can use empty exported_names to apply
  // default values.
  if (tf_metadata.tf_version == 2) {
    const tensorflow::SavedObjectGraph& object_graph_def =
        saved_model.meta_graphs()[0].object_graph_def();
    const auto& saved_object_nodes = object_graph_def.nodes();
    // According to saved_object_graph.proto, nodes[0] indicates root node.
    for (const auto& child : object_graph_def.nodes()[0].children()) {
      if (saved_object_nodes[child.node_id()].has_function()) {
        tf_metadata.exported_names_vector.push_back(child.local_name());
      }
    }
  }
  return tf_metadata;
}

absl::Status DeserializeVhloToStablehlo(mlir::ModuleOp module_op) {
  mlir::PassManager pm(module_op.getContext());
  mlir::stablehlo::createStablehloDeserializePipeline(pm);
  mlir::LogicalResult result = pm.run(module_op);
  if (mlir::failed(result)) {
    return absl::InternalError("Failed to run stablehlo deserialize pipeline.");
  }
  return absl::OkStatus();
}

absl::Status RunTFShapeInference(mlir::ModuleOp module_op) {
  mlir::PassManager pm(module_op.getContext());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  mlir::LogicalResult result = pm.run(module_op);
  if (mlir::failed(result)) {
    return absl::InternalError("Failed to run shape inference pass.");
  }
  return absl::OkStatus();
}

// Runs all passes involved in transforming or optimizing a TF MLIR graph
// without any target specialization. Referred logic from
// compiler/mlir/tensorflow/transforms/bridge.cc.
absl::Status RunStandardPipeline(mlir::ModuleOp module_op) {
  mlir::PassManager bridge(module_op.getContext());
  mlir::TF::StandardPipelineOptions pipeline_options;
  CreateTFStandardPipeline(bridge, pipeline_options);
  mlir::LogicalResult result = bridge.run(module_op);
  if (mlir::failed(result)) {
    return absl::InternalError("Failed to run standard pipeline.");
  }
  return absl::OkStatus();
}

// Iterates through all functions in the module and returns true if any of them
// has tf.XlaCallModule op.
bool HasXlaCallModule(mlir::ModuleOp module) {
  const auto walk_result =
      module->walk([&](mlir::func::FuncOp fop) -> mlir::WalkResult {
        auto it = fop.getOps<mlir::TF::XlaCallModuleOp>();
        if (!it.empty()) {
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
  if (walk_result.wasInterrupted()) {
    return true;
  }
  return false;
}

// Deserializes tf.XlaCallModule ops and converts it to stablehlo module. Input
// module op is assumed to be already a tf dialect module.
absl::Status ConvertToStablehloModule(mlir::ModuleOp module_op) {
  mlir::PassManager bridge(module_op.getContext());
  bridge.addPass(mlir::odml::CreateRenameEntrypointToMainPass());
  bridge.addPass(mlir::odml::CreateLegalizeTFXlaCallModuleToStablehloPass());
  mlir::LogicalResult result = bridge.run(module_op);
  if (mlir::failed(result)) {
    return absl::InternalError(
        "Failed to convert tf_executor dialect to tf & stablehlo dialect MLIR "
        "module.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> ModuleOpToJson(const VisualizeConfig& config,
                                           mlir::Operation* module_op) {
  mlir::PassManager pm(module_op->getContext());
  pm.addPass(CreateUniqueOpNamesPass());
  mlir::LogicalResult result = pm.run(module_op);
  if (mlir::failed(result)) {
    return absl::InternalError("Failed to run unique op names pass.");
  }

  std::string json_output;
  llvm::raw_string_ostream json_ost(json_output);
  ASSIGN_OR_RETURN(Graph graph, MlirToGraph(config, module_op));
  GraphCollection collection;
  collection.graphs.push_back(std::move(graph));
  llvm::json::Value json_result(collection.Json());
  json_ost << llvm::formatv("{0:2}", json_result);

  return json_output;
}

}  // namespace

absl::StatusOr<std::string> ConvertSavedModelToJson(
    const VisualizeConfig& config, absl::string_view model_path) {
  ASSIGN_OR_RETURN(TfMetadata tf_metadata, ObtainTfMetadata(model_path));
  std::unordered_set<std::string> tags =
      absl::StrSplit(tf_metadata.saved_model_tags, ',');
  absl::Span<std::string> exported_names(tf_metadata.exported_names_vector);

  mlir::MLIRContext context;
  // Enable parsing of MLIR modules with unregistered dialects. This is safe as
  // Model Explorer does not execute operations, only visualizes them.
  context.allowUnregisteredDialects(true);
  mlir::OwningOpRef<mlir::ModuleOp> module_op;
  if (tf_metadata.tf_version == 1) {
    LOG(INFO) << "Converting SavedModel V1 to MLIR module...";
    tensorflow::MLIRImportOptions import_options;
    import_options.upgrade_legacy = true;

    // Converts SavedModel V1 to MLIR module.
    ASSIGN_OR_RETURN(module_op, tensorflow::SavedModelSignatureDefsToMlirImport(
                                    model_path, tags, exported_names, &context,
                                    import_options));

    RETURN_IF_ERROR(RunTFShapeInference(*module_op));
  } else {
    LOG(INFO) << "Converting SavedModel V2 to MLIR module...";
    // Converts SavedModel V2 to MLIR module.
    ASSIGN_OR_RETURN(module_op,
                     tensorflow::SavedModelObjectGraphToMlirImport(
                         model_path, tags, exported_names, &context));

    // Converts tf_executor dialect to tf dialect MLIR module.
    RETURN_IF_ERROR(RunStandardPipeline(*module_op));
  }

  // Converts MLIR module to JSON string.
  if (HasXlaCallModule(*module_op)) {
    // This indicates it's a JAX converted SavedModel. There are stablehlo ops
    // serialized within tf.XlaCallModule op, we want to deserialize it before
    // proceeding.
    RETURN_IF_ERROR(ConvertToStablehloModule(*module_op));
  }

  return ModuleOpToJson(config, *module_op);
}

absl::StatusOr<std::string> ConvertFlatbufferToJson(
    const VisualizeConfig& config, absl::string_view model_path_or_buffer,
    bool is_modelpath) {
  std::unique_ptr<llvm::MemoryBuffer> input;
  std::string model_content;
  if (is_modelpath) {
    RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(),
                                          std::string(model_path_or_buffer),
                                          &model_content));

    input = llvm::MemoryBuffer::getMemBuffer(model_content,
                                             /*BufferName=*/"flatbuffer",
                                             /*RequiresNullTerminator=*/false);
  } else {
    input = llvm::MemoryBuffer::getMemBuffer(model_path_or_buffer,
                                             /*BufferName=*/"flatbuffer",
                                             /*RequiresNullTerminator=*/false);
  }

  if (input == nullptr) {
    return absl::InternalError("Can't get llvm::MemoryBuffer");
  }

  mlir::MLIRContext context;
  context.printOpOnDiagnostic(true);
  // Enable parsing of MLIR modules with unregistered dialects. This is safe as
  // Model Explorer does not execute operations, only visualizes them.
  context.allowUnregisteredDialects(true);
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());

  // Converts Flatbuffer to MLIR module.
  mlir::OwningOpRef<mlir::ModuleOp> module_op =
      FlatBufferFileToMlirTranslation(&sourceMgr, &context);
  if (!module_op || mlir::failed(mlir::verify(*module_op))) {
    return absl::InternalError("Failed to convert Flatbuffer to MLIR module.");
  }

  return ModuleOpToJson(config, *module_op);
}

absl::StatusOr<std::string> ConvertMlirToJson(const VisualizeConfig& config,
                                              absl::string_view model_path) {
  mlir::DialectRegistry registry;
  // Note: This is more dialects than is currently visualized, but does include
  // what is commonly produced by different frameworks. So this would parse
  // correctly but then fail in visualization. This should result in a better
  // user experience than failing to parse here.
  registry.insert<mlir::TFL::TensorFlowLiteDialect, mlir::TF::TensorFlowDialect,
                  mlir::stablehlo::StablehloDialect, mlir::chlo::ChloDialect,
                  mlir::mhlo::MhloDialect, mlir::vhlo::VhloDialect,
                  mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::shape::ShapeDialect, mlir::scf::SCFDialect,
                  mlir::stablehlo::check::CheckDialect, mlir::sdy::SdyDialect,
                  mlir::tosa::TosaDialect>();
  mlir::MLIRContext context(registry);
  // Enable parsing of MLIR modules with unregistered dialects. This is safe as
  // Model Explorer does not execute operations, only visualizes them.
  context.allowUnregisteredDialects(true);

  std::string diagnostic_messages;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic& diag) {
    llvm::raw_string_ostream os(diagnostic_messages);
    os << diag;
  });

  mlir::ParserConfig parser_config(&context);
  std::string model_content;
  RETURN_IF_ERROR(tsl::ReadFileToString(
      tsl::Env::Default(), std::string(model_path), &model_content));
  auto module_op =
      mlir::parseSourceString<::mlir::ModuleOp>(model_content, parser_config);
  if (!module_op) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse MLIR module: ", diagnostic_messages));
  }

  RETURN_IF_ERROR(DeserializeVhloToStablehlo(*module_op));

  if (HasXlaCallModule(*module_op)) {
    RETURN_IF_ERROR(ConvertToStablehloModule(*module_op));
  }

  return ModuleOpToJson(config, *module_op);
}

}  // namespace visualization_client
}  // namespace tooling
