#include "converter/model_json_graph_convert.h"

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
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "converter/status_macros.h"
#include "converter/translations.h"
#include "converter/visualize_config.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_tf_xla_call_module_to_stablehlo_pass.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/rename_entrypoint_to_main.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantize_preprocess.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_import_options.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/core/protobuf/saved_model.proto.h"
#include "tensorflow/core/protobuf/saved_object_graph.proto.h"
#include "tensorflow/core/protobuf/trackable_object_graph.proto.h"
#include "tensorflow/tsl/platform/env.h"

namespace tooling {
namespace visualization_client {
namespace {

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

// Obtains saved model tags and exported names from SavedModel proto.
absl::Status AssignTagsAndExportedNames(
    absl::string_view model_path, int tf_version, std::string& tags_str,
    std::vector<std::string>& exported_names) {
  tensorflow::SavedModel saved_model;
  RETURN_IF_ERROR(tensorflow::ReadSavedModel(model_path, &saved_model));
  if (saved_model.meta_graphs_size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Only `SavedModel`s with 1 MetaGraph are supported. Instead, it has ",
        saved_model.meta_graphs_size()));
  }
  const tensorflow::MetaGraphDef::MetaInfoDef& meta_info_def =
      saved_model.meta_graphs()[0].meta_info_def();
  tags_str = absl::StrJoin(meta_info_def.tags(), ",");

  // Only TF2 model needs it, TF1 model can use empty exported_names to apply
  // default values.
  if (tf_version == 2) {
    const tensorflow::SavedObjectGraph& object_graph_def =
        saved_model.meta_graphs()[0].object_graph_def();
    const auto& saved_object_nodes = object_graph_def.nodes();
    // According to saved_object_graph.proto, nodes[0] indicates root node.
    for (const auto& child : object_graph_def.nodes()[0].children()) {
      if (saved_object_nodes[child.node_id()].has_function()) {
        exported_names.push_back(child.local_name());
      }
    }
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

// Checks if module has tf.XlaCallModule ops.
bool HasXlaCallModule(mlir::ModuleOp module) {
  for (auto fn : module.getOps<mlir::func::FuncOp>()) {
    auto it = fn.getOps<mlir::TF::XlaCallModuleOp>();
    if (!it.empty()) {
      return true;
    }
  }
  return false;
}

// Deserializes tf.XlaCallModule ops and convert it to stablehlo module. Input
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

}  // namespace

absl::StatusOr<std::string> ConvertSavedModelV1ToJson(
    const VisualizeConfig& config, absl::string_view model_path) {
  std::string saved_model_tags;
  std::vector<std::string> exported_names_vector;
  RETURN_IF_ERROR(AssignTagsAndExportedNames(
      model_path, /*tf_version=*/1, saved_model_tags, exported_names_vector));
  std::unordered_set<std::string> tags = absl::StrSplit(saved_model_tags, ',');
  absl::Span<std::string> exported_names(exported_names_vector);

  mlir::MLIRContext context;
  tensorflow::MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;

  // Converts SavedModel V1 to MLIR module.
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module_op =
      tensorflow::SavedModelSignatureDefsToMlirImport(
          model_path, tags, exported_names, &context, import_options);
  RETURN_IF_ERROR(module_op.status());

  std::string json_output;
  llvm::raw_string_ostream json_ost(json_output);

  // Converts MLIR module to JSON string.
  if (HasXlaCallModule(**module_op)) {
    // This indicates it's a JAX converted SavedModel. There are stablehlo ops
    // serialized within tf.XlaCallModule op, we want to deserialize it before
    // proceeding.
    RETURN_IF_ERROR(ConvertToStablehloModule(**module_op));

    mlir::LogicalResult result =
        JaxConvertedMlirToJsonTranslate(**module_op, json_ost);
    if (mlir::failed(result)) {
      return absl::InternalError(
          "Failed to convert JAX converted MLIR module to JSON string.");
    }
  } else {
    mlir::LogicalResult result =
        TfMlirToJsonTranslateImpl(config, **module_op, json_ost);
    if (mlir::failed(result)) {
      return absl::InternalError(
          "Failed to convert TF MLIR module to JSON string.");
    }
  }

  return json_output;
}

absl::StatusOr<std::string> ConvertSavedModelV2ToJson(
    const VisualizeConfig& config, absl::string_view model_path) {
  std::string saved_model_tags;
  std::vector<std::string> exported_names_vector;
  RETURN_IF_ERROR(AssignTagsAndExportedNames(
      model_path, /*tf_version=*/2, saved_model_tags, exported_names_vector));
  std::unordered_set<std::string> tags = absl::StrSplit(saved_model_tags, ',');
  absl::Span<std::string> exported_names(exported_names_vector);
  mlir::MLIRContext context;

  // Converts SavedModel V2 to MLIR module.
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module_op =
      tensorflow::SavedModelObjectGraphToMlirImport(model_path, tags,
                                                    exported_names, &context);
  // TODO(b/310682603): Since v2 conversion oftentimes fail due to the multiple
  // input signature error, we want to enable two separate threads for running
  // v1 and v2 conversion at the same time.
  RETURN_IF_ERROR(module_op.status());

  // Converts tf_executor dialect to tf dialect MLIR module.
  RETURN_IF_ERROR(RunStandardPipeline(**module_op));

  std::string json_output;
  llvm::raw_string_ostream json_ost(json_output);

  // Converts MLIR module to JSON string.
  if (HasXlaCallModule(**module_op)) {
    // This indicates it's a JAX converted SavedModel. There are stablehlo ops
    // serialized within tf.XlaCallModule op, we want to deserialize it before
    // proceeding.
    RETURN_IF_ERROR(ConvertToStablehloModule(**module_op));

    mlir::LogicalResult result =
        JaxConvertedMlirToJsonTranslate(**module_op, json_ost);
    if (mlir::failed(result)) {
      return absl::InternalError(
          "Failed to convert JAX converted MLIR module to JSON string.");
    }
  } else {
    mlir::LogicalResult result =
        TfMlirToJsonTranslateImpl(config, **module_op, json_ost);
    if (mlir::failed(result)) {
      return absl::InternalError(
          "Failed to convert TF MLIR module to JSON string.");
    }
  }

  return json_output;
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
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());

  // Converts Flatbuffer to MLIR module.
  mlir::OwningOpRef<mlir::ModuleOp> module_op =
      FlatBufferFileToMlirTranslation(&sourceMgr, &context);
  if (!module_op || mlir::failed(mlir::verify(*module_op))) {
    return absl::InternalError("Failed to convert Flatbuffer to MLIR module.");
  }

  // Converts tfl dialect MLIR module to JSON string.
  std::string json_output;
  llvm::raw_string_ostream json_ost(json_output);
  mlir::LogicalResult result =
      TfliteMlirToJsonTranslateImpl(config, *module_op, json_ost);
  if (mlir::failed(result)) {
    return absl::InternalError("Failed to convert MLIR module to JSON string.");
  }

  return json_output;
}

}  // namespace visualization_client
}  // namespace tooling
