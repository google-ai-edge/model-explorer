#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "direct_flatbuffer_to_json_graph_convert.h"
#include "model_json_graph_convert.h"
#include "visualize_config.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tsl/platform/env.h"

constexpr char kInputFileFlag[] = "i";
constexpr char kOutputFileFlag[] = "o";
constexpr char kTfV2Flag[] = "tf_v2";
constexpr char kConstElementCountLimitFlag[] = "const_element_count_limit";
constexpr char kDisableMlirFlag[] = "disable_mlir";

namespace {

using ::tooling::visualization_client::ConvertFlatbufferDirectlyToJson;
using ::tooling::visualization_client::ConvertFlatbufferToJson;
using ::tooling::visualization_client::ConvertSavedModelV1ToJson;
using ::tooling::visualization_client::ConvertSavedModelV2ToJson;

enum ModelFormat {
  kFlatbuffer,
  kSavedModelV1,
  kSavedModelV2,
  kStablehloMlir,
};

}  // namespace

int main(int argc, char* argv[]) {
  tensorflow::InitMlir y(&argc, &argv);

  // Creates and parses flags.
  std::string input_file, output_file;
  bool use_tf_v2 = false;
  int const_element_count_limit = 16;
  bool disable_mlir = false;

  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kInputFileFlag, &input_file,
                               "Input filename or directory",
                               tflite::Flag::kRequired),
      tflite::Flag::CreateFlag(kOutputFileFlag, &output_file, "Output filename",
                               tflite::Flag::kRequired),
      tflite::Flag::CreateFlag(
          kTfV2Flag, &use_tf_v2,
          "Enable v2 object graph conversion for TF v2.x SavedModel",
          tflite::Flag::kOptional),
      tflite::Flag::CreateFlag(
          kConstElementCountLimitFlag, &const_element_count_limit,
          "The maximum number of constant elements. If the number exceeds this "
          "threshold, the rest of data will be elided. If the flag is not set, "
          "the default threshold is 16 (use -1 to print all)",
          tflite::Flag::kOptional),
      tflite::Flag::CreateFlag(
          kDisableMlirFlag, &disable_mlir,
          "Disable the MLIR-based conversion. If set to true, the conversion "
          "becomes from model directly to graph json",
          tflite::Flag::kOptional),
  };
  tflite::Flags::Parse(&argc, const_cast<const char**>(argv), flag_list);

  if (input_file.empty() || output_file.empty()) {
    LOG(ERROR) << "Input or output files cannot be empty.";
    return 1;
  }

  if (output_file.substr(output_file.size() - 4, 4) != "json") {
    LOG(WARNING) << "Please specify output format to be JSON.";
  }

  auto dot_idx = input_file.rfind('.');
  int n = input_file.size();
  ModelFormat model_format;
  if (dot_idx == std::string::npos) {
    // TF or JAX SavedModel
    if (use_tf_v2) {
      model_format = kSavedModelV2;
    } else {
      model_format = kSavedModelV1;
    }
  } else {
    std::string extension = input_file.substr(dot_idx, n - dot_idx);
    if (extension == ".tflite") {
      // TFLite Flatbuffer
      model_format = kFlatbuffer;
    } else if (extension == ".mlirbc" || extension == ".mlir") {
      // StableHLO module represented using MLIR textual or bytecode format.
      model_format = kStablehloMlir;
    } else {
      LOG(ERROR) << "Unsupported model format.";
      return 1;
    }
  }

  // Creates visualization config.
  tooling::visualization_client::VisualizeConfig config(
      const_element_count_limit);

  absl::StatusOr<std::string> json_output;
  switch (model_format) {
    case kFlatbuffer: {
      if (disable_mlir) {
        json_output = ConvertFlatbufferDirectlyToJson(config, input_file);
      } else {
        json_output =
            ConvertFlatbufferToJson(config, input_file, /*is_modelpath=*/true);
      }
      break;
    }
    case kStablehloMlir: {
      json_output = ConvertStablehloMlirToJson(config, input_file);
      break;
    }
    case kSavedModelV1: {
      json_output = ConvertSavedModelV1ToJson(config, input_file);
      break;
    }
    case kSavedModelV2: {
      json_output = ConvertSavedModelV2ToJson(config, input_file);
      break;
    }
    default: {
      LOG(ERROR) << "Unknown model format.";
      return 1;
    }
  }

  if (!json_output.ok()) {
    LOG(ERROR) << json_output.status();
    return 1;
  }

  absl::Status status =
      tsl::WriteStringToFile(tsl::Env::Default(), output_file, *json_output);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return 1;
  }

  return 0;
}
