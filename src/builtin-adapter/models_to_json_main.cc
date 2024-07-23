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

#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "direct_flatbuffer_to_json_graph_convert.h"
#include "direct_saved_model_to_json_graph_convert.h"
#include "model_json_graph_convert.h"
#include "visualize_config.h"
#include "tensorflow/compiler/mlir/lite/tools/command_line_flags.h"
#include "tensorflow/core/platform/logging.h"
#include "tsl/platform/env.h"

constexpr char kInputFileFlag[] = "i";
constexpr char kOutputFileFlag[] = "o";
constexpr char kConstElementCountLimitFlag[] = "const_element_count_limit";
constexpr char kDisableMlirFlag[] = "disable_mlir";

namespace {

using ::tooling::visualization_client::ConvertFlatbufferDirectlyToJson;
using ::tooling::visualization_client::ConvertFlatbufferToJson;
using ::tooling::visualization_client::ConvertSavedModelDirectlyToJson;
using ::tooling::visualization_client::ConvertSavedModelToJson;

enum ModelFormat {
  kFlatbuffer,
  kSavedModel,
  kMlir,
  kFlatbufferDirect,
  kSavedModelDirect,
  kGraphDefDirect,
};

}  // namespace

int main(int argc, char* argv[]) {
  tensorflow::InitMlir y(&argc, &argv);

  // Creates and parses flags.
  std::string input_file, output_file;
  int const_element_count_limit = 16;
  bool disable_mlir = false;

  std::vector<mlir::Flag> flag_list = {
      mlir::Flag::CreateFlag(kInputFileFlag, &input_file,
                             "Input filename or directory",
                             mlir::Flag::kRequired),
      mlir::Flag::CreateFlag(kOutputFileFlag, &output_file, "Output filename",
                             mlir::Flag::kRequired),
      mlir::Flag::CreateFlag(
          kConstElementCountLimitFlag, &const_element_count_limit,
          "The maximum number of constant elements. If the number exceeds this "
          "threshold, the rest of data will be elided. If the flag is not set, "
          "the default threshold is 16 (use -1 to print all)",
          mlir::Flag::kOptional),
      mlir::Flag::CreateFlag(
          kDisableMlirFlag, &disable_mlir,
          "Disable the MLIR-based conversion. If set to true, the conversion "
          "becomes from model directly to graph json",
          mlir::Flag::kOptional),
  };
  mlir::Flags::Parse(&argc, const_cast<const char**>(argv), flag_list);

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
    if (disable_mlir) {
      model_format = kSavedModelDirect;
    } else {
      model_format = kSavedModel;
    }
  } else {
    std::string extension = input_file.substr(dot_idx, n - dot_idx);
    if (extension == ".tflite") {
      // TFLite Flatbuffer
      if (disable_mlir) {
        model_format = kFlatbufferDirect;
      } else {
        model_format = kFlatbuffer;
      }
    } else if (extension == ".mlirbc" || extension == ".mlir") {
      // StableHLO module represented using MLIR textual or bytecode format.
      model_format = kMlir;
    } else if (extension == ".pb" || extension == ".pbtxt" ||
               extension == ".graphdef") {
      model_format = kGraphDefDirect;
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
      json_output =
          ConvertFlatbufferToJson(config, input_file, /*is_modelpath=*/true);
      break;
    }
    case kMlir: {
      json_output = ConvertMlirToJson(config, input_file);
      break;
    }
    case kSavedModel: {
      json_output = ConvertSavedModelToJson(config, input_file);
      break;
    }
    case kFlatbufferDirect: {
      json_output = ConvertFlatbufferDirectlyToJson(config, input_file);
      break;
    }
    case kSavedModelDirect: {
      json_output = ConvertSavedModelDirectlyToJson(config, input_file);
      break;
    }
    case kGraphDefDirect: {
      json_output = ConvertGraphDefDirectlyToJson(config, input_file);
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
