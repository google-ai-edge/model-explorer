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

#include "models_to_json_lib.h"

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "direct_flatbuffer_to_json_graph_convert.h"
#include "direct_saved_model_to_json_graph_convert.h"
#include "mediapipe_adapter/mediapipe_to_json.h"
#include "model_json_graph_convert.h"
#include "status_macros.h"
#include "visualize_config.h"
#include "tensorflow/core/platform/logging.h"
#include "tsl/platform/path.h"

namespace tooling {
namespace visualization_client {

namespace {

enum ModelFormat {
  kFlatbuffer,
  kSavedModel,
  kMlir,
  kFlatbufferDirect,
  kSavedModelDirect,
  kGraphDefDirect,
  kMediapipePipeline,
};

absl::StatusOr<ModelFormat> GetModelFormat(absl::string_view input_file,
                                           const bool disable_mlir) {
  absl::string_view extension = tsl::io::Extension(input_file);
  if (extension.empty()) {
    // TF or JAX SavedModel.
    if (disable_mlir) {
      return kSavedModelDirect;
    } else {
      return kSavedModel;
    }
  }
  if (extension == "tflite") {
    // TFLite Flatbuffer.
    if (disable_mlir) {
      return kFlatbufferDirect;
    } else {
      return kFlatbuffer;
    }
  }
  if (extension == "mlirbc" || extension == "mlir") {
    // StableHLO module represented using MLIR textual or bytecode format.
    return kMlir;
  }
  if (extension == "pb" || extension == "pbtxt" || extension == "graphdef") {
    // TF GraphDef.
    return kGraphDefDirect;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported file extension: ", extension));
}

}  // namespace

absl::StatusOr<std::string> ConvertModelToJson(const VisualizeConfig& config,
                                               absl::string_view input_file,
                                               const bool disable_mlir) {
  ASSIGN_OR_RETURN(ModelFormat model_format,
                   GetModelFormat(input_file, disable_mlir));

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
    case kMediapipePipeline: {
      json_output = ConvertMediapipeToJson(config, input_file);
      break;
    }
    default: {
      // Should never happen.
      return absl::InternalError("Unknown model format.");
    }
  }

  return json_output;
}

}  // namespace visualization_client
}  // namespace tooling
