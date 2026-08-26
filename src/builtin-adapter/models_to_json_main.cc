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

#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "common/visualize_config.h"
#include "models_to_json_lib.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"

ABSL_FLAG(std::string, input_file, "", "Input filename or directory");
ABSL_FLAG(std::string, output_file, "", "Output filename");
ABSL_FLAG(std::string, i, "", "Alias for --input_file");
ABSL_FLAG(std::string, o, "", "Alias for --output_file");
ABSL_FLAG(int, const_element_count_limit, 16,
          "The maximum number of constant elements. If the number exceeds this "
          "threshold, the rest of data will be elided. If the flag is not set, "
          "the default threshold is 16 (use -1 to print all)");
ABSL_FLAG(bool, disable_mlir, false,
          "Disable the MLIR-based conversion. If set to true, the conversion "
          "becomes from model directly to graph json");

namespace {

using ::model_explorer::adapter::ConvertModelToJson;

}  // namespace

int main(int argc, char* argv[]) {
  constexpr char kUsage[] =
      "Converts ML models (TFLite, SavedModel, MLIR, LiteRT-LM) to Model "
      "Explorer JSON format.\nUsage: models_to_json "
      "--input_file=<input_model> --output_file=<output_json>";
  tensorflow::port::InitMain(kUsage, &argc, &argv);
  absl::ParseCommandLine(argc, argv);

  std::string input_file = absl::GetFlag(FLAGS_input_file);
  if (input_file.empty()) {
    input_file = absl::GetFlag(FLAGS_i);
  }
  std::string output_file = absl::GetFlag(FLAGS_output_file);
  if (output_file.empty()) {
    output_file = absl::GetFlag(FLAGS_o);
  }
  const int const_element_count_limit =
      absl::GetFlag(FLAGS_const_element_count_limit);
  const bool disable_mlir = absl::GetFlag(FLAGS_disable_mlir);

  if (input_file.empty() || output_file.empty()) {
    ABSL_LOG(ERROR) << "Input or output files cannot be empty.";
    return 1;
  }

  if (!absl::EndsWith(output_file, ".json")) {
    ABSL_LOG(WARNING) << "Please specify output format to be JSON.";
  }

  // Creates visualization config.
  model_explorer::adapter::VisualizeConfig config(const_element_count_limit);

  const absl::StatusOr<std::string> json_output =
      ConvertModelToJson(config, input_file, disable_mlir);
  if (!json_output.ok()) {
    ABSL_LOG(ERROR) << json_output.status();
    return 1;
  }

  absl::Status status =
      tsl::WriteStringToFile(tsl::Env::Default(), output_file, *json_output);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << status;
    return 1;
  }

  return 0;
}
