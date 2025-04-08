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
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "models_to_json_lib.h"
#include "visualize_config.h"
#include "tensorflow/compiler/mlir/lite/tools/command_line_flags.h"
#include "xla/tsl/platform/env.h"

constexpr char kInputFileFlag[] = "i";
constexpr char kOutputFileFlag[] = "o";
constexpr char kConstElementCountLimitFlag[] = "const_element_count_limit";
constexpr char kDisableMlirFlag[] = "disable_mlir";

namespace {

using ::tooling::visualization_client::ConvertModelToJson;

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
                             mlir::Flag::kOptional),
      mlir::Flag::CreateFlag(kOutputFileFlag, &output_file, "Output filename",
                             mlir::Flag::kOptional),
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

  if (output_file.empty()) {
    LOG(ERROR) << "Output filename cannot be empty.";
    return 1;
  }

  if (output_file.substr(output_file.size() - 4, 4) != "json") {
    LOG(WARNING) << "Please specify output format to be JSON.";
  }

  // Creates visualization config.
  tooling::visualization_client::VisualizeConfig config(
      const_element_count_limit);

  const absl::StatusOr<std::string> json_output =
      ConvertModelToJson(config, input_file, disable_mlir);
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
