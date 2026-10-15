// Copyright 2026 The AI Edge Model Explorer Authors.
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

// What is LiteRT-LM's alone: its runtime options. Needs no model.

#include "model_debugger/runner/llm/litert_lm/litert_lm_runtime.h"

#include <filesystem>
#include <fstream>
#include <string>

#include "absl/status/status.h"
#include "google/protobuf/struct.pb.h"
#include "gtest/gtest.h"
#include "model_debugger/v1/runner.pb.h"

namespace model_debugger::runner::llm {
namespace {

class LiteRtLmTest : public testing::Test {
 protected:
  LiteRtLmTest() {
    dir_ = std::filesystem::temp_directory_path() /
           "model_debugger_runner_tests" / "litert_lm";
    std::filesystem::create_directories(dir_);
    model_ = dir_ / "empty.litertlm";
    std::ofstream(model_) << "";
    request_.set_device(v1::CPU);
    request_.set_context_length(1024);
  }

  google::protobuf::Value& Option(const char* name) {
    return (*request_.mutable_runtime_options()->mutable_fields())[name];
  }

  absl::Status Load() { return runtime_.Load(model_, dir_, request_).status(); }

  std::filesystem::path dir_;
  std::filesystem::path model_;
  v1::LoadRequest request_;
  LiteRtLmRuntime runtime_;
};

TEST_F(LiteRtLmTest, InfoDeclaresEveryLoadOption) {
  const std::string& schema = runtime_.Info().load_options_schema();
  for (const char* name :
       {"cpuThreads", "audioCpuThreads", "prefillChunkSize", "loraRank",
        "audioLoraRank", "enableYnnpack", "parallelFileSectionLoading",
        "singleThreadedExecution", "gpuMetalResidencySet", "visionBackend",
        "audioBackend", "activationDataType"}) {
    EXPECT_NE(schema.find(name), std::string::npos) << name;
  }
  EXPECT_EQ(schema.find("cacheDir"), std::string::npos)
      << "where the runtime writes is the Runner's, not an option";
}

TEST_F(LiteRtLmTest, AnOptionOfTheWrongTypeIsRefused) {
  Option("cpuThreads").set_string_value("four");
  EXPECT_TRUE(absl::IsInvalidArgument(Load()));
}

TEST_F(LiteRtLmTest, AnOptionThatIsNotAWholeNumberIsRefused) {
  Option("cpuThreads").set_number_value(2.5);
  EXPECT_TRUE(absl::IsInvalidArgument(Load()));
}

TEST_F(LiteRtLmTest, AValueOutsideTheSchemaIsRefused) {
  Option("activationDataType").set_string_value("fp16");
  EXPECT_TRUE(absl::IsInvalidArgument(Load()));
  Option("activationDataType").set_string_value("float16");
  Option("visionBackend").set_string_value("npu");
  EXPECT_TRUE(absl::IsInvalidArgument(Load()));
}

TEST_F(LiteRtLmTest, AnOptionTheRuntimeDoesNotHaveIsRefused) {
  Option("artisanMode").set_bool_value(true);
  EXPECT_TRUE(absl::IsInvalidArgument(Load()));
}

TEST_F(LiteRtLmTest, AMissingDataDirectoryIsAPreconditionNotAMissingModel) {
  EXPECT_TRUE(absl::IsFailedPrecondition(
      runtime_.Load(model_, dir_ / "no_such_directory", request_).status()));
}

}  // namespace
}  // namespace model_debugger::runner::llm
