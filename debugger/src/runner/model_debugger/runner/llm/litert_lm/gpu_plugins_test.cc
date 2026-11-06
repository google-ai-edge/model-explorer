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

#include "model_debugger/runner/llm/litert_lm/gpu_plugins.h"

#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>

#include "litert_lm/plugins.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"

namespace model_debugger::runner::llm {
namespace {

std::filesystem::path EmptyDirectory() {
  const std::filesystem::path dir =
      std::filesystem::path(::testing::TempDir()) / "plugins";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir;
}

TEST(GpuPluginsTest, AMissingPluginIsAPrecondition) {
  const std::filesystem::path dir = EmptyDirectory();
  EXPECT_FALSE(HasLiteRtGpuPlugins(dir.string()));
  EXPECT_TRUE(absl::IsFailedPrecondition(LoadLiteRtGpuPlugins(dir.string())));
}

TEST(GpuPluginsTest, APluginThatDoesNotLoadIsAPreconditionToo) {
  const std::filesystem::path dir = EmptyDirectory();
  for (const char* name : litert_lm::kGpuPlugins) {
    std::ofstream(dir / name) << "not a library";
  }
  EXPECT_TRUE(HasLiteRtGpuPlugins(dir.string()));
  const absl::Status status = LoadLiteRtGpuPlugins(dir.string());
  EXPECT_TRUE(absl::IsFailedPrecondition(status)) << status;
}

}  // namespace
}  // namespace model_debugger::runner::llm
