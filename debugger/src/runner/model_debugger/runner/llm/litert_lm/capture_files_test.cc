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

#include "model_debugger/runner/llm/litert_lm/capture_files.h"

#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <set>
#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace model_debugger::runner::llm {
namespace {

// A safetensors file as the RuntimeDebugger writes one: a tensor of `data`,
// whose offsets end at `end`.
std::string Safetensors(const std::string& data, uint64_t end) {
  const std::string header = absl::StrCat(
      "{\"t\":{\"dtype\":\"U8\",\"shape\":[", end, "],\"data_offsets\":[0,",
      end, "]},\"__metadata__\":{\"step\":\"1\"}}");
  const uint64_t size = header.size();
  std::string file(sizeof(size), '\0');
  std::memcpy(file.data(), &size, sizeof(size));
  return absl::StrCat(file, header, data);
}

// A fresh directory holding `files`, by name.
std::filesystem::path Capture(
    const std::string& name,
    const std::set<std::pair<std::string, std::string>>& files) {
  const std::filesystem::path dir =
      std::filesystem::path(::testing::TempDir()) / name;
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  for (const auto& [file, contents] : files) {
    std::ofstream(dir / file, std::ios::binary) << contents;
  }
  return dir;
}

TEST(CaptureFilesTest, WholeFilesPassAndOtherNamesAreNotLookedAt) {
  const std::filesystem::path dir =
      Capture("whole", {{"a.safetensors", Safetensors("abcd", 4)},
                        {"generated_tokens.jsonl", "{\"token_ids\":[[1]]"}});
  EXPECT_TRUE(
      CheckWholeFiles(dir, {"a.safetensors", "generated_tokens.jsonl"}).ok());
}

TEST(CaptureFilesTest, AFileCutShortOnADiskWithRoomIsInternal) {
  const std::string file = Safetensors("ab", 4);
  const std::filesystem::path dir = Capture("short", {{"a.safetensors", file}});
  const absl::Status status = CheckWholeFiles(dir, {"a.safetensors"});
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(status.message(), "a.safetensors is cut short: it holds " +
                                  std::to_string(file.size()) +
                                  " bytes of at least " +
                                  std::to_string(file.size() + 2) + ".");
}

TEST(CaptureFilesTest, AFileCutShortInItsHeaderIsInternal) {
  const std::string file = Safetensors("abcd", 4);
  const std::filesystem::path dir =
      Capture("header", {{"a.safetensors", file.substr(0, 12)},
                         {"b.safetensors", file.substr(0, 3)}});
  EXPECT_EQ(CheckWholeFiles(dir, {"a.safetensors"}).message(),
            "a.safetensors is cut short: it holds 12 bytes of at least " +
                std::to_string(file.size() - 4) + ".");
  EXPECT_EQ(CheckWholeFiles(dir, {"b.safetensors"}).message(),
            "b.safetensors is cut short: it holds 3 bytes of at least 8.");
}

TEST(CaptureFilesTest, AFileThatIsNotSafetensorsIsInternal) {
  const std::filesystem::path dir =
      Capture("other", {{"a.safetensors", std::string(8, '\0') + "x"}});
  const absl::Status status = CheckWholeFiles(dir, {"a.safetensors"});
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(status.message(), "a.safetensors has no safetensors header.");
}

TEST(CaptureFilesDeathTest, AFileCutShortByMoreThanTheDiskHoldsEndsTheRunner) {
  // Offsets past any disk's size: the rest of the file can never fit.
  const std::filesystem::path dir = Capture(
      "full", {{"a.safetensors", Safetensors("ab", uint64_t{1} << 62)}});
  EXPECT_EXIT(CheckWholeFiles(dir, {"a.safetensors"}).IgnoreError(),
              ::testing::ExitedWithCode(1),
              "runner: no space left on device while writing "
              ".*/full/a\\.safetensors: [0-9]+ bytes did not fit");
}

}  // namespace
}  // namespace model_debugger::runner::llm
