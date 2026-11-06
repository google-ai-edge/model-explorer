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

#include <algorithm>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <system_error>  // NOLINT(build/c++11)

#include "model_debugger/runner/util/disk_space.h"
#include "model_debugger/runner/util/json.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"

namespace model_debugger::runner::llm {
namespace {

// The ending of the files the RuntimeDebugger captures tensors in.
constexpr std::string_view kSuffix = ".safetensors";

// The largest header the safetensors format allows.
constexpr uint64_t kMaxHeaderBytes = 100'000'000;

// Checks one file, as CheckWholeFiles.
absl::Status CheckWholeFile(const std::filesystem::path& file) {
  const std::string name = file.filename().string();
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(file, error);
  std::ifstream in(file, std::ios::binary);
  if (error || !in.is_open()) {
    return absl::InternalError(absl::StrCat("Cannot read ", name, "."));
  }
  // What the file needs at least: the header's size as a little-endian
  // uint64, then the header, then the data its tensors' offsets end at.
  uint64_t header_size = 0;
  std::uintmax_t needs = sizeof(header_size);
  if (size >= needs) {
    in.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    if (header_size > kMaxHeaderBytes) {
      return absl::InternalError(absl::StrCat(name, " is not safetensors."));
    }
    needs += header_size;
  }
  if (size >= needs) {
    std::string text(header_size, '\0');
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    absl::StatusOr<Json> header = ParseJson(text);
    if (!in || !header.ok() || !header->is_object()) {
      return absl::InternalError(
          absl::StrCat(name, " has no safetensors header."));
    }
    uint64_t end = 0;
    for (const auto& [key, tensor] : header->items()) {
      if (key == "__metadata__") continue;
      if (!tensor.is_object() || !tensor.contains("data_offsets") ||
          !tensor.at("data_offsets").is_array() ||
          tensor.at("data_offsets").size() != 2 ||
          !tensor.at("data_offsets")[1].is_number_unsigned()) {
        return absl::InternalError(
            absl::StrCat(name, " has a malformed header."));
      }
      end = std::max(end, tensor.at("data_offsets")[1].get<uint64_t>());
    }
    if (end > std::numeric_limits<std::uintmax_t>::max() - needs) {
      return absl::InternalError(
          absl::StrCat(name, " has a malformed header."));
    }
    needs += end;
  }
  if (size >= needs) return absl::OkStatus();
  ExitIfOutOfRoom(file.string(), needs - size);
  return absl::InternalError(absl::StrCat(name, " is cut short: it holds ",
                                          size, " bytes of at least ", needs,
                                          "."));
}

}  // namespace

absl::Status CheckWholeFiles(const std::filesystem::path& dir,
                             const std::set<std::string>& names) {
  for (const std::string& name : names) {
    if (!absl::EndsWith(name, kSuffix)) continue;
    if (absl::Status status = CheckWholeFile(dir / name); !status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

}  // namespace model_debugger::runner::llm
