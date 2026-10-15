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

#include "model_debugger/runner/llm/litert_lm/ml_drift_kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <optional>
#include <string>
#include <system_error>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "model_debugger/runner/util/json.h"

namespace model_debugger::runner::llm {
namespace {

// Bytes per element of the dtypes a KV cache is captured in.
std::optional<size_t> ElementSize(absl::string_view dtype) {
  if (dtype == "I8") return 1;
  if (dtype == "F16") return 2;
  if (dtype == "F32") return 4;
  return std::nullopt;
}

// The element of the kernel's buffer that holds channel `c` of head `h` at
// position `p`: add_values_to_cache_kernel.cc's k_index and v_index, which
// count vec4s of four channels.
size_t KernelIndex(bool key, size_t h, size_t p, size_t c, size_t positions,
                   size_t channels) {
  const size_t vec4 =
      key ? ((h * (channels / 4) + c / 4) * (positions / 4) + p / 4) * 4 + p % 4
          : ((h * (positions / 4) + p / 4) * (channels / 4) + c / 4) * 4 +
                p % 4;
  return vec4 * 4 + c % 4;
}

// The host element the read-back made of buffer element `i`, unpacking the
// buffer as PHWC4 of the declared [1, `heads`, `width`, `depth`].
size_t HostIndex(size_t i, size_t heads, size_t width, size_t depth) {
  const size_t c4 = i % 4;
  i /= 4;
  const size_t w = i % width;
  i /= width;
  const size_t h = i % heads;
  const size_t slice = i / heads;
  return (h * width + w) * depth + slice * 4 + c4;
}

// Writes `contents` to `file` whole or not at all.
absl::Status Replace(const std::filesystem::path& file,
                     const std::string& contents) {
  std::filesystem::path part = file;
  part += ".unpacking";
  std::ofstream out(part, std::ios::binary);
  out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  out.close();
  std::error_code error;
  if (out.fail()) {
    std::filesystem::remove(part, error);
    return absl::InternalError(absl::StrCat("Cannot write ", part.string()));
  }
  std::filesystem::rename(part, file, error);
  if (error) {
    std::filesystem::remove(part, error);
    return absl::InternalError(
        absl::StrCat("Cannot replace ", file.string(), ": ", error.message()));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status UnpackMlDriftKvCache(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  std::string bytes((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
  if (!in.is_open() || in.bad()) {
    return absl::InternalError(absl::StrCat("Cannot read ", file.string()));
  }
  const auto refuse = [&file](absl::string_view why) {
    return absl::InvalidArgumentError(
        absl::StrCat(file.filename().string(), " ", why));
  };

  // Safetensors: the header's size as a little-endian uint64, which every
  // platform of the Runner is, then the header, then the data.
  uint64_t header_size = 0;
  if (bytes.size() < sizeof(header_size)) return refuse("is not safetensors.");
  std::memcpy(&header_size, bytes.data(), sizeof(header_size));
  if (header_size > bytes.size() - sizeof(header_size)) {
    return refuse("is not safetensors.");
  }
  const size_t data_start = sizeof(header_size) + header_size;
  absl::StatusOr<Json> header = ParseJson(
      absl::string_view(bytes).substr(sizeof(header_size), header_size));
  if (!header.ok() || !header->is_object()) {
    return refuse("has no safetensors header.");
  }
  const Json* tensor = nullptr;
  std::string name;
  for (const auto& [key, value] : header->items()) {
    if (key == "__metadata__") continue;
    if (tensor != nullptr) return refuse("holds more than one tensor.");
    tensor = &value;
    name = key;
  }
  const bool key = absl::StrContains(name, "kv_cache_k_");
  if (tensor == nullptr || !(key || absl::StrContains(name, "kv_cache_v_"))) {
    return refuse("holds no KV cache.");
  }
  if (!tensor->is_object() || !tensor->contains("dtype") ||
      !tensor->at("dtype").is_string() || !tensor->contains("shape") ||
      !tensor->contains("data_offsets")) {
    return refuse("has a malformed header.");
  }
  const std::string dtype = tensor->at("dtype").get<std::string>();
  const std::optional<size_t> element = ElementSize(dtype);
  if (!element) return refuse(absl::StrCat("has dtype ", dtype, "."));
  const Json& shape = tensor->at("shape");
  const Json& offsets = tensor->at("data_offsets");
  if (!shape.is_array() || shape.size() != 4 || !offsets.is_array() ||
      offsets.size() != 2) {
    return refuse("is not a [1, heads, x, y] tensor.");
  }
  for (const Json& value :
       {shape[0], shape[1], shape[2], shape[3], offsets[0], offsets[1]}) {
    if (!value.is_number_unsigned()) return refuse("has a malformed header.");
  }
  const size_t heads = shape[1].get<size_t>();
  const size_t width = shape[2].get<size_t>();
  const size_t depth = shape[3].get<size_t>();
  if (shape[0].get<size_t>() != 1 || heads == 0 || width == 0 ||
      width % 4 != 0 || depth == 0 || depth % 4 != 0) {
    return refuse("is not [1, heads, x, y] with x and y multiples of 4.");
  }
  // The size, never past the file's: the product cannot overflow.
  size_t size = *element;
  for (const size_t dimension : {heads, width, depth}) {
    if (dimension > bytes.size() / size) {
      return refuse("holds data of the wrong size.");
    }
    size *= dimension;
  }
  const size_t begin = offsets[0].get<size_t>();
  const size_t end = offsets[1].get<size_t>();
  if (begin > end || end - begin != size || end > bytes.size() - data_start) {
    return refuse("holds data of the wrong size.");
  }

  // K is declared [1, heads, positions, channels], V [1, heads, channels,
  // positions].
  const size_t positions = key ? width : depth;
  const size_t channels = key ? depth : width;
  const char* data = bytes.data() + data_start + begin;
  std::string unpacked(size, '\0');
  for (size_t h = 0; h < heads; ++h) {
    for (size_t p = 0; p < positions; ++p) {
      for (size_t c = 0; c < channels; ++c) {
        const size_t from =
            HostIndex(KernelIndex(key, h, p, c, positions, channels), heads,
                      width, depth);
        const size_t to = key ? (h * positions + p) * channels + c
                              : (h * channels + c) * positions + p;
        std::memcpy(&unpacked[to * *element], data + from * *element, *element);
      }
    }
  }
  // The kernel stores q + 128 as uint8; flipping the top bit gives q.
  if (dtype == "I8") {
    for (char& byte : unpacked) byte = static_cast<char>(byte ^ 0x80);
  }
  bytes.replace(data_start + begin, size, unpacked);
  return Replace(file, bytes);
}

}  // namespace model_debugger::runner::llm
