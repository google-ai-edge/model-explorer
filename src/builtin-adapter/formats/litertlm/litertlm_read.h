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

#ifndef FORMATS_LITERTLM_LITERTLM_READ_H_
#define FORMATS_LITERTLM_LITERTLM_READ_H_

#include <cstddef>
#include <cstdint>
#include <istream>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/verifier.h"
#include "formats/litertlm/litertlm_header_schema_generated.h"

namespace litert {
namespace lm {
namespace schema {

bool IsLiteRTLMFile(absl::string_view content);

bool IsLiteRTLMFile(std::istream& stream);

struct LitertlmHeader {
  std::unique_ptr<uint8_t[]> buffer;
  const LiteRTLMMetaData* metadata;
  uint32_t major_version = 0;
  uint32_t minor_version = 0;
  uint32_t patch_version = 0;

  LitertlmHeader() : buffer(nullptr), metadata(nullptr) {}

  explicit LitertlmHeader(std::unique_ptr<uint8_t[]> buffer_, size_t size = 0)
      : buffer(nullptr), metadata(nullptr) {
    reset(std::move(buffer_), size);
  }

  LitertlmHeader(const LitertlmHeader&) = delete;
  LitertlmHeader& operator=(const LitertlmHeader&) = delete;

  LitertlmHeader(LitertlmHeader&& other) noexcept
      : buffer(std::move(other.buffer)),
        metadata(other.metadata),
        major_version(other.major_version),
        minor_version(other.minor_version),
        patch_version(other.patch_version) {
    other.metadata = nullptr;
    other.major_version = 0;
    other.minor_version = 0;
    other.patch_version = 0;
  }

  LitertlmHeader& operator=(LitertlmHeader&& other) noexcept {
    if (this != &other) {
      buffer = std::move(other.buffer);
      metadata = other.metadata;
      major_version = other.major_version;
      minor_version = other.minor_version;
      patch_version = other.patch_version;
      other.metadata = nullptr;
      other.major_version = 0;
      other.minor_version = 0;
      other.patch_version = 0;
    }
    return *this;
  }

  void reset(std::unique_ptr<uint8_t[]> buffer_, size_t size = 0) {
    buffer = std::move(buffer_);
    if (buffer && size >= 4) {
      flatbuffers::Verifier verifier(buffer.get(), size);
      if (VerifyLiteRTLMMetaDataBuffer(verifier)) {
        metadata = GetLiteRTLMMetaData(buffer.get());
        return;
      }
    }
    metadata = nullptr;
  }

  ~LitertlmHeader() = default;
};

absl::Status ReadHeaderFromLiteRTLM(void* data, size_t length,
                                    LitertlmHeader* header);

absl::Status ReadHeaderFromLiteRTLM(std::istream& litertlm_stream,
                                    LitertlmHeader* header);

}  // namespace schema
}  // namespace lm
}  // namespace litert

#endif  // FORMATS_LITERTLM_LITERTLM_READ_H_
