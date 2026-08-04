// Copyright 2026 The AI Edge Model Explorer Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "formats/litertlm/litertlm_read.h"

#include <cstddef>
#include <cstdint>
#include <ios>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

namespace litert {
namespace lm {
namespace schema {

namespace {

class MemoryStreamBuf : public std::streambuf {
 public:
  MemoryStreamBuf(char* data, std::size_t length) {
    setg(data, data, data + length);
  }

 protected:
  std::streampos seekoff(std::streamoff off, std::ios_base::seekdir dir,
                         std::ios_base::openmode which) override {
    if (!(which & std::ios_base::in)) {
      return -1;
    }
    char* new_gptr = nullptr;
    if (dir == std::ios_base::beg) {
      new_gptr = eback() + off;
    } else if (dir == std::ios_base::cur) {
      new_gptr = gptr() + off;
    } else if (dir == std::ios_base::end) {
      new_gptr = egptr() + off;
    } else {
      return -1;
    }
    if (new_gptr < eback() || new_gptr > egptr()) {
      return -1;
    }
    setg(eback(), new_gptr, egptr());
    return gptr() - eback();
  }

  std::streampos seekpos(std::streampos sp,
                         std::ios_base::openmode which) override {
    return seekoff(sp - pos_type(off_type(0)), std::ios_base::beg, which);
  }
};

constexpr char kLiteRtlmMagic[] = "LITERTLM";
constexpr size_t kLiteRtlmMagicLen = 8;

}  // namespace

bool IsLiteRTLMFile(absl::string_view content) {
  if (content.size() < kLiteRtlmMagicLen) {
    return false;
  }
  return content.substr(0, kLiteRtlmMagicLen) == kLiteRtlmMagic;
}

bool IsLiteRTLMFile(std::istream& stream) {
  std::streampos start_pos = stream.tellg();
  char magic_number[kLiteRtlmMagicLen];
  stream.read(magic_number, kLiteRtlmMagicLen);
  bool is_valid =
      (stream.gcount() == kLiteRtlmMagicLen &&
       absl::string_view(magic_number, kLiteRtlmMagicLen) == kLiteRtlmMagic);
  if (start_pos != -1) {
    stream.seekg(start_pos);
  }
  return is_valid;
}

absl::Status ReadHeaderFromLiteRTLM(std::istream& litertlm_stream,
                                    LitertlmHeader* header) {
  char magic_number[kLiteRtlmMagicLen];
  litertlm_stream.read(magic_number, kLiteRtlmMagicLen);
  if (litertlm_stream.gcount() != kLiteRtlmMagicLen ||
      absl::string_view(magic_number, kLiteRtlmMagicLen) != kLiteRtlmMagic) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid magic number or failed to read: %s",
                        std::string(magic_number, litertlm_stream.gcount())));
  }

  litertlm_stream.read(reinterpret_cast<char*>(&header->major_version),
                       sizeof(uint32_t));
  litertlm_stream.read(reinterpret_cast<char*>(&header->minor_version),
                       sizeof(uint32_t));
  litertlm_stream.read(reinterpret_cast<char*>(&header->patch_version),
                       sizeof(uint32_t));

  if (!litertlm_stream) {
    return absl::InternalError("Failed to read version bytes.");
  }

  litertlm_stream.ignore(4);
  if (!litertlm_stream) {
    return absl::InternalError("Failed to skip padding after version.");
  }

  uint64_t header_end_offset;
  litertlm_stream.read(reinterpret_cast<char*>(&header_end_offset),
                       sizeof(uint64_t));
  if (!litertlm_stream) {
    return absl::InternalError("Failed to read header end offset.");
  }

  std::streampos current_position = litertlm_stream.tellg();
  if (current_position == -1) {
    return absl::InternalError("Failed to get current stream position.");
  }
  if (header_end_offset < static_cast<uint64_t>(current_position)) {
    return absl::InvalidArgumentError(
        "Invalid header end offset: smaller than current position.");
  }
  uint64_t header_size =
      header_end_offset - static_cast<uint64_t>(current_position);

  constexpr uint64_t kMaxHeaderSize = 100 * 1024 * 1024;  // 100 MB
  if (header_size > kMaxHeaderSize) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid header size: ", header_size,
                     " bytes exceeds maximum allowed limit."));
  }

  std::unique_ptr<uint8_t[]> header_buffer = nullptr;
  if (header_size > 0) {
    if (header_size < 4) {
      return absl::InvalidArgumentError(
          "Invalid header size: non-empty header metadata must be at least 4 "
          "bytes.");
    }
    header_buffer = std::make_unique<uint8_t[]>(header_size);
    litertlm_stream.read(reinterpret_cast<char*>(header_buffer.get()),
                         header_size);
    if (!litertlm_stream) {
      return absl::InternalError("Failed to read header data.");
    }
  }

  header->reset(std::move(header_buffer), header_size);
  return absl::OkStatus();
}

absl::Status ReadHeaderFromLiteRTLM(void* data, std::size_t length,
                                    LitertlmHeader* header) {
  char* char_data = static_cast<char*>(data);
  MemoryStreamBuf sbuf(char_data, length);
  std::istream input_stream(&sbuf);
  return ReadHeaderFromLiteRTLM(input_stream, header);
}

}  // namespace schema
}  // namespace lm
}  // namespace litert
