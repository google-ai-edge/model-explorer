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
#include <string>
#include <vector>

#include "model_debugger/runner/util/json.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace model_debugger::runner::llm {
namespace {

// A KV cache as the GPU leaves it: what the kernel writes, then what the
// read-back makes of it, each step as ML Drift's source states it.
struct Case {
  bool key;
  size_t heads;
  size_t positions;
  size_t channels;
  bool int8;  // I8, else F32.
};

// The logical cache's element (h, p, c): its index, distinct in each case,
// shifted into the I8 range for an I8 case of at most 256 elements.
int Value(const Case& test, size_t h, size_t p, size_t c) {
  const int index =
      static_cast<int>((h * test.positions + p) * test.channels + c);
  return test.int8 ? index - 128 : index;
}

// The declared shape: K [1, heads, positions, channels], V [1, heads,
// channels, positions].
std::vector<size_t> Shape(const Case& test) {
  return test.key
             ? std::vector<size_t>{1, test.heads, test.positions, test.channels}
             : std::vector<size_t>{1, test.heads, test.channels,
                                   test.positions};
}

// Stores `value` as element `at` of `bytes`, in the case's dtype.
void Put(const Case& test, float value, std::string& bytes, size_t at) {
  if (test.int8) {
    bytes[at] = static_cast<char>(static_cast<int8_t>(value));
  } else {
    std::memcpy(&bytes[at * sizeof(float)], &value, sizeof(float));
  }
}

// The data of the logical cache, in the declared order.
std::string Logical(const Case& test) {
  const size_t element = test.int8 ? 1 : sizeof(float);
  std::string bytes(test.heads * test.positions * test.channels * element,
                    '\0');
  for (size_t h = 0; h < test.heads; ++h) {
    for (size_t p = 0; p < test.positions; ++p) {
      for (size_t c = 0; c < test.channels; ++c) {
        const size_t at = test.key
                              ? (h * test.positions + p) * test.channels + c
                              : (h * test.channels + c) * test.positions + p;
        Put(test, static_cast<float>(Value(test, h, p, c)), bytes, at);
      }
    }
  }
  return bytes;
}

// The data the GPU capture holds for the same cache.
std::string Captured(const Case& test) {
  const size_t element = test.int8 ? 1 : sizeof(float);
  // add_values_to_cache_kernel.cc: the vec4 of channels c..c+3 goes to
  // k_index or v_index; an int8 cache holds q + 128 as uint8.
  std::string buffer(test.heads * test.positions * test.channels * element,
                     '\0');
  for (size_t h = 0; h < test.heads; ++h) {
    for (size_t p = 0; p < test.positions; ++p) {
      for (size_t c = 0; c < test.channels; ++c) {
        const size_t i_slices =
            test.key ? test.channels / 4 : test.positions / 4;
        const size_t o_slices =
            test.key ? test.positions / 4 : test.channels / 4;
        const size_t vec4 =
            test.key ? ((h * i_slices + c / 4) * o_slices + p / 4) * 4 + p % 4
                     : ((h * i_slices + p / 4) * o_slices + c / 4) * 4 + p % 4;
        const int value = Value(test, h, p, c);
        if (test.int8) {
          buffer[vec4 * 4 + c % 4] = static_cast<char>(value + 128);
        } else {
          Put(test, static_cast<float>(value), buffer, vec4 * 4 + c % 4);
        }
      }
    }
  }
  // buffer_handler_utils.cc: host element (h, w, c) of the declared
  // [1, H, W, C] is buffer element ((c/4 * H + h) * W + w) * 4 + c%4.
  const std::vector<size_t> shape = Shape(test);
  const size_t heads = shape[1];
  const size_t width = shape[2];
  const size_t depth = shape[3];
  std::string host(buffer.size(), '\0');
  for (size_t h = 0; h < heads; ++h) {
    for (size_t w = 0; w < width; ++w) {
      for (size_t c = 0; c < depth; ++c) {
        const size_t from = ((c / 4 * heads + h) * width + w) * 4 + c % 4;
        std::memcpy(&host[((h * width + w) * depth + c) * element],
                    &buffer[from * element], element);
      }
    }
  }
  return host;
}

// Writes `data` as a safetensors file holding the tensor `name`.
absl::StatusOr<std::filesystem::path> Write(const std::string& file_name,
                                            const std::string& name,
                                            const std::string& dtype,
                                            const std::vector<size_t>& shape,
                                            const std::string& data) {
  Json header = {{name,
                  {{"dtype", dtype},
                   {"shape", shape},
                   {"data_offsets", {0, data.size()}}}},
                 {"__metadata__", {{"signature", "decode"}, {"step", "9"}}}};
  const absl::StatusOr<std::string> dumped = Dump(header);
  if (!dumped.ok()) return dumped.status();
  const std::string& text = *dumped;
  const uint64_t size = text.size();
  const std::filesystem::path path =
      std::filesystem::path(::testing::TempDir()) / file_name;
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(&size), sizeof(size));
  out << text << data;
  return path;
}

std::string Read(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

// The data part of the safetensors file at `path`.
std::string Data(const std::filesystem::path& path) {
  const std::string bytes = Read(path);
  uint64_t size = 0;
  std::memcpy(&size, bytes.data(), sizeof(size));
  return bytes.substr(sizeof(size) + size);
}

class UnpackMlDriftKvCacheTest : public ::testing::TestWithParam<Case> {};

TEST_P(UnpackMlDriftKvCacheTest, GivesBackTheCacheTheKernelWrote) {
  const Case& test = GetParam();
  const absl::StatusOr<std::filesystem::path> path = Write(
      "cache.safetensors", test.key ? "post_kv_cache_k_3" : "post_kv_cache_v_3",
      test.int8 ? "I8" : "F32", Shape(test), Captured(test));
  ASSERT_TRUE(path.ok()) << path.status();
  ASSERT_TRUE(UnpackMlDriftKvCache(*path).ok());
  EXPECT_EQ(Data(*path), Logical(test));
}

INSTANTIATE_TEST_SUITE_P(
    HeadsAndDtypes, UnpackMlDriftKvCacheTest,
    ::testing::Values(Case{true, 1, 16, 8, true}, Case{false, 1, 16, 8, true},
                      Case{true, 2, 16, 8, true}, Case{false, 2, 16, 8, true},
                      Case{true, 3, 8, 8, true}, Case{false, 3, 8, 8, true},
                      Case{true, 2, 16, 8, false},
                      Case{false, 2, 16, 8, false}));

TEST(UnpackMlDriftKvCache, RefusesDimensionsThatAreNotMultiplesOfFour) {
  const std::string data(1 * 1 * 6 * 8, '\x01');
  const absl::StatusOr<std::filesystem::path> path =
      Write("odd.safetensors", "post_kv_cache_k_0", "I8", {1, 1, 6, 8}, data);
  ASSERT_TRUE(path.ok()) << path.status();
  const std::string before = Read(*path);
  EXPECT_EQ(UnpackMlDriftKvCache(*path).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(Read(*path), before);
}

TEST(UnpackMlDriftKvCache, RefusesASizeBeyondTheFile) {
  const absl::StatusOr<std::filesystem::path> path =
      Write("huge.safetensors", "post_kv_cache_k_0", "I8",
            {1, size_t{1} << 62, 4, 4}, std::string());
  ASSERT_TRUE(path.ok()) << path.status();
  EXPECT_EQ(UnpackMlDriftKvCache(*path).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(UnpackMlDriftKvCache, RefusesATensorThatIsNotAKvCache) {
  const std::string data(1 * 1 * 4 * 8, '\x01');
  const absl::StatusOr<std::filesystem::path> path =
      Write("logits.safetensors", "post_logits", "I8", {1, 1, 4, 8}, data);
  ASSERT_TRUE(path.ok()) << path.status();
  EXPECT_EQ(UnpackMlDriftKvCache(*path).code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace model_debugger::runner::llm
