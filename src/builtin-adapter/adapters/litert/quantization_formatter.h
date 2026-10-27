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
// =============================================================================

#ifndef MODEL_EXPLORER_BACKEND_ADAPTERS_LITERT_QUANTIZATION_FORMATTER_H_
#define MODEL_EXPLORER_BACKEND_ADAPTERS_LITERT_QUANTIZATION_FORMATTER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

namespace model_explorer {
namespace adapter {

// Returns the raw bytes backing a tensor. An OK, empty span means the tensor
// has no data in the model (e.g. unloaded external weights); an error means the
// tensor's buffer reference is malformed.
using TensorBufferResolver =
    absl::FunctionRef<absl::StatusOr<absl::Span<const uint8_t>>(
        const tflite::TensorT&)>;

// Display metadata derived from one tensor's quantization parameters.
struct QuantizationMetadata {
  // Ordered (key, value) attributes for the tensor's node metadata entry.
  std::vector<std::pair<std::string, std::string>> attrs;
  // The first problem found while decoding the parameters, if any. Structural
  // attributes (e.g. `quantized_dimension` or `block_size`) are kept in `attrs`
  // even when `issue` is set.
  std::optional<std::string> issue;
};

// Builds the display metadata for `tensor`'s quantization parameters.
//
// Handles affine (per-tensor / per-channel) parameters as well as
// `BlockwiseQuantization` and `MultiAxisQuantization` details, reading scale
// and zero-point tensors from `tensors` via `resolve_buffer`.
//
// The `quantization` attribute lists up to `formula_limit` dequantization
// formulas (all when negative, none when zero), one per broadcast
// (scale, zero point) pair, e.g. `0.5 * (q - 2),0.25 * q`, with a trailing
// `…(+K more)` entry when truncated. Returns empty metadata when `tensor` is
// not quantized.
QuantizationMetadata BuildQuantizationMetadata(
    const tflite::TensorT& tensor,
    absl::Span<const std::unique_ptr<tflite::TensorT>> tensors,
    TensorBufferResolver resolve_buffer, int64_t formula_limit);

// Building blocks of `BuildQuantizationMetadata`, exposed for unit testing.
namespace quantization_internal {

// Formats `scale * q` or `scale * (q -/+ |zero_point|)`.
std::string FormatAffineFormula(float scale, int64_t zero_point);

// Joins `formulas` with commas, appending a `…(+K more)` entry when
// `total_count` exceeds the number of formulas shown.
std::string JoinFormulas(absl::Span<const std::string> formulas,
                         int64_t total_count);

// Returns the storage width of one `type` element in bits, or 0 if `type` is
// not a supported scale or zero-point quantization parameter type.
int GetElementBitWidth(tflite::TensorType type);

// Returns the number of elements in a fully static `shape`, or 0 if `shape` is
// empty, has a non-positive dimension, or overflows `int64_t`.
int64_t GetStaticElementCount(absl::Span<const int32_t> shape);

// Reads the little-endian float scale at element `index` of `data` (FLOAT16,
// BFLOAT16, FLOAT32, or FLOAT64). Returns `std::nullopt` for other types or an
// out-of-range `index`.
std::optional<float> ReadScaleAt(tflite::TensorType type,
                                 absl::Span<const uint8_t> data, int64_t index);

// Reads the little-endian integer zero point at element `index` of `data`.
// Sub-byte types (INT4, UINT4, INT2) are unpacked low bits first. Returns
// `std::nullopt` for unsupported types, an out-of-range `index`, or a UINT64
// value above `INT64_MAX`.
std::optional<int64_t> ReadZeroPointAt(tflite::TensorType type,
                                       absl::Span<const uint8_t> data,
                                       int64_t index);

// Returns the expected block grid `ceil(tensor_shape / block_shape)` for
// blockwise scale and zero-point tensors. An empty `block_shape` defaults to
// `[1, ..., 1, block_size]`. Returns `std::nullopt` when `tensor_shape` is not
// static or the block description is invalid.
std::optional<std::vector<int64_t>> ComputeBlockGrid(
    absl::Span<const int32_t> tensor_shape,
    absl::Span<const int32_t> block_shape, int32_t block_size);

// Checks that `param_shape` matches `block_grid`, right-aligned, where each
// axis either equals the grid extent or is 1 (broadcast). Missing leading axes
// count as 1. Rank-0 parameters always match.
absl::Status CheckParamShapeMatchesBlockGrid(
    absl::Span<const int32_t> param_shape,
    absl::Span<const int64_t> block_grid);

// Shape information for one quantization parameter tensor.
struct ParamShape {
  // The tensor's declared shape; an empty shape is a scalar. Views
  // `tflite::TensorT::shape` (or `shape_signature`), which must outlive this
  // struct (do not bind a named `ParamShape` variable to a temporary array).
  absl::Span<const int32_t> shape;
  // Number of elements readable from the tensor's buffer.
  int64_t element_count = 0;
  // Extra trailing elements that may be sub-byte padding rather than data
  // (non-zero only when `element_count` comes from the buffer size).
  int64_t padding_elements = 0;
};

// Pairs each scale with its zero point via right-aligned (NumPy-style)
// broadcasting, with fast paths for equal shapes and scalar operands.
class BroadcastPlan {
 public:
  struct IndexPair {
    int64_t scale_index = 0;
    int64_t zero_point_index = 0;

    friend bool operator==(const IndexPair&, const IndexPair&) = default;

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const IndexPair& pair) {
      absl::Format(&sink, "{scale: %d, zero_point: %d}", pair.scale_index,
                   pair.zero_point_index);
    }
  };

  // Returns the plan for `scales` and `zero_points`, or `std::nullopt` if
  // their shapes cannot be broadcast together.
  static std::optional<BroadcastPlan> Create(const ParamShape& scales,
                                             const ParamShape& zero_points);

  // Number of (scale, zero point) pairs.
  int64_t size() const { return size_; }

  // Maps a flat index in [0, size()) to element indices of the two tensors.
  IndexPair Map(int64_t flat_index) const;

 private:
  enum class Mode { kOneToOne, kScalarZeroPoint, kScalarScale, kBroadcast };

  BroadcastPlan(Mode mode, int64_t size) : mode_(mode), size_(size) {}

  Mode mode_;
  int64_t size_;
  // Per-axis extents of the broadcast shape and of each operand, right-aligned
  // to the same rank. Only used in `Mode::kBroadcast`.
  llvm::SmallVector<int64_t, 4> broadcast_dims_;
  llvm::SmallVector<int64_t, 4> scale_dims_;
  llvm::SmallVector<int64_t, 4> zero_point_dims_;
};

}  // namespace quantization_internal
}  // namespace adapter
}  // namespace model_explorer

#endif  // MODEL_EXPLORER_BACKEND_ADAPTERS_LITERT_QUANTIZATION_FORMATTER_H_
