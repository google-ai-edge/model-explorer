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

#include "adapters/litert/quantization_formatter.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/functional/function_ref.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Endian.h"
#include "utils/convert_type.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

namespace model_explorer {
namespace adapter {
namespace quantization_internal {
namespace {

// Returns the bit width of `type` if it is a supported scale element type, or
// 0 otherwise.
int ScaleBitWidth(tflite::TensorType type) {
  switch (type) {
    case tflite::TensorType_FLOAT64:
      return 64;
    case tflite::TensorType_FLOAT32:
      return 32;
    case tflite::TensorType_FLOAT16:
    case tflite::TensorType_BFLOAT16:
      return 16;
    default:
      return 0;
  }
}

// Returns the bit width of `type` if it is a supported zero-point element
// type, or 0 otherwise.
int ZeroPointBitWidth(tflite::TensorType type) {
  switch (type) {
    case tflite::TensorType_INT64:
    case tflite::TensorType_UINT64:
      return 64;
    case tflite::TensorType_INT32:
    case tflite::TensorType_UINT32:
      return 32;
    case tflite::TensorType_INT16:
    case tflite::TensorType_UINT16:
      return 16;
    case tflite::TensorType_INT8:
    case tflite::TensorType_UINT8:
      return 8;
    case tflite::TensorType_INT4:
    case tflite::TensorType_UINT4:
      return 4;
    case tflite::TensorType_INT2:
      return 2;
    default:
      return 0;
  }
}

// Reads the little-endian `T` at element `index` of `data`.
template <typename T>
std::optional<T> ReadLittleEndianAt(absl::Span<const uint8_t> data,
                                    int64_t index) {
  if (index < 0 || static_cast<size_t>(index) >= data.size() / sizeof(T)) {
    return std::nullopt;
  }
  return llvm::support::endian::read<T, llvm::endianness::little>(
      data.data() + static_cast<size_t>(index) * sizeof(T));
}

// Reads the little-endian integer `T` at element `index` of `data`, widened
// to `int64_t`.
template <typename T>
std::optional<int64_t> ReadIntAt(absl::Span<const uint8_t> data,
                                 int64_t index) {
  std::optional<T> value = ReadLittleEndianAt<T>(data, index);
  if (!value.has_value()) return std::nullopt;
  return static_cast<int64_t>(*value);
}

// Reads the `bit_width`-bit element at `index` of tightly packed `data`, low
// bits first, sign-extending it when `is_signed`.
std::optional<int64_t> ReadPackedIntAt(absl::Span<const uint8_t> data,
                                       int64_t index, int bit_width,
                                       bool is_signed) {
  if (index < 0) return std::nullopt;
  const int64_t elements_per_byte = 8 / bit_width;
  const size_t byte_index = static_cast<size_t>(index / elements_per_byte);
  if (byte_index >= data.size()) return std::nullopt;
  const int shift = static_cast<int>(index % elements_per_byte) * bit_width;
  const int64_t raw = (data[byte_index] >> shift) & ((1 << bit_width) - 1);
  if (is_signed && raw >= (int64_t{1} << (bit_width - 1))) {
    return raw - (int64_t{1} << bit_width);
  }
  return raw;
}

}  // namespace

std::string FormatAffineFormula(float scale, int64_t zero_point) {
  if (zero_point == 0) {
    return absl::StrFormat("%g * q", scale);
  }
  const char sign = zero_point < 0 ? '+' : '-';
  // Negating in unsigned arithmetic avoids overflow for INT64_MIN.
  const uint64_t magnitude = zero_point < 0
                                 ? 0ULL - static_cast<uint64_t>(zero_point)
                                 : static_cast<uint64_t>(zero_point);
  return absl::StrFormat("%g * (q %c %u)", scale, sign, magnitude);
}

std::string JoinFormulas(absl::Span<const std::string> formulas,
                         int64_t total_count) {
  std::string joined = absl::StrJoin(formulas, ",");
  const int64_t omitted = total_count - static_cast<int64_t>(formulas.size());
  if (omitted > 0) {
    absl::StrAppend(&joined, formulas.empty() ? "" : ",", "\u2026(+", omitted,
                    " more)");
  }
  return joined;
}

int GetElementBitWidth(tflite::TensorType type) {
  const int scale_bits = ScaleBitWidth(type);
  return scale_bits > 0 ? scale_bits : ZeroPointBitWidth(type);
}

int64_t GetStaticElementCount(absl::Span<const int32_t> shape) {
  if (shape.empty()) return 0;
  int64_t count = 1;
  for (const int32_t dim : shape) {
    if (dim <= 0 || count > std::numeric_limits<int64_t>::max() / dim) {
      return 0;
    }
    count *= dim;
  }
  return count;
}

std::optional<float> ReadScaleAt(tflite::TensorType type,
                                 absl::Span<const uint8_t> data,
                                 int64_t index) {
  switch (type) {
    case tflite::TensorType_FLOAT16: {
      std::optional<uint16_t> bits = ReadLittleEndianAt<uint16_t>(data, index);
      if (!bits.has_value()) return std::nullopt;
      return static_cast<float>(Eigen::numext::bit_cast<Eigen::half>(*bits));
    }
    case tflite::TensorType_BFLOAT16: {
      std::optional<uint16_t> bits = ReadLittleEndianAt<uint16_t>(data, index);
      if (!bits.has_value()) return std::nullopt;
      return static_cast<float>(
          Eigen::numext::bit_cast<Eigen::bfloat16>(*bits));
    }
    case tflite::TensorType_FLOAT32: {
      std::optional<uint32_t> bits = ReadLittleEndianAt<uint32_t>(data, index);
      if (!bits.has_value()) return std::nullopt;
      return llvm::bit_cast<float>(*bits);
    }
    case tflite::TensorType_FLOAT64: {
      std::optional<uint64_t> bits = ReadLittleEndianAt<uint64_t>(data, index);
      if (!bits.has_value()) return std::nullopt;
      return static_cast<float>(llvm::bit_cast<double>(*bits));
    }
    default:
      return std::nullopt;
  }
}

std::optional<int64_t> ReadZeroPointAt(tflite::TensorType type,
                                       absl::Span<const uint8_t> data,
                                       int64_t index) {
  switch (type) {
    case tflite::TensorType_INT64:
      return ReadIntAt<int64_t>(data, index);
    case tflite::TensorType_UINT64: {
      std::optional<uint64_t> value = ReadLittleEndianAt<uint64_t>(data, index);
      if (!value.has_value() ||
          *value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return std::nullopt;
      }
      return static_cast<int64_t>(*value);
    }
    case tflite::TensorType_INT32:
      return ReadIntAt<int32_t>(data, index);
    case tflite::TensorType_UINT32:
      return ReadIntAt<uint32_t>(data, index);
    case tflite::TensorType_INT16:
      return ReadIntAt<int16_t>(data, index);
    case tflite::TensorType_UINT16:
      return ReadIntAt<uint16_t>(data, index);
    case tflite::TensorType_INT8:
      return ReadIntAt<int8_t>(data, index);
    case tflite::TensorType_UINT8:
      return ReadIntAt<uint8_t>(data, index);
    case tflite::TensorType_INT4:
      return ReadPackedIntAt(data, index, /*bit_width=*/4, /*is_signed=*/true);
    case tflite::TensorType_UINT4:
      return ReadPackedIntAt(data, index, /*bit_width=*/4, /*is_signed=*/false);
    case tflite::TensorType_INT2:
      return ReadPackedIntAt(data, index, /*bit_width=*/2, /*is_signed=*/true);
    default:
      return std::nullopt;
  }
}

std::optional<std::vector<int64_t>> ComputeBlockGrid(
    absl::Span<const int32_t> tensor_shape,
    absl::Span<const int32_t> block_shape, int32_t block_size) {
  if (GetStaticElementCount(tensor_shape) == 0) return std::nullopt;
  const size_t rank = tensor_shape.size();

  std::vector<int32_t> resolved_block_shape(block_shape.begin(),
                                            block_shape.end());
  if (resolved_block_shape.empty()) {
    if (block_size <= 0) return std::nullopt;
    resolved_block_shape.assign(rank, 1);
    resolved_block_shape.back() = block_size;
  }
  if (resolved_block_shape.size() != rank) return std::nullopt;

  std::vector<int64_t> grid;
  grid.reserve(rank);
  for (size_t axis = 0; axis < rank; ++axis) {
    const int64_t block_extent = resolved_block_shape[axis];
    if (block_extent <= 0) return std::nullopt;
    grid.push_back((tensor_shape[axis] + block_extent - 1) / block_extent);
  }
  return grid;
}

absl::Status CheckParamShapeMatchesBlockGrid(
    absl::Span<const int32_t> param_shape,
    absl::Span<const int64_t> block_grid) {
  if (param_shape.size() > block_grid.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "shape [%s] has more axes than the expected block grid [%s]",
        absl::StrJoin(param_shape, ","), absl::StrJoin(block_grid, ",")));
  }
  const size_t offset = block_grid.size() - param_shape.size();
  for (size_t axis = 0; axis < param_shape.size(); ++axis) {
    const int64_t extent = param_shape[axis];
    // Unknown (dynamic) extents cannot be checked.
    if (extent <= 0) continue;
    if (extent != 1 && extent != block_grid[offset + axis]) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "shape [%s] does not match the expected block grid [%s]",
          absl::StrJoin(param_shape, ","), absl::StrJoin(block_grid, ",")));
    }
  }
  return absl::OkStatus();
}

std::optional<BroadcastPlan> BroadcastPlan::Create(
    const ParamShape& scales, const ParamShape& zero_points) {
  if (scales.element_count <= 0 || zero_points.element_count <= 0) {
    return std::nullopt;
  }
  const int64_t scale_static_count = GetStaticElementCount(scales.shape);
  const int64_t zero_point_static_count =
      GetStaticElementCount(zero_points.shape);
  auto is_scalar = [](const ParamShape& param, int64_t static_count) {
    return param.element_count == 1 &&
           (param.shape.empty() || static_count == 1);
  };

  if (is_scalar(zero_points, zero_point_static_count)) {
    return BroadcastPlan(Mode::kScalarZeroPoint, scales.element_count);
  }
  if (is_scalar(scales, scale_static_count)) {
    return BroadcastPlan(Mode::kScalarScale, zero_points.element_count);
  }

  if (scale_static_count == 0 || zero_point_static_count == 0) {
    // Without two static shapes, pair element-wise up to `padding_elements` of
    // trailing sub-byte padding.
    const int64_t count =
        std::min(scales.element_count, zero_points.element_count);
    const int64_t excess =
        std::max(scales.element_count, zero_points.element_count) - count;
    const int64_t allowed_padding =
        scales.element_count > zero_points.element_count
            ? scales.padding_elements
            : zero_points.padding_elements;
    if (excess > allowed_padding) return std::nullopt;
    return BroadcastPlan(Mode::kOneToOne, count);
  }

  if (scales.shape == zero_points.shape) {
    return BroadcastPlan(Mode::kOneToOne, scale_static_count);
  }

  // Right-align both shapes and broadcast extent-1 axes.
  const size_t rank = std::max(scales.shape.size(), zero_points.shape.size());
  const size_t scale_offset = rank - scales.shape.size();
  const size_t zero_point_offset = rank - zero_points.shape.size();
  BroadcastPlan plan(Mode::kBroadcast, /*size=*/1);
  plan.broadcast_dims_.reserve(rank);
  plan.scale_dims_.reserve(rank);
  plan.zero_point_dims_.reserve(rank);
  for (size_t axis = 0; axis < rank; ++axis) {
    const int64_t scale_dim =
        axis >= scale_offset ? scales.shape[axis - scale_offset] : 1;
    const int64_t zero_point_dim =
        axis >= zero_point_offset ? zero_points.shape[axis - zero_point_offset]
                                  : 1;
    if (scale_dim != zero_point_dim && scale_dim != 1 && zero_point_dim != 1) {
      return std::nullopt;
    }
    const int64_t broadcast_dim = std::max(scale_dim, zero_point_dim);
    if (plan.size_ > std::numeric_limits<int64_t>::max() / broadcast_dim) {
      return std::nullopt;
    }
    plan.size_ *= broadcast_dim;
    plan.broadcast_dims_.push_back(broadcast_dim);
    plan.scale_dims_.push_back(scale_dim);
    plan.zero_point_dims_.push_back(zero_point_dim);
  }
  return plan;
}

BroadcastPlan::IndexPair BroadcastPlan::Map(int64_t flat_index) const {
  switch (mode_) {
    case Mode::kOneToOne:
      return {.scale_index = flat_index, .zero_point_index = flat_index};
    case Mode::kScalarZeroPoint:
      return {.scale_index = flat_index, .zero_point_index = 0};
    case Mode::kScalarScale:
      return {.scale_index = 0, .zero_point_index = flat_index};
    case Mode::kBroadcast:
      break;
  }
  // Decompose `flat_index` into row-major coordinates, from the trailing axis,
  // and re-linearize them per operand, skipping broadcast (extent-1) axes.
  IndexPair pair;
  int64_t remainder = flat_index;
  int64_t scale_stride = 1;
  int64_t zero_point_stride = 1;
  for (int64_t axis = static_cast<int64_t>(broadcast_dims_.size()) - 1;
       axis >= 0; --axis) {
    const int64_t coordinate = remainder % broadcast_dims_[axis];
    remainder /= broadcast_dims_[axis];
    if (scale_dims_[axis] > 1) pair.scale_index += coordinate * scale_stride;
    if (zero_point_dims_[axis] > 1) {
      pair.zero_point_index += coordinate * zero_point_stride;
    }
    scale_stride *= scale_dims_[axis];
    zero_point_stride *= zero_point_dims_[axis];
  }
  return pair;
}

}  // namespace quantization_internal

namespace {

using ::model_explorer::adapter::quantization_internal::BroadcastPlan;
using ::model_explorer::adapter::quantization_internal::
    CheckParamShapeMatchesBlockGrid;
using ::model_explorer::adapter::quantization_internal::ComputeBlockGrid;
using ::model_explorer::adapter::quantization_internal::FormatAffineFormula;
using ::model_explorer::adapter::quantization_internal::GetStaticElementCount;
using ::model_explorer::adapter::quantization_internal::JoinFormulas;
using ::model_explorer::adapter::quantization_internal::ParamShape;
using ::model_explorer::adapter::quantization_internal::ReadScaleAt;
using ::model_explorer::adapter::quantization_internal::ReadZeroPointAt;
using ::model_explorer::adapter::quantization_internal::ScaleBitWidth;
using ::model_explorer::adapter::quantization_internal::ZeroPointBitWidth;

// Node metadata attribute keys emitted for quantized tensors.
constexpr absl::string_view kQuantization = "quantization";
constexpr absl::string_view kQuantizedDimension = "quantized_dimension";
constexpr absl::string_view kQuantizedDimensions = "quantized_dimensions";
constexpr absl::string_view kBlockSize = "block_size";
constexpr absl::string_view kBlockShape = "block_shape";
constexpr absl::string_view kScalesTensor = "scales_tensor";
constexpr absl::string_view kZeroPointsTensor = "zero_points_tensor";

constexpr absl::string_view kScalesRole = "scales";
constexpr absl::string_view kZeroPointsRole = "zero_points";

// The fields of `BlockwiseQuantization` or `MultiAxisQuantization` details.
struct ParamTensorRefs {
  bool is_blockwise = false;
  int32_t scales_index = -1;
  // -1 means all zero points are 0.
  int32_t zero_points_index = -1;
  int32_t block_size = 0;
  // Blockwise only.
  std::vector<int32_t> block_shape;
  // Multi-axis only.
  std::vector<int32_t> quantized_dimensions;
};

// Requires `quant.details` to hold blockwise or multi-axis quantization.
ParamTensorRefs GetParamTensorRefs(
    const tflite::QuantizationParametersT& quant) {
  if (const tflite::BlockwiseQuantizationT* blockwise =
          quant.details.AsBlockwiseQuantization()) {
    return {.is_blockwise = true,
            .scales_index = blockwise->scales,
            .zero_points_index = blockwise->zero_points,
            .block_size = blockwise->block_size,
            .block_shape = blockwise->block_shape};
  }
  const tflite::MultiAxisQuantizationT& multi_axis =
      *quant.details.AsMultiAxisQuantization();
  return {.scales_index = multi_axis.scales,
          .zero_points_index = multi_axis.zero_points,
          .block_size = multi_axis.block_size,
          .quantized_dimensions = multi_axis.quantized_dimensions};
}

// Returns `tensor`'s effective shape, preferring `shape_signature` when
// present (where dynamic dimensions are encoded as `-1` instead of placeholder
// `1`s in `shape`).
absl::Span<const int32_t> GetTensorShape(const tflite::TensorT& tensor) {
  return !tensor.shape_signature.empty()
             ? absl::MakeConstSpan(tensor.shape_signature)
             : absl::MakeConstSpan(tensor.shape);
}

// Returns `values` formatted as `[a, b, c]`.
std::string FormatIntList(absl::Span<const int32_t> values) {
  return absl::StrCat("[", absl::StrJoin(values, ", "), "]");
}

std::string DescribeTensor(int index, const tflite::TensorT& tensor) {
  return absl::StrFormat("tensor #%d (%s)", index,
                         StringifyTensorShape(tensor));
}

// Returns the maximum number of formulas to list for `total_count` formulas.
int64_t ClampToFormulaLimit(int64_t total_count, int64_t formula_limit) {
  return formula_limit < 0 ? total_count : std::min(total_count, formula_limit);
}

// The bytes and shape of one scale or zero-point tensor. Views into the model
// buffer and the `tflite::TensorT`, which must outlive it.
struct ParamTensor {
  absl::Span<const uint8_t> data;
  ParamShape shape;
};

// Resolves `tensor`'s bytes and the number of elements they hold. Returns an
// empty `ParamTensor` (`data.empty()`) when the data is not stored in the
// model, and an error when the tensor cannot be decoded.
absl::StatusOr<ParamTensor> ResolveParamTensor(
    const tflite::TensorT& tensor, absl::string_view role,
    absl::FunctionRef<int(tflite::TensorType)> bit_width_fn,
    TensorBufferResolver resolve_buffer) {
  absl::StatusOr<absl::Span<const uint8_t>> data = resolve_buffer(tensor);
  if (!data.ok()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("cannot read %s tensor '%s': %s", role, tensor.name,
                        data.status().message()));
  }
  const int bit_width = bit_width_fn(tensor.type);
  if (bit_width <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("unsupported type for %s tensor '%s' (%s)", role,
                        tensor.name, StringifyTensorShape(tensor)));
  }
  const absl::Span<const int32_t> effective_shape = GetTensorShape(tensor);
  for (const int32_t dim : effective_shape) {
    if (dim == 0) {
      return absl::InvalidArgumentError(
          absl::StrFormat("%s tensor '%s' (%s) has no elements", role,
                          tensor.name, StringifyTensorShape(tensor)));
    }
  }
  if (data->empty()) return ParamTensor{};

  const int64_t capacity = static_cast<int64_t>(data->size()) * 8 / bit_width;
  const int64_t static_count = GetStaticElementCount(effective_shape);
  // Scalar and dynamically shaped tensors need at least one element.
  const int64_t required = std::max<int64_t>(static_count, 1);
  if (capacity < required) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "truncated %s tensor '%s' (%s): buffer holds %d of %d elements", role,
        tensor.name, StringifyTensorShape(tensor), capacity, required));
  }

  ParamTensor param{.data = *data, .shape = {.shape = effective_shape}};
  if (static_count > 0) {
    param.shape.element_count = static_count;
  } else if (effective_shape.empty()) {
    param.shape.element_count = 1;  // Rank-0 scalar.
  } else {
    // Dynamic shape: trust the buffer size, which for sub-byte types may
    // include padding in the last byte.
    param.shape.element_count = capacity;
    param.shape.padding_elements = bit_width < 8 ? 8 / bit_width - 1 : 0;
  }
  return param;
}

// Formats the dequantization formulas of blockwise / multi-axis parameters.
// Every parameter value is validated, but only the first `formula_limit` are
// formatted. Returns an empty string when the parameter data is not stored in
// the model or `formula_limit` is 0.
absl::StatusOr<std::string> FormatParamTensorFormulas(
    const tflite::TensorT& scales_tensor,
    const tflite::TensorT* absl_nullable zero_points_tensor,
    TensorBufferResolver resolve_buffer, int64_t formula_limit) {
  absl::StatusOr<ParamTensor> scales = ResolveParamTensor(
      scales_tensor, kScalesRole, ScaleBitWidth, resolve_buffer);
  if (!scales.ok()) return scales.status();

  ParamTensor zero_points;
  if (zero_points_tensor != nullptr) {
    absl::StatusOr<ParamTensor> resolved =
        ResolveParamTensor(*zero_points_tensor, kZeroPointsRole,
                           ZeroPointBitWidth, resolve_buffer);
    if (!resolved.ok()) return resolved.status();
    zero_points = *resolved;
  }
  if (scales->data.empty() ||
      (zero_points_tensor != nullptr && zero_points.data.empty())) {
    return "";
  }

  // Without a zero-point tensor every zero point is 0; pair each scale with a
  // virtual scalar zero point.
  const ParamShape zero_point_shape =
      zero_points_tensor != nullptr
          ? zero_points.shape
          : ParamShape{.shape = {}, .element_count = 1};
  std::optional<BroadcastPlan> plan =
      BroadcastPlan::Create(scales->shape, zero_point_shape);
  if (!plan.has_value()) {
    // A scalar zero point always broadcasts, so a zero-point tensor exists.
    ABSL_DCHECK(zero_points_tensor != nullptr);
    return absl::InvalidArgumentError(
        absl::StrFormat("cannot broadcast scales (%s) with zero_points (%s)",
                        StringifyTensorShape(scales_tensor),
                        StringifyTensorShape(*zero_points_tensor)));
  }

  auto out_of_range = [](int64_t index, absl::string_view role,
                         const tflite::TensorT& param) {
    return absl::InvalidArgumentError(
        absl::StrFormat("out-of-range value at index %d of %s tensor '%s'",
                        index, role, param.name));
  };
  // Element counts were validated above, so reads only fail on values that
  // `int64_t` cannot represent: UINT64 zero points above `INT64_MAX`. Check all
  // of them, not just the ones formatted below.
  if (zero_points_tensor != nullptr &&
      zero_points_tensor->type == tflite::TensorType_UINT64) {
    for (int64_t i = 0; i < zero_points.shape.element_count; ++i) {
      if (!ReadZeroPointAt(zero_points_tensor->type, zero_points.data, i)) {
        return out_of_range(i, kZeroPointsRole, *zero_points_tensor);
      }
    }
  }

  const int64_t shown = ClampToFormulaLimit(plan->size(), formula_limit);
  if (shown == 0) return "";
  std::vector<std::string> formulas;
  formulas.reserve(static_cast<size_t>(shown));
  for (int64_t i = 0; i < shown; ++i) {
    const BroadcastPlan::IndexPair pair = plan->Map(i);
    std::optional<float> scale =
        ReadScaleAt(scales_tensor.type, scales->data, pair.scale_index);
    if (!scale.has_value()) {
      return out_of_range(pair.scale_index, kScalesRole, scales_tensor);
    }
    int64_t zero_point = 0;
    if (zero_points_tensor != nullptr) {
      std::optional<int64_t> value = ReadZeroPointAt(
          zero_points_tensor->type, zero_points.data, pair.zero_point_index);
      if (!value.has_value()) {
        return out_of_range(pair.zero_point_index, kZeroPointsRole,
                            *zero_points_tensor);
      }
      zero_point = *value;
    }
    formulas.push_back(FormatAffineFormula(*scale, zero_point));
  }
  return JoinFormulas(formulas, plan->size());
}

// Checks blockwise scale / zero-point shapes against the block grid implied by
// the tensor shape and block description. Unknown shapes are not checked.
absl::Status CheckBlockGrid(
    const tflite::TensorT& tensor, const ParamTensorRefs& refs,
    const tflite::TensorT& scales_tensor,
    const tflite::TensorT* absl_nullable zero_points_tensor) {
  const absl::Span<const int32_t> tensor_shape = GetTensorShape(tensor);
  if (refs.block_shape.empty()) {
    if (refs.block_size <= 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "block_size (%d) must be positive for tensor '%s' "
          "(%s)",
          refs.block_size, tensor.name, StringifyTensorShape(tensor)));
    }
  } else {
    if (!tensor_shape.empty() &&
        refs.block_shape.size() != tensor_shape.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "block_shape %s does not match the rank of tensor '%s' (%s)",
          FormatIntList(refs.block_shape), tensor.name,
          StringifyTensorShape(tensor)));
    }
    for (const int32_t extent : refs.block_shape) {
      if (extent <= 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "block_shape %s has non-positive extent for tensor '%s' (%s)",
            FormatIntList(refs.block_shape), tensor.name,
            StringifyTensorShape(tensor)));
      }
    }
  }
  std::optional<std::vector<int64_t>> grid =
      ComputeBlockGrid(tensor_shape, refs.block_shape, refs.block_size);
  if (!grid.has_value()) return absl::OkStatus();
  for (const auto& [role, param] :
       {std::pair(kScalesRole, &scales_tensor),
        std::pair(kZeroPointsRole, zero_points_tensor)}) {
    if (param == nullptr) continue;
    if (absl::Status status =
            CheckParamShapeMatchesBlockGrid(GetTensorShape(*param), *grid);
        !status.ok()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s tensor '%s' %s", role, param->name, status.message()));
    }
  }
  return absl::OkStatus();
}

QuantizationMetadata BuildAffineMetadata(
    const tflite::QuantizationParametersT& quant, int64_t formula_limit) {
  QuantizationMetadata metadata;
  if (quant.scale.empty() && quant.zero_point.empty()) return metadata;
  const std::string quantized_dimension =
      absl::StrCat(quant.quantized_dimension);

  if (quant.scale.size() != quant.zero_point.size()) {
    metadata.attrs.emplace_back(kQuantizedDimension, quantized_dimension);
    metadata.issue = absl::StrFormat("scale(%d) != zp(%d)", quant.scale.size(),
                                     quant.zero_point.size());
    return metadata;
  }

  const int64_t total = static_cast<int64_t>(quant.scale.size());
  const int64_t shown = ClampToFormulaLimit(total, formula_limit);
  if (shown > 0) {
    std::vector<std::string> formulas;
    formulas.reserve(static_cast<size_t>(shown));
    for (size_t i = 0; i < static_cast<size_t>(shown); ++i) {
      formulas.push_back(
          FormatAffineFormula(quant.scale[i], quant.zero_point[i]));
    }
    metadata.attrs.emplace_back(kQuantization, JoinFormulas(formulas, total));
  }
  metadata.attrs.emplace_back(kQuantizedDimension, quantized_dimension);
  return metadata;
}

QuantizationMetadata BuildParamTensorMetadata(
    const tflite::TensorT& tensor,
    absl::Span<const std::unique_ptr<tflite::TensorT>> tensors,
    TensorBufferResolver resolve_buffer, int64_t formula_limit) {
  const tflite::QuantizationParametersT& quant = *tensor.quantization;
  const ParamTensorRefs refs = GetParamTensorRefs(quant);

  std::vector<std::pair<std::string, std::string>> structure_attrs;
  if (refs.is_blockwise) {
    structure_attrs.emplace_back(kQuantizedDimension,
                                 absl::StrCat(quant.quantized_dimension));
  } else {
    structure_attrs.emplace_back(kQuantizedDimensions,
                                 FormatIntList(refs.quantized_dimensions));
  }
  if (!refs.block_shape.empty()) {
    structure_attrs.emplace_back(kBlockShape, FormatIntList(refs.block_shape));
  } else if (refs.block_size > 0) {
    structure_attrs.emplace_back(kBlockSize, absl::StrCat(refs.block_size));
  }

  QuantizationMetadata metadata;
  auto is_valid_index = [&](int32_t index) {
    return index >= 0 && static_cast<size_t>(index) < tensors.size() &&
           tensors[index] != nullptr;
  };
  if (!is_valid_index(refs.scales_index) ||
      (refs.zero_points_index != -1 &&
       !is_valid_index(refs.zero_points_index))) {
    metadata.attrs = std::move(structure_attrs);
    metadata.issue = absl::StrFormat(
        "invalid quantization tensor index (scales=%d, zero_points=%d, "
        "tensors_size=%d)",
        refs.scales_index, refs.zero_points_index, tensors.size());
    return metadata;
  }
  const tflite::TensorT& scales_tensor = *tensors[refs.scales_index];
  const tflite::TensorT* absl_nullable zero_points_tensor =
      refs.zero_points_index == -1 ? nullptr
                                   : tensors[refs.zero_points_index].get();

  // Parameter tensors are validated even when no formulas are shown.
  absl::StatusOr<std::string> formulas = FormatParamTensorFormulas(
      scales_tensor, zero_points_tensor, resolve_buffer, formula_limit);
  if (!formulas.ok()) {
    metadata.issue = std::string(formulas.status().message());
  } else if (!formulas->empty()) {
    metadata.attrs.emplace_back(kQuantization, *formulas);
  }
  metadata.attrs.insert(metadata.attrs.end(),
                        std::make_move_iterator(structure_attrs.begin()),
                        std::make_move_iterator(structure_attrs.end()));
  metadata.attrs.emplace_back(kScalesTensor,
                              DescribeTensor(refs.scales_index, scales_tensor));
  if (zero_points_tensor != nullptr) {
    metadata.attrs.emplace_back(
        kZeroPointsTensor,
        DescribeTensor(refs.zero_points_index, *zero_points_tensor));
  }

  // The schema only defines the parameter shapes of blockwise quantization.
  if (refs.is_blockwise && !metadata.issue.has_value()) {
    if (absl::Status status =
            CheckBlockGrid(tensor, refs, scales_tensor, zero_points_tensor);
        !status.ok()) {
      metadata.issue = std::string(status.message());
    }
  }
  return metadata;
}

}  // namespace

QuantizationMetadata BuildQuantizationMetadata(
    const tflite::TensorT& tensor,
    absl::Span<const std::unique_ptr<tflite::TensorT>> tensors,
    TensorBufferResolver resolve_buffer, int64_t formula_limit) {
  if (tensor.quantization == nullptr) return {};
  const tflite::QuantizationParametersT& quant = *tensor.quantization;
  if (quant.details.AsBlockwiseQuantization() != nullptr ||
      quant.details.AsMultiAxisQuantization() != nullptr) {
    return BuildParamTensorMetadata(tensor, tensors, resolve_buffer,
                                    formula_limit);
  }
  return BuildAffineMetadata(quant, formula_limit);
}

}  // namespace adapter
}  // namespace model_explorer
