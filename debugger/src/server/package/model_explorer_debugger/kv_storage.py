# Copyright 2026 The AI Edge Model Explorer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Explicit logical views of immutable KV capacity buffers."""

import math
from .runtime.litert_storage import require_storage

LAYOUT = ['batch', 'kv_head', 'sequence', 'head_dim']


def storage_view(record):
  """Return validated storage-axis slices and their shape.

  Padding is never inferred.
  """
  require_storage(record)
  shape, layout = record.get('shape'), record.get('layout')
  if (
      not isinstance(shape, list)
      or len(shape) != 4
      or any(type(size) is not int or size <= 0 for size in shape)
      or not isinstance(layout, list)
      or len(layout) != 4
      or any(not isinstance(axis, str) for axis in layout)
      or set(layout) != set(LAYOUT)
  ):
    raise ValueError('layout_unavailable')
  declaration = record.get('storage_view')
  if declaration is None:
    return tuple(slice(0, size) for size in shape), list(shape)
  if not isinstance(declaration, list) or len(declaration) != len(shape):
    raise ValueError('invalid_storage_view')
  slices, selected = [], []
  for axis, size, item in zip(layout, shape, declaration):
    if not isinstance(item, dict) or set(item) != {'start', 'stop', 'step'}:
      raise ValueError('invalid_storage_view')
    start, stop, step = item['start'], item['stop'], item['step']
    if (
        any(type(value) is not int for value in (start, stop, step))
        or step != 1
        or not 0 <= start <= stop <= size
        or (axis != 'sequence' and (start != 0 or stop != size))
    ):
      raise ValueError('invalid_storage_view')
    slices.append(slice(start, stop, step))
    selected.append(stop - start)
  start, end, valid = (
      record.get('logical_start'),
      record.get('logical_end'),
      record.get('valid_length'),
  )
  if (
      type(start) is not int
      or type(end) is not int
      or type(valid) is not int
      or not 0 <= start <= end
      or valid != end - start
      or selected[layout.index('sequence')] != valid
  ):
    raise ValueError('logical_range_mismatch')
  return tuple(slices), selected


def dequantization(record):
  """A scalar comparison transform must be explicitly requested and recorded."""
  quant = record.get('dequantization')
  if quant is None:
    return None
  if (
      not isinstance(quant, dict)
      or record.get('dtype') not in ('int8', 'uint8', 'int16')
      or type(quant.get('scale')) not in (int, float)
      or not math.isfinite(quant['scale'])
      or quant['scale'] <= 0
      or type(quant.get('zero_point')) is not int
  ):
    raise ValueError('invalid_dequantization')
  return quant


def comparison_view(value, record):
  slices, shape = storage_view(record)
  if list(value.shape) != record['shape']:
    raise ValueError('tensor_metadata_mismatch')
  value = value[slices]
  if list(value.shape) != shape:
    raise ValueError('tensor_metadata_mismatch')
  quant = dequantization(record)
  if quant:
    value = (value.astype('float64') - quant['zero_point']) * quant['scale']
  return value
