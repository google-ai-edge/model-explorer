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

"""Reads the runner's tensor files: safetensors holding one tensor each.

A file is an 8-byte little-endian header length, a JSON header naming each
tensor's dtype, shape and byte range plus a string-to-string `__metadata__`,
then the data. The runtime's dump writes exactly one tensor per file.

numpy has no bfloat16 nor 8-bit floats, which a PyTorch runner may write
(BF16, F8_E4M3, F8_E5M2): such a tensor is read as its raw bits and returned
as float32, which holds every value of theirs exactly.
"""

import dataclasses
import functools
import json
import os
import pathlib
import struct

import numpy
import safetensors

# The safetensors format refuses headers above 100 MB; so do we, before
# reading one.
_HEADER_LIMIT = 100 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class TensorHeader:
  """What a tensor file's header says about its one tensor.

  Attributes:
      key: The tensor's name in the file, e.g. `post_logits`.
      dtype: The safetensors dtype name, e.g. `F32`, `I8`, `BOOL`.
      shape: The tensor's shape.
      metadata: The file's `__metadata__`, empty when it has none.
  """

  key: str
  dtype: str
  shape: tuple[int, ...]
  metadata: dict[str, str]


def read_header(path: pathlib.Path) -> TensorHeader:
  """Returns the header of a tensor file.

  Args:
      path: A safetensors file holding exactly one tensor.

  Raises:
      ValueError: The file is not safetensors or holds other than one tensor.
  """
  with open(path, 'rb') as file:
    prefix = file.read(8)
    if len(prefix) != 8:
      raise ValueError(f'{path.name} is not a safetensors file.')
    (length,) = struct.unpack('<Q', prefix)
    if length > _HEADER_LIMIT:
      raise ValueError(f'{path.name} has a header of {length} bytes.')
    try:
      header = json.loads(file.read(length))
    except (UnicodeDecodeError, json.JSONDecodeError):
      raise ValueError(f'{path.name} has no JSON header.') from None
  if not isinstance(header, dict):
    raise ValueError(f'{path.name} has no JSON header.')
  metadata = header.pop('__metadata__', None) or {}
  if len(header) != 1:
    raise ValueError(f'{path.name} holds {len(header)} tensors, not one.')
  ((key, spec),) = header.items()
  return TensorHeader(
      key=key,
      dtype=spec['dtype'],
      shape=tuple(spec['shape']),
      metadata=dict(metadata),
  )


def read_tensor(path: pathlib.Path) -> numpy.ndarray:
  """Returns the one tensor of a tensor file, in its own dtype.

  A dtype numpy lacks (_WIDENED) is returned as float32.

  Args:
      path: The tensor file.
  """
  widened = _widened(path)
  if widened is not None:
    return widened(())
  with safetensors.safe_open(path, framework='numpy') as file:
    (key,) = file.keys()
    return file.get_tensor(key)


def read_slice(
    path: pathlib.Path, index: tuple[slice | int, ...]
) -> numpy.ndarray:
  """Returns part of the one tensor of a tensor file, reading only that part.

  Args:
      path: The tensor file.
      index: One slice or int per leading axis, as in numpy indexing.
  """
  widened = _widened(path)
  if widened is not None:
    return widened(index)
  with safetensors.safe_open(path, framework='numpy') as file:
    (key,) = file.keys()
    return file.get_slice(key)[index]


def widened_dtype(path: pathlib.Path) -> str | None:
  """The dtype a tensor file holds, by torch's name, if numpy lacks it.

  What read_tensor() returns is float32 then; None for a dtype numpy has.

  Args:
      path: The tensor file.

  Returns:
      The torch name of the file's dtype, or None for a dtype numpy has.
  """
  return _TORCH_NAMES.get(read_header(path).dtype)


def _bf16(bits: numpy.ndarray) -> numpy.ndarray:
  """bfloat16 bits as float32: the top half of a float32's bits, exactly."""
  return (bits.astype(numpy.uint32) << 16).view(numpy.float32)


def _f8_table(exponent_bits: int, finite_only: bool) -> numpy.ndarray:
  """The float32 value of each of the 256 bit patterns of an 8-bit float.

  Args:
      exponent_bits: 4 for E4M3, 5 for E5M2.
      finite_only: E4M3's 'fn' encoding: no infinity, NaN only at all ones.

  Returns:
      The 256 float32 values, indexed by bit pattern.
  """
  mantissa_bits = 7 - exponent_bits
  bias = 2 ** (exponent_bits - 1) - 1
  top = 2**exponent_bits - 1
  values = numpy.empty(256, dtype=numpy.float32)
  for bits in range(256):
    sign = -1.0 if bits & 0x80 else 1.0
    exponent = (bits >> mantissa_bits) & top
    mantissa = bits & (2**mantissa_bits - 1)
    if finite_only and exponent == top and mantissa == 2**mantissa_bits - 1:
      value = float('nan')
    elif not finite_only and exponent == top:
      value = float('inf') if mantissa == 0 else float('nan')
    elif exponent == 0:
      value = mantissa / 2**mantissa_bits * 2.0 ** (1 - bias)
    else:
      value = (1 + mantissa / 2**mantissa_bits) * 2.0 ** (exponent - bias)
    values[bits] = sign * value
  return values


_F8_E4M3 = _f8_table(4, True)
_F8_E5M2 = _f8_table(5, False)
# The dtypes numpy lacks, by torch's names.
_TORCH_NAMES = {
    'BF16': 'bfloat16',
    'F8_E4M3': 'float8_e4m3fn',
    'F8_E5M2': 'float8_e5m2',
}
# The dtypes numpy lacks: their bits' numpy dtype, and how they become
# float32.
_WIDENED = {
    'BF16': (numpy.uint16, _bf16),
    'F8_E4M3': (numpy.uint8, lambda bits: _F8_E4M3[bits]),
    'F8_E5M2': (numpy.uint8, lambda bits: _F8_E5M2[bits]),
}


def _widened(path: pathlib.Path):
  """A reader of a tensor of a dtype numpy lacks, or None for another.

  The reader takes an index, as read_slice() does, reads only what it
  selects, and returns it as float32.

  Args:
      path: The tensor file.

  Returns:
      The reader, or None for a dtype numpy has.

  Raises:
      ValueError: As read_header(), or the data is not where the header
        says.
  """
  stat = os.stat(path)
  return _widened_file(str(path), stat.st_size, stat.st_mtime_ns)


@functools.lru_cache(maxsize=4096)
def _widened_file(path: str, size: int, mtime_ns: int):
  """_widened() of the file at `path` of that size and modification time."""
  del size, mtime_ns  # The cache's key: a file rewritten is read anew.
  path = pathlib.Path(path)
  with open(path, 'rb') as file:
    prefix = file.read(8)
    if len(prefix) != 8:
      raise ValueError(f'{path.name} is not a safetensors file.')
    (length,) = struct.unpack('<Q', prefix)
    if length > _HEADER_LIMIT:
      raise ValueError(f'{path.name} has a header of {length} bytes.')
    try:
      header = json.loads(file.read(length))
    except (UnicodeDecodeError, json.JSONDecodeError):
      raise ValueError(f'{path.name} has no JSON header.') from None
  header.pop('__metadata__', None)
  if len(header) != 1:
    raise ValueError(f'{path.name} holds {len(header)} tensors, not one.')
  ((_, spec),) = header.items()
  if spec['dtype'] not in _WIDENED:
    return None
  bits_dtype, widen = _WIDENED[spec['dtype']]
  shape = tuple(spec['shape'])
  start, end = spec['data_offsets']
  count = int(numpy.prod(shape, dtype=numpy.int64))
  if end - start != count * numpy.dtype(bits_dtype).itemsize:
    raise ValueError(
        f'{path.name} holds {end - start} bytes of a {spec["dtype"]} '
        f'tensor of shape {list(shape)}.'
    )

  def read(index: tuple[slice | int, ...]) -> numpy.ndarray:
    if count == 0:
      return numpy.zeros(shape, dtype=numpy.float32)[index]
    bits = numpy.memmap(
        path,
        dtype=bits_dtype,
        mode='r',
        offset=8 + length + start,
        shape=shape,
    )
    return widen(numpy.array(bits[index]))

  return read
