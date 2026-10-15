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
"""

import dataclasses
import json
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
  if not isinstance(metadata, dict):
    raise ValueError(f'{path.name} has invalid metadata.')
  if len(header) != 1:
    raise ValueError(f'{path.name} holds {len(header)} tensors, not one.')
  ((key, spec),) = header.items()
  if (
      not isinstance(spec, dict)
      or not isinstance(spec.get('dtype'), str)
      or not isinstance(spec.get('shape'), list)
  ):
    raise ValueError(f'{path.name} has an invalid tensor entry.')
  return TensorHeader(
      key=key,
      dtype=spec['dtype'],
      shape=tuple(spec['shape']),
      metadata=dict(metadata),
  )


def read_tensor(path: pathlib.Path) -> numpy.ndarray:
  """Returns the one tensor of a tensor file, in its own dtype."""
  try:
    with safetensors.safe_open(path, framework='numpy') as file:
      keys = list(file.keys())
      if len(keys) != 1:
        raise ValueError(f'{path.name} holds {len(keys)} tensors, not one.')
      return file.get_tensor(keys[0])
  except (safetensors.SafetensorError, TypeError, KeyError) as error:
    raise ValueError(f'{path.name} could not be read: {error}') from error


def read_slice(
    path: pathlib.Path, index: tuple[slice | int, ...]
) -> numpy.ndarray:
  """Returns part of the one tensor of a tensor file, reading only that part.

  Args:
    path: The tensor file.
    index: One slice or int per leading axis, as in numpy indexing.
  """
  try:
    with safetensors.safe_open(path, framework='numpy') as file:
      keys = list(file.keys())
      if len(keys) != 1:
        raise ValueError(f'{path.name} holds {len(keys)} tensors, not one.')
      return file.get_slice(keys[0])[index]
  except (safetensors.SafetensorError, TypeError, KeyError) as error:
    raise ValueError(f'{path.name} could not be read: {error}') from error
