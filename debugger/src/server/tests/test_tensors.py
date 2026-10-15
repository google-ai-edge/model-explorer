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

"""Tests of reading the runner's tensor files."""

import pathlib

from model_debugger_server import tensors
import numpy
import pytest
from safetensors import numpy as safetensors_numpy

_METADATA = {'signature': 'decode', 'step': '17'}


def _write(
    path: pathlib.Path, values: dict, metadata=_METADATA
) -> pathlib.Path:
  safetensors_numpy.save_file(values, str(path), metadata=metadata)
  return path


def test_read_header_names_the_one_tensor(tmp_path):
  """Read header names the one tensor."""
  path = _write(
      tmp_path / 'a.safetensors',
      {'post_logits': numpy.zeros((1, 1, 5), numpy.float32)},
  )

  header = tensors.read_header(path)

  assert header == tensors.TensorHeader(
      key='post_logits', dtype='F32', shape=(1, 1, 5), metadata=_METADATA
  )


def test_read_header_without_metadata(tmp_path):
  """Read header without metadata."""
  path = _write(
      tmp_path / 'a.safetensors', {'x': numpy.zeros(2, numpy.int8)}, None
  )

  assert tensors.read_header(path).metadata == {}


def test_read_header_refuses_two_tensors(tmp_path):
  """Read header refuses two tensors."""
  path = _write(
      tmp_path / 'a.safetensors',
      {'a': numpy.zeros(1, numpy.int8), 'b': numpy.zeros(1, numpy.int8)},
  )

  with pytest.raises(ValueError, match='holds 2 tensors'):
    tensors.read_header(path)


@pytest.mark.parametrize('data', [b'', b'1234', b'\x04' + b'\0' * 7 + b'nope'])
def test_read_header_refuses_other_files(tmp_path, data):
  """Read header refuses other files."""
  path = tmp_path / 'a.safetensors'
  path.write_bytes(data)

  with pytest.raises(ValueError):
    tensors.read_header(path)


@pytest.mark.parametrize(
    'value',
    [
        numpy.arange(-3, 3, dtype=numpy.int8).reshape(1, 1, 2, 3),
        numpy.array([[True, False, True]]),
        numpy.array([16], numpy.int32),
        numpy.linspace(-1, 1, 6, dtype=numpy.float32).reshape(1, 6),
    ],
)
def test_read_tensor_keeps_dtype_and_values(tmp_path, value):
  """Read tensor keeps dtype and values."""
  path = _write(tmp_path / 'a.safetensors', {'pre_x': value})

  result = tensors.read_tensor(path)

  assert result.dtype == value.dtype
  numpy.testing.assert_array_equal(result, value)


def test_read_slice_reads_part(tmp_path):
  """Read slice reads part."""
  value = numpy.arange(2 * 3 * 8, dtype=numpy.int8).reshape(1, 2, 8, 3)
  path = _write(tmp_path / 'a.safetensors', {'post_kv_cache_k_0': value})

  part = tensors.read_slice(path, (0, 1, slice(2, 5)))

  numpy.testing.assert_array_equal(part, value[0, 1, 2:5])
