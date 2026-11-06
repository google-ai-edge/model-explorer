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

from collections.abc import Mapping
import json
import pathlib
import types

from absl.testing import parameterized
from model_debugger.server.capture import capture_index
from model_debugger.server.capture import tensor_file
from model_debugger.server.testing import capture_fixtures
from model_debugger.server.testing import test_main
import numpy
from safetensors.numpy import save_file

_METADATA = {'signature': 'decode', 'step': '17'}


def _write(
    path: pathlib.Path,
    values: dict[str, numpy.ndarray],
    metadata: Mapping[str, str] | None = types.MappingProxyType(_METADATA),
) -> pathlib.Path:
  save_file(
      values, str(path), metadata=None if metadata is None else dict(metadata)
  )
  return path


class TensorFileTest(parameterized.TestCase):
  """Tests of reading the runner's tensor files."""

  def setUp(self):
    """Gives each test an empty directory of its own."""
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)

  def test_read_header_names_the_one_tensor(self):
    """Read header names the one tensor."""
    path = _write(
        self.tmp_path / 'a.safetensors',
        {'post_logits': numpy.zeros((1, 1, 5), numpy.float32)},
    )

    header = tensor_file.read_header(path)

    self.assertEqual(
        header,
        tensor_file.TensorHeader(
            key='post_logits',
            dtype='F32',
            shape=(1, 1, 5),
            metadata=_METADATA,
        ),
    )

  def test_read_header_without_metadata(self):
    """Read header without metadata."""
    path = _write(
        self.tmp_path / 'a.safetensors',
        {'x': numpy.zeros(2, numpy.int8)},
        None,
    )

    self.assertEqual(tensor_file.read_header(path).metadata, {})

  def test_read_header_refuses_two_tensors(self):
    """Read header refuses two tensors."""
    path = _write(
        self.tmp_path / 'a.safetensors',
        {'a': numpy.zeros(1, numpy.int8), 'b': numpy.zeros(1, numpy.int8)},
    )

    with self.assertRaisesRegex(ValueError, 'holds 2 tensors'):
      tensor_file.read_header(path)

  @parameterized.parameters(
      (b'',),
      (b'1234',),
      (b'\x04' + b'\0' * 7 + b'nope',),
  )
  def test_read_header_refuses_other_files(self, data):
    """Read header refuses other files."""
    path = self.tmp_path / 'a.safetensors'
    path.write_bytes(data)

    with self.assertRaises(ValueError):
      tensor_file.read_header(path)

  @parameterized.parameters(
      (numpy.arange(-3, 3, dtype=numpy.int8).reshape(1, 1, 2, 3),),
      (numpy.array([[True, False, True]]),),
      (numpy.array([16], numpy.int32),),
      (numpy.linspace(-1, 1, 6, dtype=numpy.float32).reshape(1, 6),),
  )
  def test_read_tensor_keeps_dtype_and_values(self, value):
    """Read tensor keeps dtype and values."""
    path = _write(self.tmp_path / 'a.safetensors', {'pre_x': value})

    result = tensor_file.read_tensor(path)

    self.assertEqual(result.dtype, value.dtype)
    numpy.testing.assert_array_equal(result, value)

  def test_read_slice_reads_part(self):
    """Read slice reads part."""
    value = numpy.arange(2 * 3 * 8, dtype=numpy.int8).reshape(1, 2, 8, 3)
    path = _write(self.tmp_path / 'a.safetensors', {'post_kv_cache_k_0': value})

    part = tensor_file.read_slice(path, (0, 1, slice(2, 5)))

    numpy.testing.assert_array_equal(part, value[0, 1, 2:5])

  def test_reads_bf16_as_float32_exactly(self):
    """BF16, which numpy lacks, reads as the float32 of its bits."""
    value = numpy.array(
        [[1.0, -2.5, 3.140625], [0.0, -0.0, 65280.0]], numpy.float32
    )
    path = self.tmp_path / 'bf16.safetensors'
    path.write_bytes(
        capture_fixtures.save_bf16('post_logits', value, _METADATA)
    )

    self.assertEqual(tensor_file.read_header(path).dtype, 'BF16')
    self.assertEqual(tensor_file.widened_dtype(path), 'bfloat16')
    whole = tensor_file.read_tensor(path)
    part = tensor_file.read_slice(path, (1, slice(1, 3)))

    self.assertEqual(whole.dtype, numpy.float32)
    numpy.testing.assert_array_equal(whole, value)
    numpy.testing.assert_array_equal(part, value[1, 1:3])

  def test_bf16_keeps_the_top_half_of_float32_bits(self):
    """A value BF16 cannot hold reads as its truncation to BF16."""
    value = numpy.array([1.0 + 2**-10, numpy.inf, numpy.nan], numpy.float32)
    path = self.tmp_path / 'cut.safetensors'
    path.write_bytes(capture_fixtures.save_bf16('x', value, {}))

    read = tensor_file.read_tensor(path)

    self.assertEqual(read[0], 1.0)
    self.assertEqual(read[1], numpy.inf)
    self.assertTrue(numpy.isnan(read[2]))

  @parameterized.parameters(
      ('F8_E4M3', [0x00, 0x38, 0xB8, 0x7E, 0x01], [0, 1, -1, 448, 2**-9]),
      ('F8_E5M2', [0x00, 0x3C, 0xBC, 0x7B, 0x7C], [0, 1, -1, 57344, 'inf']),
  )
  def test_reads_8_bit_floats(self, dtype, bits, expected):
    """The 8-bit floats torch writes read as float32."""
    data = bytes(bits)
    header = json.dumps(
        {'x': {'dtype': dtype, 'shape': [5], 'data_offsets': [0, 5]}}
    ).encode()
    path = self.tmp_path / 'f8.safetensors'
    path.write_bytes(len(header).to_bytes(8, 'little') + header + data)

    read = tensor_file.read_tensor(path)

    numpy.testing.assert_array_equal(
        read, numpy.array([float(e) for e in expected], numpy.float32)
    )

  def test_refuses_bf16_data_of_another_size(self):
    """A BF16 tensor whose bytes are not its shape's is refused."""
    header = json.dumps(
        {'x': {'dtype': 'BF16', 'shape': [4], 'data_offsets': [0, 6]}}
    ).encode()
    path = self.tmp_path / 'short.safetensors'
    path.write_bytes(len(header).to_bytes(8, 'little') + header + bytes(6))

    with self.assertRaisesRegex(ValueError, '6 bytes'):
      tensor_file.read_tensor(path)

  def test_tiny_chat_turn_and_tied_greedy_choice(self):
    """TinyChat writes readable tensor files; completion_dict and tie work."""
    chat = capture_fixtures.TinyChat(system='be brief')
    self.assertEqual(chat.load_response().start_token, capture_fixtures.BOS)
    turn = chat.send('hi', ['there'])
    stored = capture_index.completion_dict(turn.completion)
    self.assertEqual(stored['finish_reason'], 'stop')
    self.assertNotEmpty(stored['fed_tokens'])
    name, data = next(iter(turn.files.items()))
    path = self.tmp_path / name
    path.write_bytes(data)
    self.assertNotEmpty(tensor_file.read_header(path).shape)
    pick, top, why = capture_index.greedy_choice(
        numpy.array([1.0, 3.0, 3.0], numpy.float32)
    )
    self.assertIsNone(pick)
    self.assertEqual(top, [1, 2])
    self.assertIn('2 logits tied at the top', why or '')


if __name__ == '__main__':
  test_main.main()
