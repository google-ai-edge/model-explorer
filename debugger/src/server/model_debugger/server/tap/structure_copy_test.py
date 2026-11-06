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

"""Tests of the structure copies a graph is converted from."""

from typing import Any

from absl.testing import absltest
from model_debugger.server.tap import structure_copy
from model_debugger.server.tap import tflite_schema_generated as tflite
from model_debugger.server.testing import tap_fixtures
from model_debugger.server.testing import test_main


def _model(data: bytes) -> tflite.ModelT:
  return tflite.ModelT.InitFromObj(tflite.Model.GetRootAs(data, 0))


def _bytes(buffer: tflite.BufferT) -> bytes:
  return b'' if buffer.data is None else bytes(buffer.data)


def _tensor(tensor: tflite.TensorT) -> tuple[Any, ...]:
  quantization = tensor.quantization
  return (
      tensor.name,
      list(tensor.shape),
      tensor.type,
      tensor.buffer,
      None if quantization is None else list(quantization.scale),
  )


def _operator(op: tflite.OperatorT) -> tuple[Any, ...]:
  return (
      op.opcodeIndex,
      list(op.inputs),
      list(op.outputs),
      op.builtinOptionsType,
  )


class StructureCopyTest(absltest.TestCase):
  """Tests of copies(), on the fixture's text model."""

  def setUp(self):
    """The fixture's text model, and its object tree."""
    super().setUp()
    self.data = tap_fixtures.text_model()
    self.source = _model(self.data)

  def _copy(self, index: int, **kwargs) -> tflite.ModelT:
    found = list(
        structure_copy.copies(memoryview(self.data), [index], **kwargs)
    )
    self.assertEqual([i for i, _ in found], [index])
    return _model(found[0][1])

  def test_a_copy_is_its_subgraph_whole_and_the_others_empty(self):
    """Its subgraph and signature, the others by name, every buffer."""
    for index, name in enumerate(tap_fixtures.SIGNATURES):
      copy = self._copy(index)
      self.assertLen(copy.subgraphs, len(self.source.subgraphs))
      for i, (mine, theirs) in enumerate(
          zip(copy.subgraphs, self.source.subgraphs)
      ):
        self.assertEqual(mine.name, theirs.name)
        if i != index:
          for part in ('tensors', 'operators', 'inputs', 'outputs'):
            found = getattr(mine, part)
            self.assertEqual(0 if found is None else len(found), 0)
          continue
        self.assertEqual(
            [_tensor(t) for t in mine.tensors],
            [_tensor(t) for t in theirs.tensors],
        )
        self.assertEqual(
            [_operator(op) for op in mine.operators],
            [_operator(op) for op in theirs.operators],
        )
        self.assertEqual(list(mine.inputs), list(theirs.inputs))
        self.assertEqual(list(mine.outputs), list(theirs.outputs))
      self.assertEqual(
          [d.signatureKey.decode() for d in copy.signatureDefs], [name]
      )
      self.assertLen(copy.operatorCodes, len(self.source.operatorCodes))
      # Every buffer by its index; the small ones' bytes inline.
      self.assertEqual(
          [_bytes(b) for b in copy.buffers],
          [b'', tap_fixtures.WEIGHT, tap_fixtures.BIAS, b''],
      )
      self.assertEqual({b.offset for b in copy.buffers}, {0})

  def test_a_buffer_over_the_limit_is_emptied(self):
    """The weights go; a buffer at the limit stays."""
    copy = self._copy(0, kept_bytes=len(tap_fixtures.WEIGHT))
    self.assertEqual(
        [_bytes(b) for b in copy.buffers],
        [b'', tap_fixtures.WEIGHT, b'', b''],
    )
    self.assertEqual([b.size for b in copy.buffers], [0, 0, 0, 0])
    # The tensor still names its buffer.
    self.assertEqual(copy.subgraphs[0].tensors[5].buffer, 2)

  def test_large_custom_options_move_inline(self):
    """The CAST of decode keeps its options, as custom options."""
    cast = self._copy(0).subgraphs[0].operators[3]
    self.assertEqual(bytes(cast.customOptions), tap_fixtures.CUSTOM_OPTIONS)
    self.assertEqual(
        (cast.largeCustomOptionsOffset, cast.largeCustomOptionsSize),
        (0, 0),
    )

  def test_each_subgraph_is_copied_once(self):
    """A subgraph named twice, by two signatures say, is one copy."""
    found = structure_copy.copies(memoryview(self.data), [2, 0, 2])
    self.assertEqual([i for i, _ in found], [2, 0])

  def test_refuses_what_it_cannot_copy(self):
    """Not a TFLite model; a subgraph it lacks."""
    with self.assertRaisesRegex(ValueError, 'not a TFLite model'):
      list(structure_copy.copies(memoryview(b'x' * 64), [0]))
    with self.assertRaisesRegex(ValueError, 'no subgraph 9'):
      list(structure_copy.copies(memoryview(self.data), [9]))

  def test_identity_names_the_format_and_buffer_limit(self):
    """identity() names the format version and kept-buffer byte limit."""
    self.assertEqual(
        structure_copy.identity(),
        f'structure copy {structure_copy.FORMAT_VERSION}, buffers over '
        f'{structure_copy.KEPT_BYTES} bytes emptied',
    )


if __name__ == '__main__':
  test_main.main()
