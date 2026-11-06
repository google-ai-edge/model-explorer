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

"""Tests of the copy of a model that observes tensors, and of its check."""

from collections.abc import Callable
import hashlib
import pathlib

from absl.testing import parameterized
import flatbuffers
from model_debugger.server.tap import litertlm_file
from model_debugger.server.tap import tap_model
from model_debugger.server.tap import tap_scan
from model_debugger.server.tap import tflite_schema_generated as tflite
from model_debugger.server.testing import tap_fixtures
from model_debugger.server.testing import test_main

_POINTS = ['prefill_16:2:0', 'decode:0:0', 'decode:1:0']
# SHA-256 of the copy of the fixture model that observes _POINTS. When this
# module's code writes other bytes, raise tap_model.FORMAT_VERSION, then set
# this to the new copy's digest; when a schema pin or flatbuffers changed
# them instead, tap_model.identity() already names the copies anew.
_COPY_SHA256 = (
    'bae9309291bb025a55784c6bfb67f2d0097687d1dd21d365ebb05cfa3c851d23'
)


def _outputs(model: tflite.Model, signature: int) -> list[tuple[str, int]]:
  definition = model.SignatureDefs(signature)
  return [
      (
          definition.Outputs(i).Name().decode(),
          definition.Outputs(i).TensorIndex(),
      )
      for i in range(definition.OutputsLength())
  ]


def _sha256(path: pathlib.Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_model(path: pathlib.Path) -> tflite.ModelT:
  """The object tree of a file's text model, its bytes copied."""
  with litertlm_file.mapped(path) as model:
    data = bytes(model.bytes_of(model.prefill_decode()))
  return tflite.ModelT.InitFromObj(tflite.Model.GetRootAs(data, 0))


class TapModelTest(parameterized.TestCase):
  """Tests of the copy, written from the fixture model."""

  def setUp(self):
    """Writes the fixture model."""
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)
    self.source = self.tmp_path / 'model.litertlm'
    tap_fixtures.write_model(self.source)
    self.sha256 = _sha256(self.source)
    self.messages = []

  def _write(self, name: str, points: list[str]) -> tap_model.Copy:
    return tap_model.write(
        self.source,
        tap_scan.scan(self.source),
        points,
        self.tmp_path / name,
        self.sha256,
        self.messages.append,
    )

  def _changed(
      self, copy: pathlib.Path, change: Callable[[tflite.ModelT], None]
  ) -> pathlib.Path:
    """The copy with its text model's tree changed, the same size.

    Args:
        copy: A copy.
        change: Changes a value of the tree that keeps its size.

    Returns:
        Another copy, in which the change is all that differs.
    """
    with litertlm_file.mapped(copy) as written:
      section = written.prefill_decode()
      data = bytes(written.bytes_of(section))
    model = tflite.ModelT.InitFromObj(tflite.Model.GetRootAs(data, 0))
    model.buffers = [
        tap_model.AlignedBuffer.of(buffer) for buffer in model.buffers
    ]
    same = tap_model.pack(model)
    self.assertEqual(same, data[: len(same)])
    change(model)
    header = tap_model.pack(model)
    self.assertLen(header, len(same))
    changed = self.tmp_path / 'changed.litertlm'
    changed.unlink(missing_ok=True)
    with litertlm_file.mapped(copy) as written, open(changed, 'xb') as out:
      litertlm_file.write_copy(
          written,
          written.prefill_decode(),
          len(data),
          lambda stream: stream.write(header + data[len(header) :]),
          written.system['uuid'],
          _sha256(copy),
          out,
      )
    return changed

  def test_the_copy_outputs_the_points(self):
    """Each point is an output of its signature and of its subgraph."""
    copy = self._write('copy.litertlm', _POINTS)
    self.assertEqual(
        [(o.point.id, o.output_name, o.added) for o in copy.observed],
        [
            ('decode:0:0', 'tap_sg0_op0_out0', True),
            ('decode:1:0', 'activations', False),
            ('prefill_16:2:0', 'tap_sg2_op2_out0', True),
        ],
    )
    checked = dict(copy.checked)
    self.assertGreater(checked.pop('values'), 200)
    self.assertEqual(
        checked,
        {
            'sections': 3,
            'outside_bytes': (
                self.source.stat().st_size - len(tap_fixtures.text_model())
            ),
            'buffers': 4,
            'custom_options': 1,
            'points': 3,
        },
    )
    self.assertEqual(
        self.messages,
        [
            'Adding 3 outputs',
            'Writing the copy',
            'Checking the copy against the model',
            'Hashing the copy',
        ],
    )
    path = self.tmp_path / 'copy.litertlm'
    self.assertEqual(copy.sha256, hashlib.sha256(path.read_bytes()).hexdigest())
    with litertlm_file.mapped(path) as written:
      self.assertEqual(
          written.system['uuid'],
          tap_model.copy_uuid(self.sha256, _POINTS),
      )
      with written.bytes_of(written.prefill_decode()) as data:
        model = tflite.Model.GetRootAs(bytes(data), 0)
    self.assertEqual(
        _outputs(model, 0), [('activations', 3), ('tap_sg0_op0_out0', 2)]
    )
    self.assertEqual(_outputs(model, 1), [('activations', 3)])
    self.assertEqual(
        _outputs(model, 2), [('activations', 3), ('tap_sg2_op2_out0', 6)]
    )
    self.assertEqual(model.Subgraphs(0).OutputsAsNumpy().tolist(), [3, 2])
    self.assertEqual(copy.observed[0].capture_key(), 'post_tap_sg0_op0_out0')

  def test_names_each_points_key_in_a_capture(self):
    """An output added or one the signature has; read back from a key."""
    scan = tap_scan.scan(self.source)
    self.assertEqual(
        tap_model.capture_key(scan.point('prefill_16:2:0')),
        'post_tap_sg2_op2_out0',
    )
    self.assertEqual(
        tap_model.capture_key(scan.point('decode:1:0')), 'post_activations'
    )
    self.assertEqual(tap_model.added_point('post_tap_sg2_op2_out0'), (2, 2, 0))
    for key in ('post_activations', 'pre_tap_sg0_op1_out0', 'post_tap_x'):
      self.assertIsNone(tap_model.added_point(key))

  def test_the_copy_is_the_same_for_the_same_points(self):
    """Whatever the order chosen; other points make another copy."""
    first = self._write('first.litertlm', _POINTS)
    again = self._write('again.litertlm', list(reversed(_POINTS)))
    other = self._write('other.litertlm', ['decode:0:0'])
    self.assertEqual(first.sha256, again.sha256)
    self.assertNotEqual(first.sha256, other.sha256)
    self.assertNotEqual(
        tap_model.copy_uuid(self.sha256, _POINTS),
        tap_model.copy_uuid(self.sha256, ['decode:0:0']),
    )

  def test_the_copy_is_the_bytes_this_format_writes(self):
    """Other bytes for the same model and points need FORMAT_VERSION."""
    copy = self._write('copy.litertlm', _POINTS)
    self.assertEqual(
        copy.sha256,
        _COPY_SHA256,
        'The copy changed: raise tap_model.FORMAT_VERSION if the code did '
        'it, then set _COPY_SHA256.',
    )

  def test_the_identity_names_the_code_that_writes_copies(self):
    """The format, the flatbuffers runtime, and both schema bindings."""
    version, runtime, *bindings = tap_model.identity().split(':')
    self.assertEqual(version, str(tap_model.FORMAT_VERSION))
    self.assertEqual(runtime, flatbuffers.__version__)
    self.assertLen(bindings, 2)
    for digest in bindings:
      self.assertLen(digest, 64)

  def test_out_of_line_payloads_move_whatever_their_size(self):
    """An empty buffer and large custom options are moved, kept, checked."""
    self._write('copy.litertlm', _POINTS)
    source = _text_model(self.source)
    copied = _text_model(self.tmp_path / 'copy.litertlm')
    empty = copied.buffers[3]
    self.assertEqual(empty.size, 0)
    self.assertNotEqual(empty.offset, source.buffers[3].offset)
    op = copied.subgraphs[0].operators[3]
    self.assertNotEqual(
        op.largeCustomOptionsOffset,
        source.subgraphs[0].operators[3].largeCustomOptionsOffset,
    )
    with litertlm_file.mapped(self.tmp_path / 'copy.litertlm') as written:
      data = bytes(written.bytes_of(written.prefill_decode()))
    offset = op.largeCustomOptionsOffset
    self.assertEqual(
        data[offset : offset + op.largeCustomOptionsSize],
        tap_fixtures.CUSTOM_OPTIONS,
    )
    self.assertLessEqual(empty.offset, len(data))

  def test_a_model_that_changed_since_it_was_registered_is_refused(self):
    """The source is checked against its SHA-256 as it is read."""
    with self.assertRaisesRegex(ValueError, 'SHA-256 changed'):
      tap_model.write(
          self.source,
          tap_scan.scan(self.source),
          _POINTS,
          self.tmp_path / 'copy.litertlm',
          'f' * 64,
          self.messages.append,
      )
    self.assertFalse((self.tmp_path / 'copy.litertlm').exists())

  def test_bytes_between_sections_are_kept(self):
    """The source's bytes around its sections move with them."""
    tap_fixtures.write_model(self.source, between=b'kept, not zeros')
    self.sha256 = _sha256(self.source)
    copy = self._write('copy.litertlm', _POINTS)
    self.assertEqual(copy.checked['sections'], 3)
    written = (self.tmp_path / 'copy.litertlm').read_bytes()
    self.assertEqual(
        written.count(b'kept, not zeros'),
        self.source.read_bytes().count(b'kept, not zeros'),
    )
    self.assertEqual(written.count(b'kept, not zeros'), 3)

  def test_bytes_after_the_text_model_are_refused(self):
    """They would have nowhere to go: the text model grows over them."""
    tap_fixtures.write_model(self.source, after_text=b'no place')
    self.sha256 = _sha256(self.source)
    with self.assertRaisesRegex(ValueError, 'cannot place'):
      self._write('copy.litertlm', _POINTS)
    self.assertFalse((self.tmp_path / 'copy.litertlm').exists())

  @parameterized.named_parameters(
      (
          'quantization',
          lambda model: setattr(
              model.subgraphs[0].tensors[1].quantization, 'scale', [0.25]
          ),
          r'subgraphs\[0\]\.tensors\[1\]\.quantization\.scale',
      ),
      (
          'builtin_option',
          lambda model: setattr(
              model.subgraphs[1].operators[0].builtinOptions,
              'fusedActivationFunction',
              tflite.ActivationFunctionType.RELU6,
          ),
          r'subgraphs\[1\]\.operators\[0\]\.builtinOptions\.'
          'fusedActivationFunction',
      ),
      (
          'shape',
          lambda model: setattr(
              model.subgraphs[2].tensors[2], 'shape', [1, 1, 5]
          ),
          r'subgraphs\[2\]\.tensors\[2\]\.shape',
      ),
  )
  def test_the_check_finds_a_changed_value(self, change, where):
    """Any value of the text model's tree, not only its graph."""
    copy = self._write('copy.litertlm', _POINTS)
    changed = self._changed(self.tmp_path / 'copy.litertlm', change)
    with self.assertRaisesRegex(ValueError, 'changed the model.' + where):
      tap_model.check_copy(self.source, changed, copy.observed)
    unchanged = self._changed(self.tmp_path / 'copy.litertlm', id)
    tap_model.check_copy(self.source, unchanged, copy.observed)

  def test_an_observed_point_keeps_its_shape_in_the_model(self):
    """A point's shape is the model's, the KV length's axis included."""
    (observed,) = self._write('copy.litertlm', ['decode:2:0']).observed
    self.assertEqual(observed.point.shape, (1, 1, tap_fixtures.KV_LENGTH, 2))

  def test_the_check_finds_a_changed_byte(self):
    """A changed weight, inline or out of line, or another section."""
    copy = self._write('copy.litertlm', _POINTS)
    path = self.tmp_path / 'copy.litertlm'
    good = path.read_bytes()
    for found, what in (
        (
            good.index(tap_fixtures.WEIGHT),
            'Buffer 1 of the rewritten model',
        ),
        (
            good.index(tap_fixtures.BIAS) + 5,
            'Buffer 2 of the rewritten model',
        ),
        (
            good.index(tap_fixtures.CUSTOM_OPTIONS) + 3,
            'custom options of operator 3 of subgraph 0',
        ),
        (good.index(tap_fixtures.EMBEDDER), 'Section 2 differs'),
        (good.index(tap_fixtures.METADATA), 'Section 0 differs'),
    ):
      with self.subTest(what=what):
        changed = bytearray(good)
        changed[found] ^= 1
        path.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, what):
          tap_model.check_copy(self.source, path, copy.observed)
    path.write_bytes(good)
    tap_model.check_copy(self.source, path, copy.observed)

  def test_the_check_finds_an_output_missing(self):
    """A copy that lacks a point's output is not the copy of it."""
    copy = self._write('copy.litertlm', ['decode:0:0'])
    self._write('other.litertlm', ['prefill_16:2:0'])
    with self.assertRaisesRegex(ValueError, 'Subgraph 0 has other outputs'):
      tap_model.check_copy(
          self.source, self.tmp_path / 'other.litertlm', copy.observed
      )

  def test_refused_points_write_nothing(self):
    """None, too many, twice, unknown, or not observable."""
    many = [f'decode:{i}:0' for i in range(tap_model.MAX_POINTS + 1)]
    for points, why in (
        ([], f'Choose 1 to {tap_model.MAX_POINTS}'),
        (many, f'Choose 1 to {tap_model.MAX_POINTS}'),
        (['decode:0:0', 'decode:0:0'], 'chosen twice'),
        (['decode:7:0'], 'no tensor output decode:7:0'),
        (['decode:3:0'], 'decode:3:0 cannot be observed: Its type'),
    ):
      with self.subTest(points=points):
        with self.assertRaisesRegex(ValueError, why):
          self._write('refused.litertlm', points)
        self.assertFalse((self.tmp_path / 'refused.litertlm').exists())

  def test_check_points_against_a_scan(self):
    """check_points() is what write() refuses with."""
    scan = tap_scan.scan(self.source)
    tap_model.check_points(scan, ['verify:0:0'])
    with self.assertRaises(ValueError):
      tap_model.check_points(scan, ['verify:3:0'])

  def test_copies_and_checks_a_payload_free_model_with_float_options(self):
    """A model with inline-only buffers and scalar float options is checked."""
    model = tflite.ModelT.InitFromObj(
        tflite.Model.GetRootAs(tap_fixtures.layered_model(1), 0)
    )
    code = tflite.OperatorCodeT()
    code.builtinCode = tflite.BuiltinOperator.SOFTMAX
    code.deprecatedBuiltinCode = tflite.BuiltinOperator.SOFTMAX
    model.operatorCodes.append(code)
    op = model.subgraphs[0].operators[0]
    op.opcodeIndex = len(model.operatorCodes) - 1
    op.inputs = [0]
    op.builtinOptionsType = tflite.BuiltinOptions.SoftmaxOptions
    op.builtinOptions = tflite.SoftmaxOptionsT()
    op.builtinOptions.beta = 1.0
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    tap_fixtures.write_model(self.source, text=bytes(builder.Output()))
    self.sha256 = _sha256(self.source)
    copy = self._write('copy.litertlm', ['decode:0:0'])
    self.assertEqual(copy.checked['buffers'], 2)
    self.assertEqual(copy.checked['custom_options'], 0)
    changed = self._changed(
        self.tmp_path / 'copy.litertlm',
        lambda m: setattr(
            m.subgraphs[0].operators[0].builtinOptions, 'beta', 2.0
        ),
    )
    with self.assertRaisesRegex(
        ValueError, r'subgraphs\[0\]\.operators\[0\]\.builtinOptions\.beta'
    ):
      tap_model.check_copy(self.source, changed, copy.observed)


if __name__ == '__main__':
  test_main.main()
