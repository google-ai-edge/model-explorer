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

"""Tests of the scan of what a model offers to observe."""

import hashlib
import json
import os
import pathlib

from absl.testing import absltest
import flatbuffers
from model_debugger.server.tap import tap_scan
from model_debugger.server.tap import tflite_schema_generated as tflite
from model_debugger.server.testing import tap_fixtures
from model_debugger.server.testing import test_main


class TapScanTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.path = pathlib.Path(self.create_tempdir().full_path) / 'model.litertlm'
    tap_fixtures.write_model(self.path)
    self.scan = tap_scan.scan(
        self.path, {name: 'layer' for name in tap_fixtures.SIGNATURES}
    )

  def test_lists_every_operator_output_of_every_signature(self):
    """Four outputs per signature, in the model's order."""
    self.assertLen(self.scan.points, 4 * len(tap_fixtures.SIGNATURES))
    self.assertEqual(
        [point.id for point in self.scan.points[:4]],
        ['decode:0:0', 'decode:1:0', 'decode:2:0', 'decode:3:0'],
    )
    point = self.scan.point('prefill_16:0:0')
    self.assertEqual(
        (
            point.signature,
            point.subgraph,
            point.tensor,
            point.tensor_name,
            point.shape,
            point.dtype,
            point.operator,
        ),
        (
            'prefill_16',
            2,
            2,
            'prefill_16/layer_0/sum',
            (1, 1, 4),
            'float32',
            'ADD',
        ),
    )
    self.assertTrue(point.selectable)
    self.assertIsNone(point.output_name)
    self.assertEqual(
        self.scan.section_sha256,
        hashlib.sha256(tap_fixtures.text_model()).hexdigest(),
    )
    self.assertEqual(self.scan.recommended, ())
    with self.assertRaisesRegex(ValueError, 'no tensor output decode:9:0'):
      self.scan.point('decode:9:0')

  def test_a_scan_kept_as_json_reads_back_the_same(self):
    """to_json() and from_json() keep every fact, through JSON text."""
    kept = json.loads(json.dumps(tap_scan.to_json(self.scan)))
    self.assertEqual(tap_scan.from_json(kept), self.scan)
    with self.assertRaisesRegex(ValueError, 'not a kept scan'):
      tap_scan.from_json({'points': []})

  def test_a_scan_kept_with_layer_stems_still_reads(self):
    """A Session exported before kept each scan's `layer_stems`."""
    kept = json.loads(json.dumps(tap_scan.to_json(self.scan)))
    kept['layer_stems'] = {'decode': 'layer'}
    self.assertEqual(tap_scan.from_json(kept), self.scan)

  def test_marks_when_each_signature_runs(self):
    """The decode and smallest prefill always run; the others may not."""
    self.assertEqual(
        [(s.name, s.runs) for s in self.scan.signatures],
        [
            ('decode', tap_scan.RUNS_ALWAYS),
            ('prefill_4', tap_scan.RUNS_ALWAYS),
            ('prefill_16', tap_scan.RUNS_LONG_PROMPTS),
            ('verify', tap_scan.RUNS_SPECULATIVE),
        ],
    )

  def test_marks_the_kv_length_and_existing_outputs(self):
    """The K cache's sequence axis, and an output the signature has."""
    self.assertEqual(self.scan.kv_length, tap_fixtures.KV_LENGTH)
    self.assertEqual(self.scan.point('decode:2:0').context_axes, (2,))
    self.assertEqual(self.scan.point('decode:0:0').context_axes, ())
    self.assertEqual(self.scan.point('decode:1:0').output_name, 'activations')

  def test_says_why_a_point_cannot_be_observed(self):
    """A string tensor is not captured."""
    point = self.scan.point('decode:3:0')
    self.assertFalse(point.selectable)
    self.assertEqual(point.reason, 'Its type, string, is not captured.')

  def test_reads_each_tensors_layer_and_the_layer_outputs(self):
    """The layer of a name's scope; the last tensor the next layer reads."""
    _, points, _ = tap_scan.read_model(
        tap_fixtures.layered_model(3), {'decode': 'layer'}
    )
    found = {p.id: (p.layer, p.layer_output) for p in points}
    self.assertEqual(
        found,
        {
            'decode:0:0': (0, False),
            'decode:1:0': (0, False),
            'decode:2:0': (0, True),
            'decode:3:0': (1, False),
            'decode:4:0': (1, False),
            'decode:5:0': (1, True),
            'decode:6:0': (2, False),
            'decode:7:0': (2, False),
            # The last layer: no next one reads it; named as the others.
            'decode:8:0': (2, True),
            'decode:9:0': (None, False),
        },
    )
    self.assertEqual(self.scan.point('decode:0:0').layer, 0)
    self.assertFalse(self.scan.point('decode:1:0').layer_output)

  def test_a_tensor_named_by_no_layer_is_in_its_operators_group(self):
    """Layer 0's `StatefulPartitionedCall:1`: in the layer its op is."""
    model = tap_fixtures.layered_model(3, unnamed=True)
    stems = {'decode': 'layer'}
    _, alone, _ = tap_scan.read_model(model, stems)
    _, grouped, _ = tap_scan.read_model(model, stems, {'decode': {1: 0}})
    alone = {p.id: p for p in alone}
    grouped = {p.id: p for p in grouped}
    cache = grouped.pop('decode:1:1')
    self.assertEqual(cache.tensor_name, 'StatefulPartitionedCall:1')
    self.assertEqual(cache.output_name, 'kv_cache_k_0')
    self.assertEqual((cache.layer, cache.layer_stem), (0, 'layer'))
    self.assertFalse(cache.layer_output)
    loose = alone.pop('decode:1:1')
    self.assertEqual((loose.layer, loose.layer_stem), (None, ''))
    # Every other point, named by its layer, is where its name says.
    self.assertEqual(
        {i: (p.layer, p.layer_output) for i, p in grouped.items()},
        {i: (p.layer, p.layer_output) for i, p in alone.items()},
    )
    _, other, _ = tap_scan.read_model(model, stems, {'decode': {1: 2}})
    other = {p.id: p.layer for p in other}
    self.assertEqual((other['decode:1:0'], other['decode:1:1']), (0, 2))
    # Without a stem no point is in a layer, whatever the groups.
    _, none, _ = tap_scan.read_model(model, {}, {'decode': {1: 0}})
    self.assertTrue(all(p.layer is None for p in none))

  def test_without_stems_no_point_is_in_a_layer(self):
    """The stems come from the model's graph; a scan without has none."""
    plain = tap_scan.scan(self.path)
    self.assertTrue(all(p.layer is None for p in plain.points))
    self.assertFalse(any(p.layer_output for p in plain.points))
    _, points, _ = tap_scan.read_model(
        tap_fixtures.layered_model(3), {'prefill_4': 'layer'}
    )
    self.assertTrue(all(p.layer is None for p in points))

  def test_names_a_layer_by_the_first_scope_of_a_fused_name(self):
    """`a/layer_1/x;b/layer_0/y` is of layer 1; `layer_10.x` of none."""
    self.assertEqual(tap_scan.layer_of('a/layer_1/x;b/layer_0/y', 'layer'), 1)
    self.assertEqual(
        tap_scan.layer_of('t/layer_17/layer_17.post/mul', 'layer'), 17
    )
    self.assertIsNone(tap_scan.layer_of('embedder/layer_norm', 'layer'))
    self.assertIsNone(tap_scan.layer_of('a/b', 'layer'))
    # Any stem the graph names its layers by, such as Qwen3's.
    stem = 'm.Qwen3DecoderLayer'
    self.assertEqual(tap_scan.layer_of('a/m.Qwen3DecoderLayer_3', stem), 3)
    self.assertEqual(
        tap_scan.layer_of('a/m.Qwen3DecoderLayer_27/b;c', stem), 27
    )
    self.assertIsNone(tap_scan.layer_of('a/m.Qwen3DecoderLayer_3x', stem))

  def test_marks_shared_subgraphs_and_composite_operators(self):
    """Shared subgraphs are unobservable; StableHLO composites name the op."""
    model = tflite.ModelT.InitFromObj(
        tflite.Model.GetRootAs(tap_fixtures.layered_model(1), 0)
    )
    code = tflite.OperatorCodeT()
    code.builtinCode = tflite.BuiltinOperator.STABLEHLO_COMPOSITE
    code.deprecatedBuiltinCode = 127
    model.operatorCodes.append(code)
    op = tflite.OperatorT()
    op.opcodeIndex = len(model.operatorCodes) - 1
    op.inputs = [0]
    op.outputs = [len(model.subgraphs[0].tensors)]
    op.builtinOptions2Type = tflite.BuiltinOptions2.StableHLOCompositeOptions
    op.builtinOptions2 = tflite.StableHLOCompositeOptionsT()
    op.builtinOptions2.name = 'odml.rms_norm'
    norm = tflite.TensorT()
    norm.name = 'rms'
    norm.shape = [1, 1, 4]
    norm.type = tflite.TensorType.FLOAT32
    norm.buffer = 0
    model.subgraphs[0].tensors.append(norm)
    model.subgraphs[0].operators.append(op)
    alias = tflite.SignatureDefT()
    alias.signatureKey = 'prefill_4'
    alias.subgraphIndex = 0
    alias.inputs = list(model.signatureDefs[0].inputs)
    alias.outputs = list(model.signatureDefs[0].outputs)
    model.signatureDefs.append(alias)
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    _, points, _ = tap_scan.read_model(bytes(builder.Output()))
    by_id = {point.id: point for point in points}
    composite = by_id['decode:4:0']
    self.assertEqual(composite.operator, 'STABLEHLO_COMPOSITE odml.rms_norm')
    self.assertFalse(composite.selectable)
    self.assertEqual(
        composite.reason,
        'Its graph is shared by several signatures, which '
        'this server does not observe.',
    )

  def test_the_real_model(self):
    """Gemma 4 E2B, if RUNNER_MODEL names it: as experiment X1 found."""
    model = os.environ.get('RUNNER_MODEL')
    if not model:
      self.skipTest('RUNNER_MODEL names no model.')
    # Its graph names its layers `layer_<N>` (graph/model_graph).
    scan = tap_scan.scan(
        pathlib.Path(model),
        {
            name: 'layer'
            for name in ('decode', 'prefill_128', 'prefill_1024', 'verify')
        },
    )
    if scan.section_sha256 != tap_scan.REVIEWED_SECTION_SHA256:
      self.skipTest('RUNNER_MODEL is not the reviewed Gemma 4 E2B.')
    self.assertLen(scan.points, 6679)
    self.assertEqual(scan.kv_length, 32003)
    self.assertLen([p for p in scan.points if p.context_axes], 474)
    self.assertEqual(scan.recommended, tap_scan.REVIEWED_POINTS)
    self.assertEqual(
        scan.point('decode:10:0').operator,
        'STABLEHLO_COMPOSITE odml.rms_norm',
    )
    self.assertLen([p for p in scan.points if p.layer is not None], 6214)
    outputs = {
        signature: [
            p.id
            for p in scan.points
            if p.layer_output and p.signature == signature
        ]
        for signature in ('decode', 'prefill_128')
    }
    # Prefill holds layers 0-13 and layer 14's K and V only.
    self.assertLen(outputs['decode'], 35)
    self.assertLen(outputs['prefill_128'], 14)
    for point_id, layer in (
        ('decode:100:0', 0),
        ('decode:1212:0', 17),
        ('decode:2062:0', 34),
        ('prefill_128:111:0', 0),
    ):
      point = scan.point(point_id)
      self.assertEqual((point.layer, point.layer_output), (layer, True))
    self.assertFalse(scan.point('decode:10:0').layer_output)


if __name__ == '__main__':
  test_main.main()
