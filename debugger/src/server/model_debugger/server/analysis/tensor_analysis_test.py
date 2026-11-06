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

"""Tests of the intermediate tensors' comparison."""

import collections
import dataclasses
import json
import pathlib
from unittest import mock

from absl.testing import absltest
from model_debugger.server.analysis import kv_analysis
from model_debugger.server.analysis import metrics
from model_debugger.server.analysis import tensor_analysis as ta
from model_debugger.server.capture import capture_index
from model_debugger.server.capture import tensor_file
from model_debugger.server.graph import counterparts
from model_debugger.server.graph import pytorch_binding
from model_debugger.server.testing import capture_fixtures as fixtures
from model_debugger.server.testing import test_main
import numpy as np
from safetensors.numpy import save_file

_SHA = 'ab' * 32
_PREFILL_OUT = f'{fixtures.SIGNATURE}:3:0'
_PREFILL_7 = f'{fixtures.SIGNATURE}:7:0'
# The PyTorch runner's decoder layers of Gemma 4.
_LAYERS = 'model.language_model.layers.{}'
_QPROJ = _LAYERS + '.self_attn.q_proj'


def _observed(points=fixtures.TAP_POINTS, target_sha=_SHA) -> ta.Observed:
  return ta.Observed({
      'ref': ta.Side(_SHA, tuple(points)),
      'target': ta.Side(target_sha, tuple(points)),
  })


_FAILED_REASON = (
    'collect() failed at '
    'decode step 12: MemoryError: Snapshot memory limit '
    "exceeded at ('output',)"
)


class TensorAnalysisTest(absltest.TestCase):
  """Tests of places, cells, bins and one tensor, on tiny captures."""

  def setUp(self):
    """Gives each test an empty directory of its own."""
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)

  def _capture(self, target_seed=1, exchanges=None):
    return fixtures.capture(
        self.tmp_path,
        exchanges or [(' '.join(['w'] * 10), ['a', 'b', 'c'], True)],
        target_seed=target_seed,
        observe=True,
    )

  def test_a_label_drops_the_layers_scope_by_its_stem(self):
    """Gemma 4's layer_N and Qwen3's ...Qwen3DecoderLayer_N alike."""
    gemma = dataclasses.replace(
        fixtures.TAP_POINTS[3],
        tensor_name='stack/layer_7/layer_7.attn/add;fused/other',
        layer=7,
    )
    self.assertEqual(ta.label(gemma), 'attn/add · decode:5:0')
    stem = 'transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer'
    qwen = dataclasses.replace(
        gemma,
        tensor_name=f'Export/ModuleList_layers/{stem}_3/Linear_q_proj',
        layer=3,
        layer_stem=stem,
    )
    self.assertEqual(ta.label(qwen), 'Linear_q_proj · decode:5:0')
    # Without the stem, nothing of the layer's scope is dropped.
    self.assertEqual(
        ta.label(dataclasses.replace(qwen, layer_stem='')),
        f'{stem}_3/Linear_q_proj · decode:5:0',
    )
    self.assertEqual(
        ta.label(dataclasses.replace(gemma, layer_stem='')),
        'layer_7.attn/add · decode:5:0',
    )

  def test_places_are_read_from_the_points(self):
    """A layer's outputs of both graphs are one place; the rest by id."""
    places = ta.layout(_observed())
    self.assertIsNone(places.reason)
    self.assertEqual(places.layers, (0, 1))
    self.assertEqual(places.anchors, ('output', 'decode:5:0'))
    self.assertEqual(places.labels['output'], 'Layer output')
    self.assertEqual(places.labels['decode:5:0'], 'scores · decode:5:0')
    self.assertEqual(
        sorted(places.points['ref'][0, 'output']),
        ['decode', fixtures.SIGNATURE],
    )
    # The decode output that is in no layer has no place.
    self.assertNotIn(
        'decode:9:0',
        [p.id for p in places.place_points()],
    )

  def test_two_models_are_not_paired(self):
    """Points of models that differ are not compared."""
    places = ta.layout(_observed(target_sha='cd' * 32))
    self.assertEqual(places.reason['code'], ta.NOT_PAIRED)
    self.assertEqual(
        ta.nothing().reason,
        metrics.reason(ta.NOT_OBSERVED, 'This Session observes no tensors.'),
    )

  def test_two_models_pair_by_the_pairs_made_by_hand_alone(self):
    """Each pair a place of its own; nothing else of either is placed."""
    point = next(p for p in fixtures.TAP_POINTS if p.layer_output)
    place = (point.layer, ta.manual_anchor('p1'))
    observed = _observed(target_sha='cd' * 32)
    sides = {
        name: dataclasses.replace(
            side,
            places={point.id: (place,)},
            labels={place[1]: 'by hand'},
        )
        for name, side in observed.sides.items()
    }

    places = ta.layout(ta.Observed(sides))

    self.assertIsNone(places.reason)
    self.assertEqual(places.anchors, (place[1],))
    self.assertEqual(places.labels, {place[1]: 'by hand'})
    self.assertEqual({p.id for p in places.place_points()}, {point.id})

  def _rewrite(self, index, side, key, change):
    """Rewrites one side's file of a key at turn 1's second decode step."""
    # The second decode step: the first feeds the input's last token.
    forward = [
        f
        for f in index.turn(1)['sides'][side]['forwards']
        if f['phase'] == 'decode'
    ][1]
    path = (
        index.root
        / index.turn(1)['sides'][side]['dir']
        / forward['tensors'][key]
    )
    header = tensor_file.read_header(path)
    values = change(tensor_file.read_tensor(path))
    save_file({key: values}, str(path), metadata=header.metadata)
    return forward

  def test_the_same_tensors_compare_exactly_equal(self):
    """Equal sides: every cell exactly 0, the junk never read."""
    index = self._capture(target_seed=0)
    for stage in ('prefill', 'decode'):
      found = ta.stage(index, 1, stage, _observed(), None)
      self.assertEqual(found.reasons, [])
      self.assertNotEmpty(found.cells)
      for cell in found.cells:
        if len(cell.sources) != 2:
          # A column whose graph holds no tensor of the place.
          self.assertEqual(cell.status, 'unavailable')
          self.assertNotEmpty(cell.reasons)
          continue
        self.assertEqual(cell.status, metrics.OK, str(cell.reasons))
        self.assertEqual(cell.metrics['cosine_similarity'], 1.0)
        self.assertEqual(cell.metrics['relative_l2'], 0.0)
        self.assertEqual(cell.metrics['max_abs'], 0.0)

  def test_each_position_is_read_from_the_graph_that_produced_it(self):
    """Prefill chunks' rows that fed the token before; decode steps."""
    index = self._capture()
    side = index.turn(1)['sides']['ref']
    last = side['input'][-1]['position']
    found = ta.stage(index, 1, 'prefill', _observed(), None)
    # The first input token, which no step of the turn produced, is no
    # column of the stage.
    self.assertEqual((found.start, found.end), (1, last + 1))
    sources = {
        cell.column: cell.sources['ref']
        for cell in found.cells
        if cell.layer == 0 and cell.sources
    }
    # Column p is the chunk's row that fed p - 1.
    self.assertEqual(sorted(sources), list(range(1, last + 1)))
    for column, source in sources.items():
      self.assertEqual(source.point.id, _PREFILL_OUT)
      self.assertEqual(source.read, column - 1)
      self.assertEqual(
          source.forward['input_positions'][0] + source.row, column - 1
      )
    # Decode's points hold nothing of Prefill: the step that reads the
    # input's last token produces the first token generated.
    self.assertEqual({c.anchor for c in found.cells}, {'output'})
    self.assertEqual(found.unheld['decode:5:0'][0]['code'], ta.NOT_IN_STAGE)
    for cell in found.cells:
      if not cell.sources:
        continue
      self.assertEqual(cell.status, metrics.OK)
      self.assertLess(cell.metrics['cosine_similarity'], 1.0)
      self.assertGreater(cell.metrics['cosine_similarity'], 0.99)
      self.assertGreater(cell.metrics['max_abs'], 0.0)
    decode = ta.stage(index, 1, 'decode', _observed(), None)
    scores = [c for c in decode.cells if c.anchor == 'decode:5:0']
    positions = [t['position'] for t in side['output']]
    self.assertEqual([c.column for c in scores], positions)
    for cell in scores:
      # The context the step read: 0 to the position it fed.
      self.assertEqual(cell.sources['ref'].read, cell.column - 1)
      self.assertEqual(cell.shape, [2, cell.column])
      self.assertEqual(cell.status, metrics.OK)

  def test_each_column_is_the_step_final_logits_shows(self):
    """A Decode column p: the step that reads p - 1 and produces p."""
    index = self._capture()
    found = ta.stage(index, 1, 'decode', _observed(), None)
    for side in ('ref', 'target'):
      entry = index.turn(1)['sides'][side]
      producers = ta.produced(entry)
      for token in entry['output']:
        # Final Logits reads the token's own forward pass.
        self.assertEqual(
            producers[token['position']].forward['id'],
            token['forward'],
        )
        for cell in found.cells:
          if cell.column == token['position']:
            self.assertEqual(cell.sources[side].forward['id'], token['forward'])
      self.assertNotIn(entry['input'][0]['position'], producers)

  def test_two_sides_first_input_tokens_are_named_in_the_plural(self):
    """Sides whose turns start apart: both positions, one sentence."""

    class Apart:

      def positions(self, column):
        return {'ref': 20 + column, 'target': 24 + column}

    sides = {
        'ref': {'mine': {20, 21}, 'produced': {21: object()}},
        'target': {'mine': {24, 25}, 'produced': {25: object()}},
    }
    self.assertEqual(
        ta._produced_columns(0, 2, sides, Apart()),
        (
            1,
            2,
            [
                metrics.reason(
                    ta.POSITION_NOT_PRODUCED,
                    'Positions 20, 24 are not shown: each holds the '
                    'turn’s first input token, which no forward pass of '
                    'this turn produced.',
                )
            ],
        ),
    )

  def test_the_first_input_token_is_produced_by_no_step(self):
    """Prefill leaves out the first input token's column; Decode is full."""
    index = self._capture()
    found = ta.stage(index, 1, 'prefill', _observed(), None)
    self.assertEqual(found.start, 1)
    self.assertEqual([c for c in found.cells if c.column == 0], [])
    answer = ta.analysis(found)
    self.assertEqual(answer['rows'][0]['position'], 1)
    self.assertEqual(answer['columns'][0]['position'], 1)
    # It says why, once.
    self.assertEqual(
        answer['omitted'],
        [
            metrics.reason(
                ta.POSITION_NOT_PRODUCED,
                'Position 0 is not shown: it holds the turn’s first input '
                'token, which no forward pass of this turn produced.',
            )
        ],
    )
    self.assertEqual(
        ta.columns(index, 1, 'prefill', _observed(), None),
        {'start': 1, 'end': found.end, 'omitted': answer['omitted']},
    )
    # The last token generated is produced by the last decode step.
    decode = ta.stage(index, 1, 'decode', _observed(), None)
    positions = [t['position'] for t in index.turn(1)['sides']['ref']['output']]
    self.assertEqual(
        sorted({c.column for c in decode.cells if c.sources}), positions
    )

  def test_a_layer_one_side_computed_is_left_out(self):
    """A layer the other side's graph of the stage lacks: one reason."""
    index = self._capture()
    # The target's prefill graph has no point of layer 1.
    lacking = tuple(p for p in fixtures.TAP_POINTS if p.id != _PREFILL_7)
    observed = ta.Observed({
        'ref': ta.Side(_SHA, fixtures.TAP_POINTS),
        'target': ta.Side(_SHA, lacking),
    })
    found = ta.stage(index, 1, 'prefill', observed, None)
    self.assertEqual({c.layer for c in found.cells}, {0})
    answer = ta.analysis(found)
    self.assertEqual(answer['layers'], [0])
    self.assertEqual(
        answer['omitted'][-1],
        metrics.reason(
            ta.LAYERS_ONE_SIDE,
            'Layer 1 is not shown: the Target (LiteRT-LM) has no such '
            f'layer in its Prefill graph ({fixtures.SIGNATURE}); only the '
            'Reference (LiteRT-LM) computed it, so there is nothing to '
            'compare.',
        ),
    )
    ranged = ta.tensor_range(
        found, found.start, found.end, 1, None, 'mean', None
    )
    self.assertEqual(ranged['layers'], [0])
    # Decode, whose graph has the layer on both sides, keeps it.
    decode = ta.stage(index, 1, 'decode', observed, None)
    self.assertEqual({c.layer for c in decode.cells}, {0, 1})
    self.assertEqual(decode.omitted, [])

  def test_an_alignment_reads_each_side_at_its_own_position(self):
    """A column of an alignment is read at each side's position."""
    index = self._capture()
    side = index.turn(1)['sides']['ref']
    first = side['output'][0]['position']
    alignment = kv_analysis.Alignment(
        100, np.array([first, -1]), np.array([first, first])
    )
    found = ta.stage(index, 1, 'decode', _observed(), alignment)
    self.assertEqual((found.start, found.end), (100, 102))
    one_sided = [c for c in found.cells if c.column == 101]
    self.assertNotEmpty(one_sided)
    for cell in one_sided:
      self.assertEqual(set(cell.sources), {'target'})
      self.assertEqual(cell.reasons[0]['code'], ta.POSITION_MISSING)
      self.assertEqual(cell.reasons[0]['side'], 'ref')

  def test_alignments_compared_at_once_answer_as_each_alone(self):
    """One pass gives each alignment's answers, byte for byte, once read."""
    index = self._capture(
        exchanges=[(' '.join(['w'] * 10), ['a', 'b', 'c', 'd'], True)]
    )
    side = index.turn(1)['sides']['ref']
    p = [token['position'] for token in side['output']]
    # Comparing finds a reason of its own where a dtype differs.
    self._rewrite(
        index,
        'target',
        'post_tap_sg0_op3_out0',
        lambda values: values.astype(np.float16),
    )
    # Shared columns, columns of other positions, one side missing.
    alignments = [
        kv_analysis.Alignment(
            100, np.array([p[0], p[1], p[2]]), np.array([p[0], p[1], p[2]])
        ),
        kv_analysis.Alignment(
            100,
            np.array([p[0], p[2], -1, p[1]]),
            np.array([p[0], p[1], p[2], p[1]]),
        ),
        None,
    ]
    observed = _observed()

    def answers(found: ta.Stage) -> list[str]:
      ranged = ta.tensor_range(
          found, found.start, found.end, 2, None, 'mean', None
      )
      return [json.dumps(ta.analysis(found)), json.dumps(ranged)]

    alone = [
        answers(ta.stage(index, 1, 'decode', observed, alignment))
        for alignment in alignments
    ]
    with mock.patch.object(
        tensor_file, 'read_tensor', wraps=tensor_file.read_tensor
    ) as read:
      together = ta.stages(index, 1, 'decode', observed, alignments)
    self.assertEqual([answers(found) for found in together], alone)
    files = set()
    for found in together:
      for cell in found.cells:
        files.update(
            (side, source.file) for side, source in cell.sources.items()
        )
    self.assertLen(files, read.call_count)
    self.assertTrue(all(found.cells for found in together))

  def test_availability_says_why_a_stage_has_none(self):
    """No points, a graph that did not run, one that fed no position."""
    index = self._capture()
    self.assertEqual(ta.availability(index, 1, 'decode', _observed()), [])
    self.assertEqual(
        ta.availability(index, 1, 'decode', ta.nothing())[0]['code'],
        ta.NOT_OBSERVED,
    )
    prefill_only = [p for p in fixtures.TAP_POINTS if p.signature != 'decode']
    (why,) = ta.availability(index, 1, 'decode', _observed(prefill_only))
    self.assertEqual(why['code'], ta.NOT_IN_STAGE)
    self.assertIsNone(why['side'])
    self.assertIn(f'{fixtures.SIGNATURE} produced no position', why['text'])
    decode_only = [p for p in fixtures.TAP_POINTS if p.signature == 'decode']
    (why,) = ta.availability(index, 1, 'prefill', _observed(decode_only))
    self.assertEqual(why['code'], ta.NOT_IN_STAGE)
    self.assertIn('decode produced no position', why['text'])
    verify = dataclasses.replace(
        fixtures.TAP_POINTS[2], id='verify:3:0', signature='verify'
    )
    (why,) = ta.availability(index, 1, 'prefill', _observed([verify]))
    self.assertEqual(why['code'], ta.GRAPH_NOT_RUN)
    self.assertEqual(
        why['text'], 'verify:3:0: verify did not run in this turn.'
    )
    nameless = [p for p in fixtures.TAP_POINTS if p.layer is None]
    (why,) = ta.availability(index, 1, 'decode', _observed(nameless))
    self.assertEqual(why['code'], ta.NO_LAYER)

  def test_analysis_lists_places_points_and_rows(self):
    """position-analysis: layers, anchors, points, a row per cell."""
    index = self._capture()
    answer = ta.analysis(ta.stage(index, 1, 'decode', _observed(), None))
    generated = len(index.turn(1)['sides']['ref']['output'])
    self.assertEqual(answer['layers'], [0, 1])
    self.assertEqual(
        [a['id'] for a in answer['anchors']], ['output', 'decode:5:0']
    )
    self.assertEqual(answer['pairing'], {'status': 'paired', 'reasons': []})
    points = {p['id']: p for p in answer['points']}
    self.assertEqual(points['decode:5:0']['context_axes'], [3])
    self.assertEqual(points['decode:5:0']['cells'], generated)
    self.assertEqual(points['decode:5:0']['reasons'], [])
    # Its node in the model's graph: the signature's graph, op, output.
    self.assertEqual(
        points['decode:5:0']['graph_node'],
        {'graph': 'decode', 'node': '5', 'output': '0'},
    )
    row = answer['rows'][0]
    self.assertEqual(
        set(row),
        {
            'position',
            'layer',
            'anchor',
            'status',
            'reasons',
            'shape',
            'metrics',
            'points',
        },
    )
    self.assertEqual(row['shape'], [fixtures.HIDDEN])
    # The point each side's value is of.
    held = next(r for r in answer['rows'] if r['anchor'] == 'decode:5:0')
    self.assertEqual(
        held['points'], {'ref': 'decode:5:0', 'target': 'decode:5:0'}
    )
    # Per column, the forward pass each side reads it from.
    columns = answer['columns']
    self.assertEqual(
        [c['position'] for c in columns],
        list(range(answer['start'], answer['end'])),
    )
    feeds = ta.produced(index.turn(1)['sides']['ref'])
    for column in columns:
      forward = feeds[column['position']].forward
      self.assertEqual(
          column['ref'],
          {'signature': forward['signature'], 'step': forward['step']},
      )
      self.assertEqual(column['target'], column['ref'])
    self.assertEqual(answer['not_observed']['code'], ta.POINT_NOT_OBSERVED)
    self.assertEqual(set(row['metrics']), set(ta.METRICS))
    self.assertEqual(row['metrics']['max_abs']['reasons'], [])
    # Prefill holds none of decode's points, and says why.
    prefill = ta.analysis(ta.stage(index, 1, 'prefill', _observed(), None))
    self.assertEqual([a['id'] for a in prefill['anchors']], ['output'])
    points = {p['id']: p for p in prefill['points']}
    self.assertEqual(points['decode:5:0']['cells'], 0)
    self.assertEqual(
        points['decode:5:0']['reasons'][0]['code'], ta.NOT_IN_STAGE
    )

  def test_range_bins_each_layer_over_the_anchors_chosen(self):
    """Each layer's reduction over anchors, binned; the extremum's."""
    index = self._capture()
    found = ta.stage(index, 1, 'decode', _observed(), None)
    answer = ta.tensor_range(
        found, found.start, found.end, 2, None, 'min', None
    )
    self.assertEqual([r['layer'] for r in answer['rows']], [0, 1])
    self.assertLen(answer['bins'], 2)
    last = found.end - 1
    layer1 = answer['totals'][1]['bins'][0]['metrics']['cosine_similarity']
    lowest = min(
        c.metrics['cosine_similarity']
        for c in found.cells
        if c.layer == 1 and c.sources
    )
    self.assertEqual(layer1['min'], lowest)
    self.assertEqual(layer1['valid_count'], found.end - found.start)
    at = layer1['min_position']
    values = {
        c.anchor: c.metrics['cosine_similarity']
        for c in found.cells
        if c.layer == 1 and c.column == at
    }
    self.assertEqual(layer1['min_anchor'], min(values, key=values.get))
    self.assertEqual(answer['metadata']['output']['dtype'], 'float64')
    self.assertEqual(
        answer['metadata']['output']['shape'], [found.end - found.start, 2]
    )
    only = ta.tensor_range(
        found, last, last + 1, 1, ['decode:5:0'], 'mean', [0, 1]
    )
    self.assertEqual(only['rows'][0]['status'], 'unavailable')
    self.assertEqual(only['rows'][0]['reasons'][0]['code'], ta.PLACE_MISSING)
    scores = found.cell(last, 1, 'decode:5:0').metrics['max_abs']
    bin_ = only['rows'][1]['bins'][0]['metrics']['max_abs']
    self.assertEqual((bin_['min'], bin_['max']), (scores, scores))
    self.assertEqual(bin_['reasons'], [])
    self.assertEqual(answer['metadata']['ref']['dtype'], 'F32')

  def test_columns_without_a_value_say_why_in_their_bins(self):
    """Columns another graph produced: their bins say why."""
    index = self._capture()
    # The second chunk ran another graph, which holds none of the points.
    chunks = []
    for side in ('ref', 'target'):
      forward = index.turn(1)['sides'][side]['forwards'][1]
      self.assertEqual(forward['phase'], 'prefill')
      forward['signature'] = 'prefill_other'
      chunks.append(forward)
    found = ta.stage(index, 1, 'prefill', _observed(), None)
    last = found.end - 1
    answer = ta.tensor_range(
        found,
        found.start,
        found.end,
        found.end,
        ['output'],
        'mean',
        [1],
    )
    (row,) = answer['rows']
    self.assertEqual((row['status'], row['reasons']), ('ok', []))
    # Column p is produced by the row that fed p - 1; the bins start at
    # the stage's first column.
    other = chunks[0]['input_positions'][0] + 1
    step = chunks[0]['step']
    self.assertEqual(
        row['bins'][other - found.start]['metrics']['cosine_similarity'][
            'reasons'
        ],
        [
            metrics.reason(
                ta.OTHER_GRAPH,
                f'prefill_other step {step} produced position {other}; '
                'it holds no tensor of this sync point.',
            )
        ],
    )
    self.assertEqual(
        row['bins'][0]['metrics']['cosine_similarity']['reasons'], []
    )
    self.assertGreater(
        row['bins'][0]['metrics']['cosine_similarity']['valid_count'], 0
    )
    whole = answer['totals'][0]['bins'][0]['metrics']['max_abs']
    self.assertLen(whole['reasons'], ta.MAX_REASONS)
    self.assertGreater(last, other)

  def test_one_tensor_element_by_element(self):
    """position-tensor: both sides' elements, their deltas, statistics."""
    index = self._capture()
    found = ta.stage(index, 1, 'decode', _observed(), None)
    first = found.start
    answer = ta.tensor(found.cell(first, 1, 'decode:5:0'))
    self.assertTrue(answer['comparable'])
    # The step that reads the input's last token: its context up to it.
    self.assertLen(answer['channels'], 2 * first)
    channel = answer['channels'][0]
    self.assertAlmostEqual(channel['delta'], channel['target'] - channel['ref'])
    self.assertEqual(answer['reference']['shape'], [2, first])
    self.assertEqual(answer['reference']['reads'], first - 1)
    self.assertEqual(answer['reference']['step'], first)
    self.assertEqual(
        answer['reference']['source_shape'], [1, 1, 2, fixtures.KV_LENGTH]
    )
    self.assertEqual(answer['reference']['dtype'], 'F32')
    self.assertEqual(
        answer['statistics']['abs_delta']['max'],
        max(c['abs_delta'] for c in answer['channels']),
    )
    alone = ta.cell(
        index, 1, 'decode', _observed(), None, first, 1, 'decode:5:0'
    )
    self.assertEqual(ta.tensor(alone), answer)
    # A column no step produced is no column of the stage.
    prefill = ta.stage(index, 1, 'prefill', _observed(), None)
    with self.assertRaisesRegex(ValueError, 'no tensor of layer'):
      prefill.cell(0, 1, 'output')
    with self.assertRaisesRegex(ValueError, 'no tensor of layer'):
      found.cell(found.end, 1, 'decode:5:0')
    with self.assertRaisesRegex(ValueError, 'no tensor of layer'):
      ta.cell(index, 1, 'prefill', _observed(), None, 0, 1, 'decode:5:0')

  def test_too_large_a_tensor_is_not_listed(self):
    """Over MAX_ELEMENTS: no elements, and why."""
    index = self._capture()
    found = ta.stage(index, 1, 'decode', _observed(), None)
    with mock.patch.object(ta, 'MAX_ELEMENTS', 2):
      answer = ta.tensor(found.cells[0])
    self.assertEqual(answer['status'], ta.TOO_LARGE)
    self.assertEqual(answer['channels'], [])
    self.assertEqual(answer['reasons'][-1]['code'], ta.TOO_LARGE)
    self.assertIsNotNone(answer['metrics']['max_abs']['value'])

  def test_a_shape_mismatch_pads_with_null(self):
    """Two sides at two positions of a context axis: nulls past one."""
    index = self._capture()
    first = index.turn(1)['sides']['ref']['output'][0]['position']
    alignment = kv_analysis.Alignment(
        100, np.array([first]), np.array([first + 1])
    )
    cell = ta.cell(
        index, 1, 'decode', _observed(), alignment, 100, 1, 'decode:5:0'
    )
    self.assertEqual(cell.status, metrics.SHAPE_MISMATCH)
    answer = ta.tensor(cell)
    self.assertLen(answer['channels'], 2 * (first + 1))
    self.assertIsNone(answer['channels'][-1]['ref'])
    self.assertIsNotNone(answer['channels'][-1]['target'])
    self.assertIsNone(answer['channels'][0]['delta'])
    self.assertEqual(answer['statistics']['ref']['valid_count'], 2 * first)

  def test_two_stored_dtypes_are_not_compared(self):
    """dtype_mismatch, as KV's; each side's stored dtype in the range."""
    index = self._capture()
    forward = self._rewrite(
        index,
        'target',
        'post_tap_sg0_op3_out0',
        lambda values: values.astype(np.float16),
    )
    found = ta.stage(index, 1, 'decode', _observed(), None)
    # The step produces the token after the one it reads.
    column = forward['input_positions'][0] + 1
    cell = found.cell(column, 0, 'output')
    self.assertEqual(cell.status, ta.DTYPE_MISMATCH)
    self.assertEqual(
        cell.reasons,
        [
            metrics.reason(
                ta.DTYPE_MISMATCH, 'Ref stores F32 and Target stores F16.'
            )
        ],
    )
    self.assertIsNone(cell.metrics['max_abs'])
    answer = ta.tensor_range(
        found, found.start, found.end, 1, ['output'], 'mean', [0]
    )
    self.assertEqual(answer['metadata']['ref']['dtype'], 'F32')
    self.assertEqual(answer['metadata']['target']['dtype'], ['F16', 'F32'])
    tensor = ta.tensor(cell)
    self.assertEqual(tensor['target']['dtype'], 'F16')
    self.assertFalse(tensor['comparable'])

  def test_a_file_not_of_the_models_shape_is_not_read(self):
    """SHAPE_UNEXPECTED, from the file's header."""
    index = self._capture()
    wider = [
        dataclasses.replace(p, shape=(1, fixtures.PREFILL_SIZE, 7))
        if p.signature == fixtures.SIGNATURE
        else p
        for p in fixtures.TAP_POINTS
    ]
    found = ta.stage(index, 1, 'prefill', _observed(wider), None)
    cell = found.cell(1, 0, 'output')
    self.assertEqual(cell.status, 'unavailable')
    self.assertEqual(
        [(r['code'], r['side']) for r in cell.reasons],
        [(ta.SHAPE_UNEXPECTED, 'ref'), (ta.SHAPE_UNEXPECTED, 'target')],
    )
    self.assertIn('the model makes it [1, 8, 7]', cell.reasons[0]['text'])

  def test_a_tensor_with_no_token_axis_is_not_split(self):
    """NO_TOKEN_AXIS: its cells say so; with no other, nothing compares."""
    index = self._capture()
    square = dataclasses.replace(
        fixtures.TAP_POINTS[0],
        shape=(fixtures.PREFILL_SIZE, fixtures.PREFILL_SIZE),
    )
    observed = _observed([square])
    found = ta.stage(index, 1, 'prefill', observed, None)
    cell = found.cell(1, 0, 'output')
    self.assertEqual(cell.sources, {})
    self.assertEqual(cell.reasons[0]['code'], ta.NO_TOKEN_AXIS)
    self.assertIsNone(cell.reasons[0]['side'])
    (why,) = ta.availability(index, 1, 'prefill', observed)
    self.assertEqual(why['code'], ta.NO_TOKEN_AXIS)
    self.assertIsNone(why['side'])
    self.assertEqual(found.reasons, [why])
    self.assertEqual(ta.batches(index, observed), [])

  def test_a_thinking_turn_reads_its_reply_not_its_rewrite(self):
    """The rewrite's chunks feed the reply again; decode is read."""
    index = self._capture(
        exchanges=[
            ('Hi there', ['a', 'b'], True, ['x', 'y']),
            ('More', ['c'], True),
        ]
    )
    side = index.turn(1)['sides']['ref']
    rewritten = {token['forward'] for token in side['rewrite']}
    self.assertNotEmpty(rewritten)
    # The rewrite's chunks hold the observed tensors too.
    for forward_id in rewritten:
      self.assertIn(
          'post_tap_sg1_op3_out0', side['forwards'][forward_id]['tensors']
      )
    found = ta.stage(index, 1, 'decode', _observed(), None)
    read = set()
    for cell in found.cells:
      read.update(source.forward['id'] for source in cell.sources.values())
    self.assertNotEmpty(read)
    self.assertEqual(read & rewritten, set())
    for cell in found.cells:
      for source in cell.sources.values():
        self.assertEqual(source.forward['phase'], 'decode')
    self.assertEqual(ta.availability(index, 2, 'prefill', _observed()), [])
    self.assertIn(
        (2, 'prefill', fixtures.SIGNATURE),
        [
            (b['turn'], b['phase'], b['signature'])
            for b in ta.batches(index, _observed())
        ],
    )

  def test_batches_are_the_graphs_that_hold_observed_tensors(self):
    """One per turn, stage and graph that produced it on both sides."""
    index = self._capture()
    found = ta.batches(index, _observed())
    self.assertEqual(
        [(b['turn'], b['phase'], b['signature']) for b in found],
        [
            (1, 'prefill', fixtures.SIGNATURE),
            (1, 'decode', 'decode'),
        ],
    )
    self.assertEqual([b['batch'] for b in found], [0, 1])
    # Decode's first step is the one that reads the input's last token.
    output = index.turn(1)['sides']['ref']['output']
    self.assertEqual(found[1]['step'], output[0]['position'])
    self.assertTrue(
        found[1]['comparison_basis'].startswith(f'{len(output)} decode steps')
    )
    self.assertEqual(ta.batches(index, ta.nothing()), [])

  # Two runtimes of one architecture, paired by anchor.

  def _cross_capture(self, captured=True, module=False):
    """Ref as the PyTorch runner writes it, Target a LiteRT-LM tap.

    Ref's layer outputs are its tap points' values, in BF16, under the
    anchor keys, its tap files gone; `captured` False leaves it no
    anchors and says why, as its runner does. With `module`, each is
    also the output of the module _QPROJ, by heads ([2, 3]), under its
    module key.

    Args:
      captured: Whether Ref's runner captured its anchors.
      module: Whether Ref also has the module _QPROJ's outputs.

    Returns:
      The capture index of the one-turn chat.
    """
    chats = {
        'ref': fixtures.TinyChat(seed=0, observe=True),
        'target': fixtures.TinyChat(seed=1, observe=True, junk=1.0),
    }
    index = capture_index.CaptureIndex.create(
        self.tmp_path,
        {side: fixtures.side_info(chat) for side, chat in chats.items()},
    )
    outputs = {
        f'post_tap_sg{p.subgraph}_op{p.op}_out0': p.layer
        for p in fixtures.TAP_POINTS
        if p.layer_output
    }
    completions = {}
    for side, chat in chats.items():
      directory = index.turn_directory(side, 1)
      completions[side] = fixtures.write_turn(
          directory, chat.send(' '.join(['w'] * 10), ['a', 'b', 'c'])
      )
      if side != 'ref':
        continue
      for path in sorted(directory.glob('*_post_tap_*')):
        header = tensor_file.read_header(path)
        layer = outputs.get(header.key)
        if layer is not None and captured:
          signature = header.metadata['signature']
          step = header.metadata['step']
          key = f'post_anchor_{layer}_output'
          (
              directory / f'{signature}_{key}_step_{step}.safetensors'
          ).write_bytes(
              fixtures.save_bf16(
                  key,
                  tensor_file.read_tensor(path),
                  {
                      'signature': signature,
                      'step': step,
                      'layer': str(layer),
                      'anchor': 'output',
                      'module': _LAYERS.format(layer),
                      'edge': 'out',
                      'path': 'output',
                  },
              )
          )
        if layer is not None and captured and module:
          name = _QPROJ.format(layer)
          key = capture_index.module_key(name, 'out')
          values = tensor_file.read_tensor(path)
          (
              directory / f'{signature}_{key}_step_{step}.safetensors'
          ).write_bytes(
              fixtures.save_bf16(
                  key,
                  values.reshape(*values.shape[:-1], 2, 3),
                  {
                      'signature': signature,
                      'step': step,
                      'module': name,
                      'edge': 'out',
                      'path': 'output',
                  },
              )
          )
        path.unlink()
      (directory / 'observation.json').write_text(
          json.dumps({
              'observe': 'anchors',
              'memory_limit_bytes': 1000,
              'captured': captured,
              'reason': '' if captured else _FAILED_REASON,
              'snapshot_bytes': 0,
          })
      )
    index.add_turn(1, 'w', completions)
    return index

  def _cross_observed(self, modules=False, **target) -> ta.Observed:
    bound = pytorch_binding.Bound(
        architecture='gemma4',
        width=fixtures.HIDDEN,
        signatures=(fixtures.SIGNATURE, 'decode'),
        anchors=tuple(
            pytorch_binding.Anchor(
                layer,
                'output',
                'layer output',
                _LAYERS.format(layer),
                'out',
                'output',
            )
            for layer in range(fixtures.LAYERS)
        ),
    )
    bindings = collections.defaultdict(list)
    for point in fixtures.TAP_POINTS:
      if point.layer_output:
        bindings[point.signature].append({
            'layer': point.layer,
            'anchors': {'output': {'bound': True, 'point': point.id}},
        })
    side = ta.Side(
        _SHA,
        fixtures.TAP_POINTS,
        architecture='gemma4',
        places=ta.anchor_places(bindings, {p.id for p in fixtures.TAP_POINTS}),
        labels={'output': 'layer output'},
    )
    if not modules:
      return ta.Observed({
          'ref': ta.pytorch_side('ef' * 32, bound, 'bfloat16'),
          'target': dataclasses.replace(side, **target),
      })
    # Each layer output taken for q_proj's too: the place of a semantic
    # operator, its module's tensor by heads.
    paired = [
        counterparts.Counterpart(
            point_id=point.id,
            layer=point.layer,
            op='attn/q_proj',
            label='Attention q_proj',
            module=counterparts.Module(
                _QPROJ.format(point.layer), 'out', 'output', (2, 3)
            ),
        )
        for point in fixtures.TAP_POINTS
        if point.layer_output
    ]
    labels = {'attn/q_proj': 'Attention q_proj', 'output': 'layer output'}
    places = {
        point_id: (*found, (found[0][0], 'attn/q_proj'))
        for point_id, found in side.places.items()
    }
    note = metrics.reason(
        ta.NO_COUNTERPART, 'PyTorch has no counterpart for decode:5:0.'
    )
    return ta.Observed(
        {
            'ref': ta.pytorch_side(
                'ef' * 32, bound, 'bfloat16', paired, labels
            ),
            'target': dataclasses.replace(
                side, places=places, labels=labels, **target
            ),
        },
        notes=(note,),
    )

  def test_two_runtimes_pair_by_anchor_across_dtypes(self):
    """PyTorch's BF16 anchors against LiteRT-LM's F32 points there."""
    index = self._cross_capture()
    observed = self._cross_observed()
    places = ta.layout(observed)
    self.assertIsNone(places.reason)
    self.assertTrue(places.cross)
    self.assertEqual(places.anchors, ('output',))
    self.assertEqual(places.labels, {'output': 'layer output'})
    self.assertEqual(places.layers, (0, 1))
    for stage in ('prefill', 'decode'):
      self.assertEqual(ta.availability(index, 1, stage, observed), [])
      found = ta.stage(index, 1, stage, observed, None)
      held = [cell for cell in found.cells if len(cell.sources) == 2]
      self.assertNotEmpty(held)
      for cell in held:
        self.assertEqual(cell.status, metrics.OK, cell.reasons)
        self.assertEqual(cell.dtypes, {'ref': 'BF16', 'target': 'F32'})
        self.assertGreater(cell.metrics['cosine_similarity'], 0.99)
        # BF16 keeps 8 bits of values about 10: off by up to 0.06.
        self.assertLess(cell.metrics['max_abs'], 0.25)
      tensor = ta.tensor(held[0])
      self.assertTrue(tensor['comparable'])
      self.assertEqual(tensor['reference']['dtype'], 'BF16')
      self.assertEqual(
          tensor['reference']['tensor_name'],
          'model.language_model.layers.0 (returns output)',
      )
      answer = ta.analysis(found)
      ref_points = [p for p in answer['points'] if p['graph_node'] is None]
      self.assertLen(ref_points, 2 * fixtures.LAYERS)
      self.assertEqual(ref_points[0]['shape'], [1, ta.ROWS, 6])
      self.assertEqual(
          {p['id'] for p in answer['points'] if p['graph_node']},
          {p.id for p in fixtures.TAP_POINTS if p.layer_output},
      )
    self.assertEqual(
        [b['signature'] for b in ta.batches(index, observed)],
        [fixtures.SIGNATURE, 'decode'],
    )

  def test_a_semantic_operator_pairs_with_its_module_flat(self):
    """A LiteRT-LM point placed at q_proj, against its module by heads."""
    index = self._cross_capture(module=True)
    observed = self._cross_observed(modules=True)
    places = ta.layout(observed)
    self.assertIsNone(places.reason)
    self.assertEqual(places.anchors, ('attn/q_proj', 'output'))
    self.assertEqual(places.labels['attn/q_proj'], 'Attention q_proj')
    for stage in ('prefill', 'decode'):
      found = ta.stage(index, 1, stage, observed, None)
      cells = [
          cell
          for cell in found.cells
          if cell.anchor == 'attn/q_proj' and len(cell.sources) == 2
      ]
      self.assertNotEmpty(cells)
      for cell in cells:
        self.assertEqual(cell.status, metrics.OK, cell.reasons)
        self.assertEqual(cell.shape, [fixtures.HIDDEN])
        self.assertGreater(cell.metrics['cosine_similarity'], 0.99)
        self.assertEqual(
            cell.sources['ref'].point.id,
            f'{cell.sources["ref"].point.signature}:'
            f'{_QPROJ.format(cell.layer)}:out',
        )
      answer = ta.analysis(found)
      self.assertEqual(
          [a['id'] for a in answer['anchors']], ['attn/q_proj', 'output']
      )
      (note,) = answer['unpaired']
      self.assertEqual(note['code'], ta.NO_COUNTERPART)
      module = next(
          p
          for p in answer['points']
          if p['id'].endswith('self_attn.q_proj:out')
      )
      self.assertEqual(module['shape'], [1, ta.ROWS, 2, 3])
      self.assertEqual(module['anchor'], 'attn/q_proj')

  def test_a_pair_made_by_hand_is_a_place_of_its_own(self):
    """Its two tensors in its signature; a module not asked says so."""
    index = self._cross_capture(module=True)
    observed = self._cross_observed()
    outputs = {
        p.layer: p
        for p in fixtures.TAP_POINTS
        if p.layer_output and p.signature == 'decode'
    }
    ref, target = observed.sides['ref'], observed.sides['target']
    added = {
        # q_proj's output, by heads, against layer 1's output.
        'p1': (
            ta.module_point(
                'decode',
                _QPROJ.format(1),
                'out',
                (2, 3),
                'bfloat16',
                layer=1,
                op=100,
            ),
            outputs[1],
        ),
        # o_proj's input, which the runner was not asked for.
        'p2': (
            ta.module_point(
                'decode',
                _LAYERS.format(0) + '.self_attn.o_proj',
                'in',
                (fixtures.HIDDEN,),
                'bfloat16',
                layer=0,
                op=101,
            ),
            outputs[0],
        ),
    }
    for pair_id, (module, point) in added.items():
      place = (module.layer, ta.manual_anchor(pair_id))
      ref = dataclasses.replace(
          ref,
          points=(*ref.points, module),
          places={**ref.places, module.id: (place,)},
          labels={**ref.labels, place[1]: f'by hand {pair_id}'},
          keys={
              **ref.keys,
              module.id: capture_index.module_key(
                  module.tensor_name.split(' ')[0],
                  'in' if module.id.endswith(':in') else 'out',
              ),
          },
      )
      target = dataclasses.replace(
          target,
          places={
              **target.places,
              point.id: (*target.places[point.id], place),
          },
          labels={**target.labels, place[1]: f'by hand {pair_id}'},
      )
    observed = ta.Observed({'ref': ref, 'target': target})

    places = ta.layout(observed)

    self.assertIsNone(places.reason)
    self.assertEqual(places.anchors, ('output', 'manual:p1', 'manual:p2'))
    found = ta.stage(index, 1, 'decode', observed, None)
    by_hand = [c for c in found.cells if c.anchor == 'manual:p1']
    self.assertNotEmpty(by_hand)
    for cell in by_hand:
      self.assertEqual(cell.layer, 1)
      self.assertEqual(cell.status, metrics.OK, cell.reasons)
      self.assertEqual(cell.shape, [fixtures.HIDDEN])
      self.assertGreater(cell.metrics['cosine_similarity'], 0.99)
    missing = [c for c in found.cells if c.anchor == 'manual:p2']
    self.assertNotEmpty(missing)
    for cell in missing:
      self.assertEqual(set(cell.sources), {'target'})
      (reason,) = cell.reasons
      self.assertEqual(reason['code'], ta.NOT_CAPTURED)
      self.assertEqual(
          reason['text'],
          'The PyTorch runner was not asked for '
          'model.language_model.layers.0.self_attn.o_proj, what it '
          'reads, when this Session started; a Session of these models '
          'started from now on captures it.',
      )
    answer = ta.analysis(found)
    self.assertIn(
        {'id': 'manual:p1', 'label': 'by hand p1', 'default': False},
        answer['anchors'],
    )
    # The architecture's anchor is a default place; a pair by hand is not.
    self.assertEqual(
        [a['id'] for a in answer['anchors'] if a['default']], ['output']
    )
    # A prefill runs another signature: the pair has no place there.
    prefill = ta.stage(index, 1, 'prefill', observed, None)
    self.assertFalse([c for c in prefill.cells if ta.is_manual(c.anchor)])

  def test_a_turn_its_pytorch_runner_captured_nothing_of_says_why(self):
    """The runner's reason, as its observation.json gives it."""
    index = self._cross_capture(captured=False)
    (reason,) = ta.availability(index, 1, 'decode', self._cross_observed())
    self.assertEqual(reason['code'], ta.NOT_CAPTURED)
    self.assertEqual(reason['side'], 'ref')
    self.assertIn('MemoryError: Snapshot memory limit exceeded', reason['text'])

  def test_two_runtimes_need_anchors_of_one_architecture(self):
    """No anchor point, another architecture, or no point at all."""
    cases = {
        ta.NO_ANCHOR: {'places': {}},
        ta.OTHER_ARCHITECTURE: {'architecture': None},
        ta.SIDE_NOT_OBSERVED: {'points': ()},
    }
    for code, change in cases.items():
      reason = ta.layout(self._cross_observed(**change)).reason
      self.assertEqual(reason['code'], code)
      self.assertIn(code, metrics.REASONS | ta.REASONS)
    self.assertEqual(
        ta.layout(self._cross_observed(places={})).reason['side'], 'target'
    )


if __name__ == '__main__':
  test_main.main()
