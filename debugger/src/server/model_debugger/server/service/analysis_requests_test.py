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

"""Tests of the analysis routes' requests and the rows they name."""

import math
import pathlib
import re
from typing import Any
from unittest import mock

from absl.testing import parameterized
from model_debugger.server.analysis import kv_analysis
from model_debugger.server.analysis import tensor_analysis
from model_debugger.server.analysis import token_analysis
from model_debugger.server.capture import capture_index
from model_debugger.server.service import analysis_requests
from model_debugger.server.service import pair_values
from model_debugger.server.service import session_view
from model_debugger.server.testing import capture_fixtures as fixtures
from model_debugger.server.testing import test_main
from model_debugger.server.util import errors

_DECODE = {'turn': 1, 'stage': 'decode', 'alignment': 'content'}
_PAGE = {'transform': 'raw', 'offset': 0, 'limit': 4, 'order': 'index'}


class AnalysisRequestsTest(parameterized.TestCase):
  """A turn that diverges: the target says ' x' where ref goes on."""

  def setUp(self):
    """Ref replies 'a b', target 'a x b': the rows part at row 1."""
    super().setUp()
    root = pathlib.Path(self.create_tempdir().full_path)
    self.chats = {
        side: fixtures.TinyChat('', 8, seed=seed)
        for seed, side in enumerate(capture_index.SIDES)
    }
    self.index = capture_index.CaptureIndex.create(
        root,
        {side: fixtures.side_info(chat) for side, chat in self.chats.items()},
    )
    replies = {'ref': ['a', 'b'], 'target': ['a', ' x', 'b']}
    completions = {
        side: fixtures.write_turn(
            self.index.turn_directory(side, 1),
            chat.send('Hi', replies[side], True),
        )
        for side, chat in self.chats.items()
    }
    self.index.add_turn(1, 'Hi', completions)
    self.views = session_view.SessionViews()
    self.compared = self.views.compared('chat', 1, self.index)
    self._comparison = analysis_requests.Comparison(
        lambda: self.views.compared('chat', 1, self.index)
    )

  def test_rows_are_the_comparisons_rows(self):
    """Each pair is the steps its row names, and says which row."""
    content = self.compared.comparisons[1]['alignments']['content']
    answer = analysis_requests.token_analysis(
        self.index, self._comparison, {**_DECODE, 'rows': [0, 1, 2]}
    )

    self.assertEqual(
        {key: answer[key] for key in ('turn', 'stage', 'alignment')},
        _DECODE,
    )
    self.assertEqual(
        [
            (pair['row'], pair['ref_step'], pair['target_step'])
            for pair in answer['pairs']
        ],
        [
            (k, row['ref'], row['target'])
            for k, row in enumerate(content['rows'][:3])
        ],
    )
    expected = token_analysis.token_analysis(
        self.index, 1, [(0, 0), (None, 1), (1, 2)]
    )
    for pair, alone in zip(answer['pairs'], expected, strict=True):
      self.assertEqual({key: pair[key] for key in alone}, alone, pair['row'])
      self.assertEqual(
          (pair['stage'], pair['alignment'], pair['first_divergence']),
          ('decode', 'content', False),
      )

  def test_the_first_divergence_is_filled_in_by_step(self):
    """Its row holds the target's ' x' alone; the pair is step 1 of both."""
    answer = analysis_requests.token_analysis(
        self.index,
        self._comparison,
        {**_DECODE, 'rows': [], 'first_divergence': True},
    )

    (pair,) = answer['pairs']
    self.assertEqual((pair['row'], pair['first_divergence']), (1, True))
    self.assertEqual((pair['ref_step'], pair['target_step']), (1, 1))
    page = analysis_requests.final_logits(
        self.index,
        self._comparison,
        {**_DECODE, 'row': 1, 'first_divergence': True, **_PAGE},
    )
    self.assertEqual(
        (page['row'], page['ref_step'], page['target_step']), (1, 1, 1)
    )
    with self.assertRaisesRegex(ValueError, 'not the first divergence'):
      analysis_requests.final_logits(
          self.index,
          self._comparison,
          {**_DECODE, 'row': 0, 'first_divergence': True, **_PAGE},
      )

  def test_final_logits_of_a_row(self):
    """The page, summary, bins and points of a row's pair, and its row."""
    page = analysis_requests.final_logits(
        self.index, self._comparison, {**_DECODE, 'row': 2, **_PAGE}
    )
    expected = token_analysis.final_logits(
        self.index, 1, 1, 2, 'raw', 0, 4, 'index', 'ascending'
    )
    self.assertEqual({key: page[key] for key in expected}, expected)
    self.assertEqual((page['row'], page['first_divergence']), (2, False))

    row = {**_DECODE, 'alignment': 'steps', 'row': 1, 'transform': 'raw'}
    for route, payload, made in (
        (
            analysis_requests.final_logits_summary,
            row,
            lambda: token_analysis.final_logits_summary(
                self.index, 1, 1, 1, 'raw'
            ),
        ),
        (
            analysis_requests.final_logits_bins,
            {
                **row,
                'start': 1,
                'end': 4,
                'capacity': 2,
                'reduction': 'min',
            },
            lambda: token_analysis.final_logits_bins(
                self.index, 1, 1, 1, 'raw', (1, 4), 2, 'min'
            ),
        ),
        (
            analysis_requests.final_logits_points,
            {**row, 'start': -1, 'end': 2.5},
            lambda: token_analysis.final_logits_points(
                self.index, 1, 1, 1, 'raw', (-1.0, 2.5)
            ),
        ),
    ):
      with self.subTest(route.__name__):
        answer = route(self.index, self._comparison, payload)
        expected = made()
        self.assertEqual({key: answer[key] for key in expected}, expected)
        self.assertEqual(answer['alignment'], 'steps')

  @parameterized.parameters(
      ({'start': 1},),
      ({'start': 2, 'end': 2},),
      ({'start': -1, 'end': 2},),
      ({'start': 0.5, 'end': 2},),
      ({'capacity': 0},),
      ({'capacity': 8193},),
      ({'capacity': 2.0},),
      ({'reduction': 'median'},),
  )
  def test_final_logits_bins_refuses_malformed_payloads(self, fields):
    """A range of indices, both ends or neither; a known reduction."""
    payload = {
        **_DECODE,
        'row': 0,
        'transform': 'raw',
        'capacity': 4,
        'reduction': 'mean',
        **fields,
    }
    with self.assertRaises(ValueError):
      analysis_requests.final_logits_bins(self.index, self._comparison, payload)

  @parameterized.parameters(
      ({'end': 1.0},),
      ({'start': 1.0, 'end': 1.0},),
      ({'start': 'NaN', 'end': 1.0},),
      ({'start': 0.0, 'end': math.inf},),
      ({'start': True, 'end': 2},),
      ({'start': 0, 'end': 10**400},),
      ({'start': -(10**400), 'end': 0.0},),
      ({'capacity': 4},),
  )
  def test_final_logits_points_refuses_malformed_payloads(self, fields):
    """A finite, increasing value range, both ends or neither."""
    with self.assertRaises(ValueError):
      analysis_requests.final_logits_points(
          self.index,
          self._comparison,
          {**_DECODE, 'row': 0, 'transform': 'raw', **fields},
      )

  def test_a_gone_client_stops_the_charts_before_they_read(self):
    """The charts read no vector for a request nobody waits for."""
    gone = analysis_requests.Comparison(
        lambda: self.views.compared('chat', 1, self.index), lambda: True
    )
    row = {**_DECODE, 'row': 0, 'transform': 'raw'}
    for route, payload, made in (
        (
            analysis_requests.final_logits_bins,
            {**row, 'capacity': 4, 'reduction': 'mean'},
            'final_logits_bins',
        ),
        (analysis_requests.final_logits_points, row, 'final_logits_points'),
    ):
      with self.subTest(route.__name__):
        with mock.patch.object(token_analysis, made) as drawing:
          with self.assertRaises(pair_values.AbandonedError):
            route(self.index, gone, payload)
        drawing.assert_not_called()

  def test_token_range_of_the_stages_columns(self):
    """token-range bins the rows' own pairs, row k at column first + k."""
    start = self.compared.comparisons[1]['alignments']['content']['positions'][
        'start'
    ]
    answer = analysis_requests.token_range(
        self.index,
        self._comparison,
        {
            **_DECODE,
            'analysis': 'final_logits',
            'metric': 'rmse',
            'start': start,
            'end': start + 4,
            'bins': 4,
        },
    )

    self.assertEqual(
        {key: answer[key] for key in ('analysis', 'metric', 'start')},
        {'analysis': 'final_logits', 'metric': 'rmse', 'start': start},
    )
    rows = self.compared.comparisons[1]['alignments']['content']['rows']

    def get_step(row, side):
      if row[side] is None:
        return None
      return self.compared.conversation[1, side]['tokens'][row[side]]['step']

    steps = [
        tuple(get_step(row, side) for side in capture_index.SIDES)
        for row in rows
    ]
    pairs = token_analysis.token_analysis(self.index, 1, steps)
    (series,) = answer['rows']
    self.assertEqual(
        [cell['max'] for cell in series['bins']],
        [pair['final_logits']['raw']['rmse']['value'] for pair in pairs],
    )
    self.assertEqual(
        [cell['start'] for cell in series['bins']],
        list(range(start, start + 4)),
    )
    self.assertIn('peak_kl', answer)

  def test_token_range_reads_what_token_analysis_made(self):
    """Pairs token-analysis made are kept for token-range."""
    analysis_requests.token_analysis(
        self.index, self._comparison, {**_DECODE, 'rows': [0, 1, 2, 3]}
    )
    start = self.compared.comparisons[1]['alignments']['content']['positions'][
        'start'
    ]

    with mock.patch.object(
        token_analysis,
        'token_analysis',
        wraps=token_analysis.token_analysis,
    ) as made:
      analysis_requests.token_range(
          self.index,
          self._comparison,
          {
              **_DECODE,
              'analysis': 'sampling',
              'metric': 'entropy',
              'start': start,
              'end': start + 4,
              'bins': 1,
          },
      )

    # Only the first divergence's pair, filled in by step, is new.
    self.assertEqual(made.call_count, 1)
    self.assertEqual(made.call_args.args[2], [(1, 1)])

  def _range(self, **fields):
    """token-range of the whole Decode, by content, unless told."""
    positions = self.compared.comparisons[1]['alignments']['content'][
        'positions'
    ]
    start = positions['start']
    payload = {
        **_DECODE,
        'analysis': 'final_logits',
        'metric': 'kl',
        'start': start,
        'end': start + len(positions['pairs']),
        'bins': 4,
    }
    return analysis_requests.token_range(
        self.index, self._comparison, {**payload, **fields}
    )

  def test_token_analysis_reads_what_token_range_made(self):
    """token-range keeps less than a pair: token-analysis makes it."""
    self._range()
    with mock.patch.object(
        token_analysis,
        'token_analysis',
        wraps=token_analysis.token_analysis,
    ) as made:
      answer = analysis_requests.token_analysis(
          self.index, self._comparison, {**_DECODE, 'rows': [0, 2]}
      )
      self._range(bins=1)

    self.assertEqual([pair['row'] for pair in answer['pairs']], [0, 2])
    # token-analysis made its two pairs again; token-range, none.
    self.assertEqual(made.call_count, 1)
    self.assertEqual(made.call_args.args[2], [(0, 0), (1, 2)])

  def test_token_range_past_the_stage_is_refused(self):
    """Its columns are the stage's rows: no column past them."""
    whole = self._range()
    with self.assertRaisesRegex(ValueError, "stage's columns"):
      self._range(end=whole['end'] + 1)
    with self.assertRaisesRegex(ValueError, "stage's columns"):
      self._range(start=whole['start'] - 1)

  def test_an_unreadable_row_is_missing_from_its_bin(self):
    """A row whose logits cannot be read is left out; the rest is drawn."""
    whole = self._range(bins=4, metric='rmse')
    forward = self.index.turn(1)['sides']['target']['forwards'][-1]
    self.index.path('target', 1, forward['tensors']['post_logits']).unlink()
    self.views = session_view.SessionViews()  # Nothing kept.

    answer = self._range(bins=4, metric='rmse')

    drawn = [cell['valid_count'] for cell in answer['rows'][0]['bins']]
    self.assertEqual(
        drawn,
        [cell['valid_count'] for cell in whole['rows'][0]['bins'][:-1]] + [0],
    )

  @parameterized.parameters(
      ({'analysis': 'kv'},),
      ({'metric': 'entropy'},),
      ({'start': 5, 'end': 5},),
      ({'start': -1},),
      ({'bins': 0},),
      ({'bins': 1025},),
      ({'bins': True},),
      ({'stage': 'prefill'},),
      ({'extra': 1},),
  )
  def test_token_range_refuses_malformed_payloads(self, fields):
    """An analysis and one of its metrics, a range and its bins."""
    with self.assertRaises(ValueError):
      analysis_requests.token_range(
          self.index,
          self._comparison,
          {
              **_DECODE,
              'analysis': 'final_logits',
              'metric': 'kl',
              'start': 0,
              'end': 4,
              'bins': 4,
              **fields,
          },
      )

  def test_match_rate_of_chosen_rows(self):
    """Three of the four content rows are the same."""
    rate = analysis_requests.match_rate(self.index, self._comparison, _DECODE)
    self.assertEqual(rate, {**_DECODE, 'match_rate': 0.75})
    rate = analysis_requests.match_rate(
        self.index, self._comparison, {**_DECODE, 'rows': [0, 2]}
    )
    self.assertEqual(rate['match_rate'], 1.0)
    prefill = analysis_requests.match_rate(
        self.index,
        self._comparison,
        {**_DECODE, 'stage': 'prefill', 'rows': []},
    )
    self.assertIsNone(prefill['match_rate'])

  def test_kv_reads_through_the_stages_positions(self):
    """The KV columns are the rows the comparison published."""
    context = kv_analysis.kv_analysis(self.index, 1)['contexts'][-1]
    query = {
        'turn': '1',
        'context': context['id'],
        'kind': 'key',
        'head': 'max',
        'stage': 'decode',
        'alignment': 'content',
    }
    columns = self.compared.comparisons[1]['alignments']['content']['positions']
    start = columns['start']
    query.update(start=str(start), end=str(start + 4), bins='4')

    answer = analysis_requests.kv_range(self.index, self._comparison, query)

    expected = kv_analysis.kv_range(
        self.index,
        1,
        context['id'],
        start,
        start + 4,
        4,
        'key',
        'max',
        None,
        kv_analysis.Alignment.of(columns),
    )
    self.assertEqual({key: answer[key] for key in expected}, expected)
    self.assertEqual(
        (answer['stage'], answer['alignment']), ('decode', 'content')
    )
    head = analysis_requests.kv_head(
        self.index,
        self._comparison,
        {
            'turn': '1',
            'context_id': context['id'],
            'layer': '0',
            'kind': 'key',
            'position': str(start + 1),
            'stage': 'decode',
            'alignment': 'content',
        },
    )
    self.assertEqual(
        head['selection']['positions'],
        {'ref': None, 'target': columns['pairs'][1][1]},
    )
    self.assertEqual(head['selection']['stage'], 'decode')
    self.assertNotEqual(head['heads'], [])

  @parameterized.parameters(
      ({'turn': 0},),
      ({'turn': True},),
      ({'stage': 'both'},),
      ({'alignment': 'tokens'},),
      ({'stage': 'prefill', 'alignment': 'steps'},),
      ({'rows': [0] * 129},),
      ({'rows': [-1]},),
      ({'rows': [False]},),
      ({'rows': [4]},),
      ({'rows': None},),
      ({'first_divergence': 1},),
      ({'pairs': [{'ref': 0, 'target': 0}]},),
      ({'stage': 'prefill', 'rows': [0]},),
      ({'stage': 'prefill', 'rows': []},),
  )
  def test_token_analysis_refuses_malformed_payloads(self, fields):
    """A request names rows of the comparison, and only that."""
    with self.assertRaises(ValueError):
      analysis_requests.token_analysis(
          self.index, self._comparison, {**_DECODE, 'rows': [0], **fields}
      )

  def test_token_routes_refuse_a_prefill_by_the_stage_rule(self):
    """Each token route says why as token_analysis.stage_reason does."""
    why = token_analysis.stage_reason('prefill')['text']
    row = {'row': 0, 'transform': 'raw'}
    for route, payload in (
        (analysis_requests.token_analysis, {'rows': [0]}),
        (analysis_requests.final_logits, {**row, **_PAGE}),
        (analysis_requests.final_logits_summary, row),
        (
            analysis_requests.final_logits_bins,
            {**row, 'capacity': 4, 'reduction': 'mean'},
        ),
        (analysis_requests.final_logits_points, row),
    ):
      with self.subTest(route.__name__):
        with self.assertRaisesRegex(ValueError, re.escape(why)):
          route(
              self.index,
              self._comparison,
              {**_DECODE, **payload, 'stage': 'prefill'},
          )

  def test_an_uncaptured_turn_is_not_found(self):
    """As index.turn says."""
    with self.assertRaises(errors.NotFound):
      analysis_requests.token_analysis(
          self.index, self._comparison, {**_DECODE, 'turn': 2, 'rows': []}
      )

  def _kv_queries(self, n: int) -> list[tuple[Any, ...]]:
    """A kv-range and a kv-head query of turn `n`, through its Decode."""
    context = kv_analysis.kv_analysis(self.index, 1)['contexts'][0]['id']
    aligned = {'turn': str(n), 'stage': 'decode', 'alignment': 'content'}
    return [
        (
            analysis_requests.kv_range,
            {**aligned, 'context': context, 'start': '0', 'end': '1'},
        ),
        (
            analysis_requests.kv_head,
            {
                **aligned,
                'context_id': context,
                'layer': '0',
                'kind': 'key',
                'position': '0',
            },
        ),
    ]

  def test_an_aligned_kv_read_of_an_uncaptured_turn_is_not_found(self):
    """The turn is looked up before the stage's rows."""
    for read, query in self._kv_queries(99):
      with self.assertRaisesRegex(errors.NotFound, 'Turn 99 has no capture.'):
        read(self.index, self._comparison, query)

  def test_an_aligned_kv_read_of_a_rejected_turn_is_a_conflict(self):
    """It says why the turn was rejected, as index.turn does."""
    completions = {
        side: fixtures.write_turn(
            self.index.turn_directory(side, 2),
            chat.send('More', ['c'], True),
        )
        for side, chat in self.chats.items()
    }
    completions['target']['usage']['completion_tokens'] += 1
    with self.assertRaises(capture_index.Rejected):
      self.index.add_turn(2, 'More', completions)

    for read, query in self._kv_queries(2):
      with self.assertRaisesRegex(errors.Conflict, 'rejected turn 2'):
        read(self.index, self._comparison, query)

  def test_kv_columns_are_made_once_per_view(self):
    """A stage's alignment is made once for the comparison it is of."""
    context = kv_analysis.kv_analysis(self.index, 1)['contexts'][-1]['id']
    start = self.compared.comparisons[1]['alignments']['content']['positions'][
        'start'
    ]
    query = {
        'turn': '1',
        'context': context,
        'start': str(start),
        'end': str(start + 2),
        'stage': 'decode',
        'alignment': 'content',
    }

    with mock.patch.object(
        kv_analysis.Alignment, 'of', wraps=kv_analysis.Alignment.of
    ) as made:
      first = analysis_requests.kv_range(self.index, self._comparison, query)
      again = analysis_requests.kv_range(self.index, self._comparison, query)
      analysis_requests.kv_range(
          self.index, self._comparison, {**query, 'alignment': 'steps'}
      )

    self.assertEqual(again, first)
    self.assertEqual(made.call_count, 2)

  @parameterized.parameters(
      ({'limit': 0},),
      ({'limit': 257},),
      ({'offset': -1},),
      ({'order': 'probability'},),
      ({'direction': 'up'},),
      ({'transform': 'softmax'},),
      ({'row': 9},),
      ({'ref_step': 0},),
      ({'extra': 1},),
  )
  def test_final_logits_refuses_malformed_payloads(self, fields):
    """A row, its page, and nothing else."""
    with self.assertRaises(ValueError):
      analysis_requests.final_logits(
          self.index,
          self._comparison,
          {**_DECODE, 'row': 0, **_PAGE, **fields},
      )

  def test_match_rate_refuses_repeated_rows(self):
    """Each row counts once."""
    with self.assertRaisesRegex(ValueError, 'once'):
      analysis_requests.match_rate(
          self.index, self._comparison, {**_DECODE, 'rows': [0, 0]}
      )

  def test_metadata_needs_its_mode(self):
    """Only mode=metadata, of a turn by number."""
    with self.assertRaises(ValueError):
      analysis_requests.kv_metadata(self.index, {'turn': '1'})
    with self.assertRaises(ValueError):
      analysis_requests.kv_metadata(
          self.index, {'mode': 'metadata', 'turn': 'x'}
      )

  def test_kv_range_refuses_bad_requests(self):
    """Malformed ranges, and an alignment named by halves."""
    context = kv_analysis.kv_analysis(self.index, 1)['contexts'][0]['id']
    for values in (
        {'start': '2', 'end': '2'},
        {'bins': '0'},
        {'bins': '1025'},
        {'kind': 'both'},
        {'head': 'median'},
        {'head': '1', 'heads': '[0]'},
        {'heads': '[0, 0]'},
        {'layers': '[0, 0]'},
        {'layers': '{"0": 1}'},
        {'formula': 'max_abs > 1'},
        {'stage': 'decode'},
        {'stage': 'prefill', 'alignment': 'steps'},
    ):
      query = {'turn': '1', 'context': context, 'start': '0', 'end': '5'}
      with self.assertRaises(ValueError, msg=values):
        analysis_requests.kv_range(
            self.index, self._comparison, {**query, **values}
        )


class TensorRequestsTest(parameterized.TestCase):
  """The intermediate tensors' routes, of a chat that observes tensors."""

  def setUp(self):
    """One turn whose sides observe the tiny model's points."""
    super().setUp()
    root = pathlib.Path(self.create_tempdir().full_path)
    self.index = fixtures.capture(
        root, [('Hi there', ['a', 'b', 'c'], True)], observe=True
    )
    sides = {
        side: tensor_analysis.Side('ab' * 32, fixtures.TAP_POINTS)
        for side in capture_index.SIDES
    }
    views = session_view.SessionViews(
        observed=lambda record_id: tensor_analysis.Observed(sides)
    )
    self.comparison = analysis_requests.Comparison(
        lambda: views.compared('chat', 1, self.index)
    )
    self.view = views.compared('chat', 1, self.index)

  def test_analysis_echoes_the_stage_and_reads_its_columns(self):
    """With an alignment, the stage's rows; without, its positions."""
    aligned = analysis_requests.position_analysis(
        self.index,
        self.comparison,
        {'turn': '1', 'stage': 'decode', 'alignment': 'content'},
    )
    columns = self.view.columns(1, 'decode', 'content')
    self.assertEqual(
        (aligned['turn'], aligned['stage'], aligned['alignment']),
        (1, 'decode', 'content'),
    )
    self.assertEqual(
        (aligned['start'], aligned['end']), (columns.start, columns.end())
    )
    plain = analysis_requests.position_analysis(
        self.index, self.comparison, {'turn': '1', 'stage': 'prefill'}
    )
    self.assertIsNone(plain['alignment'])
    # The first input token's column, which no step produced, is not one.
    self.assertEqual(plain['start'], 1)
    self.assertEqual(plain['layers'], [0, 1])

  def test_range_bins_and_tensor_inspects(self):
    """position-range's bins and totals; position-tensor's elements."""
    query = {'turn': '1', 'stage': 'prefill', 'alignment': 'content'}
    found = self.view.tensors(1, 'prefill', 'content')
    answer = analysis_requests.position_range(
        self.index,
        self.comparison,
        {
            **query,
            'start': str(found.start),
            'end': str(found.end),
            'bins': '3',
            'anchors': '["output"]',
            'reduction': 'min',
            'layers': '[1]',
        },
    )
    self.assertEqual(answer['layers'], [1])
    self.assertEqual(answer['anchors'], ['output'])
    self.assertLen(answer['bins'], 3)
    self.assertLen(answer['totals'][0]['bins'], 1)
    last = found.end - 1
    tensor = analysis_requests.position_tensor(
        self.index,
        self.comparison,
        {**query, 'position': str(last), 'layer': '1', 'anchor': 'output'},
    )
    self.assertEqual(
        (tensor['position'], tensor['layer'], tensor['anchor']),
        (last, 1, 'output'),
    )
    self.assertLen(tensor['channels'], fixtures.HIDDEN)
    # The prefill chunk's row that fed the token before produced it.
    self.assertEqual(tensor['reference']['point'], f'{fixtures.SIGNATURE}:7:0')
    self.assertEqual(tensor['reference']['reads'], last - 1)

  @parameterized.parameters(
      {'start': '0', 'end': '0'},
      {'bins': '0'},
      {'bins': '1025'},
      {'reduction': 'median'},
      {'anchors': '[]'},
      {'anchors': '["output", "output"]'},
      {'layers': '[0, 0]'},
      {'end': '1000'},
      {'stage': 'prefill', 'alignment': 'steps'},
      {'stage': 'both'},
  )
  def test_range_refuses_bad_requests(self, **values):
    """Malformed ranges, anchors, layers and stages."""
    query = {
        'turn': '1',
        'stage': 'decode',
        'alignment': 'content',
        'start': '0',
        'end': '1',
    }
    found = self.view.tensors(1, 'decode', 'content')
    query.update(start=str(found.start), end=str(found.start + 1))
    with self.assertRaises(ValueError):
      analysis_requests.position_range(
          self.index, self.comparison, {**query, **values}
      )

  def test_tensor_of_no_cell_is_refused(self):
    """A column out of the stage, or a place it has none of."""
    query = {'turn': '1', 'stage': 'prefill', 'layer': '1'}
    for position, anchor in (('1000', 'decode:5:0'), ('0', 'nope')):
      with self.assertRaisesRegex(ValueError, 'no tensor of layer'):
        analysis_requests.position_tensor(
            self.index,
            self.comparison,
            {**query, 'position': position, 'anchor': anchor},
        )

  def test_a_column_no_step_produced_is_not_the_stages(self):
    """No step produced the first input token: no column, and why."""
    query = {'turn': '1', 'stage': 'prefill'}
    with self.assertRaisesRegex(ValueError, 'no tensor of layer'):
      analysis_requests.position_tensor(
          self.index,
          self.comparison,
          {**query, 'position': '0', 'layer': '1', 'anchor': 'output'},
      )
    answer = analysis_requests.position_analysis(
        self.index, self.comparison, query
    )
    self.assertEqual(
        [r['code'] for r in answer['omitted']],
        [tensor_analysis.POSITION_NOT_PRODUCED],
    )

  def test_one_tensor_never_compares_the_stage(self):
    """position-tensor compares its one cell, once while kept."""
    query = {
        'turn': '1',
        'stage': 'decode',
        'position': str(self.view.columns(1, 'decode', 'content').start),
        'layer': '0',
        'anchor': 'output',
    }
    real = tensor_analysis.cell
    with (
        mock.patch.object(tensor_analysis, 'stages') as stages,
        mock.patch.object(tensor_analysis, 'cell', side_effect=real) as cell,
    ):
      first = analysis_requests.position_tensor(
          self.index, self.comparison, query
      )
      again = analysis_requests.position_tensor(
          self.index, self.comparison, query
      )
    stages.assert_not_called()
    self.assertEqual(cell.call_count, 1)
    self.assertEqual(first, again)
    self.assertTrue(first['comparable'])

  @parameterized.parameters(
      ({'anchors': '["nope"]'}, 'no sync point nope; it observes output, '),
      ({'layers': '[0, 7]'}, 'no layer 7; it observes 0, 1.'),
  )
  def test_range_refuses_what_the_session_does_not_observe(self, values, words):
    """An anchor or a layer of no place: 400, naming what there is."""
    found = self.view.tensors(1, 'decode', None)
    query = {
        'turn': '1',
        'stage': 'decode',
        'start': str(found.start),
        'end': str(found.end),
    }
    with self.assertRaisesRegex(ValueError, words):
      analysis_requests.position_range(
          self.index, self.comparison, {**query, **values}
      )

  def test_a_session_observing_nothing_has_no_tensor_to_inspect(self):
    """position-tensor: a conflict, in the reason's words."""
    views = session_view.SessionViews(
        observed=lambda record_id: tensor_analysis.nothing()
    )
    comparison = analysis_requests.Comparison(
        lambda: views.compared('chat', 1, self.index)
    )
    with self.assertRaisesRegex(
        errors.Conflict, 'This Session observes no tensors.'
    ):
      analysis_requests.position_tensor(
          self.index,
          comparison,
          {
              'turn': '1',
              'stage': 'decode',
              'position': '0',
              'layer': '0',
              'anchor': 'output',
          },
      )


if __name__ == '__main__':
  test_main.main()
