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

"""128k captured-file acceptance checks for bounded KV ranges and Find."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from inline_asgi import client
from model_explorer_debugger import kv_range
from model_explorer_debugger.kv_formula import compile_kv_formula
from model_explorer_debugger.kv_head import inspect_kv_head
from model_explorer_debugger.kv_reader import KvTensor, MAX_SLICE_ELEMENTS
from model_explorer_debugger.metrics import compare
from model_explorer_debugger.store import SessionStore
import numpy as np
import test_kv_head as head_fixtures
from test_kv_reader import SliceOnly


class KvFormulaTests(unittest.TestCase):

  def setUp(self):
    self.metrics = dict(
        relative_l2=np.array([[np.nan, 0.0, 0.5, 1.0, 2.0]]),
        max_abs=np.array([[np.nan, 0.0, 2.0, 4.0, 8.0]]),
        cosine_distance=np.array([[np.nan, 0.0, 0.1, 0.2, 0.4]]),
    )

  def matches(self, formula):
    return compile_kv_formula(formula)(self.metrics)[0].tolist()

  def test_percent_alias_and_operator_precedence(self):
    self.assertEqual(
        self.matches('relative_l2 > 100%'), [False, False, False, False, True]
    )
    self.assertEqual(
        self.matches('max_abs_delta >= 4'), [False, False, False, True, True]
    )
    self.assertEqual(
        self.matches('relative_l2 = 0 OR relative_l2 >= 50% AND max_abs >= 4'),
        [False, True, False, True, True],
    )
    self.assertEqual(
        self.matches(
            '(relative_l2 = 0 OR relative_l2 >= 50%) AND max_abs >= 4'
        ),
        [False, False, False, True, True],
    )

  def test_unknown_survives_not_and_boolean_combinations(self):
    self.assertEqual(
        self.matches('NOT (relative_l2 > 100%)'),
        [False, True, True, True, False],
    )
    self.assertEqual(
        self.matches('NOT NOT (relative_l2 > 100%)'),
        [False, False, False, False, True],
    )
    self.assertEqual(self.matches('relative_l2 > 100% OR true'), [True] * 5)
    self.assertEqual(
        self.matches('NOT (relative_l2 > 100% AND false)'), [True] * 5
    )
    self.assertEqual(
        self.matches('NOT (relative_l2 > 100% OR false)'),
        [False, True, True, True, False],
    )
    self.assertEqual(self.matches('false'), [False] * 5)

  def test_invalid_formula_and_budgets_fail_before_evaluation(self):
    for text in (
        'rmse > 1',
        'max_abs > 10%',
        'relative_l2 > true',
        'relative_l2',
        'relative_l2 > 1e999',
        'true > false',
        'relative_l2 > 1; true',
        '(' * 33 + 'true' + ')' * 33,
        'x' * 1001,
        ' OR '.join(['true'] * 102),
    ):
      with self.subTest(formula=text[:60]), self.assertRaises(ValueError):
        compile_kv_formula(text)


class KvNumericalSemanticsTests(unittest.TestCase):

  def test_overflow_is_unavailable_in_compare_head_and_range(self):
    # BLAS norms can produce inf without raising an IEEE overflow warning.
    # Identical huge vectors must not then acquire a false cosine of zero.
    values = np.full((1, 2, 1, 4), 1e200, dtype=np.float64)
    with self.assertRaisesRegex(ValueError, '^numeric_overflow$'):
      compare(values[0, 0, 0], values[0, 0, 0])
    fixture = head_fixtures.KvHeadTests()
    self.addCleanup(fixture.doCleanups)
    store, selection = fixture.array_store(values, values.copy())
    store.tensors = []
    head = inspect_kv_head(store, **selection)
    token = kv_range.selection_range(
        store, 1, selection['context_id'], 10, 11, layer=0
    )['rows'][0]['token_metrics'][0]
    for result in (head, token):
      self.assertTrue(
          all(value is None for value in result['metrics'].values())
      )
      self.assertTrue(
          all(
              status == 'numeric_overflow'
              for status in result['metric_status'].values()
          )
      )
      json.dumps(result, allow_nan=False)


class KvRangeTests(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.temporary = tempfile.TemporaryDirectory()
    cls.root = Path(cls.temporary.name) / 'capture'
    script = Path(__file__).resolve().parent / 'fixtures/kv_128k_fixture.py'
    spec = importlib.util.spec_from_file_location('kv_128k_fixture', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls.manifest = module.create_fixture(cls.root)

  @classmethod
  def tearDownClass(cls):
    cls.temporary.cleanup()

  def setUp(self):
    self.store = SessionStore(self.root)
    self.context = kv_range.metadata(self.store, 1)['contexts'][0]['id']

  def row(self, reply, layer, kind='key'):
    return next(
        row
        for row in reply['rows']
        if row['layer'] == layer and row['kind'] == kind
    )

  def check_work(self, reply):
    self.assertLessEqual(reply['work']['max_side_elements'], MAX_SLICE_ELEMENTS)
    self.assertEqual(reply['work']['slice_element_limit'], MAX_SLICE_ELEMENTS)

  def test_metadata_covers_all_128k_positions_without_numerical_cells(self):
    with patch(
        'model_explorer_debugger.kv_reader.safe_open',
        side_effect=AssertionError('metadata payload read'),
    ):
      reply = kv_range.metadata(self.store, 1)
    context = reply['contexts'][0]
    self.assertEqual(len(context['layers']), 6)
    for row in context['layers']:
      self.assertEqual(
          (row['position_start'], row['position_count'], row['head_count']),
          (0, 131072, 4),
      )
      self.assertEqual(row['cells'], [])
      self.assertEqual(row['token_metrics'], [])
    missing = next(
        row
        for row in context['layers']
        if row['layer'] == 2 and row['kind'] == 'value'
    )
    self.assertEqual(missing['status'], 'unavailable')
    self.assertIsNotNone(missing['token_structure']['ref'])
    self.assertIsNone(missing['token_structure']['target'])

  def test_128_bins_keep_spike_value_original_position_and_all_counts(self):
    SliceOnly.reads.clear()
    with (
        patch('model_explorer_debugger.kv_reader.safe_open', SliceOnly),
        patch.object(
            self.store, 'load_resource', side_effect=AssertionError('full read')
        ),
    ):
      reply = kv_range.chart_range(
          self.store,
          1,
          self.context,
          0,
          131072,
          bins=128,
          kind='key',
          head='max',
      )
    self.assertEqual(len(reply['bins']), 128)
    self.assertEqual(reply['bins'][64], dict(start=65536, end=66560))
    row = self.row(reply, 0)
    spike = row['bins'][64]
    self.assertEqual(spike['compared_count'], 1024)
    self.assertEqual(
        spike['metrics']['relative_l2'],
        dict(
            min=0.0,
            max=8.0,
            min_position=65536,
            max_position=65537,
            valid_count=1024,
        ),
    )
    self.assertEqual(spike['metrics']['max_abs']['max'], 16.0)
    self.assertEqual(
        sum(cell['compared_count'] for cell in row['bins']), 131072
    )
    self.assertEqual(
        self.row(reply, 2)['bins'][0]['metrics']['relative_l2']['max_position'],
        17,
    )
    self.assertEqual(self.row(reply, 1)['status'], 'partial')
    self.check_work(reply)
    self.assertTrue(SliceOnly.reads)
    self.assertTrue(
        all(index[2].stop - index[2].start <= 4096 for index in SliceOnly.reads)
    )
    self.assertLess(len(json.dumps(reply)), 400_000)

  def test_single_position_zoom_and_head_modes_use_original_coordinates(self):
    for head, expected in (('max', 8.0), ('mean', 2.0), ('2', 8.0), ('0', 0.0)):
      with self.subTest(head=head):
        reply = kv_range.chart_range(
            self.store, 1, self.context, 65537, 65538, bins=128, head=head
        )
        self.assertEqual(reply['bins'], [dict(start=65537, end=65538)])
        stat = self.row(reply, 0)['bins'][0]['metrics']['relative_l2']
        self.assertEqual(
            (
                stat['min'],
                stat['max'],
                stat['min_position'],
                stat['max_position'],
                stat['valid_count'],
            ),
            (expected, expected, 65537, 65537, 1),
        )
        self.check_work(reply)

  def test_dense_bins_above_128_preserve_tail_spike_and_complete_coverage(self):
    self.assertEqual(
        kv_range.metadata(self.store, 1)['limits']['max_bins'], 1024
    )
    for bins in (129, 257, 1024):
      with self.subTest(bins=bins):
        reply = kv_range.chart_range(
            self.store,
            1,
            self.context,
            0,
            131072,
            bins=bins,
            kind='value',
            head='max',
        )
        self.assertEqual(len(reply['bins']), bins)
        self.assertEqual(reply['bins'][0]['start'], 0)
        self.assertEqual(reply['bins'][-1]['end'], 131072)
        self.assertTrue(
            all(
                left['end'] == right['start']
                for left, right in zip(reply['bins'], reply['bins'][1:])
            )
        )
        row = self.row(reply, 1, 'value')
        self.assertEqual(
            sum(cell['compared_count'] for cell in row['bins']), 131072
        )
        tail = row['bins'][-1]
        self.assertEqual(tail['compared_count'], tail['end'] - tail['start'])
        self.assertEqual(tail['metrics']['relative_l2']['max'], 16.0)
        self.assertEqual(tail['metrics']['relative_l2']['max_position'], 131071)
        self.assertEqual(tail['metrics']['max_abs']['max'], 32.0)
        self.assertEqual(tail['metrics']['max_abs']['max_position'], 131071)
        self.assertEqual(
            sum(
                cell['metrics']['relative_l2']['valid_count']
                for cell in row['bins']
            ),
            131072,
        )
        self.check_work(reply)

  def test_1024_bins_on_dense_zoom_preserve_half_open_tail_and_matches(self):
    reply = kv_range.chart_range(
        self.store,
        1,
        self.context,
        126976,
        131072,
        bins=1024,
        kind='value',
        head='max',
        formula='relative_l2 > 100%',
    )
    self.assertEqual(len(reply['bins']), 1024)
    self.assertEqual(reply['bins'][-1], dict(start=131068, end=131072))
    row = self.row(reply, 1, 'value')
    self.assertTrue(all(cell['compared_count'] == 4 for cell in row['bins']))
    self.assertEqual(sum(cell['match_count'] for cell in row['bins']), 1)
    self.assertEqual(
        row['bins'][-1]['metrics']['relative_l2']['max_position'], 131071
    )
    self.assertEqual(
        row['bins'][-1]['metrics']['relative_l2']['valid_count'], 4
    )
    for bins in (0, -1, 1025, 1.5):
      with (
          self.subTest(bins=bins),
          patch.object(
              KvTensor,
              'read',
              side_effect=AssertionError('invalid bin count read'),
          ),
          self.assertRaises(ValueError),
      ):
        kv_range.chart_range(self.store, 1, self.context, 0, 131072, bins=bins)

  def test_full_range_one_bin_preserves_color_scale_extrema_and_cache(self):
    options = dict(bins=1, kind='value', head='max')
    baseline = kv_range.chart_range(
        self.store, 1, self.context, 0, 131072, **options
    )
    dense = kv_range.chart_range(
        self.store,
        1,
        self.context,
        0,
        131072,
        bins=1024,
        kind='value',
        head='max',
    )
    expected = compare(np.ones(4), np.array([33.0, 1.0, 1.0, 1.0]))
    expected_maxima = dict(
        relative_l2=16.0,
        max_abs=32.0,
        cosine_distance=1 - expected['CosSim']['value'],
    )
    full = self.row(baseline, 1, 'value')['bins'][0]
    for metric, expected_maximum in expected_maxima.items():
      self.assertAlmostEqual(full['metrics'][metric]['max'], expected_maximum)
      self.assertEqual(full['metrics'][metric]['max_position'], 131071)
      self.assertEqual(full['metrics'][metric]['valid_count'], 131072)
      finite = [
          cell['metrics'][metric]
          for cell in self.row(dense, 1, 'value')['bins']
          if cell['metrics'][metric]['valid_count']
      ]
      self.assertEqual(
          full['metrics'][metric]['max'], max(stat['max'] for stat in finite)
      )
      self.assertEqual(
          full['metrics'][metric]['valid_count'],
          sum(stat['valid_count'] for stat in finite),
      )
    missing = self.row(baseline, 2, 'value')['bins'][0]
    self.assertTrue(
        all(
            stat['max'] is None and stat['valid_count'] == 0
            for stat in missing['metrics'].values()
        )
    )
    with patch.object(
        KvTensor,
        'read',
        side_effect=AssertionError('full-range baseline cache missed'),
    ):
      cached = kv_range.chart_range(
          self.store, 1, self.context, 0, 131072, **options
      )
    self.assertEqual(cached['rows'], baseline['rows'])
    self.assertTrue(cached['cache_hit'])
    self.assertEqual(cached['work']['read_chunks'], 0)

  def test_selected_token_metrics_compare_all_head_vectors_directly(self):
    reply = kv_range.selection_range(
        self.store, 1, self.context, 65537, 65538, layer=0
    )
    row = self.row(reply, 0)
    expected_ref = np.ones((4, 4))
    expected_target = expected_ref.copy()
    expected_target[2, 1] += 16
    expected = compare(expected_ref, expected_target)
    self.assertEqual(row['compared_count'], 1)
    for key, label in (
        ('relative_l2', 'Relative L2'),
        ('rmse', 'RMSE'),
        ('max_abs', 'Max abs error'),
        ('cosine_similarity', 'CosSim'),
    ):
      self.assertAlmostEqual(
          row['metrics'][key]['value'], expected[label]['value']
      )
      self.assertEqual(row['metrics'][key]['position'], 65537)
    self.assertEqual(row['metrics']['relative_l2']['value'], 4.0)
    self.assertEqual(row['token_metrics'][0]['position'], 65537)
    self.assertEqual(row['token_metrics'][0]['metrics']['relative_l2'], 4.0)
    self.assertEqual(len(reply['rows']), 2)

  def test_column_summary_keeps_layers_and_kv_independent(self):
    reply = kv_range.selection_range(self.store, 1, self.context, 0, 131072)
    self.assertEqual(len(reply['rows']), 6)
    for layer, kind, expected, position in (
        (0, 'key', 4.0, 65537),
        (1, 'value', 8.0, 131071),
        (2, 'key', 2.0, 17),
    ):
      row = self.row(reply, layer, kind)
      self.assertEqual(
          (
              row['metrics']['relative_l2']['value'],
              row['metrics']['relative_l2']['position'],
          ),
          (expected, position),
      )
    self.assertEqual(
        self.row(reply, 0, 'value')['metrics']['relative_l2']['valid_count'],
        131071,
    )
    self.assertEqual(
        self.row(reply, 1, 'key')['metrics']['relative_l2']['valid_count'],
        131071,
    )
    self.assertEqual(self.row(reply, 1, 'key')['status'], 'partial')
    missing = self.row(reply, 2, 'value')
    self.assertEqual(missing['compared_count'], 0)
    self.assertTrue(
        all(metric['value'] is None for metric in missing['metrics'].values())
    )
    self.check_work(reply)

  def test_zero_norm_and_nan_have_distinct_truthful_metric_status(self):
    zero = self.row(
        kv_range.selection_range(
            self.store, 1, self.context, 2048, 2049, layer=0
        ),
        0,
        'value',
    )['token_metrics'][0]
    self.assertEqual(
        zero['metric_status']['relative_l2'], 'undefined_zero_norm'
    )
    self.assertIsNone(zero['metrics']['cosine_similarity'])
    self.assertEqual(zero['metrics']['rmse'], 0.0)
    nan = self.row(
        kv_range.selection_range(
            self.store, 1, self.context, 4096, 4097, layer=1
        ),
        1,
    )['token_metrics'][0]
    self.assertEqual(nan['metric_status']['relative_l2'], 'non_finite_tensor')
    self.assertTrue(all(value is None for value in nan['metrics'].values()))
    json.dumps(nan, allow_nan=False)

  def test_find_scans_global_positions_and_returns_40_plus_23(self):
    first = kv_range.find_matches(
        self.store, 1, self.context, 'relative_l2 > 100%', offset=0, limit=40
    )
    second = kv_range.find_matches(
        self.store, 1, self.context, 'relative_l2 > 100%', offset=40, limit=40
    )
    self.assertEqual(
        (first['total'], len(first['results']), first['has_more']),
        (63, 40, True),
    )
    self.assertEqual(
        (second['total'], len(second['results']), second['has_more']),
        (63, 23, False),
    )
    expected = sorted(
        (p['position'], p['layer'], p['kind'], p['head'])
        for p in self.manifest['spikes']
    )
    results = first['results'] + second['results']
    actual = [
        (row['position'], row['layer'], row['kind'], row['heads'][0])
        for row in results
    ]
    self.assertEqual(actual, expected)
    self.assertEqual(results[-1]['position'], 131071)
    self.assertFalse(first['complete'])
    self.assertIn(
        (2, 'value'),
        [(row['layer'], row['kind']) for row in first['unavailable_rows']],
    )
    self.check_work(first)
    self.check_work(second)
    self.assertLess(second['work']['read_chunks'], first['work']['read_chunks'])

  def test_find_beyond_last_offset_and_reply_cache_are_bounded(self):
    first = kv_range.find_matches(
        self.store, 1, self.context, 'relative_l2 > 100%', offset=0
    )
    with patch.object(
        KvTensor, 'read', side_effect=AssertionError('cached request read')
    ):
      again = kv_range.find_matches(
          self.store, 1, self.context, 'relative_l2 > 100%', offset=0
      )
    self.assertEqual(first['results'], again['results'])
    self.assertEqual(again['work']['read_chunks'], 0)
    end = kv_range.find_matches(
        self.store, 1, self.context, 'relative_l2 > 100%', offset=1000
    )
    self.assertEqual(
        (end['total'], end['results'], end['has_more']), (63, [], False)
    )

  def test_range_counts_find_matches_independently_from_display_head(self):
    reply = kv_range.chart_range(
        self.store,
        1,
        self.context,
        65536,
        65539,
        bins=1,
        head='0',
        formula='relative_l2 > 100%',
    )
    cell = self.row(reply, 0)['bins'][0]
    self.assertEqual(cell['metrics']['relative_l2']['max'], 0.0)
    self.assertEqual(cell['match_count'], 1)
    self.assertEqual(cell['compared_count'], 3)

  def test_nonzero_logical_origin_and_outside_ranges_do_not_rebase(self):
    for record in self.store._telemetry['resources']:
      if record.get('scope') == 'kv':
        record.update(
            logical_start=100, logical_end=131172, processed_token_count=131172
        )
    context = kv_range.metadata(self.store, 1)['contexts'][0]
    self.assertEqual(context['layers'][0]['position_start'], 100)
    reply = kv_range.chart_range(self.store, 1, self.context, 99, 102, bins=3)
    self.assertEqual(
        [cell['compared_count'] for cell in self.row(reply, 0)['bins']],
        [0, 1, 1],
    )
    self.assertIsNone(
        self.row(reply, 0)['bins'][0]['metrics']['relative_l2']['max']
    )
    outside = kv_range.chart_range(
        self.store, 1, self.context, 131172, 131175, bins=3
    )
    self.assertTrue(
        all(row['status'] == 'position_unavailable' for row in outside['rows'])
    )

  def test_http_routes_validate_arguments_and_sessions_listing_survives(self):
    with client(self.store) as http:

      def get(path, **query):
        reply = http.get(path + ('?' + urlencode(query) if query else ''))
        self.assertEqual(reply.status_code, 200, reply.text)
        return reply.json()

      listing = get('/api/sessions')
      self.assertIn('sessions', listing)
      get('/api/kv-analysis', turn=1, mode='metadata')
      query = dict(turn=1, context=self.context, start=65537, end=65538)
      self.assertEqual(
          get('/api/kv-range', **query)['bins'], [dict(start=65537, end=65538)]
      )
      self.assertEqual(
          self.row(get('/api/kv-selection', **query), 0)['metrics'][
              'relative_l2'
          ]['value'],
          4.0,
      )
      self.assertEqual(
          get(
              '/api/kv-find',
              turn=1,
              context=self.context,
              formula='relative_l2 > 100%',
              limit=1,
          )['total'],
          63,
      )
      self.assertEqual(get('/api/sessions'), listing)
      self.assertEqual(
          get('/api/kv-range', **query, bins=1024)['bins'],
          [dict(start=65537, end=65538)],
      )
      invalid = [
          ('/api/kv-range', dict(query, bins=1025)),
          ('/api/kv-range', dict(query, end=65537)),
          ('/api/kv-range', dict(query, context='missing')),
          ('/api/kv-range', dict(query, head='bad')),
          ('/api/kv-selection', dict(query, layer=-1)),
          ('/api/kv-find', dict(turn=1, context=self.context, limit=41)),
          ('/api/kv-find', dict(turn=1, context=self.context, offset=-1)),
          (
              '/api/kv-find',
              dict(turn=1, context=self.context, formula='max_abs > 10%'),
          ),
      ]
      for path, params in invalid:
        with self.subTest(path=path, params=params):
          reply = http.get(path + '?' + urlencode(params))
          self.assertEqual(reply.status_code, 400)
          self.assertIn('error', reply.json())


if __name__ == '__main__':
  unittest.main()
