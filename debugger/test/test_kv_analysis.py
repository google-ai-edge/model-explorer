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

"""Exercise truthful cache-position analysis through saved evidence and HTTP."""

from copy import deepcopy
import json
import math
import unittest
from unittest.mock import patch

from inline_asgi import client
from model_explorer_debugger.capture_telemetry import empty_telemetry
from model_explorer_debugger.cross_runtime_pairs import compare_pair
from model_explorer_debugger.kv_analysis import analyze_kv
from model_explorer_debugger.store import SessionStore
import numpy as np
import test_cross_runtime_kv as explicit_fixtures
import test_pytorch_telemetry as telemetry_fixtures


class KvAnalysisTests(unittest.TestCase):

  def snapshot_store(self, divergent=False):
    fixture = telemetry_fixtures.PyTorchTelemetryTests()
    fixture.setUp()
    self.addCleanup(fixture.doCleanups)
    job, results = fixture.fixture(divergent=divergent)
    return fixture.publish(job, results)

  def terminal(self, result):
    return next(
        context
        for context in result['contexts']
        if context['moment'] == 'terminal'
    )

  def test_saved_positions_keep_forward_context_and_metric_units(self):
    store = self.snapshot_store()
    result = analyze_kv(store, 1)
    self.assertEqual(len(result['contexts']), 3)
    terminal = self.terminal(result)
    self.assertEqual(
        (
            terminal['turn'],
            terminal['phase'],
            terminal['step'],
            terminal['forward_id'],
        ),
        (1, 'decode', 1, 1),
    )
    self.assertEqual(
        [row['kind'] for row in terminal['layers']], ['key', 'value']
    )
    for role in ('ref', 'target'):
      recorded = next(
          snapshot
          for snapshot in store.telemetry(turn=1)['kv_snapshots']
          if snapshot['moment'] == 'terminal' and snapshot['run'] == role
      )
      self.assertEqual(terminal['snapshots'][role][0]['id'], recorded['id'])
      self.assertEqual(
          terminal['snapshots'][role][0]['snapshot_id'], recorded['snapshot_id']
      )
    self.assertNotEqual(
        terminal['snapshots']['ref'][0]['id'],
        terminal['snapshots']['target'][0]['id'],
    )
    row = terminal['layers'][0]
    self.assertEqual(
        (row['status'], row['head_count'], row['position_count']), ('ok', 1, 3)
    )
    self.assertEqual([cell['position'] for cell in row['cells']], [0, 1, 2])
    self.assertEqual(row['metrics']['relative_l2'], 1.0)
    self.assertEqual(
        set(row['summary_metrics']),
        {'CosSim', 'Max abs error', 'Mean abs error', 'RMSE', 'Relative L2'},
    )
    self.assertEqual(row['summary_metrics']['RMSE']['value'], 1.0)
    for side in row['token_structure'].values():
      self.assertEqual(
          side,
          dict(
              status='ok',
              shape=[1, 2],
              head_count=1,
              channel_count=2,
              batch_count=1,
              position_start=0,
              position_count=3,
              dtype='float32',
          ),
      )
    for cell in row['cells']:
      self.assertEqual(cell['metrics']['relative_l2'], 1.0)
      self.assertEqual(cell['metrics']['max_abs'], 1.0)
      self.assertAlmostEqual(cell['metrics']['cosine_distance'], 0.0)
    self.assertEqual(
        [token['position'] for token in row['token_metrics']], [0, 1, 2]
    )
    for token in row['token_metrics']:
      self.assertEqual(token['status'], 'ok')
      self.assertAlmostEqual(token['metrics']['cosine_similarity'], 1.0)
      self.assertEqual(
          {
              key: token['metrics'][key]
              for key in ('relative_l2', 'rmse', 'max_abs')
          },
          {'relative_l2': 1.0, 'rmse': 1.0, 'max_abs': 1.0},
      )
      self.assertEqual(set(token['metric_status'].values()), {'ok'})

  def test_snapshot_token_metrics_preserve_nonzero_logical_offset(self):
    store = self.snapshot_store()
    for resource in store._resources.values():
      if resource.get('scope') == 'kv' and resource.get('moment') == 'terminal':
        resource.update(logical_start=10, logical_end=13)
    for row in self.terminal(analyze_kv(store, 1))['layers']:
      self.assertEqual(
          [token['position'] for token in row['token_metrics']], [10, 11, 12]
      )
      self.assertEqual(
          [token['metrics']['rmse'] for token in row['token_metrics']],
          [1.0] * 3,
      )

  def test_different_input_context_preserves_ids_without_metric_cells(self):
    store = self.snapshot_store(divergent=True)
    terminal = self.terminal(analyze_kv(store, 1))
    for row in terminal['layers']:
      self.assertEqual(row['status'], 'sample_mismatch')
      self.assertTrue(row['reference'] and row['target'])
      self.assertEqual(row['cells'], [])
      self.assertEqual(row['token_metrics'], [])
      self.assertTrue(
          all(
              side['status'] == 'ok' and side['shape'] == [1, 2]
              for side in row['token_structure'].values()
          )
      )

  def test_unknown_axes_and_capacity_are_not_given_position_coordinates(self):
    for change, expected in (
        ('layout', 'layout_unavailable'),
        ('capacity', 'logical_mapping_unavailable'),
    ):
      with self.subTest(change=change):
        store = self.snapshot_store()
        for resource in store._resources.values():
          if (
              resource.get('scope') == 'kv'
              and resource.get('moment') == 'terminal'
          ):
            if change == 'layout':
              resource['layout'] = ['batch', 'head', 'unknown', 'dimension']
            else:
              resource['shape'] = [1, 1, 8, 2]
        rows = self.terminal(analyze_kv(store, 1))['layers']
        self.assertTrue(
            all(row['status'] == expected and not row['cells'] for row in rows)
        )
        for row in rows:
          self.assertEqual(row['token_metrics'], [])
          for side in row['token_structure'].values():
            self.assertEqual(side['status'], expected)
            self.assertIsNone(side['shape'])
            self.assertIsNone(side['head_count'])
            self.assertIsNone(side['position_count'])
            self.assertEqual(side['dtype'], 'float32')

  def test_missing_side_and_analysis_limit_remain_explicit(self):
    store = self.snapshot_store()
    store._telemetry['kv_snapshots'] = [
        s for s in store._telemetry['kv_snapshots'] if s['run'] == 'ref'
    ]
    context = self.terminal(analyze_kv(store, 1))
    self.assertEqual(context['snapshots']['target'], [])
    self.assertEqual(len(context['snapshots']['ref']), 1)
    rows = context['layers']
    self.assertTrue(
        all(
            row['status'] == 'unavailable'
            and row['reference']
            and 'target' not in row
            for row in rows
        )
    )
    for row in rows:
      self.assertEqual(row['token_metrics'], [])
      self.assertIsNone(row['token_structure']['target'])
      self.assertEqual(row['token_structure']['ref']['shape'], [1, 2])
      self.assertEqual(row['token_structure']['ref']['status'], 'ok')
    store = self.snapshot_store()
    with patch('model_explorer_debugger.kv_analysis.MAX_CELLS', 1):
      result = analyze_kv(store, 1)
    rows = [row for context in result['contexts'] for row in context['layers']]
    self.assertTrue(
        all(
            row['status'] == 'analysis_limit' and row['cells'] == []
            for row in rows
        )
    )
    self.assertTrue(all(row['token_metrics'] == [] for row in rows))
    self.assertTrue(
        all(row['token_structure']['ref']['status'] == 'ok' for row in rows)
    )
    self.assertEqual(result['limits']['max_cells'], 1)

  def test_token_structure_keeps_each_sides_batch_heads_channels_on_mismatch(
      self,
  ):
    store = self.snapshot_store()
    # The index can describe different valid token geometries; no
    # correspondence is implied.
    for resource in store._resources.values():
      if resource.get('scope') == 'kv' and resource.get('moment') == 'terminal':
        if resource['run'] == 'ref':
          resource['shape'] = [2, 3, 3, 2]
        else:
          resource['shape'] = [1, 2, 4, 3]
          resource['layout'] = ['batch', 'kv_head', 'head_dim', 'sequence']
    rows = self.terminal(analyze_kv(store, 1))['layers']
    for row in rows:
      self.assertEqual(row['status'], 'shape_mismatch')
      self.assertEqual(row['cells'], [])
      ref, target = (
          row['token_structure']['ref'],
          row['token_structure']['target'],
      )
      self.assertEqual(
          (ref['shape'], ref['batch_count'], ref['position_count']),
          ([3, 2], 2, 3),
      )
      self.assertEqual(
          (target['shape'], target['batch_count'], target['position_count']),
          ([2, 4], 1, 3),
      )
      self.assertEqual((ref['head_count'], target['channel_count']), (3, 4))

  def test_invalid_valid_length_does_not_claim_token_structure(self):
    store = self.snapshot_store()
    for resource in store._resources.values():
      if (
          resource.get('scope') == 'kv'
          and resource.get('moment') == 'terminal'
          and resource['run'] == 'target'
      ):
        resource['valid_length'] = 2
    rows = self.terminal(analyze_kv(store, 1))['layers']
    for row in rows:
      self.assertEqual(
          row['token_structure']['target']['status'], 'logical_range_mismatch'
      )
      self.assertIsNone(row['token_structure']['target']['shape'])
      self.assertEqual(row['token_structure']['ref']['shape'], [1, 2])

  def test_explicit_value_cells_use_verified_permutation_and_dequantization(
      self,
  ):
    fixture = explicit_fixtures.CrossRuntimeKVTests()
    fixture.setUp()
    self.addCleanup(fixture.doCleanups)
    manifest = fixture.fixture('value')
    report = compare_pair(manifest, fixture.roots)
    store = object.__new__(SessionStore)
    store.root = fixture.root / 'saved-capture'
    store._telemetry = empty_telemetry()
    store.session = {
        'turns': [{'n': 1}],
        'runs': [
            {
                'id': role,
                'runtime': run['runtime'],
                'provenance': {'model_sha256': run['model']['artifact_sha256']},
            }
            for role, run in report['runs'].items()
        ],
    }
    store.tensors = [
        {**run['observation_tensor'], 'run': role, 'turn': 1, 'phase': 'decode'}
        for role, run in report['runs'].items()
    ]
    saved = store.register_pair(manifest, fixture.roots)
    raw_before = store.load_pair_tensor(
        saved['pair_id'], 'target', 'original'
    ).copy()
    context = analyze_kv(store, 1)['contexts'][0]
    self.assertEqual(context['source'], 'explicit')
    self.assertEqual(context['observations']['target']['step'], 4)
    ref_snapshot, target_snapshot = (
        context['snapshots']['ref'][0],
        context['snapshots']['target'][0],
    )
    self.assertEqual(
        ref_snapshot['snapshot_id'], manifest['runs']['ref']['snapshot_id']
    )
    self.assertIsNone(ref_snapshot['id'])
    self.assertIsNone(target_snapshot['snapshot_id'])
    self.assertIsNone(target_snapshot['id'])
    self.assertEqual(
        {key: target_snapshot[key] for key in ('signature', 'step', 'edge')},
        manifest['runs']['target']['snapshot'],
    )
    row = context['layers'][0]
    self.assertEqual(row['shape'], [1, 1, 4, 2])
    self.assertEqual(row['pair_id'], saved['pair_id'])
    self.assertEqual(row['metrics']['max_abs'], 1.0)
    self.assertEqual(row['summary_metrics'], saved['metrics'])
    for role in ('ref', 'target'):
      side = row['token_structure'][role]
      self.assertEqual(side['status'], 'ok')
      self.assertEqual(side['shape'], [1, 2])
      self.assertEqual(
          (side['batch_count'], side['position_start'], side['position_count']),
          (1, 0, 4),
      )
      self.assertEqual(
          side['dtype'], report['runs'][role]['original_tensor']['dtype']
      )
    self.assertEqual(row['token_structure']['target']['dtype'], 'int8')
    self.assertAlmostEqual(
        row['cells'][0]['metrics']['relative_l2'], math.sqrt(2)
    )
    self.assertAlmostEqual(
        row['cells'][3]['metrics']['relative_l2'], math.sqrt(2 / 85)
    )
    self.assertEqual(
        [cell['metrics']['max_abs'] for cell in row['cells']], [1.0] * 4
    )
    self.assertEqual(
        [token['position'] for token in row['token_metrics']], [0, 1, 2, 3]
    )
    self.assertEqual(
        [token['metrics']['rmse'] for token in row['token_metrics']], [1.0] * 4
    )
    np.testing.assert_array_equal(
        store.load_pair_tensor(saved['pair_id'], 'target', 'original'),
        raw_before,
    )
    self.assertEqual(raw_before.dtype.name, 'int8')

  def test_undefined_cells_are_null_and_heads_are_independent(self):
    # Known arrays test the public contract's axis reduction and missing values.
    x = np.array([[[[0.0, 0.0], [1.0, 2.0]], [[3.0, 4.0], [1.0, np.nan]]]])
    y = np.array([[[[1.0, 0.0], [2.0, 4.0]], [[3.0, 5.0], [1.0, 2.0]]]])
    pair = dict(
        scope='kv',
        owner_layer=0,
        kind='key',
        pair_id='known-axes',
        observation={'turn': 1, 'phase': 'decode'},
        compared_position_range=[8, 10],
        shape=[1, 2, 2, 2],
        canonical_layout=['batch', 'kv_head', 'sequence', 'head_dim'],
        runs={'ref': {}, 'target': {}},
        metrics={},
    )

    class Store:
      session = {'turns': [{'n': 1}]}

      def explicit_pair_entries(self):
        return [{
            'pair_id': pair['pair_id'],
            'manifest': {
                'kind': 'explicit_cross_runtime_terminal_kv',
                'observation': {'turn': 1},
            },
        }]

      def compare_pair(self, identity):
        return deepcopy(pair)

      def _load_validated_pair_tensor(self, report, role, mode):
        return x if role == 'ref' else y

      def telemetry(self, turn):
        return empty_telemetry()

    result = analyze_kv(Store(), 1)
    row = result['contexts'][0]['layers'][0]
    cells = {(cell['position'], cell['head']): cell for cell in row['cells']}
    self.assertEqual(row['status'], 'partial')
    self.assertIsNone(cells[8, 0]['metrics']['relative_l2'])
    self.assertEqual(cells[8, 0]['metrics']['max_abs'], 1.0)
    self.assertAlmostEqual(cells[8, 1]['metrics']['relative_l2'], 0.2)
    self.assertAlmostEqual(cells[9, 0]['metrics']['relative_l2'], 1.0)
    self.assertTrue(
        all(value is None for value in cells[9, 1]['metrics'].values())
    )
    tokens = {token['position']: token for token in row['token_metrics']}
    self.assertEqual(tokens[8]['status'], 'ok')
    self.assertAlmostEqual(
        tokens[8]['metrics']['relative_l2'], math.sqrt(2 / 25)
    )
    self.assertEqual(tokens[9]['status'], 'non_finite_tensor')
    self.assertTrue(
        all(value is None for value in tokens[9]['metrics'].values())
    )
    self.assertEqual(
        set(tokens[9]['metric_status'].values()), {'non_finite_tensor'}
    )
    json.dumps(result, allow_nan=False)

  def test_token_metrics_compare_all_heads_and_batches_instead_of_head_means(
      self,
  ):
    x = np.array([
        [[[1.0, 0.0], [0.0, 0.0]], [[10.0, 0.0], [0.0, 0.0]]],
        [[[2.0, 0.0], [0.0, 0.0]], [[20.0, 0.0], [0.0, 0.0]]],
    ])
    y = x.copy()
    y[:, 0, 0, 0] *= 2
    pair = dict(
        scope='kv',
        owner_layer=0,
        kind='key',
        pair_id='all-heads-and-batches',
        observation={'turn': 1, 'phase': 'decode'},
        compared_position_range=[8, 10],
        shape=list(x.shape),
        canonical_layout=['batch', 'kv_head', 'sequence', 'head_dim'],
        runs={'ref': {}, 'target': {}},
        metrics={},
    )

    class Store:
      session = {'turns': [{'n': 1}]}

      def explicit_pair_entries(self):
        return [{
            'pair_id': pair['pair_id'],
            'manifest': {
                'kind': 'explicit_cross_runtime_terminal_kv',
                'observation': {'turn': 1},
            },
        }]

      def compare_pair(self, identity):
        return deepcopy(pair)

      def _load_validated_pair_tensor(self, report, role, mode):
        return x if role == 'ref' else y

      def telemetry(self, turn):
        return empty_telemetry()

    row = analyze_kv(Store(), 1)['contexts'][0]['layers'][0]
    first, zero = row['token_metrics']
    self.assertEqual(first['position'], 8)
    self.assertEqual(first['status'], 'ok')
    self.assertAlmostEqual(first['metrics']['relative_l2'], math.sqrt(5 / 505))
    self.assertAlmostEqual(
        first['metrics']['cosine_similarity'], 510 / math.sqrt(505 * 520)
    )
    self.assertAlmostEqual(first['metrics']['rmse'], math.sqrt(5 / 8))
    self.assertEqual(first['metrics']['max_abs'], 2.0)
    mean_of_head_ratios = (
        sum(
            cell['metrics']['relative_l2']
            for cell in row['cells']
            if cell['position'] == 8
        )
        / 2
    )
    self.assertEqual(mean_of_head_ratios, 0.5)
    self.assertNotAlmostEqual(
        first['metrics']['relative_l2'], mean_of_head_ratios
    )
    self.assertEqual((zero['position'], zero['status']), (9, 'partial'))
    self.assertIsNone(zero['metrics']['cosine_similarity'])
    self.assertIsNone(zero['metrics']['relative_l2'])
    self.assertEqual(
        zero['metric_status']['relative_l2'], 'undefined_zero_norm'
    )
    self.assertEqual(zero['metrics']['rmse'], 0.0)
    self.assertEqual(zero['metrics']['max_abs'], 0.0)

  def test_public_endpoint_validates_turn_and_serializes_saved_cells(self):
    with client(self.snapshot_store()) as http:
      result = http.get('/api/kv-analysis?turn=1').json()
      self.assertEqual(
          self.terminal(result)['layers'][0]['cells'][0]['metrics']['max_abs'],
          1.0,
      )
      self.assertEqual(
          self.terminal(result)['layers'][0]['token_structure']['ref']['shape'],
          [1, 2],
      )
      self.assertEqual(
          self.terminal(result)['layers'][0]['token_metrics'][0]['metrics'][
              'rmse'
          ],
          1.0,
      )
      for suffix in ('', '?turn=-1', '?turn=2', '?turn=NaN'):
        with self.subTest(suffix=suffix):
          self.assertEqual(
              http.get('/api/kv-analysis' + suffix).status_code, 400
          )

  def test_unrelated_or_invalid_explicit_pairs_do_not_hide_snapshot_evidence(
      self,
  ):
    store = self.snapshot_store()
    entries = [
        {
            'pair_id': 'other-turn',
            'manifest': {
                'kind': 'explicit_cross_runtime_terminal_kv',
                'observation': {'turn': 2},
            },
        },
        {
            'pair_id': 'module',
            'manifest': {
                'kind': 'explicit_cross_runtime_prefill',
                'observation': {'turn': 1},
            },
        },
        {
            'pair_id': 'bad-manifest',
            'error': 'registered manifest identity mismatch',
        },
        {
            'pair_id': 'bad-proof',
            'manifest': {
                'kind': 'explicit_cross_runtime_terminal_kv',
                'observation': {'turn': 1},
                'owner_layer': 0,
                'kind_of_tensor': 'key',
            },
        },
    ]
    with (
        patch.object(store, 'explicit_pair_entries', return_value=entries),
        patch.object(
            store, 'compare_pair', side_effect=ValueError('invalid proof')
        ) as compare,
    ):
      result = analyze_kv(store, 1)
    compare.assert_called_once_with('bad-proof')
    self.assertIn('invalid proof', result['notice'])
    self.assertIn('manifest identity mismatch', result['notice'])
    self.assertEqual(
        self.terminal(result)['layers'][0]['cells'][0]['metrics']['max_abs'],
        1.0,
    )

  def test_evidence_budget_rejects_before_full_tensor_validation(self):
    store = self.snapshot_store()
    entry = {
        'pair_id': 'large-evidence',
        'manifest': {
            'kind': 'explicit_cross_runtime_terminal_kv',
            'observation': {'turn': 1},
            'owner_layer': 0,
            'kind_of_tensor': 'value',
            'runs': {'target': {'tensor': {'shape': [1, 1, 4096, 256]}}},
        },
    }
    with (
        patch.object(store, 'explicit_pair_entries', return_value=[entry]),
        patch.object(store, 'compare_pair') as compare,
        patch('model_explorer_debugger.kv_analysis.MAX_EVIDENCE_ELEMENTS', 100),
    ):
      result = analyze_kv(store, 1)
    compare.assert_not_called()
    row = result['contexts'][0]['layers'][0]
    self.assertEqual(
        (row['status'], row['pair_id'], row['cells']),
        ('analysis_limit', 'large-evidence', []),
    )
    self.assertEqual(self.terminal(result)['layers'][0]['status'], 'ok')

  def test_duplicate_explicit_coordinates_cannot_overwrite_one_another(self):
    store = self.snapshot_store()
    entries = [
        {
            'pair_id': identity,
            'manifest': {
                'kind': 'explicit_cross_runtime_terminal_kv',
                'observation': {'turn': 1},
            },
        }
        for identity in ('pair-a', 'pair-b')
    ]

    def report(identity):
      return dict(
          pair_id=identity,
          observation={'turn': 1, 'phase': 'decode'},
          runs={'ref': {}, 'target': {}},
          scope='kv',
          owner_layer=0,
          kind='key',
      )

    with (
        patch.object(store, 'explicit_pair_entries', return_value=entries),
        patch.object(store, 'compare_pair', side_effect=report),
        patch.object(store, '_load_validated_pair_tensor') as load,
    ):
      result = analyze_kv(store, 1)
    load.assert_not_called()
    rows = result['contexts'][0]['layers']
    self.assertEqual(len(rows), 1)
    self.assertEqual(rows[0]['status'], 'ambiguous_observation')
    self.assertEqual(rows[0]['pair_ids'], ['pair-a', 'pair-b'])
    self.assertEqual(rows[0]['cells'], [])


if __name__ == '__main__':
  unittest.main()
