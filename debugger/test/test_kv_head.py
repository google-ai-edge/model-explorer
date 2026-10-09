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

"""Exact selected-head vectors and source transformations.

Also covers unavailable comparisons.
"""

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from inline_asgi import client
from model_explorer_debugger.capture_telemetry import empty_telemetry
from model_explorer_debugger.cross_runtime_pairs import compare_pair
from model_explorer_debugger.kv_analysis import analyze_kv
from model_explorer_debugger.kv_head import inspect_kv_head
from model_explorer_debugger.node_details import file_digest
from model_explorer_debugger.store import SessionStore
import numpy as np
from safetensors.numpy import save_file
import test_cross_runtime_kv as explicit_fixtures
import test_kv_analysis as analysis_fixtures


class KvHeadTests(unittest.TestCase):
  snapshot_store = analysis_fixtures.KvAnalysisTests.snapshot_store
  terminal = analysis_fixtures.KvAnalysisTests.terminal

  def snapshot_selection(self, store):
    context = self.terminal(analyze_kv(store, 1))
    return dict(
        turn=1,
        context_id=context['id'],
        layer=0,
        kind='key',
        position=0,
        head=0,
    )

  def explicit_store(self):
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
    pair = store.register_pair(manifest, fixture.roots)
    context = analyze_kv(store, 1)['contexts'][0]
    return (
        store,
        pair,
        dict(
            turn=1,
            context_id=context['id'],
            layer=0,
            kind='value',
            position=3,
            head=0,
        ),
    )

  def test_explicit_value_vector_uses_permutation_and_dequantization_once(self):
    store, pair, selection = self.explicit_store()
    raw = store.load_pair_tensor(pair['pair_id'], 'target', 'original').copy()
    result = inspect_kv_head(store, **selection)
    self.assertEqual(result['status'], 'ok')
    self.assertEqual(result['selection']['batch'], 0)
    self.assertEqual(
        (result['shape'], result['batch_count'], result['channel_count']),
        ([2], 1, 2),
    )
    self.assertEqual(
        [
            (row['ref'], row['target'], row['delta'], row['abs_delta'])
            for row in result['channels']
        ],
        [(6.0, 7.0, 1.0, 1.0), (7.0, 8.0, 1.0, 1.0)],
    )
    self.assertAlmostEqual(result['metrics']['relative_l2'], math.sqrt(2 / 85))
    self.assertEqual(result['metrics']['rmse'], 1.0)
    self.assertEqual(result['sources']['target']['source_dtype'], 'int8')
    self.assertEqual(result['sources']['target']['source_shape'], [1, 1, 2, 8])
    self.assertEqual(result['sources']['target']['comparison_dtype'], 'float64')
    self.assertEqual(result['sources']['target']['comparison_shape'], [2])
    self.assertEqual(result['sources']['target']['axis_order'], [0, 1, 3, 2])
    self.assertEqual(
        result['sources']['target']['dequantization']['scale'], 0.25
    )
    np.testing.assert_array_equal(
        store.load_pair_tensor(pair['pair_id'], 'target', 'original'), raw
    )

  def test_single_side_snapshot_preserves_readable_channels_without_deltas(
      self,
  ):
    store = self.snapshot_store()
    selection = self.snapshot_selection(store)
    store._telemetry['kv_snapshots'] = [
        row for row in store._telemetry['kv_snapshots'] if row['run'] == 'ref'
    ]
    result = inspect_kv_head(store, **selection)
    self.assertEqual(result['status'], 'unavailable')
    self.assertIsNone(result['sources']['target'])
    self.assertEqual([row['ref'] for row in result['channels']], [1.0, 1.0])
    self.assertTrue(
        all(
            row['target'] is None
            and row['delta'] is None
            and row['abs_delta'] is None
            for row in result['channels']
        )
    )
    self.assertTrue(all(value is None for value in result['metrics'].values()))

  def test_unproven_correspondence_retains_raw_sides_and_blocks_comparison(
      self,
  ):
    store = self.snapshot_store(divergent=True)
    result = inspect_kv_head(store, **self.snapshot_selection(store))
    self.assertEqual(result['status'], 'sample_mismatch')
    self.assertEqual(
        [(row['ref'], row['target']) for row in result['channels']],
        [(1.0, 2.0), (1.0, 2.0)],
    )
    self.assertTrue(all(row['delta'] is None for row in result['channels']))
    self.assertTrue(all(value is None for value in result['metrics'].values()))

  def test_unknown_layout_has_metadata_but_no_guessed_channels(self):
    store = self.snapshot_store()
    selection = self.snapshot_selection(store)
    for resource in store._resources.values():
      if resource.get('scope') == 'kv':
        resource['layout'] = ['unknown'] * 4
    result = inspect_kv_head(store, **selection)
    self.assertEqual(result['status'], 'layout_unavailable')
    self.assertEqual(result['channels'], [])
    self.assertIsNotNone(result['sources']['ref'])
    self.assertIsNone(result['sources']['ref']['comparison_shape'])

  def array_store(self, x, y):
    store = object.__new__(SessionStore)
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    store.root = Path(temporary.name)
    store.session = {'turns': [{'n': 1}]}
    store._telemetry = empty_telemetry()
    store._resources = {}
    for role, value in (('ref', x), ('target', y)):
      path = store.root / (role + '.safetensors')
      save_file({'cache': value}, path)
      record = dict(
          id=role,
          run=role,
          turn=1,
          runtime='PyTorch',
          model_sha256='same-model',
          scope='kv',
          sample='proven-same-input',
          moment='terminal',
          layer=0,
          kind='key',
          state='available',
          preparation_status='prepared',
          terminal_status='completed',
          layout=['batch', 'kv_head', 'sequence', 'head_dim'],
          logical_start=10,
          logical_end=10 + value.shape[2],
          valid_length=value.shape[2],
          processed_token_count=10 + value.shape[2],
          shape=list(value.shape),
          dtype=value.dtype.name,
          path=path.name,
          format='safetensors',
          key='cache',
          sha256=file_digest(path),
      )
      store._resources[role] = record
      store._telemetry['resources'].append(record)
      store._telemetry['kv_snapshots'].append(
          dict(
              run=role,
              turn=1,
              runtime='PyTorch',
              phase='decode',
              step=1,
              forward_id=1,
              moment='terminal',
              layers=[{
                  'layer': 0,
                  'tensors': [{'kind': 'key', 'resource_id': role}],
              }],
          )
      )
    store.explicit_pair_entries = lambda: []
    store.load_resource = lambda identity: x if identity == 'ref' else y
    context = self.terminal(analyze_kv(store, 1))
    return store, dict(
        turn=1,
        context_id=context['id'],
        layer=0,
        kind='key',
        position=10,
        head=0,
    )

  def test_multiple_batches_require_explicit_selection_and_exact_head(self):
    x = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    y = x + np.array([1.0, 3.0], dtype=np.float32).reshape(2, 1, 1, 1)
    store, selection = self.array_store(x, y)
    selection.update(position=11, head=1)
    result = inspect_kv_head(store, **selection)
    self.assertEqual(result['status'], 'batch_selection_required')
    self.assertEqual(result['batch_count'], 2)
    self.assertEqual(result['channels'], [])
    self.assertIsNone(result['selection']['batch'])
    result = inspect_kv_head(store, **selection, batch=1)
    self.assertEqual(
        [(row['ref'], row['target']) for row in result['channels']],
        [(14.0, 17.0), (15.0, 18.0)],
    )
    self.assertEqual(result['metrics']['max_abs'], 3.0)
    self.assertAlmostEqual(
        result['metrics']['relative_l2'], math.sqrt(18 / 421)
    )

  def test_nonfinite_and_zero_norm_do_not_become_zero_error(self):
    x = np.array([[[[0.0, 0.0], [1.0, np.nan]]]], dtype=np.float32)
    y = np.array([[[[1.0, 2.0], [2.0, 3.0]]]], dtype=np.float32)
    store, selection = self.array_store(x, y)
    result = inspect_kv_head(store, **selection)
    self.assertEqual(result['status'], 'partial')
    self.assertIsNone(result['metrics']['relative_l2'])
    self.assertEqual(
        result['metric_status']['relative_l2'], 'undefined_zero_norm'
    )
    self.assertEqual(result['metrics']['max_abs'], 2.0)
    result = inspect_kv_head(store, **{**selection, 'position': 11})
    self.assertEqual(result['channels'][1]['ref'], 'nan')
    self.assertIsNone(result['channels'][1]['delta'])
    self.assertEqual(result['channels'][0]['delta'], 1.0)
    self.assertEqual(result['largest_channel'], 0)
    self.assertTrue(all(value is None for value in result['metrics'].values()))
    json.dumps(result, allow_nan=False)

  def test_api_checks_selection_bounds_and_uncaptured_positions(self):
    store = self.snapshot_store()
    selection = self.snapshot_selection(store)
    with client(store) as http:
      result = http.get('/api/kv-head?' + urlencode(selection)).json()
      self.assertEqual(result['metrics']['mean_abs'], 1.0)
      unavailable = http.get(
          '/api/kv-head?' + urlencode({**selection, 'position': 3})
      ).json()
      self.assertEqual(unavailable['status'], 'position_unavailable')
      self.assertIsNotNone(unavailable['sources']['ref'])
      self.assertEqual(unavailable['channels'], [])
      for change in (
          {'head': 4},
          {'batch': 1},
          {'turn': 2},
          {'kind': 'unknown'},
          {'layer': -1},
      ):
        with self.subTest(change=change):
          self.assertEqual(
              http.get(
                  '/api/kv-head?' + urlencode({**selection, **change})
              ).status_code,
              400,
          )

  def test_largest_channel_reports_signed_target_minus_reference(self):
    x = np.array([[[[4.0, 5.0, -2.0]]]])
    y = np.array([[[[3.0, 1.0, 1.0]]]])
    store, selection = self.array_store(x, y)
    result = inspect_kv_head(store, **selection)
    self.assertEqual(result['largest_channel'], 1)
    self.assertEqual(
        [row['delta'] for row in result['channels']], [-1.0, -4.0, 3.0]
    )
    self.assertEqual(result['metrics']['max_abs'], 4.0)

  def test_workspace_endpoint_uses_only_the_selected_session(self):
    x = np.array([[[[1.0, 2.0]]]])
    first, selection = self.array_store(x, x + 1)
    second, _ = self.array_store(x, x + 3)

    class Registry:

      def capture_store(self, identity):
        return {'first': first, 'second': second}[identity]

    with (
        patch('model_explorer_debugger.jobs.JobManager'),
        patch('model_explorer_debugger.tap_scans.TapScans'),
        client(registry=Registry(), stores=(first, second)) as http,
    ):
      for identity, expected in (('first', 1.0), ('second', 3.0)):
        query = urlencode({**selection, 'session_id': identity})
        self.assertEqual(
            http.get('/api/kv-head?' + query).json()['metrics']['max_abs'],
            expected,
        )


if __name__ == '__main__':
  unittest.main()
