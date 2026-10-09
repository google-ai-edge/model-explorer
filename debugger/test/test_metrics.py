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
"""Tests for numerical tensor error metrics and reduction rules."""

import json
from pathlib import Path
import tempfile
import unittest
from inline_asgi import client
from model_explorer_debugger.metrics import compare
from model_explorer_debugger.store import SessionStore
import numpy as np
from safetensors.numpy import save_file


class MetricsTest(unittest.TestCase):

  def test_hand_calculated(self):
    m = compare(np.array([3.0, 4.0]), np.array([0.0, 4.0]))
    for k, value in {
        'CosSim': 0.8,
        'Max abs error': 3.0,
        'Mean abs error': 1.5,
        'RMSE': np.sqrt(4.5),
        'Relative L2': 0.6,
    }.items():
      self.assertAlmostEqual(m[k]['value'], value)

  def test_zero_norm_is_metric_specific(self):
    m = compare(np.zeros(2), np.ones(2))
    self.assertIsNone(m['CosSim']['value'])
    self.assertIsNone(m['Relative L2']['value'])
    self.assertEqual(m['RMSE']['value'], 1)

  def test_invalid_arrays(self):
    for x, y, reason in [
        (np.ones((2, 1)), np.ones(2), 'shape_mismatch'),
        (np.array([]), np.array([]), 'empty_tensor'),
        (np.array([np.nan]), np.ones(1), 'non_finite_tensor'),
        (np.array([np.inf]), np.ones(1), 'non_finite_tensor'),
        (np.array(['x']), np.array(['x']), 'unsupported_dtype'),
    ]:
      with (
          self.subTest(reason=reason),
          self.assertRaisesRegex(ValueError, reason),
      ):
        compare(x, y)


class StoreTest(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    session = {
        'batches': [{'batch': 0, 'turn': 1, 'phase': 'decode', 'step': 1}]
    }
    semantic = {
        'layers': [{'def': 0}],
        'semantic_graph': [{'anchors': [{'id': 'a'}, {'id': 'b'}]}],
    }
    execution = {
        'executions': [
            {
                'id': run,
                'graphs': [{
                    'id': 'g',
                    'nodes': [{'id': 'n', 'outputsMetadata': [{'id': '0'}]}],
                }],
            }
            for run in ('ref', 'target')
        ]
    }
    records = []
    for run, data in [('ref', [3.0, 4.0]), ('target', [0.0, 4.0])]:
      save_file({'value': np.array(data)}, self.root / (run + '.safetensors'))
      records.append(
          dict(
              id=run,
              run=run,
              graph='g',
              node='n',
              output='0',
              layer=0,
              anchor='a',
              batch=0,
              turn=1,
              phase='decode',
              step=1,
              sample='sample-0',
              path=run + '.safetensors',
              format='safetensors',
              key='value',
              shape=[2],
              dtype='float64',
          )
      )
    for name, data in [
        ('session', session),
        ('semantic', semantic),
        ('execution', execution),
        ('tensor_index', {'tensors': records}),
    ]:
      (self.root / (name + '.json')).write_text(json.dumps(data))
    self.store = SessionStore(self.root)

  def test_worst_metric_direction(self):
    save_file(
        {'value': np.array([0.0, -4.0])}, self.root / 'negative.safetensors'
    )
    self.store.tensors += [
        {
            **t,
            'id': t['id'] + 'b',
            'anchor': 'b',
            'path': (
                'negative.safetensors' if t['run'] == 'target' else t['path']
            ),
        }
        for t in self.store.tensors.copy()
    ]
    metrics = self.store.compare_batch(0)['layers'][0]['metrics']
    self.assertAlmostEqual(metrics['CosSim']['value'], -0.8)
    self.assertEqual(metrics['Max abs error']['value'], 8)
    self.assertEqual(metrics['CosSim']['valid'], 2)

  def test_overview_preserves_coverage(self):
    result = self.store.overview()['batches'][0]['metrics']['CosSim']
    self.assertEqual(result, {'value': 0.8, 'valid': 1, 'total': 2})

  def test_partial_coverage(self):
    result = self.store.compare_batch(0)
    self.assertEqual(result['rows'][0]['status'], 'ok')
    self.assertEqual(result['rows'][1]['status'], 'missing_tensor')
    self.assertEqual(
        result['layers'][0]['metrics']['CosSim'],
        {'value': 0.8, 'valid': 1, 'total': 2},
    )

  def test_no_cross_sample_or_step(self):
    self.store.tensors[1]['sample'] = 'other'
    self.assertEqual(
        self.store.compare_batch(0)['rows'][0]['status'], 'sample_mismatch'
    )
    self.store.tensors[1]['step'] = 2
    self.assertEqual(
        self.store.compare_batch(0)['rows'][0]['status'], 'missing_tensor'
    )

  def test_duplicate_is_ambiguous(self):
    self.store.tensors.append(self.store.tensors[0].copy())
    self.assertEqual(
        self.store.compare_batch(0)['rows'][0]['status'], 'ambiguous_tensor'
    )

  def test_metadata_identity_and_path(self):
    original = self.store.tensors[0].copy()
    for field, value, status in [
        ('shape', [1, 2], 'tensor_metadata_mismatch'),
        ('node', 'wrong', 'execution_identity_mismatch'),
        ('path', '../outside.safetensors', 'invalid_tensor_path'),
    ]:
      with self.subTest(field=field):
        self.store.tensors[0] = {**original, field: value}
        self.assertEqual(
            self.store.compare_batch(0)['rows'][0]['status'], status
        )

  def test_missing_file_does_not_break_batch(self):
    (self.root / 'ref.safetensors').unlink()
    self.assertEqual(
        self.store.compare_batch(0)['rows'][0]['status'], 'FileNotFoundError'
    )

  def test_http_contract(self):
    with client(self.store) as http:
      result = http.get('/api/comparisons?batch=0').json()
      self.assertEqual(
          result['rows'][0]['metrics']['RMSE']['value'], np.sqrt(4.5)
      )
      self.assertEqual(http.get('/api/comparisons?batch=99').status_code, 404)


if __name__ == '__main__':
  unittest.main()
