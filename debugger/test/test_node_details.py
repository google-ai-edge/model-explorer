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
"""Tests for model node details and graph attribute resolution."""

import json
from pathlib import Path
import tempfile
import unittest
from inline_asgi import client
from model_explorer_debugger.node_details import file_digest
from model_explorer_debugger.store import SessionStore
import numpy as np
from safetensors.numpy import save_file


class NodeDetailsTest(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    session = {
        'batches': [{'batch': 0, 'turn': 1, 'phase': 'decode', 'step': 1}]
    }
    semantic = {
        'layers': [{'def': 0, 'attrs': {'n': {'ops': {'op': {'eps': 1e-6}}}}}],
        'semantic_graph': [{
            'inputs': [],
            'nodes': [{'id': 'n', 'label': 'Norm'}],
            'anchors': [{'id': 'a', 'of': 'n:0'}],
        }],
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
    tensors = []
    for run, data in [('ref', [3.0, 4.0]), ('target', [0.0, 4.0])]:
      save_file({'value': np.array(data)}, self.root / (run + '.safetensors'))
      tensors.append(
          dict(
              id=run,
              run=run,
              graph='g',
              node='n',
              output='0',
              layer=0,
              anchor='unmapped',
              batch=0,
              turn=1,
              phase='decode',
              step=1,
              sample='s',
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
        ('tensor_index', {'tensors': tensors}),
    ]:
      (self.root / (name + '.json')).write_text(json.dumps(data))
    self.store = SessionStore(self.root)
    self.payload = dict(
        layer=0, batch=0, semantic='anchor:a', reference='ref', target='target'
    )

  def test_explicit_selection_and_persistence(self):
    self.assertEqual(
        self.store.compare_batch(0)['rows'][0]['status'], 'missing_tensor'
    )
    saved = self.store.node_details.save(self.payload)
    self.assertEqual(saved['status'], 'saved')
    restored = SessionStore(self.root)
    row = restored.compare_batch(0)['rows'][0]
    self.assertEqual(row['mapping'], 'manual')
    self.assertAlmostEqual(row['metrics']['CosSim']['value'], 0.8)
    self.assertEqual(
        restored.node_details.saved(0, 0, 'anchor:a')['status'], 'saved'
    )
    self.assertEqual(
        restored.node_details.saved(0, 0, 'n')['status'], 'not_saved'
    )

  def test_remove_restores_automatic_behavior(self):
    self.store.node_details.save(self.payload)
    self.store.node_details.remove(self.payload)
    self.assertEqual(
        SessionStore(self.root).node_details.saved(0, 0, 'anchor:a')['status'],
        'not_saved',
    )
    self.assertEqual(
        self.store.compare_batch(0)['rows'][0]['status'], 'missing_tensor'
    )

  def test_tensor_change_is_stale(self):
    self.store.node_details.save(self.payload)
    save_file({'value': np.array([5.0, 6.0])}, self.root / 'target.safetensors')
    self.assertEqual(
        self.store.node_details.saved(0, 0, 'anchor:a')['status'],
        'stale_tensor',
    )
    self.assertEqual(self.store.compare_batch(0)['rows'][0]['metrics'], {})

  def test_graph_change_is_stale(self):
    self.store.node_details.save(self.payload)
    execution = json.loads((self.root / 'execution.json').read_text())
    execution['executions'][0]['revision'] = 'changed'
    (self.root / 'execution.json').write_text(json.dumps(execution))
    self.assertEqual(
        SessionStore(self.root).node_details.saved(0, 0, 'anchor:a')['status'],
        'stale_dataset',
    )

  def test_bad_selection_never_saved(self):
    for changes in [
        {'reference': 'target'},
        {'target': 'unknown'},
        {'semantic': 'unknown'},
        {'batch': 99},
        {'layer': -1},
    ]:
      with self.subTest(changes=changes), self.assertRaises(ValueError):
        self.store.node_details.save({**self.payload, **changes})
    self.assertFalse((self.root / 'mappings.json').exists())

  def test_context_sample_shape_rejected(self):
    for field, value, reason in [
        ('step', 5, 'tensor_context_mismatch'),
        ('sample', 'other', 'sample_mismatch'),
        ('shape', [1, 2], 'tensor_metadata_mismatch'),
    ]:
      old = self.store.tensors[1][field]
      self.store.tensors[1][field] = value
      with (
          self.subTest(field=field),
          self.assertRaisesRegex(ValueError, reason),
      ):
        self.store.node_details.compare(self.payload)
      self.store.tensors[1][field] = old

  def test_verified_source_and_hash_mismatch(self):
    source = self.root / 'source.py'
    source.write_text('def norm(x):\n    return x\n')
    evidence = {
        'sources': [{'file': 'source.py', 'sha256': file_digest(source)}],
        'claims': [{
            'path': '/semantic_graph/0/nodes/0',
            'source': {'file': 'source.py', 'start': 1, 'end': 2},
        }],
    }
    (self.root / 'semantic_evidence.json').write_text(json.dumps(evidence))
    details = self.store.node_details.details(0, 0, 'n')
    self.assertEqual(details['sources'][0]['status'], 'verified')
    self.assertEqual(
        details['sources'][0]['code'], 'def norm(x):\n    return x'
    )
    source.write_text('changed')
    self.assertEqual(
        self.store.node_details.details(0, 0, 'n')['sources'][0]['status'],
        'source_hash_mismatch',
    )

  def test_http_compare_save_and_origin(self):
    with client(self.store) as http:
      for path, status in [
          ('/api/selection/compare', 'ok'),
          ('/api/mappings', 'saved'),
      ]:
        self.assertEqual(
            http.post(path, json=self.payload).json()['status'], status
        )
      self.assertEqual(
          http.post(
              '/api/mappings',
              json=self.payload,
              headers={'Origin': 'https://other.invalid'},
          ).status_code,
          403,
      )


if __name__ == '__main__':
  unittest.main()
