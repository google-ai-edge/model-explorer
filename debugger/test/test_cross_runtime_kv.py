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

"""Synthetic terminal KV views, owner proof, quantization and persistence."""

from copy import deepcopy
import json
import shutil
import unittest

from model_explorer_debugger.cross_runtime_pairs import (
    compare_pair,
    read_pair,
    register_pair,
)
from model_explorer_debugger.store import SessionStore
import numpy as np
from safetensors.numpy import save_file
import test_cross_runtime_pairs as fixtures


class CrossRuntimeKVTests(unittest.TestCase):
  setUp = fixtures.CrossRuntimePairTests.setUp
  file = fixtures.CrossRuntimePairTests.file
  tensor = fixtures.CrossRuntimePairTests.tensor
  write_json = fixtures.CrossRuntimePairTests.write_json
  change_result = fixtures.CrossRuntimePairTests.change_result
  position_fixture = fixtures.CrossRuntimePairTests.position_fixture

  def fixture(self, kind):
    base = self.position_fixture(3)
    ref_index = json.loads(
        (self.roots['ref'] / 'export/capture_index.json').read_text()
    )
    raw = np.arange(8, dtype=np.float32).reshape(1, 1, 4, 2)
    save_file(
        {'key': raw, 'value': raw},
        self.roots['ref'] / 'raw/terminal-kv.safetensors',
    )
    resources = [
        {
            'scope': 'kv',
            'forward_id': 1,
            'snapshot_id': 2,
            'layer': 0,
            'kind': k,
            'capture_key': 'kv:' + k,
            'path': 'raw/terminal-kv.safetensors',
            'key': k,
            'format': 'safetensors',
            'shape': list(raw.shape),
            'dtype': 'float32',
        }
        for k in ('key', 'value')
    ]
    ref_index['resources'].extend(resources)
    layer = {
        'layer': 0,
        'state': 'available',
        'layout': ['batch', 'kv_head', 'sequence', 'head_dim'],
        'capacity': 4,
        'valid_length': 4,
        'logical_start': 0,
        'logical_end': 4,
        'processed_token_count': 4,
        'tensors': [
            {
                'kind': k,
                'key': 'kv:' + k,
                'shape': list(raw.shape),
                'dtype': 'torch.float32',
                'storage_status': 'stored',
            }
            for k in ('key', 'value')
        ],
    }
    ref_index['kv_snapshots'] = [{
        'snapshot_id': 2,
        'moment': 'terminal',
        'forward_id': 1,
        'phase': 'decode',
        'turn': 1,
        'state': 'available',
        'terminal_status': 'completed',
        'preparation_status': 'prepared',
        'processed_token_count': 4,
        'layers': [layer],
    }]
    ref_index['generation'] = {
        'status': 'completed',
        'processed_token_count': 4,
        'pending_token_ids': [10],
    }
    base['prefill']['runs']['ref']['capture_index'] = self.write_json(
        'ref', 'export/capture_index.json', ref_index
    )
    result = json.loads((self.roots['ref'] / 'result.json').read_text())
    result['processed_token_count'] = 4
    base['prefill']['runs']['ref']['result'] = self.write_json(
        'ref', 'result.json', result
    )
    native = []
    quantization_sources = []
    for k, scale, tensor_id in (('key', 0.5, 100), ('value', 0.25, 101)):
      values = np.zeros((1, 1, 8, 2), dtype=np.int8)
      values[:, :, :4, :] = ((raw + 1) / scale).astype(np.int8)
      layout = ['batch', 'kv_head', 'sequence', 'head_dim']
      if k == 'value':
        values = values.transpose(0, 1, 3, 2).copy()
        layout = ['batch', 'kv_head', 'head_dim', 'sequence']
      slot = 'post_kv_cache_' + ('k' if k == 'key' else 'v') + '_0'
      record = self.tensor(
          'target',
          'raw/' + slot + '.safetensors',
          slot,
          values,
          metadata={'signature': 'decode', 'step': '4'},
      )
      record['file_sha256'] = record.pop('sha256')
      record.update(
          slot=0,
          kind=k,
          owner_layer=0,
          layout=layout,
          sequence_axis=layout.index('sequence'),
          head_dim=2,
          capacity=8,
          valid_length=4,
          logical_start=0,
          logical_end=4,
          quantization={
              'scale': scale,
              'zero_point': 0,
              'formula': 'real_value = (stored_int8 - zero_point) * scale',
          },
          source={
              'signature': 'decode',
              'composite': 'odml.cache_update',
              'output_tensor': tensor_id,
              'owner_input_tensor_names': [
                  'model/layer_0/input-key',
                  'model/layer_0/input-value',
              ],
              'composite_attributes': {
                  'cache_size': 16,
                  'head_size': 2,
                  'scale_k': 0.5,
                  'scale_v': 0.25,
              },
          },
      )
      native.append(record)
      quantization_sources.append({
          'name': slot.removeprefix('post_'),
          'tensor': tensor_id,
          'dtype': 'INT8',
          'quantization': {'scale': [scale], 'zero_point': [0]},
      })
    conversion = json.loads(
        (self.roots['target'] / 'conversion.json').read_text()
    )
    conversion['decode_cache_outputs'] = quantization_sources
    base['prefill']['runs']['target']['model']['source_chain'][1] = (
        self.write_json('target', 'conversion.json', conversion)
    )
    target_result = json.loads(
        (self.roots['target'] / 'result.json').read_text()
    )
    identity = {'signature': 'decode', 'step': 4, 'edge': 'post'}
    kv_index = {
        'format_version': 1,
        'model_tflite_sha256': '7' * 64,
        'tapped_container_sha256': 'b' * 64,
        'replay_result_sha256': base['prefill']['runs']['target']['result'][
            'sha256'
        ],
        'replay_proof_sha256': target_result['token_replay_proof'][
            'proof_sha256'
        ],
        'terminal_snapshot': identity,
        'pending_token_id': 10,
        'pending_token_position': 4,
        'snapshots': [
            {**identity, 'processed_token_count': 4, 'tensors': native}
        ],
    }
    kv_ref = self.write_json('target', 'kv_evidence.json', kv_index)
    manifest = {
        'format_version': 3,
        'kind': 'explicit_cross_runtime_terminal_kv',
        'terminal_position': base,
        'observation': {'turn': 1, 'phase': 'decode'},
        'owner_layer': 0,
        'kind_of_tensor': kind,
        'runs': {},
    }
    for role, tensor, layout in (
        ('ref', resources[0 if kind == 'key' else 1], layer['layout']),
        (
            'target',
            native[0 if kind == 'key' else 1],
            native[0 if kind == 'key' else 1]['layout'],
        ),
    ):
      view = [{'start': 0, 'stop': size, 'step': 1} for size in tensor['shape']]
      view[layout.index('sequence')]['stop'] = 4
      row = {
          'tensor': {**tensor, **self.file(role, tensor['path'])},
          'layout': layout,
          'view': view,
          'axis_order': [
              layout.index(name)
              for name in ['batch', 'kv_head', 'sequence', 'head_dim']
          ],
          'dequantization': None if role == 'ref' else tensor['quantization'],
      }
      if role == 'ref':
        row['snapshot_id'] = 2
      else:
        row.update(kv_index=kv_ref, snapshot=identity)
      manifest['runs'][role] = row
    return manifest

  def test_raw_key_is_int8_and_comparison_dequantizes_without_rewriting(self):
    manifest = self.fixture('key')
    raw = (
        self.roots['target'] / manifest['runs']['target']['tensor']['path']
    ).read_bytes()
    report = compare_pair(manifest, self.roots)
    self.assertEqual(
        (report['scope'], report['owner_layer'], report['kind']),
        ('kv', 0, 'key'),
    )
    self.assertEqual(report['shape'], [1, 1, 4, 2])
    self.assertEqual(report['compared_position_range'], [0, 4])
    self.assertEqual(report['metrics']['Max abs error']['value'], 1.0)
    self.assertEqual(
        (
            self.roots['target'] / manifest['runs']['target']['tensor']['path']
        ).read_bytes(),
        raw,
    )

  def test_value_permutation_and_previews_are_explicit_and_saved(self):
    manifest = self.fixture('value')
    report = compare_pair(manifest, self.roots)
    self.assertEqual(report['runs']['target']['axis_order'], [0, 1, 3, 2])
    self.assertEqual(report['metrics']['Max abs error']['value'], 1.0)
    store = object.__new__(SessionStore)
    store.root = self.root / 'capture'
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
    saved = store.register_pair(manifest, self.roots)
    shutil.rmtree(self.roots['ref'])
    shutil.rmtree(self.roots['target'])
    self.assertEqual(store.compare_pair(saved['pair_id']), saved)
    raw = store.load_pair_tensor(saved['pair_id'], 'target', 'original')
    compared = store.load_pair_tensor(saved['pair_id'], 'target', 'comparison')
    self.assertEqual((raw.dtype.name, list(raw.shape)), ('int8', [1, 1, 2, 8]))
    self.assertEqual(list(compared.shape), [1, 1, 4, 2])
    np.testing.assert_array_equal(
        compared, np.arange(8).reshape(1, 1, 4, 2) + 1
    )
    store.session['runs'][1]['runtime'] = 'PyTorch'
    with self.assertRaisesRegex(ValueError, 'session_model_mismatch'):
      store.compare_pair(saved['pair_id'])

  def test_range_owner_quantization_and_axes_cannot_be_guessed(self):
    manifest = self.fixture('value')
    for cause in (
        'range',
        'owner',
        'scale',
        'axes',
        'boolean_axis',
        'snapshot',
    ):
      with self.subTest(cause=cause):
        altered = deepcopy(manifest)
        if cause == 'range':
          altered['runs']['target']['view'][3]['stop'] = 3
        elif cause == 'owner':
          altered['owner_layer'] = 1
        elif cause == 'scale':
          altered['runs']['target']['dequantization']['scale'] = 0.5
        elif cause == 'axes':
          altered['runs']['target']['axis_order'] = [0, 1, 2, 3]
        elif cause == 'boolean_axis':
          altered['runs']['target']['axis_order'] = [False, 1, 3, 2]
        else:
          altered['runs']['target']['snapshot']['step'] = 3
        with self.assertRaises(ValueError):
          compare_pair(altered, self.roots)

  def test_cache_owner_and_quantization_require_original_source_mapping(self):
    manifest = self.fixture('key')
    index = json.loads((self.roots['target'] / 'kv_evidence.json').read_text())
    index['snapshots'][0]['tensors'][0]['source'][
        'owner_input_tensor_names'
    ] = ['model/layer_1/key', 'model/layer_1/value']
    manifest['runs']['target']['kv_index'] = self.write_json(
        'target', 'kv_evidence.json', index
    )
    with self.assertRaisesRegex(ValueError, 'owner lacks'):
      compare_pair(manifest, self.roots)

  def test_aborted_pytorch_snapshot_cannot_be_reported_as_normal_terminal(self):
    manifest = self.fixture('key')
    index = json.loads(
        (self.roots['ref'] / 'export/capture_index.json').read_text()
    )
    index['kv_snapshots'][0]['terminal_status'] = 'aborted'
    manifest['terminal_position']['prefill']['runs']['ref']['capture_index'] = (
        self.write_json('ref', 'export/capture_index.json', index)
    )
    with self.assertRaisesRegex(ValueError, 'terminal state mismatch'):
      compare_pair(manifest, self.roots)


if __name__ == '__main__':
  unittest.main()
