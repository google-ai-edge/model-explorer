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

"""Synthetic all-forward publication and independent execution resources."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest

from model_explorer_debugger.capture_importer import publish_capture
from model_explorer_debugger.capture_telemetry import proof_digest
from model_explorer_debugger.fsutil import atomic_json
from model_explorer_debugger.store import SessionStore
import numpy as np
from safetensors.numpy import save_file
import test_pytorch_import as fixtures

# Real-model fixtures are Runner-owned; import them from the Runner test tree.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / 'src/runner/python/tests')
)
import test_pytorch_worker as worker_fixtures  # noqa: E402


class PyTorchTelemetryTests(unittest.TestCase):
  setUp = fixtures.PyTorchImportTests.setUp
  publish = fixtures.PyTorchImportTests.publish

  def fixture(self, job_id='job-all', divergent=False):
    job, results = fixtures.PyTorchImportTests.fixture(self, job_id)
    for run in ('ref', 'target'):
      directory = job / run
      index = json.loads((directory / 'export/capture_index.json').read_text())
      for item in index['tensors']:
        item.update(forward_id=0, module_call_id=0, scope='module', turn=5)
      save_file(
          {
              'slot0': np.array([[3, 4]], dtype=np.float32),
              'slot1': np.array([[5, 6 + (run == 'target')]], dtype=np.float32),
          },
          directory / 'raw/decode.safetensors',
      )
      index['tensors'].extend(
          {
              **item,
              'forward_id': 1,
              'phase': 'decode',
              'step': 1,
              'path': 'raw/decode.safetensors',
              'call_seq': item['call_seq'] + 2,
              'module_call_id': 1,
          }
          for item in index['tensors'].copy()
      )
      resources, forwards, proofs, boundary_payload = [], [], [], {}
      for fid in (0, 1):
        ids = (
            [[4, 5]]
            if fid == 0
            else [[43 if divergent and run == 'target' else 42]]
        )
        inputs = {
            'input_ids': np.array(ids, dtype=np.int64),
            'attention_mask': np.ones((1, 2 + fid), dtype=np.int64),
            'cache_position': np.arange(
                0 if fid == 0 else 2, 2 + fid, dtype=np.int64
            ),
        }
        forward = {
            'forward_id': fid,
            'phase': 'prefill' if fid == 0 else 'decode',
            'step': fid,
            'turn': 5,
            'pos_offset': 0 if fid == 0 else 2,
            'status': 'completed',
            'inputs': [],
            'outputs': [],
            'cache_before': {
                'state': 'not_allocated' if fid == 0 else 'available',
                'processed_token_count': None if fid == 0 else 2,
            },
        }
        proof = {
            'scope': 'same_logical_context',
            'tensors': {},
            'cache_context': {
                'processed_token_count': 0 if fid == 0 else 2,
                'token_ids': [] if fid == 0 else [4, 5],
                'attention_mask': [] if fid == 0 else [1, 1],
            },
            'tokenizer': {'vocab_sha256': 'synthetic-vocabulary'},
            'serialization': 'synthetic',
        }
        for name, tensor in list(inputs.items()) + [
            ('logits', np.array([[1, 2 + (run == 'target')]], dtype=np.float32))
        ]:
          edge = 'outputs' if name == 'logits' else 'inputs'
          path = ['output', 'logits'] if edge == 'outputs' else ['kwargs', name]
          slot, key = f'{fid}_{name}', f'boundary:{fid}:{name}'
          boundary_payload[slot] = tensor
          forward[edge].append({
              'key': key,
              'output_path': path,
              'shape': list(tensor.shape),
              'dtype': 'torch.' + tensor.dtype.name,
          })
          resources.append({
              'scope': 'boundary',
              'forward_id': fid,
              'capture_key': key,
              'format': 'safetensors',
              'path': 'raw/boundaries.safetensors',
              'key': slot,
              'shape': list(tensor.shape),
              'dtype': tensor.dtype.name,
              'output_path': path,
              'when': 'out' if edge == 'outputs' else 'in',
          })
          if edge == 'inputs':
            proof['tensors'][name] = {
                'shape': list(tensor.shape),
                'dtype': tensor.dtype.name,
                'values': tensor.tolist(),
            }
        proofs.append({
            'forward_id': fid,
            'input_proof': proof,
            'input_identity': proof_digest(proof),
            'comparison_basis': 'logical_context',
        })
        forwards.append(forward)
        if fid == 0:
          save_file(inputs, directory / 'raw/inputs.safetensors')
          results[run].update(
              input_proof=proof, input_identity=proof_digest(proof)
          )
          index['input_identity'] = proof_digest(proof)
      save_file(boundary_payload, directory / 'raw/boundaries.safetensors')
      kv_payload, snapshots = {}, [{
          'snapshot_id': 0,
          'forward_id': 0,
          'moment': 'prefill_pre',
          'phase': 'prefill',
          'step': 0,
          'turn': 5,
          'state': 'not_allocated',
          'layers': [],
      }]
      for sid, fid, moment, length in (
          (1, 0, 'prefill_post', 2),
          (2, 1, 'terminal', 3),
      ):
        layer = {
            'layer': 0,
            'layer_type': 'DynamicLayer',
            'state': 'available',
            'layout': ['batch', 'kv_head', 'sequence', 'head_dim'],
            'capacity': length,
            'valid_length': length,
            'logical_start': 0,
            'logical_end': length,
            'processed_token_count': length,
            'tensors': [],
        }
        for kind in ('key', 'value'):
          slot, key = f'{sid}_{kind}', f'kv:{sid}:{kind}'
          tensor = np.ones((1, 1, length, 2), dtype=np.float32) + (
              run == 'target'
          )
          kv_payload[slot] = tensor
          layer['tensors'].append({
              'kind': kind,
              'key': key,
              'shape': list(tensor.shape),
              'dtype': 'torch.float32',
          })
          resources.append({
              'scope': 'kv',
              'forward_id': fid,
              'snapshot_id': sid,
              'layer': 0,
              'kind': kind,
              'capture_key': key,
              'format': 'safetensors',
              'path': 'raw/kv.safetensors',
              'key': slot,
              'shape': list(tensor.shape),
              'dtype': 'float32',
          })
        snapshots.append({
            'snapshot_id': sid,
            'forward_id': fid,
            'moment': moment,
            'phase': 'prefill' if fid == 0 else 'decode',
            'step': fid,
            'turn': 5,
            'state': 'available',
            'processed_token_count': length,
            'layers': [layer],
            'preparation_status': 'prepared',
            **(
                {'terminal_status': 'completed'} if moment == 'terminal' else {}
            ),
        })
      save_file(kv_payload, directory / 'raw/kv.safetensors')
      events = [
          {
              'event_id': 0,
              'forward_id': 0,
              'candidate': 0,
              'output_index': 0,
              'token_ids': [42],
              'text': 'first chunk',
              'texts': ['X'],
              'scores': [-0.2],
              'score_kind': 'logprob',
              'consumption': [{'consumed_by_forward_id': 1, 'position': 2}],
          },
          {
              'event_id': 1,
              'forward_id': 0,
              'candidate': 1,
              'output_index': 0,
              'token_ids': [70, 71],
              'text': 'alternate chunk',
              'texts': ['a', 'b'],
              'scores': None,
              'score_kind': None,
              'consumption': (
                  [{'consumed_by_forward_id': None, 'position': None}] * 2
              ),
          },
          {
              'event_id': 2,
              'forward_id': 1,
              'candidate': 0,
              'output_index': 1,
              'token_ids': [44],
              'text': 'tail',
              'texts': ['Y'],
              'scores': None,
              'score_kind': None,
              'consumption': [
                  {'consumed_by_forward_id': None, 'position': None}
              ],
          },
      ]
      index.update(
          export_scope='all',
          capture_scope='generation',
          forwards=forwards,
          resources=resources,
          forward_input_proofs=proofs,
          kv_snapshots=snapshots,
          token_records=events,
          generation={
              'status': 'completed',
              'stop_reason': 'max_output_tokens',
              'generated_token_count': 2,
              'processed_token_count': 3,
              'pending_token_ids': [44],
          },
      )
      atomic_json(directory / 'export/capture_index.json', index)
    return job, results

  def pair(self, telemetry, **selection):
    return [
        next(
            row
            for row in telemetry['resources']
            if row['run'] == run
            and all(row.get(key) == value for key, value in selection.items())
        )['id']
        for run in ('ref', 'target')
    ]

  def test_sampled_token_links_use_source_forward_not_generation_step(self):
    job, results = self.fixture()
    for result in results.values():
      result['tokens'] = [
          {'id': 42, 'text': 'X', 'step': 0},
          {'id': 44, 'text': 'Y', 'step': 1},
          {'id': 99, 'text': 'unknown', 'step': 2},
      ]
    store = self.publish(job, results)
    for conversation in store.session['conversation']:
      first, second, unknown = conversation['tokens']
      self.assertEqual((first['source_forward_id'], first['batch']), (0, 0))
      self.assertEqual(
          store.session['batches'][first['batch']]['phase'], 'prefill'
      )
      self.assertEqual((second['source_forward_id'], second['batch']), (1, 1))
      self.assertNotIn('batch', unknown)
    job2, results2 = self.fixture('turn-two')
    for result in results2.values():
      result['tokens'] = [
          {'id': 42, 'text': 'X', 'step': 0},
          {'id': 44, 'text': 'Y', 'step': 1, 'source_forward_id': 99},
      ]
    store2 = self.publish(job2, results2)
    for conversation in store2.session['conversation'][2:]:
      first, conflict = conversation['tokens']
      self.assertEqual(first['batch'], 2)
      self.assertNotIn('batch', conflict)

  def test_old_saved_capture_links_are_recovered_without_rewriting_json(self):
    job, results = self.fixture()
    for result in results.values():
      result['tokens'] = [
          {'id': 42, 'text': 'X', 'step': 0},
          {'id': 44, 'text': 'Y', 'step': 1},
      ]
    store = self.publish(job, results)
    session_path = store.root / 'session.json'
    saved = json.loads(session_path.read_text())
    for conversation in saved['conversation']:
      for token in conversation['tokens']:
        token.pop('source_forward_id', None)
        token['batch'] = 999
    session_path.write_text(json.dumps(saved))
    before = session_path.read_bytes()
    restored = SessionStore(store.root)
    self.assertEqual(session_path.read_bytes(), before)
    import hashlib
    from uuid import NAMESPACE_URL, uuid5
    from model_explorer_debugger.session_metadata import SessionMetadata

    fingerprint = hashlib.sha256(
        json.dumps(saved, sort_keys=True).encode()
    ).hexdigest()
    self.assertEqual(
        SessionMetadata(restored).identity,
        str(
            uuid5(NAMESPACE_URL, 'model-debugger:saved-capture:' + fingerprint)
        ),
    )
    for conversation in restored.session['conversation']:
      self.assertEqual(
          [t['source_forward_id'] for t in conversation['tokens']], [0, 1]
      )
      self.assertEqual([t['batch'] for t in conversation['tokens']], [0, 1])

  def test_all_forward_graphs_and_independent_resources_compare_with_proof(
      self,
  ):
    job, results = self.fixture()
    store = self.publish(job, results)
    data = store.telemetry(turn=1)
    self.assertEqual(len(data['forwards']), 4)
    self.assertEqual(len(data['resources']), 24)
    self.assertEqual(len(data['kv_snapshots']), 6)
    self.assertEqual(len(data['token_records']), 6)
    self.assertEqual({row['runtime_turn'] for row in data['forwards']}, {5})
    self.assertEqual(
        [batch['forward_id'] for batch in store.session['batches']], [0, 1]
    )
    self.assertTrue(
        all(
            len(execution['graphs']) == 2
            for execution in store.execution['executions']
        )
    )
    self.assertTrue(all('graph' not in row for row in data['resources']))
    for batch in (0, 1):
      self.assertTrue(
          all(
              row['status'] == 'ok'
              for row in store.compare_batch(batch)['rows']
          )
      )
    ids = self.pair(data, scope='kv', moment='terminal', kind='key')
    self.assertEqual(store.compare_resources(*ids)['status'], 'ok')
    self.assertEqual(
        store.compare_resources(*ids)['metrics']['Max abs error']['value'], 1.0
    )
    event = next(
        row
        for row in data['token_records']
        if row['run'] == 'ref' and row['candidate'] == 1
    )
    self.assertEqual(
        (event['token_ids'], event['text'], event['texts'], event['scores']),
        ([70, 71], 'alternate chunk', ['a', 'b'], None),
    )

  def test_divergent_decode_ids_pair_prefill_but_not_decode_or_terminal(self):
    job, results = self.fixture(divergent=True)
    store = self.publish(job, results)
    self.assertTrue(
        all(row['status'] == 'ok' for row in store.compare_batch(0)['rows'])
    )
    self.assertTrue(
        all(
            row['status'] == 'sample_mismatch'
            for row in store.compare_batch(1)['rows']
        )
    )
    self.assertEqual(
        store.compare_resources(
            *self.pair(
                store.telemetry(), scope='kv', moment='terminal', kind='key'
            )
        )['status'],
        'sample_mismatch',
    )

  def test_all_turn_resources_keep_original_shard_bytes_after_jobs_are_deleted(
      self,
  ):
    job, results = self.fixture()
    original = {
        (run, filename): (job / run / 'raw' / filename).read_bytes()
        for run in ('ref', 'target')
        for filename in ('boundaries.safetensors', 'kv.safetensors')
    }
    first = self.publish(job, results)
    saved = {
        row['path']: (first.root / row['path']).read_bytes()
        for row in first.telemetry()['resources']
    }
    self.assertEqual(set(saved.values()), set(original.values()))
    self.assertEqual(len(saved), 4)
    shutil.rmtree(job)
    job, results = self.fixture('second-turn')
    store = self.publish(job, results)
    shutil.rmtree(job)
    reloaded = SessionStore(store.root)
    self.assertEqual(len(reloaded.telemetry()['forwards']), 8)
    self.assertEqual(len(reloaded.telemetry()['generations']), 4)
    for path, raw in saved.items():
      self.assertEqual((reloaded.root / path).read_bytes(), raw)
    for turn in (1, 2):
      data = reloaded.telemetry(turn=turn)
      self.assertEqual(len(data['resources']), 24)
      self.assertEqual(
          reloaded.compare_resources(
              *self.pair(data, scope='kv', moment='terminal', kind='value')
          )['status'],
          'ok',
      )
      for resource in data['resources']:
        self.assertEqual(
            list(reloaded.load_resource(resource['id']).shape),
            resource['shape'],
        )

  def test_forged_or_incomplete_forward_proofs_cannot_pair(self):
    for cause in ('payload', 'hash', 'prefix_unknown', 'position_unknown'):
      with self.subTest(cause=cause):
        job, results = self.fixture('proof-' + cause)
        path = job / 'target/export/capture_index.json'
        index = json.loads(path.read_text())
        row = index['forward_input_proofs'][1]
        if cause == 'payload':
          row['input_proof']['tensors']['input_ids']['values'] = [[99]]
        elif cause == 'hash':
          row['input_identity'] = 'forged'
        elif cause == 'prefix_unknown':
          row['input_proof']['cache_context'] = None
        else:
          # Explicitly stale cache position proof is true to the raw tensor but
          # not contiguous.
          row['input_proof']['cache_context']['processed_token_count'] = 3
          row['input_proof']['cache_context']['token_ids'].append(40)
          row['input_proof']['cache_context']['attention_mask'].append(1)
        if cause != 'hash':
          row['input_identity'] = proof_digest(row['input_proof'])
        atomic_json(path, index)
        if cause in ('payload', 'hash'):
          with self.assertRaises(ValueError):
            publish_capture(
                self.registry, self.record, job, results, self.artifacts
            )
        else:
          store = self.publish(job, results)
          batch = store.session['batches'][-1]['batch']
          self.assertTrue(
              all(
                  row['status'] == 'sample_mismatch'
                  for row in store.compare_batch(batch)['rows']
              )
          )

  def test_resource_path_and_identity_tampering_reject_publication(self):
    for cause in ('path', 'shape', 'duplicate', 'forward'):
      with self.subTest(cause=cause):
        job, results = self.fixture('bad-resource-' + cause)
        path = job / 'ref/export/capture_index.json'
        index = json.loads(path.read_text())
        resource = index['resources'][0]
        if cause == 'path':
          resource['path'] = '../../outside.safetensors'
        elif cause == 'shape':
          resource['shape'] = [999]
        elif cause == 'duplicate':
          index['resources'].append(deepcopy(resource))
        else:
          resource['forward_id'] = 99
        atomic_json(path, index)
        with self.assertRaises(ValueError):
          publish_capture(
              self.registry, self.record, job, results, self.artifacts
          )

  def test_kv_range_and_layout_are_not_silently_sliced_or_guessed(self):
    job, results = self.fixture()
    store = self.publish(job, results)
    ids = self.pair(
        store.telemetry(), scope='kv', moment='terminal', kind='key'
    )
    target = store._resources[ids[1]]
    target['logical_start'] = 1
    self.assertEqual(
        store.compare_resources(*ids)['status'], 'logical_range_mismatch'
    )
    target['logical_start'] = 0
    target['layout'] = ['batch', 'sequence', 'kv_head', 'head_dim']
    self.assertEqual(store.compare_resources(*ids)['status'], 'layout_mismatch')
    target['layout'] = ['batch', 'kv_head', 'sequence', 'head_dim']
    target['terminal_status'] = 'aborted'
    self.assertEqual(store.compare_resources(*ids)['status'], 'unavailable')


@unittest.skipUnless(
    os.environ.get('RUNNER_TEST_RUNTIME_ROOT')
    and importlib.util.find_spec('ai_edge_debugger_pytorch'),
    'Requires RUNNER_TEST_RUNTIME_ROOT and the configured PyTorch capture'
    ' environment',
)
class RealPyTorchTelemetryTests(unittest.TestCase):
  """Random tiny local Llama proves the actual worker/export/import contract."""

  setUpClass = classmethod(
      worker_fixtures.PyTorchWorkerCaptureTest.setUpClass.__func__
  )
  tearDownClass = classmethod(
      worker_fixtures.PyTorchWorkerCaptureTest.tearDownClass.__func__
  )
  request = worker_fixtures.PyTorchWorkerCaptureTest.request

  def test_actual_all_forward_resources_survive_job_deletion(self):
    from model_debugger_runner.pytorch_worker import execute

    self.root = self.root.resolve()
    record = {
        'id': 'actual',
        'name': 'Random tiny Llama',
        'model': 'Random tiny Llama',
        'runs': [
            {'id': run, 'runtime': 'PyTorch'} for run in ('ref', 'target')
        ],
    }
    registry = SimpleNamespace(
        root=self.root,
        capture_store=lambda _: SessionStore(self.root / record['capture']),
    )
    job = self.root / 'sessions/actual/jobs/first'
    job.mkdir(parents=True)
    results, originals = {}, set()
    for run in ('ref', 'target'):
      request = self.request(f'sessions/actual/jobs/first/{run}', turn=1)
      results[run] = execute(request, lambda *args, **kwargs: None)
      index = json.loads((job / run / 'export/capture_index.json').read_text())
      originals.update(
          (job / run / item['path']).read_bytes() for item in index['resources']
      )
    record['capture'] = publish_capture(
        registry,
        record,
        job,
        results,
        {run: {'name': 'Random tiny Llama'} for run in results},
    )
    store = SessionStore(self.root / record['capture'])
    data = store.telemetry()
    self.assertEqual(len(data['forwards']), 6)
    self.assertTrue(all(row['comparison_eligible'] for row in data['forwards']))
    self.assertEqual(
        [batch['forward_id'] for batch in store.session['batches']], [0, 1, 2]
    )
    self.assertEqual(
        {
            (store.root / item['path']).read_bytes()
            for item in data['resources']
        },
        originals,
    )
    for batch in range(3):
      self.assertTrue(
          all(
              row['status'] == 'ok'
              for row in store.compare_batch(batch)['rows']
          )
      )
    terminal = [
        next(
            item['id']
            for item in data['resources']
            if item['run'] == run
            and item['scope'] == 'kv'
            and item['moment'] == 'terminal'
            and item['kind'] == 'key'
            and item['layer'] == 0
        )
        for run in ('ref', 'target')
    ]
    self.assertEqual(store.compare_resources(*terminal)['status'], 'ok')
    self.assertEqual(
        store.compare_resources(*terminal)['metrics']['Max abs error']['value'],
        0.0,
    )
    shutil.rmtree(job)
    reloaded = SessionStore(store.root)
    for item in reloaded.telemetry()['resources']:
      self.assertEqual(
          list(reloaded.load_resource(item['id']).shape), item['shape']
      )
    self.assertEqual(reloaded.compare_resources(*terminal)['status'], 'ok')
    self.assertEqual(
        [
            row['generated_token_count']
            for row in reloaded.telemetry()['generations']
        ],
        [3, 3],
    )


if __name__ == '__main__':
  unittest.main()
