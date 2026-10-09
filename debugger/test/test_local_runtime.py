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

"""Behavioral checks for workspace isolation, provenance, and recovery."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from model_explorer_debugger.capture_importer import publish_capture
from model_explorer_debugger.capture_importer import trace_tokens
from model_explorer_debugger.fsutil import atomic_json
from model_explorer_debugger.jobs import JobManager
from model_explorer_debugger.session_registry import SessionRegistry
import numpy as np
from safetensors.numpy import save_file

LITERT_CAPABILITIES = {
    'available': True,
    'runtimes': [{
        'id': 'LiteRT-LM',
        'available': True,
        'backends': ['CPU', 'GPU'],
        'reason': '',
    }],
}


class LocalRuntimeTests(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    self.registry = SessionRegistry(self.root / 'workspace')
    self.model = self.root / 'model.litertlm'
    self.model.write_bytes(b'synthetic-model-for-tests')
    self.semantic = self.root / 'semantic.json'
    atomic_json(
        self.semantic,
        {
            'semantic_graph': [{
                'kind': 'decoder',
                'inputs': [{'id': 'input', 'shape': [1, 1, 2]}],
                'nodes': [{
                    'id': 'norm.attn.in',
                    'label': 'Norm',
                    'namespace': '',
                    'incomingEdges': [],
                }],
                'anchors': [],
            }],
            'layers': [{'def': 0, 'attrs': {}}],
        },
    )
    self.artifact = self.registry.register_model(
        self.model, semantic=self.semantic
    )

  def create(self, name='Session'):
    return self.registry.manage(
        'create',
        {
            'name': name,
            'model': 'Test model',
            'runs': [
                dict(
                    id=run,
                    runtime='LiteRT-LM',
                    backend=backend,
                    device='macos:00000000-0000-4000-8000-00000000000'
                    + ('1' if run == 'ref' else '2'),
                    artifact=self.artifact,
                    source='registered',
                )
                for run, backend in [('ref', 'CPU'), ('target', 'CPU')]
            ],
        },
    )

  def test_arbitrary_browser_path_is_not_a_registered_model(self):
    config = dict(runtime='LiteRT-LM', artifact=str(self.model), source='path')
    with self.assertRaisesRegex(ValueError, 'registered'):
      self.registry.resolve_run(config)

  def test_unsupported_configuration_is_not_silently_ignored(self):
    with self.assertRaisesRegex(ValueError, 'forceF32'):
      self.registry.resolve_run(
          dict(runtime='LiteRT-LM', artifact=self.artifact, forceF32=True)
      )

  def test_delete_restore_and_restart_preserve_independent_sessions(self):
    a, b = self.create('A'), self.create('B')
    self.registry.manage('delete', {'id': a['id']})
    reopened = SessionRegistry(self.registry.root)
    self.assertEqual([s['name'] for s in reopened.listing()['sessions']], ['B'])
    reopened.manage('restore', {'id': a['id']})
    self.assertEqual(len(reopened.listing()['sessions']), 2)
    with self.assertRaisesRegex(ValueError, 'No completed capture'):
      reopened.capture_store(b['id'])

  def test_restart_invalidates_loaded_state(self):
    record = self.create()
    self.registry.update(record['id'], initialized=True, status='running')
    reopened = SessionRegistry(self.registry.root)
    self.assertFalse(reopened.get(record['id'])['initialized'])
    self.assertEqual(reopened.get(record['id'])['status'], 'failed')

  def test_duplicate_has_no_capture_or_native_state(self):
    record = self.create()
    self.registry.update(
        record['id'],
        has_capture=True,
        capture='some-capture',
        initialized=True,
        status='saved',
    )
    copy = self.registry.manage('duplicate', {'id': record['id']})
    self.assertFalse(copy['has_capture'])
    self.assertFalse(copy['initialized'])
    self.assertNotIn('capture', copy)

  def test_chat_context_has_independent_capture_and_survives_restart(self):
    parent = self.create()
    self.registry.update(
        parent['id'],
        has_capture=True,
        capture='parent-capture',
        initialized=True,
        status='saved',
    )
    child = self.registry.manage(
        'chat', {'id': parent['id'], 'name': 'New chat 1'}
    )
    self.assertEqual(child['parent_session_id'], parent['id'])
    self.assertFalse(child['has_capture'])
    self.assertFalse(child['initialized'])
    self.assertNotIn('capture', child)
    self.assertEqual(child['runs'], parent['runs'])
    reopened = SessionRegistry(self.registry.root)
    self.assertEqual(
        reopened.get(child['id'])['parent_session_id'], parent['id']
    )
    self.assertEqual(reopened.get(parent['id'])['capture'], 'parent-capture')

  def test_new_chat_configuration_is_atomic(self):
    parent = self.create()
    before = self.registry.listing()['sessions']
    with self.assertRaises(ValueError):
      self.registry.manage(
          'chat',
          {
              'id': parent['id'],
              'generation': {'ref': {'maxOutputTokens': 0}, 'target': {}},
          },
      )
    self.assertEqual(self.registry.listing()['sessions'], before)
    child = self.registry.manage(
        'chat',
        {
            'id': parent['id'],
            'generation': {
                'ref': {'systemPrompt': 'Reference', 'temperature': 0.4},
                'target': {'systemPrompt': 'Target', 'temperature': 0.6},
            },
        },
    )
    reopened = SessionRegistry(self.registry.root).get(child['id'])
    self.assertEqual(reopened['generation']['ref']['systemPrompt'], 'Reference')
    self.assertEqual(reopened['generation']['target']['temperature'], 0.6)
    self.assertFalse(reopened['initialized'])

  def test_chat_generation_validation_and_capture_immutability(self):
    record = self.create()
    values = {
        'ref': {
            'temperature': 0.2,
            'maxOutputTokens': 16,
            'systemPrompt': 'Answer briefly',
        },
        'target': {'thinking': 'off', 'topK': 1},
    }
    saved = self.registry.manage(
        'chat-config', {'id': record['id'], 'generation': values}
    )
    self.assertEqual(saved['generation']['ref']['maxOutputTokens'], 16)
    self.assertFalse(saved['initialized'])
    for value in (0, 257, float('nan'), True):
      with self.assertRaises(ValueError):
        self.registry.manage(
            'chat-config',
            {
                'id': record['id'],
                'generation': {'ref': {'maxOutputTokens': value}, 'target': {}},
            },
        )
    self.registry.update(record['id'], has_capture=True)
    with self.assertRaisesRegex(ValueError, 'immutable'):
      self.registry.manage(
          'chat-config', {'id': record['id'], 'generation': values}
      )

  def test_trace_chunks_are_not_invented_tokens(self):
    trace = self.root / 'trace.jsonl'
    trace.write_text(
        json.dumps({'token_ids': [[1, 2]], 'texts': ['two tokens']})
        + '\n'
        + json.dumps({'token_ids': [[3]], 'texts': ['!']})
        + '\n'
    )
    tokens = trace_tokens(trace)
    self.assertEqual([token['id'] for token in tokens], [1, 2, 3])
    # The released chunk proves both IDs, but not their individual text.
    self.assertEqual([token['text'] for token in tokens], [None, None, '!'])
    self.assertEqual([token['step'] for token in tokens], [0, 1, 2])

  def fixture_capture(self, record, divergent=False):
    job = (
        self.registry.root
        / 'sessions'
        / record['id']
        / 'jobs'
        / 'synthetic-job'
    )
    results = {}
    artifacts = {
        run: self.registry.resolve_run(r)
        for run, r in [(r['id'], r) for r in record['runs']]
    }
    for run in ['ref', 'target']:
      export = job / run / 'export'
      (export / 'tensors').mkdir(parents=True)
      indexed = []
      for i, phase in enumerate(['prefill', 'decode']):
        array = np.array([[1, 2]], dtype=np.float32)
        save_file({'tap': array}, export / f'tensors/{i}.safetensors')
        indexed.append(
            dict(
                signature=phase,
                section_offset=100,
                subgraph=i,
                op=1,
                output=0,
                tensor=2,
                tensor_name='model/layer_0/pre_attention_norm/output',
                output_name='tap',
                phase=phase,
                step=15 + i,
                path=f'tensors/{i}.safetensors',
                format='safetensors',
                key='tap',
                shape=[1, 2],
                dtype='float32',
            )
        )
      atomic_json(
          export / 'capture_index.json',
          {'format_version': 2, 'tensor_root': 'export', 'tensors': indexed},
      )
      results[run] = dict(
          model_sha256='same',
          input='hello',
          output='hello',
          messages=[]
          if not divergent or run == 'ref'
          else [{'role': 'assistant', 'content': 'different'}],
      )
    return job, results, artifacts

  def test_verified_prefill_pairs_but_decode_is_not_assumed_comparable(self):
    record = self.create()
    job, results, artifacts = self.fixture_capture(record)
    capture = publish_capture(self.registry, record, job, results, artifacts)
    self.registry.update(record['id'], capture=capture, has_capture=True)
    store = self.registry.capture_store(record['id'])
    self.assertEqual(store.compare_batch(0)['rows'][0]['status'], 'ok')
    self.assertEqual(
        store.compare_batch(1)['rows'][0]['status'], 'sample_mismatch'
    )
    self.assertEqual(store.tensors[0]['runtime_step'], 15)
    self.assertIsNone(store.tensors[0]['token_range'])
    self.assertFalse(
        any(
            'batch' in token
            for c in store.session['conversation']
            for token in c['tokens']
        )
    )

  def test_capture_turn_preserves_gap_after_text_only_turns(self):
    record = self.create()
    job, results, artifacts = self.fixture_capture(record)
    capture = publish_capture(
        self.registry,
        {**record, 'publication_turn': 3},
        job,
        results,
        artifacts,
    )
    self.registry.update(record['id'], capture=capture, has_capture=True)
    store = self.registry.capture_store(record['id'])
    self.assertEqual([turn['n'] for turn in store.session['turns']], [3])
    self.assertEqual(
        {row['turn'] for row in store.session['conversation']}, {3}
    )
    self.assertTrue(all(tensor['turn'] == 3 for tensor in store.tensors))

  def test_different_history_blocks_prefill_numeric_pairing(self):
    record = self.create()
    job, results, artifacts = self.fixture_capture(record, divergent=True)
    capture = publish_capture(self.registry, record, job, results, artifacts)
    self.registry.update(record['id'], capture=capture, has_capture=True)
    self.assertEqual(
        self.registry.capture_store(record['id']).compare_batch(0)['rows'][0][
            'status'
        ],
        'sample_mismatch',
    )

  def test_multiple_selected_outputs_of_one_operator_are_all_imported(self):
    record = self.create()
    job, results, artifacts = self.fixture_capture(record)
    for run in ('ref', 'target'):
      path = job / run / 'export/capture_index.json'
      index = json.loads(path.read_text())
      index['tensors'].append({
          **index['tensors'][0],
          'output': 1,
          'tensor': 3,
          'output_name': 'tap_second',
      })
      atomic_json(path, index)
    capture = publish_capture(self.registry, record, job, results, artifacts)
    self.registry.update(record['id'], capture=capture, has_capture=True)
    store = self.registry.capture_store(record['id'])
    self.assertEqual(len(store.tensors), 6)
    node = store.execution['executions'][0]['graphs'][0]['nodes'][0]
    self.assertEqual({o['id'] for o in node['outputsMetadata']}, {'0', '1'})

  def test_http_capture_routes_are_scoped_to_session_and_loopback(self):
    from inline_asgi import client

    a, b = self.create('Captured'), self.create('Empty')
    job, results, artifacts = self.fixture_capture(a)
    capture = publish_capture(self.registry, a, job, results, artifacts)
    self.registry.update(a['id'], capture=capture, has_capture=True)
    with client(registry=self.registry) as http:
      self.assertEqual(
          http.get('/api/session?session_id=' + a['id']).json()['name'],
          'Captured',
      )
      for path in [
          'session',
          'session?session_id=' + b['id'],
          'session?session_id=../outside',
      ]:
        self.assertEqual(http.get('/api/' + path).status_code, 400)
      self.assertEqual(
          http.get(
              '/api/sessions', headers={'Host': 'unrelated.example'}
          ).status_code,
          403,
      )

  def test_shutdown_reaps_a_native_process(self):
    import subprocess
    import sys

    record = self.create()
    manager = JobManager(self.registry)
    with (
        patch.object(
            self.registry, 'capabilities', return_value=LITERT_CAPABILITIES
        ),
        patch('threading.Thread.start'),
    ):
      job = manager.start(
          record['id'], 'initialize', {'request_id': 'shutdown'}
      )
    manager.thread = None
    process = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(30)']
    )
    manager.processes[job['id']] = process
    try:
      manager.close()
      self.assertIsNotNone(process.poll())
      self.assertTrue((manager.directory(job) / 'cancel').exists())
      with self.assertRaisesRegex(ValueError, 'shutting down'):
        manager.start(
            record['id'], 'initialize', {'request_id': 'after-shutdown'}
        )
    finally:
      if process.poll() is None:
        process.kill()
        process.wait()

  def test_http_long_prompt_reaches_runtime_without_truncation(self):
    from inline_asgi import client

    record = self.create()
    prompt = '界 token ' * 131072
    with (
        client(registry=self.registry) as http,
        patch.object(
            JobManager, 'start', return_value={'accepted': True}
        ) as start,
    ):
      response = http.post(
          f'/api/sessions/{record["id"]}/turns',
          json={'prompt': prompt, 'max_output_tokens': 1},
      )
      self.assertEqual(response.status_code, 200, response.text)
      self.assertEqual(start.call_args.args[2]['prompt'], prompt)

  def test_long_prompt_is_preserved_and_utf8_limit_is_explicit(self):
    record = self.create()
    self.registry.update(record['id'], initialized=True)
    manager = JobManager(self.registry)
    manager.reserve_runs(record['id'], record)
    manager._phase(
        record['id'], 'active', activeChatId=record['id'], newChatAllowed=True
    )
    prompt = '上下文 ' + ('token ' * 131072)
    with (
        patch.object(
            self.registry, 'capabilities', return_value=LITERT_CAPABILITIES
        ),
        patch('threading.Thread.start'),
    ):
      job = manager.start(
          record['id'],
          'generate',
          {'request_id': 'long', 'prompt': prompt, 'max_output_tokens': 1},
      )
      self.assertEqual(job['status'], 'running')
      manager.cancel(job['id'])
    # UTF-8 byte count, not code points: Chinese text crosses the bound sooner.
    manager = JobManager(self.registry)
    self.registry.update(record['id'], initialized=True)
    manager.reserve_runs(record['id'], record)
    manager._phase(
        record['id'], 'active', activeChatId=record['id'], newChatAllowed=True
    )
    with (
        patch.object(
            self.registry, 'capabilities', return_value=LITERT_CAPABILITIES
        ),
        patch('threading.Thread.start'),
    ):
      with self.assertRaisesRegex(ValueError, '4 MiB'):
        manager.start(
            record['id'],
            'generate',
            {
                'request_id': 'too-large',
                'prompt': '界' * (4 * 1024 * 1024 // 3 + 1),
                'max_output_tokens': 1,
            },
        )

  def test_idempotency_rejects_changed_payload_and_preserves_job(self):
    record = self.create()
    manager = JobManager(self.registry)
    with (
        patch.object(
            self.registry, 'capabilities', return_value=LITERT_CAPABILITIES
        ),
        patch('threading.Thread.start'),
    ):
      first = manager.start(record['id'], 'initialize', {'request_id': 'one'})
      again = manager.start(record['id'], 'initialize', {'request_id': 'one'})
      self.assertEqual(first['id'], again['id'])
      with self.assertRaisesRegex(ValueError, 'different request'):
        manager.start(
            record['id'], 'initialize', {'request_id': 'one', 'changed': True}
        )
      with self.assertRaisesRegex(ValueError, 'active'):
        manager.start(record['id'], 'initialize', {'request_id': 'two'})
      with self.assertRaisesRegex(ValueError, 'cannot be cancelled'):
        manager.cancel(first['id'])
      self.assertFalse((manager.directory(first) / 'cancel').exists())
      self.assertTrue(manager.events(first['id'], 0))
      self.assertEqual(manager.events(first['id'], 999), [])

  def test_explicit_tap_profile_is_prepared_inside_initialize_chain(self):
    from model_explorer_debugger.runtime.tap_profiles import PROFILE_ID

    record = self.create()
    self.assertEqual(record['tap_profile'], '')
    with self.assertRaisesRegex(ValueError, 'Unknown tensor capture profile'):
      self.registry.manage(
          'update', {**record, 'tap_profile': 'arbitrary-model'}
      )
    record = self.registry.manage(
        'update', {**record, 'tap_profile': PROFILE_ID}
    )
    manager = JobManager(self.registry)
    with (
        patch.object(
            self.registry, 'capabilities', return_value=LITERT_CAPABILITIES
        ),
        patch('threading.Thread.start'),
    ):
      job = manager.start(record['id'], 'initialize', {'request_id': 'init'})
      self.assertEqual(job['status'], 'initializing')
      with self.assertRaisesRegex(ValueError, 'Stop the active task'):
        self.registry.manage('update', record)
      reopened = SessionRegistry(self.registry.root)
      self.assertEqual(reopened.get(record['id'])['status'], 'failed')
      self.assertFalse(reopened.get(record['id'])['tap_prepared'])
      self.assertEqual(reopened.get(record['id'])['runs'], record['runs'])

  def test_prepare_deduplicates_and_publishes_both_runs_only_after_success(
      self,
  ):
    from model_explorer_debugger.runtime.tap_profiles import PROFILE_ID

    record = self.create()
    record = self.registry.manage(
        'update', {**record, 'tap_profile': PROFILE_ID}
    )
    self.registry.runtime_root = self.root
    manager = JobManager(self.registry)
    result = {
        'model': str(self.model),
        'manifest': str(self.root / 'manifest.json'),
        'verification': {'synthetic': True},
        'reused': False,
    }

    def launch(command, **kwargs):
      request = json.loads(Path(command[-1]).read_text())
      atomic_json(Path(request['output']) / 'result.json', result)
      from unittest.mock import Mock

      return Mock(poll=lambda: 0, returncode=0)

    with (
        patch.object(
            self.registry, 'capabilities', return_value=LITERT_CAPABILITIES
        ),
        patch(
            'model_explorer_debugger.prepare_worker.subprocess.Popen',
            side_effect=launch,
        ) as process,
    ):
      job = manager.start(record['id'], 'prepare', {'request_id': 'prepare'})
      manager.thread.join(5)
    self.assertEqual(manager.get(job['id'])['status'], 'completed')
    self.assertEqual(process.call_count, 1)
    self.assertTrue(self.registry.get(record['id'])['tap_prepared'])
    self.assertTrue(
        any(
            e.get('message') == 'Reusing Reference debug model'
            for e in manager.events(job['id'])
        )
    )

  def test_prepare_failure_keeps_original_artifacts_and_allows_retry(self):
    from model_explorer_debugger.runtime.tap_profiles import PROFILE_ID

    record = self.create()
    record = self.registry.manage(
        'update', {**record, 'tap_profile': PROFILE_ID}
    )
    self.registry.runtime_root = self.root
    manager = JobManager(self.registry)
    with (
        patch.object(
            self.registry, 'capabilities', return_value=LITERT_CAPABILITIES
        ),
        patch(
            'model_explorer_debugger.prepare_worker.subprocess.Popen',
            side_effect=OSError('preparation failed'),
        ),
    ):
      job = manager.start(record['id'], 'prepare', {'request_id': 'failed'})
      manager.thread.join(5)
    self.assertEqual(manager.get(job['id'])['status'], 'failed')
    self.assertEqual(self.registry.get(record['id'])['runs'], record['runs'])
    self.assertFalse(self.registry.get(record['id'])['tap_prepared'])
    self.assertIsNone(manager.active)
    self.assertEqual(self.registry.manage('update', record)['status'], 'draft')

  def custom_scan(self):
    scan = dict(
        status='completed',
        model_sha256=self.registry.resolve_run(
            {'runtime': 'LiteRT-LM', 'artifact': self.artifact}
        )['sha256'],
        section_sha256='section-digest',
        points=[
            {'id': 'decode:7:0', 'selectable': True},
            {'id': 'decode:8:0', 'selectable': False},
        ],
        recommended=[],
    )
    atomic_json(
        self.registry.root / 'tap-scans' / self.artifact / 'result.json', scan
    )
    return scan

  def test_custom_selection_requires_exact_scanned_model_and_available_output(
      self,
  ):
    from model_explorer_debugger.runtime.tap_profiles import CUSTOM_PROFILE

    record = self.create()
    config = {
        **record,
        'tap_profile': CUSTOM_PROFILE,
        'tap_points': {self.artifact: ['decode:7:0']},
    }
    with self.assertRaisesRegex(ValueError, 'scanning'):
      self.registry.manage('update', config)
    scan = self.custom_scan()
    result = self.registry.manage('update', config)
    self.assertEqual(
        self.registry.resolve_taps(result['runs'][0], result)['points'],
        [scan['points'][0]],
    )
    for ids in (['decode:8:0'], ['decode:999:0'], ['decode:7:0'] * 2, []):
      with self.assertRaises(ValueError):
        self.registry.manage(
            'update', {**config, 'tap_points': {self.artifact: ids}}
        )
    scan['model_sha256'] = 'a different model'
    atomic_json(
        self.registry.root / 'tap-scans' / self.artifact / 'result.json', scan
    )
    with self.assertRaisesRegex(ValueError, 'exact model'):
      self.registry.manage('update', config)

  def test_changed_model_requires_its_own_custom_selection(self):
    from model_explorer_debugger.runtime.tap_profiles import CUSTOM_PROFILE

    record = self.create()
    self.custom_scan()
    other = self.root / 'other.litertlm'
    other.write_bytes(b'different synthetic model')
    record['runs'][1]['artifact'] = self.registry.register_model(other)
    with self.assertRaisesRegex(ValueError, 'select capture points'):
      self.registry.manage(
          'update',
          {
              **record,
              'tap_profile': CUSTOM_PROFILE,
              'tap_points': {self.artifact: ['decode:7:0']},
          },
      )

  def test_scanner_queues_once_and_rejects_stale_cache_or_unknown_artifact(
      self,
  ):
    from model_explorer_debugger.tap_scans import TapScans

    scans = TapScans(self.registry)
    self.addCleanup(scans.close)
    with (
        patch.object(
            self.registry, 'capabilities', return_value={'tap_profiles': [{}]}
        ),
        patch.object(scans.pool, 'submit') as submit,
    ):
      self.assertEqual(scans.start(self.artifact)['status'], 'scanning')
      self.assertEqual(scans.start(self.artifact)['status'], 'scanning')
      self.assertEqual(submit.call_count, 1)
    with self.assertRaisesRegex(ValueError, 'Unknown registered model'):
      scans.start('../outside')
    scans.pending.clear()
    scan = self.custom_scan()
    stat = self.model.stat()
    scan['source_stat'] = [stat.st_size, stat.st_mtime_ns]
    atomic_json(
        self.registry.root / 'tap-scans' / self.artifact / 'result.json', scan
    )
    self.assertEqual(scans.get(self.artifact)['status'], 'completed')
    self.model.write_bytes(b'changed source')
    self.assertEqual(scans.get(self.artifact)['status'], 'failed')


if __name__ == '__main__':
  unittest.main()
