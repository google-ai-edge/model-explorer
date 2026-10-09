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

"""Local model provenance and runtime dispatch without loading model weights."""

import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from model_debugger_contracts.model_identity import describe_model
from model_debugger_runner.pytorch_environment import probe
from model_explorer_debugger.fsutil import atomic_json
from model_explorer_debugger.jobs import JobManager
from model_explorer_debugger.runtime.tap_profiles import PROFILE_ID
from model_explorer_debugger.session_registry import SessionRegistry

CAPABILITIES = {
    'available': True,
    'runtimes': [
        {
            'id': 'LiteRT-LM',
            'available': True,
            'backends': ['CPU', 'GPU'],
            'reason': '',
        },
        {'id': 'PyTorch', 'available': True, 'backends': ['CPU'], 'reason': ''},
    ],
}


class PyTorchRuntimeTest(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    self.model = self.root / 'synthetic-hf-model'
    self.model.mkdir()
    # These synthetic bytes test registration identities, not model inference.
    (self.model / 'config.json').write_text('{"model_type":"synthetic"}')
    (self.model / 'tokenizer.json').write_text('{"synthetic":true}')
    (self.model / 'model.safetensors').write_bytes(
        b'synthetic-checkpoint-bytes'
    )
    self.registry = SessionRegistry(
        self.root / 'workspace',
        self.root / 'litert-runtime',
        self.root / 'pytorch-runtime',
    )
    self.artifact = self.registry.register_pytorch_model(self.model)
    self.litert = self.root / 'synthetic.litertlm'
    self.litert.write_bytes(b'synthetic-litert-model')
    self.litert_artifact = self.registry.register_model(self.litert)

  def run_config(self, run='ref', **changes):
    return dict(
        id=run,
        runtime='PyTorch',
        backend='CPU',
        device='macos:00000000-0000-4000-8000-00000000000'
        + ('1' if run == 'ref' else '2'),
        artifact=self.artifact,
        source='registered',
        **changes,
    )

  def create(self, runs=None, **changes):
    return self.registry.manage(
        'create',
        dict(
            name='Test',
            model='Synthetic registration fixture',
            runs=runs or [self.run_config('ref'), self.run_config('target')],
            **changes,
        ),
    )

  def test_registration_persists_exact_file_manifest_and_runtime(self):
    artifact = self.registry.resolve_run(self.run_config())
    files, checksum = describe_model(self.model)
    self.assertEqual(artifact['runtime'], 'PyTorch')
    self.assertEqual(artifact['model_files'], files)
    self.assertEqual(artifact['sha256'], checksum)
    reopened = SessionRegistry(self.registry.root)
    self.assertEqual(
        reopened.resolve_run(self.run_config())['model_files'], files
    )
    self.assertEqual(
        [entry['path'] for entry in files],
        sorted(entry['path'] for entry in files),
    )

  def test_runtime_and_artifact_cannot_be_mixed_or_browser_paths_used(self):
    for config in (
        {**self.run_config(), 'artifact': str(self.model)},
        {**self.run_config(), 'artifact': self.litert_artifact},
        {**self.run_config(), 'runtime': 'LiteRT-LM'},
        {**self.run_config(), 'source': 'huggingface'},
    ):
      with self.subTest(config=config), self.assertRaises(ValueError):
        self.registry.resolve_run(config)

  def test_pytorch_options_are_explicit(self):
    for precision in ('', 'default', 'float32', 'float16', 'bfloat16'):
      self.registry.resolve_run(self.run_config(precision=precision))
    for changes in (
        {'backend': 'GPU'},
        {'backend': 'CPU', 'device': 'ios:phone'},
        {'precision': 'int8'},
        {'cpuThreads': '4', 'backend': 'MPS'},
        {'forceF32': True},
    ):
      with self.subTest(changes=changes), self.assertRaises(ValueError):
        self.registry.resolve_run({**self.run_config(), **changes})
    with patch.object(self.registry, 'capabilities', return_value=CAPABILITIES):
      self.registry.require_runtime(self.run_config())
      with self.assertRaisesRegex(ValueError, 'unavailable'):
        self.registry.require_runtime({**self.run_config(), 'backend': 'MPS'})

  def test_configured_worker_commands_do_not_cross_environments(self):
    request = self.root / 'request.json'
    for operation in ('initialize', 'generate'):
      with self.assertRaisesRegex(ValueError, 'owned by Runner'):
        self.registry.worker_command(self.run_config(), operation, request)
    command = self.registry.worker_command(
        {'runtime': 'LiteRT-LM'}, 'prepare', request
    )
    self.assertEqual(
        command[0], str(self.registry.runtime_root / '.venv/bin/python')
    )
    self.assertEqual(command[2], 'model_explorer_debugger.runtime.prepare_tap')

  def test_pytorch_initialization_uses_resident_workers_and_provenance(self):
    record = self.create()
    manager = JobManager(self.registry)
    self.addCleanup(manager.close)
    requests = []

    def execute(request, emit, cancelled):
      requests.append(request)
      return {
          'debug_enabled': True,
          'captured_tensors': 0,
          'worker_pid': 123,
          'model_load_count': 1,
      }

    with (
        patch.object(self.registry, 'capabilities', return_value=CAPABILITIES),
        patch(
            'model_explorer_debugger.prepare_worker.subprocess.Popen'
        ) as launch,
        patch.object(manager.runners, 'execute', side_effect=execute) as mac,
    ):
      job = manager.start(
          record['id'], 'initialize', {'request_id': 'initialize'}
      )
      manager.thread.join(5)
    self.assertEqual(manager.get(job['id'])['status'], 'completed')
    self.assertEqual(len(requests), 2)
    self.assertEqual(
        {request['run']['id'] for request in requests}, {'ref', 'target'}
    )
    self.assertTrue(
        all(
            request['model_files'] == describe_model(self.model)[0]
            for request in requests
        )
    )
    self.assertTrue(
        all(
            request['pytorch_root'] == str(self.registry.pytorch_root)
            for request in requests
        )
    )
    self.assertEqual(mac.call_count, 2)
    launch.assert_not_called()
    self.assertTrue(
        all(
            request['turn'] == 1 and request['request_id'] == job['id']
            for request in requests
        )
    )
    self.assertTrue(self.registry.get(record['id'])['initialized'])

  def test_pytorch_prepare_is_unnecessary_and_does_not_start_job(self):
    record = self.create()
    manager = JobManager(self.registry)
    self.addCleanup(manager.close)
    with patch.object(self.registry, 'capabilities', return_value=CAPABILITIES):
      with self.assertRaisesRegex(ValueError, 'not required'):
        manager.start(record['id'], 'prepare', {'request_id': 'prepare'})
    self.assertEqual(manager.jobs, {})

  def test_failed_session_publication_preserves_previous_capture_pointer(self):
    record = self.create()
    previous = self.registry.update(
        record['id'], capture='saved-before', has_capture=True
    )
    with patch.object(
        self.registry,
        '_save',
        side_effect=OSError('registry publication failed'),
    ):
      with self.assertRaisesRegex(OSError, 'publication failed'):
        self.registry.update(
            record['id'], capture='new-uncommitted', initialized=True
        )
    self.assertEqual(self.registry.get(record['id']), previous)
    reopened = SessionRegistry(self.registry.root)
    self.assertEqual(reopened.get(record['id'])['capture'], 'saved-before')

  def test_generation_failure_ends_chat_but_debug_failure_keeps_text(self):
    for fault in ('target', 'publish'):
      with self.subTest(fault=fault):
        record = self.create()
        self.registry.update(
            record['id'],
            capture='saved-before',
            has_capture=True,
            initialized=True,
        )
        previous = dict(
            turns=[dict(n=1)],
            conversation=[
                dict(run=run, input='old', output='answer')
                for run in ('ref', 'target')
            ],
        )
        manager = JobManager(self.registry)
        manager.reserve_runs(record['id'], record)
        manager._phase(
            record['id'],
            'active',
            activeChatId=record['id'],
            newChatAllowed=True,
        )
        requests = []

        def execute(request, emit, cancelled):
          requests.append(request)
          if request['operation'] == 'preflight':
            return {}
          if fault == 'target' and request['run']['id'] == 'target':
            raise RuntimeError('target generation failed')
          if fault == 'cancel':
            manager.cancel(request['request_id'])
          return dict(output='partial output', captured_tensors=0)

        try:
          with (
              patch.object(
                  self.registry, 'capabilities', return_value=CAPABILITIES
              ),
              patch.object(
                  self.registry,
                  'capture_store',
                  return_value=SimpleNamespace(session=previous),
              ),
              patch.object(manager.runners, 'execute', side_effect=execute),
              patch.object(manager.runners, 'close_session') as released,
              patch.object(manager.runners, 'reset_session') as reset,
              patch(
                  'model_explorer_debugger.publication.publish_capture',
                  side_effect=OSError('capture publication failed'),
              ) as publish,
          ):
            job = manager.start(
                record['id'],
                'generate',
                dict(
                    request_id='turn-' + fault,
                    prompt='next',
                    max_output_tokens=2,
                ),
            )
            manager.thread.join(5)
            final = manager.get(job['id'])
            self.assertEqual(
                final['status'], 'completed' if fault == 'publish' else 'failed'
            )
            released.assert_not_called()
            if fault == 'target':
              reset.assert_called_once_with(record['id'])
            else:
              reset.assert_not_called()
            after = self.registry.get(record['id'])
            self.assertEqual(after['capture'], 'saved-before')
            self.assertTrue(after['has_capture'])
            self.assertEqual(after['initialized'], fault == 'publish')
            if fault == 'publish':
              self.assertEqual(after['successful_turns'][0]['n'], 2)
              self.assertEqual(
                  after['successful_turns'][0]['debug_data']['status'],
                  'unavailable',
              )
            else:
              self.assertFalse(after.get('successful_turns'))
            self.assertTrue(
                all(
                    r['turn'] == 2 and len(r['messages']) == 0 for r in requests
                )
            )
            self.assertEqual(publish.call_count, 1 if fault == 'publish' else 0)
            duplicate = manager.start(
                record['id'],
                'generate',
                dict(
                    request_id='turn-' + fault,
                    prompt='next',
                    max_output_tokens=2,
                ),
            )
            self.assertEqual(duplicate['id'], job['id'])
            if fault == 'target':
              with self.assertRaisesRegex(ValueError, 'read-only'):
                manager.start(
                    record['id'],
                    'generate',
                    dict(
                        request_id='retry-new',
                        prompt='next',
                        max_output_tokens=2,
                    ),
                )
        finally:
          manager.close()

  def test_pytorch_rejects_stale_taps_and_unknown_generation_controls(self):
    with self.assertRaisesRegex(ValueError, 'clear the LiteRT tap profile'):
      self.create(tap_profile=PROFILE_ID)
    self.assertEqual(self.registry.state['sessions'], [])
    record = self.create()
    with self.assertRaisesRegex(ValueError, 'Unsupported generation settings'):
      self.registry.manage(
          'chat-config',
          dict(
              id=record['id'],
              generation={'ref': {'stop': ['END']}, 'target': {}},
          ),
      )

  def test_mixed_prepare_changes_only_litert_run(self):
    runs = [
        self.run_config('ref'),
        dict(
            id='target',
            runtime='LiteRT-LM',
            backend='CPU',
            artifact=self.litert_artifact,
            source='registered',
        ),
    ]
    record = self.create(runs=runs, tap_profile=PROFILE_ID)
    manager = JobManager(self.registry)
    self.addCleanup(manager.close)
    requests = []

    def launch(command, **kwargs):
      request = json.loads(Path(command[-1]).read_text())
      requests.append(request)
      self.assertEqual(
          command[2], 'model_explorer_debugger.runtime.prepare_tap'
      )
      atomic_json(
          Path(request['output']) / 'result.json',
          dict(
              model=str(self.litert),
              manifest=str(self.root / 'manifest.json'),
              verification={'synthetic': True},
              reused=False,
          ),
      )
      return SimpleNamespace(poll=lambda: 0, returncode=0)

    with (
        patch.object(self.registry, 'capabilities', return_value=CAPABILITIES),
        patch(
            'model_explorer_debugger.prepare_worker.subprocess.Popen',
            side_effect=launch,
        ),
    ):
      job = manager.start(record['id'], 'prepare', {'request_id': 'prepare'})
      manager.thread.join(5)
    self.assertEqual(manager.get(job['id'])['status'], 'completed')
    self.assertEqual(
        [request['run']['runtime'] for request in requests], ['LiteRT-LM']
    )
    updated = self.registry.get(record['id'])
    self.assertEqual(updated['runs'][0], record['runs'][0])
    self.assertTrue(updated['tap_prepared'])
    self.assertEqual(set(updated['tap_preparation']), {'target'})

  def test_environment_probe_is_unavailable_when_interpreter_missing(self):
    result = probe(self.root / 'missing')
    self.assertFalse(result['available'])
    self.assertEqual(result['id'], 'PyTorch')

  def test_environment_probe_reports_only_detected_backends_and_handles_failure(
      self,
  ):
    runtime = self.root / 'probe-runtime'
    interpreter = runtime / '.venv/bin/python'
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    with patch(
        'model_debugger_runner.pytorch_environment.subprocess.run',
        return_value=SimpleNamespace(
            stdout='{"torch":"test","transformers":"test","backends":["CPU"]}\n'
        ),
    ) as process:
      result = probe(runtime)
      self.assertTrue(result['available'])
      self.assertEqual(result['backends'], ['CPU'])
      self.assertEqual(
          process.call_args.kwargs['env']['PYTHONPATH'], str(runtime)
      )
    with patch(
        'model_debugger_runner.pytorch_environment.subprocess.run',
        side_effect=subprocess.TimeoutExpired('probe', 45),
    ):
      self.assertFalse(probe(runtime)['available'])


if __name__ == '__main__':
  unittest.main()
