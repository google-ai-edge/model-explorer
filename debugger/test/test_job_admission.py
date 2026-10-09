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

"""Fault injection at the pre-Runner admission interface; no model execution."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch
from uuid import uuid4

from model_explorer_debugger.jobs import JobManager
from model_explorer_debugger.session_registry import SessionRegistry


class FakeRunners:

  def __init__(self):
    self.identity = {'id': 'fixture-server', 'name': 'Simulated Runner'}
    self.on_lifecycle = None
    self.calls = []
    self.resets = []
    self.block = None
    self.entered = threading.Event()

  def resolve_runs(self, runs):
    return [{**run, '_runner_slot': run['id']} for run in runs]

  def device_key(self, device):
    return device

  def execute(self, request, emit, cancelled):
    self.calls.append((request['operation'], request['run']['id']))
    self.entered.set()
    if self.block is not None and not self.block.wait(3):
      raise TimeoutError('Test did not release simulated Runner')
    if request['operation'] == 'preflight':
      return {}
    return dict(
        output='simulated answer',
        generation_status='completed',
        output_confirmed=True,
        debug_data={'status': 'unavailable', 'error': 'Test has no capture'},
    )

  def reset_session(self, identity):
    self.resets.append(identity)

  def close_session(self, identity):
    pass

  def abandon_session(self, identity):
    pass

  def close(self):
    pass


class JobAdmissionTests(unittest.TestCase):
  faults = (
      'mkdir',
      'job_write',
      'event_partial',
      'thread_construct',
      'thread_start',
      'thread_started_then_failed',
      'workspace_save',
  )

  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name)
    self.registry = SessionRegistry(self.root / 'workspace')
    model = self.root / 'fixture.litertlm'
    model.write_bytes(b'synthetic model; never executed')
    artifact = self.registry.register_model(model)
    caps = patch.object(
        self.registry,
        'capabilities',
        return_value={
            'available': True,
            'runtimes': [
                {'id': 'LiteRT-LM', 'available': True, 'backends': ['CPU']}
            ],
        },
    )
    caps.start()
    self.addCleanup(caps.stop)
    self.record = self.registry.manage(
        'create',
        dict(
            name='Session',
            model='Fixture',
            runs=[
                dict(
                    id=role,
                    device='macos:' + str(uuid4()),
                    runtime='LiteRT-LM',
                    backend='CPU',
                    artifact=artifact,
                    source='registered',
                )
                for role in ('ref', 'target')
            ],
        ),
    )
    self.runners = FakeRunners()
    with patch(
        'model_explorer_debugger.runtime.runners.Runners',
        return_value=self.runners,
    ):
      self.manager = JobManager(self.registry)
    self.addCleanup(self.manager.close)

  def finish(self, job):
    thread = self.manager.threads[job['id']]
    thread.join(3)
    self.assertFalse(thread.is_alive(), 'Job did not finish')
    result = self.manager.get(job['id'])
    self.assertEqual(result['status'], 'completed', result)
    return result

  def initialize(self):
    return self.finish(
        self.manager.start(
            self.record['id'], 'initialize', {'request_id': 'initialize'}
        )
    )

  def snapshot(self):
    return dict(
        workspace=self.registry.path.read_bytes(),
        registry=deepcopy(self.registry.state),
        executions=deepcopy(self.manager.executions),
        devices=dict(self.manager.device_owners),
        jobs=deepcopy(self.manager.jobs),
        active=dict(self.manager.active_jobs),
        threads=dict(self.manager.threads),
        thread=self.manager.thread,
        paths=set(self.registry.root.rglob('job.json')),
        calls=list(self.runners.calls),
        resets=list(self.runners.resets),
    )

  def assert_unchanged(self, before, *, files=True):
    after = self.snapshot()
    if not files:
      after.pop('paths')
      before = {key: value for key, value in before.items() if key != 'paths'}
    self.assertEqual(after, before)

  @contextmanager
  def inject(self, fault):
    started = []
    real_thread = threading.Thread
    real_start = real_thread.start
    real_mkdir = Path.mkdir
    original_event = self.manager.record_event

    def mkdir(path, *args, **kwargs):
      if path.parent.name == 'jobs':
        raise OSError('injected mkdir')
      return real_mkdir(path, *args, **kwargs)

    def partial_event(job, *args, **kwargs):
      original_event(job, *args, **kwargs)
      with (self.manager.directory(job) / 'events.jsonl').open('a') as file:
        file.write('{"incomplete":')
      raise OSError('injected event_partial')

    def start(thread):
      if thread.name.startswith('job-'):
        if fault == 'thread_started_then_failed':
          real_start(thread)
          started.append(thread)
        raise RuntimeError('injected ' + fault)
      return real_start(thread)

    patches = {
        'mkdir': lambda: patch.object(Path, 'mkdir', mkdir),
        'job_write': lambda: patch(
            'model_explorer_debugger.job_admission.atomic_json',
            side_effect=OSError('injected job_write'),
        ),
        'event_partial': lambda: patch.object(
            self.manager, 'record_event', side_effect=partial_event
        ),
        'thread_construct': lambda: patch(
            'model_explorer_debugger.jobs.threading.Thread',
            side_effect=RuntimeError('injected thread_construct'),
        ),
        'thread_start': lambda: patch.object(real_thread, 'start', start),
        'thread_started_then_failed': lambda: patch.object(
            real_thread, 'start', start
        ),
        'workspace_save': lambda: patch.object(
            self.registry,
            '_save',
            side_effect=OSError('injected workspace_save'),
        ),
    }
    try:
      with patches[fault]():
        yield
    finally:
      for thread in started:
        thread.join(3)
        self.assertFalse(thread.is_alive(), 'Aborted worker did not exit')

  def rejection_matrix(self, identity, operation, payload):
    before = self.snapshot()
    for fault in self.faults:
      with self.subTest(operation=operation, fault=fault):
        with (
            self.inject(fault),
            self.assertRaisesRegex((OSError, RuntimeError), 'injected'),
        ):
          self.manager.start(identity, operation, payload)
        self.assert_unchanged(before)
    # The rejected request key remains usable; no hidden orphan is selected.
    return self.finish(self.manager.start(identity, operation, payload))

  def test_initialize_rejections_release_devices_and_allow_retry(self):
    self.rejection_matrix(
        self.record['id'], 'initialize', {'request_id': 'retry'}
    )
    self.assertEqual(
        sorted(self.runners.calls),
        [('initialize', 'ref'), ('initialize', 'target')],
    )

  def test_generate_rejections_preserve_resident_chat(self):
    self.initialize()
    self.rejection_matrix(
        self.record['id'],
        'generate',
        {'request_id': 'retry', 'prompt': 'hello', 'max_output_tokens': 1},
    )
    self.assertEqual(
        len(self.registry.get(self.record['id'])['successful_turns']), 1
    )
    self.assertEqual(self.runners.resets, [])

  def test_initialize_rejections_preserve_another_sessions_ownership(self):
    self.initialize()
    other = self.registry.manage('duplicate', {'id': self.record['id']})
    other = self.registry.manage(
        'update',
        {
            **other,
            'runs': [
                {**run, 'device': 'macos:' + str(uuid4())}
                for run in other['runs']
            ],
        },
    )
    self.rejection_matrix(other['id'], 'initialize', {'request_id': 'other'})
    self.assertEqual(
        self.manager.execution(self.record['id'])['phase'], 'active'
    )
    self.assertEqual(
        set(self.manager.device_owners.values()),
        {self.record['id'], other['id']},
    )

  def test_new_chat_rejections_preserve_previous_chat_until_commit(self):
    self.initialize()
    candidate = self.registry.stage_chat(
        {'id': self.record['id'], 'name': 'Candidate'}
    )
    self.rejection_matrix(candidate['id'], 'new_chat', {'request_id': 'retry'})
    self.assertEqual(
        self.manager.execution(self.record['id'])['activeChatId'],
        candidate['id'],
    )
    self.assertTrue(self.registry.get(self.record['id'])['read_only'])
    self.assertEqual(self.runners.resets, [self.record['id']])

  def test_chat_management_retry_keeps_previous_chat_until_admitted(self):
    self.initialize()
    previous = self.registry.get(self.record['id'])
    payload = {
        'id': self.record['id'],
        'request_id': 'new-chat',
        'name': 'Next Chat',
    }
    with (
        self.inject('thread_start'),
        self.assertRaisesRegex(RuntimeError, 'injected'),
    ):
      self.manager.manage('chat', payload)
    self.assertEqual(self.registry.get(self.record['id']), previous)
    self.assertEqual(len(self.registry.listing()['sessions']), 1)
    self.assertEqual(self.runners.resets, [])
    child = self.manager.manage('chat', payload)
    self.assertEqual(child['parent_session_id'], self.record['id'])
    self.assertEqual(self.runners.resets, [self.record['id']])

  def test_worker_cannot_execute_until_workspace_commit(self):
    original_save = self.registry._save

    def save():
      self.assertEqual(self.runners.calls, [])
      self.assertTrue(self.manager.thread.is_alive())
      original_save()
      self.assertEqual(self.runners.calls, [])

    with patch.object(self.registry, '_save', side_effect=save):
      job = self.manager.start(
          self.record['id'], 'initialize', {'request_id': 'commit'}
      )
    self.finish(job)
    self.assertEqual(len(self.runners.calls), 2)

  def test_concurrent_duplicate_requests_dispatch_one_paired_job(self):
    barrier = threading.Barrier(3)
    self.runners.block = threading.Event()

    def submit():
      barrier.wait(3)
      return self.manager.start(
          self.record['id'], 'initialize', {'request_id': 'duplicate'}
      )

    try:
      with ThreadPoolExecutor(max_workers=2) as callers:
        futures = [callers.submit(submit) for _ in range(2)]
        barrier.wait(3)
        first, second = [future.result(3) for future in futures]
      self.assertEqual(first['id'], second['id'])
      self.assertEqual(len(self.manager.jobs), 1)
      self.assertEqual(len(self.manager.threads), 1)
    finally:
      self.runners.block.set()
    self.finish(first)
    self.assertEqual(
        sorted(self.runners.calls),
        [('initialize', 'ref'), ('initialize', 'target')],
    )

  def test_rejected_job_with_failed_cleanup_is_ignored_after_restart(self):
    before = self.snapshot()
    with (
        self.inject('workspace_save'),
        patch(
            'model_explorer_debugger.jobs.shutil.rmtree',
            side_effect=OSError('injected cleanup'),
        ),
    ):
      with self.assertRaisesRegex(OSError, 'workspace_save.*cleanup'):
        self.manager.start(
            self.record['id'], 'initialize', {'request_id': 'retry'}
        )
    self.assert_unchanged(before, files=False)
    self.assertEqual(len(list(self.registry.root.rglob('job.json'))), 1)
    reopened = SessionRegistry(self.registry.root)
    with patch(
        'model_explorer_debugger.runtime.runners.Runners',
        return_value=FakeRunners(),
    ):
      recovered = JobManager(reopened)
    self.addCleanup(recovered.close)
    self.assertEqual(recovered.jobs, {})
    self.assertEqual(recovered.device_owners, {})

  def test_committed_pending_job_recovers_failed_without_replay(self):
    # Stop at the dispatch seam and copy the on-disk state, simulating a
    # process loss after acceptance but before the first worker event.
    with patch.object(self.manager, 'execute'):
      accepted = self.manager.start(
          self.record['id'], 'initialize', {'request_id': 'accepted'}
      )
      self.manager.threads[accepted['id']].join(3)
    recovery_root = self.root / 'restart'
    shutil.copytree(self.registry.root, recovery_root)
    reopened = SessionRegistry(recovery_root)
    runners = FakeRunners()
    with patch(
        'model_explorer_debugger.runtime.runners.Runners', return_value=runners
    ):
      recovered = JobManager(reopened)
    self.addCleanup(recovered.close)
    self.assertEqual(recovered.get(accepted['id'])['status'], 'failed')
    self.assertNotIn('admission', recovered.get(accepted['id']))
    self.assertEqual(runners.calls, [])

  def test_http_failure_retry_and_sse_keep_same_job_identity(self):
    from inline_asgi import client

    with patch(
        'model_explorer_debugger.jobs.JobManager', return_value=self.manager
    ):
      http = client(registry=self.registry)
    with http:

      def submit():
        return http.post(
            '/api/sessions/' + self.record['id'] + '/initialize',
            json={'request_id': 'http-retry'},
        )

      before = self.snapshot()
      with (
          self.inject('workspace_save'),
          self.assertLogs('model_explorer_debugger.http', 'ERROR'),
      ):
        self.assertEqual(submit().status_code, 500)
      self.assert_unchanged(before)
      accepted = submit().json()
      self.finish(accepted)
      self.assertEqual(submit().json()['id'], accepted['id'])
      self.assertEqual(len(self.runners.calls), 2)
      response = http.get('/api/jobs/' + accepted['id'] + '/events')
      self.assertTrue(
          response.headers['content-type'].startswith('text/event-stream')
      )
      self.assertIn('"status": "completed"', response.text)


if __name__ == '__main__':
  unittest.main()
