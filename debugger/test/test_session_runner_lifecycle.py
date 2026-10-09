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

"""Synthetic end-to-end Session ownership, heartbeat, and recovery tests."""

import collections
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from model_explorer_debugger.jobs import JobManager
from model_explorer_debugger.runtime import runner_channel
from model_explorer_debugger.runtime.runner_channel import pipeline
from model_explorer_debugger.runtime.runner_channel import RunnerChannel
from model_explorer_debugger.runtime.runner_device import RunnerDevices
from model_explorer_debugger.session_registry import SessionRegistry


class Socket:

  def __init__(self):
    self.incoming = queue.Queue()
    self.sent = []
    self.closed = False
    self.reader_threads = set()

  def recv(self, timeout):
    self.reader_threads.add(threading.get_ident())
    try:
      value = self.incoming.get(timeout=timeout)
    except queue.Empty:
      raise TimeoutError from None
    if isinstance(value, BaseException):
      raise value
    return json.dumps(value)

  def send(self, message):
    self.sent.append(json.loads(message))

  def close(self):
    self.closed = True
    self.incoming.put(ConnectionError('socket closed'))


def wait_for(predicate, timeout=2):
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if predicate():
      return
    time.sleep(0.005)
  raise AssertionError('Condition did not become true')


class ChannelTests(unittest.TestCase):

  def channel(self):
    channel = RunnerChannel()
    channel.socket = Socket()
    channel.heartbeat_interval = 0.02
    channel.heartbeat_timeout = 0.15
    channel.start_channel({'protocolVersion': 3})
    self.addCleanup(channel.socket.close)
    self.addCleanup(channel.stop_channel)
    return channel

  def activate(self, channel):
    owner = dict(serverId='server', sessionId=str(uuid4()), runId='ref')

    def reply():
      wait_for(lambda: bool(channel.socket.sent))
      channel.socket.incoming.put(
          dict(
              type='activated',
              requestId=channel.socket.sent[0]['requestId'],
              owner=owner,
          )
      )

    threading.Thread(target=reply).start()
    channel.activate_owner(owner)
    return owner

  def test_heartbeat_and_local_actions_never_pollute_request_stream(self):
    channel = self.channel()
    events = []
    channel.on_lifecycle = lambda kind, fields: events.append((kind, fields))
    owner = self.activate(channel)
    wait_for(lambda: any(v['type'] == 'heartbeat' for v in channel.socket.sent))
    heartbeat = next(v for v in channel.socket.sent if v['type'] == 'heartbeat')
    channel.socket.incoming.put(dict(heartbeat, type='heartbeat_ack'))
    channel.socket.incoming.put(dict(type='stop_requested', **owner))
    channel.socket.incoming.put(
        dict(type='delta', requestId='request', text='hello')
    )
    self.assertEqual(channel.receive(1)['type'], 'delta')
    wait_for(lambda: len(events) == 2)
    self.assertEqual(
        [event[0] for event in events], ['heartbeat', 'stop_requested']
    )
    self.assertEqual(len(channel.socket.reader_threads), 1)

  def test_idle_disconnect_is_reported_without_a_generate(self):
    channel = self.channel()
    events = []
    channel.on_lifecycle = lambda kind, fields: events.append((kind, fields))
    self.activate(channel)
    channel.socket.incoming.put(ConnectionError('cable removed'))
    wait_for(lambda: bool(events))
    self.assertEqual(events[0][0], 'disconnected')
    with self.assertRaisesRegex(ConnectionError, 'cable removed'):
      channel.receive(0.1)

  def test_wrong_owner_ack_does_not_extend_lease(self):
    channel = self.channel()
    events = []
    channel.on_lifecycle = lambda kind, fields: events.append((kind, fields))
    self.activate(channel)
    wait_for(lambda: any(v['type'] == 'heartbeat' for v in channel.socket.sent))
    for sequence in range(1, 4):
      channel.socket.incoming.put(
          dict(
              type='heartbeat_ack',
              serverId='other',
              sessionId=channel.owner['sessionId'],
              sequence=sequence,
          )
      )
    wait_for(lambda: channel.socket.closed)
    self.assertEqual(events[-1][0], 'disconnected')
    self.assertIsNone(channel.last_seen if not channel.owner else None)

  def test_channel_reads_the_socket_directly_until_started(self):
    channel = RunnerChannel()
    channel.socket = Socket()
    channel.socket.incoming.put({'type': 'hello'})
    self.assertEqual(channel.receive(1)['type'], 'hello')
    channel.send({'type': 'ping'})
    self.assertEqual(channel.socket.sent, [{'type': 'ping'}])
    channel.stop_channel()


class FakeClock:
  """A monotonic clock that only moves when a fake Runner takes time."""

  def __init__(self):
    self.now = 0.0

  def monotonic(self):
    return self.now


class OrderedConnection:
  """Answers requests in order, each reply taking `delay` fake seconds."""

  def __init__(self, delay, clock):
    self.delay = delay
    self.clock = clock
    self.unanswered = collections.deque()
    self.closed = False

  def send(self, message):
    self.unanswered.append(message)

  def receive(self, timeout):
    if self.closed:
      raise ConnectionError('Runner connection is closed')
    if timeout < self.delay:
      self.clock.now += timeout
      raise TimeoutError('Waiting for Runner response')
    self.clock.now += self.delay
    message = self.unanswered.popleft()
    return {'type': message['type'], 'requestId': message['requestId']}

  def close(self):
    self.closed = True


class PipelineTests(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.clock = FakeClock()
    patcher = patch.object(runner_channel, 'time', self.clock)
    patcher.start()
    self.addCleanup(patcher.stop)

  def abandon_after_first_reply(self, connection, *, timeout):
    replies = pipeline(
        connection,
        (('upload_chunk', {'offset': n}, None) for n in range(5)),
        window=5,
        timeout=timeout,
    )
    self.assertEqual(next(replies)['type'], 'upload_chunk')
    started = self.clock.monotonic()
    replies.close()
    return self.clock.monotonic() - started

  def test_abandoned_transfer_drains_outstanding_replies(self):
    connection = OrderedConnection(delay=0.01, clock=self.clock)
    self.abandon_after_first_reply(connection, timeout=5)
    self.assertFalse(connection.unanswered)
    self.assertFalse(connection.closed)

  def test_slow_drain_shares_one_deadline_and_closes_the_connection(self):
    connection = OrderedConnection(delay=0.2, clock=self.clock)
    with self.assertLogs(level='WARNING') as logs:
      elapsed = self.abandon_after_first_reply(connection, timeout=0.3)
    # Four outstanding replies at 0.2 s each would take 0.8 s to drain; one
    # shared deadline stops at 0.3 s.
    self.assertAlmostEqual(elapsed, 0.3)
    self.assertTrue(connection.closed)
    self.assertIn('abandoned', logs.output[0])


class SessionLifecycleTests(unittest.TestCase):

  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name)
    model = self.root / 'model.litertlm'
    model.write_bytes(b'synthetic fixture')
    self.registry = SessionRegistry(self.root / 'workspace')
    self.artifact = self.registry.register_model(model)
    capability = dict(
        available=True,
        runtimes=[dict(id='LiteRT-LM', available=True, backends=['CPU'])],
    )
    patcher = patch.object(
        self.registry, 'capabilities', return_value=capability
    )
    patcher.start()
    self.addCleanup(patcher.stop)
    self.runs = [
        dict(
            id=role,
            device='macos:' + str(uuid4()),
            runtime='LiteRT-LM',
            backend='CPU',
            artifact=self.artifact,
            source='registered',
        )
        for role in ('ref', 'target')
    ]
    self.record = self.registry.manage(
        'create', dict(name='Session', model='Fixture', runs=self.runs)
    )
    self.manager = JobManager(self.registry)
    self.addCleanup(self.manager.close)
    self.requests = []
    patcher = patch.object(
        self.manager.runners, 'execute', side_effect=self.execute
    )
    patcher.start()
    self.addCleanup(patcher.stop)

  def execute(self, request, emit, cancelled):
    self.requests.append(request)
    return dict(output='fixture', debug_enabled=True)

  def start(self):
    job = self.manager.start(
        self.record['id'], 'initialize', {'request_id': 'start'}
    )
    self.manager.thread.join(2)
    self.assertEqual(self.manager.get(job['id'])['status'], 'completed')
    return job

  def test_two_runner_owners_and_session_identity_persist(self):
    self.start()
    state = self.manager.execution(self.record['id'])
    self.assertEqual(state['phase'], 'active')
    self.assertEqual(
        {runner['role'] for runner in state['runners']}, {'ref', 'target'}
    )
    self.assertTrue(
        all(
            runner['phase'] == 'ready' and runner['connected']
            for runner in state['runners']
        )
    )
    self.assertTrue(
        all(
            request['owner']['sessionId'] == self.record['id']
            for request in self.requests
        )
    )
    from model_explorer_debugger.runtime.runners import Runners

    second = Runners(self.registry.root)
    self.assertEqual(second.identity, self.manager.runners.identity)

  def test_public_owner_projection_identifies_server_keeps_wire_owner(self):
    self.start()
    with self.manager.lifecycle_lock:
      self.manager.executions[self.record['id']]['runners'][1]['owner'][
          'serverId'
      ] = 'another-server'
    public = self.registry.get(self.record['id'])['execution']
    self.assertTrue(public['runners'][0]['owner']['isCurrentServer'])
    self.assertFalse(public['runners'][1]['owner']['isCurrentServer'])
    self.assertEqual(
        self.registry.listing()['sessions'][0]['execution'], public
    )
    self.assertTrue(
        all(
            'isCurrentServer' not in request['owner']
            for request in self.requests
        )
    )
    self.assertTrue(
        all(
            'isCurrentServer' not in runner['owner']
            for runner in self.manager.executions[self.record['id']]['runners']
        )
    )
    public['runners'][0]['owner']['serverId'] = 'changed-by-caller'
    self.assertTrue(
        self.manager.execution(self.record['id'])['runners'][0]['owner'][
            'isCurrentServer'
        ]
    )

  def test_same_mac_uses_distinct_runner_slots(self):
    self.record = self.registry.manage(
        'update',
        {
            **self.record,
            'runs': [
                {**run, 'device': self.runs[0]['device']} for run in self.runs
            ],
        },
    )
    self.start()
    self.assertEqual(
        {r['run']['_runner_slot'] for r in self.requests}, {'ref', 'target'}
    )
    self.assertEqual(len(self.manager.device_owners), 1)

  def test_occupied_device_cannot_be_acquired_by_another_session(self):
    self.start()
    other = self.registry.manage('duplicate', dict(id=self.record['id']))
    with self.assertRaisesRegex(ValueError, 'occupied'):
      self.manager.start(other['id'], 'initialize', {'request_id': 'other'})
    self.assertEqual(len(self.requests), 2)

  def test_end_returns_ending_until_both_runners_release(self):
    self.start()
    gate = threading.Event()
    entered = threading.Event()

    def release(identity):
      entered.set()
      gate.wait(2)

    with patch.object(
        self.manager.runners, 'close_session', side_effect=release
    ):
      result = self.manager.close_session(self.record['id'])
      self.assertTrue(entered.wait(1))
      self.assertEqual(result['execution']['phase'], 'ending')
      self.assertTrue(self.manager.device_owners)
      gate.set()
      wait_for(
          lambda: self.manager.execution(self.record['id'])['phase'] == 'ended'
      )
    self.assertFalse(self.manager.device_owners)
    self.assertFalse(self.registry.get(self.record['id'])['initialized'])

  def test_stop_ends_chat_without_replay_or_model_release(self):
    self.start()
    self.requests.clear()

    def execute(request, emit, cancelled):
      self.requests.append(request)
      if request['operation'] == 'generate':
        raise InterruptedError('User pressed Stop')
      return {}

    with (
        patch.object(self.manager.runners, 'execute', side_effect=execute),
        patch.object(self.manager.runners, 'reset_session') as reset,
        patch.object(self.manager.runners, 'close_session') as close,
    ):
      job = self.manager.start(
          self.record['id'],
          'generate',
          dict(
              request_id='turn', prompt='cancelled prompt', max_output_tokens=2
          ),
      )
      self.manager.threads[job['id']].join(2)
      self.assertEqual(self.manager.get(job['id'])['status'], 'cancelled')
      reset.assert_called_once_with(self.record['id'])
      close.assert_not_called()
    self.assertFalse(any(r['operation'] == 'initialize' for r in self.requests))
    self.assertTrue(self.registry.get(self.record['id'])['read_only'])
    self.assertTrue(self.manager.execution(self.record['id'])['newChatAllowed'])
    self.assertEqual(
        self.manager.execution(self.record['id'])['phase'], 'active'
    )

  def test_end_cancels_active_generation_without_recovery(self):
    self.start()
    entered = threading.Event()

    def execute(request, emit, cancelled):
      entered.set()
      wait_for(cancelled)
      raise InterruptedError('Cancelled')

    with (
        patch.object(self.manager.runners, 'execute', side_effect=execute),
        patch.object(self.manager.runners, 'reset_session') as reset,
    ):
      job = self.manager.start(
          self.record['id'],
          'generate',
          dict(request_id='turn', prompt='hello', max_output_tokens=2),
      )
      self.assertTrue(entered.wait(1))
      self.manager.close_session(self.record['id'])
      wait_for(
          lambda: self.manager.execution(self.record['id'])['phase'] == 'ended'
      )
      self.assertEqual(self.manager.get(job['id'])['status'], 'cancelled')
      reset.assert_not_called()

  def test_idle_disconnect_interrupts_parent_and_child(self):
    self.start()
    child = self.registry.manage('chat', dict(id=self.record['id']))
    self.manager.lifecycle_events.put((
        self.record['id'],
        'target',
        'disconnected',
        {'error': 'cable removed'},
    ))
    wait_for(
        lambda: self.manager.execution(self.record['id'])['phase']
        == 'unavailable'
    )
    self.assertEqual(
        self.registry.get(child['id'])['execution']['phase'], 'unavailable'
    )
    self.assertFalse(self.registry.get(child['id'])['initialized'])

  def test_ended_ios_shell_can_start_a_new_owner_without_build_replacement(
      self,
  ):
    from model_explorer_debugger.runtime.runners import Runners

    runs = [
        {**run, 'device': 'ios:00000000-000000000000000' + str(index)}
        for index, run in enumerate(self.runs)
    ]
    self.record = self.registry.manage('update', {**self.record, 'runs': runs})
    shell = dict(
        protocolVersion=4,
        lifecycle='ended',
        controlConnected=False,
        busy=False,
        residentSessions=0,
        environment={'platform': 'iOS'},
        buildId='installed-cl',
    )
    real_dispatch = Runners.execute.__get__(self.manager.runners, Runners)
    with (
        patch(
            'model_explorer_debugger.runtime.runner_discovery.inspect',
            return_value={
                'status': 'ready',
                'runnerState': 'idle',
                'runner': shell,
            },
        ) as inspection,
        patch(
            'model_explorer_debugger.runtime.runner_builds._registered',
            side_effect=AssertionError(
                'Do not select a replacement for the running build'
            ),
        ),
        patch.object(
            self.manager.runners.ios, 'execute', side_effect=self.execute
        ),
        patch.object(
            self.manager.runners, 'execute', side_effect=real_dispatch
        ),
    ):
      self.start()
      first = self.manager.execution(self.record['id'])['instanceId']
      self.manager.close_session(self.record['id'])
      wait_for(
          lambda: self.manager.execution(self.record['id'])['phase'] == 'ended'
      )
      job = self.manager.start(
          self.record['id'], 'initialize', {'request_id': 'restart'}
      )
      self.manager.thread.join(2)
      self.assertEqual(self.manager.get(job['id'])['status'], 'completed')
      self.assertEqual(inspection.call_count, 4)
      self.assertNotEqual(
          self.manager.execution(self.record['id'])['instanceId'], first
      )
    self.assertEqual(
        self.manager.execution(self.record['id'])['phase'], 'active'
    )

  def test_stale_connection_event_cannot_interrupt_a_new_session_run(self):
    self.start()
    self.manager.lifecycle_events.put((
        self.record['id'],
        'ref',
        'disconnected',
        {'error': 'old connection', 'executionId': 'previous-start'},
    ))
    self.manager.lifecycle_events.put(
        (self.record['id'], 'ref', 'heartbeat', {'lastSeen': 'current'})
    )
    wait_for(
        lambda: self.manager.execution(self.record['id'])['runners'][0].get(
            'lastSeen'
        )
        == 'current'
    )
    self.assertEqual(
        self.manager.execution(self.record['id'])['phase'], 'active'
    )

  def test_delayed_activation_and_heartbeat_do_not_restore_loading(self):
    self.start()
    self.manager.lifecycle_events.put((
        self.record['id'],
        'ref',
        'activated',
        {
            'deviceId': self.runs[0]['device'],
            'owner': self.requests[0]['owner'],
        },
    ))
    self.manager.lifecycle_events.put((
        self.record['id'],
        'ref',
        'heartbeat',
        {'lastSeen': 'after-activation'},
    ))
    wait_for(
        lambda: self.manager.execution(self.record['id'])['runners'][0].get(
            'lastSeen'
        )
        == 'after-activation'
    )
    self.assertEqual(
        self.manager.execution(self.record['id'])['runners'][0]['phase'],
        'ready',
    )

  def test_new_chat_initializes_fresh_context_and_old_chat_is_read_only(self):
    self.start()
    self.requests.clear()
    with patch.object(self.manager.runners, 'reset_session') as reset:
      child = self.manager.manage(
          'chat', dict(id=self.record['id'], name='Next Chat')
      )
      reset.assert_called_once_with(self.record['id'])
    self.assertEqual(
        [r['operation'] for r in self.requests], ['initialize', 'initialize']
    )
    self.assertTrue(all(r['messages'] == [] for r in self.requests))
    self.assertTrue(
        all(
            r['session_id'] == self.record['id'] and r['chat_id'] == child['id']
            for r in self.requests
        )
    )
    self.assertTrue(self.registry.get(self.record['id'])['read_only'])
    self.assertFalse(child['pending_chat'])
    self.assertTrue(child['initialized'])
    with self.assertRaisesRegex(ValueError, 'read-only'):
      self.manager.start(
          self.record['id'],
          'generate',
          dict(request_id='old', prompt='no', max_output_tokens=2),
      )

  def generate(self, key='turn'):
    return self.manager.start(
        self.record['id'],
        'generate',
        dict(request_id=key, prompt='hello', max_output_tokens=2),
    )

  def wait_job(self, job):
    self.manager.threads[job['id']].join(3)
    self.assertFalse(self.manager.threads[job['id']].is_alive())
    return self.manager.get(job['id'])

  def test_different_sessions_and_both_sides_execute_concurrently(self):
    other = self.registry.manage('duplicate', dict(id=self.record['id']))
    self.registry.manage(
        'update',
        {
            **other,
            'runs': [
                {**r, 'device': 'macos:' + str(uuid4())} for r in other['runs']
            ],
        },
    )
    barrier = threading.Barrier(4)

    def execute(request, emit, cancelled):
      barrier.wait(2)
      return {'output': 'fixture'}

    with patch.object(self.manager.runners, 'execute', side_effect=execute):
      a = self.manager.start(
          self.record['id'], 'initialize', {'request_id': 'A'}
      )
      b = self.manager.start(other['id'], 'initialize', {'request_id': 'B'})
      self.assertEqual(self.wait_job(a)['status'], 'completed')
      self.assertEqual(self.wait_job(b)['status'], 'completed')

  def test_one_active_operation_per_session(self):
    self.start()
    gate = threading.Event()
    entered = threading.Event()

    def execute(request, emit, cancelled):
      if request['operation'] == 'preflight':
        return {}
      entered.set()
      gate.wait(2)
      return {'output': 'fixture', 'debug_data': {'status': 'unavailable'}}

    with patch.object(self.manager.runners, 'execute', side_effect=execute):
      job = self.generate()
      self.assertTrue(entered.wait(1))
      with self.assertRaisesRegex(ValueError, 'active'):
        self.generate('other')
      gate.set()
      self.wait_job(job)

  def test_runner_phases_cover_generation_and_capture_transfer(self):
    self.start()
    entered = threading.Barrier(3)
    sampled = threading.Barrier(3)
    sample = threading.Event()
    returned = threading.Event()

    def execute(request, emit, cancelled):
      if request['operation'] == 'preflight':
        return {}
      entered.wait(2)
      sample.wait(2)
      emit('generation_completed', output='answer', output_confirmed=True)
      sampled.wait(2)
      returned.wait(2)
      return dict(
          output='answer',
          generation_status='completed',
          output_confirmed=True,
          debug_data={
              'status': 'unavailable',
              'error': 'Synthetic missing capture',
          },
      )

    with patch.object(self.manager.runners, 'execute', side_effect=execute):
      job = self.generate()
      try:
        entered.wait(2)
        self.assertEqual(
            {
                r['phase']
                for r in self.manager.execution(self.record['id'])['runners']
            },
            {'generating'},
        )
        sample.set()
        sampled.wait(2)
        self.assertEqual(
            {
                r['phase']
                for r in self.manager.execution(self.record['id'])['runners']
            },
            {'transferring'},
        )
      finally:
        sample.set()
        returned.set()
      self.assertEqual(self.wait_job(job)['status'], 'completed')
    self.assertEqual(
        {
            r['phase']
            for r in self.manager.execution(self.record['id'])['runners']
        },
        {'ready'},
    )

  def test_capacity_rejection_does_not_mutate_or_close_chat(self):
    from model_debugger_contracts.errors import InputRejected

    self.start()
    calls = []

    def execute(request, emit, cancelled):
      calls.append(request['operation'])
      if request['run']['id'] == 'target':
        raise InputRejected('Context capacity exceeded')
      return {'input_tokens': 12, 'context_limit': 1024}

    with (
        patch.object(self.manager.runners, 'execute', side_effect=execute),
        patch.object(self.manager.runners, 'reset_session') as reset,
    ):
      result = self.wait_job(self.generate())
    self.assertEqual(calls, ['preflight', 'preflight'])
    self.assertEqual(result['error_code'], 'input_rejected')
    self.assertEqual(result['status'], 'failed')
    self.assertTrue(result['input_rejected'])
    reset.assert_not_called()
    self.assertTrue(self.registry.get(self.record['id'])['initialized'])

  def test_successful_text_survives_missing_dump(self):
    self.start()

    def execute(request, emit, cancelled):
      return (
          {}
          if request['operation'] == 'preflight'
          else dict(
              output=request['run']['id'] + ' answer',
              generation_status='completed',
              output_confirmed=True,
              debug_data={'status': 'unavailable', 'error': 'missing dump'},
          )
      )

    with (
        patch.object(self.manager.runners, 'execute', side_effect=execute),
        patch('model_explorer_debugger.publication.publish_capture') as publish,
    ):
      result = self.wait_job(self.generate())
    self.assertEqual(result['status'], 'completed')
    publish.assert_not_called()
    record = self.registry.get(self.record['id'])
    self.assertFalse(record['has_capture'])
    self.assertEqual(len(record['successful_turns']), 1)
    self.assertEqual(
        record['successful_turns'][0]['output'],
        {'ref': 'ref answer', 'target': 'target answer'},
    )
    self.assertEqual(
        record['successful_turns'][0]['debug_data']['status'], 'unavailable'
    )
    self.assertTrue(record['initialized'])

  def test_debug_import_failure_does_not_end_chat(self):
    self.start()
    with patch(
        'model_explorer_debugger.publication.publish_capture',
        side_effect=ValueError('bad capture'),
    ):
      result = self.wait_job(self.generate())
    self.assertEqual(result['status'], 'completed')
    self.assertEqual(result['debug_data']['status'], 'unavailable')
    self.assertTrue(self.registry.get(self.record['id'])['initialized'])

  def test_runner_loss_after_both_confirmed_keeps_text_but_session_unavailable(
      self,
  ):
    self.start()
    barrier = threading.Barrier(2)

    def execute(request, emit, cancelled):
      if request['operation'] == 'preflight':
        return {}
      emit(
          'generation_completed',
          output=request['run']['id'],
          output_confirmed=True,
      )
      barrier.wait(2)
      return dict(
          output=request['run']['id'],
          generation_status='completed',
          output_confirmed=True,
          connection_lost=True,
          debug_data={
              'status': 'unavailable',
              'error': 'receipt connection lost',
          },
      )

    with patch.object(self.manager.runners, 'execute', side_effect=execute):
      result = self.wait_job(self.generate())
    self.assertEqual(result['status'], 'completed')
    self.assertEqual(
        len(self.registry.get(self.record['id'])['successful_turns']), 1
    )
    self.assertEqual(
        self.manager.execution(self.record['id'])['phase'], 'unavailable'
    )
    self.assertTrue(self.registry.get(self.record['id'])['read_only'])
    self.assertFalse(
        any(
            r['phase'] == 'ready'
            for r in self.manager.execution(self.record['id'])['runners']
        )
    )

  def test_generation_error_prohibits_new_chat(self):
    self.start()

    def execute(request, emit, cancelled):
      if request['operation'] == 'preflight':
        return {}
      return dict(
          generation_status='failed',
          output_confirmed=False,
          error='native error',
      )

    with patch.object(self.manager.runners, 'execute', side_effect=execute):
      result = self.wait_job(self.generate())
    self.assertEqual(result['status'], 'failed')
    self.assertFalse(
        self.manager.execution(self.record['id'])['newChatAllowed']
    )
    self.assertEqual(
        {
            r['phase']
            for r in self.manager.execution(self.record['id'])['runners']
        },
        {'failed'},
    )
    self.assertFalse(
        self.registry.get(self.record['id']).get('successful_turns')
    )
    with self.assertRaisesRegex(ValueError, 'cannot create'):
      self.manager.manage('chat', dict(id=self.record['id']))

  def test_failed_candidate_is_never_published(self):
    self.start()
    before = len(self.registry.listing()['sessions'])

    def execute(request, emit, cancelled):
      self.assertTrue(self.registry.get(self.record['id'])['read_only'])
      self.assertEqual(len(self.registry.listing()['sessions']), before)
      if request['run']['id'] == 'target':
        raise RuntimeError('candidate failure')
      return {}

    with (
        patch.object(self.manager.runners, 'execute', side_effect=execute),
        patch.object(self.manager.runners, 'reset_session'),
    ):
      with self.assertRaisesRegex(ValueError, 'candidate failure'):
        self.manager.manage('chat', dict(id=self.record['id']))
    self.assertEqual(len(self.registry.listing()['sessions']), before)
    self.assertTrue(self.registry.get(self.record['id'])['read_only'])
    self.assertTrue(self.manager.execution(self.record['id'])['newChatAllowed'])

  def test_stop_before_success_discards_late_completion(self):
    self.start()
    barrier = threading.Barrier(3)
    gate = threading.Event()

    def execute(request, emit, cancelled):
      if request['operation'] == 'preflight':
        return {}
      barrier.wait(2)
      gate.wait(2)
      emit('generation_completed', output='late', output_confirmed=True)
      return dict(
          output='late', generation_status='completed', output_confirmed=True
      )

    with (
        patch.object(self.manager.runners, 'execute', side_effect=execute),
        patch.object(self.manager.runners, 'reset_session') as reset,
    ):
      job = self.generate()
      barrier.wait(2)
      self.manager.cancel(job['id'])
      gate.set()
      result = self.wait_job(job)
    self.assertEqual(result['status'], 'cancelled')
    reset.assert_called_once()
    self.assertFalse(
        self.registry.get(self.record['id']).get('successful_turns')
    )
    self.assertEqual(
        {
            r['phase']
            for r in self.manager.execution(self.record['id'])['runners']
        },
        {'interrupted'},
    )

  def test_stop_after_success_keeps_turn_and_ends_chat(self):
    self.start()
    barrier = threading.Barrier(3)
    gate = threading.Event()

    def execute(request, emit, cancelled):
      if request['operation'] == 'preflight':
        return {}
      emit('generation_completed', output='confirmed', output_confirmed=True)
      barrier.wait(2)
      gate.wait(2)
      return dict(
          output='confirmed',
          generation_status='completed',
          output_confirmed=True,
          debug_data={'status': 'unavailable'},
      )

    with (
        patch.object(self.manager.runners, 'execute', side_effect=execute),
        patch.object(self.manager.runners, 'reset_session') as reset,
    ):
      job = self.generate()
      barrier.wait(2)
      self.manager.cancel(job['id'])
      gate.set()
      result = self.wait_job(job)
    self.assertEqual(result['status'], 'completed')
    reset.assert_called_once()
    self.assertEqual(
        len(self.registry.get(self.record['id'])['successful_turns']), 1
    )
    self.assertTrue(self.registry.get(self.record['id'])['read_only'])
    self.assertTrue(self.manager.execution(self.record['id'])['newChatAllowed'])

  def test_new_chat_request_replay_returns_same_published_chat(self):
    self.start()
    payload = dict(id=self.record['id'], name='Once', request_id='create-once')
    with patch.object(self.manager.runners, 'reset_session') as reset:
      first = self.manager.manage('chat', payload)
      again = self.manager.manage('chat', payload)
    self.assertEqual(first['id'], again['id'])
    reset.assert_called_once()
    self.assertEqual(len(self.registry.listing()['sessions']), 2)
    with self.assertRaisesRegex(ValueError, 'different Chat request'):
      self.manager.manage('chat', {**payload, 'name': 'Changed'})

  def test_disconnect_during_stop_cleanup_never_reactivates_session(self):
    self.start()

    def execute(request, emit, cancelled):
      if request['operation'] == 'preflight':
        return {}
      return dict(generation_status='stopped', output_confirmed=False)

    def reset(identity):
      self.manager._unavailable(identity, 'lost during cleanup')

    with (
        patch.object(self.manager.runners, 'execute', side_effect=execute),
        patch.object(self.manager.runners, 'reset_session', side_effect=reset),
    ):
      self.wait_job(self.generate())
    self.assertEqual(
        self.manager.execution(self.record['id'])['phase'], 'unavailable'
    )
    self.assertFalse(
        self.manager.execution(self.record['id'])['newChatAllowed']
    )

  def test_child_active_at_restart_marks_parent_execution_unavailable(self):
    self.start()
    with patch.object(self.manager.runners, 'reset_session'):
      child = self.manager.manage('chat', dict(id=self.record['id']))
    restarted = SessionRegistry(self.registry.root)
    manager = JobManager(restarted)
    self.addCleanup(manager.close)
    self.assertEqual(
        manager.execution(self.record['id'])['phase'], 'unavailable'
    )
    self.assertTrue(restarted.get(child['id'])['read_only'])

  def test_initialize_terminal_connection_loss_cannot_publish_first_chat(self):
    def execute(request, emit, cancelled):
      return {'connection_lost': request['run']['id'] == 'target'}

    with patch.object(self.manager.runners, 'execute', side_effect=execute):
      job = self.manager.start(
          self.record['id'], 'initialize', {'request_id': 'lost-init'}
      )
      result = self.wait_job(job)
    self.assertEqual(result['status'], 'failed')
    self.assertFalse(self.registry.get(self.record['id'])['initialized'])
    self.assertEqual(self.registry.get(self.record['id'])['status'], 'failed')
    self.assertIn('initialization_error', self.registry.get(self.record['id']))
    self.assertFalse(
        any(
            r['phase'] == 'ready'
            for r in self.manager.execution(self.record['id'])['runners']
        )
    )

  def test_initialize_failure_or_interruption_never_reports_ready(self):
    for error, status in [
        (RuntimeError('Load failed'), 'failed'),
        (InterruptedError('Load interrupted'), 'cancelled'),
    ]:
      with (
          self.subTest(status=status),
          patch.object(self.manager.runners, 'execute', side_effect=error),
      ):
        job = self.manager.start(
            self.record['id'], 'initialize', {'request_id': 'init-' + status}
        )
        result = self.wait_job(job)
      self.assertEqual(result['status'], status)
      self.assertFalse(self.registry.get(self.record['id'])['initialized'])
      execution = self.manager.execution(self.record['id'])
      self.assertEqual(execution['phase'], 'interrupted')
      self.assertFalse(
          any(
              r['phase'] == 'ready' or r['connected']
              for r in execution['runners']
          )
      )

  def test_selected_capture_is_prepared_before_both_default_first_chats(self):
    from model_explorer_debugger.runtime.tap_profiles import PROFILE_ID

    self.record = self.registry.manage(
        'update', {**self.record, 'tap_profile': PROFILE_ID}
    )
    order = []

    def prepare(job, record, artifacts):
      order.append('prepare')
      self.registry.update(record['id'], tap_prepared=True)
      return {}

    def execute(request, emit, cancelled):
      self.assertEqual(order[0], 'prepare')
      self.assertEqual(request['generation'], {})
      self.assertEqual(request['messages'], [])
      return {}

    with (
        patch.object(self.manager, '_prepare', side_effect=prepare),
        patch.object(self.manager.runners, 'execute', side_effect=execute),
    ):
      job = self.manager.start(
          self.record['id'], 'initialize', {'request_id': 'whole-init'}
      )
      result = self.wait_job(job)
    self.assertEqual(result['status'], 'completed')
    self.assertTrue(self.registry.get(self.record['id'])['tap_prepared'])


if __name__ == '__main__':
  unittest.main()
