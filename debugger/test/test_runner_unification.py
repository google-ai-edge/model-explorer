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

"""Exercise Runner routing and supervisor lifecycle without a model library."""

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4
from model_debugger_runner.python_runner_host import PythonRunnerHost
from model_explorer_debugger.runtime.macos_device import MacDevices
from model_explorer_debugger.runtime.runners import Runners, resolve_device


class Workers:

  def __init__(self):
    self.requests = []
    self.closed = []

  def execute(self, request, emit, cancelled):
    self.requests.append(request)
    emit('delta', text='synthetic')
    return {'processed_token_count': 7, 'effective': {'contextLength': 64}}

  def close(self):
    self.closed.append('all')


class RunnerUnificationTests(unittest.TestCase):

  def test_server_dispatches_both_runtimes_to_same_runner(self):
    identity = str(uuid4())
    with patch(
        'model_explorer_debugger.runtime.macos_device.configurations',
        return_value={identity: {'host': '127.0.0.1'}},
    ):
      runners = Runners('/unused')
      with patch.object(
          runners.mac, 'execute', return_value={'ok': True}
      ) as call:
        for runtime in ('LiteRT-LM', 'PyTorch'):
          request = {'run': {'runtime': runtime, 'device': 'Server host'}}
          runners.execute(request, lambda *a: None, lambda: False)
          self.assertEqual(
              call.call_args.args[0]['run']['device'], 'macos:' + identity
          )
          self.assertEqual(request['run']['device'], 'Server host')
      with patch(
          'model_explorer_debugger.runtime.macos_device.configurations',
          return_value={},
      ):
        with self.assertRaisesRegex(ValueError, 'Open the local Runner'):
          resolve_device('/unused', 'local')

  def test_local_candidate_preserves_explicit_device_identity(self):
    from model_explorer_debugger.runtime.runner_discovery import candidates

    with (
        patch(
            'model_explorer_debugger.runtime.macos_device.list_devices',
            return_value=[
                {'id': 'macos:example', 'name': 'Mac', 'transport': 'Local'}
            ],
        ),
        patch(
            'model_explorer_debugger.runtime.ios_device.list_devices',
            return_value=[],
        ),
    ):
      devices = candidates('/unused')['devices']
      self.assertEqual(len(devices), 1)
      self.assertEqual(devices[0]['id'], 'local')
      self.assertEqual(devices[0]['runnerId'], 'macos:example')

  def test_supervisor_owns_runtime_root_history_and_close(self):
    with tempfile.TemporaryDirectory() as folder:
      workers = Workers()
      events = []
      host = PythonRunnerHost(folder, events.append, workers)
      session = str(uuid4())

      def command(operation):
        return dict(
            type=operation,
            requestId=str(uuid4()),
            request=dict(
                operation=operation,
                session_id=session,
                run={'id': 'ref', 'runtime': 'PyTorch'},
                pytorch_root='/not/allowed',
                output=folder + '/' + str(uuid4()),
                messages=[{'role': 'user', 'content': 'old'}],
            ),
        )

      first = command('initialize')
      host.accept(first)
      host.thread.join(2)
      host.accept(command('generate'))
      host.thread.join(2)
      self.assertEqual(
          workers.requests[0]['pytorch_root'], str(Path(folder).resolve())
      )
      self.assertIn('messages', workers.requests[0])
      self.assertNotIn('messages', workers.requests[1])
      with self.assertRaisesRegex(ValueError, 'already executed'):
        host.accept(first)
      # The App never sends a per-session close; shutdown is stdin EOF via
      # host.close().
      with self.assertRaisesRegex(ValueError, 'Unsupported'):
        host.accept(
            dict(type='close', requestId=str(uuid4()), sessionId=session)
        )
      host.close()
      self.assertEqual(workers.closed[-1], 'all')

  def test_supervisor_cancellation_and_busy_are_request_scoped(self):
    with tempfile.TemporaryDirectory() as folder:
      entered = threading.Event()
      done = threading.Event()
      workers = Workers()

      def execute(request, emit, cancelled):
        entered.set()
        while not cancelled():
          if done.wait(0.01):
            raise AssertionError('Expected cancellation')
        raise InterruptedError('Cancelled')

      workers.execute = execute
      events = []
      host = PythonRunnerHost(folder, events.append, workers)
      identity = str(uuid4())
      host.accept(
          dict(
              type='initialize',
              requestId=identity,
              request=dict(
                  operation='initialize',
                  session_id=str(uuid4()),
                  run={'id': 'ref', 'runtime': 'PyTorch'},
                  output=folder + '/capture',
              ),
          )
      )
      try:
        self.assertTrue(entered.wait(2))
        with self.assertRaisesRegex(ValueError, 'No matching'):
          host.accept(dict(type='cancel', requestId=str(uuid4())))
        with self.assertRaisesRegex(ValueError, 'busy'):
          host.accept(dict(type='reset', requestId=str(uuid4())))
        host.accept(dict(type='cancel', requestId=identity))
        host.thread.join(2)
        self.assertEqual(events[-1]['type'], 'error')
        self.assertTrue((Path(folder) / 'cancel').exists())
      finally:
        done.set()
        host.close()

  def test_pytorch_missing_capability_has_no_worker_fallback(self):
    identity = str(uuid4())
    peer = MagicMock()
    peer.runtimes = ['LiteRT-LM']
    devices = MacDevices('/unused')
    request = {
        'operation': 'initialize',
        'session_id': str(uuid4()),
        'run': {
            'id': 'ref',
            'runtime': 'PyTorch',
            'device': 'macos:' + identity,
        },
    }
    with (
        patch(
            'model_explorer_debugger.runtime.macos_device.configurations',
            return_value={identity: {'host': '127.0.0.1'}},
        ),
        patch.object(devices, 'connect', return_value=peer),
    ):
      with self.assertRaisesRegex(ValueError, 'Configure PyTorch'):
        devices.execute(request, lambda *a: None, lambda: False)
    peer.send.assert_not_called()
    peer.close.assert_called_once()

  def test_acknowledged_failure_keeps_peer_for_other_role_close(self):
    identity = str(uuid4())
    session = str(uuid4())
    sent = []
    peer = MagicMock()
    peer.runtimes = ['PyTorch']
    peer.is_open = True
    peer.send.side_effect = sent.append
    peer.receive.side_effect = lambda timeout: dict(
        type='error', requestId=sent[-1]['requestId'], error='synthetic failure'
    )
    devices = MacDevices('/unused')
    devices.connections[identity] = peer
    devices.sessions = {
        (session, 'ref', identity): {},
        (session, 'target', identity): {},
    }
    request = {
        'operation': 'generate',
        'session_id': session,
        'run': {
            'id': 'ref',
            'runtime': 'PyTorch',
            'device': 'macos:' + identity,
        },
    }
    with patch(
        'model_explorer_debugger.runtime.macos_device.configurations',
        return_value={identity: {'host': '127.0.0.1'}},
    ):
      result = devices.execute(request, lambda *a: None, lambda: False)
      self.assertEqual(result['generation_status'], 'failed')
      self.assertFalse(result['connection_lost'])
      self.assertIn('synthetic failure', result['error'])
    peer.close.assert_not_called()
    self.assertEqual(list(devices.sessions), [(session, 'target', identity)])
    peer.receive.side_effect = lambda timeout: dict(
        type='closed', requestId=sent[-1]['requestId']
    )
    devices.close_session(session)
    self.assertFalse(devices.sessions)
    self.assertEqual(sent[-1]['type'], 'close')

  def test_python_workers_reject_litert_execution(self):
    from model_debugger_runner.python_workers import PythonWorkers

    with tempfile.TemporaryDirectory() as folder:
      with self.assertRaisesRegex(ValueError, 'PyTorch only'):
        PythonWorkers().execute(
            {
                'operation': 'initialize',
                'session_id': 'a',
                'output': folder + '/out',
                'run': {'id': 'ref', 'runtime': 'LiteRT-LM'},
            },
            lambda *a: None,
            lambda: False,
        )

  def test_runner_device_adapter_protocol_and_remote_runtime_seam(self):
    from model_explorer_debugger.runtime.runner_device import (
        RunnerDeviceAdapter,
        RunnerDevices,
    )

    self.assertTrue(getattr(RunnerDeviceAdapter, '_is_protocol', False))
    with tempfile.TemporaryDirectory() as folder:
      runners = Runners(folder)
      self.assertIsInstance(runners.mac, RunnerDeviceAdapter)
      self.assertIsInstance(runners.ios, RunnerDeviceAdapter)
      custom_adapter = RunnerDevices(folder)
      custom_adapter.DEVICE_PREFIXES = ('custom:',)
      custom_adapter.SUPPORTED_RUNTIMES = frozenset({'LiteRT-LM'})
      runners.adapters = (*runners.adapters, custom_adapter)
      with self.assertRaisesRegex(
          ValueError,
          'PyTorch currently requires local shared files on the server Mac',
      ):
        runners.execute(
            {
                'operation': 'initialize',
                'session_id': 's1',
                'run': {
                    'id': 'ref',
                    'runtime': 'PyTorch',
                    'device': 'custom:node-1',
                },
            },
            lambda *a: None,
            lambda: False,
        )


if __name__ == '__main__':
  unittest.main()
