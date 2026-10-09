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

"""Synthetic fixtures for non-owning discovery and connection recovery."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from model_explorer_debugger.runtime.runner_device import RunnerDevices
from model_explorer_debugger.runtime.runner_discovery import (
    INSPECTION_PROTOCOL,
    candidates,
    descriptor,
    inspect,
    read_observation,
)


def observation(**changes):
  return dict(
      version=1,
      protocolVersion=4,
      runnerInstance=str(uuid4()),
      controlConnected=False,
      busy=False,
      residentSessions=0,
      capabilities=dict(
          runtime='LiteRT-LM',
          backends=['CPU'],
          modalities=['text'],
          tensorCapture=True,
          persistentConversation=True,
          contextLengths=[1024],
          maxOutputTokens=32,
          maxCapturePoints=16,
      ),
      environment=dict(
          capturedAt='2026-09-14T00:00:00Z',
          platform='Synthetic',
          operatingSystem='Synthetic',
          architecture='arm64',
      ),
      state=dict(capturedAt='2026-09-14T00:00:00Z'),
      **changes,
  )


class RunnerDiscoveryTests(unittest.TestCase):

  def test_candidates_never_claim_online_or_open_a_connection(self):
    with (
        patch(
            'model_explorer_debugger.runtime.macos_device.list_devices',
            return_value=[{'id': 'macos:x', 'name': 'Synthetic Mac'}],
        ),
        patch(
            'model_explorer_debugger.runtime.ios_device.list_devices',
            return_value=[{'id': 'ios:x', 'name': 'Synthetic phone'}],
        ),
        patch(
            'model_explorer_debugger.runtime.device_access.adb_devices',
            return_value=[],
        ),
        patch(
            'model_explorer_debugger.runtime.device_access.registered_devices',
            return_value=[],
        ),
        patch(
            'model_explorer_debugger.runtime.runner_discovery.read_observation',
            side_effect=AssertionError('Discovery must not connect'),
        ),
    ):
      devices = candidates(Path('/unused'))['devices']
      self.assertEqual(
          [d['status'] for d in devices], ['unknown', 'unknown', 'unknown']
      )

  def test_descriptor_omits_unavailable_fields_and_unknown_sensitive_fields(
      self,
  ):
    value = observation()
    value['token'] = 'never expose'
    value['environment']['token'] = 'never expose'
    result = descriptor(value)
    self.assertNotIn('token', json.dumps(result))
    self.assertNotIn('processMemoryHeadroomBytes', result['state'])
    self.assertNotIn('chip', result['environment'])

  def test_invalid_or_unsupported_descriptors_fail_closed(self):
    for changes in [
        {'version': 2},
        {'version': True},
        {'runnerInstance': None},
        {'capabilities': None},
        {'busy': 1},
        {'residentSessions': -1},
        {'state': {'capturedAt': 'x', 'processResidentBytes': -5}},
    ]:
      with (
          self.subTest(changes=changes),
          self.assertRaises((ValueError, TypeError, KeyError)),
      ):
        descriptor(observation() | changes)

  def test_inspection_negotiates_its_own_protocol_and_sends_no_commands(self):
    peer = MagicMock()
    peer.__enter__.return_value = peer
    peer.subprotocol = INSPECTION_PROTOCOL
    peer.recv.return_value = json.dumps(observation())
    with patch('websockets.sync.client.connect', return_value=peer) as connect:
      read_observation('127.0.0.1', 8769, 'a' * 64)
      self.assertEqual(
          connect.call_args.kwargs['subprotocols'], [INSPECTION_PROTOCOL]
      )
      self.assertIsNone(connect.call_args.kwargs['proxy'])
      peer.send.assert_not_called()
      peer.subprotocol = None
      with self.assertRaises(ValueError):
        read_observation('127.0.0.1', 8769, 'a' * 64)
      self.assertEqual(connect.call_count, 2)  # No fallback control handshake.

  def test_mac_inspection_does_not_acquire_control_or_invalidate_sessions(self):
    identity = str(uuid4())
    config = dict(id=identity, host='127.0.0.1', port=8769, token='a' * 64)
    value = observation() | {
        'controlConnected': True,
        'busy': True,
        'residentSessions': 2,
    }
    with (
        patch(
            'model_explorer_debugger.runtime.macos_device.configurations',
            return_value={identity: config},
        ),
        patch(
            'model_explorer_debugger.runtime.macos_device.MacConnection',
            side_effect=AssertionError('No control acquisition'),
        ),
        patch(
            'model_explorer_debugger.runtime.runner_discovery.read_observation',
            return_value=value,
        ),
    ):
      result = inspect('/unused', 'macos:' + identity)
      self.assertEqual(result['status'], 'busy')
      self.assertEqual(result['runner']['residentSessions'], 2)
      self.assertNotIn('token', json.dumps(result))

  def test_unpaired_phone_is_not_launched_or_reprovisioned(self):
    with (
        tempfile.TemporaryDirectory() as root,
        patch(
            'model_explorer_debugger.runtime.ios_device.devicectl',
            side_effect=AssertionError('No App lifecycle operations'),
        ),
        patch(
            'model_explorer_debugger.runtime.ios_device.USBConnection',
            side_effect=AssertionError('No control acquisition'),
        ),
    ):
      result = inspect(root, 'ios:00000000-0000000000000000')
      self.assertEqual(result['status'], 'unpaired')
      self.assertFalse(list(Path(root).iterdir()))

  def test_owner_is_public_and_identifies_this_server_without_acquiring_control(
      self,
  ):
    server_id, session_id, identity = str(uuid4()), str(uuid4()), str(uuid4())
    owner = dict(
        serverId=server_id.upper(),
        serverName='Synthetic server',
        sessionId=session_id,
        sessionName='Synthetic Session',
        runId='ref',
        secret='never expose',
    )
    value = descriptor(observation() | {'owner': owner, 'lifecycle': 'active'})
    with tempfile.TemporaryDirectory() as root:
      (Path(root) / 'server-identity.json').write_text(
          json.dumps({'id': server_id})
      )
      with (
          patch(
              'model_explorer_debugger.runtime.macos_device.configurations',
              return_value={
                  identity: dict(host='127.0.0.1', port=8769, token='a' * 64)
              },
          ),
          patch(
              'model_explorer_debugger.runtime.runner_discovery'
              '.read_observation',
              return_value=value,
          ),
      ):
        result = inspect(root, 'macos:' + identity)
    self.assertEqual(result['runnerState'], 'active')
    self.assertTrue(result['runner']['owner']['isCurrentServer'])
    self.assertIn('This server', result['reason'])
    self.assertNotIn('secret', json.dumps(result))

  def test_protocol_version_constant_governs_support(self):
    from model_explorer_debugger.runtime.protocol import PROTOCOL_VERSION

    identity = str(uuid4())
    for version, status in (
        (PROTOCOL_VERSION, 'ready'),
        (PROTOCOL_VERSION + 1, 'unsupported'),
        (None, 'unsupported'),
    ):
      with (
          self.subTest(version=version),
          patch(
              'model_explorer_debugger.runtime.macos_device.configurations',
              return_value={
                  identity: dict(host='127.0.0.1', port=8769, token='a' * 64)
              },
          ),
          patch(
              'model_explorer_debugger.runtime.runner_discovery'
              '.read_observation',
              return_value=observation() | {'protocolVersion': version},
          ),
      ):
        self.assertEqual(
            inspect('/unused', 'macos:' + identity)['status'], status
        )

  def test_legacy_control_protocol_is_not_selectable(self):
    identity = str(uuid4())
    with (
        patch(
            'model_explorer_debugger.runtime.macos_device.configurations',
            return_value={
                identity: dict(host='127.0.0.1', port=8769, token='a' * 64)
            },
        ),
        patch(
            'model_explorer_debugger.runtime.runner_discovery.read_observation',
            return_value=observation() | {'protocolVersion': 2},
        ),
    ):
      result = inspect('/unused', 'macos:' + identity)
    self.assertEqual(result['status'], 'unsupported')

  def test_unreachable_is_not_free_without_positive_process_evidence(self):
    identity = str(uuid4())
    with (
        patch(
            'model_explorer_debugger.runtime.macos_device.configurations',
            return_value={
                identity: dict(host='127.0.0.1', port=8769, token='a' * 64)
            },
        ),
        patch(
            'model_explorer_debugger.runtime.runner_discovery.read_observation',
            side_effect=OSError('offline'),
        ),
        patch(
            'model_explorer_debugger.runtime.runner_builds.process_state',
            return_value={
                'connectionState': 'unknown',
                'runnerState': 'unknown',
            },
        ) as process_state,
    ):
      self.assertEqual(
          inspect('/unused', 'macos:' + identity)['runnerState'], 'unknown'
      )
      process_state.return_value = {
          'connectionState': 'connected',
          'runnerState': 'not_running',
      }
      result = inspect('/unused', 'macos:' + identity)
      self.assertEqual(result['runnerState'], 'not_running')
      self.assertEqual(result['status'], 'ready')

  def test_closed_connection_is_dropped_before_explicit_initialization(self):
    manager = RunnerDevices('/unused')
    closed = []
    manager.connections['a'] = SimpleNamespace(
        is_open=False, close=lambda: closed.append(True)
    )
    manager.sessions = {
        ('first', 'ref', 'a'): {},
        ('second', 'target', 'a'): {},
        ('other', 'ref', 'b'): {},
    }
    with self.assertRaisesRegex(ConnectionError, 'reconnection is disabled'):
      manager.reusable_connection('a', initializing=True)
    self.assertEqual(closed, [True])
    self.assertNotIn('a', manager.connections)
    self.assertEqual(list(manager.sessions), [('other', 'ref', 'b')])

  def test_closed_generate_requires_explicit_reinitialization_without_replay(
      self,
  ):
    manager = RunnerDevices('/unused')
    peer = MagicMock()
    peer.is_open = False
    manager.connections['a'] = peer
    with self.assertRaisesRegex(ConnectionError, 'reconnection is disabled'):
      manager.reusable_connection('a', initializing=False)
    peer.send.assert_not_called()

  def test_healthy_cached_connection_is_retained(self):
    manager = RunnerDevices('/unused')
    peer = MagicMock()
    peer.is_open = True
    manager.connections['a'] = peer
    self.assertIs(manager.reusable_connection('a', initializing=True), peer)
    peer.close.assert_not_called()


if __name__ == '__main__':
  unittest.main()
