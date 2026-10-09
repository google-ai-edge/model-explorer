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

"""Synthetic transport failures; native App proof is recorded separately."""

import base64
import hashlib
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from model_explorer_debugger.runtime.macos_device import (
    MacConnection,
    bridge_endpoint,
    configurations,
    connect_bridge,
    device_id,
    read_config,
    register,
)


class MacDeviceTests(unittest.TestCase):

  @unittest.skipUnless(
      sys.platform == 'darwin', 'Darwin IP_BOUND_IF socket contract'
  )
  def test_connected_socket_has_no_connection_timeout(self):
    # Exercise the actual preconnected-socket handoff used by websockets.
    # Its supplied-socket path preserves our mode, so a TCP connect timeout
    # would otherwise also terminate an idle, healthy WebSocket Session.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
      listener.bind(('127.0.0.1', 0))
      listener.listen(1)
      with patch(
          'model_explorer_debugger.runtime.macos_device.bridge_endpoint',
          return_value=(socket.if_nametoindex('lo0'), 'lo0', '127.0.0.1'),
      ):
        peer, _ = connect_bridge('127.0.0.1', listener.getsockname()[1])
      try:
        self.assertIsNone(peer.gettimeout())
      finally:
        peer.close()

  def test_bridge_selection_ignores_wifi_and_inactive_interfaces(self):
    def info(command, **kwargs):
      return (
          'inet 169.254.43.3 netmask 0xffff0000\nstatus: active'
          if command[1] == 'bridge0'
          else 'inet 169.254.1.2 netmask 0xffff0000\nstatus: inactive'
      )

    with (
        patch(
            'model_explorer_debugger.runtime.macos_device.socket.if_nameindex',
            return_value=[(14, 'en0'), (15, 'bridge0'), (17, 'bridge1')],
        ),
        patch(
            'model_explorer_debugger.runtime.macos_device.subprocess'
            '.check_output',
            side_effect=info,
        ),
    ):
      self.assertEqual(
          bridge_endpoint('169.254.53.21'), (15, 'bridge0', '169.254.43.3')
      )
      with self.assertRaises(ValueError):
        bridge_endpoint('10.1.2.3')

  def test_socket_is_bound_to_bridge_and_failure_cannot_fallback(self):
    from unittest.mock import MagicMock

    peer = MagicMock()
    peer.connect.side_effect = ConnectionRefusedError('synthetic closed peer')
    with (
        patch(
            'model_explorer_debugger.runtime.macos_device.sys.platform',
            'darwin',
        ),
        patch(
            'model_explorer_debugger.runtime.macos_device.bridge_endpoint',
            return_value=(15, 'bridge0', '169.254.43.3'),
        ),
        patch(
            'model_explorer_debugger.runtime.macos_device.socket.socket',
            return_value=peer,
        ),
    ):
      with self.assertRaises(ConnectionRefusedError):
        connect_bridge('169.254.53.21', 8769)
      import socket

      peer.setsockopt.assert_called_once_with(socket.IPPROTO_IP, 25, 15)
      peer.bind.assert_called_once_with(('169.254.43.3', 0))
      peer.connect.assert_called_once_with(('169.254.53.21', 8769))
      peer.close.assert_called_once()

  def config(self):
    return dict(
        id=str(uuid4()),
        name='Synthetic Mac',
        host='169.254.10.2',
        port=8769,
        token='a' * 64,
    )

  def test_invalid_device_identifiers(self):
    for value in ('macos:../../a', 'macos:--help', 'ios:' + str(uuid4()), None):
      with self.subTest(value=value), self.assertRaises(ValueError):
        device_id(value)

  def test_remote_registration_is_private_and_discovery_does_not_expose_key(
      self,
  ):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      source = root / 'connection.json'
      config = self.config()
      source.write_text(json.dumps(config))
      identity = register(root, source)
      self.assertEqual(identity, 'macos:' + config['id'])
      saved = root / 'macos-devices' / (config['id'] + '.json')
      self.assertEqual(saved.stat().st_mode & 0o777, 0o600)
      with patch(
          'model_explorer_debugger.runtime.macos_device.LOCAL_CONFIG',
          root / 'absent.json',
      ):
        from model_explorer_debugger.runtime.macos_device import list_devices

        self.assertNotIn('token', list_devices(root)[0])
        self.assertEqual(configurations(root)[config['id']], config)

  def test_invalid_connection_file_fails_closed(self):
    with tempfile.TemporaryDirectory() as temporary:
      source = Path(temporary) / 'connection.json'
      for change in (
          {'host': '0.0.0.0'},
          {'host': '8.8.8.8'},
          {'host': 'example.com'},
          {'port': 22},
          {'token': ''},
      ):
        source.write_text(json.dumps(self.config() | change))
        with self.subTest(change=change), self.assertRaises(ValueError):
          read_config(source)

  def test_cancelled_upload_never_commits(self):
    with tempfile.TemporaryDirectory() as temporary:
      path = Path(temporary) / 'model'
      path.write_bytes(b'synthetic model')
      peer = MacConnection.__new__(MacConnection)
      sent = []
      peer.exchange = lambda kind, **fields: sent.append(kind) or {
          'available': False
      }
      closed = []
      peer.close = lambda: closed.append(True)
      with self.assertRaises(InterruptedError):
        peer.ensure_model(
            path, 'a' * 64, lambda *args, **kwargs: None, lambda: True
        )
      self.assertEqual(sent, ['model_info', 'upload_begin'])
      self.assertTrue(closed)

  def test_unacknowledged_chunk_never_commits(self):
    with tempfile.TemporaryDirectory() as temporary:
      path = Path(temporary) / 'model'
      path.write_bytes(b'synthetic model')
      peer = MacConnection.__new__(MacConnection)
      sent = []
      peer.exchange = lambda kind, **fields: sent.append(kind) or {
          'available': False,
          'offset': 999,
      }
      peer.close = lambda: None
      with self.assertRaisesRegex(ValueError, 'offset'):
        peer.ensure_model(path, 'a' * 64, lambda *args, **kwargs: None)
      self.assertNotIn('upload_finish', sent)

  def test_capture_traversal_and_corruption_are_rejected(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      sha = hashlib.sha256(b'good').hexdigest()
      for i, name in enumerate(
          ('../escape', '/tmp/escape', 'tensors/../../escape', 'job.json')
      ):
        peer = MacConnection.__new__(MacConnection)
        files = [
            dict(path=name, size=4, sha256=sha),
            dict(path='result.json', size=4, sha256=sha),
            dict(path='runtime-build.json', size=4, sha256=sha),
        ]
        peer.exchange = (
            lambda kind, **fields: {
                'files': files,
                'manifestId': str(uuid4()),
                'generationStatus': 'succeeded',
            }
            if kind == 'run_files'
            else {'offset': 0, 'data': base64.b64encode(b'bad!').decode()}
        )
        with self.subTest(name=name), self.assertRaises(ValueError):
          peer.copy_from('Documents/Runs/' + str(uuid4()), root / str(i))
      self.assertFalse((root / 'escape').exists())


if __name__ == '__main__':
  unittest.main()
