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

"""Transport failure tests. Synthetic peers never count as device evidence."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model_explorer_debugger.runtime.ios_device import (
    IOSDevices,
    USBConnection,
    device_id,
    list_devices,
    validate_request,
)
from native_fixtures import synthetic_manifest

UDID = '00000000-0000000000000000'


class SyntheticPeer:

  def __init__(self, mode, cancelled):
    self.mode = mode
    self.cancelled = cancelled
    self.sent = []
    self.closed = False

  def ensure_model(self, *args, **kwargs):
    pass

  def send(self, message):
    self.sent.append(message)

  def receive(self, timeout):
    identity = self.sent[0]['requestId']
    if self.mode == 'wrong_id':
      return {'type': 'delta', 'requestId': 'wrong-request', 'text': 'unsafe'}
    if self.sent[-1]['type'] == 'cancel':
      return {
          'type': 'error',
          'requestId': identity,
          'error': 'Native operation cancelled',
          'generationStatus': 'stopped',
          'dumpStatus': 'unavailable',
      }
    self.cancelled[0] = True
    return {'type': 'delta', 'requestId': identity, 'text': 'first token'}

  def close(self):
    self.closed = True


class IOSUSBTests(unittest.TestCase):

  def request(self, root):
    model = root / 'model.litertlm'
    model.write_bytes(b'explicitly synthetic model')
    return dict(
        session_id='11111111-1111-4111-8111-111111111111',
        run={'id': 'ref', 'device': 'ios:' + UDID, 'backend': 'CPU'},
        operation='generate',
        generation={},
        max_output_tokens=2,
        prompt='hello',
        manifest='synthetic-manifest',
        runtime_root=str(root),
        model=str(model),
        model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        output=str(root / 'output'),
    )

  def test_device_identifiers_cannot_be_paths_or_options(self):
    for value in ('ios:../../file', 'ios:--help', 'mac:1234567890123456', None):
      with self.subTest(value=value), self.assertRaises(ValueError):
        device_id(value)

  def test_discovery_excludes_simulator_and_wireless_devices(self):
    devices = [
        {
            'hardwareProperties': {'platform': 'iOS', 'udid': UDID + str(i)},
            'connectionProperties': {'transportType': kind},
        }
        for i, kind in enumerate(('wired', 'sameMachine', 'network'))
    ]
    with (
        patch(
            'model_explorer_debugger.runtime.ios_device.devicectl',
            return_value={'devices': devices},
        ),
        patch(
            'model_explorer_debugger.runtime.ios_device.shutil.which',
            return_value='/xcrun',
        ),
    ):
      self.assertEqual(len(list_devices()), 1)
      self.assertEqual(list_devices()[0]['connection'], 'wired')

  def test_unsupported_options_fail_before_connecting(self):
    with tempfile.TemporaryDirectory() as temporary:
      request = self.request(Path(temporary))
      for change in (
          {'generation': {'temperature': 0.7}},
          {'max_output_tokens': 256},
          {'manifest': None},
          {'run': {'device': 'ios:' + UDID, 'backend': 'GPU'}},
      ):
        with self.subTest(change=change), self.assertRaises(ValueError):
          validate_request(request | change)

  def test_interrupted_model_transfer_is_not_committed_to_cache(self):
    peer = USBConnection.__new__(USBConnection)
    sent = []
    peer.send = sent.append
    peer.receive = lambda timeout: {
        'type': 'model_info',
        'requestId': sent[0]['requestId'],
        'available': False,
    }

    def interrupted(*args, **kwargs):
      raise InterruptedError('USB cable removed during transfer')

    peer.copy_to = interrupted
    with self.assertRaises(InterruptedError):
      peer.ensure_model(
          Path('synthetic-model'), 'a' * 64, lambda *args, **kwargs: None
      )
    self.assertEqual([command['type'] for command in sent], ['model_info'])

  def run_failure(self, mode):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      request = self.request(root)
      cancelled = [False]
      peer = SyntheticPeer(mode, cancelled)
      manager = IOSDevices(root)
      manager.connections[UDID] = peer
      manager.sessions[(request['session_id'], 'ref', UDID)] = {
          'config': {
              k: request.get(k)
              for k in (
                  'model',
                  'model_sha256',
                  'manifest',
                  'generation',
                  'run',
              )
          },
          'manifest': synthetic_manifest(),
          'messages': [],
      }
      events = []
      with patch(
          'model_explorer_debugger.runtime.litert_lm_adapter.verify_taps',
          return_value=synthetic_manifest(),
      ):
        result = manager.execute(
            request,
            lambda kind, **data: events.append((kind, data)),
            lambda: cancelled[0],
        )
      if mode == 'cancel':
        self.assertFalse(peer.closed)
        self.assertIs(manager.connections[UDID], peer)
        self.assertFalse(manager.sessions)
      else:
        self.assertTrue(peer.closed)
        self.assertNotIn(UDID, manager.connections)
      self.assertEqual(
          result['generation_status'],
          'stopped' if mode == 'cancel' else 'failed',
      )
      self.assertFalse(result['output_confirmed'])
      self.assertFalse(result['dump_complete'])
      self.assertEqual(result['connection_lost'], mode != 'cancel')
      self.assertFalse((root / 'output/export').exists())
      self.assertEqual(
          sum(message['type'] == 'generate' for message in peer.sent), 1
      )
      return peer, events

  def test_cancel_reaches_phone_and_never_publishes_partial_capture(self):
    peer, events = self.run_failure('cancel')
    self.assertEqual([m['type'] for m in peer.sent], ['generate', 'cancel'])
    self.assertEqual(events[0], ('delta', {'text': 'first token'}))

  def test_wrong_request_event_closes_connection_without_replaying_prompt(self):
    _, events = self.run_failure('wrong_id')
    self.assertEqual(events, [])


if __name__ == '__main__':
  unittest.main()
