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

"""Terminal dump durability, cancellation and receipt failure boundaries."""

import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from model_explorer_debugger.runtime.device_access import slot_config
from model_explorer_debugger.runtime.macos_device import (
    MacConnection,
    MacDevices,
)
from model_explorer_debugger.runtime.runner_artifacts import RunnerArtifacts
from model_explorer_debugger.runtime.runners import Runners
from native_fixtures import synthetic_manifest


class Peer(RunnerArtifacts):

  def __init__(self, status='succeeded'):
    self.run, self.manifest_id = str(uuid4()), str(uuid4())
    self.payload = {
        'terminal.json': (
            json.dumps({'jobID': self.run, 'generationStatus': status}).encode()
        ),
        'raw/partial.data': b'raw evidence',
    }
    self.manifest = dict(
        manifestId=self.manifest_id,
        generationStatus=status,
        files=[
            dict(
                path=path,
                size=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
            for path, data in self.payload.items()
        ],
    )
    self.sent = []
    self.is_open = True
    self.receipt_error = None
    self.corrupt = False
    self.destination = None

  def exchange(self, kind, **fields):
    self.sent.append((kind, fields))
    if kind == 'run_files':
      return self.manifest
    if kind == 'file_chunk':
      assert fields['manifestId'] == self.manifest_id
      data = self.payload[fields['path']][fields['offset'] :]
      return dict(
          offset=fields['offset'],
          data=base64.b64encode(
              b'!' * len(data) if self.corrupt else data
          ).decode(),
      )
    assert kind == 'run_received'
    assert fields == dict(run=self.run, manifestId=self.manifest_id)
    assert self.destination.is_dir()
    assert (self.destination / 'received-manifest.json').is_file()
    for path, data in self.payload.items():
      assert (self.destination / path).read_bytes() == data
    if self.receipt_error:
      raise self.receipt_error
    return dict(deleted=True)

  def receive_dump(self, root, cancelled=lambda: False):
    self.destination = Path(root) / 'dump'
    return self.copy_from(
        'Documents/Runs/' + self.run, self.destination, cancelled=cancelled
    )


class ArtifactTests(unittest.TestCase):

  def test_stopped_dump_is_fully_received_despite_generation_cancel_flag(self):
    with tempfile.TemporaryDirectory() as root:
      peer = Peer('stopped')
      result = peer.receive_dump(root, cancelled=lambda: True)
      self.assertTrue(result['dump_complete'])
      self.assertEqual(result['dump_generation_status'], 'stopped')
      self.assertEqual(peer.sent[-1][0], 'run_received')

  def test_failed_dump_with_failure_record_is_received(self):
    # NativeRunner writes failure.json on every failed/stopped run and
    # RunnerFiles lists it in run_files; the Server allowlist must accept it.
    with tempfile.TemporaryDirectory() as root:
      peer = Peer('failed')
      peer.payload['failure.json'] = json.dumps(
          {'status': 'failed', 'error': 'engine error'}
      ).encode()
      peer.manifest['files'] = [
          dict(
              path=path, size=len(data), sha256=hashlib.sha256(data).hexdigest()
          )
          for path, data in peer.payload.items()
      ]
      result = peer.receive_dump(root)
      self.assertTrue(result['dump_complete'])
      self.assertEqual(result['dump_generation_status'], 'failed')
      self.assertEqual(
          json.loads((peer.destination / 'failure.json').read_text())['status'],
          'failed',
      )
      self.assertEqual(peer.sent[-1][0], 'run_received')

  def test_integrity_failure_never_receipts_or_publishes_staging(self):
    with tempfile.TemporaryDirectory() as root:
      peer = Peer()
      peer.corrupt = True
      with self.assertRaisesRegex(ValueError, 'checksum'):
        peer.receive_dump(root)
      self.assertNotIn('run_received', [kind for kind, _ in peer.sent])
      self.assertEqual(list(Path(root).iterdir()), [])

  def test_fsync_failure_never_sends_receipt(self):
    with tempfile.TemporaryDirectory() as root:
      peer = Peer()
      with patch(
          'model_explorer_debugger.runtime.runner_artifacts.os.fsync',
          side_effect=OSError('disk failure'),
      ):
        with self.assertRaises(OSError):
          peer.receive_dump(root)
      self.assertNotIn('run_received', [kind for kind, _ in peer.sent])

  def test_receipt_disconnect_preserves_complete_raw_without_retry(self):
    with tempfile.TemporaryDirectory() as root:
      peer = Peer()
      peer.receipt_error = ConnectionError('lost')
      result = peer.receive_dump(root)
      self.assertTrue(result['dump_complete'])
      self.assertTrue(result['connection_lost'])
      self.assertTrue(peer.destination.is_dir())
      self.assertEqual(sum(kind == 'run_received' for kind, _ in peer.sent), 1)

  def test_deletion_failure_preserves_data_and_records_only_cleanup_error(self):
    with tempfile.TemporaryDirectory() as root:
      peer = Peer()
      peer.receipt_error = ValueError('cannot delete')
      result = peer.receive_dump(root)
      self.assertTrue(result['dump_complete'])
      self.assertFalse(result['connection_lost'])
      self.assertEqual(result['cleanup_error'], 'cannot delete')

  def test_traversal_or_duplicate_manifest_is_rejected_before_transfer(self):
    for unsafe in ('../x', '/tmp/x', 'raw/../../x', 'raw//x'):
      with self.subTest(unsafe=unsafe), tempfile.TemporaryDirectory() as root:
        peer = Peer()
        peer.manifest['files'][1]['path'] = unsafe
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
          peer.receive_dump(root)
        self.assertEqual([kind for kind, _ in peer.sent], ['run_files'])


class SlotTests(unittest.TestCase):

  def test_two_roles_have_distinct_endpoints_credentials_and_connection_keys(
      self,
  ):
    config = dict(host='127.0.0.1', port=8769, token='a' * 64)
    self.assertEqual(slot_config(config, 'ref'), config)
    self.assertEqual(slot_config(config, 'target')['port'], 8770)
    self.assertNotEqual(slot_config(config, 'target')['token'], config['token'])
    devices = MacDevices('/unused')
    device = 'macos:' + str(uuid4())
    keys = [
        devices.connection_key(
            {'run': {'device': device, '_runner_slot': role}}
        )
        for role in ('ref', 'target')
    ]
    self.assertNotEqual(*keys)

  def test_same_mac_two_slots_but_phone_rejected(self):
    runners = Runners('/unused')
    mac = 'macos:' + str(uuid4())
    with patch(
        'model_explorer_debugger.runtime.macos_device.configurations',
        return_value={},
    ):
      runs = runners.resolve_runs(
          [dict(id=role, device=mac) for role in ('ref', 'target')]
      )
    self.assertEqual([r['_runner_slot'] for r in runs], ['ref', 'target'])
    with self.assertRaisesRegex(ValueError, 'only one Runner'):
      runners.resolve_runs([
          dict(id=role, device='ios:00000000-0000000000000000')
          for role in ('ref', 'target')
      ])

  def test_lost_execution_cannot_reacquire_even_with_initialize(self):
    devices = MacDevices('/unused')
    identity = str(uuid4())
    session = str(uuid4())
    request = dict(
        operation='initialize',
        session_id=session,
        execution_id='attempt',
        run=dict(id='ref'),
    )
    devices.lost_executions.add((identity, 'attempt'))
    with patch.object(devices, 'connect') as connect:
      with self.assertRaisesRegex(ConnectionError, 'reconnection is disabled'):
        devices.acquire(identity, request)
      connect.assert_not_called()

  def test_legacy_v3_handshake_rejected_before_ownership(self):
    socket = MagicMock()
    socket.recv.return_value = json.dumps(
        dict(
            type='hello',
            protocolVersion=3,
            platform='macOS',
            debuggerEnabled=True,
            fileTransfer=True,
        )
    )
    with patch('websockets.sync.client.connect', return_value=socket):
      with self.assertRaisesRegex(ValueError, 'compatible v4'):
        MacConnection(dict(host='127.0.0.1', port=8769, token='a' * 64))
    socket.send.assert_not_called()


class NativeTerminalTests(unittest.TestCase):

  def _confirmed_terminal(self, failure, connection_open):
    from model_explorer_debugger.runtime.runner_device import RunnerDevices

    class Adapter(RunnerDevices):
      platform = 'macOS'
      transport = 'synthetic'
      capture_folder = 'raw'
      device_id = staticmethod(lambda value: value)
      validate = staticmethod(lambda request: None)

    with tempfile.TemporaryDirectory() as root:
      adapter = Adapter(root)
      session = str(uuid4())
      device = 'device'
      request = dict(
          operation='generate',
          session_id=session,
          execution_id='execution',
          runtime_root=root,
          model=root + '/model',
          model_sha256='a' * 64,
          manifest='manifest',
          output=root + '/turn/ref',
          prompt='hello',
          run=dict(id='ref', device=device),
          generation={},
      )
      config = {
          k: request.get(k)
          for k in ('model', 'model_sha256', 'manifest', 'generation', 'run')
      }
      adapter.sessions[(session, 'ref', device)] = dict(
          config=config,
          manifest=synthetic_manifest(),
          messages=[],
          runtime_instance='runtime',
          token_count=0,
          turn_sequence=0,
      )
      peer = MagicMock()
      peer.is_open = True
      peer.transport_info = {}

      def receive(timeout):
        wire = peer.send.call_args.args[0]['requestId']
        return dict(
            type='completed',
            requestId=wire,
            generationStatus='succeeded',
            output='complete answer',
            runDirectory='Documents/Runs/' + wire.upper(),
            result=dict(
                jobID=wire,
                modelSHA256='a' * 64,
                platform='macOS',
                backend='CPU',
                debuggerEnabled=True,
                input='hello',
                contextLength=1024,
                maxOutputTokens=32,
                thinkingEnabled=False,
                speculativeDecodingEnabled=False,
                sampler=dict(seed=0, temperature=0, topK=1, topP=1),
                runtimeInstance='runtime',
                modelLoadCount=1,
                tokenCountBefore=0,
                tokenCount=1,
                turnSequence=1,
                output='complete answer',
                tensors=[],
                elapsedSecondsWithCapture=1,
            ),
        )

      peer.receive.side_effect = receive

      def failed_copy(*args, **kwargs):
        peer.is_open = connection_open
        raise failure

      peer.copy_from.side_effect = failed_copy
      adapter.connections[device] = peer
      events = []
      with patch(
          'model_explorer_debugger.runtime.runner_device.subprocess.run'
      ) as importer:
        result = adapter.execute(
            request,
            lambda kind, **data: events.append((kind, data)),
            lambda: False,
        )
        importer.assert_not_called()
      self.assertEqual(result['output'], 'complete answer')
      self.assertEqual(result['generation_status'], 'completed')
      self.assertTrue(result['output_confirmed'])
      self.assertEqual(result['connection_lost'], not connection_open)
      self.assertFalse(result['dump_complete'])
      self.assertEqual(events[0][0], 'generation_completed')
      self.assertEqual(
          (device, 'execution') in adapter.lost_executions, not connection_open
      )
      if connection_open:
        self.assertIs(adapter.connections[device], peer)

  def test_confirmed_text_survives_disconnect_during_dump_and_import_never_runs(
      self,
  ):
    self._confirmed_terminal(ConnectionError('lost during transfer'), False)

  def test_server_disk_failure_with_live_channel_preserves_runner_and_text(
      self,
  ):
    self._confirmed_terminal(OSError('No space left on Server'), True)

  def test_stopped_terminal_still_retrieves_raw_without_cancel_predicate(self):
    from model_explorer_debugger.runtime.runner_device import RunnerDevices

    class Adapter(RunnerDevices):
      platform = 'macOS'
      transport = 'synthetic'
      capture_folder = 'raw'
      device_id = staticmethod(lambda value: value)
      validate = staticmethod(lambda request: None)

    with tempfile.TemporaryDirectory() as root:
      adapter = Adapter(root)
      session = str(uuid4())
      device = 'device'
      request = dict(
          operation='generate',
          session_id=session,
          execution_id='execution',
          runtime_root=root,
          model=root + '/model',
          model_sha256='a' * 64,
          manifest='manifest',
          output=root + '/turn/ref',
          prompt='hello',
          run=dict(id='ref', device=device),
          generation={},
      )
      config = {
          k: request.get(k)
          for k in ('model', 'model_sha256', 'manifest', 'generation', 'run')
      }
      adapter.sessions[(session, 'ref', device)] = dict(
          config=config, manifest=synthetic_manifest(), messages=[]
      )
      peer = MagicMock()
      peer.is_open = True
      peer.transport_info = {}
      sent = []
      peer.send.side_effect = sent.append
      peer.receive.side_effect = lambda timeout: dict(
          type='error',
          requestId=sent[0]['requestId'],
          generationStatus='stopped',
          runDirectory='Documents/Runs/' + sent[0]['requestId'].upper(),
          error='Stopped',
      )
      peer.copy_from.return_value = {'dump_complete': True}
      adapter.connections[device] = peer
      result = adapter.execute(request, lambda *a, **kw: None, lambda: True)
      self.assertEqual(result['generation_status'], 'stopped')
      self.assertTrue(result['dump_complete'])
      self.assertEqual(peer.copy_from.call_args.kwargs, {})
      self.assertEqual(
          [event['type'] for event in sent], ['generate', 'cancel']
      )
      self.assertIs(adapter.connections[device], peer)


class TerminalBoundaryTests(unittest.TestCase):

  def test_provenance_mismatch_never_confirms_output_but_raw_is_retrieved(self):
    from model_explorer_debugger.runtime.runner_device import RunnerDevices

    class Adapter(RunnerDevices):
      platform = 'macOS'
      transport = 'synthetic'
      capture_folder = 'raw'
      device_id = staticmethod(lambda value: value)
      validate = staticmethod(lambda request: None)

    with tempfile.TemporaryDirectory() as root:
      manager = Adapter(root)
      device = 'device'
      session = str(uuid4())
      request = dict(
          operation='generate',
          session_id=session,
          runtime_root=root,
          model='model',
          model_sha256='a' * 64,
          manifest='manifest',
          output=root + '/out',
          prompt='hello',
          run=dict(id='ref', device=device),
          generation={},
      )
      config = {
          k: request.get(k)
          for k in ('model', 'model_sha256', 'manifest', 'generation', 'run')
      }
      manager.sessions[(session, 'ref', device)] = dict(
          config=config, manifest=synthetic_manifest(), messages=[]
      )
      peer = MagicMock()
      peer.is_open = True
      peer.transport_info = {}
      sent = []
      peer.send.side_effect = sent.append
      peer.receive.side_effect = lambda timeout: dict(
          type='completed',
          requestId=sent[0]['requestId'],
          generationStatus='succeeded',
          runDirectory='Documents/Runs/' + sent[0]['requestId'].upper(),
          output='wrong model output',
          result=dict(jobID=sent[0]['requestId'], modelSHA256='b' * 64),
      )
      peer.copy_from.return_value = dict(
          dump_complete=True, dump_generation_status='succeeded'
      )
      manager.connections[device] = peer
      events = []
      result = manager.execute(
          request, lambda kind, **data: events.append(kind), lambda: False
      )
      self.assertEqual(result['generation_status'], 'failed')
      self.assertFalse(result['output_confirmed'])
      self.assertEqual(result['output'], '')
      self.assertNotIn('generation_completed', events)
      peer.copy_from.assert_called_once()
      self.assertFalse(result['connection_lost'])
      self.assertIs(manager.connections[device], peer)

  def test_unverified_initialize_reports_reason_and_keeps_link_for_close(self):
    import hashlib
    from model_explorer_debugger.runtime.runner_device import RunnerDevices
    from model_explorer_debugger.runtime.turn_outcome import RunnerRefused

    peer = MagicMock()
    peer.is_open = True
    peer.transport_info = {}
    sent = []
    peer.send.side_effect = sent.append
    peer.receive.side_effect = lambda timeout: dict(
        type='completed',
        requestId=sent[0]['requestId'],
        generationStatus='succeeded',
        output='',
        result=dict(jobID=sent[0]['requestId'], modelSHA256='b' * 64),
    )

    class Adapter(RunnerDevices):
      platform = 'macOS'
      transport = 'synthetic'
      capture_folder = 'raw'
      device_id = staticmethod(lambda value: value)
      validate = staticmethod(lambda request: None)
      connect = lambda self, identity: peer

    with tempfile.TemporaryDirectory() as root:
      model = Path(root) / 'model'
      model.write_bytes(b'explicitly synthetic model')
      manager = Adapter(root)
      session = str(uuid4())
      device = 'device'
      request = dict(
          operation='initialize',
          session_id=session,
          runtime_root=root,
          model=str(model),
          model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
          manifest='manifest',
          output=root + '/out',
          run=dict(id='ref', device=device),
          generation={},
      )
      with patch(
          'model_explorer_debugger.runtime.litert_lm_adapter.verify_taps',
          return_value=synthetic_manifest(),
      ):
        with self.assertRaisesRegex(RunnerRefused, 'provenance mismatch'):
          manager.execute(request, lambda *a, **kw: None, lambda: False)
      self.assertIs(manager.connections[device], peer)
      peer.close.assert_not_called()
      self.assertFalse(manager.sessions)

  def test_capacity_refusal_and_runtime_preflight_error_are_distinct(self):
    from model_explorer_debugger.runtime.runner_device import RunnerDevices
    from model_debugger_contracts.errors import InputRejected

    class Adapter(RunnerDevices):
      platform = 'macOS'
      transport = 'synthetic'
      capture_folder = 'raw'
      device_id = staticmethod(lambda value: value)
      validate = staticmethod(lambda request: None)

    for kind, code, exception in [
        ('preflight', None, InputRejected),
        ('error', 'preflight_failed', RuntimeError),
    ]:
      with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
        manager = Adapter(root)
        session = str(uuid4())
        device = 'device'
        request = dict(
            operation='preflight',
            session_id=session,
            runtime_root=root,
            model='model',
            model_sha256='a' * 64,
            manifest='manifest',
            output=root + '/out',
            prompt='hello',
            run=dict(id='ref', device=device),
            generation={},
        )
        config = {
            k: request.get(k)
            for k in ('model', 'model_sha256', 'manifest', 'generation', 'run')
        }
        manager.sessions[(session, 'ref', device)] = dict(
            config=config, manifest=synthetic_manifest(), messages=[]
        )
        peer = MagicMock()
        peer.is_open = True
        sent = []
        peer.send.side_effect = sent.append
        peer.receive.side_effect = lambda timeout: dict(
            type=kind,
            requestId=sent[0]['requestId'],
            accepted=False,
            errorCode=code,
            error='reason',
        )
        manager.connections[device] = peer
        with self.assertRaises(exception):
          manager.execute(request, lambda *a, **kw: None, lambda: False)
        self.assertIs(manager.connections[device], peer)
        self.assertEqual(len(sent), 1)
        if exception is InputRejected:
          self.assertIn((session, 'ref', device), manager.sessions)
        else:
          self.assertNotIn((session, 'ref', device), manager.sessions)


if __name__ == '__main__':
  unittest.main()
