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

"""The Server's side of the wire, driven against an in-memory Runner.

Every message the Server sends and every reply it is given is validated against
src/contracts' runner-wire schema, so the real code paths cannot drift from the
contract.
"""

from collections import deque
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from model_debugger_contracts.schema import validate_wire
from model_explorer_debugger.runtime import macos_device, runner_artifacts
from model_explorer_debugger.runtime.macos_device import MacConnection
from model_explorer_debugger.runtime.runner_artifacts import RunnerArtifacts
from model_explorer_debugger.runtime.runner_channel import (
    decode_frame,
    encode_frame,
)
from model_explorer_debugger.runtime.runner_device import RunnerDevices
from native_fixtures import synthetic_manifest


class WireRunner:
  """Answers like the Runner App.

  Models are under `models` and sealed run exports under `runs`.
  """

  def __init__(
      self, protocol=5, exports=None, link_root=None, refuses_links=False
  ):
    self.refuses_links = refuses_links
    self.protocol, self.hello = protocol, {
        'protocolVersion': protocol,
        'fileLink': link_root is not None,
    }
    self.transport_info = {'peer_ip': '127.0.0.1'}
    self.exports, self.link_root = exports or {}, link_root
    self.models, self.upload, self.deleted = {}, None, []
    self.replies, self.sent, self.most_outstanding = deque(), [], 0

  def send(self, message):
    self._accept(message, None)

  def send_binary(self, header, payload):
    header, data = decode_frame(encode_frame(header, payload)), payload
    header.pop('data')
    self._accept(header, data)

  def _accept(self, message, payload):
    validate_wire('server', message)
    if self.protocol < 5:
      assert (
          payload is None
          and 'encoding' not in message
          and 'link' not in message
      ), 'v5 message sent to a v4 Runner'
      assert not self.replies, 'a v4 Runner is given one file command at a time'
    self.sent.append(message)
    for reply in self._answer(message, payload):
      self.replies.append(dict(reply, requestId=message['requestId']))
    self.most_outstanding = max(self.most_outstanding, len(self.replies))

  def receive(self, timeout):
    if not self.replies:
      raise TimeoutError('Waiting for Runner response')
    reply = self.replies.popleft()
    validate_wire(
        'runner',
        {
            key: value
            for key, value in reply.items()
            if not isinstance(value, bytes)
        },
    )
    return reply

  def _answer(self, message, payload):
    kind = message['type']
    if kind == 'model_info':
      yield dict(type=kind, available=message['modelSHA256'] in self.models)
    elif kind == 'model_link':
      data = Path(message['path']).read_bytes()
      if (
          self.refuses_links
          or self.link_root is None
          or hashlib.sha256(data).hexdigest() != message['modelSHA256']
      ):
        yield dict(type='error', error='Linked model checksum mismatch.')
      else:
        self.models[message['modelSHA256']] = data
        yield dict(type=kind, available=True)
    elif kind == 'upload_begin':
      self.upload = (message['modelSHA256'], bytearray())
      yield dict(type=kind, offset=0)
    elif kind == 'upload_chunk':
      import base64

      data = (
          payload if payload is not None else base64.b64decode(message['data'])
      )
      assert message['offset'] == len(self.upload[1])
      self.upload[1].extend(data)
      yield dict(type=kind, offset=len(self.upload[1]))
    elif kind == 'upload_finish':
      self.models[self.upload[0]] = bytes(self.upload[1])
      yield dict(type=kind, available=True)
    elif kind == 'run_files':
      run = self.exports[message['run']]
      reply = dict(
          type=kind,
          manifestId=run['manifestId'],
          generationStatus='succeeded',
          files=[
              dict(
                  path=path,
                  size=len(data),
                  sha256=hashlib.sha256(data).hexdigest(),
              )
              for path, data in run['files'].items()
          ],
      )
      if message.get('link'):
        reply['root'] = str(self.link_root / 'Runs' / message['run'].upper())
      yield reply
    elif kind == 'file_chunk':
      import base64

      size = runner_artifacts.CHUNK
      data = self.exports[message['run']]['files'][message['path']][
          message['offset'] : message['offset'] + size
      ]
      reply = dict(
          type=kind, offset=message['offset'], manifestId=message['manifestId']
      )
      yield dict(
          reply,
          data=data
          if message.get('encoding') == 'binary'
          else base64.b64encode(data).decode(),
      )
    elif kind == 'run_received':
      self.deleted.append(message['run'])
      yield dict(type=kind, manifestId=message['manifestId'], deleted=True)
    elif kind == 'generate':
      job = message['job']
      yield dict(type='delta', text='hel')
      yield dict(type='delta', text='lo')
      yield dict(
          type='completed',
          generationStatus='succeeded',
          dumpStatus='ready',
          output='hello',
          runDirectory='Documents/Runs/' + job['id'].upper(),
          result=dict(
              jobID=job['id'].upper(),
              modelSHA256=job['modelSHA256'],
              platform='macOS',
              backend=job['backend'],
              debuggerEnabled=True,
              input=job['prompt'],
              output='hello',
              contextLength=job['contextLength'],
              maxOutputTokens=job['maxOutputTokens'],
              thinkingEnabled=False,
              speculativeDecodingEnabled=False,
              sampler=dict(seed=0, temperature=0, topK=1, topP=1),
              runtimeInstance='runtime',
              modelLoadCount=1,
              tokenCountBefore=0,
              tokenCount=9,
              turnSequence=1,
              tensors=[],
              elapsedSecondsWithCapture=0.1,
          ),
      )
    elif kind in ('close', 'reset'):
      yield dict(
          type={'close': 'closed', 'reset': 'reset'}[kind],
          sessionId=message['sessionId'],
      )
    else:
      raise AssertionError('Unexpected command ' + kind)


class Mac(WireRunner, MacConnection):
  is_open = True

  def close(self):
    pass


class Link(WireRunner, RunnerArtifacts):
  links_files, is_open = False, True


def export(root, run, files):
  """A sealed export, both in the Runner's memory and on the shared disk.

  The shared-disk copy is used for links.
  """
  directory = Path(root) / 'Runs' / run.upper()
  for path, data in files.items():
    (directory / path).parent.mkdir(parents=True, exist_ok=True)
    (directory / path).write_bytes(data)
  return {run: dict(manifestId=str(uuid4()), files=files)}


class ModelTransferTests(unittest.TestCase):

  def transfer(self, peer, data=bytes(range(200)) * 5):
    with (
        tempfile.TemporaryDirectory() as root,
        patch.object(macos_device, 'CHUNK', 64),
    ):
      model = Path(root) / 'model.litertlm'
      model.write_bytes(data)
      sha, events = hashlib.sha256(data).hexdigest(), []
      peer.ensure_model(
          model, sha, lambda kind, **fields: events.append(fields['message'])
      )
      self.assertEqual(peer.models[sha], data)
      return [message['type'] for message in peer.sent], events

  def test_a_runner_on_this_filesystem_links_the_model_instead_of_receiving_it(
      self,
  ):
    kinds, _ = self.transfer(Mac(link_root=Path('/')))
    self.assertEqual(kinds, ['model_info', 'model_link'])

  def test_a_refused_link_falls_back_to_sending_the_bytes(self):
    kinds, events = self.transfer(Mac(link_root=Path('/'), refuses_links=True))
    self.assertEqual(kinds[:3], ['model_info', 'model_link', 'upload_begin'])
    self.assertTrue(any('Model link unavailable' in event for event in events))

  def test_v5_sends_binary_chunks_inside_a_bounded_window(self):
    peer = Mac()
    kinds, _ = self.transfer(peer)
    self.assertEqual(
        kinds,
        ['model_info', 'upload_begin']
        + ['upload_chunk'] * 16
        + ['upload_finish'],
    )
    self.assertTrue(all('data' not in message for message in peer.sent))
    self.assertEqual(peer.most_outstanding, macos_device.FILE_WINDOW)

  def test_v4_is_still_driven_with_one_base64_chunk_at_a_time(self):
    peer = Mac(protocol=4)
    kinds, _ = self.transfer(peer)
    self.assertEqual(kinds.count('upload_chunk'), 16)
    self.assertEqual(peer.most_outstanding, 1)


class DumpReceptionTests(unittest.TestCase):
  files = {
      'terminal.json': json.dumps({'generationStatus': 'succeeded'}).encode(),
      'raw/000000.safetensors': bytes(range(251)) * 3,
  }

  def receive(self, protocol, link):
    with (
        tempfile.TemporaryDirectory() as root,
        patch.object(runner_artifacts, 'CHUNK', 100),
    ):
      run = str(uuid4())
      peer = Link(
          protocol=protocol,
          exports=export(root, run, self.files),
          link_root=Path(root) if link else None,
      )
      peer.links_files = link
      receipt = peer.copy_from(
          'Documents/Runs/' + run.upper(), Path(root) / 'received'
      )
      self.assertTrue(receipt['dump_complete'])
      self.assertEqual(peer.deleted, [run])
      for path, data in self.files.items():
        self.assertEqual((Path(root) / 'received' / path).read_bytes(), data)
      return peer

  def test_v5_pipelines_binary_chunks(self):
    peer = self.receive(5, link=False)
    chunks = [
        message for message in peer.sent if message['type'] == 'file_chunk'
    ]
    self.assertEqual(
        [
            message['offset']
            for message in chunks
            if message['path'].startswith('raw/')
        ],
        list(range(0, 753, 100)),
    )
    self.assertTrue(all(message['encoding'] == 'binary' for message in chunks))
    self.assertGreater(peer.most_outstanding, 1)

  def test_v4_requests_one_base64_chunk_at_a_time(self):
    self.assertEqual(self.receive(4, link=False).most_outstanding, 1)

  def test_a_shared_export_is_cloned_and_still_verified_and_acknowledged(self):
    peer = self.receive(5, link=True)
    self.assertEqual(
        [message['type'] for message in peer.sent],
        ['run_files', 'run_received'],
    )

  def test_a_shared_export_with_other_bytes_is_rejected(self):
    with tempfile.TemporaryDirectory() as root:
      run = str(uuid4())
      peer = Link(exports=export(root, run, self.files), link_root=Path(root))
      peer.links_files = True
      (
          Path(root) / 'Runs' / run.upper() / 'raw/000000.safetensors'
      ).write_bytes(b'changed after sealing')
      with self.assertRaisesRegex(ValueError, 'checksum'):
        peer.copy_from('Documents/Runs/' + run.upper(), Path(root) / 'received')
      self.assertFalse((Path(root) / 'received').exists())
      self.assertEqual(peer.deleted, [])

  def test_an_abandoned_pipeline_leaves_no_reply_for_the_next_request(self):
    with (
        tempfile.TemporaryDirectory() as root,
        patch.object(runner_artifacts, 'CHUNK', 100),
    ):
      run = str(uuid4())
      peer = Link(exports=export(root, run, self.files))
      original = peer._answer

      def answer(message, payload):
        # The sealed manifest promises more bytes than the file still has.
        for reply in original(message, payload):
          if reply['type'] == 'run_files':
            reply['files'][1]['size'] += 400
          yield reply

      peer._answer = answer
      with self.assertRaisesRegex(ValueError, 'Invalid capture chunk'):
        peer.copy_from('Documents/Runs/' + run.upper(), Path(root) / 'received')
      self.assertGreater(peer.most_outstanding, 1)
      self.assertFalse(peer.replies)


class OperationTests(unittest.TestCase):

  def test_generate_and_session_commands_are_valid_wire_messages(self):
    class Adapter(RunnerDevices):
      platform = 'macOS'
      transport = 'synthetic'
      capture_folder = 'raw'
      device_id = staticmethod(lambda value: value)
      validate = staticmethod(lambda request: None)

    with tempfile.TemporaryDirectory() as root:
      manager, session, device = Adapter(root), str(uuid4()), 'device'
      request = dict(
          operation='generate',
          session_id=session,
          chat_id=str(uuid4()),
          runtime_root=root,
          model='model',
          model_sha256='a' * 64,
          manifest='manifest',
          output=root + '/out',
          prompt='hello',
          run=dict(id='ref', device=device, backend='CPU'),
          generation={},
      )
      config = {
          key: request.get(key)
          for key in ('model', 'model_sha256', 'manifest', 'generation', 'run')
      }
      manager.sessions[(session, 'ref', device)] = dict(
          config=config,
          manifest=synthetic_manifest(),
          messages=[],
          runtime_instance='runtime',
          token_count=0,
          turn_sequence=0,
      )
      peer = Link()
      original = peer._answer

      def answer(message, payload):
        if message['type'] == 'generate':
          run = message['job']['id']
          peer.exports.update(
              export(
                  root,
                  run,
                  {
                      'terminal.json': (
                          json.dumps(
                              {'generationStatus': 'succeeded', 'jobID': run}
                          ).encode()
                      )
                  },
              )
          )
        return original(message, payload)

      peer._answer = answer
      manager.connections[device] = peer
      events = []
      result = manager.execute(
          request, lambda kind, **data: events.append(kind), lambda: False
      )
      self.assertEqual(
          (
              result['output'],
              result['output_confirmed'],
              result['dump_complete'],
          ),
          ('hello', True, True),
      )
      self.assertEqual(result['debug_data']['status'], 'available')
      self.assertEqual(events[:3], ['delta', 'delta', 'generation_completed'])
      generate = next(
          message for message in peer.sent if message['type'] == 'generate'
      )
      self.assertEqual(
          generate['job']['expect'],
          dict(runtimeInstance='runtime', turnSequence=0, tokenCount=0),
      )
      self.assertEqual(
          manager.sessions[(session, 'ref', device)]['turn_sequence'], 1
      )
      manager.reset_session(session)
      manager.sessions[(session, 'ref', device)] = {}
      manager.close_session(session)
      self.assertEqual(
          [message['type'] for message in peer.sent][-2:], ['reset', 'close']
      )


if __name__ == '__main__':
  unittest.main()
