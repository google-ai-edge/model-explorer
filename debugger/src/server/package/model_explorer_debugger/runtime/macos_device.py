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

"""Native macOS App transport: authenticated WebSocket, including file chunks.

Loopback is discovered locally. Remote bridge connections are explicitly
registered using a connection file exported by the target App.
"""

import argparse
import base64
import collections
import ipaddress
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import uuid

from model_debugger_contracts import errors
from model_explorer_debugger.runtime import device_access
from model_explorer_debugger.runtime import protocol
from model_explorer_debugger.runtime import runner_artifacts
from model_explorer_debugger.runtime import runner_channel
from model_explorer_debugger.runtime import runner_device
from model_explorer_debugger.runtime import turn_outcome
from websockets import protocol as ws_protocol
from websockets.sync import client as ws_client

CHUNK = protocol.FILE_CHUNK
FILE_WINDOW = protocol.FILE_WINDOW

LOCAL_DEVICE = 'local'
LOCAL_CONFIG = (
    pathlib.Path.home()
    / 'Library/Application Support/Model Debugger Runner/connection.json'
)


def split_slot(identity):
  """Splits a slot-qualified identity (<device>#<slot>) into (device, slot)."""
  base, separator, slot = identity.partition('#')
  if separator and slot not in ('ref', 'target'):
    raise ValueError('Invalid Runner slot')
  return base, slot if separator else 'ref'


def device_id(value):
  """Validates and normalizes a macOS or SSH Runner device identifier."""
  if isinstance(value, str) and re.fullmatch(
      r'ssh:[A-Za-z0-9_.-]{1,100}', value
  ):
    return value
  if not isinstance(value, str) or not value.startswith('macos:'):
    raise ValueError('Choose a registered macOS Runner')
  try:
    return str(uuid.UUID(value[6:]))
  except ValueError:
    raise ValueError('Invalid macOS Runner identifier') from None


def read_config(path):
  """Reads and validates a Mac Runner WebSocket connection config file."""
  if path.stat().st_size > 16384:
    raise ValueError('Connection file is too large')
  config = json.loads(path.read_text())
  identity = str(uuid.UUID(config['id']))
  host = ipaddress.IPv4Address(config['host'])
  if (
      host.is_unspecified
      or host.is_multicast
      or not (host.is_private or host.is_link_local or host.is_loopback)
  ):
    raise ValueError('Use a local or private bridge IPv4 address')
  if config.get('port') != 8769 or not re.fullmatch(
      '[0-9a-f]{64}', config.get('token', '')
  ):
    raise ValueError('Invalid Runner port or access key')
  name = config.get('name', 'Mac Runner')
  if not isinstance(name, str) or not 1 <= len(name) <= 120:
    raise ValueError('Invalid Runner name')
  return dict(
      id=identity, host=str(host), port=8769, token=config['token'], name=name
  )


def configurations(root):
  """Returns registered macOS Runner configurations keyed by UUID."""
  values = {}
  paths = sorted((pathlib.Path(root) / 'macos-devices').glob('*.json'))
  if LOCAL_CONFIG.is_file():
    paths.append(LOCAL_CONFIG)
  for path in paths:
    config = read_config(path)
    # A connection exported by the local App only represents this machine
    # when it is bound to loopback. Bridge exports are registered explicitly.
    if path == LOCAL_CONFIG and config['host'] != '127.0.0.1':
      continue
    values[config['id']] = config
  return values


def list_devices(root):
  """Lists available local and wired-bridge macOS Runner devices."""
  result = []
  for identity, config in configurations(root).items():
    result.append(
        dict(
            id='macos:' + identity,
            name=config['name'] + ' · Native App',
            platform='macOS',
            transport='Local'
            if config['host'] == '127.0.0.1'
            else 'Wired bridge',
        )
    )
  return result


def register(root, source):
  """Registers an exported remote Mac bridge connection file under root."""
  config = read_config(pathlib.Path(source))
  if config['host'] == '127.0.0.1':
    raise ValueError(
        'For another Mac, export its bridge address instead of localhost'
    )
  directory = pathlib.Path(root) / 'macos-devices'
  directory.mkdir(parents=True, exist_ok=True)
  destination = directory / (config['id'] + '.json')
  # Never expose credentials through APIs or command arguments.
  descriptor = os.open(
      destination, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
  )
  with os.fdopen(descriptor, 'w') as output:
    json.dump(config, output)
  destination.chmod(0o600)
  return 'macos:' + config['id']


def validate_request(request):
  """Validates a macOS Runner execution request."""
  device_id(request['run'].get('device'))
  runner_device.validate_native_request(request)


def bridge_endpoint(host):
  """Choose an active bridge on a matching IPv4 subnet, never a Wi-Fi route."""
  candidates = []
  address = ipaddress.IPv4Address(host)
  for index, name in socket.if_nameindex():
    if not re.fullmatch(r'bridge\d+', name):
      continue
    info = subprocess.check_output(
        ['/sbin/ifconfig', name], text=True, timeout=3
    )
    if not re.search(r'status: active\b', info):
      continue
    for local, mask in re.findall(
        r'\binet ([0-9.]+) netmask (0x[0-9a-f]+)', info
    ):
      netmask = str(ipaddress.IPv4Address(int(mask, 16)))
      if address in ipaddress.IPv4Network((local, netmask), strict=False):
        candidates.append((index, name, local))
  if len(candidates) != 1:
    raise ValueError(
        'Select an unambiguous active Thunderbolt bridge subnet for the'
        ' remote Mac'
    )
  return candidates[0]


def connect_bridge(host, port):
  """Connects a TCP socket bound to the active Thunderbolt bridge interface."""
  if sys.platform != 'darwin':
    raise ValueError(
        'Wired Mac Runner connections currently require a macOS server'
    )
  index, interface, local = bridge_endpoint(host)
  peer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  try:
    # Darwin netinet/in.h: IP_BOUND_IF = 25. Binding only the source address
    # is insufficient when Wi-Fi also owns a link-local route.
    peer.setsockopt(socket.IPPROTO_IP, 25, index)
    peer.settimeout(5)
    peer.bind((local, 0))
    peer.connect((host, port))
    # websockets preserves the timeout on a supplied socket. Limit only
    # TCP connection establishment; WebSocket deadlines and heartbeats
    # must own the lifetime of the resident Conversation.
    peer.settimeout(None)
    return peer, {
        'interface': interface,
        'local_ip': peer.getsockname()[0],
        'peer_ip': host,
        'interface_index': peer.getsockopt(socket.IPPROTO_IP, 25),
    }
  except BaseException:
    peer.close()
    raise


class MacConnection(
    runner_artifacts.RunnerArtifacts, runner_channel.RunnerChannel
):
  """Authenticated WebSocket channel to a local or Thunderbolt-bridged Mac."""

  @property
  def is_open(self):
    return (
        self.socket is not None and self.socket.state is ws_protocol.State.OPEN
    )

  def __init__(self, config):
    super().__init__()
    self.socket = None
    raw = None
    self.transport_info = {
        'interface': 'lo0',
        'local_ip': '127.0.0.1',
        'peer_ip': config['host'],
    }
    if config.get('transport') == 'SSH':
      self.transport_info['tunnel'] = 'SSH'
    try:
      if not ipaddress.IPv4Address(config['host']).is_loopback:
        raw, self.transport_info = connect_bridge(
            config['host'], config['port']
        )
      self.socket = ws_client.connect(
          f"ws://{config['host']}:{config['port']}",
          proxy=None,
          compression=None,
          additional_headers={'Authorization': 'Bearer ' + config['token']},
          open_timeout=5,
          close_timeout=1,
          ping_interval=None,
          max_size=1024 * 1024,
          **({'sock': raw} if raw else {}),
      )
      hello = self.receive(5)
      self.runtimes = hello.get('runtimes', ['LiteRT-LM'])
      if (
          hello.get('type') != 'hello'
          or hello.get('protocolVersion')
          not in protocol.SUPPORTED_PROTOCOL_VERSIONS
          or hello.get('platform') != 'macOS'
          or not hello.get('debuggerEnabled')
          or not hello.get('fileTransfer')
      ):
        raise ValueError(
            'Unsupported macOS Runner protocol; install a compatible v4 Runner'
            ' build'
        )
      self.start_channel(hello)
    except BaseException:
      self.close()
      if raw:
        raw.close()
      raise

  @property
  def links_files(self):
    """Whether the Runner shares this Server's filesystem.

    The Runner reads and writes the filesystem this Server sees: loopback,
    not an SSH tunnel.
    """
    return (
        self.protocol >= 5
        and self.hello.get('fileLink') is True
        and 'tunnel' not in self.transport_info
        and self.transport_info.get('peer_ip') == '127.0.0.1'
    )

  def ensure_model(self, model, sha, emit, cancelled=None):
    if self.exchange('model_info', modelSHA256=sha).get('available'):
      emit('progress', message='Using model cached in the native Mac App')
      return
    if getattr(self, 'links_files', False):
      try:
        emit('progress', message='Linking model into the Mac App')
        if self.exchange(
            'model_link',
            modelSHA256=sha,
            path=str(pathlib.Path(model).resolve()),
        ).get('available'):
          return
      except ValueError as error:
        # The Runner answered and keeps no partial file; the bytes can still be
        # sent.
        emit(
            'progress',
            message=f'Model link unavailable ({error}); transferring instead',
        )
    size = pathlib.Path(model).stat().st_size
    self.exchange('upload_begin', modelSHA256=sha, size=size)
    try:
      expected = collections.deque()

      def chunks(source):
        offset = 0
        while data := source.read(CHUNK):
          if cancelled and cancelled():
            raise InterruptedError('Cancelled model transfer')
          expected.append(offset + len(data))
          yield offset, data
          offset += len(data)

      with pathlib.Path(model).open('rb') as source:
        if getattr(self, 'protocol', 4) >= 5:
          replies = runner_channel.pipeline(
              self,
              (
                  ('upload_chunk', {'offset': offset}, data)
                  for offset, data in chunks(source)
              ),
              window=FILE_WINDOW,
          )
        else:
          replies = (
              self.exchange(
                  'upload_chunk',
                  offset=offset,
                  data=base64.b64encode(data).decode('ascii'),
              )
              for offset, data in chunks(source)
          )
        next_progress = 0
        for result in replies:
          offset = expected.popleft()
          if result.get('offset') != offset:
            raise ValueError('Model upload offset mismatch')
          if offset >= next_progress:
            emit(
                'progress',
                message=(
                    f'Transferring model to Mac App · {offset * 100 // size}%'
                ),
            )
            next_progress = offset + 128 * 1024 * 1024
      if cancelled and cancelled():
        raise InterruptedError('Cancelled model transfer')
      if not self.exchange('upload_finish').get('available'):
        raise ValueError('Model transfer was not committed')
    except BaseException:
      # Disconnect also deletes partial uploads. Never finalize after a
      # cancellation, missing acknowledgement, or ambiguous transfer.
      self.close()
      raise

  def close(self):
    self.stop_channel()
    if self.socket:
      self.socket.close()
      self.socket = None


class MacDevices(runner_device.RunnerDevices):
  DEVICE_PREFIXES = ('macos:', 'ssh:')
  SUPPORTED_RUNTIMES = frozenset({'LiteRT-LM', 'PyTorch'})
  SUPPORTS_MULTI_SLOT = True
  platform = 'macOS'
  transport = 'Native Mac WebSocket'
  capture_folder = 'mac-capture'
  device_id = staticmethod(device_id)
  validate = staticmethod(validate_request)

  def execute(self, request, emit, cancelled):
    if request['run']['runtime'] != 'PyTorch':
      return super().execute(request, emit, cancelled)

    identity = self.connection_key(request)
    self.check_owner(identity, request)
    config = configurations(self.root).get(split_slot(identity)[0])
    if not config or config['host'] != '127.0.0.1':
      raise ValueError(
          'PyTorch currently requires the local Mac Runner with its Python'
          ' adapter configured'
      )
    initializing = request['operation'] == 'initialize'
    key = request['session_id'], request['run']['id'], identity
    connection = self.reusable_connection(identity, initializing=initializing)
    if not initializing and key not in self.sessions:
      raise ValueError('Runner Chat is closed')
    confirmed = None
    try:
      if initializing or connection is None:
        connection = self.acquire(identity, request)
      if 'PyTorch' not in connection.runtimes:
        raise ValueError(
            'Configure PyTorch in the local Runner App and restart it'
        )
      payload = dict(request)
      if not initializing:
        payload.pop('messages', None)
      message = dict(
          type=request['operation'],
          runtime='PyTorch',
          requestId=str(uuid.uuid4()),
          request=payload,
      )
      # Text the Runner already confirmed is kept; only an unconfirmed
      # generation is cancelled.
      stop = (
          (lambda: cancelled() and confirmed is None)
          if request['operation'] == 'generate'
          else None
      )
      for reply in runner_channel.request(connection, message, cancelled=stop):
        kind = reply.get('type')
        if kind == 'runtime_event':
          event = dict(reply['event'])
          event_kind = event.pop('type')
          if event_kind == 'generation_completed':
            confirmed = dict(event.get('result') or event)
            confirmed.update(
                turn_outcome.text(
                    'completed',
                    event.get('output', confirmed.get('output', '')),
                )
            )
            emit(
                'generation_completed',
                output=confirmed['output'],
                output_confirmed=True,
                generation_status='completed',
            )
          elif event_kind not in ('error', 'finished'):
            emit(event_kind, **event)
        elif kind == 'preflight':
          if reply.get('accepted') is False:
            raise errors.InputRejected(
                reply.get('error', 'Input exceeds available context')
            )
          return reply.get('result', reply)
        elif kind == 'error':
          if reply.get('errorCode') == errors.InputRejected.code:
            raise errors.InputRejected(
                reply.get('error', 'Input exceeds available context')
            )
          if initializing or request['operation'] == 'preflight':
            raise turn_outcome.RunnerRefused(
                reply.get('error', 'Python Runner failed')
            )
          status = (
              'completed'
              if confirmed
              else (
                  'stopped'
                  if reply.get('generationStatus') == 'stopped'
                  else 'failed'
              )
          )
          result = dict(reply.get('result') or confirmed or {})
          result.setdefault('output', '')
          result.update(
              generation_status=status,
              output_confirmed=bool(confirmed),
              connection_lost=False,
              error=reply.get('error', 'Python generation failed'),
          )
          result.setdefault(
              'debug_data', turn_outcome.debug(result['error'])['debug_data']
          )
          if status != 'completed':
            self.sessions.pop(key, None)
          return result
        elif kind == 'completed':
          result = reply.get('result')
          if not isinstance(result, dict):
            result = json.loads(
                (pathlib.Path(request['output']) / 'result.json').read_text()
            )
          result = dict(
              result,
              runner_transport='local Mac App / Python subprocess',
              output_confirmed=True,
              generation_status='completed',
              connection_lost=False,
          )
          result.setdefault(
              'dump_complete',
              result.get('debug_data', {}).get('status') != 'unavailable',
          )
          result.setdefault('debug_data', turn_outcome.debug()['debug_data'])
          if not initializing and confirmed is None:
            emit(
                'generation_completed',
                output=result.get('output', ''),
                output_confirmed=True,
                generation_status='completed',
            )
          self.sessions[key] = {'runtime': 'PyTorch'}
          return result
    except errors.InputRejected:
      raise
    except BaseException as error:
      if (
          isinstance(error, turn_outcome.RunnerRefused)
          and connection is not None
          and getattr(connection, 'is_open', True)
      ):
        self.sessions.pop(key, None)
        raise
      # Any other failure leaves a request in flight: the link is fenced even
      # when text was confirmed.
      self._forget_connection(identity, request)
      if confirmed is not None:
        return dict(
            confirmed,
            **turn_outcome.dump_failed(error, lost=True),
            dump_complete=False,
        )
      if request['operation'] == 'generate':
        return turn_outcome.unconfirmed(error)
      raise

  def supports_run(self, run):
    if (
        str(run.get('device', '')).startswith('ssh:')
        and run.get('runtime') == 'PyTorch'
    ):
      return False
    return super().supports_run(run)

  def import_result(self, request, result):
    return (
        result
        if request['run']['runtime'] == 'PyTorch'
        else super().import_result(request, result)
    )

  def physical_device_key(self, resolved_device: str) -> str:
    """Returns LOCAL_DEVICE for loopback Macs or macos-host:<ip> for bridges."""
    if str(resolved_device).startswith('macos:'):
      config = configurations(self.root).get(device_id(resolved_device), {})
      if config.get('host') == '127.0.0.1':
        return LOCAL_DEVICE
      # All exports for the same bridge address represent one physical Mac.
      if config.get('host'):
        return 'macos-host:' + config['host']
    return resolved_device

  def connection_key(self, request):
    identity = device_id(request['run']['device'])
    return identity + (
        '#target' if request['run'].get('_runner_slot') == 'target' else ''
    )

  def connect(self, identity):
    identity, slot = split_slot(identity)
    if identity.startswith('ssh:'):
      config = device_access.connection_config(self.root, identity, slot=slot)
    else:
      config = configurations(self.root).get(identity)
    if config is None:
      raise ValueError('Mac Runner is not registered or is unavailable')
    return MacConnection(
        config
        if identity.startswith('ssh:')
        else device_access.slot_config(config, slot)
    )


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Register a remote Mac Runner connection exported by the App'
  )
  parser.add_argument('--workspace', type=pathlib.Path, required=True)
  parser.add_argument('--connection', type=pathlib.Path, required=True)
  args = parser.parse_args()
  print(register(args.workspace, args.connection))
