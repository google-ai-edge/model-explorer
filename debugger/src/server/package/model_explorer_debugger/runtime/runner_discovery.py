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

"""Candidate discovery and authenticated, non-owning Runner inspection.

Inspection never uses a control connection, launches an App, rotates credentials
or sends a native command. Unsupported inspection never falls back to control.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
from uuid import UUID

from .protocol import (
    DESCRIPTOR_VERSION,
    INSPECTION_PROTOCOL,
    NATIVE_BACKENDS,
    NATIVE_CONTEXT_LENGTHS,
    NATIVE_MAX_CAPTURE_POINTS,
    NATIVE_MAX_OUTPUT_TOKENS,
    SUPPORTED_PROTOCOL_VERSIONS,
)


def candidates(root):
  from .macos_device import list_devices as list_macs
  from .ios_device import list_devices as list_phones

  devices = list_macs(root)
  local = next((d for d in devices if d.get('transport') == 'Local'), None)
  devices = [d for d in devices if d.get('transport') != 'Local']
  devices.insert(
      0,
      dict(
          local or {},
          id='local',
          runnerId=local.get('id') if local else None,
          name='This computer',
          platform='macOS',
          transport='Local',
      ),
  )
  error = None
  try:
    devices += list_phones()
  except (OSError, RuntimeError, TimeoutError):
    error = (
        'USB discovery is unavailable. Registered Mac devices are still listed.'
    )
  from .device_access import registered_devices, adb_devices

  try:
    extras = {d['id']: d for d in adb_devices()}
    extras.update({d['id']: d for d in registered_devices(root)})
    devices += list(extras.values())
  except (OSError, ValueError, subprocess.SubprocessError):
    error = 'Some device access registrations could not be checked.'
  return {
      'devices': [
          dict(
              device,
              status='unknown',
              runnerState='unknown',
              reason='Runner has not been checked.',
          )
          for device in devices
      ],
      **({'notice': error} if error else {}),
  }


def descriptor(value):
  """Validate and project only the public versioned descriptor fields."""
  if (
      not isinstance(value, dict)
      or type(value.get('version')) is not int
      or value['version'] != DESCRIPTOR_VERSION
  ):
    raise ValueError('Unsupported Runner descriptor')
  if not isinstance(value.get('runnerInstance'), str):
    raise ValueError('Invalid Runner identity')
  UUID(value['runnerInstance'])
  caps, environment, state = (
      value['capabilities'],
      value['environment'],
      value['state'],
  )
  if not isinstance(caps, dict):
    raise ValueError('Invalid Runner capabilities')
  if (
      caps.get('runtime') != 'LiteRT-LM'
      or caps.get('backends')
      not in ([NATIVE_BACKENDS[0]], list(NATIVE_BACKENDS))
      or caps.get('modalities') != ['text']
      or type(caps.get('tensorCapture')) is not bool
      or caps.get('persistentConversation') is not True
      or caps.get('contextLengths')
      not in ([NATIVE_CONTEXT_LENGTHS[0]], list(NATIVE_CONTEXT_LENGTHS))
      or caps.get('maxOutputTokens') != NATIVE_MAX_OUTPUT_TOKENS
      or caps.get('maxCapturePoints') != NATIVE_MAX_CAPTURE_POINTS
  ):
    raise ValueError('Unsupported Runner capabilities')
  if any(
      type(value.get(key)) is not bool for key in ('controlConnected', 'busy')
  ):
    raise ValueError('Invalid Runner state')
  if (
      type(value.get('residentSessions')) is not int
      or not 0 <= value['residentSessions'] <= 1024
  ):
    raise ValueError('Invalid resident Session count')

  def fields(source, strings, numbers, booleans=()):
    if not isinstance(source, dict):
      raise ValueError('Invalid Runner environment')
    result = {}
    for key in strings:
      if key in source:
        if not isinstance(source[key], str) or not 0 < len(source[key]) <= 256:
          raise ValueError('Invalid environment text')
        result[key] = source[key]
    for key in numbers:
      if key in source:
        if type(source[key]) is not int or not 0 <= source[key] <= 2**53 - 1:
          raise ValueError('Invalid environment measurement')
        result[key] = source[key]
    for key in booleans:
      if key in source:
        if type(source[key]) is not bool:
          raise ValueError('Invalid environment state')
        result[key] = source[key]
    return result

  environment = fields(
      environment,
      (
          'capturedAt',
          'platform',
          'operatingSystem',
          'architecture',
          'modelIdentifier',
          'chip',
          'appVersion',
          'appBuild',
          'runtimeCommit',
          'runtimeSourceDiffSHA256',
      ),
      ('physicalMemoryBytes', 'logicalProcessorCount'),
  )
  state = fields(
      state,
      ('capturedAt', 'thermalState'),
      ('processResidentBytes', 'processMemoryHeadroomBytes'),
      ('lowPowerMode',),
  )
  if (
      not {'capturedAt', 'platform', 'operatingSystem', 'architecture'}
      <= environment.keys()
      or 'capturedAt' not in state
  ):
    raise ValueError('Missing Runner environment')
  runtimes = value.get(
      'runtimes',
      [{'id': 'LiteRT-LM', 'backends': ['CPU'], 'transport': 'native'}],
  )
  if not isinstance(runtimes, list) or len(runtimes) > 2:
    raise ValueError('Invalid Runner runtime list')
  projected = []
  for runtime in runtimes:
    if not isinstance(runtime, dict) or runtime.get('id') not in (
        'LiteRT-LM',
        'PyTorch',
    ):
      raise ValueError('Invalid runtime capability')
    backends = runtime.get('backends')
    allowed = (
        ('CPU', 'GPU')
        if runtime.get('id') == 'LiteRT-LM'
        else ('CPU', 'MPS', 'CUDA')
    )
    if (
        not isinstance(backends, list)
        or not backends
        or len(set(backends)) != len(backends)
        or any(b not in allowed for b in backends)
    ):
      raise ValueError('Invalid Runner backends')
    if runtime.get('transport') not in ('native', 'localFiles'):
      raise ValueError('Invalid runtime transport')
    projected.append(
        {key: runtime[key] for key in ('id', 'backends', 'transport')}
    )
  extra = {}
  if 'protocolVersion' in value:
    if (
        type(value['protocolVersion']) is not int
        or value['protocolVersion'] < 1
    ):
      raise ValueError('Invalid Runner protocol version')
    extra['protocolVersion'] = value['protocolVersion']
  if 'lifecycle' in value:
    if value['lifecycle'] not in (
        'waiting',
        'starting',
        'active',
        'ending',
        'ended',
        'interrupted',
    ):
      raise ValueError('Invalid Runner lifecycle')
    extra['lifecycle'] = value['lifecycle']
  if value.get('owner') is not None:
    owner = value['owner']
    if not isinstance(owner, dict):
      raise ValueError('Invalid Runner owner')
    for key in ('serverId', 'sessionId'):
      UUID(owner[key])
    if owner.get('runId') not in ('ref', 'target'):
      raise ValueError('Invalid Runner role')
    if any(
        not isinstance(owner.get(k), str) or not 0 < len(owner[k]) <= 256
        for k in ('serverName', 'sessionName')
    ):
      raise ValueError('Invalid Runner owner names')
    extra['owner'] = {
        key: owner[key]
        for key in (
            'serverId',
            'serverName',
            'sessionId',
            'sessionName',
            'runId',
        )
    }
    for key in ('serverId', 'sessionId'):
      extra['owner'][key] = str(UUID(extra['owner'][key]))
  for key in ('buildId',):
    if value.get(key) is not None:
      if not isinstance(value[key], str) or not 0 < len(value[key]) <= 256:
        raise ValueError('Invalid Runner build identity')
      extra[key] = value[key]
  if value.get('build') is not None:
    extra['build'] = fields(
        value['build'], ('id', 'label', 'version', 'cl'), ()
    )
  return {
      key: value[key]
      for key in (
          'version',
          'runnerInstance',
          'controlConnected',
          'busy',
          'residentSessions',
      )
  } | {
      'capabilities': {
          key: caps[key]
          for key in (
              'runtime',
              'backends',
              'modalities',
              'tensorCapture',
              'persistentConversation',
              'contextLengths',
              'maxOutputTokens',
              'maxCapturePoints',
          )
      },
      'environment': environment,
      'state': state,
      'runtimes': projected,
      **extra,
  }


def read_observation(host, port, token, raw=None, deadline=4):
  from websockets.sync.client import connect

  with connect(
      f'ws://{host}:{port}',
      proxy=None,
      compression=None,
      subprotocols=[INSPECTION_PROTOCOL],
      additional_headers={'Authorization': 'Bearer ' + token},
      open_timeout=deadline,
      close_timeout=0.2,
      ping_interval=None,
      max_size=65536,
      **({'sock': raw} if raw else {}),
  ) as peer:
    if peer.subprotocol != INSPECTION_PROTOCOL:
      raise ValueError('Read-only inspection is not supported')
    return descriptor(json.loads(peer.recv(timeout=deadline)))


@contextmanager
def usb_inspection_tunnel(udid):
  program = shutil.which('iproxy')
  if not program:
    raise FileNotFoundError('USB proxy is unavailable')
  with socket.socket() as reservation:
    reservation.bind(('127.0.0.1', 0))
    port = reservation.getsockname()[1]
  process = subprocess.Popen(
      [
          program,
          '--local',
          '--udid',
          udid,
          '--source',
          '127.0.0.1',
          f'{port}:8769',
      ],
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
  )
  try:
    yield port, process
  finally:
    process.terminate()
    try:
      process.wait(timeout=2)
    except subprocess.TimeoutExpired:
      process.kill()
      process.wait(timeout=2)


def inspect(root, identity, slot=None):
  from .device_access import slot_config
  from .macos_device import device_id as mac_id, configurations, connect_bridge
  from .ios_device import device_id as phone_id
  from websockets.exceptions import InvalidHandshake, ConnectionClosed

  checked = datetime.now(timezone.utc).isoformat()
  original_identity = identity
  result = {'id': identity, 'checkedAt': checked, 'runnerState': 'unknown'}
  raw = None

  def unavailable(reason, status='unreachable'):
    from .runner_builds import process_state

    try:
      presence = process_state(root, original_identity or 'local', slot=slot)
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        subprocess.SubprocessError,
    ):
      presence = {'connectionState': 'unknown', 'runnerState': 'unknown'}
    if presence.get('runnerState') == 'not_running':
      return (
          result
          | presence
          | {
              'status': 'ready',
              'reason': (
                  'Runner is not running. Select an installed build for this'
                  ' Session.'
              ),
          }
      )
    return result | presence | {'status': status, 'reason': reason}

  try:
    if identity in ('local', 'Server host', None, ''):
      from .device_ids import resolve_device

      try:
        identity = resolve_device(root, identity)
      except ValueError:
        return unavailable('No local Runner endpoint is available.')
    if isinstance(identity, str) and identity.startswith('macos:'):
      config = configurations(root).get(mac_id(identity))
      if config is None:
        return unavailable('Register this Mac Runner first.', 'unpaired')
      config = slot_config(config, slot or 'ref')
      if config['host'] != '127.0.0.1':
        raw, _ = connect_bridge(config['host'], config['port'])
      observation = read_observation(
          config['host'], config['port'], config['token'], raw
      )
    elif isinstance(identity, str) and identity.startswith('ssh:'):
      from .device_access import connection_config

      config = connection_config(root, identity, slot=slot or 'ref')
      observation = read_observation(
          config['host'], config['port'], config['token']
      )
    elif isinstance(identity, str) and identity.startswith('adb:'):
      return unavailable(
          'Android Runner execution requires an installed native adapter.',
          'unsupported',
      )
    else:
      udid = phone_id(identity)
      config_path = Path(root) / 'ios-devices' / udid / 'inspection.json'
      if not config_path.is_file():
        return unavailable(
            'USB device found. Register and select its installed Runner build.',
            'unpaired',
        )
      config = json.loads(config_path.read_text())
      if not re.fullmatch('[0-9a-f]{64}', config.get('token', '')):
        raise ValueError('Invalid saved pairing')
      with usb_inspection_tunnel(udid) as (port, process):
        deadline = time.monotonic() + 4
        while True:
          try:
            observation = read_observation(
                '127.0.0.1', port, config['token'], deadline=2
            )
            break
          except (OSError, TimeoutError):
            if process.poll() is not None or time.monotonic() >= deadline:
              raise
            time.sleep(0.1)
    status = (
        'busy'
        if observation['busy']
        else 'connected'
        if observation['controlConnected']
        else 'ready'
    )
    owner = observation.get('owner')
    if owner:
      try:
        current = json.loads(
            (Path(root) / 'server-identity.json').read_text()
        ).get('id')
      except (OSError, ValueError):
        current = None
      owner['isCurrentServer'] = owner['serverId'] == current
    if (
        not observation['capabilities']['tensorCapture']
        or observation.get('protocolVersion') not in SUPPORTED_PROTOCOL_VERSIONS
    ):
      status = 'unsupported'
    reason = (
        f"Occupied by {owner['sessionName']} ·"
        f" {'This server' if owner['isCurrentServer'] else owner['serverName']}"
        if owner
        else None
    )
    return result | {
        'status': status,
        'runner': observation,
        'connectionState': 'connected',
        'runnerState': (
            'active'
            if owner or observation['controlConnected'] or observation['busy']
            else 'idle'
        ),
        'reason': (
            reason
            or {
                'busy': 'Runner is executing a task.',
                'connected': 'Runner is connected to a debugger.',
                'ready': 'Runner is ready for a Session.',
                'unsupported': (
                    'This Runner build does not support the required protocol'
                    ' or capture capability.'
                ),
            }[status]
        ),
    }
  except (InvalidHandshake, ValueError, KeyError, TypeError):
    return result | {
        'status': 'unsupported',
        'reason': (
            'Read-only inspection or pairing is incompatible. Update or pair'
            ' the Runner.'
        ),
    }
  except (ConnectionClosed, OSError, TimeoutError, RuntimeError):
    return unavailable(
        'Runner could not be reached. Check the device connection.'
    )
  finally:
    if raw is not None:
      raw.close()
