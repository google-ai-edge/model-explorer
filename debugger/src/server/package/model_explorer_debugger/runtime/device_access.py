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

"""Device access is independent from a Session's Runner control connection.

Only administrator-registered endpoints and build paths are executed. Discovery
never launches a Runner. SSH uses the user's existing host-key/authentication
configuration; credentials and forwarded endpoints are never API fields.
"""

import atexit
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import threading
import time

_tunnels = {}
_lock = threading.RLock()


def slot_config(config, slot='ref'):
  """Derive the Target slot endpoint from a registered Ref connection.

  RunnerPlatform.swift mirrors this.
  """
  if slot not in ('ref', 'target'):
    raise ValueError('Invalid Runner slot')
  if slot == 'ref':
    return dict(config)
  return dict(
      config,
      port=8770,
      token=hashlib.sha256((config['token'] + ':target').encode()).hexdigest(),
  )


def _read(path, key):
  if not path.is_file():
    return []
  if path.stat().st_size > 1024 * 1024:
    raise ValueError('Device registration is too large')
  value = json.loads(path.read_text())
  if (
      not isinstance(value, dict)
      or value.get('version') != 1
      or not isinstance(value.get(key), list)
  ):
    raise ValueError('Unsupported device registration format')
  return value[key]


def configurations(root):
  result = {}
  for entry in _read(Path(root) / 'devices.json', 'devices'):
    if not isinstance(entry, dict):
      raise ValueError('Invalid registered device')
    identity = entry.get('id', '')
    if not isinstance(identity, str) or not re.fullmatch(
        r'(ssh|adb):[A-Za-z0-9_.:@-]{1,160}', identity
    ):
      raise ValueError('Invalid registered device ID')
    if identity in result:
      raise ValueError('Duplicate registered device ID')
    if (
        not isinstance(entry.get('name'), str)
        or not 0 < len(entry['name']) <= 160
    ):
      raise ValueError('Invalid device name')
    if entry.get('platform') not in ('macOS', 'Android', 'Linux'):
      raise ValueError('Invalid registered platform')
    if identity.startswith('ssh:'):
      host = entry.get('host', '')
      if not isinstance(host, str) or not re.fullmatch(
          r'[A-Za-z0-9_][A-Za-z0-9_.:@-]{0,250}', host
      ):
        raise ValueError('Use a registered SSH host or SSH config alias')
    result[identity] = entry
  return result


def registered_devices(root):
  return [
      dict(
          id=identity,
          name=value['name'],
          platform=value['platform'],
          transport='SSH' if identity.startswith('ssh:') else 'ADB',
          connectionState='unknown',
      )
      for identity, value in configurations(root).items()
  ]


def adb_devices():
  if not shutil.which('adb'):
    return []
  result = subprocess.run(
      ['adb', 'devices', '-l'], capture_output=True, text=True, timeout=8
  )
  if result.returncode:
    return []
  devices = []
  for line in result.stdout.splitlines()[1:]:
    fields = line.split()
    if len(fields) < 2 or not re.fullmatch(
        r'[A-Za-z0-9_.:@-]{1,160}', fields[0]
    ):
      continue
    details = dict(part.split(':', 1) for part in fields[2:] if ':' in part)
    devices.append(
        dict(
            id='adb:' + fields[0],
            name=details.get('model', fields[0]).replace('_', ' '),
            platform='Android',
            transport='ADB',
            connectionState='connected' if fields[1] == 'device' else fields[1],
        )
    )
  return devices


def device(root, identity):
  value = configurations(root).get(identity)
  if value:
    return value
  if isinstance(identity, str) and identity.startswith('adb:'):
    found = next(
        (item for item in adb_devices() if item['id'] == identity), None
    )
    if found:
      return found
  raise ValueError('Register or connect this device first')


def command(root, identity, arguments, *, timeout=8):
  """Run a fixed argv operation over a registered device access channel."""
  target = device(root, identity)
  if not arguments or any(
      not isinstance(a, str) or '\x00' in a for a in arguments
  ):
    raise ValueError('Device commands require a nonempty string argument list')
  if identity.startswith('ssh:'):
    argv = [
        'ssh',
        '-oBatchMode=yes',
        '-oConnectTimeout=5',
        target['host'],
        shlex.join(arguments),
    ]
  elif identity.startswith('adb:'):
    argv = ['adb', '-s', identity[4:], 'shell', shlex.join(arguments)]
  else:
    raise ValueError('Unsupported device access transport')
  return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def parse_process_snapshot(output):
  """A nonempty complete `ps pid,comm` snapshot is required to prove absence."""
  if (
      not isinstance(output, str)
      or not output.strip()
      or len(output) > 4 * 1024 * 1024
  ):
    raise ValueError('Invalid process snapshot')
  processes = []
  for line in output.splitlines():
    match = re.fullmatch(r'\s*([1-9][0-9]*)\s+(.+)', line)
    if not match or not match[2].strip():
      raise ValueError('Unrecognized process snapshot schema')
    processes.append(match[2].strip())
  return processes


def runner_observation(root, identity, executables=None):
  """Query the access channel; unsupported or failed probes remain unknown."""
  target = device(root, identity)
  unknown = {'connectionState': 'unknown', 'runnerState': 'unknown'}
  try:
    if identity.startswith('ssh:'):
      probe = command(root, identity, ['true'])
      if probe.returncode:
        return {'connectionState': 'unreachable', 'runnerState': 'unknown'}
      if target.get('platform') != 'macOS':
        return {'connectionState': 'connected', 'runnerState': 'unknown'}
      expected = list(executables or [])
      if target.get('runnerExecutable'):
        expected.append(target['runnerExecutable'])
      if not expected:
        return {'connectionState': 'connected', 'runnerState': 'unknown'}
      for executable in expected:
        _absolute(executable)
        if command(root, identity, ['test', '-x', executable]).returncode:
          return {'connectionState': 'connected', 'runnerState': 'unknown'}
      check = command(
          root, identity, ['/bin/ps', '-ax', '-o', 'pid=', '-o', 'comm=']
      )
      if check.returncode:
        return unknown
      processes = parse_process_snapshot(check.stdout)
      running = any(
          value in expected
          or Path(value).name in ('ModelDebuggerMac', 'ModelDebuggerRunner')
          for value in processes
      )
      return {
          'connectionState': 'connected',
          'runnerState': 'active' if running else 'not_running',
      }
    # Android access is supported, but there is no native execution adapter.
    # `pidof` exit 1 also covers missing packages and cannot certify a Runner.
    probe = command(root, identity, ['true'])
    return {
        'connectionState': (
            'connected' if probe.returncode == 0 else 'unreachable'
        ),
        'runnerState': 'unknown',
    }
  except (OSError, ValueError, KeyError, subprocess.SubprocessError):
    return unknown


def _absolute(value):
  if (
      not isinstance(value, str)
      or not value.startswith('/')
      or any(ord(c) < 32 for c in value)
  ):
    raise ValueError('Use an absolute installed path')
  return value


def connection_config(root, identity, slot='ref'):
  """Forward a registered macOS Runner's authenticated endpoint over SSH."""
  target = device(root, identity)
  if not identity.startswith('ssh:') or target['platform'] != 'macOS':
    raise ValueError('This device has no implemented Runner execution adapter')
  path = _absolute(target.get('runnerConnectionFile'))
  result = command(root, identity, ['cat', path])
  if result.returncode or len(result.stdout) > 16384:
    raise ValueError('Could not read the registered Runner connection file')
  remote = json.loads(result.stdout)
  if not isinstance(remote, dict):
    raise ValueError('Invalid Runner connection configuration')
  if (
      remote.get('host') != '127.0.0.1'
      or remote.get('port') != 8769
      or not isinstance(remote.get('token'), str)
      or not re.fullmatch('[0-9a-f]{64}', remote['token'])
  ):
    raise ValueError('SSH Runner must export its loopback connection')
  remote = slot_config(remote, slot)
  key = str(Path(root).resolve()), identity, slot
  with _lock:
    current = _tunnels.get(key)
    if current is None or current[0].poll() is not None:
      with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
      process = subprocess.Popen(
          [
              'ssh',
              '-N',
              '-oBatchMode=yes',
              '-oConnectTimeout=5',
              '-oExitOnForwardFailure=yes',
              '-oServerAliveInterval=5',
              '-oServerAliveCountMax=3',
              '-L',
              f"127.0.0.1:{port}:127.0.0.1:{remote['port']}",
              target['host'],
          ],
          stdin=subprocess.DEVNULL,
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
      )
      deadline = time.monotonic() + 6
      while time.monotonic() < deadline:
        if process.poll() is not None:
          raise ValueError('SSH forwarding failed')
        try:
          with socket.create_connection(('127.0.0.1', port), timeout=0.2):
            break
        except OSError:
          time.sleep(0.05)
      else:
        process.terminate()
        process.wait(timeout=2)
        raise TimeoutError('SSH forwarding did not become ready')
      _tunnels[key] = process, port
    else:
      process, port = current
  return dict(
      id=identity,
      host='127.0.0.1',
      port=port,
      token=remote['token'],
      name=target['name'],
      transport='SSH',
      platform='macOS',
  )


def close_tunnels():
  with _lock:
    for process, _ in _tunnels.values():
      if process.poll() is None:
        process.terminate()
        try:
          process.wait(timeout=2)
        except subprocess.TimeoutExpired:
          process.kill()
          process.wait(timeout=2)
    _tunnels.clear()


atexit.register(close_tunnels)
