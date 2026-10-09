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

"""Installed Runner builds, checked without starting or taking control.

workspace/runner-builds.json is an administrator-owned version-1 catalog. API
requests carry only its build ID, never a path or shell command. A build starts
only as part of an explicit Session initialization.
"""

import json
import os
from pathlib import Path
import plistlib
import re
import subprocess
import sys

from . import device_access
from .protocol import SUPPORTED_PROTOCOL_VERSIONS


def _builds(root):
  values = device_access._read(Path(root) / 'runner-builds.json', 'builds')
  seen = set()
  for value in values:
    if not isinstance(value, dict):
      raise ValueError('Invalid Runner build registration')
    identity = value.get('id', '')
    if not isinstance(identity, str) or not re.fullmatch(
        r'[A-Za-z0-9_.:@+-]{1,160}', identity
    ):
      raise ValueError('Invalid registered Runner build ID')
    devices = value.get('devices')
    if (
        not isinstance(devices, list)
        or not devices
        or any(not isinstance(d, str) for d in devices)
    ):
      raise ValueError('Register the devices that can launch this Runner build')
    keys = {
        (identity, 'local' if device in ('local', 'Server host') else device)
        for device in devices
    }
    if keys.intersection(seen):
      raise ValueError('Duplicate Runner build registration for a device')
    seen.update(keys)
    if value.get('kind') not in (
        'macos-app',
        'ssh-macos-app',
        'ios-app',
        'adb-app',
    ):
      raise ValueError('Unsupported Runner launch kind')
  return values


def _local_aliases(root, identity):
  aliases = {identity}
  if identity in ('local', 'Server host', None, ''):
    aliases.update(('local', 'Server host'))
  elif isinstance(identity, str) and identity.startswith('macos:'):
    from .macos_device import configurations

    if configurations(root).get(identity[6:], {}).get('host') == '127.0.0.1':
      aliases.add('local')
  return aliases


def _registered(root, identity):
  aliases = _local_aliases(root, identity)
  values = [
      value for value in _builds(root) if aliases.intersection(value['devices'])
  ]
  if len({value['id'] for value in values}) != len(values):
    raise ValueError(
        'Duplicate Runner build registration through device aliases'
    )
  return values


def _app(path):
  path = Path(device_access._absolute(path))
  with (path / 'Contents/Info.plist').open('rb') as source:
    info = plistlib.load(source)
  if not isinstance(info, dict):
    raise ValueError('Invalid Runner application metadata')
  executable = info.get('CFBundleExecutable', '')
  if (
      not isinstance(executable, str)
      or not executable
      or executable in ('.', '..')
      or Path(executable).name != executable
  ):
    raise ValueError('Invalid Runner application executable')
  binary = path / 'Contents/MacOS' / executable
  if not binary.is_file() or not os.access(binary, os.X_OK):
    raise ValueError('Runner application executable is missing')
  return binary, info


def _build_identity(info, provenance=None):
  """Match RunnerModel.build exactly; a CL is never guessed from app version."""
  explicit = info.get('ModelDebuggerBuildID')
  if isinstance(explicit, str) and explicit:
    return explicit
  version = info.get('CFBundleVersion')
  version = version if isinstance(version, str) and version else None
  provenance = provenance or {}
  commit = provenance.get('litertLMCommit')
  if isinstance(commit, str) and commit:
    diff = provenance.get('nativeSourceDiffSHA256')
    return (
        (version or 'unknown')
        + '@'
        + commit
        + ('+' + diff if isinstance(diff, str) and diff else '')
    )
  return version


def _local_identity(path, info):
  provenance = Path(path) / 'Contents/Resources/runtime-build.json'
  payload = json.loads(provenance.read_text()) if provenance.is_file() else {}
  if not isinstance(payload, dict):
    raise ValueError('Invalid Runner build provenance')
  return _build_identity(info, payload)


def _ssh_app(root, identity, value):
  app = device_access._absolute(value.get('path'))
  info_result = device_access.command(
      root,
      identity,
      ['plutil', '-convert', 'json', '-o', '-', app + '/Contents/Info.plist'],
  )
  if info_result.returncode or len(info_result.stdout) > 1024 * 1024:
    raise ValueError('Could not verify installed application metadata')
  info = json.loads(info_result.stdout)
  if not isinstance(info, dict):
    raise ValueError('Invalid Runner application metadata')
  executable = info.get('CFBundleExecutable')
  if (
      not isinstance(executable, str)
      or not executable
      or executable in ('.', '..')
      or Path(executable).name != executable
  ):
    raise ValueError('Invalid Runner application executable')
  binary = app + '/Contents/MacOS/' + executable
  if value.get('executable') not in (None, binary):
    raise ValueError('Registered executable does not match the installed app')
  if device_access.command(root, identity, ['test', '-x', binary]).returncode:
    raise ValueError('Runner executable is not installed or executable')
  resource = app + '/Contents/Resources/runtime-build.json'
  exists = device_access.command(root, identity, ['test', '-f', resource])
  if exists.returncode not in (0, 1):
    raise ValueError('Could not inspect installed build provenance')
  provenance = {}
  if exists.returncode == 0:
    result = device_access.command(root, identity, ['cat', resource])
    if result.returncode or len(result.stdout) > 1024 * 1024:
      raise ValueError('Could not read installed build provenance')
    provenance = json.loads(result.stdout)
    if not isinstance(provenance, dict):
      raise ValueError('Invalid Runner build provenance')
  return binary, _build_identity(info, provenance)


def _available(root, identity, value):
  try:
    if value.get('protocolVersion') not in SUPPORTED_PROTOCOL_VERSIONS:
      return (
          False,
          'This build must support Runner protocol '
          + ' or '.join(map(str, SUPPORTED_PROTOCOL_VERSIONS))
          + '.',
      )
    kind = value['kind']
    if kind == 'macos-app':
      if sys.platform != 'darwin':
        return False, 'Launching a macOS Runner requires a Mac server.'
      if identity not in (
          'local',
          'Server host',
          None,
          '',
      ) and 'local' not in _local_aliases(root, identity):
        return False, 'A local app can launch only on this server computer.'
      _, info = _app(value.get('path'))
      actual = _local_identity(value['path'], info)
      if actual != value['id']:
        return False, 'Installed build identity differs from the registration.'
    elif kind == 'ssh-macos-app':
      if device_access.device(root, identity)['platform'] != 'macOS':
        return False, 'This build requires a registered macOS SSH device.'
      _, actual = _ssh_app(root, identity, value)
      if actual != value['id']:
        return False, 'Installed build identity differs from the registration.'
    elif kind == 'ios-app':
      if not identity.startswith('ios:'):
        return False, 'This build requires an iOS device.'
      package = value.get('bundleId', '')
      if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.]+', package):
        return False, 'Invalid installed Runner bundle ID.'
      # Existing Apple USB adapter uses this app container for files.
      if package != 'dev.modeldebugger.runner':
        return (
            False,
            (
                'This USB adapter supports the installed Model Debugger Runner'
                ' bundle.'
            ),
        )
      from .ios_device import devicectl

      apps = devicectl('device', 'info', 'apps', '--device', identity[4:]).get(
          'apps'
      )
      if not isinstance(apps, list) or any(
          not isinstance(item, dict) for item in apps
      ):
        return False, 'Could not verify installed iOS applications.'
      installed = next(
          (item for item in apps if item.get('bundleIdentifier') == package),
          None,
      )
      if installed is None:
        return False, 'Runner build is not installed on this device.'
      if installed.get('ModelDebuggerBuildID') != value['id']:
        return (
            False,
            (
                'This device does not expose the installed CL build identity.'
                ' Open the Runner on the device to inspect it.'
            ),
        )
    else:
      return (
          False,
          (
              'Android device discovery is supported; an Android Runner adapter'
              ' is not installed.'
          ),
      )
    return True, ''
  except (
      OSError,
      ValueError,
      KeyError,
      RuntimeError,
      subprocess.SubprocessError,
  ):
    return False, 'Could not verify the installed Runner build.'


def catalog(root, identity):
  builds = []
  for value in _registered(root, identity):
    available, reason = _available(root, identity, value)
    builds.append(
        {
            key: value[key]
            for key in ('id', 'version', 'cl', 'platform')
            if key in value
        }
        | {
            'label': str(value.get('label') or value['id']),
            'available': available,
            **({'reason': reason} if reason else {}),
        }
    )
  return {
      'deviceId': identity,
      'builds': builds,
      'canLaunch': any(b['available'] for b in builds),
      **(
          {
              'reason': (
                  'No installed Runner builds are registered for this device.'
              )
          }
          if not builds
          else {}
      ),
  }


def process_state(root, identity, slot=None):
  """Positive OS process evidence; a failed Runner socket is not sufficient."""
  identity = identity or 'local'
  values = _registered(root, identity)
  if slot in ('ref', 'target') and not identity.startswith(('ios:', 'adb:')):
    # Inspect actual process arguments across all registered build paths.
    # A Ref process does not occupy the Target endpoint, or vice versa.
    try:
      if identity.startswith('ssh:'):
        snapshot = device_access.command(
            root, identity, ['/bin/ps', '-ax', '-o', 'command=']
        )
      elif identity.startswith('macos:') and 'local' not in _local_aliases(
          root, identity
      ):
        return {'connectionState': 'unknown', 'runnerState': 'unknown'}
      else:
        snapshot = subprocess.run(
            ['/bin/ps', '-ax', '-o', 'command='],
            capture_output=True,
            text=True,
            timeout=5,
        )
      if snapshot.returncode:
        raise ValueError('Process query failed')
      paths = [str(v.get('path', '')) + '/Contents/MacOS/' for v in values]
      lines = [
          line
          for line in snapshot.stdout.splitlines()
          if any(path in line for path in paths)
          or '/Contents/MacOS/ModelDebugger' in line
      ]
      active = any(
          ('--runner-slot target' in line) == (slot == 'target')
          for line in lines
      )
      return {
          'connectionState': 'connected',
          'runnerState': 'active' if active else 'not_running',
      }
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
      return {'connectionState': 'unknown', 'runnerState': 'unknown'}
  if identity.startswith('macos:') and 'local' not in _local_aliases(
      root, identity
  ):
    return {'connectionState': 'unknown', 'runnerState': 'unknown'}
  if identity.startswith(('ssh:', 'adb:')):
    executables = []
    try:
      for value in values:
        if value['kind'] == 'ssh-macos-app':
          binary, actual = _ssh_app(root, identity, value)
          if actual != value['id']:
            return {'connectionState': 'connected', 'runnerState': 'unknown'}
          executables.append(binary)
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        subprocess.SubprocessError,
    ):
      return {'connectionState': 'unknown', 'runnerState': 'unknown'}
    return device_access.runner_observation(root, identity, executables)
  if identity.startswith('ios:'):
    if not values:
      return {'connectionState': 'unknown', 'runnerState': 'unknown'}
    from .ios_device import devicectl

    try:
      processes = devicectl(
          'device', 'info', 'processes', '--device', identity[4:]
      ).get('runningProcesses')
      bundles = {value.get('bundleId') for value in values}
      # An explicit bundle identifier is required: never infer absence
      # from a process list whose schema omits app identity.
      if (
          not isinstance(processes, list)
          or not processes
          or any(
              not isinstance(p, dict)
              or not isinstance(p.get('bundleIdentifier'), str)
              for p in processes
          )
      ):
        return {'connectionState': 'connected', 'runnerState': 'unknown'}
      running = any(p.get('bundleIdentifier') in bundles for p in processes)
      if running:
        return {'connectionState': 'connected', 'runnerState': 'active'}
      apps = devicectl('device', 'info', 'apps', '--device', identity[4:]).get(
          'apps'
      )
      if (
          not bundles
          or None in bundles
          or not isinstance(apps, list)
          or any(not isinstance(app, dict) for app in apps)
          or not bundles.issubset({app.get('bundleIdentifier') for app in apps})
      ):
        return {'connectionState': 'connected', 'runnerState': 'unknown'}
      return {'connectionState': 'connected', 'runnerState': 'not_running'}
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
      return {'connectionState': 'unknown', 'runnerState': 'unknown'}
  binaries = []
  for value in values:
    if value['kind'] != 'macos-app':
      continue
    try:
      binaries.append(str(_app(value.get('path'))[0]))
    except (OSError, ValueError):
      return {'connectionState': 'connected', 'runnerState': 'unknown'}
  if not binaries:
    return {'connectionState': 'connected', 'runnerState': 'unknown'}
  try:
    result = subprocess.run(
        ['/bin/ps', '-ax', '-o', 'pid=', '-o', 'comm='],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode:
      raise ValueError('Process query failed')
    processes = device_access.parse_process_snapshot(result.stdout)
  except (OSError, ValueError, subprocess.SubprocessError):
    return {'connectionState': 'unknown', 'runnerState': 'unknown'}
  active = any(
      value in binaries
      or Path(value).name in ('ModelDebuggerMac', 'ModelDebuggerRunner')
      for value in processes
  )
  return {
      'connectionState': 'connected',
      'runnerState': 'active' if active else 'not_running',
  }
