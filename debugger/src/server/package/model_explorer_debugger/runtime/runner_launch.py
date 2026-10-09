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

"""Start an installed Runner build for a Session.

Uses catalog + discovery, never the reverse.
"""

import subprocess
import time

from model_explorer_debugger.runtime import device_access
from model_explorer_debugger.runtime import runner_builds, runner_discovery
from model_explorer_debugger.runtime.protocol import SUPPORTED_PROTOCOL_VERSIONS


def ensure_started(root, run, owner):
  """Start an installed build for a Session.

  Native activate enforces ownership.
  """
  identity = run.get('device') or 'local'
  selected = run.get('runnerBuild')
  slot = (
      run.get('_runner_slot') if not str(identity).startswith('ios:') else None
  )
  state = runner_discovery.inspect(root, identity, slot=slot)
  existing = state.get('runner') or {}
  if existing:
    active_owner = existing.get('owner')
    if active_owner and any(
        active_owner.get(k) != owner.get(k)
        for k in ('serverId', 'sessionId', 'runId')
    ):
      raise ValueError('Device is occupied by another Session or Runner side')
    reusable_ios_shell = (
        identity.startswith('ios:')
        and existing.get('lifecycle') == 'ended'
        and existing.get('environment', {}).get('platform')
        in ('iOS', 'iOS Simulator')
        and not active_owner
        and existing.get('controlConnected') is False
        and existing.get('busy') is False
        and existing.get('residentSessions') == 0
    )
    if (
        (
            not active_owner
            and (existing.get('controlConnected') or existing.get('busy'))
        )
        or existing.get('lifecycle') in ('ending', 'interrupted')
        or (existing.get('lifecycle') == 'ended' and not reusable_ios_shell)
    ):
      raise ValueError('The running Runner is not available for this Session')
    if (
        existing.get('protocolVersion') not in SUPPORTED_PROTOCOL_VERSIONS
        or state.get('status') == 'unsupported'
    ):
      raise ValueError('The running Runner is not compatible with this server')
    actual = existing.get('build', {}).get('id') or existing.get('buildId')
    if selected and actual != selected:
      raise ValueError(
          'A different Runner build is already running. End it before switching'
          ' builds.'
      )
    return
  if not selected:
    raise ValueError(
        'No available Runner was identified. Check the device and select an'
        ' installed Runner build before starting.'
    )
  value = next(
      (
          b
          for b in runner_builds._registered(root, identity)
          if b['id'] == selected
      ),
      None,
  )
  if value is None:
    raise ValueError('Selected Runner build is not registered on this device')
  available, reason = runner_builds._available(root, identity, value)
  if not available:
    raise ValueError(reason)
  presence = runner_builds.process_state(root, identity, slot=slot)
  if presence.get('runnerState') != 'not_running':
    raise ValueError(
        'Runner absence could not be confirmed. Check the device before'
        ' starting a build.'
    )
  kind = value['kind']
  arguments = ['--runner-session', owner['sessionId']] + (
      ['--runner-slot', slot] if slot else []
  )
  if kind == 'macos-app':
    subprocess.run(
        ['open', '-n', value['path'], '--args', *arguments],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=10,
    )
  elif kind == 'ssh-macos-app':
    app_path = device_access._absolute(value.get('path'))
    result = device_access.command(
        root,
        identity,
        ['open', '-n', app_path, '--args', *arguments],
        timeout=10,
    )
    if result.returncode:
      raise ValueError('Could not start the selected Runner build over SSH')
  elif kind == 'ios-app':
    # USBConnection provisions its existing app container and launches it.
    # It must not use --terminate-existing on a running Runner.
    return
  else:
    raise ValueError('No execution adapter is installed for this Runner build')
  deadline = time.monotonic() + 20
  while time.monotonic() < deadline:
    result = runner_discovery.inspect(root, identity, slot=slot)
    if result.get('runner'):
      runner = result['runner']
      active_owner = runner.get('owner')
      if active_owner and any(
          active_owner.get(k) != owner.get(k)
          for k in ('serverId', 'sessionId', 'runId')
      ):
        raise ValueError(
            'Another Session acquired the Runner while it was starting'
        )
      if (
          runner.get('protocolVersion') not in SUPPORTED_PROTOCOL_VERSIONS
          or result.get('status') == 'unsupported'
      ):
        raise ValueError('Started Runner is not compatible with this server')
      actual = runner.get('build', {}).get('id') or runner.get('buildId')
      if actual != selected:
        raise ValueError('Started Runner reported a different build identity')
      return
    time.sleep(0.2)
  raise TimeoutError('Selected Runner did not become reachable')
