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

"""Execution routing: Server dispatches to a Runner, never owns workers."""

import copy
import json
import pathlib
import socket
from typing import Any
import uuid

from model_explorer_debugger import fsutil
from model_explorer_debugger.runtime import device_access
from model_explorer_debugger.runtime import device_ids
from model_explorer_debugger.runtime import ios_device
from model_explorer_debugger.runtime import macos_device
from model_explorer_debugger.runtime import runner_device
from model_explorer_debugger.runtime import runner_launch

LOCAL_DEVICE = device_ids.LOCAL_DEVICE
resolve_device = device_ids.resolve_device

__all__ = [
    'LOCAL_DEVICE',
    'Runners',
    'local_python_capability',
    'resolve_device',
]


LOCAL_HOST_DEVICES = frozenset({None, '', LOCAL_DEVICE, 'Server host'})


class Runners:
  """Routes session execution and lifecycle across RunnerDevices."""

  def __init__(self, root):
    self.root = pathlib.Path(root)
    identity_path = self.root / 'server-identity.json'
    try:
      self.identity = json.loads(identity_path.read_text())
    except FileNotFoundError:
      self.identity = {'id': str(uuid.uuid4()), 'name': socket.gethostname()}
      if self.root.is_dir():
        fsutil.atomic_json(identity_path, self.identity)
    self.on_lifecycle = None
    self.mac = macos_device.MacDevices(root)
    self.ios = ios_device.IOSDevices(root)
    self.adapters: tuple[runner_device.RunnerDeviceAdapter, ...] = (
        self.mac,
        self.ios,
    )
    for adapter in self.adapters:
      adapter.on_lifecycle = self._lifecycle

  def _lifecycle(self, session, role, kind, fields):
    if self.on_lifecycle:
      self.on_lifecycle(session, role, kind, fields)

  def _adapter_for_device(
      self, device: str
  ) -> runner_device.RunnerDeviceAdapter | None:
    device_str = str(device or '')
    for adapter in self.adapters:
      if adapter.matches_device(device_str):
        return adapter
    return None

  def _resolve_adapter(
      self,
      device: str | None,
      *,
      fallback: runner_device.RunnerDeviceAdapter | None = None,
  ) -> runner_device.RunnerDeviceAdapter:
    return self._adapter_for_device(str(device or '')) or fallback or self.mac

  def resolve_runs(self, runs):
    """Resolves device aliases, assigns slots, and checks device occupancy."""
    values = copy.deepcopy(runs)
    for run in values:
      try:
        run['device'] = device_ids.resolve_device(self.root, run.get('device'))
      except ValueError:
        if run.get('device') not in LOCAL_HOST_DEVICES:
          raise
        run['device'] = LOCAL_DEVICE
    for run in values:
      adapter = self._adapter_for_device(run['device'])
      if (adapter and adapter.SUPPORTS_MULTI_SLOT) or (
          run['device'] == LOCAL_DEVICE
      ):
        run['_runner_slot'] = run['id']
      else:
        run['_runner_slot'] = 'ref'
    devices = [self.device_key(run['device']) for run in values]
    if len(set(devices)) != len(devices) and any(
        str(device).startswith(('ios:', 'adb:')) for device in devices
    ):
      raise ValueError('This physical device supports only one Runner')
    return values

  def device_key(self, value):
    """Resolves a device identifier to its physical-host occupancy key."""
    try:
      resolved = device_ids.resolve_device(self.root, value)
    except ValueError:
      return LOCAL_DEVICE if value in LOCAL_HOST_DEVICES else value
    adapter = self._adapter_for_device(resolved)
    return adapter.physical_device_key(resolved) if adapter else resolved

  def execute(self, request, emit, cancelled):
    """Validates runtime support and dispatches an execution request."""
    request = copy.deepcopy(request)
    raw_device = str(request['run'].get('device', ''))
    initial_adapter = self._adapter_for_device(raw_device)
    if (
        request['run'].get('runtime') == 'PyTorch'
        and raw_device not in LOCAL_HOST_DEVICES
        and not (
            initial_adapter is not None
            and initial_adapter.supports_run(request['run'])
        )
    ):
      raise ValueError(
          'PyTorch currently requires local shared files on the server Mac;'
          ' remote Runner execution is unavailable'
      )
    if (initial_adapter and initial_adapter.SUPPORTS_MULTI_SLOT) or request[
        'run'
    ].get('device') in LOCAL_HOST_DEVICES:
      request['run']['_runner_slot'] = request['run'].get('id', 'ref')
    if request.get('owner') and request.get('operation') == 'initialize':
      try:
        resolved = dict(
            request,
            run=dict(
                request['run'],
                device=device_ids.resolve_device(
                    self.root, request['run'].get('device')
                ),
            ),
        )
        # Unmatched device strings fall back to self.ios during initialize
        # pre-checks so malformed USB UDIDs surface IOSDevices validation
        # errors.
        adapter = self._resolve_adapter(
            resolved['run']['device'], fallback=self.ios
        )
        key = adapter.connection_key(resolved)
        existing = adapter.connections.get(key)
        if (
            key,
            request.get('execution_id', request['session_id']),
        ) in adapter.lost_executions:
          raise ConnectionError(
              'Runner execution was lost; reconnection is disabled'
          )
      except ValueError:
        existing = None
      if existing is None:
        runner_launch.ensure_started(
            self.root, request['run'], request['owner']
        )
    request['run']['device'] = device_ids.resolve_device(
        self.root, request['run'].get('device')
    )
    # Intentional behavior guard: verify the resolved device adapter supports
    # the requested runtime (SUPPORTED_RUNTIMES) before dispatching execution.
    adapter = self._adapter_for_device(request['run']['device'])
    if adapter is not None and adapter.supports_run(request['run']):
      return adapter.execute(request, emit, cancelled)
    raise ValueError('The selected Runner does not support this runtime')

  def import_result(self, request, result):
    """Imports and validates raw Runner capture artifacts for a turn."""
    adapter = self._resolve_adapter(request['run'].get('device', ''))
    return adapter.import_result(request, result)

  def close_session(self, identity):
    """Closes the session across all registered Runner device adapters."""
    errors = []
    for adapter in self.adapters:
      try:
        adapter.close_session(identity)
      except Exception as error:  # pylint: disable=broad-exception-caught
        errors.append(str(error))
    if errors:
      raise ConnectionError('; '.join(errors))

  def abandon_session(self, identity):
    """Fences an unresponsive session across all Runner device adapters."""
    for adapter in self.adapters:
      adapter.abandon_session(identity)

  def reset_session(self, identity):
    """Resets conversation state for a session across all device adapters."""
    for adapter in self.adapters:
      adapter.reset_session(identity)

  def close(self):
    """Closes all device adapters and active SSH tunnels."""
    for adapter in self.adapters:
      adapter.close()
    device_access.close_tunnels()


def local_python_capability(
    home_dir: pathlib.Path | str | None = None,
) -> dict[str, Any]:
  """Reads local Python Runner capabilities exported by the macOS Runner App."""
  base = dict(
      id='PyTorch',
      available=False,
      backends=['CPU'],
      precisions=['', 'float32', 'float16', 'bfloat16'],
      supported_options=['backend', 'precision', 'cpuThreads', 'contextLength'],
      reason="Configure the local Runner App's PyTorch environment.",
  )
  if home_dir and str(home_dir).strip():
    resolved_home = pathlib.Path(home_dir)
  else:
    resolved_home = pathlib.Path.home()
  path = (
      resolved_home
      / 'Library/Application Support/Model Debugger Runner/python-runner.json'
  )
  try:
    value = json.loads(path.read_text())
    if (
        value.get('version') not in (1, 2)
        or not pathlib.Path(value['executable']).is_file()
    ):
      return base
    backends = value['backends']
    if (
        not isinstance(backends, list)
        or not backends
        or any(b not in ('CPU', 'MPS', 'CUDA') for b in backends)
    ):
      return base
    return dict(base, available=True, backends=backends, reason='')
  except (OSError, ValueError, KeyError, TypeError):
    return base
