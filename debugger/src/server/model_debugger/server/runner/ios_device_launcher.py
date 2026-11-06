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

"""Paired iPhones and iPads as devices: the App, over the CoreDevice tunnel.

`xcrun devicectl` lists the devices, launches the App and stops it. The
tunnel CoreDevice keeps to a paired device is an IPv6 link between this Mac
and the device: the App listens on the device's end of it, and the server
connects there. No port is forwarded.
"""

import json
from typing import Any

from absl import logging
from model_debugger.server.runner import app_version
from model_debugger.server.runner import launch_records
from model_debugger.server.runner import launcher
from model_debugger.server.runner import tool

# The port the App listens on, on the device's end of the tunnel.
_IOS_PORT = 8769


class IosDeviceLauncher(launcher.Launcher):
  """Runs the App on the paired iOS devices that have it.

  Each launch is recorded by the device's identifier and the App's pid and
  executable there.
  """

  KIND = 'ios'

  def __init__(
      self,
      xcrun: str,
      records: launch_records.LaunchRecords | None = None,
  ) -> None:
    """Reaches devices through `xcrun devicectl`.

    Args:
        xcrun: The xcrun command.
        records: Where its launches are recorded; None for its own.
    """
    self._xcrun = xcrun
    self._records = records or launch_records.LaunchRecords()

  def devices(self) -> list[launcher.Device]:
    """Returns the paired physical devices, each with its App, if any.

    The App's version is what devicectl lists of it, read without
    launching it.
    """
    try:
      listing = _devicectl(self._xcrun, ['list', 'devices'])
    except RuntimeError as error:
      logging.info('No iOS devices: %s', error)
      return []
    found = []
    for device in listing['devices']:
      hardware = device.get('hardwareProperties', {})
      connection = device.get('connectionProperties', {})
      if (
          hardware.get('reality') != 'physical'
          or hardware.get('platform') != 'iOS'
          or connection.get('pairingState') != 'paired'
      ):
        continue  # a simulator, another platform, or not paired
      identifier = device['identifier']
      try:
        apps = _devicectl(
            self._xcrun,
            [
                'device',
                'info',
                'apps',
                '--device',
                identifier,
                '--bundle-id',
                launcher.APP_ID,
            ],
        )
      except RuntimeError:
        continue  # unreachable now: locked, or gone
      app = app_version.App(installed=False)
      if apps['apps']:
        # `version` and `bundleVersion` are the App's
        # CFBundleShortVersionString and CFBundleVersion.
        listed = apps['apps'][0]
        app = app_version.from_ios_bundle(
            str(listed.get('version', '')),
            str(listed.get('bundleVersion', '')),
        )
      found.append(
          launcher.Device(
              f'ios:{identifier}',
              device.get('deviceProperties', {}).get('name')
              or hardware.get('marketingName')
              or identifier,
              'iOS',
              'devicectl',
              False,
              app=app,
          )
      )
    return found

  def launch(
      self,
      device: launcher.Device,
      deadline: float,
      build: str | None,
      model_dir: Any = None,
  ) -> launcher.LaunchedRunner:
    """Starts the App afresh, on the device's end of the tunnel."""
    del build, model_dir  # The App installed is the runner.
    identifier = device.id.removeprefix('ios:')
    # Asking for the details brings the tunnel up, if it was not.
    details = _devicectl(
        self._xcrun, ['device', 'info', 'details', '--device', identifier]
    )
    tunnel = details['connectionProperties'].get('tunnelIPAddress')
    if not tunnel:
      raise RuntimeError(f'{device.name} has no CoreDevice tunnel.')
    # The tunnel's end alone, not every interface: the runner is not
    # offered to the device's Wi-Fi.
    address = f'[{tunnel}]:{_IOS_PORT}'
    record = self._records.add(
        {'kind': self.KIND, 'identifier': identifier},
        lambda: _terminate(self._xcrun, identifier, record.fields.get('pid')),
    )
    try:
      # The App serves one client per process: a fresh process, never
      # the old.
      launched = _devicectl(
          self._xcrun,
          [
              'device',
              'process',
              'launch',
              '--device',
              identifier,
              '--terminate-existing',
              launcher.APP_ID,
              '--listen',
              address,
          ],
      )
      process = launched['process']
      pid = process['processIdentifier']
      record.update(pid=pid, executable=process.get('executable'))
    except BaseException:
      record.stop()
      raise
    return _DeviceRunner(device, address, self._xcrun, identifier, pid, record)

  def stop_leftover(self, fields: dict[str, Any]) -> None:
    """Terminates the App a previous server left, if it still runs.

    Only while the device runs a process with the pid and the executable
    recorded: a pid alone may be another process's by now.

    Args:
      fields: Dictionary containing leftover launch fields.

    Raises:
        RuntimeError: devicectl failed.
    """
    identifier, pid, executable = (
        fields.get('identifier'),
        fields.get('pid'),
        fields.get('executable'),
    )
    if not isinstance(identifier, str) or not pid or not executable:
      return
    if not any(
        process.get('processIdentifier') == pid
        and process.get('executable') == executable
        for process in _processes(self._xcrun, identifier)
    ):
      return
    logging.info(
        'Terminating the App on %s, left by a previous server', identifier
    )
    _terminate(self._xcrun, identifier, pid)


class _DeviceRunner(launcher.LaunchedRunner):
  """The App on a paired iOS device, reached over the CoreDevice tunnel."""

  def __init__(
      self,
      device: launcher.Device,
      address: str,
      xcrun: str,
      identifier: str,
      pid: int,
      record: launch_records.Record,
  ) -> None:
    super().__init__(device, address)
    self._xcrun = xcrun
    self._identifier = identifier
    self._pid = pid
    self._record = record

  def _alive(self) -> bool:
    if self._record.stopped:
      return False
    try:
      running = _processes(self._xcrun, self._identifier)
    except RuntimeError:
      return False
    return any(
        process.get('processIdentifier') == self._pid for process in running
    )

  def _stop(self) -> None:
    self._record.stop()

  def _gone_while_starting(self) -> str:
    return (
        f'No Runner App is running on {self.device.name}: check it is '
        'installed, its developer is trusted and the screen is unlocked.'
    )


def _processes(xcrun: str, identifier: str) -> list[dict[str, Any]]:
  """The processes a device runs, as devicectl lists them.

  Args:
    xcrun: The xcrun executable path or command.
    identifier: The device identifier.

  Returns:
    A list of running process dictionaries.

  Raises:
      RuntimeError: devicectl failed.
  """
  running = _devicectl(
      xcrun, ['device', 'info', 'processes', '--device', identifier]
  )
  return running['runningProcesses']


def _terminate(xcrun: str, identifier: str, pid: int | None) -> None:
  """Terminates the App's process, once its pid is known."""
  if pid is None:
    return  # Not launched yet; the launch's own failure stops nothing.
  try:
    _devicectl(
        xcrun,
        [
            'device',
            'process',
            'terminate',
            '--device',
            identifier,
            '--pid',
            str(pid),
        ],
    )
  except RuntimeError as error:
    # Stopping must not fail its caller (tool.run_logged).
    logging.warning('%s', error)


def _devicectl(xcrun: str, args: list[str]) -> dict[str, Any]:
  """Runs `xcrun devicectl` and returns the `result` of its JSON output.

  Args:
    xcrun: The xcrun executable path or command.
    args: Additional command line arguments to pass to devicectl.

  Returns:
    The result dictionary from devicectl's JSON output.

  Raises:
      RuntimeError: devicectl failed.
  """
  # --quiet leaves the JSON alone on standard output ('-'). The options go
  # first: after a launch's bundle id, every argument is the App's.
  output = tool.run(
      [xcrun, 'devicectl', '--quiet', '--json-output', '-', *args]
  )
  return json.loads(output)['result']
