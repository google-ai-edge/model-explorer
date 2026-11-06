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

"""Booted iOS simulators as devices: the Runner App, on this Mac's loopback."""

import json
import os
import pathlib
import plistlib
import shutil
import socket
from typing import Any

from absl import logging
from model_debugger.server.runner import app_version
from model_debugger.server.runner import launch_records
from model_debugger.server.runner import launcher
from model_debugger.server.runner import tool

# Lists the processes listening on a port; macOS has it in /usr/sbin.
_LSOF = shutil.which('lsof') or '/usr/sbin/lsof'


class IosSimulatorLauncher(launcher.Launcher):
  """Runs the App in the booted simulators that have it.

  Each launch is recorded by the simulator's udid and the App's pid and
  start time, a process of this Mac.
  """

  KIND = 'ios-simulator'

  def __init__(
      self,
      xcrun: str,
      records: launch_records.LaunchRecords | None = None,
  ) -> None:
    """Reaches simulators through `xcrun simctl`.

    Args:
        xcrun: The xcrun command.
        records: Where its launches are recorded; None for its own.
    """
    self._xcrun = xcrun
    self._records = records or launch_records.LaunchRecords()

  def devices(self) -> list[launcher.Device]:
    """Returns the booted simulators, each with its App, if installed.

    The App's version is its bundle's, read from the simulator's copy of
    the App without launching it.
    """
    try:
      listing = json.loads(
          tool.run([
              self._xcrun,
              'simctl',
              'list',
              'devices',
              'booted',
              '--json',
          ])
      )
    except RuntimeError as error:
      logging.info('No iOS simulators: %s', error)
      return []
    found = []
    for simulators in listing['devices'].values():
      for simulator in simulators:
        if simulator['state'] != 'Booted':
          continue
        udid = simulator['udid']
        try:
          bundle = tool.run([
              self._xcrun,
              'simctl',
              'get_app_container',
              udid,
              launcher.APP_ID,
          ]).strip()
        except RuntimeError:
          # The App is not installed there.
          app = app_version.App(installed=False)
        else:
          app = _bundle_app(bundle)
        found.append(
            launcher.Device(
                f'ios-simulator:{udid}',
                simulator['name'],
                'iOS Simulator',
                'simulator',
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
    """Starts the App afresh, told to listen on a port of this server's."""
    del build, model_dir  # The App installed is the runner.
    udid = device.id.removeprefix('ios-simulator:')
    # The simulator shares this Mac's loopback: a port of the server's
    # choosing keeps the App off any other listener's, and the launch waits
    # for the App itself to listen there (_listens).
    address = f'127.0.0.1:{_free_port()}'
    terminate = [self._xcrun, 'simctl', 'terminate', udid, launcher.APP_ID]
    # Recorded before the App starts: a stop meanwhile terminates it.
    record = self._records.add(
        {'kind': self.KIND, 'udid': udid},
        lambda: _terminate(terminate, record.fields.get('pid')),
    )
    try:
      output = tool.run([
          self._xcrun,
          'simctl',
          'launch',
          '--terminate-running-process',
          udid,
          launcher.APP_ID,
          '--listen',
          address,
      ])
      # simctl prints '<bundle id>: <pid>'.
      _, pid_text = output.strip().rsplit(':', 1)
      pid = int(pid_text)
      record.update(pid=pid, started=tool.started(pid))
    except BaseException:
      record.stop()
      raise
    return _SimulatorRunner(device, address, pid, record)

  def stop_leftover(self, fields: dict[str, Any]) -> None:
    """Terminates the App a previous server left, if it still runs.

    Only while the App's process is the one recorded: its pid, started
    when the record says.

    Args:
      fields: Dictionary containing runner record fields (udid, pid, started).
    """
    udid, pid, when = (
        fields.get('udid'),
        fields.get('pid'),
        fields.get('started'),
    )
    if (
        not isinstance(udid, str)
        or not isinstance(pid, int)
        or not when
        or tool.started(pid) != when
    ):
      return
    logging.info(
        'Terminating the App in simulator %s, left by a previous server',
        udid,
    )
    tool.run_logged([self._xcrun, 'simctl', 'terminate', udid, launcher.APP_ID])


class _SimulatorRunner(launcher.LaunchedRunner):
  """The App in a booted iOS simulator: a process of this Mac."""

  def __init__(
      self,
      device: launcher.Device,
      address: str,
      pid: int,
      record: launch_records.Record,
  ) -> None:
    super().__init__(device, address)
    self._pid = pid
    self._record = record

  def _sees_process(self) -> bool:
    return True

  def _alive(self) -> bool:
    return not self._record.stopped and _alive(self._pid)

  def _listens(self) -> bool:
    # The port was free when chosen, but anything on this Mac may take it
    # before the App binds it; then the App's bind fails and it is not the
    # one that answers.
    port = self.address.rsplit(':', 1)[1]
    try:
      listing = tool.run([_LSOF, '-nP', f'-iTCP:{port}', '-sTCP:LISTEN', '-t'])
    except RuntimeError:
      return False  # lsof fails when nothing listens there
    return str(self._pid) in listing.split()

  def _stop(self) -> None:
    self._record.stop()


def _alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  return True


def _terminate(terminate: list[str], pid: int | None) -> None:
  """Terminates the App, unless its process, once known, is gone."""
  if pid is None or _alive(pid):
    tool.run_logged(terminate)


def _bundle_app(bundle: str) -> app_version.App:
  """The App as its bundle's Info.plist describes it."""
  try:
    with open(pathlib.Path(bundle) / 'Info.plist', 'rb') as file:
      info = plistlib.load(file)
  except (OSError, plistlib.InvalidFileException, ValueError):
    info = None
  if not isinstance(info, dict):
    return app_version.App(True)
  return app_version.from_ios_bundle(*(
      str(info.get(key, ''))
      for key in (
          'CFBundleShortVersionString',
          'CFBundleVersion',
          'LiteRTLMCommit',
      )
  ))


def _free_port() -> int:
  with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    return probe.getsockname()[1]
