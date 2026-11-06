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

"""Android devices as devices: the Runner App, reached through adb forward."""

import os
import pathlib
import shutil
from typing import Any

from absl import logging
from model_debugger.server.runner import app_version
from model_debugger.server.runner import launch_records
from model_debugger.server.runner import launcher
from model_debugger.server.runner import tool

# The Android App's activity, which starts its runner.
_ACTIVITY = f'{launcher.APP_ID}/.RunnerActivity'

# The port the Android App listens on, on the device.
_ANDROID_PORT = 8769


def find_adb() -> str | None:
  """Returns adb: the Android SDK's, else the one on PATH; None if neither."""
  candidates = []
  if 'ANDROID_HOME' in os.environ:
    candidates.append(pathlib.Path(os.environ['ANDROID_HOME']))
  candidates.append(pathlib.Path.home() / 'Library/Android/sdk')
  for sdk in candidates:
    adb = sdk / 'platform-tools' / 'adb'
    if adb.is_file():
      return str(adb)
  return shutil.which('adb')


class AndroidLauncher(launcher.Launcher):
  """Runs the App on the authorized Android devices that have it.

  Every authorized device is listed, with the App's version, or with none
  installed.

  Each launch is recorded by the device's serial, the App's pid there and
  the local port forwarded to it.
  """

  KIND = 'android'

  def __init__(
      self,
      adb: str,
      records: launch_records.LaunchRecords | None = None,
  ) -> None:
    """Reaches devices through `adb`.

    Args:
        adb: The adb command.
        records: Where its launches are recorded; None for its own.
    """
    self._adb = adb
    self._records = records or launch_records.LaunchRecords()

  def devices(self) -> list[launcher.Device]:
    """Returns the authorized devices, each with its App, if installed.

    The App's version is what `dumpsys package` says of it, read without
    starting it.
    """
    try:
      listing = tool.run([self._adb, 'devices'])
    except RuntimeError as error:
      logging.info('No Android devices: %s', error)
      return []
    found = []
    for line in listing.splitlines()[1:]:
      fields = line.split()
      if len(fields) != 2 or fields[1] != 'device':
        continue  # offline, unauthorized, or not a device line
      serial = fields[0]
      adb = (self._adb, '-s', serial)
      try:
        app = app_version.from_android_package(
            launcher.APP_ID,
            tool.run([*adb, 'shell', 'dumpsys', 'package', launcher.APP_ID]),
        )
        name = tool.run([*adb, 'shell', 'getprop', 'ro.product.model']).strip()
      except RuntimeError:
        continue  # unreachable now
      found.append(
          launcher.Device(
              f'android:{serial}',
              name or serial,
              'Android',
              'adb',
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
      model_dir: pathlib.Path | None = None,
  ) -> launcher.LaunchedRunner:
    """Starts the App afresh and forwards a free local port to it."""
    del build, model_dir  # The App installed is the runner.
    serial = device.id.removeprefix('android:')
    adb = (self._adb, '-s', serial)
    # Recorded before anything starts: a stop meanwhile sees what started.
    record = self._records.add(
        {'kind': self.KIND, 'serial': serial},
        lambda: _stop_app_and_forward(adb, record.fields.get('port')),
    )
    try:
      # The App serves one client per process: a fresh process, never
      # the old.
      _stop_android_app(adb)
      # -W: returns once the App's process runs, so alive() can be asked.
      tool.run([*adb, 'shell', 'am', 'start', '-W', '-n', _ACTIVITY])
      record.update(pid=_app_pid(adb))
      # tcp:0 picks a free local port, and adb prints it.
      port = int(tool.run([*adb, 'forward', 'tcp:0', f'tcp:{_ANDROID_PORT}']))
      record.update(port=port)
    except Exception:
      record.stop()
      raise
    return _AndroidRunner(device, adb, port, record)

  def stop_leftover(self, fields: dict[str, Any]) -> None:
    """Removes the forward a previous server left, and stops its App.

    The forward only while adb still lists it from that port to the
    App's; the App only while its process is the one recorded.

    Args:
      fields: Process fields recorded by a previous server run.

    Raises:
      RuntimeError: adb failed.
    """
    serial, port, pid = (
        fields.get('serial'),
        fields.get('port'),
        fields.get('pid'),
    )
    if not isinstance(serial, str):
      return
    adb = [self._adb, '-s', serial]
    if isinstance(port, int):
      # Each line: '<serial> tcp:<local port> tcp:<device port>'.
      forward = [serial, f'tcp:{port}', f'tcp:{_ANDROID_PORT}']
      listed = tool.run([self._adb, 'forward', '--list'])
      if any(line.split() == forward for line in listed.splitlines()):
        logging.info(
            'Removing the forward of %s from port %d, left by a '
            'previous server',
            serial,
            port,
        )
        tool.run_logged([*adb, 'forward', '--remove', f'tcp:{port}'])
    if pid and _app_pid(adb) == pid:
      logging.info('Stopping the App on %s, left by a previous server', serial)
      _stop_android_app(adb)


class _AndroidRunner(launcher.LaunchedRunner):
  """The App on an Android device, reached through an adb forward."""

  def __init__(
      self,
      device: launcher.Device,
      adb: list[str],
      local_port: int,
      record: launch_records.Record,
  ) -> None:
    super().__init__(device, f'127.0.0.1:{local_port}')
    self._adb = adb  # adb -s <serial>
    self._record = record

  def _alive(self) -> bool:
    # pidof prints nothing, and fails, once the App's process is gone.
    return not self._record.stopped and bool(_app_pid(self._adb))

  def _stop(self) -> None:
    self._record.stop()

  def _gone_while_starting(self) -> str:
    return (
        f'No Runner App is running on {self.device.name}: check it is '
        'installed and the screen is unlocked.'
    )


def _app_pid(adb: list[str]) -> str:
  """The App's process on the device, as pidof says it; '' for none."""
  try:
    return tool.run([*adb, 'shell', 'pidof', launcher.APP_ID]).strip()
  except RuntimeError:
    return ''  # pidof fails when there is no such process


def _stop_app_and_forward(adb: list[str], local_port: int | None) -> None:
  """Stops the App and removes the forward to it from `local_port`."""
  _stop_android_app(adb)
  if local_port is not None:
    tool.run_logged([*adb, 'forward', '--remove', f'tcp:{local_port}'])


def _stop_android_app(adb: list[str]) -> None:
  tool.run_logged([*adb, 'shell', 'am', 'force-stop', launcher.APP_ID])
