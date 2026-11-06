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

"""Tests of the iOS simulator launcher, with xcrun and lsof stubbed."""

import json
import os
import pathlib
import plistlib
import shutil
import socket
import subprocess
import sys
from unittest import mock

from absl.testing import absltest
from model_debugger.server.runner import android_launcher
from model_debugger.server.runner import app_version
from model_debugger.server.runner import ios_simulator_launcher
from model_debugger.server.runner import launch_records
from model_debugger.server.runner import launcher
from model_debugger.server.runner import launchers
from model_debugger.server.runner import runner_builds
from model_debugger.server.runner import tool
from model_debugger.server.testing import fake_runner
from model_debugger.server.testing import fake_tools
from model_debugger.server.testing import test_main

_APP = launcher.APP_ID
# This computer has no runner build: only the tools give devices.
_NO_BUILDS = runner_builds.RunnerBuilds(None, None)
_XCRUN = '/usr/bin/xcrun'
_UDID = '5B1F0C4E-3C47-4F77-9D63-2C7A2D1C0A11'
_OTHER = '9A9A9A9A-0000-4000-8000-000000000000'
# The App's commit, as its build writes it into its Info.plist.
_COMMIT = '61dcfc77225019e067580dceb3ea268483d1938c'


def _simulator_outputs(
    pid: int, port: int, listener: int | None = None, bundle: str = '/x/App.app'
) -> dict[tuple[str, ...], str]:
  """The tools' answers; `listener` listens on the port (the App's pid).

  The App is installed in _UDID, as `bundle`; _OTHER has none.

  Args:
    pid: The process ID of the app.
    port: The port forwarded or listened to.
    listener: Optional process ID that listens on `port`.
    bundle: The app bundle path on disk.

  Returns:
    A mapping of xcrun command tuples to their output strings.
  """
  listing = {
      'devices': {
          'com.apple.CoreSimulator.SimRuntime.iOS-26-0': [
              {'udid': _UDID, 'name': 'iPhone 17', 'state': 'Booted'},
              {'udid': _OTHER, 'name': 'iPhone Air', 'state': 'Booted'},
          ]
      }
  }
  return {
      (_XCRUN, 'simctl', 'list', 'devices', 'booted', '--json'): json.dumps(
          listing
      ),
      (_XCRUN, 'simctl', 'get_app_container', _UDID, _APP): f'{bundle}\n',
      (
          _XCRUN,
          'simctl',
          'launch',
          '--terminate-running-process',
          _UDID,
          _APP,
          '--listen',
          f'127.0.0.1:{port}',
      ): f'{_APP}: {pid}\n',
      (_XCRUN, 'simctl', 'terminate', _UDID, _APP): '',
      (
          ios_simulator_launcher._LSOF,
          '-nP',
          f'-iTCP:{port}',
          '-sTCP:LISTEN',
          '-t',
      ): f'{pid if listener is None else listener}\n',
  }


def _simulator(
    test: absltest.TestCase, outputs, port: int
) -> fake_tools.FakeTools:
  tools = fake_tools.FakeTools(outputs)
  test.enter_context(mock.patch.object(tool, 'run', tools))
  test.enter_context(
      mock.patch.object(android_launcher, 'find_adb', lambda: None)
  )
  test.enter_context(mock.patch.object(shutil, 'which', lambda tool: _XCRUN))
  test.enter_context(
      mock.patch.object(ios_simulator_launcher, '_free_port', lambda: port)
  )
  return tools


class IosSimulatorLauncherTest(absltest.TestCase):
  """Tests of the iOS simulator launcher, with xcrun and lsof stubbed."""

  def test_simulators_are_the_booted_ones_with_their_app(self):
    """Simulators are the booted ones, with their App's version or none."""
    bundle = pathlib.Path(self.create_tempdir().full_path) / 'App.app'
    bundle.mkdir()
    with open(bundle / 'Info.plist', 'wb') as file:
      plistlib.dump(
          {
              'CFBundleShortVersionString': '2026.10.15',
              'CFBundleVersion': str(int(_COMMIT[:12], 16)),
              'LiteRTLMCommit': _COMMIT,
          },
          file,
          fmt=plistlib.FMT_BINARY,
      )
    tools = _simulator(self, _simulator_outputs(1, 1, bundle=str(bundle)), 1)
    self.assertEqual(
        launchers.Launchers(_NO_BUILDS).devices(),
        [
            launcher.Device(
                f'ios-simulator:{_UDID}',
                'iPhone 17',
                'iOS Simulator',
                'simulator',
                False,
                app=app_version.App(True, _COMMIT, '2026-10-15'),
            ),
            launcher.Device(
                f'ios-simulator:{_OTHER}',
                'iPhone Air',
                'iOS Simulator',
                'simulator',
                False,
                app=app_version.App(installed=False),
            ),
        ],
    )
    # Read without launching the App.
    self.assertFalse(any('launch' in command for command in tools.commands))

  def test_an_unreadable_bundle_has_no_version(self):
    """An App whose Info.plist cannot be read has no version."""
    _simulator(self, _simulator_outputs(1, 1, bundle='/no/App.app'), 1)
    device = launchers.Launchers(_NO_BUILDS).devices()[0]
    self.assertEqual(device.app, app_version.App(True))

  def test_simulator_launch_listens_where_told_and_terminates(self):
    """A simulator launch tells the App its port; stop terminates it."""
    # The App's process stands in as a child of this test.
    app = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(60)']
    )
    try:
      with fake_runner.FakeRunner(fake_runner.no_reply) as runner:
        port = int(runner.address.rsplit(':', 1)[1])
        tools = _simulator(self, _simulator_outputs(app.pid, port), port)
        launched = launchers.Launchers(_NO_BUILDS).launch(
            f'ios-simulator:{_UDID}'
        )
        self.assertEqual(launched.address, runner.address)
        self.assertTrue(launched.alive())
        tools.commands.clear()
        launched.stop()
        self.assertEqual(
            tools.commands,
            [(_XCRUN, 'simctl', 'terminate', _UDID, _APP)],
        )
        self.assertFalse(launched.alive())
    finally:
      app.kill()
      app.wait()

  def test_simulator_launch_refuses_another_listener_on_its_port(self):
    """A runner of another process on the App's port is not the App."""
    self.enter_context(mock.patch.object(launcher, 'LAUNCH_SECONDS', 1))
    app = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(60)']
    )
    try:
      with fake_runner.FakeRunner(fake_runner.no_reply) as runner:
        port = int(runner.address.rsplit(':', 1)[1])
        outputs = _simulator_outputs(app.pid, port, listener=os.getpid())
        tools = _simulator(self, outputs, port)
        with self.assertRaisesRegex(RuntimeError, 'Another process listens'):
          launchers.Launchers(_NO_BUILDS).launch(f'ios-simulator:{_UDID}')
        self.assertEqual(
            tools.commands[-1],
            (
                _XCRUN,
                'simctl',
                'terminate',
                _UDID,
                _APP,
            ),
        )
    finally:
      app.kill()
      app.wait()

  def test_lsof_names_the_process_that_listens(self):
    """A simulator runner listens only if its own process holds the port."""
    device = launcher.Device(
        f'ios-simulator:{_UDID}',
        'iPhone 17',
        'iOS Simulator',
        'simulator',
        False,
    )
    with socket.socket() as listener:
      listener.bind(('127.0.0.1', 0))
      listener.listen()
      address = '127.0.0.1:%d' % listener.getsockname()[1]
      record = launch_records.LaunchRecords().add({}, lambda: None)
      mine = ios_simulator_launcher._SimulatorRunner(
          device, address, os.getpid(), record
      )
      self.assertTrue(mine._listens())
      other = ios_simulator_launcher._SimulatorRunner(
          device, address, 1, record
      )
      self.assertFalse(other._listens())
    self.assertFalse(mine._listens())

  def test_a_leftover_app_is_terminated_while_it_is_the_same(self):
    """A record's App goes only while its pid started when recorded."""
    app = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(60)']
    )
    try:
      started = tool.started(app.pid)
      self.assertTrue(started)
      for when, terminated in ((started, True), ('Thu Jan  1', False)):
        records = pathlib.Path(self.create_tempdir().full_path)
        launch_records.LaunchRecords().keep_in(records)
        (records / 'left.json').write_text(
            json.dumps({
                'kind': 'ios-simulator',
                'udid': _UDID,
                'pid': app.pid,
                'started': when,
            })
        )
        tools = _simulator(self, _simulator_outputs(app.pid, 1), 1)
        found = launchers.Launchers(_NO_BUILDS)
        tools.commands.clear()
        found.record_in(records)
        self.assertEqual(
            (_XCRUN, 'simctl', 'terminate', _UDID, _APP) in tools.commands,
            terminated,
        )
        self.assertEqual(list(records.iterdir()), [])
    finally:
      app.kill()
      app.wait()


if __name__ == '__main__':
  test_main.main()
