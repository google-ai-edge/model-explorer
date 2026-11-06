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

"""Tests of the Android launcher, with adb stubbed."""

import pathlib
import shutil
import socket
import time
from unittest import mock

from absl.testing import absltest
from model_debugger.server.runner import android_launcher
from model_debugger.server.runner import app_version
from model_debugger.server.runner import launch_records
from model_debugger.server.runner import launcher
from model_debugger.server.runner import launchers
from model_debugger.server.runner import runner_builds
from model_debugger.server.runner import tool
from model_debugger.server.testing import fake_runner
from model_debugger.server.testing import fake_tools
from model_debugger.server.testing import test_main
from model_debugger.server.util import json_file

_APP = launcher.APP_ID
# This computer has no runner build: only the tools give devices.
_NO_BUILDS = runner_builds.RunnerBuilds(None, None)
_ADB = fake_tools.ADB


def _android(
    test: absltest.TestCase, outputs: dict[tuple[str, ...], str]
) -> fake_tools.FakeTools:
  tools = fake_tools.FakeTools(outputs)
  test.enter_context(mock.patch.object(tool, 'run', tools))
  test.enter_context(
      mock.patch.object(android_launcher, 'find_adb', lambda: _ADB)
  )
  test.enter_context(mock.patch.object(shutil, 'which', lambda _tool: None))
  return tools


def _android_outputs(port: int) -> dict[tuple[str, ...], str]:
  return fake_tools.android_outputs(port)


class AndroidLauncherTest(absltest.TestCase):
  """Tests of the Android launcher, with adb stubbed."""

  def test_android_devices_are_the_authorized_ones_with_their_app(self):
    """Each authorized device is listed, with its App's version or none."""
    tools = _android(self, _android_outputs(1))
    self.assertEqual(
        launchers.Launchers(_NO_BUILDS).devices(),
        [
            launcher.Device(
                'android:R5CW',
                'SM-S928B',
                'Android',
                'adb',
                False,
                app=app_version.App(
                    True,
                    '61dcfc77225019e067580dceb3ea268483d1938c',
                    '2026-10-15',
                ),
            ),
            launcher.Device(
                'android:emulator-5554',
                'sdk_gphone64',
                'Android',
                'adb',
                False,
                app=app_version.App(installed=False),
            ),
        ],
    )
    # ZY22 is not authorized: adb asks it nothing.
    self.assertFalse(any('ZY22' in command for command in tools.commands))
    # Read without starting the App.
    self.assertFalse(any('am' in command for command in tools.commands))

  def test_a_device_without_the_app_launches_none(self):
    """A launch on a device without the App refuses, starting nothing."""
    tools = _android(self, _android_outputs(1))
    found = launchers.Launchers(_NO_BUILDS)
    tools.commands.clear()
    with self.assertRaisesRegex(RuntimeError, 'No Runner App'):
      found.launch('android:emulator-5554')
    self.assertFalse(any('am' in command for command in tools.commands))

  def test_android_launch_starts_the_app_afresh_and_forwards(self):
    """An Android launch restarts the App and forwards a free port."""
    with fake_runner.FakeRunner(fake_runner.no_reply) as runner:
      port = int(runner.address.rsplit(':', 1)[1])
      tools = _android(self, _android_outputs(port))
      found = launchers.Launchers(_NO_BUILDS)
      tools.commands.clear()
      launched = found.launch('android:R5CW')
      self.assertEqual(launched.address, runner.address)
      adb = (_ADB, '-s', 'R5CW')
      launch_commands = [cmd for cmd in tools.commands if cmd[:3] == adb][-4:]
      self.assertEqual(
          launch_commands,
          [
              (*adb, 'shell', 'am', 'force-stop', _APP),
              (
                  *adb,
                  'shell',
                  'am',
                  'start',
                  '-W',
                  '-n',
                  f'{_APP}/.RunnerActivity',
              ),
              (*adb, 'shell', 'pidof', _APP),
              (*adb, 'forward', 'tcp:0', 'tcp:8769'),
          ],
      )
      self.assertTrue(launched.alive())

      tools.commands.clear()
      launched.stop()
      launched.stop()
      self.assertEqual(
          tools.commands,
          [
              (*adb, 'shell', 'am', 'force-stop', _APP),
              (*adb, 'forward', '--remove', f'tcp:{port}'),
          ],
      )
      self.assertFalse(launched.alive())

  def test_android_launch_without_an_answer_cleans_up(self):
    """An Android launch nobody answers stops the App and the forward."""
    self.enter_context(mock.patch.object(launcher, 'LAUNCH_SECONDS', 1))
    with socket.socket() as unused:
      unused.bind(('127.0.0.1', 0))
      port = unused.getsockname()[1]
    tools = _android(self, _android_outputs(port))
    with self.assertRaisesRegex(RuntimeError, 'No runner answered'):
      launchers.Launchers(_NO_BUILDS).launch('android:R5CW')
    self.assertEqual(
        tools.commands[-2:],
        [
            (_ADB, '-s', 'R5CW', 'shell', 'am', 'force-stop', _APP),
            (_ADB, '-s', 'R5CW', 'forward', '--remove', f'tcp:{port}'),
        ],
    )

  def test_an_app_gone_while_starting_says_what_to_check(self):
    """A start whose App is not running says what to check on the phone."""
    _android(self, {})  # pidof finds no App.
    device = launcher.Device('android:R5CW', 'Pixel 8', 'Android', 'adb', False)
    launched = android_launcher._AndroidRunner(
        device,
        [_ADB, '-s', 'R5CW'],
        1,
        launch_records.LaunchRecords().add({}, lambda: None),
    )
    with self.assertRaises(RuntimeError) as raised:
      launched.wait_until_serving(time.monotonic() + 5)
    self.assertEqual(
        str(raised.exception),
        'No Runner App is running on Pixel 8: check it is installed and '
        'the screen is unlocked.',
    )

  def test_a_launch_is_recorded_until_stopped(self):
    """The record names the serial, the App's pid and the forward."""
    records = pathlib.Path(self.create_tempdir().full_path)
    with fake_runner.FakeRunner(fake_runner.no_reply) as runner:
      port = int(runner.address.rsplit(':', 1)[1])
      _android(self, _android_outputs(port))
      found = launchers.Launchers(_NO_BUILDS)
      found.record_in(records)
      launched = found.launch('android:R5CW')
      (path,) = records.glob('*.json')
      self.assertEqual(
          json_file.read_json(path),
          {
              'kind': 'android',
              'serial': 'R5CW',
              'pid': fake_tools.APP_PID,
              'port': port,
          },
      )
      launched.stop()
      self.assertEqual(list(records.iterdir()), [])

  def test_stop_all_stops_the_app_and_removes_the_forward(self):
    """As the server exits, a runner still running is stopped."""
    with fake_runner.FakeRunner(fake_runner.no_reply) as runner:
      port = int(runner.address.rsplit(':', 1)[1])
      tools = _android(self, _android_outputs(port))
      found = launchers.Launchers(_NO_BUILDS)
      launched = found.launch('android:R5CW')
      tools.commands.clear()
      found.stop_all()
      adb = (_ADB, '-s', 'R5CW')
      self.assertEqual(
          tools.commands,
          [
              (*adb, 'shell', 'am', 'force-stop', _APP),
              (*adb, 'forward', '--remove', f'tcp:{port}'),
          ],
      )
      self.assertFalse(launched.alive())
      with self.assertRaisesRegex(RuntimeError, 'stopping'):
        found.launch('android:R5CW')

  def test_a_leftover_forward_and_app_are_stopped(self):
    """What a killed server recorded goes, once adb shows it is still so."""
    records = pathlib.Path(self.create_tempdir().full_path)
    recorded = launch_records.LaunchRecords()
    recorded.keep_in(records)
    recorded.add(
        {
            'kind': 'android',
            'serial': 'R5CW',
            'pid': fake_tools.APP_PID,
            'port': 5000,
        },
        lambda: None,
    )
    tools = _android(self, _android_outputs(5000))
    found = launchers.Launchers(_NO_BUILDS)
    tools.commands.clear()
    found.record_in(records)
    adb = (_ADB, '-s', 'R5CW')
    self.assertEqual(
        tools.commands,
        [
            (_ADB, 'forward', '--list'),
            (*adb, 'forward', '--remove', 'tcp:5000'),
            (*adb, 'shell', 'pidof', _APP),
            (*adb, 'shell', 'am', 'force-stop', _APP),
        ],
    )
    self.assertEqual(list(records.iterdir()), [])

  def test_a_leftover_no_longer_there_is_left_alone(self):
    """Another forward, or a new App process, is not the one recorded."""
    records = pathlib.Path(self.create_tempdir().full_path)
    recorded = launch_records.LaunchRecords()
    recorded.keep_in(records)
    recorded.add(
        {'kind': 'android', 'serial': 'R5CW', 'pid': '99', 'port': 5000},
        lambda: None,
    )
    # adb forwards another port to the App, and the App is pid 4242.
    tools = _android(self, _android_outputs(6000))
    found = launchers.Launchers(_NO_BUILDS)
    tools.commands.clear()
    found.record_in(records)
    self.assertEqual(
        tools.commands,
        [
            (_ADB, 'forward', '--list'),
            (_ADB, '-s', 'R5CW', 'shell', 'pidof', _APP),
        ],
    )


if __name__ == '__main__':
  test_main.main()
