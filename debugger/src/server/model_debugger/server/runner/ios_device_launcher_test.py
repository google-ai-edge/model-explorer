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

"""Tests of the iOS device launcher, with devicectl stubbed."""

import json
import pathlib
import shutil
import time
from unittest import mock

from absl.testing import absltest
from model_debugger.server.runner import android_launcher
from model_debugger.server.runner import app_version
from model_debugger.server.runner import ios_device_launcher
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
_DEVICECTL = (_XCRUN, 'devicectl', '--quiet', '--json-output', '-')
_PHONE = '367AA7B7-C0EC-5774-8D9B-D0DB34D7B588'
_PID = 4242
_EXECUTABLE = 'file:///var/containers/Bundle/Application/A/Runner.app/Runner'
# The App's commit, and its versions as its build writes them.
_COMMIT = '61dcfc77225019e067580dceb3ea268483d1938c'
_VERSION = '2026.10.15'
_BUNDLE_VERSION = str(int(_COMMIT[:12], 16))


def _device(identifier, reality='physical', pairing='paired', name='lzr-15'):
  return {
      'identifier': identifier,
      'connectionProperties': {'pairingState': pairing},
      'deviceProperties': {'name': name},
      'hardwareProperties': {
          'reality': reality,
          'platform': 'iOS',
          'marketingName': 'iPhone 17 Pro Max',
      },
  }


def _result(result) -> str:
  return json.dumps({'info': {'outcome': 'success'}, 'result': result})


def _outputs(tunnel: str, port: int, running: bool = True):
  """The answers of devicectl: the phone has the App, no other device."""
  listing = {
      'devices': [
          _device(_PHONE),
          _device('SIM', reality='simulated', name='iPhone 17'),
          _device('UNPAIRED', pairing='unpaired', name='other'),
          _device('NOAPP', name='no app'),
      ]
  }

  def apps(identifier, found):
    app_list = []
    if found:
      app_list.append({
          'bundleIdentifier': _APP,
          'bundleVersion': _BUNDLE_VERSION,
          'name': 'Model Debugger Runner',
          'version': _VERSION,
      })
    return (
        (
            *_DEVICECTL,
            'device',
            'info',
            'apps',
            '--device',
            identifier,
            '--bundle-id',
            _APP,
        ),
        _result({'apps': app_list}),
    )

  processes = [{'processIdentifier': 1, 'executable': 'file:///sbin/init'}]
  if running:
    processes.append({'processIdentifier': _PID, 'executable': _EXECUTABLE})
  return dict([
      ((*_DEVICECTL, 'list', 'devices'), _result(listing)),
      apps(_PHONE, True),
      apps('NOAPP', False),
      (
          (*_DEVICECTL, 'device', 'info', 'details', '--device', _PHONE),
          _result({'connectionProperties': {'tunnelIPAddress': tunnel}}),
      ),
      (
          (
              *_DEVICECTL,
              'device',
              'process',
              'launch',
              '--device',
              _PHONE,
              '--terminate-existing',
              _APP,
              '--listen',
              f'[{tunnel}]:{port}',
          ),
          _result({
              'process': {
                  'processIdentifier': _PID,
                  'executable': _EXECUTABLE,
              }
          }),
      ),
      (
          (
              *_DEVICECTL,
              'device',
              'info',
              'processes',
              '--device',
              _PHONE,
          ),
          _result({'runningProcesses': processes}),
      ),
      (
          (
              *_DEVICECTL,
              'device',
              'process',
              'terminate',
              '--device',
              _PHONE,
              '--pid',
              str(_PID),
          ),
          _result({}),
      ),
  ])


def _devicectl(test: absltest.TestCase, outputs) -> fake_tools.FakeTools:
  tools = fake_tools.FakeTools(outputs)
  test.enter_context(mock.patch.object(tool, 'run', tools))
  test.enter_context(
      mock.patch.object(android_launcher, 'find_adb', lambda: None)
  )
  test.enter_context(mock.patch.object(shutil, 'which', lambda tool: _XCRUN))
  return tools


class IosDeviceLauncherTest(absltest.TestCase):
  """Tests of the iOS device launcher, with devicectl stubbed."""

  def test_devices_are_the_paired_physical_ones_with_their_app(self):
    """Devices are the paired physical iOS devices, with their App."""
    tools = _devicectl(self, _outputs('fd00::1', 1))
    self.assertEqual(
        launchers.Launchers(_NO_BUILDS).devices(),
        [
            launcher.Device(
                f'ios:{_PHONE}',
                'lzr-15',
                'iOS',
                'devicectl',
                False,
                app=app_version.App(True, _COMMIT[:12], '2026-10-15'),
            ),
            launcher.Device(
                'ios:NOAPP',
                'no app',
                'iOS',
                'devicectl',
                False,
                app=app_version.App(installed=False),
            ),
        ],
    )
    # Read without launching the App.
    self.assertFalse(any('launch' in command for command in tools.commands))

  def test_no_devicectl_is_no_devices(self):
    """Without devicectl's listing, there are no devices."""
    _devicectl(self, {})
    self.assertEqual(launchers.Launchers(_NO_BUILDS).devices(), [])

  def test_launch_listens_on_the_tunnel_and_terminates(self):
    """The App listens on the tunnel's end; stop terminates its process."""
    # The fake runner listens on this Mac's IPv6 loopback, the tunnel's
    # stand-in.
    runner = fake_runner.FakeRunner(fake_runner.no_reply).start(host='[::1]')
    self.addCleanup(runner.stop)
    port = int(runner.address.rsplit(':', 1)[1])
    self.enter_context(
        mock.patch.object(ios_device_launcher, '_IOS_PORT', port)
    )
    tools = _devicectl(self, _outputs('::1', port))
    launched = launchers.Launchers(_NO_BUILDS).launch(f'ios:{_PHONE}')
    self.assertEqual(launched.address, runner.address)
    self.assertTrue(launched.alive())
    tools.commands.clear()
    launched.stop()
    self.assertEqual(
        tools.commands,
        [(
            *_DEVICECTL,
            'device',
            'process',
            'terminate',
            '--device',
            _PHONE,
            '--pid',
            str(_PID),
        )],
    )
    self.assertFalse(launched.alive())

  def test_a_process_gone_from_the_device_is_not_alive(self):
    """A runner whose process the device no longer lists is not alive."""
    _devicectl(self, _outputs('fd00::1', 1, running=False))
    device = launcher.Device(
        f'ios:{_PHONE}', 'lzr-15', 'iOS', 'devicectl', False
    )
    launched = ios_device_launcher._DeviceRunner(
        device,
        '[fd00::1]:1',
        _XCRUN,
        _PHONE,
        _PID,
        launch_records.LaunchRecords().add({}, lambda: None),
    )
    self.assertFalse(launched.alive())

  def test_an_app_gone_while_starting_says_what_to_check(self):
    """A start whose App is not running says what to check on the phone."""
    _devicectl(self, _outputs('fd00::1', 1, running=False))
    device = launcher.Device(
        f'ios:{_PHONE}', 'lzr-15', 'iOS', 'devicectl', False
    )
    launched = ios_device_launcher._DeviceRunner(
        device,
        '[::1]:1',
        _XCRUN,
        _PHONE,
        _PID,
        launch_records.LaunchRecords().add({}, lambda: None),
    )
    with self.assertRaises(RuntimeError) as raised:
      launched.wait_until_serving(time.monotonic() + 5)
    self.assertEqual(
        str(raised.exception),
        'No Runner App is running on lzr-15: check it is installed, its '
        'developer is trusted and the screen is unlocked.',
    )

  def test_a_leftover_app_is_terminated_while_it_is_the_same(self):
    """A record's App goes only while its pid runs its executable."""
    for executable, terminated in ((_EXECUTABLE, True), ('other', False)):
      records = pathlib.Path(self.create_tempdir().full_path)
      launch_records.LaunchRecords().keep_in(records)
      (records / 'left.json').write_text(
          json.dumps({
              'kind': 'ios',
              'identifier': _PHONE,
              'pid': _PID,
              'executable': executable,
          })
      )
      tools = _devicectl(self, _outputs('fd00::1', 1))
      found = launchers.Launchers(_NO_BUILDS)
      tools.commands.clear()
      found.record_in(records)
      terminate = (
          *_DEVICECTL,
          'device',
          'process',
          'terminate',
          '--device',
          _PHONE,
          '--pid',
          str(_PID),
      )
      self.assertEqual(terminate in tools.commands, terminated)
      self.assertEqual(list(records.iterdir()), [])


if __name__ == '__main__':
  test_main.main()
