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

"""Tests of Launchers, every launcher of this computer as one."""

import pathlib
import shutil
from unittest import mock

from absl.testing import absltest
from model_debugger.server.runner import android_launcher
from model_debugger.server.runner import launchers
from model_debugger.server.runner import runner_builds
from model_debugger.server.testing import test_main
from model_debugger.server.util import errors


class LaunchersTest(absltest.TestCase):
  """Tests of Launchers, every launcher of this computer as one."""

  def setUp(self):
    """Gives each test an empty directory of its own."""
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)

  def test_no_binary_no_local_device(self):
    """Without the binary and the tools there is no device."""
    self.enter_context(
        mock.patch.object(android_launcher, 'find_adb', lambda: None)
    )
    self.enter_context(mock.patch.object(shutil, 'which', lambda tool: None))
    none = runner_builds.RunnerBuilds(None, None)
    absent = runner_builds.RunnerBuilds(
        self.tmp_path / 'absent', self.tmp_path / 'runner'
    )
    self.assertEqual(launchers.Launchers(none).devices(), [])
    self.assertEqual(launchers.Launchers(absent).devices(), [])
    with self.assertRaises(errors.NotFound):
      launchers.Launchers(none).launch('local', runner_builds.DEVELOPMENT)
    with self.assertRaises(errors.NotFound):
      launchers.Launchers(none).builds('local')

  def test_this_computer_lists_its_builds_and_chooses_one(self):
    """With a build, this computer is a device that chooses its build."""
    self.enter_context(
        mock.patch.object(android_launcher, 'find_adb', lambda: None)
    )
    self.enter_context(mock.patch.object(shutil, 'which', lambda tool: None))
    runner = self.tmp_path / 'runner'
    runner.write_text('#!/bin/sh\n')
    found = launchers.Launchers(runner_builds.RunnerBuilds(None, runner))
    (device,) = found.devices()
    self.assertEqual(device.id, 'local')
    self.assertTrue(device.chooses_build)
    self.assertEqual(
        [build.id for build in found.builds('local')],
        [runner_builds.DEVELOPMENT],
    )

  def test_a_leftover_no_launcher_here_can_stop_is_dropped(self):
    """A record of a kind no launcher here has (no adb now) is dropped."""
    self.enter_context(
        mock.patch.object(android_launcher, 'find_adb', lambda: None)
    )
    self.enter_context(mock.patch.object(shutil, 'which', lambda tool: None))
    records = self.tmp_path / 'runners'
    records.mkdir()
    (records / 'a.json').write_text('{"kind": "android", "serial": "X"}')
    found = launchers.Launchers(runner_builds.RunnerBuilds(None, None))
    with self.assertLogs(level='WARNING'):
      found.record_in(records)
    self.assertEqual(list(records.iterdir()), [])


if __name__ == '__main__':
  test_main.main()
