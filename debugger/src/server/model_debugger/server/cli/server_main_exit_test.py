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

"""However the command ends, no runner it launched outlives it.

The command runs in this process, on a port of its own, with adb stubbed:
an Android phone runs the App, and a script stands in for this computer's
runner. Once it serves, a runner is launched on each, and the command is
stopped: by SIGINT, SIGTERM or SIGHUP, or by an exception as it starts.
A command killed outright (SIGKILL) stops nothing; the next one on its
workspace stops what it recorded, and nothing else.
"""

import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import threading
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from model_debugger.server import server as server_lib
from model_debugger.server.cli import server_main
from model_debugger.server.runner import android_launcher
from model_debugger.server.runner import launcher
from model_debugger.server.runner import launchers
from model_debugger.server.runner import tool
from model_debugger.server.testing import fake_runner
from model_debugger.server.testing import fake_tools
from model_debugger.server.testing import test_main

_APP = launcher.APP_ID
_ADB = (fake_tools.ADB, '-s', 'R5CW')


def _gone(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return True
  return False


def _orphan() -> int:
  """Starts a process that is no child of this one, as a killed server's.

  Returns:
      Its pid; it sleeps for a minute.
  """
  shell = subprocess.run(
      ['/bin/sh', '-c', 'sleep 60 >/dev/null 2>&1 & echo $!'],
      capture_output=True,
      text=True,
      check=True,
  )
  return int(shell.stdout)


class ServerMainExitTest(absltest.TestCase):
  """However the command ends, it stops every runner it launched."""

  def setUp(self) -> None:
    """A phone over adb, a stand-in runner, and the signals restored."""
    super().setUp()
    self.tmp = pathlib.Path(self.create_tempdir().full_path)
    self.workspace = self.tmp / 'workspace'
    (self.tmp / 'kept').mkdir()
    runner = self.enter_context(fake_runner.FakeRunner(fake_runner.no_reply))
    self.port = int(runner.address.rsplit(':', 1)[1])
    # This computer's runner: says the fake runner's port, then lives.
    self.script = self.tmp / 'runner'
    self.script.write_text(
        '#!/bin/sh\n'
        f'echo "runner: listening on port {self.port}"\n'
        'exec sleep 60\n'
    )
    self.script.chmod(0o755)
    self.tools = fake_tools.FakeTools(fake_tools.android_outputs(self.port))
    self.enter_context(mock.patch.object(tool, 'run', self.tools))
    self.enter_context(
        mock.patch.object(
            android_launcher,
            'find_adb',
            side_effect=lambda: fake_tools.ADB,
            autospec=True,
            spec_set=True,
        )
    )
    # No xcrun: no simulator, no iPhone.
    self.enter_context(
        mock.patch.object(
            shutil,
            'which',
            side_effect=lambda name, *args, **kwargs: None,
            autospec=True,
            spec_set=True,
        )
    )
    made = self.made = []

    class Recorded(launchers.Launchers):

      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        made.append(self)

    self.enter_context(
        mock.patch.object(server_lib.launchers, 'Launchers', Recorded)
    )
    for signum in server_main._ENDING_SIGNALS:
      self.addCleanup(signal.signal, signum, signal.getsignal(signum))
    self.launched: list[launcher.LaunchedRunner] = []

  def _launch(self) -> None:
    """Launches a runner on this computer and one on the phone."""
    self.assertLen(self.made, 1)
    found = self.made[0]
    self.launched.append(found.launch('local', 'dev'))
    self.launched.append(found.launch('android:R5CW'))
    self.assertLen(list((self.workspace / 'runners').iterdir()), 2)

  def _main(self) -> None:
    """Runs the command as `model-debugger-server` would, in here."""
    with flagsaver.flagsaver(
        workspace=str(self.workspace),
        port=0,
        runner=str(self.script),
        kept_builds=str(self.tmp / 'kept'),
        pytorch_runner='',
    ):
      server_main.main(['model-debugger-server'])

  def _serve_then(self, then) -> None:
    """Has `then` run on a thread of its own once the server serves."""
    serve = server_lib.Server.wait

    def wait(server):
      port = server.state().port

      def drive():
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
          try:
            socket.create_connection(('127.0.0.1', port)).close()
            break
          except OSError:
            time.sleep(0.05)
        then()

      threading.Thread(target=drive, daemon=True).start()
      serve(server)

    self.enter_context(mock.patch.object(server_lib.Server, 'wait', wait))

  def _assert_all_stopped(self) -> None:
    self.assertLen(self.launched, 2)
    local, phone = self.launched[0], self.launched[1]
    self.assertTrue(_gone(local._process.pid))
    self.assertFalse(local.alive())
    self.assertFalse(phone.alive())
    # After the launch: the forward is removed, and the App stopped.
    after = self.tools.commands[
        self.tools.commands.index((*_ADB, 'forward', 'tcp:0', 'tcp:8769')) :
    ]
    self.assertIn((*_ADB, 'shell', 'am', 'force-stop', _APP), after)
    self.assertIn((*_ADB, 'forward', '--remove', f'tcp:{self.port}'), after)
    self.assertEqual(list((self.workspace / 'runners').iterdir()), [])

  def _assert_a_signal_stops_every_runner(
      self, signum: int, status: int
  ) -> None:
    failed = []

    def launch_then_signal():
      try:
        self._launch()
      except BaseException as error:  # pylint: disable=broad-exception-caught
        failed.append(error)
      finally:
        os.kill(os.getpid(), signum)

    self._serve_then(launch_then_signal)
    with self.assertRaises(SystemExit) as ended:
      self._main()
    if failed:
      raise failed[0]
    self.assertEqual(ended.exception.code, status)
    self._assert_all_stopped()

  def test_ctrl_c_stops_every_runner(self):
    """SIGINT (Ctrl-C): every runner stops, and the command exits 130."""
    self._assert_a_signal_stops_every_runner(signal.SIGINT, 130)

  def test_sigterm_stops_every_runner(self):
    """SIGTERM: every runner stops, and the command exits 143."""
    self._assert_a_signal_stops_every_runner(signal.SIGTERM, 143)

  def test_sighup_stops_every_runner(self):
    """SIGHUP (the terminal closed): as SIGTERM."""
    self._assert_a_signal_stops_every_runner(signal.SIGHUP, 143)

  def test_an_exception_as_it_starts_stops_every_runner(self):
    """An unexpected exception still stops every runner, and is raised."""

    def state(server):
      del server  # Unused.
      self._launch()
      raise RuntimeError('Something unexpected.')

    self.enter_context(mock.patch.object(server_lib.Server, 'state', state))
    with self.assertRaisesRegex(RuntimeError, 'Something unexpected.'):
      self._main()
    self._assert_all_stopped()

  def test_the_next_start_stops_what_a_killed_server_recorded(self):
    """A server killed outright: the next one stops what it recorded.

    Only that: a process it did not record, or one whose pid was given
    to another since, is left alone.
    """
    recorded, unrecorded, reused = _orphan(), _orphan(), _orphan()
    records = self.workspace / 'runners'
    records.mkdir(parents=True)
    left = [
        {
            'kind': 'local',
            'pid': recorded,
            'started': tool.started(recorded),
        },
        {'kind': 'local', 'pid': reused, 'started': 'Thu Jan  1'},
        {
            'kind': 'android',
            'serial': 'R5CW',
            'pid': fake_tools.APP_PID,
            'port': self.port,
        },
    ]
    for number, fields in enumerate(left):
      (records / f'{number}.json').write_text(json.dumps(fields))
    try:
      self._serve_then(lambda: os.kill(os.getpid(), signal.SIGTERM))
      with self.assertRaises(SystemExit):
        self._main()
      self.assertIsNone(tool.started(recorded))
      self.assertIsNotNone(tool.started(unrecorded))
      self.assertIsNotNone(tool.started(reused))
      self.assertIn(
          (*_ADB, 'forward', '--remove', f'tcp:{self.port}'),
          self.tools.commands,
      )
      self.assertIn(
          (*_ADB, 'shell', 'am', 'force-stop', _APP), self.tools.commands
      )
      self.assertEqual(list(records.iterdir()), [])
    finally:
      for pid in (recorded, unrecorded, reused):
        if tool.started(pid) is not None:
          os.kill(pid, signal.SIGKILL)


if __name__ == '__main__':
  test_main.main()
