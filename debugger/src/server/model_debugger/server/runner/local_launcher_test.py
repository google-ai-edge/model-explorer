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

"""Tests of the local launcher.

The command-line runner is the real one when it is built; a script stands in
for it otherwise, as the development build or a kept one.
"""

import logging as python_logging
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
from unittest import mock

from absl.testing import absltest
from model_debugger.protocol import runner_pb2
from model_debugger.server.runner import launch_records
from model_debugger.server.runner import launcher
from model_debugger.server.runner import local_launcher
from model_debugger.server.runner import runner_builds
from model_debugger.server.runner import runner_client
from model_debugger.server.runner import tool
from model_debugger.server.testing import fake_builds
from model_debugger.server.testing import fake_runner
from model_debugger.server.testing import real_runner
from model_debugger.server.testing import test_main
from model_debugger.server.util import json_file


def _development(
    runner: pathlib.Path,
    records: launch_records.LaunchRecords | None = None,
) -> local_launcher.LocalLauncher:
  """This computer, with `runner` as its development build alone."""
  return local_launcher.LocalLauncher(
      runner_builds.RunnerBuilds(None, runner), records
  )


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


def _launch(
    found: local_launcher.LocalLauncher,
    build: str = runner_builds.DEVELOPMENT,
) -> launcher.LaunchedRunner:
  """Launches a build on this computer; returns once the runner serves."""
  (device,) = found.devices()
  deadline = time.monotonic() + launcher.LAUNCH_SECONDS
  launched = found.launch(device, deadline, build)
  try:
    launched.wait_until_serving(deadline)
  except BaseException:
    launched.stop()
    raise
  return launched


def _gone(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return True
  return False


def _script(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
  script = tmp_path / 'runner'
  script.write_text(f'#!{sys.executable}\nimport sys, time\n{body}\n')
  script.chmod(0o755)
  return script


class LocalLauncherTest(absltest.TestCase):
  """Tests of the local launcher."""

  def setUp(self):
    """Gives each test an empty directory of its own."""
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)

  @absltest.skipUnless(real_runner.BUILT, 'the runner is not built')
  def test_the_real_runner_launches_serves_and_stops(self):
    """The built runner starts, serves clients over idle time, and stops."""
    found = real_runner.launcher()
    self.assertEqual(
        found.devices(),
        [
            launcher.Device(
                launcher.LOCAL_DEVICE_ID,
                'This computer',
                'macOS',
                'local',
                True,
                chooses_build=True,
            )
        ],
    )

    launched = _launch(found)
    pid = launched._process.pid
    self.assertTrue(launched.alive())
    client = runner_client.RunnerClient.connect(launched)
    self.assertEqual(
        client.info.protocol_version, runner_client.PROTOCOL_VERSION
    )
    # The runner serves on after an idle while, and serves a second client.
    time.sleep(2)
    second = runner_client.RunnerClient.connect(launched)
    self.assertEqual(second.info, client.info)
    second.close()
    client.close()

    launched.stop()
    launched.stop()
    self.assertFalse(launched.alive())
    self.assertTrue(_gone(pid))

  @absltest.skipUnless(real_runner.BUILT, 'the runner is not built')
  def test_two_real_runners_on_this_computer_serve_a_client_each(self):
    """Ref and target on this computer: two processes, a client each."""
    found = real_runner.launcher()
    ref, target = _launch(found), _launch(found)
    try:
      self.assertNotEqual(ref.address, target.address)
      clients = [runner_client.RunnerClient.connect(r) for r in (ref, target)]
      for client in clients:
        self.assertEqual(
            client.info.protocol_version, runner_client.PROTOCOL_VERSION
        )
        client.close()
    finally:
      ref.stop()
      target.stop()

  def test_a_local_runner_is_reached_at_the_port_it_prints(self):
    """A local runner is reached at the port it prints, and stopped."""
    # A stand-in for the binary: prints the fake runner's port, then lives.
    with fake_runner.FakeRunner(fake_runner.no_reply) as runner:
      port = runner.address.rsplit(':', 1)[1]
      script = _script(
          self.tmp_path,
          'print("loading", flush=True)\n'
          f'print("runner: listening on port {port}", flush=True)\n'
          'time.sleep(60)',
      )
      launched = _launch(_development(script))
      self.assertEqual(launched.address, runner.address)
      pid = launched._process.pid
      # Stopped as asked: no failure.
      with self.assertNoLogs(level='ERROR'):
        launched.stop()
      self.assertTrue(_gone(pid))

  def test_a_runners_lines_keep_the_level_it_gave_them(self):
    """The prefixes of absl and TFLite say a line's level; else INFO."""
    level = local_launcher.line_level
    self.assertEqual(
        level('W1022 12:00:00.000000 1 loader.cc:3] Section not found'),
        python_logging.WARNING,
    )
    self.assertEqual(
        level('E0000 00:00:1792710632.940790 2 a.cc:1] Failed'),
        python_logging.ERROR,
    )
    self.assertEqual(
        level('F1022 12:00:00.000000 1 a.cc:1] Check failed'),
        python_logging.ERROR,
    )
    self.assertEqual(
        level('WARNING: [gpu_registry.cc:160] GPU accelerator'),
        python_logging.WARNING,
    )
    self.assertEqual(level('ERROR: failed'), python_logging.ERROR)
    self.assertEqual(
        level('INFO: Created TensorFlow Lite XNNPACK delegate'),
        python_logging.INFO,
    )
    self.assertEqual(level('runner: listening on port 1'), python_logging.INFO)
    self.assertEqual(level('Warm-up done'), python_logging.INFO)

  def test_a_local_runner_that_ends_unasked_says_how_and_its_last_lines(self):
    """Its exit is an ERROR line, with what it said last, whatever shows.

    Only its errors show here, as the command line shows them by default.
    """
    local_launcher.OUTPUT_LOG.setLevel(python_logging.ERROR)
    self.addCleanup(local_launcher.OUTPUT_LOG.setLevel, python_logging.NOTSET)
    with fake_runner.FakeRunner(fake_runner.no_reply) as runner:
      port = runner.address.rsplit(':', 1)[1]
      script = _script(
          self.tmp_path,
          f'print("runner: listening on port {port}", flush=True)\n'
          'for n in range(8):\n'
          '    print(f"line {n}", flush=True)\n'
          'print("F1022 12:00:00.000000 1 a.cc:1] Check failed",'
          ' flush=True)\n'
          'time.sleep(0.5)\n'
          'sys.exit(3)',
      )
      with self.assertLogs(level='INFO') as logs:
        launched = _launch(_development(script))
        pid = launched._process.pid
        deadline = time.monotonic() + 10
        while not any(
            'unasked' in record.getMessage() for record in logs.records
        ):
          self.assertLess(time.monotonic(), deadline)
          time.sleep(0.05)
      launched.stop()
    shown = [
        (record.levelname, record.getMessage())
        for record in logs.records
        if record.levelname == 'ERROR'
    ]
    self.assertEqual(
        shown,
        [
            (
                'ERROR',
                f'runner {pid}: F1022 12:00:00.000000 1 a.cc:1] Check failed',
            ),
            (
                'ERROR',
                (
                    f'runner {pid} at {runner.address} ended unasked (exit '
                    'status 3); its last lines:\nline 4\nline 5\nline 6\n'
                    'line 7\nF1022 12:00:00.000000 1 a.cc:1] Check failed'
                ),
            ),
        ],
    )

  def test_a_local_runner_that_exits_for_a_reason_has_it_as_last_words(
      self,
  ):
    """Its own last line, once it exited with a failure status."""
    with fake_runner.FakeRunner(fake_runner.no_reply) as runner:
      port = runner.address.rsplit(':', 1)[1]
      words = 'runner: no space left on device while writing a: full'
      script = _script(
          self.tmp_path,
          f'print("runner: listening on port {port}", flush=True)\n'
          'time.sleep(0.3)\n'
          f'print({words!r}, flush=True)\n'
          'sys.exit(1)',
      )
      launched = _launch(_development(script))
      launched._process.wait(10)
      self.assertEqual(launched.last_words(), words)
      launched.stop()

  def test_a_local_runner_killed_or_exiting_well_has_no_last_words(self):
    """A signal, a status of 0 or a line not its own says nothing."""
    with fake_runner.FakeRunner(fake_runner.no_reply) as runner:
      port = runner.address.rsplit(':', 1)[1]
      for ending in (
          'print("runner: bye", flush=True)\nsys.exit(0)',
          'print("F1022 a.cc:1] Check failed", flush=True)\nsys.exit(3)',
          'import os, signal\nos.kill(os.getpid(), signal.SIGKILL)',
      ):
        script = _script(
            self.tmp_path,
            f'print("runner: listening on port {port}", flush=True)\n'
            f'time.sleep(0.3)\n{ending}',
        )
        launched = _launch(_development(script))
        launched._process.wait(10)
        self.assertEqual(launched.last_words(), '', ending)
        launched.stop()

  def test_a_local_runner_that_exits_says_why(self):
    """A local runner exiting before listening fails with what it said."""
    script = _script(
        self.tmp_path,
        'print("runner: Cannot listen.", flush=True)\nsys.exit(1)',
    )
    with self.assertRaisesRegex(RuntimeError, 'Cannot listen.'):
      _launch(_development(script))

  def test_a_local_runner_that_never_listens_is_stopped(self):
    """A runner that never says its port is stopped at the deadline."""
    self.enter_context(mock.patch.object(launcher, 'LAUNCH_SECONDS', 1))
    pid_file = self.tmp_path / 'pid'
    script = _script(
        self.tmp_path,
        f'import os\n'
        f'open({str(pid_file)!r}, "w").write(str(os.getpid()))\n'
        f'time.sleep(60)',
    )
    with self.assertRaisesRegex(RuntimeError, 'did not say its port'):
        _launch(_development(script))
    self.assertTrue(_gone(int(pid_file.read_text())))

  def test_a_local_runner_is_recorded_by_its_pid_and_start(self):
    """Its record names its pid and start time, until it is stopped."""
    records = launch_records.LaunchRecords()
    directory = self.tmp_path / 'records'
    records.keep_in(directory)
    with fake_runner.FakeRunner(fake_runner.no_reply) as runner:
      port = runner.address.rsplit(':', 1)[1]
      script = _script(
          self.tmp_path,
          f'print("runner: listening on port {port}", flush=True)\n'
          'time.sleep(60)',
      )
      launched = _launch(_development(script, records))
      pid = launched._process.pid
      (path,) = directory.glob('*.json')
      self.assertEqual(
          json_file.read_json(path),
          {'kind': 'local', 'pid': pid, 'started': tool.started(pid)},
      )
      launched.stop()
      self.assertEqual(list(directory.iterdir()), [])
      self.assertTrue(_gone(pid))

  def test_stop_all_stops_a_runner_still_starting(self):
    """A runner that has not said its port yet is stopped, and fails."""
    records = launch_records.LaunchRecords()
    pid_file = self.tmp_path / 'pid'
    script = _script(
        self.tmp_path,
        'import os\n'
        f'open({str(pid_file)!r}, "w").write(str(os.getpid()))\n'
        'time.sleep(60)',
    )
    failed = []

    def launch():
      try:
        _launch(_development(script, records))
      except RuntimeError as error:
        failed.append(str(error))

    thread = threading.Thread(target=launch)
    thread.start()
    deadline = time.monotonic() + 10
    while not pid_file.exists() or not pid_file.read_text():
      self.assertLess(time.monotonic(), deadline)
      time.sleep(0.05)
    records.stop_all()
    thread.join(10)
    self.assertLen(failed, 1, failed)
    self.assertTrue(_gone(int(pid_file.read_text())))
    with self.assertRaisesRegex(RuntimeError, 'stopping'):
      _launch(_development(script, records))

  def test_a_leftover_runner_is_stopped_while_it_is_the_same(self):
    """A recorded process goes only while it started when recorded."""
    found = _development(self.tmp_path / 'runner')
    pid = _orphan()
    try:
      started = tool.started(pid)
      self.assertTrue(started)
      found.stop_leftover(
          {'kind': 'local', 'pid': pid, 'started': 'Thu Jan  1'}
      )
      self.assertEqual(tool.started(pid), started)
      found.stop_leftover({'kind': 'local', 'pid': pid, 'started': started})
      self.assertIsNone(tool.started(pid))
    finally:
      if tool.started(pid) is not None:
        os.kill(pid, signal.SIGKILL)

  def test_the_kept_build_chosen_is_the_one_launched(self):
    """Each build runs its own runner, from its own directory."""
    kept = self.tmp_path / 'kept'
    with fake_runner.FakeRunner(fake_runner.no_reply) as runner:
      port = runner.address.rsplit(':', 1)[1]
      for commit in (fake_builds.OLD, fake_builds.NEW):
        fake_builds.keep(
            kept,
            commit,
            runner=(
                f'#!{sys.executable}\n'
                'import os, pathlib, time\n'
                'pathlib.Path(os.environ["RAN"]).write_text(__file__)\n'
                f'print("runner: listening on port {port}", flush=True)\n'
                'time.sleep(60)\n'
            ),
        )
      ran = self.tmp_path / 'ran'
      self.enter_context(mock.patch.dict(os.environ, RAN=str(ran)))
      found = local_launcher.LocalLauncher(
          runner_builds.RunnerBuilds(kept, None)
      )
      for commit in (fake_builds.NEW, fake_builds.OLD):
        launched = _launch(found, commit)
        launched.stop()
        self.assertEqual(ran.read_text(), str(kept / commit / 'runner'))

  def test_a_build_not_kept_or_changed_is_refused_and_not_run(self):
    """No longer kept, or a file changed since: refused, nothing runs."""
    kept = self.tmp_path / 'kept'
    ran = self.tmp_path / 'ran'
    directory = fake_builds.keep(
        kept,
        fake_builds.OLD,
        runner=f'#!/bin/sh\ntouch {ran}\nexit 1\n',
    )
    found = local_launcher.LocalLauncher(runner_builds.RunnerBuilds(kept, None))
    with self.assertRaisesRegex(RuntimeError, 'no longer kept'):
      _launch(found, fake_builds.NEW)
    library = directory / 'liblitert-lm.dylib'
    library.write_bytes(library.read_bytes() + b'!')
    with self.assertRaisesRegex(
        RuntimeError, 'liblitert-lm.dylib does not match the SHA-256'
    ):
      _launch(found, fake_builds.OLD)
    with self.assertRaisesRegex(RuntimeError, 'No runner version'):
      _launch(found, None)
    self.assertFalse(ran.exists())

  @absltest.skipUnless(
      real_runner.BUILT
      and (real_runner.RUNNER.parent / 'build.json').is_file(),
      'the runner is not a kept build',
  )
  def test_a_real_kept_build_launches_and_reports_its_commit(self):
    """A kept build is checked, launched from its directory, and named."""
    directory = real_runner.RUNNER.parent
    found = local_launcher.LocalLauncher(
        runner_builds.RunnerBuilds(directory.parent, None)
    )
    launched = _launch(found, directory.name)
    try:
      client = runner_client.RunnerClient.connect(launched)
      self.assertEqual(client.info.runtime.version, directory.name)
      client.close()
    finally:
      launched.stop()

  @absltest.skipUnless(real_runner.BUILT, 'the runner is not built')
  def test_a_real_runner_killed_mid_upload_fails_the_start(self):
    """A runner dying mid-upload fails the start: its connection closed."""
    launched = _launch(real_runner.launcher())
    model = self.tmp_path / 'model.litertlm'
    model.write_bytes(os.urandom(64 << 20))
    request = runner_pb2.LoadRequest(
        sha256='0' * 64, device=runner_pb2.CPU, context_length=128
    )
    try:
      client = runner_client.RunnerClient.connect(launched)

      def kill(sent, total):
        del total  # Unused.
        if sent >= 8 << 20:
          os.kill(launched._process.pid, signal.SIGKILL)

      with self.assertRaisesRegex(runner_client.Error, 'connection to the'):
        client.load(request, model, threading.Lock(), kill)
      client.close()
    finally:
      launched.stop()


if __name__ == '__main__':
  test_main.main()
