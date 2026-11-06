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

"""This computer as a device: the command-line runner, a child process.

Each start runs the runner build its Session chose (runner_builds): a kept
build from its own directory, once each of its files is checked, or the
development build.

What a runner prints goes to OUTPUT_LOG, a logger of its own, each line at
the level the runner gave it: a line its runtime marks as a warning or an
error (absl's `W1022 ...`, TFLite's `WARNING: ...`) at WARNING or ERROR, any
other at INFO. The command line sets how much of it shows (--runner_log). A
runner that fails unasked, once it listened, is logged at ERROR with its
exit status and, unless OUTPUT_LOG shows every line already, its last lines.
"""

import collections
import logging as python_logging
import os
import pathlib
import platform
import re
import signal
import subprocess
import threading
import time
import types
from typing import Any

from absl import logging
from model_debugger.server.runner import launch_records
from model_debugger.server.runner import launcher
from model_debugger.server.runner import runner_builds
from model_debugger.server.runner import tool

# How long a stopped process has after SIGTERM before it is killed.
_STOP_SECONDS = 5

# What this computer's operating system is called in a Device.
_PLATFORMS = types.MappingProxyType({'Darwin': 'macOS'})

# What the command-line runner prints once it listens.
_LISTENING = re.compile(r'runner: listening on port (\d+)')

# The runners' output, by line; a logger of its own, so the command line sets
# how much of it shows.
OUTPUT_LOG = python_logging.getLogger('model_debugger.runner')

# How many of a runner's last lines its unasked exit shows.
_LAST_LINES = 5

# How long last_words() waits for a runner that is ending to have ended: its
# connection closes as it exits, and its output ends a moment later.
_ENDING_SECONDS = 1.0

# The level a runner gave a line: absl's prefix (`W1022 12:00:00.000000 ...`)
# or TFLite's (`WARNING: ...`).
_ABSL_PREFIX = re.compile(r'([IWEF])\d{4} \d\d:\d\d:\d\d')
_TFLITE_PREFIX = re.compile(r'(INFO|WARNING|ERROR|FATAL): ')
_LEVELS = {
    'I': python_logging.INFO,
    'W': python_logging.WARNING,
    'E': python_logging.ERROR,
    'F': python_logging.ERROR,
}


def line_level(line: str) -> int:
  """The level a runner's line is logged at: the one the runner gave it.

  Args:
      line: One line the runner printed.

  Returns:
      WARNING or ERROR for a line its runtime marks so, INFO for any other.
  """
  match = _ABSL_PREFIX.match(line) or _TFLITE_PREFIX.match(line)
  return _LEVELS[match.group(1)[0]] if match else python_logging.INFO


class LocalLauncher(launcher.Launcher):
  """Runs the command-line runner on this computer, on a port of its own.

  Each runner process is recorded by its pid and start time.
  """

  KIND = 'local'

  def __init__(
      self,
      builds: runner_builds.RunnerBuilds,
      records: launch_records.LaunchRecords | None = None,
  ) -> None:
    """Runs the builds of `builds`; with none, it is no device.

    Args:
        builds: This computer's runner builds.
        records: Where its launches are recorded; None for its own.
    """
    self._builds = builds
    self._records = records or launch_records.LaunchRecords()

  def devices(self) -> list[launcher.Device]:
    """Returns this computer, while it has a build, launchable or not."""
    if not self._builds.list():
      return []
    system = platform.system()
    return [
        launcher.Device(
            launcher.LOCAL_DEVICE_ID,
            'This computer',
            _PLATFORMS.get(system, system),
            'local',
            True,
            chooses_build=True,
        )
    ]

  def builds(self, device: launcher.Device) -> list[runner_builds.Build]:
    """Returns this computer's builds, read anew."""
    del device  # Its only device.
    return self._builds.list()

  def launch(
      self,
      device: launcher.Device,
      deadline: float,
      build: str | None,
      model_dir: pathlib.Path | None = None,
  ) -> launcher.LaunchedRunner:
    """Starts a build's runner; returns once it said its port.

    The PyTorch runner reads its model in place: it is given the model's
    directory.

    Args:
      device: The target device to launch the runner on.
      deadline: The monotonic time deadline by which the runner must be ready.
      build: The build identifier or None.
      model_dir: Optional model directory for runners that need it.

    Returns:
      The launched runner instance.

    Raises:
      RuntimeError: The build cannot be launched (see
        RunnerBuilds.runner), or its runner did not say its port.
    """
    command = [*self._builds.command(build), '--listen', '127.0.0.1:0']
    if build == runner_builds.PYTORCH:
      if model_dir is None:
        raise RuntimeError('The PyTorch runner needs its model.')
      command += ['--model_dir', str(model_dir)]
    asked_to_exit = threading.Event()
    with self._records.starting():
      process = subprocess.Popen(
          command,
          stdin=subprocess.DEVNULL,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          text=True,
          errors='replace',
          # Lets Python spawn the runner (posix_spawn) instead of forking
          # this process, whose running gRPC threads would otherwise go
          # through gRPC's fork handlers in the child. Python and gRPC
          # open every descriptor close-on-exec, so the runner inherits
          # none of them.
          close_fds=False,
      )

      def stop() -> None:
        asked_to_exit.set()
        _terminate(process)

      record = self._records.add(
          {
              'kind': self.KIND,
              'pid': process.pid,
              'started': tool.started(process.pid),
          },
          stop,
      )
    output = _RunnerOutput(process, asked_to_exit)
    if not output.ready.wait(max(0, deadline - time.monotonic())):
      record.stop()
      raise RuntimeError(
          f'The runner did not say its port within {launcher.LAUNCH_SECONDS} s.'
      )
    if output.port is None:
      record.stop()
      raise RuntimeError(
          'The runner exited before listening: ' + '\n'.join(output.lines)
      )
    return _LocalRunner(
        device, f'127.0.0.1:{output.port}', process, record, output
    )

  def stop_leftover(self, fields: dict[str, Any]) -> None:
    """Stops a runner process a previous server left, if it still runs.

    The process with the pid recorded is stopped only while it started
    when the record says: a later process given the same pid is not it.

    Args:
      fields: Dictionary containing runner record fields ('pid', 'started').
    """
    pid, when = fields.get('pid'), fields.get('started')
    if not isinstance(pid, int) or not when or tool.started(pid) != when:
      return
    logging.info('Stopping runner %d, left by a previous server', pid)
    _stop_process(pid, when)


class _LocalRunner(launcher.LaunchedRunner):
  """The command-line runner, a child process of this server."""

  def __init__(
      self,
      device: launcher.Device,
      address: str,
      process: subprocess.Popen[Any],
      record: launch_records.Record,
      output: '_RunnerOutput',
  ) -> None:
    super().__init__(device, address)
    self._process = process
    self._record = record
    self._output = output

  def last_words(self) -> str:
    """The runner's last line, once it ended unasked with a failure."""
    if not self._output.ended.wait(_ENDING_SECONDS):
      return ''
    return self._output.last_words

  def _sees_process(self) -> bool:
    return True

  def _alive(self) -> bool:
    return self._process.poll() is None

  def _stop(self) -> None:
    self._record.stop()


def _terminate(process: subprocess.Popen[Any]) -> None:
  """Stops a child: SIGTERM, then SIGKILL if it is still there after 5 s."""
  process.terminate()
  try:
    process.wait(_STOP_SECONDS)
  except subprocess.TimeoutExpired:
    process.kill()
    process.wait()


def _stop_process(pid: int, started: str) -> None:
  """Stops a process not this server's child, as _terminate() does a child.

  Signals it only while it is still the process that started at `started`.

  Args:
    pid: Process ID of the target process.
    started: The start time string recorded when the process was started.
  """
  for signum in (signal.SIGTERM, signal.SIGKILL):
    try:
      os.kill(pid, signum)
    except ProcessLookupError:
      return
    deadline = time.monotonic() + _STOP_SECONDS
    while time.monotonic() < deadline:
      if tool.started(pid) != started:
        return
      time.sleep(0.1)


class _RunnerOutput:
  """Reads the runner's output: its port, and every line into OUTPUT_LOG.

  Attributes:
      port: The port it listens on, once it said it.
      lines: What it said before its port.
      ready: Set once the port is known, or never will be.
      asked_to_exit: Set once this server stops it: its exit is no failure.
      ended: Set once its output ended and its process exited.
      last_words: Its last line, once it exited unasked with a failure status
        after it listened, if the line is its own (`runner: ...`); else ''.
  """

  def __init__(
      self, process: subprocess.Popen[Any], asked_to_exit: threading.Event
  ) -> None:
    self.port: int | None = None
    self.lines: list[str] = []
    self.ready = threading.Event()
    self.asked_to_exit = asked_to_exit
    self.ended = threading.Event()
    self.last_words = ''
    self._process = process
    self._last = collections.deque(maxlen=_LAST_LINES)
    threading.Thread(
        target=self._read, name=f'runner-{process.pid}-output', daemon=True
    ).start()

  def _read(self) -> None:
    """Reads stdout lines from the runner process and detects listening port."""
    pid = self._process.pid
    for line in self._process.stdout:
      line = line.rstrip('\n')
      OUTPUT_LOG.log(line_level(line), 'runner %d: %s', pid, line)
      self._last.append(line)
      if self.ready.is_set():
        continue
      match = _LISTENING.fullmatch(line)
      if match:
        self.port = int(match.group(1))
        self.ready.set()
      else:
        self.lines.append(line)
    self.ready.set()
    status = self._process.wait()
    # A runner that never listened fails its launch, which says why; one
    # that exits well, its connection closed, is stopped as asked.
    if self.port is None or self.asked_to_exit.is_set() or status == 0:
      self.ended.set()
      return
    # A runner that exits for a reason prints it last, as `runner: ...`;
    # one killed by a signal had no say.
    if status > 0 and self._last and self._last[-1].startswith('runner: '):
      self.last_words = self._last[-1]
    self.ended.set()
    how = (
        f'killed by signal {-status}' if status < 0 else f'exit status {status}'
    )
    last = ''
    if self._last and not OUTPUT_LOG.isEnabledFor(python_logging.INFO):
      last = '; its last lines:\n' + '\n'.join(self._last)
    logging.error(
        'runner %d at 127.0.0.1:%d ended unasked (%s)%s',
        pid,
        self.port,
        how,
        last,
    )
