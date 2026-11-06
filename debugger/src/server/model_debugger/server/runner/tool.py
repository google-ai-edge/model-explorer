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

"""Runs the tools that reach devices and processes: adb, xcrun, lsof, ps."""

import os
import shutil
import subprocess

from absl import logging

# How long one adb or xcrun command may take.
_TOOL_SECONDS = 30

# Says when a process started; macOS and Linux have it in /bin.
_PS = shutil.which('ps') or '/bin/ps'


def run_logged(args: list[str]) -> None:
  """Runs a tool that stops something; a failure is logged, not raised.

  Stopping must not fail its caller: a device that went away has nothing
  left to stop.

  Args:
    args: Command arguments to run.
  """
  try:
    run(args)
  except RuntimeError as error:
    logging.warning('%s', error)


def run(args: list[str]) -> str:
  """Runs a tool to completion and returns its standard output.

  Args:
    args: Command arguments to run.

  Returns:
    Standard output of the tool.

  Raises:
    RuntimeError: The tool failed; the message holds what it said.
  """
  try:
    # close_fds=False: spawned, not forked, as the runner is
    # (local_launcher).
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=_TOOL_SECONDS,
        close_fds=False,
        check=False,
    )
  except subprocess.TimeoutExpired:
    raise RuntimeError(
        f'{" ".join(args)} did not finish in {_TOOL_SECONDS} s.'
    ) from None
  if result.returncode != 0:
    said = (result.stderr or result.stdout).strip()
    raise RuntimeError(
        f'{" ".join(args)} failed with status {result.returncode}: {said}'
    )
  return result.stdout


def started(pid: int) -> str | None:
  """When a process of this computer started, as ps tells it to the second.

  With its pid, what tells a process apart from a later one given the same
  pid: a launch records both, and a later server stops the process only if
  both still match.

  Args:
    pid: Process ID to inspect.

  Returns:
    ps's start time of the process, or None if there is no such process
    (or ps cannot say).
  """
  try:
    # C: the same words whatever the server's locale.
    result = subprocess.run(
        [_PS, '-o', 'lstart=', '-p', str(pid)],
        capture_output=True,
        text=True,
        timeout=_TOOL_SECONDS,
        close_fds=False,
        env={**os.environ, 'LC_ALL': 'C'},
        check=False,
    )
  except (OSError, subprocess.TimeoutExpired):
    return None
  return (result.returncode == 0 and result.stdout.strip()) or None
