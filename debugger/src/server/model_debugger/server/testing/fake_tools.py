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

"""A stand-in for adb, xcrun and lsof: canned output per command, recorded.

Usage:
  tools = FakeTools({('adb', 'devices'): 'List of devices attached'})
  self.enter_context(mock.patch.object(tool, 'run', tools))
"""

from model_debugger.server.runner import launcher

# The adb of android_outputs(), as find_adb() finds it.
ADB = '/sdk/platform-tools/adb'
# The pid android_outputs() gives the App on R5CW.
APP_PID = '4242'
# The version of the App on R5CW: LiteRT-LM 61dcfc77, the server's.
APP_VERSION = '61dcfc77225019e067580dceb3ea268483d1938c (2026-10-15)'


class FakeTools:
  """Answers `tool.run` from a table; a command not in it fails.

  Attributes:
      outputs: Each command's standard output, by its arguments.
      commands: Every command run, in order.
  """

  def __init__(self, outputs: dict[tuple[str, ...], str]) -> None:
    """Answers each command in `outputs` with its output."""
    self.outputs = outputs
    self.commands: list[tuple[str, ...]] = []

  def __call__(self, args: list[str]) -> str:
    """Runs one command as `tool.run` does: its output, or RuntimeError."""
    command = tuple(args)
    self.commands.append(command)
    output = self.outputs.get(command)
    if output is None:
      raise RuntimeError(f'{" ".join(args)} failed with status 1: no')
    return output


def android_outputs(port: int) -> dict[tuple[str, ...], str]:
  """What adb answers: R5CW has the App, which a launch forwards `port` to.

  The App on R5CW is of APP_VERSION; emulator-5554 has no App; ZY22 is not
  authorized. Once launched, adb lists the forward from `port` and pidof
  gives the App APP_PID.

  Args:
    port: The port forwarded to the App on the device.

  Returns:
    A mapping of adb command tuples to their simulated stdout outputs.
  """
  app = launcher.APP_ID
  r5cw = (ADB, '-s', 'R5CW')
  emulator = (ADB, '-s', 'emulator-5554')
  return {
      (ADB, 'devices'): (
          'List of devices attached\n'
          'R5CW\tdevice\n'
          'emulator-5554\tdevice\n'
          'ZY22\tunauthorized\n\n'
      ),
      (*r5cw, 'shell', 'dumpsys', 'package', app): (
          f'Packages:\n  Package [{app}] (5a3c9e1):\n    versionCode=1 '
          f'minSdk=31 targetSdk=35\n    versionName={APP_VERSION}\n'
      ),
      (*emulator, 'shell', 'dumpsys', 'package', app): 'Dexopt state:\n',
      (*emulator, 'shell', 'getprop', 'ro.product.model'): 'sdk_gphone64\n',
      (*r5cw, 'shell', 'getprop', 'ro.product.model'): 'SM-S928B\n',
      (*r5cw, 'shell', 'am', 'force-stop', app): '',
      (
          *r5cw,
          'shell',
          'am',
          'start',
          '-W',
          '-n',
          f'{app}/.RunnerActivity',
      ): 'Starting: Intent { cmp=... }\nStatus: ok\n',
      (*r5cw, 'forward', 'tcp:0', 'tcp:8769'): f'{port}\n',
      (*r5cw, 'forward', '--remove', f'tcp:{port}'): '',
      (ADB, 'forward', '--list'): f'R5CW tcp:{port} tcp:8769\n',
      (*r5cw, 'shell', 'pidof', app): f'{APP_PID}\n',
  }
