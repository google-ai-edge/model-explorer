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

"""Tests of the launchers.

The command-line runner is the real one when it is built; adb and xcrun are
stubbed, as no device or simulator can be assumed.
"""

import json
import os
import pathlib
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time

import fake_runner
import grpc
from model_debugger_server import errors, launchers, main, runner_client
import pytest

# RUNNER_BINARY, else the runner Bazel built last; None when neither is.
_RUNNER = (
    pathlib.Path(os.environ['RUNNER_BINARY'])
    if os.environ.get('RUNNER_BINARY')
    else main.built_runner()
)
_BUILT = _RUNNER is not None and _RUNNER.is_file()
_APP = launchers.APP_ID


def _no_reply(turn, prompt):
  raise AssertionError('No Send is expected.')


def _gone(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return True
  return False


@pytest.mark.skipif(not _BUILT, reason='the runner is not built')
def test_the_real_runner_launches_serves_one_client_and_stops():
  """The built runner starts, serves one client over idle time, and stops."""
  found = launchers.Launchers(_RUNNER)
  local = [d for d in found.devices() if d.id == 'local']
  expected_platform = launchers._PLATFORMS.get(
      platform.system(), platform.system()
  )
  assert local == [
      launchers.Device(
          'local', 'This computer', expected_platform, 'local', True, True
      )
  ]

  launched = found.launch('local')
  try:
    pid = launched._process.pid
    assert launched.alive()
    client = runner_client.RunnerClient(launched.address)
    info = client.get_info()
    assert info.protocol_version == runner_client.PROTOCOL_VERSION
    # The connection survives an idle while; a second client is refused.
    time.sleep(2)
    assert client.get_info().runtime.version == info.runtime.version
    other = runner_client.RunnerClient(launched.address)
    with pytest.raises(runner_client.RunnerError) as refused:
      other.get_info()
    assert refused.value.code == grpc.StatusCode.FAILED_PRECONDITION
    other.close()
    client.close()
  finally:
    launched.stop()
  launched.stop()
  assert not launched.alive()
  assert _gone(pid)


@pytest.mark.skipif(not _BUILT, reason='the runner is not built')
def test_two_real_runners_on_this_computer_serve_a_client_each():
  """Ref and target on this computer: two processes, a client each."""
  found = launchers.Launchers(_RUNNER)
  ref, target = found.launch('local'), found.launch('local')
  try:
    assert ref.address != target.address
    clients = [runner_client.RunnerClient(r.address) for r in (ref, target)]
    for _ in range(2):
      for client in clients:
        assert client.get_info().protocol_version == '1.0.0'
    for client in clients:
      client.close()
  finally:
    ref.stop()
    target.stop()


def _script(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
  path = tmp_path / 'runner'
  path.write_text(f'#!{sys.executable}\nimport sys, time\n{body}\n')
  path.chmod(0o755)
  return path


def test_a_local_runner_is_reached_at_the_port_it_prints(tmp_path):
  """A local runner is reached at the port it prints, and stopped."""
  # A stand-in for the binary: prints the fake runner's port, then lives.
  with fake_runner.FakeRunner(_no_reply) as runner:
    port = runner.address.rsplit(':', 1)[1]
    script = _script(
        tmp_path,
        'print("loading", flush=True)\n'
        f'print("runner: listening on port {port}", flush=True)\n'
        'time.sleep(60)',
    )
    launched = launchers.Launchers(script).launch('local')
    assert launched.address == runner.address
    pid = launched._process.pid
    launched.stop()
    assert _gone(pid)


def test_a_local_runner_that_exits_says_why(tmp_path):
  """A local runner that exits before listening fails with what it said."""
  script = _script(
      tmp_path, 'print("runner: Cannot listen.", flush=True)\nsys.exit(1)'
  )
  with pytest.raises(RuntimeError, match='Cannot listen.'):
    launchers.Launchers(script).launch('local')


def test_a_local_runner_that_never_listens_is_stopped(tmp_path, monkeypatch):
  """A local runner that never says its port is stopped at the deadline."""
  monkeypatch.setattr(launchers, '_LAUNCH_SECONDS', 1)
  pid_file = tmp_path / 'pid'
  script = _script(
      tmp_path,
      'import os\n'
      f'open({str(pid_file)!r}, "w").write(str(os.getpid()))\n'
      'time.sleep(60)',
  )
  with pytest.raises(RuntimeError, match='did not say its port'):
    launchers.Launchers(script).launch('local')
  assert _gone(int(pid_file.read_text()))


def test_no_binary_no_local_device(monkeypatch, tmp_path):
  """Without the binary and the tools there is no device."""
  monkeypatch.setattr(launchers, '_find_adb', lambda: None)
  monkeypatch.setattr(launchers.shutil, 'which', lambda tool: None)
  assert launchers.Launchers(None).devices() == []
  assert launchers.Launchers(tmp_path / 'absent').devices() == []
  with pytest.raises(errors.NotFound):
    launchers.Launchers(None).launch('local')


def test_serving_needs_a_grpc_server():
  """Only a gRPC server counts as serving, and probing claims nothing."""
  with fake_runner.FakeRunner(_no_reply) as runner:
    assert launchers._serving(runner.address)
    # The probe claimed nothing: the first RPC's peer is still admitted.
    assert runner.peers == []
  # A listener that accepts and closes, as an adb forward does with no App.
  with socket.socket() as listener:
    listener.bind(('127.0.0.1', 0))
    listener.listen()
    address = f'127.0.0.1:{listener.getsockname()[1]}'

    def accept_and_close():
      connection, _ = listener.accept()
      connection.close()

    threading.Thread(target=accept_and_close, daemon=True).start()
    assert not launchers._serving(address)
  assert not launchers._serving(address)


class _Tools:
  """A stand-in for adb and xcrun: canned output per command, recorded."""

  def __init__(self, outputs: dict[tuple[str, ...], str]) -> None:
    self.outputs = outputs
    self.commands: list[tuple[str, ...]] = []

  def __call__(self, args: list[str]) -> str:
    command = tuple(args)
    self.commands.append(command)
    output = self.outputs.get(command)
    if output is None:
      raise RuntimeError(f'{" ".join(args)} failed with status 1: no')
    return output


_ADB = '/sdk/platform-tools/adb'


def _android(monkeypatch, outputs) -> _Tools:
  tools = _Tools(outputs)
  monkeypatch.setattr(launchers, '_run', tools)
  monkeypatch.setattr(launchers, '_find_adb', lambda: _ADB)
  monkeypatch.setattr(launchers.shutil, 'which', lambda tool: None)
  return tools


def _android_outputs(port: int) -> dict[tuple[str, ...], str]:
  return {
      (_ADB, 'devices'): (
          'List of devices attached\n'
          'R5CW\tdevice\n'
          'emulator-5554\tdevice\n'
          'ZY22\tunauthorized\n\n'
      ),
      (
          _ADB,
          '-s',
          'R5CW',
          'shell',
          'pm',
          'path',
          _APP,
      ): 'package:/a/base.apk\n',
      (
          _ADB,
          '-s',
          'R5CW',
          'shell',
          'getprop',
          'ro.product.model',
      ): 'SM-S928B\n',
      (_ADB, '-s', 'R5CW', 'shell', 'am', 'force-stop', _APP): '',
      (
          _ADB,
          '-s',
          'R5CW',
          'shell',
          'am',
          'start',
          '-W',
          '-n',
          f'{_APP}/.RunnerActivity',
      ): 'Starting: Intent { cmp=... }\nStatus: ok\n',
      (_ADB, '-s', 'R5CW', 'forward', 'tcp:0', 'tcp:8769'): f'{port}\n',
      (_ADB, '-s', 'R5CW', 'forward', '--remove', f'tcp:{port}'): '',
      (_ADB, '-s', 'R5CW', 'shell', 'pidof', _APP): '4242\n',
  }


def test_android_devices_are_those_with_the_app(monkeypatch):
  """Android devices are the authorized ones with the App installed."""
  tools = _android(monkeypatch, _android_outputs(1))
  assert launchers.Launchers(None).devices() == [
      launchers.Device(
          'android:R5CW', 'SM-S928B', 'Android', 'adb', False, True
      )
  ]
  # emulator-5554 has no App (`pm path` fails); ZY22 is not authorized.
  assert (_ADB, '-s', 'ZY22', 'shell', 'pm', 'path', _APP) not in tools.commands


def test_android_launch_starts_the_app_afresh_and_forwards(monkeypatch):
  """An Android launch restarts the App and forwards a free port."""
  with fake_runner.FakeRunner(_no_reply) as runner:
    port = int(runner.address.rsplit(':', 1)[1])
    tools = _android(monkeypatch, _android_outputs(port))
    found = launchers.Launchers(None)
    tools.commands.clear()
    launched = found.launch('android:R5CW')
    assert launched.address == runner.address
    adb = (_ADB, '-s', 'R5CW')
    launch_commands = [c for c in tools.commands if c[:3] == adb][-3:]
    assert launch_commands == [
        (*adb, 'shell', 'am', 'force-stop', _APP),
        (*adb, 'shell', 'am', 'start', '-W', '-n', f'{_APP}/.RunnerActivity'),
        (*adb, 'forward', 'tcp:0', 'tcp:8769'),
    ]
    assert launched.alive()

    tools.commands.clear()
    launched.stop()
    launched.stop()
    assert tools.commands == [
        (*adb, 'shell', 'am', 'force-stop', _APP),
        (*adb, 'forward', '--remove', f'tcp:{port}'),
    ]
    assert not launched.alive()


def test_android_launch_without_an_answer_cleans_up(monkeypatch):
  """An Android launch nobody answers stops the App and the forward."""
  monkeypatch.setattr(launchers, '_LAUNCH_SECONDS', 1)
  with socket.socket() as unused:
    unused.bind(('127.0.0.1', 0))
    port = unused.getsockname()[1]
  tools = _android(monkeypatch, _android_outputs(port))
  with pytest.raises(RuntimeError, match='No runner answered'):
    launchers.Launchers(None).launch('android:R5CW')
  assert tools.commands[-2:] == [
      (_ADB, '-s', 'R5CW', 'shell', 'am', 'force-stop', _APP),
      (_ADB, '-s', 'R5CW', 'forward', '--remove', f'tcp:{port}'),
  ]


_XCRUN = '/usr/bin/xcrun'
_UDID = '5B1F0C4E-3C47-4F77-9D63-2C7A2D1C0A11'
_OTHER = '9A9A9A9A-0000-4000-8000-000000000000'


def _simulator_outputs(
    pid: int, port: int, listener: int | None = None
) -> dict[tuple[str, ...], str]:
  """The tools' answers; `listener` listens on the port (the App's pid)."""
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
      (_XCRUN, 'simctl', 'get_app_container', _UDID, _APP): '/x/App.app\n',
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
          launchers._LSOF,
          '-nP',
          f'-iTCP:{port}',
          '-sTCP:LISTEN',
          '-t',
      ): f'{pid if listener is None else listener}\n',
  }


def _simulator(monkeypatch, outputs, port: int) -> _Tools:
  tools = _Tools(outputs)
  monkeypatch.setattr(launchers, '_run', tools)
  monkeypatch.setattr(launchers, '_find_adb', lambda: None)
  monkeypatch.setattr(launchers.shutil, 'which', lambda tool: _XCRUN)
  monkeypatch.setattr(launchers, '_free_port', lambda: port)
  return tools


def test_simulators_are_the_booted_ones_with_the_app(monkeypatch):
  """Simulators are the booted ones with the App installed."""
  _simulator(monkeypatch, _simulator_outputs(1, 1), 1)
  assert launchers.Launchers(None).devices() == [
      launchers.Device(
          f'ios-simulator:{_UDID}',
          'iPhone 17',
          'iOS Simulator',
          'simulator',
          False,
          False,
      )
  ]


def test_simulator_launch_listens_where_told_and_terminates(monkeypatch):
  """A simulator launch tells the App its port; stop terminates it."""
  # The App's process stands in as a child of this test.
  app = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
  try:
    with fake_runner.FakeRunner(_no_reply) as runner:
      port = int(runner.address.rsplit(':', 1)[1])
      tools = _simulator(monkeypatch, _simulator_outputs(app.pid, port), port)
      launched = launchers.Launchers(None).launch(f'ios-simulator:{_UDID}')
      assert launched.address == runner.address
      assert launched.alive()
      tools.commands.clear()
      launched.stop()
      assert tools.commands == [(_XCRUN, 'simctl', 'terminate', _UDID, _APP)]
      assert not launched.alive()
  finally:
    app.kill()
    app.wait()


def test_simulator_launch_refuses_another_listener_on_its_port(monkeypatch):
  """A runner of another process on the App's port is not the App."""
  monkeypatch.setattr(launchers, '_LAUNCH_SECONDS', 1)
  app = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
  try:
    with fake_runner.FakeRunner(_no_reply) as runner:
      port = int(runner.address.rsplit(':', 1)[1])
      outputs = _simulator_outputs(app.pid, port, listener=os.getpid())
      tools = _simulator(monkeypatch, outputs, port)
      with pytest.raises(RuntimeError, match='Another process listens'):
        launchers.Launchers(None).launch(f'ios-simulator:{_UDID}')
      assert tools.commands[-1] == (
          _XCRUN,
          'simctl',
          'terminate',
          _UDID,
          _APP,
      )
  finally:
    app.kill()
    app.wait()


@pytest.mark.skipif(
    shutil.which('lsof') is None, reason='lsof is not installed'
)
def test_lsof_names_the_process_that_listens():
  """A simulator runner listens only if its own process holds the port."""
  device = launchers.Device(
      f'ios-simulator:{_UDID}',
      'iPhone 17',
      'iOS Simulator',
      'simulator',
      False,
      False,
  )
  with socket.socket() as listener:
    listener.bind(('127.0.0.1', 0))
    listener.listen()
    address = '127.0.0.1:%d' % listener.getsockname()[1]
    mine = launchers._SimulatorRunner(
        device, address, _XCRUN, _UDID, os.getpid()
    )
    assert mine._listens()
    other = launchers._SimulatorRunner(device, address, _XCRUN, _UDID, 1)
    assert not other._listens()
  assert not mine._listens()
