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

"""Where runners can run, and starting and stopping one there.

A runner serves one client for its life, so every chat needs a fresh runner
process: the server launches each one itself. Four kinds of device:
- this computer: the command-line runner, on a port of its choosing;
- an Android device over adb: the App, reached through `adb forward`;
- a connected iOS device over devicectl: the App, on the CoreDevice tunnel;
- a booted iOS simulator: the App, on the Mac's own loopback.

This module is the only place that knows the platforms apart.
"""

import abc
import dataclasses
import json
import logging
import os
import pathlib
import platform
import re
import shutil
import socket
import subprocess
import threading
import time
import types

from model_debugger_server import errors

_LOG = logging.getLogger(__name__)

# The App's Android package and iOS bundle identifier.
APP_ID = 'com.google.ai.edge.modeldebugger.runner'

# The Android App's activity, which starts its runner.
_ACTIVITY = f'{APP_ID}/.RunnerActivity'

# The port the Android and iOS Apps listen on, on the device.
_DEVICE_PORT = 8769

# How long a launch may take until the runner accepts connections.
_LAUNCH_SECONDS = 30

# How long a stopped process has after SIGTERM before it is killed.
_STOP_SECONDS = 5

# How long one adb or xcrun command may take.
_TOOL_SECONDS = 30

# What this computer's operating system is called in a Device.
_PLATFORMS = types.MappingProxyType({'Darwin': 'macOS'})

# Lists the processes listening on a port; macOS has it in /usr/sbin.
_LSOF = shutil.which('lsof') or '/usr/sbin/lsof'

# What the command-line runner prints once it listens.
_LISTENING = re.compile(r'runner: listening on port (\d+)')


@dataclasses.dataclass(frozen=True)
class Device:
  """A place a runner can run.

  Attributes:
    id: 'local', 'android:<serial>', 'ios:<udid>' or 'ios-simulator:<udid>'.
    name: what the user knows the device by.
    platform: 'macOS', 'Android', 'iOS' or 'iOS Simulator'.
    transport: 'local', 'adb', 'devicectl' or 'simulator'.
    shared: whether it can run several runners at once (only 'local').
    gpu: whether a runner there can use a GPU its host reports; the iOS
      simulator has no Metal device the runtime can use.
  """

  id: str
  name: str
  platform: str
  transport: str
  shared: bool
  gpu: bool


class LaunchedRunner(abc.ABC):
  """A runner this server started, listening at `address`.

  Attributes:
    device: where it runs.
    address: 'host:port' to connect to.
  """

  def __init__(self, device: Device, address: str) -> None:
    """Records where the runner runs and listens."""
    self.device = device
    self.address = address
    self._stopped = False
    self._lock = threading.Lock()

  def alive(self) -> bool:
    """Whether the runner's process still runs."""
    return not self._stopped and self._alive()

  def _listens(self) -> bool:
    """Whether what answers at `address` is this runner, not another."""
    return True

  def stop(self) -> None:
    """Stops the runner. Idempotent."""
    with self._lock:
      if self._stopped:
        return
      self._stopped = True
      self._stop()

  @abc.abstractmethod
  def _alive(self) -> bool:
    ...

  @abc.abstractmethod
  def _stop(self) -> None:
    ...


class _LocalRunner(LaunchedRunner):
  """The command-line runner, a child process of this server."""

  def __init__(
      self, device: Device, address: str, process: subprocess.Popen
  ) -> None:
    super().__init__(device, address)
    self._process = process

  def _alive(self) -> bool:
    return self._process.poll() is None

  def _stop(self) -> None:
    _terminate(self._process)


class _AndroidRunner(LaunchedRunner):
  """The App on an Android device, reached through an adb forward."""

  def __init__(self, device: Device, adb: list[str], local_port: int) -> None:
    super().__init__(device, f'127.0.0.1:{local_port}')
    self._adb = adb  # adb -s <serial>
    self._local_port = local_port

  def _alive(self) -> bool:
    # pidof prints nothing, and fails, once the App's process is gone.
    try:
      return bool(_run([*self._adb, 'shell', 'pidof', APP_ID]).strip())
    except RuntimeError:
      return False

  def _stop(self) -> None:
    _stop_android_app(self._adb)
    _run_logged([*self._adb, 'forward', '--remove', f'tcp:{self._local_port}'])


class _IosRunner(LaunchedRunner):
  """The App on a connected iOS device, reached over the CoreDevice tunnel."""

  def __init__(
      self, device: Device, address: str, xcrun: str, udid: str, pid: int
  ) -> None:
    super().__init__(device, address)
    self._processes = [
        xcrun,
        'devicectl',
        'device',
        'info',
        'processes',
        '--device',
        udid,
        '--json-output',
        '-',
    ]
    self._terminate = [
        xcrun,
        'devicectl',
        'device',
        'process',
        'terminate',
        '--device',
        udid,
        '--pid',
        str(pid),
    ]
    self._pid = pid

  def _alive(self) -> bool:
    try:
      processes = _devicectl_json(_run(self._processes))['result'][
          'runningProcesses'
      ]
    except (RuntimeError, ValueError, KeyError):
      return False
    return any(p.get('processIdentifier') == self._pid for p in processes)

  def _stop(self) -> None:
    _run_logged(self._terminate)


class _SimulatorRunner(LaunchedRunner):
  """The App in a booted iOS simulator: a process of this Mac."""

  def __init__(
      self, device: Device, address: str, xcrun: str, udid: str, pid: int
  ) -> None:
    super().__init__(device, address)
    self._terminate = [xcrun, 'simctl', 'terminate', udid, APP_ID]
    self._pid = pid

  def _alive(self) -> bool:
    try:
      os.kill(self._pid, 0)
    except ProcessLookupError:
      return False
    return True

  def _listens(self) -> bool:
    # The port was free when chosen, but anything on this Mac may take it
    # before the App binds it; then the App's bind fails and it is not the
    # one that answers.
    port = self.address.rsplit(':', 1)[1]
    try:
      listing = _run([_LSOF, '-nP', f'-iTCP:{port}', '-sTCP:LISTEN', '-t'])
    except RuntimeError:
      return False  # lsof fails when nothing listens there
    return str(self._pid) in listing.split()

  def _stop(self) -> None:
    if self._alive():
      _run_logged(self._terminate)


def _terminate(process: subprocess.Popen) -> None:
  """Stops a child: SIGTERM, then SIGKILL if it is still there after 5 s."""
  process.terminate()
  try:
    process.wait(_STOP_SECONDS)
  except subprocess.TimeoutExpired:
    process.kill()
    process.wait()


def _stop_android_app(adb: list[str]) -> None:
  _run_logged([*adb, 'shell', 'am', 'force-stop', APP_ID])


def _run_logged(args: list[str]) -> None:
  """Runs a tool that stops something; a failure is logged, not raised.

  Stopping must not fail its caller: a device that went away has nothing
  left to stop.
  """
  try:
    _run(args)
  except RuntimeError as error:
    _LOG.warning('%s', error)


def _run(args: list[str]) -> str:
  """Runs a tool to completion and returns its standard output.

  Raises:
    RuntimeError: the tool failed; the message holds what it said.
  """
  try:
    # close_fds=False: spawned, not forked, as the runner is (_launch_local).
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=_TOOL_SECONDS,
        close_fds=False,
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


def _devicectl_json(output: str) -> dict:
  """Parses devicectl's JSON output, skipping any progress lines before it."""
  start = output.find('{')
  if start < 0:
    raise ValueError('No JSON object in devicectl output.')
  return json.loads(output[start:])


def _free_port() -> int:
  with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    return probe.getsockname()[1]


def _serving(address: str) -> bool:
  """Whether a gRPC server answers at `address`, without claiming it.

  A gRPC server sends its HTTP/2 settings as soon as it accepts a
  connection, before any RPC; a runner admits its client only at the first
  RPC. An adb forward accepts the connection itself and closes it when the
  device side does not, so accepting alone proves nothing.
  """
  host, port = address.rsplit(':', 1)
  host = host.strip('[]')
  try:
    with socket.create_connection((host, int(port)), timeout=1) as probe:
      probe.settimeout(1)
      return bool(probe.recv(1))
  except OSError:
    return False


def _wait_until_serving(launched: LaunchedRunner, deadline: float) -> None:
  address = launched.address
  while True:
    answered = _serving(address)
    if answered and launched._listens():
      return
    if not launched.alive():
      raise RuntimeError(f'The runner for {address} stopped while starting.')
    if time.monotonic() > deadline:
      if answered:
        raise RuntimeError(
            f'Another process listens at {address}, where the runner was to.'
        )
      raise RuntimeError(
          f'No runner answered at {address} within {_LAUNCH_SECONDS} s.'
      )
    time.sleep(0.2)


class Launchers:
  """The devices this computer can launch runners on."""

  def __init__(self, runner_binary: pathlib.Path | None) -> None:
    """Finds the tools; a missing one simply contributes no devices.

    Args:
      runner_binary: the command-line runner, or None for no local device.
    """
    self._runner_binary = runner_binary
    self._adb = _find_adb()
    self._xcrun = shutil.which('xcrun')

  def devices(self) -> list[Device]:
    """Returns every device a runner can be launched on now."""
    found = []
    if self._runner_binary is not None and self._runner_binary.is_file():
      system = platform.system()
      found.append(
          Device(
              'local',
              'This computer',
              _PLATFORMS.get(system, system),
              'local',
              True,
              True,
          )
      )
    if self._adb is not None:
      found += self._android_devices()
    if self._xcrun is not None:
      found += self._ios_devices()
      found += self._simulators()
    return found

  def launch(self, device_id: str) -> LaunchedRunner:
    """Starts a fresh runner on the device; returns once it accepts calls.

    Whatever runner of this server's App ran on the device before is stopped.

    Raises:
      NotFound: no such device now.
      RuntimeError: the runner did not start within 30 s, or a tool failed.
    """
    device = next((d for d in self.devices() if d.id == device_id), None)
    if device is None:
      raise errors.NotFound(f'No device {device_id} is available.')
    deadline = time.monotonic() + _LAUNCH_SECONDS
    if device.transport == 'local':
      launched = self._launch_local(device, deadline)
    elif device.transport == 'adb':
      launched = self._launch_android(device)
    elif device.transport == 'devicectl':
      launched = self._launch_ios(device)
    else:
      launched = self._launch_simulator(device)
    try:
      _wait_until_serving(launched, deadline)
    except BaseException:
      launched.stop()
      raise
    return launched

  def _launch_local(self, device: Device, deadline: float) -> _LocalRunner:
    process = subprocess.Popen(
        [str(self._runner_binary), '--listen', '127.0.0.1:0'],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors='replace',
        # Lets Python spawn the runner (posix_spawn) instead of forking this
        # process, whose running gRPC threads would otherwise go through
        # gRPC's fork handlers in the child. Python and gRPC open every
        # descriptor close-on-exec, so the runner inherits none of them.
        close_fds=False,
    )
    output = _RunnerOutput(process)
    if not output.ready.wait(max(0, deadline - time.monotonic())):
      _terminate(process)
      raise RuntimeError(
          f'The runner did not say its port within {_LAUNCH_SECONDS} s.'
      )
    if output.port is None:
      _terminate(process)
      raise RuntimeError(
          'The runner exited before listening: ' + '\n'.join(output.lines)
      )
    return _LocalRunner(device, f'127.0.0.1:{output.port}', process)

  def _launch_android(self, device: Device) -> _AndroidRunner:
    adb = [self._adb, '-s', device.id.removeprefix('android:')]
    # The App serves one client per process: a fresh process, never the old.
    _stop_android_app(adb)
    try:
      # -W: returns once the App's process runs, so alive() can be asked.
      _run([*adb, 'shell', 'am', 'start', '-W', '-n', _ACTIVITY])
      # tcp:0 picks a free local port, and adb prints it.
      port = int(_run([*adb, 'forward', 'tcp:0', f'tcp:{_DEVICE_PORT}']))
    except BaseException:
      _stop_android_app(adb)
      raise
    return _AndroidRunner(device, adb, port)

  def _launch_ios(self, device: Device) -> _IosRunner:
    udid = device.id.removeprefix('ios:')
    tunnel = self._ios_tunnel_address(udid)
    address = (
        f'[{tunnel}]:{_DEVICE_PORT}'
        if ':' in tunnel
        else f'{tunnel}:{_DEVICE_PORT}'
    )
    launched = _devicectl_json(
        _run([
            self._xcrun,
            'devicectl',
            'device',
            'process',
            'launch',
            '--terminate-existing',
            '--json-output',
            '-',
            '--device',
            udid,
            APP_ID,
            '--listen',
            address,
        ])
    )
    pid = int(launched['result']['process']['processIdentifier'])
    return _IosRunner(device, address, self._xcrun, udid, pid)

  def _launch_simulator(self, device: Device) -> _SimulatorRunner:
    udid = device.id.removeprefix('ios-simulator:')
    # The simulator shares this Mac's loopback: a port of the server's
    # choosing keeps the App off any other listener's, and the launch waits
    # for the App itself to listen there (_listens).
    address = f'127.0.0.1:{_free_port()}'
    output = _run([
        self._xcrun,
        'simctl',
        'launch',
        '--terminate-running-process',
        udid,
        APP_ID,
        '--listen',
        address,
    ])
    # simctl prints '<bundle id>: <pid>'.
    pid = int(output.strip().rsplit(':', 1)[1])
    return _SimulatorRunner(device, address, self._xcrun, udid, pid)

  def _android_devices(self) -> list[Device]:
    try:
      listing = _run([self._adb, 'devices'])
    except RuntimeError as error:
      _LOG.info('No Android devices: %s', error)
      return []
    found = []
    for line in listing.splitlines()[1:]:
      fields = line.split()
      if len(fields) != 2 or fields[1] != 'device':
        continue  # offline, unauthorized, or not a device line
      serial = fields[0]
      adb = [self._adb, '-s', serial]
      try:
        if not _run([*adb, 'shell', 'pm', 'path', APP_ID]).strip():
          continue
        name = _run([*adb, 'shell', 'getprop', 'ro.product.model']).strip()
      except RuntimeError:
        continue  # `pm path` fails when the App is not installed
      found.append(
          Device(
              f'android:{serial}', name or serial, 'Android', 'adb', False, True
          )
      )
    return found

  def _ios_devices(self) -> list[Device]:
    try:
      listing = _devicectl_json(
          _run([
              self._xcrun,
              'devicectl',
              'list',
              'devices',
              '--json-output',
              '-',
          ])
      )
    except (RuntimeError, ValueError) as error:
      _LOG.info('No iOS devices: %s', error)
      return []
    found = []
    for entry in listing.get('result', {}).get('devices', []):
      hardware = entry.get('hardwareProperties', {})
      connection = entry.get('connectionProperties', {})
      properties = entry.get('deviceProperties', {})
      udid = hardware.get('udid')
      if (
          hardware.get('platform') != 'iOS'
          or hardware.get('reality') != 'physical'
          or connection.get('pairingState') != 'paired'
          or properties.get('developerModeStatus') != 'enabled'
          or not connection.get('tunnelIPAddress')
          or not udid
      ):
        continue
      found.append(
          Device(
              f'ios:{udid}',
              properties.get('name') or udid,
              'iOS',
              'devicectl',
              False,
              True,
          )
      )
    return found

  def _ios_tunnel_address(self, udid: str) -> str:
    listing = _devicectl_json(
        _run([
            self._xcrun,
            'devicectl',
            'list',
            'devices',
            '--json-output',
            '-',
        ])
    )
    for entry in listing.get('result', {}).get('devices', []):
      if entry.get('hardwareProperties', {}).get('udid') == udid:
        address = entry.get('connectionProperties', {}).get('tunnelIPAddress')
        if address:
          return address
    raise RuntimeError(f'Device {udid} has no tunnel address.')

  def _simulators(self) -> list[Device]:
    try:
      listing = json.loads(
          _run([self._xcrun, 'simctl', 'list', 'devices', 'booted', '--json'])
      )
    except RuntimeError as error:
      _LOG.info('No iOS simulators: %s', error)
      return []
    found = []
    for simulators in listing['devices'].values():
      for simulator in simulators:
        if simulator['state'] != 'Booted':
          continue
        udid = simulator['udid']
        try:
          _run([self._xcrun, 'simctl', 'get_app_container', udid, APP_ID])
        except RuntimeError:
          continue  # the App is not installed there
        found.append(
            Device(
                f'ios-simulator:{udid}',
                simulator['name'],
                'iOS Simulator',
                'simulator',
                False,
                False,
            )
        )
    return found


class _RunnerOutput:
  """Reads the runner's output: its port, and every line into the log."""

  def __init__(self, process: subprocess.Popen) -> None:
    self.port: int | None = None
    self.lines: list[str] = []  # What it said before its port.
    self.ready = threading.Event()  # Set once the port is known or never.
    self._process = process
    threading.Thread(
        target=self._read, name=f'runner-{process.pid}-output', daemon=True
    ).start()

  def _read(self) -> None:
    for line in self._process.stdout:
      line = line.rstrip('\n')
      _LOG.info('runner %d: %s', self._process.pid, line)
      if self.ready.is_set():
        continue
      match = _LISTENING.fullmatch(line)
      if match:
        self.port = int(match.group(1))
        self.ready.set()
      else:
        self.lines.append(line)
    self.ready.set()


def _find_adb() -> str | None:
  candidates = []
  if 'ANDROID_HOME' in os.environ:
    candidates.append(pathlib.Path(os.environ['ANDROID_HOME']))
  candidates.append(pathlib.Path.home() / 'Library/Android/sdk')
  for sdk in candidates:
    adb = sdk / 'platform-tools' / 'adb'
    if adb.is_file():
      return str(adb)
  return shutil.which('adb')
