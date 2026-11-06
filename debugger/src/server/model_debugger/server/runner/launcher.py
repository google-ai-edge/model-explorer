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

"""Where a runner can run, and the runner launched there.

A runner holds one chat for its life, so every chat needs a fresh runner
process: the server launches each one itself. A Launcher is one way of doing
that (this computer, Android over adb, an iOS simulator, an iOS device).
Devices is what the layers above use: every device of every Launcher, by id.
"""

import abc
import dataclasses
import pathlib
import socket
import threading
import time
from typing import Any

from absl import logging
from model_debugger.server.runner import app_version
from model_debugger.server.runner import runner_builds

# The App's Android package and iOS bundle identifier.
APP_ID = 'com.google.ai.edge.modeldebugger.runner'

# How long a launch may take until the runner accepts connections.
LAUNCH_SECONDS = 30

# The id of this computer as a device.
LOCAL_DEVICE_ID = 'local'


@dataclasses.dataclass(frozen=True)
class Device:
  """A place a runner can run.

  Attributes:
      id: LOCAL_DEVICE_ID, 'android:<serial>', 'ios-simulator:<udid>' or
        'ios:<CoreDevice identifier>'.
      name: What the user knows the device by.
      platform: 'macOS', 'Android', 'iOS Simulator' or 'iOS'.
      transport: 'local', 'adb', 'simulator' or 'devicectl'.
      shared: Whether it can run several runners at once (only this computer).
      chooses_build: Whether each start there launches the runner build its
        Session chose (only this computer, from its builds()); elsewhere the
        runner is the App installed.
      app: The Runner App of a phone, installed or not, as read without
        launching it; None on a device that chooses_build.

  Whether a runner there can use a GPU is not the device's to say: only
  the runner's GetInfo does, once launched.
  """

  id: str
  name: str
  platform: str
  transport: str
  shared: bool
  chooses_build: bool = False
  app: app_version.App | None = None

  @property
  def without_app(self) -> bool:
    """Whether it is a phone with no Runner App: no runner runs there."""
    return self.app is not None and not self.app.installed


class LaunchedRunner(abc.ABC):
  """A runner this server started, listening at `address`.

  Attributes:
      device: Where it runs.
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

  def exited(self) -> bool:
    """Whether the runner is known to have stopped, asking no device.

    Quick, and safe to ask under a lock: where seeing the process takes a
    device's tool (adb), it says False, and the connection to the runner
    tells instead.

    Returns:
      True if the runner is known to have stopped, False otherwise.
    """
    return self._stopped or (self._sees_process() and not self._alive())

  def last_words(self) -> str:
    """Why the runner said it exited, if it ended unasked with a failure.

    A runner that cannot go on says why in its last line, `runner: ...`,
    then exits with a failure status. Waits a moment for a process that
    is ending to have ended.

    Returns:
        The line; '' for a runner that runs, was stopped, exited well or
        was killed, or whose output this server does not see (only a
        runner on this computer's).
    """
    return ''

  def wait_until_serving(self, deadline: float) -> None:
    """Returns once this runner, and no other process, serves at `address`.

    Args:
        deadline: The time.monotonic() by which it must serve.

    Raises:
        RuntimeError: It stopped, another process listens there, or nothing
          answered by the deadline.
    """
    address = self.address
    while True:
      answered = _serving(address)
      if answered and self._listens():
        return
      if not self.alive():
        raise RuntimeError(self._gone_while_starting())
      if time.monotonic() > deadline:
        if answered:
          raise RuntimeError(
              f'Another process listens at {address}, where the runner was to.'
          )
        raise RuntimeError(
            f'No runner answered at {address} within {LAUNCH_SECONDS} s.'
        )
      time.sleep(0.2)

  def _gone_while_starting(self) -> str:
    """Why a start failed whose runner's process is gone, for the user.

    A phone's App says what to check on the phone.

    Returns:
      A message indicating the runner process stopped while starting.
    """
    return f'The runner for {self.address} stopped while starting.'

  def _sees_process(self) -> bool:
    """Whether _alive() asks nothing beyond this computer, for exited()."""
    return False

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
    logging.info(
        'Stopped the runner on %s at %s', self.device.name, self.address
    )

  @abc.abstractmethod
  def _alive(self) -> bool:
    ...

  @abc.abstractmethod
  def _stop(self) -> None:
    ...


class Launcher(abc.ABC):
  """One way of starting runners, and the devices it reaches now.

  Attributes:
      KIND: What its launch records name it by (launch_records): 'local',
        'android', 'ios-simulator' or 'ios'.
  """

  KIND: str

  @abc.abstractmethod
  def devices(self) -> list[Device]:
    """Returns every device it reaches now.

    A phone without the Runner App is one too (Device.without_app): no
    runner launches there, and the UI says why.
    """

  def builds(self, device: Device) -> list[runner_builds.Build]:
    """Returns the runner builds a Session may choose on a device.

    None but on a device that chooses_build: elsewhere the runner is the
    App installed.

    Args:
      device: The target device to query builds for.

    Returns:
      A list of available runner builds on the device.
    """
    del device  # The App installed is the only runner.
    return []

  @abc.abstractmethod
  def launch(
      self,
      device: Device,
      deadline: float,
      build: str | None,
      model_dir: pathlib.Path | None = None,
  ) -> LaunchedRunner:
    """Starts a fresh runner on one of its devices.

    Whatever runner of this server's App ran on the device before is
    stopped. The runner returned may not serve yet: see wait_until_serving.

    Args:
        device: One of its devices.
        deadline: The time.monotonic() by which the runner must serve.
        build: The id of the build to launch, of builds(), on a device that
          chooses_build; ignored elsewhere, where the runner is the App
          installed.
        model_dir: The model directory a runner that reads its model in place
          (the PyTorch runner) is started with; None for others.

    Raises:
        RuntimeError: The runner did not start, its build cannot be
          launched, a tool failed, or the server is stopping.
    """

  @abc.abstractmethod
  def stop_leftover(self, fields: dict[str, Any]) -> None:
    """Stops what a previous server's launch record names, if still there.

    Only what the record names, and only once it is sure that is what
    runs there now (the same process, the same forward): never another.

    Args:
        fields: The record's fields, of this launcher's KIND.

    Raises:
        RuntimeError: A tool failed.
    """


class Devices(abc.ABC):
  """Every device a runner can be launched on now, whatever its platform."""

  @abc.abstractmethod
  def devices(self) -> list[Device]:
    """Returns every device reached now, as Launcher.devices()."""

  @abc.abstractmethod
  def builds(self, device_id: str) -> list[runner_builds.Build]:
    """Returns the runner builds a Session may choose on a device.

    Read without launching anything; none but on a device that
    chooses_build.

    Raises:
        NotFound: No such device now.
    """

  @abc.abstractmethod
  def launch(
      self,
      device_id: str,
      build: str | None = None,
      model_dir: pathlib.Path | None = None,
  ) -> LaunchedRunner:
    """Starts a fresh runner on a device; returns once it accepts calls.

    Whatever runner of this server's App ran on the device before is
    stopped.

    Args:
        device_id: The Device's id.
        build: The id of the build to launch, of builds(), on a device that
          chooses_build; ignored elsewhere.
        model_dir: As Launcher.launch().

    Raises:
        NotFound: No such device now.
        RuntimeError: The runner did not start in time, its build cannot
          be launched, a tool failed, or the server is stopping.
    """

  def record_in(self, directory: pathlib.Path) -> None:
    """Stops what a previous server recorded in `directory` and left.

    From then on, records each launch there, until it is stopped. Called
    once the workspace is held, before anything launches.

    Args:
      directory: The directory where launches are recorded.
    """
    del directory  # Records nothing.

  def stop_all(self) -> None:
    """Stops every runner launched and not stopped; nothing launches after.

    Bounded: what does not stop is killed, or logged.
    """


def _serving(address: str) -> bool:
  """Whether a gRPC server answers at `address`, without an RPC.

  A gRPC server sends its HTTP/2 settings as soon as it accepts a
  connection, before any RPC. An adb forward accepts the connection itself
  and closes it when the device side does not, so accepting alone proves
  nothing.

  Args:
    address: The address to check.

  Returns:
    True if a gRPC server is answering at the address, False otherwise.
  """
  host, port = address.rsplit(':', 1)
  try:
    # An IPv6 host is bracketed, as gRPC takes it: '[fd00::1]:8769'.
    with socket.create_connection(
        (host.strip('[]'), int(port)), timeout=1
    ) as probe:
      probe.settimeout(1)
      return bool(probe.recv(1))
  except OSError:
    return False
