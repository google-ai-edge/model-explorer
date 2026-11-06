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

"""The Server: the Model Debugger's service, listening on loopback.

The facade over the layers below; every launch form shares it. The command
line (cli/) assembles nothing itself: it starts this and shows what it
reports.

Nothing a Server launched outlives it: close() stops every runner, and so
does the interpreter's exit (atexit) if close() was never called; what a
Server killed outright left, the next one on its workspace stops as it
starts.
"""

import atexit
import dataclasses
import errno
import pathlib
import socket
import threading

from model_debugger.server.runner import launchers
from model_debugger.server.runner import local_launcher
from model_debugger.server.runner import runner_builds
from model_debugger.server.service import server_service
from model_debugger.server.session import model_graphs
from model_debugger.server.session import workspace
import uvicorn

HOST = '127.0.0.1'
DEFAULT_PORT = 8080
LAST_DEFAULT_PORT = 8099
# The seconds converting one model's graph may take, unless told otherwise.
DEFAULT_GRAPH_TIMEOUT_SECONDS = model_graphs.DEFAULT_TIMEOUT_SECONDS
# The output of the runners on this computer, by line, each at the level the
# runner gave it: its logger's level sets how much of it shows.
RUNNER_OUTPUT_LOG = local_launcher.OUTPUT_LOG


# Another server holds the workspace.
WorkspaceInUseError = workspace.InUseError


class CannotListenError(Exception):
  """The port asked for cannot be bound, though not in use (one below 1024).

  Attributes:
      port: The port asked for.
  """

  def __init__(self, port: int, reason: str) -> None:
    """Names the port and why it cannot be bound."""
    super().__init__(f'Cannot listen on port {port}: {reason}')
    self.port = port


class PortInUseError(Exception):
  """Nothing to listen on: the port asked for, or every default one, is taken.

  Attributes:
      port: The port asked for; None when the default ones were tried.
  """

  def __init__(self, port: int | None) -> None:
    """Names the port asked for, or None for the default ones."""
    super().__init__(
        f'Port {port} is in use'
        if port is not None
        else f'Ports {DEFAULT_PORT}-{LAST_DEFAULT_PORT} are in use'
    )
    self.port = port


@dataclasses.dataclass(frozen=True)
class State:
  """What a Server shows of itself.

  Attributes:
      port: The port it listens on, on HOST.
      workspace: The workspace directory, resolved.
      ui: The built UI it serves, or None for the API alone.
      kept_builds: The directory of the runner builds kept on this computer, or
        None for none.
      builds: This computer's runner builds, kept ones and the development one,
        as listed now; it is a device while there is one.
  """

  port: int
  workspace: pathlib.Path
  ui: pathlib.Path | None
  kept_builds: pathlib.Path | None
  builds: tuple[runner_builds.Build, ...]


def default_workspace() -> pathlib.Path:
  """Returns the workspace a Server uses when none is named."""
  return workspace.default_root()


def default_kept_builds() -> pathlib.Path:
  """Returns where src/runner's cli:keep keeps this computer's runners."""
  return runner_builds.DEFAULT_KEPT


class Server:
  """The service on HOST, from start() until close()."""

  @classmethod
  def start(  # pylint: disable=redefined-outer-name
      cls,
      *,
      port: int | None,
      workspace: pathlib.Path | None,
      runner: pathlib.Path | None,
      ui: pathlib.Path | None,
      kept_builds: pathlib.Path | None = None,
      graph_timeout: float = DEFAULT_GRAPH_TIMEOUT_SECONDS,
      pytorch_runner: tuple[str, ...] | None = None,
      pytorch_setup: str = '',
  ) -> 'Server':
    """Binds the port and opens the service; accepts nothing until wait().

    Args:
        port: The port to listen on; 0 picks a free one; None the first free one
          of DEFAULT_PORT to LAST_DEFAULT_PORT.
        workspace: The workspace directory, created if missing; None for
          default_workspace().
        runner: The development build of the command-line runner, offered
          besides the kept builds; None for none.
        ui: The built UI's directory, or None to serve the API alone.
        kept_builds: The directory of the runner builds kept on this computer
          (runner_builds.DEFAULT_KEPT), read anew whenever they are listed; None
          for none.
        graph_timeout: The seconds converting one model's graph may take, as a
          Session of it is initialized.
        pytorch_runner: The command that runs the PyTorch runner on this
          computer; None for none.
        pytorch_setup: How to make that command, as a clause, for the reason it
          cannot be launched while it is not there.

    Raises:
        PortInUseError: The port, or every default one, is in use.
        CannotListenError: The port cannot be bound for another reason
          (one below 1024, say).
        WorkspaceInUseError: Another server holds the workspace.
    """
    listener = _listen(port)
    builds = runner_builds.RunnerBuilds(
        kept_builds, runner, pytorch_runner, pytorch_setup
    )
    try:
      service = server_service.ServerService.create(
          workspace if workspace is not None else default_workspace(),
          launchers.Launchers(builds),
          ui,
          graph_timeout,
      )
    except BaseException:
      listener.close()
      raise
    server = cls(listener, service, ui, builds)
    # The last resort, should no close() come: an embedder that forgets
    # it, an exception that skips it.
    atexit.register(server.close)
    return server

  def __init__(
      self,
      listener: socket.socket,
      service: server_service.ServerService,
      ui: pathlib.Path | None,
      builds: runner_builds.RunnerBuilds,
  ) -> None:
    """Serves `service` on `listener`; use start()."""
    self._listener = listener
    self._port = listener.getsockname()[1]
    self._service = service
    self._ui = ui
    self._builds = builds
    self._closing = threading.Lock()
    self._closed = False

  def state(self) -> State:
    """What the Server is now."""
    return State(
        self._port,
        self._service.workspace_root,
        self._ui,
        self._builds.kept_directory,
        tuple(self._builds.list()),
    )

  def wait(self) -> None:
    """Serves on this thread until a signal stops it.

    Connections are accepted from here on. Once shut down, uvicorn raises
    again the signal that stopped it.
    """
    # uvicorn logs where this process does, at WARNING and above: the
    # service writes its own access log (server_service.logs_access()).
    config = uvicorn.Config(
        self._service,
        log_config=None,
        log_level='warning',
        access_log=False,
    )
    uvicorn.Server(config).run(sockets=[self._listener])

  def close(self) -> None:
    """Stops every runner and releases the workspace and the port.

    Every runner it launched is stopped, within a bounded wait, then
    killed (runner/launch_records). Idempotent.
    """
    with self._closing:
      if self._closed:
        return
      self._closed = True
      atexit.unregister(self.close)
      try:
        self._service.close()
      finally:
        self._listener.close()


def _listen(port: int | None) -> socket.socket:
  """A socket bound on HOST: the port asked for, or the first free default.

  Bound without SO_REUSEADDR: with it, macOS lets a socket bind HOST on a
  port another one listens on for every address (0.0.0.0 or ::), and take
  that port's connections on loopback from it. Without it, a port with
  connections that closed in the last half minute (TIME_WAIT) is taken
  too.

  Args:
    port: The port to listen on; 0 picks a free one; None tries default ports.

  Returns:
    The bound socket listener.

  Raises:
    CannotListenError: If binding failed with access or availability errors.
    PortInUseError: If the specified port or all default ports are in use.
  """
  ports = (
      [port] if port is not None else range(DEFAULT_PORT, LAST_DEFAULT_PORT + 1)
  )
  for candidate in ports:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
      listener.bind((HOST, candidate))
    except OSError as error:
      listener.close()
      if error.errno == errno.EADDRINUSE:
        continue
      if port is not None and error.errno in (
          errno.EACCES,
          errno.EADDRNOTAVAIL,
      ):
        raise CannotListenError(port, error.strerror) from error
      raise
    return listener
  raise PortInUseError(port)
