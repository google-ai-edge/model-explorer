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

"""One side of a Model Server: its runner, its connection, its chats.

A side launches its runner on its device, connects to it, loads the model,
then creates a chat, sends its turns and takes their captures, and may
create another chat in its place, one call at a time; cancel() and close()
may come from any thread. What both sides do
together, and when, is the Model Server's to decide (see model_server).

A side that chose a kept runner build runs only a runner that reports that
LiteRT-LM commit, on any device. A build refused says how to choose another:
by editing the Session, whose start it refused.
"""

from collections.abc import Callable
import pathlib
import threading

from model_debugger.protocol import runner_pb2
from model_debugger.server.runner import launcher
from model_debugger.server.runner import runner_builds
from model_debugger.server.runner import runner_client


class Side:
  """One side's runner, as its Model Server holds it.

  The lock its Model Server hands it guards this state and the Model
  Server's alike, so one critical section may read both; no runner call is
  made with it held.

  Attributes:
      role: 'ref' or 'target'.
      device: Where its runner runs.
      build: The runner build its Session chose (runner_builds), if any.
      model_dir: The model directory its runner reads in place (a PyTorch
        model's); None for a runner the model is uploaded to.
      phase: The side's phase, as its Model Server sets it.
      runner: The runner launched for it, if any.
      client: The connection to the runner, once connected.
      info: What the runner answered GetInfo with, once connected.
  """

  def __init__(
      self,
      role: str,
      device: launcher.Device,
      build: str | None,
      lock: threading.RLock,
      model_dir: pathlib.Path | None = None,
  ) -> None:
    """Holds a side that has launched nothing yet.

    Args:
        role: 'ref' or 'target'.
        device: Where its runner is to run.
        build: The runner build its Session chose: a LiteRT-LM commit,
          runner_builds.DEVELOPMENT, or None for the runner installed.
        lock: Its Model Server's lock.
        model_dir: The model directory its runner reads in place (a PyTorch
          model's); None for a runner the model is uploaded to.
    """
    self.role = role
    self.device = device
    self.build = build
    self.model_dir = model_dir
    self.phase = 'starting'
    self.runner: launcher.LaunchedRunner | None = None
    self.client: runner_client.RunnerClient | None = None
    self.info: runner_pb2.RunnerInfo | None = None
    self._lock = lock

  def launch(
      self, launchers: launcher.Devices, aborted: Callable[[], bool]
  ) -> bool:
    """Launches the side's runner on its device.

    Args:
        launchers: What launches runners.
        aborted: Whether the Model Server was turned off, or its other side
          failed, meanwhile; called with the lock held, together with recording
          the runner, so a stop either sees the runner or this sees the stop.

    Returns:
        False if it was aborted meanwhile; the runner is then stopped.

    Raises:
        RuntimeError: As launchers.launch(); a build not there says how to
          choose another.
    """
    try:
      if self.model_dir is not None:
        launched = launchers.launch(self.device.id, self.build, self.model_dir)
      else:
        launched = launchers.launch(self.device.id, self.build)
    except runner_builds.NotThereError as error:
      raise RuntimeError(
          f'{error.problem}; {self._way_out(error.fix)}.'
      ) from error
    with self._lock:
      self.runner = launched
      stop = aborted()
    if stop:
      launched.stop()
      return False
    return True

  def connect(self) -> runner_pb2.RunnerInfo:
    """Connects to the runner launched.

    Returns:
        What the runner answered GetInfo with.

    Raises:
        runner_client.Error: As RunnerClient.connect().
        RuntimeError: The side chose a LiteRT-LM commit, and the runner
          reports another; the connection stays, for stop() to close.
    """
    client = runner_client.RunnerClient.connect(self.runner)
    info = client.info
    with self._lock:
      self.client = client
      self.info = info
    reported = info.runtime.version
    chosen = self.build
    pytorch = info.runtime.name == runner_pb2.RuntimeInfo.PYTORCH
    if pytorch != (chosen == runner_builds.PYTORCH):
      name = runner_pb2.RuntimeInfo.Name.Name(info.runtime.name)
      raise RuntimeError(
          f'{self.device.name} runs a runner of {name}, which is not '
          "the runtime of this side's model; create a new Session."
      )
    if (
        chosen
        and chosen not in (runner_builds.DEVELOPMENT, runner_builds.PYTORCH)
        and reported != chosen
    ):
      # Only this computer launches the build chosen; elsewhere the
      # runner is the App installed, and no version is chosen there.
      if self.device.chooses_build:
        way_out = self._way_out('keep it again')
      else:
        way_out = self._way_out(
            f'install the App built from LiteRT-LM {chosen[:12]}',
            'choose the device again',
        )
      raise RuntimeError(
          f'{self.device.name} runs LiteRT-LM '
          f'{reported[:12] or "(no version)"}, but this Session chose '
          f'{chosen[:12]}; {way_out}.'
      )
    return info

  def _way_out(self, fix: str, change: str = '') -> str:
    """How to get past a runner build refused now, as a clause.

    Args:
        fix: How to bring the build chosen back, as a clause, or ''.
        change: What to choose in a new Session instead, as a clause; by
          default, choosing a runner version.

    Returns:
        A human-readable explanation of how to resolve the build conflict.
    """
    if not change:
      version = 'another version' if self.build else 'a version'
      change = f'choose {version}'
    choose = f'create a new Session and {change}'
    return f'{fix}, or {choose}' if fix else choose

  def load(
      self,
      request: runner_pb2.LoadRequest,
      model: pathlib.Path,
      upload_lock: threading.Lock,
      on_progress: Callable[[int, int], None],
  ) -> runner_pb2.LoadResponse:
    """Loads the model, uploading it first if the runner lacks it.

    Args:
        request: The Load; its sha256 is that of the file at `model`.
        model: The model file.
        upload_lock: Held while uploading, since runners may share a store.
        on_progress: Called with (bytes sent, total) during an upload.

    Returns:
        What the runner answered.

    Raises:
        runner_client.Error: As RunnerClient.load().
    """
    return self.client.load(request, model, upload_lock, on_progress)

  def create_chat(self, request: runner_pb2.CreateChatRequest) -> None:
    """Creates a chat; the runner's chat before, if any, is closed.

    Args:
        request: The create chat request.

    Raises:
        runner_client.Error: The runner refused or failed it; it then
          holds no chat.
    """
    self.client.create_chat(request)

  def send(
      self, prompt: str, on_chunk: Callable[[runner_pb2.Chunk], None]
  ) -> runner_pb2.Completion:
    """Runs the user's next turn.

    Args:
        prompt: What the user says to this side.
        on_chunk: Called with each chunk of the reply as it arrives.

    Returns:
        The turn's Completion.

    Raises:
        runner_client.Error: The reply did not complete; the chat is over.
    """
    return self.client.send(prompt, on_chunk)

  def take_capture(
      self,
      turn: int,
      directory: pathlib.Path,
      holder: pathlib.Path | None = None,
      on_progress: Callable[[int], None] | None = None,
  ) -> None:
    """Copies a completed turn's capture into `directory`, then releases it.

    Args:
        turn: The turn.
        directory: Where the files go.
        holder: What names the disk if it has no room.
        on_progress: Called with the bytes received so far, as
          RunnerClient.take_capture() calls it.

    Raises:
        runner_client.Error: As RunnerClient.take_capture().
    """
    self.client.take_capture(turn, directory, holder, on_progress)

  def cancel(self) -> None:
    """Stops the turn in flight, its Send or its capture; the chat is over.

    Or the next one, until another chat is created.
    """
    self.client.cancel()

  def close(self) -> None:
    """Closes the connection; calls in flight fail. Idempotent."""
    self.client.close()

  def stop(self) -> None:
    """Closes the connection and stops the runner, if any; idempotent."""
    # The channel first: what it carries ends as stopped, not as gone.
    if self.client is not None:
      self.client.close()
    if self.runner is not None:
      self.runner.stop()
