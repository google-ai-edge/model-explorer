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

"""Launchers of in-process fake runners, for the service's tests.

Each launch starts a fresh FakeRunner that replies with the tiny model of
capture_fixtures; a Script, its `respond`, can hold or refuse its replies.
This computer chooses its build: its runners report the commit they were
launched as.
"""

import pathlib
import threading

import grpc
from model_debugger.protocol import runner_pb2
from model_debugger.server.runner import launcher
from model_debugger.server.runner import runner_builds
from model_debugger.server.testing import capture_fixtures
from model_debugger.server.testing import fake_runner
from model_debugger.server.util import errors

LOCAL = launcher.Device(
    launcher.LOCAL_DEVICE_ID,
    'This computer',
    'macOS',
    'local',
    True,
    chooses_build=True,
)
# The commit fake runners report unless launched as another: the build a
# Session of the tests chooses on this computer.
COMMIT = fake_runner.DEFAULT_INFO.runtime.version
PHONE = launcher.Device('android:A', 'Pixel', 'Android', 'adb', False)
TABLET = launcher.Device('android:B', 'Tablet', 'Android', 'adb', False)
# Every turn's reply, from the tiny model of capture_fixtures.
REPLY = ['It', ' is', ' 4.']
REPLY_TEXT = 'It is 4.'
# How long a test waits for what a runner or a job does.
TIMEOUT = 20


class Script:
  """What one fake runner replies: the tiny model's turns, held or refused.

  Attributes:
      runner: The runner it replies for.
      hold: If set, each reply sends its first chunk, then waits for it.
      refuse: If set, each Send fails with this status.
      no_room: If set, (tokens, left): each Send is refused as needing `tokens`
        with `left` left in the context; the chat goes on.
      thought: The words each reply thinks first; the tiny model then rewrites
        the reply without them as its turn ends.
      files: More files each turn's capture holds, by name, beside the tiny
        model's: an execution graph a PyTorch runner exports, say.
  """

  def __init__(self) -> None:
    """Replies with the tiny model's turns until told otherwise."""
    self.runner: fake_runner.FakeRunner | None = None
    self.hold: threading.Event | None = None
    self.refuse: grpc.StatusCode | None = None
    self.no_room: tuple[int, int] | None = None
    self.thought: list[str] = []
    self.files: dict[str, bytes] = {}
    self._chat: capture_fixtures.TinyChat | None = None
    # The CreateChat of the chat `_chat` replies in: a new one starts it
    # anew.
    self._request: runner_pb2.CreateChatRequest | None = None

  def __call__(self, turn: int, prompt: str) -> fake_runner.Reply:
    """Answers one Send as the fake runner's `respond`."""
    if self.refuse is not None:
      raise fake_runner.AbortError(self.refuse, f'Refused turn {turn}.')
    if self.no_room is not None:
      raise fake_runner.NoRoomError(*self.no_room)
    if self._chat is None or self._request is not self.runner.chat_request:
      request = self._request = self.runner.chat_request
      system = [
          m.content
          for m in request.messages
          if m.role == runner_pb2.ChatMessage.SYSTEM
      ]
      pytorch = self.runner.info.runtime.name == runner_pb2.RuntimeInfo.PYTORCH
      self._chat = capture_fixtures.TinyChat(
          system[0] if system else '',
          request.options.max_completion_tokens,
          pytorch=pytorch,
      )
    reply = self._chat.send(prompt, REPLY, thought=self.thought)
    chunks = [
        runner_pb2.Chunk(reasoning_content=text)
        for text in reply.reasoning_chunks
    ] + [runner_pb2.Chunk(content=text) for text in reply.chunks]
    hold = self.hold
    if hold is not None:

      def held():
        yield chunks[0]
        hold.wait(TIMEOUT)
        yield from chunks[1:]

      return held(), reply.completion, {**reply.files, **self.files}
    return chunks, reply.completion, {**reply.files, **self.files}


def runner_build(commit: str, reason: str = '') -> runner_builds.Build:
  """A build of this computer as RunnerBuilds lists it; DEVELOPMENT too.

  runner_builds.PYTORCH is the PyTorch runner.

  Args:
    commit: The commit or build identifier.
    reason: Optional reason string describing the build status.

  Returns:
    The matching Build record.
  """
  if commit == runner_builds.PYTORCH:
    return runner_builds.Build(
        commit,
        pathlib.Path('/usr/bin/uv'),
        '',
        '',
        '',
        {},
        reason,
        runtime=runner_builds.PYTORCH_RUNTIME,
        command=('/usr/bin/uv', 'run', 'model-debugger-runner-pytorch'),
    )
  if commit == runner_builds.DEVELOPMENT:
    return runner_builds.Build(
        commit, pathlib.Path('/bazel-bin/runner'), COMMIT, '', '', {}, ''
    )
  return runner_builds.Build(
      commit,
      pathlib.Path(f'/kept/{commit}/runner'),
      commit,
      '2026-10-12T22:51:09-07:00',
      '998385386',
      {'runner': '0' * 64},
      reason,
  )


class FakeLaunchedRunner(launcher.LaunchedRunner):
  """A fake runner, as a launcher hands it out.

  Attributes:
      runner: The fake runner.
      script: What it replies.
      build: The build it was launched as, if its device chooses one.
      model_dir: The model directory it was started with (the PyTorch runner's),
        or None.
  """

  model_dir: pathlib.Path | None = None

  def __init__(
      self,
      device: launcher.Device,
      runner: fake_runner.FakeRunner,
      build: str | None,
  ) -> None:
    """Hands out `runner`, running on `device` as `build`.

    As the real launchers, it sees the runner stop at once, but over adb.

    Args:
      device: The device the runner is associated with.
      runner: The fake runner being handed out.
      build: The build identifier, if any.
    """
    super().__init__(device, runner.address)
    self.runner = runner
    self.script = runner._respond
    self.build = build

  def _sees_process(self) -> bool:
    return self.device.transport != 'adb'

  def _alive(self) -> bool:
    return not self.runner.stopped

  def _stop(self) -> None:
    self.runner.stop()


class FakeLaunchers(launcher.Devices):
  """Launches a fresh fake runner per launch.

  Each device has a model store of its own, which every runner launched on
  it shares: only this computer runs several at once.

  Attributes:
      devices_now: The devices offered now.
      builds_now: The builds of the devices that choose one, by device id: a
        kept build of COMMIT and the development build, for one not named.
      stores: Each device's model store, by its id.
      infos: What the runners of a device answer GetInfo with, by its id; for a
        device not named, fake_runner.DEFAULT_INFO with the commit of the kept
        build launched.
      loaded: What the runners of a device answer Load with, by its id; for a
        device not named, the tiny model's (capture_fixtures.TinyChat).
      launched: Every runner launched, oldest first.
      stopped_all: Whether stop_all() was called.
      records: The directory record_in() was given, if called.
  """

  def __init__(self, devices: list[launcher.Device]) -> None:
    """Offers `devices` until a test changes `devices_now`."""
    self.devices_now = list(devices)
    self.builds_now: dict[str, list[runner_builds.Build]] = {}
    self.stores: dict[str, set[str]] = {}
    self.infos: dict[str, runner_pb2.RunnerInfo] = {}
    self.loaded: dict[str, runner_pb2.LoadResponse] = {}
    self.launched: list[FakeLaunchedRunner] = []
    self.stopped_all = False
    self.records: pathlib.Path | None = None
    self._lock = threading.Lock()

  def devices(self) -> list[launcher.Device]:
    """Returns the devices offered now."""
    return list(self.devices_now)

  def builds(self, device_id: str) -> list[runner_builds.Build]:
    """Returns the builds of a device offered now.

    Args:
      device_id: Target device identifier.

    Returns:
      A list of available builds for the device.

    Raises:
      NotFound: No such device now.
    """
    device = next((d for d in self.devices_now if d.id == device_id), None)
    if device is None:
      raise errors.NotFound(f'No device {device_id} is available.')
    if not device.chooses_build:
      return []
    return list(
        self.builds_now.get(
            device_id,
            [runner_build(COMMIT), runner_build(runner_builds.DEVELOPMENT)],
        )
    )

  def launch(
      self,
      device_id: str,
      build: str | None = None,
      model_dir: pathlib.Path | None = None,
  ) -> FakeLaunchedRunner:
    """Starts a fresh fake runner on a device, as `build` if it chooses.

    Args:
      device_id: Target device identifier.
      build: Build identifier to launch, if device chooses builds.
      model_dir: Model directory for PyTorch runner, if applicable.

    Returns:
      The newly launched fake runner instance.

    Raises:
      RuntimeError: The device chooses its build, and has no launchable
        build `build`, or stop_all() was called.
    """
    if self.stopped_all:
      raise RuntimeError('The server is stopping.')
    device = next(d for d in self.devices_now if d.id == device_id)
    info = fake_runner.DEFAULT_INFO
    if device.chooses_build:
      chosen = next((b for b in self.builds(device_id) if b.id == build), None)
      if chosen is None or not chosen.available:
        raise RuntimeError(f'No launchable build {build} is kept.')
      info = runner_pb2.RunnerInfo()
      if chosen.runtime == runner_builds.PYTORCH_RUNTIME:
        info.CopyFrom(fake_runner.PYTORCH_INFO)
      else:
        info.CopyFrom(fake_runner.DEFAULT_INFO)
        info.runtime.version = chosen.commit
    else:
      build = None
    script = Script()
    with self._lock:
      store = self.stores.setdefault(device.id, set())
      info = self.infos.get(device.id, info)
      loaded = self.loaded.get(
          device.id, capture_fixtures.TinyChat().load_response()
      )
    script.runner = fake_runner.FakeRunner(
        script, store=store, info=info, load_response=loaded
    ).start()
    launched = FakeLaunchedRunner(device, script.runner, build)
    launched.model_dir = model_dir
    with self._lock:
      self.launched.append(launched)
    return launched

  def record_in(self, directory: pathlib.Path) -> None:
    """Keeps the directory, which it records nothing in."""
    self.records = directory

  def stop_all(self) -> None:
    """Stops every runner launched; refuses launches from now on."""
    with self._lock:
      self.stopped_all = True
      launched = list(self.launched)
    for runner in launched:
      runner.stop()

  def running(self, device_id: str) -> list[FakeLaunchedRunner]:
    """Returns the runners of a device not stopped, oldest first."""
    with self._lock:
      return [
          launched
          for launched in self.launched
          if (
              launched.device.id == device_id
              and not launched._stopped  # pylint: disable=protected-access
          )
      ]

  def runners(self, device_id: str) -> list[fake_runner.FakeRunner]:
    """Returns every runner launched on a device, oldest first."""
    with self._lock:
      return [
          launched.runner
          for launched in self.launched
          if launched.device.id == device_id
      ]
