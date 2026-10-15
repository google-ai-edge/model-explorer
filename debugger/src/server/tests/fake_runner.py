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

"""An in-process Runner for tests: the protocol's rules, no model.

It keeps the rules the real runner keeps (runner_service.cc), with its status
codes and messages:
- it serves the TCP peer of its first RPC and refuses any other;
- Load once; NOT_FOUND until the model's sha256 is in its store;
- CreateChat once, after Load;
- one Send at a time; a Send that fails other than INVALID_ARGUMENT, or is
  cancelled, ends the chat;
- a turn's capture is listed, read and released by its number.

What a turn replies and captures comes from a callback,
`respond(turn, prompt) -> (chunks, completion, files)`; it may raise `Abort`
to fail the Send, and `chunks` may be a generator that blocks, to hold a
Send open.

Usage:
  with FakeRunner(respond, store={sha256}) as runner:
    client = RunnerClient(runner.address)
"""

from collections.abc import Callable, Iterable
import concurrent.futures
import hashlib
import json
import pathlib
import posixpath
import re
import threading
import time

import capture_fixtures
import grpc
from model_debugger.v1 import runner_pb2, runner_pb2_grpc
from model_debugger_server import (
    execution,
    jobs,
    launchers,
    models,
    sessions,
    workspace,
)

MODEL_BYTES = b'a model, as far as the fake runner can tell' * 1000
LOCAL = launchers.Device('local', 'This computer', 'macOS', 'local', True, True)
PHONE = launchers.Device('android:A', 'Pixel', 'Android', 'adb', False, True)
TABLET = launchers.Device('android:B', 'Tablet', 'Android', 'adb', False, True)
# Every turn's reply, from the tiny model of capture_fixtures.
REPLY = ['It', ' is', ' 4.']
REPLY_TEXT = 'It is 4.'
TIMEOUT = 20

# What a Send replies: the chunks, the completion, the turn's capture files.
Reply = tuple[
    Iterable[runner_pb2.Chunk], runner_pb2.Completion, dict[str, bytes]
]

# The options schemas of the real runner's LiteRT-LM (litert_lm_runtime.cc).
LOAD_OPTIONS_SCHEMA = json.dumps({
    'type': 'object',
    'properties': {
        'cpuThreads': {'type': 'integer', 'minimum': 1},
        'audioCpuThreads': {'type': 'integer', 'minimum': 1},
        'prefillChunkSize': {'type': 'integer', 'minimum': 1},
        'loraRank': {'type': 'integer', 'minimum': 0},
        'audioLoraRank': {'type': 'integer', 'minimum': 0},
        'enableYnnpack': {'type': 'boolean'},
        'parallelFileSectionLoading': {'type': 'boolean'},
        'singleThreadedExecution': {'type': 'boolean'},
        'gpuMetalResidencySet': {'type': 'boolean'},
        'visionBackend': {'enum': ['CPU', 'GPU']},
        'audioBackend': {'enum': ['CPU', 'GPU']},
        'activationDataType': {'enum': ['float32', 'float16', 'int16', 'int8']},
    },
    'additionalProperties': False,
})
CHAT_OPTIONS_SCHEMA = json.dumps({
    'type': 'object',
    'properties': {
        'thinking': {'type': 'boolean'},
        'thinking_token_budget': {'type': 'integer', 'minimum': -1},
        'apply_prompt_template': {'type': 'boolean'},
        'speculative_decoding': {'type': 'boolean'},
    },
    'additionalProperties': False,
})

# What the real runner answered GetInfo with on an M2 Max.
DEFAULT_INFO = runner_pb2.RunnerInfo(
    protocol_version='1.0.0',
    host=runner_pb2.HostInfo(
        platform=runner_pb2.HostInfo.MACOS,
        os_version='27.2',
        os_build='26B5086k',
        architecture='arm64',
        model='Mac14,6',
        chip='Apple M2 Max',
        cpu_count=12,
        memory_bytes=68719476736,
        gpu=runner_pb2.HostInfo.Gpu(name='Apple M2 Max'),
    ),
    runtime=runner_pb2.RuntimeInfo(
        name=runner_pb2.RuntimeInfo.LITERT_LM,
        version='a0f1e97361155c8bb9cd3d6aa045c04320d4334b',
        load_options_schema=LOAD_OPTIONS_SCHEMA,
        chat_options_schema=CHAT_OPTIONS_SCHEMA,
    ),
)

# What the real runner's Load answered for Gemma 4 E2B.
DEFAULT_LOAD_RESPONSE = runner_pb2.LoadResponse(
    start_token=2,
    stop_tokens=[
        runner_pb2.StopToken(ids=runner_pb2.TokenIds(ids=[1])),
        runner_pb2.StopToken(ids=runner_pb2.TokenIds(ids=[50])),
        runner_pb2.StopToken(ids=runner_pb2.TokenIds(ids=[106])),
    ],
)

_SHA256 = re.compile(r'[0-9a-f]{64}')
_UPLOAD_DATA_BYTES = 1 << 20
_CAPTURE_DATA_BYTES = 1 << 20
_FAILED = grpc.StatusCode.FAILED_PRECONDITION
_INVALID = grpc.StatusCode.INVALID_ARGUMENT


class Abort(Exception):
  """Raised by `respond` to fail a Send with a status."""

  def __init__(self, code: grpc.StatusCode, message: str) -> None:
    """Initializes the status the Send fails with."""
    super().__init__(message)
    self.code = code
    self.message = message


class FakeRunner(runner_pb2_grpc.RunnerServicer):
  """A Runner on 127.0.0.1, in this process.

  Attributes:
    address: 'host:port' it listens on, once started.
    store: sha256 of the models it holds; runners may share one set.
    peers: the peer of every RPC, in order, refused ones included.
    uploads: sha256 of every model uploaded, in order.
    load_request: the Load that succeeded, if any.
    chat_request: the CreateChat that succeeded, if any.
    prompts: the content of every completed turn, in order.
    released: the turns released, in order.
    chat_over: whether a cancelled or failed Send ended the chat.
  """

  def __init__(
      self,
      respond: Callable[[int, str], Reply],
      store: set[str] | None = None,
      info: runner_pb2.RunnerInfo = DEFAULT_INFO,
      load_response: runner_pb2.LoadResponse = DEFAULT_LOAD_RESPONSE,
  ) -> None:
    """Prepares the runner; `start` serves it."""
    self.address = ''
    self.store = store if store is not None else set()
    self.peers: list[str] = []
    self.uploads: list[str] = []
    self.load_request: runner_pb2.LoadRequest | None = None
    self.chat_request: runner_pb2.CreateChatRequest | None = None
    self.prompts: list[str] = []
    self.released: list[int] = []
    self.chat_over = False
    self._respond = respond
    self._info = info
    self._load_response = load_response
    self._lock = threading.Lock()
    self._client: str | None = None
    self._sending = False
    self._turns = 0
    self._captures: dict[int, dict[str, bytes]] = {}
    self._server: grpc.Server | None = None

  def start(self) -> 'FakeRunner':
    """Serves on a free port of 127.0.0.1; returns self."""
    self._server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=8)
    )
    runner_pb2_grpc.add_RunnerServicer_to_server(self, self._server)
    port = self._server.add_secure_port(
        '127.0.0.1:0',
        grpc.local_server_credentials(grpc.LocalConnectionType.LOCAL_TCP),
    )
    self._server.start()
    self.address = f'127.0.0.1:{port}'
    return self

  def stop(self) -> None:
    """Stops serving at once; calls in flight fail."""
    if self._server is not None:
      self._server.stop(grace=None).wait()

  def __enter__(self) -> 'FakeRunner':
    """Starts the runner."""
    return self.start()

  def __exit__(self, *exc_info: object) -> None:
    """Stops the runner."""
    self.stop()

  # ---- The protocol

  def GetInfo(self, request, context):  # noqa: D102
    with self._lock:
      self._admit(context)
    return self._info

  def UploadModel(self, request_iterator, context):  # noqa: D102
    with self._lock:
      self._admit(context)
    first = next(request_iterator, None)
    if first is None or not first.HasField('header'):
      context.abort(_INVALID, 'The stream does not start with a header.')
    header = first.header
    if not _SHA256.fullmatch(header.sha256):
      context.abort(_INVALID, 'sha256 is not 64 lower-case hexadecimal digits.')
    if header.size < 0:
      context.abort(_INVALID, 'size is negative.')
    with self._lock:
      if header.sha256 in self.store:
        return runner_pb2.UploadModelResponse()  # the stream unread
    digest = hashlib.sha256()
    received = 0
    for request in request_iterator:
      if not request.HasField('data'):
        context.abort(_INVALID, 'A second header arrived in the stream.')
      if len(request.data) > _UPLOAD_DATA_BYTES:
        context.abort(_INVALID, 'A data message exceeds 1 MiB.')
      if len(request.data) > header.size - received:
        context.abort(
            _INVALID,
            f'More than the {header.size} bytes the header announced arrived.',
        )
      digest.update(request.data)
      received += len(request.data)
    if received != header.size:
      context.abort(
          grpc.StatusCode.DATA_LOSS,
          f'The header announced {header.size} bytes; {received} arrived.',
      )
    if digest.hexdigest() != header.sha256:
      context.abort(
          grpc.StatusCode.DATA_LOSS,
          f'The bytes hash to {digest.hexdigest()}, not to {header.sha256}.',
      )
    with self._lock:
      self.store.add(header.sha256)
      self.uploads.append(header.sha256)
    return runner_pb2.UploadModelResponse()

  def Load(self, request, context):  # noqa: D102
    with self._lock:
      self._admit(context)
      if self.load_request is not None:
        context.abort(_FAILED, 'A model is loaded; a runner loads one.')
      if not _SHA256.fullmatch(request.sha256):
        context.abort(
            _INVALID, 'sha256 is not 64 lower-case hexadecimal digits.'
        )
      if request.sha256 not in self.store:
        context.abort(
            grpc.StatusCode.NOT_FOUND,
            f'No model with sha256 {request.sha256} in the model store.',
        )
      if request.context_length <= 0:
        context.abort(_INVALID, f'context_length is {request.context_length}.')
      if request.device not in (runner_pb2.CPU, runner_pb2.GPU):
        context.abort(_INVALID, 'The device is not CPU or GPU.')
      _check_options(
          context, 'load', request.runtime_options, LOAD_OPTIONS_SCHEMA
      )
      self.load_request = request
    return self._load_response

  def CreateChat(self, request, context):  # noqa: D102
    with self._lock:
      self._admit(context)
      if self.load_request is None:
        context.abort(_FAILED, 'No model is loaded.')
      if self.chat_request is not None:
        context.abort(_FAILED, 'The chat exists; a runner holds one.')
      options = request.options
      if options.max_completion_tokens <= 0:
        context.abort(
            _INVALID,
            f'max_completion_tokens is {options.max_completion_tokens}.',
        )
      _check_options(
          context, 'chat', options.runtime_options, CHAT_OPTIONS_SCHEMA
      )
      for i, message in enumerate(request.messages):
        if message.role == runner_pb2.ChatMessage.SYSTEM and i != 0:
          context.abort(
              _INVALID, 'A system message may only be the first message.'
          )
      self.chat_request = request
    return runner_pb2.CreateChatResponse()

  def Send(self, request, context):  # noqa: D102
    with self._lock:
      self._admit(context)
      if self.chat_request is None:
        context.abort(_FAILED, 'No chat exists.')
      if self.chat_over:
        context.abort(
            _FAILED,
            'The chat is over; generation on this connection ended with a '
            'cancelled or failed turn.',
        )
      if self._sending:
        context.abort(_FAILED, 'A Send is in progress.')
      if request.message.role != runner_pb2.ChatMessage.USER:
        context.abort(_INVALID, 'A turn is a user message.')
      self._sending = True
      turn = self._turns + 1
    completed = refused = False
    try:
      try:
        chunks, completion, files = self._respond(turn, request.message.content)
        for chunk in chunks:
          if not context.is_active():
            break
          yield runner_pb2.SendResponse(chunk=chunk)
      except Abort as abort:
        refused = abort.code == _INVALID
        context.abort(abort.code, abort.message)
      if not context.is_active():
        return
      with self._lock:
        self._turns = turn
        self._captures[turn] = dict(files)
        self.prompts.append(request.message.content)
      completed = True
      yield runner_pb2.SendResponse(completion=completion)
    finally:
      # Cancelled, failed, or abandoned by its client at a yield.
      with self._lock:
        self._sending = False
        if not completed and not refused:
          self.chat_over = True

  def ListCaptures(self, request, context):  # noqa: D102
    with self._lock:
      self._admit(context)
      files = self._capture(request.turn, context)
      return runner_pb2.ListCapturesResponse(
          files=[
              runner_pb2.CaptureFile(
                  path=name,
                  size=len(data),
                  sha256=hashlib.sha256(data).hexdigest(),
              )
              for name, data in sorted(files.items())
          ]
      )

  def ReadCapture(self, request, context):  # noqa: D102
    with self._lock:
      self._admit(context)
      files = self._capture(request.turn, context)
      path = request.path
      if not path or path != posixpath.basename(path):
        context.abort(
            _INVALID, "path is the name of a file in the turn's capture."
        )
      if path not in files:
        context.abort(
            grpc.StatusCode.NOT_FOUND, f'No file {path} in turn {request.turn}.'
        )
      data = files[path]
    for start in range(0, len(data), _CAPTURE_DATA_BYTES):
      yield runner_pb2.CaptureData(
          data=data[start : start + _CAPTURE_DATA_BYTES]
      )

  def ReleaseCaptures(self, request, context):  # noqa: D102
    with self._lock:
      self._admit(context)
      self._capture(request.turn, context)
      del self._captures[request.turn]
      self.released.append(request.turn)
    return runner_pb2.ReleaseCapturesResponse()

  # ---- Rules

  def _admit(self, context: grpc.ServicerContext) -> None:
    """Serves the first RPC's peer only; called with the lock held."""
    peer = context.peer()
    self.peers.append(peer)
    if self._client is None:
      self._client = peer
    elif peer != self._client:
      context.abort(_FAILED, 'This runner is serving another client.')

  def _capture(
      self, turn: int, context: grpc.ServicerContext
  ) -> dict[str, bytes]:
    """Returns a turn's files; called with the lock held."""
    if turn not in self._captures:
      context.abort(grpc.StatusCode.NOT_FOUND, f'Turn {turn} has no capture.')
    return self._captures[turn]


def _check_options(
    context: grpc.ServicerContext, kind: str, options, schema: str
) -> None:
  """Refuses a runtime option the schema does not name, as LiteRT-LM does."""
  known = json.loads(schema)['properties']
  for name in options.keys():
    if name not in known:
      context.abort(_INVALID, f'LiteRT-LM has no {kind} option {name}.')


class Script:
  """What one fake runner replies: the tiny model's turns, held or refused.

  Attributes:
    runner: the runner it replies for.
    hold: if set, each reply sends its first chunk, then waits for it.
    refuse: if set, each Send fails with this status.
  """

  def __init__(self) -> None:
    """Replies with the tiny model's turns until told otherwise."""
    self.runner: FakeRunner | None = None
    self.hold: threading.Event | None = None
    self.refuse: grpc.StatusCode | None = None
    self._chat: capture_fixtures.TinyChat | None = None

  def __call__(self, turn: int, prompt: str) -> Reply:
    """Answers one Send as the fake runner's `respond`."""
    if self.refuse is not None:
      raise Abort(self.refuse, f'Refused turn {turn}.')
    if self._chat is None:
      request = self.runner.chat_request
      system = [
          m.content
          for m in request.messages
          if m.role == runner_pb2.ChatMessage.SYSTEM
      ]
      self._chat = capture_fixtures.TinyChat(
          system[0] if system else '',
          request.options.max_completion_tokens,
      )
    reply = self._chat.send(prompt, REPLY)
    chunks = [runner_pb2.Chunk(content=text) for text in reply.chunks]
    hold = self.hold
    if hold is not None:

      def held() -> Iterable[runner_pb2.Chunk]:
        yield chunks[0]
        hold.wait(TIMEOUT)
        yield from chunks[1:]

      return held(), reply.completion, reply.files
    return chunks, reply.completion, reply.files


class Launched(launchers.LaunchedRunner):
  """A fake runner, as a launcher hands it out."""

  def __init__(self, device: launchers.Device, runner: FakeRunner) -> None:
    """Hands out `runner`, running on `device`."""
    super().__init__(device, runner.address)
    self.runner = runner
    self.script = runner._respond

  def _alive(self) -> bool:
    return True

  def _stop(self) -> None:
    self.runner.stop()


class Launchers:
  """Launches a fresh fake runner per launch; they share one model store."""

  def __init__(self, devices: list[launchers.Device]) -> None:
    """Offers `devices` until a test changes `devices_now`."""
    self.devices_now = list(devices)
    self.store: set[str] = set()
    self.launched: list[Launched] = []
    self._lock = threading.Lock()

  def devices(self) -> list[launchers.Device]:
    """Returns the devices offered now."""
    return list(self.devices_now)

  def launch(self, device_id: str) -> Launched:
    """Starts a fresh fake runner on a device."""
    device = next(d for d in self.devices_now if d.id == device_id)
    script = Script()
    script.runner = FakeRunner(
        script,
        store=self.store,
        load_response=capture_fixtures.TinyChat().load_response(),
    ).start()
    launched = Launched(device, script.runner)
    with self._lock:
      self.launched.append(launched)
    return launched

  def running(self, device_id: str) -> list[Launched]:
    """Returns the runners of a device not stopped, oldest first."""
    with self._lock:
      return [
          launched
          for launched in self.launched
          if launched.device.id == device_id and not launched._stopped
      ]

  def runners(self, device_id: str) -> list[FakeRunner]:
    """Returns every runner launched on a device, oldest first."""
    with self._lock:
      return [
          launched.runner
          for launched in self.launched
          if launched.device.id == device_id
      ]


class World:
  """Every store an Executions needs, in a temporary workspace."""

  def __init__(
      self, root: pathlib.Path, devices: list[launchers.Device]
  ) -> None:
    """Makes the stores under `root`, offering `devices`."""
    self.workspace = workspace.Workspace(root / 'workspace')
    self.models = models.ModelStore(self.workspace)
    path = root / 'gemma.litertlm'
    path.write_bytes(MODEL_BYTES)
    self.model = self.models.register(path)
    self.sessions = sessions.SessionStore(self.workspace, self.models)
    self.jobs = jobs.Jobs()
    self.launchers = Launchers(devices)
    self.executions = execution.Executions(
        self.sessions, self.models, self.jobs, self.launchers, self.workspace
    )

  def config(self, ref: str = 'local', target: str = 'local') -> dict:
    """Returns what the UI's editor posts for a Session on two devices."""
    return {
        'name': 'CPU vs CPU',
        'model': 'Gemma',
        'runs': [
            {
                'id': side,
                'runtime': 'LiteRT-LM',
                'artifact': self.model.artifact,
                'source': 'registered',
                'device': device,
                'backend': 'CPU',
                'contextLength': str(capture_fixtures.CONTEXT_LENGTH),
            }
            for side, device in (('ref', ref), ('target', target))
        ],
    }

  def create(self, ref: str = 'local', target: str = 'local') -> dict:
    """Creates a Session on two devices."""
    return self.sessions.create(self.config(ref, target))

  def wait(self, job: dict) -> dict:
    """Returns the job once it ended."""
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
      job = self.jobs.get(job['id'])
      if job['status'] in jobs.TERMINAL:
        return job
      time.sleep(0.01)
    raise AssertionError(f'job {job["id"]} did not end: {job}')

  def start(self, record: dict, request_id: str = 'start') -> dict:
    """Turns a Session's Model Server on; returns the completed job."""
    job = self.wait(self.executions.initialize(record['id'], request_id))
    assert job['status'] == 'completed', job['error']
    return job

  def turn(self, record_id: str, prompt: str, request_id: str) -> dict:
    """Sends a turn; returns its job once it ended."""
    return self.wait(self.executions.generate(record_id, request_id, prompt))

  def scripts(self, device_id: str = 'local') -> list[Script]:
    """The scripts of the runners running on a device now."""
    return [launched.script for launched in self.launchers.running(device_id)]

  def events(self, job_id: str) -> list[dict]:
    """Returns every event of a job so far."""
    return self.jobs._events[job_id]

  def close(self) -> None:
    """Stops every runner and releases the workspace."""
    self.executions.shutdown()
    self.workspace.close()
