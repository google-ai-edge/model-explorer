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
- it serves any connection;
- Load once; NOT_FOUND until the model's sha256 is in its store;
- a Load on the GPU, where GetInfo reports none, refused as the real
  runtime's CanRunOnGpu refuses it, before the options and the store;
- CreateChat after Load, with ChatOptions in their ranges, as often as
  asked: each replaces the chat, the one before closed and its unreleased
  captures deleted, and the new chat's turns count from 1; refused with
  FAILED_PRECONDITION during a Send;
- runtime_options as the runtime's schemas in GetInfo allow them, refused as
  the real runner's ValidateOptions (options_schema.cc) refuses them;
- one Load or CreateChat at a time;
- a Load that fails, other than with NOT_FOUND, or is cancelled, is final:
  its result is discarded and any later one refused; a CreateChat that
  fails or is cancelled leaves the runner with no chat, until another one
  succeeds;
- one Send at a time; a Send that fails other than INVALID_ARGUMENT, or is
  cancelled, ends the chat;
- a turn's capture is streamed file by file, by name, and released by its
  number;
- a runner of PYTORCH_INFO (the PyTorch runner) refuses every upload with
  FAILED_PRECONDITION: its store is the model directory it was started
  with.

What a turn replies and captures comes from a callback,
`respond(turn, prompt) -> (chunks, completion, files)`; it may raise
`AbortError`
to fail the Send, and `chunks` may be a generator that blocks, to hold a
Send open. Any RPC goes through `before(name)`: it may raise `AbortError` to
fail
the RPC, or block to stall it (as a slow load); on Load and CreateChat it
runs where the runtime does, after the refusals for the runner's state and
the request, and its AbortError is the runtime's failure. Each message of a
capture's stream goes through `before_chunk(chunk)` the same way. stop() is
the runner's process exiting: calls in flight fail with UNAVAILABLE.
drop_connections() is a lost connection.

Usage:
  with FakeRunner(respond, store={sha256}) as runner:
    ... runner.address ...
"""

from collections.abc import Callable, Iterable, Iterator
import concurrent.futures
import decimal
import hashlib
import json
import math
import re
import threading
import time
from typing import Any

import grpc
from model_debugger.protocol import runner_pb2
from model_debugger.protocol import runner_pb2_grpc
# pyformat: disable
from google.protobuf import struct_pb2
# pyformat: enable

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
    protocol_version=runner_pb2.DESCRIPTOR.GetOptions().Extensions[
        runner_pb2.protocol_version
    ],
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

# The options schemas of the PyTorch runner (src/runner_pytorch).
PYTORCH_LOAD_OPTIONS_SCHEMA = json.dumps({
    'type': 'object',
    'properties': {
        'cpuThreads': {'type': 'integer', 'minimum': 1},
        'dtype': {'enum': ['default', 'float32', 'float16', 'bfloat16']},
        'observe': {'enum': ['none', 'anchors']},
        'observeMemoryLimitBytes': {'type': 'integer', 'minimum': 1},
        'observeModules': {'type': 'array'},
    },
    'additionalProperties': False,
})
PYTORCH_CHAT_OPTIONS_SCHEMA = json.dumps(
    {'type': 'object', 'properties': {}, 'additionalProperties': False}
)
# What a PyTorch runner answers GetInfo with on the same Mac (MPS).
PYTORCH_INFO = runner_pb2.RunnerInfo()
PYTORCH_INFO.CopyFrom(DEFAULT_INFO)
PYTORCH_INFO.runtime.CopyFrom(
    runner_pb2.RuntimeInfo(
        name=runner_pb2.RuntimeInfo.PYTORCH,
        version='2.9.0',
        load_options_schema=PYTORCH_LOAD_OPTIONS_SCHEMA,
        chat_options_schema=PYTORCH_CHAT_OPTIONS_SCHEMA,
    )
)
# The PyTorch runner's refusal of an upload: it reads its model in place.
NO_UPLOAD = (
    'This runner reads the model directory it was started with; it takes '
    'no upload.'
)

# What the real runner's Load answered for Gemma 4 E2B, which reports no
# sampling of its own (sampler_defaults unset; LiteRT-LM 61dcfc77). A test
# whose model reports some passes its own load_response.
DEFAULT_LOAD_RESPONSE = runner_pb2.LoadResponse(
    start_token=2,
    stop_tokens=[
        runner_pb2.StopToken(ids=runner_pb2.TokenIds(ids=[1])),
        runner_pb2.StopToken(ids=runner_pb2.TokenIds(ids=[50])),
        runner_pb2.StopToken(ids=runner_pb2.TokenIds(ids=[106])),
    ],
)

# Where a fake runner's LiteRT-LM library is, as its GPU refusal names it.
LIBRARY_DIRECTORY = '/fake/runner/lib'
# What the real runner's CanRunOnGpu says on the iOS simulator.
IOS_SIMULATOR_GPU_REFUSAL = (
    "The iOS simulator's Metal lacks what LiteRT-LM's GPU backend needs "
    '(argument buffers, and textures bound past index 30); load the model '
    'on the CPU.'
)

_SHA256 = re.compile(r'[0-9a-f]{64}')
_CAPTURE_CHUNK_BYTES = 1 << 20
_FAILED = grpc.StatusCode.FAILED_PRECONDITION
_INVALID = grpc.StatusCode.INVALID_ARGUMENT
_INTERNAL = grpc.StatusCode.INTERNAL
# The refusal of a Load or a CreateChat while one runs.
_PREPARING = 'A Load or CreateChat is in progress.'
# The refusal of a Load after one failed.
_LOAD_FAILED = 'A Load failed; a runner loads one model, once.'


class AbortError(Exception):
  """A status to fail an RPC with, as `respond` and `before` raise it."""

  def __init__(self, code: grpc.StatusCode, message: str) -> None:
    """Initializes the status the RPC fails with."""
    super().__init__(message)
    self.code = code
    self.message = message


class NoRoomError(AbortError):
  """A turn the context has no room for, as `respond` refuses it.

  The Send fails with RESOURCE_EXHAUSTED and the two counts as trailing
  metadata; the chat goes on. Raised by `before('CreateChat')`, it refuses
  the chat's start the same way.
  """

  def __init__(self, tokens: int, left: int) -> None:
    """Initializes the refusal of a turn of `tokens` with `left` left."""
    super().__init__(
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        f'This turn needs {tokens} tokens; the context has {left} left.',
    )
    self.tokens = tokens
    self.left = left


def _no_room_metadata(
    context: grpc.ServicerContext, refused: NoRoomError
) -> None:
  """Gives the call the trailing metadata of `refused`, as a runner does."""
  context.set_trailing_metadata((
      ('turn-tokens', str(refused.tokens)),
      ('tokens-left', str(refused.left)),
  ))


def no_reply(turn: int, prompt: str) -> Reply:
  """A `respond` for a runner that is never sent a turn."""
  raise AssertionError('No Send is expected.')


class FakeRunner(runner_pb2_grpc.RunnerServicer):
  """A Runner on 127.0.0.1, in this process.

  Attributes:
      address: 'host:port' it listens on, once started.
      store: The sha256 of the models it holds; runners may share one set.
      info: What GetInfo answers.
      stopped: Whether stop() was called: the runner's process exited.
      before: If set, called with each RPC's name, outside the lock; it may
        raise AbortError to fail the RPC, or block to stall it. A Send then
        follows the Send's rules. On a Load or a CreateChat it is the
        runtime: it runs once the runner's refusals have passed, and its
          AbortError is the runtime's failure: a Load's is final (its NOT_FOUND
          becomes INTERNAL, as the model is in the store), a CreateChat's leaves
          no chat.
      before_chunk: If set, called with each CaptureChunk ReadCaptures is about
        to send, outside the lock; it may raise AbortError to fail the stream
        there, or block to stall it.
      peers: The peer of every RPC, in order.
      uploads: The sha256 of every model uploaded, in order.
      upload_spans: (sha256, start, end) of every UploadModel that sent a
        header, stored or not, in time.monotonic().
      deadlines: The seconds each RPC had left when it arrived, by its name;
        about 9.2e18 for one without a deadline.
      load_request: The Load that succeeded, if any.
      chat_request: The CreateChat of the chat the runner holds, if any.
      chats: Every CreateChat that succeeded, in order.
      prompts: The content of every completed turn, in order, of every chat.
      captures: The files of each completed turn of the chat not yet released,
        by turn.
      released: The turns released, in order, of every chat.
      chat_over: Whether a cancelled or failed Send ended the chat.
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
    self.info = info
    self.stopped = False
    self.before: Callable[[str], None] | None = None
    self.before_chunk: Callable[[runner_pb2.CaptureChunk], None] | None = None
    self.peers: list[str] = []
    self.uploads: list[str] = []
    self.upload_spans: list[tuple[str, float, float]] = []
    self.deadlines: dict[str, float] = {}
    self.load_request: runner_pb2.LoadRequest | None = None
    self.chat_request: runner_pb2.CreateChatRequest | None = None
    self.chats: list[runner_pb2.CreateChatRequest] = []
    self.prompts: list[str] = []
    self.captures: dict[int, dict[str, bytes]] = {}
    self.released: list[int] = []
    self.chat_over = False
    self._respond = respond
    self._load_response = load_response
    self._lock = threading.Lock()
    self._sending = False
    # A Load or a CreateChat is past its refusals, in the runtime.
    self._preparing = False
    self._turns = 0
    # A Load failed: any later one is refused.
    self._load_failed = False
    self._server: grpc.Server | None = None

  def start(self, port: int = 0, host: str = '127.0.0.1') -> 'FakeRunner':
    """Serves on `port` of `host`, or a free one; returns self.

    An IPv6 host is bracketed: '[::1]'.

    Args:
      port: Port to serve on; 0 picks a free port.
      host: Host address to serve on.

    Returns:
      This FakeRunner instance.
    """
    self._server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=8)
    )
    runner_pb2_grpc.add_RunnerServicer_to_server(self, self._server)
    # pylint: disable=line-too-long
    port = self._server.add_insecure_port(f'{host}:{port}')
    # pylint: enable=line-too-long
    self._server.start()
    self.address = f'{host}:{port}'
    return self

  def stop(self) -> None:
    """Stops serving at once, as the runner's process exiting would.

    Calls in flight fail with UNAVAILABLE; the client's channel leaves
    READY.
    """
    self.stopped = True
    if self._server is not None:
      self._server.stop(grace=None).wait()

  def drop_connections(self) -> None:
    """Closes every connection and serves on at the same address.

    As a lost transport would: calls in flight fail; the state stays.
    """
    self._server.stop(grace=None).wait()
    host, port = self.address.rsplit(':', 1)
    self.start(int(port), host)

  def __enter__(self) -> 'FakeRunner':
    """Starts the runner."""
    return self.start()

  def __exit__(self, *exc_info: object) -> None:
    """Stops the runner."""
    self.stop()

  # ---- The protocol

  def GetInfo(self, request, context):  # noqa: D102
    with self._lock:
      self._note(context)
    self._before(context, 'GetInfo')
    return self.info

  def UploadModel(self, request_iterator, context):  # noqa: D102
    started = time.monotonic()
    with self._lock:
      self._note(context)
    self._before(context, 'UploadModel')
    if self.info.runtime.name == runner_pb2.RuntimeInfo.PYTORCH:
      context.abort(_FAILED, NO_UPLOAD)
    first = next(request_iterator, None)
    if first is None or not first.HasField('header'):
      context.abort(_INVALID, 'The stream does not start with a header.')
    header = first.header
    if not _SHA256.fullmatch(header.sha256):
      context.abort(_INVALID, 'sha256 is not 64 lower-case hexadecimal digits.')
    if header.size < 0:
      context.abort(_INVALID, 'size is negative.')
    try:
      return self._upload(header, request_iterator, context)
    finally:
      with self._lock:
        self.upload_spans.append((header.sha256, started, time.monotonic()))

  def _upload(
      self,
      header: runner_pb2.UploadModelHeader,
      request_iterator: Iterable[runner_pb2.UploadModelRequest],
      context: grpc.ServicerContext,
  ) -> runner_pb2.UploadModelResponse:
    """Stores the model the stream after `header` carries."""
    with self._lock:
      if header.sha256 in self.store:
        return runner_pb2.UploadModelResponse()  # the stream unread
    digest = hashlib.sha256()
    received = 0
    for request in request_iterator:
      if not request.HasField('data'):
        context.abort(_INVALID, 'A second header arrived in the stream.')
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
      self._note(context)
      self.deadlines['Load'] = context.time_remaining()
      # Refusals for the state, which change nothing.
      if self.load_request is not None:
        context.abort(_FAILED, 'A model is loaded; a runner loads one.')
      if self._preparing:
        context.abort(_FAILED, _PREPARING)
      if self._load_failed:
        context.abort(_FAILED, _LOAD_FAILED)
      # Any failure from here on but NOT_FOUND is final.
      self._load_failed = True
      if not _SHA256.fullmatch(request.sha256):
        context.abort(
            _INVALID, 'sha256 is not 64 lower-case hexadecimal digits.'
        )
      if request.context_length <= 0:
        context.abort(
            _INVALID,
            'LoadRequest.context_length is '
            f'{request.context_length}; it must be positive.',
        )
      if request.device not in (runner_pb2.CPU, runner_pb2.GPU):
        context.abort(_INVALID, 'LoadRequest.device is not CPU or GPU.')
      # Before the store, as what follows: a Load that would be refused
      # is worth no upload.
      has_gpu = self.info.host.HasField('gpu')
      if request.device == runner_pb2.GPU and not has_gpu:
        context.abort(_FAILED, _gpu_refusal(self.info.host))
      _refuse(
          context,
          validate_options(
              self.info.runtime.load_options_schema,
              request.runtime_options,
              'LoadRequest.runtime_options',
          ),
      )
      if request.sha256 not in self.store:
        self._load_failed = False  # It asks for an upload.
        context.abort(
            grpc.StatusCode.NOT_FOUND,
            f'No model with sha256 {request.sha256} in the model store.',
        )
      self._preparing = True
    failure = self._runtime('Load')
    with self._lock:
      self._preparing = False
      if not context.is_active():
        context.abort(
            grpc.StatusCode.CANCELLED,
            'The Load was cancelled; the model is discarded.',
        )
      if failure is not None:
        # The model is in the store; what the runtime misses is the
        # runner's.
        code = failure.code
        if code == grpc.StatusCode.NOT_FOUND:
          code = _INTERNAL
        context.abort(code, failure.message)
      self._load_failed = False
      self.load_request = request
    return self._load_response

  def CreateChat(self, request, context):  # noqa: D102
    with self._lock:
      self._note(context)
      self.deadlines['CreateChat'] = context.time_remaining()
      # Refusals for the state, which change nothing.
      if self._preparing:
        context.abort(_FAILED, _PREPARING)
      if self.load_request is None:
        context.abort(_FAILED, 'No model is loaded.')
      if self._sending:
        context.abort(_FAILED, 'A Send is in progress.')
      # The chat it holds is closed, and its captures not released go:
      # a CreateChat that fails from here on leaves no chat.
      self.chat_request = None
      self.chat_over = False
      self.captures.clear()
      self._turns = 0
      refusal = _chat_options_refusal(request.options)
      if refusal:
        context.abort(_INVALID, refusal)
      _refuse(
          context,
          validate_options(
              self.info.runtime.chat_options_schema,
              request.options.runtime_options,
              'ChatOptions.runtime_options',
          ),
      )
      self._preparing = True
    # LiteRT-LM checks the messages before anything else.
    failure = _messages_refusal(request.messages) or self._runtime('CreateChat')
    with self._lock:
      self._preparing = False
      if not context.is_active():
        context.abort(
            grpc.StatusCode.CANCELLED,
            'The CreateChat was cancelled; the chat is discarded.',
        )
      if failure is not None:
        if isinstance(failure, NoRoomError):
          _no_room_metadata(context, failure)
        context.abort(failure.code, failure.message)
      self.chat_request = request
      self.chats.append(request)
    return runner_pb2.CreateChatResponse()

  def Send(self, request, context):  # noqa: D102
    with self._lock:
      self._note(context)
      if self.chat_request is None:
        context.abort(_FAILED, 'No chat exists.')
      if self.chat_over:
        context.abort(
            _FAILED,
            'The chat has ended with a cancelled or failed turn; a '
            'CreateChat starts another.',
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
        self.deadlines['Send'] = context.time_remaining()
        if self.before is not None:
          self.before('Send')
        chunks, completion, files = self._respond(turn, request.message.content)
        for chunk in chunks:
          if not context.is_active():
            break
          yield runner_pb2.SendResponse(chunk=chunk)
      except AbortError as abort:
        refused = abort.code == _INVALID or isinstance(abort, NoRoomError)
        if isinstance(abort, NoRoomError):
          _no_room_metadata(context, abort)
        context.abort(abort.code, abort.message)
      if not context.is_active():
        return
      with self._lock:
        self._turns = turn
        self.captures[turn] = dict(files)
        self.prompts.append(request.message.content)
      completed = True
      yield runner_pb2.SendResponse(completion=completion)
    finally:
      # Cancelled, failed, or abandoned by its client at a yield.
      with self._lock:
        self._sending = False
        if not completed and not refused:
          self.chat_over = True

  def ReadCaptures(self, request, context):  # noqa: D102
    with self._lock:
      self._note(context)
    self._before(context, 'ReadCaptures')
    with self._lock:
      files = dict(self._capture(request.turn, context))
    for chunk in _capture_chunks(files):
      if self.before_chunk is not None:
        try:
          self.before_chunk(chunk)
        except AbortError as abort:
          context.abort(abort.code, abort.message)
      yield chunk

  def ReleaseCaptures(self, request, context):  # noqa: D102
    with self._lock:
      self._note(context)
    self._before(context, 'ReleaseCaptures')
    with self._lock:
      self._capture(request.turn, context)
      del self.captures[request.turn]
      self.released.append(request.turn)
    return runner_pb2.ReleaseCapturesResponse()

  # ---- Rules

  def _note(self, context: grpc.ServicerContext) -> None:
    """Notes the RPC's peer; called with the lock held."""
    self.peers.append(context.peer())

  def _before(self, context: grpc.ServicerContext, method: str) -> None:
    """Runs `before` for an RPC; an AbortError it raises fails the RPC."""
    self.deadlines[method] = context.time_remaining()
    if self.before is None:
      return
    try:
      self.before(method)
    except AbortError as abort:
      context.abort(abort.code, abort.message)

  def _runtime(self, method: str) -> AbortError | None:
    """Runs `before` as the runtime of a Load or a CreateChat.

    Called without the lock, as the real runner runs its runtime, while
    `_preparing` holds off another Load or CreateChat.

    Args:
        method: 'Load' or 'CreateChat'.

    Returns:
        The AbortError `before` raised, the runtime's failure, or None.
    """
    try:
      if self.before is not None:
        self.before(method)
    except AbortError as abort:
      return abort
    except BaseException:
      with self._lock:
        self._preparing = False
      raise
    return None

  def _capture(
      self, turn: int, context: grpc.ServicerContext
  ) -> dict[str, bytes]:
    """Returns a turn's files; called with the lock held."""
    if turn not in self.captures:
      context.abort(grpc.StatusCode.NOT_FOUND, f'Turn {turn} has no capture.')
    return self.captures[turn]


def _capture_chunks(
    files: dict[str, bytes],
) -> Iterator[runner_pb2.CaptureChunk]:
  """Yields a capture's stream: by name, each file, its data, its sha256."""
  for name, data in sorted(files.items()):
    yield runner_pb2.CaptureChunk(
        file=runner_pb2.CaptureFile(path=name, size=len(data))
    )
    for start in range(0, len(data), _CAPTURE_CHUNK_BYTES):
      yield runner_pb2.CaptureChunk(
          data=data[start : start + _CAPTURE_CHUNK_BYTES]
      )
    yield runner_pb2.CaptureChunk(sha256=hashlib.sha256(data).hexdigest())


def overlapping_uploads(runners: Iterable['FakeRunner']) -> list[str]:
  """Returns the sha256 of each model two of `runners` received at once."""
  all_spans = []
  for runner in runners:
    all_spans.extend(runner.upload_spans)
  spans = sorted(all_spans)
  return [
      curr_span[0]
      for curr_span, next_span in zip(spans, spans[1:])
      if curr_span[0] == next_span[0] and next_span[1] < curr_span[2]
  ]


def _chat_options_refusal(options: runner_pb2.ChatOptions) -> str:
  """Why the runner refuses ChatOptions out of range, or '' if it does not.

  As the real runner's CheckChatOptions (runner_service.cc), NaN included.

  Args:
    options: The chat options to check.

  Returns:
    An error message if the options are refused, or '' if allowed.
  """
  if options.max_completion_tokens < 1:
    return (
        'ChatOptions.max_completion_tokens is '
        f'{options.max_completion_tokens}; it must be at least 1.'
    )
  # Written so that NaN, which compares false, is refused too.
  if options.HasField('temperature') and not options.temperature >= 0:
    return (
        f'ChatOptions.temperature is {options.temperature:g}; it must be '
        'at least 0.'
    )
  if options.HasField('top_p') and not 0 <= options.top_p <= 1:
    return f'ChatOptions.top_p is {options.top_p:g}; it must be from 0 to 1.'
  if options.HasField('top_k') and options.top_k < 1:
    return f'ChatOptions.top_k is {options.top_k}; it must be at least 1.'
  return ''


def _gpu_refusal(host: runner_pb2.HostInfo) -> str:
  """Why a runner that reports no GPU refuses a Load on the GPU.

  The real runner reports a GPU only where its runtime's CanRunOnGpu
  (litert_lm_runtime.cc) passes, and refuses a GPU Load with what that
  says: on the iOS simulator, which reports no GPU, its Metal; elsewhere,
  the GPU libraries it misses.

  Args:
    host: Host information reported by the runner.

  Returns:
    An error message describing why GPU loading is refused.
  """
  if host.platform == runner_pb2.HostInfo.IOS:
    return IOS_SIMULATOR_GPU_REFUSAL
  return (
      'The GPU libraries are not beside the LiteRT-LM library, in '
      f'{LIBRARY_DIRECTORY}.'
  )


def _messages_refusal(
    messages: Iterable[runner_pb2.ChatMessage],
) -> AbortError | None:
  """The runtime's refusal of a chat's messages, as LiteRT-LM's, or None."""
  for i, message in enumerate(messages):
    if message.role == runner_pb2.ChatMessage.SYSTEM and i != 0:
      return AbortError(
          _INVALID, 'A system message may only be the first message.'
      )
  return None


def _refuse(context: grpc.ServicerContext, refusal: AbortError | None) -> None:
  """Fails the RPC with `refusal`, if there is one."""
  if refusal is not None:
    context.abort(refusal.code, refusal.message)


# ---- The real runner's ValidateOptions (options_schema.cc)
#
# Values are JSON as nlohmann::json holds them: a Struct's numbers are all
# floats, a schema's whole numbers ints; objects are read in key order.

# The types the keyword "type" may name.
_JSON_TYPES = (
    'null',
    'boolean',
    'object',
    'array',
    'number',
    'string',
    'integer',
)


def validate_options(
    schema: str, options: struct_pb2.Struct, field: str
) -> AbortError | None:
  """Why the real runner refuses runtime options, or None if it does not.

  As its ValidateOptions: the schema must be JSON and use only the keywords
  it checks; then every number must be finite; then the options must match
  the schema, the first mismatch in key order answered.

  Args:
      schema: The runtime's JSON Schema for the options, as GetInfo has it.
      options: The options a request carries.
      field: The request's field, which names the options in messages, as
        'LoadRequest.runtime_options'.

  Returns:
      INTERNAL for a schema it cannot check, INVALID_ARGUMENT for options
      the schema does not allow, or None.
  """
  try:
    parsed = json.loads(schema, parse_constant=_no_constant)
  except ValueError:
    return AbortError(
        _INTERNAL, f"The runtime's schema for {field} is not JSON."
    )
  value = _struct_json(options)
  return (
      _check_keywords(parsed, '#')
      or _check_finite(value, field)
      or _check(parsed, value, field)
  )


def _no_constant(name: str) -> None:
  """Refuses NaN and Infinity, which JSON has not."""
  raise ValueError(f'{name} is not JSON.')


def _value_json(value: struct_pb2.Value) -> object:
  """A Value as JSON: None, bool, float, str, list or dict."""
  kind = value.WhichOneof('kind')
  if kind == 'bool_value':
    return value.bool_value
  if kind == 'number_value':
    return value.number_value
  if kind == 'string_value':
    return value.string_value
  if kind == 'struct_value':
    return _struct_json(value.struct_value)
  if kind == 'list_value':
    return [_value_json(item) for item in value.list_value.values]
  return None  # A null, or a value of no kind, which JSON writes as null.


def _struct_json(struct: struct_pb2.Struct) -> dict[str, object]:
  """A Struct as a JSON object."""
  return {name: _value_json(value) for name, value in struct.fields.items()}


def _is_number(value: object) -> bool:
  """Whether a JSON value is a number; a bool is not."""
  return isinstance(value, (int, float)) and not isinstance(value, bool)


def _show(value: object) -> str:
  """A JSON value as the real runner writes it in a message: compact."""
  if value is None:
    return 'null'
  if isinstance(value, bool):
    return 'true' if value else 'false'
  if isinstance(value, int):
    return str(value)
  if isinstance(value, float):
    return _show_float(value)
  if isinstance(value, str):
    return json.dumps(value, ensure_ascii=False)
  if isinstance(value, list):
    return '[' + ','.join(_show(item) for item in value) + ']'
  return (
      '{'
      + ','.join(f'{_show(k)}:{_show(v)}' for k, v in sorted(value.items()))
      + '}'
  )


def _show_float(number: float) -> str:
  """A finite float as nlohmann::json writes it.

  The shortest digits that read back as the number, fixed for a decimal
  point from 10^-4 to 10^15, else as an exponent; a whole number keeps
  '.0'. 2.0 is '2.0', 1e15 '1e+15', 1e-5 '1e-05'.

  Args:
    number: The finite float to format.

  Returns:
    The formatted string representation of the float.
  """
  if number == 0:
    return '-0.0' if math.copysign(1, number) < 0 else '0.0'
  sign = '-' if number < 0 else ''
  exact = decimal.Decimal(repr(abs(number))).normalize().as_tuple()
  digits = ''.join(str(digit) for digit in exact.digits)
  # The number is 0.digits * 10^point.
  max_decimal_point = 15
  min_decimal_point = -4
  k, point = len(digits), len(digits) + exact.exponent
  if k <= point <= max_decimal_point:
    text = digits + '0' * (point - k) + '.0'
  elif 0 < point <= max_decimal_point:
    text = f'{digits[:point]}.{digits[point:]}'
  elif min_decimal_point < point <= 0:
    text = '0.' + '0' * -point + digits
  else:
    mantissa = digits if k == 1 else f'{digits[0]}.{digits[1:]}'
    exponent = point - 1
    text = f'{mantissa}e{"-" if exponent < 0 else "+"}{abs(exponent):02d}'
  return sign + text


def _not_understood(where: str, what: str) -> AbortError:
  """The refusal of a schema the runner does not check."""
  return AbortError(
      _INTERNAL,
      f"The runtime's options schema {what} at {where}, which the runner "
      'does not check.',
  )


def _is_type_name(name: object) -> bool:
  """Whether a value names a JSON Schema type."""
  return isinstance(name, str) and name in _JSON_TYPES


def _check_keywords(schema: object, where: str) -> AbortError | None:
  """Refuses a schema that uses a keyword, or a form of one, it cannot check.

  The schema is found at `where`; a schema inside it is checked too.

  Args:
    schema: The schema object to check.
    where: Path/location description of the schema.

  Returns:
    An AbortError if an unsupported keyword is found, or None if valid.
  """
  if not isinstance(schema, dict):
    return _not_understood(where, 'is not an object')
  for keyword, value in sorted(schema.items()):
    at = f'{where}/{keyword}'
    if keyword == 'type':
      understood = _is_type_name(value) or (
          isinstance(value, list) and all(map(_is_type_name, value))
      )
      if not understood:
        return _not_understood(at, 'names no type it knows')
    elif keyword == 'properties':
      if not isinstance(value, dict):
        return _not_understood(at, 'is not an object')
      for name, property_schema in sorted(value.items()):
        refusal = _check_keywords(property_schema, f'{at}/{name}')
        if refusal:
          return refusal
    elif keyword == 'required':
      if not isinstance(value, list) or not all(
          isinstance(name, str) for name in value
      ):
        return _not_understood(at, 'is not a list of names')
    elif keyword == 'additionalProperties':
      if isinstance(value, bool):
        continue
      refusal = _check_keywords(value, at)
      if refusal:
        return refusal
    elif keyword == 'enum':
      if not isinstance(value, list):
        return _not_understood(at, 'is not a list')
    elif keyword in ('minimum', 'maximum'):
      if not _is_number(value):
        return _not_understood(at, 'is not a number')
    else:
      return _not_understood(at, 'uses a keyword')
  return None


def _check_finite(value: object, name: str) -> AbortError | None:
  """Refuses a value that is or holds a number that is not finite.

  JSON has no NaN or infinity, though a Struct, read from binary, may.

  Args:
    value: The JSON value to check.
    name: The name or path of the value.

  Returns:
    An AbortError if a non-finite number is found, or None if valid.
  """
  if isinstance(value, float) and not math.isfinite(value):
    return AbortError(_INVALID, f'{name} is not a finite number.')
  if isinstance(value, dict):
    for key, member in sorted(value.items()):
      refusal = _check_finite(member, f'{name}.{key}')
      if refusal:
        return refusal
  elif isinstance(value, list):
    for i, item in enumerate(value):
      refusal = _check_finite(item, f'{name}[{i}]')
      if refusal:
        return refusal
  return None


def _has_type(value: object, type_name: str) -> bool:
  """Whether a value is of a JSON Schema type; 2.0 is an integer."""
  if type_name == 'null':
    return value is None
  if type_name == 'boolean':
    return isinstance(value, bool)
  if type_name == 'object':
    return isinstance(value, dict)
  if type_name == 'array':
    return isinstance(value, list)
  if type_name == 'number':
    return _is_number(value)
  if type_name == 'string':
    return isinstance(value, str)
  return _is_number(value) and float(value).is_integer()  # 'integer'


def _json_equal(a: object, b: object) -> bool:
  """Whether two JSON values are equal as nlohmann::json compares them."""
  if isinstance(a, bool) or isinstance(b, bool):
    return type(a) is type(b) and a == b
  if _is_number(a) and _is_number(b):
    return a == b
  if isinstance(a, list) and isinstance(b, list):
    return len(a) == len(b) and all(map(_json_equal, a, b))
  if isinstance(a, dict) and isinstance(b, dict):
    return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
  return type(a) is type(b) and a == b


def _check(
    schema: dict[str, Any], value: object, name: str
) -> AbortError | None:
  """Refuses a value, named `name`, the schema does not allow.

  The schema's keywords are understood, and every number in `value` is
  finite.

  Args:
    schema: The schema definition.
    value: The value to validate.
    name: The name or path of the value being checked.

  Returns:
    An AbortError if validation fails, or None if valid.
  """
  if 'type' in schema:
    types = schema['type']
    names = [types] if isinstance(types, str) else types
    if not any(_has_type(value, each) for each in names):
      return AbortError(
          _INVALID,
          f'{name} is {_show(value)}; it must be of type {_show(types)}.',
      )
  if 'enum' in schema:
    if not any(_json_equal(each, value) for each in schema['enum']):
      return AbortError(
          _INVALID,
          f'{name} is {_show(value)}; it must be one of '
          f'{_show(schema["enum"])}.',
      )
  if _is_number(value):
    if 'minimum' in schema and value < schema['minimum']:
      return AbortError(
          _INVALID,
          f'{name} is {_show(value)}; it must be at least '
          f'{_show(schema["minimum"])}.',
      )
    if 'maximum' in schema and value > schema['maximum']:
      return AbortError(
          _INVALID,
          f'{name} is {_show(value)}; it must be at most '
          f'{_show(schema["maximum"])}.',
      )
  if not isinstance(value, dict):
    return None
  for member in schema.get('required', []):
    if member not in value:
      return AbortError(_INVALID, f'{name} lacks {member}, which is required.')
  properties = schema.get('properties')
  additional = schema.get('additionalProperties')
  for key, member in sorted(value.items()):
    if properties is not None and key in properties:
      refusal = _check(properties[key], member, f'{name}.{key}')
      if refusal:
        return refusal
      continue
    if additional is None or (isinstance(additional, bool) and additional):
      continue
    if isinstance(additional, bool) and not additional:
      return AbortError(
          _INVALID,
          f'{name} has {key}, which is not an option of this runtime.',
      )
    refusal = _check(additional, member, f'{name}.{key}')
    if refusal:
      return refusal
  return None
