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

"""The Runner service: the protocol's rules around the PyTorch runtime.

The rules, status codes and messages are the LiteRT runner's
(runner_service.cc), as the server's fake runner keeps them, but for the
model: this runner runs the one model directory it was started with, whose
id (model_id) is the only one Load finds, and takes no upload.

- Load once; a failure, or a cancel, is final, but NOT_FOUND.
- CreateChat after Load, with ChatOptions in their ranges; one chat at a
  time: it closes the chat there is first, deleting its captures, and a
  failure leaves none.
- One Load or CreateChat at a time, and one Send; no CreateChat during a
  Send.
- A Send that fails other than INVALID_ARGUMENT or FAILED_PRECONDITION, or
  is cancelled, ends the chat.
- A turn's capture is streamed file by file, by name, each with its sha256,
  and released by its number.
"""

from collections.abc import Callable
import hashlib
import math
import pathlib
import re
import shutil
import threading

import grpc
from model_debugger.protocol import runner_pb2
from model_debugger.protocol import runner_pb2_grpc
from model_debugger.runner_pytorch.llm import gemma4_runtime
from model_debugger.runner_pytorch.service import host_info
from model_debugger.runner_pytorch.service import options_schema

_SHA256 = re.compile(r'[0-9a-f]{64}')
_CAPTURE_CHUNK_BYTES = 1 << 20
_FAILED = grpc.StatusCode.FAILED_PRECONDITION
_INVALID = grpc.StatusCode.INVALID_ARGUMENT
_INTERNAL = grpc.StatusCode.INTERNAL
_CANCELLED = grpc.StatusCode.CANCELLED
_EXHAUSTED = grpc.StatusCode.RESOURCE_EXHAUSTED
_PREPARING = 'A Load or CreateChat is in progress.'
_LOAD_FAILED = 'A Load failed; a runner loads one model, once.'
_SENDING = 'A Send is in progress.'
_NO_UPLOAD = (
    'This runner loads the model it was started with; it takes no upload.'
)
_NO_GPU = 'PyTorch reports no GPU here: neither MPS nor CUDA is available.'


class RunnerService(runner_pb2_grpc.RunnerServicer):
  """Serves the protocol for one model directory."""

  def __init__(
      self,
      model_dir: pathlib.Path,
      model_id: Callable[[], str],
      data_dir: pathlib.Path,
  ) -> None:
    """Prepares the service.

    Args:
        model_dir: The Hugging Face model directory it runs.
        model_id: Returns the directory's id; called once, by the first Load
          that needs it. It may raise OSError or ValueError.
        data_dir: Where the turns' captures go, each in turn-<n>/; made.
    """
    self._model_dir = model_dir
    self._model_id = model_id
    self._data_dir = data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    self._gpu = host_info.gpu_device()
    self._info = runner_pb2.RunnerInfo(
        protocol_version=runner_pb2.DESCRIPTOR.GetOptions().Extensions[
            runner_pb2.protocol_version
        ],
        host=host_info.measure_host(),
        runtime=runner_pb2.RuntimeInfo(
            name=runner_pb2.RuntimeInfo.PYTORCH,
            version=gemma4_runtime.version(),
            load_options_schema=gemma4_runtime.LOAD_OPTIONS_SCHEMA,
            chat_options_schema=gemma4_runtime.CHAT_OPTIONS_SCHEMA,
        ),
    )
    self._lock = threading.Lock()
    self._id_lock = threading.Lock()
    self._id: str | None = None
    self._model: gemma4_runtime.Model | None = None
    self._chat: gemma4_runtime.Chat | None = None
    self._preparing = False
    self._load_failed = False
    self._sending = False
    self._chat_over = False
    self._turns = 0
    # The CreateChats begun, each closing the chat before: a
    # ReadCaptures that sees the count change finds its turn gone.
    self._chats = 0

  # ---- The protocol

  def GetInfo(self, request, context):  # noqa: D102
    return self._info

  def UploadModel(self, request_iterator, context):  # noqa: D102
    context.abort(_FAILED, _NO_UPLOAD)

  def Load(self, request, context):  # noqa: D102
    with self._lock:
      if self._model is not None:
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
      if request.device == runner_pb2.GPU and self._gpu is None:
        context.abort(_FAILED, _NO_GPU)
      _refuse(
          context,
          options_schema.validate_options(
              self._info.runtime.load_options_schema,
              request.runtime_options,
              'LoadRequest.runtime_options',
          ),
      )
      self._preparing = True
    # Without the lock: hashing and loading take seconds. A cancel does
    # not stop them; the model is then discarded.
    model, code, message = None, None, ''
    try:
      model_id = self._known_id()
      if request.sha256 != model_id:
        code = grpc.StatusCode.NOT_FOUND
        message = (
            f'No model with sha256 {request.sha256} in the model '
            f'store; this runner runs {model_id}.'
        )
      else:
        device = 'cpu' if request.device == runner_pb2.CPU else self._gpu
        model = gemma4_runtime.Model(self._model_dir, device, request)
    except Exception as error:  # pylint: disable=broad-exception-caught
      code, message = _INTERNAL, f'{type(error).__name__}: {error}'
    with self._lock:
      self._preparing = False
      if code == grpc.StatusCode.NOT_FOUND:
        self._load_failed = False  # It asks for the model.
    if code is not None:
      context.abort(code, message)
    with self._lock:
      if not context.is_active():
        context.abort(
            _CANCELLED,
            'The Load was cancelled; the model is discarded.',
        )
      self._load_failed = False
      self._model = model
    return model.info()

  def CreateChat(self, request, context):  # noqa: D102
    with self._lock:
      if self._preparing:
        context.abort(_FAILED, _PREPARING)
      if self._model is None:
        context.abort(_FAILED, 'No model is loaded.')
      if self._sending:
        context.abort(_FAILED, _SENDING)
      # The chat there is, if any, is closed whatever follows, its
      # cache freed before the next chat takes any memory.
      self._chat = None
      self._chat_over = False
      for turn in range(1, self._turns + 1):
        shutil.rmtree(self._data_dir / f'turn-{turn}', ignore_errors=True)
      self._turns = 0
      self._chats += 1
      refusal = _chat_options_refusal(request.options)
      if refusal:
        context.abort(_INVALID, refusal)
      _refuse(
          context,
          options_schema.validate_options(
              self._info.runtime.chat_options_schema,
              request.options.runtime_options,
              'ChatOptions.runtime_options',
          ),
      )
      try:
        chat = self._model.create_chat(request)
      except ValueError as error:
        context.abort(_INVALID, str(error))
      except gemma4_runtime.NoRoom as refused:
        _abort_no_room(context, refused)
      if not context.is_active():
        context.abort(
            _CANCELLED,
            'The CreateChat was cancelled; the chat is discarded.',
        )
      self._chat = chat
    return runner_pb2.CreateChatResponse()

  def Send(self, request, context):  # noqa: D102
    with self._lock:
      if self._chat is None:
        context.abort(_FAILED, 'No chat exists.')
      if self._chat_over:
        context.abort(
            _FAILED,
            'The chat has ended with a cancelled or failed turn; a '
            'CreateChat starts another.',
        )
      if self._sending:
        context.abort(_FAILED, _SENDING)
      if request.message.role != runner_pb2.ChatMessage.USER:
        context.abort(_INVALID, 'A turn is a user message.')
      self._sending = True
      turn = self._turns + 1
    directory = self._data_dir / f'turn-{turn}'
    completed = False
    # Set when the turn is refused for want of room: the chat goes on.
    refused: list[bool] = []
    chunks: list[str] = []
    try:
      # The model runs in this thread; its chunks are sent as they come.
      completion = None
      for item in self._run_turn(request, directory, context, chunks, refused):
        if isinstance(item, runner_pb2.Completion):
          completion = item
        else:
          yield runner_pb2.SendResponse(chunk=runner_pb2.Chunk(content=item))
      if not context.is_active():
        context.abort(_CANCELLED, 'The Send was cancelled.')
      with self._lock:
        self._turns = turn
      completed = True
      yield runner_pb2.SendResponse(completion=completion)
    finally:
      # Cancelled, failed, or abandoned by its client at a yield.
      with self._lock:
        self._sending = False
        if not completed:
          self._chat_over = self._chat_over or not refused
          self._turns = turn - 1
          shutil.rmtree(directory, ignore_errors=True)

  def _run_turn(self, request, directory, context, chunks, refused):
    """Yields a turn's chunks as the model makes them, then its completion.

    The model runs between yields, in the caller's thread. A turn the
    context has no room for is refused with RESOURCE_EXHAUSTED and the
    trailing metadata "turn-tokens" and "tokens-left", `refused` then
    set.

    Args:
      request: The send request.
      directory: Directory where the turn runs.
      context: The gRPC servicer context.
      chunks: List where chunks are collected.
      refused: List to record whether the turn was refused.
    """
    pending: list[str] = []
    completion = []
    error = []

    def run() -> None:
      try:
        completion.append(
            self._chat.send(
                request.message.content,
                directory,
                pending.append,
                lambda: not context.is_active(),
            )
        )
      except BaseException as raised:  # pylint: disable=broad-exception-caught
        error.append(raised)

    # The turn runs in a thread of its own, so its chunks go out while it
    # runs; this generator waits on it.
    done = threading.Event()
    worker = threading.Thread(
        target=lambda: (run(), done.set()), name='send', daemon=True
    )
    worker.start()
    sent = 0
    try:
      while True:
        finished = done.wait(0.05)
        while sent < len(pending):
          chunks.append(pending[sent])
          yield pending[sent]
          sent += 1
        if finished:
          break
    finally:
      # Closed at a yield by a cancel, the turn still runs until the
      # model sees it: the turn ends, and its directory goes, only after.
      worker.join()
    if error:
      raised = error[0]
      if isinstance(raised, gemma4_runtime.Cancelled):
        context.abort(_CANCELLED, 'The Send was cancelled.')
      if isinstance(raised, gemma4_runtime.NoRoom):
        refused.append(True)
        _abort_no_room(context, raised)
      context.abort(_INTERNAL, f'{type(raised).__name__}: {raised}')
    yield completion[0]

  def ReadCaptures(self, request, context):  # noqa: D102
    with self._lock:
      directory = self._turn_directory(request.turn, context)
      chats = self._chats
    names = sorted(path.name for path in directory.iterdir())
    for name in names:
      if not (directory / name).is_file():
        context.abort(_INTERNAL, f'{directory / name} is not a regular file.')
    for name in names:
      path = directory / name
      try:
        file = open(path, 'rb')
      except OSError:
        self._finish(request.turn, chats, context)
        context.abort(_INTERNAL, f'Cannot read {path}')
      with file:
        size = path.stat().st_size
        yield runner_pb2.CaptureChunk(
            file=runner_pb2.CaptureFile(path=name, size=size)
        )
        digest = hashlib.sha256()
        while data := file.read(_CAPTURE_CHUNK_BYTES):
          digest.update(data)
          yield runner_pb2.CaptureChunk(data=data)
        yield runner_pb2.CaptureChunk(sha256=digest.hexdigest())
    self._finish(request.turn, chats, context)

  def ReleaseCaptures(self, request, context):  # noqa: D102
    with self._lock:
      directory = self._turn_directory(request.turn, context)
      try:
        shutil.rmtree(directory)
      except OSError as error:
        context.abort(_INTERNAL, f'Cannot delete turn {request.turn}: {error}')
    return runner_pb2.ReleaseCapturesResponse()

  # ---- Rules

  def _known_id(self) -> str:
    """Returns the model directory's id, computing it once."""
    with self._id_lock:
      if self._id is None:
        self._id = self._model_id()
      return self._id

  def _turn_directory(
      self, turn: int, context: grpc.ServicerContext
  ) -> pathlib.Path:
    """Returns a completed turn's capture; called with the lock held."""
    directory = self._data_dir / f'turn-{turn}'
    if not 1 <= turn <= self._turns or not directory.is_dir():
      context.abort(grpc.StatusCode.NOT_FOUND, f'Turn {turn} has no capture.')
    return directory

  def _finish(
      self, turn: int, chats: int, context: grpc.ServicerContext
  ) -> None:
    """Fails a read whose turn was released, or its chat closed, meanwhile.

    Args:
        turn: The turn read.
        chats: self._chats when the read began.
        context: The read's.
    """
    with self._lock:
      if self._chats != chats:
        context.abort(grpc.StatusCode.NOT_FOUND, f'Turn {turn} has no capture.')
      self._turn_directory(turn, context)


def _abort_no_room(
    context: grpc.ServicerContext, refused: gemma4_runtime.NoRoom
) -> None:
  """Fails the RPC for want of room, the two counts as trailing metadata."""
  context.set_trailing_metadata((
      ('turn-tokens', str(refused.needed)),
      ('tokens-left', str(refused.left)),
  ))
  context.abort(_EXHAUSTED, str(refused))


def _refuse(
    context: grpc.ServicerContext, refusal: options_schema.Refusal | None
) -> None:
  """Fails the RPC with `refusal`, if there is one."""
  if refusal is not None:
    context.abort(refusal.code, refusal.message)


def _chat_options_refusal(options: runner_pb2.ChatOptions) -> str:
  """Why ChatOptions out of range are refused, or '' if they are not.

  As the LiteRT runner's CheckChatOptions (runner_service.cc), NaN
  included.

  Args:
    options: The chat options to validate.

  Returns:
    An error string describing why options were refused, or '' if valid.
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
  if options.HasField('temperature') and math.isinf(options.temperature):
    return 'ChatOptions.temperature is not finite.'
  return ''
