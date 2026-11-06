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

"""One runner, which holds one model for its whole life, and a chat at a time.

This module is the only reader of a runner's status codes: a call succeeds
or raises an Error. Once a chat exists, a failed call ends it, but for a
turn the runner refused as having no room in the context (NoRoomError); another
CreateChat replaces it. A CreateChat refused for the same want of room is
NoRoomToStartError.
"""

from collections.abc import Callable, Iterator
import pathlib
import threading
import time
import types
from typing import Any

import grpc
from model_debugger.protocol import runner_pb2
from model_debugger.protocol import runner_pb2_grpc
from model_debugger.server.runner import capture_writer
from model_debugger.server.runner import launcher

# The protocol version this server speaks, the proto's file option; a runner
# must speak the same.
PROTOCOL_VERSION = runner_pb2.DESCRIPTOR.GetOptions().Extensions[
    runner_pb2.protocol_version
]

_MIB = 1024 * 1024

# Most bytes in one UploadModelRequest data message (the protocol's limit).
_UPLOAD_CHUNK_BYTES = _MIB

# How long a connection may take to come back before its runner counts as
# gone. gRPC closes a connection idle for a while (30 min by default), and any
# connection may drop; the channel then reconnects (IDLE, then CONNECTING).
# That takes about 0.2 s on this computer, gRPC's Python polling the channel
# every 0.2 s; an adb forward or a device tunnel adds a few round trips over
# its link. A runner gone usually fails the reconnect at once
# (TRANSIENT_FAILURE): nothing listens, or the forward closes the connection.
# A device that vanished behind a tunnel may leave the connect hanging until
# gRPC gives up, 20 s later: this bounds that wait to the UI's 3 s refresh.
_RECONNECT_SECONDS = 3.0

# Deadlines in seconds, by RPC; Send, UploadModel and ReadCaptures take none.
_TIMEOUT_SECONDS = {
    'GetInfo': 10,
    'Load': 600,
    'CreateChat': 60,
    'ReleaseCaptures': 60,
}


# The runtimes this server drives.
_RUNTIMES = (runner_pb2.RuntimeInfo.LITERT_LM, runner_pb2.RuntimeInfo.PYTORCH)


class Error(Exception):
  """A runner call that did not succeed; `str(error)` says why."""


class NoRoomError(Error):
  """A turn the runner refused: it needs more tokens than the context has.

  Nothing changed: the chat takes the next turn as if this one had not been
  sent.

  Attributes:
      tokens: The tokens the turn needs.
      left: The tokens the context has left.
  """

  def __init__(self, tokens: int, left: int) -> None:
    """Holds the two counts; the text says them."""
    super().__init__(
        f'This turn needs {tokens} tokens, the chat has {left} left.'
    )
    self.tokens = tokens
    self.left = left


class NoRoomToStartError(Error):
  """A chat the runner refused: its system prompt does not fit the context.

  The runner counts the system prompt in the chat template, as the first
  turn's prompt for an empty message, so the count holds the template's own
  tokens too. The runner holds no chat.

  Attributes:
      tokens: The tokens the start needs.
      left: The tokens the context holds.
  """

  def __init__(self, tokens: int, left: int) -> None:
    """Holds the two counts; the text says them."""
    super().__init__(
        f'The system prompt and the chat template around it need {tokens} '
        f'tokens; the context holds {left}. Shorten it and start again.'
    )
    self.tokens = tokens
    self.left = left


class RunnerClient:
  """The server's side of one runner, made by RunnerClient.connect().

  Then load, create_chat, and per turn send and take_capture, one call at a
  time; create_chat again starts another chat in place of the last.
  cancel(), close() and online() may come from any thread.

  Attributes:
      info: What the runner answered GetInfo with.
  """

  def __init__(self, launched: launcher.LaunchedRunner) -> None:
    """Opens a channel to a runner; use connect().

    Args:
        launched: The runner, serving.
    """
    self.info = runner_pb2.RunnerInfo()
    self._launched = launched
    self._device = launched.device
    self._lock = threading.Lock()  # Guards the seven fields below.
    self._closed = False
    # Whether the chat's turn was stopped (cancel()), until the next chat.
    self._cancelled = False
    # The Send or ReadCaptures in flight.
    self._reply: grpc.Call | None = None
    # The channel's state, as its subscription last said.
    self._connectivity = grpc.ChannelConnectivity.IDLE
    # When the channel was last READY or left it (time.monotonic()), and
    # whether a connect failed since: what online() reads while it is not.
    self._unready_since = time.monotonic()
    self._failed = False
    # What asks the channel to reconnect, once it went idle.
    self._reconnect: grpc.Future | None = None
    # No keepalive pings: the runner's gRPC server closes a connection
    # that pings while idle. Completions may exceed 4 MiB.
    self._channel = grpc.insecure_channel(
        launched.address,
        options=[('grpc.max_receive_message_length', -1)],
    )
    self._stub = runner_pb2_grpc.RunnerStub(self._channel)

  @classmethod
  def connect(cls, launched: launcher.LaunchedRunner) -> 'RunnerClient':
    """Connects to a runner and checks that this server can drive it.

    Args:
        launched: The runner, serving.

    Returns:
        The client, with the runner's `info`.

    Raises:
        Error: The runner did not answer, or it speaks another protocol
          or another version of it, or hosts a runtime this server does
          not drive.
    """
    client = cls(launched)
    try:
      client._identify()
    except BaseException:
      client.close()
      raise
    client._channel.subscribe(client._on_connectivity)
    return client

  def online(self) -> bool:
    """Whether the runner is still there, as far as can be told now.

    It asks the runner nothing. A runner is gone once this server closed
    the connection, its process is known to have exited, or its connection
    is lost and did not come back: a reconnect failed, or it was not back
    within _RECONNECT_SECONDS. A connection gRPC closed for being idle, or
    that dropped, is not a runner gone while it reconnects.

    Returns:
      True if the runner is online, False otherwise.
    """
    if self._launched.exited():
      return False
    with self._lock:
      if self._closed or self._failed:
        return False
      if self._connectivity == grpc.ChannelConnectivity.READY:
        return True
      return time.monotonic() - self._unready_since < _RECONNECT_SECONDS

  def close(self) -> None:
    """Closes the connection; calls in flight fail. Idempotent."""
    with self._lock:
      self._closed = True
    self._channel.close()

  def cancel(self) -> None:
    """Stops the turn in flight: its Send, or its capture as it arrives.

    Or the next Send, or capture, of the chat: the chat is then over,
    until create_chat() starts another.
    """
    with self._lock:
      self._cancelled = True
      reply = self._reply
    if reply is not None:
      reply.cancel()

  def load(
      self,
      request: runner_pb2.LoadRequest,
      model: pathlib.Path,
      upload_lock: threading.Lock,
      on_progress: Callable[[int, int], None],
  ) -> runner_pb2.LoadResponse:
    """Loads the model, uploading it first if the runner's store lacks it.

    A PyTorch runner is never uploaded to: it reads the model directory
    it was started with.

    Args:
        request: The Load; its sha256 is that of the file at `model`.
        model: The model file.
        upload_lock: Held while uploading, since runners may share a store.
        on_progress: Called with (bytes sent, total) during an upload, on a
          thread of gRPC's.

    Returns:
        What the runner answered.

    Raises:
        Error: A GPU is asked for and the runner reports none, or the
          upload or the Load failed.
    """
    if request.device == runner_pb2.GPU and not self.info.host.HasField('gpu'):
      raise Error(
          f'{self._device.name} cannot run this runner on a GPU; choose CPU.'
      )

    # The PyTorch runner reads its model in place and takes no upload: its
    # NOT_FOUND is a failure like any other.
    uploads = self.info.runtime.name != runner_pb2.RuntimeInfo.PYTORCH

    def try_load() -> runner_pb2.LoadResponse | None:
      timeout = _TIMEOUT_SECONDS['Load']
      try:
        return self._stub.Load(request, timeout=timeout)
      except (grpc.RpcError, ValueError) as error:
        if uploads and not self._closed and isinstance(error, grpc.RpcError):
          if error.code() == grpc.StatusCode.NOT_FOUND:
            return None  # Only the runner's store lacks the model.
        raise self._to_error(error, 'Load', timeout) from None

    loaded = try_load()
    if loaded is None:
      with upload_lock:
        # A runner sharing the store may have received it meanwhile.
        loaded = try_load()
        if loaded is None:
          self._upload_model(request.sha256, model, on_progress)
      if loaded is None:
        loaded = try_load()
    if loaded is None:
      raise Error('The runner lost the model it received.')
    return loaded

  def create_chat(self, request: runner_pb2.CreateChatRequest) -> None:
    """Creates a chat; the runner's chat before, if any, is closed.

    A turn stopped in the chat before (cancel()) stops none of this one.

    Args:
        request: The chat's starting messages and options.

    Raises:
        NoRoomToStartError: The runner refused it: the context has no room
          for its system prompt. It then holds no chat.
        Error: The runner refused or failed it; it then holds no chat.
    """
    with self._lock:
      self._cancelled = False
    self._call('CreateChat', request)

  def send(
      self, content: str, on_chunk: Callable[[runner_pb2.Chunk], None]
  ) -> runner_pb2.Completion:
    """Runs the user's next turn.

    Args:
        content: What the user says.
        on_chunk: Called with each chunk of the reply as it arrives.

    Returns:
        The turn's Completion.

    Raises:
        NoRoomError: The runner refused the turn: the context has no room for
          it. The chat goes on.
        Error: The reply did not complete; the chat is over.
    """
    request = runner_pb2.SendRequest(
        message=runner_pb2.ChatMessage(
            role=runner_pb2.ChatMessage.USER, content=content
        )
    )
    try:
      reply = self._stub.Send(request)
    except ValueError as error:  # close() came meanwhile.
      raise self._to_error(error, 'Send', None) from None
    with self._lock:
      self._reply = reply
      if self._cancelled:
        reply.cancel()
    completion = None
    try:
      for response in reply:
        if response.HasField('chunk'):
          on_chunk(response.chunk)
        else:
          completion = response.completion
    except grpc.RpcError as error:
      # A Completion that arrived stands, even if cancelled since: the
      # runner counted the turn.
      if completion is None:
        raise self._to_error(error, 'Send', None) from None
    except BaseException:  # on_chunk failed: stop the reply.
      reply.cancel()
      raise
    finally:
      with self._lock:
        self._reply = None
    if completion is None:
      raise Error('The runner ended the reply without a Completion.')
    return completion

  def take_capture(
      self,
      turn: int,
      directory: pathlib.Path,
      holder: pathlib.Path | None = None,
      on_progress: Callable[[int], None] | None = None,
  ) -> None:
    """Copies a completed turn's capture into `directory`, then releases it.

    The capture comes as one stream with no deadline, however large; the
    runner is told to delete it only once every file is held.

    Args:
        turn: The turn of the last Completion.
        directory: Where the files go.
        holder: What names the disk if it has no room (CaptureWriter).
        on_progress: Called with the bytes of files received so far, after each
          piece of a file; the protocol announces each file's size as it starts,
          never the capture's.

    Raises:
        Error: A call failed, or the stream broke the protocol's order,
          ended mid-file, or brought a file not as the runner said or
          that cannot be written, its disk without room for it included.
          Nothing is released; the files completed so far stay, no .part
          file does.
    """
    request = runner_pb2.ReadCapturesRequest(turn=turn)
    writer = capture_writer.CaptureWriter(directory, holder)
    received = 0
    try:
      call = self._stub.ReadCaptures(request)
      with self._lock:
        self._reply = call
        if self._cancelled:
          call.cancel()
      for chunk in call:
        writer.write(chunk)
        if on_progress is not None and chunk.HasField('data'):
          received += len(chunk.data)
          on_progress(received)
      writer.close()
    except capture_writer.Error as error:
      call.cancel()
      raise Error(str(error)) from None
    except (grpc.RpcError, ValueError) as error:
      raise self._to_error(error, 'ReadCaptures', None) from None
    finally:
      with self._lock:
        self._reply = None
      writer.discard()
    self._call('ReleaseCaptures', runner_pb2.ReleaseCapturesRequest(turn=turn))

  def _on_connectivity(self, state: grpc.ChannelConnectivity) -> None:
    """Records the channel's state; gRPC calls it on a thread of its own.

    A channel whose connection was lost goes idle, and would reconnect
    only for the next call: it is asked to reconnect now, so that online()
    says whether the runner is still there. A failed connect counts until
    the channel is READY again.

    Args:
      state: The new connectivity state of the channel.
    """
    ready = grpc.ChannelConnectivity.READY
    with self._lock:
      if (state == ready) != (self._connectivity == ready):
        self._unready_since = time.monotonic()
      self._connectivity = state
      if state == ready:
        self._failed = False
      elif state in (
          grpc.ChannelConnectivity.TRANSIENT_FAILURE,
          grpc.ChannelConnectivity.SHUTDOWN,
      ):
        self._failed = True
      # A runner whose process is gone is not reconnected to: the
      # attempt would outlive the channel, which is closed next.
      if (
          state != grpc.ChannelConnectivity.IDLE
          or self._closed
          or self._launched.exited()
      ):
        return
      previous = self._reconnect
      self._reconnect = grpc.channel_ready_future(self._channel)
    if previous is not None:
      previous.cancel()

  def _identify(self) -> None:
    """Asks the runner who it is and checks that this server can drive it.

    Raises:
        Error: As connect().
    """
    info = self._call('GetInfo', runner_pb2.GetInfoRequest())
    if info.protocol_version != PROTOCOL_VERSION:
      raise Error(
          'The runner speaks protocol '
          f'{info.protocol_version or "(none)"}; this server speaks '
          f'{PROTOCOL_VERSION}.'
      )
    if info.runtime.name not in _RUNTIMES:
      name = runner_pb2.RuntimeInfo.Name.Name(info.runtime.name)
      raise Error(
          f'The runner hosts {name}; this server runs LiteRT-LM and PyTorch.'
      )
    self.info = info

  def _upload_model(
      self,
      sha256: str,
      path: pathlib.Path,
      on_progress: Callable[[int, int], None],
  ) -> None:
    """Uploads the model file at `path` to the runner's store."""

    def requests() -> Iterator[runner_pb2.UploadModelRequest]:
      total = path.stat().st_size
      yield runner_pb2.UploadModelRequest(
          header=runner_pb2.ModelHeader(sha256=sha256, size=total)
      )
      sent = 0
      with path.open('rb') as model:
        while data := model.read(_UPLOAD_CHUNK_BYTES):
          yield runner_pb2.UploadModelRequest(data=data)
          sent += len(data)
          on_progress(sent, total)

    self._call('UploadModel', requests())

  def _call(self, rpc: str, request: Any) -> Any:
    """Calls a unary or client-streaming RPC, with its deadline if any.

    Args:
        rpc: The RPC's name, as the stub has it.
        request: Its request, or an iterator of requests.

    Returns:
        The RPC's response.

    Raises:
        Error: The call failed.
    """
    timeout = _TIMEOUT_SECONDS.get(rpc)
    try:
      return getattr(self._stub, rpc)(request, timeout=timeout)
    except (grpc.RpcError, ValueError) as error:
      raise self._to_error(error, rpc, timeout) from None

  def _to_error(
      self, error: Exception, rpc: str, timeout: float | None
  ) -> Error:
    """Returns the Error a failed call stands for.

    This is the only reading of a runner's status codes.

    Args:
        error: The grpc.RpcError, or the ValueError of a call on a channel
          close() closed.
        rpc: The call, to name in the reason.
        timeout: Its deadline, if any.

    Returns:
        The error to raise.
    """
    if self._closed:
      return Error('This server closed the connection.')
    if isinstance(error, ValueError):
      return Error(str(error))
    code = error.code()
    if code == grpc.StatusCode.RESOURCE_EXHAUSTED and rpc in _NO_ROOM:
      counts = _no_room_counts(error)
      if counts is not None:
        return _NO_ROOM[rpc](*counts)
    if code == grpc.StatusCode.CANCELLED and self._cancelled:
      return Error('Cancelled.')
    if code == grpc.StatusCode.UNAVAILABLE and self.info.protocol_version:
      # It answered GetInfo, so it was there; one that ended for a
      # reason said it last.
      words = self._launched.last_words()
      if words:
        return Error(f'The runner exited, saying: {words}')
      return Error('The connection to the runner closed.')
    if code == grpc.StatusCode.UNIMPLEMENTED and not self.info.protocol_version:
      # Not even GetInfo: it serves no Runner service of this package.
      return Error(
          'The runner speaks another protocol (an older or newer '
          'build); rebuild it, or keep it again.'
      )
    if code == grpc.StatusCode.DEADLINE_EXCEEDED and timeout is not None:
      return Error(
          f'The runner did not finish {rpc} within {round(timeout, 1):g} s.'
      )
    return Error(error.details() or code.name)


# The error a RESOURCE_EXHAUSTED with both counts stands for, by RPC.
_NO_ROOM = types.MappingProxyType(
    {'Send': NoRoomError, 'CreateChat': NoRoomToStartError}
)


def _no_room_counts(error: grpc.RpcError) -> tuple[int, int] | None:
  """The tokens needed and left a refusal for no room gives, if it does.

  The protocol gives the two counts as trailing metadata, 'turn-tokens' and
  'tokens-left'; a RESOURCE_EXHAUSTED without them, gRPC's own, is None.

  Args:
    error: The RPC error to inspect.

  Returns:
    A tuple of (turn_tokens, tokens_left), or None if unavailable.
  """
  counts = dict(error.trailing_metadata() or ())
  try:
    return int(counts['turn-tokens']), int(counts['tokens-left'])
  except (KeyError, ValueError):
    return None
