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

"""One runner, over the one connection it serves for its whole life.

A runner admits the TCP peer (address and port) of its first RPC and refuses
every other one, so a reconnect from a new port loses the runner. The channel
is therefore never shared and never idles out: a RunnerClient holds one
connection until it is closed.
"""

from collections.abc import Callable, Iterator
import hashlib
import os
import pathlib
import threading

import grpc
from model_debugger.v1 import runner_pb2, runner_pb2_grpc

# The only protocol version this server speaks.
PROTOCOL_VERSION = '1.0.0'

# Most bytes in one UploadModelRequest data message (the protocol's limit).
_UPLOAD_CHUNK_BYTES = 1 << 20

# Deadlines of the RPCs, in seconds; Send streams take none because generation
# duration is unbounded and controlled by user cancellation.
_GET_INFO_SECONDS = 10
_LOAD_SECONDS = 600
_CALL_SECONDS = 60
_UPLOAD_SECONDS = 3600
_READ_CAPTURE_SECONDS = 600

# gRPC's largest int argument, which it reads as "never" for durations.
_NEVER_MS = 2**31 - 1

_CHANNEL_OPTIONS = (
    # gRPC closes a connection idle for 30 minutes by default; the next RPC
    # would come from a new port and be refused.
    ('grpc.client_idle_timeout_ms', _NEVER_MS),
    # Channels to one address otherwise share a connection: a second client
    # of the same runner would pass as the first.
    ('grpc.use_local_subchannel_pool', 1),
    # No keepalive pings: the runner's gRPC server takes a ping on an idle
    # connection as abuse and closes it ("too_many_pings"), which loses the
    # runner; and every transport here (loopback, adb forward) reports a dead
    # peer by closing the socket anyway.
)


class RunnerError(Exception):
  """A runner RPC failed; `str(error)` is the runner's message.

  Attributes:
    code: the gRPC status code.
  """

  def __init__(self, code: grpc.StatusCode, message: str) -> None:
    """Initializes the error with its status code and message."""
    super().__init__(message)
    self.code = code


def _runner_error(error: grpc.RpcError) -> RunnerError:
  return RunnerError(error.code(), error.details() or '')


class Generation:
  """One Send in flight: its chunks go to a callback on a thread of its own."""

  def __init__(
      self,
      call: Iterator[runner_pb2.SendResponse],
      on_chunk: Callable[[runner_pb2.Chunk], None],
  ) -> None:
    """Starts reading `call` on a background thread."""
    self._call = call
    self._on_chunk = on_chunk
    self._completion: runner_pb2.Completion | None = None
    self._error: BaseException | None = None
    self._thread = threading.Thread(
        target=self._read, name='runner-send', daemon=True
    )
    self._thread.start()

  def _read(self) -> None:
    try:
      for response in self._call:
        if response.HasField('chunk'):
          self._on_chunk(response.chunk)
        else:
          self._completion = response.completion
    except grpc.RpcError as error:
      # A cancel after the completion arrived came too late to stop the turn:
      # the runner had finished it, so the turn stands.
      if self._completion is None:
        self._error = _runner_error(error)
    except (KeyboardInterrupt, SystemExit):
      raise
    except BaseException as error:  # on_chunk failed: stop the reply.
      self._error = error
      self._call.cancel()

  def cancel(self) -> None:
    """Cancels the Send; the runner then ends the chat. Idempotent."""
    self._call.cancel()

  def wait(self) -> runner_pb2.Completion:
    """Returns the Completion once the Send ends.

    Raises:
      RunnerError: the Send failed, CANCELLED included.
    """
    self._thread.join()
    if self._error is not None:
      raise self._error
    if self._completion is None:
      raise RunnerError(
          grpc.StatusCode.UNKNOWN, 'The Send ended without a completion.'
      )
    return self._completion


class RunnerClient:
  """The server's side of one runner's connection."""

  def __init__(self, address: str) -> None:
    """Opens the channel to the runner at `address` ('host:port')."""
    self.address = address
    self._channel = grpc.insecure_channel(address, options=_CHANNEL_OPTIONS)
    self._stub = runner_pb2_grpc.RunnerStub(self._channel)

  def close(self) -> None:
    """Closes the channel; RPCs in flight end with CANCELLED."""
    self._channel.close()

  def get_info(self) -> runner_pb2.RunnerInfo:
    """Returns who the runner is; the first RPC claims the runner."""
    return self._unary(
        self._stub.GetInfo, runner_pb2.GetInfoRequest(), _GET_INFO_SECONDS
    )

  def upload_model(
      self,
      path: pathlib.Path,
      sha256: str,
      on_progress: Callable[[int, int], None],
  ) -> None:
    """Puts the model file at `path` in the runner's store.

    Args:
      path: the model file.
      sha256: its SHA-256, lower-case hex.
      on_progress: called with (bytes sent, total) after each data message, on a
        gRPC thread.
    """
    total = path.stat().st_size

    def requests() -> Iterator[runner_pb2.UploadModelRequest]:
      yield runner_pb2.UploadModelRequest(
          header=runner_pb2.ModelHeader(sha256=sha256, size=total)
      )
      sent = 0
      with path.open('rb') as file:
        while data := file.read(_UPLOAD_CHUNK_BYTES):
          yield runner_pb2.UploadModelRequest(data=data)
          sent += len(data)
          on_progress(sent, total)

    try:
      self._stub.UploadModel(requests(), timeout=_UPLOAD_SECONDS)
    except grpc.RpcError as error:
      raise _runner_error(error) from None

  def load(self, request: runner_pb2.LoadRequest) -> runner_pb2.LoadResponse:
    """Loads the connection's one model; NOT_FOUND asks for an upload."""
    return self._unary(self._stub.Load, request, _LOAD_SECONDS)

  def create_chat(self, request: runner_pb2.CreateChatRequest) -> None:
    """Creates the connection's one chat."""
    self._unary(self._stub.CreateChat, request, _CALL_SECONDS)

  def send(
      self, content: str, on_chunk: Callable[[runner_pb2.Chunk], None]
  ) -> Generation:
    """Starts the user's next turn; `on_chunk` runs on the Send's thread."""
    request = runner_pb2.SendRequest(
        message=runner_pb2.ChatMessage(
            role=runner_pb2.ChatMessage.USER, content=content
        )
    )
    return Generation(self._stub.Send(request), on_chunk)

  def list_captures(self, turn: int) -> list[runner_pb2.CaptureFile]:
    """Returns the files of a completed turn's capture."""
    response = self._unary(
        self._stub.ListCaptures,
        runner_pb2.ListCapturesRequest(turn=turn),
        _CALL_SECONDS,
    )
    return list(response.files)

  def read_capture(
      self,
      turn: int,
      file: runner_pb2.CaptureFile,
      destination: pathlib.Path,
  ) -> None:
    """Copies one captured file to `destination`, checked against its listing.

    The bytes go to `destination` + '.part' first, renamed once their size and
    SHA-256 match `file`'s.

    Raises:
      RunnerError: the read failed, or DATA_LOSS when the bytes do not match.
    """
    part = destination.with_name(destination.name + '.part')
    digest = hashlib.sha256()
    size = 0
    try:
      with part.open('wb') as out:
        request = runner_pb2.ReadCaptureRequest(turn=turn, path=file.path)
        for data in self._stub.ReadCapture(
            request, timeout=_READ_CAPTURE_SECONDS
        ):
          size += len(data.data)
          if size > file.size:
            break
          out.write(data.data)
          digest.update(data.data)
    except grpc.RpcError as error:
      part.unlink(missing_ok=True)
      raise _runner_error(error) from None
    if size != file.size or digest.hexdigest() != file.sha256:
      part.unlink(missing_ok=True)
      raise RunnerError(
          grpc.StatusCode.DATA_LOSS,
          f'{file.path} arrived as {size} bytes hashing to '
          f'{digest.hexdigest()}; the listing says {file.size} bytes, '
          f'{file.sha256}.',
      )
    os.replace(part, destination)

  def release_captures(self, turn: int) -> None:
    """Deletes one turn's capture from the runner."""
    self._unary(
        self._stub.ReleaseCaptures,
        runner_pb2.ReleaseCapturesRequest(turn=turn),
        _CALL_SECONDS,
    )

  def _unary(self, method, request, timeout: float):
    try:
      return method(request, timeout=timeout)
    except grpc.RpcError as error:
      raise _runner_error(error) from None
