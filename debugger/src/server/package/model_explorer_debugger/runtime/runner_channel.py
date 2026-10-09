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
# ==============================================================================

"""Runner socket reader, ownership heartbeat, and request event delivery."""

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timezone
import json
import logging
import queue
import threading
import time
from typing import Any
from uuid import uuid4

_LOGGER = logging.getLogger(__name__)


def encode_frame(header: dict[str, Any], payload: bytes) -> bytes:
  """Encodes a binary frame.

  The frame is the 4-byte big-endian header length, the JSON header, then the
  raw payload bytes.
  """
  head = json.dumps(header).encode()
  return len(head).to_bytes(4, 'big') + head + payload


def decode_frame(raw: str | bytes | bytearray | memoryview) -> dict[str, Any]:
  """Decodes a text or binary Runner frame into its JSON object.

  A text frame is its JSON object; a binary frame is its header with the
  payload bytes under `data`.
  """
  if isinstance(raw, str):
    value, payload = json.loads(raw), None
  else:
    size = int.from_bytes(raw[:4], 'big')
    if len(raw) < 4 + size:
      raise ValueError('Truncated Runner frame')
    value, payload = json.loads(raw[4 : 4 + size]), bytes(raw[4 + size :])
  if not isinstance(value, dict):
    raise ValueError('Invalid Runner response')
  if payload is not None:
    value['data'] = payload
  return value


def _settle(
    connection: Any,
    identity: str,
    kind: str,
    timeout: float,
    failure: str,
) -> dict[str, Any]:
  result = connection.receive(timeout)
  if result.get('requestId') != identity:
    raise ValueError('Runner response identity mismatch')
  if result.get('type') == 'error':
    raise ValueError(result.get('error', 'Runner rejected the request'))
  if result.get('type') != kind:
    raise ValueError(result.get('error', failure))
  return result


def call(
    connection: Any,
    kind: str,
    *,
    timeout: float = 60,
    reply: str | None = None,
    failure: str = 'Unexpected Runner response',
    **fields: Any,
) -> dict[str, Any]:
  """Sends one request and returns its reply.

  The reply must carry this request's ID and the expected type.
  """
  identity = str(uuid4())
  connection.send(dict(type=kind, requestId=identity, **fields))
  return _settle(connection, identity, reply or kind, timeout, failure)


def _drain_abandoned(connection: Any, count: int, timeout: float) -> None:
  """Reads `count` outstanding replies, all within one `timeout` deadline.

  A reply stream that cannot be drained in time can no longer be trusted, so
  the connection is closed instead of leaving late replies for the next request.
  """
  deadline = time.monotonic() + timeout
  try:
    for _ in range(count):
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        raise TimeoutError('Runner replies were not drained in time')
      connection.receive(remaining)
  except Exception as error:  # pylint: disable=broad-exception-caught
    # The transfer was already abandoned, usually because of an error the
    # caller is propagating; report this one without masking it.
    _LOGGER.warning(
        'Closing Runner connection after an abandoned transfer: %s', error
    )
    try:
      connection.close()
    except Exception as close_error:  # pylint: disable=broad-exception-caught
      _LOGGER.debug('Closing the Runner connection failed: %r', close_error)


def pipeline(
    connection: Any,
    commands: Iterable[tuple[str, dict[str, Any], bytes | None]],
    *,
    window: int,
    timeout: float = 60,
) -> Iterator[dict[str, Any]]:
  """Keeps up to `window` file requests in flight; yields replies in order.

  `commands` yields (kind, fields, payload); a payload goes out as a binary
  frame. The Runner answers file requests in the order it received them, so a
  reply that is not for the oldest outstanding request ends the transfer.

  If the caller abandons the generator, the outstanding replies are drained
  within one shared `timeout`; otherwise the connection is closed.
  """
  pending: deque[tuple[str, str]] = deque()
  try:
    for kind, fields, payload in commands:
      identity = str(uuid4())
      message = dict(type=kind, requestId=identity, **fields)
      if payload is None:
        connection.send(message)
      else:
        connection.send_binary(message, payload)
      pending.append((identity, kind))
      if len(pending) >= window:
        yield _settle(
            connection,
            *pending.popleft(),
            timeout,
            'Unexpected Runner response',
        )
    while pending:
      yield _settle(
          connection, *pending.popleft(), timeout, 'Unexpected Runner response'
      )
  finally:
    # An abandoned transfer must not leave its replies for the next request.
    if pending:
      _drain_abandoned(connection, len(pending), timeout)


def request(
    connection: Any,
    message: dict[str, Any],
    *,
    cancelled: Callable[[], bool] | None = None,
    deadline: float = 600,
    grace: float = 30,
) -> Iterator[dict[str, Any]]:
  """Sends one operation and yields each of its events.

  The caller stops at the terminal event. Cancellation is sent once and then
  waited for, so the reply stream keeps a single consumer. An event for another
  request means the stream can no longer be trusted.
  """
  identity = message['requestId']
  connection.send(message)
  began, cancel_at = time.monotonic(), None
  while True:
    if cancelled and cancel_at is None and cancelled():
      connection.send({'type': 'cancel', 'requestId': identity})
      cancel_at = time.monotonic()
    if cancel_at is not None and time.monotonic() - cancel_at > grace:
      raise ConnectionError('Runner did not finish cancellation')
    if time.monotonic() - began > deadline:
      raise TimeoutError(f'Runner task exceeded {deadline // 60} minutes')
    try:
      event = connection.receive(0.1)
    except TimeoutError:
      continue
    if event.get('requestId') != identity:
      raise ValueError('Runner response identity mismatch')
    yield event


class RunnerChannel:
  """Owns one Runner socket: a single reader thread plus ownership heartbeats.

  Subclasses set `self.socket` and call `start_channel(hello)` once the Runner
  handshake succeeds. Until then, `send` and `receive` use the socket directly
  so the handshake can run on the caller's thread. After `start_channel`, only
  the reader thread reads the socket: request replies are delivered through an
  inbox, while heartbeat acknowledgements and Runner lifecycle requests are
  handled internally and reported through `on_lifecycle`.

  Attributes:
    hello: The Runner's handshake message, set by `start_channel`.
    protocol: The negotiated Runner protocol version.
    owner: The Session ownership acknowledged by the Runner, or None.
    on_lifecycle: Optional `callback(kind, fields)` for heartbeat, disconnect,
      and Runner-initiated stop or session-end events.
    last_seen: ISO timestamp of the last acknowledged heartbeat, or None.
  """

  heartbeat_interval = 5.0
  heartbeat_timeout = 30.0

  def __init__(self) -> None:
    self.socket: Any = None
    self.hello: dict[str, Any] | None = None
    self.protocol: int = 4
    self.owner: dict[str, Any] | None = None
    self.on_lifecycle: Callable[[str, dict[str, Any]], Any] | None = None
    self.last_seen: str | None = None
    # None until start_channel: the handshake reads the socket directly.
    self._inbox: queue.Queue[dict[str, Any] | BaseException] | None = None
    self._channel_stop = threading.Event()
    self._send_lock = threading.Lock()
    self._expected_close = False
    self._last_ack = time.monotonic()
    self._last_heartbeat_send = 0.0
    self._sequence = 0
    self._last_acked_sequence = 0
    self._reader: threading.Thread | None = None
    self._heartbeats: threading.Thread | None = None

  def start_channel(self, hello: dict[str, Any]) -> None:
    """Starts the reader and heartbeat threads after a successful handshake."""
    self.hello = hello
    self.protocol = hello.get('protocolVersion', 4)
    self._last_ack = time.monotonic()
    self._inbox = queue.Queue()
    self._reader = threading.Thread(
        target=self._read_channel, daemon=True, name='runner-events'
    )
    self._heartbeats = threading.Thread(
        target=self._heartbeat_loop, daemon=True, name='runner-heartbeat'
    )
    self._reader.start()
    self._heartbeats.start()

  def send(self, value: dict[str, Any]) -> None:
    self._send_frame(json.dumps(value))

  def send_binary(self, header: dict[str, Any], payload: bytes) -> None:
    self._send_frame(encode_frame(header, payload))

  def _send_frame(self, frame: str | bytes) -> None:
    with self._send_lock:
      if self._channel_stop.is_set():
        raise ConnectionError('Runner connection is closed')
      self.socket.send(frame)

  def receive(self, timeout: float) -> dict[str, Any]:
    if self._inbox is None:
      return decode_frame(self.socket.recv(timeout=timeout))
    try:
      value = self._inbox.get(timeout=timeout)
    except queue.Empty:
      raise TimeoutError('Waiting for Runner response') from None
    if isinstance(value, BaseException):
      # Wake subsequent callers as well, without ever reading the socket twice.
      self._inbox.put(value)
      raise value
    return value

  def activate_owner(self, owner: dict[str, Any]) -> dict[str, Any]:
    reply = call(
        self,
        'activate',
        timeout=10,
        reply='activated',
        failure='Runner did not accept Session ownership',
        **owner,
    )
    expected = {key: owner[key] for key in ('serverId', 'sessionId', 'runId')}
    if any(
        (reply.get('owner') or {}).get(key) != value
        for key, value in expected.items()
    ):
      raise ValueError(
          'Runner ownership acknowledgement does not match this Session'
      )
    self.owner = dict(reply['owner'])
    self._last_ack = time.monotonic()
    self.last_seen = datetime.now(timezone.utc).isoformat()
    return reply

  def _notify(self, kind: str, **fields: Any) -> None:
    callback = self.on_lifecycle
    if callback:
      callback(kind, fields)

  def _fail(self, error: BaseException) -> None:
    if self._channel_stop.is_set():
      return
    self._channel_stop.set()
    self._inbox.put(ConnectionError(str(error)))
    if not self._expected_close:
      self._notify('disconnected', error=str(error))

  def _read_channel(self) -> None:
    try:
      while not self._channel_stop.is_set():
        try:
          value = decode_frame(self.socket.recv(timeout=0.5))
        except TimeoutError:
          continue
        kind = value.get('type')
        if kind == 'heartbeat_ack':
          owner = self.owner or {}
          sequence = value.get('sequence')
          if (
              value.get('serverId') != owner.get('serverId')
              or value.get('sessionId') != owner.get('sessionId')
              or type(sequence) is not int
              or not self._last_acked_sequence < sequence <= self._sequence
          ):
            continue
          self._last_acked_sequence = sequence
          self._last_ack = time.monotonic()
          self.last_seen = datetime.now(timezone.utc).isoformat()
          self._notify('heartbeat', lastSeen=self.last_seen)
        elif kind in ('session_end_requested', 'stop_requested'):
          owner = value.get('owner') or value
          if self.owner and all(
              owner.get(key) == self.owner.get(key)
              for key in ('serverId', 'sessionId', 'runId')
          ):
            self._notify(kind, **value)
        else:
          self._inbox.put(value)
    except BaseException as error:
      self._fail(error)

  def _heartbeat_loop(self) -> None:
    while not self._channel_stop.wait(min(self.heartbeat_interval, 0.5)):
      if not self.owner:
        continue
      elapsed = time.monotonic() - self._last_ack
      if elapsed >= self.heartbeat_timeout:
        self._fail(
            TimeoutError('Runner did not acknowledge heartbeats for 30 seconds')
        )
        try:
          self.socket.close()
        except Exception as error:  # pylint: disable=broad-exception-caught
          # The channel is already failed; closing is best-effort cleanup.
          _LOGGER.debug('Ignoring Runner socket close failure: %s', error)
        return
      if time.monotonic() - self._last_heartbeat_send < self.heartbeat_interval:
        continue
      try:
        self._sequence += 1
        self.send(
            dict(
                type='heartbeat',
                serverId=self.owner['serverId'],
                sessionId=self.owner['sessionId'],
                sequence=self._sequence,
            )
        )
        self._last_heartbeat_send = time.monotonic()
      except BaseException as error:
        self._fail(error)
        return

  def stop_channel(self) -> None:
    """Stops the channel threads and wakes any waiting receiver."""
    self._expected_close = True
    self._channel_stop.set()
    if self._inbox is not None:
      self._inbox.put(ConnectionError('Runner connection closed'))
