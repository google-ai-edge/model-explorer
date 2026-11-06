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

"""Tests of RunnerClient against the in-process fake runner.

Each test names the rule of the protocol checklist (X2) it checks.
"""

import concurrent.futures
import hashlib
import pathlib
import socket
import threading
import time
from typing import Any
from unittest import mock

from absl.testing import parameterized
import grpc
from model_debugger.protocol import runner_pb2
from model_debugger.server.runner import launcher
from model_debugger.server.runner import runner_client
from model_debugger.server.testing import fake_runner
from model_debugger.server.testing import test_main

_MODEL = b'not really a model' * 1000
_SHA256 = hashlib.sha256(_MODEL).hexdigest()
_FILES = {
    'decode_post_logits_step_17.safetensors': bytes(range(256)) * 9000,
    'generated_tokens.jsonl': b'{"token_ids":[[236778]],"texts":["4"]}\n',
}
_LOCAL = launcher.Device(
    launcher.LOCAL_DEVICE_ID, 'This computer', 'macOS', 'local', True
)
_SIMULATOR = launcher.Device(
    'ios-simulator:S', 'iPhone 17', 'iOS Simulator', 'simulator', False
)
_CHAT = runner_pb2.CreateChatRequest(
    options=runner_pb2.ChatOptions(max_completion_tokens=8, top_k=1)
)
_TIMEOUT = 10


class _Launched(launcher.LaunchedRunner):
  """A fake runner as a launcher hands it out; the test stops it."""

  def __init__(
      self,
      device,
      address: str,
      runner: fake_runner.FakeRunner | None = None,
      sees_process: bool = False,
  ) -> None:
    """Hands out the runner at `address`.

    `runner`, if any, says if it runs; `sees_process`, whether exited()
    sees it, as for a process of this computer.

    Args:
      device: The device hosting the runner.
      address: The network address where the runner listens.
      runner: The runner indicating running state, if any.
      sees_process: Whether exited() sees the process locally.
    """
    super().__init__(device, address)
    self._runner = runner
    self._sees = sees_process

  def _sees_process(self) -> bool:
    return self._sees

  def _alive(self) -> bool:
    return self._runner is None or not self._runner.stopped

  def _stop(self) -> None:
    pass


def _reply(turn: int, prompt: str) -> fake_runner.Reply:
  del prompt  # Unused.
  chunks = [runner_pb2.Chunk(content='4'), runner_pb2.Chunk(content='.')]
  completion = runner_pb2.Completion(
      content='4.',
      finish_reason=runner_pb2.FINISH_REASON_STOP,
      model_debugger=runner_pb2.ModelDebugger(turn=turn),
  )
  return chunks, completion, _FILES


def _load_request(
    sha256: str = _SHA256, device: int = runner_pb2.CPU
) -> runner_pb2.LoadRequest:
  return runner_pb2.LoadRequest(
      sha256=sha256, device=device, context_length=4096
  )


def _connect(
    runner: fake_runner.FakeRunner, device=_LOCAL, sees_process: bool = False
) -> runner_client.RunnerClient:
  return runner_client.RunnerClient.connect(
      _Launched(device, runner.address, runner, sees_process)
  )


def _offline_within(
    client: runner_client.RunnerClient, seconds: float
) -> float:
  """Returns how long until `client` is offline; fails after `seconds`."""
  start = time.monotonic()
  while client.online():
    assert time.monotonic() - start < seconds, 'still online'
    time.sleep(0.005)
  return time.monotonic() - start


def _online_throughout(
    client: runner_client.RunnerClient, seconds: float
) -> list[str]:
  """Samples online() for `seconds`; returns the channel states seen."""
  states = []
  deadline = time.monotonic() + seconds
  while time.monotonic() < deadline:
    assert client.online(), f'offline after {states}'
    state = client._connectivity.name
    if not states or states[-1] != state:
      states.append(state)
    time.sleep(0.002)
  return states


def _idle_after(milliseconds: int):
  """Makes channels opened meanwhile close a connection idle that long."""
  open_channel = grpc.insecure_channel

  def insecure_channel(target, options=(), **kwargs):
      options = [*options, ('grpc.client_idle_timeout_ms', milliseconds)]
      return open_channel(target, options=options, **kwargs)

  return mock.patch.object(grpc, 'insecure_channel', insecure_channel)


def _load(client, request=None, model=pathlib.Path('/no/model'), lock=None):
  return client.load(
      request or _load_request(),
      model,
      lock or threading.Lock(),
      lambda sent, total: None,
  )


def _ready_client(runner: fake_runner.FakeRunner) -> runner_client.RunnerClient:
  """Returns a client of `runner` with a model loaded and a chat created."""
  runner.store.add(_SHA256)
  client = _connect(runner)
  _load(client)
  client.create_chat(_CHAT)
  return client


def _eventually(condition) -> None:
  deadline = time.monotonic() + _TIMEOUT
  while not condition():
    assert time.monotonic() < deadline
    time.sleep(0.01)


def _send_on_thread(
    client: runner_client.RunnerClient, on_chunk=lambda c: None
) -> concurrent.futures.Future[Any]:
  """Runs a Send on a thread of its own, as a Model Server does."""
  future = concurrent.futures.Future()

  def send() -> None:
    try:
      future.set_result(client.send('Hi', on_chunk))
    except BaseException as error:  # pylint: disable=broad-exception-caught
      future.set_exception(error)

  threading.Thread(target=send, daemon=True).start()
  return future


def _fail(name: str, code: grpc.StatusCode, message: str):
  """A `before` that fails the RPC `name` with a status."""

  def before(method: str) -> None:
    if method == name:
      raise fake_runner.AbortError(code, message)

  return before


def _stall(name: str, release: threading.Event, reached: threading.Event):
  """A `before` that holds the RPC `name` until `release` is set."""

  def before(method: str) -> None:
    if method == name:
      reached.set()
      release.wait(_TIMEOUT)

  return before


def _stall_chunk(part: str, release: threading.Event, reached: threading.Event):
  """A `before_chunk` that holds the stream at its first `part` message."""

  def before_chunk(chunk: runner_pb2.CaptureChunk) -> None:
    if chunk.WhichOneof('part') == part and not reached.is_set():
      reached.set()
      release.wait(_TIMEOUT)

  return before_chunk


class _RecordingLock:
  """A lock that records when it is taken and given back."""

  def __init__(self, runner: fake_runner.FakeRunner) -> None:
    self.events = []
    self._runner = runner
    self._lock = threading.Lock()

  def __enter__(self):
    self._lock.acquire()
    self.events.append(('enter', len(self._runner.uploads)))

  def __exit__(self, *exc_info):
    self.events.append(('exit', len(self._runner.uploads)))
    self._lock.release()


class _Lingering(fake_runner.FakeRunner):
  """A runner whose Send ends only once `gate` is set, after Completion."""

  def __init__(self, respond) -> None:
    super().__init__(respond)
    self.gate = threading.Event()

  def Send(self, request, context):  # noqa: D102
    yield from super().Send(request, context)
    self.gate.wait(_TIMEOUT)


class _Watched:
  """A Send's call that tells when its Completion was read."""

  def __init__(self, call) -> None:
    self.completed = threading.Event()
    self._call = call

  def __iter__(self):
    return self

  def __next__(self) -> runner_pb2.SendResponse:
    response = next(self._call)
    if response.HasField('completion'):
      self.completed.set()
    return response

  def cancel(self) -> None:
    self._call.cancel()


# ---- 23-26: the capture


def _completed_turn(runner, tmp_path):
  del tmp_path  # Unused.
  client = _ready_client(runner)
  client.send('What is 2 + 2?', lambda c: None)
  return client


class RunnerClientTest(parameterized.TestCase):
  """Tests of RunnerClient against the in-process fake runner."""

  def setUp(self):
    """An empty directory, and a started fake runner."""
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)
    self.runner = self.enter_context(fake_runner.FakeRunner(_reply))

  # ---- 1, 2, 3, 27, 28: the connection
  def test_connect_asks_who_the_runner_is(self):
    """3: connect() asks GetInfo first, and keeps the answer."""
    client = _connect(self.runner)
    self.assertEqual(
        client.info.protocol_version, runner_client.PROTOCOL_VERSION
    )
    self.assertEqual(client.info.runtime.name, runner_pb2.RuntimeInfo.LITERT_LM)
    self.assertLen(self.runner.peers, 1)
    client.close()

  def test_a_dropped_connection_reconnects(self):
    """2: a connection lost between calls, the runner alive, is reopened."""
    client = _ready_client(self.runner)
    self.assertEqual(client.send('Hi', lambda c: None).model_debugger.turn, 1)
    self.runner.drop_connections()
    time.sleep(0.5)
    self.assertEqual(client.send('Hi', lambda c: None).model_debugger.turn, 2)
    self.assertLen(set(self.runner.peers), 2)
    client.close()

  def test_online_is_the_connection_to_the_runner(self):
    """2: online() follows the connection, and asks the runner nothing."""
    client = _connect(self.runner)
    _eventually(client.online)
    self.runner.drop_connections()
    time.sleep(0.5)
    # The runner is still there: the channel reconnects by itself.
    _eventually(client.online)
    self.assertLen(self.runner.peers, 1)  # No call was made meanwhile.
    self.runner.stop()
    _eventually(lambda: not client.online())
    client.close()

  def test_a_runner_whose_connections_drop_stays_online(self):
    """2: a connection closed idle, or dropped, is no runner gone.

    The channel reconnects by itself (IDLE, CONNECTING, READY), and the
    runner is online all along; a call then goes through.
    """
    with _idle_after(300):
      client = _ready_client(self.runner)
    _eventually(client.online)
    seen = _online_throughout(client, 1.2)  # Idle timeouts meanwhile.
    self.assertIn('IDLE', seen)
    for _ in range(3):
      self.runner.drop_connections()
      seen = _online_throughout(client, 0.6)
      self.assertEqual(seen[-1], 'READY')
    self.assertEqual(client.send('Hi', lambda c: None).model_debugger.turn, 1)
    client.close()

  def test_a_runner_that_exits_is_offline_once_the_reconnect_fails(self):
    """2: its reconnect fails at once: offline well before the grace."""
    client = _connect(self.runner)
    _eventually(client.online)
    self.runner.stop()
    waited = _offline_within(client, runner_client._RECONNECT_SECONDS)
    self.assertLess(waited, 1)
    self.assertEqual(client._connectivity.name, 'TRANSIENT_FAILURE')
    # It stays offline while the channel tries again.
    time.sleep(1.5)
    self.assertFalse(client.online())
    client.close()

  def test_a_runner_seen_to_exit_is_offline_at_once(self):
    """2: a process this computer sees end is gone, whatever the channel."""
    client = _connect(self.runner, sees_process=True)
    _eventually(client.online)
    self.runner.stop()
    self.assertFalse(client.online())
    client.close()

  def test_a_reconnect_that_hangs_is_offline_after_the_grace(self):
    """2: a connect that hangs is a runner gone once the grace is over.

    As a device that vanished behind a tunnel: the connect neither
    succeeds nor fails.
    """
    port = int(self.runner.address.rsplit(':', 1)[1])
    client = _connect(self.runner)
    _eventually(client.online)
    dropped = time.monotonic()
    self.runner.stop()
    # Accepts connections and never answers, as a tunnel to no device;
    # bound before the channel reconnects, 0.2 s after the drop.
    with socket.create_server(('127.0.0.1', port)) as silent:
      silent.settimeout(_TIMEOUT)
      with mock.patch.object(runner_client, '_RECONNECT_SECONDS', 0.5):
        _offline_within(client, _TIMEOUT)
        waited = time.monotonic() - dropped
        state = client._connectivity.name
        accepted, _ = silent.accept()
        accepted.close()
    self.assertEqual(state, 'CONNECTING')
    self.assertGreaterEqual(waited, 0.5)
    client.close()

  def test_a_closed_client_is_offline(self):
    """27: once closed, a client is offline, whatever the runner does."""
    client = _connect(self.runner)
    _eventually(client.online)
    client.close()
    self.assertFalse(client.online())

  def test_clients_of_two_runners_each_keep_their_own(self):
    """1: two clients of two runners each stay one peer of their runner."""
    with (
        fake_runner.FakeRunner(_reply) as first,
        fake_runner.FakeRunner(_reply) as second,
    ):
      one, two = _ready_client(first), _ready_client(second)
      for client in (one, two):
        client.send('Hi', lambda c: None)
      self.assertLen(set(first.peers), 1)
      self.assertLen(set(second.peers), 1)
      one.close()
      two.close()

  def test_a_second_client_of_one_runner_is_served(self):
    """1: a runner serves any connection."""
    first, second = _connect(self.runner), _connect(self.runner)
    self.assertEqual(second.info, first.info)
    self.runner.store.add(_SHA256)
    _load(first)
    first.close()
    second.close()

  def test_a_pytorch_runner_is_driven_and_never_uploaded_to(self):
    """A PyTorch runner's NOT_FOUND is its error: no upload follows."""
    self.runner.info = fake_runner.PYTORCH_INFO
    client = _connect(self.runner)
    self.assertEqual(client.info.runtime.name, runner_pb2.RuntimeInfo.PYTORCH)

    with self.assertRaisesRegex(runner_client.Error, 'No model'):
      _load(client)
    self.assertEqual(self.runner.uploads, [])
    client.close()

  @parameterized.parameters(
      ('2.0.0', runner_pb2.RuntimeInfo.LITERT_LM, 'protocol 2.0.0'),
      (
          runner_client.PROTOCOL_VERSION,
          runner_pb2.RuntimeInfo.NAME_UNSPECIFIED,
          'hosts NAME_UNSPECIFIED',
      ),
  )
  def test_connect_refuses_a_runner_it_cannot_drive(self, version, name, text):
    """3: another protocol version, or another runtime, fails the start."""
    info = runner_pb2.RunnerInfo()
    info.CopyFrom(fake_runner.DEFAULT_INFO)
    info.protocol_version = version
    info.runtime.name = name
    self.runner.info = info
    with self.assertRaisesRegex(runner_client.Error, text):
      _connect(self.runner)
    self.assertLen(self.runner.peers, 1)

  def test_connect_to_nothing_fails(self):
    """3: nothing listening fails the start."""
    with socket.socket() as probe:
      probe.bind(('127.0.0.1', 0))
      address = f'127.0.0.1:{probe.getsockname()[1]}'
    with self.assertRaises(runner_client.Error):
      runner_client.RunnerClient.connect(_Launched(_LOCAL, address))

  def test_connect_refuses_a_runner_of_another_protocol(self):
    """3: a runner that serves no Runner of this package fails the start.

    A runner built before the package was renamed serves the same RPCs
    under another service name, so it answers even GetInfo UNIMPLEMENTED.
    """
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(1))
    port = server.add_insecure_port(
        '127.0.0.1:0'
    )
    server.start()
    self.addCleanup(server.stop, None)
    with self.assertRaisesRegex(runner_client.Error, 'another protocol'):
      runner_client.RunnerClient.connect(_Launched(_LOCAL, f'127.0.0.1:{port}'))

  def test_a_runner_that_exits_between_calls_is_gone_at_the_next(self):
    """1: the next call to an exited runner ends as gone."""
    client = _ready_client(self.runner)
    self.runner.stop()
    with self.assertRaisesRegex(runner_client.Error, 'connection to the'):
      client.send('Hi', lambda c: None)
    client.close()

  def test_a_runner_that_exits_mid_send_is_gone(self):
    """1, 18: a runner that exits during a reply is gone."""
    release = threading.Event()

    def held(turn, prompt):
      del turn, prompt  # Unused.

      def chunks():
        yield runner_pb2.Chunk(content='4')
        release.wait(_TIMEOUT)

      return chunks(), runner_pb2.Completion(), {}

    with fake_runner.FakeRunner(held) as runner:
      client = _ready_client(runner)
      started = threading.Event()
      reply = _send_on_thread(client, lambda c: started.set())
      try:
        self.assertTrue(started.wait(_TIMEOUT))
        runner.stop()
        with self.assertRaisesRegex(runner_client.Error, 'connection to the'):
          reply.result(_TIMEOUT)
      finally:
        release.set()
      client.close()

  def test_calls_after_close_end_as_stopped(self):
    """27: a closed client reaches no runner; a Send ends as stopped."""
    client = _ready_client(self.runner)
    client.close()
    peers = list(self.runner.peers)
    with self.assertRaises(runner_client.Error) as over:
      client.send('Hi', lambda c: None)
    self.assertEqual(str(over.exception), 'This server closed the connection.')
    self.assertEqual(self.runner.peers, peers)

  def test_close_during_a_send_stops_it(self):
    """28: closing the connection stops the reply; the chat is over."""
    release = threading.Event()

    def held(turn, prompt):
      del turn, prompt  # Unused.

      def chunks():
        yield runner_pb2.Chunk(content='4')
        release.wait(_TIMEOUT)
        yield runner_pb2.Chunk(content='.')

      return chunks(), runner_pb2.Completion(), {}

    with fake_runner.FakeRunner(held) as runner:
      client = _ready_client(runner)
      started = threading.Event()
      reply = _send_on_thread(client, lambda c: started.set())
      try:
        self.assertTrue(started.wait(_TIMEOUT))
        client.close()
        with self.assertRaises(runner_client.Error) as over:
          reply.result(_TIMEOUT)
        self.assertEqual(
            str(over.exception), 'This server closed the connection.'
        )
      finally:
        release.set()
      _eventually(lambda: runner.chat_over)

  # ---- 5-10: the model
  def test_load_uploads_a_missing_model_then_loads(self):
    """5, 6: NOT_FOUND is answered by uploading 1 MiB pieces, then Load."""
    model = self.tmp_path / 'model.litertlm'
    model.write_bytes(_MODEL * 200)  # 3.6 MB: four data messages
    sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    client = _connect(self.runner)
    progress = []
    loaded = client.load(
        _load_request(sha256),
        model,
        threading.Lock(),
        lambda *p: progress.append(p),
    )
    self.assertEqual(list(loaded.stop_tokens[2].ids.ids), [106])
    size = model.stat().st_size
    self.assertEqual(self.runner.uploads, [sha256])
    self.assertEqual(progress[-1], (size, size))
    self.assertLen(progress, 4)
    steps = [sent for sent, _ in progress]
    self.assertTrue(all(b - a <= 1 << 20 for a, b in zip([0] + steps, steps)))
    self.assertEqual(self.runner.load_request.sha256, sha256)
    client.close()

  def test_load_uploads_only_under_the_lock_it_is_given(self):
    """8: the upload runs in the model's lock, the last Load after it."""
    model = self.tmp_path / 'model.litertlm'
    model.write_bytes(_MODEL)
    lock = _RecordingLock(self.runner)
    _load(_connect(self.runner), model=model, lock=lock)
    self.assertEqual(lock.events, [('enter', 0), ('exit', 1)])

  def test_a_model_in_the_store_is_not_uploaded(self):
    """5: a runner that has the model loads it; the lock is never taken."""
    self.runner.store.add(_SHA256)
    lock = _RecordingLock(self.runner)
    _load(_connect(self.runner), lock=lock)
    self.assertEqual(lock.events, [])
    self.assertEqual(self.runner.uploads, [])

  def test_runners_sharing_a_store_get_one_upload(self):
    """8: runners of one store, one lock: one upload, never two at once."""
    model = self.tmp_path / 'model.litertlm'
    model.write_bytes(_MODEL * 200)
    sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    store, lock = set(), threading.Lock()
    with (
        fake_runner.FakeRunner(_reply, store=store) as first,
        fake_runner.FakeRunner(_reply, store=store) as second,
    ):
      for runner in (first, second):
        runner.before = lambda method: time.sleep(
            0.3 if method == 'UploadModel' else 0
        )
      clients = [_connect(first), _connect(second)]
      threads = [
          threading.Thread(
              target=_load,
              args=(client, _load_request(sha256), model, lock),
          )
          for client in clients
      ]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join(_TIMEOUT)
      self.assertEqual(first.uploads + second.uploads, [sha256])
      self.assertEqual(fake_runner.overlapping_uploads([first, second]), [])
      self.assertTrue(first.load_request and second.load_request)

  def test_an_upload_without_room_fails_the_start(self):
    """7: the runner's reason reaches the start's failure."""
    model = self.tmp_path / 'model.litertlm'
    model.write_bytes(_MODEL)
    self.runner.before = _fail(
        'UploadModel',
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        'Cannot write the model: No space left on device',
    )
    with self.assertRaisesRegex(runner_client.Error, 'No space left'):
      _load(_connect(self.runner), model=model)
    self.assertIsNone(self.runner.load_request)

  def test_an_upload_of_other_bytes_fails_the_start(self):
    """7: bytes that do not hash to the sha256 named are DATA_LOSS."""
    model = self.tmp_path / 'model.litertlm'
    model.write_bytes(_MODEL)
    with self.assertRaisesRegex(runner_client.Error, 'hash'):
      _load(
          _connect(self.runner),
          request=_load_request('0' * 64),
          model=model,
      )
    self.assertEqual(self.runner.store, set())

  def test_a_model_file_it_cannot_read_fails_the_start(self):
    """7: a model file that is gone fails the start."""
    with self.assertRaises(runner_client.Error):
      _load(_connect(self.runner), model=self.tmp_path / 'gone.litertlm')

  @parameterized.parameters(
      (grpc.StatusCode.FAILED_PRECONDITION, 'Missing GPU library: x.'),
      (
          grpc.StatusCode.INVALID_ARGUMENT,
          'LoadRequest.context_length is 0; it must be positive.',
      ),
      (grpc.StatusCode.INTERNAL, 'Out of memory.'),
      (grpc.StatusCode.UNIMPLEMENTED, 'No such device.'),
  )
  def test_load_failures_fail_the_start(self, code, text):
    """9: any Load failure but NOT_FOUND fails the start, with its text."""
    self.runner.store.add(_SHA256)
    self.runner.before = _fail('Load', code, text)
    with self.assertRaises(runner_client.Error) as failed:
      _load(_connect(self.runner))
    self.assertEqual(str(failed.exception), text)
    self.assertEqual(self.runner.uploads, [])

  def test_a_load_slower_than_its_deadline_fails_the_start(self):
    """26: a Load past its deadline fails the start."""
    self.enter_context(
        mock.patch.dict(runner_client._TIMEOUT_SECONDS, {'Load': 0.3})
    )
    self.runner.store.add(_SHA256)
    release, reached = threading.Event(), threading.Event()
    self.runner.before = _stall('Load', release, reached)
    try:
      with self.assertRaisesRegex(runner_client.Error, 'within 0.3 s'):
        _load(_connect(self.runner))
    finally:
      release.set()

  @parameterized.parameters(_SIMULATOR, _LOCAL)
  def test_gpu_where_the_runner_reports_none_fails_before_load(self, device):
    """10: GPU needs a runner that reports a GPU, on any device."""
    info = runner_pb2.RunnerInfo()
    info.CopyFrom(fake_runner.DEFAULT_INFO)
    info.host.ClearField('gpu')
    self.runner.info = info
    client = _connect(self.runner, device=device)
    with self.assertRaisesRegex(runner_client.Error, 'choose CPU'):
      _load(client, request=_load_request(device=runner_pb2.GPU))
    self.assertIsNone(self.runner.load_request)
    self.assertEqual(self.runner.uploads, [])
    self.assertLen(self.runner.peers, 1)

  def test_gpu_where_the_runner_reports_one_loads(self):
    """10: the runner's GetInfo decides, not the kind of device."""
    self.runner.store.add(_SHA256)
    client = _connect(self.runner, device=_SIMULATOR)
    _load(client, request=_load_request(device=runner_pb2.GPU))
    self.assertEqual(self.runner.load_request.device, runner_pb2.GPU)

  # ---- 11-21: the chat
  @parameterized.parameters(
      (
          grpc.StatusCode.INVALID_ARGUMENT,
          'ChatOptions.max_completion_tokens is 0; it must be at least 1.',
      ),
      (grpc.StatusCode.INTERNAL, 'The runtime failed.'),
      (
          grpc.StatusCode.FAILED_PRECONDITION,
          'The chat exists; a runner holds one.',
      ),
  )
  def test_create_chat_failures_fail_the_start(self, code, text):
    """11, 18: CreateChat failures fail the start with the runner's text."""
    self.runner.store.add(_SHA256)
    client = _connect(self.runner)
    _load(client)
    self.runner.before = _fail('CreateChat', code, text)
    with self.assertRaises(runner_client.Error) as failed:
      client.create_chat(_CHAT)
    self.assertEqual(str(failed.exception), text)

  def test_a_chat_with_no_room_to_start_is_refused_with_both_counts(self):
    """RESOURCE_EXHAUSTED at CreateChat, with both counts: NoRoomToStartError."""
    self.runner.store.add(_SHA256)
    client = _connect(self.runner)
    _load(client)

    def before(method):
      if method == 'CreateChat':
        raise fake_runner.NoRoomError(10151, 4096)

    self.runner.before = before
    with self.assertRaises(runner_client.NoRoomToStartError) as refused:
      client.create_chat(_CHAT)
    self.assertEqual(
        (refused.exception.tokens, refused.exception.left), (10151, 4096)
    )
    self.assertEqual(
        str(refused.exception),
        'The system prompt and the chat template around it need 10151 '
        'tokens; the context holds 4096. Shorten it and start again.',
    )
    self.assertIsNone(self.runner.chat_request)

  def test_send_streams_chunks_then_completion(self):
    """15, 20, 21: chunks of both kinds in order, then the Completion."""

    def reply(turn, prompt):
      del prompt  # Unused.
      chunks = [
          runner_pb2.Chunk(reasoning_content='Add.'),
          runner_pb2.Chunk(content='4'),
      ]
      completion = runner_pb2.Completion(
          content='4',
          reasoning_content='Add.',
          model_debugger=runner_pb2.ModelDebugger(turn=turn),
      )
      return chunks, completion, _FILES

    with fake_runner.FakeRunner(reply) as runner:
      client = _ready_client(runner)
      chunks = []
      completion = client.send('What is 2 + 2?', chunks.append)
      self.assertEqual(
          [c.WhichOneof('delta') for c in chunks],
          [
              'reasoning_content',
              'content',
          ],
      )
      self.assertEqual(completion.model_debugger.turn, 1)
      self.assertEqual(runner.prompts, ['What is 2 + 2?'])
      self.assertEqual(
          client.send('Again', lambda c: None).model_debugger.turn,
          2,
      )
      client.close()

  @parameterized.parameters(
      (grpc.StatusCode.INVALID_ARGUMENT, 'Something else.'),
      (grpc.StatusCode.INTERNAL, 'Something else.'),
      (grpc.StatusCode.UNKNOWN, 'Something else.'),
      (grpc.StatusCode.RESOURCE_EXHAUSTED, 'Something else.'),
      (grpc.StatusCode.FAILED_PRECONDITION, 'Something else.'),
      (grpc.StatusCode.UNAVAILABLE, 'The connection to the runner closed.'),
  )
  def test_a_failed_send_ends_the_chat(self, code, text):
    """17, 18: any failure, a refusal too, ends the chat with its reason."""

    def fail(turn, prompt):
      raise fake_runner.AbortError(code, 'Something else.')

    with fake_runner.FakeRunner(fail) as runner:
      client = _ready_client(runner)
      with self.assertRaises(runner_client.Error) as ended:
        client.send('Hello', lambda c: None)
      self.assertEqual(str(ended.exception), text)
      client.close()

  def test_a_turn_with_no_room_is_refused_and_the_chat_goes_on(self):
    """17: RESOURCE_EXHAUSTED with both counts refuses the turn alone."""
    refuse = [True]

    def reply(turn, prompt):
      if refuse[0]:
        raise fake_runner.NoRoomError(900, 700)
      return _reply(turn, prompt)

    with fake_runner.FakeRunner(reply) as runner:
      client = _ready_client(runner)
      with self.assertRaises(runner_client.NoRoomError) as refused:
        client.send('Hello', lambda c: None)
      self.assertEqual(
          (refused.exception.tokens, refused.exception.left), (900, 700)
      )
      self.assertEqual(
          str(refused.exception),
          'This turn needs 900 tokens, the chat has 700 left.',
      )
      self.assertFalse(runner.chat_over)
      refuse[0] = False
      self.assertEqual(client.send('Hi', lambda c: None).model_debugger.turn, 1)
      client.close()

  def test_a_runner_that_exited_saying_why_fails_with_its_words(self):
    """A runner gone that said why it exited: its line is the reason."""
    words = 'runner: no space left on device while writing a: full'

    def fail(turn, prompt):
      raise fake_runner.AbortError(grpc.StatusCode.UNAVAILABLE, 'Gone.')

    with fake_runner.FakeRunner(fail) as runner:
      runner.store.add(_SHA256)
      launched = _Launched(_LOCAL, runner.address, runner)
      launched.last_words = lambda: words
      client = runner_client.RunnerClient.connect(launched)
      _load(client)
      client.create_chat(_CHAT)
      with self.assertRaises(runner_client.Error) as ended:
        client.send('Hello', lambda c: None)
      self.assertEqual(
          str(ended.exception), f'The runner exited, saying: {words}'
      )
      client.close()

  def test_cancel_ends_the_chat(self):
    """19: a cancelled Send ends as stopped, and so does the chat."""
    release = threading.Event()

    def held(turn, prompt):
      del turn, prompt  # Unused.

      def chunks():
        yield runner_pb2.Chunk(content='4')
        release.wait(_TIMEOUT)
        yield runner_pb2.Chunk(content='.')

      return chunks(), runner_pb2.Completion(content='4.'), {}

    with fake_runner.FakeRunner(held) as runner:
      client = _ready_client(runner)
      first_chunk = threading.Event()
      reply = _send_on_thread(client, lambda c: first_chunk.set())
      self.assertTrue(first_chunk.wait(_TIMEOUT))
      client.cancel()
      client.cancel()
      with self.assertRaises(runner_client.Error) as cancelled:
        reply.result(_TIMEOUT)
      self.assertEqual(str(cancelled.exception), 'Cancelled.')
      release.set()
      _eventually(lambda: runner.chat_over)
      client.close()

  def test_a_cancel_before_the_send_stops_it(self):
    """19: a cancel that comes before the Send stops it all the same."""
    client = _ready_client(self.runner)
    client.cancel()
    with self.assertRaises(runner_client.Error) as cancelled:
      client.send('Hi', lambda c: None)
    self.assertEqual(str(cancelled.exception), 'Cancelled.')
    client.close()

  def test_a_completion_then_cancelled_keeps_the_turn(self):
    """19: a cancel after the Completion came too late: the turn stands.

    Its capture is not taken: the cancel stops that too, until the next
    chat, whose turns it stops none of.
    """
    with _Lingering(_reply) as runner:
      client = _ready_client(runner)
      send, calls = client._stub.Send, []

      def watched_send(request):
        calls.append(_Watched(send(request)))
        return calls[-1]

      client._stub.Send = watched_send
      reply = _send_on_thread(client)
      _eventually(lambda: calls and calls[0].completed.is_set())
      client.cancel()
      self.assertEqual(reply.result(_TIMEOUT).model_debugger.turn, 1)
      runner.gate.set()
      with self.assertRaisesRegex(runner_client.Error, 'Cancelled'):
        client.take_capture(1, self.tmp_path / 'chat-1')
      self.assertEqual(runner.released, [])
      client.create_chat(_CHAT)
      self.assertEqual(
          _send_on_thread(client).result(_TIMEOUT).model_debugger.turn,
          1,
      )
      client.take_capture(1, self.tmp_path / 'chat-2')
      self.assertEqual(runner.released, [1])
      client.close()

  def test_a_send_without_a_completion_ends_the_chat(self):
    """20: a reply that ends without its Completion ends the chat."""

    class _Short(fake_runner.FakeRunner):

      def Send(self, request, context):  # noqa: D102
        yield runner_pb2.SendResponse(chunk=runner_pb2.Chunk(content='4'))

    with _Short(_reply) as runner:
      client = _ready_client(runner)
      with self.assertRaisesRegex(runner_client.Error, 'without a Completion'):
        client.send('Hi', lambda c: None)

  def test_a_completion_over_4_mib_arrives(self):
    """U1: replies larger than gRPC's default 4 MiB still arrive."""
    big = 'x' * (5 << 20)

    def reply(turn, prompt):
      del prompt  # Unused.
      completion = runner_pb2.Completion(
          content=big, model_debugger=runner_pb2.ModelDebugger(turn=turn)
      )
      return [], completion, {}

    with fake_runner.FakeRunner(reply) as runner:
      client = _ready_client(runner)
      self.assertEqual(client.send('Hi', lambda c: None).content, big)
      client.close()

  def test_take_capture_copies_every_file_then_releases(self):
    """23, 24: every file is copied and checked, then the turn released."""
    client = _completed_turn(self.runner, self.tmp_path)
    received = []
    client.take_capture(1, self.tmp_path, on_progress=received.append)
    self.assertEqual(
        {p.name: p.read_bytes() for p in self.tmp_path.iterdir()}, _FILES
    )
    self.assertEqual(self.runner.released, [1])
    # The bytes of the files received so far, after each piece.
    self.assertEqual(received, sorted(received))
    self.assertEqual(received[-1], sum(len(data) for data in _FILES.values()))
    client.close()

  def test_a_file_in_a_subdirectory_is_copied_there(self):
    """23: a path with directories lands in them, made as needed."""

    def reply(turn, prompt):
      chunks, completion, _ = _reply(turn, prompt)
      return chunks, completion, {'layers/0/k.safetensors': b'k'}

    with fake_runner.FakeRunner(reply) as runner:
      client = _completed_turn(runner, self.tmp_path)
      client.take_capture(1, self.tmp_path)
      self.assertEqual(
          (self.tmp_path / 'layers/0/k.safetensors').read_bytes(), b'k'
      )
      client.close()

  @parameterized.parameters('sha256', 'size')
  def test_a_file_unlike_its_announcement_ends_the_chat(self, field):
    """23: bytes not as announced are not kept, nor released."""

    class _Wrong(fake_runner.FakeRunner):

      def ReadCaptures(self, request, context):  # noqa: D102
        for chunk in super().ReadCaptures(request, context):
          if field == 'sha256' and chunk.HasField('sha256'):
            chunk = runner_pb2.CaptureChunk(sha256='0' * 64)
          if field == 'size' and chunk.HasField('file'):
            chunk.file.size += 1
          yield chunk

    with _Wrong(_reply) as runner:
      client = _completed_turn(runner, self.tmp_path)
      with self.assertRaisesRegex(runner_client.Error, 'arrived as'):
        client.take_capture(1, self.tmp_path)
      self.assertFalse(list(self.tmp_path.iterdir()))
      self.assertEqual(runner.released, [])
      client.close()

  def test_a_stream_that_ends_mid_file_ends_the_chat(self):
    """23: a capture that stops inside a file is not taken."""

    class _Truncated(fake_runner.FakeRunner):

      def ReadCaptures(self, request, context):  # noqa: D102
        for chunk in super().ReadCaptures(request, context):
          yield chunk
          if chunk.HasField('data'):
            return

    with _Truncated(_reply) as runner:
      client = _completed_turn(runner, self.tmp_path)
      with self.assertRaisesRegex(
          runner_client.Error,
          'The capture ended in the middle of '
          'decode_post_logits_step_17.safetensors.',
      ):
        client.take_capture(1, self.tmp_path)
      self.assertFalse(list(self.tmp_path.iterdir()))
      self.assertEqual(runner.released, [])
      client.close()

  @parameterized.named_parameters(
      ('data_first', [runner_pb2.CaptureChunk(data=b'x')], 'data outside'),
      (
          'sha256_first',
          [runner_pb2.CaptureChunk(sha256='0' * 64)],
          'sha256 outside',
      ),
      (
          'file_in_a_file',
          [
              runner_pb2.CaptureChunk(
                  file=runner_pb2.CaptureFile(path='a', size=1)
              ),
              runner_pb2.CaptureChunk(
                  file=runner_pb2.CaptureFile(path='b', size=1)
              ),
          ],
          'began b before ending a',
      ),
      ('empty', [runner_pb2.CaptureChunk()], 'an empty CaptureChunk'),
  )
  def test_a_chunk_out_of_order_ends_the_chat(self, chunks, reason):
    """23: a stream out of the protocol's order is not taken."""

    class _Disordered(fake_runner.FakeRunner):

      def ReadCaptures(self, request, context):  # noqa: D102
        yield from chunks

    with _Disordered(_reply) as runner:
      client = _completed_turn(runner, self.tmp_path)
      with self.assertRaisesRegex(runner_client.Error, reason):
        client.take_capture(1, self.tmp_path)
      self.assertFalse(list(self.tmp_path.glob('*.part')))
      self.assertEqual(runner.released, [])
      client.close()

  def test_a_file_that_cannot_be_written_ends_the_chat(self):
    """23: a file this server cannot write is not taken, nor released."""
    (self.tmp_path / 'generated_tokens.jsonl.part').mkdir()
    client = _completed_turn(self.runner, self.tmp_path)
    with self.assertRaisesRegex(
        runner_client.Error, 'Cannot write generated_tokens.jsonl'
    ):
      client.take_capture(1, self.tmp_path)
    self.assertEqual(self.runner.released, [])
    client.close()

  def test_no_release_until_every_file_is_held(self):
    """24: a stream that fails midway keeps the capture on the runner."""
    client = _completed_turn(self.runner, self.tmp_path)
    files = []

    def second_file_fails(chunk):
      if chunk.HasField('file'):
        files.append(chunk.file.path)
        if len(files) == 2:
          raise fake_runner.AbortError(grpc.StatusCode.NOT_FOUND, 'No file.')

    self.runner.before_chunk = second_file_fails
    with self.assertRaisesRegex(runner_client.Error, 'No file.'):
      client.take_capture(1, self.tmp_path)
    self.assertEqual(self.runner.released, [])
    self.assertIn(1, self.runner.captures)
    self.assertEqual(
        [p.name for p in self.tmp_path.iterdir()], [sorted(_FILES)[0]]
    )

  @parameterized.product(
      rpc=['ReadCaptures', 'ReleaseCaptures'],
      code=[
          grpc.StatusCode.NOT_FOUND,
          grpc.StatusCode.INTERNAL,
          grpc.StatusCode.FAILED_PRECONDITION,
      ],
  )
  def test_capture_failures(self, rpc, code):
    """25: a failed capture call ends the chat."""
    client = _completed_turn(self.runner, self.tmp_path)
    self.runner.before = _fail(rpc, code, 'Failed.')
    with self.assertRaisesRegex(runner_client.Error, 'Failed.'):
      client.take_capture(1, self.tmp_path)
    if rpc != 'ReleaseCaptures':
      self.assertEqual(self.runner.released, [])

  def test_a_capture_has_no_deadline(self):
    """26: no deadline on Send or ReadCaptures, however large."""
    client = _completed_turn(self.runner, self.tmp_path)
    client.take_capture(1, self.tmp_path)
    self.assertGreater(self.runner.deadlines['Send'], 1e9)
    self.assertGreater(self.runner.deadlines['ReadCaptures'], 1e9)

  def test_close_during_take_capture_releases_nothing(self):
    """28: closing stops a capture coming; nothing is released."""
    client = _completed_turn(self.runner, self.tmp_path)
    release, reached = threading.Event(), threading.Event()
    self.runner.before_chunk = _stall_chunk('sha256', release, reached)
    outcome = []

    def take():
      try:
        client.take_capture(1, self.tmp_path)
      except runner_client.Error as error:
        outcome.append(error)

    thread = threading.Thread(target=take)
    thread.start()
    try:
      self.assertTrue(reached.wait(_TIMEOUT))
      client.close()
      thread.join(_TIMEOUT)
    finally:
      release.set()
    self.assertEqual(str(outcome[0]), 'This server closed the connection.')
    self.assertEqual(self.runner.released, [])
    self.assertFalse(list(self.tmp_path.glob('*.part')))

  def test_a_runner_that_exits_mid_capture_is_gone(self):
    """25: a runner that exits while its capture comes is gone."""
    client = _completed_turn(self.runner, self.tmp_path)
    release, reached = threading.Event(), threading.Event()
    self.runner.before_chunk = _stall_chunk('sha256', release, reached)
    outcome = []

    def take():
      try:
        client.take_capture(1, self.tmp_path)
      except runner_client.Error as error:
        outcome.append(error)

    thread = threading.Thread(target=take)
    thread.start()
    try:
      self.assertTrue(reached.wait(_TIMEOUT))
      self.runner.stop()
      thread.join(_TIMEOUT)
    finally:
      release.set()
    self.assertEqual(str(outcome[0]), 'The connection to the runner closed.')
    self.assertFalse(list(self.tmp_path.glob('*.part')))


if __name__ == '__main__':
  test_main.main()
