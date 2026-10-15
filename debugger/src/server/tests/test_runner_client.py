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

"""Tests of RunnerClient against the in-process fake runner."""

import hashlib
import socket
import threading
import time

import fake_runner
import grpc
from model_debugger.v1 import runner_pb2, runner_pb2_grpc
from model_debugger_server import runner_client
import pytest

_MODEL = b'not really a model' * 1000
_SHA256 = hashlib.sha256(_MODEL).hexdigest()
_FILES = {
    'decode_post_logits_step_17.safetensors': bytes(range(256)) * 9000,
    'generated_tokens.jsonl': b'{"token_ids":[[236778]],"texts":["4"]}\n',
}


def _reply(turn: int, prompt: str) -> fake_runner.Reply:
  chunks = [runner_pb2.Chunk(content='4'), runner_pb2.Chunk(content='.')]
  completion = runner_pb2.Completion(
      content='4.',
      finish_reason=runner_pb2.FINISH_REASON_STOP,
      model_debugger=runner_pb2.ModelDebugger(turn=turn),
  )
  return chunks, completion, _FILES


def _ready_client(runner: fake_runner.FakeRunner) -> runner_client.RunnerClient:
  """Returns a client of `runner` with a model loaded and a chat created."""
  runner.store.add(_SHA256)
  client = runner_client.RunnerClient(runner.address)
  client.load(
      runner_pb2.LoadRequest(
          sha256=_SHA256, device=runner_pb2.CPU, context_length=4096
      )
  )
  client.create_chat(
      runner_pb2.CreateChatRequest(
          options=runner_pb2.ChatOptions(max_completion_tokens=8, top_k=1)
      )
  )
  return client


@pytest.fixture
def runner():
  """A started fake runner."""
  with fake_runner.FakeRunner(_reply) as started:
    yield started


def test_get_info(runner):
  """GetInfo answers with the runner's protocol version and runtime."""
  client = runner_client.RunnerClient(runner.address)
  info = client.get_info()
  assert info.protocol_version == runner_client.PROTOCOL_VERSION
  assert info.runtime.name == runner_pb2.RuntimeInfo.LITERT_LM
  client.close()


def test_an_idle_channel_keeps_its_connection(runner):
  """Calls separated by idle time come from one peer."""
  # The control: gRPC's idle timeout (1s at least) makes the next call
  # reconnect from a new port, which a runner refuses.
  with fake_runner.FakeRunner(_reply) as control:
    channel = grpc.secure_channel(
        control.address,
        grpc.local_channel_credentials(grpc.LocalConnectionType.LOCAL_TCP),
        options=[
            ('grpc.client_idle_timeout_ms', 1000),
            ('grpc.use_local_subchannel_pool', 1),
        ],
    )
    stub = runner_pb2_grpc.RunnerStub(channel)
    stub.GetInfo(runner_pb2.GetInfoRequest())
    time.sleep(3)
    with pytest.raises(grpc.RpcError) as refused:
      stub.GetInfo(runner_pb2.GetInfoRequest())
    assert refused.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert control.peers[0] != control.peers[1]
    channel.close()

  client = runner_client.RunnerClient(runner.address)
  client.get_info()
  time.sleep(3)
  client.get_info()
  time.sleep(1)
  client.get_info()
  assert len(set(runner.peers)) == 1
  client.close()


def test_clients_of_two_runners_each_keep_their_own():
  """Two clients of two runners each stay one peer of their runner."""
  with (
      fake_runner.FakeRunner(_reply) as first,
      fake_runner.FakeRunner(_reply) as second,
  ):
    one = runner_client.RunnerClient(first.address)
    two = runner_client.RunnerClient(second.address)
    for _ in range(3):
      one.get_info()
      two.get_info()
    assert len(first.peers) == len(second.peers) == 3
    assert len(set(first.peers)) == len(set(second.peers)) == 1
    one.close()
    two.close()


def test_a_second_client_of_one_runner_is_refused(runner):
  """A second client of a runner is a second peer, and refused."""
  # Channels to one address do not share a connection: the second client is
  # a second peer, as it would be to a real runner.
  first = runner_client.RunnerClient(runner.address)
  second = runner_client.RunnerClient(runner.address)
  first.get_info()
  with pytest.raises(runner_client.RunnerError) as refused:
    second.get_info()
  assert refused.value.code == grpc.StatusCode.FAILED_PRECONDITION
  assert str(refused.value) == 'This runner is serving another client.'
  first.get_info()
  first.close()
  second.close()


def test_load_not_found_then_upload(runner, tmp_path):
  """NOT_FOUND from Load is answered by an upload, with progress."""
  model = tmp_path / 'model.litertlm'
  model.write_bytes(_MODEL * 200)  # 3.6 MB: four data messages
  sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
  request = runner_pb2.LoadRequest(
      sha256=sha256, device=runner_pb2.CPU, context_length=4096
  )
  client = runner_client.RunnerClient(runner.address)
  with pytest.raises(runner_client.RunnerError) as missing:
    client.load(request)
  assert missing.value.code == grpc.StatusCode.NOT_FOUND

  progress = []
  client.upload_model(model, sha256, lambda *p: progress.append(p))
  size = model.stat().st_size
  assert progress[-1] == (size, size)
  assert [sent for sent, _ in progress] == sorted(sent for sent, _ in progress)
  assert len(progress) == 4
  assert runner.uploads == [sha256]

  response = client.load(request)
  assert list(response.stop_tokens[2].ids.ids) == [106]
  with pytest.raises(runner_client.RunnerError) as again:
    client.load(request)
  assert again.value.code == grpc.StatusCode.FAILED_PRECONDITION
  client.close()


def test_upload_of_bytes_that_do_not_match_is_data_loss(runner, tmp_path):
  """An upload whose bytes do not hash to its sha256 is DATA_LOSS."""
  model = tmp_path / 'model.litertlm'
  model.write_bytes(_MODEL)
  client = runner_client.RunnerClient(runner.address)
  with pytest.raises(runner_client.RunnerError) as lost:
    client.upload_model(model, '0' * 64, lambda *p: None)
  assert lost.value.code == grpc.StatusCode.DATA_LOSS
  assert runner.store == set()
  client.close()


def test_upload_of_chunk_exceeding_one_mib_is_invalid_argument(runner):
  """An UploadModel data message larger than 1 MiB is INVALID_ARGUMENT."""
  oversized = b'x' * ((1 << 20) + 1)
  sha256 = hashlib.sha256(oversized).hexdigest()
  client = runner_client.RunnerClient(runner.address)

  def requests():
    yield runner_pb2.UploadModelRequest(
        header=runner_pb2.ModelHeader(sha256=sha256, size=len(oversized))
    )
    yield runner_pb2.UploadModelRequest(data=oversized)

  with pytest.raises(grpc.RpcError) as refused:
    client._stub.UploadModel(requests())
  assert refused.value.code() == grpc.StatusCode.INVALID_ARGUMENT
  assert runner.store == set()
  client.close()


def test_send_streams_chunks_then_completion(runner):
  """Send delivers its chunks in order, then the completion."""
  client = _ready_client(runner)
  chunks = []
  completion = client.send('What is 2 + 2?', chunks.append).wait()
  assert [c.content for c in chunks] == ['4', '.']
  assert completion.content == '4.'
  assert completion.model_debugger.turn == 1
  assert runner.prompts == ['What is 2 + 2?']
  assert (
      client.send('And times 3?', lambda c: None).wait().model_debugger.turn
      == 2
  )
  client.close()


def test_cancel_ends_the_chat():
  """A cancelled Send fails with CANCELLED and ends the chat."""
  release = threading.Event()

  def held(turn, prompt):
    def chunks():
      yield runner_pb2.Chunk(content='4')
      release.wait(10)
      yield runner_pb2.Chunk(content='.')

    return chunks(), runner_pb2.Completion(content='4.'), {}

  with fake_runner.FakeRunner(held) as runner:
    client = _ready_client(runner)
    first_chunk = threading.Event()
    generation = client.send('What is 2 + 2?', lambda c: first_chunk.set())
    assert first_chunk.wait(10)
    generation.cancel()
    generation.cancel()
    with pytest.raises(runner_client.RunnerError) as cancelled:
      generation.wait()
    assert cancelled.value.code == grpc.StatusCode.CANCELLED
    release.set()
    deadline = time.monotonic() + 10
    while not runner.chat_over and time.monotonic() < deadline:
      time.sleep(0.05)
    assert runner.chat_over
    with pytest.raises(runner_client.RunnerError) as over:
      client.send('Again?', lambda c: None).wait()
    assert over.value.code == grpc.StatusCode.FAILED_PRECONDITION
    client.close()


def test_invalid_argument_leaves_the_chat():
  """A Send refused with INVALID_ARGUMENT leaves the chat as it was."""

  def refuse_first(turn, prompt):
    if prompt == 'refused':
      raise fake_runner.Abort(grpc.StatusCode.INVALID_ARGUMENT, 'No.')
    return _reply(turn, prompt)

  with fake_runner.FakeRunner(refuse_first) as runner:
    client = _ready_client(runner)
    with pytest.raises(runner_client.RunnerError) as refused:
      client.send('refused', lambda c: None).wait()
    assert refused.value.code == grpc.StatusCode.INVALID_ARGUMENT
    assert str(refused.value) == 'No.'
    completion = client.send('accepted', lambda c: None).wait()
    assert completion.model_debugger.turn == 1
    client.close()


def test_a_failed_send_ends_the_chat():
  """A Send that fails otherwise ends the chat."""

  def fail(turn, prompt):
    raise fake_runner.Abort(grpc.StatusCode.INTERNAL, 'The runtime failed.')

  with fake_runner.FakeRunner(fail) as runner:
    client = _ready_client(runner)
    with pytest.raises(runner_client.RunnerError) as failed:
      client.send('Hello', lambda c: None).wait()
    assert failed.value.code == grpc.StatusCode.INTERNAL
    with pytest.raises(runner_client.RunnerError) as over:
      client.send('Hello', lambda c: None).wait()
    assert over.value.code == grpc.StatusCode.FAILED_PRECONDITION
    client.close()


def test_captures_are_listed_read_and_released(runner, tmp_path):
  """A turn's capture is listed, copied byte for byte and released."""
  client = _ready_client(runner)
  client.send('What is 2 + 2?', lambda c: None).wait()
  files = client.list_captures(1)
  assert [f.path for f in files] == sorted(_FILES)
  for file in files:
    client.read_capture(1, file, tmp_path / file.path)
    assert (tmp_path / file.path).read_bytes() == _FILES[file.path]
  assert sorted(p.name for p in tmp_path.iterdir()) == sorted(_FILES)

  client.release_captures(1)
  assert runner.released == [1]
  with pytest.raises(runner_client.RunnerError) as released:
    client.list_captures(1)
  assert released.value.code == grpc.StatusCode.NOT_FOUND
  assert str(released.value) == 'Turn 1 has no capture.'
  client.close()


def test_a_capture_file_that_does_not_match_its_listing_is_data_loss(
    runner, tmp_path
):
  """A file that does not match its listing is DATA_LOSS and not kept."""
  client = _ready_client(runner)
  client.send('What is 2 + 2?', lambda c: None).wait()
  file = client.list_captures(1)[0]
  wrong = runner_pb2.CaptureFile(path=file.path, size=file.size, sha256='0')
  destination = tmp_path / file.path
  with pytest.raises(runner_client.RunnerError) as lost:
    client.read_capture(1, wrong, destination)
  assert lost.value.code == grpc.StatusCode.DATA_LOSS
  assert list(tmp_path.iterdir()) == []
  client.close()


def test_an_unknown_capture_file_is_not_found(runner, tmp_path):
  """Reading a file the turn does not have is NOT_FOUND."""
  client = _ready_client(runner)
  client.send('What is 2 + 2?', lambda c: None).wait()
  missing = runner_pb2.CaptureFile(path='nope.safetensors', size=0, sha256='')
  with pytest.raises(runner_client.RunnerError) as not_found:
    client.read_capture(1, missing, tmp_path / 'nope.safetensors')
  assert not_found.value.code == grpc.StatusCode.NOT_FOUND
  assert list(tmp_path.iterdir()) == []
  client.close()


def test_no_runner_is_unavailable():
  """A call to an address nobody listens on is UNAVAILABLE."""
  with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    address = f'127.0.0.1:{probe.getsockname()[1]}'
  client = runner_client.RunnerClient(address)
  with pytest.raises(runner_client.RunnerError) as unavailable:
    client.get_info()
  assert unavailable.value.code == grpc.StatusCode.UNAVAILABLE
  client.close()


def test_read_capture_aborts_immediately_when_stream_exceeds_listed_size(
    runner, tmp_path
):
  """A capture stream that exceeds file.size is aborted before reading more."""
  client = _ready_client(runner)
  client.send('What is 2 + 2?', lambda c: None).wait()
  file = client.list_captures(1)[0]
  smaller = runner_pb2.CaptureFile(path=file.path, size=1, sha256=file.sha256)
  chunks_yielded = 0

  def oversized_stream(_request, timeout=None):
    assert timeout == runner_client._READ_CAPTURE_SECONDS
    nonlocal chunks_yielded
    for chunk in (b'ab', b'cd', b'ef'):
      chunks_yielded += 1
      yield runner_pb2.CaptureData(data=chunk)

  client._stub.ReadCapture = oversized_stream
  with pytest.raises(runner_client.RunnerError) as lost:
    client.read_capture(1, smaller, tmp_path / file.path)
  assert lost.value.code == grpc.StatusCode.DATA_LOSS
  assert chunks_yielded == 1
  assert list(tmp_path.iterdir()) == []
  client.close()
