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

"""The runner over gRPC, with a tiny random Gemma 4: rules, feeding, capture."""

# pylint: disable=redefined-outer-name

import gzip
import json
import pathlib
import threading

import grpc
from model_debugger.protocol import runner_pb2
from model_debugger.protocol import runner_pb2_grpc
from model_debugger.runner_pytorch.cli import runner_main
from model_debugger.runner_pytorch.llm import gemma4_runtime
from model_debugger.runner_pytorch.service import model_id
from model_debugger.runner_pytorch.testing import runner_client
from model_debugger.runner_pytorch.testing import server_capture
from model_debugger.runner_pytorch.testing import test_main
from model_debugger.runner_pytorch.testing import tiny_gemma4
import pytest
from safetensors import safe_open

# pyformat: disable
from google.protobuf import struct_pb2
# pyformat: enable

_PREFILL = runner_pb2.FedToken.PREFILL
_DECODE = runner_pb2.FedToken.DECODE
_GREEDY = runner_pb2.ChatOptions(max_completion_tokens=12, top_k=1)


@pytest.fixture(scope='module')
def model_dir(tmp_path_factory) -> pathlib.Path:
  """The tiny model's directory."""
  return tiny_gemma4.write(tmp_path_factory.mktemp('model'))


@pytest.fixture
def runner(model_dir, tmp_path):
  """A runner of the tiny model, its stub and its model's id."""
  server, port, _ = runner_main.serve(
      model_dir, '127.0.0.1:0', tmp_path / 'cache'
  )
  # pylint: disable=line-too-long
  channel = grpc.insecure_channel(f'127.0.0.1:{port}')
  # pylint: enable=line-too-long
  try:
    yield (
        runner_pb2_grpc.RunnerStub(channel),
        model_id.model_id(model_dir, None),
    )
  finally:
    channel.close()
    server.stop(grace=None).wait()


def _load(stub, sha256, **fields):
  return stub.Load(
      runner_pb2.LoadRequest(
          sha256=sha256,
          device=fields.pop('device', runner_pb2.CPU),
          context_length=fields.pop('context_length', 256),
          **fields,
      )
  )


def _code(call) -> grpc.StatusCode:
  with pytest.raises(grpc.RpcError) as error:
    call()
  return error.value.code()


def test_get_info(runner):
  """Get info."""
  stub, _ = runner
  info = stub.GetInfo(runner_pb2.GetInfoRequest())
  assert info.protocol_version == '1.0.0'
  assert info.runtime.name == runner_pb2.RuntimeInfo.PYTORCH
  # pylint: disable=line-too-long
  assert info.runtime.version.startswith('torch 2.9.0, transformers 5.17.0')
  # pylint: enable=line-too-long
  assert info.host.cpu_count > 0
  assert json.loads(info.runtime.load_options_schema)['properties']


def test_upload_is_refused(runner):
  """Upload is refused."""
  stub, sha256 = runner
  header = runner_pb2.UploadModelRequest(
      header=runner_pb2.ModelHeader(sha256=sha256, size=0)
  )
  code = _code(lambda: stub.UploadModel(iter([header])))
  assert code == grpc.StatusCode.FAILED_PRECONDITION


def test_load_of_another_model_is_not_found_and_not_final(runner):
  """Load of another model is not found and not final."""
  stub, sha256 = runner
  assert _code(lambda: _load(stub, 'a' * 64)) == grpc.StatusCode.NOT_FOUND
  loaded = _load(stub, sha256)
  assert loaded.start_token == tiny_gemma4.BOS
  assert [list(token.ids.ids) for token in loaded.stop_tokens] == [
      [tiny_gemma4.EOS],
      [tiny_gemma4.TURN_END],
  ]
  assert loaded.sampler_defaults.top_k == 8
  assert _code(lambda: _load(stub, sha256)) == (
      grpc.StatusCode.FAILED_PRECONDITION
  )


def test_wrong_options_fail_the_load_for_good(runner):
  """Wrong options fail the load for good."""
  stub, sha256 = runner
  options = struct_pb2.Struct()
  options.update({'dtype': 'int4'})
  code = _code(lambda: _load(stub, sha256, runtime_options=options))
  assert code == grpc.StatusCode.INVALID_ARGUMENT
  assert _code(lambda: _load(stub, sha256)) == (
      grpc.StatusCode.FAILED_PRECONDITION
  )


def test_chat_options_out_of_range_are_refused(runner):
  """Chat options out of range are refused."""
  stub, sha256 = runner
  _load(stub, sha256)
  request = runner_pb2.CreateChatRequest(
      options=runner_pb2.ChatOptions(max_completion_tokens=0)
  )
  code = _code(lambda: stub.CreateChat(request))
  assert code == grpc.StatusCode.INVALID_ARGUMENT


def test_turns_feed_and_capture_as_litert_lm(runner, tmp_path):
  """Turns feed and capture as litert lm."""
  stub, sha256 = runner
  load = runner_pb2.LoadRequest(
      sha256=sha256, device=runner_pb2.CPU, context_length=256
  )
  loaded = stub.Load(load)
  chat = runner_pb2.CreateChatRequest(options=_GREEDY)
  stub.CreateChat(chat)
  prompts = ['Name three colors.', 'More?']
  turns = []
  for n, prompt in enumerate(prompts, 1):
    content, completion = runner_client.send(stub, prompt)
    assert content == completion.content
    debugger = completion.model_debugger
    assert debugger.turn == n
    fed = list(debugger.fed_tokens)
    decodes = [token for token in fed if token.phase == _DECODE]
    usage = completion.usage
    assert usage.prompt_tokens == decodes[0].position + 1
    assert usage.completion_tokens == len(decodes)
    assert usage.total_tokens == usage.prompt_tokens + len(decodes)
    assert debugger.context_token_count == fed[-1].position + 2
    positions = [token.position for token in fed]
    assert positions == list(range(positions[0], positions[-1] + 1))
    assert all(token.phase == _PREFILL for token in fed[: -len(decodes)])
    if n == 1:
      assert positions[0] == 0
      assert fed[0].id == tiny_gemma4.BOS
    directory = tmp_path / f'turn-{n}'
    names = runner_client.read_captures(stub, n, directory)
    assert 'generated_tokens.jsonl' in names
    assert f'decode_post_logits_step_{usage.prompt_tokens}.safetensors' in (
        names
    )
    _check_kv_metadata(directory, decodes[-1].position + 1)
    turns.append((prompt, completion, directory))
  info = stub.GetInfo(runner_pb2.GetInfoRequest())
  entries = server_capture.check_turns(
      tmp_path / 'index',
      {'info': info, 'load': load, 'chat': chat, 'loaded': loaded},
      turns,
  )
  for entry in entries:
    kv = entry['sides']['ref']['kv']
    assert kv[-1]['moment'] == 'terminal'
    assert len(kv[-1]['layers']) == tiny_gemma4.CACHED_LAYERS
  stub.ReleaseCaptures(runner_pb2.ReleaseCapturesRequest(turn=1))
  code = _code(
      lambda: list(stub.ReadCaptures(runner_pb2.ReadCapturesRequest(turn=1)))
  )
  assert code == grpc.StatusCode.NOT_FOUND


def _observing(sha256, **options) -> runner_pb2.LoadRequest:
  """A Load that observes the decoder layers' anchors."""
  struct = struct_pb2.Struct()
  struct.update({'observe': 'anchors', **options})
  return runner_pb2.LoadRequest(
      sha256=sha256,
      device=runner_pb2.CPU,
      context_length=256,
      runtime_options=struct,
  )


def test_a_load_that_observes_captures_every_layers_anchors(runner, tmp_path):
  """Each forward pass's four anchors of every layer, in LiteRT's layout."""
  stub, sha256 = runner
  load = _observing(sha256)
  loaded = stub.Load(load)
  chat = runner_pb2.CreateChatRequest(options=_GREEDY)
  stub.CreateChat(chat)
  turns = []
  for n, prompt in enumerate(['Name three colors.', 'More?'], 1):
    _, completion = runner_client.send(stub, prompt)
    directory = tmp_path / f'turn-{n}'
    names = runner_client.read_captures(stub, n, directory)
    observation = json.loads((directory / 'observation.json').read_text())
    assert observation['captured']
    assert not observation['reason']
    assert observation['memory_limit_bytes'] == 8 * 2**30
    assert observation['snapshot_bytes'] > 0
    assert not observation['modules']
    # The first turn on the CPU is captured fully and writes the
    # execution graph of each signature; a later one is captured
    # module-only, and writes none.
    assert observation['capture'] == ('full' if n == 1 else 'modules')
    assert not observation['execution_graph_reason']
    if n == 1:
      assert observation['execution_graph'] == ['decode', 'prefill']
      data = (directory / 'execution_graph.json.gz').read_bytes()
      graph = json.loads(gzip.decompress(data))
      assert [g['id'] for g in graph['graphs']] == ['prefill', 'decode']
    else:
      assert not observation['execution_graph']
      assert 'execution_graph.json.gz' not in names
    fed = completion.model_debugger.fed_tokens
    decodes = [token for token in fed if token.phase == _DECODE]
    prefill_rows = len(fed) - len(decodes)
    layers = len(tiny_gemma4.LAYER_TYPES)
    anchors = sorted(name for name in names if '_post_anchor_' in name)
    assert len(anchors) == (1 + len(decodes)) * layers * 4
    step = decodes[0].position + 1
    name = f'prefill_post_anchor_5_intermediate.1_step_{step}.safetensors'
    with safe_open(directory / name, 'pt') as file:
      assert file.metadata() == {
          'signature': 'prefill',
          'step': str(step),
          'layer': '5',
          'anchor': 'intermediate.1',
          'module': 'model.language_model.layers.5.pre_feedforward_layernorm',
          'edge': 'in',
          'path': 'args.0',
      }
      assert file.keys() == ['post_anchor_5_intermediate.1']
      shape = file.get_slice(file.keys()[0]).get_shape()
    assert shape == [1, prefill_rows, 64]
    turns.append((prompt, completion, directory))
  info = stub.GetInfo(runner_pb2.GetInfoRequest())
  entries = server_capture.check_turns(
      tmp_path / 'index',
      {'info': info, 'load': load, 'chat': chat, 'loaded': loaded},
      turns,
  )
  for entry in entries:
    side = entry['sides']['ref']
    assert side['observation']['captured']
    for forward in side['forwards']:
      assert 'post_anchor_0_input' in forward['tensors']


def test_a_load_captures_the_modules_it_asks_for(runner, tmp_path):
  """Each module asked for, per forward pass, beside the anchors."""
  stub, sha256 = runner
  modules = [
      {
          'module': 'model.language_model.layers.1.self_attn.q_proj',
          'edge': 'out',
      },
      {'module': 'model.language_model.layers.0.mlp.down_proj', 'edge': 'in'},
  ]
  load = _observing(sha256, observeModules=modules)
  loaded = stub.Load(load)
  chat = runner_pb2.CreateChatRequest(options=_GREEDY)
  stub.CreateChat(chat)
  turns = []
  for n, prompt in enumerate(['Name three colors.', 'More?'], 1):
    _, completion = runner_client.send(stub, prompt)
    directory = tmp_path / f'turn-{n}'
    names = runner_client.read_captures(stub, n, directory)
    observation = json.loads((directory / 'observation.json').read_text())
    assert observation['captured'], observation['reason']
    assert observation['modules'] == [
        f'{m["module"]}:{m["edge"]}' for m in modules
    ]
    fed = completion.model_debugger.fed_tokens
    passes = 1 + sum(token.phase == _DECODE for token in fed)
    found = sorted(name for name in names if '_post_module_' in name)
    assert len(found) == passes * len(modules)
    step = next(t for t in fed if t.phase == _DECODE).position + 1
    name = (
        'decode_post_module_model.language_model.layers.0.mlp.down_proj'
        f'_in_step_{step}.safetensors'
    )
    with safe_open(directory / name, 'pt') as file:
      assert file.metadata() == {
          'signature': 'decode',
          'step': str(step),
          'module': 'model.language_model.layers.0.mlp.down_proj',
          'edge': 'in',
          'path': 'args.0',
      }
      key = 'post_module_model.language_model.layers.0.mlp.down_proj_in'
      assert file.keys() == [key]
      assert file.get_slice(key).get_shape()[:2] == [1, 1]
    turns.append((prompt, completion, directory))
  info = stub.GetInfo(runner_pb2.GetInfoRequest())
  server_capture.check_turns(
      tmp_path / 'index',
      {'info': info, 'load': load, 'chat': chat, 'loaded': loaded},
      turns,
  )


def test_a_load_that_asks_for_no_module_of_the_model_fails(runner):
  """A module the model does not have fails the Load, saying which."""
  stub, sha256 = runner
  code = _code(
      lambda: stub.Load(
          _observing(
              sha256,
              observeModules=[{'module': 'layers.99', 'edge': 'out'}],
          )
      )
  )
  assert code == grpc.StatusCode.INTERNAL


def test_a_budget_too_small_completes_the_turn_unobserved(
    runner, model_dir, tmp_path
):
  """The turn is the one an unobserved runner runs; the capture says why."""
  stub, sha256 = runner
  stub.Load(_observing(sha256, observeMemoryLimitBytes=1000))
  stub.CreateChat(runner_pb2.CreateChatRequest(options=_GREEDY))
  content, completion = runner_client.send(stub, 'Name three colors.')
  names = runner_client.read_captures(stub, 1, tmp_path / 'turn-1')
  assert not [name for name in names if '_post_anchor_' in name]
  observation = json.loads(
      (tmp_path / 'turn-1' / 'observation.json').read_text()
  )
  assert not observation['captured']
  assert observation['memory_limit_bytes'] == 1000
  assert observation['reason'].startswith('collect() failed at prefill step')
  assert 'MemoryError: Snapshot memory limit exceeded' in (
      observation['reason']
  )
  server, port, _ = runner_main.serve(
      model_dir, '127.0.0.1:0', tmp_path / 'plain'
  )
  # pylint: disable=line-too-long
  channel = grpc.insecure_channel(f'127.0.0.1:{port}')
  # pylint: enable=line-too-long
  try:
    plain = runner_pb2_grpc.RunnerStub(channel)
    _load(plain, sha256)
    plain.CreateChat(runner_pb2.CreateChatRequest(options=_GREEDY))
    expected, unobserved = runner_client.send(plain, 'Name three colors.')
    names = runner_client.read_captures(plain, 1, tmp_path / 'plain-1')
    assert 'observation.json' not in names
  finally:
    channel.close()
    server.stop(grace=None).wait()
  assert content == expected
  assert list(completion.model_debugger.fed_tokens) == list(
      unobserved.model_debugger.fed_tokens
  )


def test_a_later_turn_feeds_from_the_held_token(runner):
  """A later turn feeds from the held token."""
  stub, sha256 = runner
  _load(stub, sha256)
  # Ended by length, the reply renders as it was fed: the cache is kept.
  stub.CreateChat(
      runner_pb2.CreateChatRequest(
          options=runner_pb2.ChatOptions(max_completion_tokens=1, top_k=1)
      )
  )
  _, first = runner_client.send(stub, 'Hi')
  _, second = runner_client.send(stub, 'Hi')
  assert first.finish_reason == runner_pb2.FINISH_REASON_LENGTH
  held = second.model_debugger.fed_tokens[0]
  assert held.position == first.model_debugger.context_token_count - 1
  assert held.phase == _PREFILL
  assert held.text == first.content


def test_a_cancelled_send_ends_the_chat_until_a_create_chat(runner):
  """A cancelled send ends the chat, until a CreateChat replaces it."""
  stub, sha256 = runner
  _load(stub, sha256, context_length=512)
  stub.CreateChat(
      runner_pb2.CreateChatRequest(
          options=runner_pb2.ChatOptions(max_completion_tokens=400, top_k=1)
      )
  )
  call = stub.Send(
      runner_pb2.SendRequest(
          message=runner_pb2.ChatMessage(
              role=runner_pb2.ChatMessage.USER, content='Count.'
          )
      )
  )
  # Once a chunk came, the runner runs the turn.
  assert next(call).HasField('chunk')
  call.cancel()
  with pytest.raises(grpc.RpcError):
    list(call)
  # The runner sees the cancel between two forward passes.
  for _ in range(100):
    code = _code(lambda: runner_client.send(stub, 'Again.'))
    if code != grpc.StatusCode.FAILED_PRECONDITION:
      break
    if 'ended' in _details(lambda: runner_client.send(stub, 'Again.')):
      break
  assert 'The chat has ended' in _details(
      lambda: runner_client.send(stub, 'Again.')
  )
  stub.CreateChat(runner_pb2.CreateChatRequest(options=_GREEDY))
  _, completion = runner_client.send(stub, 'Again.')
  assert completion.model_debugger.turn == 1


def _chat_with_system(system: str) -> runner_pb2.CreateChatRequest:
  """A greedy chat that starts with the system message `system`."""
  return runner_pb2.CreateChatRequest(
      messages=[
          runner_pb2.ChatMessage(
              role=runner_pb2.ChatMessage.SYSTEM, content=system
          )
      ],
      options=_GREEDY,
  )


def _read_code(stub, turn: int) -> grpc.StatusCode:
  """The status code of reading turn `turn`'s capture."""
  return _code(
      lambda: list(stub.ReadCaptures(runner_pb2.ReadCapturesRequest(turn=turn)))
  )


def test_a_create_chat_replaces_the_chat(runner, tmp_path):
  """The old chat's captures go; the new one starts from position 0."""
  stub, sha256 = runner
  _load(stub, sha256)
  stub.CreateChat(_chat_with_system('Be brief.'))
  for _ in range(2):
    runner_client.send(stub, 'Name three colors.')
  stub.CreateChat(_chat_with_system('Be kind.'))
  assert _read_code(stub, 1) == grpc.StatusCode.NOT_FOUND
  assert _read_code(stub, 2) == grpc.StatusCode.NOT_FOUND
  _, completion = runner_client.send(stub, 'Hi')
  debugger = completion.model_debugger
  assert debugger.turn == 1
  fed = list(debugger.fed_tokens)
  assert (fed[0].position, fed[0].id) == (0, tiny_gemma4.BOS)
  prompt = ''.join(token.text for token in fed if token.phase == _PREFILL)
  assert 'Be kind.' in prompt
  assert 'Be brief.' not in prompt
  assert 'Name three colors.' not in prompt
  names = runner_client.read_captures(stub, 1, tmp_path / 'turn-1')
  assert 'generated_tokens.jsonl' in names


def test_a_create_chat_during_a_send_is_refused(runner, monkeypatch):
  """A CreateChat during a Send is refused, and the chat goes on."""
  stub, sha256 = runner
  _load(stub, sha256)
  stub.CreateChat(runner_pb2.CreateChatRequest(options=_GREEDY))
  started, go = threading.Event(), threading.Event()
  send = gemma4_runtime.Chat.send

  def held(self, *args):
    started.set()
    go.wait()
    return send(self, *args)

  monkeypatch.setattr(gemma4_runtime.Chat, 'send', held)
  replies = []
  sending = threading.Thread(
      target=lambda: replies.append(runner_client.send(stub, 'Hi'))
  )
  sending.start()
  try:
    assert started.wait(30)
    request = runner_pb2.CreateChatRequest(options=_GREEDY)
    assert _details(lambda: stub.CreateChat(request)) == (
        'A Send is in progress.'
    )
  finally:
    go.set()
    sending.join()
  assert replies[0][1].model_debugger.turn == 1
  _, second = runner_client.send(stub, 'Hi')
  assert second.model_debugger.turn == 2


def test_a_failed_create_chat_leaves_no_chat(runner):
  """A refused CreateChat closes the old chat, and the next may succeed."""
  stub, sha256 = runner
  _load(stub, sha256)
  stub.CreateChat(runner_pb2.CreateChatRequest(options=_GREEDY))
  runner_client.send(stub, 'Hi')
  request = runner_pb2.CreateChatRequest(
      options=runner_pb2.ChatOptions(max_completion_tokens=0)
  )
  assert _code(lambda: stub.CreateChat(request)) == (
      grpc.StatusCode.INVALID_ARGUMENT
  )
  assert _details(lambda: runner_client.send(stub, 'Hi')) == 'No chat exists.'
  assert _read_code(stub, 1) == grpc.StatusCode.NOT_FOUND
  stub.CreateChat(runner_pb2.CreateChatRequest(options=_GREEDY))
  _, completion = runner_client.send(stub, 'Hi')
  assert completion.model_debugger.turn == 1


def test_a_create_chat_with_no_room_to_start_is_refused_with_its_counts(
    runner,
):
  """A system prompt the context cannot hold is refused at CreateChat."""
  stub, sha256 = runner
  _load(stub, sha256, context_length=64)
  request = runner_pb2.CreateChatRequest(
      messages=[
          runner_pb2.ChatMessage(
              role=runner_pb2.ChatMessage.SYSTEM, content='one ' * 128
          )
      ],
      options=_GREEDY,
  )
  with pytest.raises(grpc.RpcError) as error:
    stub.CreateChat(request)
  assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
  counts = dict(error.value.trailing_metadata())
  assert int(counts['turn-tokens']) > 64
  assert counts['tokens-left'] == '64'
  assert _details(lambda: runner_client.send(stub, 'Hi')) == 'No chat exists.'
  del request.messages[0]
  request.messages.add(role=runner_pb2.ChatMessage.SYSTEM, content='Be terse.')
  stub.CreateChat(request)
  _, completion = runner_client.send(stub, 'Hi')
  assert completion.model_debugger.turn == 1


def test_a_turn_with_no_room_is_refused_and_the_chat_goes_on(runner):
  """RESOURCE_EXHAUSTED with both counts; the next turn is turn 1."""
  stub, sha256 = runner
  _load(stub, sha256, context_length=64)
  stub.CreateChat(runner_pb2.CreateChatRequest(options=_GREEDY))
  with pytest.raises(grpc.RpcError) as error:
    runner_client.send(stub, 'one ' * 128)
  assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
  counts = dict(error.value.trailing_metadata())
  assert counts['tokens-left'] == '64'
  assert int(counts['turn-tokens']) > 64
  _, completion = runner_client.send(stub, 'Hi')
  assert completion.model_debugger.turn == 1


def _details(call) -> str:
  with pytest.raises(grpc.RpcError) as error:
    call()
  return error.value.details()


def _check_kv_metadata(directory: pathlib.Path, step: int) -> None:
  """Checks the last decode step's KV files: layout, and sliding positions."""
  for layer, kind in enumerate(tiny_gemma4.LAYER_TYPES[:4]):
    name = f'decode_post_kv_cache_k_{layer}_step_{step}.safetensors'
    with safe_open(directory / name, 'pt') as file:
      metadata = file.metadata()
      length = file.get_slice(file.keys()[0]).get_shape()[2]
    assert metadata['layout'] == 'k:bhsd,v:bhsd'
    if kind == 'sliding_attention':
      first, last = map(int, metadata['kv_positions'].split('-'))
      assert (last, last - first + 1) == (step - 1, length)
      assert length == tiny_gemma4.SLIDING_WINDOW - 1
    else:
      assert 'kv_positions' not in metadata
      assert length == step


if __name__ == '__main__':
  test_main.main()
