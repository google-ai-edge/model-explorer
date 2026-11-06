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

"""The runner command with a real Gemma 4, as the server's launcher runs it.

Runs only when RUNNER_PYTORCH_MODEL names a Gemma 4 model directory, such as
Gemma 4 E2B's Hugging Face snapshot. RUNNER_PYTORCH_DEVICE is CPU or GPU
(default GPU where there is one), RUNNER_PYTORCH_DTYPE a dtype option
(default the checkpoint's), RUNNER_PYTORCH_OBSERVE the `observe` option
(default anchors). The capture is checked by the server's capture index, in
the capture's own dtype, BF16 too. Observed, each forward pass has every
layer's four anchors and the two modules asked for (_MODULES), on the CPU
and the GPU alike; on the CPU the first turn is captured fully and holds the
execution graph of each signature, the second module-only.
"""

# pylint: disable=redefined-outer-name

import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

import grpc
from model_debugger.protocol import runner_pb2
from model_debugger.protocol import runner_pb2_grpc
from model_debugger.runner_pytorch.service import model_id
from model_debugger.runner_pytorch.testing import runner_client
from model_debugger.runner_pytorch.testing import server_capture
from model_debugger.runner_pytorch.testing import test_main
import pytest

# pyformat: disable
from google.protobuf import struct_pb2
# pyformat: enable

_MODEL = os.environ.get('RUNNER_PYTORCH_MODEL', '')
_OBSERVE = os.environ.get('RUNNER_PYTORCH_OBSERVE') or 'anchors'
# The modules an observing Load asks for beside the anchors: what layer 3's
# q_proj returns, what layer 20's down_proj reads.
_MODULES = [
    {'module': 'model.language_model.layers.3.self_attn.q_proj', 'edge': 'out'},
    {'module': 'model.language_model.layers.20.mlp.down_proj', 'edge': 'in'},
]
# What the server's local launcher waits for (local_launcher.py).
_LISTENING = re.compile(r'runner: listening on port (\d+)')

pytestmark = pytest.mark.skipif(
    not _MODEL, reason='RUNNER_PYTORCH_MODEL names no model directory.'
)


@pytest.fixture
def runner():
  """The runner command of the model, its stub and its info."""
  process = subprocess.Popen(
      [
          sys.executable,
          '-m',
          'model_debugger.runner_pytorch.cli.runner_main',
          '--model_dir',
          _MODEL,
          '--listen',
          '127.0.0.1:0',
      ],
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
  )
  port = None
  for line in process.stdout:
    match = _LISTENING.fullmatch(line.rstrip('\n'))
    if match:
      port = int(match.group(1))
      break
  assert port is not None, 'The runner exited before listening.'
  # pylint: disable=line-too-long
  channel = grpc.insecure_channel(f'127.0.0.1:{port}')
  # pylint: enable=line-too-long
  try:
    stub = runner_pb2_grpc.RunnerStub(channel)
    yield stub, stub.GetInfo(runner_pb2.GetInfoRequest())
  finally:
    channel.close()
    process.terminate()
    process.wait(30)


def test_a_turn_of_the_real_model(runner, tmp_path):
  """A turn of the real model."""
  stub, info = runner
  has_gpu = info.host.HasField('gpu')
  device = os.environ.get('RUNNER_PYTORCH_DEVICE', 'GPU' if has_gpu else 'CPU')
  options = struct_pb2.Struct()
  options.update({'observe': _OBSERVE})
  if _OBSERVE == 'anchors':
    options.update({'observeModules': _MODULES})
  if os.environ.get('RUNNER_PYTORCH_DTYPE'):
    options.update({'dtype': os.environ['RUNNER_PYTORCH_DTYPE']})
  load = runner_pb2.LoadRequest(
      sha256=model_id.model_id(pathlib.Path(_MODEL), None),
      device=runner_pb2.Device.Value(device),
      context_length=1024,
      runtime_options=options,
  )
  loaded = stub.Load(load)
  chat = runner_pb2.CreateChatRequest(
      options=runner_pb2.ChatOptions(max_completion_tokens=32, top_k=1)
  )
  stub.CreateChat(chat)
  turns = []
  for n, prompt in enumerate(['Name three colors.', 'One more.'], 1):
    content, completion = runner_client.send(stub, prompt)
    assert content == completion.content
    fed = completion.model_debugger.fed_tokens
    if n == 1:
      assert fed[0].position == 0 and fed[0].id == loaded.start_token
    else:
      # The cache is kept: the turn feeds from the token held back.
      held = turns[-1][1].model_debugger.context_token_count - 1
      assert fed[0].position == held
    directory = tmp_path / f'turn-{n}'
    names = runner_client.read_captures(stub, n, directory)
    report = _report(info, loaded, completion, names)
    if _OBSERVE == 'anchors':
      observation = json.loads((directory / 'observation.json').read_text())
      report['observation'] = observation
      anchors = [name for name in names if '_post_anchor_' in name]
      modules = [name for name in names if '_post_module_' in name]
      report['anchor_files'] = len(anchors)
      report['module_files'] = len(modules)
      # The first turn on the CPU is captured fully, every other turn
      # and every one on the GPU module-only; each holds every pass's
      # anchors of every layer and the modules asked for.
      full = device == 'CPU' and n == 1
      assert observation['capture'] == ('full' if full else 'modules')
      assert observation['captured'], observation['reason']
      forwards = {
          (name.split('_post_anchor_')[0], name.rsplit('_step_')[1])
          for name in anchors
      }
      assert len(anchors) == len(forwards) * 35 * 4
      assert len(modules) == len(forwards) * len(_MODULES)
      # Only the full turn writes both signatures' execution graphs.
      graphs = observation['execution_graph']
      assert graphs == (['decode', 'prefill'] if full else [])
      assert not observation['execution_graph_reason']
      if full:
        assert 'execution_graph.json.gz' in names
      else:
        assert 'execution_graph.json.gz' not in names
    report['files'] = sorted(
        name
        for name in names
        if '_post_anchor_' not in name and '_post_module_' not in name
    )
    print(json.dumps(report, indent=1))
    turns.append((prompt, completion, directory))
  server_capture.check_turns(
      tmp_path / 'index',
      {'info': info, 'load': load, 'chat': chat, 'loaded': loaded},
      turns,
  )


def _report(info, loaded, completion, names) -> dict[str, Any]:
  """Returns what the run showed, for the log."""
  debugger = completion.model_debugger
  return {
      'runtime': info.runtime.version,
      'host': str(info.host),
      'loaded': str(loaded),
      'content': completion.content,
      'finish_reason': completion.finish_reason,
      'usage': str(completion.usage),
      'context_token_count': debugger.context_token_count,
      'fed_tokens': [
          [token.position, token.phase, token.id, token.text]
          for token in debugger.fed_tokens
      ],
      'files': names,
  }


if __name__ == '__main__':
  test_main.main()
