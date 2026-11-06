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

"""A real turn: what the real runner captures of a real model, as it runs.

For the tests of the runner's real files. The turn is captured afresh, once
per test process and backend, by the real command-line runner on this
computer (RUNNER_BINARY, or the one Bazel built last) from the model
RUNNER_MODEL names; without both, AVAILABLE is False and the tests skip, as
cli/server_main_test.py does. Nothing of it is kept in the repository.

Usage:
  @absltest.skipUnless(real_capture.AVAILABLE, real_capture.UNAVAILABLE)
  def test_...(self):
    turn = real_capture.turn('CPU')
    ... turn.directory, turn.completion, turn.side ...
"""

import dataclasses
import functools
import hashlib
import os
import pathlib
import queue
import re
import subprocess
import tempfile
import threading
from typing import Any

from absl.testing import absltest
from model_debugger.protocol import runner_pb2
from model_debugger.server.capture import capture_index
from model_debugger.server.runner import launcher
from model_debugger.server.runner import runner_client
from model_debugger.server.testing import real_runner

# The model the turn runs, from RUNNER_MODEL.
MODEL = pathlib.Path(os.environ.get('RUNNER_MODEL', ''))
# Whether a real turn can be captured here.
AVAILABLE = real_runner.BUILT and MODEL.is_file()
# Why the tests that need one skip when it cannot.
UNAVAILABLE = 'needs the built runner and RUNNER_MODEL'
# The turn: one greedy reply, cut short.
PROMPT = 'What is 2 + 2?'
MAX_OUTPUT_TOKENS = 8
CONTEXT_LENGTH = 4096

# This computer, where the runner runs.
_DEVICE = launcher.Device(
    launcher.LOCAL_DEVICE_ID, 'This computer', 'macOS', 'local', True
)
# What the command-line runner prints once it listens.
_LISTENING = re.compile(r'runner: listening on port (\d+)')


@dataclasses.dataclass(frozen=True)
class Turn:
  """Turn 1 of a real chat, as the runner sent it.

  Attributes:
      directory: The turn's files, as `ReadCaptures` brought them.
      completion: `capture_index.completion_dict` of the turn's Completion.
      side: `capture_index.side_entry` of the chat's side.
  """

  directory: pathlib.Path
  completion: dict[str, Any]
  side: dict[str, Any]


@functools.cache
def turn(backend: str) -> Turn:
  """Returns turn 1 of a greedy chat on `backend`, captured once.

  The runner runs as a child of this process, as the local launcher runs
  it, until the turn is captured.

  Args:
      backend: 'CPU' or 'GPU'.

  Raises:
      runner_client.Error: The runner failed the turn.
  """
  process = subprocess.Popen(
      [str(real_runner.RUNNER), '--listen', '127.0.0.1:0'],
      stdin=subprocess.DEVNULL,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      errors='replace',
      close_fds=False,
  )
  ports = queue.Queue()
  reader = threading.Thread(target=_read, args=(process, ports), daemon=True)
  reader.start()
  try:
    try:
      port = ports.get(timeout=launcher.LAUNCH_SECONDS)
    except queue.Empty:
      port = None
    if port is None:
      raise RuntimeError('The runner said no port.')
    client = runner_client.RunnerClient.connect(
        _Runner(_DEVICE, f'127.0.0.1:{port}')
    )
    try:
      return _capture(client, backend)
    finally:
      client.close()
  finally:
    process.terminate()
    process.wait()
    reader.join()
    process.stdout.close()


@dataclasses.dataclass(frozen=True)
class _Runner:
  """What RunnerClient needs of a launched runner: where, and on what."""

  device: launcher.Device
  address: str


def _read(process: subprocess.Popen[Any], ports: queue.Queue[Any]) -> None:
  """Reads the runner's output to its end; puts its port, then None."""
  for line in process.stdout:
    if match := _LISTENING.fullmatch(line.rstrip('\n')):
      ports.put(int(match.group(1)))
  ports.put(None)


def _capture(client: runner_client.RunnerClient, backend: str) -> Turn:
  """Runs the turn on a connected runner and takes its capture."""
  sha256 = _sha256(MODEL)
  load = runner_pb2.LoadRequest(
      sha256=sha256,
      device=runner_pb2.Device.Value(backend),
      context_length=CONTEXT_LENGTH,
  )
  loaded = client.load(load, MODEL, threading.Lock(), lambda *_: None)
  chat = runner_pb2.CreateChatRequest()
  chat.options.top_k = 1
  chat.options.max_completion_tokens = MAX_OUTPUT_TOKENS
  client.create_chat(chat)
  completion = client.send({'role': 'user', 'content': PROMPT}, lambda _: None)
  directory = pathlib.Path(
      tempfile.mkdtemp(dir=absltest.TEST_TMPDIR.value), f'{backend}-turn-1'
  )
  directory.mkdir()
  client.take_capture(completion.model_debugger.turn, directory)
  side = capture_index.side_entry(
      model_sha256=sha256,
      model_name=MODEL.name,
      model_size=MODEL.stat().st_size,
      device_id=_DEVICE.id,
      device_name=_DEVICE.name,
      info=client.info,
      load=load,
      chat=chat,
      loaded=loaded,
  )
  return Turn(directory, capture_index.completion_dict(completion), side)


def _sha256(path: pathlib.Path) -> str:
  """Returns a file's SHA-256, in hex."""
  digest = hashlib.sha256()
  with path.open('rb') as file:
    while block := file.read(1 << 22):
      digest.update(block)
  return digest.hexdigest()
