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

"""A client of the runner for tests: a turn's reply, a turn's capture."""

import hashlib
import pathlib

from model_debugger.protocol import runner_pb2
from model_debugger.protocol import runner_pb2_grpc


def send(
    stub: runner_pb2_grpc.RunnerStub, content: str
) -> tuple[str, runner_pb2.Completion]:
  """Sends a user message; returns the chunks' content and the completion."""
  chunks, completion = [], None
  for response in stub.Send(
      runner_pb2.SendRequest(
          message=runner_pb2.ChatMessage(
              role=runner_pb2.ChatMessage.USER, content=content
          )
      )
  ):
    if response.HasField('chunk'):
      chunks.append(response.chunk.content)
    else:
      completion = response.completion
  return ''.join(chunks), completion


def read_captures(
    stub: runner_pb2_grpc.RunnerStub, turn: int, directory: pathlib.Path
) -> list[str]:
  """Writes a turn's capture into `directory`; returns its files' names.

  Each file's sha256 is checked against the one the stream ends it with.

  Args:
    stub: The runner gRPC stub.
    turn: The turn index to read captures for.
    directory: Directory to write captured files into.

  Returns:
    List of relative paths of the written files.
  """
  directory.mkdir(parents=True)
  names, file, digest = [], None, None
  for chunk in stub.ReadCaptures(runner_pb2.ReadCapturesRequest(turn=turn)):
    part = chunk.WhichOneof('part')
    if part == 'file':
      names.append(chunk.file.path)
      file = open(directory / chunk.file.path, 'wb')
      digest = hashlib.sha256()
    elif part == 'data':
      file.write(chunk.data)
      digest.update(chunk.data)
    else:
      file.close()
      if chunk.sha256 != digest.hexdigest():
        raise ValueError(f'{names[-1]} does not hash to its sha256.')
  return names
