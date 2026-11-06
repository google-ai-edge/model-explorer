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

"""The server's capture index, to check a turn's capture as the server will.

The index is src/server's capture_index, imported from the source tree. Of
the server's tap module it needs only `added_point`, whose real module needs
a schema Bazel generates; a stand-in that knows no added point, as for a
model nobody tapped, takes its place.
"""

import pathlib
import shutil
import sys
import types
from typing import Any

# pylint: disable=line-too-long
_SERVER = pathlib.Path(__file__).resolve().parents[4] / 'server'
# pylint: enable=line-too-long


# pylint: disable=g-import-not-at-top
def _import_capture_index() -> types.ModuleType:
  """Returns the server's capture_index module."""
  import model_debugger

  server_package = _SERVER / 'model_debugger'
  if str(server_package) not in model_debugger.__path__:
    model_debugger.__path__.append(str(server_package))
  if 'model_debugger.server.tap.tap_model' not in sys.modules:
    tap = types.ModuleType('model_debugger.server.tap.tap_model')
    tap.added_point = lambda key: None
    sys.modules[tap.__name__] = tap
    import model_debugger.server.tap as tap_package

    tap_package.tap_model = tap
  from model_debugger.server.capture import capture_index

  return capture_index


# pylint: enable=g-import-not-at-top


def check_turns(
    root: pathlib.Path,
    sides_info: dict[str, Any],
    turns: list[tuple[str, object, pathlib.Path]],
) -> list[dict[str, Any]]:
  """Indexes turns as the server does; returns their derived entries.

  Args:
    root: An empty directory for the index.
    sides_info: The keyword arguments of capture_index.side_entry, but the
      model's sha256, name and size and the device's id and name.
    turns: Per turn, its prompt, its Completion and its capture's directory.

  Returns:
    The list of derived turn entries in the index.

  Raises:
    capture_index.Rejected: The index rejected a turn.
  """
  capture_index = _import_capture_index()
  side = capture_index.side_entry(
      model_sha256='0' * 64,
      model_name='model',
      model_size=0,
      device_id='local',
      device_name='This computer',
      **sides_info,
  )
  index = capture_index.CaptureIndex.create(root, {'ref': side})
  for n, (prompt, completion, directory) in enumerate(turns, 1):
    target = index.turn_directory('ref', n)
    for path in directory.iterdir():
      shutil.copy(path, target / path.name)
    index.add_turn(
        n, prompt, {'ref': capture_index.completion_dict(completion)}
    )
  return index.turns
