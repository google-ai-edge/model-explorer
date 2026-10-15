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

"""The server's workspace: where models, records and captures are kept.

One server owns a workspace at a time; an exclusive lock on `root/.lock`
keeps a second one out, since both would write the same records.
"""

import fcntl
import json
import os
import pathlib
import sys
from typing import Any
import uuid


class Workspace:
  """The directories of one workspace, held under an exclusive lock.

  Attributes:
    root: The workspace directory.
    models: Uploaded model files, named by their SHA-256.
    records: One JSON file per Session or Chat record.
    captures: One directory per Chat holding its capture.
    uploads: Uploads in progress; a finished upload moves to `models`.
  """

  def __init__(self, root: pathlib.Path) -> None:
    """Creates the directories and takes the workspace's lock.

    Args:
      root: The workspace directory; created if missing.

    Raises:
      RuntimeError: Another server holds the workspace.
    """
    self.root = pathlib.Path(root).expanduser().resolve()
    self.models = self.root / 'models'
    self.records = self.root / 'records'
    self.captures = self.root / 'captures'
    self.uploads = self.root / 'uploads'
    self.root.mkdir(parents=True, exist_ok=True)
    self._lock = open(self.root / '.lock', 'a')
    try:
      fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      self._lock.close()
      raise RuntimeError(
          f'Another Model Debugger server is using the workspace {self.root}.'
      ) from None
    for directory in (self.models, self.records, self.captures, self.uploads):
      directory.mkdir(exist_ok=True)
    # An upload the previous server did not finish is of no use to anyone.
    for leftover in self.uploads.iterdir():
      leftover.unlink()

  def close(self) -> None:
    """Releases the lock; the workspace may then be opened again."""
    if not self._lock.closed:
      fcntl.flock(self._lock, fcntl.LOCK_UN)
      self._lock.close()


def write_json(path: pathlib.Path, value: Any) -> None:
  """Writes `value` as JSON so that a reader sees the old file or the new one.

  Args:
    path: The file to replace.
    value: What to write; must be JSON serializable.
  """
  temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
  try:
    with open(temporary, 'w', encoding='utf-8') as file:
      json.dump(value, file, ensure_ascii=False, indent=1)
      file.flush()
      os.fsync(file.fileno())
    os.replace(temporary, path)
  except BaseException:
    temporary.unlink(missing_ok=True)
    raise
  directory = os.open(path.parent, os.O_RDONLY)
  try:
    os.fsync(directory)
  finally:
    os.close(directory)


def read_json(path: pathlib.Path) -> Any:
  """Returns the JSON value in `path`."""
  with open(path, encoding='utf-8') as file:
    return json.load(file)


def default_root() -> pathlib.Path:
  """Returns where the workspace lives when none is named.

  The platform's place for application data: Application Support on macOS,
  `$XDG_DATA_HOME` (default `~/.local/share`) elsewhere.
  """
  if sys.platform == 'darwin':
    base = pathlib.Path.home() / 'Library' / 'Application Support'
  else:
    base = pathlib.Path(
        os.environ.get('XDG_DATA_HOME') or pathlib.Path.home() / '.local/share'
    )
  return base / 'model_debugger' / 'server'
