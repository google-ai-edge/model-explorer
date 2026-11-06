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

"""The server's workspace: where models, records and captures live.

The server reads and writes model files here alone: those uploaded, in
`models/`, and those downloaded from Hugging Face, in `hf/`, a
huggingface_hub cache of its own (model_store, hf_models).
Also the copies of models that observe tensors, in `prepared/`, the
graphs of models, in `graphs/`, the tensor pairs made by hand, in
`mappings/`, and what the files of models imported without them said, in
`facts/`.

A Session's own files are its record, `records/<id>.json`, and its chats'
captures, `captures/<id>/chat-<n>/` (chat_directory()); a Session deleted
loses its record first, then its captures (remove_session_files()), and the
captures a crash left, no record naming them, go as the next server starts
(remove_orphan_session_files()).
The `bookmarks/` directory an older server kept goes as the workspace is
opened: bookmarks are no more.

One server owns a workspace at a time; an exclusive lock on `root/.lock`
keeps a second one out, since both would write the same records.
"""

from collections.abc import Collection
import fcntl
import os
import pathlib
import shutil
import sys
import uuid

# What the directory of a copy being written in `prepared`, or of a graph in
# `graphs`, starts with.
PENDING = '.pending-'


def chat_directory(n: int) -> str:
  """The directory of a Session's chat `n` in its captures, `chat-<n>`."""
  return f'chat-{n}'


class InUseError(RuntimeError):
  """Another server holds the workspace."""


class Workspace:
  """The directories of one workspace, held under an exclusive lock.

  Attributes:
      root: The workspace directory.
      models: Uploaded model files, named by their SHA-256.
      hf: The models downloaded from Hugging Face: huggingface_hub's cache
        layout (`models--<owner>--<name>/`), and its Xet cache in `xet/`.
      records: One JSON file per Session record.
      captures: One directory per Session holding its chats' captures, one
        directory each (chat_directory()).
      uploads: Uploads in progress; a finished upload moves to `models`, a
        Session imported is unpacked here first (session_archive).
      prepared: The copies of models that observe tensors, one directory per
        model and points (prepared_models); a copy being written is in a
        `.pending-` directory there until it is checked.
      graphs: The graphs of models, one directory per model, by its SHA-256
        (model_graphs); one being written is in a `.pending-` directory there.
      mappings: The tensor pairs made by hand, one JSON file per pair of models
        (mappings).
      facts: What the files of models that came with an imported Session said,
        one directory per model, by its SHA-256 (model_store): what the
        Session's debugging reads of a model whose file is not here.
      runners: What the server launched and has not stopped yet, one JSON file
        per runner (runner/launch_records): what a server killed left running,
        the next one stops.
  """

  def __init__(self, root: pathlib.Path) -> None:
    """Creates the directories and takes the workspace's lock.

    Args:
        root: The workspace directory; created if missing.

    Raises:
        InUseError: Another server holds the workspace.
    """
    self.root = pathlib.Path(root).expanduser().resolve()
    self.models = self.root / 'models'
    self.hf = self.root / 'hf'
    self.records = self.root / 'records'
    self.captures = self.root / 'captures'
    self.uploads = self.root / 'uploads'
    self.prepared = self.root / 'prepared'
    self.graphs = self.root / 'graphs'
    self.mappings = self.root / 'mappings'
    self.facts = self.root / 'facts'
    self.runners = self.root / 'runners'
    self.root.mkdir(parents=True, exist_ok=True)
    self._lock = open(self.root / '.lock', 'a')
    try:
      fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      self._lock.close()
      raise InUseError(
          f'Another Model Debugger server is using the workspace {self.root}.'
      ) from None
    for directory in (
        self.models,
        self.hf,
        self.records,
        self.captures,
        self.uploads,
        self.prepared,
        self.graphs,
        self.mappings,
        self.facts,
        self.runners,
    ):
      directory.mkdir(exist_ok=True)
    # An upload or copy the previous server did not finish is of no use to
    # anyone.
    for leftover in self.uploads.iterdir():
      _remove(leftover)
    for leftover in (
        *self.prepared.glob(PENDING + '*'),
        *self.graphs.glob(PENDING + '*'),
    ):
      shutil.rmtree(leftover, ignore_errors=True)
    # An older server's bookmarks, which nothing reads any more.
    _remove(self.root / 'bookmarks')

  def remove_session_files(self, record_id: str) -> None:
    """Removes a Session's captures, every chat's, its record gone.

    The captures are first renamed to a `.pending-` directory, so that it
    leaves `captures/` at once, then removed. A crash meanwhile leaves
    what remove_orphan_session_files() removes next.

    Args:
        record_id: The id of a record the store had, never a path.
    """
    capture = self.captures / record_id
    if not capture.exists():
      return
    gone = self.captures / (PENDING + uuid.uuid4().hex)
    capture.rename(gone)
    _remove(gone)

  def remove_orphan_session_files(self, record_ids: Collection[str]) -> None:
    """Removes every capture no record names.

    What a delete or an import the previous server did not finish left:
    a capture moved in before its record was written (session_archive),
    or one whose record was removed before it. Called as the server
    starts, before anything writes a Session's files.

    Args:
        record_ids: The ids of the records kept.
    """
    kept = set(record_ids)
    for entry in self.captures.iterdir():
      if entry.name not in kept:
        _remove(entry)

  def close(self) -> None:
    """Releases the lock; the workspace may then be opened again."""
    if not self._lock.closed:
      fcntl.flock(self._lock, fcntl.LOCK_UN)
      self._lock.close()


def _remove(path: pathlib.Path) -> None:
  """Removes a file, or a directory with all it holds."""
  if path.is_dir() and not path.is_symlink():
    shutil.rmtree(path, ignore_errors=True)
  else:
    path.unlink(missing_ok=True)


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
