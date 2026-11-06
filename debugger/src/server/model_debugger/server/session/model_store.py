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

"""The model files the server holds, by SHA-256: all in its workspace.

The browser and the server are on different computers, so a model comes to
the server one of two ways, and is kept in its workspace, never elsewhere:
a LiteRT-LM `.litertlm` file uploaded from the browser, kept as
`models/<sha256>.litertlm` (upload()); or a download from Hugging Face
(hf_models) into the workspace's own huggingface_hub cache, `hf/`, recorded
where the cache keeps it (add_download(), add_download_directory()), with
where it came from (`source`: repo, file, revision and the commit it
resolved to). A PyTorch model only comes from Hugging Face: a repository's
snapshot directory, which the PyTorch runner reads in place. One copy of a
model serves every Session that names it.
The runner loads a LiteRT-LM model by its SHA-256, so that is its identity.
A PyTorch model's identity is the SHA-256 of its manifest, one line
`<relative path> TAB <size> TAB <sha256 of the file>` per regular file,
sorted, as the PyTorch runner computes it to answer Load.
The UI calls either an artifact, `model-<sha256[:24]>`. What a runner
reports of the model at Load that is the model's own, whatever the device or
build, is kept with it: its sampling defaults.

A model lives as long as a Session names it (collect()): the last Session
of it deleted, it is deleted too, its file and what the workspace keeps of
it; and as the server starts, so is every model no Session names, an upload
or download no Session came to use, with any file of `models/`, `hf/` or
`facts/` no model holds.

A model may also come with a Session imported from another computer
(session_archive) without its file: what its file said that a Session's
debugging reads, kept in `facts/<sha256>/` of the workspace (Model.facts):
its model_facts description (DESCRIBED), a PyTorch model's config.json
(CONFIG), and its scans (tap_scans). Such a model has no path: it is never
loaded, and a Session cannot be created or started with it. What the
server reads of a model's files it reads from these when they are there
(Model.described(), Model.config()). An older server's record of a file
outside the workspace (a path given on its command line, or the shared
Hugging Face cache) is read as such a model too, without its path
(upgrade()).
"""

import asyncio
import collections
from collections.abc import AsyncIterator, Collection, Iterable, Iterator
import contextlib
import dataclasses
import hashlib
import os
import pathlib
import re
import shutil
import threading
from typing import Any
import uuid

from absl import logging
from google.protobuf import json_format
from model_debugger.protocol import runner_pb2
from model_debugger.server.session import workspace
from model_debugger.server.tap import model_facts
from model_debugger.server.util import errors
from model_debugger.server.util import json_file

UPLOAD_LIMIT_BYTES = 20 * 1024**3
# How long an upload may wait for its next bytes before it is dropped.
UPLOAD_IDLE_TIMEOUT_SECONDS = 60
# The only model format the LiteRT-LM runner loads.
MODEL_SUFFIX = '.litertlm'
# The runtimes a model is for, as the UI names them.
LITERT_LM = 'LiteRT-LM'
PYTORCH = 'PyTorch'
# Disk an upload leaves free, beyond its own size.
_FREE_MARGIN_BYTES = 64 * 1024**2
# Uploads are written in pieces of at least this many bytes.
_WRITE_BYTES = 1024**2
# A record's key for its Model's sampler_defaults.
_DEFAULTS = 'sampler_defaults'
# A record's key for its Model's source.
_SOURCE = 'source'
# A record's key for its Model's facts.
_FACTS = 'facts'
# The files of a model's facts (Model.facts): its model_facts description,
# and a PyTorch model's config.json.
DESCRIBED = 'described.json'
CONFIG = 'config.json'
_SHA256 = re.compile(r'[0-9a-f]{64}')
_NAME_LIMIT = 512


@dataclasses.dataclass(frozen=True)
class Model:
  """One model file.

  Attributes:
      artifact: The id the UI names it by: 'model-' and its SHA-256's first 24
        hex digits.
      sha256: SHA-256 of the whole file, lower-case hex.
      name: The file name to show.
      size: Bytes.
      path: Where the file is; a PyTorch model's directory. None for a model
        imported without its file: it is never loaded.
      sampler_defaults: The sampling the model sets, as the last runner to load
        it reported (LoadResponse.sampler_defaults, by its field names); {} if
        it reported none, None until a runner loaded it.
      runtime: The runtime it is for: LITERT_LM or PYTORCH.
      source: Where it was fetched from, if from Hugging Face: `{repo, file,
        revision, commit}` (`file` '' for a PyTorch model's whole repository);
        None for an upload.
      facts: The directory of what its file said, kept as a Session that came
        with it was imported (DESCRIBED, CONFIG, and its scans); None for a
        model whose file the server read itself.
  """

  artifact: str
  sha256: str
  name: str
  size: int
  path: pathlib.Path | None
  sampler_defaults: dict[str, Any] | None = None
  runtime: str = LITERT_LM
  source: dict[str, Any] | None = None
  facts: pathlib.Path | None = None

  def present(self) -> bool:
    """Whether the model is still where it was recorded."""
    if self.path is None:
      return False
    if self.runtime == PYTORCH:
      return self.path.is_dir()
    return self.path.is_file()

  def described(self) -> dict[str, Any]:
    """What its files say (model_facts.describe()): kept, or read once.

    Its facts' when it has them, else model_facts.described() of its file;
    a model with neither says nothing is known (every value None).

    Returns:
      Described model facts dict.
    """
    if self.facts is not None and (self.facts / DESCRIBED).is_file():
      return json_file.read_json(self.facts / DESCRIBED)
    if self.path is None:
      return {'vocabulary': None, 'kv_quantization': None}
    return model_facts.described(self.path, self.sha256)

  def pieces(self) -> tuple[bytes, ...] | None:
    """Its tokenizer's pieces in id order (model_facts.pieces()).

    Read once per file; None without its file, or when they cannot be
    read.

    Returns:
      Tokenizer pieces in id order, or None if unavailable.
    """
    if not self.present():
      return None
    return model_facts.described_pieces(self.path, self.sha256)

  def config(self) -> dict[str, Any]:
    """A PyTorch model's config.json, parsed: its facts' or its own.

    Returns:
      Parsed config dict.

    Raises:
      OSError: An OS error occurred while reading.
      ValueError: The file content cannot be parsed as JSON.
      FileNotFoundError: The config.json is not on this computer.
    """
    if self.facts is not None and (self.facts / CONFIG).is_file():
      return json_file.read_json(self.facts / CONFIG)
    if self.path is None:
      raise FileNotFoundError(
          f'The config.json of {self.name} is not on this computer.'
      )
    return json_file.read_json(self.path / CONFIG)


def _artifact(sha256: str) -> str:
  return 'model-' + sha256[:24]


def _hash_file(path: pathlib.Path) -> str:
  digest = hashlib.sha256()
  with open(path, 'rb') as file:
    while block := file.read(_WRITE_BYTES):
      digest.update(block)
  return digest.hexdigest()


def _within(path: pathlib.Path, directory: pathlib.Path) -> bool:
  """Whether `path` is `directory` or in it, by its name alone."""
  return pathlib.Path(os.path.normpath(path)).is_relative_to(directory)


def _inside(path: pathlib.Path, directory: pathlib.Path) -> bool:
  """Whether `path` is in `directory`, its directories' links resolved.

  `path` itself is not resolved: a link of a Hugging Face cache's snapshot
  is in `hf/`, wherever its blob is.

  Args:
    path: Path to test.
    directory: Destination directory.

  Returns:
    True if path is inside directory, False otherwise.
  """
  try:
    named = path.parent.resolve() / path.name
  except OSError:
    return False
  return named != directory.resolve() and named.is_relative_to(
      directory.resolve()
  )


# The directories of the workspace a record's path may be in: uploads, and
# downloads from Hugging Face.
_FILE_DIRECTORIES = ('models', 'hf')
# The directory a record's facts are in.
_FACTS_DIRECTORIES = ('facts',)


def _relative(
    path: str, root: pathlib.Path, directories: tuple[str, ...]
) -> str | None:
  """A record's path as it is kept: relative to the workspace's `root`.

  A path is kept relative, so that the workspace may move. An absolute one
  is an older server's: one in the workspace is made relative; so is one
  of a workspace moved since, whose part from one of `directories` on is
  here.

  Args:
      path: The path the record holds.
      root: The workspace's directory.
      directories: The directories of the workspace it may be in.

  Returns:
      The path from `root`, in one of `directories`; None if it is in none
      of them: a file outside the workspace.
  """
  named = pathlib.PurePath(path)
  if named.is_absolute():
    named = pathlib.PurePath(os.path.normpath(named))
    if named.is_relative_to(root):
      named = named.relative_to(root)
    else:
      parts = named.parts
      index = max(
          (i for i, part in enumerate(parts) if part in directories),
          default=None,
      )
      if index is None:
        return None
      named = pathlib.PurePath(*parts[index:])
      if not os.path.lexists(root / named):
        return None
  parts = named.parts
  if len(parts) < 2 or parts[0] not in directories or '..' in parts:
    return None
  return named.as_posix()


def upgrade(records: dict[str, Any], root: pathlib.Path) -> dict[str, Any]:
  """Returns the model records of an older server as this one keeps them.

  Paths are kept relative to the workspace (_relative()): an older
  server's absolute ones are made so, of a workspace moved since too. An
  older server also recorded models by their paths elsewhere on its
  computer. Such a record loses its path: it is a model without its file,
  as one imported is (Model.facts), which a Session that names it reads
  what is kept of, and cannot start with; one no Session names is deleted
  as the server starts (ModelStore.collect()). Facts outside the
  workspace's `facts/` are dropped alike.

  Args:
      records: The records, by artifact, as `models.json` holds them.
      root: The workspace's directory.

  Returns:
      The records; `records` itself if none changes.
  """
  upgraded = {}
  for artifact, record in records.items():
    record = dict(record)
    if record.get('path'):
      path = _relative(record['path'], root, _FILE_DIRECTORIES)
      if path is None:
        record.pop('path')
        record.pop('mtime_ns', None)
      else:
        record['path'] = path
    if _FACTS in record:
      facts = _relative(record[_FACTS], root, _FACTS_DIRECTORIES)
      if facts is None:
        record.pop(_FACTS)
      else:
        record[_FACTS] = facts
    upgraded[artifact] = record
  return records if upgraded == records else upgraded


def _remove(path: pathlib.Path) -> None:
  """Removes a file or link, or a directory with all it holds."""
  if path.is_dir() and not path.is_symlink():
    shutil.rmtree(path, ignore_errors=True)
  else:
    path.unlink(missing_ok=True)


def _files(directory: pathlib.Path) -> list[pathlib.Path]:
  """The files and links in a directory, at any depth; none if missing."""
  if not directory.is_dir():
    return []
  return [
      path
      for path in directory.rglob('*')
      if path.is_symlink() or path.is_file()
  ]


def _collect_repository(
    repository: pathlib.Path, kept: Iterable[pathlib.Path] | None
) -> None:
  """Removes what a repository of a huggingface_hub cache no model holds.

  A cache keeps each file once, in `blobs/`, and links to it from each
  snapshot that has it, `snapshots/<commit>/<path>`. The blobs no link
  names go, then the empty directories, and the repository with them once
  no snapshot holds a file.

  Args:
      repository: Its directory, `models--<owner>--<name>`.
      kept: The paths of the models kept: every other link, and every incomplete
        blob of a download that stopped, goes too. None to remove no link, and
        no incomplete blob, as while a download may be writing one.
  """
  snapshots = repository / 'snapshots'
  if kept is not None:
    kept = list(kept)
    for path in _files(snapshots):
      if not any(_within(path, model) for model in kept):
        path.unlink(missing_ok=True)
  named = {path.resolve() for path in _files(snapshots)}
  blobs = repository / 'blobs'
  writing = False
  if blobs.is_dir():
    for blob in blobs.iterdir():
      if blob.name.endswith('.incomplete') and kept is None:
        writing = True
      elif blob.resolve() not in named:
        _remove(blob)
  if not named and not writing:
    shutil.rmtree(repository, ignore_errors=True)
    return
  if not snapshots.is_dir():
    return
  for directory in sorted(
      (path for path in snapshots.rglob('*') if path.is_dir()),
      key=lambda path: len(path.parts),
      reverse=True,
  ):
    if not any(directory.iterdir()):
      directory.rmdir()


def _directory_name(path: pathlib.Path) -> str:
  """What a model directory is shown as: its repository, if a cache's.

  Hugging Face's cache keeps a snapshot at
  `models--<owner>--<name>/snapshots/<revision>`; that is `<owner>/<name>`.

  Args:
    path: The model directory path.

  Returns:
    The display name of the model directory.
  """
  for part in reversed(path.parts):
    if part.startswith('models--'):
      return '/'.join(part.split('--')[1:])
  return path.name


def _check_name(name: str) -> None:
  if (
      not isinstance(name, str)
      or not name
      or name in ('.', '..')
      or pathlib.PurePath(name).name != name
      or '\\' in name
      or any(ord(char) < 32 for char in name)
  ):
    raise ValueError('The file name is not valid.')
  if not name.endswith(MODEL_SUFFIX) or name == MODEL_SUFFIX:
    raise ValueError(f'Choose a {MODEL_SUFFIX} model.')


class ModelStore:
  """The models of a workspace, kept in `root/models.json`.

  An uploaded model is `models/<sha256>.litertlm`. A model downloaded
  stays where the workspace's Hugging Face cache put it; its record holds
  its path and the size and modification time it was hashed at, so that
  the same download recorded again does not read the whole file again.
  """

  def __init__(self, workspace_: workspace.Workspace) -> None:
    """Loads the models recorded in the workspace.

    Args:
        workspace_: The workspace the models are kept in.
    """
    self._workspace = workspace_
    self._path = workspace_.root / 'models.json'
    self._lock = threading.Lock()
    self._records: dict[str, dict[str, Any]] = {}
    # The repositories of `hf/` downloads write in now, by name.
    self._downloading: collections.Counter[str] = collections.Counter()
    if self._path.exists():
      records = json_file.read_json(self._path)['models']
      self._records = upgrade(records, workspace_.root)
      if self._records is not records:
        self._save()

  def add_download(
      self, path: pathlib.Path, source: dict[str, Any] | None = None
  ) -> Model:
    """Records a `.litertlm` file downloaded into the workspace's `hf/`.

    Args:
        path: The file, in `hf/`.
        source: Where it was fetched from, as Model.source says.

    The file is kept by its own name: a link, such as one of a Hugging
    Face cache's snapshots to the blob it holds, is not followed to
    that blob's.

    Returns:
        The model.

    Raises:
        ValueError: `path` is not a `.litertlm` file in `hf/`.
    """
    path = self._downloaded(path)
    if not path.is_file():
      raise ValueError(f'{path} is not a file.')
    return self._register(
        path, path.stat(), path.name, LITERT_LM, source=source
    )

  def add_download_directory(
      self, path: pathlib.Path, source: dict[str, Any] | None = None
  ) -> Model:
    """Records a Hugging Face model directory downloaded into `hf/`.

    It is for the PyTorch runner, which reads it in place. Each file is
    hashed once: the workspace keeps its hash by its path, size and
    modification time (`file_hashes.json`).

    Args:
        path: The directory, in `hf/`, with a config.json.
        source: Where it was fetched from, as Model.source says.

    Returns:
        The model.

    Raises:
        ValueError: `path` is not such a directory.
    """
    path = self._downloaded(path)
    if not (path / 'config.json').is_file():
      raise ValueError(
          f'{path} is not a Hugging Face model directory (no config.json).'
      )
    sha256, size, mtime_ns = self._manifest(path)
    return self._register(
        path,
        (size, mtime_ns),
        _directory_name(path),
        PYTORCH,
        sha256,
        source,
    )

  def _downloaded(self, path: pathlib.Path) -> pathlib.Path:
    """`path`, its directory's links resolved, once it is in `hf/`.

    Args:
      path: Path to resolve.

    Returns:
      Resolved path within hf/.

    Raises:
      ValueError: It is not in the workspace's `hf/`.
    """
    path = pathlib.Path(path).absolute()
    path = path.parent.resolve() / path.name
    if not _within(path, self._workspace.hf):
      raise ValueError(
          f'{path} is not a download of the workspace: a model is '
          'uploaded or downloaded from Hugging Face.'
      )
    return path

  def _manifest(self, root: pathlib.Path) -> tuple[str, int, int]:
    """The SHA-256 of a directory's manifest, its size and latest mtime."""
    hashes_path = self._workspace.root / 'file_hashes.json'
    with self._lock:
      hashes = (
          json_file.read_json(hashes_path)['files']
          if hashes_path.exists()
          else {}
      )
    # As the PyTorch runner names the directory (its model_id.py): every
    # regular file, or link to one, none under a name starting with '.'.
    files = sorted(
        (file.relative_to(root).as_posix(), file)
        for file in root.rglob('*')
        if file.is_file()
        and not any(
            part.startswith('.') for part in file.relative_to(root).parts
        )
    )
    lines = []
    size = 0
    mtime_ns = 0
    changed = False
    for relative, file in files:
      stat = file.stat()
      key = str(file.resolve())
      known = hashes.get(key)
      if not (
          known
          and known['size'] == stat.st_size
          and known['mtime_ns'] == stat.st_mtime_ns
      ):
        known = {
            'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns,
            'sha256': _hash_file(file),
        }
        hashes[key] = known
        changed = True
      lines.append(f'{relative}\t{stat.st_size}\t{known["sha256"]}\n')
      size += stat.st_size
      mtime_ns = max(mtime_ns, stat.st_mtime_ns)
    if changed:
      with self._lock:
        json_file.write_json(hashes_path, {'files': hashes})
    manifest = ''.join(lines).encode()
    return hashlib.sha256(manifest).hexdigest(), size, mtime_ns

  def _register(
      self,
      path: pathlib.Path,
      stat: os.stat_result | tuple[int, int],
      name: str,
      runtime: str,
      sha256: str | None = None,
      source: dict[str, Any] | None = None,
  ) -> Model:
    """Records a model downloaded; hashes a file not known.

    A download recorded again keeps the source it had.

    Args:
      path: Path to the model file or directory.
      stat: File stat result or (size, mtime_ns) tuple.
      name: Model name.
      runtime: Runtime type string.
      sha256: Optional SHA-256 digest string.
      source: Optional source metadata dict.

    Returns:
      The registered Model instance.
    """
    if runtime == LITERT_LM:
      _check_name(path.name)
      size, mtime_ns = stat.st_size, stat.st_mtime_ns
    else:
      size, mtime_ns = stat
    kept = path.relative_to(self._workspace.root).as_posix()
    with self._lock:
      known = next(
          (
              record
              for record in self._records.values()
              if record.get('path') == kept
              and record['size'] == size
              and record.get('mtime_ns') == mtime_ns
          ),
          None,
      )
    if sha256 is None:
      sha256 = known['sha256'] if known else _hash_file(path)
    record = {
        'sha256': sha256,
        'name': name,
        'size': size,
        'path': kept,
        'mtime_ns': mtime_ns,
    }
    if runtime != LITERT_LM:
      record['runtime'] = runtime
    artifact = _artifact(sha256)
    with self._lock:
      # What runners reported is the bytes', which may be known already.
      previous = self._records.get(artifact, {})
      if _DEFAULTS in previous:
        record[_DEFAULTS] = previous[_DEFAULTS]
      if source:
        record[_SOURCE] = dict(source)
      elif _SOURCE in previous:
        record[_SOURCE] = previous[_SOURCE]
      if _FACTS in previous:
        record[_FACTS] = previous[_FACTS]
      # The file at this path may have changed since it was registered.
      for other in [
          a for a, r in self._records.items() if r.get('path') == kept
      ]:
        del self._records[other]
      self._records[artifact] = record
      self._save()
    return self._model(artifact, record)

  async def upload(
      self, name: str, size: int, chunks: AsyncIterator[bytes]
  ) -> Model:
    """Stores an uploaded model in the workspace.

    The bytes are hashed while they are written. A file whose bytes the
    store already holds is not kept twice: the model already known is
    returned.

    Args:
        name: The file's name as the browser gave it; must end in `.litertlm`.
        size: The bytes announced; exactly these many must arrive.
        chunks: The bytes, in order.

    Returns:
        The model.

    Raises:
        ValueError: The name or size is refused, the disk is too full, or
          the bytes that arrived are not `size` bytes.
    """
    _check_name(name)
    if not 0 < size <= UPLOAD_LIMIT_BYTES:
      raise ValueError(
          'Choose a non-empty file of at most '
          f'{UPLOAD_LIMIT_BYTES // 1024**3} GiB.'
      )
    if shutil.disk_usage(self._workspace.root).free < size + _FREE_MARGIN_BYTES:
      raise ValueError('There is not enough disk space for this model.')
    staging = self._workspace.uploads / f'{uuid.uuid4().hex}{MODEL_SUFFIX}'
    digest = hashlib.sha256()
    received = 0
    try:
      with open(staging, 'xb') as file:
        pending = bytearray()

        def write(data: bytes) -> None:
          digest.update(data)
          file.write(data)

        async for chunk in chunks:
          received += len(chunk)
          if received > size:
            raise ValueError(f'The upload sent more than {size} bytes.')
          pending += chunk
          if len(pending) >= _WRITE_BYTES:
            await asyncio.to_thread(write, bytes(pending))
            pending.clear()
        if received != size:
          raise ValueError(
              f'The upload ended after {received} of {size} bytes.'
          )
        await asyncio.to_thread(write, bytes(pending))
        await asyncio.to_thread(file.flush)
        await asyncio.to_thread(os.fsync, file.fileno())
      return await asyncio.to_thread(
          self._keep_upload, staging, name, size, digest.hexdigest()
      )
    finally:
      staging.unlink(missing_ok=True)

  def _keep_upload(
      self, staging: pathlib.Path, name: str, size: int, sha256: str
  ) -> Model:
    """Moves a staged upload into the model store under its SHA-256.

    Args:
      staging: Path to the temporary staging file.
      name: Model name.
      size: File size in bytes.
      sha256: Hex-encoded SHA-256 digest of the file.

    Returns:
      The registered Model instance.
    """
    artifact = _artifact(sha256)
    with self._lock:
      known = self._records.get(artifact)
      if known and self._model(artifact, known).present():
        return self._model(artifact, known)
      path = self._workspace.models / f'{sha256}{MODEL_SUFFIX}'
      os.replace(staging, path)
      record = {
          'sha256': sha256,
          'name': name,
          'size': size,
          'path': path.relative_to(self._workspace.root).as_posix(),
      }
      if known and _FACTS in known:
        record[_FACTS] = known[_FACTS]
      self._records[artifact] = record
      self._save()
      return self._model(artifact, record)

  def adopt(self, metadata: Any, facts: pathlib.Path) -> Model:
    """Registers a model that came with an imported Session, without file.

    A model already known keeps its record, file and all; it is given the
    facts unless it has some.

    Args:
        metadata: What the export said of it: `{sha256, name, size, runtime,
          source, sampler_defaults}` (`source` and `sampler_defaults` may be
          None).
        facts: The directory of what its file said, in the workspace's `facts/`
          (Model.facts).

    Returns:
        The model.

    Raises:
        ValueError: The metadata is not valid, or `facts` is not in
          `facts/`.
    """
    if not isinstance(metadata, dict):
      raise ValueError('A model of the Session is not described.')
    kept_facts = _relative(str(facts), self._workspace.root, _FACTS_DIRECTORIES)
    if kept_facts is None or not _inside(facts, self._workspace.facts):
      raise ValueError(f'{facts} is not in the facts of the workspace.')
    sha256 = metadata.get('sha256')
    name = metadata.get('name')
    size = metadata.get('size')
    runtime = metadata.get('runtime')
    source = metadata.get('source')
    defaults = metadata.get('sampler_defaults')
    if (
        not isinstance(sha256, str)
        or not _SHA256.fullmatch(sha256)
        or not isinstance(name, str)
        or not name
        or len(name) > _NAME_LIMIT
        or any(ord(char) < 32 for char in name)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or runtime not in (LITERT_LM, PYTORCH)
        or not isinstance(source, dict | None)
        or not isinstance(defaults, dict | None)
    ):
      raise ValueError(f'The model {name!r} is not described validly.')
    artifact = _artifact(sha256)
    with self._lock:
      known = self._records.get(artifact)
      if known is not None:
        if _FACTS not in known:
          known[_FACTS] = kept_facts
          self._save()
        return self._model(artifact, known)
      record = {
          'sha256': sha256,
          'name': name,
          'size': size,
          _FACTS: kept_facts,
      }
      if runtime != LITERT_LM:
        record['runtime'] = runtime
      if source:
        record[_SOURCE] = dict(source)
      if defaults is not None:
        record[_DEFAULTS] = dict(defaults)
      self._records[artifact] = record
      self._save()
      return self._model(artifact, record)

  def collect(
      self, used: Collection[str], among: Collection[str] | None = None
  ) -> list[Model]:
    """Deletes the models no Session names, their files and facts too.

    Args:
        used: The artifacts the Sessions name.
        among: The models to delete if unused: those of a Session just deleted.
          None for every model, as the server starts: then every file of
          `models/`, `hf/` and `facts/` no model kept holds goes too, what an
          upload or a download that stopped, or a deletion a crash cut short,
          left.

    Returns:
        The models deleted. What else the server keeps of a model, by
        its SHA-256 (its graph), is the caller's to delete.
    """
    with self._lock:
      gone = [
          artifact
          for artifact in self._records
          if artifact not in used and (among is None or artifact in among)
      ]
      removed = [
          self._model(artifact, self._records.pop(artifact))
          for artifact in gone
      ]
      if removed:
        self._save()
      kept = [
          self._model(artifact, record)
          for artifact, record in self._records.items()
      ]
    paths = [m.path for m in kept if m.path is not None]
    facts = {m.facts for m in kept if m.facts is not None}
    hf = self._workspace.hf
    for model in removed:
      logging.info(
          'Deleting the model %s (%s): no Session uses it',
          model.name,
          model.artifact,
      )
      if model.path is not None and (
          _inside(model.path, self._workspace.models) or _inside(model.path, hf)
      ):
        # Of a snapshot, only the files no model kept holds: one
        # commit may hold a LiteRT-LM file beside a PyTorch model.
        files = (
            _files(model.path)
            if model.path.is_dir() and not model.path.is_symlink()
            else [model.path]
        )
        for path in files:
          if not any(_within(path, other) for other in paths):
            path.unlink(missing_ok=True)
      if (
          model.facts is not None
          and model.facts not in facts
          and _inside(model.facts, self._workspace.facts)
      ):
        _remove(model.facts)
    if among is None:
      self._sweep(paths, facts)
    else:
      with self._lock:
        downloading = set(self._downloading)
      for model in removed:
        if model.path is None or not _within(model.path, hf):
          continue
        relative = model.path.relative_to(hf).parts
        # A repository a download writes in is left as it is: what
        # it no longer needs goes as the server next starts.
        if (
            relative
            and relative[0].startswith('models--')
            and relative[0] not in downloading
        ):
          _collect_repository(hf / relative[0], None)
    if removed or among is None:
      self._forget_hashes()
    return removed

  @contextlib.contextmanager
  def downloading(self, repository: str) -> Iterator[None]:
    """Marks a repository of `hf/` as written in, while in the context.

    collect() then leaves the repository's other files as they are.

    Args:
        repository: Its directory's name, `models--<owner>--<name>`.
    """
    with self._lock:
      self._downloading[repository] += 1
    try:
      yield
    finally:
      with self._lock:
        self._downloading[repository] -= 1
        if not self._downloading[repository]:
          del self._downloading[repository]

  def _sweep(self, paths: list[pathlib.Path], facts: set[pathlib.Path]) -> None:
    """Removes every model file and facts no model kept holds.

    Called as the server starts, when no upload or download is under
    way: the Xet cache and huggingface_hub's locks in `hf/` go too.

    Args:
        paths: The paths of the models kept.
        facts: Their facts.
    """
    for path in self._workspace.models.iterdir():
      if path not in paths:
        _remove(path)
    for path in self._workspace.facts.iterdir():
      if path not in facts:
        _remove(path)
    for path in self._workspace.hf.iterdir():
      if path.name.startswith('models--') and path.is_dir():
        _collect_repository(path, paths)
      elif not any(_within(model, path) for model in paths):
        _remove(path)

  def _forget_hashes(self) -> None:
    """Drops the kept hashes of the files no longer there."""
    hashes_path = self._workspace.root / 'file_hashes.json'
    with self._lock:
      if not hashes_path.exists():
        return
      hashes = json_file.read_json(hashes_path)['files']
      there = {
          key: value
          for key, value in hashes.items()
          if pathlib.Path(key).is_file()
          and _within(pathlib.Path(key), self._workspace.root)
      }
      if there != hashes:
        json_file.write_json(hashes_path, {'files': there})

  def get(self, artifact: str) -> Model:
    """Returns the model the UI names `artifact`.

    Args:
      artifact: The model artifact identifier.

    Raises:
      NotFound: No such model.
    """
    with self._lock:
      record = self._records.get(artifact)
    if record is None:
      raise errors.NotFound(f'There is no model {artifact}.')
    return self._model(artifact, record)

  def report_loaded(
      self, artifact: str, loaded: runner_pb2.LoadResponse
  ) -> None:
    """Keeps what a runner's Load reported of the model's own.

    Args:
      artifact: The model loaded.
      loaded: What the runner answered Load with.
    """
    defaults = json_format.MessageToDict(
        loaded.sampler_defaults, preserving_proto_field_name=True
    )
    with self._lock:
      record = self._records.get(artifact)
      if record is None or record.get(_DEFAULTS) == defaults:
        return
      record[_DEFAULTS] = defaults
      self._save()

  def pieces(self, table_sha256: str) -> tuple[bytes, ...] | None:
    """The tokenizer pieces of a vocabulary, by its table's digest.

    Of the first model recorded whose file is here and whose vocabulary
    (Model.described()) has the digest: one table, one id -> piece
    mapping, whichever model file holds it (a copy that observes tensors
    is no model of the store). None when no model has it, or its
    tokenizer cannot be read.

    Args:
      table_sha256: The vocabulary's `table_sha256` (model_facts).

    Returns:
      Tokenizer pieces tuple, or None if unavailable.
    """
    for model in self.list():
      if not model.present():
        continue
      vocabulary = model.described().get('vocabulary') or {}
      if vocabulary.get('table_sha256') == table_sha256:
        return model.pieces()
    return None

  def list(self) -> list[Model]:
    """Returns every model, in the order they were first recorded."""
    with self._lock:
      return [self._model(a, r) for a, r in self._records.items()]

  def _model(self, artifact: str, record: dict[str, Any]) -> Model:
    return Model(
        artifact=artifact,
        sha256=record['sha256'],
        name=record['name'],
        size=record['size'],
        path=(
            self._workspace.root / record['path']
            if record.get('path')
            else None
        ),
        sampler_defaults=(
            dict(record[_DEFAULTS]) if _DEFAULTS in record else None
        ),
        runtime=record.get('runtime', LITERT_LM),
        source=dict(record[_SOURCE]) if _SOURCE in record else None,
        facts=(
            self._workspace.root / record[_FACTS] if _FACTS in record else None
        ),
    )

  def _save(self) -> None:
    json_file.write_json(self._path, {'models': self._records})
