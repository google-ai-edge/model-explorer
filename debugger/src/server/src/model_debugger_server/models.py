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

"""The model files the server knows, by SHA-256.

A model is registered from a path on the server's machine (`--model`), kept
where it is, or uploaded from the browser into the workspace. Either way the
runner loads it by its SHA-256, so that is its identity; the UI calls it an
artifact, `model-<sha256[:24]>`.
"""

import asyncio
from collections.abc import AsyncIterator
import dataclasses
import hashlib
import os
import pathlib
import shutil
import threading
import uuid

from model_debugger_server import errors, workspace

UPLOAD_LIMIT_BYTES = 20 * 1024**3
# How long an upload may wait for its next bytes before it is dropped.
UPLOAD_IDLE_TIMEOUT_SECONDS = 60
# The only model format the runner loads.
MODEL_SUFFIX = '.litertlm'
# Disk an upload leaves free, beyond its own size.
_FREE_MARGIN_BYTES = 64 * 1024**2
# Uploads are written in pieces of at least this many bytes.
_WRITE_BYTES = 1024**2


@dataclasses.dataclass(frozen=True)
class Model:
  """One model file.

  Attributes:
    artifact: The id the UI names it by: 'model-' and its SHA-256's first 24 hex
      digits.
    sha256: SHA-256 of the whole file, lower-case hex.
    name: The file name to show.
    size: Bytes.
    path: Where the file is.
  """

  artifact: str
  sha256: str
  name: str
  size: int
  path: pathlib.Path


def _artifact(sha256: str) -> str:
  return 'model-' + sha256[:24]


def _hash_file(path: pathlib.Path) -> str:
  digest = hashlib.sha256()
  with open(path, 'rb') as file:
    while block := file.read(_WRITE_BYTES):
      digest.update(block)
  return digest.hexdigest()


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

  A registered model stays where it is; its record holds its path and the
  size and modification time it was hashed at, so that registering it again
  at the next start does not read the whole file again. An uploaded model is
  `models/<sha256>.litertlm`.
  """

  def __init__(self, workspace_: workspace.Workspace) -> None:
    """Loads the models recorded in the workspace.

    Args:
      workspace_: The workspace the models are kept in.
    """
    self._workspace = workspace_
    self._path = workspace_.root / 'models.json'
    self._lock = threading.Lock()
    self._records: dict[str, dict] = {}
    if self._path.exists():
      self._records = workspace.read_json(self._path)['models']

  def register(self, path: pathlib.Path) -> Model:
    """Registers a model file on this machine, by reference.

    Args:
      path: An existing `.litertlm` file.

    Returns:
      The model.

    Raises:
      ValueError: `path` is not an existing `.litertlm` file.
    """
    path = pathlib.Path(path).expanduser().resolve()
    if not path.is_file():
      raise ValueError(f'{path} is not a file.')
    _check_name(path.name)
    stat = path.stat()
    with self._lock:
      known = next(
          (
              record
              for record in self._records.values()
              if record['path'] == str(path)
              and record['size'] == stat.st_size
              and record.get('mtime_ns') == stat.st_mtime_ns
          ),
          None,
      )
    sha256 = known['sha256'] if known else _hash_file(path)
    record = {
        'sha256': sha256,
        'name': path.name,
        'size': stat.st_size,
        'path': str(path),
        'mtime_ns': stat.st_mtime_ns,
    }
    artifact = _artifact(sha256)
    with self._lock:
      # The file at this path may have changed since it was registered.
      for other in [
          a for a, r in self._records.items() if r['path'] == str(path)
      ]:
        del self._records[other]
      self._records[artifact] = record
      self._save()
    return self._model(artifact, record)

  async def upload(
      self, name: str, size: int, chunks: AsyncIterator[bytes]
  ) -> Model:
    """Stores an uploaded model in the workspace.

    The bytes are hashed while they are written. A file whose bytes the store
    already holds is not kept twice: the model already known is returned.

    Args:
      name: The file's name as the browser gave it; must end in `.litertlm`.
      size: The bytes announced; exactly these many must arrive.
      chunks: The bytes, in order.

    Returns:
      The model.

    Raises:
      ValueError: The name or size is refused, the disk is too full, or the
        bytes that arrived are not `size` bytes.
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
    artifact = _artifact(sha256)
    with self._lock:
      known = self._records.get(artifact)
      if known and pathlib.Path(known['path']).is_file():
        return self._model(artifact, known)
      path = self._workspace.models / f'{sha256}{MODEL_SUFFIX}'
      os.replace(staging, path)
      record = {'sha256': sha256, 'name': name, 'size': size, 'path': str(path)}
      self._records[artifact] = record
      self._save()
      return self._model(artifact, record)

  def get(self, artifact: str) -> Model:
    """Returns the model the UI names `artifact`.

    Raises:
      NotFound: No such model.
    """
    with self._lock:
      record = self._records.get(artifact)
    if record is None:
      raise errors.NotFound(f'There is no model {artifact}.')
    return self._model(artifact, record)

  def list(self) -> list[Model]:
    """Returns every model, in the order they were first recorded."""
    with self._lock:
      return [self._model(a, r) for a, r in self._records.items()]

  def _model(self, artifact: str, record: dict) -> Model:
    return Model(
        artifact=artifact,
        sha256=record['sha256'],
        name=record['name'],
        size=record['size'],
        path=pathlib.Path(record['path']),
    )

  def _save(self) -> None:
    workspace.write_json(self._path, {'models': self._records})
