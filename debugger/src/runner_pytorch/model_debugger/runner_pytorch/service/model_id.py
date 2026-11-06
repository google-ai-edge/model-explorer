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

r"""The id of a model directory: the sha256 of its manifest.

The manifest has one line per file of the directory, sorted by path:

  <relative path>\t<size>\t<sha256 of the file>\n

The files are every path under the directory, at any depth, that is a regular
file or a symbolic link to one (a Hugging Face snapshot holds links into its
blobs), but none with a component starting with '.'. The path uses '/',
whatever the system; size and sha256 are of the file's bytes, the link's
target's for a link. The id is the sha256 of the manifest's UTF-8 bytes, in
lower-case hex. The server computes the same id to name the model in Load.

Hashing gigabytes takes seconds, so each file's sha256 is kept in a cache, a
JSON file keyed by the file's resolved path, size and modification time.
"""

import hashlib
import json
import os
import pathlib

_CHUNK_BYTES = 1 << 20


def manifest(directory: pathlib.Path, cache_path: pathlib.Path | None) -> str:
  """Returns the manifest of a model directory.

  Args:
      directory: The model directory.
      cache_path: The JSON file of known file hashes, read and rewritten; None
        for no cache.

  Raises:
      OSError: A file cannot be read.
      ValueError: The directory holds no file.
  """
  cache = _read_cache(cache_path)
  lines = []
  changed = False
  for relative in _files(directory):
    path = directory / relative
    stat = path.stat()
    key = f'{path.resolve()}\t{stat.st_size}\t{stat.st_mtime_ns}'
    digest = cache.get(key)
    if digest is None:
      digest = _sha256(path)
      cache[key] = digest
      changed = True
    lines.append(f'{relative}\t{stat.st_size}\t{digest}\n')
  if not lines:
    raise ValueError(f'{directory} holds no file.')
  if changed and cache_path is not None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f'{cache_path.name}.{os.getpid()}')
    temporary.write_text(json.dumps(cache, indent=1, sort_keys=True))
    os.replace(temporary, cache_path)
  return ''.join(lines)


def model_id(directory: pathlib.Path, cache_path: pathlib.Path | None) -> str:
  """Returns the id of a model directory: its manifest's sha256.

  Args:
      directory: The model directory.
      cache_path: As for `manifest`.
  """
  text = manifest(directory, cache_path)
  return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _files(directory: pathlib.Path) -> list[str]:
  """Returns the manifest's files of `directory`, relative, sorted."""
  found = []
  for root, dirs, names in os.walk(directory):
    dirs[:] = [name for name in dirs if not name.startswith('.')]
    for name in names:
      path = pathlib.Path(root) / name
      if name.startswith('.') or not path.is_file():
        continue
      found.append(path.relative_to(directory).as_posix())
  return sorted(found)


def _sha256(path: pathlib.Path) -> str:
  """Returns the sha256 of a file's bytes."""
  digest = hashlib.sha256()
  with open(path, 'rb') as file:
    while chunk := file.read(_CHUNK_BYTES):
      digest.update(chunk)
  return digest.hexdigest()


def _read_cache(cache_path: pathlib.Path | None) -> dict[str, str]:
  """Returns the known hashes; none if the cache is missing or unreadable."""
  if cache_path is None:
    return {}
  try:
    cache = json.loads(cache_path.read_text())
  except (OSError, ValueError):
    return {}
  return cache if isinstance(cache, dict) else {}
