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
# ==============================================================================

"""Stdlib-only file and hashing helpers shared by every layer.

This module must not import anything else from the package: subprocess workers
in the external runtime venv import it without the metadata/SQLite stack.
"""

import datetime
import hashlib
import json
import os
import pathlib
import shutil
import tempfile
from typing import Any

PathLike = str | os.PathLike[str]


def atomic_json(path: PathLike, value: Any) -> None:
  target = pathlib.Path(path)
  target.parent.mkdir(parents=True, exist_ok=True)
  temporary = None
  try:
    with tempfile.NamedTemporaryFile(
        'w', dir=target.parent, delete=False
    ) as file:
      temporary = pathlib.Path(file.name)
      json.dump(value, file, ensure_ascii=False, allow_nan=False)
      file.flush()
      os.fsync(file.fileno())
    os.replace(temporary, target)
  finally:
    if temporary and temporary.exists():
      temporary.unlink()


def now() -> str:
  return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(value: Any) -> str:
  """Canonical JSON digest with default separators.

  Persisted mapping fingerprints depend on it.
  """
  return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def clone_file(source: PathLike, destination: PathLike) -> None:
  """Copy a file to a new path, as an APFS clone when both share a volume.

  A clone moves no bytes.
  """
  try:
    import ctypes

    clonefile = ctypes.CDLL(None, use_errno=True).clonefile
    if clonefile(os.fsencode(source), os.fsencode(destination), 0) == 0:
      return
  except (AttributeError, OSError):
    pass
  with (
      pathlib.Path(source).open('rb') as reader,
      pathlib.Path(destination).open('xb') as writer,
  ):
    shutil.copyfileobj(reader, writer, 1024 * 1024)


def file_digest(path: PathLike) -> str:
  with pathlib.Path(path).open('rb') as stream:
    checksum = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
      checksum.update(chunk)
    return checksum.hexdigest()


def proof_digest(proof: Any, *, ensure_ascii: bool = False) -> str:
  """Compact canonical JSON digest used for capture proofs.

  Defaults to `ensure_ascii=False` and rejects NaN/Infinity literals.
  """
  return hashlib.sha256(
      json.dumps(
          proof,
          sort_keys=True,
          separators=(',', ':'),
          ensure_ascii=ensure_ascii,
          allow_nan=False,
      ).encode()
  ).hexdigest()
