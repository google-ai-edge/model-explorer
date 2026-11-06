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

"""A turn's capture written to disk as a runner's ReadCaptures stream comes.

Each file arrives as a file message, its data messages, then its SHA-256. It
is written beside its name as `<path>.part`, checked against the size and
SHA-256 the runner sent, then renamed.

A file is begun only while its disk has room for it and SPARE_BYTES more: the
protocol announces each file's size as it starts, not the capture's. A disk
that fills all the same, as other programs write to it, fails the capture with
the same reason.
"""

from collections.abc import Iterator
import contextlib
import errno
import hashlib
import pathlib
import shutil
from typing import BinaryIO

from model_debugger.protocol import runner_pb2

# What a capture leaves free on its disk, for everything else that writes to
# it.
SPARE_BYTES = 10**9

# The errors of a write its disk has no room for.
_NO_SPACE = (errno.ENOSPC, errno.EDQUOT)


class Error(Exception):
  """A stream that breaks the protocol's order or that cannot be written."""


class CaptureWriter:
  """Writes the files of one capture's stream under a directory.

  Feed it each message with write(), then call close(); after a failure,
  discard() removes the file left half written.
  """

  def __init__(
      self, directory: pathlib.Path, holder: pathlib.Path | None = None
  ) -> None:
    """Starts a capture.

    Args:
        directory: The turn's directory; files go under it.
        holder: The directory a disk without room is named by, such as the
          workspace `directory` is in; `directory` if None.
    """
    self._directory = directory
    self._holder = holder or directory
    self._path = ''  # The file being written; empty between files.
    self._size = 0
    self._received = 0
    self._digest = hashlib.sha256()
    self._part = pathlib.Path()
    self._out: BinaryIO | None = None

  def write(self, chunk: runner_pb2.CaptureChunk) -> None:
    """Writes the stream's next message.

    Args:
        chunk: The message.

    Raises:
        Error: The message is out of the protocol's order, a file is not
          as the runner said, or a file cannot be written.
    """
    part = chunk.WhichOneof('part')
    if part == 'file':
      if self._path:
        raise Error(
            f'The runner began {chunk.file.path} before ending {self._path}.'
        )
      self._begin(chunk.file)
    elif part is None:
      raise Error('The runner sent an empty CaptureChunk.')
    elif not self._path:
      raise Error(f'The runner sent {part} outside a file.')
    elif part == 'data':
      if self._received + len(chunk.data) > self._size:
        raise Error(
            f'{self._path} exceeded its announced size of {self._size} bytes.'
        )
      with self._writing():
        self._out.write(chunk.data)
      self._digest.update(chunk.data)
      self._received += len(chunk.data)
    else:
      self._end(chunk.sha256)

  def close(self) -> None:
    """Checks the stream ended between files.

    Raises:
        Error: It ended in the middle of a file.
    """
    if self._path:
      raise Error(f'The capture ended in the middle of {self._path}.')

  def discard(self) -> None:
    """Removes the .part file of a file that will not be finished."""
    if self._out is not None:
      with contextlib.suppress(OSError):
        self._out.close()
        self._part.unlink(missing_ok=True)
      self._out = None

  def _begin(self, announced: runner_pb2.CaptureFile) -> None:
    """Opens `<path>.part` for a file, making its parent directories."""
    self._path = announced.path
    self._size = announced.size
    self._received = 0
    self._digest = hashlib.sha256()
    destination = self._directory / announced.path
    self._part = destination.with_name(destination.name + '.part')
    with self._writing():
      destination.parent.mkdir(parents=True, exist_ok=True)
      free = shutil.disk_usage(destination.parent).free
      if announced.size + SPARE_BYTES > free:
        raise Error(self._no_space(announced.size, free))
      self._out = self._part.open('wb')

  def _end(self, sha256: str) -> None:
    """Checks a file against what the runner said, then names it."""
    with self._writing():
      self._out.close()
    digest = self._digest.hexdigest()
    if self._received != self._size or digest != sha256:
      raise Error(
          f'{self._path} arrived as {self._received} bytes hashing to '
          f'{digest}; the runner announced {self._size} bytes, {sha256}.'
      )
    with self._writing():
      self._part.replace(self._directory / self._path)
    self._out = None
    self._path = ''

  @contextlib.contextmanager
  def _writing(self) -> Iterator[None]:
    """Turns an OSError of writing the file into an Error."""
    try:
      yield
    except OSError as error:
      if error.errno not in _NO_SPACE:
        raise Error(f'Cannot write {self._path}: {error}') from None
      try:
        free = shutil.disk_usage(self._directory).free
      except OSError:
        free = 0
      needed = max(self._size - self._received, 0)
      raise Error(self._no_space(needed, free)) from None

  def _no_space(self, needed: int, free: int) -> str:
    """Why a capture with `needed` bytes still to write does not fit."""
    return (
        "Not enough disk space for this turn's capture: it needs at "
        f'least {_gb(needed + SPARE_BYTES)} GB more, {_gb(SPARE_BYTES)} '
        f'GB of it kept spare; the disk holding {self._holder} has '
        f'{_gb(free)} GB free.'
    )


def _gb(size: int) -> str:
  """A number of bytes in GB, as a message shows it."""
  return f'{size / 10**9:.2f}'.rstrip('0').rstrip('.')
