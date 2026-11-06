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

"""Unit tests for streaming tensor capture writing."""

import errno
import hashlib
import pathlib
import shutil
import types
from unittest import mock

from absl.testing import absltest
from model_debugger.protocol import runner_pb2
from model_debugger.server.runner import capture_writer
from model_debugger.server.testing import test_main


def _file(path: str, data: bytes) -> list[runner_pb2.CaptureChunk]:
  """Returns the messages of one file: its header, data and SHA-256."""
  return [
      runner_pb2.CaptureChunk(
          file=runner_pb2.CaptureFile(path=path, size=len(data))
      ),
      runner_pb2.CaptureChunk(data=data[:3]),
      runner_pb2.CaptureChunk(data=data[3:]),
      runner_pb2.CaptureChunk(sha256=hashlib.sha256(data).hexdigest()),
  ]


def _free(*free: int):
  """Returns a shutil.disk_usage reporting each of `free` bytes free in turn.

  The last stays; every disk is alike.

  Args:
    *free: Byte counts of free space to report in turn.

  Returns:
    A disk usage reporting callable.
  """
  left = list(free)

  def disk_usage(path):
    del path  # Every disk alike.
    now = left.pop(0) if len(left) > 1 else left[0]
    return types.SimpleNamespace(total=2 * now, used=now, free=now)

  return disk_usage


class _FullDisk:
  """A file whose disk fills once `room` bytes are written to it."""

  def __init__(self, file, room: int) -> None:
    self._file = file
    self._room = room

  def write(self, data: bytes) -> int:
    if len(data) > self._room:
      self._file.write(data[: self._room])
      self._room = 0
      raise OSError(errno.ENOSPC, 'No space left on device')
    self._room -= len(data)
    return self._file.write(data)

  def close(self) -> None:
    self._file.close()


class CaptureWriterTest(absltest.TestCase):
  """CaptureWriter writes and checks a capture as it arrives."""

  def setUp(self):
    """A fresh directory for the capture."""
    super().setUp()
    self.directory = pathlib.Path(self.create_tempdir().full_path)

  def test_files_are_written_under_their_paths(self):
    """Each file lands at its path, parents made; no .part stays."""
    writer = capture_writer.CaptureWriter(self.directory)
    for chunk in _file('a.safetensors', b'abcdef') + _file('d/b', b'xy'):
      writer.write(chunk)
    writer.close()
    self.assertEqual((self.directory / 'a.safetensors').read_bytes(), b'abcdef')
    self.assertEqual((self.directory / 'd/b').read_bytes(), b'xy')
    self.assertFalse(list(self.directory.rglob('*.part')))

  def test_a_file_unlike_what_the_runner_said_is_refused(self):
    """A wrong SHA-256 fails, and discard() removes the .part file."""
    chunks = _file('a', b'abcdef')
    chunks[-1] = runner_pb2.CaptureChunk(sha256='0' * 64)
    writer = capture_writer.CaptureWriter(self.directory)
    for chunk in chunks[:-1]:
      writer.write(chunk)
    with self.assertRaisesRegex(capture_writer.Error, 'arrived as 6'):
      writer.write(chunks[-1])
    writer.discard()
    self.assertFalse(list(self.directory.iterdir()))

  def test_a_stream_ending_mid_file_is_refused(self):
    """close() fails while a file is open."""
    writer = capture_writer.CaptureWriter(self.directory)
    for chunk in _file('a', b'abcdef')[:2]:
      writer.write(chunk)
    with self.assertRaisesRegex(capture_writer.Error, 'middle of a'):
      writer.close()
    writer.discard()
    self.assertFalse(list(self.directory.iterdir()))

  def test_a_file_its_disk_has_no_room_for_is_not_begun(self):
    """A file and the spare room not free: refused, nothing written."""
    writer = capture_writer.CaptureWriter(
        self.directory / 'turn-1', self.directory
    )
    chunks = _file('a', b'abcdef')
    free = capture_writer.SPARE_BYTES + 5
    with mock.patch.object(shutil, 'disk_usage', _free(free)):
      with self.assertRaises(capture_writer.Error) as raised:
        writer.write(chunks[0])
    writer.discard()
    self.assertEqual(
        str(raised.exception),
        "Not enough disk space for this turn's capture: it needs at "
        'least 1 GB more, 1 GB of it kept spare; the disk holding '
        f'{self.directory} has 1 GB free.',
    )
    self.assertFalse(list(self.directory.rglob('*.part')))
    self.assertFalse(list(self.directory.rglob('a')))

  def test_a_file_with_room_and_spare_left_is_written(self):
    """Exactly the file and the spare room free is enough."""
    writer = capture_writer.CaptureWriter(self.directory)
    free = capture_writer.SPARE_BYTES + 6
    with mock.patch.object(shutil, 'disk_usage', _free(free)):
      for chunk in _file('a', b'abcdef'):
        writer.write(chunk)
    writer.close()
    self.assertEqual((self.directory / 'a').read_bytes(), b'abcdef')

  def test_a_disk_that_fills_mid_file_fails_for_want_of_room(self):
    """ENOSPC mid-file says so, and discard() removes the .part file."""
    writer = capture_writer.CaptureWriter(self.directory)
    chunks = _file('a', b'abcdef')
    opened = pathlib.Path.open

    def open_full(path, *args, **kwargs):
      return _FullDisk(opened(path, *args, **kwargs), room=4)

    with (
        mock.patch.object(pathlib.Path, 'open', open_full),
        mock.patch.object(
            shutil, 'disk_usage', _free(capture_writer.SPARE_BYTES + 6, 0)
        ),
    ):
      writer.write(chunks[0])
      writer.write(chunks[1])  # 3 bytes, which fit.
      with self.assertRaises(capture_writer.Error) as raised:
        writer.write(chunks[2])
    writer.discard()
    self.assertEqual(
        str(raised.exception),
        "Not enough disk space for this turn's capture: it needs at "
        'least 1 GB more, 1 GB of it kept spare; the disk holding '
        f'{self.directory} has 0 GB free.',
    )
    self.assertFalse(list(self.directory.iterdir()))

  def test_messages_out_of_order_are_refused(self):
    """Data outside a file, and a file begun inside one, fail."""
    writer = capture_writer.CaptureWriter(self.directory)
    with self.assertRaisesRegex(capture_writer.Error, 'outside a file'):
      writer.write(runner_pb2.CaptureChunk(data=b'x'))
    writer.write(_file('a', b'abcdef')[0])
    with self.assertRaisesRegex(capture_writer.Error, 'before ending a'):
      writer.write(_file('b', b'xy')[0])
    writer.discard()


if __name__ == '__main__':
  test_main.main()
