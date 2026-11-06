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

"""Tests of LaunchRecords, what a server launched and has not stopped."""

import pathlib
import threading
import time
from typing import Any

from absl.testing import absltest
from model_debugger.server.runner import launch_records
from model_debugger.server.testing import test_main
from model_debugger.server.util import json_file


class LaunchRecordsTest(absltest.TestCase):
  """Tests of LaunchRecords."""

  def setUp(self):
    """Gives each test a directory of records of its own."""
    super().setUp()
    self.directory = pathlib.Path(self.create_tempdir().full_path) / 'r'

  def _files(self) -> list[dict[str, Any]]:
    return [
        json_file.read_json(path)
        for path in sorted(self.directory.glob('*.json'))
    ]

  def test_a_record_is_a_file_until_stopped(self):
    """Each Record is a file, updated as it learns, gone once stopped."""
    records = launch_records.LaunchRecords()
    self.assertEqual(records.keep_in(self.directory), [])
    stopped = []
    record = records.add({'kind': 'k', 'a': 1}, lambda: stopped.append(1))
    self.assertEqual(self._files(), [{'kind': 'k', 'a': 1}])
    record.update(b=2)
    self.assertEqual(self._files(), [{'kind': 'k', 'a': 1, 'b': 2}])
    record.stop()
    record.stop()
    self.assertEqual(stopped, [1])
    self.assertEqual(self._files(), [])

  def test_records_before_keep_in_are_written_by_it(self):
    """A Record added before keep_in() is written by it."""
    records = launch_records.LaunchRecords()
    records.add({'kind': 'k'}, lambda: None)
    records.keep_in(self.directory)
    self.assertEqual(self._files(), [{'kind': 'k'}])

  def test_keep_in_returns_and_removes_what_was_left(self):
    """The files a previous server left are read once, then removed."""
    self.directory.mkdir()
    json_file.write_json(self.directory / 'a.json', {'kind': 'local'})
    (self.directory / 'b.json').write_text('{not json')
    (self.directory / '.c.json.1.tmp').write_text('{')
    left = launch_records.LaunchRecords().keep_in(self.directory)
    self.assertEqual(left, [{'kind': 'local'}])
    self.assertEqual(list(self.directory.iterdir()), [])

  def test_stop_all_stops_everything_and_then_nothing_starts(self):
    """stop_all() stops every Record at once; nothing is added after."""
    records = launch_records.LaunchRecords()
    records.keep_in(self.directory)
    stopped = []
    both = threading.Barrier(2, timeout=5)

    def stop(name):
      both.wait()  # Each waits for the other: they stop at once.
      stopped.append(name)

    records.add({'kind': 'a'}, lambda: stop('a'))
    records.add({'kind': 'b'}, lambda: stop('b'))
    records.stop_all()
    self.assertCountEqual(stopped, ['a', 'b'])
    self.assertEqual(self._files(), [])
    with self.assertRaisesRegex(RuntimeError, 'stopping'):
      records.add({'kind': 'c'}, lambda: None)
    with self.assertRaisesRegex(RuntimeError, 'stopping'):
      with records.starting():
        pass

  def test_a_launch_that_goes_on_after_stop_all_is_stopped_again(self):
    """What started after stop_all() stopped its Record is stopped too."""
    records = launch_records.LaunchRecords()
    stops = []
    record = records.add({'kind': 'k'}, lambda: stops.append(dict(fields)))
    fields = record.fields
    records.stop_all()
    with self.assertRaisesRegex(RuntimeError, 'stopping'):
      record.update(pid=7)
    self.assertEqual(stops, [{'kind': 'k'}, {'kind': 'k', 'pid': 7}])

  def test_stop_all_waits_a_bounded_time(self):
    """A stop that hangs is given up on, and logged."""
    records = launch_records.LaunchRecords()
    hang = threading.Event()
    records.add({'kind': 'k'}, hang.wait)
    began = time.monotonic()
    with self.assertLogs(level='ERROR'):
      records.stop_all(seconds=0.2)
    self.assertLess(time.monotonic() - began, 5)
    hang.set()

  def test_a_failing_stop_is_logged_and_forgotten(self):
    """A stop that raises does not stop the others."""
    records = launch_records.LaunchRecords()
    records.keep_in(self.directory)

    def fail():
      raise RuntimeError('adb: no device')

    stopped = []
    records.add({'kind': 'a'}, fail)
    records.add({'kind': 'b'}, lambda: stopped.append('b'))
    with self.assertLogs(level='ERROR'):
      records.stop_all()
    self.assertEqual(stopped, ['b'])
    self.assertEqual(self._files(), [])


if __name__ == '__main__':
  test_main.main()
