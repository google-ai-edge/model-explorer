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

"""What this server launched and has not stopped yet, so nothing outlives it.

Each launcher adds a Record as it starts something (a runner process, an
App, an adb forward), with how to stop it, and stops it through the Record.
stop_all() stops every Record left, as the server exits, however it exits;
from then on nothing more can start.

Each Record is also a JSON file in the workspace (keep_in()), saying how to
find what it started: a server killed (SIGKILL) stops nothing, and the next
one to open the workspace reads the files it left and stops what they name,
and nothing else (launchers.Launchers.record_in()).
"""

from collections.abc import Callable, Iterator
import contextlib
import pathlib
import threading
import time
from typing import Any
import uuid

from absl import logging
from model_debugger.server.util import json_file

# How long stop_all() waits for every Record to stop. Each stop is bounded
# itself: a runner process is killed 5 s after SIGTERM, a device's tool
# times out after 30 s.
STOP_ALL_SECONDS = 70


class Record:
  """One thing a launcher started, until it is stopped.

  Attributes:
      fields: How a later server finds it: 'kind' (the launcher's KIND), then
        what that launcher records.
  """

  def __init__(
      self,
      records: 'LaunchRecords',
      fields: dict[str, Any],
      stop: Callable[[], None],
  ) -> None:
    """Use LaunchRecords.add()."""
    self.fields = dict(fields)
    self._records = records
    self._stop = stop
    self._name = f'{uuid.uuid4().hex}.json'
    self._stopped = False
    self._lock = threading.Lock()

  @property
  def stopped(self) -> bool:
    """Whether it was stopped: by its launcher, or by stop_all()."""
    return self._stopped

  def update(self, **fields) -> None:
    """Adds to what it records, as the launch learns more.

    Args:
      **fields: Fields to update the launch record with.

    Raises:
      RuntimeError: The server is stopping; what was started is stopped
        again, since stop_all() may have come before it started.
    """
    with self._records._lock:  # pylint: disable=protected-access
      closed = self._records._closed  # pylint: disable=protected-access
      if not closed:
        self.fields.update(fields)
        self._records._write(self)  # pylint: disable=protected-access
    if closed:
      self.fields.update(fields)
      with self._lock:
        self._stopped = True
        self._stop_now()
      self._records._forget(self)  # pylint: disable=protected-access
      raise RuntimeError('The server is stopping.')

  def stop(self) -> None:
    """Stops what it started, then forgets it. Idempotent.

    A stop that fails is logged; the Record is forgotten all the same.
    """
    with self._lock:
      if self._stopped:
        return
      self._stopped = True
      self._stop_now()
    self._records._forget(self)  # pylint: disable=protected-access

  def _stop_now(self) -> None:
    try:
      self._stop()
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception('Could not stop %s', self.fields)


class LaunchRecords:
  """Every Record of this server not stopped yet."""

  def __init__(self) -> None:
    """Records in memory alone until keep_in()."""
    self._directory: pathlib.Path | None = None
    self._records: list[Record] = []
    self._closed = False
    # Reentrant: add() takes it inside starting().
    self._lock = threading.RLock()

  def keep_in(self, directory: pathlib.Path) -> list[dict[str, Any]]:
    """Keeps each Record as a file in `directory` from now on.

    Args:
        directory: The workspace's directory of them; created if missing.

    Returns:
        What the files a previous server left there recorded, oldest
        first; the files are removed. A file that cannot be read is
        removed too.
    """
    directory.mkdir(parents=True, exist_ok=True)
    left = []
    for path in sorted(
        directory.glob('*.json'), key=lambda path: path.stat().st_mtime
    ):
      try:
        fields = json_file.read_json(path)
      except (OSError, ValueError) as error:
        logging.warning('Unreadable runner record %s: %s', path, error)
        fields = None
      if isinstance(fields, dict):
        left.append(fields)
      path.unlink(missing_ok=True)
    # What a write cut short left.
    for path in directory.glob('.*.tmp'):
      path.unlink(missing_ok=True)
    with self._lock:
      self._directory = directory
      for record in self._records:
        self._write(record)
    return left

  @contextlib.contextmanager
  def starting(self) -> Iterator[None]:
    """Holds off stop_all() while something starts and is added.

    For what has no record before it exists (a process, by its pid):
    stop_all() either comes first, and nothing starts, or sees it.

    Raises:
        RuntimeError: The server is stopping.
    """
    with self._lock:
      if self._closed:
        raise RuntimeError('The server is stopping.')
      yield

  def add(self, fields: dict[str, Any], stop: Callable[[], None]) -> Record:
    """Records something a launcher is starting, or started.

    Args:
      fields: How a later server finds it; see Record.fields.
      stop: Stops it, whatever of it was started.

    Returns:
      The newly created Record instance.

    Raises:
      RuntimeError: The server is stopping. Nothing is started yet, or
        it is started inside starting(), which raises first.
    """
    record = Record(self, fields, stop)
    with self._lock:
      if self._closed:
        raise RuntimeError('The server is stopping.')
      self._records.append(record)
      self._write(record)
    return record

  def stop_all(self, seconds: float = STOP_ALL_SECONDS) -> None:
    """Stops every Record, all at once; nothing starts from now on.

    Waits at most `seconds` for them.

    Args:
      seconds: Maximum seconds to wait for runners to stop.
    """
    with self._lock:
      self._closed = True
      records = list(self._records)
    if not records:
      return
    logging.info('Stopping %d runner launch(es) left', len(records))
    threads = [
        threading.Thread(target=record.stop, name='stop-runner', daemon=True)
        for record in records
    ]
    for thread in threads:
      thread.start()
    deadline = time.monotonic() + seconds
    for thread in threads:
      thread.join(max(0, deadline - time.monotonic()))
    with self._lock:
      left = [record.fields for record in self._records]
    if left:
      logging.error('Runners not stopped within %d s: %s', seconds, left)

  def _write(self, record: Record) -> None:
    """Writes a Record's file. Called with the lock held."""
    if self._directory is None:
      return
    try:
      json_file.write_json(
          self._directory / record._name,  # pylint: disable=protected-access
          record.fields,
      )
    except OSError as error:
      logging.warning('Could not record a runner: %s', error)

  def _forget(self, record: Record) -> None:
    with self._lock:
      if record in self._records:
        self._records.remove(record)
      if self._directory is not None:
        (
            self._directory / record._name  # pylint: disable=protected-access
        ).unlink(missing_ok=True)
