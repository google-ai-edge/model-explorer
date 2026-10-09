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

"""Batches streaming job events into periodic SQLite commits.

Streaming job events are deltas and progress updates.
"""

import logging
import threading

logger = logging.getLogger(__name__)
FLUSH_INTERVAL_SECONDS = 0.05


class EventJournal:
  """Cheap in-memory appends; `flush()` commits everything buffered so far.

  A daemon thread flushes shortly after each append. Any reader that needs the
  durable rows (SSE pages, shutdown, a following non-streaming event) calls
  `flush()` itself from its own thread, so no thread ever waits on the flusher
  while holding a database lock.
  """

  def __init__(self, database, interval=FLUSH_INTERVAL_SECONDS):
    self.database = database
    self.interval = interval
    self.lock = threading.Lock()
    self.pending = []
    self.wake = threading.Event()
    self.closed = False
    self.thread = threading.Thread(
        target=self._run, daemon=True, name='event-journal'
    )
    self.thread.start()

  def append(self, job_id, sequence, event):
    with self.lock:
      self.pending.append((job_id, sequence, event))
    self.wake.set()

  def flush(self):
    """Commit the buffered rows in order.

    On failure the rows stay buffered for the next attempt.

    The database lock is taken first so two flushers never write the same rows
    and a thread already inside a transaction (which holds that lock) can flush
    without waiting on this thread.
    """
    with self.database.lock:
      with self.lock:
        rows = list(self.pending)
      if not rows:
        return
      self.database.append_events(rows)
      with self.lock:
        del self.pending[: len(rows)]

  def close(self):
    self.closed = True
    self.wake.set()
    self.thread.join(timeout=5)
    self.flush()

  def _run(self):
    while not self.closed:
      self.wake.wait()
      self.wake.clear()
      if self.closed:
        return
      self.wake.wait(self.interval)
      self.wake.clear()
      try:
        self.flush()
      except Exception as error:
        logger.warning('Streaming event commit deferred: %s', error)
        self.wake.wait(self.interval)
        self.wake.set()
