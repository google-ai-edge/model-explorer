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
"""Tests for session event journal append and retrieval."""

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from model_explorer_debugger.event_journal import EventJournal
from model_explorer_debugger.persistence import WorkspaceDatabase


class EventJournalTests(unittest.TestCase):

  def setUp(self):
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    root = Path(directory.name)
    (root / 'workspace.json').write_text(
        json.dumps({'sessions': [], 'artifacts': {}})
    )
    self.db = WorkspaceDatabase(root)
    self.job = dict(
        id='job',
        session_id='session',
        request_id='request',
        request_digest='digest',
        sequence=0,
        status='running',
        output={'ref': '', 'target': ''},
    )
    self.db.save_job(self.job)
    self.journal = EventJournal(self.db, interval=0.01)
    self.addCleanup(self.journal.close)

  def delta(self, sequence, text='t'):
    return dict(
        sequence=sequence,
        jobId='job',
        sessionId='session',
        turnId=1,
        type='delta',
        runId='ref',
        text=text,
    )

  def test_flush_commits_buffered_rows_in_order_and_advances_job_sequence(self):
    for sequence in range(1, 51):
      self.journal.append('job', sequence, self.delta(sequence))
    self.journal.flush()
    self.assertEqual(
        [e['sequence'] for e in self.db.events('job', 0, limit=100)],
        list(range(1, 51)),
    )
    self.assertEqual(self.db.jobs()[0]['sequence'], 50)
    self.assertEqual(self.db.jobs()[0]['output']['ref'], 't' * 50)

  def test_background_flush_makes_rows_durable_without_a_reader(self):
    self.journal.append('job', 1, self.delta(1))
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not self.db.events('job', 0):
      time.sleep(0.01)
    self.assertEqual(len(self.db.events('job', 0)), 1)

  def test_flush_from_a_thread_holding_a_write_transaction_does_not_deadlock(
      self,
  ):
    self.journal.append('job', 1, self.delta(1))
    finished = threading.Event()

    def writer():
      with self.db.transaction():
        self.journal.flush()
        self.db.save_job(
            {**self.job, 'sequence': 2, 'status': 'completed'},
            dict(
                sequence=2,
                jobId='job',
                sessionId='session',
                turnId=1,
                type='status',
                status='completed',
            ),
        )
      finished.set()

    threading.Thread(target=writer, daemon=True).start()
    self.assertTrue(
        finished.wait(5),
        'flush inside a transaction must reuse that transaction',
    )
    self.assertEqual(
        [e['type'] for e in self.db.events('job', 0)], ['delta', 'status']
    )

  def test_failed_commit_keeps_rows_for_the_next_flush(self):
    real = self.db.append_events
    failures = {'count': 0}

    def flaky(rows):
      if failures['count'] == 0:
        failures['count'] += 1
        raise OSError('disk busy')
      real(rows)

    self.db.append_events = flaky
    self.journal.append('job', 1, self.delta(1))
    with self.assertRaises(OSError):
      self.journal.flush()
    self.journal.flush()
    self.assertEqual(len(self.db.events('job', 0)), 1)


class ReadPathTests(unittest.TestCase):

  def test_reads_do_not_wait_for_the_writer_lock(self):
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    root = Path(directory.name)
    (root / 'workspace.json').write_text(
        json.dumps({'sessions': [], 'artifacts': {}})
    )
    db = WorkspaceDatabase(root)
    holding, release = threading.Event(), threading.Event()

    def hold():
      with db.transaction():
        holding.set()
        release.wait(5)

    threading.Thread(target=hold, daemon=True).start()
    self.assertTrue(holding.wait(5))
    try:
      started = time.monotonic()
      self.assertEqual(db.workspace()['sessions'], [])
      self.assertEqual(db.events('missing', 0), [])
      self.assertLess(
          time.monotonic() - started,
          1,
          'readers must not queue behind an open write transaction',
      )
    finally:
      release.set()


if __name__ == '__main__':
  unittest.main()
