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
"""Tests for workspace SQLite database transactions and schemas."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from model_explorer_debugger.persistence import WorkspaceDatabase


class WorkspaceDatabaseTests(unittest.TestCase):

  def setUp(self):
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    self.root = Path(directory.name)
    self.state = {'sessions': [], 'artifacts': {}}
    (self.root / 'workspace.json').write_text(json.dumps(self.state))
    self.db = WorkspaceDatabase(self.root)

  def job(self):
    return dict(
        id='job',
        session_id='session',
        request_id='request',
        request_digest='digest',
        sequence=0,
        status='running',
        output={'ref': '', 'target': ''},
    )

  def test_migration_preserves_original_and_database_is_authoritative(self):
    self.assertEqual(
        json.loads((self.root / '.legacy-metadata/workspace.json').read_text()),
        self.state,
    )
    changed = {**self.state, 'extra': 'committed'}
    self.db.save_workspace(changed)
    (self.root / 'workspace.json').write_text('invalid obsolete export')
    reopened = WorkspaceDatabase(self.root)
    self.assertEqual(reopened.workspace(), changed)
    self.assertEqual(reopened.check(), 'ok')

  def test_workspace_job_and_event_abort_together(self):
    with self.assertRaisesRegex(OSError, 'fault'):
      with self.db.transaction():
        self.db.save_workspace({**self.state, 'accepted': 'job'})
        job = self.job()
        job['sequence'] = 1
        self.db.save_job(job, {'sequence': 1, 'type': 'status'})
        raise OSError('fault')
    self.assertEqual(self.db.workspace(), self.state)
    self.assertEqual(self.db.jobs(), [])
    self.assertEqual(self.db.events('job', 0), [])

  def test_duplicate_request_constraint_and_event_sequence_are_atomic(self):
    job = self.job()
    job['sequence'] = 1
    self.db.save_job(job, {'sequence': 1, 'type': 'status'})
    with self.assertRaises(sqlite3.IntegrityError):
      self.db.save_job({**job, 'id': 'second'})
    with self.assertRaises(sqlite3.IntegrityError):
      self.db.save_job(
          {**job, 'status': 'bad'}, {'sequence': 1, 'type': 'status'}
      )
    self.assertEqual(self.db.jobs()[0]['status'], 'running')
    self.assertEqual(len(self.db.jobs()), 1)

  def test_streaming_appends_deltas_and_reads_only_requested_page(self):
    job = self.job()
    self.db.save_job(job)
    for sequence in range(1, 101):
      job['sequence'] = sequence
      job['output']['ref'] += 'token'
      self.db.save_job(
          job, dict(sequence=sequence, type='delta', runId='ref', text='token')
      )
    with self.db.transaction() as sql:
      snapshot = json.loads(
          sql.execute('SELECT payload FROM jobs').fetchone()[0]
      )
    self.assertEqual(snapshot['output']['ref'], '')
    self.assertEqual(self.db.jobs()[0]['output']['ref'], 'token' * 100)
    self.assertEqual(
        [e['sequence'] for e in self.db.events('job', 90, limit=5)],
        list(range(91, 96)),
    )

  def test_process_loss_rolls_back_uncommitted_metadata(self):
    source = """
import os,sys
from model_explorer_debugger.persistence import WorkspaceDatabase
db=WorkspaceDatabase(sys.argv[1])
with db.transaction():
    db.save_workspace({'sessions':[], 'artifacts':{}, 'uncommitted':True})
    os._exit(17)
"""
    result = subprocess.run(
        [sys.executable, '-c', source, str(self.root)],
        env=os.environ.copy(),
        timeout=10,
    )
    self.assertEqual(result.returncode, 17)
    reopened = WorkspaceDatabase(self.root)
    self.assertEqual(reopened.workspace(), self.state)
    self.assertEqual(reopened.check(), 'ok')


class CaptureRecoveryTests(unittest.TestCase):

  def fixture(self):
    from test_local_runtime import LocalRuntimeTests

    case = LocalRuntimeTests()
    case.setUp()
    self.addCleanup(case.doCleanups)
    record = case.create()
    directory, results, artifacts = case.fixture_capture(record)
    publication = dict(
        job_id=directory.name,
        session_id=record['id'],
        turn=1,
        status='pending',
        record=deepcopy(record),
        results=results,
        artifacts=artifacts,
        requests={},
    )
    with case.registry.transaction():
      case.registry.update(
          record['id'],
          successful_turns=[
              dict(
                  n=1,
                  job_id=directory.name,
                  input='hello',
                  output={'ref': 'hello', 'target': 'hello'},
                  debug_data={'status': 'pending'},
              )
          ],
      )
      case.registry.database.publication(directory.name, publication)
    return case, record, directory, results, artifacts

  def recover(self, case):
    from unittest.mock import patch
    from model_explorer_debugger.jobs import JobManager
    from test_job_admission import FakeRunners

    runners = FakeRunners()
    with patch(
        'model_explorer_debugger.runtime.runners.Runners', return_value=runners
    ):
      manager = JobManager(case.registry)
    self.addCleanup(manager.close)
    self.assertEqual(runners.calls, [])
    return manager

  def test_published_files_are_adopted_after_metadata_commit_was_lost(self):
    from model_explorer_debugger.capture_importer import publish_capture
    from unittest.mock import patch

    case, record, directory, results, artifacts = self.fixture()
    capture = publish_capture(
        case.registry, record, directory, results, artifacts
    )
    with patch(
        'model_explorer_debugger.publication.publish_capture'
    ) as publish:
      self.recover(case)
      publish.assert_not_called()
    saved = case.registry.get(record['id'])
    self.assertEqual(saved['capture'], capture)
    self.assertEqual(
        saved['successful_turns'][0]['debug_data']['status'], 'available'
    )
    self.assertEqual(
        case.registry.database.publications()[directory.name]['status'],
        'complete',
    )

  def test_incomplete_staging_is_rebuilt_using_received_files(self):
    case, record, directory, results, artifacts = self.fixture()
    partial = (
        case.registry.root
        / 'sessions'
        / record['id']
        / 'captures'
        / (directory.name + '.pending')
    )
    partial.mkdir(parents=True)
    (partial / 'session.json').write_text('{')
    self.recover(case)
    saved = case.registry.get(record['id'])
    self.assertEqual(
        saved['successful_turns'][0]['debug_data']['status'], 'available'
    )
    self.assertFalse(partial.exists())

  def test_missing_received_files_preserve_text_and_resolve_pending(self):
    import shutil

    case, record, directory, results, artifacts = self.fixture()
    shutil.rmtree(directory)
    self.recover(case)
    saved = case.registry.get(record['id'])
    self.assertEqual(saved['successful_turns'][0]['output']['ref'], 'hello')
    self.assertEqual(
        saved['successful_turns'][0]['debug_data']['status'], 'unavailable'
    )


class NormalizedWorkspaceTests(unittest.TestCase):

  def setUp(self):
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    self.root = Path(directory.name)

  def state(self):
    turn = lambda n: dict(
        n=n,
        job_id=f'job{n}',
        input='hi',
        output={'ref': 'a' * 10, 'target': 'b' * 10},
    )
    return {
        'sessions': [
            dict(
                id='a',
                name='A',
                status='saved',
                successful_turns=[turn(1), turn(2), turn(3)],
            ),
            dict(
                id='b',
                name='B',
                status='saved',
                parent_session_id='a',
                successful_turns=[turn(1)],
            ),
            dict(id='c', name='C', status='draft'),
        ],
        'artifacts': {
            'model-1': {'path': '/m1', 'name': 'm1'},
            'model-2': {'path': '/m2', 'name': 'm2'},
        },
        'extra': {'kept': True},
    }

  def test_version_1_blob_migrates_to_rows_and_round_trips(self):
    state = self.state()
    with sqlite3.connect(self.root / 'workspace.sqlite3') as legacy:
      legacy.execute(
          'CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)'
      )
      legacy.execute(
          'INSERT INTO metadata VALUES (?,?)', ('workspace', json.dumps(state))
      )
      legacy.execute('PRAGMA user_version=1')
    db = WorkspaceDatabase(self.root)
    self.assertEqual(db.workspace(), state)
    with db.read() as sql:
      self.assertEqual(sql.execute('PRAGMA user_version').fetchone()[0], 2)
      self.assertEqual(
          sql.execute('SELECT count(*) FROM sessions').fetchone()[0], 3
      )
      self.assertEqual(
          sql.execute('SELECT count(*) FROM turns').fetchone()[0], 4
      )
      self.assertEqual(
          sql.execute('SELECT count(*) FROM artifacts').fetchone()[0], 2
      )
      self.assertEqual(
          json.loads(
              sql.execute(
                  "SELECT value FROM metadata WHERE key='workspace'"
              ).fetchone()[0]
          ),
          {'extra': {'kept': True}},
      )
    self.assertEqual(
        WorkspaceDatabase(self.root).workspace(),
        state,
        'a second open changes nothing',
    )

  def test_save_rewrites_only_the_changed_session_and_its_turns(self):
    (self.root / 'workspace.json').write_text(
        json.dumps({'sessions': [], 'artifacts': {}})
    )
    db = WorkspaceDatabase(self.root)
    state = self.state()
    db.save_workspace(state)
    with db.read() as sql:
      rows_a = sql.execute(
          "SELECT rowid FROM turns WHERE session_id='a' ORDER BY position"
      ).fetchall()
      row_a = sql.execute("SELECT rowid FROM sessions WHERE id='a'").fetchone()
    writer = db._connect('writer')
    before = writer.total_changes
    state['sessions'][1]['successful_turns'].append(
        dict(
            n=2,
            job_id='job2',
            input='again',
            output={'ref': 'x', 'target': 'y'},
        )
    )
    state['sessions'][2]['name'] = 'C renamed'
    db.save_workspace(state)
    # B: one session row, its one old turn deleted and two written; C: one row.
    # A and the artifacts: nothing.
    self.assertEqual(writer.total_changes - before, 1 + 1 + 2 + 1)
    self.assertEqual(db.workspace(), state)
    with db.read() as sql:
      self.assertEqual(
          sql.execute(
              "SELECT rowid FROM turns WHERE session_id='a' ORDER BY position"
          ).fetchall(),
          rows_a,
      )
      self.assertEqual(
          sql.execute("SELECT rowid FROM sessions WHERE id='a'").fetchone(),
          row_a,
      )
    del state['sessions'][0]
    db.save_workspace(state)
    self.assertEqual(db.workspace(), state)
    with db.read() as sql:
      self.assertEqual(
          sql.execute(
              "SELECT count(*) FROM turns WHERE session_id='a'"
          ).fetchone()[0],
          0,
      )
