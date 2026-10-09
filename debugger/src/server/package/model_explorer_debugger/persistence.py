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

"""Transactional local metadata.

JSON files are migration inputs/exports only.
"""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import threading
import uuid


def encode(value):
  return json.dumps(
      value, ensure_ascii=False, allow_nan=False, separators=(',', ':')
  )


SCHEMA_VERSION = 2
TURNS_IN_TABLE = '__turns_in_table__'


class WorkspaceDatabase:
  """One SQLite file; writers serialize on `lock`, readers use WAL snapshots.

  Each thread keeps one writer and one reader connection open. `transaction()`
  is the only write path (BEGIN IMMEDIATE); `read()` never takes the lock and
  sees the calling thread's own open transaction when there is one.
  """

  def __init__(self, root):
    self.root = Path(root)
    self.path = self.root / 'workspace.sqlite3'
    self.lock = threading.RLock()
    self.local = threading.local()
    with self.transaction() as db:
      version = db.execute('PRAGMA user_version').fetchone()[0]
      if version not in (0, 1, SCHEMA_VERSION):
        raise ValueError(f'Unsupported workspace database version: {version}')
      db.execute(
          'CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value'
          ' TEXT NOT NULL)'
      )
      db.execute("""CREATE TABLE IF NOT EXISTS sessions (
              id TEXT PRIMARY KEY,
              parent_session_id TEXT,
              position INTEGER NOT NULL,
              payload TEXT NOT NULL,
              turns_digest TEXT NOT NULL DEFAULT '')""")
      db.execute("""CREATE TABLE IF NOT EXISTS turns (
              session_id TEXT NOT NULL REFERENCES sessions(id),
              position INTEGER NOT NULL,
              payload TEXT NOT NULL,
              PRIMARY KEY(session_id, position))""")
      db.execute(
          'CREATE TABLE IF NOT EXISTS artifacts (id TEXT PRIMARY KEY, position'
          ' INTEGER NOT NULL, payload TEXT NOT NULL)'
      )
      db.execute("""CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL,
              request_id TEXT NOT NULL,
              request_digest TEXT NOT NULL,
              sequence INTEGER NOT NULL,
              snapshot_sequence INTEGER NOT NULL,
              payload TEXT NOT NULL,
              UNIQUE(session_id, request_id))""")
      db.execute("""CREATE TABLE IF NOT EXISTS events (
              job_id TEXT NOT NULL REFERENCES jobs(id),
              sequence INTEGER NOT NULL,
              payload TEXT NOT NULL,
              PRIMARY KEY(job_id, sequence))""")
      db.execute(
          'CREATE TABLE IF NOT EXISTS publications (job_id TEXT PRIMARY KEY,'
          ' payload TEXT NOT NULL)'
      )
      if version < SCHEMA_VERSION:
        blob = db.execute(
            "SELECT value FROM metadata WHERE key='workspace'"
        ).fetchone()
        if blob is None:
          self._migrate(db)
        else:
          # Version 1 kept the workspace as one JSON value; split it into rows.
          self._save_rows(db, json.loads(blob[0]))
      db.execute(f'PRAGMA user_version={SCHEMA_VERSION}')

  @contextmanager
  def transaction(self):
    with self.lock:
      current = getattr(self.local, 'connection', None)
      if current is not None:
        name = 'nested_' + uuid.uuid4().hex
        current.execute('SAVEPOINT ' + name)
        try:
          yield current
          current.execute('RELEASE ' + name)
        except BaseException:
          current.execute('ROLLBACK TO ' + name)
          current.execute('RELEASE ' + name)
          raise
        return
      db = self._connect('writer')
      try:
        db.execute('BEGIN IMMEDIATE')
        self.local.connection = db
        yield db
        db.execute('COMMIT')
      except BaseException:
        if db.in_transaction:
          db.execute('ROLLBACK')
        raise
      finally:
        self.local.connection = None

  @contextmanager
  def read(self):
    """A consistent read snapshot.

    Inside a write transaction it reads that transaction.
    """
    current = getattr(self.local, 'connection', None)
    if current is not None:
      yield current
      return
    db = self._connect('reader')
    db.execute('BEGIN')
    try:
      yield db
    finally:
      db.execute('COMMIT')

  def _connect(self, role):
    db = getattr(self.local, role, None)
    if db is None:
      db = sqlite3.connect(
          self.path, timeout=5, isolation_level=None, check_same_thread=False
      )
      db.execute('PRAGMA foreign_keys=ON')
      db.execute('PRAGMA journal_mode=WAL')
      db.execute('PRAGMA synchronous=FULL')
      if role == 'reader':
        db.execute('PRAGMA query_only=ON')
      setattr(self.local, role, db)
    return db

  def close(self):
    """Close this thread's connections.

    Other threads close theirs when they exit.
    """
    for role in ('writer', 'reader'):
      db = getattr(self.local, role, None)
      if db is not None:
        db.close()
        setattr(self.local, role, None)

  def _migrate(self, db):
    path = self.root / 'workspace.json'
    state = (
        json.loads(path.read_text())
        if path.exists()
        else {'sessions': [], 'artifacts': {}}
    )
    if not isinstance(state.get('sessions'), list) or not isinstance(
        state.get('artifacts'), dict
    ):
      raise ValueError('Invalid legacy workspace metadata')
    accepted = {(s['id'], s.get('job_id')) for s in state['sessions']}
    backup = self.root / '.legacy-metadata'
    paths = [path] if path.exists() else []
    for job_path in sorted((self.root / 'sessions').glob('*/jobs/*/job.json')):
      job = json.loads(job_path.read_text())
      if (
          job.pop('admission', None) == 'pending'
          and (job['session_id'], job['id']) not in accepted
      ):
        continue
      events_path = job_path.with_name('events.jsonl')
      events = []
      if events_path.exists():
        lines = events_path.read_text().splitlines()
        for index, line in enumerate(lines):
          try:
            event = json.loads(line)
          except ValueError:
            if index != len(lines) - 1:
              raise ValueError(
                  f'Corrupt legacy event log: {events_path}'
              ) from None
            break  # an interrupted final append is not a committed event
          events.append(event)
        paths.append(events_path)
      paths.append(job_path)
      sequence = max([job.get('sequence', 0)] + [e['sequence'] for e in events])
      for event in events:
        if event['sequence'] > job.get('sequence', 0):
          if event['type'] == 'delta':
            job['output'][event['runId']] += event['text']
          elif event['type'] == 'progress':
            job['progress'] = event.get('message', '')
      job['sequence'] = sequence
      self._insert_job(db, job)
      for event in events:
        db.execute(
            'INSERT INTO events VALUES (?,?,?)',
            (job['id'], event['sequence'], encode(event)),
        )
    # Never overwrite the original migration evidence on a retry.
    for source in paths:
      target = backup / source.relative_to(self.root)
      if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    self._save_rows(db, state)

  def workspace(self):
    with self.read() as db:
      return self._read_rows(db)

  def save_workspace(self, state):
    with self.transaction() as db:
      self._save_rows(db, state)

  def _read_rows(self, db):
    row = db.execute(
        "SELECT value FROM metadata WHERE key='workspace'"
    ).fetchone()
    extra = json.loads(row[0]) if row else {}
    turns = {}
    for session_id, payload in db.execute(
        'SELECT session_id, payload FROM turns ORDER BY session_id, position'
    ):
      turns.setdefault(session_id, []).append(json.loads(payload))
    sessions = []
    for identity, payload in db.execute(
        'SELECT id, payload FROM sessions ORDER BY position'
    ):
      record = json.loads(payload)
      if record.pop(TURNS_IN_TABLE, False):
        record['successful_turns'] = turns.get(identity, [])
      sessions.append(record)
    artifacts = {
        identity: json.loads(payload)
        for identity, payload in db.execute(
            'SELECT id, payload FROM artifacts ORDER BY position'
        )
    }
    return {**extra, 'sessions': sessions, 'artifacts': artifacts}

  def _save_rows(self, db, state):
    """Upsert only the rows that differ from the stored ones.

    Unchanged Sessions cost no write.
    """
    extra = encode({
        key: value
        for key, value in state.items()
        if key not in ('sessions', 'artifacts')
    })
    stored_extra = db.execute(
        "SELECT value FROM metadata WHERE key='workspace'"
    ).fetchone()
    if stored_extra is None:
      db.execute('INSERT INTO metadata VALUES (?,?)', ('workspace', extra))
    elif stored_extra[0] != extra:
      db.execute("UPDATE metadata SET value=? WHERE key='workspace'", (extra,))
    stored = {
        identity: (position, payload, digest)
        for identity, position, payload, digest in db.execute(
            'SELECT id, position, payload, turns_digest FROM sessions'
        )
    }
    wanted = {}
    for position, record in enumerate(state['sessions']):
      payload = {
          key: value
          for key, value in record.items()
          if key != 'successful_turns'
      }
      turns = None
      if 'successful_turns' in record:
        payload[TURNS_IN_TABLE] = True
        turns = [encode(turn) for turn in record['successful_turns']]
      digest = (
          hashlib.sha256('\n'.join(turns).encode()).hexdigest()
          if turns is not None
          else ''
      )
      wanted[record['id']] = (
          position,
          encode(payload),
          digest,
          turns,
          record.get('parent_session_id'),
      )
    for identity in stored.keys() - wanted.keys():
      db.execute('DELETE FROM turns WHERE session_id=?', (identity,))
      db.execute('DELETE FROM sessions WHERE id=?', (identity,))
    for identity, (position, payload, digest, turns, parent) in wanted.items():
      previous = stored.get(identity)
      if previous != (position, payload, digest):
        db.execute(
            """INSERT INTO sessions
                (id, parent_session_id, position, payload, turns_digest)
                VALUES (?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                parent_session_id=excluded.parent_session_id,
                position=excluded.position,
                payload=excluded.payload,
                turns_digest=excluded.turns_digest""",
            (identity, parent, position, payload, digest),
        )
      if previous is None or previous[2] != digest:
        db.execute('DELETE FROM turns WHERE session_id=?', (identity,))
        if turns:
          db.executemany(
              'INSERT INTO turns VALUES (?,?,?)',
              [(identity, index, turn) for index, turn in enumerate(turns)],
          )
    stored_artifacts = {
        identity: (position, payload)
        for identity, position, payload in db.execute(
            'SELECT id, position, payload FROM artifacts'
        )
    }
    wanted_artifacts = {
        identity: (position, encode(value))
        for position, (identity, value) in enumerate(state['artifacts'].items())
    }
    for identity in stored_artifacts.keys() - wanted_artifacts.keys():
      db.execute('DELETE FROM artifacts WHERE id=?', (identity,))
    for identity, (position, payload) in wanted_artifacts.items():
      if stored_artifacts.get(identity) != (position, payload):
        db.execute(
            """INSERT INTO artifacts VALUES (?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                position=excluded.position, payload=excluded.payload""",
            (identity, position, payload),
        )

  def _insert_job(self, db, job):
    value = deepcopy(job)
    value.pop('admission', None)
    db.execute(
        'INSERT INTO jobs VALUES (?,?,?,?,?,?,?)',
        (
            job['id'],
            job['session_id'],
            job['request_id'],
            job['request_digest'],
            job['sequence'],
            job['sequence'],
            encode(value),
        ),
    )

  def save_job(self, job, event=None):
    with self.transaction() as db:
      exists = db.execute(
          'SELECT 1 FROM jobs WHERE id=?', (job['id'],)
      ).fetchone()
      if not exists:
        self._insert_job(db, job)
      elif event and event['type'] in ('delta', 'progress'):
        # Append streaming data without rewriting the growing output.
        db.execute(
            'UPDATE jobs SET sequence=? WHERE id=?',
            (job['sequence'], job['id']),
        )
      else:
        value = {k: v for k, v in job.items() if k != 'admission'}
        db.execute(
            'UPDATE jobs SET sequence=?,snapshot_sequence=?,payload=? WHERE'
            ' id=?',
            (job['sequence'], job['sequence'], encode(value), job['id']),
        )
      if event:
        db.execute(
            'INSERT INTO events VALUES (?,?,?)',
            (job['id'], event['sequence'], encode(event)),
        )

  def append_events(self, rows):
    """Commit buffered streaming events in one transaction.

    Each row is a `(job_id, sequence, event)` tuple.
    """
    if not rows:
      return
    with self.transaction() as db:
      db.executemany(
          'INSERT INTO events VALUES (?,?,?)',
          [
              (job_id, sequence, encode(event))
              for job_id, sequence, event in rows
          ],
      )
      latest = {}
      for job_id, sequence, _ in rows:
        latest[job_id] = max(latest.get(job_id, 0), sequence)
      db.executemany(
          'UPDATE jobs SET sequence=MAX(sequence,?) WHERE id=?',
          [(sequence, job_id) for job_id, sequence in latest.items()],
      )

  def jobs(self):
    with self.read() as db:
      result = []
      for identity, sequence, snapshot, payload in db.execute(
          'SELECT id,sequence,snapshot_sequence,payload FROM jobs'
      ):
        job = json.loads(payload)
        for (raw,) in db.execute(
            'SELECT payload FROM events WHERE job_id=? AND sequence>? ORDER BY'
            ' sequence',
            (identity, snapshot),
        ):
          event = json.loads(raw)
          if event['type'] == 'delta':
            job['output'][event['runId']] += event['text']
          elif event['type'] == 'progress':
            job['progress'] = event.get('message', '')
        job['sequence'] = sequence
        result.append(job)
      return result

  def events(self, identity, after, limit=256):
    with self.read() as db:
      return [
          json.loads(row[0])
          for row in db.execute(
              'SELECT payload FROM events WHERE job_id=? AND sequence>? ORDER'
              ' BY sequence LIMIT ?',
              (identity, after, limit),
          )
      ]

  def publication(self, identity, value):
    with self.transaction() as db:
      db.execute(
          'INSERT OR REPLACE INTO publications VALUES (?,?)',
          (identity, encode(value)),
      )

  def publications(self):
    with self.read() as db:
      return {
          key: json.loads(value)
          for key, value in db.execute(
              'SELECT job_id,payload FROM publications'
          )
      }

  def check(self):
    with self.read() as db:
      return db.execute('PRAGMA quick_check').fetchone()[0]
