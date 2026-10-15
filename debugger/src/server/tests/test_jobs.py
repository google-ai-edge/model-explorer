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

"""Tests of jobs and their events."""

import asyncio
import threading
import time

from model_debugger_server import errors, jobs
import pytest


def _collect(store, job_id, after=0):
  async def collect():
    return [event async for event in store.events(job_id, after)]

  return asyncio.run(asyncio.wait_for(collect(), 5))


def test_create_is_idempotent_per_request():
  """Create is idempotent per request."""
  store = jobs.Jobs()
  job, created = store.create('s', 'generate', 'r1', 'd', turn=2)
  assert created
  assert job == {
      'id': job['id'],
      'session_id': 's',
      'operation': 'generate',
      'status': 'running',
      'error': '',
      'sequence': 0,
      'output': {'ref': '', 'target': ''},
      'turn': 2,
      'progress': '',
      'created_at': job['created_at'],
  }
  again, created = store.create('s', 'generate', 'r1', 'd', turn=2)
  assert (again, created) == (job, False)
  with pytest.raises(ValueError, match='different request'):
    store.create('s', 'generate', 'r1', 'other')
  other, created = store.create('t', 'initialize', 'r1', 'other')
  assert created and other['status'] == 'initializing'
  assert store.create('s', 'new_chat', None, 'd')[0]['id'] != (
      store.create('s', 'new_chat', None, 'd')[0]['id']
  )


def test_emit_updates_the_job():
  """Emit updates the job."""
  store = jobs.Jobs()
  job, _ = store.create('s', 'generate', None, 'd', turn=1)
  store.emit(job['id'], 'delta', runId='ref', text='Hel')
  store.emit(job['id'], 'delta', runId='ref', text='lo')
  store.emit(job['id'], 'delta', runId='target', text='Hi')
  store.emit(job['id'], 'progress', runId='ref', message='Retrieving capture')
  store.finish(job['id'], 'failed', 'Runner lost.')
  got = store.get(job['id'])
  assert got['output'] == {'ref': 'Hello', 'target': 'Hi'}
  assert got['progress'] == 'Retrieving capture'
  assert (got['status'], got['error'], got['sequence']) == (
      'failed',
      'Runner lost.',
      5,
  )
  events = _collect(store, job['id'], after=3)
  assert events == [
      {
          'sequence': 4,
          'jobId': job['id'],
          'sessionId': 's',
          'turnId': 1,
          'type': 'progress',
          'runId': 'ref',
          'message': 'Retrieving capture',
      },
      {
          'sequence': 5,
          'jobId': job['id'],
          'sessionId': 's',
          'turnId': 1,
          'type': 'status',
          'status': 'failed',
          'error': 'Runner lost.',
      },
  ]
  assert _collect(store, job['id'], after=5) == []


def test_get_returns_copies():
  """Get returns copies."""
  store = jobs.Jobs()
  job, _ = store.create('s', 'generate', None, 'd')
  store.get(job['id'])['output']['ref'] = 'changed'
  assert store.get(job['id'])['output']['ref'] == ''


def test_finish_needs_a_terminal_status():
  """Finish needs a terminal status."""
  store = jobs.Jobs()
  job, _ = store.create('s', 'initialize', None, 'd')
  with pytest.raises(ValueError):
    store.finish(job['id'], 'running')


def test_unknown_job():
  """Unknown job."""
  store = jobs.Jobs()
  with pytest.raises(errors.NotFound):
    store.get('missing')
  with pytest.raises(errors.NotFound):
    store.emit('missing', 'progress', message='x')
  with pytest.raises(errors.NotFound):
    _collect(store, 'missing')


def test_events_wake_promptly_on_emits_from_threads():
  """Events wake promptly on emits from threads."""
  store = jobs.Jobs()
  job, _ = store.create('s', 'generate', None, 'd', turn=1)
  received = []

  def work():
    for text in ('a', 'b', 'c'):
      time.sleep(0.05)
      store.emit(job['id'], 'delta', runId='ref', text=text)
    time.sleep(0.05)
    store.finish(job['id'], 'completed')

  async def follow():
    worker = threading.Thread(target=work)
    worker.start()
    async for event in store.events(job['id'], 0):
      received.append((event['sequence'], time.monotonic()))
    worker.join()

  started = time.monotonic()
  asyncio.run(asyncio.wait_for(follow(), 5))
  assert [sequence for sequence, _ in received] == [1, 2, 3, 4]
  # Four emits 50 ms apart: no polling interval adds to that.
  assert received[-1][1] - started < 0.5


def test_two_readers_and_a_reader_that_leaves():
  """Two readers and a reader that leaves."""
  store = jobs.Jobs()
  job, _ = store.create('s', 'generate', None, 'd')

  async def scenario():
    async def read_all():
      return [e['sequence'] async for e in store.events(job['id'], 0)]

    async def read_one():
      async for event in store.events(job['id'], 0):
        return event['sequence']

    everything = asyncio.create_task(read_all())
    first = asyncio.create_task(read_one())
    await asyncio.sleep(0.01)
    store.emit(job['id'], 'delta', runId='ref', text='x')
    assert await first == 1
    await asyncio.to_thread(store.finish, job['id'], 'cancelled', 'Stopped.')
    return await everything

  assert asyncio.run(asyncio.wait_for(scenario(), 5)) == [1, 2]
