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
from typing import Any

from absl.testing import absltest
from model_debugger.server.session import jobs
from model_debugger.server.testing import test_main
from model_debugger.server.util import errors


def _collect(store, job_id, after=0):
  async def collect():
    return [event async for event in store.events(job_id, after)]

  return asyncio.run(asyncio.wait_for(collect(), 5))


class JobsTest(absltest.TestCase):
  """Tests of jobs and their events."""

  def test_create_makes_a_new_job_each_call(self):
    """Every create is a new job."""
    store = jobs.Jobs()
    job = store.create('s', 'generate', turn=2, prompt='Hi', chat=3)
    self.assertEqual(
        job,
        {
            'id': job['id'],
            'session_id': 's',
            'operation': 'generate',
            'status': 'running',
            'error': '',
            'sequence': 0,
            'output': {'ref': '', 'target': ''},
            'reasoning': {'ref': '', 'target': ''},
            'chat': 3,
            'turn': 2,
            'prompt': 'Hi',
            'note': '',
            'status_text': '',
            'progress': '',
            'created_at': job['created_at'],
        },
    )
    self.assertEqual(store.create('t', 'initialize')['status'], 'initializing')
    self.assertNotEqual(
        store.create('s', 'initialize')['id'],
        store.create('s', 'initialize')['id'],
    )

  def test_emit_updates_the_job(self):
    """Emit updates the job."""
    store = jobs.Jobs()
    job = store.create('s', 'generate', turn=1)
    store.emit(job['id'], 'delta', runId='ref', channel='content', text='Hel')
    store.emit(job['id'], 'delta', runId='ref', channel='content', text='lo')
    store.emit(job['id'], 'delta', runId='target', channel='content', text='Hi')
    store.emit(job['id'], 'progress', runId='ref', message='Retrieving capture')
    store.finish(job['id'], 'failed', 'Runner lost.')
    got = store.get(job['id'])
    self.assertEqual(got['output'], {'ref': 'Hello', 'target': 'Hi'})
    self.assertEqual(got['reasoning'], {'ref': '', 'target': ''})
    self.assertEqual(got['progress'], 'Retrieving capture')
    self.assertEqual(
        (got['status'], got['error'], got['sequence']),
        (
            'failed',
            'Runner lost.',
            5,
        ),
    )
    events = _collect(store, job['id'], after=3)
    self.assertEqual(
        events,
        [
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
                'note': '',
                'status_text': 'Generation failed',
            },
        ],
    )
    self.assertEqual(_collect(store, job['id'], after=5), [])

  def test_brief_is_a_job_without_its_outputs(self):
    """What a listing reads: no reply is copied."""
    store = jobs.Jobs()
    job = store.create('s1', 'generate', 1, 'Hi')
    store.emit(
        job['id'], 'delta', runId='ref', channel='content', text='x' * 99
    )
    store.emit(job['id'], 'progress', message='Generating')
    self.assertEqual(
        store.brief(job['id']),
        {
            'id': job['id'],
            'operation': 'generate',
            'status': 'running',
            'error': '',
            'progress': 'Generating',
            'created_at': job['created_at'],
        },
    )
    with self.assertRaises(errors.NotFound):
      store.brief('none')

  def test_progress_is_each_runs_latest(self):
    """Each run keeps its latest progress message; the job, the last."""
    store = jobs.Jobs()
    job = store.create('s', 'initialize')
    self.assertEqual(store.progress(job['id']), {})
    for run, message in (
        ('ref', 'Starting runner'),
        ('target', 'Starting runner'),
        ('ref', 'Uploading model · 45%'),
        ('target', 'Loading model'),
    ):
      store.emit(job['id'], 'progress', runId=run, message=message)
    self.assertEqual(
        store.progress(job['id']),
        {'ref': 'Uploading model · 45%', 'target': 'Loading model'},
    )
    self.assertEqual(store.get(job['id'])['progress'], 'Loading model')
    with self.assertRaises(errors.NotFound):
      store.progress('missing')

  def test_a_reasoning_delta_adds_to_the_thinking_not_the_reply(self):
    """Each channel's pieces are joined apart, and the events say which."""
    store = jobs.Jobs()
    job = store.create('s', 'generate', turn=1)
    store.emit(job['id'], 'delta', runId='ref', channel='reasoning', text='Hm')
    store.emit(job['id'], 'delta', runId='ref', channel='reasoning', text='m')
    store.emit(job['id'], 'delta', runId='ref', channel='content', text='4')
    got = store.get(job['id'])
    self.assertEqual(got['reasoning'], {'ref': 'Hmm', 'target': ''})
    self.assertEqual(got['output'], {'ref': '4', 'target': ''})
    store.finish(job['id'], 'completed')
    self.assertEqual(
        [
            (event['channel'], event['text'])
            for event in _collect(store, job['id'], after=0)
            if event['type'] == 'delta'
        ],
        [('reasoning', 'Hm'), ('reasoning', 'm'), ('content', '4')],
    )

  def test_a_cancelled_generate_says_its_turn_is_not_captured(self):
    """The note and status text come from the status, job and event."""
    store = jobs.Jobs()
    job = store.create('s', 'generate', turn=1, prompt='Hi')
    store.emit(job['id'], 'delta', runId='ref', channel='content', text='Hel')
    store.finish(job['id'], 'cancelled', 'Stopped.')
    got = store.get(job['id'])
    self.assertEqual(
        (
            got['status'],
            got['error'],
            got['note'],
            got['status_text'],
            got['output'],
        ),
        (
            'cancelled',
            'Stopped.',
            jobs.STOPPED_TURN,
            'Generation stopped',
            {'ref': 'Hel', 'target': ''},
        ),
    )
    (event,) = _collect(store, job['id'], after=1)
    self.assertEqual(
        (event['note'], event['status_text']),
        (jobs.STOPPED_TURN, 'Generation stopped'),
    )

  def test_only_a_generate_has_a_status_text(self):
    """A failed generate is labelled; a failed initialize is not."""
    store = jobs.Jobs()
    generate = store.create('s', 'generate', turn=1)
    initialize = store.create('s', 'initialize')
    for job in (generate, initialize):
      store.finish(job['id'], 'failed', 'Broken.')
    self.assertEqual(
        [
            (got['note'], got['status_text'])
            for got in map(store.get, (generate['id'], initialize['id']))
        ],
        [('', 'Generation failed'), ('', '')],
    )

  def test_get_returns_copies(self):
    """Get returns copies."""
    store = jobs.Jobs()
    job = store.create('s', 'generate')
    store.get(job['id'])['output']['ref'] = 'changed'
    self.assertEqual(store.get(job['id'])['output']['ref'], '')

  def test_finish_needs_a_terminal_status(self):
    """Finish needs a terminal status."""
    store = jobs.Jobs()
    job = store.create('s', 'initialize')
    with self.assertRaises(ValueError):
      store.finish(job['id'], 'running')

  def test_unknown_job(self):
    """Unknown job."""
    store = jobs.Jobs()
    with self.assertRaises(errors.NotFound):
      store.get('missing')
    with self.assertRaises(errors.NotFound):
      store.emit('missing', 'progress', message='x')
    with self.assertRaises(errors.NotFound):
      _collect(store, 'missing')

  def test_events_wake_promptly_on_emits_from_threads(self):
    """Events wake promptly on emits from threads."""
    store = jobs.Jobs()
    job = store.create('s', 'generate', turn=1)
    received = []

    def work():
      for text in ('a', 'b', 'c'):
        time.sleep(0.05)
        store.emit(
            job['id'],
            'delta',
            runId='ref',
            channel='content',
            text=text,
        )
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
    self.assertEqual([sequence for sequence, _ in received], [1, 2, 3, 4])
    # Four emits 50 ms apart: no polling interval adds to that.
    self.assertLess(received[-1][1] - started, 0.5)

  def test_a_reader_from_any_sequence_gets_the_events_after_it(self):
    """Live or after the end, from any sequence: what was emitted.

    As a reader reconnecting mid-reply or just after the end does; the
    pieces of both sides and channels interleave, as they stream.
    """
    store = jobs.Jobs()
    job = store.create('s', 'generate', turn=3, prompt='Hi')
    emitted = [
        ('progress', {'runId': 'ref', 'message': 'Generating'}),
        ('delta', {'runId': 'ref', 'channel': 'reasoning', 'text': 'Hm'}),
        ('delta', {'runId': 'target', 'channel': 'content', 'text': 'é'}),
        ('delta', {'runId': 'ref', 'channel': 'reasoning', 'text': 'm.'}),
        ('delta', {'runId': 'ref', 'channel': 'content', 'text': '4'}),
        ('delta', {'runId': 'target', 'channel': 'content', 'text': ''}),
        ('delta', {'runId': 'target', 'channel': 'content', 'text': '🙂'}),
        ('progress', {'message': 'Retrieving capture'}),
    ]
    live = []

    async def follow():
      async def read():
        async for event in store.events(job['id'], 0):
          live.append(event)

      async def join(after: int, count: int) -> list[dict[str, Any]]:
        """What a reader joining now from `after` gets first."""
        joined = store.events(job['id'], after)
        try:
          return [await anext(joined) for _ in range(count)]
        finally:
          await joined.aclose()

      reader = asyncio.create_task(read())
      for sequence, (event_type, fields) in enumerate(emitted, start=1):
        store.emit(job['id'], event_type, **fields)
        await asyncio.sleep(0.01)
        self.assertLen(live, sequence)
        # A reader joining mid-reply, from every sequence so far.
        for after in range(sequence + 1):
          self.assertEqual(await join(after, sequence - after), live[after:])
      store.finish(job['id'], 'completed')
      await reader

    asyncio.run(asyncio.wait_for(follow(), 5))
    expected = [
        {
            'sequence': n,
            'jobId': job['id'],
            'sessionId': 's',
            'turnId': 3,
            'type': type,
            **fields,
        }
        for n, (type, fields) in enumerate(emitted, start=1)
    ]
    status = live[-1]
    self.assertEqual(live[:-1], expected)
    self.assertEqual(
        (status['sequence'], status['type'], status['status']),
        (9, 'status', 'completed'),
    )
    # After the end, from every sequence, and from before the first.
    for after in range(-1, len(live) + 1):
      self.assertEqual(_collect(store, job['id'], after), live[max(after, 0) :])
    got = store.get(job['id'])
    self.assertEqual(got['reasoning'], {'ref': 'Hmm.', 'target': ''})
    self.assertEqual(got['output'], {'ref': '4', 'target': 'é🙂'})

  def test_two_readers_and_a_reader_that_leaves(self):
    """Two readers and a reader that leaves."""
    store = jobs.Jobs()
    job = store.create('s', 'generate')

    async def scenario():
      async def read_all():
        return [e['sequence'] async for e in store.events(job['id'], 0)]

      async def read_one():
        async for event in store.events(job['id'], 0):
          return event['sequence']

      everything = asyncio.create_task(read_all())
      first = asyncio.create_task(read_one())
      await asyncio.sleep(0.01)
      store.emit(job['id'], 'delta', runId='ref', channel='content', text='x')
      self.assertEqual(await first, 1)
      await asyncio.to_thread(store.finish, job['id'], 'cancelled', 'Stopped.')
      return await everything

    self.assertEqual(asyncio.run(asyncio.wait_for(scenario(), 5)), [1, 2])


if __name__ == '__main__':
  test_main.main()
