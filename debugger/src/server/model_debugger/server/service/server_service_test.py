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

"""Tests of the HTTP API, as the UI calls it, over fake runners."""

import asyncio
import json
import pathlib
import shutil
import threading
import time
from typing import Any
import unittest
from unittest import mock
import urllib.parse

from absl.testing import absltest
import httpx
from model_debugger.server.analysis import token_analysis
from model_debugger.server.architectures import catalog
from model_debugger.server.service import analysis_requests
from model_debugger.server.service import server_service
from model_debugger.server.session import jobs
from model_debugger.server.session import mappings
from model_debugger.server.session import model_comparison
from model_debugger.server.session import model_graphs
from model_debugger.server.session import model_store
from model_debugger.server.session import prepared_models
from model_debugger.server.session import session_archive
from model_debugger.server.session import session_store
from model_debugger.server.testing import anchor_fixtures
from model_debugger.server.testing import fake_launchers
from model_debugger.server.testing import pytorch_graph_fixtures
from model_debugger.server.testing import tap_fixtures
from model_debugger.server.testing import test_main
from model_debugger.server.testing import world as world_lib
from model_debugger.server.util import errors
from model_debugger.server.util import json_file

_ORIGIN = 'http://127.0.0.1:8080'


async def _job(
    client: httpx.AsyncClient, job: dict[str, Any]
) -> dict[str, Any]:
  """Polls a job over HTTP until it ended."""
  for _ in range(2000):
    job = (await client.get(f'/api/jobs/{job["id"]}')).json()
    if job['status'] in jobs.TERMINAL:
      return job
    await asyncio.sleep(0.01)
  raise AssertionError(f'job did not end: {job}')


async def _created(client, config: dict[str, Any]) -> dict[str, Any]:
  """Creates a Session over HTTP; returns it once its Model Server is on."""
  created = await client.post('/api/sessions/create', json=config)
  assert created.status_code == 200, created.text
  record = created.json()
  job = await _job(client, {'id': record['job_id']})
  assert job['status'] == 'completed', job['error']
  return record


async def _started(client, world, chat: bool = True) -> dict[str, Any]:
  """Creates a Session over HTTP, its Model Server on as it is created.

  Then starts a chat of each side's defaults, unless `chat` is False.

  Args:
    client: HTTP test client.
    world: The test world environment.
    chat: Whether to start an initial chat.

  Returns:
    The created session record.
  """
  record = await _created(client, world.config())
  if chat:
    await _chat(client, record['id'], {'ref': {}, 'target': {}})
  return record


async def _chat(
    client, record_id: str, generation: dict[str, Any]
) -> dict[str, Any]:
  """Starts a chat of a Session over HTTP; returns its job, completed."""
  response = await client.post(
      f'/api/sessions/{record_id}/chats', json={'generation': generation}
  )
  assert response.status_code == 200, response.text
  job = await _job(client, response.json())
  assert job['status'] == 'completed', job['error']
  return job


def _frames(text: str) -> list[tuple[str, dict[str, Any]]]:
  """The (id, event) of each data frame of an event stream."""
  frames = []
  for block in text.split('\n\n'):
    lines = dict(
        line.split(': ', 1) for line in block.splitlines() if ': ' in line
    )
    if 'data' in lines:
      frames.append((lines['id'], json.loads(lines['data'])))
  return frames


class ServerServiceTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):
  """Tests of the HTTP API, as the UI calls it, over fake runners.

  The tests run on asyncio, as uvicorn does.
  """

  def setUp(self):
    """A computer and two devices, and a built UI of two files."""
    super().setUp()
    self.world = world_lib.create(self)
    root = pathlib.Path(self.create_tempdir().full_path)
    self.ui = root / 'ui'
    self.ui.mkdir()
    (self.ui / 'index.html').write_text('<!doctype html><title>UI</title>')
    (self.ui / 'main.js').write_text('console.log(1);')
    (root / 'secret.txt').write_text('not served')

  async def asyncSetUp(self):
    """An HTTP client of the app, as the browser on this computer."""
    await super().asyncSetUp()
    world = self.world
    self.app = app = server_service.ServerService(
        world.workspace,
        world.devices,
        world.model_servers,
        world.catalog,
        world.listing,
        world.captures,
        world.exports,
        world.sessions,
        world.models,
        world.jobs,
        world.scans,
        world.prepared,
        self.ui,
    )
    self.client = await self.enterAsyncContext(
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=_ORIGIN,
            headers={'Origin': _ORIGIN},
        )
    )

  async def test_unknown_routes(self):
    """Routes with no caller answer 404 not_found."""
    for method, path in (
        ('GET', '/api/semantic?session_id=x'),
        ('POST', '/api/sessions/x/compile'),
        ('GET', '/api/kv-find'),
    ):
      response = await self.client.request(method, path, json={})
      self.assertEqual(response.status_code, 404)
      self.assertEqual(response.json(), {'error': 'not_found'})
    response = await self.client.get('/api/jobs/does-not-exist')
    self.assertEqual(response.status_code, 404)
    self.assertIsInstance(response.json()['error'], str)
    response = await self.client.get('/api/jobs/does-not-exist/events')
    self.assertEqual(response.status_code, 404)
    self.assertEqual(response.headers['content-type'], 'application/json')

  async def test_only_loopback_hosts_and_same_origin_writes(self):
    """Another Host, or a write from another origin, is refused."""
    response = await self.client.get(
        '/api/metrics', headers={'Host': 'evil.test'}
    )
    self.assertEqual(response.status_code, 403)
    self.assertEqual(response.json(), {'error': 'loopback_host_required'})
    response = await self.client.get(
        '/api/metrics', headers={'Host': 'localhost:8080'}
    )
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.headers['cache-control'], 'no-store')
    response = await self.client.post(
        '/api/sessions/create',
        json={},
        headers={'Origin': 'http://evil.test'},
    )
    self.assertEqual(response.status_code, 403)
    self.assertEqual(response.json(), {'error': 'origin_mismatch'})

  def test_the_access_log_keeps_changes_and_failures(self):
    """A read that succeeds has no line; a change or a failure has one."""
    logs = server_service.logs_access
    # The UI's polls and the reads of a page as it is shown.
    self.assertFalse(logs('GET', '/api/sessions', 200))
    self.assertFalse(logs('GET', '/api/devices', 200))
    self.assertFalse(logs('GET', '/api/jobs/x', 200))
    self.assertFalse(logs('GET', '/api/kv-analysis', 200))
    self.assertFalse(logs('GET', '/api/position-range', 200))
    self.assertFalse(logs('HEAD', '/', 200))
    self.assertFalse(logs('POST', '/api/token-analysis', 200))
    self.assertFalse(logs('POST', '/api/match-rate', 200))
    # Changes.
    self.assertTrue(logs('POST', '/api/sessions/x/turns', 200))
    self.assertTrue(logs('POST', '/api/sessions/x/close', 200))
    self.assertTrue(logs('DELETE', '/api/mappings/x', 200))
    # Failures, of reads too.
    self.assertTrue(logs('GET', '/api/jobs/x', 404))
    self.assertTrue(logs('POST', '/api/token-analysis', 400))
    self.assertTrue(logs('POST', '/api/sessions/x/turns', 409))
    self.assertTrue(logs('GET', '/api/metrics', 403))
    # The client gave up; a 500's traceback is logged where it failed.
    self.assertFalse(logs('POST', '/api/token-analysis', 499))
    self.assertFalse(logs('GET', '/api/sessions', 500))

  async def test_the_access_log_over_http(self):
    """Polls leave no line; a change is INFO, a failure WARNING.

    A request that fails unexpectedly is logged once, with its traceback.
    """
    with self.assertLogs(level='INFO') as logs:
      await self.client.get('/api/sessions')
      await self.client.get('/api/devices')
      await self.client.get('/api/metrics')
      created = await self.client.post(
          '/api/sessions/create', json=self.world.config()
      )
      await self.client.get('/api/jobs/does-not-exist')
      await self.client.get('/api/metrics', headers={'Host': 'evil.test'})
      with mock.patch.object(
          server_service.metric_catalog,
          'catalog',
          side_effect=RuntimeError('boom'),
      ):
        failed = await self.client.get('/api/metrics')
    self.assertEqual(created.status_code, 200)
    self.assertEqual(failed.status_code, 500)
    lines = [
        (record.levelname, record.getMessage().splitlines()[0])
        for record in logs.records
        if record.module == 'server_service'
    ]
    self.assertEqual(
        lines,
        [
            ('INFO', 'POST /api/sessions/create 200'),
            ('WARNING', 'GET /api/jobs/does-not-exist 404'),
            ('WARNING', 'GET /api/metrics 403'),
            ('ERROR', 'GET /api/metrics failed'),
        ],
    )
    (failure,) = [line for line in logs.output if 'failed' in line]
    self.assertIn('RuntimeError: boom', failure)

  async def test_bodies_must_be_small_json_objects(self):
    """Not JSON, not an object, or too large: refused with a message."""
    response = await self.client.post(
        '/api/sessions/create',
        content=b'name=x',
        headers={'Content-Type': 'application/x-www-form-urlencoded'},
    )
    self.assertEqual(response.status_code, 400)
    response = await self.client.post('/api/sessions/create', json=[1])
    self.assertEqual(response.status_code, 400)
    self.assertEqual(response.json(), {'error': 'Send a JSON object.'})
    response = await self.client.post(
        '/api/sessions/create', json={'name': 'x' * (64 * 1024)}
    )
    self.assertEqual(response.status_code, 413)

  async def test_listing_projects_each_record(self):
    """The listing has the template and each projection."""
    listing = (await self.client.get('/api/sessions')).json()
    self.assertEqual(
        set(listing),
        {'sessions', 'configuration_template', 'unavailable_reason'},
    )
    self.assertEqual(listing['sessions'], [])
    template = listing['configuration_template']
    self.assertEqual([run['id'] for run in template['runs']], ['ref', 'target'])
    self.assertEqual(template['runs'][0]['device'], 'local')
    config = self.world.config()
    # What the old runner took, which the editor carries over from the
    # record it starts from, is dropped.
    config['runs'][0]['repository'] = 'old'
    config['tap_profile'] = ''
    config['tap_points'] = {}
    record = await _created(self.client, config)
    self.assertNotIn('repository', record['runs'][0])
    self.assertEqual(
        [run['runnerBuild'] for run in record['runs']],
        [fake_launchers.COMMIT] * 2,
    )
    self.assertIs(record['has_capture'], False)
    self.assertEqual(record['tap_points'], {})
    (listed,) = (await self.client.get('/api/sessions')).json()['sessions']
    self.assertEqual(listed['id'], record['id'])
    self.assertEqual(listed['status_text'], 'Active')
    for gone in ('status', 'initialized', 'execution_ended', 'generation'):
      self.assertNotIn(gone, listed)
    self.assertEqual((listed['chats'], listed['live_chat']), ([], None))
    self.assertEqual(
        set(listed['actions']),
        {'enter', 'delete', 'new_chat', 'export'},
    )

  async def test_a_session_from_start_to_close(self):
    """Start, a turn over SSE, the capture, close."""
    record = await _started(self.client, self.world)
    (listed,) = (await self.client.get('/api/sessions')).json()['sessions']
    self.assertEqual(listed['execution']['phase'], 'active')
    self.assertEqual(listed['status_text'], 'Active')
    self.assertNotIn('newChatAllowed', listed['execution'])
    self.assertIs(listed['chats'][0]['actions']['send']['allowed'], True)
    self.assertEqual(
        listed['actions']['new_chat'], {'allowed': True, 'reason': ''}
    )

    response = await self.client.post(
        f'/api/sessions/{record["id"]}/turns',
        json={'prompt': 'What is 2 + 2?'},
    )
    self.assertEqual(response.status_code, 200, response.text)
    job = response.json()
    self.assertEqual(job['operation'], 'generate')
    self.assertEqual(job['session_id'], record['id'])
    self.assertEqual((job['chat'], job['turn']), (1, 1))
    stream = await self.client.get(f'/api/jobs/{job["id"]}/events?after=0')
    self.assertTrue(
        stream.headers['content-type'].startswith('text/event-stream')
    )
    frames = _frames(stream.text)
    self.assertEqual(
        [int(id) for id, _ in frames], list(range(1, len(frames) + 1))
    )
    self.assertTrue(all(event['jobId'] == job['id'] for _, event in frames))
    self.assertEqual(frames[-1][1]['type'], 'status')
    self.assertEqual(
        frames[-1][1]['status'], 'completed', frames[-1][1]['error']
    )
    text = ''.join(
        event['text']
        for _, event in frames
        if event['type'] == 'delta' and event['runId'] == 'ref'
    )
    self.assertEqual(text, fake_launchers.REPLY_TEXT)
    # Last-Event-ID wins over ?after.
    again = await self.client.get(
        f'/api/jobs/{job["id"]}/events?after=0',
        headers={'Last-Event-ID': str(len(frames) - 1)},
    )
    self.assertEqual([id for id, _ in _frames(again.text)], [str(len(frames))])
    job = (await self.client.get(f'/api/jobs/{job["id"]}')).json()
    self.assertEqual(job['output']['target'], fake_launchers.REPLY_TEXT)

    query = f'session_id={record["id"]}&chat=1'
    session = (await self.client.get(f'/api/session?{query}')).json()
    self.assertEqual([turn['n'] for turn in session['turns']], [1])
    self.assertEqual(
        {c['run'] for c in session['conversation']}, {'ref', 'target'}
    )
    (listed,) = (await self.client.get('/api/sessions')).json()['sessions']
    self.assertIs(listed['has_capture'], True)
    self.assertTrue(listed['actions']['enter']['allowed'])
    self.assertEqual(listed['job_id'], job['id'])
    kv = await self.client.get(f'/api/kv-analysis?mode=metadata&turn=1&{query}')
    self.assertEqual(kv.status_code, 200, kv.text)
    self.assertTrue(kv.json()['contexts'])
    row = {'turn': 1, 'stage': 'decode', 'alignment': 'content'}
    analysis = await self.client.post(
        f'/api/token-analysis?{query}', json={**row, 'rows': [0]}
    )
    self.assertEqual(analysis.status_code, 200, analysis.text)
    (pair,) = analysis.json()['pairs']
    self.assertEqual((pair['row'], pair['ref_step']), (0, 0))
    logits = await self.client.post(
        f'/api/final-logits?{query}',
        json={
            **row,
            'row': 0,
            'transform': 'raw',
            'offset': 0,
            'limit': 4,
            'order': 'index',
        },
    )
    self.assertEqual(logits.status_code, 200, logits.text)
    self.assertEqual(logits.json()['row'], 0)
    start = session['turns'][0]['stages']['decode']['positions']['ref']['start']
    binned = await self.client.post(
        f'/api/token-range?{query}',
        json={
            **row,
            'analysis': 'final_logits',
            'metric': 'kl',
            'start': start,
            'end': start + 1,
            'bins': 1,
        },
    )
    self.assertEqual(binned.status_code, 200, binned.text)
    self.assertEqual(binned.json()['rows'][0]['bins'][0]['valid_count'], 1)
    rate = await self.client.post(f'/api/match-rate?{query}', json=row)
    self.assertEqual(rate.status_code, 200, rate.text)
    self.assertEqual(rate.json()['match_rate'], 1.0)
    context = kv.json()['contexts'][0]['id']
    kv_range = (
        f'/api/kv-range?{query}&turn=1&context={context}&start=0&end=1'
        '&stage=prefill&alignment=content'
    )
    response = await self.client.get(kv_range)
    self.assertEqual(response.status_code, 200, response.text)
    self.assertEqual(response.json()['stage'], 'prefill')
    # The rows' positions come from the server's comparison, not the UI:
    # the old POST of them is a method the route does not take.
    for path in (kv_range, kv_range.replace('kv-range', 'kv-head')):
      response = await self.client.post(
          path, json={'alignment': {'start': 0, 'pairs': [[0, 0]]}}
      )
      self.assertEqual(response.status_code, 405, response.text)
      self.assertEqual(response.headers['allow'], 'GET, HEAD')
      self.assertEqual(response.json(), {'error': 'method_not_allowed'})
    response = await self.client.get(f'/api/match-rate?{query}')
    self.assertEqual(response.status_code, 405, response.text)
    self.assertEqual(response.headers['allow'], 'POST')
    # A Session that observes no tensors says so, where they would be.
    tensors = session['turns'][0]['stages']['decode']['analyses'][
        'intermediate_tensors'
    ]
    self.assertEqual(
        [r['code'] for r in tensors['reasons']], ['tensors_not_observed']
    )
    self.assertEqual(session['batches'], [])
    stage = f'{query}&turn=1&stage=decode&alignment=content'
    response = await self.client.get(f'/api/position-analysis?{stage}')
    self.assertEqual(response.status_code, 200, response.text)
    self.assertEqual(response.json()['rows'], [])
    self.assertEqual(
        response.json()['reasons'][0]['text'],
        'This Session observes no tensors.',
    )
    response = await self.client.get(
        f'/api/position-range?{stage}&start={start}&end={start + 1}'
    )
    self.assertEqual(response.status_code, 200, response.text)
    self.assertEqual(response.json()['rows'], [])
    response = await self.client.get(
        f'/api/position-tensor?{stage}&position={start}&layer=0&anchor=output'
    )
    self.assertEqual(response.status_code, 409, response.text)
    self.assertEqual(
        response.json(), {'error': 'This Session observes no tensors.'}
    )

    # Another chat: the first stays readable by its number.
    job = await _chat(
        self.client,
        record['id'],
        {'ref': {}, 'target': {'samplerType': 'topK', 'topK': 3}},
    )
    self.assertEqual((job['operation'], job['chat']), ('chat', 2))
    response = await self.client.get(f'/api/session?{query}')
    self.assertEqual(response.json()['chat'], 1)
    self.assertLen(response.json()['turns'], 1)
    second = (
        await self.client.get(f'/api/session?session_id={record["id"]}&chat=2')
    ).json()
    self.assertEqual((second['chat'], second['turns']), (2, []))
    (listed,) = (await self.client.get('/api/sessions')).json()['sessions']
    self.assertEqual(listed['live_chat'], 2)
    # A page that still shows Chat 1 sends none of its turns into Chat 2.
    response = await self.client.post(
        f'/api/sessions/{record["id"]}/turns',
        json={'prompt': 'Hi', 'chat': 1},
    )
    self.assertEqual(response.status_code, 409, response.text)
    self.assertEqual(response.json(), {'error': 'This chat has ended.'})
    # A chat the Session never had does not exist; it did not end.
    response = await self.client.post(
        f'/api/sessions/{record["id"]}/turns',
        json={'prompt': 'Hi', 'chat': 99},
    )
    self.assertEqual(response.status_code, 404, response.text)
    self.assertEqual(response.json(), {'error': 'Chat 99 does not exist.'})
    self.assertEqual(
        [chat['state'] for chat in listed['chats']], ['ended', 'live']
    )

    response = await self.client.post(
        f'/api/sessions/{record["id"]}/close', json={}
    )
    self.assertEqual(response.status_code, 200)
    closed = response.json()
    self.assertEqual(closed['execution']['phase'], 'ended')
    self.assertEqual(closed['status_text'], 'Ended')
    self.assertIsNone(closed['model_server']['button'])
    # The capture stays readable once the Model Server is off.
    self.assertEqual(
        (await self.client.get(f'/api/session?{query}')).status_code, 200
    )

  async def test_a_chats_settings_are_checked_as_it_starts(self):
    """Settings refused answer 400 and change nothing; chat-config is gone.

    A create takes none: a Session's settings are each chat's.
    """
    record = await _started(self.client, self.world, chat=False)
    path = f'/api/sessions/{record["id"]}/chats'
    for generation, error in (
        ({'ref': {}}, 'Reference and Target'),
        (
            {'ref': {'maxOutputTokens': 10**6}, 'target': {}},
            'at most the context length',
        ),
        (
            {'ref': {}, 'target': {'speculativeDecoding': 'on'}},
            'Speculative decoding',
        ),
    ):
      response = await self.client.post(path, json={'generation': generation})
      self.assertEqual(response.status_code, 400, response.text)
      self.assertIn(error, response.json()['error'])
    (listed,) = (await self.client.get('/api/sessions')).json()['sessions']
    self.assertEqual(listed['chats'], [])
    response = await self.client.post(
        '/api/sessions/chat-config',
        json={'id': record['id'], 'generation': {'ref': {}, 'target': {}}},
    )
    self.assertEqual(response.status_code, 404, response.text)
    chosen = {
        'ref': {'systemPrompt': 'Be terse.'},
        'target': {'samplerType': 'topK', 'topK': 40},
    }
    await _chat(self.client, record['id'], chosen)
    (listed,) = (await self.client.get('/api/sessions')).json()['sessions']
    self.assertEqual(
        listed['chats'][0]['generation']['ref']['systemPrompt'],
        'Be terse.',
    )
    response = await self.client.post(
        '/api/sessions/create',
        json={
            'name': 'Another',
            'model': listed['model'],
            'runs': listed['runs'],
            'generation': chosen,
        },
    )
    self.assertEqual(response.status_code, 400, response.text)
    self.assertIn('Unknown session settings', response.json()['error'])
    # Off, it starts no chat.
    await self.client.post(f'/api/sessions/{record["id"]}/close', json={})
    response = await self.client.post(path, json={'generation': chosen})
    self.assertEqual(response.status_code, 409, response.text)
    self.assertEqual(response.json(), {'error': 'This Session has ended.'})

  async def test_a_turn_the_index_rejects_shows_why(self):
    """The rejected turn is in /api/session; analyses of it say why."""
    original = self.world.launchers.launch

    def launch(device_id, build=None):
      launched = original(device_id, build)
      script = launched.runner._respond

      def respond(turn, prompt):
        chunks, completion, files = script(turn, prompt)
        completion.usage.completion_tokens += 1
        return chunks, completion, files

      launched.runner._respond = respond
      return launched

    self.world.launchers.launch = launch
    record = await _started(self.client, self.world)
    path = f'/api/sessions/{record["id"]}/turns'
    response = await self.client.post(path, json={'prompt': 'Hi'})
    self.assertEqual(response.status_code, 200, response.text)
    job = await _job(self.client, response.json())

    text = (
        'Reference: The capture index rejected turn 1: The runner counts '
        '5 completion tokens; turn 1 has 4 decode steps.'
    )
    self.assertEqual(job['status'], 'failed')
    self.assertTrue(
        job['error'].startswith('Turn 1 was rejected.'), job['error']
    )
    query = f'session_id={record["id"]}&chat=1'
    response = await self.client.get(f'/api/session?{query}')
    self.assertEqual(response.status_code, 200, response.text)
    session = response.json()
    (turn,) = session['turns']
    self.assertEqual(
        turn['rejected'],
        {'code': 'turn_rejected', 'side': 'ref', 'text': text},
    )
    # No analysis of it can be added, for the same text.
    for stage in turn['stages'].values():
      for answer in stage['analyses'].values():
        self.assertEqual(
            answer,
            {
                'available': False,
                'reasons': [{
                    'code': 'turn_rejected',
                    'side': None,
                    'text': text,
                }],
            },
        )
    self.assertEqual(session['conversation'], [])
    self.assertEqual(session['notice'], job['error'])
    for analysis in (
        self.client.get(f'/api/kv-analysis?mode=metadata&turn=1&{query}'),
        self.client.post(
            f'/api/token-analysis?{query}',
            json={
                'turn': 1,
                'stage': 'decode',
                'alignment': 'content',
                'rows': [0],
            },
        ),
    ):
      response = await analysis
      self.assertEqual(response.status_code, 409, response.text)
      self.assertEqual(response.json(), {'error': text})
    response = await self.client.post(path, json={'prompt': 'More'})
    self.assertEqual(response.status_code, 409, response.text)

  async def test_turn_requests_are_checked(self):
    """A turn needs a prompt and an active chat."""
    record = await _started(self.client, self.world, chat=False)
    path = f'/api/sessions/{record["id"]}/turns'
    response = await self.client.post(path, json={'prompt': ''})
    self.assertEqual(response.status_code, 400)
    response = await self.client.post(path, json={'prompt': 'Hi'})
    self.assertEqual(response.status_code, 409)
    self.assertEqual(response.json()['error'], 'Start a chat first.')

  async def test_cancel_of_an_initialize_is_refused(self):
    """Only a turn is cancelled; an ended job answers as it is."""
    record = await _started(self.client, self.world, chat=False)
    (job_id,) = [
        job_id
        for job_id, job in self.world.jobs._jobs.items()
        if job['session_id'] == record['id']
    ]
    response = await self.client.post(f'/api/jobs/{job_id}/cancel', json={})
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json()['status'], 'completed')

  async def test_a_models_tensors_are_scanned_for_the_editor(self):
    """POST starts the scan, GET follows it; each point says its graph."""
    path = self.world.workspace.hf / 'tap.litertlm'
    tap_fixtures.write_model(path)
    model = self.world.models.add_download(path)
    capabilities = (await self.client.get('/api/runtime/capabilities')).json()
    self.assertEqual(
        [profile['id'] for profile in capabilities['tap_profiles']],
        ['custom-outputs-v1'],
    )
    url = f'/api/artifacts/{model.artifact}/scan'
    idle = (await self.client.get(url)).json()
    self.assertEqual(idle['status'], 'idle')
    self.assertEqual(
        idle['copy_note'],
        'Starting the Session writes a copy of tap.litertlm that outputs '
        'the tensors observed, about 0.1 MiB, unless each is an output '
        'already. A Session keeps its copy until it is deleted, and '
        'Sessions that observe the same tensors share one.',
    )
    scan = (await self.client.post(url, json={})).json()
    for _ in range(500):
      if scan['status'] != 'scanning':
        break
      await asyncio.sleep(0.01)
      scan = (await self.client.get(url)).json()
    self.assertEqual(scan['status'], 'completed')
    self.assertEqual(scan['recommended'], [])
    points = {point['id']: point for point in scan['points']}
    self.assertLen(points, 16)
    self.assertEqual(
        points['decode:0:0'],
        {
            'id': 'decode:0:0',
            'signature': 'decode',
            'op': 0,
            'output': 0,
            'tensor': 2,
            'tensor_name': 'decode/layer_0/sum',
            'shape': [1, 1, 4],
            'dtype': 'float32',
            'operator': 'ADD',
            'selectable': True,
            'reason': '',
            'note': '',
        },
    )
    self.assertEqual(
        points['prefill_16:0:0']['note'],
        'prefill_16 runs only for long prompts; a short one is prefilled '
        'by prefill_4 alone, so this point may capture nothing.',
    )
    self.assertIn('speculative decoding', points['verify:0:0']['note'])
    self.assertEqual(
        points['decode:2:0']['note'],
        f"Its {tap_fixtures.KV_LENGTH} is the model's KV length; a capture "
        "holds it at the Session's context length.",
    )
    self.assertIn('as activations', points['decode:1:0']['note'])
    self.assertFalse(points['decode:3:0']['selectable'])
    self.assertEqual(
        [signature['runs'] for signature in scan['signatures']],
        ['always', 'always', 'long_prompts', 'speculative_decoding'],
    )
    response = await self.client.get('/api/artifacts/model-none/scan')
    self.assertEqual(response.status_code, 404)

  async def test_a_session_that_observes_tensors_is_prepared_as_it_starts(
      self,
  ):
    """Creating it writes the copy first, in its start's job."""
    path = self.world.workspace.hf / 'tap.litertlm'
    tap_fixtures.write_model(path)
    model = self.world.models.add_download(path)
    config = self.world.config()
    for run in config['runs']:
      run['artifact'] = model.artifact
    config['tap_profile'] = 'custom-outputs-v1'
    config['tap_points'] = {model.artifact: ['decode:9:0']}
    response = await self.client.post('/api/sessions/create', json=config)
    self.assertEqual(response.status_code, 400)
    self.assertEqual(
        response.json(),
        {'error': 'tap.litertlm: The model has no tensor output decode:9:0.'},
    )
    config['tap_points'] = {model.artifact: ['decode:0:0']}
    record = await _created(self.client, config)
    self.assertEqual(record['tap_points'], config['tap_points'])
    job = (await self.client.get(f'/api/jobs/{record["job_id"]}')).json()
    self.assertEqual(job['operation'], 'initialize')
    (listed,) = (await self.client.get('/api/sessions')).json()['sessions']
    self.assertEqual(
        listed['tap_preparation'],
        {'state': 'prepared', 'text': 'The tensor capture is prepared.'},
    )
    self.assertEqual(listed['job_id'], job['id'])
    copy = self.world.prepared.find(model, ['decode:0:0']).model
    for runner in self.world.launchers.runners('local'):
      self.assertEqual(runner.load_request.sha256, copy.sha256)
    url = f'/api/artifacts/{model.artifact}/scan'
    scan = await self.client.get(url)
    self.assertTrue(
        scan.json()['copy_note'].endswith(
            'Sessions that observe the same tensors share one.'
        )
    )

  async def test_a_sessions_copy_goes_as_it_goes(self):
    """Delete deletes the copy no other Session uses."""
    path = self.world.workspace.hf / 'tap.litertlm'
    tap_fixtures.write_model(path)
    model = self.world.models.add_download(path)
    config = self.world.config()
    for run in config['runs']:
      run['artifact'] = model.artifact
    config['tap_profile'] = 'custom-outputs-v1'
    config['tap_points'] = {model.artifact: ['decode:0:0']}
    sid = (await _created(self.client, config))['id']

    def copies() -> list[str]:
      prepared = self.world.workspace.prepared
      return sorted(p.name for p in prepared.iterdir())

    self.assertEqual(
        copies(), [prepared_models.key(model.sha256, ['decode:0:0'])]
    )
    await self.client.post(f'/api/sessions/{sid}/close', json={})
    response = await self.client.post('/api/sessions/delete', json={'id': sid})
    self.assertEqual(response.status_code, 200, response.text)
    self.assertEqual(copies(), [])

  async def test_a_session_is_deleted_for_good_once_its_server_is_off(self):
    """Refused while on; then its record and capture go."""
    record = await _started(self.client, self.world)
    sid = record['id']
    other = self.world.create()
    job = (
        await self.client.post(
            f'/api/sessions/{sid}/turns', json={'prompt': 'What is 2 + 2?'}
        )
    ).json()
    self.assertEqual((await _job(self.client, job))['status'], 'completed')
    space = self.world.workspace
    self.assertTrue((space.captures / sid / 'chat-1' / 'index.json').is_file())
    response = await self.client.post('/api/sessions/delete', json={'id': sid})
    self.assertEqual(response.status_code, 409)
    self.assertEqual(
        response.json(), {'error': 'Turn the Model Server off first.'}
    )
    self.assertTrue((space.records / f'{sid}.json').is_file())

    await self.client.post(f'/api/sessions/{sid}/close', json={})
    response = await self.client.post('/api/sessions/delete', json={'id': sid})
    self.assertEqual(response.status_code, 200, response.text)
    self.assertEqual(response.json()['id'], sid)
    listed = (await self.client.get('/api/sessions')).json()['sessions']
    self.assertEqual([r['id'] for r in listed], [other['id']])
    self.assertEqual(
        sorted(p.name for p in space.records.iterdir()),
        [f'{other["id"]}.json'],
    )
    self.assertFalse((space.captures / sid).exists())
    self.assertEqual(
        [p.name for p in space.captures.iterdir()],
        [other['id']] if (space.captures / other['id']).exists() else [],
    )
    response = await self.client.get(f'/api/session?session_id={sid}&chat=1')
    self.assertEqual(response.status_code, 404)
    response = await self.client.post('/api/sessions/delete', json={'id': sid})
    self.assertEqual(response.status_code, 404)
    # No undo.
    response = await self.client.post('/api/sessions/restore', json={'id': sid})
    self.assertIn(response.status_code, (404, 405))

  async def test_the_last_session_of_a_model_deleted_deletes_it(self):
    """Its file, record and pairs go; the dialog says so first."""
    response = await self.client.post(
        '/api/artifacts/upload',
        content=b'another model',
        headers={'X-File-Name': 'other.litertlm'},
    )
    self.assertEqual(response.status_code, 200, response.text)
    artifact = response.json()['artifact']
    model = self.world.models.get(artifact)
    config = self.world.config()
    for run in config['runs']:
      run['artifact'] = artifact
      run['source'] = 'upload'
    record = self.world.sessions.create(config)
    # A pair made by hand of it and the model another Session uses.
    pairs = self.world.workspace.mappings / mappings.file_name(
        model.sha256, self.world.model.sha256
    )
    pairs.parent.mkdir(exist_ok=True)
    json_file.write_json(pairs, {'format': mappings.FORMAT, 'pairs': []})
    listed = (await self.client.get('/api/sessions')).json()['sessions']
    notes = {r['id']: r['delete_note'] for r in listed}
    self.assertEqual(
        notes[record['id']],
        'It also deletes the model other.litertlm from the Server, '
        'freeing 0.0 MB.',
    )
    response = await self.client.post(
        '/api/sessions/delete', json={'id': record['id']}
    )
    self.assertEqual(response.status_code, 200, response.text)
    self.assertFalse(model.path.exists())
    with self.assertRaises(errors.NotFound):
      self.world.models.get(artifact)
    # Its tensor pairs go with it.
    self.assertFalse(pairs.exists())
    # The model another Session uses stays.
    self.assertTrue(self.world.model.present())

  async def test_capture_routes_need_a_capture(self):
    """No session_id or chat, an unknown record or chat: refused."""
    self.assertEqual((await self.client.get('/api/session')).status_code, 400)
    record = await _started(self.client, self.world, chat=False)
    for query, status in (
        ('session_id=nobody&chat=1', 404),
        (f'session_id={record["id"]}', 400),
        (f'session_id={record["id"]}&chat=x', 400),
        (f'session_id={record["id"]}&chat=1', 404),
    ):
      response = await self.client.get(f'/api/session?{query}')
      self.assertEqual(response.status_code, status, query)

  async def test_devices_and_capabilities(self):
    """Devices are listed and checked; capabilities sum them up."""
    devices = (await self.client.get('/api/devices')).json()['devices']
    self.assertEqual(
        [device['id'] for device in devices],
        [
            'local',
            'android:A',
            'android:B',
        ],
    )
    response = await self.client.post(
        '/api/devices/inspect', json={'id': 'local'}
    )
    self.assertEqual(response.json()['statusText'], 'Connected')
    response = await self.client.post(
        '/api/devices/inspect', json={'id': 'none'}
    )
    self.assertEqual(response.status_code, 404)
    # This computer's builds, read without launching one.
    device_builds = (
        await self.client.get('/api/devices/builds?id=local')
    ).json()
    self.assertEqual(device_builds['deviceId'], 'local')
    self.assertEqual(
        [build['id'] for build in device_builds['builds']],
        [fake_launchers.COMMIT, 'dev'],
    )
    for path, status in (
        ('/api/devices/builds?id=none', 404),
        ('/api/devices/builds', 400),
    ):
      response = await self.client.get(path)
      self.assertEqual(response.status_code, status, path)
    self.assertEqual(self.world.launchers.launched, [])
    capabilities = (await self.client.get('/api/runtime/capabilities')).json()
    self.assertIs(capabilities['available'], True)
    self.assertEqual(
        capabilities['upload_limit_bytes'], model_store.UPLOAD_LIMIT_BYTES
    )
    context = {
        option['key']: option
        for option in capabilities['runtimes'][0]['session_options']
    }['contextLength']
    self.assertEqual(context['default'], '4096')

  async def test_metrics_describes_every_metric(self):
    """GET /api/metrics is the catalog, the same for every caller."""
    response = await self.client.get('/api/metrics')

    self.assertEqual(response.status_code, 200)
    catalog_data = response.json()
    metrics = {metric['id']: metric for metric in catalog_data['metrics']}
    self.assertEqual(metrics['relative_l2']['unit'], 'percent')
    self.assertIn('kv', metrics['cosine_similarity']['analyses'])
    analyses = {
        analysis['id']: analysis for analysis in catalog_data['analyses']
    }
    self.assertEqual(analyses['kv']['default_head'], 'min')
    # It names no capture, and launches nothing.
    self.assertEqual(self.world.launchers.launched, [])

  async def test_chat_options_are_each_sides_own(self):
    """What each side's Chats may set: its reply bounded by its context."""
    record = self.world.create()
    response = await self.client.get(
        f'/api/sessions/chat-options?id={record["id"]}'
    )
    self.assertEqual(response.status_code, 200)
    sides = response.json()
    self.assertCountEqual(sides, ['ref', 'target'])
    for run in record['runs']:
      output = {
          option['key']: option for option in sides[run['id']]['chat_options']
      }['maxOutputTokens']
      self.assertEqual(output['maximum'], int(run['contextLength']))
    for path, status in (
        ('/api/sessions/chat-options?id=none', 404),
        ('/api/sessions/chat-options', 400),
    ):
      response = await self.client.get(path)
      self.assertEqual(response.status_code, status, path)
    self.assertEqual(self.world.launchers.launched, [])

  async def test_upload_stores_a_model(self):
    """The raw body becomes a model; the same bytes are one model."""
    data = b'uploaded model bytes' * 100
    name = urllib.parse.quote('my model.litertlm')
    response = await self.client.post(
        '/api/artifacts/upload',
        content=data,
        headers={
            'Content-Type': 'application/octet-stream',
            'X-File-Name': name,
        },
    )
    self.assertEqual(response.status_code, 200, response.text)
    uploaded = response.json()
    self.assertLen(data, uploaded['size'])
    self.assertEqual(uploaded['name'], 'my model.litertlm')
    self.assertLen(data, self.world.models.get(uploaded['artifact']).size)
    response = await self.client.post(
        '/api/artifacts/upload',
        content=data,
        headers={'Content-Type': 'application/octet-stream'},
    )
    self.assertEqual(response.status_code, 400)

  async def test_an_ended_session_is_exported_and_imported(self):
    """GET export streams one file; POST import makes it a Session."""
    record = await _started(self.client, self.world)
    sid = record['id']
    response = await self.client.get(f'/api/sessions/{sid}/export')
    self.assertEqual(response.status_code, 409)
    self.assertEqual(
        response.json(), {'error': 'This Session has no turn to export.'}
    )
    job = (
        await self.client.post(
            f'/api/sessions/{sid}/turns', json={'prompt': 'What is 2 + 2?'}
        )
    ).json()
    self.assertEqual((await _job(self.client, job))['status'], 'completed')
    response = await self.client.get(f'/api/sessions/{sid}/export')
    self.assertEqual(response.status_code, 409)
    self.assertEqual(
        response.json(), {'error': 'Turn the Model Server off first.'}
    )
    await self.client.post(f'/api/sessions/{sid}/close', json={})
    (listed,) = (await self.client.get('/api/sessions')).json()['sessions']
    self.assertEqual(
        listed['actions']['export'], {'allowed': True, 'reason': ''}
    )
    response = await self.client.get(f'/api/sessions/{sid}/export')
    self.assertEqual(response.status_code, 200, response.text)
    self.assertEqual(response.headers['content-type'], 'application/x-tar')
    self.assertEqual(
        response.headers['content-disposition'],
        'attachment; filename="CPU vs CPU.mdsession"; '
        "filename*=UTF-8''CPU%20vs%20CPU.mdsession",
    )
    self.assertLen(response.content, int(response.headers['content-length']))
    data = response.content
    # Sent whole: the Session is no longer exporting.
    (listed,) = (await self.client.get('/api/sessions')).json()['sessions']
    self.assertFalse(listed['exporting'])
    self.assertTrue(listed['actions']['delete']['allowed'])
    self.assertTrue(listed['actions']['export']['allowed'])

    response = await self.client.post(
        '/api/sessions/import',
        content=data,
        headers={'Content-Type': 'application/octet-stream'},
    )
    self.assertEqual(response.status_code, 200, response.text)
    imported = response.json()
    self.assertNotEqual(imported['id'], sid)
    # Its name is taken here by the Session it was exported from.
    self.assertEqual(imported['name'], record['name'] + ' (2)')
    self.assertEqual(imported['imported']['session_id'], sid)
    self.assertTrue(imported['chats'][0]['actions']['debug']['allowed'])
    self.assertFalse(imported['actions']['new_chat']['allowed'])
    self.assertIsNone(imported['model_server']['button'])
    row = {
        'turn': 1,
        'stage': 'decode',
        'alignment': 'content',
        'rows': [0],
    }
    answers = [
        (
            await self.client.post(
                f'/api/token-analysis?session_id={each}&chat=1', json=row
            )
        ).json()
        for each in (sid, imported['id'])
    ]
    self.assertEqual(answers[0], answers[1])
    response = await self.client.post(
        '/api/sessions/import',
        content=b'not an export' * 100,
        headers={'Content-Type': 'application/octet-stream'},
    )
    self.assertEqual(response.status_code, 400)
    self.assertIn('not a .mdsession file', response.json()['error'])
    response = await self.client.get('/api/sessions/gone/export')
    self.assertEqual(response.status_code, 404)

  async def test_a_session_being_exported_is_not_deleted(self):
    """A delete while its export is sent is refused, with the reason."""
    record = await _started(self.client, self.world)
    sid = record['id']
    job = (
        await self.client.post(
            f'/api/sessions/{sid}/turns', json={'prompt': 'Hi'}
        )
    ).json()
    self.assertEqual((await _job(self.client, job))['status'], 'completed')
    await self.client.post(f'/api/sessions/{sid}/close', json={})
    end = self.world.exports.begin(sid)
    response = await self.client.post('/api/sessions/delete', json={'id': sid})
    self.assertEqual(response.status_code, 409)
    self.assertEqual(response.json(), {'error': session_archive.EXPORTING})
    response = await self.client.get(f'/api/sessions/{sid}/export')
    self.assertEqual(response.status_code, 409)
    end()
    response = await self.client.post('/api/sessions/delete', json={'id': sid})
    self.assertEqual(response.status_code, 200, response.text)

  async def test_an_import_the_client_cancels_leaves_nothing(self):
    """One INFO line, no traceback; the part uploaded is removed."""
    messages = [
        {'type': 'http.request', 'body': b'x' * 1000, 'more_body': True},
        {'type': 'http.disconnect'},
    ]
    sent = []

    async def receive():
      if messages:
        return messages.pop(0)
      await asyncio.sleep(3600)

    async def send(message):
      sent.append(message)

    scope = {
        'type': 'http',
        'asgi': {'version': '3.0', 'spec_version': '2.3'},
        'http_version': '1.1',
        'method': 'POST',
        'scheme': 'http',
        'path': '/api/sessions/import',
        'raw_path': b'/api/sessions/import',
        'query_string': b'',
        'root_path': '',
        'headers': [
            (b'host', b'127.0.0.1:8080'),
            (b'content-type', b'application/octet-stream'),
            (b'content-length', b'100000'),
        ],
        'client': ('127.0.0.1', 1234),
        'server': ('127.0.0.1', 8080),
    }
    with self.assertLogs(level='INFO') as logs:
      await self.app(scope, receive, send)
    self.assertEqual(sent[0]['status'], 499)
    self.assertEqual(
        [record.getMessage() for record in logs.records],
        ['Import cancelled by the client'],
    )
    self.assertEqual(list(self.world.workspace.uploads.iterdir()), [])
    self.assertEqual(self.world.sessions.list(), [])

  async def test_static_ui(self):
    """The UI's files are served, and its index.html for its pages."""
    response = await self.client.get('/')
    self.assertEqual(response.status_code, 200)
    self.assertIn('<title>UI</title>', response.text)
    response = await self.client.get('/main.js')
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.text, 'console.log(1);')
    # Any other path is a page the UI names: it reads its address.
    for path in (
        '/sessions/s1',
        '/sessions/s1/chats/c1/debug/turns/2/decode/kv',
        '/sessions/s1/?debugPalette=teal',
        '/apis',
    ):
      response = await self.client.get(path)
      self.assertEqual(response.status_code, 200, path)
      self.assertIn('<title>UI</title>', response.text, path)
    # A file the UI lacks, whose name has a '.', is no page.
    for path in (
        '/missing.js',
        '/media/font.woff2',
        '/sessions/s1/main.js',
        '/%2E%2E/secret.txt',
    ):
      response = await self.client.get(path)
      self.assertEqual(response.status_code, 404, path)
      self.assertEqual(response.json(), {'error': 'not_found'}, path)
    # An /api path is never a page, nor is /api itself.
    for path in ('/api/sessions/s1/debug', '/api', '/api?x=1', '/api/'):
      response = await self.client.get(path)
      self.assertEqual(response.status_code, 404, path)
      self.assertEqual(response.json(), {'error': 'not_found'}, path)
    response = await self.client.post('/api', json={})
    self.assertEqual(response.status_code, 404)
    self.assertEqual(response.json(), {'error': 'not_found'})

  async def _a_turn(self) -> str:
    """A Session with one turn; its chat's ?session_id=&chat= query."""
    record = await _started(self.client, self.world)
    response = await self.client.post(
        f'/api/sessions/{record["id"]}/turns', json={'prompt': 'Hi'}
    )
    self.assertEqual(response.status_code, 200, response.text)
    job = await _job(self.client, response.json())
    self.assertEqual(job['status'], 'completed', job['error'])
    return f'session_id={record["id"]}&chat=1'

  async def test_other_routes_never_wait_behind_analyses(self):
    """With every analysis thread busy, GET /api/session still answers."""
    query = await self._a_turn()
    session = (await self.client.get(f'/api/session?{query}')).json()
    start = session['turns'][0]['comparison']['alignments']['content'][
        'positions'
    ]['start']
    release = threading.Event()
    real = token_analysis.token_analysis

    def blocked(index, n, steps, pieces=None):
      release.wait()
      return real(index, n, steps, pieces)

    row = {'turn': 1, 'stage': 'decode', 'alignment': 'content'}
    asked = 2 * analysis_requests.THREADS
    with mock.patch.object(
        token_analysis, 'token_analysis', side_effect=blocked
    ):
      analyses = []
      for k in range(asked):
        if k % 2:
          coro = self.client.post(
              f'/api/token-analysis?{query}',
              json={**row, 'rows': [k % 2]},
          )
        else:
          coro = self.client.post(
              f'/api/token-range?{query}',
              json={
                  **row,
                  'analysis': 'final_logits',
                  'metric': 'kl',
                  'start': start,
                  'end': start + 1,
                  'bins': 1,
              },
          )
        analyses.append(asyncio.create_task(coro))
      await asyncio.sleep(0.2)
      began = time.monotonic()
      response = await self.client.get(f'/api/session?{query}')
      waited = time.monotonic() - began
      self.assertFalse(any(task.done() for task in analyses))
      release.set()
      answers = await asyncio.gather(*analyses)

    self.assertEqual(response.status_code, 200, response.text)
    self.assertLess(waited, 2)
    self.assertEqual([a.status_code for a in answers], [200] * asked)

  async def test_an_analysis_whose_client_left_stops(self):
    """A request whose client is gone starts no more pairs: 499."""
    query = await self._a_turn()
    session = (await self.client.get(f'/api/session?{query}')).json()
    positions = session['turns'][0]['comparison']['alignments']['content'][
        'positions'
    ]
    body = json.dumps({
        'turn': 1,
        'stage': 'decode',
        'alignment': 'content',
        'analysis': 'final_logits',
        'metric': 'kl',
        'start': positions['start'],
        'end': positions['start'] + len(positions['pairs']),
        'bins': 1,
    }).encode()
    messages = [
        {'type': 'http.request', 'body': body, 'more_body': False},
        {'type': 'http.disconnect'},
    ]

    async def receive():
      return messages.pop(0) if len(messages) > 1 else messages[0]

    sent = []

    async def send(message):
      sent.append(message)

    scope = {
        'type': 'http',
        'asgi': {'version': '3.0'},
        'http_version': '1.1',
        'method': 'POST',
        'scheme': 'http',
        'path': '/api/token-range',
        'raw_path': b'/api/token-range',
        'query_string': query.encode(),
        'headers': [
            (b'host', b'127.0.0.1:8080'),
            (b'content-type', b'application/json'),
        ],
        'client': ('127.0.0.1', 1),
        'server': ('127.0.0.1', 8080),
    }
    made = []
    real = token_analysis.token_analysis

    def counted(index, n, steps, pieces=None):
      made.extend(steps)
      time.sleep(0.3)
      return real(index, n, steps, pieces)

    with mock.patch.object(
        token_analysis, 'token_analysis', side_effect=counted
    ):
      await self.app(scope, receive, send)

    self.assertEqual(sent[0]['status'], 499)


class GraphRoutesTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):
  """Tests of the graph routes, over a world that keeps graphs."""

  def setUp(self):
    """A world that keeps graphs, with the fixture model registered."""
    super().setUp()
    self.world = world_lib.create(self, graphs=True)
    path = self.world.workspace.hf / 'tap.litertlm'
    tap_fixtures.write_model(path)
    self.model = self.world.models.add_download(path)

  async def asyncSetUp(self):
    """An HTTP client of the app."""
    await super().asyncSetUp()
    world = self.world
    app = server_service.ServerService(
        world.workspace,
        world.devices,
        world.model_servers,
        world.catalog,
        world.listing,
        world.captures,
        world.exports,
        world.sessions,
        world.models,
        world.jobs,
        world.scans,
        world.prepared,
        None,
        world.graphs,
    )
    self.client = await self.enterAsyncContext(
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=_ORIGIN,
            headers={'Origin': _ORIGIN},
        )
    )

  def _initialize(self) -> dict[str, Any]:
    """Initializes a Session of the model: that converts its graph.

    Returns:
        The initialize job, ended.
    """
    config = self.world.config()
    for run in config['runs']:
      run['artifact'] = self.model.artifact
    record = self.world.sessions.create(config)
    return self.world.wait(self.world.model_servers.initialize(record['id']))

  async def test_the_editor_reads_the_anchors_a_model_captures(self):
    """Captured by default; not known until the graph is converted."""
    listing = (await self.client.get('/api/sessions')).json()
    self.assertEqual(
        listing['configuration_template']['tap_anchors'],
        {'ref': True, 'target': True},
    )
    url = f'/api/artifacts/{self.model.artifact}/anchors'
    pending = (await self.client.get(url)).json()
    self.assertEqual(pending['status'], 'pending')
    self.assertIn('first turns on', pending['text'])
    self.assertEqual((pending['points'], pending['copy_note']), (0, ''))

    self.assertEqual(self._initialize()['status'], 'completed')
    with mock.patch.object(
        self.world.graphs,
        'architecture',
        return_value=anchor_fixtures.architecture(),
    ):
      found = (await self.client.get(url)).json()

    self.assertEqual(found['status'], 'found')
    self.assertEqual(found['points'], 6)
    self.assertIn('writes a copy of tap.litertlm', found['copy_note'])
    self.assertEqual(
        [(s['name'], s['points']) for s in found['signatures']],
        [('decode', 3), ('prefill_16', 3), ('verify', 0)],
    )
    self.assertEqual(found['signatures'][0]['layers'], [0, 1])
    self.assertIn(
        'Layer 1 intermediate.2: Its type, string, is not a number.',
        found['signatures'][0]['skipped'],
    )
    none = (await self.client.get(url)).json()
    self.assertEqual(none['status'], 'none')
    self.assertIn('No architecture template exists', none['text'])
    scan = (
        await self.client.get(f'/api/artifacts/{self.model.artifact}/scan')
    ).json()
    self.assertEqual(scan['max_points'], 2048)
    response = await self.client.get(
        f'/api/artifacts/{self.world.model.artifact}x/anchors'
    )
    self.assertEqual(response.status_code, 404)

  async def test_the_editor_compares_its_two_models(self):
    """Its hint of two models, by the analyses' rules; 404 unknown."""
    artifact = self.model.artifact
    response = await self.client.get(
        f'/api/models/compare?ref={artifact}&target={artifact}'
    )
    self.assertEqual(response.status_code, 200, response.text)
    self.assertEqual(
        response.json(),
        {
            'code': 'same_file',
            'text': 'Same model file: every analysis compares.',
            'label': 'tap.litertlm',
            'capture_shared': model_comparison.SHARED_CAPTURE,
        },
    )
    response = await self.client.get(
        f'/api/models/compare?ref={artifact}&target=model-none'
    )
    self.assertEqual(response.status_code, 404, response.text)

  async def test_the_editor_lists_the_operators_inside_the_layers(self):
    """Not known until the graph is converted; 404 unknown."""
    artifact = self.model.artifact
    response = await self.client.get(
        f'/api/models/ops?ref={artifact}&target={artifact}&ops=attn/q_proj'
    )
    self.assertEqual(response.status_code, 200, response.text)
    found = response.json()
    self.assertEqual(found['status'], 'pending')
    self.assertIn("inside tap.litertlm's layers", found['text'])
    self.assertEqual(found['blocks'], [])
    response = await self.client.get(
        f'/api/models/ops?ref={artifact}&target=model-none'
    )
    self.assertEqual(response.status_code, 404, response.text)

  async def test_a_models_graph_its_signatures_and_architecture(self):
    """Idle, converted by a start, then each signature's graph."""
    base = f'/api/artifacts/{self.model.artifact}'
    response = await self.client.get(f'{base}/graph')
    self.assertEqual(response.status_code, 200, response.text)
    self.assertEqual(response.json(), {'status': 'idle'})
    response = await self.client.get(f'{base}/graph/decode')
    self.assertEqual(response.status_code, 409, response.text)
    self.assertIn('not ready', response.json()['error'])
    # Only a Session's initialize converts it: no request does.
    response = await self.client.post(f'{base}/graph', json={})
    self.assertEqual(response.status_code, 405, response.text)
    self.assertEqual(self.world.converter.calls, 0)
    job = await asyncio.to_thread(self._initialize)
    self.assertEqual(job['status'], 'completed', job['error'])
    found = (await self.client.get(f'{base}/graph')).json()
    self.assertEqual(found['status'], 'completed', found)
    self.assertEqual(
        [s['name'] for s in found['signatures']],
        list(tap_fixtures.SIGNATURES),
    )
    self.assertEqual(found['signatures'][0]['layers'], 1)
    self.assertEqual(found['signatures'][0]['layer_ids'], [0])
    self.assertEqual(
        found['layer_stems'],
        {name: 'layer' for name in tap_fixtures.SIGNATURES},
    )
    self.assertIn('No architecture template', found['architecture_text'])
    response = await self.client.get(f'{base}/graph/prefill_4')
    self.assertEqual(response.status_code, 200, response.text)
    collection = response.json()
    self.assertEqual(collection['label'], f'{self.model.name} · prefill_4')
    graph = collection['graphs'][0]
    self.assertEqual(graph['id'], 'prefill_4')
    self.assertEqual(
        graph['groupNodeAttributes']['']['runs'], 'runs in every chat'
    )
    outputs = {
        attr['key']: attr['value']
        for attr in graph['nodes'][0]['outputsMetadata'][0]['attrs']
    }
    self.assertEqual(outputs['point_id'], 'prefill_4:0:0')
    response = await self.client.get(f'{base}/graph/nothing')
    self.assertEqual(response.status_code, 404, response.text)
    response = await self.client.get(f'{base}/architecture')
    self.assertEqual(response.status_code, 200, response.text)
    self.assertEqual(response.json()['status'], 'completed')
    self.assertIsNone(response.json()['architecture'])
    response = await self.client.get('/api/artifacts/model-x/graph')
    self.assertEqual(response.status_code, 404, response.text)

  async def test_pairs_made_by_hand_are_kept_for_the_two_models(self):
    """Listed, saved, refused and deleted, each in the server's words."""
    path = self.world.workspace.hf / 'other.litertlm'
    tap_fixtures.write_model(path, between=b'another build')
    other = self.world.models.add_download(path)
    config = self.world.config()
    config['runs'][0]['artifact'] = self.model.artifact
    config['runs'][1]['artifact'] = other.artifact
    first = self.world.sessions.create(config)
    job = await asyncio.to_thread(
        lambda: self.world.wait(
            self.world.model_servers.initialize(first['id'])
        )
    )
    self.assertEqual(job['status'], 'completed', job['error'])
    point = 'prefill_4:0:0'
    record = self.world.sessions.create({
        **config,
        'tap_profile': session_store.TAP_PROFILE,
        'tap_points': {
            self.model.artifact: [point],
            other.artifact: [point],
        },
    })
    url = f'/api/mappings?session_id={record["id"]}'
    response = await self.client.get('/api/mappings')
    self.assertEqual(response.status_code, 400, response.text)
    listed = (await self.client.get(url)).json()
    self.assertEqual(
        listed,
        {'status': 'pairs', 'text': '', 'pairs': []},
    )
    tensor = {'signature': 'prefill_4', 'node': '0', 'output': '0'}

    response = await self.client.post(
        url, json={'ref': tensor, 'target': tensor, 'label': 'the same op'}
    )

    self.assertEqual(response.status_code, 200, response.text)
    (pair,) = response.json()['pairs']
    self.assertEqual(pair['ref']['point'], point)
    self.assertEqual(pair['label'], 'the same op')
    self.assertEqual(
        pair['place'], {'layer': 0, 'anchor': f'manual:{pair["id"]}'}
    )
    # The first Session, of the same two models, has it too: it observes
    # neither point, and says so.
    (shared,) = (
        await self.client.get(f'/api/mappings?session_id={first["id"]}')
    ).json()['pairs']
    self.assertEqual(shared['id'], pair['id'])
    self.assertIn('not observed in this Session', shared['text'])
    for payload, status, text in (
        ({'ref': tensor, 'target': tensor}, 409, 'paired already'),
        (
            {'ref': tensor, 'target': {**tensor, 'node': '404'}},
            400,
            'has no output 0 of node 404',
        ),
        ({'ref': tensor}, 400, 'Name the Target tensor'),
    ):
      response = await self.client.post(url, json=payload)
      self.assertEqual(response.status_code, status, response.text)
      self.assertIn(text, response.json()['error'])
    response = await self.client.delete(
        f'/api/mappings/{pair["id"]}?session_id={record["id"]}'
    )
    self.assertEqual(response.status_code, 200, response.text)
    self.assertEqual(response.json()['pairs'], [])
    response = await self.client.delete(
        f'/api/mappings/{pair["id"]}?session_id={record["id"]}'
    )
    self.assertEqual(response.status_code, 404, response.text)
    # One model on both sides needs none, and says so.
    config['runs'][1]['artifact'] = self.model.artifact
    same = self.world.sessions.create(config)
    url = f'/api/mappings?session_id={same["id"]}'
    listed = (await self.client.get(url)).json()
    self.assertEqual(listed['status'], 'identity')
    self.assertIn('same model', listed['text'])
    response = await self.client.post(
        url, json={'ref': tensor, 'target': tensor}
    )
    self.assertEqual(response.status_code, 400, response.text)
    self.assertIn('same model', response.json()['error'])

  async def test_a_pytorch_models_graph_is_the_one_a_turn_exported(self):
    """Unavailable with how to get it; then folded, bound, annotated."""
    directory = self.world.workspace.hf / 'gemma'
    directory.mkdir()
    (gemma4,) = [a for a in catalog.architectures() if a.id == 'gemma4']
    pinned = gemma4.entry('gemma-4-E2B-it')['pinned']['file']
    shutil.copyfile(gemma4.directory / pinned, directory / 'config.json')
    (directory / 'model.safetensors').write_bytes(b'weights')
    model = self.world.models.add_download_directory(directory)
    base = f'/api/artifacts/{model.artifact}'
    found = (await self.client.get(f'{base}/graph')).json()
    self.assertEqual(
        found,
        {'status': 'unavailable', 'text': model_graphs.NO_PYTORCH_GRAPH},
    )
    path = directory.parent / pytorch_graph_fixtures.FILE
    path.write_bytes(
        pytorch_graph_fixtures.gzipped(pytorch_graph_fixtures.document())
    )
    self.world.graphs.exported(
        model.artifact, path, {'session': 's1', 'turn': 1, 'side': 'ref'}
    )

    found = (await self.client.get(f'{base}/graph')).json()
    self.assertEqual(found['status'], 'completed', found)
    self.assertEqual(
        [s['name'] for s in found['signatures']], ['prefill', 'decode']
    )
    self.assertEqual(found['signatures'][1]['layer_ids'], list(range(35)))
    self.assertEqual(
        found['layer_stems'], {'prefill': 'layers/', 'decode': 'layers/'}
    )
    self.assertEqual(found['architecture_text'], '')
    collection = (await self.client.get(f'{base}/graph/decode')).json()
    graph = collection['graphs'][0]
    attributes = graph['groupNodeAttributes']['']
    self.assertIn('the decode pass of step 14, turn 1', attributes['exported'])
    self.assertIn('not expanded', attributes['folded'])
    self.assertIn('Gemma4RMSNorm', {node['label'] for node in graph['nodes']})
    answer = (await self.client.get(f'{base}/architecture')).json()
    layer = answer['bindings']['decode'][3]
    self.assertEqual(layer['scope'], 'layers/3')
    self.assertEqual(
        layer['anchors']['input']['point'],
        'decode:model.language_model.layers.3:in',
    )

  async def test_a_failed_graph_says_why(self):
    """Failed: the start fails; the graph says why."""
    self.world.converter.error = RuntimeError('The converter crashed.')
    job = await asyncio.to_thread(self._initialize)
    self.assertEqual(job['status'], 'failed')
    self.assertEqual(
        job['error'],
        f'The graph of {self.model.name} could not be converted, and the '
        'Session cannot start without it: The converter crashed.',
    )
    response = await self.client.get(
        f'/api/artifacts/{self.model.artifact}/graph'
    )
    self.assertEqual(
        response.json(),
        {'status': 'failed', 'error': 'The converter crashed.'},
    )


class ServerServiceLifeTest(absltest.TestCase):
  """The service from create() to close(): no runner outlives it."""

  def test_close_stops_every_runner_launched(self):
    """A runner launched outside any Session is stopped by close() too."""
    root = pathlib.Path(self.create_tempdir().full_path) / 'workspace'
    launchers = fake_launchers.FakeLaunchers([fake_launchers.PHONE])
    service = server_service.ServerService.create(root, launchers, [], None)
    self.assertEqual(launchers.records, root.resolve() / 'runners')
    launched = launchers.launch(fake_launchers.PHONE.id)
    service.close()
    self.assertTrue(launchers.stopped_all)
    self.assertFalse(launched.alive())

  def test_the_start_deletes_the_pairs_of_models_gone(self):
    """Pairs of a model the workspace no longer keeps go as it opens."""
    root = pathlib.Path(self.create_tempdir().full_path) / 'workspace'
    left = root / 'mappings' / mappings.file_name('a' * 64, 'b' * 64)
    left.parent.mkdir(parents=True)
    json_file.write_json(left, {'format': mappings.FORMAT, 'pairs': []})
    launchers = fake_launchers.FakeLaunchers([fake_launchers.PHONE])
    service = server_service.ServerService.create(root, launchers, None)
    self.addCleanup(service.close)
    self.assertFalse(left.exists())

  def test_a_create_that_fails_stops_every_runner(self):
    """A start that fails halfway launches nothing after it."""
    root = pathlib.Path(self.create_tempdir().full_path) / 'workspace'
    root.mkdir()
    (root / 'models.json').write_text('not JSON')
    launchers = fake_launchers.FakeLaunchers([fake_launchers.PHONE])
    with self.assertRaises(ValueError):
      server_service.ServerService.create(root, launchers, None)
    self.assertTrue(launchers.stopped_all)


if __name__ == '__main__':
  test_main.main()
