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
import urllib.parse

import fake_runner
import httpx
from model_debugger_server import http_app, jobs, models
import pytest

_ORIGIN = 'http://127.0.0.1:8080'


@pytest.fixture
def anyio_backend():
  """Runs the async tests on asyncio, as uvicorn does."""
  return 'asyncio'


@pytest.fixture
def world(tmp_path):
  """The stores and fake devices of fake_runner.World."""
  made = fake_runner.World(
      tmp_path,
      [fake_runner.LOCAL, fake_runner.PHONE, fake_runner.TABLET],
  )
  yield made
  made.close()


@pytest.fixture
def ui(tmp_path):
  """A built UI of two files."""
  root = tmp_path / 'ui'
  root.mkdir()
  (root / 'index.html').write_text('<!doctype html><title>UI</title>')
  (root / 'main.js').write_text('console.log(1);')
  (tmp_path / 'secret.txt').write_text('not served')
  return root


@pytest.fixture
async def client(world, ui):
  """An HTTP client of the app, as the browser on this computer."""
  app = http_app.create_app(
      world.executions, world.sessions, world.models, world.jobs, ui
  )
  transport = httpx.ASGITransport(app=app)
  async with httpx.AsyncClient(
      transport=transport, base_url=_ORIGIN, headers={'Origin': _ORIGIN}
  ) as made:
    yield made


async def _job(client: httpx.AsyncClient, job: dict) -> dict:
  """Polls a job over HTTP until it ended."""
  for _ in range(2000):
    job = (await client.get(f'/api/jobs/{job["id"]}')).json()
    if job['status'] in jobs.TERMINAL:
      return job
    await asyncio.sleep(0.01)
  raise AssertionError(f'job did not end: {job}')


async def _started(client, world) -> dict:
  """Creates a Session over HTTP and turns its Model Server on."""
  created = await client.post('/api/sessions/create', json=world.config())
  assert created.status_code == 200, created.text
  record = created.json()
  response = await client.post(
      f'/api/sessions/{record["id"]}/initialize', json={'request_id': 'on'}
  )
  assert response.status_code == 200, response.text
  job = await _job(client, response.json())
  assert job['status'] == 'completed', job['error']
  return record


def _frames(text: str) -> list[tuple[str, dict]]:
  """The (id, event) of each data frame of an event stream."""
  frames = []
  for block in text.split('\n\n'):
    lines = dict(
        line.split(': ', 1) for line in block.splitlines() if ': ' in line
    )
    if 'data' in lines:
      frames.append((lines['id'], json.loads(lines['data'])))
  return frames


@pytest.mark.anyio
async def test_health_and_unknown_routes(client):
  """Health answers; routes with no caller answer 404 not_found."""
  assert (await client.get('/api/health')).json() == {'status': 'ok'}
  for method, path in (
      ('GET', '/api/semantic?session_id=x'),
      ('GET', '/api/devices/builds?id=local'),
      ('POST', '/api/sessions/x/prepare'),
      ('GET', '/api/kv-find'),
  ):
    response = await client.request(method, path, json={})
    assert response.status_code == 404
    assert response.json() == {'error': 'not_found'}
  response = await client.get('/api/jobs/does-not-exist')
  assert response.status_code == 404
  assert isinstance(response.json()['error'], str)
  response = await client.get('/api/jobs/does-not-exist/events')
  assert response.status_code == 404
  assert response.headers['content-type'] == 'application/json'


@pytest.mark.anyio
async def test_only_loopback_hosts_and_same_origin_writes(client):
  """Another Host, or a write from another origin, is refused."""
  response = await client.get('/api/health', headers={'Host': 'evil.test'})
  assert response.status_code == 403
  assert response.json() == {'error': 'loopback_host_required'}
  response = await client.get('/api/health', headers={'Host': 'localhost:8080'})
  assert response.status_code == 200
  assert response.headers['cache-control'] == 'no-store'
  response = await client.post(
      '/api/sessions/create', json={}, headers={'Origin': 'http://evil.test'}
  )
  assert response.status_code == 403
  assert response.json() == {'error': 'origin_mismatch'}
  response = await client.post(
      '/api/sessions/create', json={}, headers={'Origin': 'null'}
  )
  assert response.status_code == 403
  assert response.json() == {'error': 'origin_mismatch'}
  response = await client.get(
      '/api/sessions', headers={'Sec-Fetch-Site': 'cross-site'}
  )
  assert response.status_code == 403
  assert response.json() == {'error': 'origin_mismatch'}


@pytest.mark.anyio
async def test_bodies_must_be_small_json_objects(client):
  """Not JSON, not an object, or too large: refused with a message."""
  response = await client.post(
      '/api/sessions/create',
      content=b'name=x',
      headers={'Content-Type': 'application/x-www-form-urlencoded'},
  )
  assert response.status_code == 400
  response = await client.post('/api/sessions/create', json=[1])
  assert response.status_code == 400
  assert response.json() == {'error': 'Send a JSON object.'}
  response = await client.post(
      '/api/sessions/create', json={'name': 'x' * (64 * 1024)}
  )
  assert response.status_code == 413


@pytest.mark.anyio
async def test_listing_projects_each_record(client, world):
  """The listing carries the template, capabilities and each projection."""
  listing = (await client.get('/api/sessions')).json()
  assert listing['sessions'] == []
  assert listing['capabilities'] == {
      'create': True,
      'rename': True,
      'duplicate': True,
      'delete': True,
  }
  assert listing['generation_available'] is True
  template = listing['configuration_template']
  assert [run['id'] for run in template['runs']] == ['ref', 'target']
  assert template['runs'][0]['device'] == 'local'
  config = world.config()
  # A duplicate carries what the old runner took; it is dropped.
  config['runs'][0]['runnerBuild'] = 'old'
  config['tap_profile'] = ''
  config['tap_points'] = {}
  record = (await client.post('/api/sessions/create', json=config)).json()
  assert 'runnerBuild' not in record['runs'][0]
  assert record['status'] == 'draft'
  assert record['has_capture'] is False
  assert record['tap_points'] == {}
  (listed,) = (await client.get('/api/sessions')).json()['sessions']
  assert listed['id'] == record['id']
  assert 'execution' not in listed
  assert listed['initialized'] is False


@pytest.mark.anyio
async def test_a_session_from_start_to_close(client, world):
  """Start, a turn over SSE, the capture, a new chat, close."""
  record = await _started(client, world)
  (listed,) = (await client.get('/api/sessions')).json()['sessions']
  assert listed['execution']['phase'] == 'active'
  assert listed['status'] == 'ready'
  assert listed['initialized'] is True
  # The configuration is fixed while the Model Server is on.
  response = await client.post(
      '/api/sessions/update', json={'id': record['id'], **world.config()}
  )
  assert response.status_code == 409

  response = await client.post(
      f'/api/sessions/{record["id"]}/turns',
      json={
          'request_id': 'turn',
          'prompt': 'What is 2 + 2?',
      },
  )
  assert response.status_code == 200, response.text
  job = response.json()
  assert job['operation'] == 'generate'
  assert job['session_id'] == record['id']
  assert job['turn'] == 1
  stream = await client.get(f'/api/jobs/{job["id"]}/events?after=0')
  assert stream.headers['content-type'].startswith('text/event-stream')
  frames = _frames(stream.text)
  assert [int(id) for id, _ in frames] == list(range(1, len(frames) + 1))
  assert all(event['jobId'] == job['id'] for _, event in frames)
  assert frames[-1][1]['type'] == 'status'
  assert frames[-1][1]['status'] == 'completed', frames[-1][1]['error']
  text = ''.join(
      event['text']
      for _, event in frames
      if event['type'] == 'delta' and event['runId'] == 'ref'
  )
  assert text == fake_runner.REPLY_TEXT
  # Last-Event-ID wins over ?after.
  again = await client.get(
      f'/api/jobs/{job["id"]}/events?after=0',
      headers={'Last-Event-ID': str(len(frames) - 1)},
  )
  assert [id for id, _ in _frames(again.text)] == [str(len(frames))]
  job = (await client.get(f'/api/jobs/{job["id"]}')).json()
  assert job['output']['target'] == fake_runner.REPLY_TEXT

  query = f'session_id={record["id"]}'
  session = (await client.get(f'/api/session?{query}')).json()
  assert [turn['n'] for turn in session['turns']] == [1]
  assert {c['run'] for c in session['conversation']} == {'ref', 'target'}
  (listed,) = (await client.get('/api/sessions')).json()['sessions']
  assert listed['has_capture'] is True
  assert listed['status'] == 'saved'
  assert listed['job_id'] == job['id']
  kv = await client.get(f'/api/kv-analysis?mode=metadata&turn=1&{query}')
  assert kv.status_code == 200, kv.text
  assert kv.json()['contexts']
  analysis = await client.post(
      f'/api/token-analysis?{query}',
      json={'turn': 1, 'pairs': [{'ref': 0, 'target': 0}]},
  )
  assert analysis.status_code == 200, analysis.text

  response = await client.post(
      '/api/sessions/chat', json={'id': record['id'], 'name': 'New chat 1'}
  )
  assert response.status_code == 200, response.text
  chat = response.json()
  assert chat['parent_session_id'] == record['id']
  assert chat['initialized'] is True
  listing = (await client.get('/api/sessions')).json()['sessions']
  assert {r['id'] for r in listing} == {record['id'], chat['id']}

  response = await client.post(f'/api/sessions/{record["id"]}/close', json={})
  assert response.status_code == 200
  closed = response.json()
  assert closed['execution']['phase'] == 'ended'
  assert closed['execution_ended'] is True
  # The capture stays readable once the Model Server is off.
  assert (await client.get(f'/api/session?{query}')).status_code == 200
  response = await client.post(
      f'/api/sessions/{record["id"]}/initialize', json={'request_id': 'again'}
  )
  assert response.status_code == 409
  assert 'duplicate' in response.json()['error']


@pytest.mark.anyio
async def test_turn_requests_are_checked(client, world):
  """A turn needs a request id, a prompt, and an active chat."""
  record = (
      await client.post('/api/sessions/create', json=world.config())
  ).json()
  path = f'/api/sessions/{record["id"]}/turns'
  response = await client.post(path, json={'prompt': 'Hi'})
  assert response.status_code == 400
  assert 'request_id' in response.json()['error']
  response = await client.post(path, json={'request_id': 'a', 'prompt': ''})
  assert response.status_code == 400
  response = await client.post(path, json={'request_id': 'b', 'prompt': 'Hi'})
  assert response.status_code == 409
  assert response.json()['error'] == 'Turn the Model Server on first.'


@pytest.mark.anyio
async def test_cancel_of_an_initialize_is_refused(client, world):
  """Only a turn is cancelled; an ended job answers as it is."""
  record = await _started(client, world)
  (job_id,) = [
      job_id
      for job_id, job in world.jobs._jobs.items()
      if job['session_id'] == record['id']
  ]
  response = await client.post(f'/api/jobs/{job_id}/cancel', json={})
  assert response.status_code == 200
  assert response.json()['status'] == 'completed'


@pytest.mark.anyio
async def test_capture_routes_need_a_capture(client, world):
  """No session_id, an unknown record, or no capture: refused."""
  assert (await client.get('/api/session')).status_code == 400
  response = await client.get('/api/session?session_id=nobody')
  assert response.status_code == 404
  record = (
      await client.post('/api/sessions/create', json=world.config())
  ).json()
  response = await client.get(f'/api/session?session_id={record["id"]}')
  assert response.status_code == 404


@pytest.mark.anyio
async def test_devices_and_capabilities(client, world):
  """Devices are listed, inspected by a runner, summed up in capabilities."""
  devices = (await client.get('/api/devices')).json()['devices']
  assert [device['id'] for device in devices] == [
      'local',
      'android:A',
      'android:B',
  ]
  response = await client.post('/api/devices/inspect', json={'id': 'local'})
  assert response.json()['runnerState'] == 'idle'
  response = await client.post('/api/devices/inspect', json={'id': 'none'})
  assert response.status_code == 404
  capabilities = (await client.get('/api/runtime/capabilities')).json()
  assert capabilities['available'] is True
  assert capabilities['upload_limit_bytes'] == models.UPLOAD_LIMIT_BYTES
  assert capabilities['upload_idle_timeout_seconds'] == 60
  assert capabilities['runtimes'][0]['max_output_tokens'] == 256


@pytest.mark.anyio
async def test_upload_stores_a_model(client, world):
  """The raw body becomes a model; the same bytes are one model."""
  data = b'uploaded model bytes' * 100
  name = urllib.parse.quote('my model.litertlm')
  response = await client.post(
      '/api/artifacts/upload',
      content=data,
      headers={
          'Content-Type': 'application/octet-stream',
          'X-File-Name': name,
      },
  )
  assert response.status_code == 200, response.text
  uploaded = response.json()
  assert uploaded['size'] == len(data)
  assert uploaded['name'] == 'my model.litertlm'
  assert world.models.get(uploaded['artifact']).size == len(data)
  response = await client.post(
      '/api/artifacts/upload',
      content=data,
      headers={'Content-Type': 'application/octet-stream'},
  )
  assert response.status_code == 400


@pytest.mark.anyio
async def test_static_ui(client):
  """The UI's files are served; nothing outside them is."""
  response = await client.get('/')
  assert response.status_code == 200
  assert '<title>UI</title>' in response.text
  assert (await client.get('/main.js')).status_code == 200
  assert (await client.get('/missing.js')).status_code == 404
  response = await client.get('/%2E%2E/secret.txt')
  assert response.status_code == 404
