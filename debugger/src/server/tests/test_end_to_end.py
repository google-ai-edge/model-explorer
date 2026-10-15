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

"""The whole server, end to end: real runners, a real model, over HTTP.

The server runs as the user runs it, a process of its own, with the
command-line runner and a model registered by path. One Session puts both
sides on this computer's CPU with the same model, so every comparison of a
turn must come out exactly equal. Skipped unless the runner is built and
RUNNER_MODEL names a .litertlm file; RUNNER_BINARY names another runner.
"""

import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import uuid

import httpx
from model_debugger_server import main
import pytest

# RUNNER_BINARY, else the runner Bazel built last; None when neither is.
_RUNNER = (
    pathlib.Path(os.environ['RUNNER_BINARY'])
    if os.environ.get('RUNNER_BINARY')
    else main.built_runner()
)
_BUILT = _RUNNER is not None and _RUNNER.is_file()
_MODEL = pathlib.Path(os.environ.get('RUNNER_MODEL', ''))

pytestmark = pytest.mark.skipif(
    not (_BUILT and _MODEL.is_file()),
    reason='needs the built runner and RUNNER_MODEL',
)

_TERMINAL = ('completed', 'failed', 'cancelled')
# Metrics of two equal vectors are 0 up to float64 rounding.
_EXACT = 1e-9


class _Server:
  """The server process and a client of its API."""

  def __init__(self, workspace: pathlib.Path) -> None:
    with socket.socket() as probe:
      probe.bind(('127.0.0.1', 0))
      port = probe.getsockname()[1]
    self.log = workspace.with_suffix('.log')
    with open(self.log, 'w') as log:
      self.process = subprocess.Popen(
          [
              sys.executable,
              '-m',
              'model_debugger_server',
              '--workspace',
              str(workspace),
              '--port',
              str(port),
              '--runner',
              str(_RUNNER),
              '--model',
              str(_MODEL),
          ],
          stdout=log,
          stderr=subprocess.STDOUT,
      )
    self.http = httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=600)
    deadline = time.monotonic() + 300  # Hashing the model takes a while.
    try:
      while True:
        assert self.process.poll() is None, self.log.read_text()
        try:
          if self.http.get('/api/health').status_code == 200:
            return
        except httpx.TransportError:
          pass
        assert time.monotonic() < deadline, 'The server did not start.'
        time.sleep(0.5)
    except BaseException:
      self.stop()
      raise

  def get(self, path: str, **params) -> dict:
    response = self.http.get('/api/' + path, params=params)
    assert response.status_code == 200, response.text
    return response.json()

  def post(self, path: str, body: dict, **params) -> dict:
    response = self.http.post('/api/' + path, json=body, params=params)
    assert response.status_code == 200, response.text
    return response.json()

  def record(self, record_id: str) -> dict:
    """The record as the listing shows it."""
    sessions = self.get('sessions')['sessions']
    return next(record for record in sessions if record['id'] == record_id)

  def wait_phase(self, owner_id: str, phase: str) -> dict:
    deadline = time.monotonic() + 300
    while True:
      record = self.record(owner_id)
      if record.get('execution', {}).get('phase') == phase:
        return record
      assert time.monotonic() < deadline, record
      time.sleep(0.3)

  def follow(self, job: dict, on_delta=None) -> tuple[dict, list[dict]]:
    """Reads a job's events until it ends; returns the job and the events."""
    events, after = [], 0
    while True:
      with self.http.stream(
          'GET', f'/api/jobs/{job["id"]}/events', params={'after': after}
      ) as stream:
        assert stream.status_code == 200
        for line in stream.iter_lines():
          if not line.startswith('data: '):
            continue
          event = json.loads(line.removeprefix('data: '))
          events.append(event)
          after = event['sequence']
          if event['type'] == 'delta' and on_delta is not None:
            on_delta(event)
      job = self.get(f'jobs/{job["id"]}')
      if job['status'] in _TERMINAL:
        return job, events

  def turn(self, chat_id: str, prompt: str, on_delta=None):
    job = self.post(
        f'sessions/{chat_id}/turns',
        {'request_id': str(uuid.uuid4()), 'prompt': prompt},
    )
    return self.follow(job, on_delta)

  def runners(self) -> set[int]:
    """The runner processes the server runs now."""
    listed = subprocess.run(
        ['pgrep', '-P', str(self.process.pid)], capture_output=True, text=True
    ).stdout
    return {int(pid) for pid in listed.split()}

  def stop(self) -> None:
    """Sends the server SIGTERM and waits for it to exit."""
    self.http.close()
    self.process.terminate()
    try:
      self.process.wait(60)
    except subprocess.TimeoutExpired:
      self.process.kill()
      self.process.wait()


@pytest.fixture(scope='module')
def server(tmp_path_factory):
  """The server, for this module; once stopped, none of its runners runs."""
  server = _Server(tmp_path_factory.mktemp('end_to_end') / 'workspace')
  yield server
  pids = server.runners()
  server.stop()
  for pid in pids:
    with pytest.raises(ProcessLookupError):
      os.kill(pid, 0)


def _create(server: _Server) -> dict:
  model = server.get('runtime/capabilities')['models'][0]
  runs = [
      {
          'id': side,
          'runtime': 'LiteRT-LM',
          'artifact': model['artifact'],
          'source': 'registered',
          'device': 'local',
          'backend': 'CPU',
          'contextLength': '4096',
      }
      for side in ('ref', 'target')
  ]
  record = server.post(
      'sessions/create', {'name': 'CPU and CPU', 'model': '', 'runs': runs}
  )
  # Long enough for the reply that is cancelled; the short ones stop first.
  generation = {'maxOutputTokens': 256}
  generation = {'ref': generation, 'target': generation}
  server.post(
      'sessions/chat-config', {'id': record['id'], 'generation': generation}
  )
  return record


def _check_conversation(view: dict, n: int, job: dict) -> list[dict]:
  """Checks turn n of the session view; returns the ref side's tokens."""
  assert [turn['n'] for turn in view['turns']] == list(range(1, n + 1))
  turn = view['turns'][n - 1]
  sides = {
      entry['run']: entry
      for entry in view['conversation']
      if entry['turn'] == n
  }
  assert set(sides) == {'ref', 'target'}
  for side, entry in sides.items():
    assert entry['output'] == job['output'][side]
    assert entry['stop_reason'] == 'eos'
    tokens = entry['tokens']
    assert len(tokens) == turn['decode_tokens']
    assert all(type(token['source_forward_id']) is int for token in tokens)
    assert tokens[-1]['stop'] and not tokens[-1]['released']
    assert all(token['released'] for token in tokens[:-1])
  assert [t['id'] for t in sides['ref']['tokens']] == [
      t['id'] for t in sides['target']['tokens']
  ]
  return sides['ref']['tokens']


def _check_tokens(server: _Server, chat_id: str, n: int, count: int) -> None:
  """Every pair of a turn of two equal sides: equal, and greedy."""
  analysis = server.post(
      'token-analysis',
      {'turn': n, 'pairs': [{'ref': i, 'target': i} for i in range(count)]},
      session_id=chat_id,
  )
  assert analysis['turn'] == n
  assert [(p['ref_step'], p['target_step']) for p in analysis['pairs']] == [
      (i, i) for i in range(count)
  ]
  for pair in analysis['pairs']:
    distribution = pair['distribution']
    assert distribution['compatible'], distribution['reason']
    assert distribution['context'] == 'same'
    for key in ('kl', 'js', 'relative_l2', 'cosine_distance', 'max_abs_error'):
      assert abs(pair['metrics'][key]['value']) < _EXACT, key
    assert pair['metrics']['norm_ratio']['value'] == pytest.approx(1)
    for transform in ('raw', 'centered'):
      assert pair['final_logits'][transform]['rmse']['value'] == 0
    sampling = pair['sampling_distribution']
    for side in ('ref', 'target'):
      assert sampling[side]['basis'] == 'reconstructed_greedy'
      assert sampling[side]['selected_probability'] == 1.0
    assert sampling['rows_complete']


def _check_logits(server: _Server, chat_id: str) -> None:
  request = {
      'turn': 1,
      'ref_step': 0,
      'target_step': 0,
      'transform': 'raw',
      'offset': 0,
      'limit': 16,
      'order': 'absolute_delta',
      'direction': 'descending',
  }
  page = server.post('final-logits', request, session_id=chat_id)
  for key in ('turn', 'ref_step', 'target_step', 'transform', 'offset'):
    assert page[key] == request[key], key
  assert page['compatible']
  assert len(page['rows']) == 16
  assert all(row['delta'] == 0 for row in page['rows'])
  request = {key: request[key] for key in ('turn', 'ref_step', 'target_step')}
  request['transform'] = 'centered'
  vector = server.post('final-logits-vector', request, session_id=chat_id)
  for key, value in request.items():
    assert vector[key] == value, key
  assert vector['complete'] is True
  assert len(vector['values']['ref']) == vector['total'] == page['total']
  assert vector['values']['ref'] == vector['values']['target']


def _check_kv(server: _Server, chat_id: str) -> None:
  contexts = {
      context['moment']: context
      for context in server.get(
          'kv-analysis', mode='metadata', turn=1, session_id=chat_id
      )['contexts']
  }
  assert {'prefill_post', 'terminal'} <= set(contexts)
  for moment in ('prefill_post', 'terminal'):
    rows = contexts[moment]['layers']
    assert rows and all(row['status'] == 'ok' for row in rows), moment
  terminal = contexts['terminal']
  end = terminal['processed_token_count']
  assert end > contexts['prefill_post']['processed_token_count']

  query = {'turn': 1, 'context': terminal['id'], 'start': 0, 'end': end}
  for kind in ('key', 'value'):
    chart = server.get(
        'kv-range', **query, bins=8, kind=kind, head='max', session_id=chat_id
    )
    assert (chart['context_id'], chart['start'], chart['end']) == (
        terminal['id'],
        0,
        end,
    )
    assert chart['rows'] and all(row['status'] == 'ok' for row in chart['rows'])
    for row in chart['rows']:
      for interval in row['bins']:
        metrics = interval['metrics']
        assert metrics['max_abs']['max'] == 0
        assert metrics['relative_l2']['max'] == 0
  selection = server.get('kv-selection', **query, session_id=chat_id)
  assert (selection['context_id'], selection['start'], selection['end']) == (
      terminal['id'],
      0,
      end,
  )
  for row in selection['rows']:
    assert row['status'] == 'ok', row
    assert row['metrics']['max_abs']['value'] == 0
    assert row['metrics']['cosine_similarity']['value'] == 1

  head = server.get(
      'kv-head',
      turn=1,
      context_id=terminal['id'],
      layer=0,
      kind='value',
      position=end - 1,
      head=0,
      session_id=chat_id,
  )
  selected = head['selection']
  assert (selected['turn'], selected['context_id'], selected['layer']) == (
      1,
      terminal['id'],
      0,
  )
  assert (selected['kind'], selected['position'], selected['head']) == (
      'value',
      end - 1,
      0,
  )
  assert head['status'] == 'ok'
  assert head['metrics']['max_abs'] == 0
  assert all(channel['delta'] == 0 for channel in head['channels'])


def test_a_session_of_two_cpu_runners_from_start_to_close(server):
  """Turn on, two turns, analyses, a cancel, a new chat, turn off."""
  session_id = _create(server)['id']
  server.post(
      f'sessions/{session_id}/initialize', {'request_id': str(uuid.uuid4())}
  )
  record = server.wait_phase(session_id, 'active')
  assert record['status'] == 'ready'
  assert len(server.runners()) == 2

  # Turn 1, its view, and every analysis of it.
  job, events = server.turn(session_id, 'What is 2 + 2?')
  assert job['status'] == 'completed', job['error']
  assert job['turn'] == 1
  for side in ('ref', 'target'):
    deltas = [
        e['text'] for e in events if e['type'] == 'delta' and e['runId'] == side
    ]
    assert ''.join(deltas) == job['output'][side] != ''
  assert job['output']['ref'] == job['output']['target']
  view = server.get('session', session_id=session_id)
  tokens = _check_conversation(view, 1, job)
  assert server.record(session_id)['has_capture']
  _check_tokens(server, session_id, 1, len(tokens))
  _check_logits(server, session_id)
  _check_kv(server, session_id)

  # Turn 2 follows from turn 1's context.
  job, _ = server.turn(session_id, 'And 3 + 3?')
  assert job['status'] == 'completed', job['error']
  view = server.get('session', session_id=session_id)
  tokens = _check_conversation(view, 2, job)
  _check_tokens(server, session_id, 2, len(tokens))

  # Cancelling a reply ends the chat but keeps its turns.
  cancelled = []

  def cancel(event):
    if not cancelled:
      cancelled.append(server.post(f'jobs/{event["jobId"]}/cancel', {}))

  job, _ = server.turn(
      session_id, 'Write a long story about a lighthouse keeper.', cancel
  )
  assert cancelled and job['status'] == 'cancelled'
  record = server.record(session_id)
  assert record['chat_over'] and record['notice']
  assert record['execution']['newChatAllowed']
  assert len(server.get('session', session_id=session_id)['turns']) == 2

  # A new chat runs on fresh runners.
  before = server.runners()
  chat = server.post('sessions/chat', {'id': session_id, 'name': 'New chat 1'})
  assert chat['parent_session_id'] == session_id
  after = server.runners()
  assert len(after) == 2 and not after & before
  job, _ = server.turn(chat['id'], 'What is 2 + 2?')
  assert job['status'] == 'completed', job['error']
  view = server.get('session', session_id=chat['id'])
  _check_conversation(view, 1, job)
  assert server.record(chat['id'])['status'] == 'saved'

  # Turning the Model Server off stops both runners for good.
  server.post(f'sessions/{session_id}/close', {})
  record = server.wait_phase(session_id, 'ended')
  assert record['execution_ended']
  assert server.runners() == set()


def test_terminating_the_server_stops_the_runners_it_runs(tmp_path):
  """SIGTERM while a Model Server is on: no runner outlives the server."""
  server = _Server(tmp_path / 'workspace')
  pids = set()
  try:
    session_id = _create(server)['id']
    server.post(
        f'sessions/{session_id}/initialize', {'request_id': str(uuid.uuid4())}
    )
    server.wait_phase(session_id, 'active')
    pids = server.runners()
    assert len(pids) == 2
  finally:
    server.stop()  # SIGTERM
  for pid in pids:
    with pytest.raises(ProcessLookupError):
      os.kill(pid, 0)
