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
command-line runner, and the model uploaded to it as the browser uploads
one. One Session puts both
sides on this computer's CPU with the same model, so every comparison of a
turn must come out exactly equal; another puts Target on its GPU, where the
index must still account for every token of both sides. Skipped unless the
runner is built and RUNNER_MODEL names a .litertlm file; RUNNER_BINARY names
another runner.
"""

import json
import math
import os
import pathlib
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

from absl import logging
from absl.testing import absltest
import httpx
from model_debugger.server.testing import real_runner
from model_debugger.server.testing import test_main

_RUNNER = real_runner.RUNNER
_BUILT = real_runner.BUILT
_MODEL = pathlib.Path(os.environ.get('RUNNER_MODEL', ''))

_TERMINAL = ('completed', 'failed', 'cancelled')
# Metrics of two equal vectors are 0 up to float64 rounding.
_EXACT = 1e-9


def _server_env() -> dict[str, str]:
  """The server's environment: it sees this computer, and no other device.

  No xcrun (in /usr/bin) or adb on its PATH, no Android SDK named, and a
  HOME of the test's, which holds none, to find a simulator or a phone by.

  Returns:
    A dictionary mapping environment variable names to their values.
  """
  env = {
      key: value
      for key, value in os.environ.items()
      if key not in ('PATH', 'ANDROID_HOME')
  }
  env.update(PATH='/bin', HOME=absltest.TEST_TMPDIR.value)
  return env


class _Server:
  """The server process and a client of its API, with the model uploaded."""

  def __init__(self, workspace: pathlib.Path) -> None:
    with socket.socket() as probe:
      probe.bind(('127.0.0.1', 0))
      port = probe.getsockname()[1]
    self.workspace = workspace
    self.log = workspace.with_suffix('.log')
    with open(self.log, 'w') as log:
      self.process = subprocess.Popen(
          [
              sys.executable,
              '-m',
              'model_debugger.server.cli.server_main',
              '--workspace',
              str(workspace),
              '--port',
              str(port),
              '--runner',
              str(_RUNNER),
              # The checkout's PyTorch runner is not this test's.
              '--pytorch_runner',
              '',
          ],
          stdout=log,
          stderr=subprocess.STDOUT,
          env=_server_env(),
      )
    self.http = httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=600)
    deadline = time.monotonic() + 300
    while True:
      assert self.process.poll() is None, self.log.read_text()
      try:
        if self.http.get('/api/metrics').status_code == 200:
          break
      except httpx.TransportError:
        pass
      assert time.monotonic() < deadline, 'The server did not start.'
      time.sleep(0.5)
    self.upload(_MODEL)

  def upload(self, path: pathlib.Path) -> dict[str, Any]:
    """Uploads a model file as the browser does; returns the model."""

    def chunks():
      with open(path, 'rb') as file:
        while chunk := file.read(1 << 22):
          yield chunk

    response = self.http.post(
        '/api/artifacts/upload',
        content=chunks(),
        headers={
            'Content-Length': str(path.stat().st_size),
            'X-File-Name': path.name,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()

  def get(self, path: str, **params) -> dict[str, Any]:
    response = self.http.get('/api/' + path, params=params)
    assert response.status_code == 200, response.text
    return response.json()

  def post(self, path: str, body: dict[str, Any], **params) -> dict[str, Any]:
    response = self.http.post('/api/' + path, json=body, params=params)
    assert response.status_code == 200, response.text
    return response.json()

  def record(self, record_id: str) -> dict[str, Any]:
    """The record as the listing shows it."""
    sessions = self.get('sessions')['sessions']
    return next(record for record in sessions if record['id'] == record_id)

  def wait_phase(self, owner_id: str, phase: str) -> dict[str, Any]:
    deadline = time.monotonic() + 300
    while True:
      record = self.record(owner_id)
      if record.get('execution', {}).get('phase') == phase:
        return record
      assert time.monotonic() < deadline, record
      time.sleep(0.3)

  def follow(
      self, job: dict[str, Any], on_delta=None
  ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reads a job's events until it ends; returns the job and events."""
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

  def turn(self, session_id: str, prompt: str, on_delta=None):
    job = self.post(
        f'sessions/{session_id}/turns',
        {'prompt': prompt},
    )
    return self.follow(job, on_delta)

  def runners(self) -> set[int]:
    """The runner processes the server runs now."""
    listed = subprocess.run(
        ['pgrep', '-P', str(self.process.pid)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return {int(pid) for pid in listed.split()}

  def graph_manifests(self) -> dict[str, int]:
    """Each kept graph's manifest and when it was written, in ns."""
    return {
        str(path): path.stat().st_mtime_ns
        for path in self.workspace.glob('graphs/*/manifest.json')
    }

  def stop(self) -> None:
    """Sends the server SIGTERM and waits for it to exit."""
    self.http.close()
    self.process.terminate()
    self.process.wait(60)


def _create(server: _Server, target: str = 'CPU') -> dict[str, Any]:
  """Creates a Session, which starts it: Reference CPU, Target `target`."""
  model = server.get('runtime/capabilities')['models'][0]
  runs = [
      {
          'id': side,
          'runtime': 'LiteRT-LM',
          'artifact': model['artifact'],
          'source': 'upload',
          'device': 'local',
          'backend': backend,
          'contextLength': '4096',
          # The runner --runner names, this computer's development build.
          'runnerBuild': 'dev',
      }
      for side, backend in (('ref', 'CPU'), ('target', target))
  ]
  return server.post(
      'sessions/create',
      {'name': f'CPU and {target}', 'model': '', 'runs': runs},
  )


def _chat(server: _Server, session_id: str) -> dict[str, Any]:
  """Starts a chat of a Session that is on; returns its job, completed."""
  # Long enough for the reply that is cancelled; the short ones stop first.
  generation = {'maxOutputTokens': 256}
  job, _ = server.follow(
      server.post(
          f'sessions/{session_id}/chats',
          {'generation': {'ref': generation, 'target': generation}},
      )
  )
  assert job['status'] == 'completed', job['error']
  return job


def _check_sides(
    view: dict[str, Any], n: int, job: dict[str, Any]
) -> dict[str, dict[str, Any]]:
  """Checks what the index says of turn n of each side; returns them.

  Every token of a side is accounted for, whatever the other side did.

  Args:
    view: The session view dict.
    n: The turn number.
    job: The job dict containing output.

  Returns:
    A mapping of side name to side entry dict.
  """
  assert [turn['n'] for turn in view['turns']] == list(range(1, n + 1))
  assert all('rejected' not in turn for turn in view['turns'])
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
    # Each side's tokens take consecutive steps and positions of their
    # own; the turn's start is the reference side's.
    if side == 'ref':
      assert tokens[0]['position'] == turn['step_start']
    assert [token['step'] for token in tokens] == list(range(len(tokens)))
    first = tokens[0]['position']
    assert [token['position'] for token in tokens] == list(
        range(first, first + len(tokens))
    )
    assert all(isinstance(token['source_forward_id'], int) for token in tokens)
    assert tokens[-1]['stop'] and not tokens[-1]['released']
    assert all(token['released'] for token in tokens[:-1])
    for token in tokens:
      if 'cannot_confirm' in token:
        assert set(token['cannot_confirm']) == {'text', 'reason'}
        reason = token['cannot_confirm']['reason']
        assert set(reason) == {'code', 'side', 'text'}, reason
        assert reason['code'] == 'greedy_unconfirmed', reason
  return sides


def _check_conversation(
    view: dict[str, Any], n: int, job: dict[str, Any]
) -> list[dict[str, Any]]:
  """Checks turn n of two equal sides; returns the ref side's tokens."""
  sides = _check_sides(view, n, job)
  assert [t['id'] for t in sides['ref']['tokens']] == [
      t['id'] for t in sides['target']['tokens']
  ]
  return sides['ref']['tokens']


def _check_tokens(
    server: _Server, session_id: str, n: int, count: int, chat: int = 1
) -> None:
  """Every pair of a turn of two equal sides: equal, and greedy."""
  address = {'turn': n, 'stage': 'decode', 'alignment': 'steps'}
  analysis = server.post(
      'token-analysis',
      {**address, 'rows': list(range(count))},
      session_id=session_id,
      chat=chat,
  )
  assert analysis['turn'] == n
  # Row i of the step alignment is step i on both sides.
  assert [(p['ref_step'], p['target_step']) for p in analysis['pairs']] == [
      (i, i) for i in range(count)
  ]
  rate = server.post('match-rate', address, session_id=session_id, chat=chat)
  assert rate['match_rate'] == 1.0
  for pair in analysis['pairs']:
    distribution = pair['distribution']
    assert distribution['compatible'], distribution['reasons']
    assert distribution['context'] == 'same'
    for key in (
        'kl',
        'js',
        'relative_l2',
        'cosine_distance',
        'max_abs',
    ):
      assert abs(pair['metrics'][key]['value']) < _EXACT, key
    assert math.isclose(pair['metrics']['norm_ratio']['value'], 1, rel_tol=1e-6)
    for transform in ('raw', 'centered'):
      assert pair['final_logits'][transform]['rmse']['value'] == 0
    sampling = pair['sampling_distribution']
    for side in ('ref', 'target'):
      assert sampling[side]['basis'] == 'logits_softmax'
      assert 0 < sampling[side]['selected_probability'] <= 1
    assert sampling['ref']['selected_probability'] == (
        sampling['target']['selected_probability']
    )
    assert len(sampling['rows']) > 1
  # The stage's KL by column, in one bin, and its peak: none differs.
  view = server.get('session', session_id=session_id, chat=chat)
  (turn,) = [turn for turn in view['turns'] if turn['n'] == n]
  columns = turn['comparison']['alignments']['steps']['positions']
  binned = server.post(
      'token-range',
      {
          **address,
          'analysis': 'final_logits',
          'metric': 'kl',
          'start': columns['start'],
          'end': columns['start'] + len(columns['pairs']),
          'bins': 1,
      },
      session_id=session_id,
      chat=chat,
  )
  (cell,) = binned['rows'][0]['bins']
  assert cell['valid_count'] == count
  assert abs(cell['max']) < _EXACT
  assert abs(binned['peak_kl']['value']) < _EXACT


def _check_logits(server: _Server, session_id: str, chat: int = 1) -> None:
  request = {
      'turn': 1,
      'stage': 'decode',
      'alignment': 'content',
      'row': 0,
      'transform': 'raw',
      'offset': 0,
      'limit': 16,
      'order': 'absolute_delta',
      'direction': 'descending',
  }
  page = server.post('final-logits', request, session_id=session_id, chat=chat)
  for key in ('turn', 'row', 'transform', 'offset'):
    assert page[key] == request[key], key
  assert (page['ref_step'], page['target_step']) == (0, 0)
  assert page['compatible']
  assert len(page['rows']) == 16
  assert all(row['delta'] == 0 for row in page['rows'])
  request = {key: request[key] for key in ('turn', 'stage', 'alignment', 'row')}
  request['transform'] = 'centered'
  summary = server.post(
      'final-logits-summary', request, session_id=session_id, chat=chat
  )
  for key, value in request.items():
    assert summary[key] == value, key
  assert summary['total'] == page['total']
  assert 'values' not in summary
  bins = server.post(
      'final-logits-bins',
      {**request, 'capacity': 128, 'reduction': 'max'},
      session_id=session_id,
      chat=chat,
  )
  assert bins['range'] == {'start': 0, 'end': page['total']}
  assert bins['visible']['ref'] == bins['visible']['target']
  points = server.post(
      'final-logits-points', request, session_id=session_id, chat=chat
  )
  assert points['paired'] == page['total']
  assert points['points']['ref'] == points['points']['target']


def _check_kv(server: _Server, session_id: str, chat: int = 1) -> None:
  contexts = {
      context['moment']: context
      for context in server.get(
          'kv-analysis',
          mode='metadata',
          turn=1,
          session_id=session_id,
          chat=chat,
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
        'kv-range',
        **query,
        bins=8,
        kind=kind,
        head='max',
        session_id=session_id,
        chat=chat,
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

  answer = server.get(
      'kv-head',
      turn=1,
      context_id=terminal['id'],
      layer=0,
      kind='value',
      position=end - 1,
      session_id=session_id,
      chat=chat,
  )
  selected = answer['selection']
  assert (selected['turn'], selected['context_id'], selected['layer']) == (
      1,
      terminal['id'],
      0,
  )
  assert (selected['kind'], selected['position']) == ('value', end - 1)
  # Every KV head of the layer, in one answer.
  assert [head['head'] for head in answer['heads']] == list(
      range(len(answer['heads']))
  )
  for head in answer['heads']:
    assert head['status'] == 'ok'
    assert head['metrics']['max_abs'] == 0
    assert all(channel['delta'] == 0 for channel in head['channels'])
    assert head['statistics']['abs_delta']['max'] == 0


def _start(server: _Server, target: str = 'CPU') -> str:
  """Creates a Session, which turns it on, and starts its first chat."""
  session_id = _create(server, target)['id']
  server.wait_phase(session_id, 'active')
  assert len(server.runners()) == 2
  _chat(server, session_id)
  return session_id


def _close(server: _Server, session_id: str) -> None:
  server.post(f'sessions/{session_id}/close', {})
  _check_ended(server, session_id)


def _check_ended(server: _Server, session_id: str) -> dict[str, Any]:
  """Waits for a Model Server to end; checks that no runner is left.

  Args:
    server: The test server instance.
    session_id: The session ID to check.

  Returns:
    The record as the listing shows it then.
  """
  record = server.wait_phase(session_id, 'ended')
  ended = 'Ended' if record['has_capture'] else 'Ended · no turn'
  assert record['status_text'] == ended, record
  assert record['model_server']['state'] == 'ended', record
  assert record['model_server']['text'] == ended, record
  assert not record['model_server']['button']['allowed'], record
  assert server.runners() == set()
  return record


@absltest.skipUnless(
    _BUILT and _MODEL.is_file(), 'needs the built runner and RUNNER_MODEL'
)
class ServerMainTest(absltest.TestCase):
  """The whole server, end to end: real runners, a real model, over HTTP."""

  @classmethod
  def setUpClass(cls):
    """Starts the server the tests share."""
    super().setUpClass()
    root = tempfile.mkdtemp(dir=absltest.TEST_TMPDIR.value)
    cls.server = _Server(pathlib.Path(root, 'workspace'))

  @classmethod
  def tearDownClass(cls):
    """Stops the server; then none of its runners runs."""
    pids = cls.server.runners()
    cls.server.stop()
    for pid in pids:
      try:
        os.kill(pid, 0)
      except ProcessLookupError:
        continue
      raise AssertionError(f'runner {pid} outlived the server')
    super().tearDownClass()

  def setUp(self):
    """Gives each test an empty directory of its own."""
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)

  def test_a_session_of_two_cpu_runners_from_start_to_close(self):
    """Create, two turns, analyses, a Stop that turns it off."""
    # Checking this computer launches no runner.
    checked = self.server.post('devices/inspect', {'id': 'local'})
    self.assertEqual(checked['statusText'], 'Connected')
    # Its builds, read without launching one: the --runner file alone.
    self.assertEqual(checked['runnerState'], 'not_running')
    (build,) = self.server.get('devices/builds', id='local')['builds']
    self.assertEqual((build['id'], build['available']), ('dev', True))
    self.assertEqual(self.server.runners(), set())
    session_id = _create(self.server)['id']
    record = self.server.wait_phase(session_id, 'active')
    self.assertEqual(record['chats'], [])
    self.assertTrue(record['actions']['new_chat']['allowed'])
    self.assertLen(self.server.runners(), 2)
    self.assertEqual(_chat(self.server, session_id)['chat'], 1)
    self.assertTrue(
        self.server.record(session_id)['chats'][0]['actions']['send']['allowed']
    )
    # The model's graph: converted by the first start, kept since.
    artifact = self.server.get('runtime/capabilities')['models'][0]['artifact']
    graph = self.server.get(f'artifacts/{artifact}/graph')
    self.assertEqual(graph['status'], 'completed', graph)
    kept = self.server.graph_manifests()
    self.assertTrue(kept)
    self.assertIn('decode', [s['name'] for s in graph['signatures']])
    decode = self.server.get(f'artifacts/{artifact}/graph/decode')
    self.assertEqual(decode['graphs'][0]['id'], 'decode')
    found = self.server.get(f'artifacts/{artifact}/architecture')
    if found['architecture'] is not None:
      for entry in found['bindings']['decode']:
        for anchor, binding in entry['anchors'].items():
          self.assertTrue(binding['bound'], (entry['layer'], anchor))
      # The layers a signature computes are the ones bound in it: a
      # prefill fewer than the decode, which computes them all.
      for signature in graph['signatures']:
        self.assertEqual(
            signature['layer_ids'],
            [
                entry['layer']
                for entry in found['bindings'][signature['name']]
                if any(b['bound'] for b in entry['anchors'].values())
            ],
            signature['name'],
        )
      decode_layers = next(
          s['layer_ids'] for s in graph['signatures'] if s['name'] == 'decode'
      )
      self.assertEqual(
          decode_layers, list(range(len(found['skeleton']['layers'])))
      )
      # Where the skeleton comes from: this repository's files.
      source = found['architecture']
      self.assertTrue(
          source['generator']['path'].endswith('architectures/generate.py')
      )
      self.assertRegex(
          source['config']['path'], r'^src/server/.*\.config\.json$'
      )
      self.assertRegex(source['config']['revision'], '^[0-9a-f]{40}$')

    # Turn 1, its view, and every analysis of it.
    job, events = self.server.turn(session_id, 'What is 2 + 2?')
    self.assertEqual(job['status'], 'completed', job['error'])
    self.assertEqual(job['turn'], 1)
    for side in ('ref', 'target'):
      deltas = [
          e['text']
          for e in events
          if e['type'] == 'delta' and e['runId'] == side
      ]
      self.assertEqual(''.join(deltas), job['output'][side])
      self.assertTrue(''.join(deltas))
    self.assertEqual(job['output']['ref'], job['output']['target'])
    view = self.server.get('session', session_id=session_id, chat=1)
    tokens = _check_conversation(view, 1, job)
    self.assertTrue(self.server.record(session_id)['has_capture'])
    _check_tokens(self.server, session_id, 1, len(tokens))
    _check_logits(self.server, session_id)
    _check_kv(self.server, session_id)

    # Turn 2 follows from turn 1's context.
    job, _ = self.server.turn(session_id, 'And 3 + 3?')
    self.assertEqual(job['status'], 'completed', job['error'])
    view = self.server.get('session', session_id=session_id, chat=1)
    tokens = _check_conversation(view, 2, job)
    _check_tokens(self.server, session_id, 2, len(tokens))

    # Stopping a reply ends the chat; the Model Server stays on, and the
    # turns stay.
    cancelled = []

    def cancel(event):
      if not cancelled:
        cancelled.append(self.server.post(f'jobs/{event["jobId"]}/cancel', {}))

    job, _ = self.server.turn(
        session_id, 'Write a long story about a lighthouse keeper.', cancel
    )
    self.assertTrue(cancelled and job['status'] == 'cancelled')
    record = self.server.record(session_id)
    self.assertEqual(record['execution']['phase'], 'active')
    (chat,) = record['chats']
    self.assertEqual(chat['state'], 'ended')
    self.assertIn('inspected in Debug', chat['notice'])
    # No more turns of it; Debug still reads the two it kept.
    self.assertEqual(
        chat['actions']['send'],
        {'allowed': False, 'reason': 'This chat has ended.'},
    )
    self.assertLen(
        self.server.get('session', session_id=session_id, chat=1)['turns'],
        2,
    )
    _check_tokens(self.server, session_id, 2, len(tokens))
    # A new chat on the same runners: its turns count from 1, from an
    # empty context, equal on both sides again.
    self.assertEqual(_chat(self.server, session_id)['chat'], 2)
    job, _ = self.server.turn(session_id, 'What is 2 + 2?')
    self.assertEqual((job['status'], job['chat']), ('completed', 2))
    view = self.server.get('session', session_id=session_id, chat=2)
    tokens = _check_conversation(view, 1, job)
    _check_tokens(self.server, session_id, 1, len(tokens), chat=2)
    _close(self.server, session_id)
    # Both devices are free: another Session turns on at once, and finds
    # the graph kept.
    _close(self.server, _start(self.server))
    self.assertEqual(self.server.graph_manifests(), kept)

  def test_a_session_of_a_cpu_and_a_gpu_runner(self):
    """Reference on the CPU, Target on the GPU: both turns indexed.

    The GPU's F16 logits may tie at the top; such a step is shown as
    unconfirmed, and the turn stands.
    """
    session_id = _start(self.server, 'GPU')
    view = self.server.get('session', session_id=session_id, chat=1)
    self.assertEqual([run['backend'] for run in view['runs']], ['CPU', 'GPU'])
    unconfirmed = []
    for n, prompt in enumerate(('What is 2 + 2?', 'And 3 + 3?'), 1):
      job, _ = self.server.turn(session_id, prompt)
      self.assertEqual(job['status'], 'completed', job['error'])
      view = self.server.get('session', session_id=session_id, chat=1)
      sides = _check_sides(view, n, job)
      self.assertTrue(self.server.record(session_id)['has_capture'])
      for side, entry in sides.items():
        unconfirmed += [
            (n, side, token['step'], token['cannot_confirm'])
            for token in entry['tokens']
            if 'cannot_confirm' in token
        ]
    self.assertFalse([u for u in unconfirmed if u[1] == 'ref'])
    logging.info(
        'CPU and GPU: %d steps cannot be confirmed: %s',
        len(unconfirmed),
        unconfirmed,
    )
    _close(self.server, session_id)

  def test_terminating_the_server_stops_the_runners_it_runs(self):
    """SIGTERM while a Model Server is on: no runner outlives the server."""
    server = _Server(self.tmp_path / 'workspace')
    pids = set()
    try:
      session_id = _create(server)['id']
      server.wait_phase(session_id, 'active')
      pids = server.runners()
      self.assertLen(pids, 2)
    finally:
      server.stop()  # SIGTERM
    for pid in pids:
      with self.assertRaises(ProcessLookupError):
        os.kill(pid, 0)

  def test_a_runner_killed_mid_turn_ends_the_session(self):
    """SIGKILL of one runner mid-reply: the turn fails and ends the chat.

    The Model Server turns itself off: the other runner is stopped too.
    """
    session_id = _start(self.server)
    killed = []

    def kill(event):
      del event  # Unused.
      if not killed:
        killed.append(min(self.server.runners()))
        os.kill(killed[0], signal.SIGKILL)

    job, _ = self.server.turn(
        session_id, 'Write a long story about a lighthouse keeper.', kill
    )
    self.assertTrue(killed and job['status'] == 'failed', job)
    self.assertIn('The connection to the runner closed.', job['error'])
    record = _check_ended(self.server, session_id)
    self.assertEqual(record['chats'][0]['state'], 'ended')
    self.assertFalse(record['chats'][0]['actions']['send']['allowed'])
    self.assertRegex(record['notice'], '^Runner closed: ')

  def test_a_runner_killed_between_turns_ends_the_session(self):
    """SIGKILL while idle: the Session ends, and the server turns off.

    The other runner is stopped too; its chat ended, saying why, and a
    turn is refused as the Session ended.
    """
    session_id = _start(self.server)
    job, _ = self.server.turn(session_id, 'What is 2 + 2?')
    self.assertEqual(job['status'], 'completed', job['error'])
    os.kill(min(self.server.runners()), signal.SIGKILL)
    record = _check_ended(self.server, session_id)
    self.assertRegex(
        record['notice'],
        '^Runner closed: the connection to the (Reference|Target) runner '
        'is lost.$',
    )
    (chat,) = record['chats']
    self.assertEqual(
        chat['notice'],
        f'{record["notice"][:-1]}. This chat is over; its turns can still '
        'be inspected in Debug.',
    )
    response = self.server.http.post(
        f'/api/sessions/{session_id}/turns', json={'prompt': 'And 3 + 3?'}
    )
    self.assertEqual(response.status_code, 409, response.text)
    self.assertEqual(response.json()['error'], 'This Session has ended.')


if __name__ == '__main__':
  test_main.main()
