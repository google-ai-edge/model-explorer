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

"""Tests of one Session's Model Server, with in-process fake runners."""

import errno
import json
import pathlib
import shutil
import threading
import time
import types
from typing import Any
from unittest import mock

from absl.testing import absltest
import grpc
from model_debugger.protocol import runner_pb2
from model_debugger.server.capture import capture_index
from model_debugger.server.runner import capture_writer
from model_debugger.server.runner import launcher
from model_debugger.server.session import jobs
from model_debugger.server.session import model_server
from model_debugger.server.session import options
from model_debugger.server.testing import capture_fixtures
from model_debugger.server.testing import fake_builds
from model_debugger.server.testing import fake_launchers
from model_debugger.server.testing import fake_runner
from model_debugger.server.testing import test_main
from model_debugger.server.testing import world as world_lib
from model_debugger.server.util import errors


def _with_runner(world, change):
  """Makes every launch go through `change(launched)` first."""
  original = world.launchers.launch

  def launch(device_id, build=None):
    launched = original(device_id, build)
    change(launched)
    return launched

  world.launchers.launch = launch


def _hold_reads(world, release, reading):
  def change(launched):
    def before(method):
      if method == 'ReadCaptures':
        reading.set()
        release.wait(fake_launchers.TIMEOUT)

    launched.runner.before = before

  _with_runner(world, change)


def _miscount_completions(world):
  """Makes every runner count one completion token more than it generated.

  The capture index rejects such a turn: the runner and its files disagree.

  Args:
    world: The test world containing launchers and runners.
  """

  def change(launched):
    script = launched.runner._respond

    def respond(turn, prompt):
      chunks, completion, files = script(turn, prompt)
      completion.usage.completion_tokens += 1
      return chunks, completion, files

    launched.runner._respond = respond

  _with_runner(world, change)


def _drop_a_phase(world):
  """Makes every runner report its first fed token of no phase."""

  def change(launched):
    script = launched.runner._respond

    def respond(turn, prompt):
      chunks, completion, files = script(turn, prompt)
      completion.model_debugger.fed_tokens[0].ClearField('phase')
      return chunks, completion, files

    launched.runner._respond = respond

  _with_runner(world, change)


def _fail_create_chat(method: str) -> None:
  """A runner's `before` whose runtime fails every CreateChat."""
  if method == 'CreateChat':
    raise fake_runner.AbortError(
        grpc.StatusCode.INTERNAL, 'The runtime failed.'
    )


def _errors(logs) -> list[str]:
  """The ERROR lines of what assertLogs() caught."""
  return [
      record.getMessage()
      for record in logs.records
      if record.levelname == 'ERROR'
  ]


def _label(record: dict[str, Any]) -> str:
  """How the log names a Session."""
  return f"Session '{record['name']}' ({record['id']})"


class ModelServerTest(absltest.TestCase):
  """Tests of one Session's Model Server, with in-process fake runners."""

  def setUp(self):
    """An empty directory, and a computer and two devices."""
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)
    self.world = world_lib.create(self)

  def _check_turned_off(self, record_id):
    """Checks that a Model Server turned off, or turned itself off.

    As Turn off leaves it: both runners stopped, both devices free, and
    the listing shows it off, its sides off, never to run again.

    Args:
      record_id: The ID of the session/record to check.

    Returns:
      The record as the listing shows it.
    """
    summary = self.world.summary(record_id)
    self.assertEqual(summary['execution']['phase'], 'ended')
    ended = 'Ended' if summary['has_capture'] else 'Ended · no turn'
    self.assertEqual(summary['status_text'], ended)
    self.assertEqual(
        [side['phaseText'] for side in summary['execution']['runners']],
        ['Off', 'Off'],
    )
    self.assertEqual(
        [side['connected'] for side in summary['execution']['runners']],
        [False, False],
    )
    model_server_view = summary['model_server']
    self.assertEqual(model_server_view['state'], 'ended')
    self.assertEqual(model_server_view['text'], ended)
    self.assertIsNone(model_server_view['button'])
    self.assertTrue(
        all(launched._stopped for launched in self.world.launchers.launched)
    )
    for side in self.world.model_servers.state(record_id).sides:
      self.assertIsNone(self.world.model_servers.holding(side.device))
    return summary

  def _chat(self, record_id, n=1):
    """A chat of a Session, as its record holds it."""
    return self.world.sessions.get(record_id)['chats'][n - 1]

  def _chat_view(self, record_id, n=1):
    """A chat of a Session, as the listing shows it."""
    return self.world.summary(record_id)['chats'][n - 1]

  def _check_chat_ended(self, record_id, notice, n=1):
    """Checks that a chat ended alone: the Model Server stays on.

    Its record and the listing say why; it takes no turn, and a new chat
    may start.

    Args:
      record_id: The ID of the session/record to check.
      notice: The expected notice or reason why the chat ended.
      n: The 1-based index of the chat.

    Returns:
      The record as the listing shows it.
    """
    self.assertTrue(self._chat(record_id, n)['over'])
    summary = self.world.summary(record_id)
    self.assertEqual(summary['execution']['phase'], 'active')
    self.assertEqual(summary['status_text'], 'Active')
    self.assertIsNone(summary['live_chat'])
    self.assertEqual(summary['notice'], '')
    chat = summary['chats'][n - 1]
    self.assertEqual(chat['state'], 'ended')
    self.assertEqual(chat['notice'], notice)
    self.assertEqual(
        chat['actions']['send'],
        {'allowed': False, 'reason': model_server.CHAT_ENDED},
    )
    self.assertTrue(summary['actions']['new_chat']['allowed'])
    self.assertFalse(
        any(launched._stopped for launched in self.world.launchers.launched)
    )
    with self.assertRaisesRegex(errors.Conflict, model_server.NO_CHAT):
      self.world.model_servers.generate(record_id, 'Hi')
    return summary

  def test_a_turns_progress_tells_its_capture_arriving(self):
    """Once the reply ended: Retrieving the capture, with what arrived."""
    self.assertEqual(model_server.retrieving(0), 'Retrieving the capture')
    self.assertEqual(
        model_server.retrieving(99 * 10**6), 'Retrieving the capture'
    )
    self.assertEqual(
        model_server.retrieving(345 * 10**6),
        'Retrieving the capture · 300 MB so far',
    )
    self.assertEqual(
        model_server.retrieving(1_234_000_000),
        'Retrieving the capture · 1.2 GB so far',
    )
    record = self.world.create()
    self.world.start(record)
    job = self.world.turn(record['id'], 'Hi')
    self.assertEqual(job['status'], 'completed', job['error'])
    self.assertEqual(job['progress'], 'Retrieving the capture')

  def test_initialize_failure_stops_every_runner(self):
    """A start that fails ends its Session, with no turn and no runner.

    Its record keeps why; it is not entered, and never starts again: a
    new Session is the way to try again.
    """
    record = self.world.create()
    bad = runner_pb2.RunnerInfo()
    bad.CopyFrom(fake_runner.DEFAULT_INFO)
    bad.protocol_version = '2.0.0'
    original = self.world.launchers.launch

    def launch(device_id, build=None):
      launched = original(device_id, build)
      launched.runner.info = bad
      return launched

    self.world.launchers.launch = launch
    job = self.world.wait(self.world.model_servers.initialize(record['id']))
    self.assertEqual(job['status'], 'failed')
    self.assertIn('protocol 2.0.0', job['error'])
    self.assertIsNone(self.world.view(record['id']))
    self.assertTrue(
        all(launched._stopped for launched in self.world.launchers.launched)
    )
    summary = self.world.listing.summary(self.world.sessions.get(record['id']))
    self.assertEqual(summary['status_text'], 'Ended · no turn')
    self.assertEqual(
        summary['model_server'],
        {
            'state': 'ended',
            'text': 'Ended · no turn',
            'error': job['error'],
            'button': None,
        },
    )
    self.assertFalse(summary['actions']['enter']['allowed'])
    self.assertIn('create a new Session', summary['actions']['enter']['reason'])
    self.assertTrue(self.world.sessions.get(record['id'])['started'])
    # Never again, though nothing fails now: another Session starts.
    self.world.launchers.launch = original
    with self.assertRaisesRegex(errors.Conflict, model_server.ENDED):
      self.world.model_servers.initialize(record['id'])
    self.world.start(self.world.create())

  def test_a_failed_start_logs_one_error_naming_the_session(self):
    """One ERROR line each: the Session, the side and the UI's reason.

    A start refused before it launches anything, a refusal and no
    failure, is told at INFO, as one that comes on, and goes off, is.
    """
    record = self.world.create()
    bad = runner_pb2.RunnerInfo()
    bad.CopyFrom(fake_runner.DEFAULT_INFO)
    bad.protocol_version = '2.0.0'
    original = self.world.launchers.launch

    def launch(device_id, build=None):
      launched = original(device_id, build)
      launched.runner.info = bad
      return launched

    self.world.launchers.launch = launch
    with self.assertLogs(level='INFO') as logs:
      job = self.world.wait(self.world.model_servers.initialize(record['id']))
    self.assertEqual(job['status'], 'failed')
    self.assertRegex(job['error'], r'^(Reference|Target): .*protocol 2\.0')
    self.assertEqual(
        _errors(logs), [f'{_label(record)}: start failed: {job["error"]}']
    )

    refused = self.world.create(target='android:C')
    with self.assertLogs(level='INFO') as logs:
      with self.assertRaises(errors.Conflict) as raised:
        self.world.model_servers.initialize(refused['id'])
    self.assertEqual(_errors(logs), [])
    self.assertIn(
        f'{_label(refused)}: start refused: {raised.exception}',
        [record.getMessage() for record in logs.records],
    )
    self.assertStartsWith(str(raised.exception), 'Target: ')

    self.world.launchers.launch = original
    record = self.world.create()
    with self.assertLogs(level='INFO') as logs:
      self.world.start(record)
      self.world.model_servers.close(record['id'])
    self.assertEqual(_errors(logs), [])
    said = [record.getMessage() for record in logs.records]
    self.assertIn(f'{_label(record)}: the Model Server is on', said)
    self.assertIn(f'{_label(record)}: the Model Server is off', said)

  def test_a_failed_turn_logs_one_error_naming_the_session(self):
    """One ERROR line for the turn; a turn refused after it is INFO."""
    record = self.world.create(ref='android:A', target='android:B')
    self.world.start(record)
    for script in self.world.scripts('android:B'):
      script.refuse = grpc.StatusCode.INTERNAL
    with self.assertLogs(level='INFO') as logs:
      job = self.world.turn(record['id'], 'Hi')
      with self.assertRaises(errors.Conflict) as refused:
        self.world.model_servers.generate(record['id'], 'Hi')
    self.assertEqual(job['status'], 'failed')
    self.assertStartsWith(job['error'], 'Target: Refused turn 1.')
    self.assertEqual(
        _errors(logs),
        [f'{_label(record)}: Chat 1 turn 1 failed: {job["error"]}'],
    )
    self.assertIn(
        f'{_label(record)}: turn refused: {model_server.NO_CHAT}',
        [record.getMessage() for record in logs.records],
    )
    self.assertEqual(str(refused.exception), model_server.NO_CHAT)

  def test_generate_streams_replies_and_stores_both_captures(self):
    """Deltas reach the job; both captures are read, released, indexed."""
    record = self.world.create()
    self.world.start(record)
    job = self.world.turn(record['id'], 'What is 2 + 2?')
    self.assertEqual(job['status'], 'completed', job['error'])
    self.assertEqual(job['turn'], 1)
    reply = fake_launchers.REPLY_TEXT
    self.assertEqual(job['output'], {'ref': reply, 'target': reply})
    self.assertEqual(job['reasoning'], {'ref': '', 'target': ''})
    deltas = [e for e in self.world.events(job['id']) if e['type'] == 'delta']
    self.assertEqual({event['runId'] for event in deltas}, {'ref', 'target'})
    self.assertEqual({event['channel'] for event in deltas}, {'content'})
    for runner in self.world.launchers.runners('local'):
      self.assertEqual(runner.released, [1])
    index = self.world.captures.get(record['id'], 1)
    self.assertEqual([turn['n'] for turn in index.turns], [1])
    self.assertEqual(index.turns[0]['prompt'], 'What is 2 + 2?')
    for side in ('ref', 'target'):
      entry = index.turns[0]['sides'][side]
      self.assertEqual(entry['completion']['content'], reply)
      for file in entry['files']:
        self.assertTrue((index.root / entry['dir'] / file['path']).is_file())
    # The same object, read again from disk, holds the same turn.
    self.assertEqual(
        capture_index.CaptureIndex.load(index.root).turns, index.turns
    )
    record = self.world.sessions.get(record['id'])
    self.assertTrue(record['chats'][0]['has_capture'])
    summary = self.world.listing.summary(record)
    self.assertEqual(summary['status_text'], 'Active')
    self.assertEqual(summary['live_chat'], 1)
    self.assertTrue(summary['chats'][0]['actions']['send']['allowed'])
    job = self.world.turn(record['id'], 'And 3 + 3?')
    self.assertEqual(job['chat'], 1)
    self.assertEqual(job['status'], 'completed', job['error'])
    self.assertEqual([turn['n'] for turn in index.turns], [1, 2])

  def test_a_chat_that_thinks_lines_up_every_turn(self):
    """Each reply that thought is rewritten; the next turn follows it."""
    record = self.world.create()
    self.world.start(record)
    for script in self.world.scripts():
      script.thought = ['hmm']

    for prompt in ('What is 2 + 2?', 'And 3 + 3?', 'Thanks.'):
      job = self.world.turn(record['id'], prompt)
      self.assertEqual(job['status'], 'completed', job['error'])

    index = self.world.captures.get(record['id'], 1)
    self.assertEqual([turn['n'] for turn in index.turns], [1, 2, 3])
    self.assertEqual(index.rejected, [])
    for side in ('ref', 'target'):
      entries = [turn['sides'][side] for turn in index.turns]
      for before, after in zip(entries, entries[1:]):
        # The next turn starts where the rewrite left the context.
        self.assertTrue(before['rewrite'])
        self.assertEqual(
            after['input'][0]['position'],
            before['rewrite'][-1]['position'] + 1,
        )
        self.assertEqual(after['input'][0]['origin'], 'carried')
        self.assertNotIn(None, [token['id'] for token in after['input']])

  def test_a_reply_that_thinks_streams_its_thoughts_into_the_job(self):
    """Reasoning chunks reach the job as they come, before the reply's."""
    record = self.world.create()
    self.world.start(record)
    release = threading.Event()
    for script in self.world.scripts():
      script.thought = ['hmm', ' yes']
      script.hold = release
    job = self.world.model_servers.generate(record['id'], 'What is 2 + 2?')
    try:
      # The first chunk, a thought, is in the job while the reply waits.
      world_lib.eventually(
          lambda: (
              self.world.jobs.get(job['id'])['reasoning'] == {
                  'ref': 'hmm',
                  'target': 'hmm',
              }
          )
      )
      self.assertEqual(
          self.world.jobs.get(job['id'])['output'],
          {'ref': '', 'target': ''},
      )
    finally:
      release.set()
    job = self.world.wait(job)
    self.assertEqual(job['status'], 'completed', job['error'])
    self.assertEqual(job['reasoning'], {'ref': 'hmm yes', 'target': 'hmm yes'})
    reply = fake_launchers.REPLY_TEXT
    self.assertEqual(job['output'], {'ref': reply, 'target': reply})
    deltas = [
        (e['runId'], e['channel'], e['text'])
        for e in self.world.events(job['id'])
        if e['type'] == 'delta'
    ]
    for side in ('ref', 'target'):
      self.assertEqual(
          [(channel, text) for run, channel, text in deltas if run == side],
          [('reasoning', 'hmm'), ('reasoning', ' yes')]
          + [('content', text) for text in fake_launchers.REPLY],
      )
    # What the job streamed is what the turn recorded.
    entry = self.world.captures.get(record['id'], 1).turns[0]['sides']['ref']
    self.assertEqual(entry['completion']['reasoning_content'], 'hmm yes')

  def test_a_turn_the_index_rejects_is_kept_and_ends_the_chat(self):
    """Its files stay as the evidence; index.json says why; chat over."""
    _miscount_completions(self.world)
    record = self.world.create()
    self.world.start(record)

    with self.assertLogs(level='INFO') as logs:
      job = self.world.turn(record['id'], 'What is 2 + 2?')

    reason = 'The runner counts 5 completion tokens; turn 1 has 4 decode steps.'
    text = f'Reference: The capture index rejected turn 1: {reason}'
    self.assertEqual(
        _errors(logs),
        [
            f'{_label(record)}: {text}',
            f'{_label(record)}: Chat 1 turn 1 failed: {job["error"]}',
        ],
    )
    self.assertEqual(job['status'], 'failed')
    # The turn is kept, rejected: Debug shows it, with why. The chat's
    # notice names the turn only, and the side that failed is the job's.
    self.assertEqual(
        job['error'],
        f'Turn 1 was rejected. {model_server.CHAT_OVER_CAPTURED}',
    )
    self.assertEqual(job['failed_sides'], ['ref'])
    # The runners released their copies: the server's are what is left.
    for runner in self.world.launchers.runners('local'):
      self.assertEqual(runner.released, [1])
    index = self.world.captures.get(record['id'], 1)
    self.assertEqual(index.turns, [])
    (rejected,) = index.rejected
    self.assertEqual(capture_index.rejection(rejected), text)
    stored = json.loads((index.root / 'index.json').read_text())
    (turn,) = stored['turns']
    self.assertEqual(turn['rejected'], {'side': 'ref', 'reason': reason})
    self.assertEqual(turn['prompt'], 'What is 2 + 2?')
    for side in capture_index.SIDES:
      files = turn['sides'][side]['files']
      self.assertIn('generated_tokens.jsonl', [file['path'] for file in files])
      for file in files:
        path = index.root / side / 'turn-1' / file['path']
        self.assertEqual(path.stat().st_size, file['size'])
    self.assertTrue(self._chat(record['id'])['has_capture'])
    self.assertEqual(self._chat(record['id'])['notice'], job['error'])
    summary = self._check_chat_ended(record['id'], job['error'])
    self.assertTrue(summary['chats'][0]['actions']['debug']['allowed'])

  def test_a_fed_token_of_no_phase_is_rejected_and_its_files_kept(self):
    """A runner breaking the protocol leaves the files that show it."""
    _drop_a_phase(self.world)
    record = self.world.create()
    self.world.start(record)

    job = self.world.turn(record['id'], 'What is 2 + 2?')

    reason = 'The runner reports a fed token at position 0 of no phase.'
    self.assertEqual(job['status'], 'failed')
    self.assertTrue(job['error'].startswith('Turn 1 was rejected.'))
    index = self.world.captures.get(record['id'], 1)
    self.assertEqual(index.turns, [])
    stored = json.loads((index.root / 'index.json').read_text())
    (turn,) = stored['turns']
    self.assertEqual(turn['rejected'], {'side': 'ref', 'reason': reason})
    for side in capture_index.SIDES:
      self.assertIsNone(
          turn['sides'][side]['completion']['fed_tokens'][0]['phase']
      )
      files = turn['sides'][side]['files']
      self.assertTrue(files)
      for file in files:
        path = index.root / side / 'turn-1' / file['path']
        self.assertEqual(path.stat().st_size, file['size'])
    self.assertTrue(self._chat(record['id'])['has_capture'])

  def test_generate_refuses_while_a_turn_runs(self):
    """One job at a time per Session."""
    record = self.world.create()
    self.world.start(record)
    release = threading.Event()
    for script in self.world.scripts():
      script.hold = release
    job = self.world.model_servers.generate(record['id'], 'Hi')
    try:
      with self.assertRaisesRegex(errors.Conflict, 'Wait'):
        self.world.model_servers.generate(record['id'], 'Hi again')
    finally:
      release.set()
    self.assertEqual(self.world.wait(job)['status'], 'completed')

  def test_a_session_comes_on_with_no_chat(self):
    """Turned on, both models load and no chat is created until asked."""
    record = self.world.create()
    self.world.start(record, None)
    for runner in self.world.launchers.runners('local'):
      self.assertIsNotNone(runner.load_request)
      self.assertEqual(runner.chats, [])
    summary = self.world.summary(record['id'])
    self.assertEqual(summary['status_text'], 'Active')
    self.assertEqual((summary['chats'], summary['live_chat']), ([], None))
    self.assertTrue(summary['actions']['enter']['allowed'])
    self.assertTrue(summary['actions']['new_chat']['allowed'])
    self.assertFalse(summary['has_capture'])
    with self.assertRaisesRegex(errors.Conflict, model_server.NO_CHAT):
      self.world.model_servers.generate(record['id'], 'Hi')
    # Turned off so, it ended with no turn: it is not entered.
    self.world.model_servers.close(record['id'])
    summary = self.world.summary(record['id'])
    self.assertEqual(summary['status_text'], 'Ended · no turn')
    self.assertFalse(summary['actions']['enter']['allowed'])

  def test_a_chat_starts_on_both_sides_with_its_own_settings(self):
    """Each side's CreateChat is its settings'; the chat is then live."""
    record = self.world.create()
    self.world.start(record, None)
    with self.assertLogs(level='INFO') as logs:
      job = self.world.chat(
          record['id'],
          {
              'ref': {'systemPrompt': 'Be brief.'},
              'target': {'maxOutputTokens': 8},
          },
      )
    self.assertEqual((job['operation'], job['chat']), ('chat', 1))
    self.assertIn(
        f'{_label(record)}: Chat 1 started',
        [line.getMessage() for line in logs.records],
    )
    progress = [
        (event['runId'], event['message'])
        for event in self.world.events(job['id'])
        if event['type'] == 'progress'
    ]
    self.assertCountEqual(
        progress,
        [
            ('ref', model_server.STARTING_CHAT),
            ('target', model_server.STARTING_CHAT),
        ],
    )
    ref, target = (
        runner.chat_request
        for runner in sorted(
            self.world.launchers.runners('local'),
            key=lambda runner: runner.load_request is None,
        )
    )
    requests = {
        request.messages[0].content if request.messages else '': request
        for request in (ref, target)
    }
    self.assertEqual(set(requests), {'Be brief.', ''})
    self.assertEqual(requests[''].options.max_completion_tokens, 8)
    summary = self.world.summary(record['id'])
    self.assertEqual(summary['live_chat'], 1)
    (chat,) = summary['chats']
    self.assertEqual(
        {key: chat[key] for key in ('n', 'title', 'state', 'notice')},
        {'n': 1, 'title': 'Chat 1', 'state': 'live', 'notice': ''},
    )
    self.assertEqual(
        chat['summary'],
        'Reference: Greedy · system prompt / Target: Greedy · no system '
        'prompt · at most 8 tokens',
    )
    self.assertEqual(
        chat['actions'],
        {
            'send': {'allowed': True, 'reason': ''},
            'debug': {
                'allowed': False,
                'reason': 'This chat has no turn to inspect.',
            },
        },
    )
    index = self.world.captures.get(record['id'], 1)
    self.assertEqual(
        index.sides['ref']['chat']['messages'],
        [{'role': 'system', 'content': 'Be brief.'}],
    )

  def test_a_new_chat_ends_the_live_one_which_stays_readable(self):
    """The runners' chats are replaced; the first chat's turns stay."""
    record = self.world.create()
    self.world.start(record)
    self.assertEqual(self.world.turn(record['id'], 'Hi')['status'], 'completed')
    job = self.world.chat(
        record['id'], {'ref': {'maxOutputTokens': 64}, 'target': {}}
    )
    self.assertEqual(job['chat'], 2)
    for runner in self.world.launchers.runners('local'):
      self.assertLen(runner.chats, 2)
    first = self._chat(record['id'], 1)
    self.assertTrue(first['over'])
    self.assertEqual(
        first['notice'],
        f'A new chat was started. {model_server.CHAT_OVER_CAPTURED}',
    )
    job = self.world.turn(record['id'], 'Hello again')
    self.assertEqual(job['status'], 'completed', job['error'])
    self.assertEqual((job['chat'], job['turn']), (2, 1))
    summary = self.world.summary(record['id'])
    self.assertEqual(summary['live_chat'], 2)
    self.assertEqual(
        [(chat['n'], chat['state']) for chat in summary['chats']],
        [(1, 'ended'), (2, 'live')],
    )
    self.assertEqual(
        summary['chats'][0]['actions']['send'],
        {'allowed': False, 'reason': model_server.CHAT_ENDED},
    )
    for chat in summary['chats']:
      self.assertTrue(chat['actions']['debug']['allowed'])
    # Each chat's capture is its own, turns from 1.
    for n, prompt in ((1, 'Hi'), (2, 'Hello again')):
      index = self.world.captures.get(record['id'], n)
      self.assertEqual(index.root.name, f'chat-{n}')
      self.assertEqual(
          [(turn['n'], turn['prompt']) for turn in index.turns],
          [(1, prompt)],
      )
    self.assertEqual(
        self.world.captures.get(record['id'], 2).sides['ref']['chat'][
            'options'
        ]['max_completion_tokens'],
        64,
    )

  def test_a_turn_of_a_chat_another_page_replaced_is_refused(self):
    """A page that still shows Chat 1 live sends no turn into Chat 2."""
    record = self.world.create()
    self.world.start(record)
    self.world.chat(record['id'])
    with self.assertRaisesRegex(errors.Conflict, model_server.CHAT_ENDED):
      self.world.model_servers.generate(record['id'], 'Hi', chat=1)
    with self.assertRaisesRegex(ValueError, 'by its number'):
      self.world.model_servers.generate(record['id'], 'Hi', chat='2')
    # A chat the Session never had is not one that ended.
    with self.assertRaisesRegex(errors.NotFound, '^Chat 99 does not exist.$'):
      self.world.model_servers.generate(record['id'], 'Hi', chat=99)
    self.assertEqual(self.world.captures.get(record['id'], 2).turns, [])
    job = self.world.wait(
        self.world.model_servers.generate(record['id'], 'Hi', chat=2)
    )
    self.assertEqual(job['status'], 'completed', job['error'])
    self.assertEqual((job['chat'], job['turn']), (2, 1))

  def test_a_chat_a_side_fails_to_create_does_not_start(self):
    """Its error names the side; the Session stays on; another may start.

    The chat that was live ended all the same: its runners' chats were
    replaced, the failing side's left with none.
    """
    record = self.world.create(ref='android:A', target='android:B')
    self.world.start(record)
    (ref,) = self.world.launchers.runners('android:A')
    (target,) = self.world.launchers.runners('android:B')
    target.before = _fail_create_chat
    with self.assertLogs(level='INFO') as logs:
      job = self.world.wait(
          self.world.model_servers.start_chat(
              record['id'], world_lib.DEFAULT_CHAT
          )
      )
    self.assertEqual(
        (job['status'], job['error'], job['chat']),
        ('failed', 'Target: The runtime failed.', 2),
    )
    self.assertEqual(
        _errors(logs),
        [f'{_label(record)}: a chat did not start: {job["error"]}'],
    )
    self.assertIsNone(target.chat_request)
    self.assertLen(ref.chats, 2)
    summary = self.world.summary(record['id'])
    self.assertEqual(summary['execution']['phase'], 'active')
    self.assertIsNone(summary['live_chat'])
    self.assertEqual(
        [(chat['n'], chat['state']) for chat in summary['chats']],
        [(1, 'ended')],
    )
    self.assertTrue(summary['actions']['new_chat']['allowed'])
    with self.assertRaisesRegex(errors.Conflict, model_server.NO_CHAT):
      self.world.model_servers.generate(record['id'], 'Hi')
    target.before = None
    self.assertEqual(self.world.chat(record['id'])['chat'], 2)
    job = self.world.turn(record['id'], 'Hi')
    self.assertEqual(job['status'], 'completed', job['error'])
    self.assertEqual((job['chat'], job['turn']), (2, 1))

  def test_a_chat_whose_capture_is_not_made_is_over(self):
    """Kept in the record already, it never comes live: it says why."""
    record = self.world.create()
    self.world.start(record, None)
    with mock.patch.object(
        self.world.captures,
        'create',
        side_effect=OSError('The disk is full.'),
    ):
      job = self.world.wait(
          self.world.model_servers.start_chat(
              record['id'], world_lib.DEFAULT_CHAT
          )
      )
    self.assertEqual(
        (job['status'], job['error']), ('failed', 'The disk is full.')
    )
    chat = self._chat(record['id'])
    self.assertTrue(chat['over'])
    self.assertEqual(
        chat['notice'], f'The disk is full. {model_server.CHAT_OVER}'
    )
    summary = self.world.summary(record['id'])
    self.assertIsNone(summary['live_chat'])
    self.assertEqual(summary['chats'][0]['state'], 'ended')
    self.assertEqual(self.world.chat(record['id'])['chat'], 2)

  def test_one_turn_or_chat_start_at_a_time(self):
    """While a turn runs, neither another turn nor a new chat starts."""
    record = self.world.create()
    self.world.start(record)
    release = threading.Event()
    for script in self.world.scripts():
      script.hold = release
    job = self.world.model_servers.generate(record['id'], 'Hi')
    try:
      summary = self.world.summary(record['id'])
      busy = {'allowed': False, 'reason': model_server.BUSY}
      self.assertEqual(summary['actions']['new_chat'], busy)
      self.assertEqual(summary['chats'][0]['actions']['send'], busy)
      with self.assertRaisesRegex(errors.Conflict, model_server.BUSY):
        self.world.model_servers.start_chat(
            record['id'], world_lib.DEFAULT_CHAT
        )
    finally:
      release.set()
    self.assertEqual(self.world.wait(job)['status'], 'completed')
    self.assertLen(self.world.sessions.get(record['id'])['chats'], 1)
    # A chat's start is not stopped.
    hold = threading.Event()
    self.addCleanup(hold.set)
    for launched in self.world.launchers.running('local'):
      launched.runner.before = lambda method: (
          hold.wait(fake_launchers.TIMEOUT) if method == 'CreateChat' else None
      )
    job = self.world.model_servers.start_chat(
        record['id'], world_lib.DEFAULT_CHAT
    )
    with self.assertRaisesRegex(errors.Conflict, 'cannot be stopped'):
      self.world.model_servers.cancel(job['id'])
    with self.assertRaisesRegex(errors.Conflict, model_server.BUSY):
      self.world.model_servers.generate(record['id'], 'Hi')
    hold.set()
    self.assertEqual(self.world.wait(job)['status'], 'completed')

  def test_a_system_prompt_with_no_room_does_not_start_a_chat(self):
    """The runner's refusal says both counts and names the side."""
    record = self.world.create(ref='android:A', target='android:B')
    self.world.start(record)
    (target,) = self.world.launchers.runners('android:B')

    def before(method):
      if method == 'CreateChat':
        raise fake_runner.NoRoomError(10151, 4096)

    target.before = before
    with self.assertLogs(level='INFO') as logs:
      job = self.world.wait(
          self.world.model_servers.start_chat(
              record['id'], world_lib.DEFAULT_CHAT
          )
      )
    # A refusal, not a failure of the server's: INFO, never ERROR.
    (said,) = [
        line
        for line in logs.records
        if 'a chat did not start' in line.getMessage()
    ]
    self.assertEqual(said.levelname, 'INFO')
    self.assertEqual(job['status'], 'failed')
    self.assertEqual(
        job['error'],
        'Target: The system prompt and the chat template around it need '
        '10151 tokens; the context holds 4096. Shorten it and start '
        'again.',
    )
    self.assertIsNone(target.chat_request)
    summary = self.world.summary(record['id'])
    self.assertEqual(summary['execution']['phase'], 'active')
    self.assertTrue(summary['actions']['new_chat']['allowed'])
    target.before = None
    self.assertEqual(self.world.chat(record['id'])['chat'], 2)

  def test_one_side_failing_ends_the_chat_alone(self):
    """A side's failure cancels the other side; the chat is then over.

    The Session and its runners stay on: a new chat starts on them.
    """
    record = self.world.create(ref='android:A', target='android:B')
    self.world.start(record)
    release = threading.Event()
    (ref,) = self.world.launchers.runners('android:A')
    (target,) = self.world.launchers.runners('android:B')
    ref._respond.hold = release
    target._respond.refuse = grpc.StatusCode.INTERNAL
    try:
      job = self.world.turn(record['id'], 'Hi')
    finally:
      release.set()
    self.assertEqual(job['status'], 'failed')
    # No turn kept: Debug is not named.
    self.assertEqual(
        job['error'], f'Target: Refused turn 1. {model_server.CHAT_OVER}'
    )
    # Only the side that failed shows the turn failed.
    self.assertEqual(job['failed_sides'], ['target'])
    self.assertEqual(job['note'], '')
    world_lib.eventually(lambda: ref.chat_over and target.chat_over)
    self.assertEqual(ref.released, [])
    self.assertEqual(target.released, [])
    self._check_chat_ended(record['id'], job['error'])
    target._respond.refuse = None
    self.assertEqual(self.world.chat(record['id'])['chat'], 2)
    job = self.world.turn(record['id'], 'Hi')
    self.assertEqual(job['status'], 'completed', job['error'])
    self.assertEqual((job['chat'], job['turn']), (2, 1))
    self.assertEqual(target.released, [1])

  def test_a_turn_neither_side_has_room_for_does_not_happen(self):
    """Nothing is captured; the chat goes on as before the turn."""
    record = self.world.create(ref='android:A', target='android:B')
    self.world.start(record)
    (ref,) = self.world.launchers.runners('android:A')
    (target,) = self.world.launchers.runners('android:B')
    ref._respond.no_room = target._respond.no_room = (900, 700)
    job = self.world.turn(record['id'], 'Hi ' * 900)
    self.assertEqual(job['status'], 'failed')
    self.assertEqual(
        job['error'],
        "This input doesn't fit: this turn needs 900 tokens, the chat "
        'has 700 left. Shorten it and send again.',
    )
    self.assertEqual(
        job['does_not_fit'],
        {
            'ref': {'tokens': 900, 'left': 700},
            'target': {'tokens': 900, 'left': 700},
        },
    )
    self.assertFalse(ref.chat_over or target.chat_over)
    self.assertEqual(self.world.captures.get(record['id'], 1).turns, [])
    self.assertFalse(self._chat(record['id'])['over'])
    self.assertFalse(self._chat(record['id'])['notice'])
    summary = self.world.summary(record['id'])
    self.assertEqual(summary['execution']['phase'], 'active')
    self.assertTrue(summary['chats'][0]['actions']['send']['allowed'])

    ref._respond.no_room = target._respond.no_room = None
    job = self.world.turn(record['id'], 'Hi')
    self.assertEqual(job['status'], 'completed', job['error'])
    self.assertEqual(job['turn'], 1)
    self.assertEqual(ref.prompts, ['Hi'])
    self.assertEqual(target.prompts, ['Hi'])
    self.assertLen(self.world.captures.get(record['id'], 1).turns, 1)

  def test_sides_left_with_different_room_name_the_one_with_less(self):
    """The text gives the side with fewer tokens left, and names it."""
    record = self.world.create(ref='android:A', target='android:B')
    self.world.start(record)
    (ref,) = self.world.launchers.runners('android:A')
    (target,) = self.world.launchers.runners('android:B')
    ref._respond.no_room = (900, 700)
    target._respond.no_room = (950, 600)
    job = self.world.turn(record['id'], 'Hi ' * 900)
    self.assertEqual(job['status'], 'failed')
    self.assertEqual(
        job['error'],
        "This input doesn't fit: on the Target side this turn needs 950 "
        'tokens, the chat has 600 left. Shorten it and send again.',
    )
    self.assertFalse(self._chat(record['id'])['over'])

  def test_a_turn_one_side_has_no_room_for_ends_the_chat(self):
    """The other side's reply began: the sides are out of step."""
    record = self.world.create(ref='android:A', target='android:B')
    self.world.start(record)
    release = threading.Event()
    (ref,) = self.world.launchers.runners('android:A')
    (target,) = self.world.launchers.runners('android:B')
    ref._respond.hold = release
    target._respond.no_room = (900, 700)
    try:
      job = self.world.turn(record['id'], 'Hi')
    finally:
      release.set()
    self.assertEqual(job['status'], 'failed')
    self.assertEqual(
        job['error'],
        'Target: This turn needs 900 tokens, the chat has 700 left. '
        f'{model_server.CHAT_OVER}',
    )
    self.assertNotIn('does_not_fit', job)
    world_lib.eventually(lambda: ref.chat_over)
    self.assertFalse(target.chat_over)
    self._check_chat_ended(record['id'], job['error'])

  def test_a_reply_that_fills_the_context_ends_the_chat(self):
    """The turn is kept; the runner refuses any next one: the chat is over.

    The record keeps why, not the text, which is derived as it is shown.
    The Model Server stays on for another chat.
    """
    record = self.world.create()
    self.world.start(record)
    # 116 words leave room for 3 of the reply's 4 tokens in the context.
    job = self.world.turn(record['id'], ' '.join(['w'] * 116))
    self.assertEqual(job['status'], 'completed', job['error'])
    (turn,) = self.world.captures.get(record['id'], 1).turns
    self.assertEqual(
        {entry['stop_reason'] for entry in turn['sides'].values()},
        {'context_full'},
    )
    chat = self._chat(record['id'])
    self.assertTrue(chat['over'])
    self.assertEqual(
        chat['over_cause'],
        {'code': model_server.CONTEXT_FULL, 'turn': 1},
    )
    self.assertEqual(chat['notice'], '')
    tokens = self.world.config()['runs'][0]['contextLength']
    full = f'Context full ({tokens} tokens).'
    over = model_server.CHAT_OVER_CAPTURED
    notice = f'Reference: {full} Target: {full} {over}'
    self._check_chat_ended(record['id'], notice)
    # Debug reads the turn from the store.
    self.assertLen(self.world.captures.get(record['id'], 1).turns, 1)
    # A later text of the same ending shows on the chat as it is.
    with mock.patch.object(
        capture_index, 'ending', lambda fixed, entry: 'Full'
    ):
      self.assertEqual(
          self._chat_view(record['id'])['notice'],
          f'Reference: Full. Target: Full. {over}',
      )
    # The next chat starts on a context of its own.
    self.world.chat(record['id'])
    job = self.world.turn(record['id'], 'Hi')
    self.assertEqual(job['status'], 'completed', job['error'])

  def test_a_notice_kept_as_text_is_shown_as_it_is(self):
    """A record that kept its notice as text, as older ones did."""
    old = (
        'Reference: Context full (64 tokens). '
        + model_server.CHAT_OVER_CAPTURED
    )

    self.assertEqual(
        model_server.notice({'over': True, 'notice': old}, None), old
    )

  def test_a_full_context_whose_capture_cannot_be_read(self):
    """The fact still says why the chat is over, without the capture."""
    chat = {
        'has_capture': True,
        'over': True,
        'notice': '',
        'over_cause': {'code': model_server.CONTEXT_FULL, 'turn': 1},
    }

    self.assertEqual(
        model_server.notice(chat, None),
        f'A reply filled its context. {model_server.CHAT_OVER_CAPTURED}',
    )

  def test_a_session_that_ended_says_so_to_a_turn(self):
    """Turned off with its chat open: a turn is refused as a start is.

    The Session cannot be turned on again, so a turn does not say to turn
    it on; the listing's reason is the request's. Its live chat ended
    with it, saying so.
    """
    record = self.world.create()
    self.world.start(record)
    self.world.model_servers.close(record['id'])

    summary = self.world.summary(record['id'])
    ended = {'allowed': False, 'reason': model_server.ENDED}
    self.assertEqual(summary['actions']['new_chat'], ended)
    (chat,) = summary['chats']
    self.assertEqual(
        chat['actions']['send'],
        {'allowed': False, 'reason': model_server.CHAT_ENDED},
    )
    self.assertEqual(
        chat['notice'],
        f'The Model Server was turned off. {model_server.CHAT_OVER}',
    )
    self.assertIsNone(summary['model_server']['button'])
    with self.assertRaisesRegex(errors.Conflict, 'has ended'):
      self.world.model_servers.generate(record['id'], 'Hi')
    with self.assertRaisesRegex(errors.Conflict, 'has ended'):
      self.world.model_servers.start_chat(record['id'], world_lib.DEFAULT_CHAT)

  def test_cancel_stops_both_sides_and_ends_the_chat(self):
    """Cancelling a turn cancels both Sends; nothing of the turn is kept."""
    record = self.world.create()
    self.world.start(record)
    release = threading.Event()
    runners = self.world.launchers.runners('local')
    for script in self.world.scripts():
      script.hold = release
    job = self.world.model_servers.generate(record['id'], 'Write a poem')
    try:
      deadline = time.monotonic() + fake_launchers.TIMEOUT
      while self.world.jobs.get(job['id'])['output'] != {
          'ref': 'It',
          'target': 'It',
      }:
        self.assertLess(time.monotonic(), deadline)
        time.sleep(0.01)
      self.world.model_servers.cancel(job['id'])
      job = self.world.wait(job)
    finally:
      release.set()
    self.assertEqual(job['status'], 'cancelled')
    self.assertEqual(
        job['error'], 'Generation stopped. ' + model_server.CHAT_OVER
    )
    self.assertEqual(job['output'], {'ref': 'It', 'target': 'It'})
    # The job keeps what the dropped turn asked, for a page loaded again,
    # and says the turn has no debug data.
    self.assertEqual(job['prompt'], 'Write a poem')
    self.assertEqual(job['note'], jobs.STOPPED_TURN)
    self.assertEqual(job['status_text'], 'Generation stopped')
    world_lib.eventually(lambda: all(runner.chat_over for runner in runners))
    self.assertEqual(self.world.captures.get(record['id'], 1).turns, [])
    self._check_chat_ended(record['id'], job['error'])
    # A job that ended answers unchanged.
    self.assertEqual(
        self.world.model_servers.cancel(job['id'])['status'], 'cancelled'
    )

  def test_close_while_starting_stops_what_was_launched(self):
    """Turning off a Model Server that is starting leaves no runner."""
    record = self.world.create()
    go = threading.Event()
    original = self.world.launchers.launch

    def launch(device_id, build=None):
      go.wait(fake_launchers.TIMEOUT)
      return original(device_id, build)

    self.world.launchers.launch = launch
    job = self.world.model_servers.initialize(record['id'])
    self.world.model_servers.close(record['id'])
    summary = self.world.summary(record['id'])
    # Its runners may still be launching: it ends once they are stopped.
    self.assertEqual(summary['execution']['phase'], 'ending')
    go.set()
    job = self.world.wait(job)
    self.assertEqual(job['status'], 'failed')
    self.assertIn('turned off', job['error'])
    summary = self.world.listing.summary(self.world.sessions.get(record['id']))
    self.assertNotIn('execution', summary)
    self.assertEqual(summary['model_server']['text'], 'Ended · no turn')
    self.assertEqual(summary['model_server']['error'], job['error'])
    self.assertTrue(
        all(launched._stopped for launched in self.world.launchers.launched)
    )
    # A start turned off ends its Session too: it never starts again.
    self.world.launchers.launch = original
    with self.assertRaisesRegex(errors.Conflict, model_server.ENDED):
      self.world.model_servers.initialize(record['id'])

  def test_a_failed_start_ends_its_job_whatever_fails_after(self):
    """A start that failed ends its job, whatever fails after.

    Neither runners that do not stop nor a record that cannot be written
    leave the job of an initialize running.
    """
    record = self.world.create()

    def launch(device_id, build=None):
      raise RuntimeError('The runner did not start within 30 s.')

    original = self.world.launchers.launch
    self.world.launchers.launch = launch
    with mock.patch.object(
        model_server.ModelServer,
        'stop_runners',
        side_effect=RuntimeError('The runners did not stop.'),
    ):
      job = self.world.wait(self.world.model_servers.initialize(record['id']))
    self.assertEqual(job['status'], 'failed')
    self.assertIsNone(self.world.view(record['id']))

    self.world.launchers.launch = original
    record = self.world.create()
    set_flags = self.world.sessions.set_flags

    def full(record_id, **flags):
      if 'execution_ended' in flags or 'notice' in flags:
        raise OSError('The disk is full.')
      set_flags(record_id, **flags)

    with mock.patch.object(self.world.sessions, 'set_flags', full):
      job = self.world.wait(self.world.model_servers.initialize(record['id']))
    self.assertEqual(
        (job['status'], job['error']), ('failed', 'The disk is full.')
    )
    self.assertIsNone(self.world.view(record['id']))

  def test_close_stops_both_runners_for_good(self):
    """Close ends the execution; the Session shows ended."""
    record = self.world.create()
    self.world.start(record)
    self.world.model_servers.close(record['id'])
    summary = self.world.summary(record['id'])
    self.assertEqual(summary['execution']['phase'], 'ended')
    # It kept no turn: nothing to open.
    self.assertEqual(summary['status_text'], 'Ended · no turn')
    self.assertFalse(summary['actions']['enter']['allowed'])
    self.assertIsNone(summary['model_server']['button'])
    self.assertTrue(
        all(launched._stopped for launched in self.world.launchers.launched)
    )
    # Idempotent.
    self.world.model_servers.close(record['id'])
    self.assertEqual(self.world.view(record['id'])['phase'], 'ended')

  def test_close_while_generating_cancels_the_turn(self):
    """Turning off during a turn cancels it."""
    record = self.world.create()
    self.world.start(record)
    release = threading.Event()
    for script in self.world.scripts():
      script.hold = release
    job = self.world.model_servers.generate(record['id'], 'Hi')
    try:
      self.world.model_servers.close(record['id'])
      job = self.world.wait(job)
    finally:
      release.set()
    self.assertEqual(job['status'], 'cancelled')
    self.assertEqual(job['note'], jobs.STOPPED_TURN)
    self.assertEqual(self.world.view(record['id'])['phase'], 'ended')

  def test_initialize_refuses_a_device_not_connected(self):
    """A run's device must be there now."""
    record = self.world.create(target='android:C')
    with self.assertRaisesRegex(errors.Conflict, 'not connected'):
      self.world.model_servers.initialize(record['id'])

  def test_a_side_on_this_computer_without_a_runner_version_is_refused(self):
    """Create refuses it, and saves nothing; no launch."""
    config = self.world.config()
    del config['runs'][1]['runnerBuild']
    with self.assertRaises(errors.Conflict) as refused:
      self.world.model_servers.create(config)
    self.assertEqual(str(refused.exception), 'Target: choose a runner version.')
    self.assertEqual(self.world.sessions.list(), [])
    self.assertEqual(self.world.launchers.launched, [])
    # A phone runs the App installed: it needs none.
    self.world.start(self.world.create(ref='android:A', target='android:B'))

  def test_two_builds_run_on_this_computer_at_once(self):
    """Each side runs the build it chose; the capture records each."""
    self.world.launchers.builds_now['local'] = [
        fake_launchers.runner_build(fake_builds.OLD),
        fake_launchers.runner_build(fake_builds.NEW),
    ]
    config = self.world.config()
    config['runs'][0]['runnerBuild'] = fake_builds.OLD
    config['runs'][1]['runnerBuild'] = fake_builds.NEW
    record = self.world.sessions.create(config)
    self.world.start(record)
    self.assertEqual(
        sorted(launched.build for launched in self.world.launchers.launched),
        [fake_builds.OLD, fake_builds.NEW],
    )
    sides = self.world.captures.get(record['id'], 1).sides
    self.assertEqual(
        {side: sides[side]['runtime']['version'] for side in sides},
        {'ref': fake_builds.OLD, 'target': fake_builds.NEW},
    )
    runners = self.world.view(record['id'])['runners']
    self.assertEqual(
        [runner['buildId'] for runner in runners],
        [fake_builds.OLD, fake_builds.NEW],
    )

  def test_a_build_no_longer_there_is_refused_at_start(self):
    """The launcher refuses it, with its reason, each time a side starts."""
    record = self.world.create()
    self.world.launchers.builds_now['local'] = []
    job = self.world.wait(self.world.model_servers.initialize(record['id']))
    self.assertEqual(job['status'], 'failed')
    self.assertRegex(
        job['error'],
        f'^(Reference|Target): No launchable build {fake_launchers.COMMIT}',
    )
    self.assertTrue(
        all(launched._stopped for launched in self.world.launchers.launched)
    )
    # Kept again, a new Session of it starts.
    del self.world.launchers.builds_now['local']
    self.world.start(self.world.create())

  def test_a_phone_that_reports_another_commit_is_refused(self):
    """A side that chose a commit on a phone runs only that commit."""
    info = runner_pb2.RunnerInfo()
    info.CopyFrom(fake_runner.DEFAULT_INFO)
    info.runtime.version = fake_builds.NEW
    self.world.launchers.infos['android:A'] = info
    config = self.world.config(target='android:A')
    config['runs'][1]['runnerBuild'] = fake_builds.OLD
    record = self.world.sessions.create(config)
    job = self.world.wait(self.world.model_servers.initialize(record['id']))
    self.assertEqual(job['status'], 'failed')
    self.assertEqual(
        job['error'],
        'Target: Pixel runs LiteRT-LM 222222222222, but this Session chose '
        '111111111111; install the App built from LiteRT-LM 111111111111, '
        'or create a new Session and choose the device again.',
    )
    (runner,) = self.world.launchers.runners('android:A')
    self.assertTrue(runner.stopped)
    self.assertEqual(runner.uploads, [])

  def test_a_runner_lost_mid_turn_ends_the_session(self):
    """A runner that goes away ends the chat, and the Session with it."""
    record = self.world.create(ref='android:A', target='android:B')
    self.world.start(record)
    release = threading.Event()
    (ref,) = self.world.launchers.running('android:A')
    (target,) = self.world.launchers.running('android:B')
    ref.script.hold = target.script.hold = release
    job = self.world.model_servers.generate(record['id'], 'Hi')
    try:
      world_lib.eventually(
          lambda: self.world.jobs.get(job['id'])['output']['ref'] == 'It'
      )
      ref.runner.stop()
      job = self.world.wait(job)
    finally:
      release.set()
    self.assertEqual(job['status'], 'failed')
    self.assertTrue(job['error'].startswith('Reference: The connection to the'))
    self.assertTrue(self._chat(record['id'])['over'])
    self.assertEqual(self._chat(record['id'])['notice'], job['error'])
    # The other runner is stopped too, as soon as the connection says
    # the runner is gone.
    world_lib.eventually(
        lambda: self.world.view(record['id'])['phase'] == 'ended'
    )
    summary = self._check_turned_off(record['id'])
    self.assertEqual(summary['notice'], model_server.runner_gone('ref'))
    self.assertEqual(
        summary['chats'][0]['actions']['send'],
        {'allowed': False, 'reason': model_server.CHAT_ENDED},
    )

  def test_a_runner_that_exits_between_turns_ends_the_session(self):
    """A runner gone while idle: the Session ends, the other one stopped.

    Its connection says so, and the Model Server turns itself off: the
    Session's notice says which runner is gone, its live chat ends saying
    so, and a turn is refused as the Session ended, as the listing says.
    """
    record = self.world.create(ref='android:A', target='android:B')
    self.world.start(record)
    self.assertEqual(self.world.turn(record['id'], 'Hi')['status'], 'completed')
    (ref,) = self.world.launchers.running('android:A')
    (target,) = self.world.launchers.running('android:B')
    with self.assertLogs(level='INFO') as logs:
      ref.runner.stop()
      # Over adb only the connection tells: its reconnect fails at once,
      # so the UI's next refresh, 3 s later at most, shows it.
      stopped = time.monotonic()
      world_lib.eventually(
          lambda: self.world.view(record['id'])['phase'] == 'ended'
      )
    self.assertLess(time.monotonic() - stopped, 3)
    self.assertEqual(
        _errors(logs),
        [f'{_label(record)}: {model_server.runner_gone("ref")}'],
    )
    self.assertTrue(target._stopped)
    summary = self._check_turned_off(record['id'])
    notice = model_server.runner_gone('ref')
    self.assertEqual(
        notice,
        'Runner closed: the connection to the Reference runner is lost.',
    )
    self.assertEqual(summary['notice'], notice)
    (chat,) = summary['chats']
    self.assertEqual(
        chat['notice'], f'{notice[:-1]}. {model_server.CHAT_OVER_CAPTURED}'
    )
    self.assertEqual(
        chat['actions']['send'],
        {'allowed': False, 'reason': model_server.CHAT_ENDED},
    )
    self.assertTrue(summary['actions']['enter']['allowed'])
    with self.assertRaisesRegex(errors.Conflict, 'has ended'):
      self.world.model_servers.generate(record['id'], 'Hi')
    # Its turn is still there to inspect.
    self.assertLen(self.world.captures.get(record['id'], 1).turns, 1)

  def test_a_runner_seen_to_exit_refuses_the_next_turn_at_once(self):
    """Runners of this computer gone: no wait for their connections.

    A turn is refused at once; the chat then ends, as the first side
    gone says.
    """
    record = self.world.create()
    self.world.start(record)
    for launched in self.world.launchers.running('local'):
      launched.runner.stop()
    self.assertFalse(
        self._chat_view(record['id'])['actions']['send']['allowed']
    )
    world_lib.eventually(
        lambda: self.world.view(record['id'])['phase'] == 'ended'
    )
    summary = self._check_turned_off(record['id'])
    self.assertEqual(summary['notice'], model_server.runner_gone('ref'))
    self.assertEqual(
        summary['chats'][0]['notice'],
        f'{model_server.runner_gone("ref")[:-1]}. {model_server.CHAT_OVER}',
    )

  def test_dropped_connections_of_healthy_runners_refuse_no_turn(self):
    """A connection lost, the runner still there, is no runner gone.

    As gRPC closing an idle connection: the channel reconnects, and a turn
    sent meanwhile is admitted and completes.
    """
    record = self.world.create(ref='android:A', target='local')
    self.world.start(record)
    runners = [
        *self.world.launchers.runners('android:A'),
        *self.world.launchers.runners('local'),
    ]
    for _ in range(3):
      for runner in runners:
        runner.drop_connections()
      deadline = time.monotonic() + 0.5  # The reconnect takes 0.2 s.
      while time.monotonic() < deadline:
        view = self.world.view(record['id'])
        self.assertTrue(all(side['connected'] for side in view['runners']))
        self.assertTrue(
            self._chat_view(record['id'])['actions']['send']['allowed']
        )
        time.sleep(0.01)
    for runner in runners:
      runner.drop_connections()
    self.assertEqual(self.world.turn(record['id'], 'Hi')['status'], 'completed')
    self.assertEqual(self.world.view(record['id'])['phase'], 'active')

  def test_a_failed_turn_leaves_the_runners_on(self):
    """The chat is over; its runners are not gone, and stay on.

    The Session's notice says nothing, its chat's says why; and the
    devices stay held, the Model Server on.
    """
    record = self.world.create(ref='android:A', target='android:B')
    self.world.start(record)
    for script in self.world.scripts('android:B'):
      script.refuse = grpc.StatusCode.INTERNAL
    job = self.world.turn(record['id'], 'Hi')
    self.assertEqual(job['status'], 'failed')
    time.sleep(3 * model_server._WATCH_SECONDS)
    self._check_chat_ended(record['id'], job['error'])
    with self.assertRaisesRegex(errors.Conflict, 'in use'):
      self.world.model_servers.create(
          self.world.config('android:A', 'android:B')
      )

  def test_a_session_turned_off_frees_its_devices_at_once(self):
    """Another Session starts there; it never runs again itself."""
    record = self.world.create(ref='android:A', target='android:B')
    self.world.start(record)
    self.world.model_servers.close(record['id'])
    self._check_turned_off(record['id'])
    other = self.world.create(ref='android:A', target='android:B')
    self.world.start(other)
    self.assertEqual(self.world.turn(other['id'], 'Hi')['status'], 'completed')
    with self.assertRaisesRegex(errors.Conflict, 'has ended'):
      self.world.model_servers.initialize(record['id'])

  def test_while_a_runner_gone_turns_off_its_sides_say_so(self):
    """Between a runner gone mid-turn and both stopped: Ending, not Ready.

    The turn's job ends only once the Model Server is off, so a page that
    sees the job end sees it off.
    """
    record = self.world.create(ref='local', target='android:B')
    self.world.start(record)
    release, stopping = threading.Event(), threading.Event()
    self.addCleanup(release.set)
    (ref,) = self.world.launchers.running('local')
    (target,) = self.world.launchers.running('android:B')
    stop = target.stop

    def held_stop():
      stopping.set()
      release.wait(fake_launchers.TIMEOUT)
      stop()

    target.stop = held_stop
    ref.script.hold = threading.Event()
    self.addCleanup(ref.script.hold.set)
    job = self.world.model_servers.generate(record['id'], 'Hi')
    world_lib.eventually(
        lambda: self.world.jobs.get(job['id'])['output']['ref'] == 'It'
    )
    # This computer sees its runner's process exit at once.
    ref.runner.stop()
    self.assertTrue(stopping.wait(fake_launchers.TIMEOUT))
    summary = self.world.summary(record['id'])
    self.assertEqual(summary['execution']['phase'], 'ending')
    self.assertEqual(
        [side['phaseText'] for side in summary['execution']['runners']],
        ['Ending', 'Ending'],
    )
    self.assertFalse(summary['chats'][0]['actions']['send']['allowed'])
    self.assertNotIn(self.world.jobs.get(job['id'])['status'], jobs.TERMINAL)
    release.set()
    self.assertEqual(self.world.wait(job)['status'], 'failed')
    self._check_turned_off(record['id'])

  def test_a_runner_that_does_not_stop_still_turns_the_server_off(self):
    """A failure while turning off is logged; the Model Server ends.

    The other runner is stopped all the same.
    """
    record = self.world.create(ref='android:A', target='android:B')
    self.world.start(record)
    (ref,) = self.world.launchers.running('android:A')
    (target,) = self.world.launchers.running('android:B')

    def broken_stop():
      raise RuntimeError('adb did not answer.')

    ref.stop = broken_stop
    with self.assertLogs(level='ERROR') as logs:
      self.world.model_servers.close(record['id'])
    self.assertIn('adb did not answer.', '\n'.join(logs.output))
    self.assertTrue(target._stopped)
    summary = self.world.summary(record['id'])
    self.assertEqual(summary['execution']['phase'], 'ended')
    self.assertEqual(summary['model_server']['state'], 'ended')
    self.assertTrue(self._chat(record['id'])['over'])
    # Nor does a failure of the whole stop keep it on.
    other = self.world.create()
    self.world.start(other)
    with mock.patch.object(
        model_server.ModelServer,
        'stop_runners',
        side_effect=RuntimeError('The runners did not stop.'),
    ):
      self.world.model_servers.close(other['id'])
    self.assertEqual(self.world.view(other['id'])['phase'], 'ended')

  def test_chat_options_a_runner_refuses_start_no_chat(self):
    """Settings a runner refuses are refused at once; nothing changes.

    The live chat goes on, and no CreateChat is sent.
    """
    info = runner_pb2.RunnerInfo()
    info.CopyFrom(fake_runner.DEFAULT_INFO)
    info.runtime.chat_options_schema = (
        '{"type": "object", "properties": {}, "additionalProperties": false}'
    )
    _with_runner(
        self.world, lambda launched: setattr(launched.runner, 'info', info)
    )
    record = self.world.sessions.create(self.world.config())
    self.world.start(record)
    with self.assertRaisesRegex(ValueError, '^Reference: .*thinking'):
      self.world.model_servers.start_chat(
          record['id'], {'ref': {'thinking': 'on'}, 'target': {}}
      )
    with self.assertRaisesRegex(ValueError, 'Speculative decoding'):
      self.world.model_servers.start_chat(
          record['id'],
          {'ref': {}, 'target': {'speculativeDecoding': 'on'}},
      )
    summary = self.world.summary(record['id'])
    self.assertEqual(summary['live_chat'], 1)
    self.assertLen(summary['chats'], 1)
    for runner in self.world.launchers.runners('local'):
      self.assertLen(runner.chats, 1)

  def _top_k_default(self, record_id):
    """What the chat settings say Top-K + Top-P's top_k defaults to."""
    (top_k,) = [
        option
        for option in self.world.catalog.chat_capabilities(record_id)['target'][
            'chat_options'
        ]
        if option['key'] == 'topK'
    ]
    return top_k['defaults']['topP']

  def test_top_k_top_p_sends_the_values_the_record_keeps(self):
    """The record keeps the top_k shown, and the runner gets it."""
    top_k_top_p = {
        'samplerType': 'topP',
        'topP': 0.9,
        'temperature': 0.8,
        'seed': 7,
    }
    record = self.world.create()
    # No runner loaded the model yet: the table's 64.
    self.assertEqual(
        self._top_k_default(record['id'])['value'],
        options.MODEL_DEFAULTS['topP']['topK'][1],
    )
    self.assertEqual(
        self.world.sessions.check_generation(
            record['id'], {'ref': {}, 'target': top_k_top_p}
        )['target']['topK'],
        64,
    )
    # Once a runner loaded it, its own: the chat's settings keep it.
    self.world.start(record, None)
    self.world.chat(record['id'], {'ref': {}, 'target': top_k_top_p})
    top_k = capture_fixtures.SAMPLER_DEFAULTS.top_k
    self.assertEqual(
        self._chat(record['id'])['generation']['target']['topK'], top_k
    )
    index = self.world.captures.get(record['id'], 1)
    self.assertEqual(
        index.sides['target']['chat']['options'],
        {
            # No reply limit is set: the reply may fill the context.
            'max_completion_tokens': index.sides['target']['load'][
                'context_length'
            ],
            'top_k': top_k,
            'top_p': 0.9,
            'temperature': 0.8,
            'seed': 7,
        },
    )
    self.assertEqual(index.sides['ref']['chat']['options']['top_k'], 1)
    # The model's own, as its Load reported it, is kept with the model
    # and is the default from now on.
    self.assertEqual(index.sides['target']['sampler_defaults']['top_k'], top_k)
    self.assertEqual(
        self.world.models.get(self.world.model.artifact).sampler_defaults[
            'top_k'
        ],
        top_k,
    )
    self.assertEqual(
        self._top_k_default(record['id']),
        {
            'value': top_k,
            'source': "the model's own, saved with these settings",
        },
    )

  def test_top_k_top_p_on_a_model_that_reports_no_top_k(self):
    """A model with no default top_k runs the table's, and it says so."""
    loaded = capture_fixtures.TinyChat().load_response()
    loaded.ClearField('sampler_defaults')
    self.world.launchers.loaded['local'] = loaded
    record = self.world.create()
    self.world.start(
        record, {'ref': {}, 'target': {'samplerType': 'topP', 'topP': 0.9}}
    )
    sides = self.world.captures.get(record['id'], 1).sides
    self.assertEqual(sides['target']['chat']['options']['top_k'], 64)
    self.assertEqual(
        self._top_k_default(record['id']),
        {
            'value': 64,
            'source': (
                "the Model Debugger's own, as the model has none; "
                'saved with these settings'
            ),
        },
    )

  def test_top_k_top_p_without_a_saved_top_k_runs_the_default_shown(self):
    """A chat that kept no top_k runs the one the dialog shows."""
    self.world.models.report_loaded(
        self.world.model.artifact,
        capture_fixtures.TinyChat().load_response(),
    )
    record = self.world.create()
    top_k = capture_fixtures.SAMPLER_DEFAULTS.top_k
    self.assertNotEqual(top_k, options.MODEL_DEFAULTS['topP']['topK'][1])
    self.assertEqual(self._top_k_default(record['id'])['value'], top_k)
    self.world.start(record, None)
    with self.world.model_servers._lock:
      model_server_ = self.world.model_servers._model_servers[record['id']]
      # As settings saved before they always kept topK have them.
      requests = model_server_.chat_requests(
          {'ref': {}, 'target': {'samplerType': 'topP', 'topP': 0.9}}
      )
    self.assertEqual(requests['target'].options.top_k, top_k)

  def test_a_typed_top_k_reaches_the_index_unchanged(self):
    """The largest top_k values typed are what index.json records."""
    record = self.world.create()
    self.world.start(record, None)
    for n, top_k in enumerate((262144, 2**31 - 1), 1):
      self.world.chat(
          record['id'],
          {
              'ref': {},
              'target': {
                  'samplerType': 'topP',
                  'topK': top_k,
                  'topP': 0.9,
              },
          },
      )
      index = self.world.captures.get(record['id'], n)
      stored = json.loads((index.root / 'index.json').read_text())
      self.assertEqual(
          stored['sides']['target']['chat']['options']['top_k'], top_k
      )

  def test_gpu_a_runner_reports_none_of_fails_before_any_upload(self):
    """GPU on a simulator whose runner reports no GPU fails before Load."""
    simulator = launcher.Device(
        'ios-simulator:S',
        'iPhone 17',
        'iOS Simulator',
        'simulator',
        False,
    )
    made = world_lib.World(self.tmp_path, [fake_launchers.LOCAL, simulator])
    try:
      info = runner_pb2.RunnerInfo()
      info.CopyFrom(fake_runner.DEFAULT_INFO)
      info.host.ClearField('gpu')
      made.launchers.infos[simulator.id] = info
      config = made.config(ref=simulator.id)
      config['runs'][0]['backend'] = 'GPU'
      record = made.sessions.create(config)
      job = made.wait(made.model_servers.initialize(record['id']))
      self.assertEqual(job['status'], 'failed')
      self.assertEqual(
          job['error'],
          'Reference: iPhone 17 cannot run this runner on a GPU; choose CPU.',
      )
      (runner,) = made.launchers.runners(simulator.id)
      self.assertEqual(runner.uploads, [])
      self.assertIsNone(runner.load_request)
    finally:
      made.close()

  def test_stop_while_retrieving_captures_drops_the_turn(self):
    """A Stop while captures come drops the turn and ends the chat."""
    release, reading = threading.Event(), threading.Event()
    _hold_reads(self.world, release, reading)
    record = self.world.create()
    self.world.start(record)
    job = self.world.model_servers.generate(record['id'], 'Hi')
    try:
      self.assertTrue(reading.wait(fake_launchers.TIMEOUT))
      self.world.model_servers.cancel(job['id'])
      job = self.world.wait(job)
    finally:
      release.set()
    self.assertEqual(job['status'], 'cancelled')
    self.assertEqual(job['note'], jobs.STOPPED_TURN)
    self.assertEqual(
        job['error'], 'Generation stopped. ' + model_server.CHAT_OVER
    )
    index = self.world.captures.get(record['id'], 1)
    self.assertEqual(index.turns, [])
    self.assertFalse(list(index.root.glob('*/turn-1/*')))
    for runner in self.world.launchers.runners('local'):
      self.assertEqual(runner.released, [])
      # The turn's capture stays on the runner until the next chat.
      self.assertEqual(list(runner.captures), [1])
    self._check_chat_ended(record['id'], job['error'])
    self.world.chat(record['id'])
    for runner in self.world.launchers.runners('local'):
      self.assertEqual(runner.captures, {})

  def test_a_capture_its_disk_has_no_room_for_fails_the_turn(self):
    """Nothing is written, nothing released; the reason names the disk."""
    record = self.world.create()
    self.world.start(record)
    free = capture_writer.SPARE_BYTES // 2

    def disk_usage(path):
      del path  # Every disk alike.
      return types.SimpleNamespace(total=2 * free, used=free, free=free)

    with (
        mock.patch.object(shutil, 'disk_usage', disk_usage),
        self.assertLogs(level='INFO') as logs,
    ):
      job = self.world.turn(record['id'], 'Hi')
    self.assertEqual(job['status'], 'failed')
    reason = (
        "Not enough disk space for this turn's capture: it needs at "
        'least 1 GB more, 1 GB of it kept spare; the disk holding '
        f'{self.world.workspace.root} has 0.5 GB free'
    )
    self.assertIn(reason, job['error'])
    self.assertTrue(job['error'].endswith(model_server.CHAT_OVER))
    self.assertEqual(
        _errors(logs),
        [f'{_label(record)}: Chat 1 turn 1 failed: {job["error"]}'],
    )
    index = self.world.captures.get(record['id'], 1)
    self.assertEqual(index.turns, [])
    self.assertFalse(list(index.root.glob('*/turn-1')))
    for runner in self.world.launchers.runners('local'):
      self.assertEqual(runner.released, [])
    self._check_chat_ended(record['id'], job['error'])

  def test_a_disk_that_fills_mid_capture_fails_the_turn_leaving_nothing(
      self,
  ):
    """ENOSPC mid-file: the reason is the disk's; no file of it stays."""
    record = self.world.create()
    self.world.start(record)
    opened = pathlib.Path.open
    written = []

    class FullDisk:
      """A .part file whose disk fills after the first bytes written."""

      def __init__(self, file):
        self._file = file

      def write(self, data):
        if written:
          raise OSError(errno.ENOSPC, 'No space left on device')
        written.append(len(data))
        return self._file.write(data)

      def close(self):
        self._file.close()

    def open_full(path, *args, **kwargs):
      file = opened(path, *args, **kwargs)
      return FullDisk(file) if path.name.endswith('.part') else file

    with mock.patch.object(pathlib.Path, 'open', open_full):
      job = self.world.turn(record['id'], 'Hi')
    self.assertEqual(job['status'], 'failed')
    self.assertIn(
        "Not enough disk space for this turn's capture: it needs at least",
        job['error'],
    )
    self.assertIn(
        f'the disk holding {self.world.workspace.root} has', job['error']
    )
    index = self.world.captures.get(record['id'], 1)
    self.assertEqual(index.turns, [])
    self.assertFalse(list(index.root.glob('*/turn-1')))
    for runner in self.world.launchers.runners('local'):
      self.assertEqual(runner.released, [])
    self._check_chat_ended(record['id'], job['error'])


if __name__ == '__main__':
  test_main.main()
