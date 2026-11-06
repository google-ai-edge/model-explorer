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

"""Tests of the listing of Sessions.

Every refused action is checked against its real request: the reason the
listing shows is the text the request is refused with.
"""

import pathlib
import threading
from typing import Any

from absl.testing import absltest
from model_debugger.server.service import session_listing
from model_debugger.server.session import model_server
from model_debugger.server.testing import fake_launchers
from model_debugger.server.testing import test_main
from model_debugger.server.testing import world as world_lib

_ALLOWED = {'allowed': True, 'reason': ''}
# Generation settings that are valid: each side's defaults.
_SETTINGS = {'ref': {}, 'target': {}}
# Settings a user chose: a system prompt, and another sampler on one side.
_CHOSEN = {
    'ref': {'systemPrompt': 'Be terse.', 'samplerType': 'topK', 'topK': 40},
    'target': {'systemPrompt': 'Be terse.'},
}


class SessionListingTest(absltest.TestCase):
  """Tests of the listing of Sessions."""

  def setUp(self):
    """A computer and two devices."""
    super().setUp()
    self.world = world_lib.create(self)

  def _actions(self, record_id: str) -> dict[str, Any]:
    return self.world.summary(record_id)['actions']

  def _chat(self, record_id: str, n: int = 1) -> dict[str, Any]:
    """A chat as the listing shows it."""
    return self.world.summary(record_id)['chats'][n - 1]

  def _connected(self, record_id: str) -> list[bool]:
    return [
        runner['connected'] for runner in self.world.view(record_id)['runners']
    ]

  def _refused(self, action: dict[str, Any], call, *args) -> None:
    """Checks that `action` is refused with what `call(*args)` raises."""
    error = world_lib.raised(call, *args)
    self.assertIsNotNone(error, 'The request was not refused.')
    self.assertEqual(
        {key: action[key] for key in ('allowed', 'reason')},
        {'allowed': False, 'reason': str(error)},
    )

  def _held_turn(
      self, record: dict[str, Any]
  ) -> tuple[dict[str, Any], threading.Event]:
    """Starts a turn whose replies wait for the event returned."""
    release = threading.Event()
    self.addCleanup(release.set)
    for device_id in ('android:A', 'android:B'):
      (launched,) = self.world.launchers.running(device_id)
      launched.script.hold = release
    job = self.world.model_servers.generate(record['id'], 'Hi')
    world_lib.eventually(
        lambda: self.world.jobs.get(job['id'])['output']['ref'] == 'It'
    )
    return job, release

  # ---- What the listing asks.

  def test_the_listing_asks_no_device(self):
    """Polling the listing asks no device: runners and devices are kept."""
    record = self.world.create(ref='android:A', target='local')
    self.world.start(record)
    world_lib.eventually(lambda: self._connected(record['id']) == [True] * 2)
    listed = []
    devices = self.world.launchers.devices

    def counted():
      listed.append(1)
      return devices()

    def alive():
      raise AssertionError('The listing asked a device.')

    self.world.launchers.devices = counted
    # A runner of this computer is seen without asking anything; one on
    # a phone could be seen only by asking it.
    for launched in self.world.launchers.launched:
      if not launched._sees_process():
        launched._alive = alive
    for _ in range(5):
      self.world.listing.listing()
    # Listed when the Session started, and kept since.
    self.assertEqual(listed, [])
    self.world.devices.devices()
    self.assertEqual(listed, [1])

  def test_a_runner_is_online_as_its_connection_says(self):
    """A runner gone shows offline, without asking its device."""
    record = self.world.create(ref='android:A', target='android:B')
    self.world.start(record)
    world_lib.eventually(lambda: self._connected(record['id']) == [True] * 2)
    (launched,) = self.world.launchers.running('android:A')
    launched._alive = lambda: True  # The device would say it runs.
    launched.runner.stop()
    world_lib.eventually(lambda: self._connected(record['id']) == [False, True])

  # ---- Each side's phase and progress.

  def test_each_side_shows_its_phase_and_latest_progress(self):
    """While starting, each side shows what it does and its last line."""
    record = self.world.create(ref='android:A', target='android:B')
    launch = self.world.launchers.launch
    release, waiting = threading.Event(), threading.Event()
    self.addCleanup(release.set)

    def held(device_id, build=None):
      if device_id == 'android:A':
        waiting.set()
        release.wait(fake_launchers.TIMEOUT)
      return launch(device_id, build)

    self.world.launchers.launch = held
    job = self.world.model_servers.initialize(record['id'])
    self.assertTrue(waiting.wait(fake_launchers.TIMEOUT))
    # The target waits for the reference before it loads anything.
    world_lib.eventually(
        lambda: (
            self.world.view(record['id'])['runners'][1]['phase'] == 'connecting'
        )
    )
    ref, target = self.world.view(record['id'])['runners']
    self.assertEqual(
        (ref['phase'], ref['phaseText'], ref['progress']),
        ('starting', 'Starting', 'Starting runner'),
    )
    self.assertEqual(
        (target['phaseText'], target['progress']),
        ('Connecting', 'Starting runner'),
    )
    # What an upload says, as the job's latest line for its side.
    self.world.jobs.emit(
        job['id'], 'progress', runId='ref', message='Uploading model · 45%'
    )
    ref, _ = self.world.view(record['id'])['runners']
    self.assertEqual(ref['progress'], 'Uploading model · 45%')
    summary = self.world.summary(record['id'])
    self.assertEqual(summary['status_text'], 'Starting')
    self.assertEqual(summary['model_server']['state'], 'starting')
    self.assertEqual(
        summary['model_server']['button'],
        {'label': 'Cancel', 'allowed': True, 'reason': ''},
    )
    self.assertEqual(
        summary['actions']['enter'],
        {'allowed': False, 'reason': 'The Model Server is starting.'},
    )
    self.assertEqual(
        summary['actions']['new_chat'],
        {'allowed': False, 'reason': model_server.STARTING},
    )
    release.set()
    self.assertEqual(self.world.wait(job)['status'], 'completed')
    for runner in self.world.view(record['id'])['runners']:
      self.assertEqual((runner['phaseText'], runner['progress']), ('Ready', ''))

  # ---- The Model Server switch.

  def test_create_is_refused_as_a_start_is(self):
    """A device another Session holds, or one not connected: no record."""
    self.world.start(self.world.create(target='android:A'))
    for target, reason in (
        ('android:A', 'in use by another Session'),
        ('android:C', 'not connected'),
    ):
      with self.assertRaisesRegex(Exception, reason):
        self.world.model_servers.create(self.world.config(target=target))
    self.assertLen(self.world.listing.listing()['sessions'], 1)

  def test_an_ended_session_cannot_be_turned_on_again(self):
    """Once ended, it never runs again: initialize's own words."""
    record = self.world.create()
    self.world.start(record)
    on = self.world.summary(record['id'])['model_server']
    self.assertEqual((on['state'], on['text']), ('on', 'On'))
    self.assertEqual(on['button']['label'], 'Turn off')
    self.assertTrue(on['button']['allowed'])
    self.world.model_servers.close(record['id'])
    summary = self.world.summary(record['id'])
    self.assertEqual(summary['status_text'], 'Ended · no turn')
    self.assertEqual(
        (summary['model_server']['state'], summary['model_server']['text']),
        ('ended', 'Ended · no turn'),
    )
    self.assertIsNone(summary['model_server']['button'])
    with self.assertRaisesRegex(Exception, model_server.ENDED):
      self.world.model_servers.initialize(record['id'])

  def test_a_failed_start_ends_the_session(self):
    """No Try again: a start that failed ends with no turn, and says why.

    It is not entered; another try is a new Session.
    """
    record = self.world.create()

    def launch(device_id, build=None):
      raise RuntimeError('The runner did not start within 30 s.')

    self.world.launchers.launch = launch
    job = self.world.wait(self.world.model_servers.initialize(record['id']))
    self.assertEqual(job['status'], 'failed')
    summary = self.world.summary(record['id'])
    self.assertEqual(summary['status_text'], 'Ended · no turn')
    self.assertNotIn('execution', summary)
    server_info = summary['model_server']
    self.assertEqual(
        (server_info['state'], server_info['text']),
        ('ended', 'Ended · no turn'),
    )
    self.assertEqual(server_info['error'], job['error'])
    self.assertIn('did not start', server_info['error'])
    self.assertIsNone(server_info['button'])
    with self.assertRaisesRegex(Exception, 'This Session has ended.'):
      self.world.model_servers.initialize(record['id'])
    enter = summary['actions']['enter']
    self.assertFalse(enter['allowed'])
    self.assertIn('create a new Session', enter['reason'])
    self.assertTrue(summary['actions']['delete']['allowed'])

  def test_a_start_whose_graph_failed_ends_the_session_too(self):
    """A start a graph conversion failed ends as any failed start does.

    The world's own model is no .litertlm: its graph cannot be
    converted, which fails the start before any runner launches.
    """
    world = world_lib.create(self, graphs=True)
    record = world.create()
    job = world.wait(world.model_servers.initialize(record['id']))
    self.assertEqual(job['status'], 'failed')
    self.assertIn('could not be converted', job['error'])
    server_info = world.summary(record['id'])['model_server']
    self.assertEqual(
        (server_info['state'], server_info['error']),
        ('ended', job['error']),
    )
    self.assertIsNone(server_info['button'])

  def test_after_a_restart_a_session_that_started_has_ended(self):
    """With no turn it is not entered; with turns, Debug.

    The records as a server that stopped meanwhile left them: one whose
    start was under way or that never took a turn, and one that ran.
    """
    started = self.world.create()
    self.world.sessions.set_flags(started['id'], started=True)
    summary = self.world.summary(started['id'])
    self.assertEqual(summary['status_text'], 'Ended · no turn')
    self.assertEqual(summary['model_server']['text'], 'Ended · no turn')
    self.assertEqual(summary['model_server']['error'], '')
    self.assertFalse(summary['actions']['enter']['allowed'])
    self.assertNotIn('started', summary)

    ran = self.world.create()
    self.world.start(ran)
    job = self.world.turn(ran['id'], 'Hi')
    self.assertEqual(job['status'], 'completed', job['error'])
    restarted = world_lib.World(
        pathlib.Path(self.create_tempdir().full_path),
        [fake_launchers.LOCAL],
    )
    self.addCleanup(restarted.close)
    listing = session_listing.SessionListing(
        self.world.sessions,
        restarted.model_servers,
        restarted.jobs,
        self.world.captures,
        self.world.prepared,
    )
    summary = listing.summary(self.world.sessions.get(ran['id']))
    self.assertEqual(summary['status_text'], 'Ended')
    self.assertEqual(
        (summary['model_server']['state'], summary['model_server']['text']),
        ('ended', 'Ended'),
    )
    self.assertEqual(summary['model_server']['error'], '')
    self.assertTrue(summary['actions']['enter']['allowed'])
    # Its chat ended with the server: read, never sent to again.
    (chat,) = summary['chats']
    self.assertEqual(
        (chat['state'], chat['notice']), ('ended', model_server.ENDED)
    )
    self.assertTrue(chat['actions']['debug']['allowed'])
    self.assertFalse(chat['actions']['send']['allowed'])
    self.assertIsNone(summary['live_chat'])

  def test_a_session_being_turned_off_refuses_another_close(self):
    """While ending, the button is refused as close refuses it."""
    record = self.world.create(ref='android:A', target='android:B')
    self.world.start(record)
    release, stopping = threading.Event(), threading.Event()
    self.addCleanup(release.set)
    (launched,) = self.world.launchers.running('android:A')
    stop = launched.stop

    def held_stop():
      stopping.set()
      release.wait(fake_launchers.TIMEOUT)
      stop()

    launched.stop = held_stop
    closing = threading.Thread(
        target=self.world.model_servers.close, args=(record['id'],)
    )
    closing.start()
    self.addCleanup(closing.join, fake_launchers.TIMEOUT)
    self.assertTrue(stopping.wait(fake_launchers.TIMEOUT))
    summary = self.world.summary(record['id'])
    self.assertEqual(summary['status_text'], 'Ending')
    button = summary['model_server']['button']
    self.assertEqual(button['label'], 'Turn off')
    self._refused(button, self.world.model_servers.close, record['id'])
    release.set()
    closing.join(fake_launchers.TIMEOUT)
    # Ended, a close does nothing, and there is no button any more.
    self.world.model_servers.close(record['id'])
    self.assertEqual(self.world.view(record['id'])['phase'], 'ended')

  def test_a_capture_that_cannot_be_read_refuses_only_its_chat(self):
    """An unreadable capture refuses Debug of its chat; the rest list."""
    record = self.world.create()
    self.world.start(record)
    self.assertEqual(self.world.turn(record['id'], 'Hi')['status'], 'completed')
    broken = self.world.create()
    self.world.sessions.set_flags(
        broken['id'], started=True, execution_ended=True
    )
    for n in (1, 2):
      self.world.sessions.add_chat(broken['id'], _SETTINGS)
      self.world.sessions.set_chat_flags(
          broken['id'], n, has_capture=True, over=True
      )
    index = (
        self.world.workspace.captures / broken['id'] / 'chat-2' / 'index.json'
    )
    index.parent.mkdir(parents=True)
    index.write_text('[]')  # Not an object: no refusal, a fault.
    listed = {
        summary['id']: summary
        for summary in self.world.listing.listing()['sessions']
    }
    self.assertEqual(listed[broken['id']]['actions']['enter'], _ALLOWED)
    first, second = listed[broken['id']]['chats']
    self.assertFalse(second['actions']['debug']['allowed'])
    self.assertIn(
        'capture cannot be read', second['actions']['debug']['reason']
    )
    # Its own capture is missing: that is a refusal of its own.
    self.assertEqual(
        first['actions']['debug'],
        {'allowed': False, 'reason': 'This chat has no capture.'},
    )
    self.assertEqual(listed[record['id']]['actions']['enter'], _ALLOWED)
    self.assertEqual(
        listed[record['id']]['chats'][0]['actions']['debug'], _ALLOWED
    )

  def test_a_session_not_on_starts_no_chat(self):
    """No Model Server: a new chat is refused as it would be."""
    record = self.world.create()
    actions = self._actions(record['id'])
    self._refused(
        actions['new_chat'],
        self.world.model_servers.start_chat,
        record['id'],
        _SETTINGS,
    )
    self.assertEqual(actions['new_chat']['reason'], model_server.NOT_ON)
    with self.assertRaisesRegex(Exception, model_server.NOT_ON):
      self.world.model_servers.generate(record['id'], 'Hi')

  def test_a_running_session_refuses_changes(self):
    """While on: delete is refused as it would be."""
    record = self.world.create()
    self.world.start(record)
    actions = self._actions(record['id'])
    for name in ('enter', 'new_chat'):
      self.assertEqual(actions[name], _ALLOWED, name)
    self.assertEqual(self._chat(record['id'])['actions']['send'], _ALLOWED)
    self._refused(
        actions['delete'],
        self.world.model_servers.require_idle,
        record['id'],
    )

  def test_a_turn_in_flight_refuses_another_and_a_new_chat(self):
    """A job runs: send and new chat wait, with their requests' words."""
    record = self.world.create(ref='android:A', target='android:B')
    self.world.start(record)
    job, release = self._held_turn(record)
    summary = self.world.summary(record['id'])
    self.assertEqual(summary['status_text'], 'Active')
    for runner in summary['execution']['runners']:
      self.assertEqual(runner['phaseText'], 'Generating')
    self._refused(
        summary['chats'][0]['actions']['send'],
        self.world.model_servers.generate,
        record['id'],
        'Hi',
    )
    self._refused(
        summary['actions']['new_chat'],
        self.world.model_servers.start_chat,
        record['id'],
        _SETTINGS,
    )
    release.set()
    self.assertEqual(self.world.wait(job)['status'], 'completed')
    self.assertEqual(self._chat(record['id'])['actions']['send'], _ALLOWED)
    self.assertEqual(self._actions(record['id'])['new_chat'], _ALLOWED)

  def test_a_chat_that_is_over_refuses_a_turn(self):
    """A reply filled the context: its send says so; a new chat may start.

    The Session is entered, and the chat read in Debug.
    """
    record = self.world.create()
    self.world.start(record)
    # 116 words leave room for 3 of the reply's 4 tokens in the context.
    job = self.world.turn(record['id'], ' '.join(['w'] * 116))
    self.assertEqual(job['status'], 'completed', job['error'])
    summary = self.world.summary(record['id'])
    chat = summary['chats'][0]
    self.assertEqual(
        chat['actions'],
        {
            'send': {'allowed': False, 'reason': model_server.CHAT_ENDED},
            'debug': _ALLOWED,
        },
    )
    self.assertIn('inspected in Debug', chat['notice'])
    self._refused(
        {'allowed': False, 'reason': model_server.NO_CHAT},
        self.world.model_servers.generate,
        record['id'],
        'Hi',
    )
    self.assertEqual(summary['actions']['enter'], _ALLOWED)
    self.assertEqual(summary['actions']['new_chat'], _ALLOWED)

  # ---- Whether a chat is read in Debug.

  def test_a_chat_is_read_in_debug_once_it_kept_a_turn(self):
    """Its Debug is refused until a turn is kept."""
    record = self.world.create()
    self.world.start(record)
    no_turn = {
        'allowed': False,
        'reason': 'This chat has no turn to inspect.',
    }
    self.assertEqual(self._chat(record['id'])['actions']['debug'], no_turn)
    job = self.world.turn(record['id'], 'Hi')
    self.assertEqual(job['status'], 'completed', job['error'])
    self.assertEqual(self._chat(record['id'])['actions']['debug'], _ALLOWED)

  def test_a_session_turned_off_after_a_turn_is_still_read(self):
    """Its Model Server off: the Session has ended; its chats are read."""
    record = self.world.create()
    self.world.start(record)
    job = self.world.turn(record['id'], 'Hi')
    self.assertEqual(job['status'], 'completed', job['error'])
    self.world.model_servers.close(record['id'])
    summary = self.world.summary(record['id'])
    self._refused(
        summary['actions']['new_chat'],
        self.world.model_servers.start_chat,
        record['id'],
        _SETTINGS,
    )
    self.assertEqual(
        summary['actions']['new_chat']['reason'], 'This Session has ended.'
    )
    self.assertEqual(summary['actions']['enter'], _ALLOWED)
    chat = summary['chats'][0]
    self.assertEqual(chat['state'], 'ended')
    self.assertEqual(chat['actions']['debug'], _ALLOWED)
    self.assertEqual(
        chat['notice'],
        model_server.ended_chat(model_server.TURNED_OFF, True),
    )

  def test_a_session_that_ended_with_no_turn_is_not_entered(self):
    """Turned off before any chat kept a turn: Ended · no turn."""
    record = self.world.create()
    self.world.start(record)
    self.world.chat(record['id'])
    self.world.model_servers.close(record['id'])
    summary = self.world.summary(record['id'])
    self.assertEqual(summary['status_text'], 'Ended · no turn')
    self.assertFalse(summary['actions']['enter']['allowed'])
    self.assertIn('create a new Session', summary['actions']['enter']['reason'])
    for chat in summary['chats']:
      self.assertFalse(chat['actions']['debug']['allowed'])

  def test_a_first_turn_stopped_ends_its_chat_with_no_turn(self):
    """A Stop ended the chat before any turn was kept: no Debug."""
    record = self.world.create(ref='android:A', target='android:B')
    self.world.start(record)
    job, release = self._held_turn(record)
    self.world.model_servers.cancel(job['id'])
    release.set()
    self.assertEqual(self.world.wait(job)['status'], 'cancelled')
    summary = self.world.summary(record['id'])
    chat = summary['chats'][0]
    self.assertEqual(chat['notice'], 'Generation stopped. This chat is over.')
    self.assertFalse(chat['actions']['debug']['allowed'])
    self.assertEqual(summary['actions']['new_chat'], _ALLOWED)
    self.assertEqual(summary['actions']['enter'], _ALLOWED)

  def test_a_session_that_is_starting_starts_no_chat(self):
    """While its Model Server starts, a new chat is refused as it is.

    The listing goes on meanwhile: the Session is listed with its job,
    and each side shows its progress.
    """
    record = self.world.create()
    inside, go = threading.Event(), threading.Event()
    self.addCleanup(go.set)
    original = self.world.launchers.launch

    def launch(device_id, build=None):
      inside.set()
      go.wait(fake_launchers.TIMEOUT)
      return original(device_id, build)

    self.world.launchers.launch = launch
    job = self.world.model_servers.initialize(record['id'])
    self.assertTrue(inside.wait(fake_launchers.TIMEOUT))
    summary = self.world.summary(record['id'])
    self.assertEqual(summary['job_id'], job['id'])
    self._refused(
        summary['actions']['new_chat'],
        self.world.model_servers.start_chat,
        record['id'],
        _SETTINGS,
    )
    world_lib.eventually(
        lambda: (
            [
                runner['progress']
                for runner in self.world.view(record['id'])['runners']
            ]
            == ['Starting runner', 'Starting runner']
        )
    )
    go.set()
    self.assertEqual(self.world.wait(job)['status'], 'completed')
    self.assertEqual(self._actions(record['id'])['new_chat'], _ALLOWED)

  def test_a_session_that_ran_may_be_entered_and_deleted(self):
    """Off after a turn: entered, and deleted."""
    record = self.world.create()
    self.world.start(record)
    job = self.world.turn(record['id'], 'Hi')
    self.assertEqual(job['status'], 'completed')
    self.world.model_servers.close(record['id'])
    actions = self._actions(record['id'])
    for name in ('enter', 'delete'):
      self.assertEqual(actions[name], _ALLOWED, name)

  def test_each_chat_lists_its_own_settings(self):
    """In a few words: once when the sides agree, else each side's."""
    record = self.world.create()
    self.world.start(record, _CHOSEN)
    self.world.chat(record['id'], _SETTINGS)
    first, second = self.world.summary(record['id'])['chats']
    self.assertEqual(
        first['summary'],
        'Reference: Top-K 40, temperature 1.0 · system prompt / Target: '
        'Greedy · system prompt',
    )
    self.assertEqual(first['generation']['ref']['systemPrompt'], 'Be terse.')
    self.assertEqual(second['summary'], 'Greedy · no system prompt')
    self.assertEqual(
        [(chat['title'], chat['state']) for chat in (first, second)],
        [('Chat 1', 'ended'), ('Chat 2', 'live')],
    )


if __name__ == '__main__':
  test_main.main()
