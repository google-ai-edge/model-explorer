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

"""Tests of Executions with in-process fake runners and a fake launcher."""

import threading
import time

import capture_fixtures
import fake_runner
import grpc
from model_debugger.v1 import runner_pb2
from model_debugger_server import (
    capture_index,
    errors,
    execution,
    jobs,
    launchers,
    models,
    sessions,
    workspace,
)
import pytest

_MODEL_BYTES = fake_runner.MODEL_BYTES
_LOCAL = fake_runner.LOCAL
_PHONE = fake_runner.PHONE
_TABLET = fake_runner.TABLET
REPLY = fake_runner.REPLY
REPLY_TEXT = fake_runner.REPLY_TEXT
_TIMEOUT = fake_runner.TIMEOUT
Script = fake_runner.Script
Launched = fake_runner.Launched
Launchers = fake_runner.Launchers
World = fake_runner.World


def _eventually(condition) -> None:
  """Waits for a fake runner's handler to see what the client did."""
  deadline = time.monotonic() + _TIMEOUT
  while not condition():
    assert time.monotonic() < deadline
    time.sleep(0.01)


@pytest.fixture
def world(tmp_path):
  """A computer and two devices that run one runner at a time."""
  made = World(tmp_path, [_LOCAL, _PHONE, _TABLET])
  yield made
  made.close()


def test_initialize_uploads_the_model_once_and_starts_the_chat(world):
  """Both runners load the model, uploaded once to their shared store."""
  record = world.create()
  job = world.start(record)
  ref, target = world.launchers.runners('local')
  assert world.launchers.store == {world.model.sha256}
  assert ref.uploads + target.uploads == [world.model.sha256]
  for runner in (ref, target):
    assert runner.load_request.sha256 == world.model.sha256
    assert runner.load_request.context_length == 128
    assert runner.chat_request.options.max_completion_tokens == 32
    assert runner.chat_request.options.top_k == 1
  progress = [
      (event['runId'], event['message'])
      for event in world.events(job['id'])
      if event['type'] == 'progress'
  ]
  assert ('ref', 'Starting runner') in progress
  assert ('target', 'Starting chat') in progress
  assert any(message == 'Uploading model · 100%' for _, message in progress)
  view = world.executions.view(record['id'])
  assert view['phase'] == 'active'
  assert view['activeChatId'] == record['id']
  assert view['newChatAllowed']
  assert [runner['phase'] for runner in view['runners']] == ['active'] * 2
  assert view['runners'][0]['buildId'] == (
      fake_runner.DEFAULT_INFO.runtime.version
  )
  index = world.executions.capture(record['id'])
  assert index.turns == []
  assert index.sides['ref']['model']['sha256'] == world.model.sha256
  assert index.sides['ref']['start_token'] == 2
  assert index.sides['ref']['stop_tokens'] == [[1], [4]]
  assert index.sides['ref']['chat']['options']['top_k'] == 1
  summary = world.executions.summary(world.sessions.get(record['id']))
  assert summary['status'] == 'ready'
  assert summary['initialized']
  assert summary['job_id'] == job['id']
  # Recorded at once, so a restart cannot reattach; shown once it is over.
  assert world.sessions.get(record['id'])['execution_ended']
  assert not summary['execution_ended']


def test_initialize_is_idempotent_per_request(world):
  """The same request id answers the same job."""
  record = world.create()
  first = world.executions.initialize(record['id'], 'once')
  again = world.executions.initialize(record['id'], 'once')
  assert again['id'] == first['id']
  world.wait(first)
  assert len(world.launchers.launched) == 2


def test_initialize_refuses_a_session_already_on_or_ended(world):
  """A Session starts once; after it ends, only a duplicate runs."""
  record = world.create()
  world.start(record)
  with pytest.raises(errors.Conflict, match='already on'):
    world.executions.initialize(record['id'], 'again')
  world.executions.close(record['id'])
  with pytest.raises(errors.Conflict, match='duplicate'):
    world.executions.initialize(record['id'], 'after close')


def test_initialize_failure_stops_every_runner(world):
  """A side that cannot load leaves the execution interrupted, no runner."""
  record = world.create()
  bad = runner_pb2.RunnerInfo()
  bad.CopyFrom(fake_runner.DEFAULT_INFO)
  bad.protocol_version = '2.0.0'
  original = world.launchers.launch

  def launch(device_id):
    launched = original(device_id)
    launched.runner._info = bad
    return launched

  world.launchers.launch = launch
  job = world.wait(world.executions.initialize(record['id'], 'start'))
  assert job['status'] == 'failed'
  assert 'protocol 2.0.0' in job['error']
  view = world.executions.view(record['id'])
  assert view['phase'] == 'interrupted'
  assert 'protocol 2.0.0' in view['error']
  assert all(launched._stopped for launched in world.launchers.launched)
  summary = world.executions.summary(world.sessions.get(record['id']))
  assert summary['status'] == 'failed'
  assert not summary['execution_ended']
  # Never on: it may try again.
  world.launchers.launch = original
  world.start(record, 'try again')


def test_generate_streams_replies_and_stores_both_captures(world):
  """Deltas reach the job; each side's capture is read, released, indexed."""
  record = world.create()
  world.start(record)
  job = world.turn(record['id'], 'What is 2 + 2?', 'turn-1')
  assert job['status'] == 'completed', job['error']
  assert job['turn'] == 1
  reply = REPLY_TEXT
  assert job['output'] == {'ref': reply, 'target': reply}
  deltas = [e for e in world.events(job['id']) if e['type'] == 'delta']
  assert {event['runId'] for event in deltas} == {'ref', 'target'}
  for runner in world.launchers.runners('local'):
    assert runner.released == [1]
  index = world.executions.capture(record['id'])
  assert [turn['n'] for turn in index.turns] == [1]
  assert index.turns[0]['prompt'] == 'What is 2 + 2?'
  for side in ('ref', 'target'):
    entry = index.turns[0]['sides'][side]
    assert entry['completion']['content'] == reply
    for file in entry['files']:
      assert (index.root / entry['dir'] / file['path']).is_file()
  # The same object, read again from disk, holds the same turn.
  assert capture_index.CaptureIndex.load(index.root).turns == index.turns
  record = world.sessions.get(record['id'])
  assert record['has_capture']
  summary = world.executions.summary(record)
  assert summary['status'] == 'saved'
  assert summary['initialized']
  job = world.turn(record['id'], 'And 3 + 3?', 'turn-2')
  assert job['status'] == 'completed', job['error']
  assert [turn['n'] for turn in index.turns] == [1, 2]


def test_generate_refuses_while_a_turn_runs(world):
  """One job at a time per Session."""
  record = world.create()
  world.start(record)
  release = threading.Event()
  for script in world.scripts():
    script.hold = release
  job = world.executions.generate(record['id'], 'one', 'Hi')
  try:
    with pytest.raises(errors.Conflict, match='Wait'):
      world.executions.generate(record['id'], 'two', 'Hi again')
    with pytest.raises(errors.Conflict, match='Wait'):
      world.executions.new_chat(record['id'], 'New chat 1', None)
  finally:
    release.set()
  assert world.wait(job)['status'] == 'completed'


def test_generate_checks_the_prompt(world):
  """An empty prompt, or one over the runner's limit, is refused at once."""
  record = world.create()
  world.start(record)
  with pytest.raises(ValueError, match='Type a prompt'):
    world.executions.generate(record['id'], 'empty', '')
  with pytest.raises(ValueError, match='the most a runner takes'):
    world.executions.generate(
        record['id'], 'long', 'x' * (execution.PROMPT_LIMIT_BYTES + 1)
    )


def test_both_sides_refusing_keeps_the_chat(world):
  """A turn both sides refuse changes nothing: the next turn is turn 1."""
  record = world.create()
  world.start(record)
  for script in world.scripts():
    script.refuse = grpc.StatusCode.INVALID_ARGUMENT
  job = world.turn(record['id'], 'Too long', 'refused')
  assert job['status'] == 'failed'
  assert job['error'] == 'Reference: Refused turn 1. Target: Refused turn 1.'
  view = world.executions.view(record['id'])
  assert view['phase'] == 'active'
  assert [runner['phase'] for runner in view['runners']] == ['active'] * 2
  assert not world.sessions.get(record['id'])['chat_over']
  for script in world.scripts():
    script.refuse = None
  job = world.turn(record['id'], 'What is 2 + 2?', 'accepted')
  assert job['status'] == 'completed', job['error']
  assert job['turn'] == 1


def test_one_side_failing_ends_the_chat(world):
  """A side's failure cancels the other side; the chat is then over."""
  record = world.create(ref='android:A', target='android:B')
  world.start(record)
  release = threading.Event()
  (ref,) = world.launchers.runners('android:A')
  (target,) = world.launchers.runners('android:B')
  ref._respond.hold = release
  target._respond.refuse = grpc.StatusCode.INTERNAL
  try:
    job = world.turn(record['id'], 'Hi', 'fails')
  finally:
    release.set()
  assert job['status'] == 'failed'
  assert job['error'].startswith('Target: Refused turn 1. This chat is over')
  _eventually(lambda: ref.chat_over and target.chat_over)
  assert ref.released == target.released == []
  record = world.sessions.get(record['id'])
  assert record['chat_over']
  assert record['notice'] == job['error']
  view = world.executions.view(record['id'])
  assert view['phase'] == 'active'
  assert view['newChatAllowed']
  summary = world.executions.summary(record)
  assert not summary['initialized']
  with pytest.raises(errors.Conflict, match='chat is over'):
    world.executions.generate(record['id'], 'next', 'Hi')


def test_one_side_refusing_while_the_other_replies_ends_the_chat(world):
  """A refusal on one side and a reply on the other leave them out of step."""
  record = world.create(ref='android:A', target='android:B')
  world.start(record)
  release = threading.Event()
  (ref,) = world.launchers.runners('android:A')
  (target,) = world.launchers.runners('android:B')
  ref._respond.hold = release
  target._respond.refuse = grpc.StatusCode.INVALID_ARGUMENT
  try:
    job = world.turn(record['id'], 'Hi', 'half')
  finally:
    release.set()
  assert job['status'] == 'failed'
  assert job['error'].startswith('Target: Refused turn 1. This chat is over')
  _eventually(lambda: ref.chat_over)
  assert not target.chat_over
  assert world.sessions.get(record['id'])['chat_over']


def test_cancel_stops_both_sides_and_ends_the_chat(world):
  """Cancelling a turn cancels both Sends; nothing of the turn is kept."""
  record = world.create()
  world.start(record)
  release = threading.Event()
  runners = world.launchers.runners('local')
  for script in world.scripts():
    script.hold = release
  job = world.executions.generate(record['id'], 'stop me', 'Write a poem')
  try:
    deadline = time.monotonic() + _TIMEOUT
    while world.jobs.get(job['id'])['output'] != {
        'ref': 'It',
        'target': 'It',
    }:
      assert time.monotonic() < deadline
      time.sleep(0.01)
    world.executions.cancel(job['id'])
    job = world.wait(job)
  finally:
    release.set()
  assert job['status'] == 'cancelled'
  assert job['error'] == 'Generation stopped. ' + execution.CHAT_OVER
  assert job['output'] == {'ref': 'It', 'target': 'It'}
  _eventually(lambda: all(runner.chat_over for runner in runners))
  assert world.executions.capture(record['id']).turns == []
  assert world.sessions.get(record['id'])['chat_over']
  assert world.executions.view(record['id'])['newChatAllowed']
  # A job that ended answers unchanged.
  assert world.executions.cancel(job['id'])['status'] == 'cancelled'


def test_cancel_refuses_starting_a_model_server(world):
  """Initialize is not cancelled; the Model Server is turned off instead."""
  record = world.create()
  go = threading.Event()
  original = world.launchers.launch

  def launch(device_id):
    go.wait(_TIMEOUT)
    return original(device_id)

  world.launchers.launch = launch
  job = world.executions.initialize(record['id'], 'start')
  try:
    with pytest.raises(errors.Conflict, match='turn it off'):
      world.executions.cancel(job['id'])
    summary = world.executions.summary(world.sessions.get(record['id']))
    assert summary['status'] == 'initializing'
    assert summary['execution']['phase'] == 'starting'
    assert summary['execution']['activeOperation'] == 'initialize'
  finally:
    go.set()
  assert world.wait(job)['status'] == 'completed'


def test_close_while_starting_stops_what_was_launched(world):
  """Turning off a Model Server that is starting leaves no runner."""
  record = world.create()
  go = threading.Event()
  original = world.launchers.launch

  def launch(device_id):
    go.wait(_TIMEOUT)
    return original(device_id)

  world.launchers.launch = launch
  job = world.executions.initialize(record['id'], 'start')
  summary = world.executions.close(record['id'])
  # Its runners may still be launching: it ends once they are stopped.
  assert summary['execution']['phase'] == 'ending'
  go.set()
  job = world.wait(job)
  assert job['status'] == 'failed'
  assert 'turned off' in job['error']
  summary = world.executions.summary(world.sessions.get(record['id']))
  assert 'execution' not in summary
  assert all(launched._stopped for launched in world.launchers.launched)
  # It never came on, so it may start again.
  world.launchers.launch = original
  world.start(record, 'start again')


def test_close_while_launching_holds_the_device_until_launched(world):
  """A device is not free while a closed start may still launch on it."""
  first = world.create(ref='android:A', target='local')
  second = world.create(ref='android:A', target='local')
  inside, go = threading.Event(), threading.Event()
  original = world.launchers.launch

  def launch(device_id):
    if device_id == 'android:A':
      inside.set()
      go.wait(_TIMEOUT)
    return original(device_id)

  world.launchers.launch = launch
  job = world.executions.initialize(first['id'], 'first')
  try:
    assert inside.wait(_TIMEOUT)
    world.executions.close(first['id'])
    with pytest.raises(errors.Conflict, match='Pixel is in use'):
      world.executions.initialize(second['id'], 'too early')
    assert world.executions.inspect('android:A')['status'] == 'busy'
  finally:
    go.set()
  assert world.wait(job)['status'] == 'failed'
  assert world.launchers.running('android:A') == []
  world.launchers.launch = original
  world.start(second)


def test_close_while_starting_a_new_chat_ends_once_launched(world):
  """Turned off during a new chat: ended only when nothing launches."""
  record = world.create(ref='android:A', target='local')
  world.start(record)
  inside, go = threading.Event(), threading.Event()
  original = world.launchers.launch

  def launch(device_id):
    inside.set()
    go.wait(_TIMEOUT)
    return original(device_id)

  world.launchers.launch = launch
  outcome = []
  thread = threading.Thread(
      target=lambda: outcome.append(
          _raised(world.executions.new_chat, record['id'], 'New chat 1', None)
      )
  )
  thread.start()
  try:
    assert inside.wait(_TIMEOUT)
    summary = world.executions.close(record['id'])
    assert summary['execution']['phase'] == 'ending'
  finally:
    go.set()
  thread.join(_TIMEOUT)
  assert isinstance(outcome[0], execution.Failed)
  assert world.executions.view(record['id'])['phase'] == 'ended'
  assert all(launched._stopped for launched in world.launchers.launched)


def _raised(call, *args):
  """Returns what `call(*args)` raised, or None."""
  try:
    call(*args)
  except Exception as error:
    return error
  return None


def test_new_chat_relaunches_both_runners(world):
  """A new chat is new runners, with the new chat's settings."""
  record = world.create()
  world.start(record)
  assert world.turn(record['id'], 'Hi', 'first')['status'] == 'completed'
  first = world.launchers.runners('local')
  chat = world.executions.new_chat(
      record['id'], 'New chat 1', {'ref': {'maxOutputTokens': 8}, 'target': {}}
  )
  assert chat['parent_session_id'] == record['id']
  assert chat['status'] == 'ready'
  assert chat['initialized']
  runners = world.launchers.runners('local')
  assert len(runners) == 4
  assert all(launched._stopped for launched in world.launchers.launched[:2])
  new_ref = next(
      r
      for r in runners[2:]
      if r.chat_request.options.max_completion_tokens == 8
  )
  assert new_ref not in first
  # The model is in the shared store already: no second upload.
  assert sum(len(runner.uploads) for runner in runners) == 1
  view = world.executions.view(record['id'])
  assert view['activeChatId'] == chat['id']
  assert world.sessions.get(record['id'])['chat_over']
  with pytest.raises(errors.Conflict, match='chat is over'):
    world.executions.generate(record['id'], 'old', 'Hi')
  job = world.turn(chat['id'], 'Hello', 'in the new chat')
  assert job['status'] == 'completed', job['error']
  assert job['turn'] == 1
  assert job['session_id'] == chat['id']
  assert [t['n'] for t in world.executions.capture(chat['id']).turns] == [1]
  assert len(world.executions.capture(record['id']).turns) == 1


def test_new_chat_needs_the_model_server_on(world):
  """No new chat without an active Model Server."""
  record = world.create()
  with pytest.raises(errors.Conflict, match='on first'):
    world.executions.new_chat(record['id'], 'New chat 1', None)


def test_new_chat_failure_interrupts_the_model_server(world):
  """A new chat that cannot start leaves no runner and a failed chat."""
  record = world.create()
  world.start(record)
  world.launchers.devices_now = [_PHONE]
  original = world.launchers.launch

  def launch(device_id):
    raise RuntimeError('The runner did not start within 30 s.')

  world.launchers.launch = launch
  with pytest.raises(execution.Failed, match='did not start'):
    world.executions.new_chat(record['id'], 'New chat 1', None)
  world.launchers.launch = original
  view = world.executions.view(record['id'])
  assert view['phase'] == 'interrupted'
  assert all(launched._stopped for launched in world.launchers.launched)
  (chat,) = [
      r
      for r in world.sessions.list()
      if r.get('parent_session_id') == record['id']
  ]
  assert chat['chat_over']
  assert 'did not start' in chat['notice']
  assert world.executions.summary(chat)['status'] == 'failed'


def test_close_stops_both_runners_for_good(world):
  """Close ends the execution; the Session shows ended."""
  record = world.create()
  world.start(record)
  summary = world.executions.close(record['id'])
  assert summary['execution']['phase'] == 'ended'
  assert summary['execution_ended']
  assert summary['status'] == 'draft'
  assert all(launched._stopped for launched in world.launchers.launched)
  # Idempotent.
  assert world.executions.close(record['id'])['execution']['phase'] == 'ended'


def test_close_while_generating_cancels_the_turn(world):
  """Turning off during a turn cancels it."""
  record = world.create()
  world.start(record)
  release = threading.Event()
  for script in world.scripts():
    script.hold = release
  job = world.executions.generate(record['id'], 'closing', 'Hi')
  try:
    world.executions.close(record['id'])
    job = world.wait(job)
  finally:
    release.set()
  assert job['status'] == 'cancelled'
  assert world.executions.view(record['id'])['phase'] == 'ended'


def test_close_of_a_session_never_on_answers_inactive(world):
  """Nothing to close: the record comes back as it is."""
  record = world.create()
  summary = world.executions.close(record['id'])
  assert 'execution' not in summary
  assert summary['status'] == 'draft'


def test_a_device_that_runs_one_runner_serves_one_side(world):
  """Android devices hold one runner, across every Session."""
  with pytest.raises(errors.Conflict, match='different devices'):
    world.executions.initialize(
        world.create(ref='android:A', target='android:A')['id'], 'same'
    )
  first = world.create(ref='local', target='android:A')
  world.start(first)
  second = world.create(ref='android:A', target='local')
  with pytest.raises(errors.Conflict, match='Pixel is in use'):
    world.executions.initialize(second['id'], 'busy')
  inspected = world.executions.inspect('android:A')
  assert inspected['status'] == 'busy'
  assert inspected['runnerState'] == 'active'
  assert inspected['runner']['owner']['sessionId'] == first['id']
  # This computer runs any number of runners.
  third = world.create()
  world.start(third)
  world.executions.close(first['id'])
  world.start(second)


def test_initialize_refuses_a_device_not_connected(world):
  """A run's device must be there now."""
  record = world.create(target='android:C')
  with pytest.raises(errors.Conflict, match='not connected'):
    world.executions.initialize(record['id'], 'start')


def test_inspect_asks_a_fresh_runner_and_stops_it(world):
  """Inspect launches, asks GetInfo, stops; capabilities then know options."""
  before = world.executions.capabilities()
  assert before['runtimes'][0]['supported_options'] == [
      'backend',
      'contextLength',
  ]
  inspected = world.executions.inspect('local')
  assert inspected['status'] == 'ready'
  assert inspected['runnerState'] == 'idle'
  runner = inspected['runner']
  version = fake_runner.DEFAULT_INFO.runtime.version
  assert runner['buildId'] == version
  assert runner['build'] == {
      'id': version,
      'label': f'LiteRT-LM {version[:12]}',
      'available': True,
  }
  assert runner['runtimes'] == [
      {'id': 'LiteRT-LM', 'backends': ['CPU', 'GPU'], 'transport': 'local'}
  ]
  assert 'cpuThreads' in runner['capabilities']['engineOptions']
  assert 'backend' not in runner['capabilities']['engineOptions']
  assert 'maxOutputTokens' not in runner['capabilities']['conversationOptions']
  assert runner['environment']['chip'] == 'Apple M2 Max'
  assert all(launched._stopped for launched in world.launchers.launched)
  after = world.executions.capabilities()
  assert after['available']
  assert 'cpuThreads' in after['runtimes'][0]['supported_options']
  assert after['runtimes'][0]['backends'] == ['CPU', 'GPU']
  assert after['models'] == [{
      'artifact': world.model.artifact,
      'name': world.model.name,
      'runtime': 'LiteRT-LM',
  }]
  assert after['tap_profiles'] == []


def test_inspect_reports_a_runner_that_does_not_start(world):
  """A launch failure reads as unreachable, with the reason."""

  def launch(device_id):
    raise RuntimeError('adb failed.')

  world.launchers.launch = launch
  inspected = world.executions.inspect('android:B')
  assert inspected['status'] == 'unreachable'
  assert inspected['reason'] == 'adb failed.'


def test_devices_lists_every_device_unchecked(world):
  """devices() is the launcher's list, each 'unknown' until inspected."""
  assert world.executions.devices()['devices'][1] == {
      'id': 'android:A',
      'name': 'Pixel',
      'platform': 'Android',
      'transport': 'adb',
      'status': 'unknown',
  }


def test_capabilities_judge_each_device_by_its_own_answer(tmp_path):
  """A device's GPU counts only by its own answer, and while connected."""
  simulator = launchers.Device(
      'ios-simulator:S', 'iPhone 17', 'iOS Simulator', 'simulator', False, False
  )
  made = World(tmp_path, [_PHONE, simulator])
  try:
    capabilities = made.executions.capabilities
    assert capabilities()['backends'] == ['CPU']
    made.executions.inspect(simulator.id)
    # Its host has a GPU, which the simulator cannot use.
    assert capabilities()['backends'] == ['CPU']
    assert 'cpuThreads' in capabilities()['supported_options']
    made.executions.inspect(_PHONE.id)
    assert capabilities()['backends'] == ['CPU', 'GPU']
    made.launchers.devices_now = [simulator]
    assert capabilities()['runtimes'][0]['backends'] == ['CPU']
  finally:
    made.close()


def test_the_listing_asks_devices_without_holding_the_lock(world):
  """A slow device answers the listing; meanwhile the lock stays free."""
  record = world.create(ref='android:A', target='local')
  world.start(record)
  asked, answer = threading.Event(), threading.Event()
  (launched,) = world.launchers.running('android:A')

  def alive():
    asked.set()
    return answer.wait(_TIMEOUT)

  launched._alive = alive
  listing = threading.Thread(
      target=world.executions.summary, args=(world.sessions.get(record['id']),)
  )
  listing.start()
  try:
    assert asked.wait(_TIMEOUT)
    refused = []
    check = threading.Thread(
        target=lambda: refused.append(
            _raised(world.executions.require_idle, record['id'])
        )
    )
    check.start()
    check.join(5)  # Long before the device answers.
    assert isinstance(refused[0], errors.Conflict)
  finally:
    answer.set()
    listing.join(_TIMEOUT)


def test_capabilities_without_devices_is_unavailable(tmp_path):
  """No device: generation is unavailable, with a reason."""
  made = World(tmp_path, [])
  try:
    capabilities = made.executions.capabilities()
    assert not capabilities['available']
    assert capabilities['reason']
  finally:
    made.close()


def test_a_runner_lost_mid_turn_leaves_the_model_server_unavailable(world):
  """A runner that goes away is no chat and no new chat: turn it off."""
  record = world.create(ref='android:A', target='android:B')
  world.start(record)
  release = threading.Event()
  (ref,) = world.launchers.running('android:A')
  (target,) = world.launchers.running('android:B')
  ref.script.hold = target.script.hold = release
  job = world.executions.generate(record['id'], 'lost', 'Hi')
  try:
    _eventually(lambda: world.jobs.get(job['id'])['output']['ref'] == 'It')
    ref.runner.stop()
    job = world.wait(job)
  finally:
    release.set()
  assert job['status'] == 'failed'
  view = world.executions.view(record['id'])
  assert view['phase'] == 'unavailable', job['error']
  assert not view['newChatAllowed']
  assert view['runners'][0]['phase'] == 'disconnected'
  summary = world.executions.summary(world.sessions.get(record['id']))
  # Still holding its devices: the UI offers to turn it off.
  assert not summary['execution_ended']
  with pytest.raises(errors.Conflict, match='Turn the Model Server off'):
    world.executions.require_idle(record['id'])
  world.executions.close(record['id'])
  assert world.executions.view(record['id'])['phase'] == 'ended'


def test_flag_write_failure_does_not_leave_job_unfinished_or_wedge_session(
    world, monkeypatch
):
  """A failing set_flags write during new_chat or turn failure does not wedge."""
  record = world.create(ref='android:A', target='android:B')
  world.start(record)

  def fail_set_flags(*_args, **_kwargs):
    raise OSError('No space left on device')

  monkeypatch.setattr(world.sessions, 'set_flags', fail_set_flags)

  second = world.executions.new_chat(record['id'], 'Second chat', None)
  assert second['id'] != record['id']
  assert world.executions.view(record['id'])['phase'] == 'active'

  for side in world.launchers.running('android:A'):
    side.script.refuse = grpc.StatusCode.INTERNAL
  job = world.wait(world.executions.generate(second['id'], 'err', 'Hi'))
  assert job['status'] == 'failed'
  world.executions.close(record['id'])
  assert world.executions.view(record['id'])['phase'] == 'ended'


def test_model_upload_locks_are_scoped_per_device(world):
  """Uploading the same model to two separate devices uses distinct locks."""
  original_launch = world.launchers.launch

  def launch_with_isolated_store(device_id: str):
    launched = original_launch(device_id)
    launched.runner.store = set()
    return launched

  world.launchers.launch = launch_with_isolated_store
  record = world.create(ref='android:A', target='android:B')
  world.start(record)

  assert ('android:A', world.model.sha256) in world.executions._upload_locks
  assert ('android:B', world.model.sha256) in world.executions._upload_locks
  assert (
      world.executions._upload_locks[('android:A', world.model.sha256)]
      is not world.executions._upload_locks[('android:B', world.model.sha256)]
  )
  world.executions.close(record['id'])
