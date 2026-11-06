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

"""Tests of one side's work, with in-process fake runners."""

import hashlib
import pathlib
import threading
import time

from absl.testing import absltest
from model_debugger.protocol import runner_pb2
from model_debugger.server.runner import runner_builds
from model_debugger.server.runner import runner_client
from model_debugger.server.session import side as side_lib
from model_debugger.server.testing import capture_fixtures
from model_debugger.server.testing import fake_builds
from model_debugger.server.testing import fake_launchers
from model_debugger.server.testing import fake_runner
from model_debugger.server.testing import test_main

_MODEL_BYTES = b'a model, as far as the fake runner can tell' * 1000


def _held_elsewhere(lock: threading.RLock) -> bool:
  """Whether another thread holds `lock` now."""
  taken = []

  def take() -> None:
    acquired = lock.acquire(blocking=False)
    taken.append(acquired)
    if acquired:
      lock.release()

  thread = threading.Thread(target=take)
  thread.start()
  thread.join(fake_launchers.TIMEOUT)
  return not taken[0]


class _KeptLaunchers(fake_launchers.FakeLaunchers):
  """Fake launchers that first refuse, as this computer does, a build gone."""

  def __init__(self, kept: pathlib.Path) -> None:
    super().__init__([fake_launchers.LOCAL, fake_launchers.PHONE])
    self.kept = runner_builds.RunnerBuilds(kept, None)

  def launch(
      self, device_id: str, build: str | None = None
  ) -> fake_launchers.FakeLaunchedRunner:
    if device_id == fake_launchers.LOCAL.id:
      self.kept.runner(build)
    return super().launch(device_id, build)


class SideTest(absltest.TestCase):
  """Tests of one side's work, with in-process fake runners."""

  def setUp(self):
    """A model file, this computer, and a side that runs there."""
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)
    self.model = self.tmp_path / 'gemma.litertlm'
    self.model.write_bytes(_MODEL_BYTES)
    self.sha256 = hashlib.sha256(_MODEL_BYTES).hexdigest()
    self.launchers = fake_launchers.FakeLaunchers([fake_launchers.LOCAL])
    self.lock = threading.RLock()
    self.side = side_lib.Side(
        'ref', fake_launchers.LOCAL, fake_launchers.COMMIT, self.lock
    )
    self.addCleanup(self.side.stop)

  def _load_request(self) -> runner_pb2.LoadRequest:
    return runner_pb2.LoadRequest(
        sha256=self.sha256,
        device=runner_pb2.CPU,
        context_length=capture_fixtures.CONTEXT_LENGTH,
    )

  def _ready(self) -> None:
    """Launches, connects, loads and creates the chat."""
    self.assertTrue(self.side.launch(self.launchers, lambda: False))
    self.side.connect()
    self.side.load(
        self._load_request(), self.model, threading.Lock(), lambda *_: None
    )
    request = runner_pb2.CreateChatRequest()
    request.options.max_completion_tokens = 32
    self.side.create_chat(request)

  def test_a_side_starts_takes_a_turn_and_its_capture(self):
    """Launch to capture: the runner sees each step, in order."""
    self.assertEqual(self.side.phase, 'starting')
    self.assertTrue(self.side.launch(self.launchers, lambda: False))
    (launched,) = self.launchers.launched
    self.assertIs(self.side.runner, launched)
    self.assertEqual(launched.build, fake_launchers.COMMIT)
    info = self.side.connect()
    self.assertEqual(info, fake_runner.DEFAULT_INFO)
    self.assertIs(self.side.info, info)
    self.assertIsNotNone(self.side.client)
    progress = []
    loaded = self.side.load(
        self._load_request(),
        self.model,
        threading.Lock(),
        lambda sent, total: progress.append((sent, total)),
    )
    runner = launched.runner
    self.assertEqual(loaded, capture_fixtures.TinyChat().load_response())
    self.assertEqual(runner.uploads, [self.sha256])
    self.assertEqual(progress[-1], (len(_MODEL_BYTES), len(_MODEL_BYTES)))
    request = runner_pb2.CreateChatRequest()
    request.options.max_completion_tokens = 32
    self.side.create_chat(request)
    self.assertEqual(runner.chat_request, request)
    chunks = []
    completion = self.side.send('Hi', chunks.append)
    self.assertEqual(
        ''.join(chunk.content for chunk in chunks),
        fake_launchers.REPLY_TEXT,
    )
    self.assertEqual(completion.content, fake_launchers.REPLY_TEXT)
    directory = self.tmp_path / 'turn-1'
    self.side.take_capture(1, directory)
    self.assertTrue(any(path.is_file() for path in directory.rglob('*')))
    self.assertEqual(runner.released, [1])
    self.side.stop()
    self.assertTrue(runner.stopped)

  def test_a_runner_of_another_commit_is_refused_at_connect(self):
    """The side chose a commit; a runner reporting another is refused."""
    info = runner_pb2.RunnerInfo()
    info.CopyFrom(fake_runner.DEFAULT_INFO)
    info.runtime.version = fake_builds.NEW
    self.launchers.infos[fake_launchers.LOCAL.id] = info
    self.assertTrue(self.side.launch(self.launchers, lambda: False))
    with self.assertRaises(RuntimeError) as refused:
      self.side.connect()
    self.assertEqual(
        str(refused.exception),
        'This computer runs LiteRT-LM 222222222222, but this Session '
        f'chose {fake_launchers.COMMIT[:12]}; keep it again, or create a '
        'new Session and choose another version.',
    )
    # The connection is the side's, for stop() to close with the runner.
    self.assertIsNotNone(self.side.client)
    self.side.stop()
    self.assertTrue(self.side.runner.runner.stopped)

  def test_a_phone_of_another_commit_is_not_told_to_keep_it(self):
    """Only this computer runs the build chosen; a phone runs its App."""
    info = runner_pb2.RunnerInfo()
    info.CopyFrom(fake_runner.DEFAULT_INFO)
    info.runtime.version = fake_builds.NEW
    launchers = fake_launchers.FakeLaunchers([fake_launchers.PHONE])
    launchers.infos[fake_launchers.PHONE.id] = info
    side = side_lib.Side(
        'target', fake_launchers.PHONE, fake_builds.OLD, self.lock
    )
    self.addCleanup(side.stop)
    self.assertTrue(side.launch(launchers, lambda: False))
    with self.assertRaises(RuntimeError) as refused:
      side.connect()
    self.assertEqual(
        str(refused.exception),
        'Pixel runs LiteRT-LM 222222222222, but this Session chose '
        '111111111111; install the App built from LiteRT-LM 111111111111, '
        'or create a new Session and choose the device again.',
    )

  def test_a_build_gone_says_how_to_choose_another(self):
    """The start it refused ends the Session: a new one chooses again."""
    kept = self.tmp_path / 'kept'
    launchers = _KeptLaunchers(kept)
    launchers.builds_now[fake_launchers.LOCAL.id] = [
        fake_launchers.runner_build(fake_builds.OLD)
    ]
    side = side_lib.Side(
        'ref', fake_launchers.LOCAL, fake_builds.OLD, self.lock
    )
    self.addCleanup(side.stop)
    with self.assertRaises(RuntimeError) as first:
      side.launch(launchers, lambda: False)
    self.assertEqual(
        str(first.exception),
        'The runner for LiteRT-LM 111111111111 is no longer kept on this '
        'computer; keep it again, or create a new Session and choose '
        'another version.',
    )
    unchosen = side_lib.Side('ref', fake_launchers.LOCAL, None, self.lock)
    with self.assertRaises(RuntimeError) as none:
      unchosen.launch(launchers, lambda: False)
    self.assertEqual(
        str(none.exception),
        'No runner version was chosen; create a new Session and choose a '
        'version.',
    )

  def test_the_development_build_and_a_phone_run_whatever_they_report(self):
    """No commit was chosen: whatever the runner reports is taken."""
    info = runner_pb2.RunnerInfo()
    info.CopyFrom(fake_runner.DEFAULT_INFO)
    info.runtime.version = fake_builds.NEW
    launchers = fake_launchers.FakeLaunchers(
        [fake_launchers.LOCAL, fake_launchers.PHONE]
    )
    for device in (fake_launchers.LOCAL, fake_launchers.PHONE):
      launchers.infos[device.id] = info
    sides = [
        side_lib.Side(
            'ref',
            fake_launchers.LOCAL,
            runner_builds.DEVELOPMENT,
            self.lock,
        ),
        side_lib.Side('target', fake_launchers.PHONE, None, self.lock),
    ]
    for side in sides:
      self.addCleanup(side.stop)
      self.assertTrue(side.launch(launchers, lambda: False))
      self.assertEqual(side.connect().runtime.version, fake_builds.NEW)
    self.assertEqual(
        [launched.build for launched in launchers.launched],
        [runner_builds.DEVELOPMENT, None],
    )

  def test_a_launch_aborted_meanwhile_stops_its_runner(self):
    """Asked with the lock held, after the runner is recorded."""
    seen = []

    def aborted() -> bool:
      seen.append((self.side.runner, _held_elsewhere(self.lock)))
      return True

    self.assertFalse(self.side.launch(self.launchers, aborted))
    (launched,) = self.launchers.launched
    self.assertEqual(seen, [(launched, True)])
    self.assertIs(self.side.runner, launched)
    self.assertTrue(launched.runner.stopped)

  def test_load_uploads_under_the_lock_it_is_handed(self):
    """No upload while another holds the upload lock."""
    self.assertTrue(self.side.launch(self.launchers, lambda: False))
    self.side.connect()
    upload_lock = threading.Lock()
    outcome = []
    with upload_lock:
      thread = threading.Thread(
          target=lambda: outcome.append(
              self.side.load(
                  self._load_request(),
                  self.model,
                  upload_lock,
                  lambda *_: None,
              )
          )
      )
      thread.start()
      time.sleep(0.2)
      self.assertEqual(self.side.runner.runner.uploads, [])
    thread.join(fake_launchers.TIMEOUT)
    self.assertLen(outcome, 1)
    self.assertEqual(self.side.runner.runner.uploads, [self.sha256])

  def test_cancel_ends_the_send_in_flight(self):
    """A cancelled Send fails, and the runner's chat is over."""
    self._ready()
    runner = self.side.runner.runner
    release, first = threading.Event(), threading.Event()
    self.side.runner.script.hold = release
    outcome = []

    def send() -> None:
      try:
        outcome.append(self.side.send('Hi', lambda _: first.set()))
      except runner_client.Error as error:
        outcome.append(error)

    thread = threading.Thread(target=send)
    thread.start()
    try:
      self.assertTrue(first.wait(fake_launchers.TIMEOUT))
      self.side.cancel()
      thread.join(fake_launchers.TIMEOUT)
    finally:
      release.set()
    self.assertIsInstance(outcome[0], runner_client.Error)
    deadline = time.monotonic() + fake_launchers.TIMEOUT
    while not runner.chat_over:
      self.assertLess(time.monotonic(), deadline)
      time.sleep(0.01)

  def test_close_fails_the_calls_that_follow(self):
    """Closing the connection leaves the runner running."""
    self._ready()
    self.side.close()
    with self.assertRaises(runner_client.Error):
      self.side.send('Hi', lambda _: None)
    self.assertFalse(self.side.runner.runner.stopped)

  def test_stop_is_idempotent(self):
    """Stop needs nothing launched, and stops a runner once."""
    self.side.stop()
    self.assertTrue(self.side.launch(self.launchers, lambda: False))
    self.side.connect()
    launched = self.side.runner
    self.side.stop()
    self.side.stop()
    self.assertTrue(launched.runner.stopped)


if __name__ == '__main__':
  test_main.main()
