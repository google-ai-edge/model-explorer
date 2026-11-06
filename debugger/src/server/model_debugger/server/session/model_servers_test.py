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

"""Tests of the Model Servers: which Session holds which device, uploads."""

import threading
import time
from typing import Any
from unittest import mock

from absl.testing import absltest
from model_debugger.server.analysis import kv_analysis
from model_debugger.server.runner import runner_builds
from model_debugger.server.session import model_graphs
from model_debugger.server.session import model_server as model_server_lib
from model_debugger.server.testing import fake_launchers
from model_debugger.server.testing import fake_runner
from model_debugger.server.testing import pytorch_graph_fixtures
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


def _slow_uploads(world):
  def change(launched):
    def before(method):
      if method == 'UploadModel':
        time.sleep(0.3)

    launched.runner.before = before

  _with_runner(world, change)


class ModelServersTest(absltest.TestCase):
  """Tests of the Model Servers, with in-process fake runners."""

  def setUp(self):
    """A computer and two devices."""
    super().setUp()
    self.world = world_lib.create(self)

  def test_initialize_uploads_the_model_once_then_a_chat_starts(self):
    """Both runners load the model, uploaded once to their shared store.

    A chat started then is created on both.
    """
    record = self.world.create()
    job = self.world.start(record, None)
    ref, target = self.world.launchers.runners('local')
    self.assertEqual((ref.chats, target.chats), ([], []))
    chat_job = self.world.chat(record['id'])
    self.assertEqual(
        self.world.launchers.stores, {'local': {self.world.model.sha256}}
    )
    self.assertEqual(ref.uploads + target.uploads, [self.world.model.sha256])
    for runner in (ref, target):
      self.assertEqual(runner.load_request.sha256, self.world.model.sha256)
      self.assertEqual(runner.load_request.context_length, 128)
      # No reply limit is set: the reply may fill the context.
      self.assertEqual(runner.chat_request.options.max_completion_tokens, 128)
      self.assertEqual(runner.chat_request.options.top_k, 1)
    progress = [
        (event['runId'], event['message'])
        for event in self.world.events(job['id'])
        if event['type'] == 'progress' and 'runId' in event
    ]
    self.assertIn(('ref', 'Starting runner'), progress)
    self.assertNotIn(('target', 'Starting chat'), progress)
    self.assertIn(
        ('target', 'Starting chat'),
        [
            (event['runId'], event['message'])
            for event in self.world.events(chat_job['id'])
            if event['type'] == 'progress'
        ],
    )
    self.assertTrue(
        any(message == 'Uploading model · 100%' for _, message in progress)
    )
    view = self.world.view(record['id'])
    self.assertEqual(view['phase'], 'active')
    self.assertNotIn('activeChatId', view)
    self.assertTrue(
        self.world.summary(record['id'])['chats'][0]['actions']['send'][
            'allowed'
        ]
    )
    self.assertEqual(
        [runner['phase'] for runner in view['runners']], ['active'] * 2
    )
    self.assertEqual(
        view['runners'][0]['buildId'],
        fake_runner.DEFAULT_INFO.runtime.version,
    )
    index = self.world.captures.get(record['id'], 1)
    self.assertEqual(index.turns, [])
    self.assertEqual(
        index.sides['ref']['model']['sha256'], self.world.model.sha256
    )
    self.assertEqual(index.sides['ref']['start_token'], 2)
    self.assertEqual(index.sides['ref']['stop_tokens'], [[1], [4]])
    self.assertEqual(index.sides['ref']['chat']['options']['top_k'], 1)
    summary = self.world.listing.summary(self.world.sessions.get(record['id']))
    self.assertEqual(summary['status_text'], 'Active')
    self.assertTrue(summary['actions']['new_chat']['allowed'])
    # The last job run for it: the chat's start.
    self.assertEqual(summary['job_id'], chat_job['id'])
    # Recorded at once, so a restart cannot reattach; it is the server's
    # own, and the switch turns the Model Server off while it is on.
    self.assertTrue(self.world.sessions.get(record['id'])['execution_ended'])
    self.assertNotIn('execution_ended', summary)
    self.assertEqual(summary['model_server']['button']['label'], 'Turn off')

  def test_initialize_refuses_a_session_already_on_or_ended(self):
    """A Session starts once; after it ends, it never runs again."""
    record = self.world.create()
    self.world.start(record)
    with self.assertRaisesRegex(errors.Conflict, 'already on'):
      self.world.model_servers.initialize(record['id'])
    self.world.model_servers.close(record['id'])
    with self.assertRaisesRegex(errors.Conflict, 'has ended'):
      self.world.model_servers.initialize(record['id'])

  def test_generate_checks_the_prompt(self):
    """An empty prompt is refused at once."""
    record = self.world.create()
    self.world.start(record)
    with self.assertRaisesRegex(ValueError, 'Type a prompt'):
      self.world.model_servers.generate(record['id'], '')

  def test_cancel_refuses_starting_a_model_server(self):
    """Initialize is not cancelled; turn the Model Server off instead."""
    record = self.world.create()
    go = threading.Event()
    original = self.world.launchers.launch

    def launch(device_id, build=None):
      go.wait(fake_launchers.TIMEOUT)
      return original(device_id, build)

    self.world.launchers.launch = launch
    job = self.world.model_servers.initialize(record['id'])
    try:
      with self.assertRaisesRegex(errors.Conflict, 'turn it off'):
        self.world.model_servers.cancel(job['id'])
      summary = self.world.listing.summary(
          self.world.sessions.get(record['id'])
      )
      self.assertEqual(summary['status_text'], 'Starting')
      self.assertEqual(summary['execution']['phase'], 'starting')
      self.assertEqual(summary['execution']['activeOperation'], 'initialize')
    finally:
      go.set()
    self.assertEqual(self.world.wait(job)['status'], 'completed')

  def test_close_while_launching_holds_the_device_until_launched(self):
    """A device is not free while a closed start may still launch on it."""
    first = self.world.create(ref='android:A', target='local')
    second = self.world.create(ref='android:A', target='local')
    inside, go = threading.Event(), threading.Event()
    original = self.world.launchers.launch

    def launch(device_id, build=None):
      if device_id == 'android:A':
        inside.set()
        go.wait(fake_launchers.TIMEOUT)
      return original(device_id, build)

    self.world.launchers.launch = launch
    job = self.world.model_servers.initialize(first['id'])
    try:
      self.assertTrue(inside.wait(fake_launchers.TIMEOUT))
      self.world.model_servers.close(first['id'])
      with self.assertRaisesRegex(errors.Conflict, 'Pixel is in use'):
        self.world.model_servers.initialize(second['id'])
      self.assertEqual(
          self.world.catalog.inspect('android:A')['statusText'], 'Busy'
      )
    finally:
      go.set()
    self.assertEqual(self.world.wait(job)['status'], 'failed')
    self.assertEqual(self.world.launchers.running('android:A'), [])
    self.world.launchers.launch = original
    self.world.start(second)

  def test_a_side_reads_uploading_while_its_model_is_sent(self):
    """Its phase follows its progress: Uploading model, then Loading."""
    seen = []
    original = model_server_lib.ModelServer._set_phase

    def record(server, role, phase):
      seen.append(phase)
      return original(server, role, phase)

    with mock.patch.object(model_server_lib.ModelServer, '_set_phase', record):
      self.world.start(self.world.create(), None)
    self.assertIn('uploading', seen)
    last_upload = len(seen) - 1 - seen[::-1].index('uploading')
    self.assertIn('loading', seen[last_upload:])

  def test_a_turn_needs_the_model_server_on(self):
    """No turn without an active Model Server."""
    record = self.world.create()
    with self.assertRaisesRegex(errors.Conflict, 'to come on'):
      self.world.model_servers.generate(record['id'], 'Hi')

  def test_create_starts_the_session_or_saves_nothing(self):
    """A Session starts as it is created; one that cannot is not saved."""
    first = self.world.model_servers.create(
        self.world.config('local', 'android:A')
    )
    self.assertTrue(first['started'])
    job_id = self.world.model_servers.last_job(first['id'])
    job = self.world.wait({'id': job_id})
    self.assertEqual(job['status'], 'completed', job['error'])
    with self.assertRaisesRegex(errors.Conflict, 'Pixel is in use'):
      self.world.model_servers.create(self.world.config('android:A', 'local'))
    with self.assertRaisesRegex(ValueError, 'Reference and Target'):
      self.world.model_servers.create({'name': 'No runs'})
    self.assertEqual(
        [record['id'] for record in self.world.sessions.list()],
        [first['id']],
    )
    self.assertEqual(
        list(self.world.workspace.records.iterdir()),
        [self.world.workspace.records / f'{first["id"]}.json'],
    )

  def test_a_device_that_runs_one_runner_serves_one_side(self):
    """Android devices hold one runner, across every Session."""
    with self.assertRaisesRegex(errors.Conflict, 'different devices'):
      self.world.model_servers.initialize(
          self.world.create(ref='android:A', target='android:A')['id']
      )
    first = self.world.create(ref='local', target='android:A')
    self.world.start(first)
    second = self.world.create(ref='android:A', target='local')
    with self.assertRaisesRegex(errors.Conflict, 'Pixel is in use'):
      self.world.model_servers.initialize(second['id'])
    inspected = self.world.catalog.inspect('android:A')
    self.assertEqual(inspected['statusText'], 'Busy')
    self.assertEqual(inspected['holder'], f'In use by {first["name"]} · Target')
    # This computer runs any number of runners.
    third = self.world.create()
    self.world.start(third)
    self.world.model_servers.close(first['id'])
    self.world.start(second)

  def test_each_phone_gets_its_own_upload_one_at_a_time(self):
    """Phones keep their own stores: one upload each, never two at once."""
    _slow_uploads(self.world)
    record = self.world.create(ref='android:A', target='android:B')
    self.world.start(record)
    (ref,) = self.world.launchers.runners('android:A')
    (target,) = self.world.launchers.runners('android:B')
    self.assertEqual(ref.uploads, [self.world.model.sha256])
    self.assertEqual(target.uploads, [self.world.model.sha256])
    self.assertEqual(fake_runner.overlapping_uploads([ref, target]), [])

  def test_runners_of_this_computer_share_one_upload(self):
    """Runners on this computer share a store: one upload for both."""
    _slow_uploads(self.world)
    self.world.start(self.world.create())
    runners = self.world.launchers.runners('local')
    self.assertEqual(sum(len(runner.uploads) for runner in runners), 1)
    self.assertEqual(fake_runner.overlapping_uploads(runners), [])


class PyTorchModelServersTest(absltest.TestCase):
  """Sessions of a PyTorch model: the PyTorch runner on this computer."""

  def setUp(self):
    """This computer with the PyTorch runner, and a model directory."""
    super().setUp()
    self.world = world_lib.create(self, graphs=True)
    self.world.launchers.builds_now['local'] = [
        fake_launchers.runner_build(fake_launchers.COMMIT),
        fake_launchers.runner_build(runner_builds.PYTORCH),
    ]
    directory = self.world.workspace.hf / 'gemma'
    directory.mkdir()
    (directory / 'config.json').write_text('{}')
    (directory / 'model.safetensors').write_bytes(b'weights')
    self.model = self.world.models.add_download_directory(directory)
    self.directory = directory.resolve()

  def _create(self, **run: str) -> dict[str, Any]:
    config = self.world.config()
    for side in config['runs']:
      side.update({
          'runtime': 'PyTorch',
          'artifact': self.model.artifact,
          'runnerBuild': runner_builds.PYTORCH,
          **run,
      })
    return self.world.sessions.create(config)

  def test_a_pytorch_session_runs_without_an_upload(self):
    """The runner reads the directory it is started with; no upload."""
    self.world.launchers.stores['local'] = {self.model.sha256}
    record = self._create(dtype='bfloat16', cpuThreads='4')

    self.world.start(record)
    job = self.world.turn(record['id'], 'Hi')

    self.assertEqual(job['status'], 'completed', job.get('error'))
    launched = self.world.launchers.launched
    self.assertEqual(
        [runner.model_dir for runner in launched], [self.directory] * 2
    )
    ref, target = self.world.launchers.runners('local')
    self.assertEqual(ref.uploads + target.uploads, [])
    self.assertEqual(ref.load_request.sha256, self.model.sha256)
    self.assertEqual(
        dict(ref.load_request.runtime_options),
        # It captures its anchors, always.
        {'dtype': 'bfloat16', 'cpuThreads': 4, 'observe': 'anchors'},
    )
    index = self.world.captures.get(record['id'], 1)
    self.assertEqual(index.sides['ref']['runtime']['name'], 'PyTorch')
    contexts = kv_analysis.kv_analysis(index, 1)['contexts']
    self.assertEqual(
        [c['moment'] for c in contexts], ['prefill_post', 'terminal']
    )
    for context in contexts:
      self.assertEqual(context['runtime'], 'PyTorch')
      self.assertEqual({row['status'] for row in context['layers']}, {'ok'})
    # A PyTorch run always observes its anchors; this directory's
    # config.json names no architecture to place them by.
    self.assertIn(
        'No architecture defines model_type',
        self.world.prepared.observed(record).reason['text'],
    )

  def test_the_execution_graph_a_turn_exports_is_kept(self):
    """Once the turn is recorded, the model's graph is the one it gave."""
    self.world.launchers.stores['local'] = {self.model.sha256}
    record = self._create(observe='anchors')
    self.world.start(record)
    # Both sides run the model, and both export its graph, as its
    # runners do; the runners launch in either order.
    for runner in self.world.launchers.runners('local'):
      runner._respond.files = {
          pytorch_graph_fixtures.FILE: pytorch_graph_fixtures.gzipped(
              pytorch_graph_fixtures.document(layers=2)
          )
      }
    self.assertEqual(
        self.world.graphs.state(self.model.artifact).status,
        model_graphs.UNAVAILABLE,
    )

    job = self.world.turn(record['id'], 'Hi')

    self.assertEqual(job['status'], 'completed', job.get('error'))
    # Of two sides that exported one model's graph, the ref side's is
    # kept: the sides are handed over in role order.
    state = self.world.graphs.state(self.model.artifact)
    self.assertEqual(state.status, model_graphs.COMPLETED)
    self.assertEqual(
        [
            (entry['name'], entry['source'])
            for entry in state.manifest['signatures']
        ],
        [
            (
                name,
                {
                    'session': record['id'],
                    'chat': 1,
                    'turn': 1,
                    'side': 'ref',
                },
            )
            for name in ('prefill', 'decode')
        ],
    )
    # The captures keep the files too, as the runners wrote them.
    index = self.world.captures.get(record['id'], 1)
    for role in ('ref', 'target'):
      self.assertIn(
          pytorch_graph_fixtures.FILE,
          [f['path'] for f in index.turn(1)['sides'][role]['files']],
      )

  def test_a_model_the_pytorch_runner_lacks_fails_without_an_upload(self):
    """A Load the PyTorch runner answers NOT_FOUND is the start's error."""
    record = self._create()

    job = self.world.wait(self.world.model_servers.initialize(record['id']))

    self.assertEqual(job['status'], 'failed')
    self.assertIn('No model with sha256', job['error'])
    for runner in self.world.launchers.runners('local'):
      self.assertEqual(runner.uploads, [])

  def test_a_pytorch_run_needs_its_runner_and_this_computer(self):
    """A PyTorch run chooses the PyTorch runner, on this computer."""
    with self.assertRaisesRegex(ValueError, 'PyTorch runner'):
      self._create(runnerBuild=fake_launchers.COMMIT)
    config = self.world.config()
    config['runs'][0].update(runtime='PyTorch')
    with self.assertRaisesRegex(ValueError, 'LiteRT-LM model'):
      self.world.sessions.create(config)
    config = self.world.config()
    config['runs'][0].update(
        artifact=self.model.artifact, runnerBuild=fake_launchers.COMMIT
    )
    with self.assertRaisesRegex(ValueError, 'PyTorch model'):
      self.world.sessions.create(config)

  def test_a_pytorch_model_has_no_graph_to_convert(self):
    """The graph of a PyTorch model is unavailable, with the reason."""
    state = model_graphs.ModelGraphs(
        self.world.workspace,
        self.world.models,
        self.world.scans,
        self.world.jobs,
        convert=self.world.converter,
    ).state(self.model.artifact)

    self.assertEqual(state.status, model_graphs.UNAVAILABLE)
    self.assertEqual(state.text, model_graphs.NO_PYTORCH_GRAPH)


if __name__ == '__main__':
  test_main.main()
