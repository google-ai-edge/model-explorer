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

"""Tests of preparing Sessions that observe tensors, and of the copies kept."""

import collections
import dataclasses
import json
import os
import threading
from typing import Any
from unittest import mock

from absl.testing import absltest
from google.protobuf import json_format
from model_debugger.server.analysis import tensor_analysis
from model_debugger.server.architectures import catalog
from model_debugger.server.graph import architecture as architecture_lib
from model_debugger.server.runner import runner_builds
from model_debugger.server.session import anchor_points
from model_debugger.server.session import op_menu
from model_debugger.server.session import prepared_models
from model_debugger.server.session import session_store
from model_debugger.server.session import side as side_lib
from model_debugger.server.session import tap_scans
from model_debugger.server.session import workspace
from model_debugger.server.tap import tap_model
from model_debugger.server.testing import anchor_fixtures
from model_debugger.server.testing import fake_launchers
from model_debugger.server.testing import tap_fixtures
from model_debugger.server.testing import test_main
from model_debugger.server.testing import world as world_lib
from model_debugger.server.util import errors

_POINTS = ['decode:0:0', 'prefill_16:2:0']
# decode's output already: the runtime captures it without a copy.
_OUTPUT = ['decode:1:0']


class PreparedModelsTest(absltest.TestCase):
  """Tests of the preparations and the copies, over fake runners."""

  def setUp(self):
    """A computer and two devices, and the fixture model registered."""
    super().setUp()
    self.world = world_lib.create(self)
    path = self.world.workspace.hf / 'tap.litertlm'
    tap_fixtures.write_model(path)
    self.model = self.world.models.add_download(path)
    self.prepared = self.world.prepared

  def _config(
      self,
      points: list[str] | None = None,
      ref: str = fake_launchers.LOCAL.id,
      target: str = fake_launchers.LOCAL.id,
  ) -> dict[str, Any]:
    """What the editor posts for the fixture model and points."""
    config = self.world.config(ref, target)
    for run in config['runs']:
      run['artifact'] = self.model.artifact
    config['tap_profile'] = session_store.TAP_PROFILE
    config['tap_points'] = {self.model.artifact: points or _POINTS}
    return config

  def _create(
      self,
      points: list[str] | None = None,
      ref: str = fake_launchers.LOCAL.id,
      target: str = fake_launchers.LOCAL.id,
  ) -> dict[str, Any]:
    """A Session whose runs load the fixture model and observe points.

    As a record written before every Session captured the anchors: the
    points alone (the anchors' own tests turn them on).

    Args:
      points: Anchor points to observe.
      ref: Reference runner identifier.
      target: Target runner identifier.

    Returns:
      The created session record.
    """
    return self.world.older(
        self.world.sessions.create(self._config(points, ref, target))
    )

  def _copies(self) -> list[str]:
    """The keys of the copies in the workspace, and what is pending."""
    return sorted(p.name for p in self.world.workspace.prepared.iterdir())

  def _key(self, points: list[str]) -> str:
    return prepared_models.key(self.model.sha256, points)

  def _prepare(self, record: dict[str, Any]) -> dict[str, Any]:
    """Starts a Session, which prepares its copies; its job, ended."""
    return self.world.wait(self.world.model_servers.initialize(record['id']))

  def _progress(self, job: dict[str, Any]) -> list[str]:
    return [
        event['message']
        for event in self.world.events(job['id'])
        if event['type'] == 'progress' and 'runId' not in event
    ]

  def _held(self) -> tuple[threading.Event, collections.Counter[str]]:
    """Holds every copy write until the event is set; counts them."""
    release = threading.Event()
    written = collections.Counter()
    write = tap_model.write

    def held(*args, **kwargs):
      written[tuple(sorted(args[2]))] += 1
      assert release.wait(fake_launchers.TIMEOUT)
      return write(*args, **kwargs)

    patch = mock.patch.object(tap_model, 'write', side_effect=held)
    patch.start()
    self.addCleanup(patch.stop)
    self.addCleanup(release.set)
    return release, written

  def test_turning_on_prepares_the_copy_both_sides_load(self):
    """One job: the copy is written, then the sides start and load it."""
    record = self._create()
    summary = self.world.summary(record['id'])
    self.assertEqual(summary['tap_points'], {self.model.artifact: _POINTS})
    self.assertEqual(summary['tap_preparation']['state'], 'needed')
    job = self.world.start(record, None)
    self.assertEqual(job['operation'], 'initialize')
    name = self.model.name
    self.assertEqual(
        self._progress(job),
        [
            f'{name}: Adding 2 outputs',
            f'{name}: Writing the copy',
            f'{name}: Checking the copy against the model',
            f'{name}: Hashing the copy',
            f'{name}: its copy is prepared',
        ],
    )
    summary = self.world.summary(record['id'])
    self.assertEqual(summary['tap_preparation']['state'], 'prepared')
    self.assertEqual(summary['job_id'], job['id'])
    found = self.prepared.find(self.model, _POINTS)
    self.assertEqual(
        found.model.path,
        self.world.workspace.prepared
        / prepared_models.key(self.model.sha256, _POINTS)
        / 'model.litertlm',
    )
    self.assertEqual(found.model.name, f'{name} (tapped, 2 points)')
    self.assertEqual(
        [point['capture_key'] for point in found.manifest['points']],
        ['post_tap_sg0_op0_out0', 'post_tap_sg2_op2_out0'],
    )
    self.assertEqual(found.manifest['source']['sha256'], self.model.sha256)
    self.assertEqual(found.manifest['identity'], tap_model.identity())
    for runner in self.world.launchers.runners('local'):
      self.assertEqual(runner.load_request.sha256, found.model.sha256)
    self.world.chat(record['id'])
    index = self.world.captures.get(record['id'], 1)
    self.assertEqual(
        index.sides['target']['model']['sha256'], found.model.sha256
    )

  def test_a_start_shows_its_preparation_until_it_is_done(self):
    """The listing says what it prepares; the start then goes on."""
    release, written = self._held()
    record = self._create()
    job = self.world.model_servers.initialize(record['id'])
    world_lib.eventually(lambda: written)
    summary = self.world.summary(record['id'])
    self.assertEqual(
        summary['tap_preparation'],
        {'state': 'preparing', 'text': 'Preparing the tensor capture'},
    )
    self.assertEqual(summary['model_server']['state'], 'starting')
    self.assertEqual(summary['job_id'], job['id'])
    self.assertEqual(self.world.launchers.runners('local'), [])
    release.set()
    job = self.world.wait(job)
    self.assertEqual(job['status'], 'completed', job['error'])
    self.assertEqual(
        self.world.summary(record['id'])['model_server']['state'], 'on'
    )

  def test_one_copy_is_written_at_a_time(self):
    """Another waits, and says so; turned off, it stops waiting."""
    release, written = self._held()
    first = self._create()
    started = self.world.model_servers.initialize(first['id'])
    world_lib.eventually(lambda: written)
    other = self._create(['decode:0:0'])
    waiting = self.world.model_servers.initialize(other['id'])
    third = self._create(
        ['verify:0:0'], fake_launchers.PHONE.id, fake_launchers.TABLET.id
    )
    stopped = self.world.model_servers.initialize(third['id'])
    for record in (other, third):
      world_lib.eventually(
          lambda record=record: (
              prepared_models.WAITING
              in self.world.summary(record['id'])['tap_preparation']['text']
          )
      )
    self.world.model_servers.close(third['id'])
    stopped = self.world.wait(stopped)
    self.assertEqual(stopped['status'], 'failed')
    self.assertIn('turned off while starting', stopped['error'])
    self.assertLen(written, 1)
    release.set()
    self.assertEqual(self.world.wait(started)['status'], 'completed')
    waiting = self.world.wait(waiting)
    self.assertEqual(waiting['status'], 'completed', waiting['error'])
    self.assertEqual(
        self._progress(waiting)[0],
        f'{self.model.name}: {prepared_models.WAITING}',
    )
    self.assertEqual(
        written,
        {tuple(sorted(_POINTS)): 1, ('decode:0:0',): 1},
    )
    self.assertIsNone(self.prepared.find(self.model, ['verify:0:0']))

  def test_the_same_points_of_the_same_model_are_prepared_once(self):
    """Another Session reuses the copy; a copy written anew is the same."""
    first = self._prepare(self._create())
    self.assertEqual(first['status'], 'completed', first['error'])
    self.assertEqual(first['operation'], 'initialize')
    found = self.prepared.find(self.model, _POINTS)
    written = found.model.path.stat().st_mtime_ns
    again = self._prepare(self._create(list(reversed(_POINTS))))
    self.assertEqual(again['status'], 'completed', again['error'])
    self.assertEqual(
        self._progress(again),
        [f'{self.model.name}: its copy is prepared already'],
    )
    self.assertEqual(found.model.path.stat().st_mtime_ns, written)
    self.assertLen(list(self.world.workspace.prepared.iterdir()), 1)
    # Deleted, it is written again: the same bytes.
    os.unlink(found.model.path)
    self.assertIsNone(self.prepared.find(self.model, _POINTS))
    self._prepare(self._create())
    self.assertEqual(
        self.prepared.find(self.model, _POINTS).model.sha256,
        found.model.sha256,
    )

  def test_a_copy_older_code_wrote_is_not_reused(self):
    """The key names the code that writes copies."""
    self._prepare(self._create())
    self.assertIsNotNone(self.prepared.find(self.model, _POINTS))
    with mock.patch.object(tap_model, 'identity', return_value='newer'):
      self.assertIsNone(self.prepared.find(self.model, _POINTS))

  def test_points_that_are_outputs_already_need_no_copy(self):
    """The model loads as it is; nothing is written."""
    record = self._create(_OUTPUT)
    job = self.world.start(record)
    self.assertEqual(
        self._progress(job)[-1],
        f'{self.model.name}: each tensor chosen is an output already; '
        'it loads as is',
    )
    self.assertEqual(list(self.world.workspace.prepared.iterdir()), [])
    for runner in self.world.launchers.runners('local'):
      self.assertEqual(runner.load_request.sha256, self.model.sha256)
    self.assertEqual(
        self.world.summary(record['id'])['tap_preparation']['state'],
        'prepared',
    )

  def test_a_copy_changed_on_disk_is_not_prepared(self):
    """A copy not the size and time it was written with is written anew."""
    record = self._create()
    self._prepare(record)
    path = self.prepared.find(self.model, _POINTS).model.path
    with open(path, 'ab') as file:
      file.write(b'x')
    self.assertIsNone(self.prepared.find(self.model, _POINTS))
    self.assertEqual(
        self.world.summary(record['id'])['tap_preparation']['state'],
        'needed',
    )
    with self.assertRaisesRegex(errors.Conflict, 'not prepared'):
      self.prepared.capture_model(record, record['runs'][0])
    self.world.start(self._create())
    self.assertIsNotNone(self.prepared.find(self.model, _POINTS))

  def test_too_little_disk_fails_the_preparation_and_leaves_nothing(self):
    """The start says why; the server goes on."""
    record = self._create()
    usage = mock.patch.object(
        prepared_models.shutil,
        'disk_usage',
        return_value=mock.Mock(free=self.model.size),
    )
    with usage:
      job = self._prepare(record)
      self.assertEqual(job['status'], 'failed')
      self.assertRegex(
          job['error'],
          r'^Preparing the tensor capture failed: There is not enough '
          r'disk space for a copy of tap.litertlm: it takes .* MiB, and '
          r'.* MiB is free\.$',
      )
    self.assertEqual(list(self.world.workspace.prepared.iterdir()), [])
    self.assertEqual(self.world.launchers.runners('local'), [])
    # That start ended its Session; with room again, another starts.
    self.world.start(self._create())

  def test_a_shared_copy_goes_with_the_last_session_that_uses_it(self):
    """One that ran keeps it too; it goes once both are deleted."""
    first = self._create()
    second = self._create(list(reversed(_POINTS)))
    job = self.world.start(first)
    self.assertEqual(job['status'], 'completed', job['error'])
    self.world.model_servers.close(first['id'])
    self.assertEqual(self.prepared.collect(), [])
    self.assertEqual(self._copies(), [self._key(_POINTS)])
    self.world.sessions.delete(first['id'])
    self.assertEqual(self.prepared.collect(), [])
    self.assertEqual(
        self.world.summary(second['id'])['tap_preparation']['state'],
        'prepared',
    )
    self.world.sessions.delete(second['id'])
    self.assertEqual(self.prepared.collect(), [self._key(_POINTS)])
    self.assertEqual(self._copies(), [])

  def test_the_server_start_deletes_the_copies_no_session_uses(self):
    """A copy left by a crash, or of older code, goes; a used one stays."""
    record = self._create()
    self._prepare(record)
    workspace_ = self.world.workspace
    orphan = workspace_.prepared / ('0' * 64)
    orphan.mkdir()
    (orphan / 'model.litertlm').write_bytes(b'left by a crash')
    older = workspace_.prepared / prepared_models.key(
        self.model.sha256, ['decode:0:0']
    )
    older.mkdir()
    (older / 'model.litertlm').write_bytes(b'of a Session long changed')
    pending = workspace_.prepared / (workspace.PENDING + 'abc')
    pending.mkdir()
    prepared_models.PreparedModels(
        workspace_,
        self.world.models,
        self.world.sessions,
        self.world.jobs,
        self.world.scans,
    )
    self.assertEqual(
        self._copies(), [workspace.PENDING + 'abc', self._key(_POINTS)]
    )
    # Newer code names its copies anew: the one older code wrote goes.
    written = self._key(_POINTS)
    with mock.patch.object(tap_model, 'identity', return_value='newer'):
      self.assertEqual(self.prepared.collect(), [written])

  def test_no_copy_is_deleted_while_a_start_writes_it(self):
    """Points named no more meanwhile: the copy goes once it is loaded."""
    release, written = self._held()
    record = self._create()
    job = self.world.model_servers.initialize(record['id'])
    world_lib.eventually(lambda: written)
    self.world.sessions.set_flags(
        record['id'], tap_anchors=False, tap_profile='', tap_points={}
    )
    self.assertEqual(self.prepared.collect(), [])
    release.set()
    self.assertEqual(self.world.wait(job)['status'], 'completed')
    self.assertEqual(self._copies(), [])

  def test_no_copy_is_deleted_while_a_start_loads_it(self):
    """A start holds its copies until both sides loaded them."""
    first = self._create()
    self._prepare(first)
    self.world.model_servers.close(first['id'])
    sha256 = self.prepared.find(self.model, _POINTS).model.sha256
    record = self._create()
    loading = threading.Event()
    release = threading.Event()
    self.addCleanup(release.set)
    load = side_lib.Side.load

    def held(side, *args, **kwargs):
      loading.set()
      assert release.wait(fake_launchers.TIMEOUT)
      return load(side, *args, **kwargs)

    self.enter_context(mock.patch.object(side_lib.Side, 'load', held))
    job = self.world.model_servers.initialize(record['id'])
    self.assertTrue(loading.wait(fake_launchers.TIMEOUT))
    # No record names the copy any more as it starts.
    for each in (first, record):
      self.world.sessions.set_flags(
          each['id'], tap_anchors=False, tap_profile='', tap_points={}
      )
    self.assertEqual(self.prepared.collect(), [])
    self.assertEqual(self._copies(), [self._key(_POINTS)])
    release.set()
    job = self.world.wait(job)
    self.assertEqual(job['status'], 'completed', job['error'])
    for runner in self.world.launchers.runners('local'):
      self.assertEqual(runner.load_request.sha256, sha256)
    # Loaded, it is let go of: no Session uses it now.
    self.assertEqual(self._copies(), [])

  def test_capture_model_is_the_runs_own_without_points(self):
    """A run of a model the Session observes nothing of loads it."""
    plain = self.world.create()
    self.assertEqual(
        self.prepared.capture_model(plain, plain['runs'][0]),
        self.world.model,
    )

  def test_observed_is_each_runs_model_and_points(self):
    """Both sides' points as the scan lists them; or why there are none."""
    record = self._create()
    observed = self.prepared.observed(record)
    self.assertIsNone(observed.reason)
    for side in ('ref', 'target'):
      self.assertEqual(observed.sides[side].model_sha256, self.model.sha256)
      self.assertEqual([p.id for p in observed.sides[side].points], _POINTS)
    self.assertEqual(
        self.prepared.observed(self.world.create()).reason['code'],
        tensor_analysis.NOT_OBSERVED,
    )
    gone = dict(record, tap_points={'model-gone': _POINTS})
    gone['runs'] = [dict(run, artifact='model-gone') for run in gone['runs']]
    why = self.prepared.observed(gone).reason
    self.assertEqual(why['code'], tensor_analysis.MODEL_UNREADABLE)
    self.assertIn('the ref side observes', why['text'])

  def test_the_scans_keep_one_scan_per_model(self):
    """A scan runs once; its state says where it is."""
    scans = tap_scans.TapScans(self.world.models)
    self.addCleanup(scans.close)
    self.assertEqual(scans.state(self.model.artifact).status, tap_scans.IDLE)
    self.assertIsNone(scans.kept(self.model))
    scans.start(self.model.artifact)
    world_lib.eventually(
        lambda: (scans.state(self.model.artifact).status == tap_scans.COMPLETED)
    )
    state = scans.state(self.model.artifact)
    self.assertLen(state.scan.points, 16)
    self.assertIs(scans.start(self.model.artifact).scan, state.scan)
    self.assertIs(scans.scan(self.model), state.scan)
    self.assertIs(scans.kept(self.model), state.scan)
    scans.start(self.world.model.artifact)
    world_lib.eventually(
        lambda: (
            scans.state(self.world.model.artifact).status == tap_scans.FAILED
        )
    )
    self.assertIn(
        'not a .litertlm', scans.state(self.world.model.artifact).error
    )
    tap_model.check_points(state.scan, _POINTS)


class AnchorsTest(absltest.TestCase):
  """Sessions that capture the anchors, which the graph binds."""

  def setUp(self):
    """A world that keeps graphs; the fixture model is Gemma 4's."""
    super().setUp()
    self.world = world_lib.create(self, graphs=True)
    path = self.world.workspace.hf / 'tap.litertlm'
    tap_fixtures.write_model(path)
    self.model = self.world.models.add_download(path)
    patch = mock.patch.object(
        self.world.graphs,
        'architecture',
        return_value=anchor_fixtures.architecture(),
    )
    patch.start()
    self.addCleanup(patch.stop)
    self.prepared = self.world.prepared

  def _create(self, **fields) -> dict[str, Any]:
    config = self.world.config()
    for run in config['runs']:
      run['artifact'] = self.model.artifact
    config.update(tap_anchors=True, **fields)
    return self.world.sessions.create(config)

  def test_a_start_finds_the_anchors_and_both_sides_load_their_copy(self):
    """Found once the graph is converted, kept, and placed by anchor."""
    record = self._create(
        tap_profile=session_store.TAP_PROFILE,
        tap_points={self.model.artifact: ['prefill_4:0:0']},
    )
    self.assertEqual(
        self.world.summary(record['id'])['tap_preparation']['text'],
        prepared_models.ANCHORS_NEEDED,
    )

    job = self.world.start(record)

    self.assertEqual(job['status'], 'completed', job.get('error'))
    found = self.world.sessions.get(record['id'])
    anchors = list(
        anchor_points.find(
            self.world.graphs, self.world.scans, self.model
        ).point_ids
    )
    self.assertEqual(found['tap_anchor_points'], {self.model.artifact: anchors})
    observed = anchors + ['prefill_4:0:0']
    self.assertEqual(
        session_store.tap_points(found), {self.model.artifact: observed}
    )
    copy = self.prepared.find(self.model, observed).model
    for runner in self.world.launchers.runners('local'):
      self.assertEqual(runner.load_request.sha256, copy.sha256)
    progress = [
        event['message']
        for event in self.world.events(job['id'])
        if event['type'] == 'progress' and 'runId' not in event
    ]
    self.assertIn(
        'tap.litertlm: its anchors are 6 tensor outputs: decode: 3 tensor '
        'outputs, the anchors of 2 layers; 2 anchors skipped; prefill_16: '
        '3 tensor outputs, the anchors of 1 layer; 5 anchors skipped; '
        'verify: skipped; it runs only with speculative decoding, which '
        'cannot be enabled for now, so its anchors would capture nothing',
        progress,
    )
    summary = self.world.summary(record['id'])
    self.assertTrue(summary['tap_anchors'])
    self.assertEqual(
        summary['tap_points'], {self.model.artifact: ['prefill_4:0:0']}
    )
    self.assertNotIn('tap_anchor_points', summary)
    side = self.prepared.observed(found).sides['ref']
    # One runtime's anchors are placed as what they are; a point chosen
    # that is no anchor by its layer.
    self.assertEqual(
        side.places['decode:1:0'],
        ((0, 'output'), (1, 'input'), (1, 'output')),
    )
    self.assertNotIn('prefill_4:0:0', side.places)
    layout = tensor_analysis.layout(self.prepared.observed(found))
    self.assertIsNone(layout.reason)
    self.assertEqual(
        layout.anchors[:4],
        ('input', 'intermediate.1', 'intermediate.2', 'output'),
    )

  def test_a_model_of_no_architecture_observes_nothing(self):
    """Its anchors found to be none, it loads its own model."""
    self.world.graphs.architecture.return_value = {
        **anchor_fixtures.architecture(),
        'architecture': None,
        'text': 'No architecture template exists for this model.',
        'bindings': {},
    }
    record = self._create()

    job = self.world.start(record)

    self.assertEqual(job['status'], 'completed', job.get('error'))
    found = self.world.sessions.get(record['id'])
    self.assertEqual(found['tap_anchor_points'], {self.model.artifact: []})
    self.assertEqual(self.prepared.view(found)['state'], 'none')
    for runner in self.world.launchers.runners('local'):
      self.assertEqual(runner.load_request.sha256, self.model.sha256)


def _tiny_gemma4_config() -> dict[str, Any]:
  """E2B's config.json, shrunk to two layers of width 4: q_proj returns 4.

  Its layer 1 shares the keys and values of layer 0.

  Returns:
    The modified Gemma 4 configuration dictionary.
  """
  (gemma4,) = [a for a in catalog.architectures() if a.id == 'gemma4']
  entry = gemma4.entry('gemma-4-E2B-it')
  config = json.loads(
      (gemma4.directory / entry['pinned']['file']).read_text('utf-8')
  )
  text = config['text_config']
  text.update(
      hidden_size=4,
      intermediate_size=8,
      num_attention_heads=1,
      head_dim=4,
      global_head_dim=4,
      num_hidden_layers=2,
      num_kv_shared_layers=1,
      hidden_size_per_layer_input=2,
      layer_types=text['layer_types'][:2],
      use_double_wide_mlp=False,
  )
  return config


class CounterpartsTest(absltest.TestCase):
  """A LiteRT-LM model's points chosen beside a PyTorch run."""

  def setUp(self):
    """The fixture LiteRT-LM model, Gemma 4's, and a tiny PyTorch one.

    The LiteRT-LM binding maps one semantic operator, q_proj, to the
    ADD of each layer (the fixture's `sum`).
    """
    super().setUp()
    self.world = world_lib.create(self, graphs=True)
    self.world.launchers.builds_now['local'] = [
        fake_launchers.runner_build(fake_launchers.COMMIT),
        fake_launchers.runner_build(runner_builds.PYTORCH),
    ]
    path = self.world.workspace.hf / 'tap.litertlm'
    tap_fixtures.write_model(path)
    self.model = self.world.models.add_download(path)
    directory = self.world.workspace.hf / 'gemma'
    directory.mkdir()
    config = _tiny_gemma4_config()
    (directory / 'config.json').write_text(json.dumps(config))
    (directory / 'model.safetensors').write_bytes(b'weights')
    self.pytorch = self.world.models.add_download_directory(directory)
    self.world.launchers.stores['local'] = {self.pytorch.sha256}
    _, document = catalog.generate(config)
    self.enter_context(
        mock.patch.object(
            self.world.graphs,
            'architecture',
            return_value={
                **anchor_fixtures.architecture(),
                'skeleton': document,
            },
        )
    )
    (known,) = [a for a in architecture_lib.architectures() if a.id == 'gemma4']
    binding = dict(
        known.binding,
        ops={'attn/q_proj': {'scope': '{layer}', 'operator': 'ADD'}},
    )
    self.enter_context(
        mock.patch.object(
            architecture_lib,
            'architectures',
            return_value=[dataclasses.replace(known, binding=binding)],
        )
    )
    self.prepared = self.world.prepared

  def _create(self, points: list[str]) -> dict[str, Any]:
    config = self.world.config()
    config['runs'][0].update(
        runtime='PyTorch',
        artifact=self.pytorch.artifact,
        runnerBuild=runner_builds.PYTORCH,
        observe='anchors',
    )
    config['runs'][1]['artifact'] = self.model.artifact
    config.update(
        tap_profile=session_store.TAP_PROFILE,
        tap_points={self.model.artifact: points},
    )
    return self.world.sessions.create(config)

  def test_a_point_of_a_semantic_operator_is_its_modules(self):
    """Placed at q_proj; the PyTorch side loads to observe its module."""
    record = self._create(['prefill_4:0:0', 'prefill_4:1:0', 'prefill_4:2:0'])

    job = self.world.start(record)

    self.assertEqual(job['status'], 'completed', job.get('error'))
    found = self.world.sessions.get(record['id'])
    module = 'model.language_model.layers.0.self_attn.q_proj'
    self.assertEqual(
        self.prepared.observe_modules(found, found['runs'][0]),
        [{'module': module, 'edge': 'out'}],
    )
    self.assertEqual(self.prepared.observe_modules(found, found['runs'][1]), [])
    loads = {
        runner.load_request.sha256: runner.load_request
        for runner in self.world.launchers.runners('local')
    }
    options = json_format.MessageToDict(
        loads[self.pytorch.sha256].runtime_options
    )
    self.assertEqual(
        options,
        {
            'observe': 'anchors',
            'observeModules': [{'module': module, 'edge': 'out'}],
        },
    )
    observed = self.prepared.observed(found)
    target = observed.sides['target']
    self.assertEqual(target.places['prefill_4:0:0'], ((0, 'attn/q_proj'),))
    self.assertNotIn('prefill_4:1:0', target.places)
    self.assertEqual(target.labels['attn/q_proj'], 'Attention q_proj')
    ref = observed.sides['ref']
    self.assertEqual(ref.places[f'decode:{module}:out'], ((0, 'attn/q_proj'),))
    self.assertEqual(
        ref.keys[f'decode:{module}:out'], f'post_module_{module}_out'
    )
    notes = [note['text'] for note in observed.notes]
    self.assertEqual(
        notes,
        [
            (
                'PyTorch has no counterpart for prefill_4:1:0 (layer 0, MUL): '
                'it is a MUL (float32) inside Attention q_proj, not its '
                'output, the ADD; only the anchors are captured there.'
            ),
            (
                'PyTorch has no counterpart for prefill_4:2:0 (layer 0, ADD): '
                'its shape [1, 1, 10, 2] is not that of what '
                'model.language_model.layers.0.self_attn.q_proj returns, '
                '[batch, rows, 4]; only the anchors are captured there.'
            ),
        ],
    )
    layout = tensor_analysis.layout(observed)
    self.assertIsNone(layout.reason)
    self.assertIn('attn/q_proj', layout.anchors)
    self.assertEqual(layout.notes, observed.notes)

  def test_the_editor_is_told_how_each_point_pairs(self):
    """An anchor, a module, or why not; nothing until the graph is in."""
    points = ['decode:1:0', 'prefill_4:0:0', 'prefill_4:1:0']
    pending = self.prepared.counterparts(self.model, self.pytorch, points)
    self.assertEqual(pending['status'], anchor_points.PENDING)
    self.world.start(self._create(points))

    found = self.prepared.counterparts(self.model, self.pytorch, points)

    self.assertEqual(found['status'], anchor_points.FOUND)
    anchor, module, unpaired = found['points']
    self.assertEqual(anchor['paired'], 'anchor')
    self.assertEqual(anchor['place'], 'output')
    self.assertEqual(
        module['text'],
        'Layer 0 Attention q_proj: paired with PyTorch '
        'model.language_model.layers.0.self_attn.q_proj, what it returns.',
    )
    self.assertEqual(unpaired['paired'], '')
    self.assertIn('not its output, the ADD;', unpaired['text'])
    # What the PyTorch side's box says it captures beside its anchors.
    self.assertEqual(
        found['pytorch_text'],
        f'Beside its anchors, {self.pytorch.name} also captures 1 module '
        f'paired with the outputs chosen of {self.model.name}; 1 output '
        'chosen has no module paired with it.',
    )

  def _by_hand_graphs(self) -> None:
    """Converts the LiteRT-LM model's graph; serves two PyTorch outputs.

    The PyTorch model's graph is none the fake runner exports: its
    outputs are given, o_proj's input of layer 0 (`attn`) and q_proj's
    output (`q`), each of the shape of the LiteRT-LM point
    prefill_4:1:0.
    """
    self.world.start(self._create(['prefill_4:1:0']))
    real = self.world.graphs.tensor
    shape = real(self.model, 'prefill_4', '1', '0')['tensor_shape']
    module = 'decode:model.language_model.layers.0.self_attn.'
    outputs = {
        'attn': {'point_id': module + 'o_proj:in', 'layer': '0'},
        'q': {'point_id': module + 'q_proj:out', 'layer': '0'},
    }

    def tensor(model, signature, node, output):
      if model.sha256 != self.pytorch.sha256:
        return real(model, signature, node, output)
      if node not in outputs:
        raise errors.NotFound(f'No node {node}.')
      return {
          **outputs[node],
          'observable': 'true',
          'tensor_shape': shape,
      }

    self.enter_context(
        mock.patch.object(self.world.graphs, 'tensor', side_effect=tensor)
    )

  def test_a_pair_made_by_hand_is_placed_and_its_module_observed(self):
    """Every Session of the two models has it; the editor is told."""
    self._by_hand_graphs()
    record = self._create(['prefill_4:1:0'])
    attn = {'signature': 'decode', 'node': 'attn', 'output': '0'}
    litert = {'signature': 'prefill_4', 'node': '1', 'output': '0'}

    answer = self.prepared.save_pair(record, attn, litert, 'by hand')

    self.assertEqual(answer['status'], 'pairs')
    (pair,) = answer['pairs']
    self.assertEqual(pair['label'], 'by hand')
    self.assertEqual(
        pair['place'], {'layer': 0, 'anchor': f'manual:{pair["id"]}'}
    )
    self.assertEqual(pair['text'], '')
    self.assertEqual(pair['target']['point'], 'prefill_4:1:0')
    module = 'model.language_model.layers.0.self_attn.o_proj'
    self.assertEqual(
        self.prepared.observe_modules(record, record['runs'][0]),
        [{'module': module, 'edge': 'in'}],
    )
    observed = self.prepared.observed(record)
    self.assertEqual(
        observed.sides['ref'].keys[f'decode:{module}:in'],
        f'post_module_{module}_in',
    )
    self.assertEqual(
        observed.sides['target'].labels[f'manual:{pair["id"]}'], 'by hand'
    )
    # It is paired by hand, so no note says PyTorch has none for it.
    self.assertEqual(observed.notes, ())
    layout = tensor_analysis.layout(observed)
    self.assertIn(f'manual:{pair["id"]}', layout.anchors)
    # Started before it was saved, its runner was not asked for it.
    self.assertIn(
        'was not asked for',
        self.prepared.manual_pairs(record, {'ref': None, 'target': None})[
            'pairs'
        ][0]['text'],
    )
    found = self.prepared.counterparts(
        self.model, self.pytorch, ['prefill_4:1:0']
    )
    self.assertEqual(found['points'][0]['paired'], 'manual')
    # A Session that does not observe the point says how to.
    other = self._create(['prefill_4:0:0'])
    (pair,) = self.prepared.manual_pairs(other)['pairs']
    self.assertIsNone(pair['place'])
    self.assertIn(
        'prefill_4:1:0 is not observed in this Session: add it in '
        'Tensor capture',
        pair['text'],
    )
    notes = self.prepared.observed(other).notes
    self.assertIn(
        tensor_analysis.MANUAL_NOT_OBSERVED, [n['code'] for n in notes]
    )
    # Deleted, it is gone for both.
    self.prepared.delete_pair(record, pair['id'])
    self.assertEqual(self.prepared.manual_pairs(other)['pairs'], [])

  def test_a_pair_the_server_makes_itself_is_refused(self):
    """A duplicate, an automatic pair, one model: each says why."""
    self._by_hand_graphs()
    record = self._create(['prefill_4:0:0', 'prefill_4:1:0'])
    attn = {'signature': 'decode', 'node': 'attn', 'output': '0'}
    litert = {'signature': 'prefill_4', 'node': '1', 'output': '0'}
    self.prepared.save_pair(record, attn, litert)
    with self.assertRaisesRegex(errors.Conflict, 'paired already'):
      self.prepared.save_pair(record, attn, litert)
    # q_proj's module is the counterpart of prefill_4:0:0 (the fixture
    # binding's ADD).
    with self.assertRaisesRegex(
        errors.Conflict, 'without a pair made by hand: Layer 0 Attention'
    ):
      self.prepared.save_pair(
          record,
          {'signature': 'decode', 'node': 'q', 'output': '0'},
          {'signature': 'prefill_4', 'node': '0', 'output': '0'},
      )
    config = self.world.config()
    config['runs'][0]['artifact'] = self.model.artifact
    config['runs'][1]['artifact'] = self.model.artifact
    same = self.world.sessions.create(config)
    answer = self.prepared.manual_pairs(same)
    self.assertEqual(answer['status'], 'identity')
    self.assertIn('same model', answer['text'])
    with self.assertRaisesRegex(ValueError, 'same model'):
      self.prepared.save_pair(same, attn, attn)

  def _operators(self) -> None:
    """The fixture binding maps q_proj and k_proj to each layer's MUL."""
    (known,) = [a for a in architecture_lib.architectures() if a.id == 'gemma4']
    rule = {'scope': '{layer}', 'operator': 'MUL'}
    binding = dict(
        known.binding, ops={'attn/q_proj': rule, 'attn/k_proj': rule}
    )
    self.enter_context(
        mock.patch.object(
            architecture_lib,
            'architectures',
            return_value=[dataclasses.replace(known, binding=binding)],
        )
    )

  def test_operators_chosen_are_captured_in_every_layer_of_both_sides(self):
    """LiteRT-LM's points and PyTorch's modules, paired by (layer, op)."""
    self._operators()
    config = self.world.config()
    config['runs'][0].update(
        runtime='PyTorch',
        artifact=self.pytorch.artifact,
        runnerBuild=runner_builds.PYTORCH,
        observe='anchors',
    )
    config['runs'][1]['artifact'] = self.model.artifact
    config['tap_ops'] = ['attn/q_proj', 'attn/k_proj']
    record = self.world.sessions.create(config)

    job = self.world.start(record)

    self.assertEqual(job['status'], 'completed', job.get('error'))
    found = self.world.sessions.get(record['id'])
    # Every signature the anchors are captured in: not verify.
    self.assertEqual(
        found['tap_op_points'],
        {
            self.model.artifact: [
                'decode:1:0',
                'prefill_4:1:0',
                'prefill_16:1:0',
            ]
        },
    )
    layers = 'model.language_model.layers'
    # Layer 1 shares layer 0's keys and values: no k_proj.
    self.assertEqual(
        self.prepared.observe_modules(found, found['runs'][0]),
        [
            {'module': f'{layers}.0.self_attn.q_proj', 'edge': 'out'},
            {'module': f'{layers}.1.self_attn.q_proj', 'edge': 'out'},
            {'module': f'{layers}.0.self_attn.k_proj', 'edge': 'out'},
        ],
    )
    observed = self.prepared.observed(found)
    # The fixture's MUL is its layer's output too, an anchor.
    self.assertEqual(
        observed.sides['target'].places['decode:1:0'][-2:],
        ((0, 'attn/q_proj'), (0, 'attn/k_proj')),
    )
    self.assertEqual(
        observed.sides['ref'].places[f'decode:{layers}.0.self_attn.k_proj:out'],
        ((0, 'attn/k_proj'),),
    )
    layout = tensor_analysis.layout(observed)
    self.assertIsNone(layout.reason)
    self.assertEqual(
        set(layout.points['ref']) & set(layout.points['target']),
        {
            (0, 'attn/q_proj'),
            (0, 'attn/k_proj'),
            *(
                place
                for place in layout.points['target']
                if not place[1].startswith('attn/')
            ),
        },
    )
    self.assertEqual(layout.labels['attn/k_proj'], 'Attention k_proj')

  def test_one_model_on_both_sides_places_its_operators_alike(self):
    """Two LiteRT-LM sides of one file pair at (layer, operator)."""
    self._operators()
    config = self.world.config()
    for run in config['runs']:
      run['artifact'] = self.model.artifact
    config['tap_ops'] = ['attn/q_proj']
    record = self.world.sessions.create(config)

    self.world.start(record)

    layout = tensor_analysis.layout(
        self.prepared.observed(self.world.sessions.get(record['id']))
    )
    self.assertIsNone(layout.reason)
    self.assertEqual(
        layout.points['ref'][(0, 'attn/q_proj')],
        layout.points['target'][(0, 'attn/q_proj')],
    )
    self.assertEqual(
        sorted(layout.points['ref'][(0, 'attn/q_proj')]),
        ['decode', 'prefill_16', 'prefill_4'],
    )

  def test_the_editor_lists_the_operators_inside_the_layers(self):
    """By block, where each side lacks one and why, and its size."""
    self._operators()
    menu = op_menu.OpMenu(
        self.world.models, self.world.graphs, self.world.scans
    )
    pending = menu.menu(self.pytorch.artifact, self.model.artifact)
    self.assertEqual(pending['status'], op_menu.PENDING)
    self.assertIn('found in its graph, which is converted', pending['text'])
    config = self.world.config()
    for run in config['runs']:
      run['artifact'] = self.model.artifact
    self.world.start(self.world.sessions.create(config))

    found = menu.menu(
        self.pytorch.artifact, self.model.artifact, ['attn/k_proj']
    )

    self.assertEqual(found['status'], op_menu.FOUND, found['text'])
    self.assertEqual(
        [block['name'] for block in found['blocks']],
        ['Attention', 'Feed Forward', 'Per-Layer Embedding'],
    )
    ops = {op['id']: op for op in found['blocks'][0]['ops']}
    q_proj, k_proj = ops['attn/q_proj'], ops['attn/k_proj']
    # The fixture's LiteRT-LM binding maps no other.
    self.assertFalse(ops['attn/o_proj']['available'])
    self.assertEqual(
        ops['attn/o_proj']['reason'],
        'LiteRT-LM: the LiteRT-LM binding maps no tensor of it',
    )
    self.assertEqual(
        (q_proj['id'], q_proj['name'], q_proj['kind']),
        ('attn/q_proj', 'q_proj', 'Linear'),
    )
    # The fixture's LiteRT-LM model has layer 0 alone.
    self.assertEqual(
        q_proj['reason'],
        'LiteRT-LM: not in layer 1: its LiteRT-LM graph has no tensor of '
        'it there',
    )
    self.assertEqual(
        k_proj['reason'], "not in layer 1 (they reuse layer 0's K and V)"
    )
    # LiteRT-LM's MUL [1, 1, 4] float32; PyTorch's k_proj 4 bfloat16.
    self.assertEqual(k_proj['size'], '24 B')
    self.assertEqual(
        k_proj['size_detail'],
        'Reference 8 B · Target 16 B per generated token',
    )
    self.assertEqual(found['total']['bytes'], 24)
    self.assertEqual(found['refusal'], '')
    everything = menu.menu(
        self.model.artifact, self.model.artifact, ['attn/q_proj']
    )
    self.assertIn(
        'tap.litertlm: 1 more tensor output beside its 6 anchors.',
        everything['total']['text'],
    )
    with mock.patch.object(tap_model, 'MAX_POINTS', 3):
      refused = menu.menu(
          self.model.artifact, self.model.artifact, ['attn/q_proj']
      )
    self.assertIn('a copy of it observes at most 3', refused['refusal'])


if __name__ == '__main__':
  test_main.main()
