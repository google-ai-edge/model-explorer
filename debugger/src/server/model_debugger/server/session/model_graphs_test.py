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

"""Tests of the models' graphs: converted as a Session starts, then kept."""

import gzip
import json
import pathlib
import shutil
import threading
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from model_debugger.server.analysis import tensor_analysis
from model_debugger.server.architectures import catalog
from model_debugger.server.graph import architecture as architecture_lib
from model_debugger.server.graph import graph_converter
from model_debugger.server.graph import model_graph
from model_debugger.server.session import model_graphs
from model_debugger.server.session import model_server as model_server_lib
from model_debugger.server.session import session_store
from model_debugger.server.testing import fake_launchers
from model_debugger.server.testing import pytorch_graph_fixtures
from model_debugger.server.testing import tap_fixtures
from model_debugger.server.testing import test_main
from model_debugger.server.testing import world as world_lib
from model_debugger.server.util import errors
from model_debugger.server.util import json_file

_POINTS = ['decode:0:0', 'prefill_16:2:0']


def _blocked(converting: threading.Event, release: threading.Event):
  """A FakeConverter.before that waits for `release`, or to be stopped."""

  def before(stopped):
    converting.set()
    while not release.is_set():
      if stopped():
        raise graph_converter.Stopped()
      threading.Event().wait(0.01)

  return before


class ModelGraphsTest(parameterized.TestCase):
  """Tests of ModelGraphs, over fake runners and the fake converter."""

  def setUp(self):
    """A world that keeps graphs, and the fixture model registered."""
    super().setUp()
    self.world = world_lib.create(self, graphs=True)
    self.root = self.world.workspace.hf
    path = self.root / 'tap.litertlm'
    tap_fixtures.write_model(path)
    self.model = self.world.models.add_download(path)
    self.graphs = self.world.graphs

  def _create(
      self,
      points: list[str] | None = None,
      model=None,
      devices=(fake_launchers.LOCAL.id, fake_launchers.LOCAL.id),
      name: str = 'CPU vs CPU',
  ) -> dict[str, Any]:
    model = model or self.model
    config = self.world.config(*devices)
    config['name'] = name
    for run in config['runs']:
      run['artifact'] = model.artifact
    if points:
      config['tap_profile'] = session_store.TAP_PROFILE
      config['tap_points'] = {model.artifact: points}
    # As a record written before every Session captured the anchors:
    # these start without finding any.
    return self.world.older(self.world.sessions.create(config))

  def _progress(self, job: dict[str, Any]) -> list[str]:
    return [
        event['message']
        for event in self.world.events(job['id'])
        if event['type'] == 'progress' and 'runId' not in event
    ]

  def _completed(self, model=None) -> bool:
    model = model or self.model
    return self.graphs.state(model.artifact).status == model_graphs.COMPLETED

  def test_a_start_converts_the_graph_once(self):
    """The first start converts it; a later Session's finds it kept."""
    self.assertEqual(
        self.graphs.state(self.model.artifact).status, model_graphs.IDLE
    )
    record = self._create()
    job = self.world.start(record)
    name = self.model.name
    progress = self._progress(job)
    self.assertEqual(
        progress[0],
        f"{name}: Converting its graph with Model Explorer's converter",
    )
    self.assertRegex(progress[1], f'^{name}: its graph is converted, in ')
    self.assertEqual(self.world.converter.calls, 1)
    self.assertEqual(
        self.world.converter.timeouts,
        [model_graphs.DEFAULT_TIMEOUT_SECONDS],
    )
    state = self.graphs.state(self.model.artifact)
    self.assertEqual(state.status, model_graphs.COMPLETED)
    manifest = state.manifest
    self.assertEqual(manifest['identity'], model_graphs.identity())
    self.assertIn(graph_converter.identity(), manifest['identity'])
    self.assertEqual(manifest['model']['sha256'], self.model.sha256)
    self.assertEqual(
        [s['name'] for s in manifest['signatures']],
        list(tap_fixtures.SIGNATURES),
    )
    # The layers are worked out as the graph is read, not kept.
    self.assertNotIn('layer_stems', manifest)
    facts = self.graphs.facts(self.model)
    self.assertEqual(
        facts.layer_stems.stems,
        {name: 'layer' for name in tap_fixtures.SIGNATURES},
    )
    self.assertEqual(
        facts.layers, {name: 1 for name in tap_fixtures.SIGNATURES}
    )
    self.assertEqual(
        facts.layer_ids, {name: (0,) for name in tap_fixtures.SIGNATURES}
    )
    directory = self.world.workspace.graphs / self.model.sha256
    self.assertEqual(
        sorted(p.name for p in directory.iterdir()),
        ['manifest.json'] + [f'signature-{i}.json.gz' for i in range(4)],
    )
    # Nothing is left of the conversion: the converter's output, its
    # pending directory.
    self.assertEqual(
        [p.name for p in self.world.workspace.graphs.iterdir()],
        [self.model.sha256],
    )
    self.world.model_servers.close(record['id'])
    again = self.world.start(self._create())
    self.assertEqual(self._progress(again), [])
    self.assertEqual(self.world.converter.calls, 1)

  def test_a_graph_goes_with_its_model(self):
    """collect(): the graphs of the models kept stay, the rest go."""
    self.world.start(self._create())
    kept = self.world.workspace.graphs / self.model.sha256
    gone = self.world.workspace.graphs / ('f' * 64)
    gone.mkdir()
    self.graphs.collect({self.model.sha256})
    self.assertTrue(kept.is_dir())
    self.assertFalse(gone.exists())
    self.graphs.collect(set())
    self.assertEqual(list(self.world.workspace.graphs.iterdir()), [])
    self.assertEqual(
        self.graphs.state(self.model.artifact).status, model_graphs.IDLE
    )

  def test_the_conversion_ends_before_the_starts_runners_launch(self):
    """Every runner event of the start comes after the conversion."""
    launched = []
    self.world.converter.before = lambda stopped: launched.append(
        len(self.world.launchers.launched)
    )
    job = self.world.start(self._create())
    self.assertEqual(launched, [0])
    events = [
        event
        for event in self.world.events(job['id'])
        if event['type'] == 'progress'
    ]
    converted = next(
        i
        for i, event in enumerate(events)
        if event['message'].endswith(' s')
        and 'its graph is converted' in event['message']
    )
    runners = [i for i, event in enumerate(events) if 'runId' in event]
    self.assertNotEmpty(runners)
    self.assertGreater(min(runners), converted)
    self.assertEqual({events[i]['runId'] for i in runners}, {'ref', 'target'})
    self.assertLen(self.world.launchers.launched, 2)

  def test_the_graph_is_each_signatures_with_its_points(self):
    """graph(): the kept graph; each output names its point."""
    self.world.start(self._create())
    graph = self.graphs.graph(self.model, 'prefill_16')
    self.assertEqual(graph['id'], 'prefill_16')
    node = graph['nodes'][2]
    facts = {a['key']: a['value'] for a in node['outputsMetadata'][0]['attrs']}
    self.assertEqual(facts['point_id'], 'prefill_16:2:0')
    self.assertEqual(facts['layer'], '0')
    # What is kept is the converter's graph alone.
    directory = self.world.workspace.graphs / self.model.sha256
    with gzip.open(directory / 'signature-2.json.gz', 'rt') as file:
      kept = json.load(file)
    self.assertNotIn(
        'point_id',
        {a['key'] for a in kept['nodes'][2]['outputsMetadata'][0]['attrs']},
    )
    with self.assertRaisesRegex(errors.NotFound, 'no signature x'):
      self.graphs.graph(self.model, 'x')
    # A model whose graph is not converted has none to give.
    with self.assertRaisesRegex(errors.Conflict, 'not ready'):
      self.graphs.graph(self.world.model, 'decode')
    answer = self.graphs.architecture(self.model)
    self.assertIsNone(answer['architecture'])
    self.assertIn('No architecture template exists', answer['text'])
    self.assertEqual(answer['error'], '')
    self.assertEqual(answer['bindings'], {})

  def test_an_architecture_that_cannot_be_bound_says_why(self):
    """Binding raises: the answer says so, and is kept."""
    self.world.start(self._create())
    with mock.patch.object(
        architecture_lib, 'detect', side_effect=ValueError('Bad rule.')
    ) as detect:
      answer = self.graphs.architecture(self.model)
      self.assertEqual(
          answer['error'],
          'The architecture could not be bound to the graph: Bad rule.',
      )
      self.assertIsNone(answer['architecture'])
      self.assertIs(self.graphs.architecture(self.model), answer)
      self.assertEqual(detect.call_count, 1)

  def _failed_start(
      self, record: dict[str, Any], reason: str
  ) -> dict[str, Any]:
    """Checks a start failed for its graph, as any failed start does."""
    job = self.world.wait(self.world.model_servers.initialize(record['id']))
    self.assertEqual(job['status'], 'failed')
    model = next(
        self.world.models.get(run['artifact']) for run in record['runs']
    )
    self.assertEqual(
        job['error'],
        f'The graph of {model.name} could not be converted, and the '
        f'Session cannot start without it: {reason}',
    )
    # No runner launched; the Session has ended with no turn, and the
    # listing says why.
    self.assertEqual(self.world.launchers.launched, [])
    self.assertIsNone(self.world.model_servers.state(record['id']))
    model_server = self.world.summary(record['id'])['model_server']
    self.assertEqual(
        (model_server['state'], model_server['error']),
        ('ended', job['error']),
    )
    self.assertIsNone(model_server['button'])
    return job

  @parameterized.named_parameters(
      ('failed', 'The converter failed: Failed to build model from buffer.'),
      (
          'timed_out',
          'The converter did not finish in 600 s, and was stopped.',
      ),
      (
          'killed',
          (
              'The converter was killed by signal 9 (SIGKILL) after using '
              "60.1 GB of this computer's 64.0 GB (out of memory?): no output"
          ),
      ),
      ('crashed', 'The converter was killed by signal 11 (SIGSEGV): x'),
      (
          'unavailable',
          (
              f'{graph_converter.UNAVAILABLE} (ai-edge-model-explorer-adapter '
              '0.2.0 on win32 AMD64: No module named '
              "'ai_edge_model_explorer_adapter')"
          ),
      ),
  )
  def test_a_failed_conversion_fails_the_start(self, reason):
    """The Session cannot start; its graph says why, and is kept."""
    self.world.converter.error = RuntimeError(reason)
    self._failed_start(self._create(), reason)
    state = self.graphs.state(self.model.artifact)
    self.assertEqual(
        (state.status, state.error, state.identity),
        (model_graphs.FAILED, reason, model_graphs.identity()),
    )
    # Kept without a manifest: the reason, the identity, the log's tail.
    directory = self.world.workspace.graphs / self.model.sha256
    self.assertEqual(
        sorted(p.name for p in directory.iterdir()),
        ['converter.log', 'failure.json'],
    )
    kept = json_file.read_json(directory / 'failure.json')
    self.assertEqual(kept['error'], reason)
    self.assertEqual(kept['identity'], model_graphs.identity())
    self.assertIn(reason, (directory / 'converter.log').read_text())

  def test_a_model_that_cannot_be_read_fails_the_start(self):
    """The world's own model is no .litertlm: its start fails so."""
    with self.assertRaises(ValueError) as raised:
      self.world.scans.scan(self.world.model)
    self._failed_start(self.world.create(), errors.message(raised.exception))
    self.assertEqual(self.world.converter.calls, 0)
    self.assertEqual(
        self.graphs.state(self.world.model.artifact).status,
        model_graphs.FAILED,
    )

  def test_a_start_its_graph_failed_is_not_tried_again(self):
    """Its Session has ended; another Session's start converts again."""
    self.world.converter.error = RuntimeError('Out of memory.')
    record = self._create()
    job = self._failed_start(record, 'Out of memory.')
    self.assertEqual(self.world.converter.calls, 1)
    # Looking at it converts nothing.
    self.graphs.state(self.model.artifact)
    self.assertEqual(self.world.converter.calls, 1)
    # Its start is refused, as of any Session that ended, and converts
    # nothing.
    with self.assertRaises(errors.Conflict) as refused:
      self.world.model_servers.initialize(record['id'])
    self.assertEqual(str(refused.exception), model_server_lib.ENDED)
    self.assertEqual(
        self.world.sessions.get(record['id'])['notice'], job['error']
    )
    self.assertEqual(self.world.converter.calls, 1)
    # Another Session of the model, once memory is free: converted, and
    # it starts.
    self.world.converter.error = None
    self.world.start(self._create(name='Another'))
    self.assertEqual(self.world.converter.calls, 2)
    self.assertTrue(self._completed())

  def test_a_failure_kept_is_tried_by_the_next_server_runs_start(self):
    """A later server run reads the failure, and a start tries once more."""
    self.world.converter.error = RuntimeError('Out of memory.')
    self._failed_start(self._create(), 'Out of memory.')
    later = model_graphs.ModelGraphs(
        self.world.workspace,
        self.world.models,
        self.world.scans,
        self.world.jobs,
        convert=self.world.converter,
    )
    self.addCleanup(later.close)
    self.assertEqual(later.state(self.model.artifact).error, 'Out of memory.')
    record = self._create()
    job_id = self.world.jobs.create(record['id'], 'initialize')['id']
    with self.assertRaisesRegex(RuntimeError, 'Out of memory.'):
      later.prepare_start(record, job_id, lambda: False)
    self.assertEqual(self.world.converter.calls, 2)
    # A converter of another identity converts anew.
    self.world.converter.error = None
    with mock.patch.object(
        graph_converter, 'ADAPTER_VERSION', 'another version'
    ):
      self.assertEqual(
          later.state(self.model.artifact).status, model_graphs.IDLE
      )
      self.assertTrue(later.prepare_start(record, job_id, lambda: False))
      self.assertEqual(self.world.converter.calls, 3)
      self.assertEqual(
          later.state(self.model.artifact).status,
          model_graphs.COMPLETED,
      )

  def test_a_start_that_waited_for_a_failure_fails_with_it(self):
    """It does not convert again what failed while it waited."""
    converting, release = threading.Event(), threading.Event()
    blocked = _blocked(converting, release)

    def before(stopped):
      blocked(stopped)
      raise RuntimeError('The converter crashed.')

    self.world.converter.before = before
    first = self._create(name='First')
    first_job = self.world.model_servers.initialize(first['id'])
    self.assertTrue(converting.wait(fake_launchers.TIMEOUT))
    second = self._create(
        devices=(fake_launchers.PHONE.id, fake_launchers.TABLET.id),
        name='Second',
    )
    waiting = threading.Event()
    emit = self.world.jobs.emit

    def emitted(job_id, kind, **fields):
      # Only the second start waits.
      if 'waiting for' in fields.get('message', ''):
        waiting.set()
      emit(job_id, kind, **fields)

    with mock.patch.object(self.world.jobs, 'emit', emitted):
      second_job = self.world.model_servers.initialize(second['id'])
      self.assertTrue(waiting.wait(fake_launchers.TIMEOUT))
    release.set()
    self.assertEqual(self.world.wait(first_job)['status'], 'failed')
    second_job = self.world.wait(second_job)
    self.assertEqual(second_job['status'], 'failed')
    self.assertEqual(
        second_job['error'],
        f'The graph of {self.model.name} could not be converted, and the '
        'Session cannot start without it: The converter crashed. (it '
        'failed while this start waited for it)',
    )
    self.assertEqual(self.world.converter.calls, 1)
    self.assertEqual(self.world.launchers.launched, [])

  def test_turning_off_a_start_stops_its_conversion(self):
    """The start is turned off while converting: nothing is kept."""
    converting, release = threading.Event(), threading.Event()
    self.world.converter.before = _blocked(converting, release)
    record = self._create()
    job = self.world.model_servers.initialize(record['id'])
    self.assertTrue(converting.wait(fake_launchers.TIMEOUT))
    self.world.model_servers.close(record['id'])
    job = self.world.wait(job)
    self.assertEqual(job['status'], 'failed')
    self.assertEqual(
        self.graphs.state(self.model.artifact).status, model_graphs.IDLE
    )
    self.assertEqual(list(self.world.workspace.graphs.iterdir()), [])
    self.assertEqual(self.world.launchers.launched, [])

  def test_one_at_a_time_the_waiting_start_says_for_what(self):
    """A second start waits for the first's, and says whose it is."""
    other_path = self.root / 'other.litertlm'
    tap_fixtures.write_model(other_path, between=b'other')
    other = self.world.models.add_download(other_path)
    converting, release = threading.Event(), threading.Event()
    self.world.converter.before = _blocked(converting, release)
    first = self._create(name='First')
    first_job = self.world.model_servers.initialize(first['id'])
    self.assertTrue(converting.wait(fake_launchers.TIMEOUT))
    second = self._create(
        model=other,
        devices=(fake_launchers.PHONE.id, fake_launchers.TABLET.id),
        name='Second',
    )
    second_job = self.world.model_servers.initialize(second['id'])
    holder = f'the graph of {self.model.name}, for First'
    world_lib.eventually(
        lambda: self.graphs.state(other.artifact).waiting == holder
    )
    self.assertEqual(
        self.graphs.state(other.artifact).status, model_graphs.CONVERTING
    )
    release.set()
    self.world.wait(first_job)
    second_job = self.world.wait(second_job)
    self.assertEqual(second_job['status'], 'completed')
    self.assertEqual(
        self._progress(second_job)[0],
        f'{other.name}: waiting for {holder} to be converted',
    )
    self.assertTrue(self._completed(other))
    self.assertEqual(self.world.converter.calls, 2)

  def test_a_start_that_waits_can_be_turned_off(self):
    """Turned off while waiting: it never converts, and leaves no state."""
    other_path = self.root / 'other.litertlm'
    tap_fixtures.write_model(other_path, between=b'other')
    other = self.world.models.add_download(other_path)
    converting, release = threading.Event(), threading.Event()
    self.world.converter.before = _blocked(converting, release)
    first_job = self.world.model_servers.initialize(self._create()['id'])
    self.assertTrue(converting.wait(fake_launchers.TIMEOUT))
    second = self._create(
        model=other,
        devices=(fake_launchers.PHONE.id, fake_launchers.TABLET.id),
    )
    second_job = self.world.model_servers.initialize(second['id'])
    world_lib.eventually(
        lambda: bool(self.graphs.state(other.artifact).waiting)
    )
    self.world.model_servers.close(second['id'])
    self.assertEqual(self.world.wait(second_job)['status'], 'failed')
    self.assertEqual(
        self.graphs.state(other.artifact).status, model_graphs.IDLE
    )
    release.set()
    self.assertEqual(self.world.wait(first_job)['status'], 'completed')
    self.assertEqual(self.world.converter.calls, 1)

  def test_close_stops_the_conversion_and_waits_for_it(self):
    """close() returns once the conversion under way has ended."""
    converting, release = threading.Event(), threading.Event()
    ended = threading.Event()
    blocked = _blocked(converting, release)

    def before(stopped):
      try:
        blocked(stopped)
      finally:
        ended.set()

    self.world.converter.before = before
    job = self.world.model_servers.initialize(self._create()['id'])
    self.assertTrue(converting.wait(fake_launchers.TIMEOUT))
    self.graphs.close()
    self.assertTrue(ended.is_set())
    self.assertEqual(self.world.wait(job)['status'], 'failed')
    self.assertEqual(list(self.world.workspace.graphs.iterdir()), [])

  def test_a_graph_another_identity_wrote_is_converted_again(self):
    """The manifest's identity names the converter, slimming, layout."""
    self.world.start(self._create())
    directory = self.world.workspace.graphs / self.model.sha256
    manifest = json_file.read_json(directory / 'manifest.json')
    manifest['identity'] = 'another converter'
    json_file.write_json(directory / 'manifest.json', manifest)
    self.assertEqual(
        self.graphs.state(self.model.artifact).status, model_graphs.IDLE
    )
    self.world.start(self._create())
    self.assertTrue(self._completed())
    self.assertEqual(self.world.converter.calls, 2)
    with gzip.open(directory / 'signature-0.json.gz', 'rt') as file:
      self.assertEqual(json.load(file)['id'], 'decode')

  def test_the_analyses_read_the_layers_and_never_convert(self):
    """No graph: a reason, and no conversion; once kept, the layers."""
    record = self._create(_POINTS)
    observed = self.world.prepared.observed(record)
    self.assertEqual(observed.reason['code'], tensor_analysis.MODEL_UNREADABLE)
    self.assertEqual(
        observed.reason['text'],
        "The layers of the ref side come from its model's graph, which "
        'is not converted: it is converted when a Session of the model '
        'is initialized.',
    )
    self.assertEqual(self.world.converter.calls, 0)
    self.assertEqual(
        self.graphs.state(self.model.artifact).status, model_graphs.IDLE
    )
    self.world.start(record)
    observed = self.world.prepared.observed(record)
    self.assertIsNone(observed.reason)
    self.assertEqual(
        [(p.id, p.layer) for p in observed.sides['ref'].points],
        [('decode:0:0', 0), ('prefill_16:2:0', 0)],
    )
    self.assertEqual(
        self.graphs.layers(self.model).stems,
        {name: 'layer' for name in tap_fixtures.SIGNATURES},
    )

  def test_a_point_named_by_no_layer_is_where_the_graph_groups_it(self):
    """The analyses and the graph place `StatefulPartitionedCall:1` alike.

    Layer 0's second output of op 1 is named by no layer; the converter
    groups op 1 in layer 0, and so its point is in layer 0, in the
    analyses' points and in the graph served.
    """
    path = self.root / 'unnamed.litertlm'
    tap_fixtures.write_model(
        path, text=tap_fixtures.layered_model(2, unnamed=True)
    )
    model = self.world.models.add_download(path)
    record = self._create(['decode:1:1', 'decode:2:0'], model=model)
    self.world.start(record)
    layers = self.graphs.layers(model)
    self.assertEqual(layers.groups, {'decode': {1: 0}})
    self.assertEqual(self.graphs.facts(model).layer_groups, layers.groups)
    observed = self.world.prepared.observed(record)
    self.assertIsNone(observed.reason)
    self.assertEqual(
        [(p.id, p.layer) for p in observed.sides['ref'].points],
        [('decode:1:1', 0), ('decode:2:0', 0)],
    )
    graph = self.graphs.graph(model, 'decode')
    node = next(n for n in graph['nodes'] if n['id'] == '1')
    self.assertEqual(node['namespace'].split('/')[2], 'layer_0')
    self.assertEqual(
        [
            (attrs['point_id'], attrs['layer'])
            for attrs in map(model_graph.attrs, node['outputsMetadata'])
        ],
        [('decode:1:0', '0'), ('decode:1:1', '0')],
    )
    # A graph not kept has no groups either.
    self.assertIsNone(self.graphs.layers(self.world.model).groups)


class PyTorchGraphsTest(absltest.TestCase):
  """A PyTorch model's graph: the one a turn of it exported, kept."""

  def setUp(self):
    """A world that keeps graphs, and Gemma 4 E2B's directory."""
    super().setUp()
    self.world = world_lib.create(self, graphs=True)
    self.root = self.world.workspace.hf
    directory = self.root / 'gemma-4-E2B-it'
    directory.mkdir()
    (gemma4,) = [a for a in catalog.architectures() if a.id == 'gemma4']
    pinned = gemma4.entry('gemma-4-E2B-it')['pinned']['file']
    shutil.copyfile(gemma4.directory / pinned, directory / 'config.json')
    (directory / 'model.safetensors').write_bytes(b'weights')
    self.model = self.world.models.add_download_directory(directory)
    self.graphs = self.world.graphs

  def _export(self, name: str = 'turn-1', **document) -> pathlib.Path:
    path = self.root / name / pytorch_graph_fixtures.FILE
    path.parent.mkdir()
    path.write_bytes(
        pytorch_graph_fixtures.gzipped(
            pytorch_graph_fixtures.document(**document)
        )
    )
    return path

  def test_no_graph_until_a_turn_exports_one_and_how_to_get_it(self):
    """Unavailable, saying which run exports it; never converted."""
    state = self.graphs.state(self.model.artifact)
    self.assertEqual(state.status, model_graphs.UNAVAILABLE)
    self.assertEqual(state.text, model_graphs.NO_PYTORCH_GRAPH)
    self.assertIn('on the CPU that observes the anchors', state.text)
    layers = self.graphs.layers(self.model)
    self.assertIsNone(layers.stems)
    self.assertIn('observes the anchors', layers.reason)
    with self.assertRaisesRegex(errors.Conflict, 'not kept yet'):
      self.graphs.graph(self.model, 'decode')

  def test_a_turns_graph_is_kept_folded_and_bound(self):
    """Kept once; served folded, its anchors bound and annotated."""
    source = {'session': 's1', 'turn': 1, 'side': 'ref'}
    kept = self.graphs.exported(self.model.artifact, self._export(), source)

    self.assertEqual(kept, ['prefill', 'decode'])
    state = self.graphs.state(self.model.artifact)
    self.assertEqual(state.status, model_graphs.COMPLETED)
    self.assertEqual(
        state.manifest['identity'], model_graphs.pytorch_identity()
    )
    self.assertEqual(
        [e['name'] for e in state.manifest['signatures']],
        ['prefill', 'decode'],
    )
    self.assertEqual(state.manifest['signatures'][1]['source'], source)
    facts = self.graphs.facts(self.model)
    self.assertEqual(
        facts.layer_stems.stems, {'prefill': 'layers/', 'decode': 'layers/'}
    )
    self.assertEqual(facts.layer_ids['decode'], tuple(range(35)))
    self.assertEqual(facts.layers['prefill'], 35)
    answer = self.graphs.architecture(self.model)
    self.assertEqual(answer['architecture']['name'], 'Gemma 4')
    self.assertEqual(answer['architecture']['skeleton'], 'gemma-4-E2B-it')
    self.assertEqual(answer['error'], '')
    layer = answer['bindings']['decode'][3]
    self.assertEqual(layer['scope'], 'layers/3')
    self.assertEqual(
        layer['anchors']['output']['point'],
        'decode:model.language_model.layers.3:out',
    )
    self.assertEqual({b['by'] for b in layer['blocks']}, {'anchors'})
    graph = self.graphs.graph(self.model, 'decode')
    by_id = {n['id']: n for n in graph['nodes']}
    output = by_id[layer['anchors']['output']['node']]
    self.assertEqual(
        model_graph.attrs(output['outputsMetadata'][0])['point_id'],
        'decode:model.language_model.layers.3:out',
    )
    norms = [n for n in graph['nodes'] if n['label'] == 'Gemma4RMSNorm']
    self.assertLen(norms, 35 * 8 + 1)
    # A semantic operator's module carries the point a Session that
    # pairs it with a LiteRT-LM point observes; every other folded
    # module too, which a pair made by hand may name (the final norm,
    # in no layer); an ATen operator's output none.
    first = model_graph.attrs(norms[0]['outputsMetadata'][0])
    self.assertEqual(
        first['point_id'],
        'decode:model.language_model.layers.0.input_layernorm:out',
    )
    self.assertEqual(first['observable'], 'true')
    self.assertEqual(first['layer'], '0')
    final = model_graph.attrs(norms[-1]['outputsMetadata'][0])
    self.assertEqual(final['point_id'], 'decode:model.language_model.norm:out')
    self.assertNotIn('layer', final)
    aten = []
    for node in graph['nodes']:
      if 'module_class' not in model_graph.attrs(node):
        for entry in node.get('outputsMetadata', []):
          if model_graph.attrs(entry).get('observable') == 'false':
            aten.append(model_graph.attrs(entry))
    self.assertNotEmpty(aten)
    self.assertEqual(aten[0]['reason'], model_graphs.NOT_AN_ANCHOR)
    # What a folded module reads is its point too: o_proj's input.
    o_proj = next(
        n
        for n in graph['nodes']
        if model_graph.attrs(n).get('module')
        == 'model.language_model.layers.3.self_attn.o_proj'
    )
    source = o_proj['incomingEdges'][0]
    read = next(
        model_graph.attrs(entry)
        for entry in by_id[source['sourceNodeId']]['outputsMetadata']
        if entry['id'] == source['sourceNodeOutputId']
    )
    self.assertEqual(
        read['point_id'],
        'decode:model.language_model.layers.3.self_attn.o_proj:in',
    )
    self.assertEqual(read['layer'], '3')
    self.assertEqual(
        self.graphs.tensor(
            self.model,
            'decode',
            source['sourceNodeId'],
            source['sourceNodeOutputId'],
        ),
        read,
    )
    with self.assertRaises(errors.NotFound):
      self.graphs.tensor(self.model, 'decode', 'no-such-node', '0')
    # A later turn's graph is not kept again.
    again = self._export('turn-2')
    self.assertEqual(
        self.graphs.exported(self.model.artifact, again, source), []
    )
    self.assertEqual(
        self.graphs.state(self.model.artifact).manifest['created_at'],
        state.manifest['created_at'],
    )

  def test_a_later_turn_adds_a_signature_the_first_did_not_run(self):
    """The signatures it lacks; those it has stay as they were."""
    first = {'session': 's1', 'turn': 1, 'side': 'ref'}
    later = {'session': 's1', 'turn': 2, 'side': 'ref'}
    self.assertEqual(
        self.graphs.exported(
            self.model.artifact,
            self._export(signatures=('decode',)),
            first,
        ),
        ['decode'],
    )
    self.assertEqual(
        self.graphs.facts(self.model).layer_stems.stems,
        {'decode': 'layers/'},
    )
    self.assertEqual(
        self.graphs.exported(
            self.model.artifact, self._export('turn-2'), later
        ),
        ['prefill'],
    )
    manifest = self.graphs.state(self.model.artifact).manifest
    self.assertEqual(
        [(e['name'], e['source']) for e in manifest['signatures']],
        [('decode', first), ('prefill', later)],
    )
    self.assertEqual(
        set(self.graphs.architecture(self.model)['bindings']),
        {'decode', 'prefill'},
    )

  def test_a_file_that_cannot_be_kept_says_why(self):
    """Nothing is kept; where the graph is says why."""
    path = self.root / pytorch_graph_fixtures.FILE
    path.write_bytes(gzip.compress(b'{"format": 2}'))
    source = {'session': 's1', 'turn': 1, 'side': 'ref'}

    self.assertEqual(
        self.graphs.exported(self.model.artifact, path, source), []
    )
    state = self.graphs.state(self.model.artifact)
    self.assertEqual(state.status, model_graphs.UNAVAILABLE)
    self.assertIn(
        'was not kept: The execution graph is not of format 1', state.text
    )
    with self.assertRaisesRegex(ValueError, 'not a PyTorch model'):
      self.graphs.keep_exported(self.world.model, path, source)


if __name__ == '__main__':
  test_main.main()
