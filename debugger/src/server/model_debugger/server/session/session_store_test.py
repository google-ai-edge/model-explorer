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

"""Tests of Session records."""

import copy
import pathlib
from typing import Any
from unittest import mock

from absl.testing import parameterized
from model_debugger.server.session import model_store
from model_debugger.server.session import session_store
from model_debugger.server.session import tap_scans
from model_debugger.server.session import workspace
from model_debugger.server.tap import tap_model
from model_debugger.server.testing import tap_fixtures
from model_debugger.server.testing import test_main
from model_debugger.server.util import errors
from model_debugger.server.util import json_file

# A LiteRT-LM commit, as a run chooses its kept runner build.
_COMMIT = '61dcfc77225019e067580dceb3ea268483d1938c'
# What a side runs with when its chat sets nothing: greedy, every sampling
# value explicit.
_GREEDY = {
    'samplerType': 'greedy',
    'topK': 1,
    'topP': 1.0,
    'temperature': 1.0,
    'seed': 0,
}


def _run(side, model, **fields):
  return {
      'id': side,
      'runtime': 'LiteRT-LM',
      'artifact': model.artifact,
      'source': 'upload',
      'device': 'local',
      'backend': 'CPU',
      'contextLength': '2048',
      **fields,
  }


def _payload(registered, **fields):
  return {
      'name': ' First ',
      'model': 'Gemma 4 E2B',
      'runs': [_run('target', registered), _run('ref', registered)],
      **fields,
  }


class SessionStoreTest(parameterized.TestCase):
  """Tests of Session records."""

  def _store(self) -> session_store.SessionStore:
    """A SessionStore of the workspace, reading its records anew."""
    models = model_store.ModelStore(self.opened)
    return session_store.SessionStore(
        self.opened, models, tap_scans.TapScans(models)
    )

  def setUp(self):
    """An open workspace, registered models, then a SessionStore."""
    super().setUp()
    tmp_path = pathlib.Path(self.create_tempdir().full_path)
    self.opened = workspace.Workspace(tmp_path / 'ws')
    self.addCleanup(self.opened.close)
    path = self.opened.hf / 'gemma.litertlm'
    path.write_bytes(b'model')
    self.model = model_store.ModelStore(self.opened).add_download(path)
    # Registered before the store reads the models.
    tapped = self.opened.hf / 'tap.litertlm'
    tap_fixtures.write_model(tapped)
    self.tapped = model_store.ModelStore(self.opened).add_download(tapped)
    self.store = self._store()

  def _started(self, payload: dict[str, Any]) -> dict[str, Any]:
    """Creates a record whose start was admitted, as model_servers does."""
    record = self.store.create(payload)
    self.store.set_flags(record['id'], started=True)
    return self.store.get(record['id'])

  def test_create_and_reload(self):
    """Create and reload."""
    record = self.store.create(_payload(self.model))
    self.assertEqual(
        record,
        {
            'id': record['id'],
            'name': 'First',
            # Named by the server: the one model both sides load.
            'model': 'gemma.litertlm',
            'runs': [
                {
                    'id': 'ref',
                    'runtime': 'LiteRT-LM',
                    'artifact': self.model.artifact,
                    'source': 'upload',
                    'device': 'local',
                    'backend': 'CPU',
                    'contextLength': '2048',
                },
                {
                    'id': 'target',
                    'runtime': 'LiteRT-LM',
                    'artifact': self.model.artifact,
                    'source': 'upload',
                    'device': 'local',
                    'backend': 'CPU',
                    'contextLength': '2048',
                },
            ],
            # Each side captures the anchors, always, found as it is
            # prepared.
            'tap_anchors': {'ref': True, 'target': True},
            'tap_ops': [],
            'tap_profile': '',
            'tap_points': {},
            'created_at': record['created_at'],
            'execution_ended': False,
            'notice': '',
            # No chat until one is started.
            'chats': [],
        },
    )
    # Its start is admitted as it is created (model_servers.create()).
    self.store.set_flags(record['id'], started=True)
    record = self.store.get(record['id'])
    reopened = self._store()
    self.assertEqual(reopened.get(record['id']), record)
    self.assertEqual(reopened.list(), [record])

  def test_a_record_an_older_server_saved_without_starting_has_ended(self):
    """Read as one whose start failed, with why; once, as it loads."""
    record = self.store.create(_payload(self.model))
    reopened = self._store().get(record['id'])
    self.assertEqual(
        reopened,
        {
            **record,
            'started': True,
            'notice': session_store.NEVER_STARTED,
        },
    )
    written = json_file.read_json(self.opened.records / f'{record["id"]}.json')
    self.assertEqual(written, reopened)

  def test_create_keeps_options_set_and_drops_old_keys(self):
    """Create keeps options set and the runner build; drops old keys."""
    # What the UI's editor posts of the record it starts from: its runs
    # with every option key, its runner build, the old runner's keys, and
    # the tap settings.
    ref = _run(
        'ref',
        self.model,
        cpuThreads='4',
        enableYnnpack='on',
        useRingbuffersLocalAttention='',
        loraRank='',
        runnerBuild=_COMMIT,
        repository='',
        revision='',
        sourceUrl='',
        precision='',
    )
    record = self.store.create(
        _payload(
            self.model,
            runs=[ref, _run('target', self.model)],
            tap_profile='',
            tap_points={},
        )
    )
    self.assertEqual(
        record['runs'][0],
        {
            'id': 'ref',
            'runtime': 'LiteRT-LM',
            'artifact': self.model.artifact,
            'source': 'upload',
            'device': 'local',
            'backend': 'CPU',
            'contextLength': '2048',
            'runnerBuild': _COMMIT,
            'cpuThreads': '4',
            'enableYnnpack': 'on',
        },
    )
    # No build, or none given, keeps none: a start asks for one where its
    # device needs one.
    self.assertNotIn('runnerBuild', record['runs'][1])
    target = _run('target', self.model, runnerBuild='')
    record = self.store.create(_payload(self.model, runs=[ref, target]))
    self.assertNotIn('runnerBuild', record['runs'][1])
    # The development build keeps its own, and so do runs posted again as
    # a record holds them.
    target = _run('target', self.model, runnerBuild='dev')
    record = self.store.create(_payload(self.model, runs=[ref, target]))
    self.assertEqual(record['runs'][1]['runnerBuild'], 'dev')
    again = self.store.create(
        _payload(self.model, runs=copy.deepcopy(record['runs']))
    )
    self.assertEqual(
        [run.get('runnerBuild') for run in again['runs']], [_COMMIT, 'dev']
    )

  @parameterized.parameters(
      ({'name': ''}, '1 to 160'),
      ({'name': 'x' * 161}, '1 to 160'),
      ({'name': 'a\tb'}, 'control characters'),
      ({'chats': []}, 'Unknown session settings: chats'),
      ({'generation': {}}, 'Unknown session settings: generation'),
      ({'runs': []}, 'Reference and Target runs'),
  )
  def test_create_refuses_config(self, change, message):
    """Create refuses config."""
    with self.assertRaisesRegex(ValueError, message):
      self.store.create(_payload(self.model, **change))

  @parameterized.parameters(
      ({'runtime': 'ONNX'}, 'Target: the runtime must be LiteRT-LM or'),
      ({'runtime': 'PyTorch'}, 'Target: gemma.litertlm is a LiteRT-LM'),
      ({'artifact': 'model-0'}, 'Target: select a model'),
      ({'device': ''}, 'Target: select a device'),
      ({'backend': ''}, 'Target: the backend must be CPU or GPU'),
      ({'contextLength': 'big'}, 'Target: The context length'),
      ({'enableYnnpack': 'yes'}, 'Target: enableYnnpack must be on, off'),
      ({'cpuThreads': 4}, 'Target cpuThreads is not valid text'),
      ({'tapPoints': ''}, 'Unknown Target settings: tapPoints'),
      ({'runnerBuild': 'build-1'}, 'Target: the runner version must be'),
      ({'runnerBuild': _COMMIT[:12]}, 'Target: the runner version must be'),
      ({'runnerBuild': 7}, 'Target runner version is not valid text'),
  )
  def test_create_refuses_run(self, fields, message):
    """Create refuses run."""
    with self.assertRaisesRegex(ValueError, message):
      self.store.create(
          _payload(
              self.model,
              runs=[
                  _run('ref', self.model),
                  _run('target', self.model, **fields),
              ],
          )
      )

  def test_a_model_deleted_while_a_record_is_checked_is_refused(self):
    """The check and the write hold the lock collect_models() takes."""
    config = self.store._config
    deleted = []

    def check_then_delete(payload):
      checked = config(payload)
      # As another tab's delete of the last Session using it would.
      deleted.extend(self.store.collect_models({self.model.artifact}))
      return checked

    self.store._config = check_then_delete
    with self.assertRaisesRegex(ValueError, 'deleted from the Server'):
      self.store.create(_payload(self.model))
    self.assertLen(deleted, 1)
    self.assertEqual(self.store.list(), [])

  def test_create_refuses_a_model_no_longer_there(self):
    """Create refuses a model whose file is not on the server."""
    self.model.path.unlink()
    with self.assertRaisesRegex(ValueError, 'not on the Server'):
      self.store.create(_payload(self.model))

  def test_the_points_a_session_observes_are_kept(self):
    """Kept with the record; without the profile, no points are."""
    points = {self.tapped.artifact: ['decode:0:0', 'prefill_16:2:0']}
    record = self.store.create(
        _payload(
            self.tapped,
            tap_profile=session_store.TAP_PROFILE,
            tap_points=points,
        )
    )
    self.assertEqual(record['tap_points'], points)
    self.assertEqual(session_store.tap_points(record), points)
    reopened = self._store()
    self.assertEqual(
        session_store.tap_points(reopened.get(record['id'])), points
    )
    plain = self.store.create(
        _payload(self.tapped, tap_profile='', tap_points=points)
    )
    self.assertEqual(plain['tap_points'], {})
    self.assertEqual(session_store.tap_points(plain), {})
    # A record written before Sessions observed tensors observes none.
    old = {key: value for key, value in plain.items() if 'tap' not in key}
    self.assertEqual(session_store.tap_points(old), {})

  @parameterized.named_parameters(
      ('unknown_profile', 'gemma4', None, 'Unknown tensor capture'),
      (
          'another_model',
          session_store.TAP_PROFILE,
          {'model-other': ['decode:0:0']},
          'LiteRT-LM models the Session loads',
      ),
      ('not_a_list', session_store.TAP_PROFILE, 'decode:0:0', 'Choose 1'),
      (
          'too_many',
          session_store.TAP_PROFILE,
          [f'decode:{i}:0' for i in range(tap_model.MAX_POINTS + 1)],
          f'Choose 1 to {tap_model.MAX_POINTS}',
      ),
      (
          'twice',
          session_store.TAP_PROFILE,
          ['decode:0:0', 'decode:0:0'],
          'chosen twice',
      ),
      (
          'not_in_the_model',
          session_store.TAP_PROFILE,
          ['decode:9:0'],
          'tap.litertlm: The model has no tensor output decode:9:0.',
      ),
      (
          'not_an_id',
          session_store.TAP_PROFILE,
          ['../x'],
          'no tensor output ../x',
      ),
      (
          'not_observable',
          session_store.TAP_PROFILE,
          ['decode:3:0'],
          'decode:3:0 cannot be observed: Its type, string',
      ),
  )
  def test_create_refuses_points(self, profile, points, message):
    """Points of every model it loads, as its scan lists them."""
    if isinstance(points, (list, str)):
      points = {self.tapped.artifact: points}
    with self.assertRaisesRegex(ValueError, message):
      self.store.create(
          _payload(
              self.tapped,
              tap_profile=profile,
              tap_points=points or {},
          )
      )

  def test_the_anchors_are_found_once_and_kept(self):
    """Captured by default; found later, beside the points chosen."""
    record = self.store.create(
        _payload(
            self.tapped,
            tap_profile=session_store.TAP_PROFILE,
            tap_points={self.tapped.artifact: ['decode:0:0']},
        )
    )
    self.assertTrue(session_store.captures_anchors(record))
    self.assertFalse(session_store.anchors_found(record))
    self.assertEqual(
        session_store.tap_points(record),
        {self.tapped.artifact: ['decode:0:0']},
    )

    found = self.store.set_anchor_points(
        record['id'],
        {self.tapped.artifact: ['prefill_16:2:0', 'decode:0:0']},
    )

    self.assertTrue(session_store.anchors_found(found))
    # The anchors, then the points chosen that are not among them.
    self.assertEqual(
        session_store.tap_points(found),
        {self.tapped.artifact: ['prefill_16:2:0', 'decode:0:0']},
    )
    self.assertEqual(
        session_store.chosen_points(found),
        {self.tapped.artifact: ['decode:0:0']},
    )
    self.store.set_flags(record['id'], started=True)
    self.assertEqual(
        self._store().get(record['id']), self.store.get(record['id'])
    )
    self.assertTrue(
        session_store.anchors_found(self._store().get(record['id']))
    )

  def test_the_operators_chosen_inside_the_layers_are_kept(self):
    """One list for both sides; its points found with the anchors."""
    record = self.store.create(
        _payload(self.tapped, tap_ops=['attn/q_proj', 'norm.ffn.in'])
    )
    self.assertEqual(
        session_store.tap_ops(record), ['attn/q_proj', 'norm.ffn.in']
    )
    self.assertEqual(session_store.op_points(record), {})

    found = self.store.set_anchor_points(
        record['id'],
        {self.tapped.artifact: ['decode:0:0']},
        {self.tapped.artifact: ['decode:1:0', 'decode:0:0']},
    )

    self.assertEqual(
        found['tap_op_points'],
        {self.tapped.artifact: ['decode:1:0', 'decode:0:0']},
    )
    # The anchors, then the operators' points, each once.
    self.assertEqual(
        session_store.tap_points(found),
        {self.tapped.artifact: ['decode:0:0', 'decode:1:0']},
    )

  @parameterized.parameters(
      ('attn/q_proj', 'Choose operators inside the layers by their ids.'),
      (['attn/q_proj', 'attn/q_proj'], 'is chosen twice'),
      (['attn/rope'], 'Unknown operators inside the layers: attn/rope.'),
  )
  def test_create_refuses_operators(self, ops, message):
    """Not a list, twice, or none a binding maps."""
    with self.assertRaisesRegex(ValueError, message):
      self.store.create(_payload(self.tapped, tap_ops=ops))

  def test_a_session_that_ran_keeps_the_anchors_it_ran_with(self):
    """set_anchor_points() changes nothing of a record that ran."""
    record = self.store.create(_payload(self.tapped))
    found = self.store.set_anchor_points(
        record['id'], {self.tapped.artifact: ['decode:0:0']}
    )
    self.store.set_flags(record['id'], execution_ended=True)

    kept = self.store.set_anchor_points(record['id'], {})

    self.assertEqual(
        session_store.tap_points(kept), session_store.tap_points(found)
    )

  def test_the_anchors_are_always_captured_and_none_chosen_beside(self):
    """No switch: off is ignored; beside them no point need be chosen."""
    off = self.store.create(_payload(self.tapped, tap_anchors=False))
    self.assertEqual(off['tap_anchors'], {'ref': True, 'target': True})
    self.assertTrue(session_store.captures_anchors(off))
    on = self.store.create(
        _payload(
            self.tapped,
            tap_profile=session_store.TAP_PROFILE,
            tap_points={self.tapped.artifact: []},
        )
    )
    self.assertEqual((on['tap_profile'], on['tap_points']), ('', {}))
    # A record written before the anchors were captures none.
    old = {k: v for k, v in on.items() if k != 'tap_anchors'}
    self.assertFalse(session_store.captures_anchors(old))

  def test_each_side_captures_its_anchors_by_itself(self):
    """Per LiteRT-LM side; an older record's one bool is both sides'."""
    both = [_run('target', self.model), _run('ref', self.tapped)]
    record = self.store.create(
        _payload(self.tapped, runs=both, tap_anchors={'ref': False})
    )
    self.assertEqual(record['tap_anchors'], {'ref': True, 'target': True})
    self.assertEqual(
        session_store.anchor_points({
            **record,
            'tap_anchors': {'ref': False, 'target': True},
            'tap_anchor_points': {
                self.model.artifact: ['a'],
                self.tapped.artifact: ['b'],
            },
        }),
        {self.model.artifact: ['a']},
    )
    # The model the listing names is both, Reference first.
    self.assertEqual(record['model'], 'tap.litertlm vs gemma.litertlm')
    # Records as written before keep what they stored.
    self.assertEqual(
        session_store.anchor_sides({**record, 'tap_anchors': True}),
        {'ref': True, 'target': True},
    )
    self.assertEqual(
        session_store.anchor_sides({**record, 'tap_anchors': False}),
        {'ref': False, 'target': False},
    )

  def test_a_pytorch_run_always_captures_its_anchors(self):
    """Its `observe` is 'anchors', whatever is posted."""
    directory = self.opened.hf / 'pytorch'
    directory.mkdir()
    (directory / 'config.json').write_text('{}')
    pytorch = model_store.ModelStore(self.opened).add_download_directory(
        directory
    )
    store = self._store()
    runs = [
        _run(
            'ref',
            pytorch,
            runtime='PyTorch',
            runnerBuild='pytorch',
            observe='',
        ),
        _run('target', self.tapped),
    ]

    record = store.create(_payload(self.tapped, runs=runs))

    self.assertEqual(record['runs'][0]['observe'], 'anchors')
    self.assertEqual(record['tap_anchors'], {'target': True})

  def test_a_hugging_face_run_holds_where_its_model_came_from(self):
    """Taken from the model store, whatever the editor posts."""
    source = {
        'repo': 'org/repo',
        'file': 'gemma.litertlm',
        'revision': 'main',
        'commit': 'c' * 40,
    }
    models = model_store.ModelStore(self.opened)
    models.add_download(self.model.path, source)
    store = session_store.SessionStore(
        self.opened, models, tap_scans.TapScans(models)
    )
    runs = [
        _run(
            'ref',
            self.model,
            source='huggingface',
            huggingface={'repo': 'other/repo'},
        ),
        _run('target', self.tapped, source='huggingface'),
    ]

    with self.assertRaisesRegex(ValueError, 'did not come from Hugging'):
      store.create(_payload(self.model, runs=runs))
    runs[1]['source'] = 'upload'
    record = store.create(_payload(self.model, runs=runs))

    self.assertEqual(record['runs'][0]['huggingface'], source)
    self.assertNotIn('huggingface', record['runs'][1])

  def test_a_side_may_choose_no_outputs(self):
    """Points of one side's model alone; none at all is no profile."""
    both = [_run('target', self.model), _run('ref', self.tapped)]
    record = self.store.create(
        _payload(
            self.tapped,
            runs=both,
            tap_profile=session_store.TAP_PROFILE,
            tap_points={self.tapped.artifact: ['decode:0:0']},
        )
    )
    self.assertEqual(
        record['tap_points'], {self.tapped.artifact: ['decode:0:0']}
    )
    none = self.store.create(
        _payload(
            self.tapped,
            tap_profile=session_store.TAP_PROFILE,
            tap_points={},
        )
    )
    self.assertEqual((none['tap_profile'], none['tap_points']), ('', {}))

  def test_an_older_editors_model_is_dropped(self):
    """The server names the models; what an editor named is dropped."""
    record = self.store.create(_payload(self.tapped, model='x' * 300))
    self.assertEqual(record['model'], 'tap.litertlm')

  def test_points_of_a_model_that_cannot_be_observed_are_refused(self):
    """A model without a text model to scan says so."""
    with self.assertRaisesRegex(ValueError, 'not a .litertlm file'):
      self.store.create(
          _payload(
              self.model,
              tap_profile=session_store.TAP_PROFILE,
              tap_points={self.model.artifact: ['decode:0:0']},
          )
      )

  def test_rename_and_delete_for_good(self):
    """A record deleted is gone, from the store and from the disk."""
    record = self.store.create(_payload(self.model))
    kept = self._started(_payload(self.model, name='Kept'))
    self.assertEqual(
        self.store.rename(record['id'], ' Renamed ')['name'], 'Renamed'
    )
    self.assertEqual(self.store.delete(record['id'])['name'], 'Renamed')
    self.assertEqual(self.store.list(), [kept])
    self.assertFalse((self.opened.records / f'{record["id"]}.json').exists())
    for call in (
        self.store.get,
        self.store.delete,
        lambda i: self.store.set_flags(i, notice='Over.'),
    ):
      with self.assertRaises(errors.NotFound):
        call(record['id'])
    self.assertFalse(hasattr(self.store, 'restore'))
    self.assertEqual(self._store().list(), [kept])

  def test_a_record_marked_deleted_is_deleted_for_good_as_it_loads(self):
    """An older server's deleted record goes, with its capture.

    A record it kept, `deleted` false, is read without the flag.
    """
    gone = self.store.create(_payload(self.model))
    kept = self._started(_payload(self.model, name='Kept'))
    for record, deleted in ((gone, True), (kept, False)):
      path = self.opened.records / f'{record["id"]}.json'
      json_file.write_json(path, {**record, 'deleted': deleted})
      (self.opened.captures / record['id']).mkdir()
      (self.opened.captures / record['id'] / 'index.json').write_text('{}')
    store = self._store()
    self.assertEqual(store.list(), [kept])
    self.assertEqual(
        sorted(p.name for p in self.opened.records.iterdir()),
        [f'{kept["id"]}.json'],
    )
    self.assertEqual(
        [p.name for p in self.opened.captures.iterdir()], [kept['id']]
    )
    with self.assertRaises(errors.NotFound):
      store.get(gone['id'])

  def test_a_record_an_older_server_exported_is_kept_without_its_flag(self):
    """Its `deleted` flag, false as it was exported, is not kept."""
    record = self.store.create(_payload(self.model))
    self.store.add_chat(record['id'], {'ref': {}, 'target': {}})
    self.store.set_chat_flags(record['id'], 1, has_capture=True)
    record = self.store.get(record['id'])
    adopted = self.store.adopt(
        {**record, 'deleted': False},
        self.store.free_id(None),
        {'session_id': record['id']},
    )
    self.assertNotIn('deleted', adopted)
    self.assertNotIn('deleted', self.store.get(adopted['id']))

  def test_files_no_record_names_go_as_the_store_loads(self):
    """What a crash left of a delete or an import goes; the rest stays."""
    record = self._started(_payload(self.model))
    orphan = '00000000-0000-4000-8000-000000000000'
    for record_id in (record['id'], orphan):
      (self.opened.captures / record_id / 'turns').mkdir(parents=True)
    # A capture renamed away as it was deleted.
    (self.opened.captures / '.pending-abc').mkdir()
    store = self._store()
    self.assertEqual(store.list(), [record])
    self.assertEqual(
        [p.name for p in self.opened.captures.iterdir()], [record['id']]
    )
    self.assertTrue((self.opened.captures / record['id'] / 'turns').is_dir())

  def test_a_chats_settings_are_read_as_saved(self):
    """Whatever a chat saved is read back as it is, top_k included."""
    record = self.store.create(_payload(self.model))
    self.store.add_chat(record['id'], {'ref': _GREEDY, 'target': _GREEDY})
    path = self.opened.records / f'{record["id"]}.json'
    saved = {
        'ref': {'samplerType': 'topP', 'topK': 2**31 - 1, 'topP': 0.9},
        'target': _GREEDY,
    }
    stored = json_file.read_json(path)
    stored['chats'][0]['generation'] = saved
    json_file.write_json(path, stored)
    store = self._store()
    self.assertEqual(store.get(record['id'])['chats'][0]['generation'], saved)

  def test_set_flags_keeps_what_changed_meanwhile(self):
    """Flags set while a record was renamed keep both changes."""
    record = self.store.create(_payload(self.model))
    self.store.rename(record['id'], 'Renamed')
    self.store.set_flags(record['id'], started=True, notice='Done.')
    stored = self.store.get(record['id'])
    self.assertEqual(stored['name'], 'Renamed')
    self.assertTrue(stored['started'] and stored['notice'] == 'Done.')
    with self.assertRaises(errors.NotFound):
      self.store.set_flags('missing', notice='Over.')

  def test_chats_are_numbered_and_kept(self):
    """Each chat added is the next number; its flags change it alone."""
    record = self.store.create(_payload(self.model))
    first = self.store.add_chat(
        record['id'],
        self.store.check_generation(
            record['id'], {'ref': {'samplerType': 'greedy'}, 'target': {}}
        ),
    )
    second = self.store.add_chat(record['id'], {'ref': {}, 'target': {}})
    self.assertEqual(
        first,
        {
            'n': 1,
            'created_at': first['created_at'],
            'generation': {'ref': _GREEDY, 'target': _GREEDY},
            'has_capture': False,
            'over': False,
            'notice': '',
        },
    )
    self.assertEqual(second['n'], 2)
    self.store.set_chat_flags(
        record['id'], 1, has_capture=True, over=True, notice='Over.'
    )
    chats = self._store().get(record['id'])['chats']
    self.assertEqual(
        [(c['n'], c['has_capture'], c['over'], c['notice']) for c in chats],
        [(1, True, True, 'Over.'), (2, False, False, '')],
    )
    self.assertTrue(session_store.has_turn(self.store.get(record['id'])))
    with self.assertRaises(errors.NotFound):
      self.store.set_chat_flags(record['id'], 3, over=True)
    with self.assertRaises(ValueError):
      self.store.set_chat_flags(record['id'], 1, started=True)

  def test_a_chats_settings_are_checked_against_the_runs(self):
    """No reply longer than the side's context; speculation refused."""
    record = self.store.create(_payload(self.model))
    with self.assertRaisesRegex(ValueError, 'at most the context length'):
      self.store.check_generation(
          record['id'], {'ref': {'maxOutputTokens': 2049}, 'target': {}}
      )
    with self.assertRaisesRegex(ValueError, 'Speculative decoding'):
      self.store.check_generation(
          record['id'],
          {'ref': {}, 'target': {'speculativeDecoding': 'on'}},
      )
    with self.assertRaises(errors.NotFound):
      self.store.check_generation('missing', {'ref': {}, 'target': {}})

  def test_a_record_of_an_older_server_is_read_as_one_ended_chat(self):
    """Its one chat is Chat 1, its capture moved under chat-1/; once."""
    record = self.store.create(_payload(self.model))
    path = self.opened.records / f'{record["id"]}.json'
    older = {key: value for key, value in record.items() if key != 'chats'}
    older.update(
        generation={'ref': _GREEDY, 'target': {'samplerType': 'topK'}},
        has_capture=True,
        started=True,
        execution_ended=True,
        chat_over=True,
        chat_over_cause={'code': 'context_full', 'turn': 2},
        notice='',
    )
    json_file.write_json(path, older)
    capture = self.opened.captures / record['id']
    (capture / 'ref' / 'turn-1').mkdir(parents=True)
    (capture / 'index.json').write_text('{}')
    store = self._store()
    upgraded = store.get(record['id'])
    self.assertEqual(
        upgraded['chats'],
        [{
            'n': 1,
            'created_at': record['created_at'],
            'generation': older['generation'],
            'has_capture': True,
            'over': True,
            'notice': '',
            'over_cause': {'code': 'context_full', 'turn': 2},
        }],
    )
    for key in ('generation', 'has_capture', 'chat_over'):
      self.assertNotIn(key, upgraded)
    self.assertEqual(sorted(p.name for p in capture.iterdir()), ['chat-1'])
    self.assertTrue((capture / 'chat-1' / 'index.json').is_file())
    self.assertTrue((capture / 'chat-1' / 'ref' / 'turn-1').is_dir())
    # Written as this server keeps it: read again, nothing moves.
    self.assertEqual(json_file.read_json(path), upgraded)
    self.assertEqual(self._store().get(record['id']), upgraded)

  def test_an_older_servers_tool_settings_are_dropped(self):
    """Tools and constrained decoding go from a chat's settings; once."""
    record = self.store.create(_payload(self.model))
    self.store.add_chat(record['id'], {'ref': _GREEDY, 'target': {}})
    path = self.opened.records / f'{record["id"]}.json'
    older = json_file.read_json(path)
    older['chats'][0]['generation']['ref'].update(
        tools='[]', constrainedDecoding='on'
    )
    older['chats'][0]['generation']['target'].update(tools='[]')
    json_file.write_json(path, older)
    upgraded = self._store().get(record['id'])
    self.assertEqual(
        upgraded['chats'][0]['generation'], {'ref': _GREEDY, 'target': {}}
    )
    self.assertEqual(json_file.read_json(path), upgraded)

  def test_an_older_record_that_never_came_on_has_no_chat(self):
    """Why its start failed stays the Session's notice."""
    record = self.store.create(_payload(self.model))
    path = self.opened.records / f'{record["id"]}.json'
    older = {key: value for key, value in record.items() if key != 'chats'}
    older.update(
        generation={'ref': {}, 'target': {}},
        has_capture=False,
        started=True,
        chat_over=False,
        notice='Target: the runner did not start.',
    )
    json_file.write_json(path, older)
    upgraded = self._store().get(record['id'])
    self.assertEqual(upgraded['chats'], [])
    self.assertEqual(upgraded['notice'], 'Target: the runner did not start.')
    self.assertNotIn('generation', upgraded)

  def test_adopt_refuses_a_chat_not_as_this_server_keeps_it(self):
    """A chat lacking what the listing reads is refused; nothing is kept."""
    record = self.store.create(_payload(self.model))
    self.store.add_chat(record['id'], {'ref': {}, 'target': {}})
    self.store.set_chat_flags(record['id'], 1, has_capture=True)
    record = self.store.get(record['id'])
    (chat,) = record['chats']
    for broken, message in (
        ({k: v for k, v in chat.items() if k != 'generation'}, 'Chat 1 '),
        ({k: v for k, v in chat.items() if k != 'has_capture'}, 'Chat 1 '),
        ({**chat, 'generation': {'ref': {}}}, 'Chat 1 '),
        ({**chat, 'n': True}, 'no number'),
        ('chat', 'no number'),
    ):
      with self.subTest(message=message, broken=str(broken)[:40]):
        with self.assertRaisesRegex(ValueError, message):
          self.store.adopt(
              {**record, 'chats': [broken]},
              self.store.free_id(None),
              {'session_id': record['id']},
          )
    with self.assertRaisesRegex(ValueError, 'twice'):
      self.store.adopt(
          {**record, 'chats': [chat, chat]},
          self.store.free_id(None),
          {'session_id': record['id']},
      )
    self.assertEqual(self.store.list(), [record])

  def test_the_last_active_is_listed_first(self):
    """Created, imported or a turn sent: whichever is latest counts.

    A Session imported counts from its import, whenever it ran where it
    was exported; Sessions as active are in the order of their ids.
    """
    times = iter(f'2026-10-28T09:00:{s:02d}.000000+00:00' for s in range(9))
    with mock.patch.object(session_store.clock, 'now', lambda: next(times)):
      first = self.store.create(_payload(self.model, name='First'))
      self.store.create(_payload(self.model, name='Second'))
      self.store.add_chat(first['id'], {'ref': {}, 'target': {}})
      self.store.set_chat_flags(first['id'], 1, has_capture=True)
      self.store.turn_sent(first['id'])
    exported = self.store.get(first['id'])
    early = self.store.adopt(
        exported,
        self.store.free_id(None),
        {'imported_at': '2026-10-27T00:00:00.000000+00:00'},
    )
    late = self.store.adopt(
        exported,
        self.store.free_id(None),
        {'imported_at': '2026-10-29T00:00:00.000000+00:00'},
    )
    self.assertEqual(
        [r['name'] for r in self.store.list()],
        [late['name'], 'First', 'Second', early['name']],
    )
    both = self.store.adopt(
        exported,
        self.store.free_id(None),
        {'imported_at': '2026-10-29T00:00:00.000000+00:00'},
    )
    ties = [late['id'], both['id']]
    self.assertEqual([r['id'] for r in self.store.list()[:2]], sorted(ties))

  def test_adopt_refuses_a_session_whose_chats_kept_no_turn(self):
    """Only a Session a chat of which kept a turn is imported."""
    record = self.store.create(_payload(self.model))
    self.store.add_chat(record['id'], {'ref': {}, 'target': {}})
    with self.assertRaisesRegex(ValueError, 'kept no turn'):
      self.store.adopt(
          self.store.get(record['id']),
          self.store.free_id(None),
          {'session_id': record['id']},
      )


if __name__ == '__main__':
  test_main.main()
