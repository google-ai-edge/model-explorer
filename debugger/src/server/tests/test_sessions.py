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

"""Tests of Session and Chat records."""

from model_debugger_server import errors, models, sessions, workspace
import pytest


@pytest.fixture
def opened(tmp_path):
  """An open workspace, closed after the test."""
  value = workspace.Workspace(tmp_path / 'ws')
  yield value
  value.close()


@pytest.fixture
def model(opened, tmp_path):
  """A registered model file."""
  path = tmp_path / 'gemma.litertlm'
  path.write_bytes(b'model')
  return models.ModelStore(opened).register(path)


@pytest.fixture
def store(opened, model):
  """A SessionStore over the workspace, after the model is registered."""
  del model  # Registered before the store reads the models.
  return sessions.SessionStore(opened, models.ModelStore(opened))


def _run(side, model, **fields):
  return {
      'id': side,
      'runtime': 'LiteRT-LM',
      'artifact': model.artifact,
      'source': 'upload',
      'device': 'local',
      'backend': 'CPU',
      'contextLength': '1024',
      **fields,
  }


def _payload(registered, **fields):
  return {
      'name': ' First ',
      'model': 'Gemma 4 E2B',
      'runs': [_run('target', registered), _run('ref', registered)],
      **fields,
  }


def test_create_and_reload(store, opened, model):
  """Create and reload."""
  record = store.create(_payload(model))
  assert record == {
      'id': record['id'],
      'name': 'First',
      'model': 'Gemma 4 E2B',
      'runs': [
          {
              'id': 'ref',
              'runtime': 'LiteRT-LM',
              'artifact': model.artifact,
              'source': 'upload',
              'device': 'local',
              'backend': 'CPU',
              'contextLength': '1024',
          },
          {
              'id': 'target',
              'runtime': 'LiteRT-LM',
              'artifact': model.artifact,
              'source': 'upload',
              'device': 'local',
              'backend': 'CPU',
              'contextLength': '1024',
          },
      ],
      'created_at': record['created_at'],
      'deleted': False,
      'generation': {'ref': {}, 'target': {}},
      'has_capture': False,
      'execution_ended': False,
      'chat_over': False,
      'notice': '',
  }
  reopened = sessions.SessionStore(opened, models.ModelStore(opened))
  assert reopened.get(record['id']) == record
  assert reopened.list() == [record]


def test_create_keeps_options_set_and_drops_old_keys(store, model):
  """Create keeps options set and drops old keys."""
  # What the UI's duplicate posts: the source's runs with every option key,
  # the old runner's keys, and the tap settings.
  ref = _run(
      'ref',
      model,
      cpuThreads='4',
      enableYnnpack='on',
      useRingbuffersLocalAttention='',
      loraRank='',
      runnerBuild='build-1',
      repository='',
      revision='',
      sourceUrl='',
      precision='',
  )
  record = store.create(
      _payload(
          model,
          runs=[ref, _run('target', model)],
          tap_profile='',
          tap_points={},
      )
  )
  assert record['runs'][0] == {
      'id': 'ref',
      'runtime': 'LiteRT-LM',
      'artifact': model.artifact,
      'source': 'upload',
      'device': 'local',
      'backend': 'CPU',
      'contextLength': '1024',
      'cpuThreads': '4',
      'enableYnnpack': 'on',
  }


@pytest.mark.parametrize(
    ('change', 'message'),
    [
        ({'name': ''}, '1 to 160'),
        ({'name': 'x' * 161}, '1 to 160'),
        ({'name': 'a\tb'}, 'control characters'),
        ({'model': 'x' * 201}, 'at most 200'),
        ({'generation': {}}, 'Unknown session settings: generation'),
        ({'runs': []}, 'Reference and Target runs'),
    ],
)
def test_create_refuses_config(store, model, change, message):
  """Create refuses config."""
  with pytest.raises(ValueError, match=message):
    store.create(_payload(model, **change))


@pytest.mark.parametrize(
    ('fields', 'message'),
    [
        ({'runtime': 'PyTorch'}, 'Target: the runtime must be LiteRT-LM'),
        ({'artifact': 'model-0'}, 'Target: select a model'),
        ({'device': ''}, 'Target: select a device'),
        ({'backend': ''}, 'Target: the backend must be CPU or GPU'),
        ({'contextLength': 'big'}, 'Target: The context length'),
        ({'enableYnnpack': 'yes'}, 'Target: enableYnnpack must be on, off'),
        ({'cpuThreads': 4}, 'Target cpuThreads is not valid text'),
        ({'tapPoints': ''}, 'Unknown Target settings: tapPoints'),
    ],
)
def test_create_refuses_run(store, model, fields, message):
  """Create refuses run."""
  with pytest.raises(ValueError, match=message):
    store.create(
        _payload(
            model, runs=[_run('ref', model), _run('target', model, **fields)]
        )
    )


def test_create_refuses_a_model_no_longer_there(store, model):
  """Create refuses a model no longer there."""
  model.path.unlink()
  with pytest.raises(ValueError, match='no longer there'):
    store.create(_payload(model))


def test_update_until_it_ran(store, model):
  """Update until it ran."""
  record = store.create(_payload(model))
  updated = store.update({**_payload(model, name='Second'), 'id': record['id']})
  assert updated['name'] == 'Second'
  assert updated['created_at'] == record['created_at']
  store.set_flags(record['id'], execution_ended=True)
  with pytest.raises(errors.Conflict):
    store.update({**_payload(model), 'id': record['id']})
  with pytest.raises(errors.NotFound):
    store.update({**_payload(model), 'id': 'missing'})


def test_rename_delete_restore(store, model):
  """Rename delete restore."""
  record = store.create(_payload(model))
  assert store.rename(record['id'], ' Renamed ')['name'] == 'Renamed'
  assert store.delete(record['id'])['deleted']
  assert store.list() == []
  with pytest.raises(errors.NotFound):
    store.get(record['id'])
  with pytest.raises(errors.NotFound):
    store.delete(record['id'])
  assert store.restore(record['id'])['name'] == 'Renamed'
  with pytest.raises(errors.NotFound):
    store.restore('missing')


def test_create_chat(store, model):
  """Create chat."""
  parent = store.create(_payload(model))
  store.set_generation(
      parent['id'], {'ref': {'maxOutputTokens': 8}, 'target': {}}
  )
  store.set_flags(parent['id'], execution_ended=True)
  copied = store.create_chat(parent['id'], 'Chat 2', None)
  assert copied['parent_session_id'] == parent['id']
  assert copied['generation'] == {'ref': {'maxOutputTokens': 8}, 'target': {}}
  assert copied['runs'] == parent['runs']
  assert copied['execution_ended']
  assert not copied['has_capture']
  # A Chat of a Chat belongs to the same Session.
  third = store.create_chat(
      copied['id'], None, {'ref': {}, 'target': {'seed': 1}}
  )
  assert third['name'] == 'New chat'
  assert third['parent_session_id'] == parent['id']
  assert third['generation'] == {'ref': {}, 'target': {'seed': 1}}
  assert [r['id'] for r in store.list()] == [
      parent['id'],
      copied['id'],
      third['id'],
  ]
  with pytest.raises(ValueError, match='Unsupported generation settings'):
    store.create_chat(parent['id'], 'x', {'ref': {'bad': 1}, 'target': {}})


def test_set_flags_keeps_what_changed_meanwhile(store, model):
  """Flags set while a record was renamed or deleted keep both changes."""
  record = store.create(_payload(model))
  store.rename(record['id'], 'Renamed')
  store.set_flags(record['id'], has_capture=True, notice='Done.')
  stored = store.get(record['id'])
  assert stored['name'] == 'Renamed'
  assert stored['has_capture'] and stored['notice'] == 'Done.'
  store.delete(record['id'])
  store.set_flags(record['id'], chat_over=True)
  assert store.list() == []
  assert store.restore(record['id'])['chat_over']
  with pytest.raises(errors.NotFound):
    store.set_flags('missing', chat_over=True)


def test_set_generation_until_captured(store, model):
  """Set generation until captured."""
  record = store.create(_payload(model))
  updated = store.set_generation(
      record['id'], {'ref': {'samplerType': 'greedy'}, 'target': {}}
  )
  assert updated['generation']['ref'] == {'samplerType': 'greedy'}
  store.set_flags(record['id'], has_capture=True)
  with pytest.raises(errors.Conflict):
    store.set_generation(record['id'], {'ref': {}, 'target': {}})
