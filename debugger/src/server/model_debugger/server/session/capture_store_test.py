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

"""Tests of the capture store."""

import pathlib

from absl.testing import absltest
from model_debugger.server.session import capture_store
from model_debugger.server.session import model_store
from model_debugger.server.session import session_store
from model_debugger.server.session import tap_scans
from model_debugger.server.session import workspace
from model_debugger.server.testing import capture_fixtures
from model_debugger.server.testing import test_main
from model_debugger.server.util import errors

_SIDE = {'model': {'sha256': '0' * 64, 'name': 'm', 'size': 1}}


class CaptureStoreTest(absltest.TestCase):
  """Tests of the capture store."""

  def setUp(self):
    """A workspace with one Session record, and its capture store."""
    super().setUp()
    tmp_path = pathlib.Path(self.create_tempdir().full_path)
    opened = workspace.Workspace(tmp_path / 'workspace')
    self.addCleanup(opened.close)
    models = model_store.ModelStore(opened)
    path = opened.hf / 'gemma.litertlm'
    path.write_bytes(b'model')
    model = models.add_download(path)
    sessions = session_store.SessionStore(
        opened, models, tap_scans.TapScans(models)
    )
    record = sessions.create({
        'name': 'A',
        'model': 'Gemma',
        'runs': [
            {
                'id': side,
                'runtime': 'LiteRT-LM',
                'artifact': model.artifact,
                'source': 'upload',
                'device': 'local',
                'backend': 'CPU',
                'contextLength': '4096',
            }
            for side in ('ref', 'target')
        ],
    })
    sessions.add_chat(record['id'], {'ref': {}, 'target': {}})
    sessions.add_chat(record['id'], {'ref': {}, 'target': {}})
    self.stores = opened, sessions, record

  def test_a_created_capture_is_the_one_got(self):
    """get() answers the index create() made, captures/<id>/chat-<n>."""
    opened, sessions, record = self.stores
    captures = capture_store.CaptureStore(opened, sessions)
    index = captures.create(record['id'], 2, {'ref': _SIDE, 'target': _SIDE})
    self.assertIs(captures.get(record['id'], 2), index)
    self.assertEqual(index.root, opened.captures / record['id'] / 'chat-2')
    self.assertEqual(index.turns, [])
    with self.assertRaises(errors.NotFound):
      captures.get(record['id'], 1)

  def test_a_capture_is_read_back_after_a_restart(self):
    """A store that did not make the capture reads it from disk, once."""
    opened, sessions, record = self.stores
    capture_store.CaptureStore(opened, sessions).create(
        record['id'], 1, {'ref': _SIDE, 'target': _SIDE}
    )
    restarted = capture_store.CaptureStore(opened, sessions)
    index = restarted.get(record['id'], 1)
    self.assertEqual(index.sides['ref'], _SIDE)
    self.assertIs(restarted.get(record['id'], 1), index)

  def test_a_capture_with_a_turn_cut_short_is_read_once(self):
    """The turn is rejected on read; the capture is kept like any other."""
    opened, sessions, record = self.stores
    root = opened.captures / record['id'] / 'chat-1'
    capture_fixtures.capture(root, [('Hi', ['a'], True), ('Yo', ['b'], True)])
    cut = min((root / 'ref' / 'turn-2').glob('*.safetensors'))
    cut.write_bytes(cut.read_bytes()[:-1])
    captures = capture_store.CaptureStore(opened, sessions)

    index = captures.get(record['id'], 1)

    self.assertEqual([turn['n'] for turn in index.turns], [1])
    self.assertEqual([turn['n'] for turn in index.rejected], [2])
    self.assertIs(captures.get(record['id'], 1), index)

  def test_no_record_chat_or_capture_is_not_found(self):
    """An unknown record or chat, or a chat with no capture, has none."""
    opened, sessions, record = self.stores
    captures = capture_store.CaptureStore(opened, sessions)
    with self.assertRaises(errors.NotFound):
      captures.get('nobody', 1)
    with self.assertRaises(errors.NotFound):
      captures.get(record['id'], 3)
    with self.assertRaises(errors.NotFound):
      captures.get(record['id'], 1)

  def test_forget_drops_every_chat_of_a_session(self):
    """A Session deleted: its chats' indexes are read anew, if asked."""
    opened, sessions, record = self.stores
    captures = capture_store.CaptureStore(opened, sessions)
    first = captures.create(record['id'], 1, {'ref': _SIDE, 'target': _SIDE})
    second = captures.create(record['id'], 2, {'ref': _SIDE, 'target': _SIDE})
    captures.forget(record['id'])
    self.assertIsNot(captures.get(record['id'], 1), first)
    self.assertIsNot(captures.get(record['id'], 2), second)


if __name__ == '__main__':
  test_main.main()
