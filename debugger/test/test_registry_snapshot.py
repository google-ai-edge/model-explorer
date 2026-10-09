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
# ==============================================================================
"""Tests for model and device registry snapshot serialization."""

from pathlib import Path
import tempfile
import unittest

from model_explorer_debugger.session_registry import SessionRegistry


class RegistrySnapshotTests(unittest.TestCase):

  def setUp(self):
    self.folder = tempfile.TemporaryDirectory()
    self.addCleanup(self.folder.cleanup)
    self.registry = SessionRegistry(Path(self.folder.name))

  def test_state_is_replaced_only_through_restore(self):
    with self.assertRaisesRegex(AttributeError, 'restore'):
      self.registry.state = {'sessions': []}
    sessions = self.registry.state['sessions']
    snapshot = self.registry.snapshot()
    self.registry.state['sessions'].append({'id': 'draft', 'status': 'draft'})
    self.registry.restore(snapshot)
    self.assertEqual(self.registry.state['sessions'], [])
    self.assertIsNot(
        self.registry.state['sessions'],
        sessions,
        'restore swaps the collections',
    )
    self.assertIsNot(
        snapshot, self.registry.state, 'the caller keeps an independent copy'
    )

  def test_failed_transaction_restores_the_snapshot_in_place(self):
    state = self.registry.state
    with self.assertRaises(RuntimeError):
      with self.registry.transaction():
        self.registry.state['artifacts']['lost'] = {'path': 'x'}
        raise RuntimeError('rollback')
    self.assertIs(
        self.registry.state,
        state,
        'references taken before the transaction stay valid',
    )
    self.assertNotIn('lost', self.registry.state['artifacts'])


if __name__ == '__main__':
  unittest.main()
