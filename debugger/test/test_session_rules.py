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
"""Tests for session validation rules and execution constraints."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from model_explorer_debugger.session_metadata import SessionMetadata
from model_explorer_debugger.session_registry import SessionRegistry
from model_explorer_debugger.session_rules import chat_config_locked, copy_name
from model_explorer_debugger.sessions_backend import SessionsBackend


class SessionRulesTests(unittest.TestCase):

  def setUp(self):
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    self.root = Path(directory.name)

  def runs(self, artifact):
    return [
        dict(
            id=role,
            device='local',
            runtime='LiteRT-LM',
            backend='CPU',
            artifact=artifact,
            source='registered',
        )
        for role in ('ref', 'target')
    ]

  def test_both_backends_satisfy_the_protocol(self):
    metadata = SessionMetadata(SimpleNamespace(root=self.root, session={}))
    registry = SessionRegistry(self.root / 'workspace')
    self.assertIsInstance(metadata, SessionsBackend)
    self.assertIsInstance(registry, SessionsBackend)

  def test_duplicates_truncate_the_name_identically_in_both_backends(self):
    long_name = 'n' * 160
    self.assertEqual(copy_name(long_name), 'n' * 150 + ' (copy)')
    metadata = SessionMetadata(SimpleNamespace(root=self.root, session={}))
    draft = metadata.manage(
        'create', dict(name=long_name, model='m', runs=self.runs('artifact'))
    )
    self.assertEqual(
        metadata.manage('duplicate', {'id': draft['id']})['name'],
        copy_name(long_name),
    )
    registry = SessionRegistry(self.root / 'workspace')
    model = self.root / 'fixture.litertlm'
    model.write_bytes(b'synthetic model; never executed')
    artifact = registry.register_model(model)
    capabilities = {
        'available': True,
        'runtimes': [
            {'id': 'LiteRT-LM', 'available': True, 'backends': ['CPU']}
        ],
    }
    with patch.object(registry, 'capabilities', return_value=capabilities):
      record = registry.manage(
          'create', dict(name=long_name, model='m', runs=self.runs(artifact))
      )
      self.assertEqual(
          registry.manage('duplicate', {'id': record['id']})['name'],
          copy_name(long_name),
      )

  def test_chat_configuration_lock_is_one_rule(self):
    self.assertFalse(chat_config_locked({'status': 'draft'}))
    for field in ('has_capture', 'read_only'):
      self.assertTrue(chat_config_locked({field: True}))
    self.assertTrue(chat_config_locked({'successful_turns': [{'n': 1}]}))
    metadata = SessionMetadata(SimpleNamespace(root=self.root, session={}))
    draft = metadata.manage(
        'create', dict(name='chat', model='m', runs=self.runs('artifact'))
    )
    draft['read_only'] = True
    with (
        patch.object(metadata, '_find', return_value=draft),
        self.assertRaisesRegex(ValueError, 'immutable'),
    ):
      metadata.manage('chat-config', {'id': draft['id'], 'generation': {}})


if __name__ == '__main__':
  unittest.main()
