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
"""Tests for session metadata persistence, renaming, and discovery."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from model_explorer_debugger.session_metadata import SessionMetadata


class SessionMetadataTest(unittest.TestCase):

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.store = SimpleNamespace(
        root=Path(self.tmp.name),
        session={
            'model': 'Test model',
            'runs': [{'precision': 'FP32'}, {'precision': 'FP16'}],
        },
    )
    self.subject = SessionMetadata(self.store)

  def test_identity_and_unknown_created_without_writes(self):
    summary = self.subject.summary()
    self.assertEqual(summary['id'], SessionMetadata(self.store).summary()['id'])
    self.assertIsNone(summary['created_at'])
    self.assertEqual(list(self.store.root.iterdir()), [])

  def test_rename_survives_restart_preserving_capture(self):
    before = json.dumps(self.store.session)
    identity = self.subject.identity
    self.subject.rename({'id': identity, 'name': '  My comparison  '})
    self.assertEqual(
        SessionMetadata(self.store).summary()['name'], 'My comparison'
    )
    self.assertEqual(self.subject.summary()['id'], identity)
    self.assertEqual(json.dumps(self.store.session), before)
    self.assertIsNone(self.subject.summary()['created_at'])

  def test_bad_names_and_identity_do_not_change_record(self):
    for name in ['', ' ', 'x' * 161, 'a\nb', None]:
      with self.assertRaises(ValueError):
        self.subject.rename({'id': self.subject.identity, 'name': name})
    with self.assertRaises(ValueError):
      self.subject.rename({'id': 'another-capture', 'name': 'Name'})
    self.assertFalse(self.subject.path.exists())

  def test_changed_capture_does_not_inherit_old_metadata(self):
    self.subject.rename({'id': self.subject.identity, 'name': 'Old name'})
    self.store.session['model'] = 'Different capture'
    other = SessionMetadata(self.store)
    self.assertNotEqual(other.identity, self.subject.identity)
    self.assertNotEqual(other.summary()['name'], 'Old name')

  def test_http_listing_rename_and_origin_rejection(self):
    from inline_asgi import client

    with client(self.store) as http:
      listing = http.get('/api/sessions').json()
      self.assertTrue(listing['capabilities']['create'])
      self.assertIsNone(listing['sessions'][0]['created_at'])
      payload = {'id': listing['sessions'][0]['id'], 'name': 'Saved name'}
      self.assertEqual(
          http.post('/api/sessions/rename', json=payload).json()['name'],
          'Saved name',
      )
      config = {
          'name': 'HTTP draft',
          'model': 'Test',
          'runs': [
              {'id': side, 'runtime': 'LiteRT-LM', 'artifact': 'model.bin'}
              for side in ('ref', 'target')
          ],
      }
      draft = http.post('/api/sessions/create', json=config).json()
      self.assertFalse(draft['has_capture'])
      for operation in ('delete', 'restore'):
        self.assertEqual(
            http.post(
                '/api/sessions/' + operation, json={'id': draft['id']}
            ).json()['id'],
            draft['id'],
        )
      artifact = http.post(
          '/api/artifacts/upload',
          content=b'HTTP fixture',
          headers={
              'Content-Type': 'application/octet-stream',
              'X-File-Name': 'fixture.bin',
          },
      ).json()
      self.assertEqual(
          (self.store.root / artifact['artifact']).read_bytes(), b'HTTP fixture'
      )
      self.assertEqual(
          http.post(
              '/api/sessions/rename',
              json=payload,
              headers={'Origin': 'https://unrelated.example'},
          ).status_code,
          403,
      )
      self.assertEqual(
          http.get('/api/sessions').json()['sessions'][0]['name'], 'Saved name'
      )

  def test_draft_management_and_capture_isolation(self):
    original = json.dumps(self.store.session)
    config = {
        'name': 'New comparison',
        'model': 'Test',
        'runs': [
            {
                'id': 'ref',
                'runtime': 'LiteRT-LM',
                'backend': 'CPU',
                'artifact': 'model.bin',
            },
            {
                'id': 'target',
                'runtime': 'LiteRT-LM',
                'backend': 'GPU',
                'artifact': 'model.bin',
            },
        ],
    }
    created = self.subject.manage('create', config)
    self.assertFalse(created['has_capture'])
    self.assertEqual(created['status'], 'draft')
    self.assertEqual(len(SessionMetadata(self.store).listing()['sessions']), 2)
    updated = self.subject.manage(
        'update', {**config, 'id': created['id'], 'name': 'Edited'}
    )
    self.assertEqual(updated['name'], 'Edited')
    duplicate = self.subject.manage('duplicate', {'id': self.subject.identity})
    self.assertFalse(duplicate['has_capture'])
    self.assertNotEqual(duplicate['id'], self.subject.identity)
    with self.assertRaises(ValueError):
      self.subject.manage('update', {**config, 'id': self.subject.identity})
    self.subject.manage('delete', {'id': created['id']})
    self.assertNotIn(
        created['id'], [r['id'] for r in self.subject.listing()['sessions']]
    )
    self.subject.manage('restore', {'id': created['id']})
    self.assertIn(
        created['id'], [r['id'] for r in self.subject.listing()['sessions']]
    )
    self.subject.manage('delete', {'id': self.subject.identity})
    self.assertEqual(self.subject.summary()['status'], 'saved')
    with self.assertRaises(ValueError):
      self.subject.rename({'id': self.subject.identity, 'name': 'Deleted'})
    self.subject.manage('delete', {'id': duplicate['id']})
    self.subject.manage('delete', {'id': created['id']})
    listing = SessionMetadata(self.store).listing()
    self.assertEqual(listing['sessions'], [])
    self.assertEqual(listing['configuration_template']['model'], 'Test model')
    self.assertEqual(json.dumps(self.store.session), original)

  def test_new_chat_configuration_is_atomic(self):
    before = self.subject.listing()['sessions']
    with self.assertRaises(ValueError):
      self.subject.manage(
          'chat',
          {
              'id': self.subject.identity,
              'generation': {'ref': {'maxOutputTokens': 0}, 'target': {}},
          },
      )
    self.assertEqual(self.subject.listing()['sessions'], before)
    child = self.subject.manage(
        'chat',
        {
            'id': self.subject.identity,
            'generation': {
                'ref': {'systemPrompt': 'Reference'},
                'target': {'temperature': 0.6},
            },
        },
    )
    saved = next(
        r
        for r in SessionMetadata(self.store).listing()['sessions']
        if r['id'] == child['id']
    )
    self.assertEqual(saved['generation']['ref']['systemPrompt'], 'Reference')
    self.assertEqual(saved['generation']['target']['temperature'], 0.6)

  def test_invalid_draft_does_not_write(self):
    with self.assertRaises(ValueError):
      self.subject.manage(
          'create', {'name': 'Bad', 'model': 'Test', 'runs': []}
      )
    self.assertFalse(self.subject.workspace_path.exists())

  def test_upload_bytes_and_invalid_paths(self):
    from io import BytesIO

    result = self.subject.upload(BytesIO(b'fixture'), 7, 'model.bin')
    self.assertEqual(
        (self.store.root / result['artifact']).read_bytes(), b'fixture'
    )
    with self.assertRaises(ValueError):
      self.subject.upload(BytesIO(b'x'), 1, '../outside')
    with self.assertRaises(ValueError):
      self.subject.upload(BytesIO(b'x'), 4, 'incomplete.bin')
    self.assertEqual(
        len(list((self.store.root / '.debugger-artifacts').glob('*/*'))), 1
    )

  def test_runtime_options_validate_and_persist(self):
    config = {
        'name': 'Options',
        'model': 'Test',
        'runs': [
            {
                'id': side,
                'runtime': 'LiteRT-LM',
                'artifact': 'model.bin',
                'backend': 'CPU',
                'cpuThreads': '4',
                'forceF32': True,
                'contextLength': '4096',
                'prefillBatchSizes': '128, 512',
            }
            for side in ('ref', 'target')
        ],
    }
    created = self.subject.manage('create', config)
    self.assertEqual(
        SessionMetadata(self.store).listing()['sessions'][1]['runs'][0][
            'contextLength'
        ],
        '4096',
    )
    self.assertTrue(created['runs'][0]['forceF32'])
    for key, value in [
        ('contextLength', '0'),
        ('cpuThreads', '1.5'),
        ('prefillBatchSizes', '128,0'),
        ('forceF32', 'false'),
        ('backend', 'TPU'),
    ]:
      invalid = json.loads(json.dumps(config))
      invalid['runs'][0][key] = value
      with self.assertRaises(ValueError):
        self.subject.manage('create', invalid)
    self.assertEqual(len(self.subject.listing()['sessions']), 2)

  def test_hf_reference_preserves_revision_without_capture(self):
    config = {
        'name': 'HF reference',
        'model': 'Test',
        'runs': [
            {
                'id': side,
                'runtime': 'LiteRT-LM',
                'source': 'huggingface',
                'repository': 'qa/example',
                'artifact': 'folder/model.litertlm',
                'revision': 'revision-123',
                'sourceUrl': (
                    'https://huggingface.co/qa/example/resolve/revision-123/folder/model.litertlm'
                ),
            }
            for side in ('ref', 'target')
        ],
    }
    created = self.subject.manage('create', config)
    saved = SessionMetadata(self.store).listing()['sessions'][1]
    self.assertEqual(saved['runs'][0]['revision'], 'revision-123')
    self.assertEqual(
        saved['runs'][0]['sourceUrl'], config['runs'][0]['sourceUrl']
    )
    self.assertFalse(created['has_capture'])
    self.assertFalse((self.store.root / '.debugger-artifacts').exists())
