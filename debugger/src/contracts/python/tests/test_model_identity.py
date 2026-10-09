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

"""Contract-only regressions extracted from the model registration suite."""

from pathlib import Path
import tempfile
import unittest
from model_debugger_contracts.model_identity import describe_model, verify_model


class ModelIdentityTests(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    self.model = self.root / 'synthetic-hf-model'
    self.model.mkdir()
    (self.model / 'config.json').write_text('{"model_type":"synthetic"}')
    (self.model / 'tokenizer.json').write_text('{"synthetic":true}')
    (self.model / 'model.safetensors').write_bytes(
        b'synthetic-checkpoint-bytes'
    )

  def test_hash_covers_tokenizer_changes_and_added_runtime_files(self):
    files, checksum = describe_model(self.model)
    for name in ('tokenizer.json', 'chat_template.jinja'):
      path = self.model / name
      original = path.read_bytes() if path.exists() else None
      path.write_text('changed runtime input')
      with (
          self.subTest(name=name),
          self.assertRaisesRegex(ValueError, 'changed'),
      ):
        verify_model(self.model, files, checksum)
      if original is None:
        path.unlink()
      else:
        path.write_bytes(original)
    self.assertEqual(
        verify_model(self.model, files, checksum), (files, checksum)
    )

  def test_hf_snapshot_symlink_hashes_target_bytes(self):
    weights = self.model / 'model.safetensors'
    blob = self.root / 'blob'
    weights.rename(blob)
    weights.symlink_to(blob)
    files, checksum = describe_model(self.model)
    self.assertEqual(
        next(
            entry['size']
            for entry in files
            if entry['path'] == 'model.safetensors'
        ),
        blob.stat().st_size,
    )
    blob.write_bytes(b'changed checkpoint')
    with self.assertRaisesRegex(ValueError, 'changed'):
      verify_model(self.model, files, checksum)

  def test_model_requires_local_safetensors_and_tokenizer(self):
    for name in ('model.safetensors', 'tokenizer.json', 'config.json'):
      path = self.model / name
      data = path.read_bytes()
      path.unlink()
      with self.subTest(name=name), self.assertRaises(ValueError):
        describe_model(self.model)
      path.write_bytes(data)
