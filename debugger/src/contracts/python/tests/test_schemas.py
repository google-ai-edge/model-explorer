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

from copy import deepcopy
import json
from pathlib import Path
import unittest

from model_debugger_contracts import schema

FIXTURE = Path(__file__).resolve().parents[2] / 'fixtures/capture-job.json'


class SchemaTests(unittest.TestCase):

  def setUp(self):
    self.job = json.loads(FIXTURE.read_text())

  def test_every_schema_loads_and_the_shared_fixture_is_a_valid_capture_job(
      self,
  ):
    for name in schema.SCHEMAS:
      self.assertEqual(schema.load_schema(name)['type'], 'object')
    self.assertIs(schema.validate('capture-job', self.job), self.job)
    schema.validate('tap-manifest', self.job['manifest'])

  def test_capture_job_violations_name_the_member(self):
    cases = [
        (lambda j: j.pop('manifest'), '$: missing required member'),
        (
            lambda j: j.update(modelSHA256='xyz'),
            '$.modelSHA256: does not match',
        ),
        (
            lambda j: j.update(contextLength=2048),
            '$.contextLength: expected one of',
        ),
        (
            lambda j: j.update(maxOutputTokens=33),
            '$.maxOutputTokens: expected at most 32',
        ),
        (lambda j: j.update(prompt=''), '$.prompt: expected at least 1'),
        (lambda j: j.update(extra=True), "$: unexpected member 'extra'"),
        (
            lambda j: j['manifest'].update(taps=j['manifest']['taps'] * 17),
            '$.manifest.taps: expected at most 16',
        ),
        (
            lambda j: j['manifest']['taps'][0].update(shape=[0]),
            '$.manifest.taps[0].shape[0]: expected at least 1',
        ),
        (
            lambda j: j['messages'][0].update(role='system'),
            '$.messages[0].role: expected one of',
        ),
        (lambda j: j.update(formatVersion=2), '$.formatVersion: expected 1'),
    ]
    for mutate, expected in cases:
      job = deepcopy(self.job)
      mutate(job)
      with self.assertRaises(schema.SchemaError) as caught:
        schema.validate('capture-job', job)
      self.assertIn(expected, str(caught.exception), expected)

  def test_capture_index_v2_accepts_both_runtimes_and_rejects_old_versions(
      self,
  ):
    record = {
        'format': 'safetensors',
        'path': 'tensors/000000.safetensors',
        'key': 'post_x',
        'dtype': 'float32',
        'shape': [1, 4],
    }
    litert = {
        'format_version': 2,
        'tensor_root': 'export',
        'tapped_sha256': 'a' * 64,
        'backend_requested': 'CPU',
        'tensors': [dict(record, signature='prefill', step=0)],
        'uncaptured_signatures': [],
    }
    pytorch = {
        'format_version': 2,
        'tensor_root': 'run',
        'runtime': 'PyTorch',
        'export_scope': 'all',
        'tensors': [dict(record, module_path='model.layers.0', layer=0)],
        'resources': [record],
        'native_trace': None,
    }
    schema.validate('capture-index-v2', litert)
    pytorch.pop('native_trace')
    schema.validate('capture-index-v2', pytorch)
    with self.assertRaisesRegex(
        schema.SchemaError, r'\$\.format_version: expected 2'
    ):
      schema.validate('capture-index-v2', dict(litert, format_version=1))
    with self.assertRaisesRegex(
        schema.SchemaError, r'\$\.tensors\[0\]: missing required member'
    ):
      schema.validate('capture-index-v2', dict(litert, tensors=[{'path': 'x'}]))
    with self.assertRaisesRegex(
        schema.SchemaError, r'\$\.tensor_root: expected one of'
    ):
      schema.validate('capture-index-v2', dict(litert, tensor_root='elsewhere'))


if __name__ == '__main__':
  unittest.main()
