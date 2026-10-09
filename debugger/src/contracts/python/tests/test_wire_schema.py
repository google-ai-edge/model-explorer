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

"""Wire fixtures are valid, and the schema rejects what the Runner would."""

import json
from pathlib import Path
import unittest

from model_debugger_contracts import schema

WIRE = Path(__file__).resolve().parents[2] / 'fixtures/wire'


class WireSchemaTests(unittest.TestCase):

  def test_every_message_has_a_fixture_and_every_fixture_is_valid(self):
    covered = set()
    for path in sorted(WIRE.glob('*/*.json')):
      with self.subTest(fixture=f'{path.parent.name}/{path.name}'):
        message = json.loads(path.read_text())
        self.assertEqual(message['type'], path.name.split('.')[0])
        schema.validate_wire(path.parent.name, message)
        covered.add(f"{path.parent.name}.{message['type']}")
    messages = {
        name
        for name in schema.load_schema('runner-wire')['$defs']
        if '.' in name
    }
    self.assertEqual(covered, messages)

  def test_violations_name_the_member(self):
    load = lambda name: json.loads((WIRE / name).read_text())
    cases = [
        (
            'server',
            dict(load('server/activate.json'), runId='both'),
            '$.runId: expected one of',
        ),
        (
            'server',
            dict(load('server/model_link.json'), path='models/relative'),
            '$.path: does not match',
        ),
        (
            'server',
            dict(load('server/file_chunk.v5.json'), encoding='base64'),
            "$.encoding: expected 'binary'",
        ),
        (
            'server',
            dict(load('server/cancel.json'), requestId='1'),
            '$.requestId: does not match',
        ),
        (
            'server',
            dict(load('server/close.json'), extra=True),
            "$: unexpected member 'extra'",
        ),
        (
            'runner',
            dict(load('runner/hello.json'), protocolVersion=3),
            '$.protocolVersion: expected one of',
        ),
        (
            'runner',
            dict(load('runner/run_files.json'), files=[]),
            '$.files: expected at least 1',
        ),
        (
            'runner',
            {'type': 'telemetry'},
            "$.type: unknown runner message 'telemetry'",
        ),
    ]
    for sender, message, expected in cases:
      with self.assertRaises(schema.SchemaError) as caught:
        schema.validate_wire(sender, message)
      self.assertIn(expected, str(caught.exception), expected)

  def test_a_generate_job_is_checked_as_a_capture_job(self):
    message = json.loads((WIRE / 'server/generate.json').read_text())
    message['job']['maxOutputTokens'] = 33
    with self.assertRaisesRegex(
        schema.SchemaError,
        r'\$\.job\.maxOutputTokens: expected at most 32|matches none',
    ):
      schema.validate_wire('server', message)


if __name__ == '__main__':
  unittest.main()
