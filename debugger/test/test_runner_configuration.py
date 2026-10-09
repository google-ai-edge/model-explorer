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

"""Verifies Server discovery of existing and extracted Runner configs."""

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

from model_explorer_debugger import generation_config
from model_explorer_debugger import session_registry
from model_explorer_debugger import session_rules
from model_explorer_debugger.runtime import protocol
from model_explorer_debugger.runtime import runners


class RunnerConfigurationTests(unittest.TestCase):
  """Verifies local Runner configuration discovery and capability reporting."""

  def test_legacy_and_independent_package_configs_advertise_same_backends(self):
    """Checks v1 and v2 Runner config schemas report identical backends."""
    with tempfile.TemporaryDirectory() as folder:
      home = pathlib.Path(folder)
      config = (
          home
          / 'Library/Application Support'
          / 'Model Debugger Runner/python-runner.json'
      )
      config.parent.mkdir(parents=True)
      common = dict(
          executable=sys.executable,
          runtimeRoot='/runtime',
          backends=['CPU', 'MPS'],
      )
      records = (
          dict(common, version=1, packageRoot='/server/package'),
          dict(
              common,
              version=2,
              packageRoot='/runner/python/package',
              contractPackageRoot='/contracts/python/package',
              module='model_debugger_runner.python_runner_host',
          ),
      )
      for record in records:
        with self.subTest(version=record['version']):
          config.write_text(json.dumps(record))
          capability = runners.local_python_capability(home_dir=home)
          self.assertTrue(capability['available'])
          self.assertEqual(capability['backends'], ['CPU', 'MPS'])
      with mock.patch.object(
          pathlib.Path,
          'home',
          autospec=True,
          spec_set=True,
          return_value=home,
      ):
        config.write_text(json.dumps(records[1]))
        self.assertTrue(
            runners.local_python_capability(home_dir='')['available']
        )
        reg = session_registry.SessionRegistry(
            home / 'workspace_empty_home',
            home_dir='',
        )
        self.assertIsNone(reg.home_dir)
        reg.close()
        config.write_text(json.dumps(dict(common, version=999)))
        self.assertFalse(runners.local_python_capability()['available'])

  def test_capabilities_report_generation_and_upload_limits(self):
    """Checks SessionRegistry capabilities include token and upload limits."""
    self.assertNotIn('atomic_json', session_registry.__all__)
    with tempfile.TemporaryDirectory() as folder:
      reg = session_registry.SessionRegistry(
          pathlib.Path(folder) / 'workspace',
          home_dir=pathlib.Path(folder),
      )
      capabilities = reg.capabilities()
      reg.close()
    runtimes = {runtime['id']: runtime for runtime in capabilities['runtimes']}
    self.assertEqual(
        runtimes['LiteRT-LM']['max_output_tokens'],
        protocol.NATIVE_MAX_OUTPUT_TOKENS,
    )
    self.assertEqual(
        runtimes['LiteRT-LM']['prompt_limit_bytes'],
        protocol.NATIVE_MAX_PROMPT_BYTES,
    )
    self.assertEqual(
        runtimes['PyTorch']['max_output_tokens'],
        generation_config.MAX_OUTPUT_TOKENS,
    )
    self.assertEqual(
        capabilities['upload_limit_bytes'], session_rules.UPLOAD_LIMIT_BYTES
    )
    self.assertEqual(
        capabilities['upload_idle_timeout_seconds'],
        session_rules.UPLOAD_IDLE_TIMEOUT_SECONDS,
    )


if __name__ == '__main__':
  unittest.main()
