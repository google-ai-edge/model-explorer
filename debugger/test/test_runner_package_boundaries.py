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

"""Server package isolation and the installed App's remaining CLI entrypoint."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from uuid import uuid4

SOURCE = Path(__file__).resolve().parents[1] / 'src'
SERVER = SOURCE / 'server/package'
CONTRACTS = SOURCE / 'contracts/python/package'


class RunnerPackageBoundaryTests(unittest.TestCase):

  def isolated(self, *arguments, roots, stdin=None):
    with tempfile.TemporaryDirectory() as folder:
      result = subprocess.run(
          [sys.executable, '-S', *arguments],
          input=stdin,
          cwd=folder,
          env={**os.environ, 'PYTHONPATH': os.pathsep.join(map(str, roots))},
          text=True,
          capture_output=True,
          timeout=20,
      )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return result.stdout

  def test_server_imports_without_runner_execution(self):
    # -S avoids editable-package .pth hooks. Add only the installed numeric
    # dependencies' directories; never add the Runner source root or run site.
    dependency_roots = sorted({
        str(Path(importlib.util.find_spec(name).origin).parent.parent)
        for name in ('numpy', 'safetensors', 'ml_dtypes')
    })
    code = (
        '\n'
        'import importlib, importlib.abc, sys\n'
        'class BlockExecution(importlib.abc.MetaPathFinder):\n'
        '    def find_spec(self, fullname, path=None, target=None):\n'
        "        if fullname.split('.')[0] in {'model_debugger_runner',"
        " 'torch', 'transformers', 'litert_lm', 'ai_edge_debugger_pytorch'}:\n"
        "            raise AssertionError('Server imported execution package:"
        " ' + fullname)\n"
        'sys.meta_path.insert(0, BlockExecution())\n'
        'sys.path.extend(DEPENDENCY_ROOTS)\n'
        "for name in ('server', 'jobs', 'session_registry',"
        " 'runtime.runners', 'runtime.prepare_tap'):\n"
        "    importlib.import_module('model_explorer_debugger.' + name)\n"
        "assert not any(name.startswith('model_debugger_runner') for name in"
        ' sys.modules)\n'
    ).replace('DEPENDENCY_ROOTS', repr(dependency_roots))
    self.isolated('-c', code, roots=(SERVER, CONTRACTS))

  def test_installed_app_cli_resolves_canonical_packages_with_only_server_path(
      self,
  ):
    # 'reset' is the only session-scoped command the App sends besides execution
    # requests.
    for kind, expected in (('reset', 'reset'),):
      with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
        identity = str(uuid4())
        command = dict(type=kind, requestId=identity)
        output = self.isolated(
            '-m',
            'model_explorer_debugger.runtime.python_runner_host',
            '--root',
            folder,
            roots=(SERVER,),
            stdin=json.dumps(command) + '\n',
        )
        (event,) = map(json.loads, output.splitlines())
        self.assertEqual(
            (event['type'], event['requestId']), (expected, identity)
        )
        self.assertEqual(event['generationStatus'], 'succeeded')
