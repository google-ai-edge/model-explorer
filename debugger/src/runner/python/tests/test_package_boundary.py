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

"""Independent Runner and contracts packages do not depend on Server."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

SOURCE = Path(__file__).resolve().parents[3]
RUNNER = SOURCE / 'runner/python/package'
CONTRACTS = SOURCE / 'contracts/python/package'


class PackageBoundaryTests(unittest.TestCase):

  def isolated(self, *arguments, roots=(RUNNER, CONTRACTS), stdin=None):
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

  def test_runner_imports_with_no_server_or_optional_model_libraries(self):
    self.isolated(
        '-c',
        (
            '\n'
            'import importlib, importlib.abc, sys\n'
            'class BlockRuntime(importlib.abc.MetaPathFinder):\n'
            '    def find_spec(self, fullname, path=None, target=None):\n'
            "        if fullname.split('.')[0] in {'model_explorer_debugger',"
            " 'torch', 'transformers', 'safetensors',"
            " 'ai_edge_debugger_pytorch'}:\n"
            "            raise AssertionError('Unexpected package dependency:"
            " ' + fullname)\n"
            'sys.meta_path.insert(0, BlockRuntime())\n'
            "for name in ('python_runner_host', 'python_workers',"
            " 'pytorch_worker', 'pytorch_session', 'pytorch_capture',"
            " 'pytorch_environment'):\n"
            "    importlib.import_module('model_debugger_runner.' + name)\n"
            'from model_debugger_runner.pytorch_worker import _messages,'
            ' _settings\n'
            "assert _messages({'prompt':'hello'}) == [{'role':'user',"
            " 'content':'hello'}]\n"
            "assert _settings({})['do_sample'] is False\n"
        ),
    )

  def test_contracts_import_without_runner_or_server(self):
    self.isolated(
        '-c',
        (
            '\n'
            'import sys\n'
            'from model_debugger_contracts.errors import InputRejected\n'
            'from model_debugger_contracts.model_identity import'
            ' describe_model, verify_model\n'
            "assert InputRejected.code == 'input_rejected'\n"
            "assert not any(name.startswith(('model_debugger_runner',"
            " 'model_explorer_debugger')) for name in sys.modules)\n"
        ),
        roots=(CONTRACTS,),
    )

  def test_supervisor_cli_preserves_reset_protocol(self):
    for module, roots in (
        ('model_debugger_runner.python_runner_host', (RUNNER, CONTRACTS)),
    ):
      with self.subTest(module=module), tempfile.TemporaryDirectory() as folder:
        identity = str(uuid4())
        output = self.isolated(
            '-m',
            module,
            '--root',
            folder,
            roots=roots,
            stdin=json.dumps(dict(type='reset', requestId=identity)) + '\n',
        )
        (event,) = map(json.loads, output.splitlines())
        self.assertEqual(
            (event['type'], event['requestId']), ('reset', identity)
        )

  def test_worker_cli_accepts_close_without_model_imports(self):
    for module, roots in (
        ('model_debugger_runner.pytorch_worker', (RUNNER, CONTRACTS)),
    ):
      with self.subTest(module=module):
        self.assertEqual(
            self.isolated(
                '-m',
                module,
                '--serve',
                roots=roots,
                stdin='{"operation":"close"}\n',
            ),
            '',
        )
    self.isolated(
        '-c',
        """
import sys
from model_debugger_runner.pytorch_worker import main
for arguments in ([], ['request.json']):
    sys.argv = ['pytorch_worker', *arguments]
    try:
        main()
    except SystemExit as error:
        assert '--serve' in str(error)
    else:
        raise AssertionError('Worker accepted the removed single-request CLI')
assert 'torch' not in sys.modules
""",
    )

  def test_spawned_worker_receives_only_runner_contracts_and_runtime_roots(
      self,
  ):
    self.isolated(
        '-c',
        (
            '\n'
            'from pathlib import Path\n'
            'import os, tempfile\n'
            'from unittest.mock import MagicMock, patch\n'
            'from model_debugger_runner.python_workers import PythonWorkers\n'
            'with tempfile.TemporaryDirectory() as folder:\n'
            '    root = Path(folder)\n'
            '    workers = PythonWorkers()\n'
            '    process = MagicMock()\n'
            "    request = dict(operation='initialize', session_id='session',\n"
            "        run=dict(id='ref', runtime='PyTorch'),"
            " output=str(root/'output'), pytorch_root=str(root))\n"
            '    with'
            " patch('model_debugger_runner.python_workers.subprocess.Popen',"
            ' return_value=process) as launch:\n'
            '        try:\n'
            '            workers.execute(request, lambda *a, **k:None,'
            ' lambda:True)\n'
            '        except InterruptedError:\n'
            '            pass\n'
            '    argv = launch.call_args.args[0]\n'
            "    assert argv[1:] == ['-m',"
            " 'model_debugger_runner.pytorch_worker', '--serve']\n"
            '    roots ='
            " launch.call_args.kwargs['env']['PYTHONPATH'].split(os.pathsep)\n"
            "    assert any((Path(path)/'model_debugger_runner').is_dir() for"
            ' path in roots)\n'
            "    assert any((Path(path)/'model_debugger_contracts').is_dir()"
            ' for path in roots)\n'
            '    assert not'
            " any((Path(path)/'model_explorer_debugger').is_dir() for path in"
            ' roots)\n'
            '    workers.close()\n'
        ),
    )

  def test_configuration_explicit_roots_write_new_schema_to_requested_output(
      self,
  ):
    spec = importlib.util.spec_from_file_location(
        'runner_configure', SOURCE / 'runner/python/tools/configure_python.py'
    )
    configure = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configure)
    with (
        tempfile.TemporaryDirectory() as folder,
        patch.object(sys, 'path', [str(RUNNER), str(CONTRACTS), *sys.path]),
    ):
      from model_debugger_runner import pytorch_environment

      output = Path(folder) / 'python-runner.json'
      with (
          patch.object(
              sys,
              'argv',
              [
                  'configure_python.py',
                  '--pytorch-root',
                  folder,
                  '--package-root',
                  str(RUNNER),
                  '--contracts-root',
                  str(CONTRACTS),
                  '--output',
                  str(output),
              ],
          ),
          patch.object(
              pytorch_environment,
              'probe',
              return_value=dict(available=True, backends=['CPU']),
          ),
      ):
        configure.main()
      config = json.loads(output.read_text())
      self.assertEqual(config['version'], 2)
      self.assertEqual(
          config['module'], 'model_debugger_runner.python_runner_host'
      )
      self.assertEqual(config['packageRoot'], str(RUNNER))
      self.assertEqual(config['contractPackageRoot'], str(CONTRACTS))
      self.assertEqual(config['runtimeRoot'], str(Path(folder).resolve()))
      self.assertEqual(output.stat().st_mode & 0o777, 0o600)


if __name__ == '__main__':
  unittest.main()
