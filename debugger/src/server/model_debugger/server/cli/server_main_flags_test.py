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

"""The command's flags, checked before anything starts.

Each common mistake is one line, and the command exits 1 with no traceback.
"""

import pathlib
import subprocess
import sys
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from model_debugger.server.cli import server_main
from model_debugger.server.session import workspace
from model_debugger.server.testing import test_main


def _problems(*args: str) -> list[str]:
  """What the command line `args` is told, as the command checks it."""
  with flagsaver.flagsaver():
    flags.FLAGS(['model-debugger-server', *args])
    return server_main.flag_problems()


class FlagProblemsTest(absltest.TestCase):
  """Each mistake in what the flags name, as one line."""

  def setUp(self):
    """A built UI and a runner, all good, and a directory of neither."""
    super().setUp()
    self.tmp = pathlib.Path(self.create_tempdir().full_path)
    self.hf = self.tmp / 'hf'
    self.hf.mkdir()
    (self.hf / 'config.json').write_text('{}')
    self.ui = self.tmp / 'ui'
    self.ui.mkdir()
    (self.ui / 'index.html').write_text('')
    self.runner = self.tmp / 'runner'
    self.runner.write_text('#!/bin/sh\n')
    self.runner.chmod(0o755)
    self.file = self.tmp / 'file.txt'
    self.file.write_text('')

  def test_what_is_there_is_no_problem(self):
    """Flags that name what they should: no problem."""
    self.assertEqual(
        _problems(
            '--ui',
            str(self.ui),
            '--runner',
            str(self.runner),
            '--pytorch_runner',
            f'{self.runner} --flag',
            '--kept_builds',
            str(self.tmp),
            '--workspace',
            str(self.tmp / 'new' / 'workspace'),
        ),
        [],
    )
    self.assertEqual(_problems('--pytorch_runner', ''), [])
    self.assertEqual(_problems('--pytorch_runner', 'sh -c true'), [])

  def test_a_ui_not_built(self):
    """--ui names a directory with index.html."""
    self.assertEqual(
        _problems('--ui', str(self.hf)),
        [f'--ui {self.hf}: no index.html in it: not a built UI'],
    )
    self.assertEqual(
        _problems('--ui', str(self.file)),
        [f'--ui {self.file}: not a directory'],
    )

  def test_a_runner_that_cannot_run(self):
    """--runner names an executable file; --pytorch_runner a command."""
    self.assertEqual(
        _problems('--runner', str(self.tmp / 'absent')),
        [f'--runner {self.tmp / "absent"}: no such file'],
    )
    self.assertEqual(
        _problems('--runner', str(self.file)),
        [f'--runner {self.file}: not executable'],
    )
    self.assertEqual(
        _problems('--pytorch_runner', 'no-such-command-here run'),
        [
            "--pytorch_runner 'no-such-command-here run': "
            'no-such-command-here: no such command'
        ],
    )
    self.assertEqual(
        _problems('--pytorch_runner', f'{self.file}'),
        [f"--pytorch_runner '{self.file}': not executable"],
    )
    self.assertEqual(
        _problems('--pytorch_runner', '"unclosed'),
        ["--pytorch_runner '\"unclosed': No closing quotation"],
    )

  def test_directories_that_cannot_be_used(self):
    """--kept_builds and --workspace must be usable directories."""
    self.assertEqual(
        _problems('--kept_builds', str(self.tmp / 'absent')),
        [f'--kept_builds {self.tmp / "absent"}: no such directory'],
    )
    self.assertEqual(
        _problems('--workspace', str(self.file)),
        [f'--workspace {self.file}: not a directory'],
    )
    self.assertEqual(
        _problems('--workspace', str(self.file / 'workspace')),
        [
            f'--workspace {self.file / "workspace"}: cannot be created: '
            f'{self.file} is not a directory'
        ],
    )

  def test_a_graph_timeout_that_is_no_number(self):
    """--graph_timeout is a number of seconds."""
    self.assertEqual(
        _problems('--graph_timeout', 'nan'),
        ['--graph_timeout nan: not a number'],
    )

  def test_every_mistake_is_a_line(self):
    """Several mistakes: a line each."""
    self.assertLen(
        _problems('--ui', str(self.file), '--runner', str(self.tmp / 'x')),
        2,
    )


class DefaultUiTest(absltest.TestCase):
  """The UI served without --ui: the wheel's copy, else the checkout's."""

  def setUp(self):
    """A package and a checkout, neither with a built UI yet."""
    super().setUp()
    tmp = pathlib.Path(self.create_tempdir().full_path)
    self.package = tmp / 'package'
    self.bundled = self.package / 'ui'
    self.checkout = tmp / 'checkout'
    for ui in (self.bundled, self.checkout):
      ui.mkdir(parents=True)
    self.enter_context(
        mock.patch.object(
            server_main.importlib.resources,
            'files',
            lambda package: self.package,
        )
    )
    self.enter_context(
        mock.patch.object(server_main, '_CHECKOUT_UI', self.checkout)
    )

  def test_the_wheels_copy_first(self):
    """An installed wheel serves the UI it carries."""
    for ui in (self.bundled, self.checkout):
      (ui / 'index.html').write_text('')
    self.assertEqual(server_main.default_ui(), self.bundled)

  def test_the_checkouts_build_else(self):
    """A checkout, whose package carries none, serves the UI built there."""
    (self.checkout / 'index.html').write_text('')
    self.assertEqual(server_main.default_ui(), self.checkout)

  def test_none_built(self):
    """Neither built: the API alone."""
    self.assertIsNone(server_main.default_ui())


class FlagExitTest(absltest.TestCase):
  """The command run with a mistake: one line on stderr, and status 1."""

  def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
    tmp = self.create_tempdir()
    return subprocess.run(
        [
            sys.executable,
            '-m',
            'model_debugger.server.cli.server_main',
            '--kept_builds',
            tmp.mkdir('kept').full_path,
            '--pytorch_runner',
            '',
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

  def _assert_one_line(self, ended, line: str) -> None:
    self.assertEqual(ended.returncode, 1, ended.stderr)
    self.assertEqual(ended.stdout, '')
    self.assertEqual(ended.stderr, f'model-debugger-server: {line}\n')

  def test_a_ui_not_there(self):
    """A UI that is not there is named, and nothing starts."""
    absent = self.create_tempdir().full_path + '/y'
    workspace_dir = pathlib.Path(self.create_tempdir().full_path) / 'w'
    ended = self._run('--ui', absent, '--workspace', str(workspace_dir))
    self._assert_one_line(ended, f'--ui {absent}: no such directory')
    self.assertFalse(workspace_dir.exists())

  def test_a_model_is_not_a_flag(self):
    """--model and --pytorch_model are gone: models come by the UI."""
    for flag in ('--model', '--pytorch_model'):
      ended = self._run(flag, '/x')
      self.assertEqual(ended.returncode, 1, ended.stderr)
      self.assertIn('Unknown command line flag', ended.stderr)

  def test_a_flag_value_out_of_range(self):
    """A port out of range is absl's error, as one line."""
    ended = self._run('--port', '70000')
    self.assertEqual(ended.returncode, 1, ended.stderr)
    self.assertRegex(
        ended.stderr,
        r'^model-debugger-server: flag --port=70000: .*\n$',
    )

  def test_an_unknown_flag(self):
    """A misspelled flag is named."""
    ended = self._run('--modle', 'x')
    self.assertEqual(ended.returncode, 1, ended.stderr)
    self.assertRegex(ended.stderr, r"^model-debugger-server: .*'modle'.*\n$")

  def test_an_argument_that_is_no_flag(self):
    """A word that is no flag is named."""
    ended = self._run('serve')
    self._assert_one_line(ended, 'unexpected arguments: serve')

  def test_a_workspace_in_use(self):
    """A workspace another server holds is named, with what to do."""
    root = pathlib.Path(self.create_tempdir().full_path) / 'w'
    held = workspace.Workspace(root)
    try:
      ended = self._run('--workspace', str(root), '--port', '0')
    finally:
      held.close()
    self._assert_one_line(
        ended,
        'Another Model Debugger server is using the workspace '
        f'{root.resolve()}. Stop that server, or pass another '
        '--workspace.',
    )


if __name__ == '__main__':
  test_main.main()
