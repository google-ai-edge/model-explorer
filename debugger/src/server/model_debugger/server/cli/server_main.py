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

"""The Server as a command: `model-debugger-server --help`.

Serves the Model Debugger on 127.0.0.1 until the process is stopped; its
address is printed on stdout. It logs to stderr: at INFO what it serves, the
lives of Sessions and runners, and each request that changes something; at
WARNING what goes wrong and is recovered from; at ERROR each start or turn
that fails. --verbosity (absl's) sets how much shows; --runner_log sets how
much of the runners' output does.

Its flags are checked before anything starts: a mistake (a --ui that is no
built UI, a --runner that is not there) is one line on stderr, and exit
status 1. It takes no model: a model is uploaded from the browser or
downloaded from Hugging Face into the workspace (session/model_store).
However it ends (Ctrl-C, SIGTERM, SIGHUP, an exception) it stops every
runner it launched first; see server.Server.
"""

from collections.abc import Sequence
import importlib.resources
import logging as python_logging
import math
import os
import pathlib
import shlex
import shutil
import signal
import sys
import types

from absl import app
from absl import flags
from absl import logging

from model_debugger.server import server as server_lib  # pylint: disable=g-import-not-at-top

# The repository this file sits in when run from a checkout:
# <root>/src/server/model_debugger/server/cli/server_main.py.
# pylint: disable=line-too-long
_ROOT = pathlib.Path(__file__).resolve().parents[5]
# pylint: enable=line-too-long
_CHECKOUT_UI = _ROOT / 'src/ui/dist/model_debugger/browser'
# Where Bazel puts the command-line runner, under the configuration it was
# built in. Not bazel-bin: that names the configuration of the last build,
# which may be an App's.
_BAZEL_OUT = _ROOT / 'src/runner/bazel-out'
# The PyTorch runner's Bazel workspace, in a checkout, and the command it
# builds there.
_PYTORCH_PROJECT = _ROOT / 'src/runner_pytorch'
_PYTORCH_TARGET = '//model_debugger/runner_pytorch/cli:runner'
_PYTORCH_RUNNER_IN_PROJECT = (
    _PYTORCH_PROJECT / 'bazel-bin/model_debugger/runner_pytorch/cli/runner'
)
# How to build it, for the reason the runner is not there.
_PYTORCH_SETUP = (
    f'build it with `cd {_PYTORCH_PROJECT} && bazel build {_PYTORCH_TARGET}`'
)
_RUNNER_IN_CONFIGURATION = 'bin/model_debugger/runner/cli/runner'
# What the command is called, as each line it fails with starts.
_PROGRAM = 'model-debugger-server'
# The signals that end the server as Ctrl-C does: each stops its runners.
_ENDING_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

_WORKSPACE = flags.DEFINE_string(
    'workspace',
    None,
    'where Sessions, models and captures are kept '
    f'(default: {server_lib.default_workspace()})',
)
_PORT = flags.DEFINE_integer(
    'port',
    None,
    f'the port to listen on (default: {server_lib.DEFAULT_PORT}, or the '
    f'next free one up to {server_lib.LAST_DEFAULT_PORT}); one in use is an '
    'error',
    lower_bound=0,
    upper_bound=65535,
)
_UI = flags.DEFINE_string(
    'ui',
    None,
    'the built UI to serve (default: the one the wheel carries; in a '
    f'checkout, {_CHECKOUT_UI}, if built)',
)
_RUNNER = flags.DEFINE_string(
    'runner',
    None,
    'the development build: a command-line runner offered on this computer '
    'besides the kept builds (default: the latest '
    f'{_BAZEL_OUT}/*/{_RUNNER_IN_CONFIGURATION}, if built)',
)
_KEPT_BUILDS = flags.DEFINE_string(
    'kept_builds',
    str(server_lib.default_kept_builds()),
    'the directory of the runner builds kept on this computer, one '
    "directory per LiteRT-LM commit, as src/runner's cli:keep writes them",
)
_PYTORCH_RUNNER = flags.DEFINE_string(
    'pytorch_runner',
    None,
    'the command that runs the PyTorch runner on this computer, as a shell '
    f'would split it (default: {_PYTORCH_RUNNER_IN_PROJECT}, in a checkout); '
    "'' for none",
)
# The levels of the runners' output --runner_log names.
_RUNNER_LOG_LEVELS = types.MappingProxyType({
    'info': python_logging.INFO,
    'warning': python_logging.WARNING,
    'error': python_logging.ERROR,
})
# By default only the runners' errors show: LiteRT-LM warns as every runner
# starts and on every turn (accelerators not found, sections a model lacks),
# none of it trouble. A runner that ends unasked is logged at ERROR with its
# last lines whatever this says.
_RUNNER_LOG = flags.DEFINE_enum(
    'runner_log',
    'error',
    list(_RUNNER_LOG_LEVELS),
    "how much of the output of this computer's runners shows: every line "
    '(info), what they mark as warnings and errors (warning), or errors '
    '(error); a runner that ends unasked always shows how, and its last '
    'lines',
)
_GRAPH_TIMEOUT = flags.DEFINE_float(
    'graph_timeout',
    server_lib.DEFAULT_GRAPH_TIMEOUT_SECONDS,
    "the seconds converting one model's graph may take, as a Session of it "
    'is initialized; the converter is stopped then',
    lower_bound=1,
)


def _path(value: str | None) -> pathlib.Path | None:
  return pathlib.Path(value) if value is not None else None


def default_ui() -> pathlib.Path | None:
  """Returns the built UI served without --ui, if any.

  That is the copy an installed wheel carries, else, in a source checkout,
  the UI built there.
  """
  # The wheel's copy: model_debugger/server/ui (BUILD.bazel's wheel).
  bundled = importlib.resources.files('model_debugger.server') / 'ui'
  for ui in (pathlib.Path(str(bundled)), _CHECKOUT_UI):
    if (ui / 'index.html').is_file():
      return ui
  return None


def built_runner() -> pathlib.Path | None:
  """Returns the command-line runner Bazel built last here, if any."""
  built = _BAZEL_OUT.glob(f'*/{_RUNNER_IN_CONFIGURATION}')
  return max(
      (path for path in built if path.is_file()),
      key=lambda path: path.stat().st_mtime,
      default=None,
  )


def pytorch_runner(command: str | None) -> tuple[str, ...] | None:
  """Returns the PyTorch runner's command: the flag's, or the checkout's."""
  if command is not None:
    return tuple(shlex.split(command)) or None
  if not (_PYTORCH_PROJECT / 'MODULE.bazel').is_file():
    return None
  return (str(_PYTORCH_RUNNER_IN_PROJECT),)


def _exit_on_signal(signum: int, frame: object) -> None:
  raise SystemExit(128 + signum)


def _hang_up(signum: int, frame: object) -> None:
  # As SIGTERM, which uvicorn, while it serves, takes to shut down first.
  del signum, frame
  signal.raise_signal(signal.SIGTERM)


def _directory_problem(path: pathlib.Path) -> str | None:
  """Why `path` is no readable directory, or None."""
  if not path.exists():
    return 'no such directory'
  if not path.is_dir():
    return 'not a directory'
  if not os.access(path, os.R_OK | os.X_OK):
    return 'cannot be read'
  return None


def _workspace_problem(path: pathlib.Path) -> str | None:
  """Why the server cannot keep a workspace in `path`, or None.

  One not there yet is created, in the nearest directory that is.

  Args:
    path: Workspace directory path to validate.

  Returns:
    An error description string, or None if valid.
  """
  if path.exists():
    if not path.is_dir():
      return 'not a directory'
    if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
      return 'not writable'
    return None
  parent = path.absolute().parent
  while not parent.exists():
    parent = parent.parent
  if not parent.is_dir():
    return f'cannot be created: {parent} is not a directory'
  if not os.access(parent, os.W_OK | os.X_OK):
    return f'cannot be created: {parent} is not writable'
  return None


def _command_problem(command: str) -> str | None:
  """Why the command `command` names, split as a shell would, cannot run."""
  try:
    words = shlex.split(command)
  except ValueError as error:
    return str(error)
  if not words:
    return None  # None at all: no PyTorch runner.
  program = words[0]
  if '/' not in program:
    if shutil.which(program) is None:
      return f'{program}: no such command'
    return None
  return _program_problem(pathlib.Path(program).expanduser())


def _program_problem(path: pathlib.Path) -> str | None:
  """Why `path` is no program this user can run, or None."""
  if not path.exists():
    return 'no such file'
  if not path.is_file():
    return 'not a file'
  if not os.access(path, os.X_OK):
    return 'not executable'
  return None


def flag_problems() -> list[str]:
  """Checks the flags; returns each mistake as a line of its own.

  What the flags name must be there now and be of its kind; relative paths
  are taken from the working directory. Read by the server only later, a
  mistake here would show only once the UI used it.

  Returns:
      '--flag value: what is wrong', for each mistake; none if all is
      well.
  """
  problems = []

  def check(flag: str, value: str, problem: str | None) -> None:
    if problem is not None:
      problems.append(f'--{flag} {value}: {problem}')

  if _UI.value is not None:
    path = pathlib.Path(_UI.value).expanduser()
    problem = _directory_problem(path)
    if problem is None and not (path / 'index.html').is_file():
      problem = 'no index.html in it: not a built UI'
    check('ui', _UI.value, problem)
  if _RUNNER.value is not None:
    check(
        'runner',
        _RUNNER.value,
        _program_problem(pathlib.Path(_RUNNER.value).expanduser()),
    )
  if _PYTORCH_RUNNER.value is not None:
    check(
        'pytorch_runner',
        repr(_PYTORCH_RUNNER.value),
        _command_problem(_PYTORCH_RUNNER.value),
    )
  # The default needs not be there: src/runner's cli:keep makes it.
  if flags.FLAGS['kept_builds'].present:
    check(
        'kept_builds',
        _KEPT_BUILDS.value,
        _directory_problem(pathlib.Path(_KEPT_BUILDS.value).expanduser()),
    )
  if _WORKSPACE.value is not None:
    check(
        'workspace',
        _WORKSPACE.value,
        _workspace_problem(pathlib.Path(_WORKSPACE.value).expanduser()),
    )
  if not math.isfinite(_GRAPH_TIMEOUT.value):
    check('graph_timeout', str(_GRAPH_TIMEOUT.value), 'not a number')
  return problems


def _fail(problem: str) -> SystemExit:
  """The exit of a mistake: one line on stderr, and status 1."""
  return SystemExit(f'{_PROGRAM}: {problem}')


def _parse_flags(argv: list[str]) -> list[str]:
  """Parses the flags as absl does; a mistake is one line, not a usage."""
  try:
    return flags.FLAGS(argv)
  except flags.Error as error:
    raise _fail(str(error)) from None


def main(argv: Sequence[str]) -> None:
  """Runs the server until interrupted.

  Args:
    argv: The command line left after the flags; the program alone.

  Raises:
    SystemExit: Command line arguments are invalid or server fails to start.
  """
  # Each signal that ends the server ends it through `finally`, never by
  # its default action, which would leave its runners running. Before
  # anything starts: a start stopped halfway stops what it started.
  signal.signal(signal.SIGINT, _exit_on_signal)
  signal.signal(signal.SIGTERM, _exit_on_signal)
  signal.signal(signal.SIGHUP, _hang_up)
  if len(argv) > 1:
    raise _fail(f'unexpected arguments: {" ".join(argv[1:])}')
  # `bazel run` starts the command in its runfiles; the paths it is given are
  # relative to where it was run.
  if working_directory := os.environ.get('BUILD_WORKING_DIRECTORY'):
    os.chdir(working_directory)
  problems = flag_problems()
  if problems:
    for problem in problems[:-1]:
      print(f'{_PROGRAM}: {problem}', file=sys.stderr)
    raise _fail(problems[-1])
  ui = _path(_UI.value) or default_ui()
  runner = _path(_RUNNER.value) or built_runner()
  server_lib.RUNNER_OUTPUT_LOG.setLevel(_RUNNER_LOG_LEVELS[_RUNNER_LOG.value])
  try:
    server = server_lib.Server.start(
        port=_PORT.value,
        workspace=_path(_WORKSPACE.value),
        runner=runner,
        ui=ui,
        kept_builds=_path(_KEPT_BUILDS.value),
        graph_timeout=_GRAPH_TIMEOUT.value,
        pytorch_runner=pytorch_runner(_PYTORCH_RUNNER.value),
        pytorch_setup=_PYTORCH_SETUP,
    )
  except server_lib.PortInUseError as error:
    # The reason alone, on stderr, and exit status 1: no traceback.
    if error.port is not None:
      advice = (
          ': something else listens on it, or connections to it closed '
          'less than half a minute ago. Pass another --port, or none for '
          f'the first free one of {server_lib.DEFAULT_PORT}-'
          f'{server_lib.LAST_DEFAULT_PORT}.'
      )
    else:
      advice = '; pass --port.'
    raise _fail(f'{error}{advice}') from None
  except server_lib.CannotListenError as error:
    raise _fail(f'{error}. Pass another --port.') from None
  except server_lib.WorkspaceInUseError as error:
    raise _fail(
        f'{error} Stop that server, or pass another --workspace.'
    ) from None
  try:
    state = server.state()
    lines = [
        'Serving:',
        f'  Workspace: {state.workspace}',
        f'  UI:        {state.ui or "(none; the API alone)"}',
        f'  Kept:      {state.kept_builds or "(none)"}',
    ]
    for build in state.builds:
      reason = f'; {build.reason}' if build.reason else ''
      lines.append(f'  Runner:    {build.label} ({build.runner}{reason})')
    if not state.builds:
      lines.append('  Runner:    (none; devices only)')
    logging.info('%s', '\n'.join(lines))
    # The address on stdout, whatever the verbosity: what to open, and
    # what a script that starts the server waits for.
    print(
        f'Model Debugger: http://{server_lib.HOST}:{state.port}/',
        flush=True,
    )
    # Once shut down, uvicorn raises again the signal that stopped it, which
    # _exit_on_signal turns into the exit through `finally`.
    server.wait()
  finally:
    # A second Ctrl-C must not cut the runners' stop short: it is bounded
    # (Server.close).
    for signum in _ENDING_SIGNALS:
      signal.signal(signum, signal.SIG_IGN)
    logging.info('Stopping the runners it launched')
    server.close()
    logging.info('The server stopped.')


def run() -> None:
  """Runs the command: the wheel's model-debugger-server."""
  # What the server does is logged at INFO (see the module's docstring).
  flags.FLAGS.set_default('verbosity', logging.INFO)
  app.run(main, flags_parser=_parse_flags)


if __name__ == '__main__':
  run()
