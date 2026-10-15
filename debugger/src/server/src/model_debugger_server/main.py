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

"""Runs the Model Debugger server: `model-debugger-server --help`."""

import argparse
import logging
import pathlib
import signal
import socket
import sys

from model_debugger_server import (
    execution,
    http_app,
    jobs,
    launchers,
    models,
    sessions,
    workspace,
)
import uvicorn

DEFAULT_PORT = 8080
LAST_DEFAULT_PORT = 8099
HOST = '127.0.0.1'

# The repository this file sits in when run from a checkout:
# <root>/src/server/src/model_debugger_server/main.py.
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DEFAULT_UI = _ROOT / 'src/ui/dist/model_debugger/browser'
# Where Bazel puts the command-line runner, under the configuration it was
# built in. Not bazel-bin: that names the configuration of the last build,
# which may be an App's.
_BAZEL_OUT = _ROOT / 'src/runner/bazel-out'
_RUNNER_IN_CONFIGURATION = 'bin/model_debugger/runner/cli/runner'


def built_runner() -> pathlib.Path | None:
  """Returns the command-line runner Bazel built last here, if any."""
  built = _BAZEL_OUT.glob(f'*/{_RUNNER_IN_CONFIGURATION}')
  return max(
      (path for path in built if path.is_file()),
      key=lambda path: path.stat().st_mtime,
      default=None,
  )


def _parse(argv: list[str] | None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      prog='model-debugger-server',
      description='Runs the Model Debugger server on 127.0.0.1.',
  )
  parser.add_argument(
      '--workspace',
      type=pathlib.Path,
      default=None,
      help=(
          'where Sessions, models and captures are kept '
          f'(default: {workspace.default_root()})'
      ),
  )
  parser.add_argument(
      '--port',
      type=int,
      default=None,
      help=(
          f'the port to listen on (default: {DEFAULT_PORT}, or the next free '
          f'one up to {LAST_DEFAULT_PORT})'
      ),
  )
  parser.add_argument(
      '--ui',
      type=pathlib.Path,
      default=None,
      help=f'the built UI to serve (default: {_DEFAULT_UI}, if built)',
  )
  parser.add_argument(
      '--runner',
      type=pathlib.Path,
      default=None,
      help=(
          'the command-line runner that runs on this computer (default: '
          f'the latest {_BAZEL_OUT}/*/{_RUNNER_IN_CONFIGURATION}, if built)'
      ),
  )
  parser.add_argument(
      '--model',
      type=pathlib.Path,
      action='append',
      default=[],
      help='a .litertlm file to offer as a model, by reference; repeatable',
  )
  return parser.parse_args(argv)


def _listen(port: int | None) -> socket.socket:
  """A socket bound on HOST: the port asked for, or the first free default."""
  ports = (
      [port] if port is not None else range(DEFAULT_PORT, LAST_DEFAULT_PORT + 1)
  )
  for candidate in ports:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
      listener.bind((HOST, candidate))
    except OSError:
      listener.close()
      continue
    return listener
  if port is not None:
    raise SystemExit(f'model-debugger-server: port {port} is in use.')
  raise SystemExit(
      f'model-debugger-server: ports {DEFAULT_PORT}-{LAST_DEFAULT_PORT} are '
      'in use; pass --port.'
  )


def _exit_on_signal(signum: int, frame: object) -> None:
  raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> int:
  """Runs the server until interrupted.

  Args:
    argv: the command-line arguments; sys.argv[1:] when None.

  Returns:
    The exit status.
  """
  args = _parse(argv)
  logging.basicConfig(
      level=logging.INFO, format='%(asctime)s %(name)s: %(message)s'
  )
  ui = args.ui
  if ui is None and (_DEFAULT_UI / 'index.html').is_file():
    ui = _DEFAULT_UI
  runner = args.runner if args.runner is not None else built_runner()
  listener = _listen(args.port)
  space = workspace.Workspace(args.workspace or workspace.default_root())
  executions = None
  try:
    store = models.ModelStore(space)
    registered = [store.register(path) for path in args.model]
    records = sessions.SessionStore(space, store)
    tasks = jobs.Jobs()
    devices = launchers.Launchers(runner)
    executions = execution.Executions(records, store, tasks, devices, space)
    # Learn the options the local runner takes before the UI asks.
    if any(device.id == 'local' for device in devices.devices()):
      executions.inspect('local')
    app = http_app.create_app(executions, records, store, tasks, ui)
    port = listener.getsockname()[1]
    print(f'Workspace: {space.root}')
    print(f'UI:        {ui or "(none; the API alone)"}')
    print(f'Runner:    {runner or "(none; devices only)"}')
    for model in registered:
      print(f'Model:     {model.name} ({model.artifact})')
    print(f'Model Debugger: http://{HOST}:{port}/', flush=True)
    server = uvicorn.Server(uvicorn.Config(app, log_level='info'))
    # Once shut down, uvicorn raises again the signal that stopped it; the
    # default action of SIGTERM would end this process before it stops its
    # runners, so SIGTERM exits as SIGINT does, through `finally`.
    signal.signal(signal.SIGTERM, _exit_on_signal)
    server.run(sockets=[listener])
  finally:
    if executions is not None:
      executions.shutdown()
    space.close()
    listener.close()
  return 0


if __name__ == '__main__':
  sys.exit(main())
