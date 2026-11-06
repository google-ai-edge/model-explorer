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

"""The PyTorch runner as a command: serves the protocol until stopped.

Usage:
  model-debugger-runner-pytorch --model_dir DIR [--listen HOST:PORT]

It prints `runner: listening on port <port>` once it listens, as the LiteRT
runner does, and stops on SIGTERM or SIGINT. The model directory's id is
computed in the background (the first time, hashing its files takes
seconds), and Load waits for it.
"""

import argparse
import concurrent.futures
import os
import pathlib
import platform
import shutil
import signal
import sys
import threading

import grpc
# pyformat: disable
# pylint: disable=g-import-not-at-top
from model_debugger.protocol import runner_pb2_grpc
from model_debugger.runner_pytorch.service import model_id
from model_debugger.runner_pytorch.service import runner_service
# pylint: enable=g-import-not-at-top
# pyformat: enable

_RUN_PREFIX = 'run-'


def cache_directory() -> pathlib.Path:
  """Returns the runner's cache directory: hashes and runs' captures."""
  home = pathlib.Path.home()
  if platform.system() == 'Darwin':
    root = home / 'Library' / 'Caches'
  else:
    root = pathlib.Path(os.environ.get('XDG_CACHE_HOME') or home / '.cache')
  return root / 'model_debugger' / 'pytorch'


def serve(
    model_dir: pathlib.Path, listen: str, cache_dir: pathlib.Path
) -> tuple[grpc.Server, int, pathlib.Path]:
  """Starts serving the runner of `model_dir`.

  Args:
      model_dir: The Hugging Face model directory.
      listen: HOST:PORT; port 0 picks a free one.
      cache_dir: Where the file hashes and the run's captures go.

  Returns:
      The server, the port it listens on, and its run's directory, which
      the caller deletes once the server stopped.

  Raises:
      RuntimeError: It cannot listen there.
  """
  _remove_orphaned_runs(cache_dir)
  data_dir = cache_dir / f'{_RUN_PREFIX}{os.getpid()}'
  shutil.rmtree(data_dir, ignore_errors=True)
  hashes = cache_dir / 'hashes.json'
  computed = concurrent.futures.Future()

  def compute() -> None:
    try:
      computed.set_result(model_id.model_id(model_dir, hashes))
    except Exception as error:  # pylint: disable=broad-exception-caught
      # Load reports it.
      computed.set_exception(error)

  threading.Thread(target=compute, name='model-id', daemon=True).start()
  service = runner_service.RunnerService(model_dir, computed.result, data_dir)
  server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=16))
  runner_pb2_grpc.add_RunnerServicer_to_server(service, server)
  port = server.add_insecure_port(listen)
  if port == 0:
    raise RuntimeError(f'Cannot listen on {listen}.')
  server.start()
  return server, port, data_dir


def main(argv: list[str] | None = None) -> int:
  """Runs the runner until SIGTERM or SIGINT; returns the exit status."""
  parser = argparse.ArgumentParser(
      description='Serves a Model Debugger runner of PyTorch.'
  )
  parser.add_argument(
      '--model_dir',
      required=True,
      type=pathlib.Path,
      help='The Hugging Face Gemma 4 model directory to run.',
  )
  parser.add_argument(
      '--listen',
      default='127.0.0.1:8769',
      help='HOST:PORT to serve on; port 0 picks a free one.',
  )
  args = parser.parse_args(argv)
  if not args.model_dir.is_dir():
    print(f'runner: {args.model_dir} is no directory.', file=sys.stderr)
    return 1
  stop = threading.Event()
  for signal_number in (signal.SIGTERM, signal.SIGINT):
    signal.signal(signal_number, lambda *_: stop.set())
  try:
    server, port, data_dir = serve(
        args.model_dir.resolve(), args.listen, cache_directory()
    )
  except RuntimeError as error:
    print(f'runner: {error}', file=sys.stderr)
    return 1
  print(f'runner: listening on port {port}', flush=True)
  while not stop.wait(1):
    pass
  server.stop(grace=None).wait()
  shutil.rmtree(data_dir, ignore_errors=True)
  return 0


def _remove_orphaned_runs(cache_dir: pathlib.Path) -> None:
  """Deletes the run directories of runner processes that are gone."""
  if not cache_dir.is_dir():
    return
  for entry in cache_dir.iterdir():
    number = entry.name.removeprefix(_RUN_PREFIX)
    if entry.name == number or not number.isdigit():
      continue
    try:
      os.kill(int(number), 0)
    except ProcessLookupError:
      shutil.rmtree(entry, ignore_errors=True)
    except PermissionError:
      pass  # Another user's process.


if __name__ == '__main__':
  sys.exit(main())
