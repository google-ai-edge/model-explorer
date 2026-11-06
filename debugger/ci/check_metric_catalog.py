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

"""Checks that the UI's copy of GET /api/metrics is the server's answer.

The UI's tests describe metrics as src/ui/src/server/testing/
metric_catalog_fixture.json has them. That file must be what the server
answers to GET /api/metrics, JSON types and all (true is not 1, and 6.0 is
not 6); with --write, it is written so. The server is src/server's, built by
Bazel (bazelisk, or bazel) and asked once: on a free port, with a workspace
of its own and no kept runner builds, and stopped then.

Usage: python3 -I ci/check_metric_catalog.py [--write] /path/to/fixture.json
"""

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request

# The server's Bazel workspace and its command.
_SERVER = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'server'
_TARGET = '//model_debugger/server/cli:server'
# What the server prints, followed by its address, once it listens.
_LISTENING = 'Model Debugger: '
# How long the server may take to listen, and then to answer.
_TIMEOUT_SECONDS = 120


class _Error(Exception):
  """The server could not be built, started or asked."""


def _served_catalog(scratch: pathlib.Path) -> object:
  """Returns the server's answer to GET /api/metrics, parsed.

  Args:
      scratch: An empty directory for the server's script, log, workspace and
        kept builds.

  Raises:
      _Error: The server could not be built, started or asked.
  """
  bazel = shutil.which('bazelisk') or shutil.which('bazel')
  if bazel is None:
    raise _Error('bazelisk is not installed (brew install bazelisk)')
  script = scratch / 'server'
  if subprocess.run(
      [bazel, 'run', f'--script_path={script}', _TARGET],
      cwd=_SERVER,
      check=False,
  ).returncode:
    raise _Error(f'bazel could not build {_TARGET}')
  kept_builds = scratch / 'kept_builds'
  kept_builds.mkdir()
  log = scratch / 'server.log'
  with log.open('w') as stderr:
    server = subprocess.Popen(
        [
            str(script),
            f'--workspace={scratch / "workspace"}',
            f'--kept_builds={kept_builds}',
            '--port=0',
        ],
        stdout=subprocess.PIPE,
        stderr=stderr,
        text=True,
    )
  # A server that never listens is killed, which ends its output.
  watchdog = threading.Timer(_TIMEOUT_SECONDS, server.kill)
  watchdog.start()
  try:
    for line in server.stdout:
      if line.startswith(_LISTENING):
        address = line[len(_LISTENING) :].strip()
        break
    else:
      raise _Error('the server stopped before it listened:\n' + log.read_text())
    # Straight to the server, whatever proxy the environment names.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
      with opener.open(
          f'{address}api/metrics', timeout=_TIMEOUT_SECONDS
      ) as response:
        return json.load(response)
    except OSError as error:
      raise _Error(f'GET {address}api/metrics: {error}') from None
  finally:
    watchdog.cancel()
    server.terminate()
    server.wait()
    server.stdout.close()


def _canonical(value: object) -> str:
  """The JSON text of `value`, the same for equal values of equal types."""
  return json.dumps(value, sort_keys=True, ensure_ascii=False)


def main(argv: list[str]) -> int:
  """Returns 0 if the file is the catalog (or was written), else 1."""
  args = argv[1:]
  write = args[:1] == ['--write']
  if write:
    args = args[1:]
  if len(args) != 1:
    print(__doc__.strip().splitlines()[-1], file=sys.stderr)
    return 2
  path = pathlib.Path(args[0]).resolve()
  try:
    with tempfile.TemporaryDirectory() as scratch:
      catalog = _served_catalog(pathlib.Path(scratch))
  except _Error as error:
    print(f'check_metric_catalog: {error}', file=sys.stderr)
    return 1
  if write:
    path.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    return 0
  # Compared as canonical JSON text, not as parsed values: Python's == takes
  # true for 1 and 6 for 6.0. The layout is free (prettier formats the
  # file).
  if path.is_file() and _canonical(
      json.loads(path.read_text('utf-8'))
  ) == _canonical(catalog):
    return 0
  print(
      f"check_metric_catalog: {path} is not the server's metric catalog; "
      f'write it with: python3 -I {pathlib.Path(__file__).resolve()} '
      f'--write {path}',
      file=sys.stderr,
  )
  return 1


if __name__ == '__main__':
  sys.exit(main(sys.argv))
