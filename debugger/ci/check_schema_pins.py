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

"""Checks that the server reads models with the schemas the Runner runs.

And that the LiteRT-LM commit the server is written for, as it tells a
phone's App from (runner/app_version.py's LITERT_LM_COMMIT), is the Runner's.

The server's tap/ reads and rewrites models with two schemas it fetches by
commit (src/server/MODULE.bazel): LiteRT-LM's .litertlm header schema, and
the TFLite schema of the LiteRT that LiteRT-LM builds with. Both commits
must be the Runner's: the header schema's, and the commit of LiteRT-LM's
WORKSPACE the server fetches, are LITERT_LM_COMMIT of
src/runner/MODULE.bazel; the TFLite schema's is LITERT_REF of that
WORKSPACE.

RUNNER and SERVER are the MODULE.bazel of src/runner and src/server,
WORKSPACE LiteRT-LM's, APP_VERSION the server's runner/app_version.py.

Usage: python3 -I ci/check_schema_pins.py RUNNER SERVER WORKSPACE APP_VERSION
"""

import pathlib
import re
import sys

_RUNNER_COMMIT = re.compile(r'^LITERT_LM_COMMIT = "([0-9a-f]{40})"$', re.M)
_LITERT_REF = re.compile(r'^LITERT_REF = "([0-9a-f]{40})"$', re.M)
_SERVER_COMMIT = re.compile(r"^LITERT_LM_COMMIT = '([0-9a-f]{40})'$", re.M)
# The URL of each http_file the server pins, by name.
_HTTP_FILE = re.compile(
    r'http_file\(\s*name = "(?P<name>[^"]+)",.*?urls = \["(?P<url>[^"]+)"\]',
    re.S,
)
_GITHUB = re.compile(
    r'https://raw\.githubusercontent\.com/google-ai-edge/'
    r'(?P<repository>[^/]+)/(?P<commit>[0-9a-f]{40})/'
)
# The server's pins: name -> the repository its URL must be of.
_PINS = {
    'litertlm_header_schema': 'LiteRT-LM',
    'litertlm_workspace': 'LiteRT-LM',
    'tflite_schema': 'LiteRT',
}


def check(
    runner: str, server: str, workspace: str, app_version: str
) -> list[str]:
  """Each way the server's pins differ from the Runner's commits.

  Args:
      runner: The text of src/runner/MODULE.bazel.
      server: The text of src/server/MODULE.bazel.
      workspace: The text of LiteRT-LM's WORKSPACE the server fetches.
      app_version: The text of the server's runner/app_version.py.

  Returns:
      What is wrong, in words; empty if nothing.
  """
  found = _RUNNER_COMMIT.search(runner)
  if found is None:
    return ['src/runner/MODULE.bazel sets no LITERT_LM_COMMIT']
  litert_lm = found.group(1)
  found = _LITERT_REF.search(workspace)
  if found is None:
    return ["LiteRT-LM's WORKSPACE sets no LITERT_REF"]
  litert = found.group(1)
  expected = {'LiteRT-LM': litert_lm, 'LiteRT': litert}
  urls = {
      match.group('name'): match.group('url')
      for match in _HTTP_FILE.finditer(server)
  }
  problems = []
  found = _SERVER_COMMIT.search(app_version)
  if found is None:
    problems.append('runner/app_version.py sets no LITERT_LM_COMMIT')
  elif found.group(1) != litert_lm:
    problems.append(
        f'runner/app_version.py is written for LiteRT-LM {found.group(1)}, '
        f'the Runner builds with {litert_lm}'
    )
  for name, repository in _PINS.items():
    match = _GITHUB.match(urls.get(name, ''))
    if match is None or match.group('repository') != repository:
      problems.append(
          f'src/server/MODULE.bazel pins {name} by no {repository} commit'
      )
    elif match.group('commit') != expected[repository]:
      problems.append(
          f'src/server/MODULE.bazel pins {name} at {repository} '
          f'{match.group("commit")}, the Runner builds with '
          f'{expected[repository]}'
      )
  return problems


def main(argv: list[str]) -> int:
  """Prints every difference and returns 1 if there is any, else 0."""
  if len(argv) != 5:
    print(__doc__.strip().splitlines()[-1], file=sys.stderr)
    return 2
  runner, server, workspace, app_version = (
      pathlib.Path(path).read_text(encoding='utf-8') for path in argv[1:]
  )
  problems = check(runner, server, workspace, app_version)
  for problem in problems:
    print(f'check_schema_pins: {problem}', file=sys.stderr)
  return 1 if problems else 0


if __name__ == '__main__':
  sys.exit(main(sys.argv))
