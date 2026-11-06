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

"""Writes each architecture's skeletons from its pinned configs, or checks.

For every skeleton of every architecture (catalog), the skeleton its
definition makes of the pinned config.json (catalog.generate()), as its file
holds it (skeleton.dumps()). It writes the skeleton files; with --check, it
writes nothing and fails unless every file is byte for byte what it would
write. Either way it fails when a pinned config is not the one its entry's
SHA-256 names, when a config is of another architecture, or when a
skeleton file is not the one its entry's SHA-256 names (an entry is
updated by hand, so that a changed skeleton is a reviewed change).

The files are the source tree's when run by `bazel run` (in
BUILD_WORKSPACE_DIRECTORY), else beside this module:

  bazel run //model_debugger/server/architectures:generate [-- --check]
"""

import hashlib
import json
import os
import pathlib
import sys

from absl import app
from absl import flags
from model_debugger.server.architectures import catalog
from model_debugger.server.architectures import skeleton

_CHECK = flags.DEFINE_bool(
    'check', False, 'Write nothing; fail unless every file is unchanged.'
)
# The package's directory under a Bazel workspace's root.
_PACKAGE = pathlib.Path('model_debugger/server/architectures')


def _sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def run(directory: pathlib.Path, check: bool) -> list[str]:
  """Writes or checks every skeleton under `directory`'s data.

  Args:
      directory: The package's directory: data/<architecture>/ in it holds each
        architecture's file, configs and skeletons.
      check: Whether to write nothing and only compare.

  Returns:
      What is wrong, one line each; [] if nothing is.
  """
  problems = []
  for known in catalog.architectures():
    here = directory / 'data' / known.id
    for entry in known.spec['skeletons']:
      pinned = here / entry['pinned']['file']
      data = pinned.read_bytes()
      if _sha256(data) != entry['pinned']['sha256']:
        problems.append(
            f'{pinned}: its SHA-256 is not {entry["pinned"]["sha256"]}'
            ', the one its entry pins.'
        )
        continue
      made_by, document = catalog.generate(json.loads(data))
      if made_by.id != known.id:
        problems.append(f'{pinned} is of {made_by.name}, not {known.name}.')
        continue
      body = skeleton.dumps(document).encode('utf-8')
      path = here / entry['file']
      if check:
        if not path.is_file() or path.read_bytes() != body:
          problems.append(
              f'{path} is not what its config makes; run generate '
              'without --check.'
          )
      elif not path.is_file() or path.read_bytes() != body:
        path.write_bytes(body)
        print(f'Wrote {path}.')
      if _sha256(body) != entry['sha256']:
        problems.append(
            f'{path}: its SHA-256 is {_sha256(body)}, but '
            f'{known.id}/architecture.json says {entry["sha256"]}; '
            'update it once the change is reviewed.'
        )
  return problems


def _directory() -> pathlib.Path:
  """The source tree's package directory under `bazel run`, else this."""
  workspace = os.environ.get('BUILD_WORKSPACE_DIRECTORY')
  if workspace:
    return pathlib.Path(workspace) / _PACKAGE
  return pathlib.Path(__file__).parent


def main(argv: list[str]) -> int:
  """Writes or checks the skeletons; 1 if anything is wrong."""
  if len(argv) > 1:
    raise app.UsageError('Too many arguments.')
  problems = run(_directory(), _CHECK.value)
  for problem in problems:
    print(f'generate: {problem}', file=sys.stderr)
  return 1 if problems else 0


if __name__ == '__main__':
  app.run(main)
