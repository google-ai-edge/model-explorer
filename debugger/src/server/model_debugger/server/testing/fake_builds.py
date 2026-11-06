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

"""Kept runner builds as src/runner's `cli:keep` writes them, for tests."""

import hashlib
import json
import pathlib

# Two LiteRT-LM commits, older first.
OLD = '1111111111111111111111111111111111111111'
NEW = '2222222222222222222222222222222222222222'
# The libraries `cli:keep` keeps beside the runner.
LIBRARIES = (
    'liblitert-lm.dylib',
    'libLiteRt.dylib',
    'libGemmaModelConstraintProvider.dylib',
    'libwebgpu_dawn.dylib',
    'libLiteRtWebGpuAccelerator.dylib',
    'libLiteRtTopKWebGpuSampler.dylib',
)


def keep(
    kept: pathlib.Path,
    commit: str,
    *,
    runner: str = '#!/bin/sh\nexit 1\n',
    commit_date: str = '2026-10-12T22:51:09-07:00',
    cl: str = '998385386',
) -> pathlib.Path:
  """Keeps a build of `commit` in `kept`, as `cli:keep` does.

  Args:
      kept: The kept builds' directory; created if missing.
      commit: The build's LiteRT-LM commit, which names its directory.
      runner: The runner's text: a script that stands in for it.
      commit_date: When the commit was made.
      cl: The commit's CL.

  Returns:
      The build's directory: the runner, its libraries and build.json.
  """
  directory = kept / commit
  directory.mkdir(parents=True)
  (directory / 'runner').write_text(runner)
  (directory / 'runner').chmod(0o755)
  for library in LIBRARIES:
    (directory / library).write_bytes(
        f'{library} of LiteRT-LM {commit}'.encode()
    )
  sha256 = {
      path.name: hashlib.sha256(path.read_bytes()).hexdigest()
      for path in sorted(directory.iterdir())
  }
  (directory / 'build.json').write_text(
      json.dumps({
          'commit': commit,
          'commit_date': commit_date,
          'cl': cl,
          'built_at': '2026-10-15T14:52:06Z',
          'sha256': sha256,
      })
  )
  return directory
