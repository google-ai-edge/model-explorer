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

"""Pick the real test runtime explicitly; never infer it from a package link."""

import os
from pathlib import Path
import sys
import unittest


def configured_runtime_root():
  value = os.environ.get('RUNNER_TEST_RUNTIME_ROOT')
  if not value:
    raise unittest.SkipTest(
        'Set RUNNER_TEST_RUNTIME_ROOT to run real model tests'
    )
  root = Path(value).expanduser().resolve()
  environment = root / '.venv'
  if not (environment / 'bin/python').is_file():
    raise RuntimeError('RUNNER_TEST_RUNTIME_ROOT must contain .venv/bin/python')
  if Path(sys.prefix).resolve() != environment.resolve():
    raise RuntimeError(
        'Run the parent tests with RUNNER_TEST_RUNTIME_ROOT/.venv/bin/python so'
        ' parent and worker use the same environment'
    )
  return root
