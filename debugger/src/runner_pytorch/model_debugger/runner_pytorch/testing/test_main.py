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

"""The main of every test module: pytest's, with a HOME of its own.

Usage, at the end of a test module:
  if __name__ == '__main__':
    test_main.main()

or, for a test module without a main of its own (pytorch_collector/tests),
with the path of the module to run.
"""

import os
import sys

import pytest


def main(module_file: str | None = None) -> None:
  """Runs the tests of a module, and exits with pytest.

  Args:
      module_file: The path of the test module; by default, the module run as
        __main__.
  """
  tmp = os.environ.get('TEST_TMPDIR')
  # Bazel sets no HOME, under which the runner keeps its cache: each test
  # gets its own, the temporary directory, as Bazel's test protocol advises.
  if 'HOME' not in os.environ and tmp:
    os.environ['HOME'] = tmp
  args = [
      module_file or sys.modules['__main__'].__file__,
      '-ra',
      '--import-mode=importlib',
      '-p',
      'no:cacheprovider',
  ]
  if tmp:
    args.append(f'--basetemp={tmp}/pytest')
  if xml := os.environ.get('XML_OUTPUT_FILE'):
    args.append(f'--junitxml={xml}')
  sys.exit(pytest.main(args + sys.argv[1:]))
