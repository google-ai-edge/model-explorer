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

"""The main of every test module: absltest's, with a HOME of its own.

Usage, at the end of a test module:
  if __name__ == '__main__':
    test_main.main()
"""

import os

from absl.testing import absltest


def main() -> None:
  """Runs the tests of the module run as __main__."""
  # Bazel sets no HOME, under which the real runner keeps its cache: each test
  # gets its own, the temporary directory, as Bazel's test protocol advises.
  if 'HOME' not in os.environ and (tmp := os.environ.get('TEST_TMPDIR')):
    os.environ['HOME'] = tmp
  absltest.main()
