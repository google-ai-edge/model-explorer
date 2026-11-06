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

"""The main of each of pytorch-collector's tests, which have none of their own.

The test files are as the library has them, pytest modules without a main:
each py_test (BUILD.bazel) runs this with the test file's path first, and
it runs that file as testing/test_main.py runs the runner's own.
"""

import sys

from model_debugger.runner_pytorch.testing import test_main

if __name__ == '__main__':
  test_main.main(sys.argv.pop(1))
