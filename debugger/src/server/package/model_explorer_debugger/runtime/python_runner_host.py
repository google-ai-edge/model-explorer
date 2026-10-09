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

"""CLI launch bridge for installed Apps still on the version-1 module name."""

from pathlib import Path
import sys


def main():
  # Existing Apps pass only server/package on PYTHONPATH. Resolve the two
  # canonical source packages here; the Server never imports Runner execution.
  source = Path(__file__).resolve().parents[4]
  packages = (
      source / 'runner/python/package',
      source / 'contracts/python/package',
  )
  if (packages[0] / 'model_debugger_runner/__init__.py').is_file() and (
      packages[1] / 'model_debugger_contracts/__init__.py'
  ).is_file():
    sys.path[:0] = [str(path) for path in packages if str(path) not in sys.path]

  from model_debugger_runner.python_runner_host import main as run

  run()


if __name__ == '__main__':
  main()
