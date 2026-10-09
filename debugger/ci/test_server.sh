#!/usr/bin/env bash
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

set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_root"
python_bin="${SERVER_PYTHON:-python3}"
# Editable installs keep the venv pointed at this checkout (README documents the same setup).
"$python_bin" -m pip install --quiet -e ./src/contracts/python -e ./src/runner/python -e './src/server[apple,test]'
# Runner-owned real-model fixtures are imported from src/runner/python/tests; the Runner
# suite itself is owned by ci/test_python_runner.sh and is not repeated here.
export PYTHONPATH="src/server/package:src/runner/python/tests"
export PYTHONDONTWRITEBYTECODE=1
"$python_bin" -m unittest discover -s test -v
