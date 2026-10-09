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
runtime_root="${RUNNER_TEST_RUNTIME_ROOT:-}"
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != --runtime-root ]]; then
    echo 'Usage: ci/test_python_runner.sh [--runtime-root PATH]' >&2
    exit 2
  fi
  runtime_root="$2"
fi

runner_python="${RUNNER_PYTHON:-python3}"
packages=(
  "${repo_root}/src/runner/python/package"
  "${repo_root}/src/contracts/python/package"
  "${repo_root}/src/capture/pytorch/package"
  "${repo_root}/src/runner/python/tests"
)
package_paths="$(IFS=:; echo "${packages[*]}")"
if [[ -n "${runtime_root}" ]]; then
  runtime_root="$(cd "${runtime_root}" && pwd -P)"
  runner_python="${runtime_root}/.venv/bin/python"
  if [[ ! -x "${runner_python}" ]]; then
    echo 'Runtime root must contain executable .venv/bin/python' >&2
    exit 2
  fi
  export RUNNER_TEST_RUNTIME_ROOT="${runtime_root}"
  package_paths="${package_paths}:${runtime_root}"
fi

export PYTHONPATH="${package_paths}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
cd "${repo_root}"

"${runner_python}" -m unittest discover \
  -s src/runner/python/tests -p test_package_boundary.py -v
if [[ -n "${runtime_root}" ]]; then
  "${runner_python}" -m unittest discover -s src/capture/pytorch/tests -v
  "${runner_python}" - <<'PY'
import json
import sys
import unittest
import torch
import transformers
import tokenizers
import safetensors
import ai_edge_debugger_pytorch
from runtime_fixture import configured_runtime_root

configured_runtime_root()
suite = unittest.defaultTestLoader.loadTestsFromNames([
    'test_pytorch_worker',
    'test_pytorch_resident',
    'test_python_runner_cancellation',
])
result = unittest.TextTestRunner(verbosity=2).run(suite)
print(
    json.dumps(
        dict(
            runtime_tests=result.testsRun,
            skipped=len(result.skipped),
            failures=len(result.failures),
            errors=len(result.errors),
            torch=torch.__version__,
            transformers=transformers.__version__,
            backend='CPU',
        )
    )
)
success = (
    result.wasSuccessful() and result.testsRun == 31 and not result.skipped
)
sys.exit(0 if success else 1)
PY
else
  if "${runner_python}" -c 'import torch, safetensors.torch' >/dev/null 2>&1; then
    "${runner_python}" -m unittest discover -s src/capture/pytorch/tests -v
  else
    echo 'Skipping src/capture/pytorch/tests in lightweight mode (torch not installed in' "${runner_python}" ').'
  fi
  "${runner_python}" -m unittest \
    test_pytorch_worker.PyTorchWorkerTest \
    test_python_runner_cancellation -v
  echo 'Lightweight checks passed. Supply --runtime-root' \
    'to also require the 21 real tiny-model tests.'
fi
