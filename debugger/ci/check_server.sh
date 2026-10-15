#!/bin/bash
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

# The server's check: compile the protocol, then ruff, then pytest.
#
# Usage: ci/check_server.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly ROOT

main() {
  cd "${ROOT}/src/server"
  local python="${ROOT}/src/server/.venv/bin/python"
  if [[ ! -x "${python}" ]]; then
    echo "check_server: no .venv; see src/server/README.md" >&2
    exit 1
  fi
  # The protocol's Python messages and stubs, as model_debugger.v1; not
  # committed, so made here and by the README's build step.
  "${python}" -m grpc_tools.protoc -I "${ROOT}/src/protocol" --python_out=src \
    --grpc_python_out=src --pyi_out=src model_debugger/v1/runner.proto
  "${python}" -m ruff check src tests
  "${python}" -m pytest -q tests
}

main "$@"
