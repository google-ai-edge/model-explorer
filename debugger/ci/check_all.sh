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

# Every check: these scripts, the runner, the PyTorch runner, the server, the
# UI. The runner comes first: the server's tests run the runner it builds,
# with RUNNER_MODEL. The protocol has no check of its own; every part
# compiles its .proto.
#
# Usage: RUNNER_MODEL=/path/to/model.litertlm
#   [RUNNER_PYTORCH_MODEL=/path/to/gemma-4-E2B-it] ci/check_all.sh [SANITIZER]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly ROOT

main() {
  if ! command -v shellcheck > /dev/null; then
    echo "check_all: shellcheck is not installed" >&2
    exit 1
  fi
  shellcheck "${ROOT}"/ci/*.sh
  "${ROOT}/ci/check_runner.sh" "$@"
  "${ROOT}/ci/check_runner_pytorch.sh"
  "${ROOT}/ci/check_server.sh"
  "${ROOT}/ci/check_ui.sh"
}

main "$@"
