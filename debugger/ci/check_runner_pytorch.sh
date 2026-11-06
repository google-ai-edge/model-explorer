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

# The PyTorch runner's check: buildifier (for Starlark) and ruff (check and
# format, with the server's ruff.toml), both at the versions Bazel fetches;
# then bazel test, which also builds every target, the runner command and the
# wheel among them. The real model's test also needs RUNNER_PYTORCH_MODEL, a
# Gemma 4 model directory such as E2B's Hugging Face snapshot; without it, it
# is skipped. The tests that read the model or the server's capture index run
# every time, not from Bazel's cache; see src/runner_pytorch/README.md.
#
# Usage: [RUNNER_PYTORCH_MODEL=/path/to/gemma-4-E2B-it]
#   ci/check_runner_pytorch.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly ROOT

err() {
  echo "check_runner_pytorch: $*" >&2
}

main() {
  # The tests run in Bazel's directories, not in this one, so a relative path
  # would find nothing and the real model's test would skip.
  if [[ -n "${RUNNER_PYTORCH_MODEL:-}" ]]; then
    if [[ ! -d "${RUNNER_PYTORCH_MODEL}" ]]; then
      err "RUNNER_PYTORCH_MODEL names no directory: ${RUNNER_PYTORCH_MODEL}"
      exit 1
    fi
    RUNNER_PYTORCH_MODEL="$(cd "${RUNNER_PYTORCH_MODEL}" && pwd -P)"
    export RUNNER_PYTORCH_MODEL
  fi
  cd "${ROOT}/src/runner_pytorch"

  # bazelisk reads .bazelversion; a bare bazel of another version refuses it.
  local bazel
  if ! bazel="$(command -v bazelisk || command -v bazel)"; then
    err "bazelisk is not installed (brew install bazelisk)"
    exit 1
  fi
  # Starlark (MODULE.bazel and BUILD files): buildifier's format and its lint,
  # prebuilt and fetched by Bazel, which runs it from elsewhere, so the files
  # go by absolute path.
  find "${PWD}" -type f \( -name MODULE.bazel -o -name BUILD.bazel \
    -o -name '*.bzl' \) -print0 \
    | xargs -0 "${bazel}" run @buildifier_prebuilt//:buildifier -- \
      --mode=check --lint=warn
  # Python: ruff from the PyPI wheel requirements_lock.txt pins, with the
  # server's settings, src/server/ruff.toml. The runner's code alone:
  # pytorch_collector/, the capture library moved here, keeps its own style.
  local config="${ROOT}/src/server/ruff.toml"
  "${bazel}" run //:ruff -- check --config="${config}" "${PWD}/model_debugger"
  "${bazel}" run //:ruff -- format --check --config="${config}" \
    "${PWD}/model_debugger"

  "${bazel}" test //...
}

main "$@"
