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

# The server's check: buildifier (for Starlark) and ruff (check and format,
# for the server's Python and that of ci/), both at the versions Bazel
# fetches; that the package's dependencies only go downward, util <
# architectures < tap < graph < capture < analysis < runner < session <
# service < server (the facade) < cli
# (ci/check_server_layers.py, which holds the order); that the architectures'
# skeletons are what their definitions make of the pinned configs (the
# generator's --check); that the schemas tap/
# reads models with, and the commit it tells a phone's App from, are the
# Runner's LiteRT-LM commit (ci/check_schema_pins.py); that the UI's metric
# catalog fixture is the server's answer to GET /api/metrics
# (ci/check_metric_catalog.py, which builds the server and asks it; its
# --write writes the fixture); then bazel test, which also builds every
# target and checks that requirements_lock.txt still matches the
# requirements (which needs the network); last, the wheel for PyPI, which
# carries the built UI: it builds the UI in src/ui (npm), then the wheel,
# and checks that the wheel holds the UI's index.html. The local launcher's
# tests run the real runner when it is built; the end-to-end test also needs
# RUNNER_MODEL, a model. The tests that read
# the runner, the model or REAL_SESSIONS run every time, not from Bazel's
# cache; see src/server/README.md.
#
# Usage: [RUNNER_MODEL=/path/to/model.litertlm] [RUNNER_BINARY=/path/to/runner]
#   [REAL_SESSIONS=/path/to/sessions] ci/check_server.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly ROOT

err() {
  echo "check_server: $*" >&2
}

# Makes the variable named $1 absolute, if it is set; $2 says what it must
# name: "file" or "directory". The tests run in Bazel's directories, not in
# this one, so a relative path would find nothing and the tests would skip.
make_absolute() {
  local var="$1"
  local kind="$2"
  local path="${!var:-}"
  if [[ -z "${path}" ]]; then
    return
  fi
  if [[ "${kind}" == file && ! -f "${path}" ]] \
    || [[ "${kind}" == directory && ! -d "${path}" ]]; then
    err "${var} names no ${kind}: ${path}"
    exit 1
  fi
  local dir
  dir="$(cd "$(dirname "${path}")" && pwd -P)"
  export "${var}=${dir}/$(basename "${path}")"
}

main() {
  make_absolute RUNNER_MODEL file
  make_absolute RUNNER_BINARY file
  make_absolute REAL_SESSIONS directory
  cd "${ROOT}/src/server"

  # bazelisk reads .bazelversion; a bare bazel of another version refuses it.
  local bazel
  if ! bazel="$(command -v bazelisk || command -v bazel)"; then
    err "bazelisk is not installed (brew install bazelisk)"
    exit 1
  fi
  # Starlark (MODULE.bazel, BUILD files and .bzl): buildifier's format and its
  # lint, prebuilt and fetched by Bazel, which runs it from elsewhere, so the
  # files go by absolute path.
  find "${PWD}" -type f \( -name MODULE.bazel -o -name BUILD.bazel \
    -o -name '*.bzl' \) -print0 \
    | xargs -0 "${bazel}" run @buildifier_prebuilt//:buildifier -- \
      --mode=check --lint=warn
  # Python: ruff from the PyPI wheel requirements_lock.txt pins, with the
  # settings in ruff.toml, for the server and for the checks in ci/.
  local python_files=("${PWD}/model_debugger" "${ROOT}"/ci/*.py)
  "${bazel}" run //:ruff -- check --config="${PWD}/ruff.toml" \
    "${python_files[@]}"
  "${bazel}" run //:ruff -- format --check --config="${PWD}/ruff.toml" \
    "${python_files[@]}"
  # The checks in ci/ need Python's standard library alone, and none of the
  # environment: python3 -I.
  if ! command -v python3 > /dev/null; then
    err "python3 is not installed"
    exit 1
  fi
  # A module imports only from its own layer and those below it; only tests
  # import testing.
  python3 -I "${ROOT}/ci/check_server_layers.py" "${PWD}/model_debugger/server"
  # Each skeleton file is byte for byte what its architecture's definition
  # makes of its pinned config.json; without --check, the generator writes
  # them.
  "${bazel}" run //model_debugger/server/architectures:generate -- --check
  # tap/ reads models with the schemas of the Runner's LiteRT-LM and LiteRT:
  # the server's pins against the Runner's commit, and LiteRT-LM's WORKSPACE
  # at it, which Bazel fetches; and app_version.py's commit against it too.
  "${bazel}" build @litertlm_workspace//file
  local workspace_file
  workspace_file="$("${bazel}" info output_base)/$("${bazel}" cquery \
    --output=files @litertlm_workspace//file 2> /dev/null)"
  python3 -I "${ROOT}/ci/check_schema_pins.py" \
    "${ROOT}/src/runner/MODULE.bazel" "${PWD}/MODULE.bazel" \
    "${workspace_file}" \
    "${PWD}/model_debugger/server/runner/app_version.py"
  # The UI's tests describe metrics as the server does.
  python3 -I "${ROOT}/ci/check_metric_catalog.py" \
    "${ROOT}/src/ui/src/server/testing/metric_catalog_fixture.json"

  "${bazel}" test //...

  # The wheel carries the UI as npm builds it in src/ui; //... leaves the
  # wheel out, since the server's own build and tests need no UI.
  if ! command -v npm > /dev/null; then
    err "npm is not installed; see src/ui/README.md"
    exit 1
  fi
  (
    cd "${ROOT}/src/ui"
    if [[ ! -d node_modules ]]; then
      npm ci --no-audit --no-fund
    fi
    npm run build
  )
  "${bazel}" build //:wheel
  local wheel
  wheel="$("${bazel}" cquery --output=files //:wheel 2> /dev/null)"
  local page=model_debugger/server/ui/index.html
  local holds='import sys, zipfile
sys.exit(sys.argv[2] not in zipfile.ZipFile(sys.argv[1]).namelist())'
  if ! python3 -I -c "${holds}" "${wheel}" "${page}"; then
    err "the wheel ${wheel} holds no ${page}"
    exit 1
  fi
}

main "$@"
