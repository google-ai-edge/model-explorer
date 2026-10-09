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

# Environment-variable wrapper over `model_explorer_debugger.server`. Unset variables
# leave the Server's own defaults in place: the checkout workspace, its built UI,
# port 8080 and the sibling debugger_runtime checkout (or the remembered root).
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="$repo_dir/src/server/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo 'Create src/server/.venv and install ./src/contracts/python ./src/server first (see src/server/README.md).' >&2
  exit 1
fi
args=()
[[ -n "${MODEL_DEBUGGER_WORKSPACE:-}" ]] && args+=(--workspace "$MODEL_DEBUGGER_WORKSPACE")
[[ -n "${LITERT_LM_ROOT:-}" ]] && args+=(--runtime-root "$LITERT_LM_ROOT")
[[ -n "${MODEL_DEBUGGER_PORT:-}" ]] && args+=(--port "$MODEL_DEBUGGER_PORT")
[[ -n "${MODEL_DEBUGGER_UI_ROOT:-}" ]] && args+=(--ui "$MODEL_DEBUGGER_UI_ROOT")
cd "$repo_dir"
exec "$python_bin" -m model_explorer_debugger.server ${args[@]+"${args[@]}"} "$@"
