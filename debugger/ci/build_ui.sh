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
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root/src/ui"
npm ci --no-audit --no-fund
npm run verify
# Browser smoke against the built UI served by the Python Server (examples/gemma4-e2b).
server_python="${SERVER_PYTHON:-$repo_root/src/server/.venv/bin/python}"
if [[ -x "$server_python" ]]; then
  SERVER_PYTHON="$server_python" npm run test:e2e
else
  echo "build_ui: test:e2e skipped; create src/server/.venv (see README) to run the browser smoke." >&2
  exit 2
fi
