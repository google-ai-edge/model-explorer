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

# The UI's check: install its packages once, then Prettier's format over
# everything it formats in src/ui (all but what src/ui/.prettierignore names),
# the UI's rules (src/ui/scripts/check_layering.mjs: the layers and what each
# may import, that only src/server/ sends requests, that details bodies draw
# no chart, where CSS custom properties are set), the production build, the
# unit tests (ng test: every src/**/*.spec.ts, vitest over jsdom), and the
# structure-graph prototype's checks, which test public/graph-arch-preview/
# with node itself. Each fails the check.
#
# Usage: ci/check_ui.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly ROOT

main() {
  cd "${ROOT}/src/ui"
  if ! command -v npm > /dev/null; then
    echo "check_ui: npm is not installed; see src/ui/README.md" >&2
    exit 1
  fi
  if [[ ! -d node_modules ]]; then
    npm ci --no-audit --no-fund
  fi
  npm run format:check
  npm run check:layering
  npm run build
  npm test
  local check
  for check in graph_pairs mapping_pair_edit model_tree_metrics; do
    if [[ -f "scripts/check_${check}.mjs" ]]; then
      node "scripts/check_${check}.mjs"
    fi
  done
}

main "$@"
