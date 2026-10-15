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

# The UI's check: install its packages once, then the production build, the
# node regression scripts and the Angular unit tests. The Playwright specs
# need a running server and are not part of it.
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
  npm run build
  npm test
  npm run test:unit
}

main "$@"
