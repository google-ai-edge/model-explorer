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

# Swift host checks. check_runner.sh links against the vendored native libraries
# (src/runner/apple/Vendor/macos_arm64 or LITERT_LM_LIBS); check_macos.sh also needs
# LITERT_LM_LIBS or LITERT_LM_ROOT for the runtime build.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
tools="$repo_root/src/runner/apple/Tools"
vendor="$repo_root/src/runner/apple/Vendor/macos_arm64"
if [[ -z "${LITERT_LM_LIBS:-}" && ! -d "$vendor" ]]; then
  echo "check_apple: native libraries missing. Set LITERT_LM_LIBS or provide $vendor (see src/runner/apple/README.md)." >&2
  exit 2
fi
bash "$tools/check_runner.sh"
if [[ -n "${LITERT_LM_LIBS:-}" || -n "${LITERT_LM_ROOT:-}" ]]; then
  bash "$tools/check_macos.sh"
else
  echo 'check_apple: check_macos.sh skipped (set LITERT_LM_LIBS or LITERT_LM_ROOT to run native validation).'
fi
