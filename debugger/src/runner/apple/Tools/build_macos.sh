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
# ==============================================================================

set -euo pipefail
runner_root="$(cd "$(dirname "$0")/.." && pwd -P)"
: "${LITERT_LM_ROOT:?Set LITERT_LM_ROOT to an explicit debugger-enabled runtime checkout}"
runtime_root="$(cd "$LITERT_LM_ROOT" && pwd -P)"
runner_derived_data="${RUNNER_DERIVED_DATA:-$runner_root/.build/mac}"
runner_native_args=()
if [[ -f "$runner_root/.build/mac-native/liblitert-lm.dylib" ]]; then
  runner_native_args=(--native-library "$runner_root/.build/mac-native/liblitert-lm.dylib")
fi
python3 "$runner_root/Tools/stage_macos.py" --runtime-root "$runtime_root" ${runner_native_args[@]+"${runner_native_args[@]}"}
xcrun swift "$runner_root/Tools/sync_icons.swift" "$runner_root"
python3 "$runner_root/Tools/generate_project.py"
xcodebuild -project "$runner_root/ModelDebuggerRunner.xcodeproj" -scheme ModelDebuggerMac \
  -configuration Debug -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$runner_derived_data" build
codesign --verify --deep --strict "$runner_derived_data/Build/Products/Debug/ModelDebuggerMac.app"
echo "$runner_derived_data/Build/Products/Debug/ModelDebuggerMac.app"
