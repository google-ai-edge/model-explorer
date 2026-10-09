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
native_root="$runtime_root/third_party/LiteRT-LM"
bazel_bin="${BAZEL_BIN:-$runtime_root/.tools/bazel-7.6.1}"
[[ "$(git -C "$native_root" rev-parse HEAD)" == a0f1e97361155c8bb9cd3d6aa045c04320d4334b ]] || { echo 'Unexpected LiteRT-LM revision'; exit 1; }
for header in engine.h conversation.h experimental.h; do
  cmp "$runner_root/CLiteRTLM/$header" "$native_root/c/$header"
done
cd "$native_root"
"$bazel_bin" build --config=macos_arm64 --macos_minimum_os=14.0 \
  --define=LITERT_LM_DEBUGGER_ENABLED=1 --define=litert_runtime_link_mode=dynamic \
  --jobs="${BAZEL_JOBS:-6}" //c:litert-lm
mkdir -p "$runner_root/.build/mac-native"
if [[ -e "$runner_root/.build/mac-native/liblitert-lm.dylib" ]]; then
  chmod u+w "$runner_root/.build/mac-native/liblitert-lm.dylib"
fi
cp bazel-bin/c/liblitert-lm.dylib "$runner_root/.build/mac-native/"
python3 "$runner_root/Tools/stage_macos.py" --runtime-root "$runtime_root" \
  --native-library "$runner_root/.build/mac-native/liblitert-lm.dylib"
