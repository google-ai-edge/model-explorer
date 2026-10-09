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
if [[ -n "${LITERT_LM_LIBS:-}" ]]; then
  native_libs="$(cd "$LITERT_LM_LIBS" && pwd -P)"
else
  : "${LITERT_LM_ROOT:?Set LITERT_LM_LIBS or LITERT_LM_ROOT explicitly}"
  runtime_root="$(cd "$LITERT_LM_ROOT" && pwd -P)"
  native_libs="$runtime_root/artifacts/python/litert_lm"
fi
runner_build_root="${RUNNER_BUILD_ROOT:-$runner_root/.build}"
runner_module_cache="${RUNNER_MODULE_CACHE:-$runner_build_root/module-cache}"
mkdir -p "$runner_build_root"
runner_test=ValidationTests
runner_name=validation-tests
if [[ "${1:-}" == --smoke || "${1:-}" == --persistent-smoke || "${1:-}" == --chat-lifecycle-smoke ]]; then
  runner_mode="$1"
  shift
  [[ $# == 3 ]] || { echo 'Usage: check_macos.sh --smoke job.json model.litertlm NEW_OUTPUT_DIRECTORY'; exit 1; }
  runner_test=NativeSmoke
  runner_name=native-smoke
  if [[ "$runner_mode" == --persistent-smoke ]]; then
    runner_test=PersistentSmoke
    runner_name=persistent-smoke
  fi
  if [[ "$runner_mode" == --chat-lifecycle-smoke ]]; then
    runner_test=ChatLifecycleSmoke
    runner_name=chat-lifecycle-smoke
  fi
fi
xcrun swiftc -swift-version 5 -I "$runner_root/CLiteRTLM" \
  -module-cache-path "$runner_module_cache" \
  "$runner_root"/Core/*.swift "$runner_root/Tests/$runner_test.swift" \
  -L "$native_libs" -llitert-lm -Xlinker -rpath -Xlinker "$native_libs" \
  -o "$runner_build_root/$runner_name"
install_name_tool -change bazel-out/darwin_arm64-opt/bin/c/liblitert-lm.dylib \
  @rpath/liblitert-lm.dylib "$runner_build_root/$runner_name"
codesign --force --sign - "$runner_build_root/$runner_name"
"$runner_build_root/$runner_name" "$@"
if [[ "$runner_test" == ValidationTests ]]; then
  xcrun swiftc -swift-version 5 -module-cache-path "$runner_module_cache" \
    "$runner_root/Core/NativeCaptureStore.swift" "$runner_root/Tests/NativeCaptureStoreTests.swift" \
    -o "$runner_build_root/capture-store-tests"
  "$runner_build_root/capture-store-tests"
fi
