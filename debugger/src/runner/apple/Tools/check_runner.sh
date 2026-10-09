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
native_libs="${LITERT_LM_LIBS:-$runner_root/Vendor/macos_arm64}"
runner_build_root="${RUNNER_BUILD_ROOT:-$runner_root/.build}"
runner_module_cache="${RUNNER_MODULE_CACHE:-$runner_build_root/module-cache}"
mkdir -p "$runner_build_root"
runner_sources=()
for runner_file in "$runner_root"/Runner/*.swift; do
  case "$(basename "$runner_file")" in RunnerApp.swift|RunnerView.swift|RunnerWebView.swift) continue ;; esac
  runner_sources+=("$runner_file")
done
xcrun swiftc -swift-version 5 -I "$runner_root/CLiteRTLM" \
  -module-cache-path "$runner_module_cache" \
  "$runner_root"/Core/*.swift "${runner_sources[@]}" "$runner_root/Tests/RunnerLifecycleTests.swift" \
  -L "$native_libs" -llitert-lm -Xlinker -rpath -Xlinker "$native_libs" \
  -o "$runner_build_root/runner-lifecycle-tests"
install_name_tool -change bazel-out/darwin_arm64-opt/bin/c/liblitert-lm.dylib \
  @rpath/liblitert-lm.dylib "$runner_build_root/runner-lifecycle-tests"
codesign --force --sign - "$runner_build_root/runner-lifecycle-tests"
"$runner_build_root/runner-lifecycle-tests"
