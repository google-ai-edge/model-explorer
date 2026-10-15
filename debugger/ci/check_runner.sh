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

# The runner's check: clang-format, swift-format, ktfmt (Google style, for the
# Android App's Kotlin) and buildifier (for Starlark), the last two fetched by
# Bazel; then bazel test on the CPU and on the GPU, which also builds the macOS
# App, then the iOS App for the simulator, and the Android App where
# ANDROID_HOME and ANDROID_NDK_HOME name an SDK and an NDK. The tests load a
# real model; a machine without one cannot say the build is green. The first
# run fetches and builds LiteRT-LM, for the machine and for the simulator,
# which takes a long time.
#
# Usage: RUNNER_MODEL=/path/to/model.litertlm ci/check_runner.sh [SANITIZER]
#   SANITIZER  thread or address.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly ROOT

err() {
  echo "check_runner: $*" >&2
}

#######################################
# Prints the clang-format to use: the one on PATH, or Xcode's.
#######################################
clang_format() {
  if command -v clang-format > /dev/null; then
    command -v clang-format
  elif command -v xcrun > /dev/null \
    && xcrun --find clang-format > /dev/null 2>&1; then
    xcrun --find clang-format
  else
    err "clang-format is not installed"
    return 1
  fi
}

main() {
  local sanitizer="${1:-}"
  case "${sanitizer}" in
    "" | thread | address) ;;
    *)
      err "unknown sanitizer '${sanitizer}' (thread or address)"
      exit 2
      ;;
  esac
  if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    err "the runner is built for Apple Silicon macOS only, for now"
    exit 1
  fi
  if [[ -z "${RUNNER_MODEL:-}" || ! -f "${RUNNER_MODEL}" ]]; then
    err "RUNNER_MODEL must name a .litertlm file:" \
      "the runner's tests load a model"
    exit 1
  fi
  # Absolute: the tests run in Bazel's directories, not in this one.
  local model_dir
  model_dir="$(cd "$(dirname "${RUNNER_MODEL}")" && pwd -P)"
  RUNNER_MODEL="${model_dir}/$(basename "${RUNNER_MODEL}")"
  export RUNNER_MODEL
  cd "${ROOT}/src/runner"

  local format
  format="$(clang_format)"
  find model_debugger -type f \( -name '*.h' -o -name '*.cc' \) \
    -print0 | xargs -0 "${format}" --dry-run --Werror
  # Xcode's swift-format, with the Google style in .swift-format.
  find model_debugger -type f -name '*.swift' -print0 \
    | xargs -0 xcrun swift-format lint --strict --configuration .swift-format

  # bazelisk reads .bazelversion; a bare bazel of another version refuses it.
  local bazel
  if ! bazel="$(command -v bazelisk || command -v bazel)"; then
    err "bazelisk is not installed (brew install bazelisk)"
    exit 1
  fi
  # The Android App's Kotlin, in Google style: ktfmt on Bazel's own Java, which
  # runs it from elsewhere, so the files go by absolute path.
  find "${PWD}/model_debugger" -type f -name '*.kt' -print0 \
    | xargs -0 "${bazel}" run //model_debugger/runner/app/android:ktfmt -- \
      --dry-run --set-exit-if-changed
  # Starlark (MODULE.bazel, BUILD files, .bzl and the BUILD templates):
  # buildifier's format and its lint, prebuilt and fetched by Bazel.
  find "${PWD}" -type f \( -name MODULE.bazel -o -name BUILD.bazel \
    -o -name '*.BUILD' -o -name '*.bzl' \) -print0 \
    | xargs -0 "${bazel}" run @buildifier_prebuilt//:buildifier -- \
      --mode=check --lint=warn

  # ${sanitizer:+...} expands to nothing when there is none; bash 3.2, which
  # macOS ships, would call an empty array unbound.
  "${bazel}" test //... ${sanitizer:+--config="${sanitizer}"}
  # The iOS App is not in //...: it is built for the simulator, on request.
  "${bazel}" build //model_debugger/runner/app/apple:ios \
    --ios_multi_cpus=sim_arm64
  # Nor is the Android App, which needs the Android SDK and NDK.
  if [[ -n "${ANDROID_HOME:-}" && -n "${ANDROID_NDK_HOME:-}" ]]; then
    "${bazel}" build --config=android_app \
      //model_debugger/runner/app/android:runner
  else
    echo "check_runner: skipping the Android App:" \
      "ANDROID_HOME and ANDROID_NDK_HOME are not set"
  fi
}

main "$@"
