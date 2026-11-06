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

# Keeps the command-line Runner Bazel has just built, under the LiteRT-LM
# commit it was built against, where the Model Debugger server finds it:
#
#   ~/Library/Application Support/model_debugger/runners/macos_arm64/<commit>/
#
# The directory holds the runner, the LiteRT-LM library, the two libraries it
# links and its three GPU plugins, which the runner finds beside itself, and
# build.json: the commit, its date and CL, when it was kept, and the SHA-256 of
# each file. The new Runner is checked first: started from its own directory
# with a HOME of its own, it must answer GetInfo with that commit. A commit
# kept again is replaced by an atomic swap, so that a reader sees the old
# directory or the new one, never a part of either.
#
# Usage: bazel run //model_debugger/runner/cli:keep
#        bazel run --repo_env=LITERT_LM_COMMIT_MACOS_ARM64=<commit> \
#          //model_debugger/runner/cli:keep
# Bazel passes the arguments: RUNNER GET_INFO RUNTIME_FILES...

set -euo pipefail

readonly PLATFORM=macos_arm64
# The renamex_np flag that swaps two paths (<stdio.h>).
readonly RENAME_SWAP=2

# What the exit trap cleans up: the directory being assembled (after a swap,
# the old build), the checked Runner's HOME, and its process.
staging=""
test_home=""
runner_pid=""

err() {
  echo "keep: $*" >&2
}

cleanup() {
  if [[ -n "${runner_pid}" ]]; then
    kill "${runner_pid}" 2> /dev/null || true
    wait "${runner_pid}" 2> /dev/null || true
  fi
  if [[ -n "${test_home}" ]]; then
    rm -rf "${test_home}"
  fi
  if [[ -n "${staging}" ]]; then
    rm -rf "${staging}"
  fi
}

#######################################
# Starts the runner of a directory, with a HOME of its own, and checks that it
# answers GetInfo with the commit; stops it afterwards.
# Arguments:
#   The directory, the get_info binary, the commit.
#######################################
check_runner() {
  local dir="$1"
  local get_info="$2"
  local commit="$3"
  test_home="$(mktemp -d)"
  HOME="${test_home}" "${dir}/runner" --listen=127.0.0.1:0 \
    > "${test_home}/runner.log" 2>&1 &
  runner_pid=$!
  local port=""
  local i
  for ((i = 0; i < 600; i++)); do
    port="$(sed -n 's/^runner: listening on port \([0-9]*\)$/\1/p' \
      "${test_home}/runner.log")"
    if [[ -n "${port}" ]] || ! kill -0 "${runner_pid}" 2> /dev/null; then
      break
    fi
    sleep 0.1
  done
  if [[ -z "${port}" ]]; then
    err "the new runner did not start:"
    cat "${test_home}/runner.log" >&2
    return 1
  fi
  local info
  info="$("${get_info}" --address="127.0.0.1:${port}")"
  local version
  version="$(jq -r '.runtime.version' <<< "${info}")"
  if [[ "${version}" != "${commit}" ]]; then
    err "the new runner reports LiteRT-LM '${version}', not ${commit}"
    return 1
  fi
  kill "${runner_pid}"
  wait "${runner_pid}" 2> /dev/null || true
  runner_pid=""
  echo "keep: the new runner started and reported LiteRT-LM ${version}"
}

#######################################
# Prints a JSON object from the name of each file of a directory to its
# SHA-256.
# Arguments:
#   The directory.
#######################################
sha256_of_files() {
  local sums='{}'
  local file
  for file in "$1"/*; do
    sums="$(jq --arg name "$(basename "${file}")" \
      --arg sum "$(shasum -a 256 "${file}" | cut -d ' ' -f 1)" \
      '. + {($name): $sum}' <<< "${sums}")"
  done
  echo "${sums}"
}

#######################################
# Puts a directory at a path in one rename: if something is there, the two
# are swapped, and what was there is left at the directory's old path.
# Arguments:
#   The directory, the path.
#######################################
swap_into_place() {
  local from="$1"
  local to="$2"
  if [[ ! -e "${to}" ]]; then
    mv "${from}" "${to}"
    return
  fi
  # macOS's renamex_np(RENAME_SWAP), which no command offers.
  python3 -c '
import ctypes, os, sys
libc = ctypes.CDLL(None, use_errno=True)
if libc.renamex_np(sys.argv[1].encode(), sys.argv[2].encode(), int(sys.argv[3])):
    sys.exit("keep: renamex_np: " + os.strerror(ctypes.get_errno()))
' "${from}" "${to}" "${RENAME_SWAP}"
}

main() {
  if (($# < 3)); then
    err "run it with bazel run //model_debugger/runner/cli:keep"
    exit 2
  fi
  local runner="$1"
  local get_info="$2"
  shift 2
  local commit_json=""
  local -a libraries=()
  local file
  for file in "$@"; do
    if [[ "$(basename "${file}")" == commit.json ]]; then
      commit_json="${file}"
    else
      libraries+=("${file}")
    fi
  done
  if [[ -z "${commit_json}" ]]; then
    err "Bazel passed no commit.json"
    exit 2
  fi
  local commit
  commit="$(jq -r '.commit' "${commit_json}")"
  if [[ ! "${commit}" =~ ^[0-9a-f]{40}$ ]]; then
    err "commit.json names no whole commit: '${commit}'"
    exit 1
  fi
  if ! otool -l "${runner}" | grep -q 'path @loader_path (offset'; then
    err "the runner does not look for its libraries beside itself" \
      "(no @loader_path rpath)"
    exit 1
  fi

  local root="${HOME}/Library/Application Support/model_debugger/runners/${PLATFORM}"
  mkdir -p "${root}"
  trap cleanup EXIT
  # Assembled beside the kept builds, so that one rename puts it in place.
  staging="$(mktemp -d "${root}/.keep.XXXXXX")"
  chmod 755 "${staging}"
  cp "${runner}" "${staging}/runner"
  cp "${libraries[@]}" "${staging}/"
  check_runner "${staging}" "${get_info}" "${commit}"

  local sums
  sums="$(sha256_of_files "${staging}")"
  jq --arg built_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --argjson sha256 "${sums}" \
    '{commit, commit_date, cl, built_at: $built_at, sha256: $sha256}' \
    "${commit_json}" > "${staging}/build.json"
  swap_into_place "${staging}" "${root}/${commit}"
  if [[ ! -e "${staging}" ]]; then
    staging=""
  fi
  echo "keep: kept the runner for LiteRT-LM ${commit} in ${root}/${commit}"
}

main "$@"
