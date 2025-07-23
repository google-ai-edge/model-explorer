#!/usr/bin/env bash
# Copyright 2024 The AI Edge Model Explorer Authors.
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

set -ex

USAGE="$(basename $0) <package-version>

Builds a pip package for the Model Explorer backend adapter.

<package-version> should be a string of the form "x.x.x", eg. "1.2.0".
"

# Define a regex pattern for the format x.x.x
PATTERN="^[0-9]+\.[0-9]+\.[0-9]+$"

if [[ -z "${1}" ]]; then
  echo "${USAGE}"
  exit 1
fi

# Check if the argument matches the pattern
if [[ "$1" =~ $PATTERN ]]; then
  export PACKAGE_VERSION="${1}"
else
  echo "Error: The package version '$1' is not in the correct format."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${CI_BUILD_PYTHON:-python3}"
PYTHON_VERSION="$(${PYTHON} --version | cut -d " " -f 2)"
IFS='.' read -ra VERSION_PARTS <<< "${PYTHON_VERSION}"
# TF only supports python version ["3.9", "3.10", "3.11", "3.12"].
export TF_PYTHON_VERSION="${VERSION_PARTS[0]}.${VERSION_PARTS[1]}"
export PROJECT_NAME=${WHEEL_PROJECT_NAME:-ai_edge_model_explorer_adapter}
BUILD_DIR="gen/adapter_pip"
# This variable will hold flags specific to the OS that are not in .bazelrc
BAZEL_PLATFORM_FLAGS=""
ARCH="$(uname -m)"

# Build source tree.
rm -rf "${BUILD_DIR}" && mkdir -p "${BUILD_DIR}/ai_edge_model_explorer_adapter"
cp -r "${SCRIPT_DIR}/MANIFEST.in" \
      "${BUILD_DIR}"
cp  "${SCRIPT_DIR}/setup_with_binary.py" "${BUILD_DIR}/setup.py"
echo "__version__ = '${PACKAGE_VERSION}'" >> "${BUILD_DIR}/ai_edge_model_explorer_adapter/__init__.py"

# Build python _pywrap_convert_wrapper.

# We need to pass down the environment variable with a possible alternate Python
# include path for Python 3.x builds to work.
export CROSSTOOL_PYTHON_INCLUDE_PATH

case "${TENSORFLOW_TARGET}" in
  windows)
    LIBRARY_EXTENSION=".pyd"
    # The .bazelrc handles all compiler flags. We just need to trigger the config.
    BAZEL_PLATFORM_FLAGS="--config=windows"
    # Override ARCH for Windows, as 'uname -m' might not be reliable.
    ARCH="x86_64"
    ;;
  *)
    LIBRARY_EXTENSION=".so"
    ;;
esac

# Set linkopt for different architectures for non-Windows platforms.
case "${ARCH}" in
  x86_64)
    # No extra flags needed for Linux/Windows x86_64
    ;;
  arm64)
    # MacOS arm64.
    BAZEL_PLATFORM_FLAGS="${BAZEL_PLATFORM_FLAGS} --linkopt="-ld_classic""
    ;;
  aarch64)
    # Linux arm64.
    BAZEL_PLATFORM_FLAGS="${BAZEL_PLATFORM_FLAGS} --config=linux_arm64"
    ;;
  *)
    echo "Unsupported architecture: ${ARCH}"
    exit 1
    ;;
esac

# Note: --config=monolithic is already set for Windows in .bazelrc, but is needed for other platforms.
bazel build -c opt -s --config=monolithic --config=noaws --config=nogcp --config=nohdfs --config=nonccl \
  ${BAZEL_PLATFORM_FLAGS} python/convert_wrapper:_pywrap_convert_wrapper
cp "bazel-bin/python/convert_wrapper/_pywrap_convert_wrapper${LIBRARY_EXTENSION}" \
   "${BUILD_DIR}/ai_edge_model_explorer_adapter"

# Bazel generates the wrapper library with r-x permissions for user.
# At least on Windows, we need write permissions to delete the file.
# Without this, setuptools fails to clean the build directory.
chmod u+w "${BUILD_DIR}/ai_edge_model_explorer_adapter/_pywrap_convert_wrapper${LIBRARY_EXTENSION}"

# Build python wheel.
cd "${BUILD_DIR}"

# Assign the wheel name based on the platform and architecture.
# This logic must remain in the script.
if [[ "${TENSORFLOW_TARGET}" == "windows" ]]; then
    WHEEL_PLATFORM_NAME="win_amd64"
elif test -e "/System/Library/CoreServices/SystemVersion.plist"; then
  if [[ "${ARCH}" == "arm64" ]]; then
    WHEEL_PLATFORM_NAME="macosx_12_0_arm64"
  else
    WHEEL_PLATFORM_NAME="macosx_10_15_x86_64"
  fi
elif test -e "/etc/lsb-release"; then
  if [[ "${ARCH}" == "aarch64" ]]; then
    WHEEL_PLATFORM_NAME="manylinux_2_17_aarch64"
  elif [[ "${ARCH}" == "x86_64" ]]; then
    WHEEL_PLATFORM_NAME="manylinux_2_17_x86_64"
  fi
fi

if [[ -n "${WHEEL_PLATFORM_NAME}" ]]; then
  ${PYTHON} setup.py sdist bdist_wheel --plat-name=${WHEEL_PLATFORM_NAME}
else
  ${PYTHON} setup.py sdist bdist_wheel
fi

echo "Output can be found here:"
find ${PWD} -name '*.whl'
