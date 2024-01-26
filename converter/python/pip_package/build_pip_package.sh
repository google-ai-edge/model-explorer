#!/usr/bin/env bash
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${CI_BUILD_PYTHON:-python3}"
VERSION_SUFFIX=${VERSION_SUFFIX:-}
export PACKAGE_VERSION="v0.1.0"
export PROJECT_NAME=${WHEEL_PROJECT_NAME:-model_explorer_converter}
BUILD_DIR="gen/converter_pip"
BAZEL_FLAGS="--copt=-O3"

# Build source tree.
rm -rf "${BUILD_DIR}" && mkdir -p "${BUILD_DIR}/model_explorer_converter"
cp -r "${SCRIPT_DIR}/MANIFEST.in" \
      "${BUILD_DIR}"
cp  "${SCRIPT_DIR}/setup_with_binary.py" "${BUILD_DIR}/setup.py"
echo "__version__ = '${PACKAGE_VERSION}'" >> "${BUILD_DIR}/model_explorer_converter/__init__.py"

# Build python _pywrap_convert_wrapper.

# We need to pass down the environment variable with a possible alternate Python
# include path for Python 3.x builds to work.
export CROSSTOOL_PYTHON_INCLUDE_PATH

case "${TENSORFLOW_TARGET}" in
  windows)
    LIBRARY_EXTENSION=".pyd"
    ;;
  *)
    LIBRARY_EXTENSION=".so"
    ;;
esac

bazel build -c opt -s --config=monolithic --config=noaws --config=nogcp --config=nohdfs --config=nonccl \
  ${BAZEL_FLAGS} python/convert_wrapper:_pywrap_convert_wrapper
cp "bazel-bin/python/convert_wrapper/_pywrap_convert_wrapper${LIBRARY_EXTENSION}" \
   "${BUILD_DIR}/model_explorer_converter"

# Bazel generates the wrapper library with r-x permissions for user.
# At least on Windows, we need write permissions to delete the file.
# Without this, setuptools fails to clean the build directory.
chmod u+w "${BUILD_DIR}/model_explorer_converter/_pywrap_convert_wrapper${LIBRARY_EXTENSION}"

# Build python wheel.
cd "${BUILD_DIR}"

if [[ -n "${WHEEL_PLATFORM_NAME}" ]]; then
  ${PYTHON} setup.py bdist --plat-name=${WHEEL_PLATFORM_NAME} \
                      bdist_wheel --plat-name=${WHEEL_PLATFORM_NAME}
else
  ${PYTHON} setup.py bdist bdist_wheel
fi

echo "Output can be found here:"
find "$PWD"
