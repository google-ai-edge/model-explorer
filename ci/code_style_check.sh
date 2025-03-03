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

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/..
cd "${ROOT_DIR}"

# Download pigweed for the license and formatting checks.
./ci/pigweed_download.sh

# Explicitly disable exit on error so that we can report all the style errors in
# one pass and clean up the temporary git repository even when one of the
# scripts fail with an error code.
set +e

# --fix_formatting to let the script fix both code and build file format error.
FIX_FORMAT_FLAG=${1}

############################################################
# License Check
############################################################
.downloads/pigweed/pw_presubmit/py/pw_presubmit/pigweed_presubmit.py \
  -p copyright_notice \
  -e .downloads \
  -e .github \
  -e venv \
  -e src/server/package/src/model_explorer/web_app \
  -e src/custom_element_demos \
  -e src/ui \
  -e "\.md" \
  -e "\.ipynb" \
  -e "\.patch" \
  -e "\.jpg" \
  -e "\.png" \
  -e "\.ico" \
  -e "\.jar" \
  -e "\.test" \
  -e "\.toml" \
  -e "BUILD" \
  -e "\.dot" \
  -e "\.in" \
  -e "\.pyi" \
  -e "\.tflite" \
  -e "\.pb" \
  -e "\.pbtxt" \
  -e "\.mlir" \
  --output-directory /tmp

LICENSE_CHECK_RESULT=$?

if [[ ${LICENSE_CHECK_RESULT} != 0 ]]; then
  echo "License check failed. Please fix the corresponding licenses."
fi

############################################################
# Python formatting
############################################################

PYINK_COMMAND="pyink --pyink-use-majority-quotes --pyink-indentation=2 --preview --unstable --line-length 80
      --extend-exclude .downloads --extend-exclude \.pyi --check ./"

echo "Testing python formatting with ${PYINK_COMMAND}"
${PYINK_COMMAND}
PYTHON_FORMAT_RESULT=$?

if [[ ${PYTHON_FORMAT_RESULT} != 0 ]]; then
  echo "Python formatting issues found."
  echo "To apply formatting automatically, run: ./format.sh"
fi
if [[ ${LICENSE_CHECK_RESULT}  != 0 || \
      ${PYTHON_FORMAT_RESULT}  != 0 \
   ]]
then
  exit 1
fi

# Re-enable exit on error now that we are done with the temporary git repo.
set -e
