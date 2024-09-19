#!/bin/bash
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

# Always call from the root of the repository: ./ci/pigweed_download.sh
#
# We are using Pigweed for formatting checks, License checks etc.

set -e
source ci/bash_helpers.sh

DOWNLOADS_DIR=.downloads
mkdir -p ${DOWNLOADS_DIR}

DOWNLOADED_PIGWEED_PATH=${DOWNLOADS_DIR}/pigweed

if [ -d ${DOWNLOADED_PIGWEED_PATH} ]; then
  echo "${DOWNLOADED_PIGWEED_PATH} already exists, skipping the download."
else
  git clone https://pigweed.googlesource.com/pigweed/pigweed ${DOWNLOADED_PIGWEED_PATH} >&2
  pushd ${DOWNLOADED_PIGWEED_PATH} > /dev/null

  git checkout 47268dff45019863e20438ca3746c6c62df6ef09 >&2
  rm -rf ${DOWNLOADED_PIGWEED_PATH}/.git
  rm -f `find . -name BUILD`

  create_git_repo ./
  apply_patch_to_folder ./ ../../ci/pigweed.patch "SDK patch"

  popd > /dev/null
fi