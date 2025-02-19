#!/bin/bash

# @license
# Copyright 2024 The Model Explorer Authors. All Rights Reserved.
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

# Check if the script is being run from the "src/ui" directory.
dir=$(pwd)
if [[ ":$dir" != *"/src/ui" ]]; then
  echo 'Must run the script from the "src/ui" directory'
  exit 1
fi

# Build app.
ng build model_explorer

# Remove old app
cd ../server/package/src/model_explorer/web_app
rm -rf *

# Copy the newly-built files.
cd -
cp -rf dist/model_explorer/browser/* ../server/package/src/model_explorer/web_app/