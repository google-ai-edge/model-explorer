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

# Fail on any error.
set -e

# Check npm version
echo
echo '#### Check npm version'

npm -v

# Install pacakges
echo
echo '#### Install packages'

npm install

# Build the python server
SERVER_DIR="model_explorer_server"
if [ -d "$SERVER_DIR" ]; then rm -r $SERVER_DIR; fi
mkdir $SERVER_DIR
cd ../pyinstaller/
./build.sh
mv ./venv/lib/python*/site-packages/model_explorer/dist/model_explorer/* ../app/model_explorer_server
cd -

# Build and package electron app.
echo
echo '#### Build and package electon app'

npm run package
