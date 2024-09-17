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

ELECTRON_BASE_DIR="${KOKORO_ARTIFACTS_DIR}/github/model-explorer/src/electron"

# Build model explorer stand-alone pacakge using pyinstaller.
echo
echo '##################################################'
echo 'Build model explorer pyinstaller package'

cd "${ELECTRON_BASE_DIR}/pyinstaller"
./build.sh


# Build electron app.
echo
echo '##################################################'
echo 'Build electron app'

# Move the pyinstaller-built package into electron app.
echo
echo '#### Move model explorer stand-alone package into electron app'

ME_SERVER_TARGET_DIR="${ELECTRON_BASE_DIR}/app/model_explorer_server"
mkdir -p "${ME_SERVER_TARGET_DIR}"
mv "${ELECTRON_BASE_DIR}"/pyinstaller/venv/lib/python*/site-packages/model_explorer/dist/* \
    "${ME_SERVER_TARGET_DIR}/"

# Install node.
echo
echo '#### Install node'

ARCH="$(uname -m)"
# Rename "x86_64" to "x64".
if [[ "$ARCH" == "x86_64" ]]; then
    ARCH="x64"
fi
sudo installer -store -pkg "${ELECTRON_BASE_DIR}/tools/node-v20.17.0-${ARCH}.pkg" -target "/"
node -v

# Build electron app.
cd "${ELECTRON_BASE_DIR}/app"
./build.sh


# Create artifact (tar) from the built electron app.
echo
echo '##################################################'
echo "Create tar'd app"

mkdir "${KOKORO_ARTIFACTS_DIR}/artifacts"
cd "${ELECTRON_BASE_DIR}"
tar -czf "${KOKORO_ARTIFACTS_DIR}/artifacts/app.tar.gz" app

echo
echo '##################################################'
echo "Done creating tar. Uploading to placer"

ls -lh "${KOKORO_ARTIFACTS_DIR}/artifacts/"*
