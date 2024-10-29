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

# Remember current directory.
SCRIPT_DIR="$(pwd)"

# Check python version.
echo
echo '#### Check python version'

python3 --version

# Create venv.
echo
echo '#### Create venv'

python3 -m venv venv
source venv/bin/activate

# Install packages.
echo
echo '#### Install model explorer packages'

pip install torch ai-edge-model-explorer pyinstaller model-explorer-onnx \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.python.org/simple

# Replace the model explorer code with the latest.
# Move to 'model-explorer/src/electron/pyinstaller/venv/lib/python3.X/site-packages'
cd venv/lib/python*/site-packages/
rm -rf model_explorer
# Copy 'model-explorer/src/server/package/src/model_explorer/'
cp -rf "${SCRIPT_DIR}/../../server/package/src/model_explorer" .
cd -

# Run pyinstaller
echo
echo '#### Run pyinstaller'

cp -f model_explorer.py venv/lib/python*/site-packages/
cd venv/lib/python*/site-packages/
pyinstaller -y \
  --hidden-import model_explorer.builtin_tflite_flatbuffer_adapter \
  --hidden-import model_explorer.builtin_tflite_mlir_adapter \
  --hidden-import model_explorer.builtin_tf_mlir_adapter \
  --hidden-import model_explorer.builtin_tf_direct_adapter \
  --hidden-import model_explorer.builtin_graphdef_adapter \
  --hidden-import model_explorer.builtin_pytorch_exportedprogram_adapter \
  --hidden-import model_explorer.builtin_mlir_adapter \
  --hidden-import model_explorer_onnx.main \
  --copy-metadata ai-edge-model-explorer \
  --add-data "web_app:model_explorer/web_app" \
  model_explorer.py
cd -

# Done.
echo
echo '#### Done building model explorer pyinstaller package'
