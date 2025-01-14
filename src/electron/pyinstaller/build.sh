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

pip install torch pyinstaller model-explorer-onnx \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.python.org/simple

# Install local model explorer
pip install ../../server/package/

# Run pyinstaller
echo
echo '#### Run pyinstaller'

cp -f model_explorer.py venv/lib/python*/site-packages/
cd venv/lib/python*/site-packages/

# This generates a spec file that may be useful to inspect when debugging
# ./venv/lib/python3.11/site-packages/model_explorer/model_explorer.spec
#
# `--python-option u` is required to make python not buffer stdio.
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
  --python-option u \
  model_explorer.py
cd -

# Done.
echo
echo '#### Done building model explorer pyinstaller package'
