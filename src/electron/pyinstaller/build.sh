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

# Check python version.
echo '#### Check python version'
python -v

# Install packages.
echo '#### Install model explorer packages'
pip install torch ai-edge-model-explorer pyinstaller \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.python.org/simple
