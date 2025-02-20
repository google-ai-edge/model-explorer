#!/bin/bash
# Copyright 2025 The AI Edge Model Explorer Authors.
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
set -x

# Install
npm install

# Clear dist.
mkdir -p dist
rm -rf dist/*
mkdir -p dist/my_worker_path

# Build.
esbuild --bundle src/script.ts --outdir=dist

# Copy index.html.
cp src/index.html dist/

# Link worker.js and static_files.
cd dist
ln -s ../../node_modules/ai-edge-model-explorer-visualizer/dist/worker.js my_worker_path/worker.js
ln -s ../node_modules/ai-edge-model-explorer-visualizer/dist/static_files static_files
cd -

# Start a local server.
npx http-server -o dist/
