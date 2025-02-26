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

# Build the custom element code and the model explorer code.
ng build custom_element
ng build model_explorer

# Build the browser-loadable script.
esbuild dist/custom_element/browser/main.js --bundle --outfile=dist/custom_element/browser/main_browser.js

# Clear.
DIST_DIR="custom_element_npm/dist"
mkdir -p "${DIST_DIR}"
rm -rf "${DIST_DIR}"/*
mkdir -p "${DIST_DIR}/static_files"

SRC_DIR="custom_element_npm/src"
mkdir -p "${SRC_DIR}"
rm -rf "${SRC_DIR}"/*
mkdir -p "${SRC_DIR}/custom_element"
mkdir -p "${SRC_DIR}/components"

# Copy files over.
cp -rf dist/custom_element/browser/static_files/* "${DIST_DIR}/static_files"
cp -rf dist/custom_element/browser/styles.css "${DIST_DIR}/static_files"
cp -rf dist/custom_element/browser/main.js "${DIST_DIR}/"
cp -rf dist/custom_element/browser/main_browser.js "${DIST_DIR}/"
cp -f dist/model_explorer/browser/worker*.js "${DIST_DIR}/worker.js"
cp -f src/custom_element/index.d.ts "${SRC_DIR}/custom_element"
cp -rf src/components/visualizer "${SRC_DIR}/components/"
