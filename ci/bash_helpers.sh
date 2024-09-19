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

# Create a git repo in a folder.
#
# Parameter(s):
#   $[1} - relative path to folder
create_git_repo() {
  pushd ${1} > /dev/null
  git init . > /dev/null
  git config user.email "unknown@google.com" --local
  git config user.name "odml" --local
  git add . >&2 2> /dev/null
  git commit -a -m "Commit for a temporary repository." > /dev/null
  git checkout -b odml > /dev/null
  popd > /dev/null
}

# Create a new commit with a patch in a folder that has a git repo.
#
# Parameter(s):
#   $[1} - relative path to folder
#   ${2} - path to patch file (relative to ${1})
#   ${3} - commit message for the patch
function apply_patch_to_folder() {
  pushd ${1} > /dev/null
  echo >&2 "Applying ${PWD}/${1}/${2} to ${PWD}/${1}"
  git apply ${2}
  git commit -a -m "${3}" > /dev/null
  popd > /dev/null
}
