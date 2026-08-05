#!/bin/bash

check_sha() {
  local repo=$1
  local tag=$2
  local expected_sha=$3
  
  echo "Checking $repo@$tag (Expected: $expected_sha)"
  # Get the SHA of the tag
  actual_sha=$(git ls-remote "https://github.com/$repo.git" "refs/tags/$tag" | awk '{print $1}')
  
  if [ -z "$actual_sha" ]; then
    echo "  -> Error: Tag not found."
  elif [ "$actual_sha" = "$expected_sha" ]; then
    echo "  -> Match: $actual_sha"
  else
    echo "  -> Mismatch! Actual: $actual_sha, Expected: $expected_sha"
  fi
}

check_sha "actions/download-artifact" "v4" "d3f86a106a0bac45b974a628896c90dbdf5c8093"
check_sha "softprops/action-gh-release" "v2" "3bb12739c298aeb8a4eeaf626c5b8d85266b0e65"
check_sha "actions/checkout" "v3" "a37ce9120846195fa4ece8f58b268e6043cb2f26"
check_sha "actions/setup-node" "v3" "3235b876344d2a9aa001b8d1453c930bba69e610"
check_sha "actions/checkout" "v4" "11d5960a326750d5838078e36cf38b85af677262"
check_sha "actions/setup-python" "v5" "a26af69be951a213d495a4c3e4e4022e16d87065"
check_sha "actions/stale" "v9" "5bef64f19d7facfb25b37b414482c7164d639639"
check_sha "actions/setup-node" "v4" "49933ea5288caeca8642d1e84afbd3f7d6820020"
check_sha "actions/upload-artifact" "v4" "ea165f8d65b6e75b540449e92b4886f43607fa02"

