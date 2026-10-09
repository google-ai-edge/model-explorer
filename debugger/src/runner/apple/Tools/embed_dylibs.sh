#!/bin/bash
# Copyright 2026 The AI Edge Model Explorer Authors.
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

set -euo pipefail
case "$PLATFORM_NAME" in
  macosx) runner_arch=macos_arm64 ;;
  iphoneos) runner_arch=ios_arm64 ;;
  iphonesimulator) runner_arch=ios_sim_arm64 ;;
  *) echo "Unsupported Runner platform: $PLATFORM_NAME" >&2; exit 1 ;;
esac
runner_frameworks="$TARGET_BUILD_DIR/$FRAMEWORKS_FOLDER_PATH"
mkdir -p "$runner_frameworks"
# Do not include the separate native Metal plugins. The simulator variant
# duplicates Objective-C classes already in libLiteRt. macOS uses WebGPU/Metal.
rm -f "$runner_frameworks/libLiteRtMetalAccelerator.dylib" "$runner_frameworks/libLiteRtTopKMetalSampler.dylib"
runner_libraries=(libLiteRt.dylib libGemmaModelConstraintProvider.dylib)
if [[ "$PLATFORM_NAME" == macosx ]]; then
  runner_libraries+=(liblitert-lm.dylib libwebgpu_dawn.dylib libLiteRtWebGpuAccelerator.dylib libLiteRtTopKWebGpuSampler.dylib)
else
  # Remove stale macOS GPU payloads if a build directory is reused.
  rm -f "$runner_frameworks/libwebgpu_dawn.dylib" "$runner_frameworks/libLiteRtWebGpuAccelerator.dylib" "$runner_frameworks/libLiteRtTopKWebGpuSampler.dylib"
fi
for runner_name in "${runner_libraries[@]}"; do
  runner_library="$SRCROOT/Vendor/$runner_arch/$runner_name"
  cp -f "$runner_library" "$runner_frameworks/"
  runner_copy="$runner_frameworks/$(basename "$runner_library")"
  chmod u+w "$runner_copy"
  if [[ "$runner_name" == liblitert-lm.dylib ]]; then
    install_name_tool -id @rpath/liblitert-lm.dylib "$runner_copy"
    install_name_tool -change bazel-out/darwin_arm64-opt/bin/c/liblitert-lm.dylib @rpath/liblitert-lm.dylib "$TARGET_BUILD_DIR/$EXECUTABLE_PATH"
  fi
  if [[ "$CODE_SIGNING_ALLOWED" != NO && -n "${EXPANDED_CODE_SIGN_IDENTITY:-}" ]]; then
    /usr/bin/codesign --force --sign "$EXPANDED_CODE_SIGN_IDENTITY" "$runner_copy"
  fi
done
if [[ "$PLATFORM_NAME" == macosx ]]; then
  runner_provenance="$TARGET_BUILD_DIR/$UNLOCALIZED_RESOURCES_FOLDER_PATH/runtime-build.json"
  # Keep upstream/staged hashes in files; signatures and install names can change
  # bytes in the App, so record its final embedded binaries separately.
  python3 - "$runner_provenance" "$runner_frameworks" "${runner_libraries[@]}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
provenance = Path(sys.argv[1])
frameworks = Path(sys.argv[2])
value = json.loads(provenance.read_text())
value['embeddedFiles'] = {}
for name in sys.argv[3:]:
    digest = hashlib.sha256()
    with (frameworks / name).open('rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            digest.update(block)
    value['embeddedFiles'][name] = digest.hexdigest()
provenance.write_text(json.dumps(value, indent=2) + '\n')
PY
fi
