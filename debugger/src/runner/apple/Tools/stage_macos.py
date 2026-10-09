#!/usr/bin/env python3
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

"""Stage the existing debugger-enabled arm64 macOS C runtime for the Mac App."""

import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess


def main() -> None:
  """Stages the arm64 macOS C runtime libraries and writes build provenance."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--runtime-root', type=pathlib.Path, required=True)
  parser.add_argument(
      '--native-library',
      type=pathlib.Path,
      help='Use a separately rebuilt C API dylib',
  )
  parser.add_argument(
      '--native-provenance',
      type=pathlib.Path,
      help='Build provenance bound to the native library SHA256',
  )
  parser.add_argument(
      '--destination',
      type=pathlib.Path,
      help='Stage into an isolated directory instead of Vendor/macos_arm64',
  )
  args = parser.parse_args()
  root = pathlib.Path(__file__).resolve().parents[1]
  destination = args.destination or root / 'Vendor/macos_arm64'
  destination.mkdir(parents=True, exist_ok=True)
  files = {}
  minimum_versions = []
  gpu_plugins = (
      'libwebgpu_dawn.dylib',
      'libLiteRtWebGpuAccelerator.dylib',
      'libLiteRtTopKWebGpuSampler.dylib',
  )
  for name in (
      'liblitert-lm.dylib',
      'libLiteRt.dylib',
      'libGemmaModelConstraintProvider.dylib',
      *gpu_plugins,
  ):
    source = args.runtime_root / 'artifacts/python/litert_lm' / name
    if name == 'liblitert-lm.dylib' and args.native_library:
      source = args.native_library
    build = subprocess.check_output(
        ['xcrun', 'vtool', '-show-build', str(source)], text=True
    )
    version = re.search(r'minos ([0-9.]+)', build)
    if not version:
      raise ValueError('Native library has no macOS minimum version')
    minimum_versions.append(version.group(1))
    target = destination / name
    if target.exists():
      target.chmod(0o755)
    shutil.copy2(source, target)
    with source.open('rb') as file:
      files[name] = hashlib.file_digest(file, 'sha256').hexdigest()
  native = args.runtime_root / 'third_party/LiteRT-LM'
  provenance = dict(
      formatVersion=1,
      platform='macOS',
      architecture='arm64',
      minimumOS=max(
          minimum_versions,
          key=lambda value: tuple(map(int, value.split('.'))),
      ),
      sourceLock=json.loads(
          (args.runtime_root / 'source-lock.json').read_text()
      ),
      litertLMCommit=subprocess.check_output(
          ['git', '-C', str(native), 'rev-parse', 'HEAD'], text=True
      ).strip(),
      nativeLibrarySource=str(
          (
              args.native_library
              or args.runtime_root
              / 'artifacts/python/litert_lm/liblitert-lm.dylib'
          ).resolve()
      ),
      files=files,
      gpuPlugins=list(gpu_plugins),
      gpuPluginAPI='WebGPU',
      gpuAdapterBackend='Metal',
      note=(
          'Staged runtime libraries; plugin presence describes availability,'
          ' not verified GPU execution. The running C API checks debugger'
          ' support.'
      ),
  )
  if args.native_provenance:
    native_build = json.loads(args.native_provenance.read_text())
    if native_build.get('library_sha256') != files['liblitert-lm.dylib']:
      raise ValueError(
          'Native build provenance does not match the staged library bytes'
      )
    provenance['nativeLibraryBuild'] = native_build
    if native_build.get('full_source_working_diff_sha256'):
      provenance['nativeSourceDiffSHA256'] = native_build[
          'full_source_working_diff_sha256'
      ]
  (destination / 'runtime-build.json').write_text(
      json.dumps(provenance, indent=2) + '\n'
  )
  print(destination)


if __name__ == '__main__':
  main()
