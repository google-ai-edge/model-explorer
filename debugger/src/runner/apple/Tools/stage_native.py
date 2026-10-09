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

"""Stage the locally built debugger-enabled iOS runtime and exact provenance."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


def sha(path):
  with path.open('rb') as file:
    return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--runtime-root', type=Path, required=True)
  args = parser.parse_args()
  root = Path(__file__).resolve().parents[1]
  native = args.runtime_root / 'third_party/LiteRT-LM'
  archive = native / 'bazel-bin/swift/CLiteRTLM.xcframework.zip'
  vendor = root / 'Vendor'
  vendor.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=vendor, prefix='.staging-') as temp:
    staging = Path(temp)
    with zipfile.ZipFile(archive) as file:
      for member in file.infolist():
        if (
            member.filename.startswith('/')
            or '..' in Path(member.filename).parts
        ):
          raise ValueError('Unsafe framework archive path')
      file.extractall(staging)
    framework = staging / 'CLiteRTLM.xcframework'
    if not framework.is_dir():
      raise ValueError('Missing XCFramework in Bazel output')
    for platform in ('ios_arm64', 'ios_sim_arm64'):
      source = native / 'prebuilt' / platform
      destination = staging / platform
      destination.mkdir()
      for library in source.glob('*.dylib'):
        if library.read_bytes()[:7] == b'version':
          raise ValueError('Git LFS pointer instead of native library')
        shutil.copy2(library, destination)
    for name in ('CLiteRTLM.xcframework', 'ios_arm64', 'ios_sim_arm64'):
      target = vendor / name
      if target.exists():
        shutil.rmtree(target)
      shutil.move(str(staging / name), target)
  files = {
      str(path.relative_to(vendor)): sha(path)
      for path in sorted(vendor.rglob('*'))
      if path.is_file() and path.name != 'runtime-build.json'
  }
  provenance = dict(
      formatVersion=1,
      platform='iOS',
      debuggerBuildDefine=True,
      litertLMCommit=subprocess.check_output(
          ['git', '-C', str(native), 'rev-parse', 'HEAD'], text=True
      ).strip(),
      sourceLock=json.loads(
          (args.runtime_root / 'source-lock.json').read_text()
      ),
      nativeSourceDiffSHA256=hashlib.sha256(
          subprocess.check_output([
              'git',
              '-C',
              str(native),
              '-c',
              'filter.lfs.required=false',
              '-c',
              'filter.lfs.clean=cat',
              'diff',
              '--',
              'c',
              'runtime',
              'swift',
          ])
      ).hexdigest(),
      xcode=subprocess.check_output(
          ['xcodebuild', '-version'], text=True
      ).strip(),
      buildCommand=[
          'build',
          '--config=ios_arm64',
          '--define=LITERT_LM_DEBUGGER_ENABLED=1',
          '--define=litert_runtime_link_mode=dynamic',
          '//swift:CLiteRTLM',
      ],
      files=files,
  )
  (vendor / 'runtime-build.json').write_text(
      json.dumps(provenance, indent=2) + '\n'
  )
  print(vendor)


if __name__ == '__main__':
  main()
