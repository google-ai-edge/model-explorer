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

"""Configure the development Mac Runner's local PyTorch adapter."""

import argparse
import json
import os
from pathlib import Path
import sys


def main():
  source = Path(__file__).resolve().parents[3]
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--pytorch-root', required=True, type=Path)
  parser.add_argument(
      '--package-root',
      type=Path,
      default=source / 'runner/python/package',
      help='Directory containing model_debugger_runner',
  )
  parser.add_argument(
      '--contracts-root',
      type=Path,
      default=source / 'contracts/python/package',
      help='Directory containing model_debugger_contracts',
  )
  parser.add_argument(
      '--output',
      type=Path,
      default=Path.home()
      / 'Library/Application Support/Model Debugger Runner/python-runner.json',
  )
  args = parser.parse_args()
  package, contracts = (
      args.package_root.resolve(),
      args.contracts_root.resolve(),
  )
  for path in (
      package / 'model_debugger_runner/python_runner_host.py',
      contracts / 'model_debugger_contracts/__init__.py',
  ):
    if not path.is_file():
      parser.error('Python package file not found: ' + str(path))
  sys.path[:0] = [str(package), str(contracts)]
  from model_debugger_runner.pytorch_environment import probe

  root = args.pytorch_root.resolve()
  capability = probe(root)
  if not capability['available']:
    raise SystemExit(capability['reason'])
  config = dict(
      version=2,
      executable=str(root / '.venv/bin/python'),
      packageRoot=str(package),
      contractPackageRoot=str(contracts),
      module='model_debugger_runner.python_runner_host',
      runtimeRoot=str(root),
      backends=capability['backends'],
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  fd = os.open(args.output, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
  with os.fdopen(fd, 'w') as file:
    json.dump(config, file, indent=2)
  print('PyTorch adapter configured. Restart the local Runner App to load it.')


if __name__ == '__main__':
  main()
