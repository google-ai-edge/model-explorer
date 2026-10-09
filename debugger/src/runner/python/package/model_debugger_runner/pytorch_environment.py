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

"""Probe the configured Runner's isolated PyTorch environment."""

import json
import os
import subprocess


def probe(root):
  base = dict(
      id='PyTorch',
      available=False,
      backends=['CPU'],
      precisions=['', 'float32', 'float16', 'bfloat16'],
      supported_options=['backend', 'precision', 'cpuThreads', 'contextLength'],
      reason='Configure the PyTorch capture package with --pytorch-root.',
  )
  if not root or not (root / '.venv/bin/python').is_file():
    return base
  code = (
      'import json, torch, transformers\n'
      'from ai_edge_debugger_pytorch import CaptureRun\n'
      'print(json.dumps(dict(torch=torch.__version__,'
      ' transformers=transformers.__version__,\n'
      " backends=['CPU'] + (['MPS'] if torch.backends.mps.is_available() else"
      " []) + (['CUDA'] if torch.cuda.is_available() else []))))"
  )
  try:
    result = subprocess.run(
        [str(root / '.venv/bin/python'), '-c', code],
        env={**os.environ, 'PYTHONPATH': str(root)},
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
    )
    details = json.loads(result.stdout.strip().splitlines()[-1])
    return {**base, **details, 'available': True, 'reason': ''}
  except (OSError, ValueError, subprocess.SubprocessError) as error:
    return {
        **base,
        'reason': (
            'PyTorch environment could not load torch, transformers and'
            f' CaptureRun: {error}'
        ),
    }
