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

"""Content identity for a local Hugging Face model.

Computing the identity does not import the model's runtime.
"""

import hashlib
import json
from pathlib import Path
from typing import Any


def _digest(path: Path) -> str:
  value = hashlib.sha256()
  with path.open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
      value.update(block)
  return value.hexdigest()


def describe_model(path: str | Path) -> tuple[list[dict[str, Any]], str]:
  root = Path(path).resolve()
  if not root.is_dir() or not (root / 'config.json').is_file():
    raise ValueError(
        'Choose a local Hugging Face model directory containing config.json'
    )
  if not any(root.glob('*.safetensors')):
    raise ValueError('The local model must contain Safetensors weights')
  if not any(
      (root / name).is_file()
      for name in (
          'tokenizer.json',
          'tokenizer.model',
          'vocab.json',
          'vocab.txt',
          'spiece.model',
      )
  ):
    raise ValueError('The local model must include its tokenizer files')
  # Hash every top-level runtime input, including custom tokenizer templates.
  # HF cache snapshots use symlinks into a shared blob store; hash their bytes.
  suffixes = {'.json', '.safetensors', '.model', '.txt', '.jinja', '.tiktoken'}
  files = [
      dict(path=file.name, size=file.stat().st_size, sha256=_digest(file))
      for file in sorted(root.iterdir())
      if file.is_file() and file.suffix in suffixes
  ]
  checksum = hashlib.sha256(
      json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
  ).hexdigest()
  return files, checksum


def verify_model(
    path: str | Path,
    expected_files: list[dict[str, Any]],
    expected_sha256: str,
) -> tuple[list[dict[str, Any]], str]:
  files, checksum = describe_model(path)
  if files != expected_files or checksum != expected_sha256:
    raise ValueError(
        'Registered PyTorch model or tokenizer changed; register the directory'
        ' again'
    )
  return files, checksum
