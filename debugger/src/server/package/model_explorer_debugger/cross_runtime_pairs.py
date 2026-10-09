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

"""Explicit pair registration and reload.

Evaluation lives in cross_runtime_common.
"""

import json
from pathlib import Path
import shutil
import tempfile

from .cross_runtime_common import _digest, _evaluate_prefill, _require, _sha

__all__ = ['compare_pair', 'read_pair', 'read_pair_manifest', 'register_pair']


def _evaluate(manifest, roots):
  if isinstance(manifest, dict) and manifest.get('format_version') == 3:
    from .cross_runtime_kv import evaluate_kv

    return evaluate_kv(manifest, roots)
  if isinstance(manifest, dict) and manifest.get('format_version') == 2:
    from .cross_runtime_positions import evaluate_position

    return evaluate_position(manifest, roots)
  return _evaluate_prefill(manifest, roots)


def compare_pair(manifest, roots):
  """Validate every source and compute an explicit view.

  Raw data is never mutated.
  """
  return _evaluate(manifest, roots)[0]


def register_pair(destination, manifest, roots):
  """Preserve validated raw files and source evidence for independent reload."""
  report, evidence = _evaluate(manifest, roots)
  destination = Path(destination).resolve()
  destination.mkdir(parents=True, exist_ok=True)
  pair_id = report['pair_id']
  final = destination / pair_id
  if final.exists():
    _require(
        final.resolve().is_relative_to(destination),
        'registered pair outside root',
    )
    saved = json.loads((final / 'manifest.json').read_text())
    _require(saved == manifest, 'registered pair identity conflict')
    return read_pair(destination, pair_id)
  temporary = Path(tempfile.mkdtemp(prefix='.pair-', dir=destination))
  try:
    for (role, relative), (source, _) in evidence.files.items():
      target = temporary / role / relative
      _require(
          target.resolve().is_relative_to((temporary / role).resolve()),
          'copy destination outside role',
      )
      target.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(source, target)
    (temporary / 'manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n'
    )
    (temporary / 'report.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n'
    )
    compare_pair(
        manifest, {role: temporary / role for role in ('ref', 'target')}
    )
    temporary.rename(final)
  except BaseException:
    shutil.rmtree(temporary, ignore_errors=True)
    raise
  return report


def read_pair_manifest(destination, pair_id):
  """Read an identity-checked declaration without loading tensor evidence."""
  _require(_sha(pair_id), 'invalid registered pair ID')
  root = Path(destination).resolve() / pair_id
  _require(
      root.resolve().is_relative_to(Path(destination).resolve()),
      'registered pair outside root',
  )
  _require(
      all(
          (root / role).resolve().is_relative_to(root.resolve())
          for role in ('ref', 'target')
      ),
      'registered evidence outside pair',
  )
  manifest = json.loads((root / 'manifest.json').read_text())
  _require(
      _digest(manifest) == pair_id, 'registered manifest identity mismatch'
  )
  return manifest


def read_pair(destination, pair_id):
  manifest = read_pair_manifest(destination, pair_id)
  root = Path(destination).resolve() / pair_id
  return compare_pair(
      manifest, {role: root / role for role in ('ref', 'target')}
  )
