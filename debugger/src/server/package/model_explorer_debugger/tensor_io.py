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

"""Load one indexed tensor without changing its stored dtype or coordinates."""

from pathlib import Path

# Registers bfloat16 with NumPy for Safetensors loading.
import ml_dtypes  # noqa: F401
import numpy as np
from safetensors import SafetensorError, safe_open


def _tensor_path(root, record):
  raw_path = record.get('path')
  if not isinstance(raw_path, str) or not raw_path or '\x00' in raw_path:
    raise ValueError('invalid_tensor_path')
  relative = Path(raw_path)
  if relative.is_absolute():
    raise ValueError('invalid_tensor_path')
  try:
    root = Path(root).resolve()
    path = (root / relative).resolve()
  except (OSError, RuntimeError, ValueError) as exc:
    raise ValueError('invalid_tensor_path') from exc
  if not path.is_relative_to(root) or path.suffix != '.safetensors':
    raise ValueError('invalid_tensor_path')
  return path


def load_tensor(root, record) -> np.ndarray:
  """Read one keyed Safetensors record relative to its session root.

  Records must explicitly declare the format, exact key and canonical NumPy
  dtype name, including ``bfloat16``. Malformed data has stable errors; missing
  files raise FileNotFoundError.
  """
  if record.get('format') != 'safetensors':
    raise ValueError('unsupported_tensor_format')
  path = _tensor_path(root, record)
  shape, dtype = record.get('shape'), record.get('dtype')
  if (
      not isinstance(shape, list)
      or any(type(size) is not int or size < 0 for size in shape)
      or not isinstance(dtype, str)
      or not dtype
  ):
    raise ValueError('tensor_metadata_mismatch')
  key = record.get('key')
  if not isinstance(key, str):
    raise ValueError('invalid_tensor_key')
  # safe_open reports absent files differently across library versions.
  if not path.exists():
    raise FileNotFoundError(path)
  missing_key = False
  try:
    with safe_open(path, framework='numpy') as shard:
      missing_key = key not in shard.keys()
      tensor = None if missing_key else shard.get_tensor(key)
  except FileNotFoundError:
    raise
  except (
      OSError,
      SafetensorError,
      ValueError,
      TypeError,
      KeyError,
      EOFError,
      OverflowError,
  ) as exc:
    raise ValueError('invalid_tensor_file') from exc
  if missing_key:
    raise ValueError('tensor_key_not_found')
  if list(tensor.shape) != shape or dtype != tensor.dtype.name:
    raise ValueError('tensor_metadata_mismatch')
  return tensor
