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

"""Explicit metric semantics; no reshaping, clipping, or implicit epsilon."""

import ml_dtypes
import numpy as np

KEYS = ('CosSim', 'Max abs error', 'Mean abs error', 'RMSE', 'Relative L2')


def compare(
    reference: np.ndarray, target: np.ndarray
) -> dict[str, dict[str, float | str | None]]:
  """Computes float64 similarity and error metrics between two matching tensors.

  Raises ValueError with a machine-readable code ('shape_mismatch',
  'empty_tensor', 'unsupported_dtype', 'non_finite_tensor', or
  'numeric_overflow') so IPC analysis workers and HTTP handlers can surface
  uniform validation errors without custom exception classes.
  """
  if reference.shape != target.shape:
    raise ValueError('shape_mismatch')
  if reference.size == 0:
    raise ValueError('empty_tensor')
  if any(
      tensor.dtype.kind not in 'fiu'
      and tensor.dtype != np.dtype(ml_dtypes.bfloat16)
      for tensor in (reference, target)
  ):
    raise ValueError('unsupported_dtype')
  # Accumulate in float64, never change tensor coordinates.
  ref_f64, target_f64 = (
      reference.astype(np.float64),
      target.astype(np.float64),
  )
  if not np.isfinite(ref_f64).all() or not np.isfinite(target_f64).all():
    raise ValueError('non_finite_tensor')
  try:
    with np.errstate(over='raise', invalid='raise', divide='raise'):
      diff = ref_f64 - target_f64
      error = np.abs(diff)
      ref_norm, target_norm = (
          np.linalg.norm(ref_f64),
          np.linalg.norm(target_f64),
      )
      if not np.isfinite(ref_norm) or not np.isfinite(target_norm):
        raise ValueError('numeric_overflow')
      values = [
          None
          if ref_norm == 0 or target_norm == 0
          else float(np.sum((ref_f64 / ref_norm) * (target_f64 / target_norm))),
          float(error.max()),
          float(error.mean()),
          float(np.sqrt(np.mean(diff * diff))),
          None if ref_norm == 0 else float(np.linalg.norm(diff) / ref_norm),
      ]
  except FloatingPointError as exc:
    raise ValueError('numeric_overflow') from exc
  if any(
      metric_value is not None and not np.isfinite(metric_value)
      for metric_value in values
  ):
    raise ValueError('numeric_overflow')
  return {
      key: {
          'value': value,
          'status': 'undefined_zero_norm' if value is None else 'ok',
      }
      for key, value in zip(KEYS, values)
  }
