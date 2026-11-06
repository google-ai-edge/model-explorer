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

"""What a chart draws of two long vectors, so the UI never reads them whole.

A vector here is one runtime's values by element index (a vocabulary index),
float64, NaN where an element has no value; a value that is NaN or infinite
is not valid. Two charts are drawn of the reference and target vectors:

* The line chart (src/ui widgets/charts/channel_bins.ts) splits a range of
  indices into bins of `bin_size` elements and reduces each runtime's bin on
  its own, to its minimum, maximum or mean (`line_bins`). A bin with an
  element that is not valid has no value: it is a gap. A bin's
  representative, the element a click on it selects, is its first element
  for the mean and its first extremum otherwise.
* The scatter plot (widgets/charts/scatter_model.ts) draws the elements both
  of whose values are valid, the paired elements, Ref against Target. When
  more than `SCATTER_POINT_LIMIT` of them lie in the value range, it draws
  one element per occupied cell of a square grid over the range: the one
  that differs most, then the lowest index (`scatter_points`). Its value
  axes span the paired values (`scatter_extent`).

Each is computed as the UI computed it from the whole vectors, operation for
operation, so what is drawn is the same to the last bit: a mean adds each
value divided by the bin's size, in index order, from zero; a minimum is -0
when the bin holds a -0 among its zeros, a maximum +0 when it holds a +0.
"""

import math
from typing import Any

import numpy as np

# The reductions of a line-chart bin.
REDUCTIONS = ('min', 'max', 'mean')
# How many points a scatter plot draws at most before it groups them; the
# UI's SCATTER_POINT_LIMIT (widgets/charts/scatter_model.ts).
SCATTER_POINT_LIMIT = 4096


def bin_size(start: int, end: int, capacity: int) -> int:
  """How many elements a bin of [start, end) holds, for `capacity` bins."""
  return max(1, math.ceil((end - start) / max(1, capacity)))


def clamped_range(start: int, end: int, total: int) -> tuple[int, int]:
  """A range of element indices kept inside [0, max(1, total)).

  As the UI keeps a saved element range inside the vector: a start at the
  last index at most, an end after the start.

  Args:
      start: The range's first index.
      end: The index after its last.
      total: How many elements the vector has.

  Returns:
      The clamped `(start, end)`.
  """
  last = max(1, total)
  start = max(0, min(last - 1, start))
  return start, max(start + 1, min(last, end))


def line_bins(
    ref: np.ndarray,
    target: np.ndarray,
    start: int,
    end: int,
    capacity: int,
    reduction: str,
) -> dict[str, Any]:
  """The line chart's bins of [start, end), each runtime reduced alone.

  Args:
      ref: The reference values by index, float64, NaN where missing; at least
        `end` long.
      target: The target's, likewise.
      start: The first index of the range.
      end: The index after its last.
      capacity: How many bins the chart has room for.
      reduction: One of REDUCTIONS.

  Returns:
      `{start, end, size, ref, target, ref_index, target_index}`: bin k
      spans [start + k * size, min(end, start + (k + 1) * size)); `ref`
      and `target` are each bin's value, None for a gap; `ref_index` and
      `target_index` each bin's representative element, None for a gap.
  """
  size = bin_size(start, end, capacity)
  result = {'start': start, 'end': end, 'size': size}
  for side, values in (('ref', ref), ('target', target)):
    reduced, chosen = _reduce(values[start:end], size, reduction)
    firsts = np.arange(start, end, size)
    result[side] = [
        float(value) if ok else None
        for value, ok in zip(reduced, ~np.isnan(chosen))
    ]
    result[side + '_index'] = [
        int(first + k) if ok else None
        for first, k, ok in zip(
            firsts.tolist(),
            np.nan_to_num(chosen).astype(np.int64).tolist(),
            (~np.isnan(chosen)).tolist(),
        )
    ]
  return result


def _reduce(
    values: np.ndarray, size: int, reduction: str
) -> tuple[np.ndarray, np.ndarray]:
  """Each bin's value and its representative's offset in the bin.

  Args:
      values: The values of the range, float64.
      size: How many elements a bin holds.
      reduction: One of REDUCTIONS.

  Returns:
      Two float64 arrays, one entry per bin: the value, and the
      representative's offset from the bin's first element; the offset is
      NaN for a bin with an element that is not valid.
  """
  count = -(-values.size // size)
  full = values.size // size
  values_out = np.zeros(count)
  offsets = np.full(count, np.nan)
  blocks = [(0, values[: full * size].reshape(full, size))]
  if full < count:
    blocks.append((full, values[full * size :].reshape(1, -1)))
  for first, block in blocks:
    rows = slice(first, first + block.shape[0])
    valid = np.isfinite(block).all(axis=1)
    if reduction == 'mean':
      # Each value divided by the bin's size, added in order from zero
      # (+ 0.0: a sum from +0 is never -0).
      reduced = np.cumsum(block / block.shape[1], axis=1)[:, -1] + 0.0
      offset = np.zeros(block.shape[0])
    else:
      pick = np.argmin if reduction == 'min' else np.argmax
      offset = pick(block, axis=1)
      reduced = np.take_along_axis(block, offset[:, None], axis=1)[:, 0]
      zeros = block == 0
      signed = zeros & np.signbit(block)
      # Math.min prefers -0 to +0, Math.max +0 to -0.
      if reduction == 'min':
        zero = np.where(signed.any(axis=1), -0.0, 0.0)
      else:
        zero = np.where((zeros & ~signed).any(axis=1), 0.0, -0.0)
      reduced = np.where(reduced == 0, zero, reduced)
    values_out[rows] = np.where(valid, reduced, 0.0)
    offsets[rows] = np.where(valid, offset, np.nan)
  return values_out, offsets


def _paired(
    ref: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """The paired elements' indices, and their reference and target values."""
  indices = np.flatnonzero(np.isfinite(ref) & np.isfinite(target))
  return indices, ref[indices], target[indices]


def scatter_extent(ref: np.ndarray, target: np.ndarray) -> dict[str, float]:
  """The scatter plot's full value range: the paired values, padded 5%.

  `{start: -1, end: 1}` when no element is paired.

  Args:
      ref: The reference values by index, float64, NaN where missing.
      target: The target's, likewise.

  Returns:
      `{start, end}`: the padded value range.
  """
  _, refs, targets = _paired(ref, target)
  if not refs.size:
    return {'start': -1.0, 'end': 1.0}
  low = float(min(refs.min(), targets.min()))
  high = float(max(refs.max(), targets.max()))
  pad = ((high - low) or abs(low) or 1) * 0.05
  return {'start': low - pad, 'end': high + pad}


def paired_runs(ref: np.ndarray, target: np.ndarray) -> list[list[int]]:
  """The paired elements' indices, as increasing half-open runs."""
  paired = np.isfinite(ref) & np.isfinite(target)
  edges = np.flatnonzero(np.diff(np.r_[False, paired, False]))
  return edges.reshape(-1, 2).tolist()


def scatter_points(
    ref: np.ndarray,
    target: np.ndarray,
    start: float,
    end: float,
    limit: int = SCATTER_POINT_LIMIT,
    comparison_allowed: bool = True,
) -> dict[str, Any]:
  """The paired elements a scatter plot of [start, end] draws.

  Every paired element when there are `limit` or fewer; else those inside
  the value range on both axes, if `limit` or fewer, or else one per
  occupied cell of a grid of floor(sqrt(limit)) cells a side over the
  range: the one whose values differ most (when the runtimes may be
  compared), then the lowest index. A selected element is not added: the
  UI adds it (scatter_model.ts, withSelected).

  Args:
      ref: The reference values by index, float64, NaN where missing.
      target: The target's, likewise.
      start: The low end of the value range on both axes.
      end: The high end of the value range on both axes.
      limit: The most points drawn before they are grouped.
      comparison_allowed: Whether the runtimes may be compared, so a cell draws
        the element whose values differ most.

  Returns:
      `{indices, ref, target, paired, visible, grouped}`: the drawn
      elements by increasing index with their values; how many elements
      are paired, and inside the range; whether the points were grouped.
  """
  indices, refs, targets = _paired(ref, target)
  inside = (
      (refs >= start) & (refs <= end) & (targets >= start) & (targets <= end)
  )
  visible = int(inside.sum())
  grouped = visible > limit
  if indices.size <= limit:
    chosen = np.arange(indices.size)
  elif not grouped:
    chosen = np.flatnonzero(inside)
  else:
    chosen = np.flatnonzero(inside)
    cells = max(1, math.floor(math.sqrt(limit)))
    span = end - start

    def cell(values: np.ndarray) -> np.ndarray:
      return np.clip(
          np.floor(((values - start) / span) * cells), 0, cells - 1
      ).astype(np.int64)

    keys = cell(refs[chosen]) * cells + cell(targets[chosen])
    score = (
        np.abs(targets[chosen] - refs[chosen])
        if comparison_allowed
        else np.zeros(chosen.size)
    )
    order = np.lexsort((chosen, -score, keys))
    first = np.r_[True, np.diff(keys[order]) != 0]
    chosen = np.sort(chosen[order[first]])
  return {
      'indices': indices[chosen].tolist(),
      'ref': refs[chosen].tolist(),
      'target': targets[chosen].tolist(),
      'paired': int(indices.size),
      'visible': visible,
      'grouped': grouped,
  }
