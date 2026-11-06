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

"""Statistics of values at positions: what Statistics tables and bins show.

This is the one computation of them. A statistic is the UI's
`KvRangeMetricSummary`: the finite values' `min`, `max` and `mean`, the
position of each extremum (the first, on ties) and how many values were
finite (`valid_count`). A value that is NaN or infinite is left out; a bin
without a finite value has None for each statistic and a count of 0. The
positions are whatever the values are of: a vocabulary index, a head
dimension, a token's column.
"""

from typing import Any

import numpy as np


def empty() -> dict[str, Any]:
  """The statistic of no finite value.

  Returns:
      The statistic: every value None and a `valid_count` of 0.
  """
  return {
      'min': None,
      'max': None,
      'mean': None,
      'min_position': None,
      'max_position': None,
      'valid_count': 0,
  }


def binned(
    values: np.ndarray, positions: np.ndarray, bin_edges: np.ndarray
) -> list[dict[str, Any]]:
  """The statistic of the values of each bin of positions.

  Args:
      values: The values, float64.
      positions: The position of each value, in order: they never decrease.
      bin_edges: The bins' increasing edges: bin i holds the positions from
        `bin_edges[i]` up to, not including, `bin_edges[i + 1]`.

  Returns:
      One statistic per bin, in order.
  """
  count = len(bin_edges) - 1
  values = np.asarray(values, dtype=np.float64)
  positions = np.asarray(positions)
  bin_ids = np.searchsorted(bin_edges, positions, side='right') - 1
  keep = np.isfinite(values) & (bin_ids >= 0) & (bin_ids < count)
  values, positions, bin_ids = values[keep], positions[keep], bin_ids[keep]
  if not values.size:
    return [empty() for _ in range(count)]
  # Positions never decrease, so each bin's values are one segment.
  starts = np.flatnonzero(np.r_[True, np.diff(bin_ids) != 0])
  counts = np.diff(np.r_[starts, values.size])
  # Dividing before summing keeps a finite mean from overflowing.
  means = np.add.reduceat(values / np.repeat(counts, counts), starts)
  columns = {'mean': means.tolist(), 'valid_count': counts.tolist()}
  for name, reduce in (('min', np.minimum), ('max', np.maximum)):
    extreme = reduce.reduceat(values, starts)
    hits = np.flatnonzero(values == np.repeat(extreme, counts))
    segments = np.searchsorted(starts, hits, side='right') - 1
    _, first = np.unique(segments, return_index=True)
    columns[name] = extreme.tolist()
    columns[name + '_position'] = positions[hits[first]].tolist()
  result = [None] * count
  for i, owner in enumerate(bin_ids[starts].tolist()):
    result[owner] = {key: column[i] for key, column in columns.items()}
  return [statistic or empty() for statistic in result]


def summary(
    values: np.ndarray, positions: np.ndarray | None = None
) -> dict[str, Any]:
  """The statistic of all the values.

  Args:
      values: The values, float64.
      positions: The position of each value, never decreasing; None: each
        value's index.

  Returns:
      The statistic of the finite values, or `empty()` when there are none.
  """
  values = np.asarray(values, dtype=np.float64).reshape(-1)
  if positions is None:
    positions = np.arange(values.size)
  if not values.size:
    return empty()
  bin_edges = np.array([positions[0], positions[-1] + 1])
  return binned(values, positions, bin_edges)[0]


def edges(start: int, end: int, bins: int) -> np.ndarray:
  """The edges of `bins` bins of [start, end), as even as integers allow.

  Fewer bins when the range holds fewer positions: each bin holds one or
  more.

  Args:
      start: The first position of the range.
      end: The end of the range, not included.
      bins: The most bins to make.

  Returns:
      The increasing integer edges, one more than the bins.
  """
  count = max(1, min(bins, end - start))
  return np.array(
      [start + i * (end - start) // count for i in range(count + 1)]
  )
