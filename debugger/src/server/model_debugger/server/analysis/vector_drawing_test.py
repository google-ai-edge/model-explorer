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

"""Tests of what the charts draw, against hand-computed values."""

import math

from absl.testing import absltest
from model_debugger.server.analysis import vector_drawing
from model_debugger.server.testing import test_main
import numpy as np

NAN = math.nan


def _bins(ref, target, start, end, capacity, reduction):
  return vector_drawing.line_bins(
      np.array(ref, dtype=np.float64),
      np.array(target, dtype=np.float64),
      start,
      end,
      capacity,
      reduction,
  )


class LineBinsTest(absltest.TestCase):
  """Tests of the line chart's bins."""

  def test_bin_size_and_range(self):
    """Bins of ceil(length / capacity), the last one shorter."""
    self.assertEqual(vector_drawing.bin_size(0, 10, 3), 4)
    self.assertEqual(vector_drawing.bin_size(5, 6, 128), 1)
    self.assertEqual(vector_drawing.bin_size(0, 10, 0), 10)
    self.assertEqual(vector_drawing.clamped_range(3, 99, 10), (3, 10))
    self.assertEqual(vector_drawing.clamped_range(12, 20, 10), (9, 10))
    self.assertEqual(vector_drawing.clamped_range(0, 5, 0), (0, 1))

  def test_reductions_and_representatives(self):
    """Each runtime's minimum, maximum or mean, and the element chosen."""
    ref = [1.0, 3.0, 3.0, 2.0, 5.0, -1.0, 4.0]
    target = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 9.0]
    self.assertEqual(
        _bins(ref, target, 0, 7, 3, 'max'),
        {
            'start': 0,
            'end': 7,
            'size': 3,
            'ref': [3.0, 5.0, 4.0],
            # The first maximum on ties.
            'ref_index': [1, 4, 6],
            'target': [1.0, 2.0, 9.0],
            'target_index': [2, 4, 6],
        },
    )
    low = _bins(ref, target, 0, 7, 3, 'min')
    self.assertEqual(low['ref'], [1.0, -1.0, 4.0])
    self.assertEqual(low['ref_index'], [0, 5, 6])
    self.assertEqual(low['target_index'], [0, 3, 6])
    mean = _bins(ref, target, 1, 7, 3, 'mean')
    self.assertEqual(mean['ref'], [3.0, 3.5, 1.5])
    # A mean's element is the bin's first.
    self.assertEqual(mean['ref_index'], [1, 3, 5])

  def test_a_value_that_is_not_valid_makes_a_gap(self):
    """NaN or an infinity: no value and no element, the other side kept."""
    ref = [1.0, NAN, 2.0, math.inf, 3.0]
    target = [1.0, 1.0, 2.0, 2.0, 3.0]
    bins = _bins(ref, target, 0, 5, 3, 'mean')
    self.assertEqual(bins['ref'], [None, None, 3.0])
    self.assertEqual(bins['ref_index'], [None, None, 4])
    self.assertEqual(bins['target'], [1.0, 2.0, 3.0])

  def test_mean_adds_each_value_over_the_size_in_order(self):
    """sum(v / n) from zero, in index order, as the UI added it."""
    values = [0.1, 0.2, 0.3, 1e16, -1e16, 0.7]
    expected = 0.0
    for value in values:
      expected = expected + value / len(values)
    bins = _bins(values, values, 0, 6, 1, 'mean')
    self.assertEqual(bins['ref'], [expected])
    self.assertNotEqual(expected, math.fsum(values) / len(values))

  def test_signed_zeros(self):
    """-0 is the minimum of zeros holding one; +0 the maximum."""
    ref = [0.0, -0.0, -0.0, 0.0]
    bins = _bins(ref, ref, 0, 4, 2, 'min')
    self.assertEqual([math.copysign(1, v) for v in bins['ref']], [-1, -1])
    bins = _bins(ref, ref, 0, 4, 2, 'max')
    self.assertEqual([math.copysign(1, v) for v in bins['ref']], [1, 1])
    only_negative = _bins([-0.0, -0.0], [-0.0, -0.0], 0, 2, 1, 'max')
    self.assertEqual(math.copysign(1, only_negative['ref'][0]), -1)
    # A sum from +0 is never -0.
    mean = _bins([-0.0, -0.0], [-0.0, -0.0], 0, 2, 1, 'mean')
    self.assertEqual(math.copysign(1, mean['ref'][0]), 1)


class ScatterTest(absltest.TestCase):
  """Tests of the scatter plot's extent, points and runs."""

  def test_extent(self):
    """The paired values, padded by 5% of their span."""
    ref = np.array([1.0, 2.0, NAN, 5.0])
    target = np.array([0.0, 3.0, 9.0, 4.0])
    extent = vector_drawing.scatter_extent(ref, target)
    self.assertEqual(extent, {'start': 0.0 - 0.25, 'end': 5.0 + 0.25})
    same = vector_drawing.scatter_extent(np.array([2.0]), np.array([2.0]))
    self.assertEqual(same, {'start': 2.0 - 0.1, 'end': 2.0 + 0.1})
    zero = vector_drawing.scatter_extent(np.array([0.0]), np.array([0.0]))
    self.assertEqual(zero, {'start': -0.05, 'end': 0.05})
    none = vector_drawing.scatter_extent(np.array([NAN]), np.array([1.0]))
    self.assertEqual(none, {'start': -1.0, 'end': 1.0})

  def test_paired_runs(self):
    """The indices with both values valid, as half-open runs."""
    ref = np.array([1.0, NAN, 1.0, 1.0, 1.0, 1.0])
    target = np.array([1.0, 1.0, 1.0, math.inf, 1.0, 1.0])
    self.assertEqual(
        vector_drawing.paired_runs(ref, target), [[0, 1], [2, 3], [4, 6]]
    )

  def test_few_points_are_all_drawn(self):
    """At most `limit` paired elements: all, inside the range or not."""
    ref = np.array([0.0, 5.0, NAN, 1.0])
    target = np.array([0.0, 5.0, 1.0, 1.0])
    drawn = vector_drawing.scatter_points(ref, target, 0.5, 1.5, limit=4)
    self.assertEqual(drawn['indices'], [0, 1, 3])
    self.assertEqual(
        (drawn['paired'], drawn['visible'], drawn['grouped']),
        (3, 1, False),
    )

  def test_visible_points_when_few_are_inside(self):
    """More paired than `limit`, few inside: those inside."""
    ref = np.arange(10.0)
    drawn = vector_drawing.scatter_points(ref, ref, 2.0, 4.0, limit=4)
    self.assertEqual(drawn['indices'], [2, 3, 4])
    self.assertEqual(drawn['ref'], [2.0, 3.0, 4.0])
    self.assertFalse(drawn['grouped'])

  def test_grouped_points_keep_the_largest_difference(self):
    """One element per grid cell: largest |delta|, then lowest index."""
    # limit 4: a 2 x 2 grid over [0, 4]; cells split at 2.
    ref = np.array([0.5, 0.5, 0.5, 3.0, 3.5, 0.5, 9.0])
    target = np.array([0.5, 1.0, 1.0, 3.0, 3.5, 3.5, 9.0])
    drawn = vector_drawing.scatter_points(ref, target, 0.0, 4.0, limit=4)
    # Cell (0, 0): 0, 1 and 2, of |delta| 0, 0.5 and 0.5: index 1.
    # Cell (1, 1): 3 and 4, both 0: index 3. Cell (0, 1): 5.
    self.assertEqual(drawn['indices'], [1, 3, 5])
    self.assertEqual(
        (drawn['paired'], drawn['visible'], drawn['grouped']),
        (7, 6, True),
    )
    same = vector_drawing.scatter_points(
        ref, target, 0.0, 4.0, limit=4, comparison_allowed=False
    )
    # Without a comparison, the lowest index of each cell.
    self.assertEqual(same['indices'], [0, 3, 5])

  def test_a_value_at_the_range_end_is_in_the_last_cell(self):
    """The grid's last cell holds the range's end."""
    ref = np.array([4.0, 4.0, 0.0, 0.0, 1.0])
    target = np.array([4.0, 3.0, 0.0, 0.1, 1.0])
    drawn = vector_drawing.scatter_points(ref, target, 0.0, 4.0, limit=4)
    self.assertEqual(drawn['indices'], [1, 3])


if __name__ == '__main__':
  test_main.main()
