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

"""Tests of the statistics against hand-computed values."""

from absl.testing import absltest
from model_debugger.server.analysis import statistics
from model_debugger.server.testing import test_main
import numpy as np


class StatisticsTest(absltest.TestCase):
  """Tests of the statistics against hand-computed values."""

  def test_summary_of_finite_values(self):
    """Minimum, maximum and mean, each extremum at its first position."""
    self.assertEqual(
        statistics.summary(np.array([3.0, -1.0, 5.0, -1.0, 5.0])),
        {
            'min': -1.0,
            'max': 5.0,
            'mean': 2.2,
            'min_position': 1,
            'max_position': 2,
            'valid_count': 5,
        },
    )

  def test_summary_leaves_out_values_that_are_not_finite(self):
    """NaN and infinities are neither counted nor an extremum."""
    stat = statistics.summary(
        np.array([np.nan, 2.0, np.inf, -np.inf, 4.0]),
        np.array([10, 11, 12, 13, 14]),
    )
    self.assertEqual(stat['min'], 2.0)
    self.assertEqual(stat['min_position'], 11)
    self.assertEqual(stat['max_position'], 14)
    self.assertEqual(stat['mean'], 3.0)
    self.assertEqual(stat['valid_count'], 2)

  def test_summary_without_a_finite_value_is_empty(self):
    """No finite value: no statistic, a count of zero."""
    self.assertEqual(statistics.summary(np.array([np.nan])), statistics.empty())
    self.assertEqual(statistics.summary(np.array([])), statistics.empty())

  def test_mean_of_huge_values_does_not_overflow(self):
    """The mean divides before it sums."""
    stat = statistics.summary(np.array([1e308, 1e308]))
    self.assertEqual(stat['mean'], 1e308)

  def test_binned_by_positions(self):
    """Each bin of positions has its own statistic; an empty one none."""
    values = np.array([1.0, 2.0, 7.0, 7.0, np.nan])
    positions = np.array([0, 1, 4, 4, 6])
    stats = statistics.binned(values, positions, np.array([0, 2, 4, 6, 8]))
    self.assertEqual(
        [(s['min'], s['max'], s['valid_count']) for s in stats],
        [(1.0, 2.0, 2), (None, None, 0), (7.0, 7.0, 2), (None, None, 0)],
    )
    self.assertEqual(stats[0]['mean'], 1.5)
    self.assertEqual(stats[2]['max_position'], 4)

  def test_positions_outside_the_bins_are_left_out(self):
    """Only positions within the edges count."""
    stats = statistics.binned(
        np.array([9.0, 1.0, 9.0]), np.array([0, 5, 9]), np.array([2, 6])
    )
    self.assertEqual(stats[0]['valid_count'], 1)
    self.assertEqual(stats[0]['max'], 1.0)

  def test_edges_split_a_range_evenly(self):
    """Edges as even as integers allow, and no empty bin."""
    np.testing.assert_array_equal(statistics.edges(16, 25, 3), [16, 19, 22, 25])
    np.testing.assert_array_equal(statistics.edges(0, 10, 4), [0, 2, 5, 7, 10])
    np.testing.assert_array_equal(statistics.edges(3, 5, 9), [3, 4, 5])


if __name__ == '__main__':
  test_main.main()
