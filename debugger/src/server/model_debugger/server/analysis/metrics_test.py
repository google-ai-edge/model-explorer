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

"""Tests of the vector metrics against hand-computed values."""

import math
from typing import Any

from absl.testing import parameterized
from model_debugger.server.analysis import metrics
from model_debugger.server.testing import test_main
import numpy as np


def _vector(*values: float) -> np.ndarray:
  return np.array(values, dtype=np.float64)


def _reasons(values: dict[str, dict[str, Any]]) -> set[tuple[str, str]]:
  """Returns the (code, text) of every reason of every value."""
  reasons = set()
  for value in values.values():
    for reason in value['reasons']:
      reasons.add((reason['code'], reason['text']))
  return reasons


class MetricsTest(parameterized.TestCase):
  """Tests of the vector metrics against hand-computed values."""

  def test_json_number_spells_non_finite_values(self):
    """Non-finite numbers are spelled as JavaScript spells them."""
    self.assertIsNone(metrics.json_number(None))
    self.assertEqual(metrics.json_number(np.float32(1.5)), 1.5)
    self.assertEqual(metrics.json_number(math.nan), 'NaN')
    self.assertEqual(metrics.json_number(math.inf), 'Infinity')
    self.assertEqual(metrics.json_number(-math.inf), '-Infinity')

  def test_distribution_of_two_logits(self):
    """softmax, ranks, entropy and margin of [0, ln 3]."""
    # softmax([0, ln 3]) = [1/4, 3/4].
    distribution = metrics.distribution(_vector(0, math.log(3)))
    np.testing.assert_allclose(distribution.p, [0.25, 0.75])
    np.testing.assert_allclose(distribution.logp, np.log([0.25, 0.75]))
    self.assertEqual(distribution.order.tolist(), [1, 0])
    self.assertEqual(distribution.ranks.tolist(), [2, 1])
    entropy = -(0.25 * math.log2(0.25) + 0.75 * math.log2(0.75))
    np.testing.assert_allclose(distribution.entropy, entropy, rtol=1e-6)
    np.testing.assert_allclose(distribution.margin, 0.5, rtol=1e-6)

  def test_distribution_ranks_ties_by_id(self):
    """Equal probabilities rank by id; -Infinity is probability 0."""
    distribution = metrics.distribution(_vector(0, 1, 1, -math.inf))
    self.assertEqual(distribution.order.tolist(), [1, 2, 0, 3])
    self.assertEqual(distribution.ranks.tolist(), [3, 1, 2, 4])
    self.assertEqual(distribution.margin, 0.0)
    # -Infinity is a zero probability, which adds nothing to the entropy.
    self.assertEqual(distribution.p[3], 0.0)
    self.assertTrue(math.isfinite(distribution.entropy))

  def test_distribution_of_one_entry_has_no_margin(self):
    """A one-entry vocabulary has no second id to compare with."""
    distribution = metrics.distribution(_vector(7))
    self.assertEqual(distribution.p.tolist(), [1.0])
    self.assertEqual(distribution.entropy, 0.0)
    self.assertIsNone(distribution.margin)

  @parameterized.parameters(
      ((0, math.nan),),
      ((0, math.inf),),
      ((-math.inf, -math.inf),),
  )
  def test_distribution_refuses_invalid_logits(self, logits):
    """NaN, +Infinity or nothing finite is refused."""
    with self.assertRaisesRegex(ValueError, 'Logits contain invalid values'):
      metrics.distribution(_vector(*logits))

  def test_kl_divergence_in_nats(self):
    """KL divergence in nats, by hand."""
    p = metrics.distribution(_vector(0, math.log(3)))
    q = metrics.distribution(_vector(0, 0))
    expected = 0.25 * math.log(0.25 / 0.5) + 0.75 * math.log(0.75 / 0.5)
    np.testing.assert_allclose(metrics.kl_divergence(p, q), expected, rtol=1e-6)
    self.assertEqual(metrics.kl_divergence(p, p), 0.0)

  def test_kl_divergence_is_infinite_when_target_rules_out_an_id(self):
    """KL is infinite where the target has zero probability."""
    p = metrics.distribution(_vector(0, math.log(3)))
    q = metrics.distribution(_vector(0, -math.inf))
    self.assertEqual(metrics.kl_divergence(p, q), math.inf)

  def test_js_divergence_in_bits(self):
    """Jensen-Shannon divergence in bits, by hand."""
    p = metrics.distribution(_vector(0, math.log(3)))
    q = metrics.distribution(_vector(0, 0))
    m = (0.375, 0.625)
    expected = 0.5 * (
        0.25 * math.log2(0.25 / m[0]) + 0.75 * math.log2(0.75 / m[1])
    ) + 0.5 * (0.5 * math.log2(0.5 / m[0]) + 0.5 * math.log2(0.5 / m[1]))
    np.testing.assert_allclose(metrics.js_divergence(p, q), expected, rtol=1e-6)
    self.assertEqual(metrics.js_divergence(p, p), 0.0)
    # Disjoint supports are exactly one bit apart.
    one = metrics.distribution(_vector(0, -math.inf))
    other = metrics.distribution(_vector(-math.inf, 0))
    np.testing.assert_allclose(
        metrics.js_divergence(one, other), 1.0, rtol=1e-6
    )

  def test_compare_by_hand(self):
    """Every metric of two vectors, by hand."""
    # ref [1, 2, 2, 0], target [1, 2, 2, 4]: |ref| 3, |target| 5, one
    # channel 4 apart.
    comparison = metrics.compare(_vector(1, 2, 2, 0), _vector(1, 2, 2, 4))
    expected = {
        'cosine_similarity': 0.6,
        'cosine_distance': 0.4,
        'relative_l2': 400 / 3,
        'norm_ratio': 5 / 3,
        'max_abs': 4.0,
        'mean_abs': 1.0,
        'rmse': 2.0,
    }
    self.assertEqual(set(comparison.values), set(metrics.VECTOR_KEYS))
    for key, value in expected.items():
      np.testing.assert_allclose(
          comparison.values[key], value, rtol=1e-12, err_msg=key
      )
      self.assertEqual(comparison.status[key].item(), metrics.OK, key)

  def test_compare_over_some_axes(self):
    """The axes left are the groups, one value per group."""
    ref = np.zeros((2, 3, 4))
    target = np.zeros((2, 3, 4))
    ref[:, 1, :] = 1
    target[:, 1, :] = 1
    target[0, 1, 3] = 3
    comparison = metrics.compare(ref, target, axes=(0, 2))
    self.assertEqual(comparison.values['max_abs'].tolist(), [0, 2, 0])
    # Group 1: |ref| sqrt(8), |target - ref| 2.
    np.testing.assert_allclose(
        comparison.values['relative_l2'][1], 100 * 2 / math.sqrt(8)
    )
    np.testing.assert_allclose(comparison.values['rmse'][1], math.sqrt(4 / 8))
    # Groups 0 and 2 are zero on both sides: no norm, no ratios.
    self.assertEqual(
        comparison.status['cosine_similarity'].tolist(),
        [metrics.ZERO_NORM, metrics.OK, metrics.ZERO_NORM],
    )
    self.assertTrue(np.isnan(comparison.values['relative_l2'][0]))
    self.assertEqual(comparison.values['rmse'][0], 0)

  def test_compare_equal_vectors_are_exactly_similar(self):
    """Equal vectors are exactly 1 similar, and 0 distant, at any scale."""
    for values in ((0.1, 0.2, 0.3), (1e-150, 3e-150), (1e300, -1e300)):
      comparison = metrics.compare(_vector(*values), _vector(*values))
      self.assertEqual(comparison.values['cosine_similarity'], 1.0)
      self.assertEqual(comparison.values['cosine_distance'], 0.0)
      self.assertEqual(comparison.values['relative_l2'], 0.0)

  def test_compare_clips_cosine(self):
    """Similarity stays within [-1, 1], distance within [0, 2]."""
    comparison = metrics.compare(
        _vector(0.1, 0.2, 0.7), _vector(-0.1, -0.2, -0.7)
    )
    self.assertEqual(comparison.values['cosine_similarity'], -1.0)
    self.assertEqual(comparison.values['cosine_distance'], 2.0)

  def test_compare_scales_before_squaring(self):
    """Huge finite values are compared without overflow."""
    comparison = metrics.compare(_vector(1e300, 1e300), _vector(1e300, -1e300))
    np.testing.assert_allclose(
        comparison.values['relative_l2'], 100 * math.sqrt(2)
    )
    np.testing.assert_allclose(comparison.values['rmse'], 2e300 / 2**0.5)
    self.assertEqual(comparison.values['cosine_similarity'], 0.0)
    self.assertEqual(
        {code.item() for code in comparison.status.values()}, {metrics.OK}
    )

  def test_compare_keeps_full_precision_on_near_equal_vectors(self):
    """A 1e-12 relative change is measured to float64 precision."""
    rng = np.random.default_rng(0)
    ref = rng.normal(size=256) * 3.7
    target = ref * (1 + 1e-12 * rng.normal(size=256))
    # target - ref is exact for values this close.
    expected = 100 * np.linalg.norm(target - ref) / np.linalg.norm(ref)
    np.testing.assert_allclose(
        metrics.compare(ref, target).values['relative_l2'],
        expected,
        rtol=1e-9,
    )

  def test_compare_mean_abs_does_not_overflow(self):
    """A mean of finite differences is finite, however large they are."""
    comparison = metrics.compare(np.zeros(4), np.full(4, 1e308))
    self.assertEqual(comparison.status['mean_abs'].item(), metrics.OK)
    np.testing.assert_allclose(comparison.values['mean_abs'], 1e308)

  def test_compare_scaling_changes_nothing_that_cannot_overflow(self):
    """Values that cannot overflow get exactly their unscaled metrics."""
    rng = np.random.default_rng(2)
    ref = rng.normal(size=(64, 96)) * 3.7
    target = ref + rng.normal(size=ref.shape) * 1e-2
    comparison = metrics.compare(ref, target, axes=(1,))
    difference = np.abs(target - ref)
    np.testing.assert_array_equal(
        comparison.values['mean_abs'], difference.mean(axis=-1)
    )
    np.testing.assert_array_equal(
        comparison.values['rmse'],
        np.sqrt(np.mean(difference**2, axis=-1)),
    )
    # A mean that dividing by the largest difference would round.
    comparison = metrics.compare(
        np.zeros(3), _vector(0.0010333333333333334, 2.1786541666666666, 1.5)
    )
    self.assertEqual(comparison.values['mean_abs'], 1.2265625)
    # The largest float64 against its negation: exactly 200 percent.
    largest = np.finfo(np.float64).max
    comparison = metrics.compare(_vector(largest), _vector(-largest))
    self.assertEqual(comparison.values['relative_l2'], 200.0)
    self.assertEqual(comparison.values['norm_ratio'], 1.0)

  def test_compare_computes_the_keys_asked_for_only(self):
    """Each metric asked for is as it is when every one is."""
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(2, 3, 5, 8)).astype(np.float32)
    target = (ref + rng.normal(size=ref.shape) * 1e-3).astype(np.float32)
    ref[0, 1, 2] = 0
    every = metrics.compare(ref, target, axes=(0, 3))
    for keys in (
        ('relative_l2', 'max_abs', 'cosine_distance', 'cosine_similarity'),
        ('norm_ratio',),
        ('mean_abs', 'rmse'),
    ):
      some = metrics.compare(ref, target, axes=(0, 3), keys=keys)
      self.assertEqual(tuple(some.values), keys)
      for key in keys:
        np.testing.assert_array_equal(
            some.values[key], every.values[key], err_msg=key
        )
        np.testing.assert_array_equal(
            some.status[key], every.status[key], err_msg=key
        )
    with self.assertRaisesRegex(ValueError, 'Unknown metrics: kl'):
      metrics.compare(ref, target, keys=('kl',))

  def test_compare_refuses_non_finite_values(self):
    """A non-finite value leaves every metric of its group missing."""
    comparison = metrics.compare(
        np.array([[1.0, 2.0], [1.0, math.nan]]),
        np.array([[1.0, 2.0], [1.0, 2.0]]),
        axes=(1,),
    )
    for key in metrics.VECTOR_KEYS:
      self.assertEqual(
          comparison.status[key].tolist(),
          [metrics.OK, metrics.NON_FINITE],
          key,
      )
      self.assertTrue(np.isnan(comparison.values[key][1]), key)
    with self.assertRaisesRegex(ValueError, 'shape_mismatch'):
      metrics.compare(_vector(1), _vector(1, 2))

  def test_compare_reports_a_difference_beyond_float64(self):
    """A difference beyond float64 is an overflow, not infinity."""
    comparison = metrics.compare(_vector(-1e308), _vector(1e308))
    self.assertEqual(comparison.status['max_abs'].item(), metrics.OVERFLOW)
    self.assertEqual(comparison.status['rmse'].item(), metrics.OVERFLOW)
    self.assertEqual(comparison.status['cosine_similarity'].item(), metrics.OK)

  def test_activation_metrics_of_equal_vectors(self):
    """Equal activations are zero apart."""
    values = metrics.activation_metrics(_vector(3, 4), _vector(3, 4))
    self.assertEqual(values['relative_l2'], {'value': 0.0, 'reasons': []})
    np.testing.assert_allclose(values['norm_ratio']['value'], 1.0, rtol=1e-6)
    np.testing.assert_allclose(
        values['cosine_distance']['value'], 0.0, atol=1e-15
    )
    self.assertEqual(values['max_abs'], {'value': 0.0, 'reasons': []})

  def test_activation_metrics_by_hand(self):
    """Activation metrics of two orthogonal vectors, by hand."""
    # |b - a| / |a| = sqrt(4 + 16) / 2, in percent; orthogonal vectors.
    values = metrics.activation_metrics(_vector(2, 0), _vector(0, 4))
    np.testing.assert_allclose(
        values['relative_l2']['value'], 100 * math.sqrt(5), rtol=1e-6
    )
    np.testing.assert_allclose(values['norm_ratio']['value'], 2.0, rtol=1e-6)
    np.testing.assert_allclose(
        values['cosine_distance']['value'], 1.0, rtol=1e-6
    )
    np.testing.assert_allclose(values['max_abs']['value'], 4.0, rtol=1e-6)
    opposite = metrics.activation_metrics(_vector(1, 1), _vector(-1, -1))
    np.testing.assert_allclose(
        opposite['cosine_distance']['value'], 2.0, rtol=1e-6
    )

  def test_activation_metrics_of_large_values_do_not_overflow(self):
    """Scaling keeps huge finite values finite."""
    values = metrics.activation_metrics(_vector(1e200, 0), _vector(0, 1e200))
    np.testing.assert_allclose(
        values['relative_l2']['value'], 100 * math.sqrt(2), rtol=1e-6
    )
    np.testing.assert_allclose(values['max_abs']['value'], 1e200, rtol=1e-6)

  def test_activation_metrics_with_zero_reference(self):
    """A zero ref norm leaves the ratios undefined."""
    values = metrics.activation_metrics(_vector(0, 0), _vector(1, 0))
    reason = metrics.reason(
        metrics.ZERO_NORM, 'Zero reference norm or non-finite result'
    )
    for key in (
        'relative_l2',
        'norm_ratio',
        'cosine_similarity',
        'cosine_distance',
    ):
      self.assertEqual(values[key], {'value': None, 'reasons': [reason]})
    self.assertEqual(values['max_abs']['value'], 1.0)

  def test_activation_metrics_refuse_shapes_and_non_finite_values(self):
    """Different shapes and NaN make every metric missing."""
    values = metrics.activation_metrics(_vector(1, 2), _vector(1, 2, 3))
    self.assertEqual(
        _reasons(values),
        {(metrics.SHAPE_MISMATCH, 'Activation shapes differ')},
    )
    values = metrics.activation_metrics(_vector(1, math.nan), _vector(1, 2))
    self.assertEqual(
        _reasons(values),
        {(metrics.NON_FINITE, 'Activation contains non-finite values')},
    )
    self.assertEqual(set(values), set(metrics.ACTIVATION_KEYS))

  def test_logits_errors_by_hand(self):
    """RMSE and largest error of two logits vectors, by hand."""
    errors = metrics.logits_errors(_vector(0, 0, 0, 0), _vector(1, -3, 0, 3))
    self.assertEqual(errors['max_abs'], {'value': 3.0, 'reasons': []})
    # The first of equal largest errors.
    self.assertEqual(errors['max_abs_index'], 1)
    np.testing.assert_allclose(
        errors['rmse']['value'], math.sqrt(19 / 4), rtol=1e-6
    )
    equal = metrics.logits_errors(_vector(1, 2), _vector(1, 2))
    self.assertEqual(
        equal,
        {
            'rmse': {'value': 0.0, 'reasons': []},
            'max_abs': {'value': 0.0, 'reasons': []},
            'max_abs_index': 0,
        },
    )

  def test_logits_errors_refuse_what_they_cannot_compute(self):
    """Shapes, non-finite inputs and overflow are refused."""
    self.assertEqual(
        metrics.logits_errors(_vector(1), _vector(1, 2)),
        metrics.missing_logits_errors(
            [metrics.reason(metrics.SHAPE_MISMATCH, 'Logits shapes differ')]
        ),
    )
    self.assertEqual(
        metrics.logits_errors(_vector(1, -math.inf), _vector(1, 2)),
        metrics.missing_logits_errors([
            metrics.reason(
                metrics.NON_FINITE,
                'Logits contain non-finite values; full-vector '
                'metrics are unavailable',
            )
        ]),
    )
    self.assertEqual(
        metrics.logits_errors(_vector(-1e308), _vector(1e308)),
        metrics.missing_logits_errors([
            metrics.reason(
                metrics.OVERFLOW,
                'Logits differences exceed finite float64 values',
            )
        ]),
    )

  def test_centered(self):
    """Centering subtracts the mean, or says why it cannot."""
    values, reason = metrics.centered(_vector(1, 2, 6))
    np.testing.assert_allclose(values, [-2, -1, 3])
    self.assertIsNone(reason)
    self.assertEqual(metrics.centered(_vector(0, 0))[0].tolist(), [0.0, 0.0])
    self.assertEqual(
        metrics.centered(_vector(1, math.inf)),
        (
            None,
            metrics.reason(
                metrics.NON_FINITE,
                'Centered logits unavailable: source contains non-finite '
                'values',
            ),
        ),
    )


if __name__ == '__main__':
  test_main.main()
