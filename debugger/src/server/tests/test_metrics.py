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

from model_debugger_server.analysis import metrics
import numpy as np
import pytest


def _vector(*values: float) -> np.ndarray:
  return np.array(values, dtype=np.float64)


def test_json_number_spells_non_finite_values():
  """Non-finite numbers are spelled as JavaScript spells them."""
  assert metrics.json_number(None) is None
  assert metrics.json_number(np.float32(1.5)) == 1.5
  assert metrics.json_number(math.nan) == 'NaN'
  assert metrics.json_number(math.inf) == 'Infinity'
  assert metrics.json_number(-math.inf) == '-Infinity'
  assert metrics.json_numbers(_vector(1, 2)) == [1.0, 2.0]
  assert metrics.json_numbers(_vector(1, -math.inf, math.nan)) == [
      1.0,
      '-Infinity',
      'NaN',
  ]


def test_distribution_of_two_logits():
  """softmax, ranks, entropy and margin of [0, ln 3]."""
  # softmax([0, ln 3]) = [1/4, 3/4].
  distribution = metrics.distribution(_vector(0, math.log(3)))
  np.testing.assert_allclose(distribution.p, [0.25, 0.75])
  np.testing.assert_allclose(distribution.logp, np.log([0.25, 0.75]))
  assert distribution.order.tolist() == [1, 0]
  assert distribution.ranks.tolist() == [2, 1]
  entropy = -(0.25 * math.log2(0.25) + 0.75 * math.log2(0.75))
  assert distribution.entropy == pytest.approx(entropy)
  assert distribution.margin == pytest.approx(0.5)


def test_distribution_ranks_ties_by_id():
  """Equal probabilities rank by id; -Infinity is probability 0."""
  distribution = metrics.distribution(_vector(0, 1, 1, -math.inf))
  assert distribution.order.tolist() == [1, 2, 0, 3]
  assert distribution.ranks.tolist() == [3, 1, 2, 4]
  assert distribution.margin == 0.0
  # -Infinity is a zero probability, which adds nothing to the entropy.
  assert distribution.p[3] == 0.0
  assert math.isfinite(distribution.entropy)


def test_distribution_of_one_entry_has_no_margin():
  """A one-entry vocabulary has no second id to compare with."""
  distribution = metrics.distribution(_vector(7))
  assert distribution.p.tolist() == [1.0]
  assert distribution.entropy == 0.0
  assert distribution.margin is None


@pytest.mark.parametrize(
    'logits',
    [(0, math.nan), (0, math.inf), (-math.inf, -math.inf)],
)
def test_distribution_refuses_invalid_logits(logits):
  """NaN, +Infinity or nothing finite is refused."""
  with pytest.raises(ValueError, match='Logits contain invalid values'):
    metrics.distribution(_vector(*logits))


def test_kl_divergence_in_nats():
  """KL divergence in nats, by hand."""
  p = metrics.distribution(_vector(0, math.log(3)))
  q = metrics.distribution(_vector(0, 0))
  expected = 0.25 * math.log(0.25 / 0.5) + 0.75 * math.log(0.75 / 0.5)
  assert metrics.kl_divergence(p, q) == pytest.approx(expected)
  assert metrics.kl_divergence(p, p) == 0.0


def test_kl_divergence_is_infinite_when_target_rules_out_an_id():
  """KL is infinite where the target has zero probability."""
  p = metrics.distribution(_vector(0, math.log(3)))
  q = metrics.distribution(_vector(0, -math.inf))
  assert metrics.kl_divergence(p, q) == math.inf


def test_js_divergence_in_bits():
  """Jensen-Shannon divergence in bits, by hand."""
  p = metrics.distribution(_vector(0, math.log(3)))
  q = metrics.distribution(_vector(0, 0))
  m = (0.375, 0.625)
  expected = 0.5 * (
      0.25 * math.log2(0.25 / m[0]) + 0.75 * math.log2(0.75 / m[1])
  ) + 0.5 * (0.5 * math.log2(0.5 / m[0]) + 0.5 * math.log2(0.5 / m[1]))
  assert metrics.js_divergence(p, q) == pytest.approx(expected)
  assert metrics.js_divergence(p, p) == 0.0
  # Disjoint supports are exactly one bit apart.
  one = metrics.distribution(_vector(0, -math.inf))
  other = metrics.distribution(_vector(-math.inf, 0))
  assert metrics.js_divergence(one, other) == pytest.approx(1.0)


def test_activation_metrics_of_equal_vectors():
  """Equal activations are zero apart."""
  values = metrics.activation_metrics(_vector(3, 4), _vector(3, 4))
  assert values['relative_l2'] == {'value': 0.0, 'reason': None}
  assert values['norm_ratio']['value'] == pytest.approx(1.0)
  assert values['cosine_distance']['value'] == pytest.approx(0.0, abs=1e-15)
  assert values['max_abs_error'] == {'value': 0.0, 'reason': None}


def test_activation_metrics_by_hand():
  """Activation metrics of two orthogonal vectors, by hand."""
  # |b - a| / |a| = sqrt(4 + 16) / 2, in percent; orthogonal vectors.
  values = metrics.activation_metrics(_vector(2, 0), _vector(0, 4))
  assert values['relative_l2']['value'] == pytest.approx(100 * math.sqrt(5))
  assert values['norm_ratio']['value'] == pytest.approx(2.0)
  assert values['cosine_distance']['value'] == pytest.approx(1.0)
  assert values['max_abs_error']['value'] == pytest.approx(4.0)
  opposite = metrics.activation_metrics(_vector(1, 1), _vector(-1, -1))
  assert opposite['cosine_distance']['value'] == pytest.approx(2.0)


def test_activation_metrics_of_large_values_do_not_overflow():
  """Scaling keeps huge finite values finite."""
  values = metrics.activation_metrics(_vector(1e200, 0), _vector(0, 1e200))
  assert values['relative_l2']['value'] == pytest.approx(100 * math.sqrt(2))
  assert values['max_abs_error']['value'] == pytest.approx(1e200)


def test_activation_metrics_with_zero_reference():
  """A zero ref norm leaves the ratios undefined."""
  values = metrics.activation_metrics(_vector(0, 0), _vector(1, 0))
  reason = 'Zero reference norm or non-finite result'
  for key in ('relative_l2', 'norm_ratio', 'cosine_distance'):
    assert values[key] == {'value': None, 'reason': reason}
  assert values['max_abs_error']['value'] == 1.0


def test_activation_metrics_refuse_shapes_and_non_finite_values():
  """Different shapes and NaN make every metric missing."""
  values = metrics.activation_metrics(_vector(1, 2), _vector(1, 2, 3))
  assert {value['reason'] for value in values.values()} == {
      'Activation shapes differ'
  }
  values = metrics.activation_metrics(_vector(1, math.nan), _vector(1, 2))
  assert {value['reason'] for value in values.values()} == {
      'Activation contains non-finite values'
  }
  assert set(values) == set(metrics.ACTIVATION_KEYS)


def test_logits_errors_by_hand():
  """RMSE and largest error of two logits vectors, by hand."""
  errors = metrics.logits_errors(_vector(0, 0, 0, 0), _vector(1, -3, 0, 3))
  assert errors['max_abs'] == {'value': 3.0, 'reason': None}
  # The first of equal largest errors.
  assert errors['max_abs_index'] == 1
  assert errors['rmse']['value'] == pytest.approx(math.sqrt(19 / 4))
  equal = metrics.logits_errors(_vector(1, 2), _vector(1, 2))
  assert equal == {
      'rmse': {'value': 0.0, 'reason': None},
      'max_abs': {'value': 0.0, 'reason': None},
      'max_abs_index': 0,
  }


def test_logits_errors_refuse_what_they_cannot_compute():
  """Shapes, non-finite inputs and overflow are refused."""
  assert metrics.logits_errors(_vector(1), _vector(1, 2)) == (
      metrics.missing_logits_errors('Logits shapes differ')
  )
  assert metrics.logits_errors(_vector(1, -math.inf), _vector(1, 2)) == (
      metrics.missing_logits_errors(
          'Logits contain non-finite values; full-vector metrics are'
          ' unavailable'
      )
  )
  assert metrics.logits_errors(_vector(-1e308), _vector(1e308)) == (
      metrics.missing_logits_errors(
          'Logits differences exceed finite float64 values'
      )
  )


def test_centered():
  """Centering subtracts the mean, or says why it cannot."""
  values, reason = metrics.centered(_vector(1, 2, 6))
  np.testing.assert_allclose(values, [-2, -1, 3])
  assert reason is None
  assert metrics.centered(_vector(0, 0))[0].tolist() == [0.0, 0.0]
  assert metrics.centered(_vector(1, math.inf)) == (
      None,
      'Centered logits unavailable: source contains non-finite values',
  )
