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

"""Metrics between a ref vector and a target vector, computed in float64.

Nothing here reshapes, clips a tensor, or adds an epsilon: a metric that is
undefined (a zero norm, a non-finite value) is reported as missing with the
reason, never as a substitute number.
"""

import dataclasses
import math

import numpy as np


def metric(value: float) -> dict:
  """A metric the UI shows: its value and no reason."""
  return {'value': value, 'reason': None}


def missing(reason: str) -> dict:
  """A metric that has no value, with the sentence that says why."""
  return {'value': None, 'reason': reason}


def json_number(value: float | None) -> float | str | None:
  """A float for JSON, with non-finite values spelled as JavaScript does."""
  if value is None:
    return None
  value = float(value)
  if math.isfinite(value):
    return value
  if math.isnan(value):
    return 'NaN'
  return 'Infinity' if value > 0 else '-Infinity'


def json_numbers(values: np.ndarray) -> list[float | str]:
  """`json_number` over a whole vector, fast when every value is finite."""
  numbers = values.astype(np.float64).tolist()
  if np.all(np.isfinite(values)):
    return numbers
  return [json_number(value) for value in numbers]


@dataclasses.dataclass(frozen=True)
class Distribution:
  """The softmax of one logits vector (temperature 1), with its ranks.

  Attributes:
    p: Probability of each vocabulary id.
    logp: Natural log of `p`, from log-sum-exp (finite where logits are).
    order: Vocabulary ids from the most to the least likely; equal probabilities
      are ordered by id.
    ranks: 1-based rank of each id in `order`.
    entropy: Entropy in bits.
    margin: Probability of the most likely id minus that of the second, or None
      for a one-entry vocabulary.
  """

  p: np.ndarray
  logp: np.ndarray
  order: np.ndarray
  ranks: np.ndarray
  entropy: float
  margin: float | None


def distribution(logits: np.ndarray) -> Distribution:
  """The softmax distribution of a float64 logits vector.

  Args:
    logits: One vocabulary vector. -Infinity is a zero probability.

  Returns:
    The distribution.

  Raises:
    ValueError: The logits hold NaN or +Infinity, or nothing finite.
  """
  if (
      np.any(np.isnan(logits))
      or np.any(np.isposinf(logits))
      or not np.any(np.isfinite(logits))
  ):
    raise ValueError('Logits contain invalid values')
  shifted = logits - np.max(logits)
  logp = shifted - np.log(np.exp(shifted).sum())
  p = np.exp(logp)
  order = np.lexsort((np.arange(p.size), -p))
  ranks = np.empty(p.size, dtype=np.int64)
  ranks[order] = np.arange(1, p.size + 1)
  positive = p > 0
  entropy = float(-np.sum(p[positive] * logp[positive]) / np.log(2))
  margin = float(p[order[0]] - p[order[1]]) if p.size > 1 else None
  return Distribution(p, logp, order, ranks, entropy, margin)


def kl_divergence(ref: Distribution, target: Distribution) -> float:
  """KL(ref || target) in nats; +Infinity when target rules out a ref id."""
  positive = ref.p > 0
  with np.errstate(over='ignore', invalid='ignore'):
    value = float(
        np.sum(ref.p[positive] * (ref.logp[positive] - target.logp[positive]))
    )
  return max(0.0, value) if math.isfinite(value) else math.inf


def js_divergence(ref: Distribution, target: Distribution) -> float:
  """Jensen-Shannon divergence in bits, within [0, 1]."""
  middle = (ref.p + target.p) / 2
  value = 0.0
  for p in (ref.p, target.p):
    positive = p > 0
    value += 0.5 * float(
        np.sum(p[positive] * (np.log2(p[positive]) - np.log2(middle[positive])))
    )
  return float(np.clip(value, 0, 1))


ACTIVATION_KEYS = (
    'relative_l2',
    'cosine_distance',
    'norm_ratio',
    'max_abs_error',
)


def activation_metrics(ref: np.ndarray, target: np.ndarray) -> dict:
  """How far target's activation vector is from ref's.

  Both vectors are divided by their largest absolute value first, so squaring
  cannot overflow; `max_abs_error` is scaled back.

  Args:
    ref: The ref side's vector.
    target: The target side's vector.

  Returns:
    `{key: metric}` for every key of `ACTIVATION_KEYS`: `relative_l2` is
    100 * |target - ref| / |ref| (a percent), `cosine_distance` is
    1 - cosine similarity within [0, 2], `norm_ratio` is |target| / |ref|,
    `max_abs_error` is the largest |target - ref|.
  """
  if ref.shape != target.shape:
    return {key: missing('Activation shapes differ') for key in ACTIVATION_KEYS}
  a, b = ref.astype(np.float64), target.astype(np.float64)
  scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
  if not math.isfinite(scale):
    return {
        key: missing('Activation contains non-finite values')
        for key in ACTIVATION_KEYS
    }
  if scale:
    a, b = a / scale, b / scale
  norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
  difference = b - a
  values = {
      'relative_l2': (
          100 * float(np.linalg.norm(difference)) / norm_a if norm_a else None
      ),
      'cosine_distance': (
          float(np.clip(1 - np.dot(a, b) / (norm_a * norm_b), 0, 2))
          if norm_a and norm_b
          else None
      ),
      'norm_ratio': norm_b / norm_a if norm_a else None,
      'max_abs_error': float(np.max(np.abs(difference))) * scale,
  }
  return {
      key: (
          metric(value)
          if value is not None and math.isfinite(value)
          else missing('Zero reference norm or non-finite result')
      )
      for key, value in values.items()
  }


def missing_logits_errors(reason: str) -> dict:
  """`logits_errors` when they cannot be computed."""
  return {
      'rmse': missing(reason),
      'max_abs': missing(reason),
      'max_abs_index': None,
  }


def logits_errors(ref: np.ndarray, target: np.ndarray) -> dict:
  """RMSE and largest absolute error over two whole logits vectors.

  No finite subset and no softmax stands in: non-finite inputs make both
  metrics missing.

  Args:
    ref: The ref side's float64 vector.
    target: The target side's float64 vector.

  Returns:
    `{rmse, max_abs, max_abs_index}`; the index is the vocabulary id of the
    largest error (the first one on ties).
  """
  if ref.shape != target.shape:
    return missing_logits_errors('Logits shapes differ')
  if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(target)):
    return missing_logits_errors(
        'Logits contain non-finite values; full-vector metrics are unavailable'
    )
  with np.errstate(over='ignore', invalid='ignore'):
    absolute = np.abs(target - ref)
  if not np.all(np.isfinite(absolute)):
    return missing_logits_errors(
        'Logits differences exceed finite float64 values'
    )
  index = int(np.argmax(absolute))
  maximum = float(absolute[index])
  # Scaled before squaring so that large finite differences cannot overflow.
  rmse = (
      maximum * float(np.sqrt(np.mean((absolute / maximum) ** 2)))
      if maximum
      else 0.0
  )
  return {
      'rmse': metric(rmse),
      'max_abs': metric(maximum),
      'max_abs_index': index,
  }


def centered(values: np.ndarray) -> tuple[np.ndarray | None, str | None]:
  """The vector minus its mean, or None and the reason it has none.

  Args:
    values: A float64 vector.

  Returns:
    `(centered, None)`, or `(None, reason)` when the input or the result
    holds a non-finite value.
  """
  if not np.all(np.isfinite(values)):
    return (
        None,
        'Centered logits unavailable: source contains non-finite values',
    )
  scale = float(np.max(np.abs(values)))
  mean = float(np.mean(values / scale)) * scale if scale else 0.0
  with np.errstate(over='ignore', invalid='ignore'):
    result = values - mean
  if not np.all(np.isfinite(result)):
    return None, 'Centered logits exceed finite float64 values'
  return result, None
