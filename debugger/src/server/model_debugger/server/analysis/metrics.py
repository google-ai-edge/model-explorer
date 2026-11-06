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

"""Metrics between a ref and a target, computed in float64.

This is the one computation of them. Token activations, final logits and
the KV cache (each position and KV head, and one head's channels) are all
compared by `compare`, so a metric means the same everywhere; each analysis
shows the ones it needs. Nothing here reshapes, clips a tensor, or adds an
epsilon: a metric that is undefined (a zero norm, a non-finite value) is
reported as missing with the reason, never as a substitute number.

A reason is `{code, side, text}` (`reason`): `code` is what the UI decides
on, `side` the side it is of (None for both, or for the pair), `text` the
words it shows. A value or a comparison missing for several reasons, one per
side say, lists each. `REASONS` gives each code of this module the words
the metric catalog lists for it (service/metric_catalog).
"""

import dataclasses
import math
from typing import Any

import numpy as np

# The dtype every metric is computed in, as the UI's metadata names it.
CALCULATION_DTYPE = 'float64'
# Every metric `compare` computes, by its one name.
VECTOR_KEYS = (
    'cosine_similarity',
    'cosine_distance',
    'relative_l2',
    'norm_ratio',
    'max_abs',
    'mean_abs',
    'rmse',
)
# The status of a value: why it is missing, or OK.
OK = 'ok'
NON_FINITE = 'non_finite_tensor'
ZERO_NORM = 'undefined_zero_norm'
OVERFLOW = 'numeric_overflow'
# Why two vectors are not compared: their shapes differ; or a tensor file
# cannot be read as the index lists it.
SHAPE_MISMATCH = 'shape_mismatch'
UNREADABLE = 'unreadable'
# The words of each code of this module, as the metric catalog lists them;
# a status the UI gets as a code alone (kv-head's `metric_status`) is shown
# in them.
REASONS = {
    NON_FINITE: 'Unavailable because the values include NaN or infinity.',
    ZERO_NORM: 'Undefined because a required vector norm is zero.',
    OVERFLOW: 'The comparison exceeds the supported numerical range.',
    SHAPE_MISMATCH: 'The two sides have different shapes.',
    UNREADABLE: 'A captured tensor file cannot be read as the index lists it.',
}
# The metrics each zero norm leaves undefined.
_NEED_REF_NORM = ('relative_l2', 'norm_ratio')
_NEED_BOTH_NORMS = ('cosine_similarity', 'cosine_distance')


def reason(code: str, text: str, side: str | None = None) -> dict[str, Any]:
  """Why a value is missing, or two sides are not compared.

  Args:
      code: What the UI decides on; the metric catalog lists every code.
      text: The words the UI shows.
      side: 'ref' or 'target' for a reason of one side; None for both, or for
        the pair.

  Returns:
      `{code, side, text}`.
  """
  return {'code': code, 'side': side, 'text': text}


def metric(value: float) -> dict[str, Any]:
  """A metric the UI shows: its value and no reasons."""
  return {'value': value, 'reasons': []}


def missing(reasons: list[dict[str, Any]]) -> dict[str, Any]:
  """A metric that has no value, with the reasons (`reason`) why."""
  return {'value': None, 'reasons': list(reasons)}


@dataclasses.dataclass(frozen=True)
class Comparison:
  """The metrics `compare` was asked for, one value per compared group.

  Attributes:
      values: `{key: values}`, NaN where the metric is undefined.
      status: `{key: codes}`, `OK` where the metric has a value, else why not
        (`NON_FINITE`, `ZERO_NORM`, `OVERFLOW`).
  """

  values: dict[str, np.ndarray]
  status: dict[str, np.ndarray]


def compare(
    ref: np.ndarray,
    target: np.ndarray,
    axes: tuple[int, ...] | None = None,
    keys: tuple[str, ...] = VECTOR_KEYS,
) -> Comparison:
  """How far target is from ref, over `axes`, for each metric of `keys`.

  Each group (the values left once `axes` are taken out) is divided by the
  power of two just above its largest absolute value, on either side, and
  its differences by the one just above their largest, before anything is
  squared or summed, so finite values never overflow. Dividing by a power
  of two changes no value unless the quotient is subnormal, so a group
  that could not overflow gets the values it would get unscaled; the
  ratios do not depend on that scale. Only the metrics of `keys` are
  computed, each as it is when all are.

  - relative_l2: 100 * |target - ref| / |ref|, a percent.
  - cosine_similarity: ref . target / (|ref| |target|), within [-1, 1];
    exactly 1 for equal vectors.
  - cosine_distance: 1 - cosine_similarity, within [0, 2].
  - norm_ratio: |target| / |ref|.
  - max_abs, mean_abs: the largest and the mean |target - ref|.
  - rmse: the root mean square of target - ref.

  Args:
      ref: The ref side's values.
      target: The target side's values, of the same shape.
      axes: The axes compared over; None for all of them.
      keys: The metrics wanted, of `VECTOR_KEYS`.

  Returns:
      The comparison of `keys`, each value shaped as the axes that are
      left.

  Raises:
      ValueError: The shapes differ, a group holds no values, or a key is
        not one of `VECTOR_KEYS`.
  """
  unknown = set(keys) - set(VECTOR_KEYS)
  if unknown:
    raise ValueError(f'Unknown metrics: {", ".join(sorted(unknown))}')
  if ref.shape != target.shape:
    raise ValueError('shape_mismatch')
  a, b = ref.astype(np.float64), target.astype(np.float64)
  if axes is None:
    axes = tuple(range(a.ndim))
  # The compared axes become the last one.
  moved = tuple(range(-len(axes), 0))
  a, b = np.moveaxis(a, axes, moved), np.moveaxis(b, axes, moved)
  groups = a.shape[: a.ndim - len(axes)]
  a, b = a.reshape(groups + (-1,)), b.reshape(groups + (-1,))
  if not a.shape[-1]:
    raise ValueError('No values to compare')
  wanted = set(keys)
  finite = np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
  values = {}
  with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
    if wanted & {'max_abs', 'mean_abs', 'rmse'}:
      absolute = np.abs(b - a)
      largest = absolute.max(axis=-1)
      values['max_abs'] = largest
      # Dividing by a power of two before the mean keeps finite inputs
      # from overflowing it, and changes nothing that cannot overflow.
      unit = _power_of_two_above(np.where(largest > 0, largest, 1.0))
      fractions = absolute / unit[..., None]
      if 'mean_abs' in wanted:
        values['mean_abs'] = unit * np.mean(fractions, axis=-1)
      if 'rmse' in wanted:
        values['rmse'] = unit * np.sqrt(np.mean(fractions**2, axis=-1))
    nx = ny = None
    if wanted - {'max_abs', 'mean_abs', 'rmse'}:
      scale = np.maximum(np.abs(a).max(axis=-1), np.abs(b).max(axis=-1))
      scale = np.where(finite & (scale > 0), scale, 1.0)
      divisor = _power_of_two_above(scale)[..., None]
      x, y = a / divisor, b / divisor
      xx = np.einsum('...i,...i->...', x, x)
      nx = np.sqrt(xx)
      if wanted & {'cosine_similarity', 'cosine_distance', 'norm_ratio'}:
        yy = np.einsum('...i,...i->...', y, y)
        ny = np.sqrt(yy)
      if wanted & {'cosine_similarity', 'cosine_distance'}:
        xy = np.einsum('...i,...i->...', x, y)
        # sqrt(xx * yy), not nx * ny, makes equal vectors exactly 1.
        similarity = np.clip(xy / np.sqrt(xx * yy), -1, 1)
        values['cosine_similarity'] = similarity
        values['cosine_distance'] = 1 - similarity
      if 'relative_l2' in wanted:
        scaled = y - x
        values['relative_l2'] = (
            100 * np.sqrt(np.einsum('...i,...i->...', scaled, scaled)) / nx
        )
      if 'norm_ratio' in wanted:
        values['norm_ratio'] = ny / nx
  result, status = {}, {}
  for key in keys:
    value = values[key]
    code = np.full(groups, OK, dtype=object)
    code[~np.isfinite(value)] = OVERFLOW
    if key in _NEED_REF_NORM:
      code[nx == 0] = ZERO_NORM
    if key in _NEED_BOTH_NORMS:
      code[(nx == 0) | (ny == 0)] = ZERO_NORM
    code[~finite] = NON_FINITE
    result[key] = np.where(code == OK, value, np.nan)
    status[key] = code
  return Comparison(result, status)


def _power_of_two_above(values: np.ndarray) -> np.ndarray:
  """The power of two just above each value; 1 for a non-finite one.

  2^1024 is not a float64: the largest values get 2^1023, which leaves
  them below 2 once divided by it.

  Args:
      values: The values, float64.

  Returns:
      The power of two of each value, float64.
  """
  return np.ldexp(1.0, np.minimum(np.frexp(values)[1], 1023))


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


@dataclasses.dataclass(frozen=True)
class Distribution:
  """The softmax of one logits vector (temperature 1), with its ranks.

  Attributes:
      p: Probability of each vocabulary id.
      logp: Natural log of `p`, from log-sum-exp (finite where logits are).
      order: Vocabulary ids from the most to the least likely; equal
        probabilities are ordered by id.
      ranks: 1-based rank of each id in `order`.
      entropy: Entropy in bits.
      margin: Probability of the most likely id minus that of the second, or
        None for a one-entry vocabulary.
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


# What token analysis shows of an activation pair, from `compare`.
ACTIVATION_KEYS = (
    'relative_l2',
    'cosine_similarity',
    'cosine_distance',
    'norm_ratio',
    'max_abs',
)
# Why an activation metric is missing, by status.
_ACTIVATION_REASONS = {
    NON_FINITE: 'Activation contains non-finite values',
    ZERO_NORM: 'Zero reference norm or non-finite result',
    OVERFLOW: 'Zero reference norm or non-finite result',
}


def activation_metrics(
    ref: np.ndarray, target: np.ndarray
) -> dict[str, dict[str, Any]]:
  """How far target's activation vector is from ref's.

  Args:
      ref: The ref side's vector.
      target: The target side's vector.

  Returns:
      `{key: metric}` for every key of `ACTIVATION_KEYS`, as `compare`
      defines them.
  """
  if ref.shape != target.shape:
    why = [reason(SHAPE_MISMATCH, 'Activation shapes differ')]
    return {key: missing(why) for key in ACTIVATION_KEYS}
  comparison = compare(ref, target, keys=ACTIVATION_KEYS)
  result = {}
  for key in ACTIVATION_KEYS:
    status = comparison.status[key].item()
    result[key] = (
        metric(comparison.values[key].item())
        if status == OK
        else missing([reason(status, _ACTIVATION_REASONS[status])])
    )
  return result


def missing_logits_errors(reasons: list[dict[str, Any]]) -> dict[str, Any]:
  """`logits_errors` when they cannot be computed, and why (`reason`)."""
  return {
      'rmse': missing(reasons),
      'max_abs': missing(reasons),
      'max_abs_index': None,
  }


def logits_errors(ref: np.ndarray, target: np.ndarray) -> dict[str, Any]:
  """RMSE and largest absolute error over two whole logits vectors.

  No finite subset and no softmax stands in: non-finite inputs make both
  metrics missing.

  Args:
      ref: The ref side's float64 vector.
      target: The target side's float64 vector.

  Returns:
      `{rmse, max_abs, max_abs_index}`, as `compare` defines them; the index
      is the vocabulary id of the largest error (the first one on ties).
  """
  if ref.shape != target.shape:
    return missing_logits_errors(
        [reason(SHAPE_MISMATCH, 'Logits shapes differ')]
    )
  comparison = compare(ref, target, keys=('rmse', 'max_abs'))
  status = {comparison.status[key].item() for key in ('rmse', 'max_abs')}
  if NON_FINITE in status:
    return missing_logits_errors([
        reason(
            NON_FINITE,
            'Logits contain non-finite values; full-vector metrics '
            'are unavailable',
        )
    ])
  if status != {OK}:
    return missing_logits_errors(
        [reason(OVERFLOW, 'Logits differences exceed finite float64 values')]
    )
  return {
      'rmse': metric(comparison.values['rmse'].item()),
      'max_abs': metric(comparison.values['max_abs'].item()),
      'max_abs_index': int(np.argmax(np.abs(target - ref))),
  }


def centered(
    values: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
  """The vector minus its mean, or None and the reason it has none.

  Args:
      values: A float64 vector.

  Returns:
      `(centered, None)`, or `(None, reason)` when the input or the result
      holds a non-finite value; the reason (`reason`) is of no side.
  """
  if not np.all(np.isfinite(values)):
    return None, reason(
        NON_FINITE,
        'Centered logits unavailable: source contains non-finite values',
    )
  scale = float(np.max(np.abs(values)))
  mean = float(np.mean(values / scale)) * scale if scale else 0.0
  with np.errstate(over='ignore', invalid='ignore'):
    result = values - mean
  if not np.all(np.isfinite(result)):
    return None, reason(
        OVERFLOW, 'Centered logits exceed finite float64 values'
    )
  return result, None
