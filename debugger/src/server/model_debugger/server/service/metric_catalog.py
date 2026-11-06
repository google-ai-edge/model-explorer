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

"""What each metric is, for the UI: the response of GET /api/metrics.

The analyses compute the metrics (`analysis/metrics.py` and the KL, JS and
distribution figures beside them); this module says what each one means and
how it is shown, so that no panel of the UI keeps a table of its own. Its
shape is the UI's `MetricCatalog` (`server/contracts/metrics.ts`):

- `metrics`: one description per metric id, the name every response uses.
  `unit` says what the value is measured in and `unit_label` how it is
  written after a number; `format` is how a value is written (`fixed`
  decimals or `significant` digits, an optional `scale` it is multiplied by
  for display and a `suffix`); `larger_is_better`, `ideal` (the value of
  two identical sides) and `bounds` say how to read it; `color` holds the
  four `thresholds` of a stepped color scale (null for a continuous one),
  `heat_full`, the value at which a token's heat is full, and `heat_labels`,
  what a legend of that heat writes at its ends (`_heat_labels`); `analyses`
  lists the analyses that show it by default.
- `analyses`: each analysis the UI shows, with the metrics it offers, in
  order, its default, the group each metric's label is prefixed with, what
  it compares (`inputs`, unless a metric names its own) and, for the KV
  cache, how KV heads are combined by default.
- `reasons`: every code a reason (metrics.reason) or a status may carry, with
  its words (`text`). A response's reason carries its own text, which the UI
  shows; the UI decides on the code, and shows these words where a response
  gives a code alone (kv-head's `metric_status`). One list, not a map per
  metric: a code (a context that differs, a cache not captured) is of an
  analysis or a stage more often than of one metric, and means the same
  for every metric it leaves missing.
"""

from typing import Any

from model_debugger.server.analysis import kv_analysis
from model_debugger.server.analysis import metrics
from model_debugger.server.analysis import tensor_analysis
from model_debugger.server.analysis import token_analysis
from model_debugger.server.service import session_view

_PERCENT = {'style': 'fixed', 'decimals': 2, 'suffix': '%'}
_DIGITS = {'style': 'significant', 'digits': 6}
_TENSOR_COLOR = {'thresholds': [1e-5, 1e-4, 1e-3, 1e-2], 'heat_full': 0.1}
_NO_COLOR = {'thresholds': None, 'heat_full': None}
_SOFTMAX_INPUTS = (
    'Full captured vocabulary logits of the token; softmax (T=1), '
    'independent of the generation settings; no top-k, top-p or sampling '
    'constraints'
)

# id -> description, in the order the UI lists them.
_METRICS = {
    'cosine_similarity': {
        'label': 'Cosine similarity',
        'unit': 'unitless',
        'unit_label': '',
        'measures': 'Direction similarity of the Reference and Target vectors',
        'formula': 'Reference · Target / (‖Reference‖₂ ‖Target‖₂)',
        'range': '−1 to 1',
        'reading': '1: same direction · 0: orthogonal · −1: opposite',
        'larger_is_better': True,
        'ideal': 1,
        'bounds': [-1, 1],
        'format': {'style': 'fixed', 'decimals': 6},
        'color': {'thresholds': None, 'heat_full': 0.98},
    },
    'cosine_distance': {
        'label': 'Cosine distance',
        'unit': 'unitless',
        'unit_label': '',
        'measures': 'How far the Reference and Target directions are apart',
        'formula': '1 − cosine similarity',
        'range': '0 to 2',
        'reading': '0: same direction · 1: orthogonal · 2: opposite',
        'larger_is_better': False,
        'ideal': 0,
        'bounds': [0, 2],
        'format': _DIGITS,
        'color': {'thresholds': None, 'heat_full': 0.02},
    },
    'relative_l2': {
        'label': 'Relative L2',
        'unit': 'percent',
        'unit_label': '%',
        'measures': 'L2 error relative to the Reference L2 norm',
        'formula': '100 · ‖Target − Reference‖₂ / ‖Reference‖₂',
        'range': '0% and above',
        'reading': '0%: identical values',
        'larger_is_better': False,
        'ideal': 0,
        'bounds': [0, None],
        'format': _PERCENT,
        'color': {'thresholds': [0.01, 0.1, 1, 10], 'heat_full': 1},
    },
    'norm_ratio': {
        'label': 'Norm ratio',
        'unit': 'ratio',
        'unit_label': '×',
        'measures': 'Target L2 norm divided by the Reference L2 norm',
        'formula': '‖Target‖₂ / ‖Reference‖₂',
        'range': '0 and above',
        'reading': '1: equal norms',
        'larger_is_better': False,
        'ideal': 1,
        'bounds': [0, None],
        'format': _DIGITS,
        'color': {'thresholds': None, 'heat_full': 1.2},
    },
    'max_abs': {
        'label': 'Max absolute error',
        'unit': 'same as the compared values',
        'unit_label': '',
        'measures': 'Largest absolute element difference',
        'formula': 'max |Target − Reference|',
        'range': '0 and above',
        'reading': '0: identical values',
        'larger_is_better': False,
        'ideal': 0,
        'bounds': [0, None],
        'format': _DIGITS,
        'color': _TENSOR_COLOR,
    },
    'mean_abs': {
        'label': 'Mean absolute error',
        'unit': 'same as the compared values',
        'unit_label': '',
        'measures': 'Mean absolute element difference',
        'formula': 'mean |Target − Reference|',
        'range': '0 and above',
        'reading': '0: identical values',
        'larger_is_better': False,
        'ideal': 0,
        'bounds': [0, None],
        'format': _DIGITS,
        'color': _TENSOR_COLOR,
    },
    'rmse': {
        'label': 'RMSE',
        'unit': 'same as the compared values',
        'unit_label': '',
        'measures': 'Root mean square error across all elements',
        'formula': '√mean((Target − Reference)²)',
        'range': '0 and above',
        'reading': '0: identical values; larger errors weigh more',
        'larger_is_better': False,
        'ideal': 0,
        'bounds': [0, None],
        'format': _DIGITS,
        'color': _TENSOR_COLOR,
    },
    'kl': {
        'label': 'Kullback–Leibler divergence',
        'unit': 'nats',
        'unit_label': 'nats',
        'measures': 'Kullback–Leibler divergence from Reference to Target',
        'formula': 'Σ p log(p / q), p Reference, q Target',
        'range': '0 and above',
        'reading': '0: identical distributions; direction Reference → Target',
        'larger_is_better': False,
        'ideal': 0,
        'bounds': [0, None],
        'format': _DIGITS,
        'color': {'thresholds': None, 'heat_full': 0.01},
        'inputs': _SOFTMAX_INPUTS,
    },
    'js': {
        'label': 'Jensen–Shannon divergence',
        'unit': 'bits',
        'unit_label': 'bits',
        'measures': (
            'Symmetric difference of the Reference and Target distributions'
        ),
        'formula': '½ KL(p ‖ m) + ½ KL(q ‖ m), m = (p + q) / 2, base 2',
        'range': '0 to 1 bit',
        'reading': '0: identical · 1: disjoint support',
        'larger_is_better': False,
        'ideal': 0,
        'bounds': [0, 1],
        'format': _DIGITS,
        'color': {'thresholds': None, 'heat_full': 1},
        'inputs': _SOFTMAX_INPUTS,
    },
    'entropy': {
        'label': 'Entropy',
        'unit': 'bits',
        'unit_label': 'bits',
        'measures': (
            'Average information of one runtime’s distribution; '
            'fractional bits are valid'
        ),
        'formula': 'H = −Σ p(token) log₂ p(token)',
        'range': '0 to log₂(vocabulary size)',
        'reading': (
            '0 bits: one token has all the probability · 1 bit: two '
            'tokens of 50% each · higher: more spread across tokens'
        ),
        'larger_is_better': None,
        'ideal': None,
        'bounds': [0, None],
        'format': _DIGITS,
        'color': _NO_COLOR,
    },
    'selected_probability': {
        'label': 'Selected probability',
        'unit': 'probability',
        'unit_label': '%',
        'measures': (
            'Probability of the selected token in one runtime’s distribution'
        ),
        'formula': 'p(selected token)',
        'range': '0% to 100%',
        'reading': '100%: the runtime was certain of its token',
        'larger_is_better': None,
        'ideal': None,
        'bounds': [0, 1],
        'format': {**_PERCENT, 'scale': 100},
        'color': _NO_COLOR,
    },
    'margin': {
        'label': 'Top-2 margin',
        'unit': 'probability',
        'unit_label': 'pp',
        'measures': (
            'Highest minus second-highest token probability of one runtime'
        ),
        'formula': 'p(first) − p(second)',
        'range': '0 to 100 percentage points',
        'reading': 'Higher: larger separation between the top two tokens',
        'larger_is_better': None,
        'ideal': None,
        'bounds': [0, 1],
        'format': {
            'style': 'fixed',
            'decimals': 2,
            'scale': 100,
            'suffix': ' pp',
        },
        'color': _NO_COLOR,
    },
}

_ACTIVATION = 'Layer Activation'
_ANALYSES = (
    {
        'id': 'kv',
        'label': 'KV cache',
        'metrics': ['cosine_similarity', 'relative_l2', 'max_abs'],
        'default_metric': 'cosine_similarity',
        'default_head': kv_analysis.DEFAULT_HEAD,
        'groups': {},
        'inputs': (
            'Stored KV values of each position and KV head, compared '
            'over the batch and channel (head_dim) axes in float64; for two '
            'models of one KV architecture (one model in two runtimes), each '
            'side read through its own layout, an int8 side dequantized by the '
            'scale its model gives the cache'
        ),
    },
    {
        'id': 'kv_head',
        'label': 'KV head',
        'metrics': ['cosine_similarity', 'relative_l2', 'rmse', 'max_abs'],
        'default_metric': 'cosine_similarity',
        'groups': {},
        'inputs': (
            'One KV head’s stored channels at one position, compared '
            'in float64; for two models, an int8 side dequantized by its '
            'model’s scale'
        ),
    },
    {
        'id': 'token_diff',
        'label': 'Token Diff',
        'metrics': [
            'kl',
            'js',
            'relative_l2',
            'cosine_similarity',
            'norm_ratio',
            'max_abs',
        ],
        'default_metric': 'kl',
        'groups': {
            'kl': 'Logits',
            'js': 'Logits',
            'relative_l2': _ACTIVATION,
            'cosine_similarity': _ACTIVATION,
            'cosine_distance': _ACTIVATION,
            'norm_ratio': _ACTIVATION,
            'max_abs': _ACTIVATION,
        },
        'inputs': (
            'The captured activations of the forward that produced the '
            'token, compared in float64'
        ),
    },
    {
        'id': 'final_logits',
        'label': 'Final Logits',
        # What token-range bins of the analysis.
        'metrics': list(token_analysis.RANGE_METRICS['final_logits']),
        'default_metric': 'rmse',
        'groups': {},
        'inputs': 'Raw captured final logits, before softmax; no centering',
    },
    {
        'id': 'sampling',
        'label': 'Sampling',
        'metrics': list(token_analysis.RANGE_METRICS['sampling']),
        'default_metric': 'entropy',
        'groups': dict.fromkeys(
            ('entropy', 'selected_probability', 'margin'), 'Per runtime'
        ),
        # What token_analysis._sampling_side does.
        'inputs': (
            'The capture holds the logits before sampling, not the sampler’s'
            ' probabilities: each runtime’s distribution is the softmax of its'
            ' full captured vocabulary logits divided by its sampler'
            ' temperature (1 when none is set), before top-k, top-p or any'
            ' other constraint; a greedy sampler’s too. Each runtime uses its'
            ' own temperature and selected token.'
        ),
    },
    {
        'id': 'intermediate_tensors',
        'label': 'Intermediate tensors',
        'metrics': ['cosine_similarity', 'relative_l2', 'max_abs'],
        'default_metric': 'cosine_similarity',
        'groups': {},
        'inputs': (
            'Captured intermediate tensors of each position and sync '
            'point, compared in float64'
        ),
    },
)


# The reason codes of every module that gives reasons; no code is in two.
_REASONS = (
    metrics.REASONS,
    token_analysis.REASONS,
    kv_analysis.REASONS,
    tensor_analysis.REASONS,
    session_view.REASONS,
)


def _number(value: float) -> str:
  """A number of a legend, as the UI wrote one: 1, 0.98, 0.01."""
  return f'{value:.6g}'


def _heat_labels(description: dict[str, Any]) -> dict[str, Any] | None:
  """What a legend of a metric's token heat writes at its two ends.

  A token's heat is its distance from the ideal value relative to
  `heat_full`'s (src/ui metric_scale.ts, tokenHeat). `none` is the ideal
  value, where the heat is none; `full` where it is full: `heat_full` with
  the unit, or, when values on both sides of the ideal reach that
  distance, the ideal ± it, in percent of a nonzero ideal: norm ratio's
  heat is full at 0.8 and at 1.2, '1 ± 20%'.

  Args:
    description: Metric description dictionary containing ideal and color scale.

  Returns:
    `{none, full}`, or None for a metric without a heat scale.
  """
  ideal, full = description['ideal'], description['color']['heat_full']
  if ideal is None or full is None:
    return None
  distance = abs(full - ideal)
  low, high = description['bounds']
  both_sides = (low is None or ideal - distance >= low) and (
      high is None or ideal + distance <= high
  )
  unit = description['unit_label']
  if both_sides and ideal:
    text = f'{_number(ideal)} ± {_number(100 * distance / abs(ideal))}%'
  elif both_sides:
    text = f'± {_number(distance)}' + (f' {unit}' if unit else '')
  else:
    text = _number(full) + (f' {unit}' if unit else '')
  return {'none': _number(ideal), 'full': text}


def catalog() -> dict[str, Any]:
  """Returns the UI's `MetricCatalog`: every metric and analysis."""
  used = {}
  for analysis in _ANALYSES:
    for key in analysis['metrics']:
      used.setdefault(key, []).append(analysis['id'])
  reasons = []
  for table in _REASONS:
    for code, text in table.items():
      reasons.append({'code': code, 'text': text})
  return {
      'metrics': [
          {
              'id': key,
              **description,
              'color': {
                  **description['color'],
                  'heat_labels': _heat_labels(description),
              },
              'inputs': description.get('inputs'),
              'analyses': used.get(key, []),
          }
          for key, description in _METRICS.items()
      ],
      'analyses': [dict(analysis) for analysis in _ANALYSES],
      'reasons': reasons,
  }
