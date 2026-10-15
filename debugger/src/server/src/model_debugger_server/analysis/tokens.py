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

"""Per-token comparison of ref and target: token-analysis and final logits.

A pair names one generated token on each side by its ordinal in the turn's
`output` (the `step` the UI holds). Each token resolves to the decode forward
whose logits produced it, and that forward's `post_logits` and
`post_activations` files are what is compared. Two logits vectors are
compared only when both sides ran the same model file (so the vocabulary is
the same) and fed exactly the same tokens before the pair; otherwise each side
still shows its own distribution, without deltas.
"""

import bisect
import dataclasses

from model_debugger_server import capture_index, tensors
from model_debugger_server.analysis import metrics
import numpy as np

SIDES = ('ref', 'target')
TOKEN_METRIC_KEYS = ('kl', 'js') + metrics.ACTIVATION_KEYS
MAX_PAIRS = 128
MAX_FINAL_LOGITS_ROWS = 256
MAX_VECTOR_VALUES = 524_288
# Candidate rows of a distribution: each side's most likely ids.
CANDIDATES = 64

CONTEXT_REASON = (
    'Matching captured logical token contexts are required; generated '
    'histories have diverged or are unproven'
)
VOCABULARY_REASON = 'Matching captured vocabulary IDs and sizes are required'
NOT_PAIRED_REASON = 'Paired full-vocabulary logits not captured'
NOT_CAPTURED_REASON = 'Token not captured on this side'


@dataclasses.dataclass(frozen=True)
class _Token:
  """One side's token of a pair, resolved to the forward that produced it.

  Attributes:
    side: 'ref' or 'target'.
    id: The token's vocabulary id; None for a sampled chat's stop token, which
      the runner does not name.
    forward: The decode forward whose logits produced the token.
    context: `(position, id)` of every token the chat fed before it; an id is
      None where the runner did not name the token.
    model: sha256 of the side's model file.
  """

  side: str
  id: int | None
  forward: dict
  context: tuple[tuple[int, int], ...]
  model: str


@dataclasses.dataclass(frozen=True)
class _Logits:
  """One side's logits vector and what the UI is told about it."""

  values: np.ndarray
  vocabulary: str
  summary: dict


class _Turn:
  """The turn a request names, read from the capture index."""

  def __init__(self, index: capture_index.CaptureIndex, n: int) -> None:
    self.index = index
    self.n = n
    self.sides = index.turn(n)['sides']
    self._fed = {}
    self._labels = {}

  def token(self, side: str, step: int | None) -> _Token:
    """The `step`-th token the side generated in this turn."""
    output = self.sides[side]['output']
    if step is None or step >= len(output):
      raise ValueError(NOT_CAPTURED_REASON)
    token = output[step]
    forward = next(
        forward
        for forward in self.sides[side]['forwards']
        if forward['id'] == token['forward']
    )
    positions, ids = self._fed_tokens(side)
    count = bisect.bisect_left(positions, token['position'])
    return _Token(
        side=side,
        id=token['id'],
        forward=forward,
        context=tuple(zip(positions[:count], ids[:count])),
        model=self.index.sides[side]['model']['sha256'],
    )

  def _fed_tokens(self, side: str) -> tuple[list[int], list[int]]:
    """Positions and ids of every token of turns 1..n, by position."""
    if side not in self._fed:
      by_position = {}
      for n in range(1, self.n + 1):
        entry = self.index.turn(n)['sides'][side]
        # A later turn's input replaces what an earlier output left at a
        # position (the carried stop token is both, with one id).
        for token in entry['input'] + entry['output']:
          by_position[token['position']] = token['id']
      positions = sorted(by_position)
      self._fed[side] = (positions, [by_position[p] for p in positions])
    return self._fed[side]

  def logits(self, token: _Token) -> _Logits:
    """The logits that produced the token, as one float64 vector."""
    values, metadata = self._vector(
        token, 'post_logits', 'Full-vocabulary logits'
    )
    # One model file has one vocabulary; its size is the logits length.
    vocabulary = f'{token.model}:{values.size}'
    summary = {
        'size': int(values.size),
        'source_forward_id': token.forward['id'],
        'reason': None,
        'logits_metadata': {**metadata, 'vocab_identity': vocabulary},
    }
    return _Logits(values, vocabulary, summary)

  def activations(self, token: _Token) -> np.ndarray:
    """The final hidden state of the forward that produced the token."""
    return self._vector(
        token, 'post_activations', 'Last hidden-state activation'
    )[0]

  def _vector(
      self, token: _Token, key: str, name: str
  ) -> tuple[np.ndarray, dict]:
    file = token.forward['tensors'].get(key)
    if file is None:
      raise ValueError(f'{name} not captured')
    tensor = tensors.read_tensor(self.index.path(token.side, self.n, file))
    # A decode forward computes one position: every axis but the last is 1.
    if tensor.ndim == 0 or any(size != 1 for size in tensor.shape[:-1]):
      raise ValueError('Token output layout is not a single causal-LM position')
    if not tensor.size or tensor.dtype.kind not in 'fiu':
      raise ValueError('Token output is not a numeric vector')
    metadata = {
        'source_shape': list(tensor.shape),
        'source_dtype': str(tensor.dtype),
        'comparison_shape': [int(tensor.shape[-1])],
    }
    return tensor.reshape(-1).astype(np.float64), metadata

  def labels(self, sides: list[str]) -> dict[int, str]:
    """Token texts the sides' captures name, by id (earlier sides first).

    Only texts the chat itself produced (prompt tokens, released tokens) are
    known; there is no tokenizer, so any other id has no label.
    """
    result = {}
    for side in sides:
      if side not in self._labels:
        texts = {}
        for turn in self.index.turns:
          entry = turn['sides'][side]
          for token in entry['input'] + entry['output']:
            if token.get('text') is not None:
              texts.setdefault(token['id'], token['text'])
        self._labels[side] = texts
      for identity, text in self._labels[side].items():
        result.setdefault(identity, text)
    return result


def _context_same(tokens: dict[str, _Token]) -> bool | None:
  """Whether both sides fed the same tokens before the pair (None: unpaired)."""
  if len(tokens) != 2:
    return None
  contexts = [tokens[side].context for side in SIDES]
  # A token the runner did not name proves nothing, not even on both sides.
  if any(id is None for context in contexts for _, id in context):
    return False
  return contexts[0] == contexts[1]


def _compatibility(
    sources: dict[str, _Logits], context: bool | None, reasons: dict[str, str]
) -> tuple[bool, bool, str | None]:
  """Whether two logits vectors share a vocabulary, and may be compared.

  Args:
    sources: The logits of each side that has them.
    context: `_context_same` of the pair.
    reasons: Why a side has no logits.

  Returns:
    `(vocabulary, compatible, reason)`: `vocabulary` is true when both sides
    have logits of one vocabulary; `compatible` also needs the same context;
    `reason` says why they are not compatible (None when they are).
  """
  both = 'ref' in sources and 'target' in sources
  vocabulary = (
      both
      and sources['ref'].vocabulary == sources['target'].vocabulary
      and sources['ref'].values.shape == sources['target'].values.shape
  )
  if both:
    reason = None if vocabulary else VOCABULARY_REASON
  else:
    reason = '; '.join(f'{side}: {text}' for side, text in reasons.items())
  if context is False:
    return vocabulary, False, CONTEXT_REASON
  return vocabulary, vocabulary, reason


def _transformed(
    source: _Logits | None, transform: str
) -> tuple[np.ndarray | None, str | None]:
  if source is None:
    return None, None
  if transform == 'raw':
    return source.values, None
  return metrics.centered(source.values)


def _side_summary(
    source: _Logits | None,
    distribution: metrics.Distribution | None,
    token: _Token | None,
    reason: str | None,
) -> dict:
  """A side's `TokenDistributionSide`."""
  summary = dict(source.summary) if source else {}
  if distribution is None:
    summary['reason'] = reason
    return summary
  known = token.id is not None
  selected = token.id if known and 0 <= token.id < distribution.p.size else None
  summary.update(
      entropy=distribution.entropy,
      selected_id=selected,
      selected_rank=(
          int(distribution.ranks[selected]) if selected is not None else None
      ),
      selected_probability=(
          float(distribution.p[selected]) if selected is not None else None
      ),
      margin=distribution.margin,
      reason=(
          None
          if selected is not None
          else 'Selected token ID is outside captured vocabulary'
          if known
          else 'The runner does not name the sampled stop token'
      ),
  )
  return summary


def _sampling_side(
    turn: _Turn,
    side: str,
    token: _Token | None,
    source: _Logits | None,
    reason: str | None,
) -> dict:
  """The post-sampler distribution of one side, where it can be rebuilt.

  The capture holds logits before sampling and the sampled id, not the
  sampler's own probabilities. Only greedy sampling (top_k 1) makes them
  known: all probability on the argmax, which must be the sampled id.
  """
  options = turn.index.sides[side]['chat']['options']
  sampler = {
      key: options.get(key) for key in ('temperature', 'top_k', 'top_p', 'seed')
  }
  result = {
      **(source.summary if source else {}),
      'entropy': None,
      'selected_probability': None,
      'margin': None,
      'selected_rank': None,
      'candidate_count': None,
      'basis': None,
      'sampler': sampler,
      'reason': None,
  }
  if token is None or source is None:
    result['reason'] = (
        reason or 'Source logits and sampled token are not captured'
    )
    return result
  result['selected_id'] = token.id
  logits = source.values
  top_k = options.get('top_k')
  if type(top_k) is not int or top_k != 1:
    result['reason'] = (
        'Post-sampler probabilities not captured; only greedy sampling '
        '(top_k 1) can be reconstructed'
    )
  elif options.get('runtime_options', {}).get('speculative_decoding') is True:
    result['reason'] = (
        'Post-sampler probabilities not captured; speculative decoding was on'
    )
  elif (
      np.any(np.isnan(logits))
      or np.any(np.isposinf(logits))
      or not np.any(np.isfinite(logits))
      or not 0 <= token.id < logits.size
      or token.id != int(np.argmax(logits))
  ):
    result['reason'] = (
        'Greedy reconstruction does not match the sampled token and source'
        ' logits'
    )
  else:
    result.update(
        basis='reconstructed_greedy',
        candidate_count=1,
        entropy=0.0,
        selected_probability=1.0,
        selected_rank=1,
        margin=1.0,
    )
  return result


def _sampling_distribution(
    turn: _Turn,
    tokens: dict[str, _Token],
    sources: dict[str, _Logits],
    reasons: dict[str, str],
    context: bool | None,
) -> dict:
  """The pair's `SamplingDistribution`."""
  sides = {
      side: _sampling_side(
          turn, side, tokens.get(side), sources.get(side), reasons.get(side)
      )
      for side in SIDES
  }
  vocabulary, _, _ = _compatibility(sources, context, reasons)
  available = {
      side: value for side, value in sides.items() if value['basis'] is not None
  }
  # Ids of two different vocabularies are never merged into one row; a
  # single available side still lists its own candidates.
  mergeable = len(available) < 2 or vocabulary
  rows = []
  if mergeable:
    for identity in dict.fromkeys(
        value['selected_id'] for value in available.values()
    ):
      owner = next(
          side
          for side, value in available.items()
          if value['selected_id'] == identity
      )
      row = {'id': identity, 'label': turn.labels([owner]).get(identity)}
      for side in SIDES:
        selected = (
            side in available and identity == available[side]['selected_id']
        )
        row[side] = (
            {
                'probability': 1.0 if selected else 0.0,
                'rank': 1 if selected else None,
                'selected': selected,
            }
            if side in available
            else None
        )
      rows.append(row)
  if not mergeable:
    reason = VOCABULARY_REASON
  else:
    reason = (
        '; '.join(
            f'{side}: {value["reason"]}'
            for side, value in sides.items()
            if value['reason']
        )
        or None
    )
  return {
      **sides,
      'context': (
          None if context is None else 'same' if context else 'different'
      ),
      'vocabulary_compatible': vocabulary,
      'rows_complete': bool(available) and mergeable,
      'reason': reason,
      'rows': rows,
  }


def _activation_metrics(
    turn: _Turn, tokens: dict[str, _Token], context: bool | None
) -> dict:
  try:
    if context is None:
      raise ValueError('Paired source forwards not captured')
    if not context:
      raise ValueError(CONTEXT_REASON)
    if tokens['ref'].model != tokens['target'].model:
      raise ValueError(
          'Matching activation coordinate spaces are not established'
      )
    return metrics.activation_metrics(
        turn.activations(tokens['ref']), turn.activations(tokens['target'])
    )
  except (OSError, ValueError) as error:
    return {key: metrics.missing(str(error)) for key in metrics.ACTIVATION_KEYS}


def _candidate_rows(
    turn: _Turn,
    tokens: dict[str, _Token],
    sources: dict[str, _Logits],
    distributions: dict[str, metrics.Distribution],
    compatible: bool,
) -> list[dict]:
  """Pinned sampled ids, then the most likely ids of either side."""
  present = [side for side in SIDES if side in distributions]
  selected = {
      side: tokens[side].id
      for side in present
      if tokens[side].id is not None
      and 0 <= tokens[side].id < distributions[side].p.size
  }
  pinned = list(dict.fromkeys(selected.values()))
  pool = {
      int(identity)
      for side in present
      for identity in distributions[side].order[:CANDIDATES]
  } - set(pinned)
  ids = pinned + sorted(
      pool,
      key=lambda identity: (
          -max(float(distributions[side].p[identity]) for side in present),
          identity,
      ),
  )
  labels = turn.labels(present)
  rows = []
  for identity in ids[:CANDIDATES]:
    row = {'id': identity, 'label': labels.get(identity), 'delta': None}
    for side in SIDES:
      if side not in distributions:
        row[side] = None
        continue
      distribution = distributions[side]
      logit = float(sources[side].values[identity])
      row[side] = {
          'probability': float(distribution.p[identity]),
          'rank': int(distribution.ranks[identity]),
          # Logits a distribution accepts are finite or -Infinity.
          'logit': metrics.json_number(logit),
          'selected': selected.get(side) == identity,
      }
    if compatible:
      row['delta'] = float(
          (
              distributions['target'].p[identity]
              - distributions['ref'].p[identity]
          )
          * 100
      )
    rows.append(row)
  return rows


def _pair(turn: _Turn, ref_step: int | None, target_step: int | None) -> dict:
  """The `TokenAnalysisPair` of one ref token and one target token."""
  steps = {'ref': ref_step, 'target': target_step}
  tokens, sources, distributions, reasons = {}, {}, {}, {}
  for side in SIDES:
    try:
      tokens[side] = turn.token(side, steps[side])
      sources[side] = turn.logits(tokens[side])
      distributions[side] = metrics.distribution(sources[side].values)
    except (OSError, ValueError) as error:
      reasons[side] = str(error)
  context = _context_same(tokens)
  vocabulary, compatible, reason = _compatibility(
      {side: sources[side] for side in distributions}, context, reasons
  )

  values = {
      key: metrics.missing('Paired token evidence not captured')
      for key in TOKEN_METRIC_KEYS
  }
  if compatible:
    ref, target = distributions['ref'], distributions['target']
    kl = metrics.kl_divergence(ref, target)
    values['kl'] = (
        metrics.metric(kl)
        if np.isfinite(kl)
        else metrics.missing(
            'Divergence is infinite: target assigns zero probability'
        )
    )
    values['js'] = metrics.metric(metrics.js_divergence(ref, target))
  else:
    values['kl'] = values['js'] = metrics.missing(
        reason or 'Full-vocabulary logits not captured'
    )
  values.update(_activation_metrics(turn, tokens, context))

  # Rows are keyed by vocabulary id, so they need one vocabulary but not one
  # context: once the generations diverge, each side shows its own
  # candidates, without a delta.
  rows = []
  if len(distributions) < 2 or vocabulary:
    rows = _candidate_rows(turn, tokens, sources, distributions, compatible)

  _, logits_compatible, logits_reason = _compatibility(
      sources, context, reasons
  )
  final = {}
  for transform in ('raw', 'centered'):
    ref_values, ref_reason = _transformed(sources.get('ref'), transform)
    target_values, target_reason = _transformed(
        sources.get('target'), transform
    )
    if not logits_compatible:
      final[transform] = metrics.missing_logits_errors(
          logits_reason or NOT_PAIRED_REASON
      )
    elif ref_reason or target_reason:
      final[transform] = metrics.missing_logits_errors(
          '; '.join(text for text in (ref_reason, target_reason) if text)
      )
    else:
      final[transform] = metrics.logits_errors(ref_values, target_values)

  return {
      'ref_step': ref_step,
      'target_step': target_step,
      'metrics': values,
      'final_logits': final,
      'sampling_distribution': _sampling_distribution(
          turn, tokens, sources, reasons, context
      ),
      'distribution': {
          'compatible': compatible,
          'reason': None if compatible else reason,
          'context': (
              None
              if len(distributions) != 2
              else 'same'
              if context
              else 'different'
          ),
          **{
              side: _side_summary(
                  sources.get(side),
                  distributions.get(side),
                  tokens.get(side),
                  reasons.get(side),
              )
              for side in SIDES
          },
          'rows': rows,
      },
  }


@dataclasses.dataclass(frozen=True)
class _Inspection:
  """Both sides' whole logits vectors of a pair, after one transform."""

  values: dict[str, np.ndarray | None]
  sides: dict[str, dict]
  compatible: bool
  reason: str | None
  labels: dict[int, str]
  metrics: dict
  total: int


def _inspection(
    turn: _Turn, ref_step: int | None, target_step: int | None, transform: str
) -> _Inspection:
  steps = {'ref': ref_step, 'target': target_step}
  tokens, sources, reasons = {}, {}, {}
  for side in SIDES:
    try:
      tokens[side] = turn.token(side, steps[side])
      sources[side] = turn.logits(tokens[side])
    except (OSError, ValueError) as error:
      reasons[side] = str(error)
  vocabulary, compatible, reason = _compatibility(
      sources, _context_same(tokens), reasons
  )
  values, sides = {}, {}
  for side in SIDES:
    source = sources.get(side)
    values[side], transform_reason = _transformed(source, transform)
    sides[side] = (
        dict(source.summary) if source else {'reason': reasons.get(side)}
    )
    if transform_reason:
      sides[side]['reason'] = reasons[side] = transform_reason
  if compatible and any(values[side] is None for side in SIDES):
    compatible = False
    reason = '; '.join(f'{side}: {text}' for side, text in reasons.items())
  if not compatible:
    reason = reason or NOT_PAIRED_REASON
    errors = metrics.missing_logits_errors(reason)
  else:
    errors = metrics.logits_errors(values['ref'], values['target'])
  # Two vocabularies may share indices, but not a label: labels come from
  # one vocabulary only.
  if vocabulary or len(sources) == 1:
    labels = turn.labels(list(sources))
  else:
    labels = {}
  return _Inspection(
      values=values,
      sides=sides,
      compatible=compatible,
      reason=None if compatible else reason,
      labels=labels,
      metrics=errors,
      total=max((source.values.size for source in sources.values()), default=0),
  )


def _order(inspection: _Inspection, order: str, direction: str) -> np.ndarray:
  """Vocabulary indices sorted by a value; missing/non-finite values last."""
  indices = np.arange(inspection.total)
  values = np.full(inspection.total, np.nan)
  if order in SIDES:
    side = inspection.values[order]
    if side is not None:
      values[: side.size] = side
  elif inspection.compatible:
    with np.errstate(over='ignore', invalid='ignore'):
      values = inspection.values['target'] - inspection.values['ref']
      if order == 'absolute_delta':
        values = np.abs(values)
  finite = np.isfinite(values)
  keys = np.where(finite, values if direction == 'ascending' else -values, 0.0)
  return np.lexsort((indices, keys, ~finite))


def _element(values: np.ndarray | None, index: int) -> float | None:
  if values is None or index >= values.size:
    return None
  return float(values[index])


def _is_step(value: object) -> bool:
  return value is None or (type(value) is int and value >= 0)


def _check_pair_payload(payload: dict) -> None:
  """Checks the turn, steps and transform every final-logits request has."""
  if type(payload['turn']) is not int or payload['turn'] < 1:
    raise ValueError('Expected a positive turn')
  if not all(_is_step(payload[key]) for key in ('ref_step', 'target_step')):
    raise ValueError('Expected original ref and target generation steps')
  if payload['transform'] not in ('raw', 'centered'):
    raise ValueError('Expected raw/centered transform')


def token_analysis(index: capture_index.CaptureIndex, payload: dict) -> dict:
  """POST /api/token-analysis: metrics and distributions of token pairs.

  Args:
    index: The chat's capture.
    payload: `{turn, pairs: [{ref, target}]}`; each value is a token's ordinal
      in its side's `output`, or None. At most 128 pairs.

  Returns:
    The UI's `TokenAnalysisResponse`: `{turn, pairs}`, one entry per
    requested pair, in order.

  Raises:
    ValueError: The payload is malformed.
    NotFound: The turn was not captured.
  """
  turn, pairs = payload.get('turn'), payload.get('pairs')
  if (
      type(turn) is not int
      or turn < 1
      or not isinstance(pairs, list)
      or len(pairs) > MAX_PAIRS
  ):
    raise ValueError(f'Expected one turn and at most {MAX_PAIRS} token pairs')
  for pair in pairs:
    if (
        not isinstance(pair, dict)
        or set(pair) != {'ref', 'target'}
        or not all(_is_step(value) for value in pair.values())
    ):
      raise ValueError('Expected original ref and target generation steps')
  view = _Turn(index, turn)
  return {
      'turn': turn,
      'pairs': [_pair(view, pair['ref'], pair['target']) for pair in pairs],
  }


def final_logits(index: capture_index.CaptureIndex, payload: dict) -> dict:
  """POST /api/final-logits: one page of a pair's sorted logits.

  Args:
    index: The chat's capture.
    payload: `{turn, ref_step, target_step, transform: raw|centered, offset,
      limit: 1..256, order: index|ref|target|delta|absolute_delta, direction?:
        ascending|descending}` (ascending by default).

  Returns:
    The UI's `FinalLogitsResponse`. Sorting covers the whole vocabulary;
    missing and non-finite values come last either way, ties by index.

  Raises:
    ValueError: The payload is malformed, or offset is past the vocabulary.
    NotFound: The turn was not captured.
  """
  keys = {
      'turn',
      'ref_step',
      'target_step',
      'transform',
      'offset',
      'limit',
      'order',
  }
  if not isinstance(payload, dict) or set(payload) not in (
      keys,
      keys | {'direction'},
  ):
    raise ValueError(
        'Expected turn, ref_step, target_step, transform, offset, limit, '
        'order and optional direction'
    )
  _check_pair_payload(payload)
  order, offset, limit = payload['order'], payload['offset'], payload['limit']
  direction = payload.get('direction', 'ascending')
  if order not in ('index', 'ref', 'target', 'delta', 'absolute_delta'):
    raise ValueError('Expected index/ref/target/delta/absolute_delta order')
  if direction not in ('ascending', 'descending'):
    raise ValueError('Expected ascending or descending direction')
  if (
      type(offset) is not int
      or offset < 0
      or type(limit) is not int
      or not 1 <= limit <= MAX_FINAL_LOGITS_ROWS
  ):
    raise ValueError(
        f'Expected offset >= 0 and limit between 1 and {MAX_FINAL_LOGITS_ROWS}'
    )
  turn, ref_step, target_step = (
      payload['turn'],
      payload['ref_step'],
      payload['target_step'],
  )
  data = _inspection(
      _Turn(index, turn), ref_step, target_step, payload['transform']
  )
  if offset > data.total:
    raise ValueError('Final logits offset exceeds the vocabulary length')
  if order == 'index':
    indices = range(data.total)
    if direction == 'descending':
      indices = indices[::-1]
    indices = indices[offset : offset + limit]
  else:
    indices = _order(data, order, direction)[offset : offset + limit]
  rows = []
  for index_value in indices:
    vocabulary_index = int(index_value)
    ref, target = (
        _element(data.values[side], vocabulary_index) for side in SIDES
    )
    delta = None
    if data.compatible:
      with np.errstate(over='ignore', invalid='ignore'):
        delta = target - ref
    rows.append({
        'index': vocabulary_index,
        'label': data.labels.get(vocabulary_index),
        'ref': metrics.json_number(ref),
        'target': metrics.json_number(target),
        'delta': metrics.json_number(delta),
        'absolute_delta': metrics.json_number(
            abs(delta) if delta is not None else None
        ),
    })
  return {
      'turn': turn,
      'ref_step': ref_step,
      'target_step': target_step,
      'transform': payload['transform'],
      'compatible': data.compatible,
      'reason': data.reason,
      'total': data.total,
      'offset': offset,
      'limit': limit,
      'rows': rows,
      'ref': data.sides['ref'],
      'target': data.sides['target'],
      'metrics': data.metrics,
  }


def final_logits_vector(
    index: capture_index.CaptureIndex, payload: dict
) -> dict:
  """POST /api/final-logits-vector: both whole logits vectors of a pair.

  Args:
    index: The chat's capture.
    payload: `{turn, ref_step, target_step, transform: raw|centered}`.

  Returns:
    The UI's `FinalLogitsVectorResponse`: `values.{ref,target}` is the whole
    vector indexed by vocabulary id, or None for a side without one.

  Raises:
    ValueError: The payload is malformed, or a vector is longer than
      `MAX_VECTOR_VALUES` (it is never truncated).
    NotFound: The turn was not captured.
  """
  keys = {'turn', 'ref_step', 'target_step', 'transform'}
  if not isinstance(payload, dict) or set(payload) != keys:
    raise ValueError('Expected turn, ref_step, target_step and transform')
  _check_pair_payload(payload)
  data = _inspection(
      _Turn(index, payload['turn']),
      payload['ref_step'],
      payload['target_step'],
      payload['transform'],
  )
  if data.total > MAX_VECTOR_VALUES:
    raise ValueError(
        f'Full final-logits vector exceeds the {MAX_VECTOR_VALUES} element '
        'budget; no values were truncated'
    )
  return {
      'turn': payload['turn'],
      'ref_step': payload['ref_step'],
      'target_step': payload['target_step'],
      'transform': payload['transform'],
      'compatible': data.compatible,
      'reason': data.reason,
      'total': data.total,
      'complete': True,
      'values': {
          side: None if values is None else metrics.json_numbers(values)
          for side, values in data.values.items()
      },
      'ref': data.sides['ref'],
      'target': data.sides['target'],
      'metrics': data.metrics,
  }
