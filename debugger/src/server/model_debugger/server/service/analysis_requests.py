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

"""The requests of the analysis routes: read, and their rows looked up.

A request names rows of the server's own comparison of a turn
(service/session_view): its `turn`, its `stage` (`prefill`, the turn's input
aligned by content, or `decode`, its generated tokens), its `alignment`
(`content` or `steps`; a Prefill has only `content`) and the rows, by index.
The tokens, steps and positions a row stands for are looked up here, from
that comparison; no request carries them. The analyses (analysis/) are then
called with plain values, and every answer repeats what its request named.

A JSON body is read for POST routes, the query string for GET routes, whose
values arrive as strings.

The analysis routes run on threads of their own (THREADS), so the other
routes never wait behind them. token-analysis and token-range share each
chat's token pairs (service/pair_values): a pair is made once, whoever asks
first, and token-range's are kept.
"""

from collections.abc import Callable
import json
import math
from typing import Any

from model_debugger.server.analysis import alignment as alignment_lib
from model_debugger.server.analysis import kv_analysis
from model_debugger.server.analysis import tensor_analysis
from model_debugger.server.analysis import token_analysis as token_analysis_lib
from model_debugger.server.analysis import vector_drawing
from model_debugger.server.capture import capture_index
from model_debugger.server.service import pair_values
from model_debugger.server.service import session_view
from model_debugger.server.util import errors

STAGES = ('prefill', 'decode')
# How many analysis requests are answered at a time; a request waiting for
# token pairs others make holds one of them (pair_values.POLL_SECONDS).
THREADS = 8


def _present() -> bool:
  return False


class Comparison:
  """The comparison of a request's chat, and whether its client is gone."""

  def __init__(
      self,
      make: Callable[[], session_view.Compared],
      gone: pair_values.Gone = _present,
  ) -> None:
    """A request's chat.

    Args:
        make: Returns the chat's comparison, made (or kept) when first asked
          for.
        gone: Whether the request's client is gone; it is asked while the
          request waits for token pairs.
    """
    self._make = make
    self.gone = gone

  def __call__(self) -> session_view.Compared:
    """The chat's comparison."""
    return self._make()


class _Rows:
  """The rows of one stage of one turn, and the tokens they index."""

  def __init__(
      self, view: session_view.Compared, n: int, stage: str, kind: str
  ) -> None:
    self.view = view
    comparison = view.comparisons[n]
    key = 'input_tokens' if stage == 'prefill' else 'tokens'
    self.tokens = {
        side: view.conversation[n, side][key] for side in capture_index.SIDES
    }
    self.compared = (
        comparison['prefill']
        if stage == 'prefill'
        else comparison['alignments'][kind]
    )
    self.rows = self.compared['rows']

  def steps(self, row: dict[str, Any]) -> tuple[int | None, int | None]:
    """Each side's step of a row's tokens, or None where it has none."""
    return tuple(
        None if row[side] is None else self.tokens[side][row[side]]['step']
        for side in capture_index.SIDES
    )

  def divergence(self) -> dict[str, Any] | None:
    """The first divergence's row, filled in by step; None without."""
    return self.compared.get('first_divergence')


# Reading requests.


def _turn(index: capture_index.CaptureIndex, value: object) -> int:
  """A turn the index read; raises NotFound or Conflict as index.turn."""
  if (not isinstance(value, int) or isinstance(value, bool)) or value < 1:
    raise ValueError('Name the turn as a positive integer.')
  index.turn(value)
  return value


def _stage_and_alignment(stage: object, kind: object) -> tuple[str, str]:
  if stage not in STAGES:
    raise ValueError('stage must be prefill or decode.')
  if kind not in alignment_lib.ALIGNMENTS:
    raise ValueError('alignment must be content or steps.')
  if stage == 'prefill' and kind != 'content':
    raise ValueError('A Prefill is aligned by content only.')
  return stage, kind


def _row(value: object, count: int) -> int:
  if (
      not isinstance(value, int) or isinstance(value, bool)
  ) or not 0 <= value < count:
    raise ValueError(f"A row is an index of the stage's {count} rows.")
  return value


def _keys(
    payload: dict[str, Any], required: set[str], optional: set[str]
) -> None:
  if not required <= set(payload) <= required | optional:
    raise ValueError(
        'Expected '
        + ', '.join(sorted(required))
        + (' and optional ' + ', '.join(sorted(optional)) if optional else '')
    )


def _address(
    index: capture_index.CaptureIndex,
    comparison: Comparison,
    payload: dict[str, Any],
    tokens: bool = False,
) -> tuple[dict[str, Any], _Rows]:
  """The turn, stage and alignment a JSON request names, and its rows.

  Args:
      index: The chat's capture.
      comparison: The chat's comparison.
      payload: The request.
      tokens: Whether the request is of a token analysis, whose stages
        token_analysis.stage_reason allows.

  Returns:
      A tuple of (echo_dict, rows).
  """
  n = _turn(index, payload['turn'])
  stage, kind = _stage_and_alignment(payload['stage'], payload['alignment'])
  why = token_analysis_lib.stage_reason(stage)
  if tokens and why is not None:
    raise ValueError(why['text'] + '.')
  echo = {'turn': n, 'stage': stage, 'alignment': kind}
  return echo, _Rows(comparison(), n, stage, kind)


def _decode_pair(rows: _Rows, row: int, divergence: bool):
  """The steps of a Decode row, or of the first divergence at that row."""
  if not divergence:
    return rows.steps(rows.rows[row])
  found = rows.divergence()
  if found is None or found['row'] != row:
    raise ValueError(f'Row {row} is not the first divergence.')
  return rows.steps(found)


def _flag(payload: dict[str, Any], name: str) -> bool:
  value = payload.get(name, False)
  if not isinstance(value, bool):
    raise ValueError(f'{name} must be true or false.')
  return value


# The token routes: JSON bodies.


def token_analysis(
    index: capture_index.CaptureIndex,
    comparison: Comparison,
    payload: dict[str, Any],
) -> dict[str, Any]:
  """POST /api/token-analysis: the metrics of rows of a turn's Decode.

  Args:
      index: The chat's capture.
      comparison: The chat's comparison.
      payload: `{turn, stage: 'decode', alignment, rows, first_divergence?}`: up
        to MAX_PAIRS rows by index, and with `first_divergence` the first
        divergence's pair too (its row, filled in by step).

  Returns:
      The UI's `TokenAnalysisResponse`: `{turn, stage, alignment, pairs}`;
      each pair says which row it is (`row`, `first_divergence`) and
      which steps it compared; the first divergence's comes last.

  Raises:
      ValueError: The payload is malformed.
      pair_values.AbandonedError: The client is gone.
      NotFound: The turn was not captured.
      Conflict: The capture index rejected the turn.
  """
  _keys(payload, {'turn', 'stage', 'alignment', 'rows'}, {'first_divergence'})
  echo, rows = _address(index, comparison, payload, tokens=True)
  wanted, divergence = payload['rows'], _flag(payload, 'first_divergence')
  if (
      not isinstance(wanted, list)
      or len(wanted) + divergence > token_analysis_lib.MAX_PAIRS
  ):
    raise ValueError(
        f'Name at most {token_analysis_lib.MAX_PAIRS} rows, the first '
        'divergence among them.'
    )
  named = [(_row(k, len(rows.rows)), False) for k in wanted]
  if divergence:
    found = rows.divergence()
    if found is None:
      raise ValueError('The turn has no first divergence to analyze.')
    named.append((found['row'], True))
  steps = [_decode_pair(rows, row, filled) for row, filled in named]
  pairs = rows.view.pairs.pairs(
      rows.view.index, echo['turn'], steps, comparison.gone
  )
  return {
      **echo,
      'pairs': [
          {
              'stage': echo['stage'],
              'alignment': echo['alignment'],
              'row': row,
              'first_divergence': filled,
              **pair,
          }
          for (row, filled), pair in zip(named, pairs, strict=True)
      ],
  }


def token_range(
    index: capture_index.CaptureIndex,
    comparison: Comparison,
    payload: dict[str, Any],
) -> dict[str, Any]:
  """POST /api/token-range: one metric of a turn's Decode rows, binned.

  Args:
      index: The chat's capture.
      comparison: The chat's comparison.
      payload: `{turn, stage: 'decode', alignment, analysis, metric, start, end,
        bins}`: `analysis` and `metric` as token_analysis.RANGE_METRICS; `start`
        and `end` columns (the KV analyses' columns: row k is the stage's first
        column + k), end past start; `bins` 1..token_analysis.MAX_RANGE_BINS.

  Returns:
      token_analysis.token_range's answer, the UI's `TokenRangeResponse`,
      with what the request named. Each pair's values are made once per
      chat (session_view.Compared.pairs).

  Raises:
      ValueError: The payload is malformed, or the range reaches past the
        stage's columns.
      pair_values.AbandonedError: The client is gone.
      NotFound: The turn was not captured.
      Conflict: The capture index rejected the turn.
  """
  _keys(
      payload,
      {
          'turn',
          'stage',
          'alignment',
          'analysis',
          'metric',
          'start',
          'end',
          'bins',
      },
      set(),
  )
  echo, rows = _address(index, comparison, payload, tokens=True)
  analysis, metric = payload['analysis'], payload['metric']
  if metric not in token_analysis_lib.RANGE_METRICS.get(analysis, ()):
    raise ValueError(
        'Name an analysis and one of its metrics: '
        + '; '.join(
            f'{name}: {", ".join(keys)}'
            for name, keys in token_analysis_lib.RANGE_METRICS.items()
        )
    )
  start, end, bins = payload['start'], payload['end'], payload['bins']
  if (
      (not isinstance(start, int) or isinstance(start, bool))
      or (not isinstance(end, int) or isinstance(end, bool))
      or start < 0
      or end <= start
  ):
    raise ValueError('The column range [start, end) must not be empty.')
  if (
      not isinstance(bins, int) or isinstance(bins, bool)
  ) or not 1 <= bins <= token_analysis_lib.MAX_RANGE_BINS:
    raise ValueError(f'Expected 1 to {token_analysis_lib.MAX_RANGE_BINS} bins.')
  n = echo['turn']
  first = rows.view.columns(n, echo['stage'], echo['alignment']).start
  if not first <= start < end <= first + len(rows.rows):
    raise ValueError(
        "The column range [start, end) must lie in the stage's columns "
        f'[{first}, {first + len(rows.rows)}).'
    )
  steps = [rows.steps(row) for row in rows.rows]
  found = rows.divergence()
  fork = found if found is not None and found.get('filled') else None
  wanted = steps + ([rows.steps(fork)] if fork is not None else [])
  values = rows.view.pairs.values(rows.view.index, n, wanted, comparison.gone)
  stage = [
      token_analysis_lib.RangeRow(
          ref=row['ref'] is not None,
          target=row['target'] is not None,
          steps=key,
          values=value,
      )
      for row, key, value in zip(rows.rows, steps, values)
  ]
  divergence = None
  if fork is not None:
    divergence = (
        fork['row'],
        token_analysis_lib.RangeRow(
            ref=fork['ref'] is not None,
            target=fork['target'] is not None,
            steps=wanted[-1],
            values=values[-1],
        ),
    )
  answer = token_analysis_lib.token_range(
      first, stage, divergence, analysis, metric, start, end, bins
  )
  return {**echo, 'analysis': analysis, 'metric': metric, **answer}


def _logits_address(
    index: capture_index.CaptureIndex,
    comparison: Comparison,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], tuple[int | None, int | None], str]:
  """Extracts logits address information from the payload.

  Args:
    index: The capture index.
    comparison: The chat comparison.
    payload: The request payload.

  Returns:
    A tuple of (echo_dict, (ref_step, target_step), transform).
  """
  echo, rows = _address(index, comparison, payload, tokens=True)
  row = _row(payload['row'], len(rows.rows))
  divergence = _flag(payload, 'first_divergence')
  steps = _decode_pair(rows, row, divergence)
  transform = payload['transform']
  if transform not in token_analysis_lib.TRANSFORMS:
    raise ValueError('Expected raw/centered transform')
  return (
      {**echo, 'row': row, 'first_divergence': divergence},
      steps,
      transform,
  )


def final_logits(
    index: capture_index.CaptureIndex,
    comparison: Comparison,
    payload: dict[str, Any],
) -> dict[str, Any]:
  """POST /api/final-logits: one page of a Decode row's sorted logits.

  Args:
      index: The chat's capture.
      comparison: The chat's comparison.
      payload: `{turn, stage: 'decode', alignment, row, first_divergence?,
        transform: raw|centered, offset, limit: 1..256,
        order: index|ref|target|delta|absolute_delta, direction?:
          ascending|descending}` (ascending by default).

  Returns:
      The UI's `FinalLogitsResponse`.

  Raises:
      ValueError: The payload is malformed, or offset is past the vocabulary.
      NotFound: The turn was not captured.
      Conflict: The capture index rejected the turn.
  """
  _keys(
      payload,
      {
          'turn',
          'stage',
          'alignment',
          'row',
          'transform',
          'offset',
          'limit',
          'order',
      },
      {'first_divergence', 'direction'},
  )
  echo, steps, transform = _logits_address(index, comparison, payload)
  order, offset, limit = payload['order'], payload['offset'], payload['limit']
  direction = payload.get('direction', 'ascending')
  if order not in token_analysis_lib.ORDERS:
    raise ValueError('Expected index/ref/target/delta/absolute_delta order')
  if direction not in token_analysis_lib.DIRECTIONS:
    raise ValueError('Expected ascending or descending direction')
  if (
      (not isinstance(offset, int) or isinstance(offset, bool))
      or offset < 0
      or (not isinstance(limit, int) or isinstance(limit, bool))
      or not 1 <= limit <= token_analysis_lib.MAX_FINAL_LOGITS_ROWS
  ):
    raise ValueError(
        'Expected offset >= 0 and limit between 1 and '
        f'{token_analysis_lib.MAX_FINAL_LOGITS_ROWS}'
    )
  page = token_analysis_lib.final_logits(
      index, echo['turn'], *steps, transform, offset, limit, order, direction
  )
  return {**echo, **page}


_ADDRESS = {'turn', 'stage', 'alignment', 'row', 'transform'}


def final_logits_summary(
    index: capture_index.CaptureIndex,
    comparison: Comparison,
    payload: dict[str, Any],
) -> dict[str, Any]:
  """POST /api/final-logits-summary: what is known of a row's logits.

  Args:
      index: The chat's capture.
      comparison: The chat's comparison.
      payload: `{turn, stage: 'decode', alignment, row, first_divergence?,
        transform: raw|centered}`.

  Returns:
      The UI's `FinalLogitsSummaryResponse`: everything the Inspection
      shows of both whole vectors but their values.

  Raises:
      ValueError: The payload is malformed.
      NotFound: The turn was not captured.
      Conflict: The capture index rejected the turn.
  """
  _keys(payload, _ADDRESS, {'first_divergence'})
  echo, steps, transform = _logits_address(index, comparison, payload)
  summary = token_analysis_lib.final_logits_summary(
      index, echo['turn'], *steps, transform
  )
  return {**echo, **summary}


def _still_asked(comparison: Comparison) -> None:
  """Stops a request whose client is gone before it reads the vectors.

  A chart asks anew as its view changes (a drag, a resize) and drops the
  request it no longer wants: reading and drawing two whole vectors for
  nobody would hold up the request that replaced it.

  Args:
      comparison: The chat comparison.

  Raises:
      pair_values.AbandonedError: The client is gone.
  """
  if comparison.gone():
    raise pair_values.AbandonedError()


def final_logits_bins(
    index: capture_index.CaptureIndex,
    comparison: Comparison,
    payload: dict[str, Any],
) -> dict[str, Any]:
  """POST /api/final-logits-bins: the Inspection line chart's bins.

  Args:
      index: The chat's capture.
      comparison: The chat's comparison.
      payload: `{turn, stage: 'decode', alignment, row, first_divergence?,
        transform: raw|centered, start?, end?, capacity, reduction:
          min|max|mean}`: the vocabulary range shown, both or neither (the whole
          vocabulary), kept inside the vocabulary; how many bins the chart has
          room for, 1..token_analysis_lib.MAX_LINE_BINS.

  Returns:
      The UI's `FinalLogitsBinsResponse`.

  Raises:
      ValueError: The payload is malformed.
      NotFound: The turn was not captured.
      Conflict: The capture index rejected the turn.
      pair_values.AbandonedError: The client is gone.
  """
  _keys(
      payload,
      _ADDRESS | {'capacity', 'reduction'},
      {'first_divergence', 'start', 'end'},
  )
  echo, steps, transform = _logits_address(index, comparison, payload)
  elements = _pair_of(payload, int)
  capacity, reduction = payload['capacity'], payload['reduction']
  if (
      not isinstance(capacity, int) or isinstance(capacity, bool)
  ) or not 1 <= capacity <= token_analysis_lib.MAX_LINE_BINS:
    raise ValueError(f'Expected 1 to {token_analysis_lib.MAX_LINE_BINS} bins.')
  if reduction not in vector_drawing.REDUCTIONS:
    raise ValueError('Expected min/max/mean reduction')
  _still_asked(comparison)
  bins = token_analysis_lib.final_logits_bins(
      index, echo['turn'], *steps, transform, elements, capacity, reduction
  )
  return {**echo, **bins}


def final_logits_points(
    index: capture_index.CaptureIndex,
    comparison: Comparison,
    payload: dict[str, Any],
) -> dict[str, Any]:
  """POST /api/final-logits-points: the Inspection scatter plot's points.

  Args:
      index: The chat's capture.
      comparison: The chat's comparison.
      payload: `{turn, stage: 'decode', alignment, row, first_divergence?,
        transform: raw|centered, start?, end?}`: the value range shown, finite
          and increasing, both or neither (the paired values' extent).

  Returns:
      The UI's `FinalLogitsPointsResponse`; of a value range, its
      `FinalLogitsZoomedPointsResponse`, without the extent and the paired
      runs, which do not change with the range.

  Raises:
      ValueError: The payload is malformed.
      NotFound: The turn was not captured.
      Conflict: The capture index rejected the turn.
      pair_values.AbandonedError: The client is gone.
  """
  _keys(payload, _ADDRESS, {'first_divergence', 'start', 'end'})
  echo, steps, transform = _logits_address(index, comparison, payload)
  values = _pair_of(payload, float)
  _still_asked(comparison)
  points = token_analysis_lib.final_logits_points(
      index, echo['turn'], *steps, transform, values
  )
  return {**echo, **points}


def _finite(value: int | float) -> bool:
  """Whether a number is finite as a float; an int too large is not."""
  try:
    return math.isfinite(value)
  except OverflowError:
    return False


def _pair_of(
    payload: dict[str, Any], kind: type[Any]
) -> tuple[Any, ...] | None:
  """A payload's `(start, end)`, start below end; None when it has neither.

  Args:
      payload: The request.
      kind: int, for element indices from 0; float, for finite values (an int is
        one).

  Returns:
      A tuple of (start, end), or None when it has neither.
  """
  if 'start' not in payload and 'end' not in payload:
    return None
  start, end = payload.get('start'), payload.get('end')
  if kind is int:
    valid = all(
        (isinstance(v, int) and not isinstance(v, bool)) and v >= 0
        for v in (start, end)
    )
  else:
    valid = all(type(v) in (int, float) and _finite(v) for v in (start, end))
  if not valid or not start < end:
    raise ValueError('Expected start below end.')
  return (start, end) if kind is int else (float(start), float(end))


def match_rate(
    index: capture_index.CaptureIndex,
    comparison: Comparison,
    payload: dict[str, Any],
) -> dict[str, Any]:
  """POST /api/match-rate: how many of a stage's rows are the same.

  Args:
      index: The chat's capture.
      comparison: The chat's comparison.
      payload: `{turn, stage, alignment, rows?}`: the rows by index, each once;
        every row of the stage without `rows`.

  Returns:
      `{turn, stage, alignment, match_rate}`: the share of the rows that
      are the same (analysis/alignment.match_rate); None when no row is
      named or one named row's sameness is unknown.

  Raises:
      ValueError: The payload is malformed.
      NotFound: The turn was not captured.
      Conflict: The capture index rejected the turn.
  """
  _keys(payload, {'turn', 'stage', 'alignment'}, {'rows'})
  echo, rows = _address(index, comparison, payload)
  selected = payload.get('rows')
  if selected is not None:
    if not isinstance(selected, list):
      raise ValueError('rows must be a list of row indices.')
    selected = [_row(k, len(rows.rows)) for k in selected]
    if len(set(selected)) != len(selected):
      raise ValueError('Name each row once.')
  return {**echo, 'match_rate': alignment_lib.match_rate(rows.rows, selected)}


# The KV routes: query strings.


_REQUIRED = object()


def _text(query: dict[str, Any], name: str) -> str:
  value = query.get(name)
  if not isinstance(value, str) or not value:
    raise ValueError(f'The request needs a {name}.')
  return value


def _integer(
    query: dict[str, Any], name: str, default=_REQUIRED, minimum: int = 0
):
  """Parses an integer from a query dictionary.

  Args:
    query: The query dictionary.
    name: The parameter name to extract.
    default: The default value if not found.
    minimum: The minimum allowed integer value.

  Returns:
    The parsed integer value.

  Raises:
    ValueError: If the value is missing or invalid.
  """
  value = query.get(name)
  if value is None:
    if default is _REQUIRED:
      raise ValueError(f'The request needs a {name}.')
    return default
  if not (isinstance(value, str) and value.isascii() and value.isdigit()):
    raise ValueError(f'{name} must be a nonnegative integer.')
  number = int(value)
  if number < minimum:
    raise ValueError(f'{name} must be at least {minimum}.')
  return number


def _kind(query: dict[str, Any], default: str | None = 'key') -> str:
  kind = query.get('kind', default)
  if kind not in kv_analysis.KINDS:
    raise ValueError('kind must be key or value.')
  return kind


def _head(value: str) -> int | str:
  if value in kv_analysis.HEAD_AGGREGATIONS:
    return value
  if not (value.isascii() and value.isdigit()):
    raise ValueError('head must be max, min, mean or a KV head index.')
  return int(value)


def _heads(value: str | None, head: int | str) -> list[int] | None:
  """Parses a JSON-encoded array of KV head indices.

  Args:
    value: The raw string representation or None.
    head: The head aggregation mode or index.

  Returns:
    A sorted list of head indices, or None if value is None.

  Raises:
    ValueError: If value is malformed or invalid.
  """
  if value is None:
    return None
  try:
    heads = json.loads(value)
  except json.JSONDecodeError as error:
    raise ValueError('heads must be a JSON array of KV heads.') from error
  if (
      not isinstance(heads, list)
      or not heads
      or any(
          (not isinstance(h, int) or isinstance(h, bool)) or h < 0
          for h in heads
      )
      or len(set(heads)) != len(heads)
      or head not in kv_analysis.HEAD_AGGREGATIONS
  ):
    raise ValueError(
        'heads must be a nonempty array of distinct KV head indices, '
        'used with head min, max or mean.'
    )
  return sorted(heads)


def _layers(value: str | None) -> list[int] | None:
  """Parses a JSON-encoded array of layer indices.

  Args:
    value: The raw string representation or None.

  Returns:
    A sorted list of layer indices, or None if value is None.

  Raises:
    ValueError: If value is malformed or invalid.
  """
  if value is None:
    return None
  try:
    layers = json.loads(value)
  except json.JSONDecodeError as error:
    raise ValueError('layers must be a JSON array of layers.') from error
  if (
      not isinstance(layers, list)
      or any(
          (not isinstance(layer, int) or isinstance(layer, bool)) or layer < 0
          for layer in layers
      )
      or len(set(layers)) != len(layers)
  ):
    raise ValueError('layers must be an array of distinct layer indices.')
  return sorted(layers)


def _kv_alignment(
    index: capture_index.CaptureIndex,
    query: dict[str, Any],
    n: int,
    comparison: Comparison,
) -> tuple[dict[str, Any], kv_analysis.Alignment | None]:
  """The stage alignment a KV request reads through, if it names one.

  `stage` and `alignment` together: columns are the stage's rows, from the
  ref side's first position of the stage (analysis/alignment.positions),
  once the turn is known to be read (NotFound or Conflict as index.turn).
  Neither: a column is the position on both sides.

  Args:
    index: The capture index.
    query: The query dictionary.
    n: The turn number.
    comparison: The chat comparison.

  Returns:
    A tuple of (echo_dict, columns).
  """
  stage, kind = query.get('stage'), query.get('alignment')
  if stage is None and kind is None:
    return {}, None
  index.turn(n)
  stage, kind = _stage_and_alignment(stage, kind)
  columns = comparison().columns(n, stage, kind)
  return {'stage': stage, 'alignment': kind}, columns


def kv_metadata(
    index: capture_index.CaptureIndex, query: dict[str, Any]
) -> dict[str, Any]:
  """GET /api/kv-analysis?mode=metadata&turn=: the KV contexts of a turn."""
  if query.get('mode') != 'metadata':
    raise ValueError('Only kv-analysis with mode=metadata is served.')
  return kv_analysis.kv_analysis(index, _integer(query, 'turn', minimum=1))


def kv_range(
    index: capture_index.CaptureIndex,
    comparison: Comparison,
    query: dict[str, Any],
) -> dict[str, Any]:
  """GET /api/kv-range: binned per-position metrics of a kind's layers.

  Args:
      index: The chat's capture.
      comparison: The chat's comparison.
      query: `turn`, `context`, `start`, `end` (columns), `bins` (128), `kind`
        ('key'), `head` ('max', 'min', 'mean' or a KV head index;
        kv_analysis.DEFAULT_HEAD), optional `heads` (a JSON array of KV heads to
        aggregate over), optional `layers` (a JSON array of the layers wanted;
        every layer without), and optional `stage` with `alignment`.

  Returns:
      kv_analysis.kv_range's answer, and the `stage` and `alignment` the
      request named.

  Raises:
      ValueError: The query is malformed.
      NotFound: The turn was not captured.
      Conflict: The capture index rejected the turn.
  """
  n = _integer(query, 'turn', minimum=1)
  context_id = _text(query, 'context')
  start, end = _integer(query, 'start'), _integer(query, 'end')
  if end <= start:
    raise ValueError('The position range [start, end) must not be empty.')
  bins = _integer(query, 'bins', default=128, minimum=1)
  if bins > kv_analysis.MAX_BINS:
    raise ValueError(f'At most {kv_analysis.MAX_BINS} chart bins are allowed.')
  kind = _kind(query)
  head = _head(query.get('head', kv_analysis.DEFAULT_HEAD))
  heads = _heads(query.get('heads'), head)
  layers = _layers(query.get('layers'))
  if query.get('formula'):
    raise ValueError('KV formulas are not supported.')
  echo, alignment = _kv_alignment(index, query, n, comparison)
  answer = kv_analysis.kv_range(
      index,
      n,
      context_id,
      start,
      end,
      bins,
      kind,
      head,
      heads,
      alignment,
      layers,
  )
  return {**answer, **echo}


def kv_head(
    index: capture_index.CaptureIndex,
    comparison: Comparison,
    query: dict[str, Any],
) -> dict[str, Any]:
  """GET /api/kv-head: every KV head's channels at one column, both sides.

  Args:
      index: The chat's capture.
      comparison: The chat's comparison.
      query: `turn`, `context_id`, `layer`, `kind`, `position` (a column),
        optional `batch`, and optional `stage` with `alignment`.

  Returns:
      kv_analysis.kv_head's answer, the UI's KvHeadsEvidence; its
      `selection` names the `stage` and `alignment` the request did.

  Raises:
      ValueError: The query is malformed.
      NotFound: The turn was not captured.
      Conflict: The capture index rejected the turn.
  """
  n = _integer(query, 'turn', minimum=1)
  context_id = _text(query, 'context_id')
  layer = _integer(query, 'layer')
  kind = _kind(query, default=None)
  position = _integer(query, 'position')
  batch = _integer(query, 'batch', default=None)
  echo, alignment = _kv_alignment(index, query, n, comparison)
  answer = kv_analysis.kv_head(
      index, n, context_id, layer, kind, position, batch, alignment
  )
  answer['selection'].update(echo)
  return answer


# The intermediate tensors' routes: query strings.


def _tensor_request(
    index: capture_index.CaptureIndex, query: dict[str, Any]
) -> tuple[int, str, str | None]:
  """The turn, stage and alignment a tensor request names.

  `turn` and `stage`, and optionally `alignment`: with it, columns are the
  stage's rows (analysis/alignment.positions); without, a column is the
  position on both sides.

  Args:
    index: The capture index.
    query: The query dictionary.

  Returns:
    A tuple of (turn, stage, alignment).
  """
  n = _integer(query, 'turn', minimum=1)
  index.turn(n)
  stage = query.get('stage')
  kind = query.get('alignment')
  if kind is None:
    if stage not in STAGES:
      raise ValueError('stage must be prefill or decode.')
  else:
    stage, kind = _stage_and_alignment(stage, kind)
  return n, stage, kind


def _tensor_stage(
    index: capture_index.CaptureIndex,
    comparison: Comparison,
    query: dict[str, Any],
) -> tuple[dict[str, Any], tensor_analysis.Stage]:
  """The stage a tensor request names, compared (Compared.tensors)."""
  n, stage, kind = _tensor_request(index, query)
  echo = {'turn': n, 'stage': stage, 'alignment': kind}
  return echo, comparison().tensors(n, stage, kind)


def _known(asked: list[Any] | None, known: tuple[Any, ...], what: str) -> None:
  """Refuses what a request names that the Session has no place of."""
  unknown = [item for item in asked or [] if item not in known]
  if unknown:
    raise ValueError(
        f'The Session observes no {what} {", ".join(map(str, unknown))}; '
        f'it observes {", ".join(map(str, known)) or "none"}.'
    )


def _anchors(value: str | None) -> list[str] | None:
  """Parses a JSON-encoded array of anchor names.

  Args:
    value: The raw string representation or None.

  Returns:
    A list of anchor names, or None if value is None.

  Raises:
    ValueError: If value is malformed or empty.
  """
  if value is None:
    return None
  try:
    anchors = json.loads(value)
  except json.JSONDecodeError as error:
    raise ValueError('anchors must be a JSON array of anchors.') from error
  if (
      not isinstance(anchors, list)
      or not anchors
      or any(not isinstance(a, str) or not a for a in anchors)
      or len(set(anchors)) != len(anchors)
  ):
    raise ValueError('anchors must be a nonempty array of distinct anchors.')
  return anchors


def position_analysis(
    index: capture_index.CaptureIndex,
    comparison: Comparison,
    query: dict[str, Any],
) -> dict[str, Any]:
  """GET /api/position-analysis: a stage's intermediate tensors, cell by cell.

  Args:
      index: The chat's capture.
      comparison: The chat's comparison.
      query: `turn`, `stage`, and optionally `alignment`.

  Returns:
      tensor_analysis.analysis's answer, with what the request named.

  Raises:
      ValueError: The query is malformed.
      NotFound: The turn was not captured.
      Conflict: The capture index rejected the turn.
  """
  echo, found = _tensor_stage(index, comparison, query)
  return {**echo, **tensor_analysis.analysis(found)}


def position_range(
    index: capture_index.CaptureIndex,
    comparison: Comparison,
    query: dict[str, Any],
) -> dict[str, Any]:
  """GET /api/position-range: each layer's metrics over columns, binned.

  Args:
      index: The chat's capture.
      comparison: The chat's comparison.
      query: `turn`, `stage`, optionally `alignment`; `start`, `end` (the
        columns, within the stage's), `bins` (1..tensor_analysis.MAX_BINS, 128
        by default), optional `anchors` (a JSON array; every anchor without),
        `reduction` (min, max or mean; mean by default) and optional `layers` (a
        JSON array; every layer without).

  Returns:
      tensor_analysis.tensor_range's answer, with what the request named.

  Raises:
      ValueError: The query is malformed.
      NotFound: The turn was not captured.
      Conflict: The capture index rejected the turn.
  """
  start, end = _integer(query, 'start'), _integer(query, 'end')
  if end <= start:
    raise ValueError('The column range [start, end) must not be empty.')
  bins = _integer(query, 'bins', default=128, minimum=1)
  if bins > tensor_analysis.MAX_BINS:
    raise ValueError(
        f'At most {tensor_analysis.MAX_BINS} chart bins are allowed.'
    )
  reduction = query.get('reduction', 'mean')
  if reduction not in tensor_analysis.REDUCTIONS:
    raise ValueError('reduction must be min, max or mean.')
  anchors = _anchors(query.get('anchors'))
  layers = _layers(query.get('layers'))
  echo, found = _tensor_stage(index, comparison, query)
  if found.places.reason is None:
    _known(anchors, found.places.anchors, 'sync point')
    _known(layers, found.places.layers, 'layer')
  if not found.start <= start < end <= found.end:
    raise ValueError(
        "The column range [start, end) must lie in the stage's columns "
        f'[{found.start}, {found.end}).'
    )
  answer = tensor_analysis.tensor_range(
      found, start, end, bins, anchors, reduction, layers
  )
  return {**echo, **answer}


def position_tensor(
    index: capture_index.CaptureIndex,
    comparison: Comparison,
    query: dict[str, Any],
) -> dict[str, Any]:
  """GET /api/position-tensor: one place at one column, element by element.

  Args:
      index: The chat's capture.
      comparison: The chat's comparison.
      query: `turn`, `stage`, optionally `alignment`; `position` (a column),
        `layer` and `anchor`.

  Returns:
      tensor_analysis.tensor's answer, with what the request named.

  Raises:
      ValueError: The query is malformed, or the stage has no such cell.
      NotFound: The turn was not captured.
      Conflict: The capture index rejected the turn, or the Session
        observes no tensor that can be compared (the reason's words).
  """
  position = _integer(query, 'position')
  layer = _integer(query, 'layer')
  anchor = _text(query, 'anchor')
  n, stage, kind = _tensor_request(index, query)
  view = comparison()
  why = tensor_analysis.layout(view.observed).reason
  if why is not None:
    raise errors.Conflict(why['text'])
  found = view.cell(n, stage, kind, position, layer, anchor)
  echo = {'turn': n, 'stage': stage, 'alignment': kind}
  return {**echo, **tensor_analysis.tensor(found)}
