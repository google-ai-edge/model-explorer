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

"""KV cache comparison: kv-analysis, kv-range and kv-head.

A context pairs the Ref and Target KV snapshot of one turn and moment
(`prefill_pre`, `prefill_post`, `terminal`). Each of its rows is one layer and
kind (key or value). Positions are conversation-global: the stored sequence
index is the position, and only `[0, processed)` holds processed tokens; the
rest of the buffer is never read. A layer whose file says it keeps a window
of positions (the capture index's `positions`) that does not start at 0 is
not compared yet, and says so.

Two sides that ran one model file store one representation: their stored
values are compared as they are (`values_mode: 'stored'`), accumulated in
float64, by the metrics every analysis shares (`metrics.compare`); the
dump holds no quantization scales, and none is needed. Two sides that ran
different model files (one model in PyTorch and in LiteRT-LM) are compared
when their KV caches have one architecture: every layer both store has the
same batch, KV heads and head dimension on both sides (else
`model_mismatch`). Each side is read through its own layout into
[batch, kv_head, position, head_dim]; a float side as stored, an integer
side dequantized, as scale * (stored - zero_point), by the scale its model
gives that KV cache tensor (the side's `model.kv_quantization`,
tap/model_facts; without one, `kv_scale_unknown`), so the values compared
(`values_mode: 'comparison'`) are what each runtime's attention reads. The
one rule of which pairs compare is `_pair`'s. With a position alignment
(the columns of a stage's rows, analysis/alignment.positions), a display
column maps to one position per side; without one, the column is the
position on both sides.

A row or head that cannot be compared has a `status` and says why as
reasons (metrics.reason), each of one side or of the pair; a status is a
code of `REASONS` too, which kv-head's `metric_status` gives alone.
"""

from collections.abc import Iterator
import dataclasses
import hashlib
import json
import math
import pathlib
import types
from typing import Any

from model_debugger.server.analysis import comparability
from model_debugger.server.analysis import metrics
from model_debugger.server.analysis import statistics
from model_debugger.server.capture import capture_index
from model_debugger.server.capture import tensor_file
from model_debugger.server.util import errors
import numpy as np

LAYOUT = ('batch', 'kv_head', 'sequence', 'head_dim')
KINDS = ('key', 'value')
# moment -> (phase, label)
MOMENTS = types.MappingProxyType({
    'prefill_pre': ('prefill', 'Before Prefill'),
    'prefill_post': ('prefill', 'After Prefill'),
    'terminal': ('decode', 'Generation end'),
})
MAX_BINS = 1024
MAX_SLICE_ELEMENTS = 262_144
MAX_CHANNELS = 16_384
# What kv-range charts of `metrics.compare`, per position and KV head.
CHART_METRICS = (
    'relative_l2',
    'max_abs',
    'cosine_distance',
    'cosine_similarity',
)
# What kv-head reports of one head's channels: every metric.
HEAD_METRICS = metrics.VECTOR_KEYS
HEAD_AGGREGATIONS = ('max', 'min', 'mean')
# How kv-range combines the KV heads when the request names none.
DEFAULT_HEAD = 'min'

# The statuses and reason codes of this module, and their words as the
# metric catalog lists them; a response's reason carries its own words,
# which may name more.
TENSOR_MISSING = 'tensor_missing'
VECTOR_MISSING = 'vector_missing'
SLIDING_WINDOW = 'sliding_window'
MODEL_MISMATCH = 'model_mismatch'
KV_SCALE_UNKNOWN = 'kv_scale_unknown'
# How an integer side's stored values become the values compared.
DEQUANTIZATION_FORMULA = 'scale × (stored − zero_point)'
# What this module's words call each side.
_SIDE_LABELS = {'ref': 'Ref', 'target': 'Target'}
REASONS = types.MappingProxyType({
    'unavailable': 'Unavailable for this selection.',
    TENSOR_MISSING: (
        'A stored tensor is required on both sides at this observation.'
    ),
    'layout_unavailable': (
        'The stored tensor does not have a batch, KV head, '
        'sequence and head dimension layout.'
    ),
    'logical_range_mismatch': (
        'The processed positions exceed the stored '
        'sequence axis, or differ between the sides without a position '
        'alignment.'
    ),
    'empty': 'This snapshot holds no processed positions.',
    SLIDING_WINDOW: (
        'The layer keeps a sliding window of positions that does '
        'not start at position 0; it is not compared yet.'
    ),
    MODEL_MISMATCH: (
        'Ref and Target ran different models whose KV caches '
        'differ in batch, KV heads or head dimension; they are not comparable.'
    ),
    KV_SCALE_UNKNOWN: (
        'Ref and Target ran different models, and a side '
        'stores integer KV values whose quantization scale its model does not '
        'give; they cannot be compared with the other model’s values.'
    ),
    'dtype_mismatch': 'Ref and Target store different dtypes.',
    'layout_mismatch': 'Ref and Target store different layouts.',
    'head_unavailable': 'The KV head is not in this layer’s stored tensor.',
    'position_unavailable': (
        'The position was not captured in this layer and kind.'
    ),
    'partial': (
        'Some metrics are undefined (zero norm, non-finite values or '
        'numeric overflow); finite counts are reported per metric.'
    ),
    'analysis_limit': f'A KV head may have at most {MAX_CHANNELS} channels.',
    'batch_selection_required': (
        'The cache holds several batches; batches are not combined.'
    ),
    VECTOR_MISSING: 'The vector is unavailable on one side.',
})


@dataclasses.dataclass(frozen=True)
class _Tensor:
  """One side's stored KV tensor of one layer and kind.

  `path` is None for a tensor judged from the capture index alone
  (snapshot_reason), which is never read.
  """

  file: str
  path: pathlib.Path | None
  dtype: str
  shape: tuple[int, ...]
  layout: tuple[str, ...] | None
  processed: int
  runtime: str | None
  positions: tuple[int, int] | None = None
  # (scale, zero point) its stored values are read with, for a side
  # compared with another model's; None reads them as stored.
  dequantization: tuple[float, int] | None = None

  @classmethod
  def of(
      cls,
      stored: dict[str, Any],
      processed: int,
      path: pathlib.Path | None = None,
      runtime: str | None = None,
  ) -> '_Tensor':
    """The tensor of a snapshot layer's `key` or `value` index entry."""
    return cls(
        file=stored['file'],
        path=path,
        dtype=stored['dtype'],
        shape=tuple(stored['shape']),
        layout=tuple(stored['layout']) if stored.get('layout') else None,
        processed=processed,
        runtime=runtime,
        positions=(
            tuple(stored['positions']) if stored.get('positions') else None
        ),
    )

  def integer(self) -> bool:
    """Whether its stored values are integers (quantized, if a KV cache)."""
    return self.dtype[:1] in ('I', 'U')

  def geometry(self) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Returns (raw axis per canonical axis, canonical shape), or None."""
    if (
        self.layout is None
        or len(self.layout) != 4
        or set(self.layout) != set(LAYOUT)
        or len(self.shape) != 4
        or any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0
            for size in self.shape
        )
    ):
      return None
    order = tuple(self.layout.index(axis) for axis in LAYOUT)
    return order, tuple(self.shape[axis] for axis in order)

  def problem(self) -> tuple[str, str] | None:
    """Why this tensor cannot be read, or None when it can."""
    geometry = self.geometry()
    if geometry is None:
      return (
          'layout_unavailable',
          (
              'The stored tensor does not have a batch, KV head, sequence '
              'and head dimension layout.'
          ),
      )
    if self.positions is not None and self.positions[0] != 0:
      first, last = self.positions
      return (
          SLIDING_WINDOW,
          (
              f'This layer keeps positions {first}-{last} only, a sliding '
              'window; a cache that does not start at position 0 is not '
              'compared yet.'
          ),
      )
    if not 0 <= self.processed <= geometry[1][2]:
      return (
          'logical_range_mismatch',
          'The processed positions exceed the stored sequence axis.',
      )
    if self.processed == 0:
      return 'empty', 'This snapshot holds no processed positions.'
    return None

  def canonical(self) -> tuple[int, ...]:
    """(batch, kv_head, processed, head_dim) of a readable tensor."""
    batch, heads, _, channels = self.geometry()[1]
    return batch, heads, self.processed, channels

  def structure(self, side: str) -> dict[str, Any]:
    """The UI's KvTokenStructure: one token's geometry on `side`."""
    result = {
        'status': 'unavailable',
        'shape': None,
        'dtype': self.dtype,
        'head_count': None,
        'channel_count': None,
        'batch_count': None,
        'position_start': None,
        'position_count': None,
        'reasons': [],
    }
    problem = self.problem()
    if problem is not None and problem[0] != 'empty':
      result.update(
          status=problem[0],
          reasons=[metrics.reason(problem[0], problem[1], side)],
      )
      return result
    batch, heads, positions, channels = self.canonical()
    result.update(
        status='empty' if problem else 'ok',
        shape=[heads, channels],
        head_count=heads,
        channel_count=channels,
        batch_count=batch,
        position_start=0,
        position_count=positions,
    )
    return result

  def source(self) -> dict[str, Any]:
    """The UI's KvHeadSource: where this side's values come from."""
    geometry = self.geometry()
    dequantization = None
    if self.dequantization is not None:
      scale, zero_point = self.dequantization
      dequantization = {
          'scale': scale,
          'zero_point': zero_point,
          'formula': DEQUANTIZATION_FORMULA,
      }
    return {
        'runtime': self.runtime,
        'source_dtype': self.dtype,
        'source_shape': list(self.shape),
        'comparison_dtype': (
            self.dtype if dequantization is None else metrics.CALCULATION_DTYPE
        ),
        'comparison_shape': list(self.canonical()) if geometry else None,
        'layout': list(self.layout) if self.layout is not None else None,
        'axis_order': list(geometry[0]) if geometry else None,
        'dequantization': dequantization,
        'values_mode': 'stored' if dequantization is None else 'comparison',
        'resource_id': self.file,
    }

  def read(
      self,
      positions: np.ndarray,
      head: int | None = None,
      batch: int | None = None,
  ) -> np.ndarray:
    """Reads increasing positions as [batch, kv_head, position, head_dim].

    Args:
        positions: Increasing positions in [0, processed).
        head: One KV head, or None for every head.
        batch: One batch, or None for every batch.

    Returns:
        The stored values in canonical axis order, in the stored dtype;
        dequantized, in float64, with `dequantization`.

    Raises:
        ValueError: One position holds more than the slice element limit, or
          the file disagrees with the index.
    """
    order, canonical = self.geometry()
    per_position = (
        (canonical[0] if batch is None else 1)
        * (canonical[1] if head is None else 1)
        * canonical[3]
    )
    if per_position > MAX_SLICE_ELEMENTS:
      raise ValueError(
          'One KV position exceeds the limit of '
          f'{MAX_SLICE_ELEMENTS} elements per read.'
      )
    chunk = MAX_SLICE_ELEMENTS // per_position
    first, last = int(positions[0]), int(positions[-1])
    if last - first < chunk:
      # One read of the whole span, then the wanted rows of it.
      value = self._read_span(order, canonical, first, last + 1, head, batch)
      if last - first + 1 != len(positions):
        value = value[:, :, positions - first, :]
      return self._dequantized(value)
    # A sparse alignment: read each run of consecutive positions.
    breaks = np.flatnonzero(np.diff(positions) != 1) + 1
    parts = []
    for run in np.split(positions, breaks):
      for low in range(int(run[0]), int(run[-1]) + 1, chunk):
        high = min(int(run[-1]) + 1, low + chunk)
        parts.append(self._read_span(order, canonical, low, high, head, batch))
    return self._dequantized(np.concatenate(parts, axis=2))

  def _dequantized(self, value: np.ndarray) -> np.ndarray:
    """Stored values as compared: as stored, or dequantized (float64)."""
    if self.dequantization is None:
      return value
    scale, zero_point = self.dequantization
    return (value.astype(np.float64) - zero_point) * scale

  def _read_span(self, order, canonical, start, end, head, batch):
    """Returns positions [start, end) of the tensor in canonical order."""
    index = [slice(None)] * 4
    for axis, value in ((0, batch), (1, head)):
      if value is not None:
        index[order[axis]] = slice(value, value + 1)
    index[order[2]] = slice(start, end)
    value = tensor_file.read_slice(self.path, tuple(index)).transpose(order)
    expected = (
        canonical[0] if batch is None else 1,
        canonical[1] if head is None else 1,
        end - start,
        canonical[3],
    )
    if value.shape != expected:
      raise ValueError(f'{self.file} does not have the shape the index lists.')
    return value


@dataclasses.dataclass(frozen=True)
class _Context:
  """The Ref and Target snapshot of one turn and moment."""

  id: str
  turn: int
  moment: str
  snapshots: dict[str, dict[str, Any]]  # side -> the index's kv entry
  rows: dict[
      tuple[int, str], dict[str, _Tensor]
  ]  # (layer, kind) -> {side: _Tensor}
  # Whether both sides ran one model file; else _architecture_reason.
  same_model: bool = True
  architecture: dict[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class _Pair:
  """A row's tensors and whether they can be compared."""

  tensors: dict[str, _Tensor]  # side -> _Tensor, for the sides that stored one
  status: str
  reasons: list[
      dict[str, Any]
  ]  # metrics.reason of each, none for a comparable pair
  same_model: bool = True

  def readable(self) -> dict[str, _Tensor]:
    """The sides whose tensor can be read."""
    return {
        side: tensor
        for side, tensor in self.tensors.items()
        if tensor.problem() is None
    }

  def basis(self) -> str | None:
    """How the values are compared, for a comparable pair."""
    if self.status != 'ok':
      return None
    if not self.same_model:
      sides = []
      for side in capture_index.SIDES:
        tensor = self.tensors[side]
        name = _SIDE_LABELS[side]
        if tensor.dequantization is None:
          sides.append(f'{name}: stored {tensor.dtype} values')
        else:
          scale, zero_point = tensor.dequantization
          sides.append(
              f'{name}: stored {tensor.dtype} values dequantized '
              f'as {scale:.6g} × (stored − {zero_point}), the '
              'scale its model gives this cache'
          )
      return (
          '; '.join(sides)
          + '. Different models of one KV '
          'architecture, each read through its own layout and compared '
          'in float64.'
      )
    dtype = self.tensors['ref'].dtype
    basis = f'Stored {dtype} values, compared as stored in float64'
    if dtype[:1] in ('I', 'U'):
      return basis + '; the capture records no quantization scales.'
    return basis + '.'


@dataclasses.dataclass(frozen=True)
class Alignment:
  """Display columns [start, start + len) and each side's position or -1."""

  start: int
  ref: np.ndarray
  target: np.ndarray

  @classmethod
  def of(cls, columns: dict[str, Any]) -> 'Alignment':
    """The alignment of `{start, pairs}` (analysis/alignment.positions).

    Raises:
        ValueError: A side's positions do not increase.
    """
    pairs = columns['pairs']
    positions = np.full((2, len(pairs)), -1, dtype=np.int64)
    for i, pair in enumerate(pairs):
      for side, position in enumerate(pair):
        if position is not None:
          positions[side, i] = position
    for side in positions:
      named = side[side >= 0]
      if np.any(np.diff(named) <= 0):
        raise ValueError('Aligned positions must increase on each side.')
    return cls(columns['start'], positions[0], positions[1])

  def end(self) -> int:
    """The column after the last."""
    return self.start + len(self.ref)

  def positions(self, column: int) -> dict[str, int | None]:
    """Each side's position of a column; None where it has none."""
    index = column - self.start
    if not 0 <= index < len(self.ref):
      return {'ref': None, 'target': None}
    return {
        side: int(values[index]) if values[index] >= 0 else None
        for side, values in (('ref', self.ref), ('target', self.target))
    }


def kv_analysis(index: capture_index.CaptureIndex, n: int) -> dict[str, Any]:
  """Returns the KV contexts of a turn (kv-analysis?mode=metadata).

  Args:
      index: The chat's capture.
      n: The turn.

  Returns:
      `{contexts}`, each context with its layer rows; no tensor is read.
  """
  return {
      'contexts': [
          _context_view(index, context) for context in _contexts(index, n)
      ]
  }


def snapshot_reason(
    snapshot: dict[str, Any], side: str
) -> dict[str, Any] | None:
  """Why kv-range and kv-head read no tensor of a side's snapshot, or None.

  By the rule those routes read by (_Tensor.problem), from the snapshot's
  index entry alone: no tensor is read. None when a tensor of some layer and
  kind can be read; else the first tensor's problem, or TENSOR_MISSING when
  the snapshot stores none.

  Args:
      snapshot: A `kv` entry of a side's turn in the capture index.
      side: 'ref' or 'target', the side the reason is of.

  Returns:
      None when some tensor can be read; else the reason (metrics.reason).
  """
  problems = []
  for layer in snapshot['layers']:
    problems += [
        _Tensor.of(layer[kind], snapshot['processed']).problem()
        for kind in KINDS
        if layer.get(kind) is not None
    ]
  if not problems:
    return metrics.reason(
        TENSOR_MISSING, 'This snapshot stores no KV tensor.', side
    )
  if None in problems:
    return None
  return metrics.reason(*problems[0], side)


def context_id(n: int, moment: str) -> str:
  """Returns the id of turn `n`'s KV context of `moment` (MOMENTS)."""
  digest = hashlib.sha256(json.dumps([n, moment]).encode()).hexdigest()
  return 'snapshot-' + digest[:24]


def kv_range(
    index: capture_index.CaptureIndex,
    n: int,
    requested_id: str,
    start: int,
    end: int,
    bins: int,
    kind: str,
    head: int | str,
    heads: list[int] | None,
    alignment: Alignment | None,
    layers: list[int] | None = None,
) -> dict[str, Any]:
  """Returns binned per-position metrics of the layers of a kind.

  Args:
      index: The chat's capture.
      n: The turn.
      requested_id: The KV context.
      start: The first column of the range.
      end: The column after it; more than `start`.
      bins: How many bins, 1..MAX_BINS.
      kind: 'key' or 'value'.
      head: A KV head, or how KV heads combine (HEAD_AGGREGATIONS).
      heads: The KV heads `head` combines, increasing; None for all.
      alignment: The columns' positions per side; None: a column is the position
        on both sides.
      layers: The layers wanted; None for every layer.

  Returns:
      `{context_id, start, end, kind, head, heads, layers, bins, rows,
      metadata}`: `metadata` is what the rows compared (`_metadata`).
  """
  context = _context(index, n, requested_id)
  edges = statistics.edges(start, end, bins)
  bounds = [
      {'start': int(low), 'end': int(high)}
      for low, high in zip(edges[:-1], edges[1:])
  ]
  keys = [
      (layer, row_kind)
      for layer, row_kind in sorted(context.rows, key=_row_order)
      if row_kind == kind and (layers is None or layer in layers)
  ]
  rows = [
      _range_row(
          context,
          layer,
          kind,
          start,
          end,
          edges,
          bounds,
          head,
          heads,
          alignment,
      )
      for layer, _ in keys
  ]
  if isinstance(head, int):
    selected = [head]
  else:
    selected = heads if heads is not None else range(_head_domain(context))
  return {
      'context_id': requested_id,
      'start': start,
      'end': end,
      'kind': kind,
      'head': head,
      'heads': heads,
      'layers': layers,
      'bins': bounds,
      'rows': rows,
      'metadata': _metadata(context, keys, start, end, selected, alignment),
  }


def kv_head(
    index: capture_index.CaptureIndex,
    n: int,
    requested_id: str,
    layer: int,
    kind: str | None,
    position: int,
    batch: int | None,
    alignment: Alignment | None,
) -> dict[str, Any]:
  """Returns every KV head's channels at one column, both sides.

  Each side's column is read once, every head at a time.

  Args:
      index: The chat's capture.
      n: The turn.
      requested_id: The KV context.
      layer: The layer.
      kind: 'key' or 'value'.
      position: The column.
      batch: The batch; None for the only one.
      alignment: The columns' positions per side; None: a column is the position
        on both sides.

  Returns:
      `{selection, heads}`: `heads` holds the UI's KvHeadEvidence of each
      KV head the layer's stored tensors have (the most either side
      stores), in order, each with the `statistics` (analysis/statistics)
      of its channels' `ref`, `target`, `delta` and `abs_delta`.

  Raises:
      ValueError: The batch is outside the stored tensor.
  """
  context = _context(index, n, requested_id)
  positions = (
      {'ref': position, 'target': position}
      if alignment is None
      else alignment.positions(position)
  )
  selection = {
      'turn': n,
      'context_id': requested_id,
      'layer': layer,
      'kind': kind,
      'position': position,
      'batch': batch,
  }
  if alignment is not None:
    selection['positions'] = positions
  pair = _pair(context, layer, kind, alignment is not None)
  count = max(
      (
          tensor.geometry()[1][1]
          for tensor in pair.tensors.values()
          if tensor.geometry() is not None
      ),
      default=0,
  )
  common = {
      'status': 'unavailable',
      'batch_count': None,
      'channel_count': None,
      'shape': None,
      'sources': {side: None for side in capture_index.SIDES},
      'comparison_basis': pair.basis(),
      'calculation_dtype': metrics.CALCULATION_DTYPE,
      'delta_definition': 'Target - Reference',
  }
  for side, tensor in pair.tensors.items():
    common['sources'][side] = tensor.source()

  def every_head(status: str, reasons: list[dict[str, Any]]) -> dict[str, Any]:
    """The answer when no head can be read, for one reason."""
    return {
        'selection': selection,
        'heads': [
            _head_failure(_head_result(common, head), status, reasons)
            for head in range(count)
        ],
    }

  readable = pair.readable()
  if not readable:
    # A pair with no readable side is not comparable, and says why.
    return every_head(pair.status, pair.reasons)
  shapes = {side: tensor.canonical() for side, tensor in readable.items()}
  for side, shape in shapes.items():
    common['sources'][side]['comparison_shape'] = [shape[3]]
  common['batch_count'] = max(shape[0] for shape in shapes.values())
  common['channel_count'] = max(shape[3] for shape in shapes.values())
  common['shape'] = [common['channel_count']]
  if common['channel_count'] > MAX_CHANNELS:
    return every_head('analysis_limit', [_reason('analysis_limit')])
  if batch is None and common['batch_count'] > 1:
    return every_head(
        'batch_selection_required', [_reason('batch_selection_required')]
    )
  batch = 0 if batch is None else batch
  selection['batch'] = batch
  if not any(batch < shape[0] for shape in shapes.values()):
    raise ValueError('The batch is outside the stored tensor.')
  # Each side's column, every head: [kv_head, head_dim].
  stored, unread = {}, {}
  for side, tensor in readable.items():
    at = positions[side]
    shape = shapes[side]
    if at is None or not (0 <= at < shape[2] and batch < shape[0]):
      continue
    try:
      stored[side] = tensor.read(np.array([at]), batch=batch)[0, :, 0, :]
    except (OSError, ValueError) as error:
      unread[side] = metrics.reason(metrics.UNREADABLE, str(error), side)
  return {
      'selection': selection,
      'heads': [
          _head(common, pair, shapes, positions, position, stored, unread, head)
          for head in range(count)
      ],
  }


def _head_result(common: dict[str, Any], head: int) -> dict[str, Any]:
  """A head's KvHeadEvidence before it is compared."""
  return {
      **common,
      'head': head,
      'sources': dict(common['sources']),
      'metrics': dict.fromkeys(HEAD_METRICS),
      'metric_status': dict.fromkeys(HEAD_METRICS, 'unavailable'),
      'channels': [],
      'largest_channel': None,
      'statistics': {key: statistics.empty() for key in _CHANNEL_VALUES},
  }


# What each head's statistics are of: its channels' values.
_CHANNEL_VALUES = ('ref', 'target', 'delta', 'abs_delta')


def _head(
    common: dict[str, Any],
    pair: _Pair,
    shapes: dict[str, tuple[int, ...]],
    positions: dict[str, int | None],
    position: int,
    stored: dict[str, np.ndarray],
    unread: dict[str, dict[str, Any]],
    head: int,
) -> dict[str, Any]:
  """One KV head's KvHeadEvidence, from each side's column (kv_head)."""
  result = _head_result(common, head)
  status = pair.status
  vectors, failures = {}, []
  for side, shape in shapes.items():
    at = positions[side]
    if at is None:
      continue
    if side in unread:
      failures.append(('unavailable', unread[side]))
      continue
    if side not in stored or head >= shape[1]:
      failures.append((
          'position_unavailable',
          metrics.reason(
              'position_unavailable',
              f'position {at}, head {head} was not captured.',
              side,
          ),
      ))
      continue
    vectors[side] = stored[side][head]
  if not vectors:
    ranges = ', '.join(
        f'{side}: [0, {shape[2]})' for side, shape in shapes.items()
    )
    return _head_failure(
        result,
        failures[0][0] if failures else 'position_unavailable',
        [
            metrics.reason(
                'position_unavailable',
                f'Cache position {position} was not captured in this '
                f'layer and kind. Captured ranges: {ranges}.',
            )
        ],
    )
  comparable = (
      status == 'ok'
      and len(vectors) == 2
      and vectors['ref'].shape == vectors['target'].shape
  )
  if comparable:
    measured = metrics.compare(vectors['ref'], vectors['target'])
    codes = {key: measured.status[key].item() for key in HEAD_METRICS}
    if metrics.NON_FINITE in codes.values():
      _head_failure(
          result,
          'partial',
          [
              metrics.reason(
                  metrics.NON_FINITE,
                  f'The head metrics are unavailable: {metrics.NON_FINITE}',
              )
          ],
      )
      result['metric_status'] = dict.fromkeys(HEAD_METRICS, metrics.NON_FINITE)
    else:
      for key, code in codes.items():
        value = measured.values[key].item()
        result['metrics'][key] = value if code == metrics.OK else None
        result['metric_status'][key] = code
      result['status'] = (
          'ok'
          if all(code == metrics.OK for code in codes.values())
          else 'partial'
      )
  else:
    why = pair.reasons + [failure[1] for failure in failures]
    _head_failure(
        result,
        status if status != 'ok' else 'unavailable',
        why or [_reason(VECTOR_MISSING)],
    )
  for channel in range(max(len(vector) for vector in vectors.values())):
    raw = {
        side: vector[channel]
        for side, vector in vectors.items()
        if channel < len(vector)
    }
    delta, channel_status = None, result['status']
    if comparable:
      channel_status = 'non_finite'
      if all(np.isfinite(value) for value in raw.values()):
        with np.errstate(over='ignore', invalid='ignore'):
          candidate = float(np.float64(raw['target']) - np.float64(raw['ref']))
        if math.isfinite(candidate):
          delta, channel_status = candidate, 'ok'
    result['channels'].append({
        'channel': channel,
        'ref': _scalar(raw['ref']) if 'ref' in raw else None,
        'target': _scalar(raw['target']) if 'target' in raw else None,
        'delta': delta,
        'abs_delta': abs(delta) if delta is not None else None,
        'status': channel_status,
    })
  finite = [row for row in result['channels'] if row['abs_delta'] is not None]
  if finite:
    result['largest_channel'] = max(finite, key=lambda row: row['abs_delta'])[
        'channel'
    ]
  result['statistics'] = _channel_statistics(result['channels'])
  return result


def _channel_statistics(channels: list[dict[str, Any]]) -> dict[str, Any]:
  """The statistics of a head's channel values (analysis/statistics).

  A value the UI gets as a number counts; one it gets as a string
  (non-finite, or an integer past 2^53) does not.

  Args:
      channels: The head's channel rows, each with `channel` and each of
        _CHANNEL_VALUES.

  Returns:
      Each of _CHANNEL_VALUES -> its statistics.summary over the channels.
  """
  positions = np.array([row['channel'] for row in channels])
  return {
      key: statistics.summary(
          np.array(
              [
                  row[key] if type(row[key]) in (int, float) else np.nan
                  for row in channels
              ],
              dtype=np.float64,
          ),
          positions,
      )
      for key in _CHANNEL_VALUES
  }


# Contexts and rows.


def _contexts(index: capture_index.CaptureIndex, n: int) -> list[_Context]:
  """Returns the snapshot contexts of turn `n`, one per moment captured."""
  turn = index.turn(n)
  sides = turn['sides']
  same = _same_model(index)
  contexts = []
  for moment in MOMENTS:
    snapshots = {}
    for side in capture_index.SIDES:
      entry = sides.get(side)
      snapshot = next(
          (
              row
              for row in (entry or {}).get('kv', [])
              if row['moment'] == moment
          ),
          None,
      )
      if snapshot is not None:
        snapshots[side] = snapshot
    if not snapshots:
      continue
    rows = {}
    for side, snapshot in snapshots.items():
      runtime = index.sides[side]['runtime']['name']
      scales = index.sides[side]['model'].get('kv_quantization') or {}
      for layer in snapshot['layers']:
        for kind in KINDS:
          stored = layer.get(kind)
          if stored is None:
            continue
          tensor = _Tensor.of(
              stored,
              snapshot['processed'],
              index.path(side, n, stored['file']),
              runtime,
          )
          scale = scales.get(_kv_input(layer['layer'], kind))
          if not same and tensor.integer() and scale is not None:
            tensor = dataclasses.replace(
                tensor,
                dequantization=(
                    float(scale['scale']),
                    int(scale['zero_point']),
                ),
            )
          rows.setdefault((layer['layer'], kind), {})[side] = tensor
    contexts.append(
        _Context(
            id=context_id(n, moment),
            turn=n,
            moment=moment,
            snapshots=snapshots,
            rows=rows,
            same_model=same,
            architecture=None if same else _architecture_reason(rows),
        )
    )
  return contexts


def _same_model(index: capture_index.CaptureIndex) -> bool:
  """Whether both sides ran one model file (comparability.same_file())."""
  return comparability.same_file(*(
      comparability.Model(index.sides[side]['model']['sha256'])
      for side in capture_index.SIDES
  ))


def _kv_input(layer: int, kind: str) -> str:
  """The name of a layer's KV cache input (tap/model_facts)."""
  return f'kv_cache_{kind[0]}_{layer}'


def _architecture_reason(
    rows: dict[tuple[int, str], dict[str, _Tensor]],
) -> dict[str, Any] | None:
  """Why two models' KV caches are of different architectures, or None.

  They are of one when every layer and kind both sides store has the same
  batch, KV heads and head dimension on both; a layer one side stores alone
  is that row's matter (_pair).

  Args:
      rows: (layer, kind) -> {side: _Tensor}, a context's rows.

  Returns:
      A MODEL_MISMATCH reason of the first row whose shapes differ, or None.
  """
  for (layer, kind), tensors in sorted(rows.items(), key=lambda row: row[0]):
    if len(tensors) != 2:
      continue
    shapes = {}
    for side, tensor in tensors.items():
      geometry = tensor.geometry()
      if geometry is None:
        break
      batch, heads, _, channels = geometry[1]
      shapes[side] = (batch, heads, channels)
    if len(shapes) == 2 and shapes['ref'] != shapes['target']:
      ref, target = (
          '{}×{}×{}'.format(*shapes[side]) for side in capture_index.SIDES
      )
      return metrics.reason(
          MODEL_MISMATCH,
          'Ref and Target ran different models whose KV caches differ: '
          f'layer {layer} {kind} holds batch × KV heads × head dimension '
          f'{ref} on Ref and {target} on Target.',
      )
  return None


def _context(
    index: capture_index.CaptureIndex, n: int, requested_id: str
) -> _Context:
  """Returns turn `n`'s KV context of `requested_id`, or raises NotFoundError."""
  for context in _contexts(index, n):
    if context.id == requested_id:
      return context
  raise errors.NotFoundError(f'Turn {n} has no KV context {requested_id}.')


def _row_order(key: tuple[int, str]) -> tuple[int, int]:
  layer, kind = key
  return layer, KINDS.index(kind)


def _pair(
    context: _Context,
    layer: int,
    kind: str,
    aligned: bool,
) -> _Pair:
  """Whether a row compares, and why not: the one rule of pairing.

  The status gates, in order; alignment lifts the range and length ones.
  One model file on both sides stores one representation: the dtypes and
  layouts must be the same, and the values are compared as stored. Two
  model files must have one KV architecture (_architecture_reason); each
  side is read through its own layout, an integer one dequantized by its
  model's scale (`_Tensor.dequantization`), so dtypes and layouts may
  differ.

  Args:
      context: The KV context.
      layer: The layer.
      kind: 'key' or 'value'.
      aligned: Whether a position alignment maps the columns.

  Returns:
      The row's _Pair: its tensors, status and reasons.
  """
  stored = context.rows.get((layer, kind), {})
  same = context.same_model

  def pair(status: str, reasons: list[dict[str, Any]]) -> _Pair:
    return _Pair(stored, status, reasons, same)

  if len(stored) != 2:
    (alone,) = stored
    other = 'target' if alone == 'ref' else 'ref'
    # Of two models, the other side's snapshot of this moment lacks the
    # layer: its runtime keeps none (else it did not capture the moment).
    if same or other not in context.snapshots:
      return pair('unavailable', [_reason(TENSOR_MISSING)])
    return pair(
        'unavailable',
        [
            metrics.reason(
                TENSOR_MISSING,
                f'Only {_SIDE_LABELS[alone]} stores layer '
                f'{layer}’s {kind} cache; '
                f'{_SIDE_LABELS[other]}’s runtime keeps none '
                'of its own for this layer (as for a layer that reuses '
                'another layer’s KV).',
                other,
            )
        ],
    )
  for side in capture_index.SIDES:
    problem = stored[side].problem()
    if problem is not None:
      return pair(problem[0], [metrics.reason(*problem, side)])
  ref, target = stored['ref'], stored['target']
  if not same:
    if context.architecture is not None:
      return pair(MODEL_MISMATCH, [context.architecture])
    for side, tensor in stored.items():
      if tensor.integer() and tensor.dequantization is None:
        return pair(
            KV_SCALE_UNKNOWN,
            [
                metrics.reason(
                    KV_SCALE_UNKNOWN,
                    f'{_SIDE_LABELS[side]} stores '
                    f'{tensor.dtype} values, and the capture records '
                    'no quantization scale of '
                    f'{_kv_input(layer, kind)} for its model; they '
                    'cannot be compared with the other model’s '
                    'values.',
                    side,
                )
            ],
        )
  elif ref.dtype != target.dtype:
    return pair(
        'dtype_mismatch',
        [
            metrics.reason(
                'dtype_mismatch',
                f'Ref stores {ref.dtype} and Target stores {target.dtype}.',
            )
        ],
    )
  elif ref.layout != target.layout:
    return pair('layout_mismatch', [_reason('layout_mismatch')])
  ref_shape, target_shape = ref.canonical(), target.canonical()
  if any(ref_shape[axis] != target_shape[axis] for axis in (0, 1, 3)):
    return pair(
        metrics.SHAPE_MISMATCH,
        [
            metrics.reason(
                metrics.SHAPE_MISMATCH,
                'The batch, KV head or head dimension sizes differ.',
            )
        ],
    )
  if not aligned and ref.processed != target.processed:
    return pair(
        'logical_range_mismatch',
        [
            metrics.reason(
                'logical_range_mismatch',
                f'Ref processed {ref.processed} positions and Target '
                f'{target.processed}; compare them through a position '
                'alignment.',
            )
        ],
    )
  return pair('ok', [])


def _context_view(
    index: capture_index.CaptureIndex, context: _Context
) -> dict[str, Any]:
  """Returns the JSON view of `context`: its label, step and layers."""
  phase, moment_label = MOMENTS[context.moment]
  steps = {snapshot['step'] for snapshot in context.snapshots.values()}
  snapshots = context.snapshots
  processed = {snapshot['processed'] for snapshot in snapshots.values()}
  runtimes = {index.sides[side]['runtime']['name'] for side in snapshots}
  label = f'Turn {context.turn} · {moment_label}'
  view = {
      'id': context.id,
      'label': label,
      'turn': context.turn,
      'phase': phase,
      'step': steps.pop() if len(steps) == 1 else None,
      'moment': context.moment,
      'source': 'snapshot',
      'runtime': runtimes.pop() if len(runtimes) == 1 else None,
      'layers': [],
  }
  if len(processed) == 1:
    view['processed_token_count'] = processed.pop()
    view['label'] += f' · {view["processed_token_count"]} processed tokens'
  for layer, kind in sorted(context.rows, key=_row_order):
    pair = _pair(context, layer, kind, False)
    row = {
        'layer': layer,
        'kind': kind,
        'status': pair.status,
        'head_count': None,
        'position_start': None,
        'position_count': None,
        'token_structure': {side: None for side in capture_index.SIDES},
        'reasons': pair.reasons,
    }
    for side, tensor in pair.tensors.items():
      row['token_structure'][side] = tensor.structure(side)
      row['reference' if side == 'ref' else 'target'] = tensor.file
    shaped = [
        tensor
        for tensor in pair.tensors.values()
        if tensor.problem() is None or tensor.problem()[0] == 'empty'
    ]
    if shaped:
      shape = shaped[0].canonical()
      row.update(
          shape=list(shape),
          canonical_layout=list(LAYOUT),
          head_count=shape[1],
          position_start=0,
          position_count=shape[2],
      )
    basis = pair.basis()
    if basis:
      row['comparison_basis'] = basis
    view['layers'].append(row)
  return view


def _head_domain(context: _Context) -> int:
  """How many KV heads the context's layers have, the most any has.

  A layer's count is that of its first side whose tensor has a shape.

  Args:
      context: The KV context.

  Returns:
      The largest KV head count of a layer; 0 when no layer has a shape.
  """
  counts = [0]
  for tensors in context.rows.values():
    shaped = [
        tensor
        for tensor in tensors.values()
        if tensor.problem() is None or tensor.problem()[0] == 'empty'
    ]
    if shaped:
      counts.append(shaped[0].canonical()[1])
  return max(counts)


# The axes of what kv-range compares, and of each metric it gives.
INPUT_DIMENSIONS = ('Position', 'Layer', 'Head', 'Head dimension')
OUTPUT_DIMENSIONS = ('Position', 'Layer')


def _shaped(tensor: _Tensor | None) -> tuple[int, ...] | None:
  """(batch, kv_head, processed, head_dim) of a tensor with a shape."""
  if tensor is None:
    return None
  problem = tensor.problem()
  if problem is not None and problem[0] != 'empty':
    return None
  return tensor.canonical()


def _metadata(
    context: _Context,
    keys: list[tuple[int, str]],
    start: int,
    end: int,
    heads,
    alignment: Alignment | None,
) -> dict[str, Any]:
  """What kv-range's rows compared: each side's input, and the output.

  Each side's input is, per layer, the positions of the columns
  [start, end) the side stored (each once), the selected KV heads its
  tensor has and its head dimension: `[positions, heads, head_dim]`; the
  output is one value per column and layer, where both sides have a shape
  of one head dimension and batch count. Across the layers, an axis is a
  size where every layer has the same, a list of the sizes where they
  differ, and None where a layer has no shape; a dtype likewise, a list
  where the layers' differ.

  Args:
      context: The KV context.
      keys: The rows, (layer, kind), in order.
      start: The first column.
      end: The column after the last.
      heads: The selected KV heads.
      alignment: The columns' positions per side; None: a column is the position
        on both sides.

  Returns:
      `{ref, target, output}`: each side's input and the output, each with
      `dimensions`, `shape`, `dtype` and per-layer `layers`.
  """
  heads = sorted(set(heads))
  result = {}
  for side_index, side in enumerate(capture_index.SIDES):
    layers = []
    for key in keys:
      tensor = context.rows[key].get(side)
      shape = _shaped(tensor)
      if shape is not None:
        if alignment is None:
          count = max(0, min(shape[2], end) - max(0, start))
        else:
          stored = (alignment.ref, alignment.target)[side_index]
          chosen = stored[
              max(0, start - alignment.start) : max(0, end - alignment.start)
          ]
          count = int(
              np.unique(chosen[(chosen >= 0) & (chosen < shape[2])]).size
          )
        shape = [
            count,
            sum(1 for head in heads if 0 <= head < shape[1]),
            shape[3],
        ]
      layers.append({
          'layer': key[0],
          'shape': shape,
          'dtype': tensor.dtype if tensor is not None else None,
      })
    result[side] = {
        'dimensions': list(INPUT_DIMENSIONS),
        'shape': _across(layers, 3),
        'dtype': _uniform([layer['dtype'] for layer in layers]),
        'layers': layers,
    }
  outputs = []
  for key in keys:
    tensors = context.rows[key]
    ref, target = (_shaped(tensors.get(side)) for side in capture_index.SIDES)
    comparable = (
        ref is not None
        and target is not None
        and ref[3] == target[3]
        and ref[0] == target[0]
    )
    outputs.append(
        {'layer': key[0], 'shape': [end - start] if comparable else None}
    )
  shaped = any(layer['shape'] is not None for layer in outputs)
  result['output'] = {
      'dimensions': list(OUTPUT_DIMENSIONS),
      'shape': _across(outputs, 1),
      'dtype': metrics.CALCULATION_DTYPE if shaped else None,
      'layers': outputs,
  }
  return result


def _across(layers: list[dict[str, Any]], axes: int) -> list[Any] | None:
  """The layers' shapes as one: each axis, then the layer count second.

  None when no layer has a shape; an axis is None where a layer has none.

  Args:
      layers: The layers, each with a `shape` (a list, or None).
      axes: How many axes each shape has.

  Returns:
      [first axis, layer count, other axes...], or None.
  """
  shapes = [layer['shape'] for layer in layers]
  if all(shape is None for shape in shapes):
    return None
  sizes = [
      None
      if any(shape is None for shape in shapes)
      else _uniform([shape[axis] for shape in shapes])
      for axis in range(axes)
  ]
  return [sizes[0], len(layers), *sizes[1:]]


def _uniform(values: list[Any]) -> Any:
  """The one value all have; a list of the distinct ones; None for none."""
  distinct = list(dict.fromkeys(values))
  if not distinct:
    return None
  return distinct[0] if len(distinct) == 1 else distinct


# Reading and measuring.


def _blocks(
    pair: _Pair, start: int, end: int, alignment: Alignment | None
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """Yields (columns, ref values, target values) for comparable columns.

  A column is compared only where both sides have a processed position. Each
  block reads at most MAX_SLICE_ELEMENTS elements per side and read.

  Args:
      pair: A comparable pair.
      start: The first column.
      end: The column after the last.
      alignment: The columns' positions per side; None: a column is the position
        on both sides.

  Yields:
      (columns, ref values, target values): the block's columns, and each
      side's values there as [batch, kv_head, position, head_dim].

  Raises:
      ValueError: One position holds more than the slice element limit.
  """
  ref, target = pair.tensors['ref'], pair.tensors['target']
  if alignment is None:
    first, stop = start, min(end, ref.processed)
  else:
    first, stop = max(start, alignment.start), min(end, alignment.end())
  batch, heads, _, channels = ref.canonical()
  per_position = batch * heads * channels
  if per_position > MAX_SLICE_ELEMENTS:
    raise ValueError(
        'One KV position exceeds the limit of '
        f'{MAX_SLICE_ELEMENTS} elements per read.'
    )
  chunk = MAX_SLICE_ELEMENTS // per_position
  for left in range(first, stop, chunk):
    columns = np.arange(left, min(stop, left + chunk))
    if alignment is None:
      ref_positions = target_positions = columns
    else:
      ref_positions = alignment.ref[columns - alignment.start]
      target_positions = alignment.target[columns - alignment.start]
    keep = (
        (ref_positions >= 0)
        & (ref_positions < ref.processed)
        & (target_positions >= 0)
        & (target_positions < target.processed)
    )
    if not keep.any():
      continue
    yield (
        columns[keep],
        ref.read(ref_positions[keep]),
        target.read(target_positions[keep]),
    )


def _measure(
    reference: np.ndarray, target: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
  """Per-position metrics of [batch, kv_head, position, head_dim] values.

  Compares over batch and channel (head_dim): one value per KV head and
  position, of CHART_METRICS only.

  Args:
      reference: Ref values.
      target: Target values, same shape.

  Returns:
      (metrics, statuses): each of CHART_METRICS as [kv_head, position], NaN
      where undefined; statuses say per entry whether the values were
      finite and stayed so ('ok'), or not. A zero norm is not a status: it
      leaves only its own metrics undefined.
  """
  measured = metrics.compare(reference, target, axes=(0, 3), keys=CHART_METRICS)
  statuses = np.full(
      measured.status[CHART_METRICS[0]].shape, metrics.OK, object
  )
  for key in CHART_METRICS:
    code = measured.status[key]
    worse = (
        (statuses == metrics.OK)
        & (code != metrics.OK)
        & (code != metrics.ZERO_NORM)
    )
    statuses[worse] = code[worse]
  return {key: measured.values[key] for key in CHART_METRICS}, statuses


def _aggregate_heads(
    head_metrics: dict[str, np.ndarray],
    head: int | str,
    heads: list[int] | None,
) -> dict[str, np.ndarray]:
  """One value per position: a KV head's, or min/max/mean over KV heads."""
  if isinstance(head, int):
    return {key: head_metrics[key][head] for key in CHART_METRICS}
  result = {}
  for key in CHART_METRICS:
    values = head_metrics[key] if heads is None else head_metrics[key][heads]
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
      if head == 'mean':
        aggregate = np.where(valid, values, 0).sum(axis=0) / count
      elif head == 'min':
        aggregate = np.where(valid, values, np.inf).min(axis=0)
      else:
        aggregate = np.where(valid, values, -np.inf).max(axis=0)
    result[key] = np.where(count > 0, aggregate, np.nan)
  return result


def _range_row(
    context,
    layer,
    kind,
    start,
    end,
    edges,
    bounds,
    head,
    heads,
    alignment,
) -> dict[str, Any]:
  """Returns the statistics row of one layer and kind over the range."""
  row = {'layer': layer, 'kind': kind, 'status': 'unavailable', 'reasons': []}
  stats = {}
  try:
    pair = _pair(context, layer, kind, alignment is not None)
    if pair.status != 'ok':
      row.update(status=pair.status, reasons=pair.reasons)
      return _with_bins(row, bounds, stats)
    head_count = pair.tensors['ref'].canonical()[1]
    if isinstance(head, int) and head >= head_count:
      row.update(
          status='head_unavailable',
          reasons=[
              metrics.reason(
                  'head_unavailable',
                  'This KV head is outside the stored tensor.',
              )
          ],
      )
      return _with_bins(row, bounds, stats)
    # The panel's head domain is the union over layers; each layer uses the
    # selected heads it has.
    layer_heads = (
        None if heads is None else [h for h in heads if h < head_count]
    )
    if layer_heads is not None and not layer_heads:
      row.update(
          status='head_unavailable',
          reasons=[
              metrics.reason(
                  'head_unavailable',
                  'None of the selected KV heads is in this layer.',
              )
          ],
      )
      return _with_bins(row, bounds, stats)
    columns, values, partial = [], {key: [] for key in CHART_METRICS}, False
    for block, reference, target in _blocks(pair, start, end, alignment):
      measured, statuses = _measure(reference, target)
      aggregated = _aggregate_heads(measured, head, layer_heads)
      columns.append(block)
      for key in CHART_METRICS:
        values[key].append(aggregated[key])
      partial = partial or bool(
          np.any(statuses != 'ok')
          or any(np.any(~np.isfinite(value)) for value in aggregated.values())
      )
    row['status'] = 'ok'
    if not columns:
      row.update(
          status='position_unavailable',
          reasons=[
              metrics.reason(
                  'position_unavailable',
                  'This layer has no comparable positions in the range.',
              )
          ],
      )
      return _with_bins(row, bounds, stats)
    positions = np.concatenate(columns)
    for key in CHART_METRICS:
      stats[key] = statistics.binned(
          np.concatenate(values[key]), positions, edges
      )
    if partial:
      row.update(status='partial', reasons=[_reason('partial')])
  except (OSError, ValueError) as error:
    row.update(
        status='unavailable',
        reasons=[metrics.reason(metrics.UNREADABLE, str(error))],
    )
  return _with_bins(row, bounds, stats)


def _with_bins(
    row: dict[str, Any],
    bounds: list[dict[str, Any]],
    stats: dict[str, Any],
) -> dict[str, Any]:
  """Attaches each bin's statistics; a metric without any is empty."""
  for key in CHART_METRICS:
    if key not in stats:
      stats[key] = [statistics.empty() for _ in bounds]
  row['bins'] = [
      {**bound, 'metrics': {key: stats[key][i] for key in CHART_METRICS}}
      for i, bound in enumerate(bounds)
  ]
  return row


def _reason(code: str) -> dict[str, Any]:
  """A reason of the pair, of `code` in the words `REASONS` gives it."""
  return metrics.reason(code, REASONS[code])


def _head_failure(
    result: dict[str, Any], status: str, reasons: list[dict[str, Any]]
) -> dict[str, Any]:
  result.update(status=status, reasons=reasons)
  result['metric_status'] = dict.fromkeys(HEAD_METRICS, status)
  return result


def _scalar(value: np.generic) -> int | float | str:
  """A JSON-safe channel value: non-finite and huge integers as strings."""
  item = value.item()
  if isinstance(item, float) and not math.isfinite(item):
    return str(item)
  if isinstance(item, int) and abs(item) > 2**53 - 1:
    return str(item)
  return item
