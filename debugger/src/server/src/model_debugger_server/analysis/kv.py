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

"""KV cache comparison: kv-analysis, kv-range, kv-selection and kv-head.

A context pairs the Ref and Target KV snapshot of one turn and moment
(`prefill_pre`, `prefill_post`, `terminal`). Each of its rows is one layer and
kind (key or value). Positions are conversation-global: the stored sequence
index is the position, and only `[0, processed)` holds processed tokens; the
rest of the buffer is never read.

The dump holds no quantization scales, so stored values are compared as they
are (`values_mode: 'stored'`), accumulated in float64. With a position
alignment, a display column maps to one position per side; without one, the
column is the position on both sides.
"""

import dataclasses
import hashlib
import json
import math
import pathlib
import types
from typing import Any

from model_debugger_server import capture_index, errors, tensors
import numpy as np

LAYOUT = ('batch', 'kv_head', 'sequence', 'head_dim')
SIDES = ('ref', 'target')
KINDS = ('key', 'value')
# moment -> (phase, label)
MOMENTS = types.MappingProxyType({
    'prefill_pre': ('prefill', 'Before Prefill'),
    'prefill_post': ('prefill', 'After Prefill'),
    'terminal': ('decode', 'Generation end'),
})
MAX_BINS = 1024
MAX_SLICE_ELEMENTS = 262_144
MAX_ALIGNMENT_COLUMNS = 262_144
MAX_CHANNELS = 16_384
CHART_METRICS = (
    'relative_l2',
    'max_abs',
    'cosine_distance',
    'cosine_similarity',
)
TOKEN_METRICS = ('cosine_similarity', 'relative_l2', 'rmse', 'max_abs')
HEAD_METRICS = (
    'cosine_similarity',
    'cosine_distance',
    'relative_l2',
    'mean_abs',
    'rmse',
    'max_abs',
)
HEAD_AGGREGATIONS = ('max', 'min', 'mean')


@dataclasses.dataclass(frozen=True)
class _Tensor:
  """One side's stored KV tensor of one layer and kind."""

  file: str
  path: pathlib.Path
  dtype: str
  shape: tuple[Any, ...]
  layout: tuple[str, ...] | None
  processed: int
  runtime: str | None

  def geometry(self) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Returns (raw axis of each canonical axis, canonical shape), or None."""
    if (
        self.layout is None
        or len(self.layout) != 4
        or set(self.layout) != set(LAYOUT)
        or len(self.shape) != 4
        or any(type(size) is not int or size <= 0 for size in self.shape)
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
              'The stored tensor does not have a batch, KV head, sequence and'
              ' head dimension layout.'
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

  def structure(self) -> dict[str, Any]:
    """The UI's KvTokenStructure: one token's geometry on this side."""
    result = {
        'status': 'unavailable',
        'shape': None,
        'dtype': self.dtype,
        'head_count': None,
        'channel_count': None,
        'batch_count': None,
        'position_start': None,
        'position_count': None,
    }
    problem = self.problem()
    if problem is not None and problem[0] != 'empty':
      result.update(status=problem[0], reason=problem[1])
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
    return {
        'runtime': self.runtime,
        'source_dtype': self.dtype,
        'source_shape': list(self.shape),
        'comparison_dtype': self.dtype,
        'comparison_shape': list(self.canonical()) if geometry else None,
        'layout': list(self.layout) if self.layout is not None else None,
        'axis_order': list(geometry[0]) if geometry else None,
        'dequantization': None,
        'values_mode': 'stored',
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
      positions: increasing positions in [0, processed).
      head: one KV head, or None for every head.
      batch: one batch, or None for every batch.

    Returns:
      The stored values in canonical axis order, in the stored dtype.

    Raises:
      ValueError: one position holds more than the slice element limit, or
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
      if last - first + 1 == len(positions):
        return value
      return value[:, :, positions - first, :]
    # A sparse alignment: read each run of consecutive positions.
    breaks = np.flatnonzero(np.diff(positions) != 1) + 1
    parts = []
    for run in np.split(positions, breaks):
      for low in range(int(run[0]), int(run[-1]) + 1, chunk):
        high = min(int(run[-1]) + 1, low + chunk)
        parts.append(self._read_span(order, canonical, low, high, head, batch))
    return np.concatenate(parts, axis=2)

  def _read_span(self, order, canonical, start, end, head, batch):
    index = [slice(None)] * 4
    for axis, value in ((0, batch), (1, head)):
      if value is not None:
        index[order[axis]] = slice(value, value + 1)
    index[order[2]] = slice(start, end)
    value = tensors.read_slice(self.path, tuple(index)).transpose(order)
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
  # side -> the index's forwards of the turn
  forwards: dict[str, list[dict[str, Any]]]
  # (layer, kind) -> {side: _Tensor}
  rows: dict[tuple[int, str], dict[str, _Tensor]]


@dataclasses.dataclass(frozen=True)
class _Pair:
  """A row's tensors and whether they can be compared."""

  tensors: dict[str, _Tensor]  # side -> _Tensor, for the sides that stored one
  status: str
  reason: str | None

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
    dtype = self.tensors['ref'].dtype
    basis = f'Stored {dtype} values, compared as stored in float64'
    if dtype[:1] in ('I', 'U'):
      return basis + '; the capture records no quantization scales.'
    return basis + '.'


@dataclasses.dataclass(frozen=True)
class _Alignment:
  """Display columns [start, start + len) and each side's position or -1."""

  start: int
  ref: np.ndarray
  target: np.ndarray

  def end(self) -> int:
    return self.start + len(self.ref)

  def positions(self, column: int) -> dict[str, int | None]:
    index = column - self.start
    if not 0 <= index < len(self.ref):
      return {'ref': None, 'target': None}
    return {
        side: int(values[index]) if values[index] >= 0 else None
        for side, values in (('ref', self.ref), ('target', self.target))
    }


def kv_analysis(index: capture_index.CaptureIndex, query: dict) -> dict:
  """Answers GET kv-analysis?mode=metadata: the contexts of a turn.

  Args:
    index: the chat's capture.
    query: `mode` ('metadata') and `turn`.

  Returns:
    `{contexts}`, each context with its layer rows; no tensor is read.
  """
  if query.get('mode') != 'metadata':
    raise ValueError('Only kv-analysis with mode=metadata is served.')
  n = _integer(query, 'turn', minimum=1)
  return {
      'contexts': [
          _context_view(index, context) for context in _contexts(index, n)
      ]
  }


def kv_range(
    index: capture_index.CaptureIndex, query: dict, payload: dict | None
) -> dict:
  """Answers kv-range: binned per-position metrics of every layer of a kind.

  Args:
    index: the chat's capture.
    query: `turn`, `context`, `start`, `end`, `bins` (128), `kind` ('key'),
      `head` ('max', 'min', 'mean' or a KV head index), optional `heads` (a JSON
      array of KV heads to aggregate over).
    payload: an optional `{alignment}`.

  Returns:
    `{context_id, start, end, kind, head, heads, bins, rows}`.
  """
  n = _integer(query, 'turn', minimum=1)
  context_id = _text(query, 'context')
  start, end = _range(query)
  bins = _integer(query, 'bins', default=128, minimum=1)
  if bins > MAX_BINS:
    raise ValueError(f'At most {MAX_BINS} chart bins are allowed.')
  kind = _kind(query)
  head = _head(query.get('head', 'max'))
  heads = _heads(query.get('heads'), head)
  if query.get('formula'):
    raise ValueError('KV formulas are not supported.')
  alignment = _alignment(payload)
  context = _context(index, n, context_id)
  count = min(bins, end - start)
  edges = np.array(
      [start + i * (end - start) // count for i in range(count + 1)]
  )
  bounds = [
      {'start': int(edges[i]), 'end': int(edges[i + 1])} for i in range(count)
  ]
  rows = [
      _range_row(
          index,
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
      for layer, row_kind in sorted(context.rows, key=_row_order)
      if row_kind == kind
  ]
  return {
      'context_id': context_id,
      'start': start,
      'end': end,
      'kind': kind,
      'head': head,
      'heads': heads,
      'bins': bounds,
      'rows': rows,
  }


def kv_selection(
    index: capture_index.CaptureIndex, query: dict, payload: dict | None
) -> dict:
  """Answers kv-selection: whole-token metrics over a range, per row.

  Each metric keeps its worst finite value in the range (lowest cosine
  similarity, largest errors) and where it is. A one-position range also
  carries that position's `token_metrics`.

  Args:
    index: the chat's capture.
    query: `turn`, `context`, `start`, `end`, optional `layer`.
    payload: an optional `{alignment}`.

  Returns:
    `{context_id, start, end, rows}`.
  """
  n = _integer(query, 'turn', minimum=1)
  context_id = _text(query, 'context')
  start, end = _range(query)
  layer = _integer(query, 'layer', default=None)
  alignment = _alignment(payload)
  context = _context(index, n, context_id)
  rows = [
      _selection_row(index, context, row_layer, kind, start, end, alignment)
      for row_layer, kind in sorted(context.rows, key=_row_order)
      if layer is None or row_layer == layer
  ]
  return {'context_id': context_id, 'start': start, 'end': end, 'rows': rows}


def kv_head(
    index: capture_index.CaptureIndex, query: dict, payload: dict | None
) -> dict:
  """Answers kv-head: one KV head's channels at one position, both sides.

  Args:
    index: the chat's capture.
    query: `turn`, `context_id`, `layer`, `kind`, `position`, `head`, optional
      `batch`.
    payload: an optional `{alignment}`.

  Returns:
    The UI's KvHeadEvidence.

  Raises:
    ValueError: the batch or head is outside the stored tensor.
  """
  n = _integer(query, 'turn', minimum=1)
  context_id = _text(query, 'context_id')
  layer = _integer(query, 'layer')
  kind = _kind(query, default=None)
  position = _integer(query, 'position')
  head = _integer(query, 'head')
  batch = _integer(query, 'batch', default=None)
  alignment = _alignment(payload)
  context = _context(index, n, context_id)
  positions = (
      {'ref': position, 'target': position}
      if alignment is None
      else alignment.positions(position)
  )
  selection = {
      'turn': n,
      'context_id': context_id,
      'layer': layer,
      'kind': kind,
      'position': position,
      'head': head,
      'batch': batch,
  }
  if alignment is not None:
    selection['positions'] = positions
  result = {
      'selection': selection,
      'status': 'unavailable',
      'batch_count': None,
      'channel_count': None,
      'shape': None,
      'metrics': dict.fromkeys(HEAD_METRICS),
      'metric_status': dict.fromkeys(HEAD_METRICS, 'unavailable'),
      'channels': [],
      'largest_channel': None,
      'sources': {side: None for side in SIDES},
      'comparison_basis': None,
      'calculation_dtype': 'float64',
      'delta_definition': 'Target - Reference',
  }
  pair = _pair(index, context, layer, kind, alignment is not None)
  for side, tensor in pair.tensors.items():
    result['sources'][side] = tensor.source()
  result['comparison_basis'] = pair.basis()
  status, reason = pair.status, pair.reason
  readable = pair.readable()
  if not readable:
    return _head_failure(
        result,
        status,
        reason or 'No stored tensor is linked to this selection.',
    )
  shapes = {side: tensor.canonical() for side, tensor in readable.items()}
  for side, shape in shapes.items():
    result['sources'][side]['comparison_shape'] = [shape[3]]
  result['batch_count'] = max(shape[0] for shape in shapes.values())
  result['channel_count'] = max(shape[3] for shape in shapes.values())
  result['shape'] = [result['channel_count']]
  if result['channel_count'] > MAX_CHANNELS:
    return _head_failure(
        result,
        'analysis_limit',
        f'A KV head may have at most {MAX_CHANNELS} channels.',
    )
  if batch is None and result['batch_count'] > 1:
    return _head_failure(
        result,
        'batch_selection_required',
        'Select a batch to inspect this position and head; batches are not '
        'combined.',
    )
  batch = 0 if batch is None else batch
  selection['batch'] = batch
  if not any(batch < s[0] and head < s[1] for s in shapes.values()):
    raise ValueError('The batch or KV head is outside the stored tensor.')
  vectors, failures = _read_head_vectors(
      readable, shapes, positions, head, batch
  )
  if not vectors:
    ranges = ', '.join(
        f'{side}: [0, {shape[2]})' for side, shape in shapes.items()
    )
    return _head_failure(
        result,
        failures[0][0] if failures else 'position_unavailable',
        f'Cache position {position} was not captured in this layer and kind. '
        f'Captured ranges: {ranges}.',
    )
  comparable = (
      status == 'ok'
      and len(vectors) == 2
      and vectors['ref'].shape == vectors['target'].shape
  )
  if comparable:
    try:
      measured = _compare(vectors['ref'], vectors['target'])
    except ValueError as error:
      _head_failure(
          result, 'partial', f'The head metrics are unavailable: {error}'
      )
      result['metric_status'] = dict.fromkeys(HEAD_METRICS, str(error))
    else:
      for key, (value, metric_status) in measured.items():
        result['metrics'][key] = value
        result['metric_status'][key] = metric_status
      result['status'] = (
          'ok'
          if all(value == 'ok' for value in result['metric_status'].values())
          else 'partial'
      )
  else:
    detail = '; '.join(
        filter(None, [reason, *(failure[1] for failure in failures)])
    )
    _head_failure(
        result,
        status if status != 'ok' else 'unavailable',
        detail or 'The vector is unavailable on one side.',
    )
  result['channels'], result['largest_channel'] = _compute_head_channels(
      vectors, comparable, result['status']
  )
  return result


def _read_head_vectors(
    readable: dict[str, _Tensor],
    shapes: dict[str, tuple[int, ...]],
    positions: dict[str, int | None],
    head: int,
    batch: int,
) -> tuple[dict[str, np.ndarray], list[tuple[str, str]]]:
  """Reads the 1-D channel vector of one head and position on each side."""
  vectors: dict[str, np.ndarray] = {}
  failures: list[tuple[str, str]] = []
  for side, tensor in readable.items():
    at = positions[side]
    if at is None:
      continue
    shape = shapes[side]
    if not (0 <= at < shape[2] and batch < shape[0] and head < shape[1]):
      failures.append((
          'position_unavailable',
          f'{side}: position {at}, head {head} was not captured.',
      ))
      continue
    try:
      value = tensor.read(np.array([at]), head=head, batch=batch)
    except (OSError, ValueError) as error:
      failures.append(('unavailable', f'{side}: {error}'))
      continue
    vectors[side] = value[0, 0, 0, :]
  return vectors, failures


def _compute_head_channels(
    vectors: dict[str, np.ndarray], comparable: bool, base_status: str
) -> tuple[list[dict[str, Any]], int | None]:
  """Computes per-channel comparisons and the largest absolute delta channel."""
  channels = []
  for channel in range(max(len(vector) for vector in vectors.values())):
    raw = {
        side: vector[channel]
        for side, vector in vectors.items()
        if channel < len(vector)
    }
    delta, channel_status = None, base_status
    if comparable:
      channel_status = 'non_finite'
      if all(np.isfinite(value) for value in raw.values()):
        with np.errstate(over='ignore', invalid='ignore'):
          candidate = float(np.float64(raw['target']) - np.float64(raw['ref']))
        if math.isfinite(candidate):
          delta, channel_status = candidate, 'ok'
    channels.append({
        'channel': channel,
        'ref': _scalar(raw['ref']) if 'ref' in raw else None,
        'target': _scalar(raw['target']) if 'target' in raw else None,
        'delta': delta,
        'abs_delta': abs(delta) if delta is not None else None,
        'status': channel_status,
    })
  finite = [row for row in channels if row['abs_delta'] is not None]
  largest = (
      max(finite, key=lambda row: row['abs_delta'])['channel']
      if finite
      else None
  )
  return channels, largest


# Contexts and rows.


def _contexts(index: capture_index.CaptureIndex, n: int) -> list[_Context]:
  turn = index.turn(n)
  sides = turn['sides']
  contexts = []
  for moment in MOMENTS:
    snapshots = {}
    for side in SIDES:
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
      for layer in snapshot['layers']:
        for kind in KINDS:
          stored = layer.get(kind)
          if stored is None:
            continue
          rows.setdefault((layer['layer'], kind), {})[side] = _Tensor(
              file=stored['file'],
              path=index.path(side, n, stored['file']),
              dtype=stored['dtype'],
              shape=tuple(stored['shape']),
              layout=tuple(stored['layout']) if stored.get('layout') else None,
              processed=snapshot['processed'],
              runtime=runtime,
          )
    digest = hashlib.sha256(json.dumps([n, moment]).encode()).hexdigest()
    contexts.append(
        _Context(
            id='snapshot-' + digest[:24],
            turn=n,
            moment=moment,
            snapshots=snapshots,
            forwards={side: sides[side]['forwards'] for side in snapshots},
            rows=rows,
        )
    )
  return contexts


def _context(
    index: capture_index.CaptureIndex, n: int, context_id: str
) -> _Context:
  for context in _contexts(index, n):
    if context.id == context_id:
      return context
  raise errors.NotFound(f'Turn {n} has no KV context {context_id}.')


def _row_order(key: tuple[int, str]) -> tuple[int, int]:
  layer, kind = key
  return layer, KINDS.index(kind)


def _pair(
    index: capture_index.CaptureIndex,
    context: _Context,
    layer: int,
    kind: str,
    aligned: bool,
) -> _Pair:
  """The status gates, in order; alignment lifts the range and length ones."""
  stored = context.rows.get((layer, kind), {})
  if len(stored) != 2:
    return _Pair(
        stored,
        'unavailable',
        'A stored tensor is required on both sides at this observation.',
    )
  for side in SIDES:
    problem = stored[side].problem()
    if problem is not None:
      return _Pair(stored, problem[0], f'{side}: {problem[1]}')
  ref, target = stored['ref'], stored['target']
  models = [index.sides[side]['model']['sha256'] for side in SIDES]
  if models[0] != models[1]:
    return _Pair(
        stored,
        'model_mismatch',
        'Ref and Target ran different model files; their caches are not '
        'comparable.',
    )
  if ref.dtype != target.dtype:
    return _Pair(
        stored,
        'dtype_mismatch',
        f'Ref stores {ref.dtype} and Target stores {target.dtype}.',
    )
  if ref.layout != target.layout:
    return _Pair(
        stored, 'layout_mismatch', 'Ref and Target store different layouts.'
    )
  ref_shape, target_shape = ref.canonical(), target.canonical()
  if any(ref_shape[axis] != target_shape[axis] for axis in (0, 1, 3)):
    return _Pair(
        stored,
        'shape_mismatch',
        'The batch, KV head or head dimension sizes differ.',
    )
  if not aligned and ref.processed != target.processed:
    return _Pair(
        stored,
        'logical_range_mismatch',
        f'Ref processed {ref.processed} positions and Target '
        f'{target.processed}; compare them through a position alignment.',
    )
  return _Pair(stored, 'ok', None)


def _context_view(index: capture_index.CaptureIndex, context: _Context) -> dict:
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
      'forward_id': None,
      'moment': context.moment,
      'source': 'snapshot',
      'runtime': runtimes.pop() if len(runtimes) == 1 else None,
      'layers': [],
      'snapshots': {side: [] for side in SIDES},
  }
  if len(processed) == 1:
    view['processed_token_count'] = processed.pop()
    view['label'] += f' · {view["processed_token_count"]} processed tokens'
  for side, snapshot in context.snapshots.items():
    view['snapshots'][side].append(
        _snapshot_identity(index, context, side, snapshot, phase)
    )
  for layer, kind in sorted(context.rows, key=_row_order):
    pair = _pair(index, context, layer, kind, False)
    row = {
        'layer': layer,
        'kind': kind,
        'status': pair.status,
        'head_count': None,
        'position_start': None,
        'position_count': None,
        'cells': [],
        'token_metrics': [],
        'token_structure': {side: None for side in SIDES},
    }
    if pair.reason:
      row['reason'] = pair.reason
    for side, tensor in pair.tensors.items():
      row['token_structure'][side] = tensor.structure()
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


def _snapshot_identity(
    index: capture_index.CaptureIndex,
    context: _Context,
    side: str,
    snapshot: dict,
    phase: str,
) -> dict:
  """The forward whose files hold this snapshot names its signature."""
  files = {
      stored['file']
      for layer in snapshot['layers']
      for stored in (layer.get('key'), layer.get('value'))
      if stored is not None
  }
  forward = next(
      (
          row
          for row in context.forwards[side]
          if files & set(row['tensors'].values())
      ),
      None,
  )
  return {
      'id': None,
      'snapshot_id': None,
      'runtime': index.sides[side]['runtime']['name'],
      'turn': context.turn,
      'phase': phase,
      'step': snapshot['step'],
      'forward_id': forward['id'] if forward else None,
      'moment': context.moment,
      'signature': forward['signature'] if forward else None,
      'edge': 'pre' if context.moment == 'prefill_pre' else 'post',
  }


# Reading and measuring.


def _blocks(pair: _Pair, start: int, end: int, alignment: _Alignment | None):
  """Yields (columns, ref values, target values) for comparable columns.

  A column is compared only where both sides have a processed position. Each
  block reads at most MAX_SLICE_ELEMENTS elements per side and read.
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
    reference: np.ndarray, target: np.ndarray, whole: bool
) -> tuple[dict, np.ndarray]:
  """Per-position metrics of [batch, kv_head, position, head_dim] values.

  Args:
    reference: Ref values.
    target: Target values, same shape.
    whole: reduce over batch, KV head and channel (one value per position);
      otherwise over batch and channel (one value per KV head and position).

  Returns:
    (metrics, statuses): each metric is [position] or [kv_head, position],
    NaN where undefined (zero norm, non-finite input, overflow); statuses
    say why per entry.
  """
  x, y = reference.astype(np.float64), target.astype(np.float64)
  subscripts = 'bhpd,bhpd->p' if whole else 'bhpd,bhpd->hp'
  axes = (0, 1, 3) if whole else (0, 3)
  count = math.prod(x.shape[axis] for axis in axes)
  with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
    delta = x - y
    xx = np.einsum(subscripts, x, x)
    yy = np.einsum(subscripts, y, y)
    xy = np.einsum(subscripts, x, y)
    errors = np.einsum(subscripts, delta, delta)
    maximum = np.abs(delta).max(axis=axes)
    nx, ny = np.sqrt(xx), np.sqrt(yy)
    # √(xx·yy) makes equal vectors exactly 1; nx·ny where xx·yy overflows.
    denominator = np.sqrt(xx * yy)
    denominator = np.where(np.isfinite(denominator), denominator, nx * ny)
    cosine = np.clip(xy / denominator, -1.0, 1.0)
    relative = np.sqrt(errors) / nx
    rmse = np.sqrt(errors / count)
  if reference.dtype.kind in 'iu' and target.dtype.kind in 'iu':
    finite = np.ones(nx.shape, dtype=bool)
  else:
    finite = np.all(np.isfinite(x) & np.isfinite(y), axis=axes)
  overflow = ~(
      np.isfinite(nx)
      & np.isfinite(ny)
      & np.isfinite(errors)
      & np.isfinite(maximum)
  )
  invalid = ~finite | overflow
  statuses = np.where(
      ~finite, 'non_finite_tensor', np.where(overflow, 'numeric_overflow', 'ok')
  )
  metrics = {
      'cosine_similarity': cosine,
      'cosine_distance': np.clip(1 - cosine, 0.0, 2.0),
      'relative_l2': relative,
      'rmse': rmse,
      'max_abs': maximum,
  }
  return {
      key: np.where(invalid | ~np.isfinite(value), np.nan, value)
      for key, value in metrics.items()
  }, statuses


def _aggregate_heads(
    metrics: dict, head: int | str, heads: list[int] | None
) -> dict:
  """One value per position: a KV head's, or min/max/mean over KV heads."""
  if isinstance(head, int):
    return {key: metrics[key][head] for key in CHART_METRICS}
  result = {}
  for key in CHART_METRICS:
    values = metrics[key] if heads is None else metrics[key][heads]
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


def _empty_stat() -> dict:
  return {
      'min': None,
      'max': None,
      'mean': None,
      'min_position': None,
      'max_position': None,
      'valid_count': 0,
  }


def _bin_stats(
    values: np.ndarray, positions: np.ndarray, bin_ids: np.ndarray, count: int
) -> list[dict]:
  """Finite min, max, mean, their first positions and counts per bin.

  `positions` increase, so each bin is one contiguous segment.
  """
  valid = np.isfinite(values)
  values, positions, bin_ids = values[valid], positions[valid], bin_ids[valid]
  if not values.size:
    return [_empty_stat() for _ in range(count)]
  starts = np.flatnonzero(np.r_[True, np.diff(bin_ids) != 0])
  counts = np.diff(np.r_[starts, values.size])
  # Dividing before summing keeps a finite mean from overflowing.
  means = np.add.reduceat(values / np.repeat(counts, counts), starts)
  columns = {'mean': means.tolist(), 'valid_count': counts.tolist()}
  for name, reduce in (('min', np.minimum), ('max', np.maximum)):
    extreme = reduce.reduceat(values, starts)
    hits = np.flatnonzero(values == np.repeat(extreme, counts))
    segments = np.searchsorted(starts, hits, side='right') - 1
    _, first = np.unique(segments, return_index=True)
    columns[name] = extreme.tolist()
    columns[name + '_position'] = positions[hits[first]].tolist()
  stats = [None] * count
  for i, owner in enumerate(bin_ids[starts].tolist()):
    stats[owner] = {key: column[i] for key, column in columns.items()}
  return [stat or _empty_stat() for stat in stats]


def _range_row(
    index,
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
) -> dict:
  row = {'layer': layer, 'kind': kind, 'status': 'unavailable'}
  stats = {}
  try:
    pair = _pair(index, context, layer, kind, alignment is not None)
    if pair.status != 'ok':
      row.update(
          status=pair.status,
          reason=pair.reason or pair.status.replace('_', ' '),
      )
      return _with_bins(row, bounds, stats)
    head_count = pair.tensors['ref'].canonical()[1]
    if isinstance(head, int) and head >= head_count:
      row.update(
          status='head_unavailable',
          reason='This KV head is outside the stored tensor.',
      )
      return _with_bins(row, bounds, stats)
    # The panel's head domain is the union over layers; each layer uses the
    # selected heads it has.
    layer_heads = (
        None if heads is None else [h for h in heads if h < head_count]
    )
    if layer_heads == []:
      row.update(
          status='head_unavailable',
          reason='None of the selected KV heads is in this layer.',
      )
      return _with_bins(row, bounds, stats)
    columns, values, partial = [], {key: [] for key in CHART_METRICS}, False
    for block, reference, target in _blocks(pair, start, end, alignment):
      measured, statuses = _measure(reference, target, whole=False)
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
          reason='This layer has no comparable positions in the range.',
      )
      return _with_bins(row, bounds, stats)
    positions = np.concatenate(columns)
    bin_ids = np.searchsorted(edges, positions, side='right') - 1
    for key in CHART_METRICS:
      stats[key] = _bin_stats(
          np.concatenate(values[key]), positions, bin_ids, len(bounds)
      )
    if partial:
      row.update(
          status='partial',
          reason=(
              'Some metrics are undefined (zero norm, non-finite values or '
              'numeric overflow); finite counts are reported per metric.'
          ),
      )
  except (OSError, ValueError) as error:
    row.update(status='unavailable', reason=str(error))
  return _with_bins(row, bounds, stats)


def _with_bins(row: dict, bounds: list[dict], stats: dict) -> dict:
  """Attaches each bin's statistics; a metric without any is empty."""
  for key in CHART_METRICS:
    if key not in stats:
      stats[key] = [_empty_stat() for _ in bounds]
  row['bins'] = [
      {**bound, 'metrics': {key: stats[key][i] for key in CHART_METRICS}}
      for i, bound in enumerate(bounds)
  ]
  return row


def _selection_row(index, context, layer, kind, start, end, alignment) -> dict:
  row = {'layer': layer, 'kind': kind, 'status': 'unavailable'}
  columns, values, statuses = [], {key: [] for key in TOKEN_METRICS}, []
  try:
    pair = _pair(index, context, layer, kind, alignment is not None)
    row['status'] = pair.status
    if pair.status != 'ok':
      row['reason'] = pair.reason or pair.status.replace('_', ' ')
    else:
      for block, reference, target in _blocks(pair, start, end, alignment):
        measured, block_statuses = _measure(reference, target, whole=True)
        columns.append(block)
        statuses.append(block_statuses)
        for key in TOKEN_METRICS:
          values[key].append(measured[key])
      if not columns:
        row.update(
            status='position_unavailable',
            reason='This layer has no comparable positions in the range.',
        )
  except (OSError, ValueError) as error:
    row.update(status='unavailable', reason=str(error))
    columns = []
  if columns:
    positions = np.concatenate(columns)
    values = {key: np.concatenate(values[key]) for key in TOKEN_METRICS}
    statuses = np.concatenate(statuses)
    if np.any(statuses != 'ok') or any(
        np.any(~np.isfinite(value)) for value in values.values()
    ):
      row.update(
          status='partial',
          reason=(
              'Some token metrics are undefined; only finite values '
              'contribute to each range summary.'
          ),
      )
    if end - start == 1:
      token = {
          key: float(values[key][0]) if np.isfinite(values[key][0]) else None
          for key in TOKEN_METRICS
      }
      metric_status = {
          key: (
              'ok'
              if token[key] is not None
              else (
                  str(statuses[0])
                  if statuses[0] != 'ok'
                  else 'undefined_zero_norm'
              )
          )
          for key in TOKEN_METRICS
      }
      row['token_metrics'] = [{
          'position': int(positions[0]),
          'status': (
              'ok'
              if all(value == 'ok' for value in metric_status.values())
              else 'partial'
          ),
          'metrics': token,
          'metric_status': metric_status,
      }]
  row['metrics'] = {}
  for key in TOKEN_METRICS:
    # The worst value: the lowest similarity, the largest error.
    side = 'min' if key == 'cosine_similarity' else 'max'
    stat = (
        _bin_stats(values[key], positions, np.zeros_like(positions), 1)[0]
        if columns
        else _empty_stat()
    )
    if stat['valid_count']:
      status = 'ok'
    elif row.get('token_metrics'):
      status = row['token_metrics'][0]['metric_status'][key]
    elif row['status'] != 'ok':
      status = row['status']
    else:
      status = 'undefined_or_not_captured'
    row['metrics'][key] = {
        'value': stat[side],
        'position': stat[side + '_position'],
        'valid_count': stat['valid_count'],
        'status': status,
    }
  return row


def _compare(reference: np.ndarray, target: np.ndarray) -> dict:
  """Metrics of two vectors as {key: (value or None, status)}.

  Raises:
    ValueError: a value is not finite, or a metric overflows.
  """
  x, y = reference.astype(np.float64), target.astype(np.float64)
  if not np.isfinite(x).all() or not np.isfinite(y).all():
    raise ValueError('non_finite_tensor')
  try:
    with np.errstate(over='raise', invalid='raise', divide='raise'):
      delta = x - y
      error = np.abs(delta)
      nx, ny = np.linalg.norm(x), np.linalg.norm(y)
      if not np.isfinite(nx) or not np.isfinite(ny):
        raise ValueError('numeric_overflow')
      cosine = None if nx == 0 or ny == 0 else _cosine(x, y, nx * ny)
      values = {
          'cosine_similarity': cosine,
          'cosine_distance': (
              None if cosine is None else float(np.clip(1 - cosine, 0.0, 2.0))
          ),
          'relative_l2': None if nx == 0 else float(np.linalg.norm(delta) / nx),
          'mean_abs': float(error.mean()),
          'rmse': float(np.sqrt(np.mean(delta * delta))),
          'max_abs': float(error.max()),
      }
  except FloatingPointError as error:
    raise ValueError('numeric_overflow') from error
  if any(v is not None and not math.isfinite(v) for v in values.values()):
    raise ValueError('numeric_overflow')
  return {
      key: (value, 'undefined_zero_norm' if value is None else 'ok')
      for key, value in values.items()
  }


def _cosine(x: np.ndarray, y: np.ndarray, norms: float) -> float:
  """x·y / √(x·x · y·y), exactly 1 for equal vectors; `norms` on overflow."""
  with np.errstate(over='ignore'):
    product = float(x @ x) * float(y @ y)
  denominator = math.sqrt(product) if math.isfinite(product) else norms
  return float(np.clip(float(x @ y) / denominator, -1.0, 1.0))


def _head_failure(result: dict, status: str, reason: str) -> dict:
  result.update(status=status, reason=reason)
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


# Request parsing. Query values arrive as strings.


def _text(query: dict, name: str) -> str:
  value = query.get(name)
  if not isinstance(value, str) or not value:
    raise ValueError(f'The request needs a {name}.')
  return value


_REQUIRED = object()


def _integer(query: dict, name: str, default=_REQUIRED, minimum: int = 0):
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


def _range(query: dict) -> tuple[int, int]:
  start, end = _integer(query, 'start'), _integer(query, 'end')
  if end <= start:
    raise ValueError('The position range [start, end) must not be empty.')
  return start, end


def _kind(query: dict, default: str | None = 'key') -> str:
  kind = query.get('kind', default)
  if kind not in KINDS:
    raise ValueError('kind must be key or value.')
  return kind


def _head(value: str) -> int | str:
  if value in HEAD_AGGREGATIONS:
    return value
  if not (value.isascii() and value.isdigit()):
    raise ValueError('head must be max, min, mean or a KV head index.')
  return int(value)


def _heads(value: str | None, head: int | str) -> list[int] | None:
  if value is None:
    return None
  try:
    heads = json.loads(value)
  except json.JSONDecodeError as error:
    raise ValueError('heads must be a JSON array of KV heads.') from error
  if (
      not isinstance(heads, list)
      or not heads
      or any(type(h) is not int or h < 0 for h in heads)
      or len(set(heads)) != len(heads)
      or head not in HEAD_AGGREGATIONS
  ):
    raise ValueError(
        'heads must be a nonempty array of distinct KV head indices, used with '
        'head min, max or mean.'
    )
  return sorted(heads)


def _alignment(payload: dict | None) -> _Alignment | None:
  """Validates `{alignment: {start, pairs}}`; positions increase per side."""
  value = (payload or {}).get('alignment')
  if value is None:
    return None
  if (
      not isinstance(value, dict)
      or type(value.get('start')) is not int
      or value['start'] < 0
  ):
    raise ValueError('An alignment needs a start and position pairs.')
  pairs = value.get('pairs')
  if not isinstance(pairs, list) or len(pairs) > MAX_ALIGNMENT_COLUMNS:
    raise ValueError(
        f'An alignment may have at most {MAX_ALIGNMENT_COLUMNS} pairs.'
    )
  columns = np.full((2, len(pairs)), -1, dtype=np.int64)
  previous = [-1, -1]
  for i, pair in enumerate(pairs):
    if not isinstance(pair, list) or len(pair) != 2 or pair == [None, None]:
      raise ValueError('Each alignment pair needs a Ref or Target position.')
    for side, position in enumerate(pair):
      if position is None:
        continue
      if type(position) is not int or position <= previous[side]:
        raise ValueError('Aligned positions must increase on each side.')
      previous[side] = columns[side, i] = position
  return _Alignment(value['start'], columns[0], columns[1])
