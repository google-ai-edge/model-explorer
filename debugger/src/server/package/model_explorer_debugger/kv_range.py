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
# ==============================================================================

"""Bounded KV charts, range details, and global Find over capture slices."""

from collections import OrderedDict
from functools import wraps
import hashlib
import json
import math
from pathlib import Path
import threading

from model_explorer_debugger import errors
import numpy as np

from .kv_analysis import TOKEN_AGGREGATION
from .kv_formula import compile_kv_formula
from .kv_reader import KvReader, MAX_SLICE_ELEMENTS

CHART_METRICS = ('relative_l2', 'max_abs', 'cosine_distance')
TOKEN_METRICS = ('cosine_similarity', 'relative_l2', 'rmse', 'max_abs')
MAX_BINS = 1024
MAX_FIND_PAGE = 40
MAX_FIND_BLOCK_RECORDS = 4096
MAX_FIND_INDEX_BLOCKS = 8192
MAX_REPLY_CACHE_BYTES = 8 * 1024 * 1024
_CACHE_LOCK = threading.RLock()
_WORKERS = threading.BoundedSemaphore(2)
_WORK_DEPTH = threading.local()


def _revision(store):
  """Cache only unchanged source declarations and immutable file identities."""
  root = Path(store.root)
  paths = {
      root / row['path']
      for row in store._telemetry['resources']
      if isinstance(row.get('path'), str)
  }
  pairs = root / 'explicit_pairs'
  if pairs.is_dir():
    for path in pairs.rglob('*'):
      if path.is_file():
        paths.add(path)
        if len(paths) > 4096:
          return None
  signatures = []
  for path in sorted(paths):
    try:
      stat = path.stat()
      signatures.append((
          str(path),
          stat.st_ino,
          stat.st_size,
          stat.st_mtime_ns,
          stat.st_ctime_ns,
      ))
    except OSError:
      signatures.append((str(path), 'unavailable'))
  declarations = dict(
      session=store.session, telemetry=store._telemetry, tensors=store.tensors
  )
  return hashlib.sha256(
      json.dumps([declarations, signatures], sort_keys=True).encode()
  ).hexdigest()


def _cached(function):
  """Bound replies/indexes by bytes; coalesce identical concurrent requests."""

  @wraps(function)
  def run(store, *args, **kwargs):
    revision = _revision(store)
    if revision is None:
      return function(store, *args, **kwargs)
    key = (
        revision,
        function.__name__,
        json.dumps([args, kwargs], sort_keys=True),
    )
    while True:
      with _CACHE_LOCK:
        if not hasattr(store, '_kv_reply_cache'):
          store._kv_reply_cache = dict(
              values=OrderedDict(), pending={}, bytes=0
          )
        cache = store._kv_reply_cache
        if key in cache['values']:
          encoded, _ = cache['values'][key]
          value = json.loads(encoded)
          cache['values'].move_to_end(key)
          return {
              **value,
              'cache_hit': True,
              'work': {
                  **value.get('work', {}),
                  'read_chunks': 0,
                  'max_side_elements': 0,
                  'compared_positions': 0,
                  'cache_hit': True,
              },
          }
        event = cache['pending'].get(key)
        if event is None:
          if len(cache['pending']) >= 16:
            raise errors.AnalysisBusy(
                'The bounded KV work queue is full; retry shortly.'
            )
          event = threading.Event()
          cache['pending'][key] = event
          break
      if not event.wait(60):
        raise errors.AnalysisBusy(
            'The identical KV request is still running; retry shortly.'
        )
    try:
      depth = getattr(_WORK_DEPTH, 'value', 0)
      if depth == 0 and not _WORKERS.acquire(timeout=60):
        raise errors.AnalysisBusy(
            'KV analysis workers are busy; retry shortly.'
        )
      _WORK_DEPTH.value = depth + 1
      try:
        result = function(store, *args, **kwargs)
      finally:
        _WORK_DEPTH.value = depth
        if depth == 0:
          _WORKERS.release()
      encoded = json.dumps(
          result, separators=(',', ':'), allow_nan=False
      ).encode()
      size = len(encoded)
      with _CACHE_LOCK:
        if result.get('_cacheable', True) and size <= MAX_REPLY_CACHE_BYTES:
          while cache['values'] and (
              cache['bytes'] + size > MAX_REPLY_CACHE_BYTES
              or len(cache['values']) >= 8
          ):
            _, (_, removed) = cache['values'].popitem(last=False)
            cache['bytes'] -= removed
          cache['values'][key] = (encoded, size)
          cache['bytes'] += size
      return result
    finally:
      with _CACHE_LOCK:
        cache['pending'].pop(key, None)
        event.set()

  return run


def reader_for(store):
  # Resolved metadata is request-local; verified-file/proof caches live on
  # store.
  return KvReader(store)


def metadata(store, turn):
  return dict(
      contexts=reader_for(store).contexts(turn),
      notice='Metrics are read in bounded slices for the selected range.',
      limits=dict(
          max_bins=MAX_BINS,
          max_slice_elements=MAX_SLICE_ELEMENTS,
          max_find_page=MAX_FIND_PAGE,
      ),
  )


def _context(store, turn, identity):
  reader = reader_for(store)
  context = next(
      (value for value in reader.contexts(turn) if value['id'] == identity),
      None,
  )
  if context is None:
    raise ValueError('Unknown KV context identity')
  return reader, context


def _bounds(start, end):
  if (
      type(start) is not int
      or type(end) is not int
      or start < 0
      or end <= start
  ):
    raise ValueError('Expected a nonempty logical position range [start, end).')


def _work():
  return dict(
      read_chunks=0,
      max_side_elements=0,
      compared_positions=0,
      slice_element_limit=MAX_SLICE_ELEMENTS,
  )


def _chunks(evidence, start, end, work):
  if evidence.status != 'ok' or set(evidence.sides) != {'ref', 'target'}:
    return
  sides = list(evidence.sides.values())
  first = max(start, *(side.start for side in sides))
  stop = min(end, *(side.end for side in sides))
  per_position = max(
      side.shape[0] * side.shape[1] * side.shape[3] for side in sides
  )
  if per_position > MAX_SLICE_ELEMENTS:
    raise ValueError(
        'One complete token exceeds the bounded slice element limit.'
    )
  chunk_positions = max(1, min(4096, MAX_SLICE_ELEMENTS // per_position))
  for left in range(first, stop, chunk_positions):
    right = min(stop, left + chunk_positions)
    values = evidence.read(left, right)
    if (
        set(values) != {'ref', 'target'}
        or values['ref'].shape != values['target'].shape
    ):
      side_errors = getattr(values, 'errors', {})
      raise ValueError(
          '; '.join(
              f'{role}: {error["reason"]}'
              for role, error in side_errors.items()
          )
          or (
              'Both sides require matching proven coordinates for numerical'
              ' comparison.'
          )
      )
    work['read_chunks'] += 1
    work['max_side_elements'] = max(
        work['max_side_elements'], *(value.size for value in values.values())
    )
    work['compared_positions'] += right - left
    yield left, values['ref'], values['target']


def _measure(reference, target, whole=False):
  """Vectorized equivalents of compare(); the unreduced axis is position."""
  x, y = reference.astype(np.float64), target.astype(np.float64)
  axes = (0, 1, 3) if whole else (0, 3)
  count = math.prod(x.shape[axis] for axis in axes)
  finite = np.all(np.isfinite(x) & np.isfinite(y), axis=axes)
  with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
    delta = x - y
    nx = np.sqrt(np.sum(x * x, axis=axes))
    ny = np.sqrt(np.sum(y * y, axis=axes))
    squared_errors = np.sum(delta * delta, axis=axes)
    maximum = np.max(np.abs(delta), axis=axes)
    norm_x = nx[None, None, :, None] if whole else nx[None, :, :, None]
    norm_y = ny[None, None, :, None] if whole else ny[None, :, :, None]
    cosine = np.sum((x / norm_x) * (y / norm_y), axis=axes)
    relative = np.sqrt(squared_errors) / nx
    rmse = np.sqrt(squared_errors / count)
  overflow = ~(
      np.isfinite(nx)
      & np.isfinite(ny)
      & np.isfinite(squared_errors)
      & np.isfinite(maximum)
  )
  invalid = ~finite | overflow
  status = np.where(
      ~finite, 'non_finite_tensor', np.where(overflow, 'numeric_overflow', 'ok')
  )
  metrics = dict(
      cosine_similarity=cosine,
      cosine_distance=1 - cosine,
      relative_l2=relative,
      rmse=rmse,
      max_abs=maximum,
  )
  return {
      key: np.where(invalid | ~np.isfinite(value), np.nan, value)
      for key, value in metrics.items()
  }, status


def _aggregate_heads(metrics, head):
  if head not in ('max', 'mean'):
    return {
        key: value[int(head)]
        for key, value in metrics.items()
        if key in CHART_METRICS
    }
  result = {}
  for key in CHART_METRICS:
    values = metrics[key]
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
      aggregate = (
          np.where(valid, values, 0).sum(axis=0) / count
          if head == 'mean'
          else np.max(np.where(valid, values, -np.inf), axis=0)
      )
    result[key] = np.where(count, aggregate, np.nan)
  return result


def _chart_stat():
  return dict(
      min=None, max=None, min_position=None, max_position=None, valid_count=0
  )


def _update_stat(stat, values, positions):
  valid = np.isfinite(values)
  if not np.any(valid):
    return
  values, positions = values[valid], positions[valid]
  stat['valid_count'] += int(values.size)
  for name, index in (
      ('min', int(np.argmin(values))),
      ('max', int(np.argmax(values))),
  ):
    value, position = float(values[index]), int(positions[index])
    previous = stat[name]
    if previous is None or (
        value < previous if name == 'min' else value > previous
    ):
      stat[name], stat[name + '_position'] = value, position


def _head(value):
  if value in ('max', 'mean'):
    return value
  if not str(value).isdigit():
    raise ValueError('Head must be max, mean or a nonnegative KV head index.')
  return int(value)


@_cached
def chart_range(
    store,
    turn,
    context_id,
    start,
    end,
    bins=128,
    kind='key',
    head='max',
    formula='',
):
  _bounds(start, end)
  if (
      type(bins) is not int
      or not 1 <= bins <= MAX_BINS
      or kind not in ('key', 'value')
  ):
    raise ValueError(f'Expected key/value and 1–{MAX_BINS} chart bins.')
  head = _head(head)
  predicate = compile_kv_formula(formula)
  reader, context = _context(store, turn, context_id)
  bins = min(bins, end - start)
  edges = [start + index * (end - start) // bins for index in range(bins + 1)]
  bounds = [
      dict(start=edges[index], end=edges[index + 1]) for index in range(bins)
  ]
  work, rows = _work(), []
  for source in context['layers']:
    if source['kind'] != kind:
      continue
    row = dict(
        layer=source['layer'],
        kind=kind,
        status='unavailable',
        bins=[
            dict(
                **interval,
                metrics={key: _chart_stat() for key in CHART_METRICS},
                match_count=0,
                compared_count=0,
            )
            for interval in bounds
        ],
    )
    rows.append(row)
    try:
      evidence = reader.resolve(turn, context_id, source['layer'], kind)
      row.update(status=evidence.status)
      if evidence.reason:
        row['reason'] = evidence.reason
      if evidence.status != 'ok':
        row.setdefault('reason', evidence.status.replace('_', ' '))
        continue
      if isinstance(head, int) and any(
          head >= side.shape[1] for side in evidence.sides.values()
      ):
        row.update(
            status='head_unavailable',
            reason='This KV head is outside the recorded tensor.',
        )
        continue
      for left, reference, target in _chunks(evidence, start, end, work):
        measured, statuses = _measure(reference, target)
        values = _aggregate_heads(measured, head)
        matches = (
            np.any(predicate(measured), axis=0)
            if formula.strip()
            else np.zeros(reference.shape[2], dtype=bool)
        )
        positions = np.arange(left, left + reference.shape[2])
        bin_ids = np.searchsorted(edges, positions, side='right') - 1
        # Positions and bin IDs are ordered; visit each contiguous span
        # once instead of rescanning the entire chunk for every bin.
        cuts = np.r_[0, np.flatnonzero(np.diff(bin_ids)) + 1, len(bin_ids)]
        for low, high in zip(cuts[:-1], cuts[1:]):
          segment = slice(low, high)
          cell = row['bins'][int(bin_ids[low])]
          cell['compared_count'] += int(high - low)
          cell['match_count'] += int(np.count_nonzero(matches[segment]))
          for metric in CHART_METRICS:
            _update_stat(
                cell['metrics'][metric],
                values[metric][segment],
                positions[segment],
            )
        if np.any(statuses != 'ok') or any(
            np.any(~np.isfinite(value)) for value in values.values()
        ):
          row['status'] = 'partial'
          row['reason'] = (
              'Some metrics are undefined (zero norm, non-finite values or'
              ' numeric overflow); finite counts are reported per metric.'
          )
      if not any(cell['compared_count'] for cell in row['bins']):
        row.update(
            status='position_unavailable',
            reason=(
                'This layer/K/V has no comparable positions in the requested'
                ' range.'
            ),
        )
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
      row.update(
          status='partial'
          if any(cell['compared_count'] for cell in row['bins'])
          else 'unavailable',
          reason=str(error),
      )
  return dict(
      context_id=context_id,
      start=start,
      end=end,
      bins=bounds,
      rows=rows,
      head=head,
      kind=kind,
      work=work,
      formula=formula,
      aggregation=(
          'Per-position display-head values; each bin preserves its finite'
          ' minimum and maximum and original positions.'
      ),
  )


@_cached
def selection_range(store, turn, context_id, start, end, layer=None):
  _bounds(start, end)
  if layer is not None and (type(layer) is not int or layer < 0):
    raise ValueError('Expected a nonnegative layer.')
  reader, context = _context(store, turn, context_id)
  work, rows = _work(), []
  for source in context['layers']:
    if layer is not None and source['layer'] != layer:
      continue
    row = dict(
        layer=source['layer'],
        kind=source['kind'],
        status='unavailable',
        compared_count=0,
        metrics={
            key: dict(
                value=None, position=None, valid_count=0, status='unavailable'
            )
            for key in TOKEN_METRICS
        },
    )
    rows.append(row)
    extrema = {key: _chart_stat() for key in TOKEN_METRICS}
    try:
      evidence = reader.resolve(
          turn, context_id, source['layer'], source['kind']
      )
      row['status'] = evidence.status
      if evidence.reason:
        row['reason'] = evidence.reason
      if evidence.status != 'ok':
        row.setdefault('reason', evidence.status.replace('_', ' '))
        for value in row['metrics'].values():
          value['status'] = evidence.status
        continue
      for left, reference, target in _chunks(evidence, start, end, work):
        row['compared_count'] += reference.shape[2]
        measured, statuses = _measure(reference, target, whole=True)
        positions = np.arange(left, left + reference.shape[2])
        for key in TOKEN_METRICS:
          _update_stat(extrema[key], measured[key], positions)
        if np.any(statuses != 'ok') or any(
            np.any(~np.isfinite(measured[key])) for key in TOKEN_METRICS
        ):
          row['status'] = 'partial'
          row['reason'] = (
              'Some token metrics are undefined; only finite values contribute'
              ' to each range summary.'
          )
        if end - start == 1:
          values = {
              key: (
                  float(measured[key][0])
                  if np.isfinite(measured[key][0])
                  else None
              )
              for key in TOKEN_METRICS
          }
          metric_status = {
              key: (
                  'ok'
                  if values[key] is not None
                  else (
                      str(statuses[0])
                      if statuses[0] != 'ok'
                      else 'undefined_zero_norm'
                  )
              )
              for key in TOKEN_METRICS
          }
          row['token_metrics'] = [
              dict(
                  position=left,
                  status='ok'
                  if all(value == 'ok' for value in metric_status.values())
                  else 'partial',
                  metrics=values,
                  metric_status=metric_status,
              )
          ]
      if row['compared_count'] == 0:
        row.update(
            status='position_unavailable',
            reason=(
                'This layer/K/V has no comparable positions in the requested'
                ' range.'
            ),
        )
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
      row.update(
          status='partial'
          if any(value['valid_count'] for value in extrema.values())
          else 'unavailable',
          reason=str(error),
      )
    for key, stat in extrema.items():
      side = 'min' if key == 'cosine_similarity' else 'max'
      unavailable_status = (
          row['token_metrics'][0]['metric_status'][key]
          if row.get('token_metrics')
          else row['status']
          if row['status'] != 'ok'
          else 'undefined_or_not_captured'
      )
      row['metrics'][key] = dict(
          value=stat[side],
          position=stat[side + '_position'],
          valid_count=stat['valid_count'],
          status='ok' if stat['valid_count'] else unavailable_status,
      )
  return dict(
      context_id=context_id,
      start=start,
      end=end,
      rows=rows,
      work=work,
      aggregation=TOKEN_AGGREGATION,
      range_aggregation=(
          'Worst finite token metric within each layer and K/V: minimum CosSim,'
          ' maximum errors. Layers remain independent.'
      ),
  )


def _find_sources(store, turn, context_id):
  reader, context = _context(store, turn, context_id)
  available, unavailable = [], []
  for row in context['layers']:
    try:
      evidence = reader.resolve(turn, context_id, row['layer'], row['kind'])
      if evidence.status != 'ok':
        unavailable.append(
            dict(
                layer=row['layer'],
                kind=row['kind'],
                status=evidence.status,
                reason=evidence.reason,
            )
        )
      else:
        available.append((row, evidence))
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
      unavailable.append(
          dict(
              layer=row['layer'],
              kind=row['kind'],
              status='unavailable',
              reason=str(error),
          )
      )
  return available, unavailable


@_cached
def _find_index(store, turn, context_id, formula):
  """Keep counts only; dense queries never create a dict for every match."""
  predicate = compile_kv_formula(formula)
  available, unavailable = _find_sources(store, turn, context_id)
  work, blocks, total, cacheable = _work(), [], 0, True
  starts = [
      max(side.start for side in evidence.sides.values())
      for _, evidence in available
  ]
  ends = [
      min(side.end for side in evidence.sides.values())
      for _, evidence in available
  ]
  if available:
    block_size = max(1, min(256, MAX_FIND_BLOCK_RECORDS // len(available)))
    if (
        math.ceil((max(ends) - min(starts)) / block_size)
        > MAX_FIND_INDEX_BLOCKS
    ):
      raise ValueError('The global Find index exceeds its bounded block limit.')
    for left in range(min(starts), max(ends), block_size):
      stop, count = min(max(ends), left + block_size), 0
      for row, evidence in available:
        try:
          for _, reference, target in _chunks(evidence, left, stop, work):
            measured, _ = _measure(reference, target)
            matching = predicate(measured)
            count += int(np.count_nonzero(np.any(matching, axis=0)))
        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            RuntimeError,
        ) as error:
          cacheable = False
          failure = dict(
              layer=row['layer'],
              kind=row['kind'],
              status='unavailable',
              reason=str(error),
          )
          if failure not in unavailable:
            unavailable.append(failure)
      blocks.append(dict(start=left, end=stop, offset=total, count=count))
      total += count
  return dict(
      total=total,
      blocks=blocks,
      unavailable_rows=unavailable,
      work=work,
      _cacheable=cacheable,
  )


@_cached
def find_matches(store, turn, context_id, formula, offset=0, limit=40):
  if (
      type(offset) is not int
      or offset < 0
      or type(limit) is not int
      or not 1 <= limit <= MAX_FIND_PAGE
  ):
    raise ValueError(
        'Expected a nonnegative offset and a Find page size of 1–40.'
    )
  predicate = compile_kv_formula(formula)
  index = _find_index(store, turn, context_id, formula)
  available, unavailable = _find_sources(store, turn, context_id)
  work, results = _work(), []
  failures = [
      *index['unavailable_rows'],
      *[row for row in unavailable if row not in index['unavailable_rows']],
  ]
  materialized = 0
  for interval in index['blocks']:
    if (
        interval['count'] == 0
        or interval['offset'] + interval['count'] <= offset
        or interval['offset'] >= offset + limit
    ):
      continue
    block = []
    for row, evidence in available:
      try:
        for first, reference, target in _chunks(
            evidence, interval['start'], interval['end'], work
        ):
          measured, _ = _measure(reference, target)
          matching = predicate(measured)
          for position_index in np.flatnonzero(np.any(matching, axis=0)):
            heads = np.flatnonzero(matching[:, position_index])
            values = _aggregate_heads(
                {
                    key: value[heads, position_index : position_index + 1]
                    for key, value in measured.items()
                },
                'max',
            )
            block.append(
                dict(
                    layer=row['layer'],
                    kind=row['kind'],
                    position=first + int(position_index),
                    heads=heads.tolist(),
                    metrics={
                        key: float(value[0]) if np.isfinite(value[0]) else None
                        for key, value in values.items()
                    },
                )
            )
      except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        failure = dict(
            layer=row['layer'],
            kind=row['kind'],
            status='unavailable',
            reason=str(error),
        )
        if failure not in failures:
          failures.append(failure)
    block.sort(key=lambda row: (row['position'], row['layer'], row['kind']))
    materialized += len(block)
    low, high = max(0, offset - interval['offset']), min(
        len(block), offset + limit - interval['offset']
    )
    results.extend(block[low:high])
  work.update(
      read_chunks=work['read_chunks'] + index['work']['read_chunks'],
      max_side_elements=max(
          work['max_side_elements'], index['work']['max_side_elements']
      ),
      compared_positions=work['compared_positions']
      + index['work']['compared_positions'],
      index_cache_hit=index.get('cache_hit', False),
      materialized_matches=materialized,
      index_blocks=len(index['blocks']),
  )
  return dict(
      context_id=context_id,
      formula=formula,
      total=index['total'],
      offset=offset,
      limit=limit,
      results=results,
      has_more=offset + len(results) < index['total'],
      complete=not failures,
      unavailable_rows=failures,
      work=work,
      _cacheable=index.get('_cacheable', True),
  )


def _common(query):
  return (
      int(query['turn'][0]),
      query.get('context', query.get('context_id', ['']))[0],
  )


def range_query(store, query):
  return chart_range(
      store,
      *_common(query),
      int(query['start'][0]),
      int(query['end'][0]),
      int(query.get('bins', ['128'])[0]),
      query.get('kind', ['key'])[0],
      query.get('head', ['max'])[0],
      query.get('formula', [''])[0],
  )


def selection_query(store, query):
  return selection_range(
      store,
      *_common(query),
      int(query['start'][0]),
      int(query['end'][0]),
      int(query['layer'][0]) if 'layer' in query else None,
  )


def find_query(store, query):
  return find_matches(
      store,
      *_common(query),
      query.get('formula', [''])[0],
      int(query.get('offset', ['0'])[0]),
      int(query.get('limit', ['40'])[0]),
  )
