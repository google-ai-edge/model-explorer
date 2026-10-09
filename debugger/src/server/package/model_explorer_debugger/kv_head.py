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

"""One exact cache position and KV head: channel values and source evidence."""

from copy import deepcopy
import math

import numpy as np

from .kv_reader import KvReader
from .metrics import compare

HEAD_METRICS = {
    'cosine_similarity': 'CosSim',
    'cosine_distance': 'CosSim',
    'relative_l2': 'Relative L2',
    'mean_abs': 'Mean abs error',
    'rmse': 'RMSE',
    'max_abs': 'Max abs error',
}
MAX_CHANNELS = 16_384


def _empty(selection):
  return dict(
      selection=selection,
      status='unavailable',
      batch_count=None,
      channel_count=None,
      shape=None,
      metrics={key: None for key in HEAD_METRICS},
      metric_status={key: 'unavailable' for key in HEAD_METRICS},
      channels=[],
      largest_channel=None,
      sources={'ref': None, 'target': None},
      calculation_dtype='float64',
      delta_definition='Target - Reference',
  )


def _failure(result, status, reason):
  result.update(status=status, reason=reason)
  result['metric_status'] = {key: status for key in HEAD_METRICS}
  return result


def _scalar(value):
  item = value.item()
  if isinstance(item, float) and not math.isfinite(item):
    return str(item)
  if isinstance(item, int) and abs(item) > 2**53 - 1:
    return str(item)
  return item


def inspect_kv_head(
    store, turn, context_id, layer, kind, position, head, batch=None
):
  if (
      type(turn) is not int
      or turn < 1
      or not any(row['n'] == turn for row in store.session['turns'])
      or not isinstance(context_id, str)
      or not context_id
      or kind not in ('key', 'value')
      or any(
          type(value) is not int or value < 0
          for value in (layer, position, head)
      )
      or (batch is not None and (type(batch) is not int or batch < 0))
  ):
    raise ValueError(
        'Expected a captured Turn, context, layer, key/value, position, head'
        ' and optional nonnegative batch.'
    )
  selection = dict(
      turn=turn,
      context_id=context_id,
      layer=layer,
      kind=kind,
      position=position,
      head=head,
      batch=batch,
  )
  result = _empty(selection)
  if not context_id.startswith(('explicit-', 'snapshot-')):
    raise ValueError('Unknown KV context identity')
  evidence = KvReader(store).resolve(turn, context_id, layer, kind)
  result['comparison_basis'] = evidence.comparison_basis
  result['sources'].update(deepcopy(evidence.sources))
  status, reason = evidence.status, evidence.reason
  if not evidence.sides:
    return _failure(
        result,
        status,
        reason or 'No stored tensor is linked to this selection.',
    )
  geometries = evidence.sides
  for role, geometry in geometries.items():
    result['sources'][role]['comparison_shape'] = [geometry.shape[3]]
  result['batch_count'] = max(
      geometry.shape[0] for geometry in geometries.values()
  )
  result['channel_count'] = max(
      geometry.shape[3] for geometry in geometries.values()
  )
  result['shape'] = [result['channel_count']]
  if result['channel_count'] > MAX_CHANNELS:
    return _failure(
        result,
        'analysis_limit',
        'Selected head exceeds the channel evidence budget.',
    )
  if batch is None and result['batch_count'] > 1:
    return _failure(
        result,
        'batch_selection_required',
        'Select a batch explicitly to inspect this position and head; batch'
        ' values are not combined.',
    )
  batch = 0 if batch is None else batch
  selection['batch'] = batch
  if not any(
      batch < geometry.shape[0] and head < geometry.shape[1]
      for geometry in geometries.values()
  ):
    raise ValueError(
        'Selected batch or KV head is outside the recorded tensor.'
    )
  if not any(
      batch < geometry.shape[0]
      and head < geometry.shape[1]
      and geometry.start <= position < geometry.end
      for geometry in geometries.values()
  ):
    ranges = ', '.join(
        f'{role}: [{geometry.start}, {geometry.end})'
        for role, geometry in geometries.items()
    )
    return _failure(
        result,
        'position_unavailable',
        f'Cache position {position} was not captured in this layer/K/V.'
        f' Recorded ranges: {ranges}.',
    )
  sliced = evidence.read(position, position + 1, batch=batch, head=head)
  vectors = {role: value[0, 0, 0, :] for role, value in sliced.items()}
  errors = [
      role + ': ' + error['reason'] for role, error in sliced.errors.items()
  ]
  if sliced.errors and not vectors:
    status = next(iter(sliced.errors.values()))['status']
  comparable = (
      status == 'ok'
      and len(vectors) == 2
      and vectors['ref'].shape == vectors['target'].shape
  )
  if comparable:
    try:
      measured = compare(vectors['ref'], vectors['target'])
      for key, name in HEAD_METRICS.items():
        value = measured[name]['value']
        result['metrics'][key] = (
            1 - value
            if key == 'cosine_distance' and value is not None
            else value
        )
        result['metric_status'][key] = measured[name]['status']
      result['status'] = (
          'ok'
          if all(value == 'ok' for value in result['metric_status'].values())
          else 'partial'
      )
    except (ValueError, TypeError, FloatingPointError, OverflowError) as error:
      _failure(
          result,
          'partial',
          'Selected head metrics are unavailable: ' + str(error),
      )
      metric_status = (
          'numeric_overflow'
          if isinstance(error, (FloatingPointError, OverflowError))
          else str(error)
      )
      result['metric_status'] = {key: metric_status for key in HEAD_METRICS}
  else:
    detail = (
        '; '.join(filter(None, [reason, *errors]))
        or 'A vector is unavailable on one side.'
    )
    _failure(result, status if status != 'ok' else 'unavailable', detail)
  if not vectors:
    return result
  for channel in range(max(len(vector) for vector in vectors.values())):
    raw = {
        role: vector[channel]
        for role, vector in vectors.items()
        if channel < len(vector)
    }
    values = {role: _scalar(value) for role, value in raw.items()}
    delta, absolute = None, None
    channel_status = result['status'] if not comparable else 'non_finite'
    if comparable and all(np.isfinite(value) for value in raw.values()):
      with np.errstate(over='ignore', invalid='ignore'):
        candidate = float(np.float64(raw['target']) - np.float64(raw['ref']))
      if math.isfinite(candidate):
        delta, absolute, channel_status = candidate, abs(candidate), 'ok'
    result['channels'].append(
        dict(
            channel=channel,
            ref=values.get('ref'),
            target=values.get('target'),
            delta=delta,
            abs_delta=absolute,
            status=channel_status,
        )
    )
  finite_channels = [
      row for row in result['channels'] if row['abs_delta'] is not None
  ]
  if finite_channels:
    result['largest_channel'] = max(
        finite_channels, key=lambda row: row['abs_delta']
    )['channel']
  return result


def kv_head_query(store, query):
  return inspect_kv_head(
      store,
      int(query['turn'][0]),
      query['context_id'][0],
      int(query['layer'][0]),
      query['kind'][0],
      int(query['position'][0]),
      int(query['head'][0]),
      int(query['batch'][0]) if 'batch' in query else None,
  )
