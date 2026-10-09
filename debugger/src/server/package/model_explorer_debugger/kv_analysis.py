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

"""Position/head metrics from proven KV correspondence; no inferred layouts."""

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math

from .kv_storage import comparison_view, storage_view
from .metrics import compare

LAYOUT = ['batch', 'kv_head', 'sequence', 'head_dim']
MAX_CELLS = 100_000
MAX_ELEMENTS = 32_000_000
MAX_EVIDENCE_ELEMENTS = 128_000_000
AGGREGATION = (
    'Across batch and head dimension at each logical position and KV head.'
)
TOKEN_AGGREGATION = (
    'Direct comparison across all stored batch, KV head and channel elements at'
    ' one logical position.'
)
METRICS = {
    'relative_l2': 'Relative L2',
    'max_abs': 'Max abs error',
    'cosine_distance': 'CosSim',
}
TOKEN_METRICS = {
    'cosine_similarity': 'CosSim',
    'relative_l2': 'Relative L2',
    'rmse': 'RMSE',
    'max_abs': 'Max abs error',
}


def _identity(value):
  return hashlib.sha256(
      json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
  ).hexdigest()[:24]


def _row(layer, kind, **evidence):
  return dict(
      layer=layer,
      kind=kind,
      status='unavailable',
      head_count=None,
      position_start=None,
      position_count=None,
      cells=[],
      token_metrics=[],
      token_structure={'ref': None, 'target': None},
      **evidence,
  )


def _token_structure(shape, layout, start, count, dtype, valid_length=None):
  """Describe one batch's token geometry, independently of pair eligibility."""
  result = dict(
      status='unavailable',
      shape=None,
      head_count=None,
      channel_count=None,
      batch_count=None,
      position_start=None,
      position_count=None,
      dtype=dtype if isinstance(dtype, str) else None,
  )
  if (
      not isinstance(layout, list)
      or len(layout) != 4
      or any(not isinstance(axis, str) for axis in layout)
      or set(layout) != set(LAYOUT)
      or not isinstance(shape, list)
      or len(shape) != 4
      or any(type(size) is not int or size <= 0 for size in shape)
  ):
    return _reject(
        result,
        'layout_unavailable',
        'A recorded batch, KV head, sequence and head dimension layout is'
        ' required.',
    )
  canonical = [shape[layout.index(axis)] for axis in LAYOUT]
  if (
      type(start) is not int
      or start < 0
      or type(count) is not int
      or count <= 0
      or canonical[2] != count
  ):
    return _reject(
        result,
        'logical_mapping_unavailable',
        'The sequence axis must match the recorded logical range before token'
        ' dimensions are assigned.',
    )
  if valid_length is not None and valid_length != count:
    return _reject(
        result,
        'logical_range_mismatch',
        'The recorded valid length differs from the logical range.',
    )
  result.update(
      status='ok',
      shape=[canonical[1], canonical[3]],
      head_count=canonical[1],
      channel_count=canonical[3],
      batch_count=canonical[0],
      position_start=start,
      position_count=count,
  )
  return result


def _resource_structure(record):
  start, end = record.get('logical_start'), record.get('logical_end')
  count = end - start if type(start) is int and type(end) is int else None
  try:
    _, shape = storage_view(record)
  except ValueError as error:
    result = _token_structure(
        record.get('shape'),
        record.get('layout'),
        start,
        count,
        record.get('dtype'),
        record.get('valid_length'),
    )
    return _reject(
        result,
        str(error),
        getattr(error, 'reason', 'The recorded KV storage view is invalid.'),
    )
  if (
      count == 0
      and record.get('valid_length') == 0
      and shape[record['layout'].index('sequence')] == 0
  ):
    canonical = [shape[record['layout'].index(axis)] for axis in LAYOUT]
    return dict(
        status='empty',
        shape=[canonical[1], canonical[3]],
        head_count=canonical[1],
        channel_count=canonical[3],
        batch_count=canonical[0],
        position_start=start,
        position_count=0,
        dtype=record.get('dtype'),
    )
  return _token_structure(
      shape,
      record.get('layout'),
      start,
      count,
      record.get('dtype'),
      record.get('valid_length'),
  )


def _snapshot_coordinates(snapshot):
  result = {
      key: snapshot.get(key)
      for key in ('turn', 'phase', 'step', 'forward_id', 'moment')
  }
  if snapshot.get('runtime') == 'LiteRT-LM' and snapshot.get(
      'logical_context_identity'
  ):
    result.update(
        step=None,
        forward_id=None,
        logical_context_identity=snapshot['logical_context_identity'],
        processed_token_count=snapshot.get('processed_token_count'),
    )
    if isinstance(snapshot.get('basis'), str):
      # An inferred context is its own kind of observation; it never pairs with
      # a recorded one.
      result.update(
          basis=snapshot['basis'], context_basis=snapshot.get('context_basis')
      )
  return result


def _metric_values(values):
  metrics = {
      key: values.get(name, {}).get('value') for key, name in METRICS.items()
  }
  if metrics['cosine_distance'] is not None:
    metrics['cosine_distance'] = 1 - metrics['cosine_distance']
  return {
      key: (
          value
          if isinstance(value, (int, float)) and math.isfinite(value)
          else None
      )
      for key, value in metrics.items()
  }


def _reject(row, status, reason):
  row.update(status=status, reason=reason)
  return row


def _dimensions(row, shape, layout, start, count, budget):
  """Require a complete logical sequence, never guess a ring-buffer mapping."""
  if (
      not isinstance(layout, list)
      or len(layout) != 4
      or any(not isinstance(axis, str) for axis in layout)
      or set(layout) != set(LAYOUT)
      or not isinstance(shape, list)
      or len(shape) != 4
      or any(type(size) is not int or size <= 0 for size in shape)
  ):
    _reject(
        row,
        'layout_unavailable',
        'A recorded batch, KV head, sequence and head dimension layout is'
        ' required.',
    )
    return None
  canonical = [shape[layout.index(axis)] for axis in LAYOUT]
  row.update(
      shape=canonical,
      canonical_layout=LAYOUT.copy(),
      head_count=canonical[1],
      position_start=start,
      position_count=count,
  )
  if (
      type(start) is not int
      or start < 0
      or type(count) is not int
      or count <= 0
      or canonical[2] != count
  ):
    _reject(
        row,
        'logical_mapping_unavailable',
        'The recorded sequence axis must exactly match the valid logical range;'
        ' capacity and ring-buffer positions are not inferred.',
    )
    return None
  cells, elements = canonical[1] * canonical[2], math.prod(canonical) * 2
  if cells > budget['cells'] or elements > budget['elements']:
    _reject(
        row,
        'analysis_limit',
        'This tensor exceeds the remaining analysis budget. Raw evidence is'
        ' retained; no positions were sampled or truncated.',
    )
    return None
  budget['cells'] -= cells
  budget['elements'] -= elements
  return [layout.index(axis) for axis in LAYOUT]


def _cells(row, reference, target, order):
  if reference.shape != target.shape:
    return _reject(row, 'shape_mismatch', 'The compared tensor shapes differ.')
  x, y = reference.transpose(order), target.transpose(order)
  if list(x.shape) != row['shape']:
    return _reject(
        row,
        'shape_mismatch',
        'The loaded tensors differ from the recorded dimensions.',
    )
  cells, tokens, partial = [], [], False
  for position in range(row['position_count']):
    logical_position = row['position_start'] + position
    try:
      values = compare(x[:, :, position, :], y[:, :, position, :])
      token = dict(
          position=logical_position,
          metrics={
              key: values[name]['value'] for key, name in TOKEN_METRICS.items()
          },
          metric_status={
              key: values[name]['status'] for key, name in TOKEN_METRICS.items()
          },
      )
      token['status'] = (
          'ok'
          if all(value == 'ok' for value in token['metric_status'].values())
          else 'partial'
      )
    except (ValueError, TypeError, FloatingPointError, OverflowError) as error:
      status = (
          str(error) if isinstance(error, ValueError) else 'numeric_overflow'
      )
      token = dict(
          position=logical_position,
          status=status,
          metrics={key: None for key in TOKEN_METRICS},
          metric_status={key: status for key in TOKEN_METRICS},
      )
    tokens.append(token)
    partial = partial or token['status'] != 'ok'
    for head in range(row['head_count']):
      try:
        values = compare(x[:, head, position, :], y[:, head, position, :])
        metrics = _metric_values(values)
        status = (
            'ok'
            if all(value is not None for value in metrics.values())
            else 'undefined_zero_norm'
        )
      except (
          ValueError,
          TypeError,
          FloatingPointError,
          OverflowError,
      ) as error:
        metrics = {key: None for key in METRICS}
        status = (
            str(error)
            if isinstance(error, ValueError)
            else type(error).__name__
        )
      partial = partial or status != 'ok'
      cells.append(
          dict(
              position=logical_position,
              head=head,
              metrics=metrics,
              status=status,
          )
      )
  row.update(
      status='partial' if partial else 'ok',
      cells=cells,
      token_metrics=tokens,
      aggregation=AGGREGATION,
      token_aggregation=TOKEN_AGGREGATION,
  )
  return row


def _context(source, coordinates, **extra):
  moment = coordinates.get('moment')
  phase = coordinates.get('phase')
  moment_label = {
      'prefill_pre': 'Before Prefill',
      'prefill_post': 'After Prefill',
      'terminal': 'Generation end',
  }.get(moment, moment or 'Stored cache')
  step = coordinates.get('step')
  runtime = extra.get('runtime')
  label = f'Turn {coordinates["turn"]} · {moment_label}'
  if runtime:
    label += f' · {runtime}'
  if step is not None:
    label += f' · step {step}'
  if coordinates.get('processed_token_count') is not None:
    label += f" · {coordinates['processed_token_count']} processed tokens"
  return dict(
      id=source + '-' + _identity([coordinates, extra]),
      label=label,
      turn=coordinates['turn'],
      phase=phase,
      step=step,
      forward_id=coordinates.get('forward_id'),
      moment=moment,
      source=source,
      layers=[],
      snapshots={'ref': [], 'target': []},
      **{
          key: coordinates[key]
          for key in (
              'processed_token_count',
              'logical_context_identity',
              'basis',
              'context_basis',
          )
          if key in coordinates
      },
      **extra,
  )


def _capture_snapshot(snapshot, runtime=None, observation=None):
  """Keep a run's recorded identity.

  An observation hash is not a snapshot ID.
  """
  if not (
      isinstance(snapshot.get('id'), str)
      or type(snapshot.get('snapshot_id')) is int
      or (
          isinstance(snapshot.get('signature'), str)
          and type(snapshot.get('step')) is int
          and isinstance(snapshot.get('edge'), str)
      )
  ):
    return None
  coordinates = {**(observation or {}), **snapshot}
  return {
      key: coordinates.get(key)
      for key in (
          'id',
          'snapshot_id',
          'turn',
          'phase',
          'step',
          'forward_id',
          'moment',
          'signature',
          'edge',
      )
  } | {'runtime': snapshot.get('runtime', runtime)}


def _append_snapshot(context, role, snapshot):
  if snapshot is not None and snapshot not in context['snapshots'][role]:
    context['snapshots'][role].append(snapshot)


def _evidence_size(value):
  """Conservatively count declared tensors before numerical proof validation."""
  if isinstance(value, list):
    return sum(_evidence_size(item) for item in value)
  if not isinstance(value, dict):
    return 0
  shape = value.get('shape')
  own = (
      math.prod(shape)
      if (
          isinstance(shape, list)
          and shape
          and all(type(size) is int and size >= 0 for size in shape)
      )
      else 0
  )
  return own + sum(
      _evidence_size(item) for key, item in value.items() if key != 'shape'
  )


def _explicit_contexts(store, turn, budget, notices):
  contexts, reports = {}, defaultdict(list)
  unavailable = []
  for entry in store.explicit_pair_entries():
    if entry.get('error'):
      notices.append(
          'A saved pair declaration could not be read: ' + entry['error']
      )
      continue
    manifest = entry['manifest']
    if (
        manifest.get('kind') != 'explicit_cross_runtime_terminal_kv'
        or manifest.get('observation', {}).get('turn') != turn
    ):
      continue
    size = _evidence_size(manifest)
    failure = None
    if size > budget['evidence_elements']:
      failure = (
          'analysis_limit',
          (
              'The recorded evidence exceeds the remaining validation budget;'
              ' no tensor values were loaded.'
          ),
      )
    else:
      budget['evidence_elements'] -= size
      try:
        pair = store.compare_pair(entry['pair_id'])
      except (KeyError, ValueError, TypeError, OSError, RuntimeError) as error:
        failure = (
            'unavailable',
            'Saved comparison evidence could not be validated: ' + str(error),
        )
    if failure:
      failed_context = _context(
          'explicit',
          {'turn': turn, 'phase': None, 'moment': None},
          runtime=None,
          pair_id=entry['pair_id'],
      )
      layer, kind = manifest.get('owner_layer'), manifest.get('kind_of_tensor')
      failed_context['label'] = (
          f'Turn {turn} · Layer {layer} {kind} · Saved comparison unavailable'
      )
      if type(layer) is int and layer >= 0 and kind in ('key', 'value'):
        failed_context['layers'].append(
            _reject(_row(layer, kind, pair_id=entry['pair_id']), *failure)
        )
      unavailable.append(failed_context)
      notices.append(failure[1])
      continue
    observations = {
        role: deepcopy(run.get('observation', pair['observation']))
        for role, run in pair['runs'].items()
    }
    coordinates = {**pair['observation'], 'moment': 'terminal'}
    key = _identity([coordinates, observations])
    if key not in contexts:
      contexts[key] = _context(
          'explicit', coordinates, runtime=None, observations=observations
      )
      contexts[key]['label'] += ' · Explicit comparison'
    for role in ('ref', 'target'):
      declaration = manifest.get('runs', {}).get(role, {})
      identity = (
          {'snapshot_id': declaration.get('snapshot_id')}
          if role == 'ref'
          else declaration.get('snapshot', {})
      )
      _append_snapshot(
          contexts[key],
          role,
          _capture_snapshot(
              identity,
              pair.get('runs', {}).get(role, {}).get('runtime'),
              observations.get(role),
          ),
      )
    reports[key, pair['owner_layer'], pair['kind']].append(pair)
  for (key, layer, kind), pairs in reports.items():
    if len(pairs) != 1:
      contexts[key]['layers'].append(
          _reject(
              _row(layer, kind, pair_ids=[p['pair_id'] for p in pairs]),
              'ambiguous_observation',
              'Multiple saved pairs claim this layer, K/V and observation; no'
              ' pair was selected.',
          )
      )
      continue
    pair = pairs[0]
    row = _row(
        layer,
        kind,
        pair_id=pair['pair_id'],
        comparison_basis=pair.get('comparison_basis'),
        weight_equivalence=pair.get('weight_equivalence'),
        metrics=_metric_values(pair.get('metrics', {})),
        summary_metrics=deepcopy(pair.get('metrics', {})),
    )
    contexts[key]['layers'].append(row)
    position_range = pair.get('compared_position_range')
    start = (
        position_range[0]
        if isinstance(position_range, list) and len(position_range) == 2
        else None
    )
    count = (
        position_range[1] - start
        if type(start) is int and type(position_range[1]) is int
        else None
    )
    for role in ('ref', 'target'):
      run = pair.get('runs', {}).get(role, {})
      row['token_structure'][role] = _token_structure(
          pair.get('shape'),
          pair.get('canonical_layout'),
          start,
          count,
          run.get('original_tensor', {}).get('dtype'),
      )
    order = _dimensions(
        row,
        pair.get('shape'),
        pair.get('canonical_layout'),
        start,
        count,
        budget,
    )
    if order is not None:
      try:
        _cells(
            row,
            store._load_validated_pair_tensor(pair, 'ref', 'comparison'),
            store._load_validated_pair_tensor(pair, 'target', 'comparison'),
            order,
        )
      except (KeyError, ValueError, TypeError, OSError, RuntimeError) as error:
        _reject(row, 'unavailable', str(error))
  return list(contexts.values()) + unavailable


def _snapshot_contexts(store, turn, budget):
  data = store.telemetry(turn=turn)
  resources = {row['id']: row for row in data['resources']}
  groups = defaultdict(list)
  for snapshot in data['kv_snapshots']:
    coordinates = {
        **_snapshot_coordinates(snapshot),
        'runtime': snapshot.get('runtime'),
    }
    groups[json.dumps(coordinates, sort_keys=True)].append(snapshot)
  contexts = []
  for encoded, snapshots in groups.items():
    coordinates = json.loads(encoded)
    runtime = coordinates.pop('runtime')
    context = _context('snapshot', coordinates, runtime=runtime)
    contexts.append(context)
    entries = defaultdict(lambda: defaultdict(list))
    for snapshot in snapshots:
      if snapshot.get('run') not in ('ref', 'target'):
        continue
      _append_snapshot(context, snapshot['run'], _capture_snapshot(snapshot))
      for layer in snapshot.get('layers', []):
        for tensor in layer.get('tensors', []):
          if tensor.get('kind') in ('key', 'value'):
            entries[layer['layer'], tensor['kind']][snapshot['run']].append(
                (layer, tensor)
            )
    for (layer, kind), sides in entries.items():
      row = _row(layer, kind)
      context['layers'].append(row)
      for role, field in (('ref', 'reference'), ('target', 'target')):
        if len(sides[role]) == 1 and sides[role][0][1].get('resource_id'):
          row[field] = sides[role][0][1]['resource_id']
          resource = resources.get(row[field])
          if resource is not None:
            row['token_structure'][role] = _resource_structure(resource)
      if any(len(sides[role]) > 1 for role in ('ref', 'target')):
        _reject(
            row,
            'ambiguous_observation',
            'Multiple stored tensors claim this observation; no pair was'
            ' selected.',
        )
        continue
      if not row.get('reference') or not row.get('target'):
        _reject(
            row,
            'unavailable',
            'A stored tensor is required on both sides at this exact'
            ' observation.',
        )
        continue
      reference, target = resources.get(row['reference']), resources.get(
          row['target']
      )
      if reference is None or target is None:
        _reject(row, 'unavailable', 'A stored resource index is missing.')
        continue
      try:
        _, ref_shape = storage_view(reference)
        _, target_shape = storage_view(target)
      except ValueError as error:
        _reject(
            row,
            str(error),
            getattr(
                error, 'reason', 'The recorded KV storage view is invalid.'
            ),
        )
        continue
      if ref_shape != target_shape:
        _reject(
            row, 'shape_mismatch', 'The complete stored tensor shapes differ.'
        )
        continue
      # Check budget and geometry before loading or comparing large resources.
      start, end = reference.get('logical_start'), reference.get('logical_end')
      count = end - start if type(start) is int and type(end) is int else None
      if (
          count == 0
          and reference.get('valid_length') == target.get('valid_length') == 0
      ):
        _reject(row, 'empty', 'This snapshot contains no processed tokens.')
        continue
      order = _dimensions(
          row, ref_shape, reference.get('layout'), start, count, budget
      )
      if order is None:
        continue
      result = store.compare_resources(row['reference'], row['target'])
      row['comparison_basis'] = result.get('comparison_basis')
      row['metrics'] = _metric_values(result.get('metrics', {}))
      row['summary_metrics'] = deepcopy(result.get('metrics', {}))
      if result['status'] not in (
          'ok',
          'non_finite_tensor',
          'numeric_overflow',
      ):
        _reject(
            row,
            result['status'],
            'Stored tensors have no proven compatible comparison: '
            + result['status'],
        )
        continue
      if (
          reference.get('valid_length') != count
          or target.get('valid_length') != count
      ):
        _reject(
            row,
            'logical_range_mismatch',
            'The complete logical range must match the recorded valid length.',
        )
        continue
      try:
        _cells(
            row,
            comparison_view(store.load_resource(row['reference']), reference),
            comparison_view(store.load_resource(row['target']), target),
            order,
        )
      except (KeyError, ValueError, TypeError, OSError, RuntimeError) as error:
        _reject(row, 'unavailable', str(error))
  return contexts


def analyze_kv(store, turn):
  """Return exact cells or explicit unavailability.

  Never samples or invents evidence.
  """
  if type(turn) is not int or turn < 1:
    raise ValueError('Expected a positive Turn number')
  if not any(row['n'] == turn for row in store.session['turns']):
    raise ValueError('Unknown Turn')
  budget = dict(
      cells=MAX_CELLS,
      elements=MAX_ELEMENTS,
      evidence_elements=MAX_EVIDENCE_ELEMENTS,
  )
  notices = []
  contexts = _explicit_contexts(
      store, turn, budget, notices
  ) + _snapshot_contexts(store, turn, budget)
  for context in contexts:
    context['layers'].sort(key=lambda row: (row['layer'], row['kind'] != 'key'))
  return dict(
      contexts=contexts,
      aggregation=AGGREGATION,
      token_aggregation=TOKEN_AGGREGATION,
      limits=dict(
          max_cells=MAX_CELLS,
          max_token_metrics=MAX_CELLS,
          max_tensor_elements=MAX_ELEMENTS,
          max_evidence_elements=MAX_EVIDENCE_ELEMENTS,
      ),
      notice=' '.join(
          dict.fromkeys([
              (
                  'Only recorded correspondence and explicit layouts produce'
                  ' metrics. Missing or undefined values remain unavailable.'
              ),
              *notices,
          ])
      ),
  )
