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

"""Terminal KV comparisons with explicit owner/layout/quantization evidence."""

from copy import deepcopy
import math
import re

import numpy as np
from safetensors import safe_open

from .cross_runtime_common import _digest, _require, _view
from .cross_runtime_positions import evaluate_position
from .metrics import compare

LAYOUT = ['batch', 'kv_head', 'sequence', 'head_dim']


def _same_tensor(left, right):
  return all(
      left.get(key) == right.get(key)
      for key in ('format', 'path', 'key', 'shape', 'dtype')
  )


def _reference(evidence, base, declaration, layer_id, kind, count):
  index = evidence.json('ref', base['prefill']['runs']['ref']['capture_index'])
  fid = base['runs']['ref']['observation']['forward_id']
  snapshots = [
      row
      for row in index.get('kv_snapshots', [])
      if row.get('snapshot_id') == declaration.get('snapshot_id')
  ]
  _require(len(snapshots) == 1, 'one PyTorch terminal snapshot required')
  snapshot = snapshots[0]
  _require(
      snapshot.get('moment') == 'terminal'
      and snapshot.get('forward_id') == fid
      and snapshot.get('phase') == 'decode'
      and snapshot.get('turn') == 1
      and snapshot.get('state') == 'available'
      and snapshot.get('terminal_status') == 'completed'
      and snapshot.get('preparation_status') == 'prepared'
      and snapshot.get('processed_token_count') == count,
      'PyTorch terminal state mismatch',
  )
  layers = [
      row for row in snapshot.get('layers', []) if row.get('layer') == layer_id
  ]
  _require(len(layers) == 1, 'PyTorch physical KV owner missing')
  layer = layers[0]
  _require(
      layer.get('state') == 'available'
      and layer.get('layout') == LAYOUT
      and layer.get('logical_start') == 0
      and layer.get('logical_end') == count
      and layer.get('valid_length') == count
      and layer.get('processed_token_count') == count,
      'PyTorch KV range/layout mismatch',
  )
  references = [
      row for row in layer.get('tensors', []) if row.get('kind') == kind
  ]
  _require(
      len(references) == 1 and references[0].get('storage_status') == 'stored',
      'PyTorch KV tensor unavailable',
  )
  resource = [
      row
      for row in index.get('resources', [])
      if row.get('scope') == 'kv'
      and row.get('capture_key') == references[0]['key']
  ]
  _require(
      len(resource) == 1
      and resource[0].get('snapshot_id') == snapshot['snapshot_id']
      and resource[0].get('forward_id') == fid
      and resource[0].get('layer') == layer_id
      and resource[0].get('kind') == kind
      and _same_tensor(resource[0], declaration.get('tensor', {}))
      and resource[0].get('shape') == references[0].get('shape')
      and resource[0].get('dtype')
      == str(references[0].get('dtype')).removeprefix('torch.'),
      'PyTorch KV index/resource mismatch',
  )
  generation = index.get('generation', {})
  result = evidence.json('ref', base['prefill']['runs']['ref']['result'])
  _require(
      generation.get('status') == 'completed'
      and generation.get('processed_token_count') == count
      and generation.get('pending_token_ids') == [base['_terminal_pending']]
      and result.get('processed_token_count') == count,
      'PyTorch generation terminal ledger mismatch',
  )
  return layer, evidence.tensor('ref', declaration['tensor'])


def _target(evidence, base, declaration, layer_id, kind, count):
  index = evidence.json('target', declaration.get('kv_index'))
  run = base['prefill']['runs']['target']
  result = evidence.json('target', run['result'])
  native = result['token_replay_proof']
  capture = evidence.json('target', run['capture_index'])
  _require(
      index.get('format_version') == 1
      and index.get('model_tflite_sha256') == capture.get('tapped_sha256')
      and index.get('tapped_container_sha256')
      == run['model']['artifact_sha256']
      and index.get('replay_result_sha256') == run['result']['sha256']
      and index.get('replay_proof_sha256') == native['proof_sha256'],
      'native KV source identity mismatch',
  )
  target = base['runs']['target']['observation']
  identity = {
      'signature': target['signature'],
      'step': target['step'],
      'edge': 'post',
  }
  _require(
      declaration.get('snapshot') == identity
      and index.get('terminal_snapshot') == identity
      and index.get('pending_token_id') == base['_terminal_pending']
      and index.get('pending_token_position') == count,
      'native terminal KV coordinates mismatch',
  )
  snapshots = [
      row
      for row in index.get('snapshots', [])
      if all(row.get(key) == value for key, value in identity.items())
  ]
  _require(
      len(snapshots) == 1
      and snapshots[0].get('processed_token_count') == count,
      'native terminal KV snapshot missing',
  )
  matches = [
      row
      for row in snapshots[0].get('tensors', [])
      if row.get('owner_layer') == layer_id and row.get('kind') == kind
  ]
  _require(len(matches) == 1, 'native physical KV owner missing')
  tensor = matches[0]
  layout = (
      LAYOUT if kind == 'key' else ['batch', 'kv_head', 'head_dim', 'sequence']
  )
  axis = layout.index('sequence')
  _require(
      tensor.get('layout') == layout
      and tensor.get('sequence_axis') == axis
      and tensor.get('logical_start') == 0
      and tensor.get('logical_end') == count
      and tensor.get('valid_length') == count
      and _same_tensor(tensor, declaration.get('tensor', {}))
      and tensor.get('file_sha256') == declaration['tensor'].get('sha256'),
      'native KV tensor/range/layout mismatch',
  )
  source = tensor.get('source', {})
  attrs = source.get('composite_attributes', {})
  names = source.get('owner_input_tensor_names', [])
  _require(
      source.get('signature') == target['signature']
      and source.get('composite') == 'odml.cache_update'
      and isinstance(names, list)
      and len(names) == 2
      and all(isinstance(name, str) for name in names)
      and all(
          re.search(r'/layer_' + str(layer_id) + r'/', name) for name in names
      ),
      'native KV owner lacks cache-update source evidence',
  )
  quant = tensor.get('quantization', {})
  scale = quant.get('scale')
  zero = quant.get('zero_point')
  _require(
      isinstance(scale, (float, int))
      and not isinstance(scale, bool)
      and math.isfinite(scale)
      and scale > 0
      and type(zero) is int
      and zero == 0
      and quant.get('formula')
      == 'real_value = (stored_int8 - zero_point) * scale'
      and scale == attrs.get('scale_k' if kind == 'key' else 'scale_v')
      and declaration.get('dequantization') == quant,
      'explicit KV dequantization differs from cache-update metadata',
  )
  documents = [
      evidence.json('target', row)
      for row in run['model']['source_chain']
      if row['path'].endswith('.json')
  ]
  constants = [
      row
      for document in documents
      for row in document.get('decode_cache_outputs', [])
      if row.get('name') == tensor['key'].removeprefix('post_')
      and row.get('tensor') == source.get('output_tensor')
  ]
  _require(
      len(constants) == 1
      and constants[0].get('dtype') == 'INT8'
      and constants[0].get('quantization', {}).get('scale') == [scale]
      and constants[0].get('quantization', {}).get('zero_point') == [zero],
      'native KV quantization lacks tensor-source proof',
  )
  raw = evidence.tensor('target', declaration['tensor'])
  _require(
      raw.dtype == np.int8
      and raw.ndim == 4
      and raw.shape[axis] == tensor.get('capacity')
      and raw.shape[layout.index('head_dim')]
      == tensor.get('head_dim')
      == attrs.get('head_size')
      and count <= tensor['capacity'] <= attrs.get('cache_size', 0),
      'native KV raw capacity/head dimensions mismatch',
  )
  with safe_open(
      evidence.file('target', declaration['tensor']), framework='numpy'
  ) as file:
    metadata = file.metadata() or {}
  _require(
      metadata.get('signature') == target['signature']
      and metadata.get('step') == str(target['step']),
      'native KV raw signature/step mismatch',
  )
  return tensor, raw


def evaluate_kv(manifest, roots):
  _require(
      manifest.get('kind') == 'explicit_cross_runtime_terminal_kv',
      'unsupported KV pair schema',
  )
  base = deepcopy(manifest.get('terminal_position'))
  report, evidence = evaluate_position(base, roots)
  count = report['processed_token_count']
  _require(
      manifest.get('observation') == report['observation']
      and report['compared_position_range'] == [count - 1, count],
      'KV pair requires the last consumed position observation',
  )
  native = evidence.json('target', base['prefill']['runs']['target']['result'])[
      'token_replay_proof'
  ]['decode']
  _require(
      native['processed_token_count'] == count
      and native['pending_token_id'] == report['pending_token_id'],
      'selected position is not terminal',
  )
  base['_terminal_pending'] = report['pending_token_id']
  layer_id, kind = manifest.get('owner_layer'), manifest.get('kind_of_tensor')
  _require(
      type(layer_id) is int
      and 0 <= layer_id < 15
      and kind in ('key', 'value')
      and set(manifest.get('runs', {})) == {'ref', 'target'},
      'explicit physical owner and K/V required',
  )
  ref_layer, ref = _reference(
      evidence, base, manifest['runs']['ref'], layer_id, kind, count
  )
  native_tensor, target = _target(
      evidence, base, manifest['runs']['target'], layer_id, kind, count
  )
  values = {}
  for role, raw in (('ref', ref), ('target', target)):
    declaration = manifest['runs'][role]
    layout = ref_layer['layout'] if role == 'ref' else native_tensor['layout']
    axis = layout.index('sequence')
    view = [{'start': 0, 'stop': size, 'step': 1} for size in raw.shape]
    view[axis]['stop'] = count
    order = [layout.index(name) for name in LAYOUT]
    _require(
        declaration.get('layout') == layout
        and declaration.get('view') == view
        and isinstance(declaration.get('axis_order'), list)
        and all(type(axis) is int for axis in declaration['axis_order'])
        and declaration['axis_order'] == order
        and (role != 'ref' or declaration.get('dequantization') is None),
        'KV view/permutation must explicitly preserve the full logical range'
        ' and heads',
    )
    value = _view(raw, declaration).transpose(order).astype(np.float64)
    if role == 'target':
      quant = declaration['dequantization']
      value = (value - quant['zero_point']) * quant['scale']
    values[role] = value
  _require(
      values['ref'].shape == values['target'].shape,
      'canonical KV shapes differ',
  )
  report.update(
      format_version=3,
      pair_id=_digest(manifest),
      scope='kv',
      owner_layer=layer_id,
      kind=kind,
      semantic=f'physical KV owner {layer_id} · {kind}',
      compared_position_range=[0, count],
      shape=list(values['ref'].shape),
      canonical_layout=LAYOUT,
      metrics=compare(values['ref'], values['target']),
  )
  for role, declaration in manifest['runs'].items():
    old = deepcopy(report['runs'][role])
    report['runs'][role].update(
        original_tensor=deepcopy(declaration['tensor']),
        observation_tensor=old['original_tensor'],
        layout=declaration['layout'],
        view=declaration['view'],
        axis_order=declaration['axis_order'],
        dequantization=declaration.get('dequantization'),
        observation={
            **old['observation'],
            'moment': 'terminal',
            'compared_position_range': [0, count],
        },
    )
  return report, evidence
