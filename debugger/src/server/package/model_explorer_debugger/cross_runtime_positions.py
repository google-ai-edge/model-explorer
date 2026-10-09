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

"""Explicit corresponding positions may come from different runtime phases."""

from copy import deepcopy

import numpy as np
from safetensors import safe_open

from .cross_runtime_common import _anchor, _digest, _evaluate_prefill, _require
from .metrics import compare


def _forward_inputs(evidence, run, index):
  observation = run['observation']
  fid = observation.get('forward_id')
  _require(type(fid) is int and fid >= 0, 'PyTorch forward identity required')
  rows = [
      row for row in index.get('forwards', []) if row.get('forward_id') == fid
  ]
  proofs = [
      row
      for row in index.get('forward_input_proofs', [])
      if row.get('forward_id') == fid
  ]
  _require(
      len(rows) == len(proofs) == 1, 'missing or ambiguous PyTorch forward'
  )
  forward, proof = rows[0], proofs[0].get('input_proof', {})
  _require(
      forward.get('status') == 'completed'
      and forward.get('phase') == observation.get('phase')
      and forward.get('turn') == observation.get('turn') == 1
      and forward.get('step') == observation.get('step')
      and forward.get('pos_offset') == observation.get('pos_offset')
      and proofs[0].get('input_identity') == _digest(proof),
      'PyTorch observation/proof mismatch',
  )
  declared = run.get('input_boundary_files', [])
  files = {item.get('path'): item for item in declared}
  _require(
      files and len(files) == len(declared), 'boundary source hashes required'
  )
  resources = {
      row.get('capture_key'): row
      for row in index.get('resources', [])
      if row.get('forward_id') == fid and row.get('scope') == 'boundary'
  }
  tensors = {}
  for item in forward.get('inputs', []):
    path = item.get('output_path')
    _require(
        isinstance(path, list) and len(path) == 2 and path[0] == 'kwargs',
        'unsupported boundary input tree',
    )
    name = path[1]
    resource = resources.get(item.get('key'), {})
    declaration = files.get(resource.get('path'), {})
    _require(
        name not in tensors
        and declaration
        and resource.get('shape') == item.get('shape')
        and resource.get('dtype')
        == str(item.get('dtype')).removeprefix('torch.'),
        'boundary index mismatch',
    )
    value = evidence.tensor(
        'ref', {**resource, 'sha256': declaration['sha256']}
    )
    actual = {
        'shape': list(value.shape),
        'dtype': value.dtype.name,
        'values': value.tolist(),
    }
    _require(
        proof.get('tensors', {}).get(name) == actual,
        'executed forward input proof mismatch',
    )
    tensors[name] = value
  _require(
      set(tensors) == set(proof.get('tensors', {})),
      'incomplete executed forward proof',
  )
  context = proof.get('cache_context', {})
  count = context.get('processed_token_count')
  prefix = context.get('token_ids')
  _require(
      type(count) is int
      and count >= 0
      and isinstance(prefix, list)
      and len(prefix) == count
      and all(type(value) is int and value >= 0 for value in prefix)
      and context.get('attention_mask') == [1] * count,
      'logical cache prefix proof required',
  )
  ids = tensors.get('input_ids')
  mask = tensors.get('attention_mask')
  positions = tensors.get('cache_position', tensors.get('position_ids'))
  _require(
      ids is not None
      and ids.ndim == 2
      and ids.shape[0] == 1
      and np.issubdtype(ids.dtype, np.integer)
      and mask is not None
      and mask.shape == (1, count + ids.shape[1])
      and np.all(mask == 1)
      and positions is not None
      and np.array_equal(
          positions.reshape(-1), np.arange(count, count + ids.shape[1])
      )
      and forward.get('pos_offset') == count,
      'executed position or mask proof mismatch',
  )
  if count:
    _require(
        forward.get('cache_before', {}).get('state') == 'available'
        and forward['cache_before'].get('processed_token_count') == count,
        'actual cache prefix length mismatch',
    )
  return forward, prefix + ids.reshape(-1).tolist(), count, ids.shape[1]


def _decode_executions(evidence, proof):
  decode = proof.get('decode', {})
  emissions, rows = decode.get('actual_output_token_ids'), decode.get(
      'executions'
  )
  _require(
      decode.get('sampler_policy') == 'fixed_token_sequence_constraint'
      and decode.get('model_computation_modified') is False
      and isinstance(rows, list)
      and rows
      and isinstance(emissions, list)
      and len(rows) == len(emissions)
      and all(type(token) is int and token >= 0 for token in emissions),
      'fixed Decode execution evidence required',
  )
  admitted = proof['actual_token_ids']
  length = proof['graph_valid_length']
  context = admitted[:-1]
  expected_input = admitted[-1]
  verified = []
  for output_id, row in zip(emissions, rows):
    _require(
        all(
            type(row.get(key)) is int
            for key in (
                'step',
                'input_token_id',
                'input_position',
                'emitted_token_id',
                'processed_token_count_before',
                'processed_token_count_after',
            )
        )
        and row.get('signature') == 'decode'
        and row.get('step') == length + 1
        and row.get('input_position') == length
        and row.get('input_token_id') == expected_input
        and row.get('emitted_token_id') == output_id
        and row.get('processed_token_count_before') == length
        and row.get('processed_token_count_after') == length + 1,
        'native Decode token/position lineage mismatch',
    )
    inputs = row.get('graph_inputs', {})
    _require(
        set(inputs) == {'position_ids', 'attention_mask'},
        'native Decode graph inputs required',
    )
    pos = evidence.tensor('target', inputs['position_ids'])
    mask = evidence.tensor('target', inputs['attention_mask'])
    _require(
        pos.dtype == np.int32
        and pos.shape == (1,)
        and pos.tolist() == [length],
        'native Decode position mismatch',
    )
    _require(
        mask.dtype == np.bool_
        and mask.ndim == 4
        and mask.shape[:3] == (1, 1, 1)
        and mask.shape[-1] >= length + 1,
        'unsupported native Decode mask layout',
    )
    expected_mask = np.zeros(mask.shape, dtype=np.bool_)
    expected_mask[..., : length + 1] = True
    _require(np.array_equal(mask, expected_mask), 'native Decode mask mismatch')
    for tensor in inputs.values():
      with safe_open(
          evidence.file('target', tensor), framework='numpy'
      ) as file:
        metadata = file.metadata() or {}
      _require(
          metadata.get('signature') == 'decode'
          and metadata.get('step') == str(row['step']),
          'native Decode raw signature/step mismatch',
      )
    context = context + [expected_input]
    verified.append({**row, 'context_token_ids': context})
    expected_input = output_id
    length += 1
  _require(
      decode.get('processed_token_count') == length
      and decode.get('pending_token_id') == expected_input
      and decode.get('pending_token_position') == length,
      'native terminal Decode state mismatch',
  )
  return verified


def evaluate_position(manifest, roots):
  _require(
      manifest.get('kind') == 'explicit_cross_runtime_position',
      'unsupported position pair schema',
  )
  base = manifest.get('prefill')
  report, evidence = _evaluate_prefill(base, roots)
  observation = manifest.get('observation', {})
  _require(
      observation.get('turn') == base['observation']['turn']
      and observation.get('phase') == 'decode',
      'position observation must identify the native Decode phase',
  )
  position = manifest.get('position')
  _require(
      type(position) is int
      and position >= 0
      and set(manifest.get('runs', {})) == {'ref', 'target'},
      'explicit corresponding position and two roles required',
  )
  runs, anchors = {}, {}
  for role in ('ref', 'target'):
    delta = manifest['runs'][role]
    _require(
        set(delta) <= {'anchor', 'observation', 'input_boundary_files'}
        and {'anchor', 'observation'} <= set(delta),
        'position pair cannot override source identity',
    )
    runs[role] = {**deepcopy(base['runs'][role]), **deepcopy(delta)}
    _require(
        runs[role]['observation'].get('turn') == observation['turn'],
        'role turn mismatch',
    )
    anchors[role] = _anchor(evidence, role, runs[role], position_pair=True)
  ref_index = evidence.json('ref', runs['ref']['capture_index'])
  forward, ref_context, start, length = _forward_inputs(
      evidence, runs['ref'], ref_index
  )
  _require(
      anchors['ref'][0].get('forward_id') == forward['forward_id'],
      'reference anchor forward mismatch',
  )
  native_result = evidence.json('target', runs['target']['result'])
  executions = _decode_executions(evidence, native_result['token_replay_proof'])
  native_observation = runs['target']['observation']
  matching = [
      row
      for row in executions
      if row['step'] == native_observation.get('step')
      and row['signature'] == native_observation.get('signature')
  ]
  _require(
      native_observation.get('phase') == 'decode' and len(matching) == 1,
      'native Decode observation mismatch',
  )
  native = matching[0]
  _require(
      anchors['target'][0]['step'] == native['step']
      and native['input_position'] == position
      and start <= position < start + length
      and ref_context[: position + 1] == native['context_token_ids'],
      'actual logical token prefixes or comparison positions differ',
  )
  for role, (_, raw, viewed) in anchors.items():
    local = position - start if role == 'ref' else 0
    declaration = runs[role]['anchor']
    _require(
        declaration['layout'] == ['batch', 'sequence', 'hidden']
        and raw.ndim == 3
        and declaration['view']
        == [
            {'start': 0, 'stop': 1, 'step': 1},
            {'start': local, 'stop': local + 1, 'step': 1},
            {'start': 0, 'stop': raw.shape[2], 'step': 1},
        ]
        and viewed.shape == (1, 1, raw.shape[2]),
        'view must select exactly the declared position and full hidden width',
    )
  _require(
      anchors['ref'][2].shape == anchors['target'][2].shape,
      'position view shape mismatch',
  )
  report.update(
      format_version=2,
      pair_id=_digest(manifest),
      observation=deepcopy(observation),
      compared_position_range=[position, position + 1],
      shape=list(anchors['ref'][2].shape),
      processed_token_ids=native['context_token_ids'],
      processed_token_count=native['processed_token_count_after'],
      pending_token_id=native['emitted_token_id'],
      pending_token_position=native['processed_token_count_after'],
      sampler_policy=(
          'LiteRT fixed token sequence constraint; PyTorch original recorded'
          ' sampler'
      ),
      mask_basis=(
          'same recorded consumed token prefix and positions; raw'
          ' unpadded/causal masks independently verified'
      ),
      metrics=compare(anchors['ref'][2], anchors['target'][2]),
  )
  for role, run in runs.items():
    report['runs'][role].update(
        original_tensor=deepcopy(run['anchor']['tensor']),
        layout=run['anchor']['layout'],
        view=run['anchor']['view'],
        observation={
            **deepcopy(run['observation']),
            'compared_position_range': [position, position + 1],
        },
    )
  return report, evidence
