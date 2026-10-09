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

"""Preserve execution, KV, and sampling evidence apart from graph outputs."""

from copy import deepcopy
import shutil

from .fsutil import digest, file_digest, proof_digest
from .tensor_io import load_tensor

COLLECTIONS = (
    'forwards',
    'kv_snapshots',
    'token_records',
    'generations',
    'resources',
)


def empty_telemetry():
  return {'format_version': 1, **{name: [] for name in COLLECTIONS}}


def preserve_shard(root, tensor, temporary, prefix, copied):
  """Validate a reference, then copy each original payload container once."""
  load_tensor(root, tensor)
  source = (root / tensor['path']).resolve()
  if source not in copied:
    name = f'tensors/{prefix}-{len(copied)}.safetensors'
    shutil.copy2(source, temporary / name)
    copied[source] = name, file_digest(temporary / name)
  return copied[source]


def _coordinates(record):
  forward_id = record.get('forward_id')
  phase, step = record.get('phase'), record.get('step')
  if (
      type(forward_id) is not int
      or forward_id < 0
      or phase not in ('prefill', 'decode', 'unknown')
      or (step is not None and (type(step) is not int or step < 0))
      or (phase != 'unknown' and step is None)
  ):
    raise ValueError('Invalid forward evidence coordinates')
  return forward_id


def _verify_proof(proof_row, forward, resources, temporary):
  proof = proof_row.get('input_proof')
  if not isinstance(proof, dict) or proof_row.get(
      'input_identity'
  ) != proof_digest(proof):
    raise ValueError('Forward input proof identity mismatch')
  tensors = proof.get('tensors')
  if not isinstance(tensors, dict) or not tensors:
    raise ValueError('Missing forward input tensors')
  actual = {}
  for item in forward.get('inputs', []):
    path = item.get('output_path', [])
    if len(path) == 2 and path[0] == 'kwargs' and item.get('resource_id'):
      actual[path[1]] = resources[item['resource_id']]
  if set(actual) != set(tensors):
    raise ValueError('Forward input proof omits or invents captured tensors')
  for name, metadata in tensors.items():
    tensor = load_tensor(temporary, actual[name])
    if (
        not isinstance(metadata, dict)
        or metadata.get('shape') != list(tensor.shape)
        or metadata.get('dtype') != tensor.dtype.name
        or metadata.get('values') != tensor.tolist()
    ):
      raise ValueError('Forward input proof differs from captured inputs')
  # A matching step number is insufficient. Logical comparison needs the
  # worker's consumed prefix and the actual call's positions and complete mask.
  context = proof.get('cache_context')
  if proof.get('scope') != 'same_logical_context' or not isinstance(
      context, dict
  ):
    return False
  count = context.get('processed_token_count')
  prefix, mask = context.get('token_ids'), context.get('attention_mask')
  if (
      type(count) is not int
      or count < 0
      or not isinstance(prefix, list)
      or not isinstance(mask, list)
      or len(prefix) != count
      or len(mask) != count
      or any(type(token) is not int or token < 0 for token in prefix)
      or any(type(value) is not int or value not in (0, 1) for value in mask)
  ):
    return False
  ids = tensors.get('input_ids', {}).get('values')
  full_mask = tensors.get('attention_mask', {}).get('values')
  if (
      not isinstance(ids, list)
      or len(ids) != 1
      or not isinstance(ids[0], list)
      or not isinstance(full_mask, list)
      or len(full_mask) != 1
      or full_mask[0][:count] != mask
      or len(full_mask[0]) != count + len(ids[0])
  ):
    return False
  positions = tensors.get('cache_position', {}).get('values')
  if positions is None and forward.get('pos_offset') is not None:
    positions = list(
        range(forward['pos_offset'], forward['pos_offset'] + len(ids[0]))
    )
  if positions != list(range(count, count + len(ids[0]))):
    return False
  cache = forward.get('cache_before', {})
  if count == 0:
    return (
        cache.get('state') in ('not_allocated', 'disabled')
        or cache.get('processed_token_count') == 0
    )
  return (
      cache.get('state') == 'available'
      and cache.get('processed_token_count') == count
  )


def append_telemetry(
    data, index, root, temporary, job_id, run, turn, result, copied
):
  """Save a run's full-generation records; resolve only committed resources."""
  if index.get('export_scope') != 'all':
    return {}
  runtime = index.get('runtime')
  if runtime not in ('PyTorch', 'LiteRT-LM'):
    raise ValueError('Full execution telemetry requires an explicit runtime')
  from .runtime.litert_dump_inference import CAPTURE_SCOPE as INFERRED_SCOPE

  inferred = (
      runtime == 'LiteRT-LM' and index.get('capture_scope') == INFERRED_SCOPE
  )
  if inferred:
    # Inferred bindings are re-derived here from the saved shards, never trusted
    # from the importer.
    from .runtime.litert_dump_inference import verify_inferred_index

    verify_inferred_index(index, root)
  elif runtime == 'LiteRT-LM':
    from .runtime.litert_trace import verify_trace_index

    verify_trace_index(index, root)
  provenance = {
      'run': run,
      'turn': turn,
      'runtime': runtime,
      'model_sha256': result.get('model_sha256'),
  }
  if inferred:
    provenance['basis'] = index['inference'].get('basis')
  if runtime == 'LiteRT-LM':
    provenance.update(
        backend_requested=result.get('backend_requested')
        or index.get('backend_requested'),
        backend_effective=result.get('backend_effective'),
    )
  forwards = {}
  for original in index.get('forwards', []):
    forward = deepcopy(original)
    forward_id = _coordinates(forward)
    if forward_id in forwards or forward.get('status') not in (
        'completed',
        'failed',
    ):
      raise ValueError('Invalid or duplicate forward evidence')
    forward.update(
        provenance,
        runtime_turn=original.get('turn'),
        id=f'{job_id}:{run}:forward:{forward_id}',
        sample=None,
    )
    forwards[forward_id] = forward
  resources, addresses = {}, {}
  for original in index.get('resources', []):
    resource = deepcopy(original)
    scope, key = resource.get('scope'), resource.get('capture_key')
    if (
        scope not in ('boundary', 'kv')
        or not isinstance(key, str)
        or not key
        or type(resource.get('forward_id')) is not int
        or resource.get('forward_id') not in forwards
        or (scope, key) in addresses
    ):
      raise ValueError('Invalid or duplicate execution resource')
    path, checksum = preserve_shard(
        root, resource, temporary, f'{job_id}-{run}-resource', copied
    )
    resource_id = 'resource-' + digest([job_id, run, scope, key])[:32]
    resource.update(
        provenance,
        id=resource_id,
        path=path,
        sha256=checksum,
        sample=None,
        runtime_turn=original.get('turn'),
        storage_status='stored',
    )
    if original.get('storage_source'):
      source = deepcopy(original['storage_source'])
      source_path, source_checksum = preserve_shard(
          root, source, temporary, f'{job_id}-{run}-storage-source', copied
      )
      if source_checksum != source.get('sha256'):
        raise ValueError('Native storage source checksum mismatch')
      source.update(path=source_path, sha256=source_checksum)
      resource['storage_source'] = source
    if runtime == 'LiteRT-LM' and original.get('native_trace'):
      resource['native_trace'] = dict(
          path=f'inputs/{job_id}-{run}-runtime-trace.jsonl',
          sha256=index['native_trace']['sha256'],
      )
    resources[resource_id] = resource
    addresses[scope, key] = resource

  def bind(item, scope, forward_id):
    resource = addresses.get((scope, item.get('key')))
    item['resource_id'] = None
    item['storage_status'] = 'unavailable'
    if resource is not None:
      if (
          resource['forward_id'] != forward_id
          or resource['shape'] != item.get('shape')
          or resource['dtype'] != str(item.get('dtype')).removeprefix('torch.')
      ):
        raise ValueError('Execution index differs from its resource')
      if (
          runtime == 'LiteRT-LM'
          and scope == 'boundary'
          and resource.get('output_path') != item.get('output_path')
      ):
        raise ValueError('Native boundary role differs from its resource')
      item.update(resource_id=resource['id'], storage_status='stored')
    return resource

  for forward in forwards.values():
    for edge in ('inputs', 'outputs'):
      for item in forward.get(edge, []):
        bind(item, 'boundary', forward['forward_id'])
  for proof_row in index.get('forward_input_proofs', []):
    forward = forwards.get(proof_row.get('forward_id'))
    if forward is None or 'input_proof' in forward:
      raise ValueError('Invalid or duplicate forward input proof')
    if runtime == 'LiteRT-LM':
      from .runtime.litert_trace import verify_input_proof

      eligible = verify_input_proof(proof_row, forward, resources, temporary)
    else:
      eligible = _verify_proof(proof_row, forward, resources, temporary)
    forward.update(
        input_proof=deepcopy(proof_row['input_proof']),
        input_identity=proof_row['input_identity'],
        comparison_basis='logical_context'
        if eligible
        else 'input equivalence not established',
        comparison_eligible=eligible and forward['status'] == 'completed',
    )
  snapshots, snapshot_ids = [], set()
  for original in index.get('kv_snapshots', []):
    snapshot = deepcopy(original)
    forward_id = snapshot.get('forward_id')
    snapshot_id = snapshot.get('snapshot_id')
    if (
        type(forward_id) is not int
        or forward_id not in forwards
        or type(snapshot_id) is not int
        or snapshot_id < 0
        or snapshot_id in snapshot_ids
        or snapshot.get('moment')
        not in ('prefill_pre', 'prefill_post', 'terminal')
    ):
      raise ValueError('Invalid KV snapshot identity')
    snapshot_ids.add(snapshot_id)
    snapshot.update(
        provenance,
        runtime_turn=original.get('turn'),
        id=f'{job_id}:{run}:kv:{snapshot_id}',
    )
    if inferred:
      # The logical context is derived, never observed: keep its identity and
      # count, no token IDs.
      forward = forwards[forward_id]
      if (
          snapshot.get('basis') != provenance.get('basis')
          or type(snapshot.get('processed_token_count')) is not int
          or snapshot['processed_token_count'] < 0
          or not isinstance(snapshot.get('logical_context_identity'), str)
          or snapshot.get('logical_token_ids') is not None
          or snapshot['moment'] == 'prefill_pre'
      ):
        raise ValueError(
            'Inferred KV snapshot lacks its derived logical context'
        )
      snapshot['comparison_eligible'] = (
          forward.get('comparison_eligible') is True
      )
    elif runtime == 'LiteRT-LM':
      from .runtime.litert_trace import state

      forward = forwards[forward_id]
      context = state(
          forward[
              'state_before'
              if snapshot['moment'] == 'prefill_pre'
              else 'state_after'
          ]
      )
      expected_identity = proof_digest(
          dict(token_ids=context, processed_token_count=len(context))
      )
      if (
          snapshot.get('logical_token_ids') != context
          or snapshot.get('processed_token_count') != len(context)
          or snapshot.get('logical_context_identity') != expected_identity
      ):
        raise ValueError(
            'Native KV logical context differs from its observed forward state'
        )
      snapshot['comparison_eligible'] = (
          forward.get('comparison_eligible') is True
      )
    for layer in snapshot.get('layers', []):
      for item in layer.get('tensors', []):
        resource = bind(item, 'kv', forward_id)
        if resource is not None:
          if (
              resource.get('snapshot_id') != snapshot_id
              or resource.get('layer') != layer.get('layer')
              or resource.get('kind') != item.get('kind')
          ):
            raise ValueError('KV resource identity mismatch')
          resource.update({
              key: deepcopy(value)
              for key, value in layer.items()
              if key != 'tensors'
          })
          resource.update(
              moment=snapshot['moment'],
              snapshot_id=snapshot_id,
              preparation_status=snapshot.get('preparation_status'),
              terminal_status=snapshot.get('terminal_status'),
          )
    snapshots.append(snapshot)
  data['forwards'].extend(forwards.values())
  data['resources'].extend(resources.values())
  data['kv_snapshots'].extend(snapshots)
  for event in index.get('token_records', []):
    if (
        type(event.get('forward_id')) is not int
        or event['forward_id'] not in forwards
        or forwards[event['forward_id']]['status'] != 'completed'
        or type(event.get('candidate')) is not int
        or event['candidate'] < 0
        or not isinstance(event.get('token_ids'), list)
        or not event['token_ids']
        or any(
            type(token) is not int or token < 0 for token in event['token_ids']
        )
        or not isinstance(event.get('consumption'), list)
        or len(event['consumption']) != len(event['token_ids'])
    ):
      raise ValueError('Token event refers to an unknown forward')
    for item in event['consumption']:
      consumer = item.get('consumed_by_forward_id')
      if consumer is not None and (
          type(consumer) is not int
          or consumer not in forwards
          or forwards[consumer]['status'] != 'completed'
          or consumer <= event['forward_id']
      ):
        raise ValueError('Invalid consumed token forward')
    data['token_records'].append({**deepcopy(event), **provenance})
  data['generations'].append(
      {**deepcopy(index.get('generation', {})), **provenance}
  )
  return forwards


def pair_forward_evidence(data, current, runtimes, results, turn):
  """Pair the same model and proven logical inputs; cache bytes may differ."""
  runtime = runtimes.get('ref')
  if runtime != runtimes.get('target') or runtime not in (
      'PyTorch',
      'LiteRT-LM',
  ):
    return
  model = results['ref'].get('model_sha256')
  if not model or model != results['target'].get('model_sha256'):
    return
  if runtime == 'LiteRT-LM':
    _pair_native(data, current, model, turn)
    return
  for forward_id, ref in current.get('ref', {}).items():
    target = current.get('target', {}).get(forward_id)
    if (
        not target
        or not ref.get('comparison_eligible')
        or not target.get('comparison_eligible')
        or ref['input_identity'] != target['input_identity']
        or any(
            ref.get(key) != target.get(key)
            for key in ('phase', 'step', 'runtime_turn')
        )
    ):
      continue
    sample = digest([turn, runtime, model, forward_id, ref['input_identity']])
    ref['sample'] = target['sample'] = sample
    for resource in data['resources']:
      if (
          resource['turn'] == turn
          and resource['run'] in ('ref', 'target')
          and resource['forward_id'] == forward_id
      ):
        resource['sample'] = sample


def _pair_native(data, current, model, turn):
  """Pair native forwards and KV snapshots by identity, not invocation ID.

  Native chunk sizes differ by backend, so invocation IDs are not alignment.
  """
  paired = {}
  for role in ('ref', 'target'):
    for forward in current.get(role, {}).values():
      if forward.get('comparison_eligible'):
        key = (
            forward['phase'],
            forward['input_identity'],
            forward.get('runtime_turn'),
        )
        paired.setdefault(key, {'ref': [], 'target': []})[role].append(forward)
  for key, sides in paired.items():
    if any(len(sides[role]) != 1 for role in ('ref', 'target')):
      continue
    sample = digest([turn, 'LiteRT-LM', model, key])
    for role in ('ref', 'target'):
      forward = sides[role][0]
      forward['sample'] = sample
      for resource in data['resources']:
        if (
            resource['turn'] == turn
            and resource['run'] == role
            and resource['forward_id'] == forward['forward_id']
            and resource['scope'] != 'kv'
        ):
          resource['sample'] = sample
  snapshots = {}
  for snapshot in data['kv_snapshots']:
    if snapshot['turn'] == turn and snapshot.get('comparison_eligible'):
      key = snapshot['moment'], snapshot['logical_context_identity']
      snapshots.setdefault(key, {'ref': [], 'target': []})[
          snapshot['run']
      ].append(snapshot)
  for key, sides in snapshots.items():
    if any(len(sides[role]) != 1 for role in ('ref', 'target')):
      continue
    sample = digest([turn, 'LiteRT-LM', model, 'kv', key])
    for role in ('ref', 'target'):
      snapshot = sides[role][0]
      snapshot['sample'] = sample
      for resource in data['resources']:
        if (
            resource['turn'] == turn
            and resource['run'] == role
            and resource['scope'] == 'kv'
            and resource.get('snapshot_id') == snapshot['snapshot_id']
        ):
          resource['sample'] = sample
