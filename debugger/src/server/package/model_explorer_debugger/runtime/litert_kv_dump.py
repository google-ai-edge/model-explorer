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

"""Index KV cache snapshots from the RuntimeDebugger dump.

The unpatched runtime writes every KV cache tensor after each Prefill invocation
and once more when generation ends (its ``ObserveTokens`` flush), never before
Prefill and never per Decode step. This module turns those shards into
``kv_snapshots``: ``prefill_post`` on the last Prefill forward and ``terminal``
on the last Decode forward. The processed token count of each snapshot comes
from the dumped ``input_pos`` and ``mask`` tensors and is cross-checked against
the Decode count and the Runner's token count; a failed check keeps no snapshot.
Logical contexts are inferred (serialized conversation, admitted input IDs,
released token IDs), so records carry the same ``basis`` as the generation
evidence.

CPU host tensors are logical as dumped. GPU tensors are host downloads of the
WebGPU delegate's physical layout with unsigned bytes; they are normalized on
the Server with the pinned conversion in ``litert_storage`` into a separate
logical shard. The storage contract binds that shard to the untouched dump
shard, the exact LiteRT revision the Runner was built from and the Runner's
backend evidence, and every value is re-validated at publication and on read.
Without that evidence the GPU bytes stay unavailable.
"""

from copy import deepcopy
import json
from pathlib import Path
import re

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from ..fsutil import file_digest, proof_digest
from .litert_storage import (
    DUMP_BINDING,
    NativeStorageError,
    REFERENCE,
    normalize_webgpu_kv,
    require_storage,
    validate_storage,
)

KV_KEY = re.compile(r'^post_(kv_cache_([kv])_(\d+))$')
CHECKS = (
    'kv_prefill_count_matches_mask',
    'kv_terminal_count_matches_decodes',
    'kv_runner_count_reconciles',
)
MOMENTS = {'prefill': 'prefill_post', 'decode': 'terminal'}
ASSUMPTION = (
    'The unpatched runtime records no buffer type. LiteRT-LM keeps WebGPU KV'
    ' caches in WebGpuBuffer I8 storage whose host download carries unsigned'
    ' bytes in the delegate layout; the pinned conversion is the one the'
    ' runtime storage contract applies for that buffer type.'
)


def _read(path, key):
  with safe_open(path, framework='numpy') as shard:
    return shard.get_tensor(key)


def _mask_columns(capture, forward):
  """Returns attention columns of a Prefill mask.

  These are every position the cache holds after the chunk.
  """
  path = (
      Path(capture)
      / f"{forward['signature']}_pre_mask_step_{forward['step']}.safetensors"
  )
  if not path.is_file():
    return None
  mask = np.asarray(_read(path, 'pre_mask'))
  if (
      mask.ndim != 4
      or mask.shape[0] != 1
      or mask.shape[1] != 1
      or mask.dtype != np.bool_
  ):
    return None
  return int(mask[0, 0].any(axis=0).sum())


def prefill_processed(capture, forward, position_path):
  """Returns positions held after a Prefill forward, and the mask's columns.

  The positions held are the first input position plus its valid rows.
  """
  positions = np.asarray(_read(position_path, 'pre_input_pos')).reshape(-1)
  if positions.size == 0:
    return None, None
  valid = 1
  while (
      valid < positions.size
      and int(positions[valid]) == int(positions[0]) + valid
  ):
    valid += 1
  return int(positions[0]) + valid, _mask_columns(capture, forward)


def terminal_processed(position_path):
  """Returns positions held after a Decode forward.

  That is the position it consumed plus one.
  """
  positions = np.asarray(_read(position_path, 'pre_input_pos')).reshape(-1)
  return int(positions[0]) + 1 if positions.size else None


def gpu_evidence(result, build):
  """Returns what the Runner and its build record about the accelerator.

  Nothing is guessed.
  """
  evidence = (
      result.get('backendEvidence')
      if isinstance(result.get('backendEvidence'), dict)
      else {}
  )
  libraries = [
      lib for lib in evidence.get('libraries') or [] if isinstance(lib, dict)
  ]
  dawn = next(
      (
          lib
          for lib in libraries
          if 'webgpu' in str(lib.get('name', '')).lower()
          and isinstance(lib.get('binarySHA256'), str)
      ),
      None,
  )
  lock = build.get('sourceLock') if isinstance(build, dict) else None
  return dict(
      litert_revision=lock.get('litert') if isinstance(lock, dict) else None,
      backend_effective=evidence.get('effectiveBackend'),
      webgpu_library=dict(name=dawn['name'], sha256=dawn['binarySHA256'])
      if dawn
      else None,
  )


def gate_reason(evidence):
  if evidence['backend_effective'] != 'GPU':
    return 'the Runner did not report an effective GPU backend'
  if evidence['litert_revision'] != REFERENCE['source_revision']:
    return (
        'the Runner was not built from the LiteRT revision the KV conversion is'
        ' pinned to'
    )
  if not evidence['webgpu_library']:
    return 'the Runner reported no loaded WebGPU library'
  return None


def _normalize_shard(
    resource, shard, capture, forward, descriptor, evidence, section_sha256
):
  """Write the logical shard next to the untouched dump shard.

  The contract that binds the two shards is validated too.
  """
  kind, shape, key = descriptor['kind'], list(shard['shape']), shard['key']
  logical = normalize_webgpu_kv(_read(shard['path'], key), kind, shape)
  storage = dict(
      version=1,
      representation='logical_tensor',
      binding=DUMP_BINDING,
      logical=dict(dtype='I8', shape=shape, layout=list(descriptor['layout'])),
      source=dict(
          path=shard['path'].name,
          key=key,
          dtype='I8',
          shape=shape,
          buffer_type='WebGpuBuffer',
          origin='runtime_dump',
      ),
      conversion=dict(
          id='webgpu_kv_u8_to_logical_i8',
          version=1,
          kind=kind,
          source_function='RearrangeK' if kind == 'key' else 'RearrangeV',
          **REFERENCE,
      ),
      model_source={
          **deepcopy(descriptor['source']),
          'quantization': deepcopy(descriptor['quantization']),
      },
      evidence=deepcopy(evidence),
      assumption=ASSUMPTION,
  )
  name = (
      shard['path'].name.removesuffix('.safetensors') + '_logical.safetensors'
  )
  target = shard['path'].with_name(name)
  save_file(
      {key: logical},
      target,
      metadata={
          'signature': forward['signature'],
          'step': str(forward['step']),
          'storage': json.dumps(storage, sort_keys=True),
      },
  )
  resource.update(
      path='raw/' + name, sha256=file_digest(target), storage=storage
  )
  validate_storage(
      resource,
      Path(capture).parent,
      descriptor=descriptor,
      model_section_sha256=section_sha256,
  )


def attach_kv(
    capture,
    *,
    basis,
    forwards,
    files,
    kv_shards,
    model,
    model_reason,
    build,
    result,
    job,
    turn,
    model_sha256,
    backend,
    released_ids,
    admitted_ids,
    generation_ok,
):
  """Return ``(status, kv_snapshots, kv_resources)``.

  Anything unproven is skipped with its reason.
  """
  capture = Path(capture)
  checks = {name: None for name in CHECKS}
  status = dict(
      basis=basis,
      status='unavailable',
      reason=None,
      checks=checks,
      moments=[],
      skipped=[],
  )

  def unavailable(reason):
    status['reason'] = reason
    return status, [], []

  if not kv_shards:
    return unavailable('the runtime dump holds no KV cache tensors')
  if model is None:
    return unavailable(
        model_reason or 'model KV cache descriptors are unavailable'
    )
  if not generation_ok:
    return unavailable(
        'generation inference did not pass, so no logical context is'
        ' established'
    )
  prefills = [f for f in forwards if f['phase'] == 'prefill']
  decodes = [f for f in forwards if f['phase'] == 'decode']
  if not prefills or not decodes:
    return unavailable('a Prefill and a Decode forward are required')
  last_prefill, last_decode = prefills[-1], decodes[-1]
  try:
    prefill_count, mask_columns = prefill_processed(
        capture,
        last_prefill,
        files[last_prefill['forward_id'], 'pre_input_pos'],
    )
    terminal_count = terminal_processed(
        files[last_decode['forward_id'], 'pre_input_pos']
    )
  except (KeyError, ValueError) as error:
    return unavailable(f'input positions unavailable: {error}')
  checks['kv_prefill_count_matches_mask'] = (
      prefill_count is not None and mask_columns == prefill_count
  )
  checks['kv_terminal_count_matches_decodes'] = (
      prefill_count is not None
      and terminal_count == prefill_count + len(decodes)
  )
  after = result.get('tokenCount')
  if type(after) is int and terminal_count is not None:
    # The engine counts the pending (unprocessed) sample; the cache does not
    # hold it.
    checks['kv_runner_count_reconciles'] = after - terminal_count in (0, 1)
  failed = [name for name in CHECKS if checks[name] is False]
  if failed:
    status['status'] = 'mismatch'
    return unavailable('failed checks: ' + ', '.join(failed))
  evidence = gpu_evidence(result, build)
  gpu = backend == 'GPU'
  blocked = gate_reason(evidence) if gpu else None
  if not gpu and evidence['backend_effective'] not in (None, 'CPU'):
    blocked = (
        'the Runner reported an effective backend other than the requested CPU'
    )
  snapshots, resources = [], []
  for forward, count in (
      (last_prefill, prefill_count),
      (last_decode, terminal_count),
  ):
    moment = MOMENTS[forward['phase']]
    shards = kv_shards.get((forward['signature'], forward['step']), [])
    if not shards:
      status['skipped'].append(
          f"{moment}: no KV shard dumped for {forward['signature']} step"
          f" {forward['step']}"
      )
      continue
    identity = proof_digest(
        dict(
            basis=basis,
            moment=moment,
            model_sha256=model_sha256,
            messages=job.get('messages', []),
            prompt=job.get('prompt', ''),
            admitted_input_ids=admitted_ids,
            processed_token_count=count,
            released_token_ids=list(released_ids)
            if moment == 'terminal'
            else [],
        )
    )
    snapshot = dict(
        snapshot_id=len(snapshots),
        forward_id=forward['forward_id'],
        turn=turn,
        phase=forward['phase'],
        step=forward['step'],
        moment=moment,
        preparation_status='prepared',
        terminal_status='completed' if moment == 'terminal' else None,
        layers=[],
        logical_token_ids=None,
        processed_token_count=count,
        logical_context_identity=identity,
        basis=basis,
        context_basis=(
            'serialized conversation, admitted input IDs and processed count'
        )
        + (', released token IDs' if moment == 'terminal' else ''),
    )
    available = 0
    for ordinal, shard in shards:
      name = KV_KEY.match(shard['key']).group(1)
      descriptor = model['kv'].get((forward['signature'], name))
      if descriptor is None:
        status['skipped'].append(
            f"{moment}: {forward['signature']} {name} has no model cache"
            ' descriptor'
        )
        continue
      layout, shape = list(descriptor['layout']), list(shard['shape'])
      axis = layout.index('sequence')
      if (
          len(shape) != 4
          or shard['dtype'] != 'int8'
          or count > shape[axis]
          or any(
              shape[i] != descriptor['model_shape'][i]
              for i in range(4)
              if i != axis
          )
      ):
        status['skipped'].append(
            f"{moment}: {forward['signature']} {name} differs from the model"
            ' cache layout'
        )
        continue
      resource = dict(
          format='safetensors',
          path='raw/' + shard['path'].name,
          key=shard['key'],
          shape=shape,
          dtype=shard['dtype'],
          sha256=file_digest(shard['path']),
          forward_id=forward['forward_id'],
          phase=forward['phase'],
          step=forward['step'],
          runtime='LiteRT-LM',
          signature=forward['signature'],
          runtime_step=forward['step'],
          capture_key=f"dump:{ordinal}:post:{shard['key']}",
          tensor_name=name,
          when='post',
          scope='kv',
          output_path=['output', name],
          basis=basis,
          snapshot_id=snapshot['snapshot_id'],
          layer=descriptor['layer'],
          kind=descriptor['kind'],
          layout=layout,
          state='available',
          logical_start=0,
          logical_end=count,
          valid_length=count,
          processed_token_count=count,
          capacity=shape[axis],
          quantization=deepcopy(descriptor['quantization']),
          model_source=deepcopy(descriptor['source']),
          model_section_sha256=model['section_sha256'],
          backend_requested=backend,
          backend_effective=evidence['backend_effective'],
          storage_view=[
              dict(start=0, stop=count if i == axis else size, step=1)
              for i, size in enumerate(shape)
          ],
      )
      if descriptor['quantization']:
        resource['dequantization'] = {
            key: descriptor['quantization'][key]
            for key in ('scale', 'zero_point')
        }
      try:
        if blocked:
          raise NativeStorageError(
              f'GPU KV bytes were not normalized: {blocked}. Raw bytes are'
              ' retained.'
          )
        if gpu:
          _normalize_shard(
              resource,
              shard,
              capture,
              forward,
              descriptor,
              evidence,
              model['section_sha256'],
          )
        require_storage(resource)
      except NativeStorageError as error:
        resource.update(
            state='unavailable',
            storage_comparison_status=error.status,
            storage_comparison_reason=error.reason,
        )
      else:
        available += 1
      layer = next(
          (
              row
              for row in snapshot['layers']
              if row['layer'] == descriptor['layer']
          ),
          None,
      )
      if layer is None:
        layer = dict(layer=descriptor['layer'], tensors=[])
        snapshot['layers'].append(layer)
      layer['tensors'].append(
          dict(
              key=resource['capture_key'],
              kind=resource['kind'],
              shape=shape,
              dtype=resource['dtype'],
          )
      )
      resources.append(resource)
    if not snapshot['layers']:
      status['skipped'].append(
          f'{moment}: no KV shard could be bound to the model cache layout'
      )
      continue
    snapshot['state'] = 'available' if available else 'unavailable'
    snapshots.append(snapshot)
    status['moments'].append(moment)
  if snapshots:
    status['status'] = 'ok'
  elif status['reason'] is None:
    status['reason'] = 'no KV shard could be bound to the model cache layout'
  return status, snapshots, resources


def verify_kv(index, root, forwards, positions):
  """Re-derive every inferred KV snapshot's context and storage.

  Both are re-derived from the saved shards.
  """
  root = Path(root)
  snapshots = index.get('kv_snapshots', [])
  kv = (index.get('inference') or {}).get('kv') or {}
  if not snapshots:
    if any(r.get('scope') == 'kv' for r in index.get('resources', [])):
      raise ValueError('Inferred KV resources without their snapshots')
    return
  if kv.get('status') != 'ok' or any(
      kv.get('checks', {}).get(name) is not True for name in CHECKS[:2]
  ):
    raise ValueError('Inferred KV snapshots require passed checks')
  ordered = {
      phase: sorted(
          (f for f in forwards.values() if f['phase'] == phase),
          key=lambda f: f['forward_id'],
      )
      for phase in ('prefill', 'decode')
  }
  resources = {
      r['capture_key']: r
      for r in index.get('resources', [])
      if r.get('scope') == 'kv'
  }
  bound = set()
  for snapshot in snapshots:
    forward = forwards.get(snapshot.get('forward_id'))
    expected = MOMENTS.get(forward['phase']) if forward else None
    if (
        forward is None
        or snapshot.get('moment') != expected
        or forward is not ordered[forward['phase']][-1]
    ):
      raise ValueError(
          'Inferred KV snapshot is not bound to the last Prefill or Decode'
          ' forward'
      )
    position = positions.get((forward['forward_id'], 'pre_input_pos'))
    if position is None:
      raise ValueError('Inferred KV snapshot forward lacks its input position')
    if expected == 'prefill_post':
      count, columns = prefill_processed(root / 'raw', forward, position)
      derived = count if columns == count else None
    else:
      derived = terminal_processed(position)
    if derived is None or derived != snapshot.get('processed_token_count'):
      raise ValueError(
          'Inferred KV processed count differs from the dumped positions'
      )
    for layer in snapshot.get('layers', []):
      for item in layer.get('tensors', []):
        resource = resources.get(item.get('key'))
        if (
            resource is None
            or resource.get('snapshot_id') != snapshot['snapshot_id']
            or resource.get('layer') != layer.get('layer')
            or resource.get('kind') != item.get('kind')
            or resource.get('forward_id') != forward['forward_id']
            or any(
                resource.get(key) != snapshot['processed_token_count']
                for key in (
                    'valid_length',
                    'logical_end',
                    'processed_token_count',
                )
            )
        ):
          raise ValueError('Inferred KV resource differs from its snapshot')
        bound.add(item['key'])
        if resource.get('state') == 'available':
          if resource.get('storage') is not None:
            validate_storage(resource, root)
          require_storage(resource)
  if bound != set(resources):
    raise ValueError('Inferred KV resources without their snapshots')
