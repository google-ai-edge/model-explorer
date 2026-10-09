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

"""Validate native execution events against unchanged captured Safetensors."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from model_explorer_debugger import fsutil
from model_explorer_debugger.runtime import litert_capture
from model_explorer_debugger.runtime import litert_storage
import numpy as np
from safetensors import safe_open

DTYPE_NAMES = {header: name for header, name in litert_capture.DTYPES.values()}


def ids(value):
  if (
      not isinstance(value, list)
      or len(value) != 1
      or not isinstance(value[0], list)
      or any(type(token) is not int or token < 0 for token in value[0])
  ):
    raise ValueError(
        'Native evidence requires actual single-candidate token IDs'
    )
  return value[0]


def state(value):
  if not isinstance(value, dict):
    raise ValueError('Missing native executor state')
  count = value.get('processed_token_count')
  tokens = ids(value.get('processed_token_ids'))
  pending = value.get('pending_token_ids')
  if (
      type(count) is not int
      or count < 0
      or count != len(tokens)
      or type(value.get('runtime_step')) is not int
      or value['runtime_step'] < 0
      or not isinstance(pending, list)
      or len(pending) > 1
      or any(type(token) is not int or token < 0 for token in pending)
  ):
    raise ValueError('Native processed state differs from its actual IDs')
  return tokens


def read_events(path):
  events, previous, session = [], -1, None
  for line in Path(path).read_text().splitlines():
    if not line.strip():
      continue
    row = json.loads(line)
    if (
        not isinstance(row, dict)
        or row.get('format_version') != 1
        or type(row.get('sequence')) is not int
        or row['sequence'] <= previous
        or type(row.get('session_id')) is not int
    ):
      raise ValueError('Invalid native trace identity or event order')
    if session is not None and session != row['session_id']:
      raise ValueError('Native trace crosses runtime sessions')
    session, previous = row['session_id'], row['sequence']
    events.append(row)
  if not events:
    raise ValueError('Native execution trace is empty')
  return events


def verify_trace_index(index, root):
  """The saved trace, not a rewritten index, supplies token identities."""
  trace = index.get('native_trace', {})
  relative = trace.get('path')
  if not isinstance(relative, str) or Path(relative).is_absolute():
    raise ValueError('Invalid native trace path')
  path = (Path(root) / relative).resolve()
  if not path.is_relative_to(Path(root).resolve()) or fsutil.file_digest(
      path
  ) != trace.get('sha256'):
    raise ValueError('Native trace checksum mismatch')
  events = read_events(path)
  originals = {
      row['invocation_id']: row
      for row in events
      if row.get('event') in ('graph_post', 'graph_error')
  }
  fields = (
      'sequence',
      'signature',
      'phase',
      'runtime_step',
      'consumed_token_ids',
      'input_positions',
      'input_rows',
      'state_before',
      'state_after',
      'status',
      'capture_complete',
      'supported',
  )
  if set(originals) != {
      row.get('forward_id') for row in index.get('forwards', [])
  }:
    raise ValueError('Native forward index differs from original trace')
  for row in index['forwards']:
    source = originals[row['forward_id']]
    if any(row.get(key) != source.get(key) for key in fields):
      raise ValueError('Native forward index differs from original trace')
  admissions = [row for row in events if row.get('event') == 'admission']
  admitted = (
      [token for row in admissions for token in ids(row['token_ids'])]
      if all(row.get('supported') is True for row in admissions)
      else None
  )
  if (
      index.get('admitted_input_ids') != admitted
      or index.get('generation', {}).get('admissions') != admissions
  ):
    raise ValueError('Native admission index differs from original trace')
  samples = [row for row in events if row.get('event') == 'sample']
  released = [row for row in events if row.get('event') == 'released_tokens']
  supported = [row for row in samples if row.get('supported') is True]
  expected_tokens = (
      _sample_tokens(supported, released)
      if len(supported) == len(samples)
      else []
  )
  if index.get('tokens') != expected_tokens or len(
      index.get('token_records', [])
  ) != len(supported):
    raise ValueError('Native sampled tokens differ from original trace')
  for ordinal, (sample, row) in enumerate(
      zip(supported, index['token_records'])
  ):
    if (
        row.get('output_index') != ordinal
        or row.get('forward_id') != sample['source_invocation_id']
        or row.get('token_ids') != ids(sample['token_ids'])
        or row.get('candidate') != 0
        or row.get('sample_id') != sample.get('sample_id')
    ):
      raise ValueError('Native sample source differs from original trace')
  tensor_events = {
      (row.get('invocation_id'), row.get('moment'), row.get('key')): row
      for row in events
      if row.get('event') == 'tensor' and row.get('status') == 'stored'
  }
  for resource in index.get('resources', []):
    source = tensor_events.get(
        (resource.get('forward_id'), resource.get('when'), resource.get('key'))
    )
    if (
        source is None
        or resource.get('path') != 'raw/' + source['path']
        or resource.get('tensor_name') != source.get('tensor_name')
        or resource.get('signature') != source.get('signature')
        or resource.get('storage') != source.get('storage')
        or resource.get('capture_key')
        != (
            f"native:{source['invocation_id']}:"
            f"{source['moment']}:{source['key']}"
        )
    ):
      raise ValueError('Native resource index differs from original trace')
    if resource.get('scope') == 'kv' and resource.get('storage') is not None:
      if (
          index.get('model_section_sha256') is not None
          and resource.get('model_source', {}).get('section_sha256')
          != index['model_section_sha256']
      ):
        raise ValueError(
            'Native storage section differs from capture model section'
        )
      litert_storage.validate_storage(resource, root, event=source)


def input_proof(forward, resources, root, tokenizer):
  """Check observed IDs/rows against the native positions and boolean mask."""
  prefix = state(forward['state_before'])
  consumed = ids(forward['consumed_token_ids'])
  positions, rows = forward.get('input_positions'), forward.get('input_rows')
  if (
      not consumed
      or not isinstance(positions, list)
      or not isinstance(rows, list)
      or any(type(value) is not int or value < 0 for value in positions + rows)
      or len(rows) != len(consumed)
      or len(set(rows)) != len(rows)
      or positions != list(range(len(prefix), len(prefix) + len(consumed)))
  ):
    raise ValueError(
        'Native graph positions/rows do not establish a contiguous logical'
        ' input'
    )
  after = state(forward['state_after'])
  if after != prefix + consumed:
    raise ValueError(
        'Native completed graph did not process exactly its recorded input'
    )
  inputs = {
      r['tensor_name']: r
      for r in resources
      if r['forward_id'] == forward['forward_id'] and r['when'] == 'pre'
  }
  if not {'input_pos', 'mask'}.issubset(inputs):
    raise ValueError(
        'Native graph is missing recorded position or attention-mask input'
    )
  tensors = {}
  for name in ('input_pos', 'mask'):
    record = inputs[name]
    with safe_open(root / record['path'], framework='numpy') as shard:
      value = shard.get_tensor(record['key'])
    tensors[name] = (
        value,
        dict(
            shape=list(value.shape),
            dtype=value.dtype.name,
            tensor_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
        ),
    )
  pos, mask = tensors['input_pos'][0], tensors['mask'][0]
  if (
      pos.dtype != np.int32
      or pos.ndim != 1
      or max(rows) >= pos.size
      or pos[rows].tolist() != positions
      or mask.dtype != np.bool_
      or mask.ndim != 4
      or mask.shape[:2] != (1, 1)
      or mask.shape[2] != pos.size
      or mask.shape[3] < len(after)
  ):
    raise ValueError(
        'Native input tensors differ from trace positions or supported mask'
        ' layout'
    )
  # Check every captured row; inactive rows cannot quietly become model input.
  row_positions = dict(zip(rows, positions))
  for index, values in enumerate(mask[0, 0]):
    stop = row_positions[index] + 1 if index in row_positions else 0
    if not np.all(values[:stop]) or np.any(values[stop:]):
      raise ValueError(
          'Native recorded attention mask differs from its logical context'
      )
  return dict(
      format_version=1,
      runtime='LiteRT-LM',
      scope='same_logical_context',
      cache_context=dict(processed_token_count=len(prefix), token_ids=prefix),
      consumed_token_ids=consumed,
      input_positions=positions,
      input_rows=rows,
      tensors={name: metadata for name, (_, metadata) in tensors.items()},
      tokenizer=deepcopy(tokenizer),
  )


def verify_input_proof(proof_row, forward, resources, root):
  proof = proof_row.get('input_proof')
  if not isinstance(proof, dict) or fsutil.proof_digest(proof) != proof_row.get(
      'input_identity'
  ):
    raise ValueError('Native input proof identity mismatch')
  original = input_proof(
      forward, list(resources.values()), root, proof.get('tokenizer', {})
  )
  if proof != original:
    raise ValueError(
        'Native input proof differs from stored execution evidence'
    )
  return (
      forward.get('supported') is True
      and forward.get('capture_complete') is True
  )


def _resource(capture, event, forward):
  relative = event.get('path')
  if (
      not isinstance(relative, str)
      or Path(relative).name != relative
      or not relative.endswith('.safetensors')
  ):
    raise ValueError('Invalid native trace tensor path')
  path = (capture / relative).resolve()
  if not path.is_relative_to(capture):
    raise ValueError('Native trace path escapes capture')
  if (
      event.get('signature') != forward.get('signature')
      or event.get('moment') not in ('pre', 'post', 'terminal')
      or event.get('key')
      != ('pre_' if event['moment'] == 'pre' else 'post_')
      + str(event.get('tensor_name'))
  ):
    raise ValueError('Native resource differs from its graph identity or edge')
  with safe_open(path, framework='numpy') as shard:
    metadata = shard.metadata() or {}
    key = event.get('key')
    if (
        key not in shard.keys()
        or metadata.get('signature') != event.get('signature')
        or metadata.get('step') != str(event.get('runtime_step'))
    ):
      raise ValueError('Native trace tensor identity differs from raw header')
    tensor = shard.get_slice(key)
    dtype = DTYPE_NAMES.get(tensor.get_dtype())
    if dtype is None:
      raise ValueError('Unsupported native raw tensor dtype')
    shape = tensor.get_shape()
  result = dict(
      format='safetensors',
      path='raw/' + relative,
      key=key,
      shape=shape,
      dtype=dtype,
      sha256=fsutil.file_digest(path),
      forward_id=forward['forward_id'],
      phase=forward['phase'],
      step=forward['step'],
      runtime='LiteRT-LM',
      signature=event['signature'],
      runtime_step=event['runtime_step'],
      capture_key=f"native:{event['invocation_id']}:{event['moment']}:{key}",
      tensor_name=event['tensor_name'],
      when=event['moment'],
      scope='boundary',
      output_path=[
          'kwargs' if event['moment'] == 'pre' else 'output',
          event['tensor_name'],
      ],
  )
  if event.get('storage') is not None:
    if json.loads(metadata.get('storage', 'null')) != event['storage']:
      raise ValueError('Native storage event differs from actual tensor header')
    result['storage'] = deepcopy(event['storage'])
  return result


def _sample_tokens(samples, released):
  """Build token entries from sampler IDs and released text.

  Sampler IDs supply identity; only unique ordered release matches supply text.
  """
  tokens = [
      dict(
          id=ids(event['token_ids'])[0],
          step=i,
          text=None,
          source_forward_id=event['source_invocation_id'],
      )
      for i, event in enumerate(samples)
  ]
  visible = []
  for event in released:
    values = ids(event.get('token_ids'))
    texts = event.get('texts', [])
    single = (
        len(values) == 1
        and isinstance(texts, list)
        and len(texts) == 1
        and isinstance(texts[0], str)
    )
    visible.extend((token, texts[0] if single else None) for token in values)

  # Both streams preserve order. Ambiguous repeated IDs do not inherit text.
  def subsequence(small, large):
    iterator = iter(large)
    return all(
        any(value == candidate for candidate in iterator) for value in small
    )

  generated = [token['id'] for token in tokens]
  released_ids = [token for token, _ in visible]
  if not subsequence(released_ids, generated):
    raise ValueError(
        'Released token IDs are not an ordered subset of sampled tokens'
    )
  for index, (token_id, text) in enumerate(visible):
    candidates = [
        i
        for i, value in enumerate(generated)
        if value == token_id
        and subsequence(released_ids[:index], generated[:i])
        and subsequence(released_ids[index + 1 :], generated[i + 1 :])
    ]
    if len(candidates) == 1 and text is not None:
      tokens[candidates[0]].update(
          text=text, text_source='unique_ordered_native_release'
      )
  return tokens


def attach_trace(index, capture, manifest, model_info, *, turn=0):
  """Attach proven generation evidence.

  Unsupported events stay explicitly unpaired.
  """
  capture = Path(capture).resolve()
  events = read_events(capture / 'runtime_trace.jsonl')
  if (
      sum(row.get('event') == 'terminal' for row in events) != 1
      or events[-1].get('event') != 'terminal'
  ):
    raise ValueError(
        'Native trace must be bound to exactly one completed generation'
    )
  pre, completed, tensor_events, admissions, samples, released = (
      {},
      {},
      [],
      [],
      [],
      [],
  )
  terminal = None
  for event in events:
    kind = event.get('event')
    if kind == 'admission':
      if event.get('supported') is True:
        ids(event.get('token_ids'))
        state(event.get('state_before'))
      admissions.append(event)
    elif kind == 'graph_pre':
      identity = event.get('invocation_id')
      if type(identity) is not int or identity < 0 or identity in pre:
        raise ValueError('Invalid or duplicate native invocation')
      pre[identity] = event
    elif kind in ('graph_post', 'graph_error'):
      identity = event.get('invocation_id')
      start = pre.get(identity)
      if (
          start is None
          or identity in completed
          or any(
              event.get(k) != start.get(k)
              for k in (
                  'signature',
                  'phase',
                  'runtime_step',
                  'consumed_token_ids',
                  'input_positions',
                  'input_rows',
                  'state_before',
              )
          )
      ):
        raise ValueError('Native graph completion differs from dispatch')
      if (
          event.get('phase') not in ('prefill', 'decode')
          or type(event.get('runtime_step')) is not int
      ):
        raise ValueError('Invalid native graph coordinates')
      completed[identity] = {
          **deepcopy(event),
          'forward_id': identity,
          'step': event['runtime_step'],
          'turn': turn,
          'status': 'completed' if kind == 'graph_post' else 'failed',
          'inputs': [],
          'outputs': [],
      }
    elif kind == 'tensor':
      tensor_events.append(event)
    elif kind == 'sample':
      samples.append(event)
    elif kind == 'released_tokens':
      released.append(event)
    elif kind == 'terminal':
      terminal = event
  if not completed or terminal is None or terminal.get('task_state') != 'done':
    raise ValueError('Native trace has no completed generation boundary')
  resources, addresses, kv_events = [], set(), []
  lookup = {
      (row['signature'], row['key'], row['step']): row
      for row in index['tensors']
  }
  selected = set()
  for event in tensor_events:
    if event.get('status') != 'stored':
      continue
    forward = completed.get(event.get('invocation_id'))
    if forward is None:
      raise ValueError('Native tensor refers to an unknown invocation')
    resource = _resource(capture, event, forward)
    identity = (event['signature'], event['key'], event['runtime_step'])
    if identity in lookup and event.get('moment') == 'post':
      if identity in selected:
        raise ValueError('Selected tensor has ambiguous native invocation')
      selected.add(identity)
      lookup[identity].update(forward_id=forward['forward_id'], turn=turn)
      continue
    if resource['capture_key'] in addresses:
      raise ValueError('Duplicate native resource identity')
    addresses.add(resource['capture_key'])
    if (event['signature'], event['tensor_name']) in model_info['kv']:
      resource['scope'] = 'kv'
      kv_events.append((event, resource))
    else:
      item = dict(
          key=resource['capture_key'],
          shape=resource['shape'],
          dtype=resource['dtype'],
          output_path=resource['output_path'],
      )
      if event['moment'] in ('pre', 'post'):
        forward['inputs' if event['moment'] == 'pre' else 'outputs'].append(
            item
        )
    resources.append(resource)
  if selected != set(lookup):
    raise ValueError('Selected captures lack unambiguous native trace bindings')
  proofs = []
  root = capture.parent
  for forward in completed.values():
    if (
        forward.get('supported') is not True
        or forward['status'] != 'completed'
        or not forward.get('capture_complete')
    ):
      continue
    proof = input_proof(forward, resources, root, model_info['tokenizer'])
    proofs.append(
        dict(
            forward_id=forward['forward_id'],
            input_proof=proof,
            input_identity=fsutil.proof_digest(proof),
        )
    )
  snapshots = {}
  for event, resource in kv_events:
    forward = completed[event['invocation_id']]
    descriptor = model_info['kv'][event['signature'], event['tensor_name']]
    resource.update(
        kind=descriptor['kind'],
        layout=descriptor['layout'],
        backend_requested=index.get('backend_requested'),
        model_source=descriptor['source'],
        quantization=descriptor['quantization'],
        native_trace=dict(
            path='raw/runtime_trace.jsonl',
            sha256=fsutil.file_digest(capture / 'runtime_trace.jsonl'),
        ),
    )
    if resource.get('storage') is not None:
      if descriptor['quantization']:
        resource['dequantization'] = {
            key: descriptor['quantization'][key]
            for key in ('scale', 'zero_point')
        }
      litert_storage.validate_storage(
          resource,
          root,
          event=event,
          descriptor=descriptor,
          model_section_sha256=model_info['section_sha256'],
      )
    if forward['status'] != 'completed' or forward.get('supported') is not True:
      continue
    moment = event['moment']
    if moment not in ('pre', 'post', 'terminal') or (
        moment != 'terminal' and forward['phase'] != 'prefill'
    ):
      continue
    point = 'terminal' if moment == 'terminal' else 'prefill_' + moment
    executor = forward['state_before' if moment == 'pre' else 'state_after']
    valid = len(state(executor))
    layout = descriptor['layout']
    shape = resource['shape']
    axis = layout.index('sequence')
    if (
        len(shape) != 4
        or valid > shape[axis]
        or any(
            shape[i] != descriptor['model_shape'][i]
            for i in range(4)
            if i != axis
        )
    ):
      raise ValueError(
          'Native KV shape/validity differs from verified model layout'
      )
    key = (forward['forward_id'], point)
    snapshot = snapshots.setdefault(
        key,
        dict(
            snapshot_id=len(snapshots),
            forward_id=forward['forward_id'],
            turn=turn,
            phase=forward['phase'],
            step=forward['step'],
            moment=point,
            preparation_status='prepared',
            terminal_status='completed' if point == 'terminal' else None,
            layers=[],
            logical_token_ids=state(executor),
            processed_token_count=valid,
            logical_context_identity=fsutil.proof_digest(
                dict(token_ids=state(executor), processed_token_count=valid)
            ),
        ),
    )
    resource.update(
        snapshot_id=snapshot['snapshot_id'],
        layer=descriptor['layer'],
        kind=descriptor['kind'],
        layout=layout,
        state='available',
        logical_start=0,
        logical_end=valid,
        valid_length=valid,
        processed_token_count=valid,
        capacity=shape[axis],
        quantization=descriptor['quantization'],
        backend_requested=index.get('backend_requested'),
        model_source=descriptor['source'],
        storage_view=[
            dict(start=0, stop=valid if i == axis else size, step=1)
            for i, size in enumerate(shape)
        ],
    )
    try:
      litert_storage.require_storage(resource)
      if descriptor['quantization']:
        resource['dequantization'] = {
            key: descriptor['quantization'][key]
            for key in ('scale', 'zero_point')
        }
    except litert_storage.NativeStorageError as error:
      resource.update(
          state='unavailable',
          storage_comparison_status=error.status,
          storage_comparison_reason=error.reason,
      )
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
  supported = []
  token_records = []
  for sample in samples:
    source = sample.get('source_invocation_id')
    forward = completed.get(source)
    if sample.get('supported') is not True:
      continue
    values = ids(sample.get('token_ids'))
    if (
        len(values) != 1
        or forward is None
        or forward['status'] != 'completed'
        or forward['phase'] != 'decode'
        or state(sample.get('state_after')) != state(forward['state_after'])
        or sample['state_after'].get('pending_token_ids') != values
    ):
      raise ValueError('Native sample has no proven completed source Decode')
    supported.append(sample)
  tokens = (
      _sample_tokens(supported, released)
      if len(supported) == len(samples)
      else []
  )
  for output_index, sample in enumerate(supported):
    source = sample['source_invocation_id']
    value = ids(sample['token_ids'])[0]
    consumers = [
        row['forward_id']
        for row in completed.values()
        if row['sequence'] > sample['sequence']
        and row['status'] == 'completed'
        and row.get('supported') is True
        and state(row['state_before']) == state(sample['state_after'])
        and row['state_before'].get('pending_token_ids') == [value]
        and ids(row['consumed_token_ids'])[:1] == [value]
    ]
    consumer = min(consumers) if consumers else None
    token_records.append(
        dict(
            forward_id=source,
            candidate=0,
            output_index=output_index,
            token_ids=[value],
            consumption=[dict(consumed_by_forward_id=consumer)],
            sample_id=sample.get('sample_id'),
            logits_stage='graph_output_before_sampling_constraints',
        )
    )
  admitted = [
      token
      for event in admissions
      if event.get('supported') is True
      for token in ids(event['token_ids'])
  ]
  index.update(
      runtime='LiteRT-LM',
      export_scope='all',
      capture_scope='native_generation',
      forwards=list(completed.values()),
      model_section_sha256=model_info['section_sha256'],
      resources=resources,
      kv_snapshots=list(snapshots.values()),
      token_records=token_records,
      forward_input_proofs=proofs,
      native_trace=dict(
          path='raw/runtime_trace.jsonl',
          sha256=fsutil.file_digest(capture / 'runtime_trace.jsonl'),
          first_sequence=events[0]['sequence'],
          last_sequence=events[-1]['sequence'],
      ),
      tokens=tokens,
      admitted_input_ids=admitted
      if all(row.get('supported') is True for row in admissions)
      else None,
      generation=dict(
          status='completed',
          admissions=admissions,
          sampled_token_count=len(samples),
          sampled_token_ids=[ids(row['token_ids']) for row in samples],
          released_tokens=released,
          terminal=terminal,
          token_count_basis=(
              'native sampler IDs, including stop tokens; pending tokens are'
              ' not yet processed KV'
          ),
      ),
  )
  return index
