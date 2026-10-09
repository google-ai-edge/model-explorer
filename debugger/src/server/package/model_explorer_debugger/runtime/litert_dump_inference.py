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

"""Derive generation evidence from the RuntimeDebugger dump alone.

Used when no native trace exists.

The debugger-enabled runtime dumps every graph invocation's boundary tensors
(``<signature>_<pre|post>_<name>_step_<N>.safetensors``) and the released tokens
(``generated_tokens.jsonl``), but records no token IDs, sampler results or
invocation identities. This module reconstructs the generation evidence Token
Diff needs from those files alone, under the one configuration the native Runner
admits today: greedy, single-candidate, non-speculative decoding
(``runner_device.validate_native_request``).

Nothing is trusted from order alone. The k-th released token must equal the
argmax of the k-th Decode invocation's logits, and Decode input positions must
advance by one per invocation and equal ``step - 1``. Anything else fails
closed: forwards and resources are kept, no ``token_records`` are written,
Decode forwards lose their comparison eligibility and ``inference.status`` says
why. Every record carries ``basis: 'inferred_greedy_argmax'`` so consumers can
tell inference from a recorded native trace
(``capture_scope: 'native_generation'``).
"""

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from ..fsutil import file_digest, proof_digest
from .litert_capture import DTYPES
from .litert_kv_dump import KV_KEY, attach_kv, verify_kv
from .litert_storage import INFERRED_BASIS as BASIS

CAPTURE_SCOPE = 'native_dump_inferred'
RESOURCE_TENSORS = {'pre': ('input_pos',), 'post': ('logits', 'activations')}
DTYPE_NAMES = {header: name for header, name in DTYPES.values()}
CHECKS = (
    'greedy_sampler',
    'single_candidate',
    'not_speculative',
    'decode_count',
    'prefill_precedes_decode',
    'input_pos_step',
    'input_pos_contiguous',
    'argmax_matches_released',
    'input_count_reconciles',
)


def admitted_input(result):
  """Returns the Runner-reported admitted input, or None.

  The admitted input is the engine-tokenized rendered template.
  """
  tokens = result.get('inputTokens')
  if not isinstance(tokens, list) or not tokens:
    return None
  records = []
  for index, token in enumerate(tokens):
    identity = token.get('id') if isinstance(token, dict) else None
    if type(identity) is not int or identity < 0:
      return None
    text = token.get('text')
    records.append(
        dict(
            id=identity,
            text=text if isinstance(text, str) else None,
            step=index,
        )
    )
  return records


def has_dump_logits(capture):
  return any(Path(capture).glob('decode_post_logits_step_*.safetensors'))


def released_tokens(path):
  """Returns released IDs and texts in order, or None.

  None when the released tokens are absent or not single-candidate.
  """
  path = Path(path)
  if not path.is_file():
    return None
  tokens = []
  for line in path.read_text().splitlines():
    if not line.strip():
      continue
    record = json.loads(line)
    ids, texts = record.get('token_ids'), record.get('texts')
    if (
        not isinstance(ids, list)
        or len(ids) != 1
        or not isinstance(ids[0], list)
        or any(type(token) is not int or token < 0 for token in ids[0])
    ):
      return None
    aligned = (
        len(ids[0]) == 1
        and isinstance(texts, list)
        and len(texts) == 1
        and isinstance(texts[0], str)
    )
    for offset, token in enumerate(ids[0]):
      tokens.append(
          dict(id=token, text=texts[0] if aligned else None, step=len(tokens))
      )
  return tokens


def stop_token_ids(result):
  """Returns single-ID stop tokens the Runner reported.

  A multi-ID stop sequence cannot match one sample.
  """
  tokens = result.get('stopTokens')
  if not isinstance(tokens, list):
    return None
  ids = set()
  for token in tokens:
    values = token.get('ids') if isinstance(token, dict) else None
    if (
        isinstance(values, list)
        and len(values) == 1
        and type(values[0]) is int
        and values[0] >= 0
    ):
      ids.add(values[0])
  return ids


def classify_input_tokens(tokens, rendered, prompt):
  """Marks admitted tokens as template, text, or special.

  Tokens are marked only when their texts tile the rendered input.
  """
  if (
      not tokens
      or not isinstance(rendered, str)
      or not isinstance(prompt, str)
      or not prompt
  ):
    return
  start = rendered.find(prompt)
  if start < 0:
    return
  end, cursor, spans = start + len(prompt), 0, []
  for token in tokens:
    text = token.get('text')
    if text is None:
      spans.append(None)
      continue
    if not rendered.startswith(text, cursor):
      return
    spans.append((cursor, cursor + len(text)))
    cursor += len(text)
  if cursor != len(rendered):
    return
  for token, span in zip(tokens, spans):
    token['kind'] = (
        'special'
        if span is None
        else 'text'
        if start <= span[0] and span[1] <= end
        else 'template'
    )


def _last_position(tensor):
  tensor = np.asarray(tensor)
  if tensor.ndim == 3 and tensor.shape[0] == 1 and tensor.shape[1] > 0:
    return tensor[0, -1]
  if tensor.ndim == 2 and tensor.shape[0] == 1:
    return tensor[0]
  if tensor.ndim == 1:
    return tensor
  raise ValueError('Native logits are not a single causal-LM sequence')


def _scan(capture, tap_keys):
  """Group every dumped shard by (signature, step).

  Only the boundary tensors we index are kept.
  """
  invocations = {}
  for path in sorted(Path(capture).glob('*.safetensors')):
    with safe_open(path, framework='numpy') as shard:
      metadata = shard.metadata() or {}
      signature = metadata.get('signature')
      try:
        step = int(metadata.get('step'))
      except (TypeError, ValueError):
        raise ValueError(
            f'Missing/invalid step metadata: {path.name}'
        ) from None
      if not isinstance(signature, str) or not signature or step < 0:
        raise ValueError(f'Missing signature metadata: {path.name}')
      group = invocations.setdefault((signature, step), [])
      for key in shard.keys():
        moment = (
            'pre'
            if key.startswith('pre_')
            else 'post'
            if key.startswith('post_')
            else None
        )
        if moment is None:
          raise ValueError(f'Unexpected native dump key {key} in {path.name}')
        name = key[len(moment) + 1 :]
        kv = moment == 'post' and KV_KEY.match(key) is not None
        if (signature, key) in tap_keys or (
            name not in RESOURCE_TENSORS[moment] and not kv
        ):
          continue
        view = shard.get_slice(key)
        dtype = DTYPE_NAMES.get(view.get_dtype())
        if dtype is None:
          raise ValueError(f'Unsupported native dtype in {path.name}')
        group.append(
            dict(
                moment=moment,
                name=name,
                key=key,
                path=path,
                shape=list(view.get_shape()),
                dtype=dtype,
                kv=kv,
            )
        )
  return invocations


def _read(path, key):
  with safe_open(path, framework='numpy') as shard:
    return shard.get_tensor(key)


def infer_generation(
    index,
    capture,
    result,
    job,
    *,
    turn=0,
    model=None,
    model_reason=None,
    build=None,
):
  """Attach inferred forwards, token records and KV snapshots to a tap index.

  Resources are attached too; ``index`` is updated in place.

  ``model`` is ``litert_evidence_model.describe_model`` output (KV cache
  descriptors) and ``build`` the Runner's ``runtime-build.json``; without them
  KV shards stay unindexed.
  """
  capture = Path(capture).resolve()
  model_sha256 = job.get('modelSHA256') or result.get('modelSHA256')
  tap_keys = {(t['signature'], t['key']) for t in index.get('tensors', [])}
  invocations = _scan(capture, tap_keys)
  if not invocations:
    raise ValueError('Native dump has no graph invocations')
  order = sorted(
      invocations, key=lambda key: (key[1], 0 if 'prefill' in key[0] else 1)
  )
  released = released_tokens(capture / 'generated_tokens.jsonl')
  sampler = result.get('sampler') or {}
  checks = {name: None for name in CHECKS}
  checks['greedy_sampler'] = (
      sampler.get('temperature') == 0 and sampler.get('topK') == 1
  )
  checks['single_candidate'] = released is not None
  checks['not_speculative'] = result.get('speculativeDecodingEnabled') is False
  released_ids = [token['id'] for token in released] if released else []
  input_tokens = admitted_input(result)
  admitted_ids = (
      [token['id'] for token in input_tokens] if input_tokens else None
  )
  classify_input_tokens(
      input_tokens, result.get('renderedInput'), job.get('prompt')
  )
  stops = stop_token_ids(result)
  forwards, resources, decodes, files, kv_shards = [], [], [], {}, {}
  prefill_index = decode_index = 0
  for ordinal, (signature, step) in enumerate(order, start=1):
    phase = 'prefill' if 'prefill' in signature else 'decode'
    forward = dict(
        forward_id=ordinal,
        signature=signature,
        phase=phase,
        step=step,
        runtime_step=step,
        turn=turn,
        status='completed',
        supported=True,
        capture_complete=True,
        runtime='LiteRT-LM',
        inputs=[],
        outputs=[],
        basis=BASIS,
    )
    for tensor in invocations[(signature, step)]:
      if tensor['kv']:
        # KV cache shards are snapshot resources, not forward boundaries
        # (litert_kv_dump).
        kv_shards.setdefault((signature, step), []).append((ordinal, tensor))
        continue
      capture_key = f"dump:{ordinal}:{tensor['moment']}:{tensor['key']}"
      output_path = [
          'kwargs' if tensor['moment'] == 'pre' else 'output',
          tensor['name'],
      ]
      # Forward items address resources by capture_key (capture_telemetry.bind),
      # as attach_trace does.
      forward['inputs' if tensor['moment'] == 'pre' else 'outputs'].append(
          dict(
              key=capture_key,
              shape=tensor['shape'],
              dtype=tensor['dtype'],
              output_path=output_path,
          )
      )
      files[ordinal, tensor['key']] = tensor['path']
      resources.append(
          dict(
              format='safetensors',
              path='raw/' + tensor['path'].name,
              key=tensor['key'],
              shape=tensor['shape'],
              dtype=tensor['dtype'],
              sha256=file_digest(tensor['path']),
              forward_id=ordinal,
              phase=phase,
              step=step,
              runtime='LiteRT-LM',
              signature=signature,
              runtime_step=step,
              capture_key=capture_key,
              tensor_name=tensor['name'],
              when=tensor['moment'],
              scope='boundary',
              output_path=output_path,
              basis=BASIS,
          )
      )
    if phase == 'prefill':
      # The Runner's engine-tokenized template is the strongest Prefill identity
      # available.
      context = dict(
          prefill_chunk=prefill_index,
          **({'admitted_input_ids': admitted_ids} if admitted_ids else {}),
      )
      prefill_index += 1
    else:
      context = dict(released_prefix=released_ids[:decode_index])
      decode_index += 1
      decodes.append(forward)
    forward['input_identity'] = proof_digest(
        dict(
            basis='serialized_conversation',
            model_sha256=model_sha256,
            messages=job.get('messages', []),
            prompt=job.get('prompt', ''),
            **context,
        )
    )
    forward['comparison_basis'] = 'serialized conversation (inferred)'
    forward['comparison_eligible'] = True
    forwards.append(forward)
  status, reason, vocab = 'ok', None, None
  argmax, positions = [], []
  if not (
      checks['greedy_sampler']
      and checks['single_candidate']
      and checks['not_speculative']
  ):
    status, reason = (
        'unsupported',
        'requires greedy single-candidate non-speculative native generation',
    )
  else:
    expected = len(released_ids)
    checks['decode_count'] = (
        len(decodes) in (expected, expected + 1) and len(decodes) > 0
    )
    prefills = [f for f in forwards if f['phase'] == 'prefill']
    checks['prefill_precedes_decode'] = (
        bool(prefills)
        and bool(decodes)
        and max(f['step'] for f in prefills) <= decodes[0]['step']
    )
    try:
      for forward in decodes:
        logits = _last_position(
            _read(files[forward['forward_id'], 'post_logits'], 'post_logits')
        )
        if vocab is None:
          vocab = int(logits.shape[-1])
        elif vocab != int(logits.shape[-1]):
          raise ValueError('Decode logits change vocabulary size')
        argmax.append(int(np.asarray(logits).argmax()))
        positions.append(
            int(
                np.asarray(
                    _read(
                        files[forward['forward_id'], 'pre_input_pos'],
                        'pre_input_pos',
                    )
                ).reshape(-1)[0]
            )
        )
    except (KeyError, ValueError) as error:
      status, reason = (
          'mismatch',
          f'decode boundary tensors unavailable: {error}',
      )
    if status == 'ok':
      checks['input_pos_step'] = all(
          p == f['step'] - 1 for p, f in zip(positions, decodes)
      )
      checks['input_pos_contiguous'] = all(
          b - a == 1 for a, b in zip(positions, positions[1:])
      )
      checks['argmax_matches_released'] = argmax[:expected] == released_ids
      before, after = result.get('tokenCountBefore'), result.get('tokenCount')
      if (
          admitted_ids is not None
          and type(before) is int
          and type(after) is int
      ):
        # Every admitted token and every Decode invocation occupies one context
        # position.
        checks['input_count_reconciles'] = after - before == len(
            admitted_ids
        ) + len(decodes)
      failed = [name for name in CHECKS if checks[name] is False]
      if failed:
        status, reason = 'mismatch', 'failed checks: ' + ', '.join(failed)
  token_records, unreleased, unreleased_sample = [], None, None
  tokens = list(released or [])
  if status == 'ok':
    for k, forward in enumerate(decodes[: len(released_ids)]):
      consumer = decodes[k + 1]['forward_id'] if k + 1 < len(decodes) else None
      token_records.append(
          dict(
              forward_id=forward['forward_id'],
              candidate=0,
              output_index=k,
              token_ids=[released_ids[k]],
              consumption=[dict(consumed_by_forward_id=consumer)],
              sample_id=None,
              logits_stage='graph_output_before_sampling_constraints',
              basis=BASIS,
          )
      )
    if len(decodes) == len(released_ids) + 1:
      unreleased = argmax[-1]
      if stops and unreleased in stops:
        # The runtime sampled a configured stop token and filtered it from the
        # text; it is a real sample with its own logits, so it is kept as a
        # special token.
        token_records.append(
            dict(
                forward_id=decodes[-1]['forward_id'],
                candidate=0,
                output_index=len(released_ids),
                token_ids=[unreleased],
                consumption=[dict(consumed_by_forward_id=None)],
                sample_id=None,
                logits_stage='graph_output_before_sampling_constraints',
                basis=BASIS,
                released=False,
                stop=True,
            )
        )
        # `<turn|>` ends a turn in the rendered template too; a stop ID the run
        # admitted as a chat-template token is one, anything else (`<eos>`)
        # stays a special token.
        template_ids = {
            token['id']
            for token in input_tokens or []
            if token.get('kind') == 'template'
        }
        tokens.append(
            dict(
                id=unreleased,
                text=None,
                step=len(released_ids),
                released=False,
                stop=True,
                kind='template' if unreleased in template_ids else 'special',
            )
        )
        unreleased_sample = dict(id=unreleased, stop=True, recorded=True)
      else:
        unreleased_sample = dict(
            id=unreleased,
            stop=None if stops is None else False,
            recorded=False,
            reason='stop tokens not reported by the Runner'
            if stops is None
            else 'not a single-ID stop token',
        )
  else:
    for forward in decodes:
      forward.update(
          comparison_eligible=False,
          comparison_basis='input equivalence not established',
      )
  if vocab is not None:
    for forward in forwards:
      forward['vocab_identity'] = f'{model_sha256}:{vocab}'
  kv_status, kv_snapshots, kv_resources = attach_kv(
      capture,
      basis=BASIS,
      forwards=forwards,
      files=files,
      kv_shards=kv_shards,
      model=model,
      model_reason=model_reason,
      build=build,
      result=result,
      job=job,
      turn=turn,
      model_sha256=model_sha256,
      backend=index.get('backend_requested'),
      released_ids=released_ids,
      admitted_ids=admitted_ids,
      generation_ok=status == 'ok',
  )
  resources.extend(kv_resources)
  lookup = {(f['signature'], f['step']): f for f in forwards}
  for tensor in index.get('tensors', []):
    forward = lookup.get((tensor['signature'], tensor['step']))
    if forward is None:
      raise ValueError('Selected tap tensor has no dumped invocation')
    tensor.update(forward_id=forward['forward_id'], turn=turn)
  rendered = result.get('renderedInput')
  index.update(
      runtime='LiteRT-LM',
      export_scope='all',
      capture_scope=CAPTURE_SCOPE,
      forwards=forwards,
      resources=resources,
      token_records=token_records,
      kv_snapshots=kv_snapshots,
      tokens=tokens,
      **(
          {'admitted_input_ids': admitted_ids, 'input_tokens': input_tokens}
          if admitted_ids
          else {}
      ),
      **({'serialized_input': rendered} if isinstance(rendered, str) else {}),
      generation=dict(
          status='completed',
          basis=BASIS,
          released_token_ids=released_ids,
          prefill_invocations=prefill_index,
          decode_invocations=len(decodes),
          unreleased_argmax=unreleased,
          unreleased_sample=unreleased_sample,
          stop_token_ids=sorted(stops) if stops is not None else None,
          vocab_size=vocab,
          token_count_basis=(
              'released native tokens; the trailing unreleased sample is not a'
              ' Turn token'
          ),
      ),
      inference=dict(
          basis=BASIS,
          status=status,
          reason=reason,
          checks=checks,
          sampler=sampler,
          speculative_decoding=result.get('speculativeDecodingEnabled'),
          kv=kv_status,
      ),
  )
  return index


def verify_inferred_index(index, root):
  """Re-check every inferred binding against the saved shards.

  Runs before publication.
  """
  root = Path(root).resolve()
  if (
      index.get('capture_scope') != CAPTURE_SCOPE
      or index.get('runtime') != 'LiteRT-LM'
  ):
    raise ValueError('Not an inferred native capture')
  forwards = {f['forward_id']: f for f in index.get('forwards', [])}
  resources = {}
  for resource in index.get('resources', []):
    path = (root / resource['path']).resolve()
    if (
        Path(resource['path']).is_absolute()
        or not path.is_relative_to(root)
        or path.suffix != '.safetensors'
    ):
      raise ValueError('Inferred resource path escapes the capture')
    with safe_open(path, framework='numpy') as shard:
      metadata = shard.metadata() or {}
      if (
          metadata.get('signature') != resource.get('signature')
          or metadata.get('step') != str(resource.get('runtime_step'))
          or resource.get('key') not in shard.keys()
      ):
        raise ValueError('Inferred resource differs from its raw header')
    if resource.get('forward_id') not in forwards:
      raise ValueError('Inferred resource refers to an unknown forward')
    resources[resource['forward_id'], resource['key']] = path
  decodes = sorted(
      (f for f in forwards.values() if f['phase'] == 'decode'),
      key=lambda f: f['forward_id'],
  )
  positions = []
  for forward in decodes:
    pos = resources.get((forward['forward_id'], 'pre_input_pos'))
    if pos is None:
      raise ValueError('Inferred decode forward lacks its input position')
    position = int(np.asarray(_read(pos, 'pre_input_pos')).reshape(-1)[0])
    if position != forward['step'] - 1:
      raise ValueError('Inferred decode position differs from its runtime step')
    positions.append(position)
  if any(b - a != 1 for a, b in zip(positions, positions[1:])):
    raise ValueError('Inferred decode positions are not contiguous')
  for record in index.get('token_records', []):
    forward = forwards.get(record.get('forward_id'))
    logits = resources.get((record.get('forward_id'), 'post_logits'))
    if (
        forward is None
        or forward['phase'] != 'decode'
        or logits is None
        or len(record.get('token_ids', [])) != 1
    ):
      raise ValueError('Inferred token record has no Decode logits')
    if (
        int(np.asarray(_last_position(_read(logits, 'post_logits'))).argmax())
        != record['token_ids'][0]
    ):
      raise ValueError('Inferred token differs from its logits argmax')
  verify_kv(index, root, forwards, resources)
  return True
