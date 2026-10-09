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

"""LiteRT tap verification and offline replay evidence validation.

The Server calls verify_taps only. token_replay_evidence, token_decode_evidence,
replay_text_evidence and validate_token_replay are the reference implementation
of
the token_replay_proof a Runner would have to record for explicit cross-runtime
comparison; no Runner in this repository produces it yet (see docs/FEATURES.md).
"""

import json
import mmap
from pathlib import Path


def verify_taps(model_path, manifest_path):
  """Validate the declared signature/op/output against the actual container.

  The old tap manifest hash identifies the TFLite section, not the enclosing
  .litertlm. Never compare that hash with a container hash or simply overwrite
  it.
  """
  from debugger_tap import tflite_generated as fb

  manifest = json.loads(Path(manifest_path).read_text())
  verified = []
  with Path(model_path).open('rb') as file:
    data = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    cursor = 0
    while (position := data.find(b'TFL3', cursor)) >= 0:
      cursor = position + 4
      try:
        model = fb.Model.GetRootAs(data, position - 4)
        signatures = {
            model.SignatureDefs(i)
            .SignatureKey()
            .decode(): model.SignatureDefs(i)
            for i in range(model.SignatureDefsLength())
        }
      except (ValueError, IndexError, TypeError, UnicodeDecodeError):
        continue
      for tap in manifest['taps']:
        sig = signatures.get(tap['signature'])
        if sig is None:
          continue
        graph = model.Subgraphs(sig.SubgraphIndex())
        tensor_id = graph.Operators(tap['op']).Outputs(tap['output'])
        tensor = graph.Tensors(tensor_id)
        outputs = {
            sig.Outputs(i).Name().decode(): sig.Outputs(i).TensorIndex()
            for i in range(sig.OutputsLength())
        }
        actual = dict(
            subgraph=sig.SubgraphIndex(),
            tensor=tensor_id,
            shape=[tensor.Shape(i) for i in range(tensor.ShapeLength())],
            tensor_type=tensor.Type(),
            tensor_name=tensor.Name().decode(),
        )
        if (
            any(actual[k] != tap[k] for k in actual)
            or outputs.get(tap['output_name']) != tensor_id
        ):
          raise ValueError(
              'Tap manifest does not match this model: ' + tap['signature']
          )
        verified.append({**tap, 'section_offset': position - 4})
    data.close()
  if len(verified) != len(manifest['taps']):
    raise ValueError('Tap signature missing or ambiguous in model')
  return {**manifest, 'taps': verified}


def validate_token_replay(replay):
  """The first replay contract is one unpadded text batch and one graph run."""
  if not isinstance(replay, dict):
    raise ValueError('token_replay must be an object')
  ids = replay.get('token_ids')
  if (
      not isinstance(ids, list)
      or not 2 <= len(ids) <= 128
      or any(
          type(token) is not int or not 0 <= token <= 2147483647
          for token in ids
      )
  ):
    raise ValueError(
        'token_replay.token_ids must contain 2 to 128 nonnegative int32 values'
    )
  return list(ids)


def token_replay_evidence(directory, capture, actual_ids, replay, provenance):
  """Verify actual graph inputs and retain the native admission-buffer readback.

  LiteRT holds the final admitted token pending; its Prefill graph processes
  only the preceding tokens. This is distinct from its current-step counter.
  The supported mask here is the CPU boolean, unpadded first text Prefill.
  """
  import hashlib
  import numpy as np
  from safetensors import safe_open
  from safetensors.numpy import save_file
  from ..fsutil import file_digest

  if actual_ids != validate_token_replay(replay):
    raise ValueError('Native admitted IDs differ from token replay request')
  directory, capture = Path(directory).resolve(), Path(capture).resolve()
  if not capture.is_relative_to(directory):
    raise ValueError('Replay capture must stay inside the run')
  valid_length = len(actual_ids) - 1
  resources, tensors = {}, {}
  for candidate in sorted(capture.glob('*.safetensors')):
    path = candidate.resolve()
    if not path.is_relative_to(capture):
      raise ValueError('Replay tensor path escapes capture')
    with safe_open(path, framework='numpy') as file:
      metadata = file.metadata() or {}
      if not str(metadata.get('signature', '')).startswith('prefill_'):
        continue
      for kind, key in [
          ('position_ids', 'pre_input_pos'),
          ('attention_mask', 'pre_mask'),
      ]:
        if key not in file.keys():
          continue
        if kind in resources:
          raise ValueError(
              'Replay requires exactly one Prefill graph input set'
          )
        if metadata.get('step') != str(len(actual_ids)):
          raise ValueError(
              'Replay graph step differs from admitted token count'
          )
        value = file.get_tensor(key)
        tensors[kind] = value
        resources[kind] = {
            'format': 'safetensors',
            'path': str(path.relative_to(directory)),
            'key': key,
            'shape': list(value.shape),
            'dtype': str(value.dtype),
            'file_sha256': file_digest(path),
            'tensor_sha256': hashlib.sha256(value.tobytes()).hexdigest(),
            'signature': metadata['signature'],
            'step': int(metadata['step']),
            'valid_length': valid_length,
        }
  if set(tensors) != {'position_ids', 'attention_mask'}:
    raise ValueError('Replay is missing actual Prefill position or mask inputs')
  positions, mask = tensors['position_ids'], tensors['attention_mask']
  if (
      positions.dtype != np.int32
      or positions.ndim != 1
      or positions.size < valid_length
      or not np.array_equal(positions[:valid_length], np.arange(valid_length))
      or np.any(positions[valid_length:] != 0)
  ):
    raise ValueError(
        'Replay graph position IDs do not match the first token range'
    )
  if (
      mask.dtype != np.bool_
      or mask.ndim != 4
      or mask.shape[:2] != (1, 1)
      or mask.shape[2] != positions.size
      or mask.shape[3] < valid_length
  ):
    raise ValueError(
        'Replay supports only a captured boolean first-Prefill attention mask'
    )
  expected = np.zeros(mask.shape, dtype=np.bool_)
  expected[0, 0, :valid_length, :valid_length] = np.tri(
      valid_length, dtype=np.bool_
  )
  if not np.array_equal(mask, expected):
    raise ValueError(
        'Replay graph attention mask is not the declared first-Prefill causal'
        ' mask'
    )
  if (
      resources['position_ids']['signature']
      != resources['attention_mask']['signature']
  ):
    raise ValueError('Replay graph input signatures differ')
  raw = directory / 'raw'
  raw.mkdir(exist_ok=True)
  input_path = raw / 'replay-inputs.safetensors'
  save_file(
      {'input_ids': np.asarray([actual_ids], dtype=np.int32)},
      str(input_path),
      metadata={
          'source': 'native preprocessed InputText TensorBuffer readback'
      },
  )
  proof = {
      'format_version': 1,
      'mode': 'first_prefill_exact_token_ids',
      'actual_token_ids': actual_ids,
      'accepted_token_count': len(actual_ids),
      'graph_valid_length': valid_length,
      'graph_token_ids': actual_ids[:-1],
      'pending_token_id': actual_ids[-1],
      'pending_token_position': valid_length,
      'input_resource': {
          'format': 'safetensors',
          'path': str(input_path.relative_to(directory)),
          'key': 'input_ids',
          'shape': [1, len(actual_ids)],
          'dtype': 'int32',
          'file_sha256': file_digest(input_path),
      },
      'graph_inputs': resources,
      'requested_source': {
          key: value for key, value in replay.items() if key != 'token_ids'
      },
      'source_scope': (
          'requester-supplied token source; not evidence of LiteRT weight'
          ' lineage'
      ),
      'weight_equivalence': 'unverified',
      'quantization_profile': 'unknown',
      'native': provenance,
  }
  proof['proof_sha256'] = hashlib.sha256(
      json.dumps(
          proof, sort_keys=True, separators=(',', ':'), allow_nan=False
      ).encode()
  ).hexdigest()
  return proof


def token_decode_evidence(directory, capture, prefill_proof, actual_ids):
  """Verify true Decode graph steps after the saved explicit-ID Prefill."""
  import hashlib
  import numpy as np
  from safetensors import safe_open
  from ..fsutil import file_digest

  directory, capture = Path(directory).resolve(), Path(capture).resolve()
  if not capture.is_relative_to(directory):
    raise ValueError('Decode capture must stay inside the run')
  if (
      not isinstance(actual_ids, list)
      or not 1 <= len(actual_ids) <= 2
      or actual_ids
      != prefill_proof.get('requested_source', {}).get('decode_token_ids')
  ):
    raise ValueError('Native Decode output IDs differ from the replay request')
  admitted = prefill_proof['actual_token_ids']
  inputs = [admitted[-1], *actual_ids[:-1]]
  executions = []
  for index, (input_token, emitted) in enumerate(zip(inputs, actual_ids)):
    position = len(admitted) - 1 + index
    # The graph callback runs before the next sampled token advances step.
    step = len(admitted) + index
    resources, values = {}, {}
    for name, key in [
        ('position_ids', 'pre_input_pos'),
        ('attention_mask', 'pre_mask'),
    ]:
      path = (capture / f'decode_{key}_step_{step}.safetensors').resolve()
      if not path.is_relative_to(capture):
        raise ValueError('Decode tensor path escapes capture')
      with safe_open(path, framework='numpy') as file:
        meta = file.metadata() or {}
        if meta.get('signature') != 'decode' or meta.get('step') != str(step):
          raise ValueError('Fixed Decode metadata does not match execution')
        value = file.get_tensor(key)
      values[name] = value
      resources[name] = {
          'format': 'safetensors',
          'path': str(path.relative_to(directory)),
          'key': key,
          'shape': list(value.shape),
          'dtype': str(value.dtype),
          'file_sha256': file_digest(path),
          'tensor_sha256': hashlib.sha256(value.tobytes()).hexdigest(),
      }
    positions, mask = values['position_ids'], values['attention_mask']
    if (
        positions.dtype != np.int32
        or positions.shape != (1,)
        or int(positions[0]) != position
    ):
      raise ValueError(
          'Fixed Decode actual position differs from token sequence'
      )
    if (
        mask.dtype != np.bool_
        or mask.ndim != 4
        or mask.shape[:3] != (1, 1, 1)
        or mask.shape[3] <= position
    ):
      raise ValueError('Fixed Decode requires a captured boolean causal mask')
    expected = np.zeros(mask.shape, dtype=np.bool_)
    expected[0, 0, 0, : position + 1] = True
    if not np.array_equal(mask, expected):
      raise ValueError('Fixed Decode actual mask differs from token context')
    executions.append({
        'signature': 'decode',
        'step': step,
        'input_token_id': input_token,
        'input_position': position,
        'emitted_token_id': emitted,
        'processed_token_count_before': position,
        'processed_token_count_after': position + 1,
        'graph_inputs': resources,
    })
  proof = dict(prefill_proof)
  proof.pop('proof_sha256', None)
  proof['decode'] = {
      'sampler_policy': 'fixed_token_sequence_constraint',
      'model_computation_modified': False,
      'actual_output_token_ids': actual_ids,
      'executions': executions,
      'pending_token_id': actual_ids[-1],
      'pending_token_position': len(admitted) + len(actual_ids) - 1,
      'processed_token_count': len(admitted) + len(actual_ids) - 1,
  }
  proof['proof_sha256'] = hashlib.sha256(
      json.dumps(
          proof, sort_keys=True, separators=(',', ':'), allow_nan=False
      ).encode()
  ).hexdigest()
  return proof


def replay_text_evidence(
    engine, actual_ids, *, trace_path=None, native_library_sha256=None
):
  """Prefer observed native callback text; label tokenizer-only projections.

  A callback chunk may contain multiple token IDs. Its whole text is usable
  as output, but assigning that string to each ID would invent token text.
  Only those per-token displays use the same native engine's tokenizer.
  """
  from ..fsutil import file_digest

  rows, observed_ids = [], []
  trace = Path(trace_path) if trace_path else None
  if trace is not None:
    for line in trace.read_text().splitlines():
      if not line.strip():
        continue
      row = json.loads(line)
      candidates, texts = row.get('token_ids', []), row.get('texts', [])
      if not candidates:
        continue
      if (
          len(candidates) != 1
          or not isinstance(candidates[0], list)
          or any(type(token) is not int for token in candidates[0])
      ):
        raise ValueError(
            'Replay text trace requires one native token candidate'
        )
      ids = candidates[0]
      if not ids:
        continue
      text = (
          texts[0]
          if isinstance(texts, list)
          and len(texts) == 1
          and isinstance(texts[0], str)
          else None
      )
      rows.append((ids, text))
      observed_ids.extend(ids)
    if observed_ids != actual_ids:
      raise ValueError(
          'Replay callback token IDs differ from verified native Decode output'
      )
  if not actual_ids:
    output, tokens, source = '', [], 'not_generated'
  elif trace is not None and rows and all(text is not None for _, text in rows):
    output, source = ''.join(text for _, text in rows), 'native_decode_callback'
    tokens = []
    for ids, text in rows:
      for token in ids:
        single = len(ids) == 1
        tokens.append({
            'id': token,
            'step': len(tokens),
            'text': text if single else engine.detokenize([token]),
            'text_source': (
                'native_decode_callback'
                if single
                else 'native_tokenizer_detokenize'
            ),
        })
  else:
    output, source = (
        engine.detokenize(actual_ids),
        'native_tokenizer_detokenize',
    )
    tokens = [
        {
            'id': token,
            'step': index,
            'text': engine.detokenize([token]),
            'text_source': 'native_tokenizer_detokenize',
        }
        for index, token in enumerate(actual_ids)
    ]
  return {
      'output': output,
      'tokens': tokens,
      'text_source': source,
      'text_evidence': {
          'format_version': 1,
          'source': source,
          'token_ids': list(actual_ids),
          'native_library_sha256': native_library_sha256,
          'trace_sha256': file_digest(trace) if trace is not None else None,
          'projection_scope': (
              'display only; token IDs come from verified native Decode output'
          ),
      },
  }
