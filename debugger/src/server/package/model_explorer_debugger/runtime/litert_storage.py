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

"""Native KV bytes require a storage contract before logical interpretation."""

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from safetensors import safe_open

from ..fsutil import file_digest

REFERENCE = dict(
    source_revision='761d99cb90e20c67efcb3fe1119a60c92381bd1a',
    source_file=(
        'ml_drift_delegate/delegate/composite/sdpa_transposed_kernel_test.cc'
    ),
    source_file_sha256=(
        '1622ff49627f8b113990f46f0e816d332d295f87004aae6be79a810e60cd8815'
    ),
)
KEY_LAYOUT = ['batch', 'kv_head', 'sequence', 'head_dim']
VALUE_LAYOUT = ['batch', 'kv_head', 'head_dim', 'sequence']
_VALIDATED = OrderedDict()
_TRACES = OrderedDict()
# A Server-normalized dump shard: bound to the untouched dump, not to a runtime
# trace event.
DUMP_BINDING = 'server_dump_normalization'
# How litert_dump_inference labels evidence it inferred from a dump; dump-bound
# storage requires it.
INFERRED_BASIS = 'inferred_greedy_argmax'


def _digest(value):
  return hashlib.sha256(
      json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
  ).hexdigest()


def _signature(path):
  value = path.stat()
  return (
      str(path),
      value.st_ino,
      value.st_size,
      value.st_mtime_ns,
      value.st_ctime_ns,
  )


def _proof_key(record):
  return _digest({
      key: record.get(key)
      for key in (
          'path',
          'key',
          'sha256',
          'shape',
          'dtype',
          'storage',
          'storage_source',
          'storage_validation',
          'native_trace',
          'model_source',
          'quantization',
          'model_sha256',
          'model_section_sha256',
          'dequantization',
          'layout',
      )
  })


def _remember(cache, key, value):
  cache[key] = value
  cache.move_to_end(key)
  while len(cache) > 4096:
    cache.popitem(last=False)


def _path(root, relative):
  if not isinstance(relative, str) or Path(relative).is_absolute():
    raise ValueError('Invalid native storage evidence path')
  path = (Path(root) / relative).resolve()
  if not path.is_relative_to(Path(root).resolve()):
    raise ValueError('Native storage evidence path escapes capture')
  return path


def _check(condition, reason):
  if not condition:
    raise NativeStorageError(reason)


class NativeStorageError(ValueError):
  status = 'storage_contract_unavailable'

  def __init__(self, reason):
    self.reason = reason
    super().__init__(self.status)


def native_kv(record):
  return record.get('runtime') == 'LiteRT-LM' and record.get('scope') == 'kv'


def require_storage(record):
  """Legacy CPU host tensors are supported; GPU layouts are never inferred."""
  if not native_kv(record):
    return
  if record.get('storage') is not None:
    storage = record['storage']
    logical = storage.get('logical', {}) if isinstance(storage, dict) else {}
    model_source = (
        storage.get('model_source') if isinstance(storage, dict) else None
    )
    quant = (
        model_source.get('quantization')
        if isinstance(model_source, dict)
        else None
    )
    _check(
        record.get('layout') in (None, logical.get('layout')),
        'Native comparison layout differs from its validated logical storage.',
    )
    if record.get('dequantization') is not None:
      _check(
          isinstance(quant, dict)
          and record['dequantization']
          == {key: quant.get(key) for key in ('scale', 'zero_point')},
          'Native comparison quantization differs from its verified model'
          ' units.',
      )
    dependencies = _VALIDATED.get(_proof_key(record))
    try:
      if dependencies and all(
          _signature(Path(value[0])) == value for value in dependencies
      ):
        return
    except OSError:
      pass
    raise NativeStorageError(
        'Native logical KV storage has not been validated against its trace,'
        ' source bytes and model metadata, or its evidence changed.'
    )
  backends = {
      str(record[key]).upper()
      for key in (
          'backend_requested',
          'backend_effective',
          'backend_configured',
      )
      if record.get(key)
  }
  if backends == {'CPU'}:
    return
  detail = (
      'GPU KV capture has no verified logical storage conversion.'
      if 'GPU' in backends
      else (
          'Native KV capture has no unambiguous CPU host storage or verified'
          ' logical storage conversion.'
      )
  )
  raise NativeStorageError(
      detail
      + ' Raw bytes are retained; model dtype, shape and quantization do not'
      ' establish their physical representation.'
  )


def prepare_saved_resources(resources, runs, *, root=None, conversation=()):
  """Annotate the read-time view of legacy captures.

  Saved files are never rewritten.
  """
  by_run = {row['id']: row for row in runs}
  for record in resources:
    if not native_kv(record):
      continue
    run = by_run.get(record.get('run'), {})
    provenance = run.get('provenance') or {}
    if run.get('backend'):
      record['backend_configured'] = run['backend']
    if not record.get('backend_requested'):
      record['backend_requested'] = run.get('backend') or provenance.get(
          'backend_requested'
      )
    if not record.get('backend_effective'):
      record['backend_effective'] = provenance.get('backend_effective')
    if not record.get('native_trace'):
      captured = next(
          (
              row
              for row in conversation
              if row.get('run') == record.get('run')
              and row.get('turn') == record.get('turn')
          ),
          {},
      )
      if captured.get('native_trace_path'):
        record['native_trace'] = dict(
            path=captured['native_trace_path'],
            sha256=captured.get('native_trace_sha256'),
        )
    try:
      if record.get('storage') is not None and root is not None:
        validate_storage(record, root)
      require_storage(record)
    except (
        NativeStorageError,
        ValueError,
        OSError,
        KeyError,
        TypeError,
    ) as error:
      record.update(
          state='unavailable',
          storage_comparison_status='storage_contract_unavailable',
          storage_comparison_reason=getattr(error, 'reason', str(error)),
      )


def _trace_event(record, root):
  trace = record.get('native_trace') or {}
  path = _path(root, trace.get('path'))
  signature = _signature(path)
  cache_key = signature, trace.get('sha256')
  events = _TRACES.get(cache_key)
  if events is None:
    _check(
        file_digest(path) == trace.get('sha256'),
        'Native storage trace checksum mismatch.',
    )
    events = {}
    for line in path.read_text().splitlines():
      row = json.loads(line)
      if row.get('event') == 'tensor' and row.get('status') == 'stored':
        key = row.get('invocation_id'), row.get('moment'), row.get('key')
        _check(key not in events, 'Ambiguous native tensor storage event.')
        events[key] = row
    _remember(_TRACES, cache_key, events)
  event = events.get(
      (record.get('forward_id'), record.get('when'), record.get('key'))
  )
  _check(
      event is not None
      and event.get('storage') == record.get('storage')
      and event.get('signature') == record.get('signature')
      and event.get('tensor_name') == record.get('tensor_name'),
      'Native logical storage differs from its original tensor event.',
  )
  return signature


def webgpu_logical_blocks(kind, shape, block=None):
  """Yield (start, stop, flat source indices) per block of logical positions.

  Indices address the pinned WebGPU KV layout.
  """
  heads = shape[1]
  sequence = shape[2 if kind == 'key' else 3]
  channels = shape[3 if kind == 'key' else 2]
  head = np.arange(heads)[:, None, None]
  channel = np.arange(channels)[None, None, :]
  block = block or max(1, 262144 // (heads * channels))
  for start in range(0, sequence, block):
    stop = min(sequence, start + block)
    position = np.arange(start, stop)[None, :, None]
    if kind == 'key':
      linear = head * (channels // 4) + channel // 4
      indices = (
          ((linear % heads) * sequence + position) * channels
          + (linear // heads) * 4
          + channel % 4
      )
    else:
      rhs = (
          (head * (sequence // 4) + position // 4) * (channels // 4)
          + channel // 4
      ) * 4 + position % 4
      temporary = rhs // channels
      indices = (
          ((temporary % heads) * channels + rhs % channels) * sequence
          + (temporary // heads) * 4
          + channel % 4
      )
    yield start, stop, indices


def _unsigned_bytes(raw):
  return np.ascontiguousarray(np.asarray(raw)).reshape(-1).view(np.uint8)


def normalize_webgpu_kv(raw, kind, shape):
  """Returns the logical int8 tensor of one WebGPU KV host download.

  The tensor is in its declared layout; the download holds unsigned bytes.
  """
  raw = _unsigned_bytes(raw)
  if (
      kind not in ('key', 'value')
      or len(shape) != 4
      or shape[0] != 1
      or raw.size != math.prod(shape)
  ):
    raise ValueError('WebGPU KV bytes differ from their logical shape')
  logical = np.empty(shape, dtype=np.int8)
  for start, stop, indices in webgpu_logical_blocks(kind, shape):
    block = (raw[indices].astype(np.int16) - 128).astype(np.int8)
    if kind == 'key':
      logical[0, :, start:stop, :] = block
    else:
      logical[0, :, :, start:stop] = block.transpose(0, 2, 1)
  return logical


def _validate_values(raw, logical, kind, shape):
  """Checks every byte with bounded vectorized inverse indices.

  The inverse indices come from the pinned reference.
  """
  raw = _unsigned_bytes(raw)
  for start, stop, indices in webgpu_logical_blocks(kind, shape):
    expected = (raw[indices].astype(np.int16) - 128).astype(np.int8)
    actual = (
        logical[:, :, start:stop, :]
        if kind == 'key'
        else logical[:, :, :, start:stop]
    )
    if kind == 'value':
      expected = expected.transpose(0, 2, 1)
    _check(
        np.array_equal(actual, expected[None]),
        'Normalized GPU KV values differ from the recorded source conversion.',
    )


def validate_storage(
    record, root, *, event=None, descriptor=None, model_section_sha256=None
):
  """Validate source/header/trace/model binding.

  Imports also prove all converted values.
  """
  storage = record.get('storage')
  declared_source = (
      storage.get('model_source') if isinstance(storage, dict) else None
  )
  # The normalized bytes belong to the text section. Runtime provenance binds
  # the outer container separately, and publishing must not change this proof.
  if isinstance(declared_source, dict) and not record.get(
      'model_section_sha256'
  ):
    record['model_section_sha256'] = declared_source.get('section_sha256')
  if event is None and descriptor is None:
    dependencies = _VALIDATED.get(_proof_key(record))
    try:
      if dependencies and all(
          _signature(Path(value[0])) == value for value in dependencies
      ):
        return record
    except OSError:
      # The validated files moved (a retained shard in a newer capture):
      # validate again below.
      pass
  _check(
      isinstance(storage, dict)
      and storage.get('version') == 1
      and storage.get('representation') == 'logical_tensor',
      'Unsupported native KV storage contract.',
  )
  logical = storage.get('logical', {})
  source = storage.get('source', {})
  conversion = storage.get('conversion', {})
  _check(
      all(isinstance(value, dict) for value in (logical, source, conversion)),
      'Invalid native storage declaration fields.',
  )
  # A Server-normalized dump shard binds to the untouched dump shard, the pinned
  # LiteRT revision and the Runner's backend evidence instead of a runtime trace
  # event.
  dump_bound = storage.get('binding') == DUMP_BINDING
  if dump_bound:
    evidence = (
        storage.get('evidence')
        if isinstance(storage.get('evidence'), dict)
        else {}
    )
    _check(
        record.get('basis') == INFERRED_BASIS
        and record.get('native_trace') is None,
        'Server-normalized KV storage requires inferred native evidence without'
        ' a trace.',
    )
    _check(
        evidence.get('litert_revision') == REFERENCE['source_revision']
        and evidence.get('backend_effective') == 'GPU'
        and isinstance(evidence.get('webgpu_library'), dict)
        and isinstance(evidence['webgpu_library'].get('sha256'), str),
        'Server-normalized KV storage lacks its pinned runtime revision and'
        ' WebGPU backend evidence.',
    )
  kind = record.get('kind') or conversion.get('kind')
  layout = KEY_LAYOUT if kind == 'key' else VALUE_LAYOUT
  shape = logical.get('shape')
  _check(
      kind in ('key', 'value')
      and logical.get('dtype') == 'I8'
      and record.get('dtype') == 'int8'
      and logical.get('layout') == layout
      and shape == record.get('shape')
      and isinstance(shape, list)
      and len(shape) == 4
      and all(type(n) is int and n > 0 for n in shape)
      and math.prod(shape) <= 128_000_000,
      'Native logical KV dtype, shape or layout is unsupported.',
  )
  model_source = storage.get('model_source')
  expected_source = (
      deepcopy(descriptor['source'])
      if descriptor
      else deepcopy(record.get('model_source'))
  )
  quant = (
      descriptor.get('quantization')
      if descriptor
      else record.get('quantization')
  )
  _check(
      isinstance(expected_source, dict)
      and model_source == {**expected_source, 'quantization': quant}
      and expected_source.get('signature') == record.get('signature')
      and expected_source.get('composite') == 'odml.cache_update'
      and isinstance(expected_source.get('section_sha256'), str)
      and len(expected_source['section_sha256']) == 64,
      'Native storage does not match the verified model cache-update source.',
  )
  _check(
      isinstance(quant, dict)
      and type(quant.get('scale')) in (int, float)
      and math.isfinite(quant['scale'])
      and quant['scale'] > 0
      and type(quant.get('zero_point')) is int
      and -128 <= quant['zero_point'] <= 127
      and type(quant.get('quantized_dimension')) is int
      and 0 <= quant['quantized_dimension'] < 4,
      'Native logical I8 KV requires verified scalar model quantization.',
  )
  if descriptor:
    axis = layout.index('sequence')
    _check(
        layout == descriptor['layout']
        and all(
            shape[i] == descriptor['model_shape'][i]
            for i in range(4)
            if i != axis
        ),
        'Native storage differs from model cache dimensions.',
    )
    _check(
        model_section_sha256 in (None, expected_source['section_sha256']),
        'Native storage section differs from the exact imported model section.',
    )
    record.update(
        model_source=deepcopy(descriptor['source']),
        quantization=deepcopy(quant),
    )
  _check(
      record.get('model_section_sha256') == expected_source['section_sha256'],
      'Native logical storage is missing its exact model section identity.',
  )
  _check(
      conversion.get('version') == 1
      and (
          conversion.get('id') == 'host_identity'
          or (
              conversion.get('kind') == kind
              and all(
                  conversion.get(key) == value
                  for key, value in REFERENCE.items()
              )
          )
      ),
      'Native storage converter is not the pinned verified implementation.',
  )
  path = _path(root, record.get('path'))
  dependencies = []
  if not dump_bound:
    dependencies.append(_trace_event(record, root))
  if event is not None:
    _check(
        event.get('storage') == storage, 'Native storage event/header mismatch.'
    )
  source_name = source.get('path')
  _check(
      isinstance(source_name, str)
      and Path(source_name).name == source_name
      and source_name.endswith('.safetensors'),
      'Native source artifact must be a flat Safetensors path.',
  )
  source_record = record.get('storage_source')
  importing = source_record is None
  if importing:
    source_path = path.parent / source_name
    source_record = dict(
        format='safetensors',
        path=str(source_path.relative_to(Path(root).resolve())),
        key=source.get('key'),
        dtype='uint8' if source.get('dtype') == 'U8' else 'int8',
        shape=source.get('shape'),
    )
  else:
    source_path = _path(root, source_record.get('path'))
  _check(
      source_record.get('key') == source.get('key')
      and source_record.get('shape') == source.get('shape')
      and source_record.get('dtype')
      == ('uint8' if source.get('dtype') == 'U8' else 'int8'),
      'Native source resource differs from its storage declaration.',
  )
  with (
      safe_open(path, framework='numpy') as normalized,
      safe_open(source_path, framework='numpy') as original,
  ):
    for shard, key, expected_dtype, expected_shape in (
        (normalized, record['key'], 'I8', shape),
        (original, source.get('key'), source.get('dtype'), source.get('shape')),
    ):
      metadata = shard.metadata() or {}
      if dump_bound and shard is original:
        _check(
            key in shard.keys()
            and 'storage' not in metadata
            and metadata.get('signature') == record.get('signature')
            and metadata.get('step') == str(record.get('runtime_step')),
            'Server-normalized KV storage must bind to the untouched runtime'
            ' dump shard.',
        )
      else:
        _check(
            key in shard.keys()
            and json.loads(metadata.get('storage', 'null')) == storage,
            'Native source and logical headers must contain the identical'
            ' storage contract.',
        )
      tensor = shard.get_slice(key)
      _check(
          tensor.get_dtype() == expected_dtype
          and tensor.get_shape() == expected_shape,
          'Native storage contract differs from actual Safetensors metadata.',
      )
    if conversion.get('id') == 'webgpu_kv_u8_to_logical_i8':
      geometry = (
          source.get('buffer_type') == 'WebGpuBuffer'
          and shape[0] == 1
          and shape[2] % 4 == shape[3] % 4 == 0
          and conversion.get('source_function')
          == ('RearrangeK' if kind == 'key' else 'RearrangeV')
          and source_path != path
      )
      if dump_bound:
        _check(
            geometry
            and source.get('dtype') == 'I8'
            and source.get('shape') == shape
            and source.get('key') == record['key']
            and source.get('origin') == 'runtime_dump',
            'Unsupported dump-bound WebGPU KV source or converter geometry.',
        )
      else:
        _check(
            geometry
            and source.get('dtype') == 'U8'
            and source.get('shape') == [math.prod(shape)]
            and source.get('key') == 'source_' + record['key'],
            'Unsupported WebGPU KV backing buffer or converter geometry.',
        )
      _validate_values(
          original.get_tensor(source.get('key')),
          normalized.get_slice(record['key']),
          kind,
          shape,
      )
    elif conversion.get('id') == 'host_identity':
      _check(
          not dump_bound
          and source_path == path
          and source.get('key') == record['key']
          and source.get('dtype') == 'I8'
          and source.get('shape') == shape
          and source.get('buffer_type') == 'HostMemory',
          'Native CPU identity storage must refer to the same host tensor.',
      )
    else:
      raise NativeStorageError('Unknown native KV storage conversion.')
  source_hash = file_digest(source_path)
  normalized_hash = file_digest(path)
  _check(
      normalized_hash == record.get('sha256'),
      'Normalized native KV checksum mismatch.',
  )
  proof = dict(
      version=2,
      method='all_values_exact'
      if conversion['id'] != 'host_identity'
      else 'identity',
      storage_sha256=_digest(storage),
      normalized_sha256=normalized_hash,
      source_sha256=source_hash,
      model_section_sha256=expected_source['section_sha256'],
      **({'binding': DUMP_BINDING} if dump_bound else {}),
  )
  if not importing:
    saved_proof = record.get('storage_validation')
    valid_proof = saved_proof == proof
    if isinstance(saved_proof, dict) and saved_proof.get('version') == 1:
      # V1 accidentally used either a manifest section hash or an already
      # published container hash. Accept only those exact known identities,
      # after independently checking every source/header/trace/value above.
      legacy_hash = saved_proof.get('model_sha256')
      known_identities = {expected_source['section_sha256']}
      if (
          isinstance(record.get('model_sha256'), str)
          and len(record['model_sha256']) == 64
      ):
        known_identities.add(record['model_sha256'])
      legacy = {
          key: value
          for key, value in proof.items()
          if key != 'model_section_sha256'
      }
      legacy.update(version=1, model_sha256=legacy_hash)
      valid_proof = legacy_hash in known_identities and saved_proof == legacy
    _check(
        source_record.get('sha256') == source_hash and valid_proof,
        'Native source checksum or recorded conversion proof differs from its'
        ' immutable evidence.',
    )
    proof = deepcopy(
        saved_proof
    )  # Keep immutable saved V1 declarations and stable cache keys.
  source_record['sha256'] = source_hash
  record.update(storage_source=source_record, storage_validation=proof)
  dependencies.extend((_signature(path), _signature(source_path)))
  _remember(_VALIDATED, _proof_key(record), dependencies)
  return record
