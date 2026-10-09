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

"""Bounded reads of recorded KV coordinates, with independently validated proof.

Snapshot metadata never loads tensor values. Explicit pairs retain the existing
proof validator, within a conservative memory budget; validated reports are
cached only while every registered file and the session binding are unchanged.
"""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import threading

import ml_dtypes  # Registers bfloat16 for NumPy / Safetensors.
import numpy as np
from safetensors import SafetensorError, safe_open

from .fsutil import file_digest
from .kv_analysis import (
    LAYOUT,
    _append_snapshot,
    _capture_snapshot,
    _context,
    _evidence_size,
    _resource_structure,
    _row,
    _snapshot_coordinates,
    _token_structure,
)
from .kv_storage import dequantization, storage_view
from .runtime.litert_storage import NativeStorageError, require_storage

MAX_SLICE_ELEMENTS = 262_144
MAX_PROOF_ELEMENTS = 8_000_000
MAX_PROOF_FILES = 4096
_CACHE_ENTRIES = 128
_CACHE_LOCK = threading.RLock()


class KvReadError(ValueError):

  def __init__(self, status, reason=None):
    self.status = status
    self.reason = reason or status.replace('_', ' ')
    super().__init__(self.reason)


def _verified_storage(record):
  try:
    require_storage(record)
  except NativeStorageError as error:
    raise KvReadError(error.status, error.reason) from error


def _signature(path):
  stat = path.stat()
  return (
      str(path),
      stat.st_dev,
      stat.st_ino,
      stat.st_size,
      stat.st_mtime_ns,
      stat.st_ctime_ns,
  )


def _fingerprint(value):
  return hashlib.sha256(
      json.dumps(
          value, sort_keys=True, separators=(',', ':'), allow_nan=False
      ).encode()
  ).hexdigest()


def _cache(store):
  with _CACHE_LOCK:
    if not hasattr(store, '_kv_reader_cache'):
      store._kv_reader_cache = {
          'hashes': OrderedDict(),
          'proofs': OrderedDict(),
      }
    return store._kv_reader_cache


def _remember(cache, key, value):
  cache[key] = value
  cache.move_to_end(key)
  while len(cache) > _CACHE_ENTRIES:
    cache.popitem(last=False)


def _path(root, record):
  name, kind = record.get('path'), record.get('format')
  if kind not in ('safetensors', 'npy'):
    raise KvReadError('unsupported_tensor_format')
  if (
      not isinstance(name, str)
      or not name
      or '\x00' in name
      or Path(name).is_absolute()
  ):
    raise KvReadError('invalid_tensor_path')
  root = Path(root).resolve()
  path = (root / name).resolve()
  if not path.is_relative_to(root) or path.suffix != (
      '.safetensors' if kind == 'safetensors' else '.npy'
  ):
    raise KvReadError('invalid_tensor_path')
  return path


def _verified_signature(store, path, checksum):
  if (
      not isinstance(checksum, str)
      or len(checksum) != 64
      or any(char not in '0123456789abcdef' for char in checksum)
  ):
    raise KvReadError(
        'resource_checksum_mismatch', 'A recorded SHA256 is required.'
    )
  signature = _signature(path)
  key = (signature, checksum)
  with _CACHE_LOCK:
    hashes = _cache(store)['hashes']
    if key not in hashes:
      # file_digest reads 1 MiB at a time, independent of tensor capacity.
      if file_digest(path) != checksum:
        raise KvReadError('resource_checksum_mismatch')
      if _signature(path) != signature:
        raise KvReadError(
            'evidence_changed',
            'The tensor changed while its checksum was verified.',
        )
      _remember(hashes, key, True)
  return signature


def _geometry(shape, layout, start, end, valid_length=None):
  if (
      not isinstance(shape, list)
      or len(shape) != 4
      or any(type(n) is not int or n <= 0 for n in shape)
      or not isinstance(layout, list)
      or len(layout) != 4
      or any(not isinstance(axis, str) for axis in layout)
      or set(layout) != set(LAYOUT)
  ):
    raise KvReadError('layout_unavailable')
  order = tuple(layout.index(axis) for axis in LAYOUT)
  canonical = tuple(shape[axis] for axis in order)
  if (
      type(start) is not int
      or type(end) is not int
      or start < 0
      or end <= start
      or canonical[2] != end - start
      or (valid_length is not None and valid_length != end - start)
  ):
    raise KvReadError(
        'logical_mapping_unavailable',
        'The stored sequence axis must exactly match its recorded logical'
        ' range.',
    )
  return canonical, order


@dataclass
class KvTensor:
  store: object
  root: Path
  record: dict
  shape: tuple
  start: int
  end: int
  order: tuple
  view: tuple
  source: dict
  structure: dict
  quantization: dict | None = None

  def read(self, start, end, *, batch=None, head=None):
    _verified_storage(self.record)
    if (
        type(start) is not int
        or type(end) is not int
        or not self.start <= start < end <= self.end
        or any(
            value is not None
            and (type(value) is not int or value < 0 or value >= size)
            for value, size in ((batch, self.shape[0]), (head, self.shape[1]))
        )
    ):
      raise KvReadError(
          'position_unavailable',
          'Requested coordinates are outside this captured tensor.',
      )
    shape = (
        self.shape[0] if batch is None else 1,
        self.shape[1] if head is None else 1,
        end - start,
        self.shape[3],
    )
    if math.prod(shape) > MAX_SLICE_ELEMENTS:
      raise KvReadError(
          'analysis_limit',
          f'A KV slice may contain at most {MAX_SLICE_ELEMENTS} elements per'
          ' side.',
      )
    slices = list(self.view)
    for canonical_axis, low, high in (
        (0, batch, None if batch is None else batch + 1),
        (1, head, None if head is None else head + 1),
        (2, start - self.start, end - self.start),
    ):
      if low is not None:
        raw_axis = self.order[canonical_axis]
        base = self.view[raw_axis].start
        slices[raw_axis] = slice(base + low, base + high)
    path = _path(self.root, self.record)
    signature = _verified_signature(
        self.store,
        path,
        self.record.get('sha256', self.record.get('file_sha256')),
    )
    try:
      if self.record['format'] == 'safetensors':
        key = self.record.get('key')
        if not isinstance(key, str):
          raise KvReadError('invalid_tensor_key')
        with safe_open(path, framework='numpy') as shard:
          if key not in shard.keys():
            raise KvReadError('tensor_key_not_found')
          tensor = shard.get_slice(key)
          if tensor.get_shape() != self.record['shape']:
            raise KvReadError('tensor_metadata_mismatch')
          value = tensor[tuple(slices)]
      else:
        mapped = np.load(path, mmap_mode='r', allow_pickle=False)
        if list(mapped.shape) != self.record['shape']:
          raise KvReadError('tensor_metadata_mismatch')
        value = np.array(mapped[tuple(slices)], copy=True)
      if value.dtype.name != self.record.get('dtype'):
        raise KvReadError('tensor_metadata_mismatch')
      if value.dtype.kind not in 'fiu' and value.dtype != np.dtype(
          ml_dtypes.bfloat16
      ):
        raise KvReadError('unsupported_dtype')
      if _signature(path) != signature:
        raise KvReadError(
            'evidence_changed', 'The tensor changed during the slice read.'
        )
      _verified_storage(self.record)
      value = value.transpose(self.order)
      if value.shape != shape:
        raise KvReadError('tensor_metadata_mismatch')
      if self.source['values_mode'] == 'comparison':
        value = value.astype(np.float64)
        if self.quantization:
          value = (value - self.quantization['zero_point']) * self.quantization[
              'scale'
          ]
      return value
    except KvReadError:
      raise
    except (
        SafetensorError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        EOFError,
        OverflowError,
    ) as error:
      raise KvReadError('invalid_tensor_file', str(error)) from error


class KvSlice(dict):
  """Available arrays, with per-side read failures retained separately."""

  def __init__(self):
    super().__init__()
    self.errors = {}


@dataclass
class KvEvidence:
  sides: dict
  status: str
  reason: str | None = None
  comparison_basis: str | None = None
  sources: dict = field(default_factory=dict)
  structures: dict = field(default_factory=dict)

  def __post_init__(self):
    for role, tensor in self.sides.items():
      self.sources[role] = deepcopy(tensor.source)
      self.structures[role] = deepcopy(tensor.structure)

  def read(self, start, end, *, batch=None, head=None):
    result = KvSlice()
    for role, tensor in self.sides.items():
      try:
        result[role] = tensor.read(start, end, batch=batch, head=head)
      except (KvReadError, OSError) as error:
        result.errors[role] = dict(
            status=getattr(error, 'status', 'unavailable'), reason=str(error)
        )
    return result


def _snapshot_status(sides):
  if len(sides) != 2:
    return (
        'unavailable',
        (
            'A stored tensor is required on both sides; readable single-side'
            ' values are retained.'
        ),
    )
  ref, target = (sides[role].record for role in ('ref', 'target'))
  if ref.get('runtime') != target.get('runtime'):
    return 'runtime_mismatch', None
  if not ref.get('model_sha256') or ref['model_sha256'] != target.get(
      'model_sha256'
  ):
    return 'model_mismatch', None
  if ref.get('scope') != target.get('scope') or ref.get('scope') != 'kv':
    return 'resource_identity_mismatch', None
  if not ref.get('sample') or ref['sample'] != target.get('sample'):
    return 'sample_mismatch', None
  if any(
      ref.get(key) != target.get(key) for key in ('moment', 'layer', 'kind')
  ):
    return 'resource_identity_mismatch', None
  if any(
      row.get('state') != 'available'
      or row.get('preparation_status') == 'failed'
      or (
          row.get('moment') == 'terminal'
          and row.get('terminal_status') != 'completed'
      )
      for row in (ref, target)
  ):
    return 'unavailable', None
  if not ref.get('layout') or ref['layout'] != target.get('layout'):
    return 'layout_mismatch', None
  if any(
      ref.get(key) is None or ref.get(key) != target.get(key)
      for key in (
          'logical_start',
          'logical_end',
          'valid_length',
          'processed_token_count',
      )
  ):
    return 'logical_range_mismatch', None
  if sides['ref'].shape != sides['target'].shape:
    return 'shape_mismatch', None
  if (
      sides['ref'].source['values_mode']
      != sides['target'].source['values_mode']
  ):
    return (
        'quantization_unavailable',
        'Both sides must use the same recorded comparison units.',
    )
  return 'ok', None


class KvReader:
  """Request-scoped catalog and access to recorded Layer × K/V tensor pairs.

  Create one reader per request. Only immutable validated proof reports and
  checksum results are shared through the store's synchronized bounded cache.
  """

  max_slice_elements = MAX_SLICE_ELEMENTS

  def __init__(self, store):
    self.store = store
    self._resolved = {}
    self._loaded_turn = None

  def _snapshot_tensor(self, record):
    try:
      view, selected_shape = storage_view(record)
      quant = dequantization(record)
    except ValueError as error:
      raise KvReadError(str(error), getattr(error, 'reason', None)) from error
    if (
        record.get('valid_length') == 0
        and selected_shape[record['layout'].index('sequence')] == 0
    ):
      raise KvReadError('empty', 'This snapshot contains no processed tokens.')
    shape, order = _geometry(
        selected_shape,
        record.get('layout'),
        record.get('logical_start'),
        record.get('logical_end'),
        record.get('valid_length'),
    )
    source = dict(
        runtime=record.get('runtime'),
        source_dtype=record.get('dtype'),
        source_shape=deepcopy(record.get('shape')),
        comparison_dtype='float64' if quant else record.get('dtype'),
        comparison_shape=list(shape),
        layout=deepcopy(record.get('layout')),
        axis_order=list(order),
        values_mode='comparison' if quant else 'stored',
        resource_id=record.get('id'),
        storage_view=deepcopy(record.get('storage_view')),
        dequantization=deepcopy(quant),
    )
    structure = _resource_structure(record)
    return KvTensor(
        self.store,
        Path(self.store.root),
        deepcopy(record),
        shape,
        record['logical_start'],
        record['logical_end'],
        order,
        view,
        source,
        structure,
        quant,
    )

  def _explicit_tensor(self, pair, role):
    run = pair['runs'][role]
    record = run['original_tensor']
    start, end = pair['compared_position_range']
    shape, _ = _geometry(pair['shape'], pair['canonical_layout'], start, end)
    order = tuple(run['axis_order'])
    view = tuple(
        slice(axis['start'], axis['stop'], axis['step']) for axis in run['view']
    )
    source = dict(
        runtime=run['runtime'],
        source_dtype=record['dtype'],
        source_shape=deepcopy(record['shape']),
        comparison_dtype='float64',
        comparison_shape=list(shape),
        layout=deepcopy(run['layout']),
        axis_order=list(order),
        dequantization=deepcopy(run.get('dequantization')),
        values_mode='comparison',
        pair_id=pair['pair_id'],
    )
    structure = _token_structure(
        pair['shape'],
        pair['canonical_layout'],
        start,
        end - start,
        record['dtype'],
    )
    return KvTensor(
        self.store,
        self.store.root / 'explicit_pairs' / pair['pair_id'] / role,
        deepcopy(record),
        shape,
        start,
        end,
        order,
        view,
        source,
        structure,
        run.get('dequantization'),
    )

  def _proof(self, entry):
    manifest, pair_id = entry['manifest'], entry['pair_id']
    if _evidence_size(manifest) > MAX_PROOF_ELEMENTS:
      raise KvReadError(
          'analysis_limit',
          'This explicit pair exceeds the bounded legacy proof-validation'
          ' budget.',
      )
    root = (Path(self.store.root) / 'explicit_pairs' / pair_id).resolve()

    def signatures():
      files = []
      for path in root.rglob('*'):
        if path.is_file():
          resolved = path.resolve()
          if not resolved.is_relative_to(root):
            raise KvReadError(
                'invalid_tensor_path',
                'Registered proof dependency escapes its pair root.',
            )
          files.append(_signature(resolved))
          if len(files) > MAX_PROOF_FILES:
            raise KvReadError(
                'analysis_limit', 'Too many registered proof dependencies.'
            )
      return tuple(sorted(files))

    binding = _fingerprint(
        {'session': self.store.session, 'tensors': self.store.tensors}
    )
    before = signatures()
    key = (str(root), _fingerprint(manifest), binding, before)
    with _CACHE_LOCK:
      cache = _cache(self.store)['proofs']
      if key in cache:
        return deepcopy(cache[key])
      # A false small declaration must not let the legacy validator load a
      # huge actual tensor before detecting its metadata mismatch.
      for signature in before:
        path = Path(signature[0])
        if path.suffix == '.safetensors':
          with safe_open(path, framework='numpy') as shard:
            for name in shard.keys():
              if (
                  math.prod(shard.get_slice(name).get_shape())
                  > MAX_PROOF_ELEMENTS
              ):
                raise KvReadError(
                    'analysis_limit',
                    'A proof tensor exceeds the bounded validator budget.',
                )
      report = self.store.compare_pair(pair_id)
      if signatures() != before:
        raise KvReadError(
            'evidence_changed', 'Registered proof changed during validation.'
        )
      _remember(cache, key, deepcopy(report))
      return report

  def contexts(self, turn):
    if type(turn) is not int or not any(
        row['n'] == turn for row in self.store.session['turns']
    ):
      raise ValueError('Expected a captured Turn.')
    contexts = OrderedDict()
    self._resolved = {}
    self._loaded_turn = None
    for entry in self.store.explicit_pair_entries():
      manifest = entry.get('manifest', {})
      if (
          manifest.get('kind') != 'explicit_cross_runtime_terminal_kv'
          or manifest.get('observation', {}).get('turn') != turn
      ):
        continue
      try:
        pair = self._proof(entry)
        observations = {
            role: deepcopy(run.get('observation', pair['observation']))
            for role, run in pair['runs'].items()
        }
        context = _context(
            'explicit',
            {**pair['observation'], 'moment': 'terminal'},
            runtime=None,
            observations=observations,
        )
        context['label'] += ' · Explicit comparison'
        context = contexts.setdefault(context['id'], context)
        layer, kind = pair['owner_layer'], pair['kind']
        sides = {
            role: self._explicit_tensor(pair, role)
            for role in ('ref', 'target')
        }
        evidence = KvEvidence(
            sides, 'ok', comparison_basis=pair.get('comparison_basis')
        )
        for role in ('ref', 'target'):
          declaration = manifest['runs'][role]
          identity = (
              {'snapshot_id': declaration.get('snapshot_id')}
              if role == 'ref'
              else declaration.get('snapshot', {})
          )
          _append_snapshot(
              context,
              role,
              _capture_snapshot(
                  identity, pair['runs'][role]['runtime'], observations[role]
              ),
          )
      except (
          KvReadError,
          KeyError,
          ValueError,
          TypeError,
          OSError,
          RuntimeError,
      ) as error:
        layer, kind = manifest.get('owner_layer'), manifest.get(
            'kind_of_tensor'
        )
        if type(layer) is not int or layer < 0 or kind not in ('key', 'value'):
          continue
        context = _context(
            'explicit',
            {'turn': turn, 'phase': None, 'moment': None},
            runtime=None,
            pair_id=entry['pair_id'],
        )
        context['label'] = (
            f'Turn {turn} · Layer {layer} {kind} · Saved comparison unavailable'
        )
        context = contexts.setdefault(context['id'], context)
        evidence = KvEvidence(
            {}, getattr(error, 'status', 'unavailable'), str(error)
        )
      self._add(context, layer, kind, evidence, pair_id=entry['pair_id'])

    data = self.store.telemetry(turn=turn)
    resources = {row['id']: row for row in data['resources']}
    grouped = {}
    for snapshot in data['kv_snapshots']:
      coordinates = _snapshot_coordinates(snapshot)
      context = _context(
          'snapshot', coordinates, runtime=snapshot.get('runtime')
      )
      context = contexts.setdefault(context['id'], context)
      role = snapshot.get('run')
      if role not in ('ref', 'target'):
        continue
      _append_snapshot(context, role, _capture_snapshot(snapshot))
      for layer in snapshot.get('layers', []):
        for tensor in layer.get('tensors', []):
          kind = tensor.get('kind')
          if kind in ('key', 'value'):
            key = (context['id'], layer['layer'], kind)
            grouped.setdefault(key, {'ref': [], 'target': []})[role].append(
                resources.get(tensor.get('resource_id'))
            )
    for (context_id, layer, kind), records in grouped.items():
      sides, failures, sources, structures = {}, [], {}, {}
      if any(len(values) > 1 for values in records.values()):
        evidence = KvEvidence(
            {},
            'ambiguous_observation',
            'Multiple stored resources claim this observation.',
        )
      else:
        for role, values in records.items():
          if values and values[0]:
            record = values[0]
            sources[role] = dict(
                runtime=record.get('runtime'),
                source_dtype=record.get('dtype'),
                source_shape=deepcopy(record.get('shape')),
                comparison_dtype=record.get('dtype'),
                comparison_shape=None,
                layout=deepcopy(record.get('layout')),
                values_mode='stored',
                resource_id=record.get('id'),
            )
            structures[role] = _resource_structure(record)
            try:
              sides[role] = self._snapshot_tensor(record)
            except (KvReadError, ValueError, TypeError, KeyError) as error:
              failures.append(error)
        status, reason = _snapshot_status(sides)
        if failures:
          status, reason = getattr(failures[0], 'status', 'unavailable'), str(
              failures[0]
          )
        evidence = KvEvidence(
            sides,
            status,
            reason,
            'same logical context; cumulative numerical difference'
            if status == 'ok'
            else None,
            sources,
            structures,
        )
      self._add(contexts[context_id], layer, kind, evidence)
    self._loaded_turn = turn
    return list(contexts.values())

  def _add(self, context, layer, kind, evidence, **extra):
    key = (context['id'], layer, kind)
    previous = next(
        (
            row
            for row in context['layers']
            if row['layer'] == layer and row['kind'] == kind
        ),
        None,
    )
    if previous is not None:
      evidence = KvEvidence(
          {},
          'ambiguous_observation',
          'Multiple saved pairs claim this layer and K/V.',
      )
      previous.update(status=evidence.status, reason=evidence.reason)
      self._resolved[key] = evidence
      return
    row = _row(layer, kind, **extra)
    row.update(
        status=evidence.status, comparison_basis=evidence.comparison_basis
    )
    if evidence.reason:
      row['reason'] = evidence.reason
    for role, structure in evidence.structures.items():
      row['token_structure'][role] = deepcopy(structure)
    for role, tensor in evidence.sides.items():
      if tensor.source.get('resource_id'):
        row['reference' if role == 'ref' else 'target'] = tensor.source[
            'resource_id'
        ]
    if evidence.sides:
      first = next(iter(evidence.sides.values()))
      row.update(
          shape=list(first.shape),
          canonical_layout=LAYOUT.copy(),
          head_count=first.shape[1],
          position_start=first.start,
          position_count=first.end - first.start,
      )
    context['layers'].append(row)
    self._resolved[key] = evidence

  def resolve(self, turn, context_id, layer, kind):
    if self._loaded_turn != turn:
      self.contexts(turn)
    return self._resolved.get(
        (context_id, layer, kind),
        KvEvidence(
            {},
            'context_unavailable',
            'This recorded KV selection is unavailable.',
        ),
    )
