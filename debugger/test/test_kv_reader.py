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

"""Bounded, immutable KV slices from actual Safetensors and recorded proofs."""

import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from ml_dtypes import bfloat16
from model_explorer_debugger.capture_telemetry import empty_telemetry
from model_explorer_debugger.kv_head import inspect_kv_head
from model_explorer_debugger.kv_reader import KvReader, MAX_SLICE_ELEMENTS
from model_explorer_debugger.node_details import file_digest
from model_explorer_debugger.store import SessionStore
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
import test_kv_head as head_fixtures


class SliceOnly:
  """A real Safetensors handle that fails immediately on a full tensor read."""

  reads = []

  def __init__(self, *args, **kwargs):
    self.handle = safe_open(*args, **kwargs)

  def __enter__(self):
    self.handle.__enter__()
    return self

  def __exit__(self, *args):
    return self.handle.__exit__(*args)

  def keys(self):
    return self.handle.keys()

  def get_tensor(self, key):
    raise AssertionError('Unbounded get_tensor is forbidden.')

  def get_slice(self, key):
    original = self.handle.get_slice(key)

    class Observed:

      def get_shape(self):
        return original.get_shape()

      def __getitem__(self, indices):
        SliceOnly.reads.append(indices)
        return original[indices]

    return Observed()


class KvReaderTests(unittest.TestCase):
  explicit_store = head_fixtures.KvHeadTests.explicit_store

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)

  def snapshot_store(
      self,
      shape=(1, 2, 131072, 4),
      layout=None,
      start=10,
      dtype=np.float32,
      kind='key',
      tensor_format='safetensors',
  ):
    layout = layout or ['batch', 'kv_head', 'sequence', 'head_dim']
    canonical = (
        np.arange(np.prod(shape), dtype=np.float32).reshape(shape).astype(dtype)
    )
    order = [
        (['batch', 'kv_head', 'sequence', 'head_dim']).index(axis)
        for axis in layout
    ]
    raw = canonical.transpose(order).copy()
    store = object.__new__(SessionStore)
    store.root = self.root
    store.session = {'turns': [{'n': 1}], 'runs': [], 'batches': []}
    store.tensors = []
    store._telemetry = empty_telemetry()
    store._resources = {}
    for role, value in (
        ('ref', raw),
        ('target', (raw.astype(np.float32) + 1).astype(dtype)),
    ):
      path = self.root / (
          role + ('.safetensors' if tensor_format == 'safetensors' else '.npy')
      )
      if tensor_format == 'safetensors':
        save_file({'cache': value}, path)
      else:
        np.save(path, value, allow_pickle=False)
      record = dict(
          id=role,
          run=role,
          turn=1,
          runtime='PyTorch',
          model_sha256='same',
          sample='same-context',
          scope='kv',
          moment='terminal',
          layer=0,
          kind=kind,
          state='available',
          preparation_status='prepared',
          terminal_status='completed',
          layout=layout,
          logical_start=start,
          logical_end=start + shape[2],
          valid_length=shape[2],
          processed_token_count=start + shape[2],
          shape=list(value.shape),
          dtype=value.dtype.name,
          path=path.name,
          format=tensor_format,
          key='cache',
          sha256=file_digest(path),
      )
      store._resources[role] = record
      store._telemetry['resources'].append(record)
      store._telemetry['kv_snapshots'].append(
          dict(
              run=role,
              turn=1,
              runtime='PyTorch',
              phase='decode',
              step=0,
              forward_id=1,
              moment='terminal',
              snapshot_id=2,
              layers=[
                  dict(layer=0, tensors=[dict(kind=kind, resource_id=role)])
              ],
          )
      )
    return store, canonical

  def evidence(self, store, kind='key'):
    reader = KvReader(store)
    context = next(c for c in reader.contexts(1) if c['moment'] == 'terminal')
    return reader, context, reader.resolve(1, context['id'], 0, kind)

  def test_128k_catalog_does_not_read_tensor_payloads(self):
    store, _ = self.snapshot_store()
    with (
        patch(
            'model_explorer_debugger.kv_reader.safe_open',
            side_effect=AssertionError('payload read'),
        ),
        patch.object(
            store, 'load_resource', side_effect=AssertionError('full read')
        ),
    ):
      contexts = KvReader(store).contexts(1)
    layer = contexts[0]['layers'][0]
    self.assertEqual(
        (layer['status'], layer['position_start'], layer['position_count']),
        ('ok', 10, 131072),
    )
    self.assertEqual(layer['token_structure']['ref']['shape'], [2, 4])
    self.assertEqual(layer['cells'], [])
    self.assertEqual(layer['token_metrics'], [])

  def test_128k_exact_range_and_head_never_use_get_tensor(self):
    store, raw = self.snapshot_store()
    reader, _, evidence = self.evidence(store)
    SliceOnly.reads.clear()
    with (
        patch('model_explorer_debugger.kv_reader.safe_open', SliceOnly),
        patch.object(
            store, 'load_resource', side_effect=AssertionError('full read')
        ),
    ):
      part = evidence.read(131080, 131082, batch=0, head=1)
    self.assertEqual(part.errors, {})
    self.assertEqual(part['ref'].shape, (1, 1, 2, 4))
    np.testing.assert_array_equal(part['ref'], raw[:, 1:2, 131070:131072, :])
    np.testing.assert_array_equal(part['target'], part['ref'] + 1)
    self.assertEqual(len(SliceOnly.reads), 2)
    self.assertTrue(
        all(index[2] == slice(131070, 131072) for index in SliceOnly.reads)
    )
    self.assertLess(part['ref'].size, reader.max_slice_elements)

  def test_slice_budget_blocks_big_range_without_opening_payload(self):
    store, _ = self.snapshot_store()
    _, _, evidence = self.evidence(store)
    with patch(
        'model_explorer_debugger.kv_reader.safe_open',
        side_effect=AssertionError('payload read'),
    ):
      result = evidence.read(10, 131082)
    self.assertFalse(result)
    self.assertEqual(
        {error['status'] for error in result.errors.values()},
        {'analysis_limit'},
    )
    self.assertEqual(MAX_SLICE_ELEMENTS, 262144)

  def test_128k_head_endpoint_uses_only_the_selected_vector(self):
    store, raw = self.snapshot_store()
    reader = KvReader(store)
    context = reader.contexts(1)[0]
    SliceOnly.reads.clear()
    with (
        patch('model_explorer_debugger.kv_reader.safe_open', SliceOnly),
        patch.object(
            store, 'load_resource', side_effect=AssertionError('full read')
        ),
    ):
      result = inspect_kv_head(store, 1, context['id'], 0, 'key', 131081, 1)
    self.assertEqual(result['status'], 'ok')
    self.assertEqual(result['metrics']['rmse'], 1.0)
    self.assertEqual(len(result['channels']), 4)
    self.assertEqual(
        [row['ref'] for row in result['channels']], raw[0, 1, -1, :].tolist()
    )
    self.assertEqual(len(SliceOnly.reads), 2)
    self.assertTrue(
        all(index[2] == slice(131071, 131072) for index in SliceOnly.reads)
    )

  def test_realistic_128k_by_256_head_exceeds_old_full_tensor_budget(self):
    store, _ = self.snapshot_store(shape=(1, 1, 3, 2), start=0)
    shape = [1, 1, 131072, 256]
    byte_count = int(np.prod(shape)) * 4
    header = json.dumps(
        {'cache': dict(dtype='F32', shape=shape, data_offsets=[0, byte_count])}
    ).encode()
    header += b' ' * (-len(header) % 8)
    for role in ('ref', 'target'):
      record = store._resources[role]
      path = self.root / record['path']
      # Sparse, valid Safetensors fixture: 128 MiB logical payload per side,
      # with only the final vector materialized by the test setup.
      with path.open('wb') as stream:
        stream.write(struct.pack('<Q', len(header)))
        stream.write(header)
        stream.truncate(8 + len(header) + byte_count)
        stream.seek(8 + len(header) + byte_count - 256 * 4)
        stream.write(
            (np.arange(256, dtype=np.float32) + (role == 'target')).tobytes()
        )
      record.update(
          shape=shape,
          logical_end=131072,
          valid_length=131072,
          processed_token_count=131072,
          sha256=file_digest(path),
      )
    context = KvReader(store).contexts(1)[0]
    SliceOnly.reads.clear()
    with (
        patch('model_explorer_debugger.kv_reader.safe_open', SliceOnly),
        patch.object(
            store, 'load_resource', side_effect=AssertionError('full read')
        ),
    ):
      result = inspect_kv_head(store, 1, context['id'], 0, 'key', 131071, 0)
    self.assertEqual(result['status'], 'ok')
    self.assertEqual(
        (len(result['channels']), result['metrics']['rmse']), (256, 1.0)
    )
    self.assertEqual(result['channels'][-1]['ref'], 255.0)
    self.assertTrue(
        all(index[2] == slice(131071, 131072) for index in SliceOnly.reads)
    )

  def test_transposed_value_axes_and_selected_batch_are_preserved(self):
    layout = ['batch', 'kv_head', 'head_dim', 'sequence']
    store, raw = self.snapshot_store(
        shape=(2, 3, 4, 2), layout=layout, kind='value'
    )
    _, _, evidence = self.evidence(store, 'value')
    result = evidence.read(11, 13, batch=1, head=2)
    self.assertEqual(result.errors, {})
    np.testing.assert_array_equal(result['ref'], raw[1:2, 2:3, 1:3, :])
    self.assertEqual(result['ref'].shape, (1, 1, 2, 2))

  def test_bfloat16_bits_remain_in_source_dtype(self):
    store, raw = self.snapshot_store(shape=(1, 1, 3, 2), dtype=bfloat16)
    _, _, evidence = self.evidence(store)
    result = evidence.read(11, 12)
    self.assertEqual(result.errors, {})
    self.assertEqual(result['ref'].dtype.name, 'bfloat16')
    np.testing.assert_array_equal(
        result['ref'].view(np.uint16), raw[:, :, 1:2, :].view(np.uint16)
    )

  def test_npy_is_read_only_mmap_and_only_slice_is_copied(self):
    store, raw = self.snapshot_store(tensor_format='npy')
    _, _, evidence = self.evidence(store)
    before = {
        role: file_digest(self.root / (role + '.npy'))
        for role in ('ref', 'target')
    }
    with patch(
        'model_explorer_debugger.kv_reader.np.load', wraps=np.load
    ) as load:
      result = evidence.read(131081, 131082, head=1)
    self.assertEqual(result.errors, {})
    self.assertEqual(load.call_count, 2)
    self.assertTrue(
        all(
            call.kwargs == dict(mmap_mode='r', allow_pickle=False)
            for call in load.call_args_list
        )
    )
    np.testing.assert_array_equal(result['ref'], raw[:, 1:2, -1:, :])
    self.assertEqual(
        before,
        {
            role: file_digest(self.root / (role + '.npy'))
            for role in ('ref', 'target')
        },
    )

  def test_checksum_is_cached_and_changed_side_does_not_hide_readable_side(
      self,
  ):
    store, _ = self.snapshot_store(shape=(1, 1, 3, 2))
    _, _, evidence = self.evidence(store)
    with patch(
        'model_explorer_debugger.kv_reader.file_digest', wraps=file_digest
    ) as digest:
      self.assertEqual(evidence.read(10, 11).errors, {})
      self.assertEqual(evidence.read(11, 12).errors, {})
      self.assertEqual(digest.call_count, 2)
      path = self.root / 'target.safetensors'
      with path.open('r+b') as stream:
        stream.seek(-1, 2)
        stream.write(b'\x01')
      result = evidence.read(10, 11)
    self.assertEqual(set(result), {'ref'})
    self.assertEqual(
        result.errors['target']['status'], 'resource_checksum_mismatch'
    )
    self.assertEqual(digest.call_count, 3)

  def test_missing_side_and_unproven_pair_preserve_available_values(self):
    store, _ = self.snapshot_store(shape=(1, 1, 3, 2))
    store._resources['target']['sample'] = 'different-context'
    _, _, evidence = self.evidence(store)
    self.assertEqual(evidence.status, 'sample_mismatch')
    self.assertEqual(set(evidence.read(10, 11)), {'ref', 'target'})
    store._telemetry['kv_snapshots'] = store._telemetry['kv_snapshots'][:1]
    _, _, evidence = self.evidence(store)
    self.assertEqual(evidence.status, 'unavailable')
    self.assertEqual(set(evidence.read(10, 11)), {'ref'})

  def test_unknown_layout_and_capacity_do_not_create_logical_mapping(self):
    for field, value in (('layout', ['unknown'] * 4), ('logical_end', 99)):
      with self.subTest(field=field):
        store, _ = self.snapshot_store(shape=(1, 1, 3, 2))
        for record in store._resources.values():
          record[field] = value
        _, _, evidence = self.evidence(store)
        self.assertIn(
            evidence.status,
            ('layout_unavailable', 'logical_mapping_unavailable'),
        )
        self.assertEqual(evidence.sides, {})

  def test_wrong_dtype_and_escaping_paths_fail_truthfully(self):
    store, _ = self.snapshot_store(shape=(1, 1, 3, 2))
    store._resources['target']['dtype'] = 'float64'
    _, _, evidence = self.evidence(store)
    result = evidence.read(10, 11)
    self.assertEqual(
        result.errors['target']['status'], 'tensor_metadata_mismatch'
    )
    store._resources['target']['path'] = '../outside.safetensors'
    _, _, evidence = self.evidence(store)
    self.assertEqual(
        evidence.read(10, 11).errors['target']['status'], 'invalid_tensor_path'
    )

  def test_explicit_proof_is_cached_and_slice_dequantizes_once(self):
    store, pair, selection = self.explicit_store()
    reader = KvReader(store)
    with patch.object(
        store, 'compare_pair', wraps=store.compare_pair
    ) as validate:
      reader.contexts(1)
      evidence = reader.resolve(1, selection['context_id'], 0, 'value')
      again = KvReader(store).resolve(1, selection['context_id'], 0, 'value')
      self.assertEqual(validate.call_count, 1)
    before = file_digest(
        store.root
        / 'explicit_pairs'
        / pair['pair_id']
        / 'target'
        / pair['runs']['target']['original_tensor']['path']
    )
    with patch('model_explorer_debugger.kv_reader.safe_open', SliceOnly):
      result = evidence.read(3, 4, batch=0, head=0)
    self.assertEqual(evidence.status, 'ok')
    self.assertEqual(again.status, 'ok')
    self.assertEqual(result.errors, {})
    np.testing.assert_array_equal(result['ref'], np.array([[[[6.0, 7.0]]]]))
    np.testing.assert_array_equal(result['target'], np.array([[[[7.0, 8.0]]]]))
    self.assertEqual(
        before,
        file_digest(
            store.root
            / 'explicit_pairs'
            / pair['pair_id']
            / 'target'
            / pair['runs']['target']['original_tensor']['path']
        ),
    )

  def test_explicit_changed_proof_dependency_and_binding_invalidate_cache(self):
    store, pair, selection = self.explicit_store()
    reader = KvReader(store)
    self.assertEqual(
        reader.resolve(1, selection['context_id'], 0, 'value').status, 'ok'
    )
    # The dependency still parses, but its hash mismatches the declaration.
    path = (
        store.root
        / 'explicit_pairs'
        / pair['pair_id']
        / 'target'
        / 'kv_evidence.json'
    )
    path.write_text(path.read_text() + ' ')
    result = reader.contexts(1)
    self.assertEqual(result[0]['layers'][0]['status'], 'unavailable')
    self.assertIn('checksum', result[0]['layers'][0]['reason'])

  def test_explicit_changed_session_binding_is_not_accepted_from_cache(self):
    store, _, selection = self.explicit_store()
    self.assertEqual(
        KvReader(store).resolve(1, selection['context_id'], 0, 'value').status,
        'ok',
    )
    store.session['runs'][1]['provenance']['model_sha256'] = 'changed'
    result = KvReader(store).contexts(1)
    self.assertEqual(result[0]['layers'][0]['status'], 'unavailable')
    self.assertIn('session_model_mismatch', result[0]['layers'][0]['reason'])

  def test_oversized_explicit_proof_is_not_assumed_valid(self):
    store, _, _ = self.explicit_store()
    with (
        patch('model_explorer_debugger.kv_reader.MAX_PROOF_ELEMENTS', 1),
        patch.object(
            store,
            'compare_pair',
            side_effect=AssertionError('unsafe proof read'),
        ),
    ):
      contexts = KvReader(store).contexts(1)
    self.assertEqual(contexts[0]['layers'][0]['status'], 'analysis_limit')
    self.assertEqual(contexts[0]['layers'][0]['cells'], [])


if __name__ == '__main__':
  unittest.main()
