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

"""Explicit native KV capacity views keep raw payloads immutable and bounded."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model_explorer_debugger.kv_analysis import analyze_kv
from model_explorer_debugger.kv_head import inspect_kv_head
from model_explorer_debugger.kv_range import chart_range, selection_range
from model_explorer_debugger.kv_reader import KvReader
from model_explorer_debugger.node_details import file_digest
from model_explorer_debugger.runtime import litert_storage
from model_explorer_debugger.store import SessionStore
import numpy as np
from safetensors.numpy import save_file
import test_kv_reader as fixtures


class NativeKvStorageTests(unittest.TestCase):
  snapshot_store = fixtures.KvReaderTests.snapshot_store
  evidence = fixtures.KvReaderTests.evidence

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)

  def capacity_store(self, *, kind='key', offset=0, count=3, dtype=np.float32):
    layout = (
        ['batch', 'kv_head', 'head_dim', 'sequence']
        if kind == 'value'
        else ['batch', 'kv_head', 'sequence', 'head_dim']
    )
    store, raw = self.snapshot_store(
        shape=(1, 2, 8, 4), layout=layout, start=10, kind=kind, dtype=dtype
    )
    store._resource_hashes = set()
    for record in store._resources.values():
      record.update(
          runtime='LiteRT-LM',
          backend_requested='CPU',
          logical_end=10 + count,
          valid_length=count,
          processed_token_count=10 + count,
          storage_view=[
              dict(
                  start=offset if axis == 'sequence' else 0,
                  stop=offset + count if axis == 'sequence' else size,
                  step=1,
              )
              for axis, size in zip(layout, record['shape'])
          ],
      )
    return store, raw

  def test_capacity_read_uses_only_recorded_valid_range(self):
    store, raw = self.capacity_store()
    before = {
        role: file_digest(self.root / (role + '.safetensors'))
        for role in ('ref', 'target')
    }
    _, _, evidence = self.evidence(store)
    self.assertEqual(evidence.status, 'ok')
    fixtures.SliceOnly.reads.clear()
    with patch(
        'model_explorer_debugger.kv_reader.safe_open', fixtures.SliceOnly
    ):
      values = evidence.read(10, 13, head=1)
    np.testing.assert_array_equal(values['ref'], raw[:, 1:2, :3, :])
    self.assertEqual(evidence.sources['ref']['source_shape'], [1, 2, 8, 4])
    self.assertEqual(evidence.sources['ref']['comparison_shape'], [1, 2, 3, 4])
    self.assertTrue(
        all(row[2] == slice(0, 3) for row in fixtures.SliceOnly.reads)
    )
    self.assertEqual(
        before,
        {
            role: file_digest(self.root / (role + '.safetensors'))
            for role in before
        },
    )

  def test_transposed_value_uses_explicit_physical_offset(self):
    store, raw = self.capacity_store(kind='value', offset=2)
    _, _, evidence = self.evidence(store, 'value')
    np.testing.assert_array_equal(
        evidence.read(11, 13)['ref'], raw[:, :, 3:5, :]
    )

  def test_missing_view_never_guesses_padding(self):
    store, _ = self.capacity_store()
    for record in store._resources.values():
      record.pop('storage_view')
    _, _, evidence = self.evidence(store)
    self.assertEqual(evidence.status, 'logical_mapping_unavailable')
    self.assertFalse(evidence.sides)

  def test_invalid_view_is_rejected_before_payload_read(self):
    for mutation in ('bounds', 'stride', 'length', 'head'):
      with self.subTest(mutation=mutation):
        store, _ = self.capacity_store()
        for record in store._resources.values():
          if mutation == 'bounds':
            record['storage_view'][2]['stop'] = 9
          elif mutation == 'stride':
            record['storage_view'][2]['step'] = 2
          elif mutation == 'length':
            record['valid_length'] = 2
          else:
            record['storage_view'][1]['stop'] = 1
        with patch(
            'model_explorer_debugger.kv_reader.safe_open',
            side_effect=AssertionError('payload read'),
        ):
          _, _, evidence = self.evidence(store)
        self.assertIn(
            evidence.status, ('invalid_storage_view', 'logical_range_mismatch')
        )

  def test_empty_snapshot_is_empty_not_zero_valued_capacity(self):
    store, _ = self.capacity_store(count=0)
    _, context, evidence = self.evidence(store)
    self.assertEqual(evidence.status, 'empty')
    self.assertFalse(evidence.sides)
    self.assertEqual(
        context['layers'][0]['token_structure']['ref']['position_count'], 0
    )

  def test_explicit_dequantization_preserves_raw_dtype_and_transforms_view(
      self,
  ):
    store, raw = self.capacity_store(dtype=np.int8)
    for record in store._resources.values():
      record['dequantization'] = {'scale': 0.25, 'zero_point': -1}
    _, _, evidence = self.evidence(store)
    values = evidence.read(10, 12)
    np.testing.assert_array_equal(
        values['ref'], (raw[:, :, :2, :].astype(np.float64) + 1) * 0.25
    )
    self.assertEqual(evidence.sources['ref']['source_dtype'], 'int8')
    self.assertEqual(evidence.sources['ref']['values_mode'], 'comparison')
    self.assertAlmostEqual(
        store.compare_resources('ref', 'target')['metrics']['RMSE']['value'],
        0.25,
    )

  def test_legacy_analysis_uses_view_for_geometry_and_values(self):
    store, _ = self.capacity_store()
    result = analyze_kv(store, 1)
    row = result['contexts'][0]['layers'][0]
    self.assertEqual(row['status'], 'ok')
    self.assertEqual(row['position_count'], 3)
    self.assertEqual(len(row['token_metrics']), 3)
    self.assertEqual(row['token_metrics'][0]['metrics']['rmse'], 1.0)

  def saved_native_store(self, backend='GPU', *, via_provenance=False):
    store, _ = self.capacity_store(dtype=np.int8)
    store.session['runs'] = [
        dict(id='ref', runtime='LiteRT-LM', backend='CPU'),
        dict(
            id='target',
            runtime='LiteRT-LM',
            **(
                {'provenance': {'backend_effective': backend}}
                if via_provenance
                else {'backend': backend}
            ),
        ),
    ]
    for record in store._resources.values():
      record.pop('backend_requested')
      record.update(
          dequantization={'scale': 0.25, 'zero_point': 0}, state='available'
      )
    for snapshot in store._telemetry['kv_snapshots']:
      snapshot['runtime'] = 'LiteRT-LM'
    documents = {
        'session.json': store.session,
        'telemetry.json': store._telemetry,
        'semantic.json': {'layers': [], 'semantic_graph': []},
        'execution.json': {'executions': []},
        'tensor_index.json': {'tensors': []},
    }
    for name, value in documents.items():
      (self.root / name).write_text(json.dumps(value))
    before = {
        path.name: file_digest(path)
        for path in self.root.iterdir()
        if path.is_file()
    }
    return SessionStore(self.root), before

  def test_old_saved_gpu_unavailable_in_catalog_details_and_all_metrics(self):
    store, before = self.saved_native_store()
    with patch.object(
        store, 'load_resource', side_effect=AssertionError('full payload read')
    ):
      reader, context, evidence = self.evidence(store)
      self.assertEqual(evidence.status, 'storage_contract_unavailable')
      self.assertEqual(set(evidence.sides), {'ref'})
      self.assertIn('physical representation', evidence.reason)
      self.assertEqual(store._resources['target']['state'], 'unavailable')
      self.assertEqual(store.compare_resources('ref', 'target')['metrics'], {})
      self.assertEqual(
          store.compare_resources('ref', 'target')['status'],
          'storage_contract_unavailable',
      )
      self.assertEqual(
          analyze_kv(store, 1)['contexts'][0]['layers'][0]['status'],
          'storage_contract_unavailable',
      )
      for result in (
          chart_range(store, 1, context['id'], 10, 13),
          selection_range(store, 1, context['id'], 10, 13),
      ):
        self.assertEqual(
            result['rows'][0]['status'], 'storage_contract_unavailable'
        )
        self.assertEqual(result['work']['read_chunks'], 0)
      fixtures.SliceOnly.reads.clear()
      with patch(
          'model_explorer_debugger.kv_reader.safe_open', fixtures.SliceOnly
      ):
        details = inspect_kv_head(store, 1, context['id'], 0, 'key', 11, 0)
      self.assertEqual(details['status'], 'storage_contract_unavailable')
      self.assertTrue(
          all(value is None for value in details['metrics'].values())
      )
      self.assertTrue(
          all(
              row['target'] is None and row['ref'] is not None
              for row in details['channels']
          )
      )
      self.assertEqual(
          len(fixtures.SliceOnly.reads),
          1,
          'only the proven CPU side may be read',
      )
    self.assertEqual(
        before,
        {
            path.name: file_digest(path)
            for path in self.root.iterdir()
            if path.is_file()
        },
    )
    self.assertEqual(
        json.loads((self.root / 'telemetry.json').read_text())['resources'][1][
            'state'
        ],
        'available',
    )

  def test_legacy_gpu_provenance_and_unknown_backend_are_not_assumed_cpu(self):
    for backend in ('GPU', None):
      with self.subTest(backend=backend):
        store, _ = self.saved_native_store(backend, via_provenance=True)
        _, _, evidence = self.evidence(store)
        self.assertEqual(evidence.status, 'storage_contract_unavailable')
        self.assertEqual(set(evidence.sides), {'ref'})

  def test_legacy_cpu_record_claim_cannot_override_saved_gpu_run(self):
    self.saved_native_store()
    data = json.loads((self.root / 'telemetry.json').read_text())
    data['resources'][1].update(
        backend_requested='CPU', backend_effective='CPU'
    )
    (self.root / 'telemetry.json').write_text(json.dumps(data))
    store = SessionStore(self.root)
    self.assertEqual(
        self.evidence(store)[2].status, 'storage_contract_unavailable'
    )

  def test_old_saved_cpu_remains_readable_and_comparable(self):
    store, _ = self.saved_native_store('CPU')
    _, _, evidence = self.evidence(store)
    self.assertEqual(evidence.status, 'ok')
    self.assertEqual(set(evidence.read(10, 11)), {'ref', 'target'})
    self.assertEqual(store.compare_resources('ref', 'target')['status'], 'ok')

  def test_backend_and_storage_booleans_cannot_override_unknown_gpu_bytes(self):
    store, _ = self.saved_native_store()
    store._resources['target'].update(
        state='available',
        storage_verified=True,
        storage={'version': 1, 'representation': 'logical_tensor'},
    )
    _, _, evidence = self.evidence(store)
    self.assertEqual(evidence.status, 'storage_contract_unavailable')
    self.assertEqual(set(evidence.sides), {'ref'})

  def normalized_store(self, kind='key', target_backend='GPU'):
    store, _ = self.saved_native_store(target_backend)
    layout = (
        litert_storage.KEY_LAYOUT
        if kind == 'key'
        else litert_storage.VALUE_LAYOUT
    )
    canonical = np.arange(64, dtype=np.int8).reshape(1, 2, 8, 4) - 32
    values = (
        canonical if kind == 'key' else canonical.transpose(0, 1, 3, 2).copy()
    )
    for role, record in store._resources.items():
      source = dict(
          section_offset=1024,
          section_sha256='a' * 64,
          signature='prefill_8',
          subgraph=1,
          op=2,
          output_tensor=3,
          composite='odml.cache_update',
          owner_input_tensor_names=['/layer_0/key', '/layer_0/cache'],
          composite_attributes={'cache_size': 8, 'head_size': 4},
      )
      quant = {'scale': 0.25, 'zero_point': 0, 'quantized_dimension': 0}
      record.update(
          shape=list(values.shape),
          layout=layout,
          kind=kind,
          model_source=source,
          quantization=quant,
          model_sha256='b' * 64,
          signature='prefill_8',
          tensor_name='kv_cache_' + kind,
          when='post',
          forward_id=1,
          state='available',
          key='post_' + kind,
          backend_requested='CPU' if role == 'ref' else target_backend,
          storage_view=[
              dict(start=0, stop=3 if name == 'sequence' else size, step=1)
              for name, size in zip(layout, values.shape)
          ],
      )
      source_record = dict(
          path=record['path'],
          key=record['key'],
          dtype='I8',
          shape=list(values.shape),
          buffer_type='HostMemory',
      )
      conversion = {'id': 'host_identity', 'version': 1}
      packed = None
      if role == 'target' and target_backend == 'GPU':
        source_record.update(
            path=role + '_source.safetensors',
            key='source_' + record['key'],
            dtype='U8',
            shape=[64],
            buffer_type='WebGpuBuffer',
        )
        conversion = dict(
            id='webgpu_kv_u8_to_logical_i8',
            version=1,
            kind=kind,
            source_function='RearrangeK' if kind == 'key' else 'RearrangeV',
            **litert_storage.REFERENCE,
        )
        packed = np.zeros(64, dtype=np.uint8)
        for head in range(2):
          for position in range(8):
            for channel in range(4):
              if kind == 'key':
                linear = head + channel // 4
                index = (
                    ((linear % 2) * 8 + position) * 4
                    + (linear // 2) * 4
                    + channel % 4
                )
              else:
                rhs = (
                    (head * 2 + position // 4) + channel // 4
                ) * 4 + position % 4
                temp = rhs // 4
                index = (
                    ((temp % 2) * 4 + rhs % 4) * 8
                    + (temp // 2) * 4
                    + channel % 4
                )
              packed[index] = int(canonical[0, head, position, channel]) + 128
      storage = dict(
          version=1,
          representation='logical_tensor',
          logical=dict(dtype='I8', shape=list(values.shape), layout=layout),
          source=source_record,
          conversion=conversion,
          model_source={**source, 'quantization': quant},
      )
      record['storage'] = storage
      metadata = {'storage': json.dumps(storage)}
      save_file(
          {record['key']: values.copy()},
          self.root / record['path'],
          metadata=metadata,
      )
      if packed is not None:
        save_file(
            {source_record['key']: packed},
            self.root / source_record['path'],
            metadata=metadata,
        )
      record['sha256'] = file_digest(self.root / record['path'])
      event = dict(
          event='tensor',
          status='stored',
          invocation_id=1,
          moment='post',
          key=record['key'],
          signature=record['signature'],
          tensor_name=record['tensor_name'],
          storage=storage,
      )
      trace = self.root / (role + '-trace.jsonl')
      trace.write_text(json.dumps(event) + '\n')
      record['native_trace'] = {
          'path': trace.name,
          'sha256': file_digest(trace),
      }
      descriptor = dict(
          source=source,
          quantization=quant,
          layout=layout,
          model_shape=list(values.shape),
      )
      litert_storage.validate_storage(
          record,
          self.root,
          event=event,
          descriptor=descriptor,
          model_section_sha256='a' * 64,
      )
    for snapshot in store._telemetry['kv_snapshots']:
      snapshot['layers'][0]['tensors'][0]['kind'] = kind
    (self.root / 'telemetry.json').write_text(json.dumps(store._telemetry))
    litert_storage._VALIDATED.clear()
    return SessionStore(self.root)

  def test_model_bound_normalized_gpu_key_and_value_survive_saved_reload(self):
    for kind in ('key', 'value'):
      with self.subTest(kind=kind):
        store = self.normalized_store(kind)
        _, context, evidence = self.evidence(store, kind)
        self.assertEqual(evidence.status, 'ok', evidence.reason)
        self.assertEqual(set(evidence.read(10, 13)), {'ref', 'target'})
        self.assertEqual(
            store.compare_resources('ref', 'target')['metrics']['RMSE'][
                'value'
            ],
            0.0,
        )
        details = inspect_kv_head(store, 1, context['id'], 0, kind, 11, 1)
        self.assertEqual(details['status'], 'ok')
        self.assertEqual(details['metrics']['rmse'], 0.0)

  def test_model_bound_cpu_identity_has_no_gpu_converter_requirement(self):
    store = self.normalized_store(target_backend='CPU')
    _, _, evidence = self.evidence(store)
    self.assertEqual(evidence.status, 'ok', evidence.reason)
    self.assertEqual(
        store.compare_resources('ref', 'target')['metrics']['RMSE']['value'],
        0.0,
    )

  def test_normalized_claim_and_proof_cannot_hide_source_or_trace_tampering(
      self,
  ):
    for change in (
        'source_bytes',
        'logical_bytes',
        'trace',
        'converter',
        'model_source',
        'buffer_type',
        'source_key',
    ):
      with self.subTest(change=change):
        store = self.normalized_store()
        data = json.loads((self.root / 'telemetry.json').read_text())
        record = data['resources'][1]
        if change in ('source_bytes', 'logical_bytes'):
          tensor = (
              record['storage_source'] if change == 'source_bytes' else record
          )
          path = self.root / tensor['path']
          raw = bytearray(path.read_bytes())
          raw[-1] ^= 1
          path.write_bytes(raw)
          # Even rewriting declared hashes/proof cannot turn an incorrect
          # conversion into valid evidence.
          tensor['sha256'] = file_digest(path)
          record['storage_validation'][
              'source_sha256'
              if change == 'source_bytes'
              else 'normalized_sha256'
          ] = tensor['sha256']
        elif change == 'trace':
          path = self.root / record['native_trace']['path']
          row = json.loads(path.read_text())
          row['storage']['version'] = 2
          path.write_text(json.dumps(row) + '\n')
          record['native_trace']['sha256'] = file_digest(path)
        elif change == 'converter':
          record['storage']['conversion']['source_revision'] = 'f' * 40
        elif change == 'model_source':
          record['model_source']['output_tensor'] = 99
        elif change == 'buffer_type':
          record['storage']['source']['buffer_type'] = 'WebGpuBufferPacked'
        else:
          record['storage_source']['key'] = 'not-the-source'
        (self.root / 'telemetry.json').write_text(json.dumps(data))
        litert_storage._VALIDATED.clear()
        loaded = SessionStore(self.root)
        _, _, evidence = self.evidence(loaded)
        self.assertEqual(evidence.status, 'storage_contract_unavailable')
        self.assertEqual(set(evidence.sides), {'ref'})
        self.assertEqual(
            loaded.compare_resources('ref', 'target')['metrics'], {}
        )

  def test_source_changes_after_validation_block_existing_reader(self):
    store = self.normalized_store()
    source = self.root / store._resources['target']['storage_source']['path']
    previous_evidence = self.evidence(store)[2]
    raw = bytearray(source.read_bytes())
    raw[-1] ^= 1
    source.write_bytes(raw)
    _, _, evidence = self.evidence(store)
    self.assertEqual(evidence.status, 'storage_contract_unavailable')
    self.assertEqual(store.compare_resources('ref', 'target')['metrics'], {})
    values = previous_evidence.read(10, 11)
    self.assertEqual(set(values), {'ref'})
    self.assertEqual(
        values.errors['target']['status'], 'storage_contract_unavailable'
    )

  def test_saved_reload_uses_only_stat_cache_and_rejects_changed_units(self):
    store = self.normalized_store()
    with (
        patch(
            'model_explorer_debugger.runtime.litert_storage.safe_open',
            side_effect=AssertionError('storage payload reopened'),
        ),
        patch(
            'model_explorer_debugger.runtime.litert_storage.file_digest',
            side_effect=AssertionError('storage dependency rehashed'),
        ),
    ):
      reloaded = SessionStore(self.root)
      self.assertEqual(self.evidence(reloaded)[2].status, 'ok')
    for field, value in [
        ('dequantization', {'scale': 99.0, 'zero_point': 0}),
        ('layout', litert_storage.VALUE_LAYOUT),
    ]:
      record = store._resources['target']
      old = deepcopy(record[field])
      record[field] = value
      self.assertEqual(
          self.evidence(store)[2].status, 'storage_contract_unavailable'
      )
      self.assertEqual(store.compare_resources('ref', 'target')['metrics'], {})
      record[field] = old

  def test_legacy_proof_section_and_container_scopes_are_exact_and_keep_cache(
      self,
  ):
    for old_hash in ('a' * 64, 'b' * 64, 'c' * 64):
      with self.subTest(old_hash=old_hash):
        self.normalized_store()
        data = json.loads((self.root / 'telemetry.json').read_text())
        for record in data['resources']:
          proof = record['storage_validation']
          proof.pop('model_section_sha256')
          proof.update(version=1, model_sha256=old_hash)
          record.pop('model_section_sha256', None)
        (self.root / 'telemetry.json').write_text(json.dumps(data))
        before = file_digest(self.root / 'telemetry.json')
        litert_storage._VALIDATED.clear()
        store = SessionStore(self.root)
        evidence = self.evidence(store)[2]
        if old_hash == 'c' * 64:
          self.assertEqual(evidence.status, 'storage_contract_unavailable')
          self.assertFalse(evidence.sides)
        else:
          self.assertEqual(evidence.status, 'ok', evidence.reason)
          self.assertEqual(
              store._resources['target']['storage_validation']['version'], 1
          )
          with patch(
              'model_explorer_debugger.runtime.litert_storage.safe_open',
              side_effect=AssertionError('v1 cache miss'),
          ):
            self.assertEqual(
                self.evidence(SessionStore(self.root))[2].status, 'ok'
            )
        self.assertEqual(file_digest(self.root / 'telemetry.json'), before)

  def test_v2_section_proof_does_not_change_when_publication_binds_container(
      self,
  ):
    self.normalized_store()
    data = json.loads((self.root / 'telemetry.json').read_text())
    for record in data['resources']:
      self.assertEqual(record['storage_validation']['version'], 2)
      self.assertEqual(
          record['storage_validation']['model_section_sha256'], 'a' * 64
      )
      self.assertNotIn('model_sha256', record['storage_validation'])
      record['model_sha256'] = (
          'd' * 64
      )  # Publication's actual container identity is independent.
    (self.root / 'telemetry.json').write_text(json.dumps(data))
    litert_storage._VALIDATED.clear()
    self.assertEqual(self.evidence(SessionStore(self.root))[2].status, 'ok')


if __name__ == '__main__':
  unittest.main()
