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

"""Synthetic PyTorch boundary captures; no model inference is claimed here."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

from ml_dtypes import bfloat16
from model_explorer_debugger.capture_importer import publish_capture
from model_explorer_debugger.fsutil import atomic_json
from model_explorer_debugger.store import SessionStore
import numpy as np
from safetensors.numpy import save_file


def identity(proof):
  return hashlib.sha256(
      json.dumps(
          proof, sort_keys=True, separators=(',', ':'), ensure_ascii=False
      ).encode()
  ).hexdigest()


class PyTorchImportTests(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name).resolve()
    self.registry = SimpleNamespace(
        root=self.root,
        capture_store=lambda _: SessionStore(
            self.root / self.record['capture']
        ),
    )
    self.record = {
        'id': 'session',
        'name': 'Synthetic module capture',
        'model': 'Synthetic model',
        'runs': [
            {'id': run, 'runtime': 'PyTorch'} for run in ('ref', 'target')
        ],
    }
    self.artifacts = {
        run: {'name': 'synthetic model'} for run in ('ref', 'target')
    }

  def fixture(self, job_id='job-1', delta=1, dtype=np.float32):
    job = self.root / 'sessions/session/jobs' / job_id
    results = {}
    for run in ('ref', 'target'):
      directory = job / run
      (directory / 'raw').mkdir(parents=True)
      (directory / 'export').mkdir()
      save_file(
          {
              'slot0': np.array([[1, 2]], dtype=dtype),
              'slot1': np.array(
                  [[3, 4 + (delta if run == 'target' else 0)]], dtype=dtype
              ),
          },
          directory / 'raw/prefill.safetensors',
      )
      tensors = [
          {
              'format': 'safetensors',
              'path': 'raw/prefill.safetensors',
              'key': f'slot{i}',
              'shape': [1, 2],
              'dtype': np.dtype(dtype).name,
              'module_path': 'model.layers.0.input_layernorm',
              'module_type': 'RMSNorm',
              'edge': edge,
              'invocation': 0,
              'call_seq': i,
              'output': 0,
              'layer': 0,
              'phase': 'prefill',
              'step': 0,
          }
          for i, edge in enumerate(('in', 'out'))
      ]
      proof = {
          'tensors': {
              'input_ids': {
                  'shape': [1, 2],
                  'dtype': 'int64',
                  'values': [[4, 5]],
              },
              'attention_mask': {
                  'shape': [1, 2],
                  'dtype': 'int64',
                  'values': [[1, 1]],
              },
          },
          'tokenizer': {'vocab_sha256': 'synthetic-vocabulary'},
          'serialization': 'exact synthetic input',
      }
      save_file(
          {
              key: np.array(metadata['values'], dtype=metadata['dtype'])
              for key, metadata in proof['tensors'].items()
          },
          directory / 'raw/inputs.safetensors',
      )
      input_identity = identity(proof)
      atomic_json(
          directory / 'export/capture_index.json',
          {
              'format_version': 2,
              'tensor_root': 'run',
              'runtime': 'PyTorch',
              'input_identity': input_identity,
              'tensors': tensors,
              'topology': {
                  'blocks_path': 'model.layers',
                  'sites': [{'module_path': 'uncaptured.module', 'layer': 0}],
              },
          },
      )
      results[run] = {
          'model_sha256': 'same synthetic model',
          'input': 'same visible input',
          'output': 'X',
          'messages': [],
          'input_identity': input_identity,
          'input_proof': proof,
          'input_tensor_path': 'raw/inputs.safetensors',
          'tokens': [{'id': 42, 'text': 'X', 'step': 0}],
          'effective': {
              'precision': np.dtype(dtype).name,
              'backend': 'CPU',
              'torch_compile': False,
          },
          'capture_scope': 'prefill_module_boundaries',
          'torch_version': 'synthetic-torch',
          'transformers_version': 'synthetic-transformers',
      }
    return job, results

  def publish(self, job, results):
    capture = publish_capture(
        self.registry, self.record, job, results, self.artifacts
    )
    self.record['capture'] = capture
    return SessionStore(self.root / capture)

  def change_index(self, job, run, change):
    path = job / run / 'export/capture_index.json'
    data = json.loads(path.read_text())
    change(data)
    atomic_json(path, data)

  def test_recorded_stop_reason_is_preserved_without_inference(self):
    job, results = self.fixture()
    results['ref']['stop_reason'] = 'max_output_tokens'
    store = self.publish(job, results)
    reference, target = store.session['conversation']
    self.assertEqual(reference['stop_reason'], 'max_output_tokens')
    self.assertNotIn('stop_reason', target)

  def test_each_turn_preserves_its_recorded_input_serialization(self):
    job, results = self.fixture()
    for result in results.values():
      result['serialized_input'] = (
          '<system>first</system><user>same visible input</user>'
      )
      result['messages'] = [
          {'role': 'system', 'content': 'first'},
          {'role': 'user', 'content': 'same visible input'},
      ]
    first = self.publish(job, results)
    self.assertEqual(
        first.session['conversation'][0]['serialized_input'],
        results['ref']['serialized_input'],
    )
    second_job, second_results = self.fixture('job-2')
    for result in second_results.values():
      result['serialized_input'] = '<user>second turn</user>'
      result['messages'] = [{'role': 'user', 'content': 'second turn'}]
    second = self.publish(second_job, second_results)
    self.assertEqual(
        second.session['conversation'][0]['messages'][0]['content'], 'first'
    )
    self.assertEqual(
        second.session['conversation'][2]['serialized_input'],
        '<user>second turn</user>',
    )

  def test_observed_module_edges_share_event_node_and_raw_shard(self):
    job, results = self.fixture(dtype=bfloat16)
    store = self.publish(job, results)
    self.assertEqual(len({tensor['path'] for tensor in store.tensors}), 2)
    self.assertEqual(
        {tensor['output'] for tensor in store.tensors}, {'in.0', 'out.0'}
    )
    for run in store.execution['executions']:
      self.assertEqual(run['runtime'], 'PyTorch')
      self.assertEqual(len(run['graphs']), 1)
      nodes = run['graphs'][0]['nodes']
      self.assertEqual(len(nodes), 1)
      self.assertEqual(nodes[0]['incomingEdges'], [])
      self.assertEqual(
          {output['id'] for output in nodes[0]['outputsMetadata']},
          {'in.0', 'out.0'},
      )
    graph = store.semantic['semantic_graph'][0]
    self.assertEqual(graph['kind'], 'PyTorch module boundaries')
    self.assertEqual(len(graph['nodes']), 1)
    self.assertEqual(
        graph['nodes'][0]['module_path'], 'model.layers.0.input_layernorm'
    )
    self.assertEqual(graph['nodes'][0]['incomingEdges'], [])
    self.assertEqual(len(graph['anchors']), 2)
    self.assertEqual(
        [anchor['label'] for anchor in graph['anchors']],
        ['input_layernorm · in[0]', 'input_layernorm · out[0]'],
    )
    self.assertEqual(
        [anchor['edge'] for anchor in graph['anchors']], ['in', 'out']
    )
    self.assertEqual(
        store.semantic['layers'][0]['source'], 'observed_module_boundaries'
    )
    details = store.node_details.details(
        0, 0, 'anchor:' + graph['anchors'][0]['id']
    )
    self.assertEqual(
        details['node']['module_path'], 'model.layers.0.input_layernorm'
    )
    self.assertEqual(len(details['tensors']), 4)
    for tensor in store.tensors:
      self.assertEqual(store.load(tensor).dtype, np.dtype(bfloat16))
      self.assertEqual(tensor['runtime_layer'], 0)
      self.assertEqual(tensor['layer'], 0)
      self.assertIsNotNone(tensor['sample'])
    self.assertEqual(
        [
            row['metrics']['Max abs error']['value']
            for row in store.compare_batch(0)['rows']
        ],
        [0.0, 1.0],
    )
    self.assertEqual(
        store.session['conversation'][0]['tokens'], results['ref']['tokens']
    )
    self.assertEqual(
        [
            (b['runtime'], b['phase'], b['step'])
            for b in store.session['batches']
        ],
        [('PyTorch', 'prefill', 0)],
    )

  def test_repeated_module_invocations_have_distinct_nodes_and_anchors(self):
    job, results = self.fixture()
    for run in ('ref', 'target'):
      self.change_index(
          job,
          run,
          lambda index: index['tensors'].extend(
              {**tensor, 'invocation': 1, 'call_seq': tensor['call_seq'] + 2}
              for tensor in index['tensors'].copy()
          ),
      )
    store = self.publish(job, results)
    self.assertEqual(
        len(store.execution['executions'][0]['graphs'][0]['nodes']), 2
    )
    self.assertEqual(len(store.semantic['semantic_graph'][0]['anchors']), 4)
    self.assertTrue(
        all(row['status'] == 'ok' for row in store.compare_batch(0)['rows'])
    )

  def test_block_node_and_anchors_have_readable_labels(self):
    job, results = self.fixture()
    for run in ('ref', 'target'):
      self.change_index(
          job,
          run,
          lambda index: [
              tensor.update(module_path='model.layers.0')
              for tensor in index['tensors']
          ],
      )
    store = self.publish(job, results)
    graph = store.semantic['semantic_graph'][0]
    self.assertEqual(graph['nodes'][0]['label'], 'block')
    self.assertEqual(graph['nodes'][0]['module_path'], 'model.layers.0')
    self.assertEqual(
        [anchor['label'] for anchor in graph['anchors']],
        ['block · in[0]', 'block · out[0]'],
    )
    self.assertTrue(
        all(
            'model.layers.0' in anchor['semantic']
            for anchor in graph['anchors']
        )
    )

  def test_run_precision_uses_effective_value_and_preserves_runtime_provenance(
      self,
  ):
    job, results = self.fixture()
    for run in self.record['runs']:
      run['precision'] = 'default'
      results[run['id']]['requested'] = {'precision': 'default'}
    store = self.publish(job, results)
    for run in store.session['runs']:
      self.assertEqual(run['precision'], 'float32')
      provenance = run['provenance']
      self.assertEqual(provenance['requested']['precision'], 'default')
      for key in (
          'effective',
          'capture_scope',
          'torch_version',
          'transformers_version',
      ):
        self.assertEqual(provenance[key], results[run['id']][key])

  def test_unknown_effective_precision_does_not_fall_back_to_requested_value(
      self,
  ):
    job, results = self.fixture()
    for run in self.record['runs']:
      run['precision'] = 'bfloat16'
      results[run['id']]['requested'] = {'precision': 'bfloat16'}
    results['ref']['effective'].pop('precision')
    results['target'].pop('effective')
    store = self.publish(job, results)
    self.assertEqual(
        [run['precision'] for run in store.session['runs']],
        ['Unknown', 'Unknown'],
    )

  def test_same_text_with_different_recorded_mask_is_not_paired(self):
    job, results = self.fixture()
    proof = results['target']['input_proof']
    proof['tensors']['attention_mask']['values'] = [[0, 1]]
    results['target']['input_identity'] = identity(proof)
    save_file(
        {
            key: np.array(metadata['values'], dtype=metadata['dtype'])
            for key, metadata in proof['tensors'].items()
        },
        job / 'target/raw/inputs.safetensors',
    )
    self.change_index(
        job,
        'target',
        lambda index: index.update(input_identity=identity(proof)),
    )
    store = self.publish(job, results)
    self.assertTrue(all(tensor['sample'] is None for tensor in store.tensors))
    self.assertTrue(
        all(
            row['status'] == 'sample_mismatch'
            for row in store.compare_batch(0)['rows']
        )
    )

  def test_different_model_or_missing_input_proof_is_not_paired(self):
    for cause in ('model', 'proof'):
      with self.subTest(cause=cause):
        job, results = self.fixture('unpaired-' + cause)
        if cause == 'model':
          results['target']['model_sha256'] = 'different model'
        else:
          results['target'].pop('input_proof')
          results['target'].pop('input_tensor_path')
        store = self.publish(job, results)
        batch = store.session['batches'][-1]['batch']
        self.assertTrue(
            all(
                row['status'] == 'sample_mismatch'
                for row in store.compare_batch(batch)['rows']
            )
        )

  def test_raw_inputs_and_capture_shards_survive_append_and_job_removal(self):
    job, results = self.fixture()
    first = self.publish(job, results)
    initial = {
        conversation['input_tensor_path']: (
            (first.root / conversation['input_tensor_path']).read_bytes()
        )
        for conversation in first.session['conversation']
    }
    shutil.rmtree(job)
    job, results = self.fixture('job-2', delta=2)
    store = self.publish(job, results)
    shutil.rmtree(job)
    self.assertEqual(len(store.session['turns']), 2)
    self.assertEqual(len(store.semantic['semantic_graph'][0]['anchors']), 2)
    self.assertEqual(len({tensor['path'] for tensor in store.tensors}), 4)
    for relative, raw in initial.items():
      self.assertEqual((store.root / relative).read_bytes(), raw)
    for batch, expected in ((0, 1.0), (1, 2.0)):
      rows = store.compare_batch(batch)['rows']
      self.assertEqual(
          [row['metrics']['Max abs error']['value'] for row in rows],
          [0.0, expected],
      )
    for conversation in store.session['conversation']:
      self.assertTrue(
          (store.root / conversation['input_tensor_path']).is_file()
      )
      self.assertEqual(
          identity(conversation['input_proof']), conversation['input_identity']
      )
    for run in store.session['runs']:
      self.assertTrue(
          (store.root / run['provenance']['input_tensor_path']).is_file()
      )

  def test_mixed_runtime_batches_never_create_implicit_cross_runtime_pairs(
      self,
  ):
    job, results = self.fixture()
    self.record['runs'][1]['runtime'] = 'LiteRT-LM'
    for key in ('input_identity', 'input_proof', 'input_tensor_path'):
      results['target'].pop(key)
    index = {
        'format_version': 2,
        'tensor_root': 'run',
        'runtime': 'LiteRT-LM',
        'tensors': [{
            'format': 'safetensors',
            'path': 'raw/prefill.safetensors',
            'key': 'slot1',
            'shape': [1, 2],
            'dtype': 'float32',
            'signature': 'prefill',
            'section_offset': 100,
            'subgraph': 0,
            'op': 1,
            'output': 0,
            'tensor': 2,
            'output_name': 'norm',
            'tensor_name': 'model/layer_0/pre_attention_norm/output',
            'phase': 'prefill',
            'step': 0,
        }],
    }
    atomic_json(job / 'target/export/capture_index.json', index)
    profile = self.root / 'reviewed.json'
    atomic_json(
        profile,
        {
            'semantic_graph': [{
                'kind': 'reviewed',
                'inputs': [],
                'anchors': [],
                'nodes': [
                    {'id': 'norm.attn.in', 'label': 'Norm', 'incomingEdges': []}
                ],
            }],
            'layers': [{'def': 0}],
        },
    )
    self.artifacts['target']['semantic'] = str(profile)
    store = self.publish(job, results)
    self.assertEqual(len(store.session['batches']), 2)
    self.assertEqual(
        {layer['runtime'] for layer in store.semantic['layers']},
        {'PyTorch', 'LiteRT-LM'},
    )
    self.assertTrue(all(tensor['sample'] is None for tensor in store.tensors))
    self.assertEqual(len({tensor['batch'] for tensor in store.tensors}), 2)
    for batch in store.session['batches']:
      self.assertTrue(
          all(
              row['status'] != 'ok'
              for row in store.compare_batch(batch['batch'])['rows']
          )
      )

  def test_reviewed_graph_is_preserved_without_guessing_module_bindings(self):
    job, results = self.fixture()
    graph = {
        'semantic_graph': [{
            'kind': 'Reviewed architecture',
            'inputs': [],
            'nodes': [{
                'id': 'norm.attn.in',
                'label': 'Reviewed norm',
                'incomingEdges': [],
            }],
            'anchors': [{'id': 'reviewed', 'of': 'norm.attn.in:0'}],
        }],
        'layers': [{'def': 0}],
        'evidence': 'reviewed source',
    }
    profile = self.root / 'pytorch-reviewed.json'
    atomic_json(profile, graph)
    self.artifacts['ref']['semantic'] = str(profile)
    store = self.publish(job, results)
    self.assertEqual(store.semantic['semantic_graph'], graph['semantic_graph'])
    self.assertEqual(store.semantic['evidence'], 'reviewed source')
    self.assertTrue(
        all(
            tensor['anchor'] is None and tensor['layer'] is None
            for tensor in store.tensors
        )
    )

  def test_mismatched_input_proof_or_export_identity_prevents_publication(self):
    for cause in (
        'proof-hash',
        'payload',
        'export-identity',
        'unclaimed-input',
    ):
      with self.subTest(cause=cause):
        job, results = self.fixture('invalid-' + cause)
        if cause == 'proof-hash':
          results['ref']['input_proof']['tokenizer']['vocab_sha256'] = 'changed'
        elif cause == 'payload':
          save_file(
              {
                  'input_ids': np.array([[6, 7]], dtype=np.int64),
                  'attention_mask': np.array([[1, 1]], dtype=np.int64),
              },
              job / 'ref/raw/inputs.safetensors',
          )
        elif cause == 'export-identity':
          self.change_index(
              job,
              'ref',
              lambda index: index.update(input_identity='another forward'),
          )
        else:
          save_file(
              {
                  'input_ids': np.array([[4, 5]], dtype=np.int64),
                  'attention_mask': np.array([[1, 1]], dtype=np.int64),
                  'position_ids': np.array([[0, 1]], dtype=np.int64),
              },
              job / 'ref/raw/inputs.safetensors',
          )
        with self.assertRaises(ValueError):
          publish_capture(
              self.registry, self.record, job, results, self.artifacts
          )
        self.assertFalse((job.parent.parent / 'captures' / job.name).exists())
        self.assertFalse(
            (job.parent.parent / 'captures' / (job.name + '.pending')).exists()
        )

  def test_invalid_or_duplicate_module_coordinates_are_rejected(self):
    for i, update in enumerate((
        {'edge': 'unknown'},
        {'invocation': True},
        {'output': -1},
        {'phase': 'decode'},
    )):
      job, results = self.fixture('bad-module-' + str(i))
      self.change_index(
          job, 'ref', lambda index: index['tensors'][0].update(update)
      )
      with self.subTest(update=update), self.assertRaises(ValueError):
        publish_capture(
            self.registry, self.record, job, results, self.artifacts
        )
    job, results = self.fixture('duplicate-module')
    self.change_index(
        job,
        'ref',
        lambda index: index['tensors'].append(deepcopy(index['tensors'][0])),
    )
    with self.assertRaisesRegex(
        ValueError, 'Duplicate captured module boundary'
    ):
      publish_capture(self.registry, self.record, job, results, self.artifacts)


if __name__ == '__main__':
  unittest.main()
