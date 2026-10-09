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

"""Synthetic evidence tests; real Gemma reports are saved separately."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from model_explorer_debugger.cross_runtime_common import (
    BASIS,
    SEMANTIC,
    _digest,
)
from model_explorer_debugger.cross_runtime_pairs import (
    compare_pair,
    read_pair,
    register_pair,
)
from model_explorer_debugger.node_details import file_digest
from model_explorer_debugger.store import SessionStore
import numpy as np
from safetensors.numpy import save_file


class CrossRuntimePairTests(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name).resolve()
    self.roots = {role: self.root / role for role in ('ref', 'target')}
    self.manifest = {
        'format_version': 1,
        'kind': 'explicit_cross_runtime_prefill',
        'observation': {'turn': 1, 'phase': 'prefill'},
        'mapping': {
            'semantic': SEMANTIC,
            'mask_relation': 'unpadded_causal_prefix',
        },
        'runs': {},
    }
    ids = np.array([[2, 4, 7]], dtype=np.int64)
    prepared = {
        'input_ids': ids,
        'attention_mask': np.ones_like(ids),
        'position_ids': np.array([[0, 1, 2]], dtype=np.int64),
    }
    proof = {
        'scope': 'same_logical_context',
        'cache_context': {
            'processed_token_count': 0,
            'token_ids': [],
            'attention_mask': [],
        },
        'tensors': {
            key: {
                'shape': list(value.shape),
                'dtype': value.dtype.name,
                'values': value.tolist(),
            }
            for key, value in prepared.items()
        },
        'tokenizer': {'vocab_sha256': 'c' * 64},
        'serialization': {'template_sha256': 'd' * 64},
    }
    for role, runtime in (('ref', 'PyTorch'), ('target', 'LiteRT-LM')):
      root = self.roots[role]
      (root / 'raw').mkdir(parents=True)
      (root / 'export').mkdir()
      model = {
          'family': 'gemma-4-E2B',
          'variant': 'it',
          'artifact_sha256': ('a' if role == 'ref' else 'b') * 64,
          'source_chain': [],
      }
      run = self.manifest['runs'][role] = {
          'runtime': runtime,
          'model': model,
          'semantic_evidence': [
              self.write_json(role, 'semantic.json', {'declaration': SEMANTIC})
          ],
      }
      if role == 'ref':
        model.update(
            revision='f' * 40,
            checkpoint_sha256='e' * 64,
            config_sha256='1' * 64,
        )
        model['source_chain'] = [
            self.write_json(
                role,
                'source.json',
                {
                    'repo_id': 'google/gemma-4-E2B-it',
                    'revision': 'f' * 40,
                    'files': [
                        {'path': 'model.safetensors', 'local_sha256': 'e' * 64},
                        {'path': 'config.json', 'local_sha256': '1' * 64},
                    ],
                },
            )
        ]
        raw = np.arange(6, dtype=np.float32).reshape(1, 3, 2)
        slot = 'norm_output'
        record = {
            'format': 'safetensors',
            'path': 'raw/anchor.safetensors',
            'key': slot,
            'shape': list(raw.shape),
            'dtype': 'float32',
            'module_path': 'model.layers.0.input_layernorm',
            'module_type': 'Gemma4RMSNorm',
            'edge': 'out',
            'forward_id': 0,
            'phase': 'prefill',
            'step': 0,
        }
        save_file({slot: raw}, root / 'raw/anchor.safetensors')
        save_file(prepared, root / 'raw/inputs.safetensors')
        save_file(prepared, root / 'raw/boundaries.safetensors')
        run['input_tensor_file'] = self.file(role, 'raw/inputs.safetensors')
        run['input_boundary_files'] = [
            self.file(role, 'raw/boundaries.safetensors')
        ]
        inputs = [
            {
                'key': key,
                'output_path': ['kwargs', key],
                'shape': list(value.shape),
                'dtype': 'torch.int64',
            }
            for key, value in prepared.items()
        ]
        index = {
            'format_version': 2,
            'tensor_root': 'run',
            'tensors': [record],
            'forwards': [{
                'forward_id': 0,
                'status': 'completed',
                'phase': 'prefill',
                'pos_offset': 0,
                'turn': 1,
                'inputs': inputs,
            }],
            'forward_input_proofs': [{
                'forward_id': 0,
                'input_identity': _digest(proof),
                'input_proof': proof,
            }],
            'resources': [
                {
                    'scope': 'boundary',
                    'capture_key': key,
                    'forward_id': 0,
                    'format': 'safetensors',
                    'path': 'raw/boundaries.safetensors',
                    'key': key,
                    'shape': list(value.shape),
                    'dtype': 'int64',
                }
                for key, value in prepared.items()
            ],
        }
        files = [
            {'path': 'model.safetensors', 'size': 100, 'sha256': 'e' * 64},
            {'path': 'config.json', 'size': 20, 'sha256': '1' * 64},
            {'path': 'tokenizer.json', 'size': 20, 'sha256': 'c' * 64},
            {'path': 'chat_template.jinja', 'size': 20, 'sha256': 'd' * 64},
        ]
        model['artifact_sha256'] = _digest(files, ensure_ascii=True)
        result = {
            'runtime': runtime,
            'turn_sequence': 1,
            'model_sha256': model['artifact_sha256'],
            'model_files': files,
            'input_identity': _digest(proof),
            'input_proof': proof,
            'input_tensor_path': 'raw/inputs.safetensors',
        }
        selector = {
            key: record[key] for key in ('module_path', 'edge', 'forward_id')
        }
      else:
        raw = np.full((1, 8, 2), 999, dtype=np.float32)
        raw[:, :2, :] = np.arange(4).reshape(1, 2, 2) + 1
        slot = 'post_norm'
        record = {
            'format': 'safetensors',
            'path': 'raw/anchor.safetensors',
            'key': slot,
            'shape': list(raw.shape),
            'dtype': 'float32',
            'signature': 'prefill_8',
            'subgraph': 2,
            'op': 12,
            'output': 0,
            'tensor': 45,
            'phase': 'prefill',
            'step': 3,
            'tensor_name': (
                'model/layer_0/layer_0.pre_qkv/pre_attention_norm/composite'
            ),
        }
        save_file(
            {slot: raw},
            root / 'raw/anchor.safetensors',
            metadata={'signature': 'prefill_8', 'step': '3'},
        )
        admitted = self.tensor(
            role, 'raw/admitted.safetensors', 'input_ids', ids.astype(np.int32)
        )
        pos = np.zeros(8, dtype=np.int32)
        pos[:2] = np.arange(2)
        mask = np.zeros((1, 1, 8, 16), dtype=np.bool_)
        mask[0, 0, :2, :2] = np.tri(2, dtype=np.bool_)
        graph = {
            name: {
                **self.tensor(
                    role,
                    'raw/' + name + '.safetensors',
                    name,
                    value,
                    metadata={'signature': 'prefill_8', 'step': '3'},
                ),
                'signature': 'prefill_8',
                'step': 3,
                'valid_length': 2,
            }
            for name, value in [('position_ids', pos), ('attention_mask', mask)]
        }
        (root / 'native-library').write_bytes(b'synthetic native library')
        run['native_library'] = self.file(role, 'native-library')
        run['native_source_lock'] = self.write_json(
            role, 'native-source.json', {'revision': 'synthetic'}
        )
        native = {
            'library_sha256': run['native_library']['sha256'],
            'source_lock': {'revision': 'synthetic'},
        }
        replay = {
            'format_version': 1,
            'mode': 'first_prefill_exact_token_ids',
            'actual_token_ids': ids[0].tolist(),
            'accepted_token_count': 3,
            'graph_valid_length': 2,
            'graph_token_ids': [2, 4],
            'pending_token_id': 7,
            'pending_token_position': 2,
            'input_resource': admitted,
            'graph_inputs': graph,
            'requested_source': {
                'source_model_sha256': 'e' * 64,
                'source_revision': 'f' * 40,
                'config_sha256': '1' * 64,
                'tokenizer': proof['tokenizer'],
                'serialization': proof['serialization'],
            },
            'native': native,
            'weight_equivalence': 'unverified',
            'quantization_profile': 'unknown',
        }
        replay['proof_sha256'] = _digest(replay, ensure_ascii=True)
        result = {
            'runtime': runtime,
            'turn_sequence': 1,
            'model_sha256': model['artifact_sha256'],
            'token_replay_proof': replay,
        }
        index = {
            'format_version': 2,
            'tensor_root': 'run',
            'tapped_sha256': '7' * 64,
            'tensors': [record],
        }
        run['semantic_evidence'] = [
            self.write_json(
                role,
                'tap-evidence.json',
                {'tapped_sha256': '7' * 64, 'taps': [record]},
            )
        ]
        model['source_chain'] = [
            self.write_json(
                role,
                'source.json',
                {
                    'repo': 'litert-community/gemma-4-E2B-it-litert-lm',
                    'revision': '9' * 40,
                    'local_original_sha256': '8' * 64,
                    'files': [{
                        'filename': 'gemma-4-E2B-it.litertlm',
                        'lfs': {'sha256': '8' * 64},
                    }],
                },
            ),
            self.write_json(
                role,
                'conversion.json',
                {
                    'original_container_sha256': '8' * 64,
                    'tapped_container_sha256': 'b' * 64,
                    'source_sha256': '7' * 64,
                },
            ),
        ]
        selector = {
            key: record[key] for key in ('signature', 'op', 'output', 'tensor')
        }
      run['result'] = self.write_json(role, 'result.json', result)
      run['capture_index'] = self.write_json(
          role, 'export/capture_index.json', index
      )
      run['anchor'] = {
          'tensor': {**record, **self.file(role, 'raw/anchor.safetensors')},
          'selector': selector,
          'layout': ['batch', 'sequence', 'hidden'],
          'view': [
              {'start': 0, 'stop': 1, 'step': 1},
              {'start': 0, 'stop': 2, 'step': 1},
              {'start': 0, 'stop': 2, 'step': 1},
          ],
      }

  def file(self, role, path):
    return {'path': path, 'sha256': file_digest(self.roots[role] / path)}

  def write_json(self, role, path, data):
    (self.roots[role] / path).write_text(json.dumps(data))
    return self.file(role, path)

  def tensor(self, role, path, key, value, **kwargs):
    save_file({key: value}, self.roots[role] / path, **kwargs)
    return {
        **self.file(role, path),
        'format': 'safetensors',
        'key': key,
        'shape': list(value.shape),
        'dtype': value.dtype.name,
    }

  def change_result(self, role, change):
    path = self.roots[role] / 'result.json'
    result = json.loads(path.read_text())
    change(result)
    if role == 'target':
      proof = result['token_replay_proof']
      proof.pop('proof_sha256', None)
      proof['proof_sha256'] = _digest(proof, ensure_ascii=True)
    self.manifest['runs'][role]['result'] = self.write_json(
        role, 'result.json', result
    )

  def position_fixture(self, position):
    ref_index = json.loads(
        (self.roots['ref'] / 'export/capture_index.json').read_text()
    )
    ref_index['forwards'][0]['step'] = 0
    first = json.loads((self.roots['ref'] / 'result.json').read_text())[
        'input_proof'
    ]
    inputs = {
        'input_ids': np.array([[9]], dtype=np.int64),
        'attention_mask': np.ones((1, 4), dtype=np.int64),
        'cache_position': np.array([3], dtype=np.int64),
    }
    save_file(inputs, self.roots['ref'] / 'raw/decode-boundaries.safetensors')
    proof = {
        **deepcopy(first),
        'cache_context': {
            'processed_token_count': 3,
            'token_ids': [2, 4, 7],
            'attention_mask': [1, 1, 1],
        },
        'tensors': {
            key: {
                'shape': list(value.shape),
                'dtype': value.dtype.name,
                'values': value.tolist(),
            }
            for key, value in inputs.items()
        },
    }
    ref_index['forwards'].append({
        'forward_id': 1,
        'phase': 'decode',
        'step': 0,
        'pos_offset': 3,
        'turn': 1,
        'status': 'completed',
        'cache_before': {'state': 'available', 'processed_token_count': 3},
        'inputs': [
            {
                'key': 'f1:' + key,
                'output_path': ['kwargs', key],
                'shape': list(value.shape),
                'dtype': 'torch.int64',
            }
            for key, value in inputs.items()
        ],
    })
    ref_index['resources'].extend(
        {
            'scope': 'boundary',
            'forward_id': 1,
            'capture_key': 'f1:' + key,
            'path': 'raw/decode-boundaries.safetensors',
            'key': key,
            'format': 'safetensors',
            'shape': list(value.shape),
            'dtype': 'int64',
        }
        for key, value in inputs.items()
    )
    ref_index['forward_input_proofs'].append({
        'forward_id': 1,
        'input_proof': proof,
        'input_identity': _digest(proof),
    })
    record = {
        **deepcopy(ref_index['tensors'][0]),
        **self.tensor(
            'ref',
            'raw/decode-anchor.safetensors',
            'norm_output',
            np.array([[[6, 7]]], dtype=np.float32),
        ),
        'forward_id': 1,
        'phase': 'decode',
        'step': 0,
    }
    record.pop('sha256')
    ref_index['tensors'].append(record)
    self.manifest['runs']['ref']['capture_index'] = self.write_json(
        'ref', 'export/capture_index.json', ref_index
    )
    target_index = json.loads(
        (self.roots['target'] / 'export/capture_index.json').read_text()
    )
    executions = []
    for pos, input_id, emission in ((2, 7, 9), (3, 9, 10)):
      mask = np.zeros((1, 1, 1, 16), dtype=np.bool_)
      mask[..., : pos + 1] = True
      meta = {'signature': 'decode', 'step': str(pos + 1)}
      graph = {
          name: self.tensor(
              'target',
              f'raw/decode-{pos}-{name}.safetensors',
              name,
              value,
              metadata=meta,
          )
          for name, value in [
              ('position_ids', np.array([pos], dtype=np.int32)),
              ('attention_mask', mask),
          ]
      }
      executions.append({
          'signature': 'decode',
          'step': pos + 1,
          'input_token_id': input_id,
          'input_position': pos,
          'emitted_token_id': emission,
          'processed_token_count_before': pos,
          'processed_token_count_after': pos + 1,
          'graph_inputs': graph,
      })
      tensor = self.tensor(
          'target',
          f'raw/decode-{pos}-anchor.safetensors',
          'post_norm',
          np.array([[[2 * pos + 1, 2 * pos + 2]]], dtype=np.float32),
          metadata=meta,
      )
      tensor.pop('sha256')
      target_index['tensors'].append({
          **tensor,
          'signature': 'decode',
          'subgraph': 0,
          'op': 10,
          'output': 0,
          'tensor': 55,
          'phase': 'decode',
          'step': pos + 1,
          'tensor_name': (
              'model/layer_0/layer_0.pre_qkv/pre_attention_norm/composite'
          ),
      })
    decode = {
        'sampler_policy': 'fixed_token_sequence_constraint',
        'model_computation_modified': False,
        'actual_output_token_ids': [9, 10],
        'executions': executions,
        'pending_token_id': 10,
        'pending_token_position': 4,
        'processed_token_count': 4,
    }
    self.change_result(
        'target',
        lambda result: result['token_replay_proof'].update(decode=decode),
    )
    self.manifest['runs']['target']['capture_index'] = self.write_json(
        'target', 'export/capture_index.json', target_index
    )
    self.manifest['runs']['target']['semantic_evidence'] = [
        self.write_json(
            'target',
            'tap-evidence.json',
            {'tapped_sha256': '7' * 64, 'taps': target_index['tensors'][:2]},
        )
    ]
    fid = 0 if position == 2 else 1
    pair = {
        'format_version': 2,
        'kind': 'explicit_cross_runtime_position',
        'prefill': deepcopy(self.manifest),
        'observation': {'turn': 1, 'phase': 'decode'},
        'position': position,
        'runs': {},
    }
    for role in ('ref', 'target'):
      if role == 'ref':
        tensor = ref_index['tensors'][fid]
        forward = ref_index['forwards'][fid]
        observation = {
            'turn': 1,
            **{
                key: forward[key]
                for key in ('phase', 'step', 'forward_id', 'pos_offset')
            },
        }
        selector = {
            key: tensor[key] for key in ('module_path', 'edge', 'forward_id')
        }
        local = position - forward['pos_offset']
      else:
        tensor = target_index['tensors'][1 + (position - 2)]
        observation = {
            'turn': 1,
            'phase': 'decode',
            'signature': 'decode',
            'step': position + 1,
        }
        selector = {
            key: tensor[key] for key in ('signature', 'op', 'output', 'tensor')
        }
        local = 0
      pair['runs'][role] = {
          'observation': observation,
          'anchor': {
              'tensor': {**tensor, **self.file(role, tensor['path'])},
              'selector': selector,
              'layout': ['batch', 'sequence', 'hidden'],
              'view': [
                  {'start': 0, 'stop': 1, 'step': 1},
                  {'start': local, 'stop': local + 1, 'step': 1},
                  {'start': 0, 'stop': 2, 'step': 1},
              ],
          },
      }
      if role == 'ref':
        pair['runs'][role]['input_boundary_files'] = [
            self.file(
                role,
                'raw/boundaries.safetensors'
                if fid == 0
                else 'raw/decode-boundaries.safetensors',
            )
        ]
    return pair

  def test_explicit_prefix_view_preserves_original_and_reports_difference(self):
    before = (self.roots['target'] / 'raw/anchor.safetensors').read_bytes()
    report = compare_pair(self.manifest, self.roots)
    self.assertEqual(report['shape'], [1, 2, 2])
    self.assertEqual(report['metrics']['Max abs error']['value'], 1.0)
    self.assertEqual(
        (report['processed_token_ids'], report['pending_token_id']), ([2, 4], 7)
    )
    self.assertEqual(report['comparison_basis'], BASIS)
    self.assertEqual(report['weight_equivalence'], 'unverified')
    self.assertEqual(
        (self.roots['target'] / 'raw/anchor.safetensors').read_bytes(), before
    )

  def test_registration_copies_all_evidence_and_reloads_without_source_runs(
      self,
  ):
    destination = self.root / 'saved'
    report = register_pair(destination, self.manifest, self.roots)
    self.assertEqual(
        register_pair(destination, self.manifest, self.roots), report
    )
    shutil.rmtree(self.roots['ref'])
    shutil.rmtree(self.roots['target'])
    self.assertEqual(read_pair(destination, report['pair_id']), report)

  def test_no_implicit_slice_or_semantic_mapping(self):
    for cause in (
        'view',
        'padded',
        'hidden',
        'semantic',
        'selector',
        'layout',
        'variant',
        'checksum',
        'outside',
    ):
      with self.subTest(cause=cause):
        manifest = deepcopy(self.manifest)
        run = manifest['runs']['target']
        if cause == 'view':
          run['anchor'].pop('view')
        elif cause == 'padded':
          run['anchor']['view'][1]['stop'] = 8
        elif cause == 'hidden':
          run['anchor']['view'][2]['stop'] = 1
        elif cause == 'semantic':
          manifest['mapping']['semantic'] = 'layer.0.another.output'
        elif cause == 'selector':
          run['anchor']['selector']['op'] = 13
        elif cause == 'layout':
          run['anchor']['layout'] = ['batch', 'hidden', 'sequence']
        elif cause == 'variant':
          run['model']['variant'] = 'base'
        elif cause == 'checksum':
          run['anchor']['tensor']['sha256'] = '0' * 64
        else:
          run['anchor']['tensor']['path'] = '../ref/raw/anchor.safetensors'
        with self.assertRaises(ValueError):
          compare_pair(manifest, self.roots)

  def test_native_readback_mismatch_is_rejected_even_with_rehashed_json(self):
    self.change_result(
        'target',
        lambda result: result['token_replay_proof'][
            'actual_token_ids'
        ].__setitem__(0, 99),
    )
    with self.assertRaises(ValueError):
      compare_pair(self.manifest, self.roots)

  def test_captured_mask_must_match_full_causal_proof_including_padding(self):
    result = json.loads((self.roots['target'] / 'result.json').read_text())
    ref = result['token_replay_proof']['graph_inputs']['attention_mask']
    mask = np.zeros(ref['shape'], dtype=np.bool_)
    mask[0, 0, :2, :2] = np.tri(2, dtype=np.bool_)
    mask[0, 0, 5, 5] = True
    updated = self.tensor(
        'target',
        ref['path'],
        ref['key'],
        mask,
        metadata={'signature': 'prefill_8', 'step': '3'},
    )
    self.change_result(
        'target',
        lambda item: item['token_replay_proof']['graph_inputs'][
            'attention_mask'
        ].update(updated),
    )
    with self.assertRaisesRegex(ValueError, 'causal prefix'):
      compare_pair(self.manifest, self.roots)

  def test_input_preparation_cannot_replace_actual_boundary_evidence(self):
    boundary = self.roots['ref'] / 'raw/boundaries.safetensors'
    save_file(
        {
            'input_ids': np.array([[9, 4, 7]], dtype=np.int64),
            'attention_mask': np.ones((1, 3), dtype=np.int64),
            'position_ids': np.array([[0, 1, 2]], dtype=np.int64),
        },
        boundary,
    )
    self.manifest['runs']['ref']['input_boundary_files'] = [
        self.file('ref', 'raw/boundaries.safetensors')
    ]
    with self.assertRaisesRegex(ValueError, 'executed PyTorch input'):
      compare_pair(self.manifest, self.roots)

  def test_store_only_attaches_to_matching_model_runtime_turn_and_raw_tensor(
      self,
  ):
    store = object.__new__(SessionStore)
    store.root = self.root / 'capture'
    store.session = {
        'turns': [{'n': 1}],
        'runs': [
            {
                'id': role,
                'runtime': run['runtime'],
                'provenance': {'model_sha256': run['model']['artifact_sha256']},
            }
            for role, run in self.manifest['runs'].items()
        ],
    }
    store.tensors = [
        {**run['anchor']['tensor'], 'run': role, 'turn': 1, 'phase': 'prefill'}
        for role, run in self.manifest['runs'].items()
    ]
    report = store.register_pair(self.manifest, self.roots)
    self.assertEqual(store.compare_pair(report['pair_id']), report)
    self.assertEqual(store.explicit_pairs(), [report])
    for field in ('runtime', 'model', 'turn', 'tensor'):
      with self.subTest(field=field):
        original_session, original_tensors = deepcopy(store.session), deepcopy(
            store.tensors
        )
        if field == 'runtime':
          store.session['runs'][1]['runtime'] = 'PyTorch'
        elif field == 'model':
          store.session['runs'][0]['provenance']['model_sha256'] = '0' * 64
        elif field == 'turn':
          store.session['turns'] = [{'n': 2}]
        else:
          store.tensors[0]['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
          store.register_pair(self.manifest, self.roots)
        store.session, store.tensors = original_session, original_tensors

  def test_position_pair_preserves_cross_phase_identity_and_one_row_range(self):
    for position in (2, 3):
      with self.subTest(position=position):
        # A fresh fixture avoids duplicate synthetic Decode records.
        if position == 3:
          self.setUp()
        pair = self.position_fixture(position)
        report = compare_pair(pair, self.roots)
        self.assertEqual(
            report['compared_position_range'], [position, position + 1]
        )
        self.assertEqual(report['shape'], [1, 1, 2])
        self.assertEqual(report['processed_token_count'], position + 1)
        self.assertEqual(
            report['runs']['ref']['observation']['phase'],
            'prefill' if position == 2 else 'decode',
        )
        self.assertEqual(
            report['runs']['target']['observation']['phase'], 'decode'
        )
        self.assertEqual(report['metrics']['Max abs error']['value'], 1.0)

  def test_position_pair_rejects_faked_phase_prefix_or_actual_mask(self):
    pair = self.position_fixture(3)
    for cause in ('phase', 'position', 'prefix', 'mask'):
      with self.subTest(cause=cause):
        mutated = deepcopy(pair)
        if cause == 'phase':
          mutated['runs']['ref']['observation']['phase'] = 'prefill'
        elif cause == 'position':
          mutated['position'] = 2
        elif cause == 'prefix':
          index = json.loads(
              (self.roots['ref'] / 'export/capture_index.json').read_text()
          )
          row = index['forward_input_proofs'][1]
          row['input_proof']['cache_context']['token_ids'][0] = 99
          row['input_identity'] = _digest(row['input_proof'])
          mutated['prefill']['runs']['ref']['capture_index'] = self.write_json(
              'ref', 'bad-index.json', index
          )
          # Export roots are defined by index location; keep the new index
          # beside the original.
          mutated['prefill']['runs']['ref']['capture_index'] = self.write_json(
              'ref', 'export/bad-index.json', index
          )
        else:
          result = json.loads(
              (self.roots['target'] / 'result.json').read_text()
          )
          proof = result['token_replay_proof']
          row = proof['decode']['executions'][1]['graph_inputs'][
              'attention_mask'
          ]
          mask = np.ones((1, 1, 1, 16), dtype=np.bool_)
          row.update(
              self.tensor(
                  'target',
                  'raw/bad-mask.safetensors',
                  row['key'],
                  mask,
                  metadata={'signature': 'decode', 'step': '4'},
              )
          )
          proof.pop('proof_sha256')
          proof['proof_sha256'] = _digest(proof, ensure_ascii=True)
          mutated['prefill']['runs']['target']['result'] = self.write_json(
              'target', 'bad-result.json', result
          )
        with self.assertRaises(ValueError):
          compare_pair(mutated, self.roots)

  def test_position_pair_registration_is_independent_of_original_runs(self):
    pair = self.position_fixture(2)
    destination = self.root / 'registered'
    report = register_pair(destination, pair, self.roots)
    shutil.rmtree(self.roots['ref'])
    shutil.rmtree(self.roots['target'])
    self.assertEqual(read_pair(destination, report['pair_id']), report)

  def test_source_path_with_parent_segments_cannot_escape_when_relocated(self):
    # This source path resolves inside its source root, but is unsafe to copy
    # under a new parent. Reject the lexical traversal before any write.
    self.manifest['runs']['target']['anchor']['tensor'][
        'path'
    ] = '../target/raw/anchor.safetensors'
    destination = self.root / 'registered'
    with self.assertRaisesRegex(ValueError, 'invalid file reference'):
      register_pair(destination, self.manifest, self.roots)
    self.assertFalse(destination.exists())

  def test_registered_role_symlink_cannot_read_outside_saved_pair(self):
    destination = self.root / 'registered'
    report = register_pair(destination, self.manifest, self.roots)
    role = destination / report['pair_id'] / 'target'
    shutil.rmtree(role)
    role.symlink_to(self.roots['target'], target_is_directory=True)
    with self.assertRaisesRegex(ValueError, 'outside pair'):
      read_pair(destination, report['pair_id'])

  def test_loaded_hf_files_are_bound_even_when_replay_uses_the_same_hash_domain(
      self,
  ):
    self.change_result('ref', lambda result: result.pop('model_files'))
    with self.assertRaisesRegex(ValueError, 'loaded HF file identity'):
      compare_pair(self.manifest, self.roots)

  def test_first_turn_evidence_cannot_be_relabelled_as_another_turn(self):
    self.manifest['observation']['turn'] = 999
    with self.assertRaises(ValueError):
      compare_pair(self.manifest, self.roots)

  def test_self_consistent_bad_semantic_index_cannot_alias_language_norm(self):
    for role in ('ref', 'target'):
      with self.subTest(role=role):
        manifest = deepcopy(self.manifest)
        run = manifest['runs'][role]
        index = json.loads(
            (self.roots[role] / 'export/capture_index.json').read_text()
        )
        if role == 'ref':
          index['tensors'][0][
              'module_path'
          ] = 'vision_tower.encoder.layers.0.input_layernorm'
          run['anchor']['selector']['module_path'] = index['tensors'][0][
              'module_path'
          ]
        else:
          index['tensors'][0]['op'] = 13
          run['anchor']['selector']['op'] = 13
        run['capture_index'] = self.write_json(
            role, 'export/semantic-mismatch.json', index
        )
        with self.assertRaises(ValueError):
          compare_pair(manifest, self.roots)

  def test_module_class_must_be_rmsnorm_even_at_the_correct_path(self):
    index = json.loads(
        (self.roots['ref'] / 'export/capture_index.json').read_text()
    )
    index['tensors'][0]['module_type'] = 'Linear'
    self.manifest['runs']['ref']['capture_index'] = self.write_json(
        'ref', 'export/wrong-class.json', index
    )
    with self.assertRaisesRegex(ValueError, 'semantic anchor'):
      compare_pair(self.manifest, self.roots)


if __name__ == '__main__':
  unittest.main()
