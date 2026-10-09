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

"""Inferred native generation evidence from RuntimeDebugger dumps.

All tensors are synthetic.
"""

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

from model_explorer_debugger import capture_importer
from model_explorer_debugger import token_analysis
from model_explorer_debugger.capture_telemetry import (
    append_telemetry,
    empty_telemetry,
    pair_forward_evidence,
)
from model_explorer_debugger.kv_reader import KvReader
from model_explorer_debugger.runtime.litert_dump_inference import (
    BASIS,
    CAPTURE_SCOPE,
    classify_input_tokens,
    infer_generation,
    released_tokens,
    verify_inferred_index,
)
from model_explorer_debugger.runtime.litert_kv_dump import DUMP_BINDING
from model_explorer_debugger.runtime.litert_storage import (
    NativeStorageError,
    REFERENCE,
    normalize_webgpu_kv,
    prepare_saved_resources,
    webgpu_logical_blocks,
)
from model_explorer_debugger.tensor_io import load_tensor
from model_explorer_debugger.token_analysis import analyze_tokens
from model_explorer_debugger.token_sources import link_token_batches
import numpy as np
from safetensors.numpy import save_file
import sentencepiece as spm

MODEL = 'ab' * 32
VOCAB, HIDDEN = 12, 4
GREEDY = {'temperature': 0, 'topK': 1, 'topP': 1, 'seed': 0}


SECTION = 'cd' * 32
CACHE, HEAD_DIM = 8, 4
KV_SLOTS = (
    (0, 'key', [1, 1, CACHE, HEAD_DIM]),
    (0, 'value', [1, 1, HEAD_DIM, CACHE]),
    (1, 'key', [1, 1, CACHE, HEAD_DIM]),
)
GPU_BUILD = {'sourceLock': {'litert': REFERENCE['source_revision']}}
GPU_EVIDENCE = {
    'effectiveBackend': 'GPU',
    'engineInitialized': True,
    'libraries': [{'name': 'libwebgpu_dawn.dylib', 'binarySHA256': 'ef' * 32}],
}


def kv_model(signatures=('prefill_128', 'decode')):
  """Model cache descriptors as describe_model reports them.

  Describes a two-slot toy cache.
  """
  descriptors = {}
  for signature in signatures:
    for layer, kind, shape in KV_SLOTS:
      name = f"kv_cache_{'k' if kind == 'key' else 'v'}_{layer}"
      layout = (
          ['batch', 'kv_head', 'sequence', 'head_dim']
          if kind == 'key'
          else ['batch', 'kv_head', 'head_dim', 'sequence']
      )
      descriptors[signature, name] = dict(
          layer=layer,
          kind=kind,
          slot=layer,
          layout=layout,
          model_shape=[1, 1, 4096, HEAD_DIM]
          if kind == 'key'
          else [1, 1, HEAD_DIM, 4096],
          quantization=dict(
              scale=0.01 * (layer + 1), zero_point=0, quantized_dimension=0
          ),
          source=dict(
              section_offset=0,
              section_sha256=SECTION,
              signature=signature,
              subgraph=0,
              op=layer,
              output_tensor=layer,
              composite='odml.cache_update',
              owner_input_tensor_names=[
                  f'/layer_{layer}/k',
                  f'/layer_{layer}/v',
              ],
              composite_attributes={
                  'cache_size': 4096,
                  'head_size': HEAD_DIM,
                  'kv_cache_batch_size': 1,
              },
          ),
      )
  return dict(
      kv=descriptors,
      section_sha256=SECTION,
      tokenizer=dict(vocab_sha256='00' * 32),
  )


def webgpu_bytes(logical, kind):
  """The delegate's physical, unsigned byte order for a logical int8 tensor.

  This is the inverse of normalization.
  """
  shape = list(logical.shape)
  raw = np.zeros(logical.size, dtype=np.uint8)
  for start, stop, indices in webgpu_logical_blocks(kind, shape):
    block = (
        logical[0, :, start:stop, :]
        if kind == 'key'
        else logical[0, :, :, start:stop].transpose(0, 2, 1)
    )
    raw[indices] = (block.astype(np.int16) + 128).astype(np.uint8)
  return raw.view(np.int8).reshape(shape)


def write_dump(
    root,
    *,
    prefill_step=3,
    decode_steps=(3, 4, 5),
    argmaxes=(7, 2, 9),
    released=(7, 2),
    positions=None,
    seed=0,
    prefill_rows=1,
    kv=False,
    gpu=False,
    mask_columns=None,
):
  """A tiny debugger dump: one Prefill chunk, Decode steps and a tap.

  Decode steps carry logits/activations/input_pos. With ``kv``, the Prefill step
  and the last Decode step also carry KV cache shards (as the unpatched runtime
  dumps them); ``gpu`` writes them in the WebGPU physical byte order.
  """
  raw = Path(root) / 'raw'
  raw.mkdir(parents=True)
  rng = np.random.default_rng(seed)

  def save(name, key, array, signature, step):
    save_file(
        {key: array},
        raw / f'{signature}_{name}_step_{step}.safetensors',
        metadata={'signature': signature, 'step': str(step)},
    )

  save(
      'pre_input_pos',
      'pre_input_pos',
      np.arange(prefill_rows, dtype=np.int32),
      'prefill_128',
      prefill_step,
  )
  if kv:
    columns = prefill_rows if mask_columns is None else mask_columns
    mask = np.zeros([1, 1, prefill_rows, CACHE], dtype=bool)
    mask[0, 0, :, :columns] = True
    save('pre_mask', 'pre_mask', mask, 'prefill_128', prefill_step)
    for signature, step in (
        ('prefill_128', prefill_step),
        ('decode', decode_steps[-1]),
    ):
      for layer, kind, shape in KV_SLOTS:
        key = f"post_kv_cache_{'k' if kind == 'key' else 'v'}_{layer}"
        logical = rng.integers(-127, 128, size=shape, dtype=np.int8)
        save(
            key,
            key,
            webgpu_bytes(logical, kind) if gpu else logical,
            signature,
            step,
        )
  save(
      'post_tap_x',
      'post_tap_x',
      rng.standard_normal([1, 2, HIDDEN]).astype(np.float32),
      'prefill_128',
      prefill_step,
  )
  positions = (
      list(positions)
      if positions is not None
      else [step - 1 for step in decode_steps]
  )
  for step, argmax, position in zip(decode_steps, argmaxes, positions):
    logits = rng.standard_normal([1, 1, VOCAB]).astype(np.float32)
    logits[0, 0, argmax] = 10.0
    save('post_logits', 'post_logits', logits, 'decode', step)
    save(
        'post_activations',
        'post_activations',
        rng.standard_normal([1, 1, HIDDEN]).astype(np.float32),
        'decode',
        step,
    )
    save(
        'pre_input_pos',
        'pre_input_pos',
        np.array([position], dtype=np.int32),
        'decode',
        step,
    )
    save(
        'post_tap_x',
        'post_tap_x',
        rng.standard_normal([1, 1, HIDDEN]).astype(np.float32),
        'decode',
        step,
    )
  (raw / 'generated_tokens.jsonl').write_text(
      ''.join(
          json.dumps(
              {'token_ids': [[token]], 'texts': [f't{token}'], 'scores': [0.0]}
          )
          + '\n'
          for token in released
      )
  )
  tensors = [
      dict(
          signature='prefill_128',
          key='post_tap_x',
          step=prefill_step,
          phase='prefill',
          path=f'raw/prefill_128_post_tap_x_step_{prefill_step}.safetensors',
          format='safetensors',
          dtype='float32',
          shape=[1, 2, HIDDEN],
      )
  ]
  tensors += [
      dict(
          signature='decode',
          key='post_tap_x',
          step=step,
          phase='decode',
          format='safetensors',
          path=f'raw/decode_post_tap_x_step_{step}.safetensors',
          dtype='float32',
          shape=[1, 1, HIDDEN],
      )
      for step in decode_steps
  ]
  return {'format_version': 2, 'tensor_root': 'export', 'tensors': tensors}


def result(sampler=GREEDY, speculative=False, gpu=False):
  outcome = {
      'sampler': dict(sampler),
      'speculativeDecodingEnabled': speculative,
      'modelSHA256': MODEL,
  }
  if gpu:
    outcome['backendEvidence'] = deepcopy(GPU_EVIDENCE)
  return outcome


JOB = {'modelSHA256': MODEL, 'messages': [], 'prompt': 'hi'}


class DumpInferenceTests(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)

  def infer(self, **dump):
    index = write_dump(self.root, **dump)
    infer_generation(index, self.root / 'raw', result(), JOB, turn=1)
    return index

  def test_greedy_dump_yields_verified_bindings(self):
    index = self.infer()
    self.assertEqual(index['inference']['status'], 'ok', index['inference'])
    self.assertEqual(
        (index['capture_scope'], index['export_scope'], index['runtime']),
        (CAPTURE_SCOPE, 'all', 'LiteRT-LM'),
    )
    phases = [
        (f['forward_id'], f['phase'], f['step']) for f in index['forwards']
    ]
    self.assertEqual(
        phases,
        [
            (1, 'prefill', 3),
            (2, 'decode', 3),
            (3, 'decode', 4),
            (4, 'decode', 5),
        ],
    )
    self.assertEqual(
        [
            (
                r['forward_id'],
                r['output_index'],
                r['token_ids'],
                r['consumption'][0]['consumed_by_forward_id'],
            )
            for r in index['token_records']
        ],
        [(2, 0, [7], 3), (3, 1, [2], 4)],
    )
    self.assertEqual(index['generation']['unreleased_argmax'], 9)
    self.assertEqual([t['forward_id'] for t in index['tensors']], [1, 2, 3, 4])
    self.assertTrue(
        all(
            f['vocab_identity'] == f'{MODEL}:{VOCAB}' for f in index['forwards']
        )
    )
    self.assertTrue(all(f['comparison_eligible'] for f in index['forwards']))
    decode = [
        f['input_identity'] for f in index['forwards'] if f['phase'] == 'decode'
    ]
    self.assertEqual(
        len(set(decode)), 3, 'each Decode has its own released-prefix context'
    )
    outputs = {
        tuple(item['output_path'])
        for f in index['forwards']
        if f['phase'] == 'decode'
        for item in f['outputs']
    }
    self.assertEqual(outputs, {('output', 'logits'), ('output', 'activations')})
    self.assertTrue(
        all(
            r['basis'] == BASIS
            for r in index['resources'] + index['token_records']
        )
    )
    self.assertEqual([t['id'] for t in index['tokens']], [7, 2])
    self.assertTrue(verify_inferred_index(index, self.root))

  def test_runner_reported_input_tokens_reconcile_with_context_growth(self):
    index = write_dump(self.root)
    tokens = [
        dict(id=2, text=None),
        dict(id=105, text='<|turn>'),
        dict(id=40654, text='Reply'),
    ]
    outcome = dict(
        result(),
        tokenCountBefore=0,
        tokenCount=len(tokens) + 3,
        inputTokens=tokens,
        renderedInput='<|turn>user\nReply',
    )
    infer_generation(index, self.root / 'raw', outcome, JOB, turn=1)
    self.assertEqual(index['inference']['status'], 'ok', index['inference'])
    self.assertTrue(index['inference']['checks']['input_count_reconciles'])
    self.assertEqual(index['admitted_input_ids'], [2, 105, 40654])
    self.assertEqual(
        [(t['id'], t['text'], t['step']) for t in index['input_tokens']],
        [(2, None, 0), (105, '<|turn>', 1), (40654, 'Reply', 2)],
    )
    self.assertEqual(index['serialized_input'], '<|turn>user\nReply')
    prefill = next(f for f in index['forwards'] if f['phase'] == 'prefill')
    with tempfile.TemporaryDirectory() as other:
      plain = write_dump(other)
      infer_generation(plain, Path(other) / 'raw', result(), JOB, turn=1)
      self.assertNotEqual(
          prefill['input_identity'],
          next(f for f in plain['forwards'] if f['phase'] == 'prefill')[
              'input_identity'
          ],
          'admitted IDs sharpen the Prefill identity',
      )
      self.assertIsNone(plain['inference']['checks']['input_count_reconciles'])
    with tempfile.TemporaryDirectory() as other:
      off = write_dump(other)
      infer_generation(
          off,
          Path(other) / 'raw',
          dict(outcome, tokenCount=len(tokens) + 2),
          JOB,
          turn=1,
      )
      self.assertEqual(off['inference']['status'], 'mismatch')
      self.assertIn('input_count_reconciles', off['inference']['reason'])

  def test_argmax_mismatch_fails_closed(self):
    index = self.infer(released=(7, 3))
    self.assertEqual(index['inference']['status'], 'mismatch')
    self.assertIn('argmax_matches_released', index['inference']['reason'])
    self.assertEqual(index['token_records'], [])
    self.assertEqual(
        len(index['forwards']), 4, 'forwards and resources are kept'
    )
    self.assertFalse(
        any(
            f['comparison_eligible']
            for f in index['forwards']
            if f['phase'] == 'decode'
        )
    )
    self.assertTrue(
        next(f for f in index['forwards'] if f['phase'] == 'prefill')[
            'comparison_eligible'
        ]
    )

  def test_decode_count_and_position_checks(self):
    short = self.infer(decode_steps=(3, 4), argmaxes=(7, 2), released=(7, 2, 5))
    self.assertEqual(short['inference']['status'], 'mismatch')
    self.assertFalse(short['inference']['checks']['decode_count'])
    with tempfile.TemporaryDirectory() as other:
      index = write_dump(other, positions=(2, 3, 5))
      infer_generation(index, Path(other) / 'raw', result(), JOB, turn=1)
      self.assertEqual(index['inference']['status'], 'mismatch')
      self.assertFalse(index['inference']['checks']['input_pos_step'])
      self.assertFalse(index['inference']['checks']['input_pos_contiguous'])

  def test_non_greedy_or_speculative_is_unsupported(self):
    for outcome in (
        result(sampler={**GREEDY, 'temperature': 0.7}),
        result(speculative=True),
    ):
      with tempfile.TemporaryDirectory() as other:
        index = write_dump(other)
        infer_generation(index, Path(other) / 'raw', outcome, JOB, turn=1)
        self.assertEqual(index['inference']['status'], 'unsupported')
        self.assertEqual(index['token_records'], [])
        self.assertEqual(len(index['forwards']), 4)

  def test_multi_candidate_release_is_unsupported(self):
    index = write_dump(self.root)
    (self.root / 'raw/generated_tokens.jsonl').write_text(
        json.dumps({'token_ids': [[7], [8]], 'texts': []}) + '\n'
    )
    self.assertIsNone(released_tokens(self.root / 'raw/generated_tokens.jsonl'))
    infer_generation(index, self.root / 'raw', result(), JOB, turn=1)
    self.assertEqual(index['inference']['status'], 'unsupported')

  def test_verification_rejects_rewritten_bindings(self):
    index = self.infer()
    tampered = deepcopy(index)
    tampered['token_records'][0]['token_ids'] = [3]
    with self.assertRaisesRegex(ValueError, 'argmax'):
      verify_inferred_index(tampered, self.root)
    escaped = deepcopy(index)
    escaped['resources'][0]['path'] = '../escape.safetensors'
    with self.assertRaisesRegex(ValueError, 'escapes'):
      verify_inferred_index(escaped, self.root)


class DumpKvTests(unittest.TestCase):
  # KV snapshots after Prefill and at generation end, covering derived
  # contexts and pinned GPU normalization.

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)

  def infer(
      self,
      root=None,
      *,
      gpu=False,
      model=None,
      build=None,
      outcome=None,
      **dump,
  ):
    root = Path(root or self.root)
    index = write_dump(root, prefill_rows=2, kv=True, gpu=gpu, **dump)
    index['backend_requested'] = 'GPU' if gpu else 'CPU'
    infer_generation(
        index,
        root / 'raw',
        outcome or result(gpu=gpu),
        JOB,
        turn=1,
        model=kv_model() if model is None else model,
        build=build,
    )
    return index

  def test_cpu_dump_yields_two_snapshots_with_dumped_counts(self):
    index = self.infer()
    self.assertEqual(index['inference']['status'], 'ok', index['inference'])
    kv = index['inference']['kv']
    self.assertEqual(
        (kv['status'], kv['moments'], kv['skipped']),
        ('ok', ['prefill_post', 'terminal'], []),
        kv,
    )
    self.assertTrue(
        all(
            kv['checks'][name]
            for name in (
                'kv_prefill_count_matches_mask',
                'kv_terminal_count_matches_decodes',
            )
        )
    )
    self.assertIsNone(
        kv['checks']['kv_runner_count_reconciles'], 'no Runner count reported'
    )
    snapshots = index['kv_snapshots']
    self.assertEqual(
        [
            (
                s['moment'],
                s['phase'],
                s['step'],
                s['processed_token_count'],
                s['state'],
            )
            for s in snapshots
        ],
        [
            ('prefill_post', 'prefill', 3, 2, 'available'),
            ('terminal', 'decode', 5, 5, 'available'),
        ],
    )
    self.assertEqual(
        snapshots[1]['forward_id'],
        index['forwards'][-1]['forward_id'],
        'terminal binds the last Decode forward',
    )
    self.assertTrue(
        all(
            s['basis'] == BASIS and s['logical_token_ids'] is None
            for s in snapshots
        )
    )
    self.assertEqual(
        sorted(
            (row['layer'], t['kind'])
            for row in snapshots[0]['layers']
            for t in row['tensors']
        ),
        [(0, 'key'), (0, 'value'), (1, 'key')],
    )
    resources = [r for r in index['resources'] if r['scope'] == 'kv']
    self.assertEqual(len(resources), 6)
    for resource in resources:
      self.assertEqual(
          (
              resource['state'],
              resource['logical_start'],
              resource['logical_end'],
              resource['valid_length'],
              resource['capacity'],
          ),
          (
              'available',
              0,
              resource['processed_token_count'],
              resource['processed_token_count'],
              CACHE,
          ),
          resource,
      )
      self.assertEqual(
          resource['storage_view'][resource['layout'].index('sequence')][
              'stop'
          ],
          resource['processed_token_count'],
      )
      self.assertEqual(
          resource['dequantization'],
          {'scale': 0.01 * (resource['layer'] + 1), 'zero_point': 0},
      )
      self.assertEqual(resource['model_section_sha256'], SECTION)
      self.assertNotIn('storage', resource)
    self.assertEqual(
        len([
            f
            for f in index['forwards']
            for item in f['outputs']
            if 'kv_cache' in item['key']
        ]),
        0,
        'KV shards are snapshot resources, not forward boundaries',
    )
    self.assertTrue(verify_inferred_index(index, self.root))

  def test_contexts_pair_by_derived_identity(self):
    first = self.infer()
    with tempfile.TemporaryDirectory() as other:
      same = self.infer(other)
      self.assertEqual(
          [s['logical_context_identity'] for s in same['kv_snapshots']],
          [s['logical_context_identity'] for s in first['kv_snapshots']],
      )
    with tempfile.TemporaryDirectory() as other:
      diverged = self.infer(other, argmaxes=(7, 5, 9), released=(7, 5))
      self.assertEqual(
          diverged['kv_snapshots'][0]['logical_context_identity'],
          first['kv_snapshots'][0]['logical_context_identity'],
          'the same admitted input pairs After Prefill',
      )
      self.assertNotEqual(
          diverged['kv_snapshots'][1]['logical_context_identity'],
          first['kv_snapshots'][1]['logical_context_identity'],
          'different released tokens never pair at generation end',
      )

  def test_counts_fail_closed(self):
    with tempfile.TemporaryDirectory() as other:
      index = self.infer(other, mask_columns=1)
      self.assertEqual(
          index['inference']['status'],
          'ok',
          'Token Diff evidence is unaffected',
      )
      self.assertEqual(index['inference']['kv']['status'], 'mismatch')
      self.assertIn(
          'kv_prefill_count_matches_mask', index['inference']['kv']['reason']
      )
      self.assertEqual(
          (
              index['kv_snapshots'],
              [r for r in index['resources'] if r['scope'] == 'kv'],
          ),
          ([], []),
      )
    with tempfile.TemporaryDirectory() as other:
      index = self.infer(other, outcome=dict(result(), tokenCount=9))
      self.assertFalse(
          index['inference']['kv']['checks']['kv_runner_count_reconciles']
      )
      self.assertEqual(index['kv_snapshots'], [])
    with tempfile.TemporaryDirectory() as other:
      index = self.infer(other, released=(7, 3))
      self.assertEqual(index['inference']['status'], 'mismatch')
      self.assertIn('no logical context', index['inference']['kv']['reason'])
      self.assertEqual(index['kv_snapshots'], [])

  def test_unknown_descriptor_is_skipped_not_guessed(self):
    model = kv_model()
    del model['kv']['decode', 'kv_cache_k_1']
    index = self.infer(model=model)
    self.assertEqual(index['inference']['kv']['status'], 'ok')
    self.assertEqual(
        index['inference']['kv']['skipped'],
        ['terminal: decode kv_cache_k_1 has no model cache descriptor'],
    )
    self.assertEqual(
        len([r for r in index['resources'] if r['scope'] == 'kv']), 5
    )
    with tempfile.TemporaryDirectory() as other:
      index = self.infer(other, model={'kv': {}, 'section_sha256': SECTION})
      self.assertEqual(index['inference']['kv']['status'], 'unavailable')
      self.assertEqual(index['kv_snapshots'], [])
    with tempfile.TemporaryDirectory() as other:
      plain = write_dump(other, prefill_rows=2, kv=True)
      infer_generation(
          plain,
          Path(other) / 'raw',
          result(),
          JOB,
          turn=1,
          model_reason='model KV cache descriptors unavailable: boom',
      )
      self.assertEqual(
          plain['inference']['kv']['reason'],
          'model KV cache descriptors unavailable: boom',
      )
      self.assertTrue(verify_inferred_index(plain, Path(other)))

  def test_gpu_bytes_are_normalized_with_the_pinned_conversion(self):
    index = self.infer(gpu=True, build=GPU_BUILD)
    kv = index['inference']['kv']
    self.assertEqual((kv['status'], kv['skipped']), ('ok', []), kv)
    resources = [r for r in index['resources'] if r['scope'] == 'kv']
    self.assertTrue(
        all(r['state'] == 'available' for r in resources),
        [r.get('storage_comparison_reason') for r in resources],
    )
    for resource in resources:
      self.assertTrue(resource['path'].endswith('_logical.safetensors'))
      storage = resource['storage']
      self.assertEqual(
          (
              storage['binding'],
              storage['conversion']['id'],
              storage['source']['origin'],
              storage['evidence']['litert_revision'],
          ),
          (
              DUMP_BINDING,
              'webgpu_kv_u8_to_logical_i8',
              'runtime_dump',
              REFERENCE['source_revision'],
          ),
      )
      self.assertEqual(
          resource['storage_source']['path'], 'raw/' + storage['source']['path']
      )
      self.assertEqual(
          resource['storage_validation']['method'], 'all_values_exact'
      )
      logical = load_tensor(self.root, dict(resource, format='safetensors'))
      dump = load_tensor(
          self.root, dict(resource['storage_source'], format='safetensors')
      )
      np.testing.assert_array_equal(
          logical,
          normalize_webgpu_kv(dump, resource['kind'], resource['shape']),
      )
      self.assertFalse(
          np.array_equal(logical, dump),
          'the dump bytes are not the logical tensor',
      )
    self.assertTrue(verify_inferred_index(index, self.root))
    tampered = [r for r in resources if r['kind'] == 'value'][0]
    source = self.root / tampered['storage_source']['path']
    data = bytearray(source.read_bytes())
    data[-1] ^= 0x7F
    source.write_bytes(bytes(data))
    with self.assertRaises((NativeStorageError, ValueError)):
      verify_inferred_index(index, self.root)

  def test_retained_gpu_shards_revalidate_after_their_capture_moves(self):
    index = self.infer(gpu=True, build=GPU_BUILD)
    self.assertTrue(verify_inferred_index(index, self.root))
    with tempfile.TemporaryDirectory() as other:
      moved = Path(other) / 'later'
      shutil.copytree(self.root, moved)
      shutil.rmtree(self.root / 'raw')
      self.assertTrue(
          verify_inferred_index(deepcopy(index), moved),
          'a stale validation cache entry must not fail the copy',
      )

  def test_gpu_without_pinned_evidence_stays_unavailable(self):
    cases = {
        'no build': (None, result(gpu=True)),
        'other revision': (
            {'sourceLock': {'litert': 'ff' * 20}},
            result(gpu=True),
        ),
        'no library': (
            GPU_BUILD,
            dict(
                result(gpu=True),
                backendEvidence={'effectiveBackend': 'GPU', 'libraries': []},
            ),
        ),
    }
    for name, (build, outcome) in cases.items():
      with tempfile.TemporaryDirectory() as other:
        index = self.infer(other, gpu=True, build=build, outcome=outcome)
        resources = [r for r in index['resources'] if r['scope'] == 'kv']
        self.assertEqual(index['inference']['kv']['status'], 'ok', name)
        self.assertTrue(
            all(
                r['state'] == 'unavailable'
                and r['storage_comparison_status']
                == 'storage_contract_unavailable'
                and 'not normalized' in r['storage_comparison_reason']
                for r in resources
            ),
            name,
        )
        self.assertTrue(
            all(s['state'] == 'unavailable' for s in index['kv_snapshots']),
            name,
        )
        self.assertFalse(
            list((Path(other) / 'raw').glob('*_logical.safetensors')), name
        )
        self.assertTrue(verify_inferred_index(index, Path(other)), name)


class InferredPublicationTests(unittest.TestCase):
  """Two inferred sides pair by logical context and feed Token Diff metrics."""

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.base = Path(self.temp.name)

  def publish(self, spec, *, kv=False, gpu_target=False):
    temporary = self.base / 'capture'
    (temporary / 'tensors').mkdir(parents=True)
    data, current, results, copied, conversation = (
        empty_telemetry(),
        {},
        {},
        {},
        [],
    )
    for role, (argmaxes, released) in spec.items():
      root = self.base / role
      gpu = gpu_target and role == 'target'
      index = write_dump(
          root,
          decode_steps=tuple(range(3, 3 + len(argmaxes))),
          argmaxes=argmaxes,
          released=released,
          seed=1 if role == 'ref' else 2,
          prefill_rows=2 if kv else 1,
          kv=kv,
          gpu=gpu,
      )
      index['backend_requested'] = 'GPU' if gpu else 'CPU'
      infer_generation(
          index,
          root / 'raw',
          result(gpu=gpu),
          JOB,
          turn=1,
          model=kv_model() if kv else None,
          build=GPU_BUILD if gpu else None,
      )
      self.assertEqual(index['inference']['status'], 'ok', index['inference'])
      backend = 'GPU' if gpu else 'CPU'
      results[role] = dict(
          model_sha256=MODEL,
          backend_requested=backend,
          backend_effective=backend,
      )
      current[role] = append_telemetry(
          data, index, root, temporary, 'job', role, 1, results[role], copied
      )
      conversation.append(dict(turn=1, run=role, tokens=index['tokens']))
    runtimes = {'ref': 'LiteRT-LM', 'target': 'LiteRT-LM'}
    pair_forward_evidence(data, current, runtimes, results, 1)
    session = dict(
        conversation=conversation,
        batches=[],
        turns=[dict(n=1)],
        runs=[
            dict(
                id=role,
                backend=results[role]['backend_requested'],
                provenance=dict(results[role]),
            )
            for role in spec
        ],
    )
    link_token_batches(session, data, 1, runtimes)
    prepare_saved_resources(
        data['resources'],
        session['runs'],
        root=temporary,
        conversation=conversation,
    )

    class Store:
      _telemetry = data
      root = temporary

      def telemetry(self, turn=None):
        return data

      def explicit_pair_entries(self):
        return []

      def load_resource(self, identity):
        return load_tensor(
            temporary, next(r for r in data['resources'] if r['id'] == identity)
        )

    Store.session = session
    return data, Store()

  def test_pairs_until_divergence_and_reports_inferred_basis(self):
    data, store = self.publish(
        {'ref': ((7, 2, 4, 9), (7, 2, 4)), 'target': ((7, 5, 4, 9), (7, 5, 4))}
    )
    self.assertTrue(all(f.get('basis') == BASIS for f in data['forwards']))
    paired = sorted(
        (f['run'], f['phase'], f['step'])
        for f in data['forwards']
        if f.get('sample')
    )
    self.assertEqual(
        paired,
        [
            ('ref', 'decode', 3),
            ('ref', 'decode', 4),
            ('ref', 'prefill', 3),
            ('target', 'decode', 3),
            ('target', 'decode', 4),
            ('target', 'prefill', 3),
        ],
    )
    self.assertTrue(
        all(
            t.get('source_forward_id')
            for row in store.session['conversation']
            for t in row['tokens']
        )
    )
    pairs = analyze_tokens(
        store, {'turn': 1, 'pairs': [{'ref': k, 'target': k} for k in range(3)]}
    )['pairs']
    self.assertEqual([p['basis'] for p in pairs], [BASIS] * 3)
    for pair in pairs[:2]:
      for key in (
          'kl',
          'js',
          'relative_l2',
          'cosine_distance',
          'norm_ratio',
          'max_abs_error',
      ):
        self.assertIsNotNone(
            pair['metrics'][key]['value'],
            (pair['ref_step'], key, pair['metrics'][key]),
        )
      self.assertTrue(pair['distribution']['compatible'])
      self.assertEqual(pair['distribution']['ref']['size'], VOCAB)
    self.assertEqual(
        (
            pairs[1]['distribution']['ref']['selected_id'],
            pairs[1]['distribution']['target']['selected_id'],
        ),
        (2, 5),
    )
    self.assertGreaterEqual(pairs[1]['metrics']['kl']['value'], 0)
    self.assertIsNone(pairs[2]['metrics']['kl']['value'])
    self.assertIn('diverged', pairs[2]['metrics']['kl']['reason'])
    # After divergence each side keeps its own candidates and confidence;
    # nothing is subtracted.
    diverged = pairs[2]['distribution']
    self.assertEqual(
        (diverged['compatible'], diverged['context']), (False, 'different')
    )
    self.assertTrue(
        diverged['rows']
        and all(row['delta'] is None for row in diverged['rows'])
    )
    self.assertTrue(
        all(row['ref'] and row['target'] for row in diverged['rows'])
    )
    self.assertEqual(pairs[0]['distribution']['context'], 'same')
    self.assertTrue(
        all(
            row['delta'] is not None for row in pairs[0]['distribution']['rows']
        )
    )
    for side in ('ref', 'target'):
      summary = diverged[side]
      self.assertGreater(summary['selected_probability'], 0.5)
      self.assertGreater(summary['margin'], 0)
      self.assertLessEqual(summary['margin'], summary['selected_probability'])
    self.assertIn('diverged', pairs[2]['metrics']['relative_l2']['reason'])

  def test_kv_snapshots_pair_after_prefill_and_compare_cpu_with_normalized_gpu(
      self,
  ):
    data, store = self.publish(
        {'ref': ((7, 2, 4, 9), (7, 2, 4)), 'target': ((7, 5, 4, 9), (7, 5, 4))},
        kv=True,
        gpu_target=True,
    )
    snapshots = data['kv_snapshots']
    self.assertEqual(
        sorted(
            (s['run'], s['moment'], s['processed_token_count'], s.get('basis'))
            for s in snapshots
        ),
        [
            ('ref', 'prefill_post', 2, BASIS),
            ('ref', 'terminal', 6, BASIS),
            ('target', 'prefill_post', 2, BASIS),
            ('target', 'terminal', 6, BASIS),
        ],
    )
    paired = {(s['run'], s['moment']): s.get('sample') for s in snapshots}
    self.assertTrue(
        paired['ref', 'prefill_post']
        and paired['ref', 'prefill_post'] == paired['target', 'prefill_post']
    )
    self.assertIsNone(
        paired['ref', 'terminal'],
        'diverged generations never pair at generation end',
    )
    gpu = [
        r
        for r in data['resources']
        if r['scope'] == 'kv' and r['run'] == 'target'
    ]
    self.assertTrue(
        all(
            r['state'] == 'available' and r.get('storage_validation')
            for r in gpu
        ),
        [r.get('storage_comparison_reason') for r in gpu],
    )
    contexts = KvReader(store).contexts(1)
    after = next(c for c in contexts if c['moment'] == 'prefill_post')
    self.assertEqual(
        sorted(
            (row['layer'], row['kind'], row['status'])
            for row in after['layers']
        ),
        [(0, 'key', 'ok'), (0, 'value', 'ok'), (1, 'key', 'ok')],
        after['layers'],
    )
    self.assertIn('2 processed tokens', after['label'])
    end = [c for c in contexts if c['moment'] == 'terminal']
    self.assertEqual(
        len(end), 2, 'each side keeps its own generation-end observation'
    )
    self.assertTrue(
        all(row['status'] == 'unavailable' for c in end for row in c['layers'])
    )


def _encode_length_delimited(field_number: int, payload: bytes) -> bytes:
  """Encodes a small (<128 B) protobuf length-delimited field (wire type 2)."""
  tag = (field_number << 3) | 2
  return bytes([tag, len(payload)]) + payload


def _tiny_tokenizer_bytes() -> bytes:
  """Returns serialized SentencePiece ModelProto bytes without disk training."""
  buffer = io.BytesIO()
  corpus = [
      'the ocean is a vast body of water',
      'the sky appears blue because of scattering',
      'hello world',
      'a mysterious and powerful ocean',
  ] * 8
  try:
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(corpus),
        model_writer=buffer,
        vocab_size=40,
        model_type='unigram',
        minloglevel=2,
    )
    return buffer.getvalue()
  except TypeError:
    # Fallback for sentencepiece builds without in-memory model_writer support:
    # Construct a minimal SentencePiece ModelProto (field 1: repeated
    # SentencePiece{piece=1, score=2, type=3}, field 2:
    # TrainerSpec{model_type=3, vocab_size=4}, field 3: NormalizerSpec{name=1}).
    pieces = [
        ('<unk>', 0.0, 2),
        ('<s>', 0.0, 3),
        ('</s>', 0.0, 3),
        ('\u2581the', -1.0, 1),
        ('\u2581ocean', -1.5, 1),
        ('\u2581sky', -2.0, 1),
        ('\u2581world', -2.5, 1),
    ]
    out = bytearray()
    for piece, score, ptype in pieces:
      entry = (
          _encode_length_delimited(1, piece.encode('utf-8'))
          + b'\x15'
          + struct.pack('<f', score)
          + b'\x18'
          + bytes([ptype])
      )
      out.extend(_encode_length_delimited(1, entry))
    trainer_spec = b'\x18\x01\x20' + bytes([len(pieces)])
    normalizer_spec = _encode_length_delimited(1, b'identity')
    out.extend(_encode_length_delimited(2, trainer_spec))
    out.extend(_encode_length_delimited(3, normalizer_spec))
    return bytes(out)


class TokenizerLabelTests(unittest.TestCase):
  # Distribution rows label any vocabulary ID from the saved tokenizer.

  def setUp(self):
    super().setUp()
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)

  def tiny_tokenizer(self):
    return _tiny_tokenizer_bytes()

  def test_labels_prefer_generated_text_then_saved_tokenizer(self):
    blob = self.tiny_tokenizer()
    digest = hashlib.sha256(blob).hexdigest()
    (self.root / 'tokenizers').mkdir()
    (self.root / f'tokenizers/{digest}.model').write_bytes(blob)
    processor = spm.SentencePieceProcessor(model_proto=blob)
    ocean = processor.PieceToId('▁ocean')
    self.assertGreater(ocean, 0)

    class Store:
      root = self.root
      _telemetry = empty_telemetry()
      session = {
          'runs': [{
              'id': 'ref',
              'tokenizer': {
                  'kind': 'sentencepiece',
                  'path': f'tokenizers/{digest}.model',
                  'sha256': digest,
              },
          }],
          'conversation': [{
              'turn': 1,
              'run': 'ref',
              'tokens': [{'id': 3, 'text': 'released text', 'step': 0}],
          }],
      }

    analysis = token_analysis.TokenAnalysis(Store())
    self.assertEqual(
        analysis.label(1, 3), 'released text', 'the runtime-released text wins'
    )
    self.assertEqual(analysis.label(1, ocean), ' ocean')
    self.assertEqual(
        analysis.label(1, processor.PieceToId('<unk>')),
        '<unk>',
        'control pieces fall back to their name',
    )
    self.assertIsNone(analysis.label(1, 10**6))
    Store.session['runs'][0]['tokenizer']['sha256'] = '0' * 64
    self.assertIsNone(
        token_analysis.TokenAnalysis(Store()).label(1, ocean),
        'a digest mismatch disables the tokenizer',
    )


class CompleteTokenTests(unittest.TestCase):
  # Every sampled token is kept, including released text, the filtered stop
  # sample, and classified input.

  def setUp(self):
    super().setUp()
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)

  def test_sampled_stop_token_becomes_a_special_token(self):
    index = write_dump(self.root)
    infer_generation(
        index,
        self.root / 'raw',
        dict(
            result(),
            stopTokens=[{'ids': [9], 'text': '<eos>'}, {'ids': [1, 2]}],
        ),
        JOB,
        turn=1,
    )
    self.assertEqual(index['inference']['status'], 'ok')
    self.assertEqual(index['generation']['stop_token_ids'], [9])
    self.assertEqual(
        index['generation']['unreleased_sample'],
        {'id': 9, 'stop': True, 'recorded': True},
    )
    last = index['token_records'][-1]
    self.assertEqual(
        (
            last['forward_id'],
            last['output_index'],
            last['token_ids'],
            last['released'],
            last['stop'],
        ),
        (4, 2, [9], False, True),
    )
    self.assertEqual(
        index['tokens'][-1],
        {
            'id': 9,
            'text': None,
            'step': 2,
            'kind': 'special',
            'released': False,
            'stop': True,
        },
    )
    self.assertTrue(verify_inferred_index(index, self.root))

  def test_stop_token_admitted_as_template_is_a_chat_template_token(self):
    index = write_dump(self.root)
    tokens = [
        dict(id=2, text=None),
        dict(id=105, text='<|turn>'),
        dict(id=40654, text='hi'),
        dict(id=9, text='<turn|>'),
    ]
    outcome = dict(
        result(),
        stopTokens=[{'ids': [9], 'text': '<turn|>'}],
        inputTokens=tokens,
        renderedInput='<|turn>hi<turn|>',
    )
    infer_generation(index, self.root / 'raw', outcome, JOB, turn=1)
    self.assertEqual(index['inference']['status'], 'ok', index['inference'])
    self.assertEqual(
        [t['kind'] for t in index['input_tokens']],
        ['special', 'template', 'text', 'template'],
    )
    self.assertEqual(
        (
            index['tokens'][-1]['id'],
            index['tokens'][-1]['kind'],
            index['tokens'][-1]['stop'],
        ),
        (9, 'template', True),
    )

  def test_unknown_trailing_sample_stays_unrecorded(self):
    for outcome, reason in (
        (result(), 'stop tokens not reported by the Runner'),
        (
            dict(result(), stopTokens=[{'ids': [1]}]),
            'not a single-ID stop token',
        ),
    ):
      with tempfile.TemporaryDirectory() as other:
        index = write_dump(other)
        infer_generation(index, Path(other) / 'raw', outcome, JOB, turn=1)
        self.assertEqual(len(index['token_records']), 2)
        self.assertEqual(len(index['tokens']), 2)
        self.assertEqual(
            index['generation']['unreleased_sample']['recorded'], False
        )
        self.assertEqual(
            index['generation']['unreleased_sample']['reason'], reason
        )

  def test_input_tokens_are_classified_when_their_texts_tile_the_rendered_input(
      self,
  ):
    tokens = [
        dict(id=2, text=None),
        dict(id=105, text='<|turn>'),
        dict(id=2364, text='user'),
        dict(id=107, text='\n'),
        dict(id=40654, text='Reply'),
        dict(id=106, text='<turn|>'),
        dict(id=107, text='\n'),
        dict(id=105, text='<|turn>'),
        dict(id=4368, text='model'),
        dict(id=107, text='\n'),
    ]
    classify_input_tokens(
        tokens, '<|turn>user\nReply<turn|>\n<|turn>model\n', 'Reply'
    )
    self.assertEqual(
        [t.get('kind') for t in tokens],
        [
            'special',
            'template',
            'template',
            'template',
            'text',
            'template',
            'template',
            'template',
            'template',
            'template',
        ],
    )
    untiled = [dict(id=1, text='<|turn>'), dict(id=2, text='oops')]
    classify_input_tokens(untiled, '<|turn>user\nReply', 'Reply')
    self.assertEqual(
        [t.get('kind') for t in untiled],
        [None, None],
        'texts that do not tile the input get no kinds',
    )

  def test_control_tokens_get_their_piece_text_at_publication(self):
    model_bytes = _tiny_tokenizer_bytes()
    path = self.root / 'tokenizer.model'
    path.write_bytes(model_bytes)
    processor = spm.SentencePieceProcessor(model_proto=model_bytes)
    word = next(
        i
        for i in range(processor.GetPieceSize())
        if processor.IdToPiece(i).startswith('\u2581')
        and len(processor.IdToPiece(i)) > 2
        and not processor.IsControl(i)
    )
    tokens = [
        dict(id=processor.PieceToId('<s>'), text=None),
        dict(id=processor.PieceToId('\u2581ocean'), text=' ocean'),
        dict(id=word, text=None),
    ]
    capture_importer.label_missing_texts(tokens, path)
    self.assertEqual(
        [t['text'] for t in tokens],
        ['<s>', ' ocean', processor.IdToPiece(word).replace('\u2581', ' ')],
    )
    self.assertEqual(
        [t.get('text_basis') for t in tokens],
        ['saved tokenizer', None, 'saved tokenizer'],
    )
    capture_importer.label_missing_texts(
        [dict(id=5, text=None)], self.root / 'missing.model'
    )


if __name__ == '__main__':
  unittest.main()
