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

"""Tests of the KV analysis over small hand-computed captures and a real one."""

import json
import pathlib
import time
from typing import Any

from absl.testing import absltest
from model_debugger.server.analysis import kv_analysis
from model_debugger.server.analysis import metrics
from model_debugger.server.capture import capture_index
from model_debugger.server.testing import capture_fixtures
from model_debugger.server.testing import real_capture
from model_debugger.server.testing import test_main
from model_debugger.server.util import errors
import numpy as np
from safetensors import numpy as safetensors_numpy
from safetensors import safe_open

K_LAYOUT = ['batch', 'kv_head', 'sequence', 'head_dim']
V_LAYOUT = ['batch', 'kv_head', 'head_dim', 'sequence']
SIGNATURES = {
    'prefill_pre': ('prefill_8', 'pre'),
    'prefill_post': ('prefill_8', 'post'),
    'terminal': ('decode', 'post'),
}

# Every Ref vector is [1, 2, 2, 0] (norm 3). Target differs from Ref only at
# layer 0, KV head 1, position 3, where it is [1, 2, 2, 4] (norm 5):
# cosine 9 / 15 = 0.6, delta norm 4, relative L2 400 / 3, RMSE 2, max abs 4.
HEADS, CAPACITY, CHANNELS = 2, 8, 4
BASE = np.array([1, 2, 2, 0], dtype=np.int8)


def _logical(processed: int, padding: int) -> np.ndarray:
  """[kv_head, sequence, head_dim]; rows past `processed` are padding."""
  values = np.full((HEADS, CAPACITY, CHANNELS), padding, dtype=np.int8)
  values[:, :processed] = BASE
  return values


def _side_info(sha256: str) -> dict[str, Any]:
  return {
      'model': {'sha256': sha256, 'name': 'model.litertlm', 'size': 1},
      'runtime': {'name': 'LiteRT-LM', 'version': '0' * 40},
      'host': {},
      'device_id': 'local',
      'device_name': 'This computer',
      'load': {'device': 'CPU', 'context_length': CAPACITY},
      'chat': {'messages': [], 'options': {'max_completion_tokens': 2}},
      'start_token': 2,
      'stop_tokens': [[1]],
  }


def _write_kv(
    directory: pathlib.Path, moment: str, step: int, layers: dict[str, Any]
) -> list[dict[str, Any]]:
  """Writes K and V of each layer in the runner's layouts.

  Args:
      directory: The side's turn directory.
      moment: The snapshot's moment.
      step: The snapshot's step label.
      layers: Layer -> logical [kv_head, sequence, head_dim] int values.

  Returns:
      The index's layer entries.
  """
  signature, edge = SIGNATURES[moment]
  entries = []
  for layer, logical in layers.items():
    entry = {'layer': layer}
    for kind, letter, raw, layout in (
        ('key', 'k', logical[None], K_LAYOUT),
        ('value', 'v', logical.transpose(0, 2, 1)[None], V_LAYOUT),
    ):
      key = f'{edge}_kv_cache_{letter}_{layer}'
      file = f'{signature}_{key}_step_{step}.safetensors'
      safetensors_numpy.save_file(
          {key: np.ascontiguousarray(raw)},
          directory / file,
          metadata={'signature': signature, 'step': str(step)},
      )
      entry[kind] = {
          'file': file,
          'dtype': 'I8',
          'shape': list(raw.shape),
          'layout': layout,
      }
    entries.append(entry)
  return entries


def _write_capture(
    root: pathlib.Path,
    target_sha256: str = 'a' * 64,
    target_terminal: int = 7,
) -> capture_index.CaptureIndex:
  """A one-turn capture: prefill of 5 positions, terminal of 7 (Ref)."""
  sides = {'ref': _side_info('a' * 64), 'target': _side_info(target_sha256)}
  turn = {'n': 1, 'prompt': 'Hi', 'sides': {}}
  for side in ('ref', 'target'):
    directory = root / side / 'turn-1'
    directory.mkdir(parents=True)
    terminal = 7 if side == 'ref' else target_terminal
    snapshots = []
    for moment, step, processed in (
        ('prefill_pre', 6, 0),
        ('prefill_post', 6, 5),
        ('terminal', 7, terminal),
    ):
      padding = 0 if side == 'ref' else 100
      layer_0 = _logical(processed, padding)
      if side == 'target' and processed > 3:
        layer_0[1, 3] = [1, 2, 2, 4]
      layers = _write_kv(
          directory,
          moment,
          step,
          {0: layer_0, 1: _logical(processed, padding)},
      )
      snapshots.append({
          'moment': moment,
          'step': step,
          'processed': processed,
          'layers': layers,
      })
    turn['sides'][side] = {
        'dir': f'{side}/turn-1',
        'kv': snapshots,
    }
  return capture_index.CaptureIndex(root, sides, [turn])


# The cross-runtime capture's scale of Target's int8 KV.
SCALE = 0.5


def _cross_runtime_capture(
    root: pathlib.Path,
    scales: bool = True,
    ref_heads: int = HEADS,
    ref_layers: tuple[int, ...] = (0, 1),
) -> capture_index.CaptureIndex:
  """One model in two runtimes: a terminal of 7 positions on each side.

  Ref is PyTorch's: float32 K and V in [batch, kv_head, sequence,
  head_dim], the sequence the 7 positions, of `ref_heads` KV heads and
  `ref_layers`. Target is LiteRT-LM's: int8 K [b, h, s, d] and V
  [b, h, d, s] over the context (CAPACITY), whose model gives each cache
  the scale SCALE when `scales`. Ref's values are Target's dequantized,
  BASE * SCALE, but at layer 0, KV head 1, position 3, where Target's are
  [1, 2, 2, 4] (dequantized [0.5, 1, 1, 2]).

  Args:
    root: Root directory for the capture.
    scales: Whether to include KV quantization scales.
    ref_heads: Number of reference KV heads.
    ref_layers: Layers to include for the reference model.

  Returns:
    The indexed capture.
  """
  sides = {'ref': _side_info('p' * 64), 'target': _side_info('l' * 64)}
  sides['ref']['runtime'] = {'name': 'PyTorch', 'version': '0'}
  if scales:
    kv_quantization = {}
    for letter in 'kv':
      for layer in (0, 1):
        kv_quantization[f'kv_cache_{letter}_{layer}'] = {
            'scale': SCALE,
            'zero_point': 0,
        }
    sides['target']['model']['kv_quantization'] = kv_quantization
  processed = 7
  turn = {'n': 1, 'prompt': 'Hi', 'sides': {}}
  for side in ('ref', 'target'):
    directory = root / side / 'turn-1'
    directory.mkdir(parents=True)
    if side == 'target':
      layers = {0: _logical(processed, 100), 1: _logical(processed, 100)}
      layers[0][1, 3] = [1, 2, 2, 4]
      entries = _write_kv(directory, 'terminal', 7, layers)
    else:
      entries = []
      for layer in ref_layers:
        logical = np.tile(BASE * SCALE, (ref_heads, processed, 1))
        entry = {'layer': layer}
        for kind, letter in (('key', 'k'), ('value', 'v')):
          key = f'post_kv_cache_{letter}_{layer}'
          file = f'decode_{key}_step_7.safetensors'
          raw = logical[None].astype(np.float32)
          safetensors_numpy.save_file({key: raw}, directory / file)
          entry[kind] = {
              'file': file,
              'dtype': 'F32',
              'shape': list(raw.shape),
              'layout': K_LAYOUT,
          }
        entries.append(entry)
    turn['sides'][side] = {
        'dir': f'{side}/turn-1',
        'kv': [{
            'moment': 'terminal',
            'step': 7,
            'processed': processed,
            'layers': entries,
        }],
    }
  return capture_index.CaptureIndex(root, sides, [turn])


def _context_id(index, moment: str, turn: int = 1) -> str:
  analysis = kv_analysis.kv_analysis(index, turn)
  return next(c['id'] for c in analysis['contexts'] if c['moment'] == moment)


def _query(index, moment='prefill_post', **values) -> dict[str, Any]:
  query = {'turn': '1', 'context': _context_id(index, moment)}
  query.update({key: str(value) for key, value in values.items()})
  return query


def _range(index, query, payload=None):
  """kv_range of a query and body as kv-range's were (strings)."""
  alignment = (payload or {}).get('alignment')
  head = query.get('head', kv_analysis.DEFAULT_HEAD)
  heads = query.get('heads')
  layers = query.get('layers')
  return kv_analysis.kv_range(
      index,
      int(query['turn']),
      query['context'],
      int(query['start']),
      int(query['end']),
      int(query.get('bins', '128')),
      query.get('kind', 'key'),
      int(head) if head.isdigit() else head,
      None if heads is None else sorted(json.loads(heads)),
      None if alignment is None else kv_analysis.Alignment.of(alignment),
      None if layers is None else sorted(json.loads(layers)),
  )


def _heads(index, query, payload=None):
  """kv_head of a query and body as kv-head's were (strings)."""
  alignment = (payload or {}).get('alignment')
  batch = query.get('batch')
  return kv_analysis.kv_head(
      index,
      int(query['turn']),
      query['context_id'],
      int(query['layer']),
      query.get('kind'),
      int(query['position']),
      None if batch is None else int(batch),
      None if alignment is None else kv_analysis.Alignment.of(alignment),
  )


def _head(index, query, payload=None):
  """The KV head `query['head']` of _heads, with the answer's selection."""
  answer = _heads(index, query, payload)
  head = answer['heads'][int(query['head'])]
  return {**head, 'selection': answer['selection']}


def _row(
    response: dict[str, Any], layer: int, kind: str = 'key'
) -> dict[str, Any]:
  return next(
      row
      for row in response['rows']
      if row['layer'] == layer and row['kind'] == kind
  )


# The real turn: Gemma, CPU, I8 KV of layers 0..14, prefill 16 positions,
# terminal 24.


def _real_capture(root: pathlib.Path, perturb: str | None = None):
  """Indexes the real turn for both sides; Target may change one file."""
  real_turn = real_capture.turn('CPU').directory
  shapes = {}
  for path in real_turn.glob('*kv_cache*.safetensors'):
    with safe_open(path, framework='numpy') as tensor_file:
      (key,) = tensor_file.keys()
      shapes[path.name] = tensor_file.get_slice(key).get_shape()
  for side in ('ref', 'target'):
    directory = root / side / 'turn-1'
    directory.mkdir(parents=True)
    for name in shapes:
      if side == 'target' and name == perturb:
        with safe_open(real_turn / name, framework='numpy') as tensor_file:
          (key,) = tensor_file.keys()
          value = tensor_file.get_tensor(key)
          metadata = tensor_file.metadata()
        # Position 10, channel 5 of the only KV head moves by one.
        value[0, 0, 10, 5] += 1 if value[0, 0, 10, 5] < 127 else -1
        safetensors_numpy.save_file({key: value}, directory / name, metadata)
      else:
        (directory / name).symlink_to(real_turn / name)

  def snapshot(moment, signature, edge, step, processed):
    layers = []
    for layer in range(15):
      entry = {'layer': layer}
      for kind, letter, layout in (
          ('key', 'k', K_LAYOUT),
          ('value', 'v', V_LAYOUT),
      ):
        name = f'{signature}_{edge}_kv_cache_{letter}_{layer}_step_{step}'
        name += '.safetensors'
        entry[kind] = {
            'file': name,
            'dtype': 'I8',
            'shape': shapes[name],
            'layout': layout,
        }
      layers.append(entry)
    return {
        'moment': moment,
        'step': step,
        'processed': processed,
        'layers': layers,
    }

  side_entry = {
      'kv': [
          snapshot('prefill_pre', 'prefill_128', 'pre', 17, 0),
          snapshot('prefill_post', 'prefill_128', 'post', 17, 16),
          snapshot('terminal', 'decode', 'post', 24, 24),
      ],
  }
  return capture_index.CaptureIndex(
      root,
      {'ref': _side_info('c' * 64), 'target': _side_info('c' * 64)},
      [{
          'n': 1,
          'prompt': 'What is 2 + 2?',
          'sides': {
              side: {**side_entry, 'dir': f'{side}/turn-1'}
              for side in ('ref', 'target')
          },
      }],
  )


needs_real_turn = absltest.skipUnless(
    real_capture.AVAILABLE, real_capture.UNAVAILABLE
)


def _random_capture(root: pathlib.Path, processed: int):
  """Gemma-shaped random I8 KV (layers 4, 9, 14 with head_dim 512)."""
  generator = np.random.default_rng(0)
  side_entries = {}
  for side in ('ref', 'target'):
    directory = root / side / 'turn-1'
    directory.mkdir(parents=True)
    layers = []
    for layer in range(15):
      channels = 512 if layer in (4, 9, 14) else 256
      entry = {'layer': layer}
      for kind, letter, shape, layout in (
          ('key', 'k', [1, 1, 4096, channels], K_LAYOUT),
          ('value', 'v', [1, 1, channels, 4096], V_LAYOUT),
      ):
        name = f'decode_post_kv_cache_{letter}_{layer}_step_9.safetensors'
        value = generator.integers(-127, 128, size=shape, dtype=np.int8)
        safetensors_numpy.save_file({'post': value}, directory / name)
        entry[kind] = {
            'file': name,
            'dtype': 'I8',
            'shape': shape,
            'layout': layout,
        }
      layers.append(entry)
    side_entries[side] = {
        'dir': f'{side}/turn-1',
        'kv': [{
            'moment': 'terminal',
            'step': 9,
            'processed': processed,
            'layers': layers,
        }],
    }
  return capture_index.CaptureIndex(
      root,
      {'ref': _side_info('d' * 64), 'target': _side_info('d' * 64)},
      [{'n': 1, 'prompt': 'p', 'sides': side_entries}],
  )


class KvAnalysisTest(absltest.TestCase):
  """Tests of the KV analysis."""

  def setUp(self):
    """An empty directory, and the small capture in another."""
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)
    self.index = _write_capture(pathlib.Path(self.create_tempdir().full_path))

  def _pytorch_capture(self, window_start: int | None = None):
    """A PyTorch chat's capture, layer 0 a window from `window_start`."""
    chats = {
        side: capture_fixtures.TinyChat(seed=seed, pytorch=True)
        for side, seed in (('ref', 0), ('target', 1))
    }
    index = capture_index.CaptureIndex.create(
        self.tmp_path,
        {s: capture_fixtures.side_info(c) for s, c in chats.items()},
    )
    completions = {}
    for side, chat in chats.items():
      directory = index.turn_directory(side, 1)
      completions[side] = capture_fixtures.write_turn(
          directory, chat.send('Hi there', ['a', 'b'])
      )
      if window_start is not None:
        for path in directory.glob('*_kv_cache_*_0_step_*'):
          shape = (
              safe_open(path, 'numpy')
              .get_slice(next(iter(safe_open(path, 'numpy').keys())))
              .get_shape()
          )
          capture_fixtures.set_metadata(
              path,
              kv_positions=f'{window_start}-{window_start + shape[2] - 1}',
          )
    index.add_turn(1, 'Hi there', completions)
    return index

  def test_pytorch_bf16_bhsd_kv_is_compared(self):
    """A PyTorch capture's BF16 cache, in HF's layout, is compared."""
    index = self._pytorch_capture()

    contexts = kv_analysis.kv_analysis(index, 1)['contexts']

    self.assertEqual(
        [c['moment'] for c in contexts], ['prefill_post', 'terminal']
    )
    for context in contexts:
      self.assertEqual(context['runtime'], 'PyTorch')
      for row in context['layers']:
        self.assertEqual(row['status'], 'ok', row)
        self.assertEqual(
            row['position_count'], context['processed_token_count']
        )
    terminal = contexts[-1]
    result = kv_analysis.kv_range(
        index,
        1,
        terminal['id'],
        0,
        terminal['processed_token_count'],
        8,
        'key',
        'min',
        None,
        None,
    )
    self.assertTrue(result['rows'])

  def test_a_window_that_does_not_start_at_0_says_so(self):
    """A layer that keeps a window not from position 0 has a reason."""
    index = self._pytorch_capture(window_start=1)

    context = kv_analysis.kv_analysis(index, 1)['contexts'][-1]

    rows = {(r['layer'], r['kind']): r for r in context['layers']}
    self.assertEqual(rows[(0, 'key')]['status'], kv_analysis.SLIDING_WINDOW)
    self.assertIn('sliding window', rows[(0, 'key')]['reasons'][0]['text'])
    self.assertEqual(rows[(1, 'key')]['status'], 'ok')

  def test_metadata_lists_each_moment_with_its_rows(self):
    """Metadata lists each moment, its rows and each row's structure."""
    analysis = kv_analysis.kv_analysis(self.index, 1)

    contexts = {c['moment']: c for c in analysis['contexts']}
    self.assertEqual(
        list(contexts), ['prefill_pre', 'prefill_post', 'terminal']
    )
    post = contexts['prefill_post']
    self.assertTrue(
        post['id'].startswith('snapshot-') and len(post['id']) == 33
    )
    self.assertEqual(post['turn'], 1)
    self.assertEqual(post['phase'], 'prefill')
    self.assertEqual(post['step'], 6)
    self.assertEqual(post['source'], 'snapshot')
    self.assertEqual(post['runtime'], 'LiteRT-LM')
    self.assertEqual(post['processed_token_count'], 5)
    self.assertEqual(
        post['label'], 'Turn 1 · After Prefill · 5 processed tokens'
    )
    self.assertEqual(
        [(row['layer'], row['kind']) for row in post['layers']],
        [
            (0, 'key'),
            (0, 'value'),
            (1, 'key'),
            (1, 'value'),
        ],
    )
    row = post['layers'][1]
    self.assertEqual(row['status'], 'ok')
    self.assertEqual(row['head_count'], 2)
    self.assertEqual(row['position_start'], 0)
    self.assertEqual(row['position_count'], 5)
    self.assertEqual(row['shape'], [1, 2, 5, 4])
    self.assertEqual(row['canonical_layout'], K_LAYOUT)
    self.assertIn('no quantization scales', row['comparison_basis'])
    self.assertEqual(
        row['token_structure']['target'],
        {
            'status': 'ok',
            'shape': [2, 4],
            'dtype': 'I8',
            'head_count': 2,
            'channel_count': 4,
            'batch_count': 1,
            'position_start': 0,
            'position_count': 5,
            'reasons': [],
        },
    )
    self.assertEqual(row['reasons'], [])
    self.assertEqual(contexts['terminal']['phase'], 'decode')

  def test_metadata_reports_an_empty_snapshot(self):
    """A snapshot of no processed positions is empty, not compared."""
    analysis = kv_analysis.kv_analysis(self.index, 1)

    pre = analysis['contexts'][0]
    self.assertEqual(pre['moment'], 'prefill_pre')
    row = pre['layers'][0]
    self.assertEqual(row['status'], 'empty')
    self.assertEqual(row['position_count'], 0)
    self.assertEqual(row['head_count'], 2)
    self.assertEqual(row['token_structure']['ref']['status'], 'empty')
    self.assertNotIn('comparison_basis', row)

  def test_metadata_of_an_unknown_turn_is_not_found(self):
    """Only a captured turn has KV contexts."""
    with self.assertRaises(errors.NotFound):
      kv_analysis.kv_analysis(self.index, 2)

  def test_one_model_in_two_runtimes_is_compared(self):
    """Each side's own layout and dtype; int8 dequantized by its scale."""
    index = _cross_runtime_capture(self.tmp_path)

    (context,) = kv_analysis.kv_analysis(index, 1)['contexts']
    self.assertIsNone(context['runtime'])
    self.assertEqual({row['status'] for row in context['layers']}, {'ok'})
    row = context['layers'][0]
    self.assertEqual(row['shape'], [1, HEADS, 7, CHANNELS])
    self.assertIn('Ref: stored F32 values', row['comparison_basis'])
    self.assertIn(
        'Target: stored I8 values dequantized as 0.5 × (stored − 0)',
        row['comparison_basis'],
    )
    response = _range(
        index, _query(index, 'terminal', start=0, end=7, bins=7, head=1)
    )
    bins = _row(response, 0)['bins']
    self.assertAlmostEqual(bins[3]['metrics']['relative_l2']['max'], 400 / 3)
    self.assertAlmostEqual(bins[3]['metrics']['max_abs']['max'], 2.0)
    self.assertEqual(bins[2]['metrics']['max_abs']['max'], 0.0)
    self.assertEqual(
        _row(response, 1)['bins'][3]['metrics']['max_abs']['max'], 0.0
    )

    head = _head(
        index,
        {
            'turn': '1',
            'context_id': context['id'],
            'layer': '0',
            'kind': 'value',
            'position': '3',
            'head': '1',
        },
    )
    self.assertEqual(head['status'], 'ok')
    self.assertEqual(head['sources']['ref']['values_mode'], 'stored')
    self.assertIsNone(head['sources']['ref']['dequantization'])
    self.assertEqual(head['sources']['target']['values_mode'], 'comparison')
    self.assertEqual(
        head['sources']['target']['dequantization'],
        {
            'scale': SCALE,
            'zero_point': 0,
            'formula': kv_analysis.DEQUANTIZATION_FORMULA,
        },
    )
    self.assertEqual(
        [channel['target'] for channel in head['channels']],
        [0.5, 1.0, 1.0, 2.0],
    )
    self.assertEqual(
        [channel['delta'] for channel in head['channels']],
        [0.0, 0.0, 0.0, 2.0],
    )

  def test_int8_without_its_scale_is_not_compared_with_another_model(self):
    """Without the model's scale, int8 values mean nothing to floats."""
    index = _cross_runtime_capture(self.tmp_path, scales=False)

    (context,) = kv_analysis.kv_analysis(index, 1)['contexts']
    self.assertEqual(
        {row['status'] for row in context['layers']},
        {kv_analysis.KV_SCALE_UNKNOWN},
    )
    (reason,) = context['layers'][0]['reasons']
    self.assertEqual(reason['side'], 'target')
    self.assertIn('kv_cache_k_0', reason['text'])

  def test_models_of_different_kv_architectures_are_not_comparable(self):
    """KV heads that differ make every row of the context incomparable."""
    index = _cross_runtime_capture(self.tmp_path, ref_heads=1)

    (context,) = kv_analysis.kv_analysis(index, 1)['contexts']
    self.assertEqual(
        {row['status'] for row in context['layers']},
        {kv_analysis.MODEL_MISMATCH},
    )
    (reason,) = context['layers'][3]['reasons']
    self.assertEqual(reason['code'], kv_analysis.MODEL_MISMATCH)
    self.assertIn('layer 0 key', reason['text'])
    self.assertIn('1×1×4 on Ref and 1×2×4 on Target', reason['text'])

  def test_a_layer_one_runtime_keeps_no_cache_of_says_so(self):
    """A layer only Target stores is that row's matter alone."""
    index = _cross_runtime_capture(self.tmp_path, ref_layers=(0,))

    (context,) = kv_analysis.kv_analysis(index, 1)['contexts']
    rows = {(r['layer'], r['kind']): r for r in context['layers']}
    self.assertEqual(rows[(0, 'key')]['status'], 'ok')
    self.assertEqual(rows[(1, 'value')]['status'], 'unavailable')
    (reason,) = rows[(1, 'value')]['reasons']
    self.assertEqual(reason['code'], kv_analysis.TENSOR_MISSING)
    self.assertEqual(reason['side'], 'ref')
    self.assertIn('Only Target stores layer 1', reason['text'])

  def test_a_moment_one_runtime_did_not_capture_is_missing(self):
    """A snapshot one side lacks is not a layer it keeps no cache of."""
    index = _cross_runtime_capture(self.tmp_path)
    turn = index.turn(1)
    terminal = turn['sides']['target']['kv'][0]
    turn['sides']['target']['kv'].append(
        {**terminal, 'moment': 'prefill_pre', 'processed': 0}
    )

    contexts = kv_analysis.kv_analysis(index, 1)['contexts']
    pre = next(c for c in contexts if c['moment'] == 'prefill_pre')
    missing = metrics.reason(
        kv_analysis.TENSOR_MISSING,
        kv_analysis.REASONS[kv_analysis.TENSOR_MISSING],
    )
    for row in pre['layers']:
      self.assertEqual(row['reasons'], [missing])

  def test_the_same_model_ignores_the_scales_of_its_files(self):
    """One model file stores one representation: compared as stored."""
    sides = dict(self.index.sides)
    sides['target'] = {
        **sides['target'],
        'model': {
            **sides['target']['model'],
            'kv_quantization': {
                'kv_cache_k_0': {'scale': 3.0, 'zero_point': 1}
            },
        },
    }
    index = capture_index.CaptureIndex(
        self.index.root, sides, [self.index.turn(1)]
    )
    before = kv_analysis.kv_analysis(self.index, 1)
    self.assertEqual(kv_analysis.kv_analysis(index, 1), before)
    query = _query(index, start=0, end=5, bins=5, head=1)
    self.assertEqual(_range(index, query), _range(self.index, query))

  def test_range_metrics_per_head(self):
    """One KV head's metrics match the hand-computed values."""
    response = _range(
        self.index, _query(self.index, start=0, end=5, bins=5, head=1)
    )

    self.assertEqual(
        response['context_id'], _context_id(self.index, 'prefill_post')
    )
    self.assertEqual((response['start'], response['end']), (0, 5))
    self.assertEqual(
        response['bins'], [{'start': i, 'end': i + 1} for i in range(5)]
    )
    self.assertEqual(response['kind'], 'key')
    self.assertEqual(response['head'], 1)
    row = _row(response, 0)
    self.assertEqual(row['status'], 'ok')
    changed = row['bins'][3]['metrics']
    np.testing.assert_allclose(
        changed['relative_l2']['max'], 400 / 3, rtol=1e-6
    )
    np.testing.assert_allclose(
        changed['cosine_similarity']['min'], 0.6, rtol=1e-6
    )
    np.testing.assert_allclose(
        changed['cosine_distance']['max'], 0.4, rtol=1e-6
    )
    np.testing.assert_allclose(changed['max_abs']['mean'], 4, rtol=1e-6)
    self.assertEqual(changed['max_abs']['valid_count'], 1)
    same = row['bins'][0]['metrics']
    self.assertEqual(same['relative_l2']['max'], 0)
    np.testing.assert_allclose(same['cosine_similarity']['max'], 1, rtol=1e-6)
    self.assertEqual(
        _row(response, 1)['bins'][3]['metrics']['max_abs']['max'], 0
    )

  def test_range_aggregates_heads(self):
    """Heads aggregate by min, max, mean and over a subset."""

    def metrics(head, heads=None):  # pylint: disable=redefined-outer-name
      values = {'start': 3, 'end': 4, 'bins': 1, 'head': head}
      if heads is not None:
        values['heads'] = json.dumps(heads)
      response = _range(self.index, _query(self.index, **values))
      return _row(response, 0)['bins'][0]['metrics']

    np.testing.assert_allclose(
        metrics('max')['relative_l2']['max'], 400 / 3, rtol=1e-6
    )
    np.testing.assert_allclose(
        metrics('max')['cosine_similarity']['max'], 1, rtol=1e-6
    )
    np.testing.assert_allclose(
        metrics('min')['cosine_similarity']['min'], 0.6, rtol=1e-6
    )
    self.assertEqual(metrics('min')['relative_l2']['min'], 0)
    np.testing.assert_allclose(
        metrics('mean')['relative_l2']['mean'], 200 / 3, rtol=1e-6
    )
    self.assertEqual(metrics('max', [0])['relative_l2']['max'], 0)
    np.testing.assert_allclose(
        metrics('max', [1, 5])['relative_l2']['max'], 400 / 3, rtol=1e-6
    )

  def test_range_aggregates_heads_by_the_minimum_by_default(self):
    """A request that names no head aggregation gets the minimum."""
    response = _range(self.index, _query(self.index, start=3, end=4, bins=1))

    self.assertEqual(response['head'], 'min')
    response_metrics = _row(response, 0)['bins'][0]['metrics']
    np.testing.assert_allclose(
        response_metrics['cosine_similarity']['min'], 0.6, rtol=1e-6
    )
    self.assertEqual(response_metrics['relative_l2']['min'], 0)

  def test_range_bin_statistics(self):
    """A bin keeps finite min, max, mean, their positions and a count."""
    response = _range(
        self.index, _query(self.index, start=0, end=5, bins=1, head='max')
    )

    stat = _row(response, 0)['bins'][0]['metrics']['relative_l2']
    np.testing.assert_allclose(stat.pop('max'), 400 / 3, rtol=1e-6)
    np.testing.assert_allclose(stat.pop('mean'), 80 / 3, rtol=1e-6)
    self.assertEqual(
        stat,
        {
            'min': 0.0,
            'min_position': 0,
            'max_position': 3,
            'valid_count': 5,
        },
    )

  def test_range_metadata_describes_what_it_compared(self):
    """Each side's input and the output, over the layers and heads named."""
    response = _range(
        self.index,
        _query(
            self.index,
            'terminal',
            start=2,
            end=9,
            bins=1,
            head='max',
            heads=json.dumps([1, 4]),
            layers=json.dumps([1]),
        ),
    )

    self.assertEqual(response['layers'], [1])
    self.assertEqual([row['layer'] for row in response['rows']], [1])
    metadata = response['metadata']
    # Columns 2..8 hold the processed positions 2..6 of 7; of the heads
    # [1, 4], the tensor has head 1.
    self.assertEqual(
        metadata['ref'],
        {
            'dimensions': [
                'Position',
                'Layer',
                'Head',
                'Head dimension',
            ],
            'shape': [5, 1, 1, 4],
            'dtype': 'I8',
            'layers': [{'layer': 1, 'shape': [5, 1, 4], 'dtype': 'I8'}],
        },
    )
    self.assertEqual(
        metadata['output'],
        {
            'dimensions': ['Position', 'Layer'],
            'shape': [7, 1],
            'dtype': 'float64',
            'layers': [{'layer': 1, 'shape': [7]}],
        },
    )

  def test_range_metadata_through_an_alignment(self):
    """Aligned columns count each side's own positions, each once."""
    alignment = {'start': 10, 'pairs': [[0, 1], [None, 2], [1, None]]}

    response = _range(
        self.index,
        _query(self.index, start=10, end=13, bins=1, head=0),
        {'alignment': alignment},
    )

    metadata = response['metadata']
    self.assertEqual(metadata['ref']['shape'], [2, 2, 1, 4])
    self.assertEqual(metadata['target']['shape'], [2, 2, 1, 4])
    self.assertEqual(metadata['output']['shape'], [3, 2])

  def test_range_metadata_of_layers_that_differ(self):
    """Sizes that differ across layers are listed; no shape is None."""
    stored = self.index.turn(1)['sides']['target']['kv'][1]['layers'][0]['key']
    stored['layout'] = None

    metadata = _range(self.index, _query(self.index, start=0, end=5, bins=1))[
        'metadata'
    ]

    self.assertEqual(metadata['target']['shape'], [None, 2, None, None])
    self.assertIsNone(metadata['target']['layers'][0]['shape'])
    self.assertEqual(metadata['output']['shape'], [None, 2])
    self.assertEqual(metadata['output']['dtype'], 'float64')
    self.assertEqual(kv_analysis._uniform([256, 512, 256]), [256, 512])
    self.assertIsNone(kv_analysis._uniform([]))

  def test_range_value_layout_matches_key(self):
    """The transposed value layout reads the same logical values."""
    key = _range(self.index, _query(self.index, start=0, end=5, bins=5))
    value = _range(
        self.index, _query(self.index, start=0, end=5, bins=5, kind='value')
    )

    self.assertEqual(value['kind'], 'value')
    self.assertEqual(_row(value, 0, 'value')['bins'], _row(key, 0)['bins'])

  def test_range_reads_only_processed_positions(self):
    """Positions past the processed ones are never compared."""
    # Target's padding (100) past position 5 would differ from Ref's (0).
    response = _range(self.index, _query(self.index, start=0, end=8, bins=8))

    row = _row(response, 1)
    self.assertEqual(row['status'], 'ok')
    self.assertEqual(
        [b['metrics']['max_abs']['valid_count'] for b in row['bins']],
        [
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
        ],
    )
    self.assertIsNone(row['bins'][6]['metrics']['max_abs']['max'])

  def test_range_outside_processed_positions(self):
    """A range of no processed positions says so."""
    response = _range(self.index, _query(self.index, start=5, end=8))

    self.assertEqual(_row(response, 0)['status'], 'position_unavailable')
    self.assertEqual(
        _row(response, 0)['reasons'],
        [
            metrics.reason(
                'position_unavailable',
                'This layer has no comparable positions in the range.',
            )
        ],
    )
    self.assertLen(response['bins'], 3)

  def test_range_of_an_unknown_context_is_not_found(self):
    """A context the turn does not have is not found."""
    with self.assertRaises(errors.NotFound):
      _range(
          self.index,
          {
              'turn': '1',
              'context': 'snapshot-x',
              'start': '0',
              'end': '1',
          },
      )

  def test_range_head_outside_tensor(self):
    """A KV head outside the tensor is unavailable."""
    response = _range(self.index, _query(self.index, start=0, end=5, head=2))

    self.assertEqual(_row(response, 0)['status'], 'head_unavailable')

  def test_processed_counts_must_match_unless_aligned(self):
    """Different processed counts compare only through an alignment."""
    index = _write_capture(self.tmp_path, target_terminal=6)
    query = _query(index, 'terminal', start=0, end=7, bins=7, head='max')

    plain = _range(index, query)
    pairs = [[i, i] for i in range(6)] + [[6, None]]
    aligned = _range(index, query, {'alignment': {'start': 0, 'pairs': pairs}})

    self.assertEqual(_row(plain, 0)['status'], 'logical_range_mismatch')
    self.assertEqual(
        [reason['code'] for reason in _row(plain, 0)['reasons']],
        ['logical_range_mismatch'],
    )
    row = _row(aligned, 0)
    self.assertEqual(row['status'], 'ok')
    self.assertEqual(row['reasons'], [])
    np.testing.assert_allclose(
        row['bins'][3]['metrics']['relative_l2']['max'], 400 / 3, rtol=1e-6
    )
    self.assertEqual(row['bins'][6]['metrics']['relative_l2']['valid_count'], 0)

  def test_alignment_maps_columns_to_side_positions(self):
    """An alignment maps each display column to a position per side."""
    # Column 10 compares Ref position 0 with Target position 1 (equal), and
    # column 13 Ref position 2 with Target position 3 (changed).
    alignment = {
        'start': 10,
        'pairs': [[0, 1], [None, 2], [1, None], [2, 3]],
    }

    response = _range(
        self.index,
        _query(self.index, start=10, end=14, bins=4, head=1),
        {'alignment': alignment},
    )
    head = _head(
        self.index,
        {
            'turn': '1',
            'context_id': _context_id(self.index, 'prefill_post'),
            'layer': '0',
            'kind': 'key',
            'position': '13',
            'head': '1',
        },
        {'alignment': alignment},
    )

    bins = _row(response, 0)['bins']
    self.assertEqual(bins[0]['metrics']['relative_l2']['max'], 0)
    np.testing.assert_allclose(
        bins[3]['metrics']['relative_l2']['max'], 400 / 3, rtol=1e-6
    )
    self.assertEqual(bins[3]['metrics']['relative_l2']['max_position'], 13)
    self.assertEqual(
        [b['metrics']['max_abs']['valid_count'] for b in bins], [1, 0, 0, 1]
    )
    self.assertEqual(head['selection']['positions'], {'ref': 2, 'target': 3})
    self.assertEqual([c['target'] for c in head['channels']], [1, 2, 2, 4])

  def test_alignment_positions_must_increase(self):
    """Columns read each side in order: the alignment says so."""
    with self.assertRaises(ValueError):
      kv_analysis.Alignment.of({'start': 0, 'pairs': [[1, 1], [1, 2]]})
    with self.assertRaises(ValueError):
      kv_analysis.Alignment.of(
          {'start': 0, 'pairs': [[0, 2], [None, None], [1, 1]]}
      )
    aligned = kv_analysis.Alignment.of(
        {'start': 4, 'pairs': [[0, None], [None, 0], [1, 1]]}
    )
    self.assertEqual(aligned.positions(5), {'ref': None, 'target': 0})
    self.assertEqual(aligned.positions(9), {'ref': None, 'target': None})

  def test_head_channels_and_metrics(self):
    """kv-head echoes the selection and compares the channels."""
    query = {
        'turn': '1',
        'context_id': _context_id(self.index, 'terminal'),
        'layer': '0',
        'kind': 'value',
        'position': '3',
        'head': '1',
    }

    head = _head(self.index, query)

    self.assertEqual(
        head['selection'],
        {
            'turn': 1,
            'context_id': query['context_id'],
            'layer': 0,
            'kind': 'value',
            'position': 3,
            'batch': 0,
        },
    )
    self.assertEqual(head['head'], 1)
    self.assertEqual(head['status'], 'ok')
    self.assertEqual(head['batch_count'], 1)
    self.assertEqual(head['channel_count'], 4)
    self.assertEqual(head['shape'], [4])
    head_metrics = head['metrics']
    np.testing.assert_allclose(
        head_metrics.pop('cosine_similarity'), 0.6, rtol=1e-6
    )
    np.testing.assert_allclose(
        head_metrics.pop('cosine_distance'), 0.4, rtol=1e-6
    )
    np.testing.assert_allclose(
        head_metrics.pop('relative_l2'), 400 / 3, rtol=1e-6
    )
    np.testing.assert_allclose(head_metrics.pop('norm_ratio'), 5 / 3, rtol=1e-6)
    self.assertEqual(
        head_metrics,
        {
            'mean_abs': 1.0,
            'rmse': 2.0,
            'max_abs': 4.0,
        },
    )
    self.assertEqual(set(head['metric_status'].values()), {'ok'})
    self.assertEqual(
        head['channels'][3],
        {
            'channel': 3,
            'ref': 0,
            'target': 4,
            'delta': 4.0,
            'abs_delta': 4.0,
            'status': 'ok',
        },
    )
    self.assertEqual(head['largest_channel'], 3)
    source = head['sources']['ref']
    self.assertEqual(source['values_mode'], 'stored')
    self.assertEqual(source['source_dtype'], 'I8')
    self.assertEqual(source['comparison_dtype'], 'I8')
    self.assertEqual(source['source_shape'], [1, 2, 4, 8])
    self.assertEqual(source['comparison_shape'], [4])
    self.assertEqual(source['layout'], V_LAYOUT)
    self.assertEqual(source['axis_order'], [0, 1, 3, 2])
    self.assertIsNone(source['dequantization'])
    self.assertEqual(head['calculation_dtype'], 'float64')
    self.assertEqual(head['delta_definition'], 'Target - Reference')
    self.assertIn('no quantization scales', head['comparison_basis'])
    # Ref [1, 2, 2, 0], Target [1, 2, 2, 4]: delta [0, 0, 0, 4].
    statistics = head['statistics']
    self.assertEqual(
        statistics['ref'],
        {
            'min': 0.0,
            'max': 2.0,
            'mean': 1.25,
            'min_position': 3,
            'max_position': 1,
            'valid_count': 4,
        },
    )
    self.assertEqual(
        (statistics['target']['max'], statistics['target']['max_position']),
        (4.0, 3),
    )
    self.assertEqual(statistics['delta']['mean'], 1.0)
    self.assertEqual(statistics['delta']['min_position'], 0)
    self.assertEqual(statistics['abs_delta']['max_position'], 3)

  def test_head_answers_every_head_at_once(self):
    """kv-head reads the column once and answers each KV head."""
    answer = _heads(
        self.index,
        {
            'turn': '1',
            'context_id': _context_id(self.index, 'terminal'),
            'layer': '0',
            'kind': 'key',
            'position': '3',
        },
    )

    self.assertEqual([head['head'] for head in answer['heads']], [0, 1])
    same, changed = answer['heads']
    self.assertEqual(same['status'], 'ok')
    self.assertEqual(same['metrics']['max_abs'], 0.0)
    self.assertEqual(same['largest_channel'], 0)
    self.assertEqual(same['statistics']['abs_delta']['max'], 0.0)
    self.assertEqual(changed['metrics']['max_abs'], 4.0)
    self.assertEqual(changed['largest_channel'], 3)

  def test_head_outside_processed_positions(self):
    """kv-head reports positions that were not processed."""
    query = {
        'turn': '1',
        'context_id': _context_id(self.index, 'prefill_post'),
        'layer': '0',
        'kind': 'key',
        'position': '6',
        'head': '0',
    }

    head = _head(self.index, query)

    self.assertEqual(head['status'], 'position_unavailable')
    self.assertEqual(head['channels'], [])
    self.assertEqual(head['metric_status']['rmse'], 'position_unavailable')
    self.assertEqual(
        [(r['code'], r['side']) for r in head['reasons']],
        [('position_unavailable', None)],
    )
    self.assertEqual(head['statistics']['ref']['valid_count'], 0)
    self.assertLen(_heads(self.index, query)['heads'], 2)
    with self.assertRaises(ValueError):
      _heads(self.index, {**query, 'position': '0', 'batch': '1'})

  @needs_real_turn
  def test_real_turn_compared_with_itself(self):
    """The real turn compared with itself differs nowhere."""
    index = _real_capture(self.tmp_path)

    contexts = kv_analysis.kv_analysis(index, 1)['contexts']
    moments = {c['moment']: c for c in contexts}
    terminal = moments['terminal']
    self.assertEqual(terminal['processed_token_count'], 24)
    self.assertLen(terminal['layers'], 30)
    self.assertEqual({row['status'] for row in terminal['layers']}, {'ok'})
    self.assertEqual({row['head_count'] for row in terminal['layers']}, {1})
    channels = {
        row['token_structure']['ref']['channel_count']
        for row in terminal['layers']
    }
    self.assertEqual(channels, {256, 512})
    self.assertEqual(
        {row['status'] for row in moments['prefill_pre']['layers']},
        {'empty'},
    )
    for kind in ('key', 'value'):
      response = _range(
          index,
          _query(index, 'terminal', start=0, end=24, bins=24, kind=kind),
      )
      for row in response['rows']:
        self.assertEqual(row['status'], 'ok', row)
        for cell in row['bins']:
          self.assertEqual(cell['metrics']['max_abs']['max'], 0)
          self.assertEqual(cell['metrics']['relative_l2']['max'], 0)
          np.testing.assert_allclose(
              cell['metrics']['cosine_similarity']['min'],
              1,
              rtol=1e-6,
          )

  @needs_real_turn
  def test_real_turn_with_one_changed_value(self):
    """One changed stored value shows at its layer and position only."""
    index = _real_capture(
        self.tmp_path,
        perturb='decode_post_kv_cache_k_3_step_24.safetensors',
    )

    response = _range(
        index, _query(index, 'terminal', start=0, end=24, bins=24)
    )
    head = _head(
        index,
        {
            'turn': '1',
            'context_id': _context_id(index, 'terminal'),
            'layer': '3',
            'kind': 'key',
            'position': '10',
            'head': '0',
        },
    )
    prefill = _range(
        index, _query(index, 'prefill_post', start=0, end=16, bins=1)
    )

    for row in response['rows']:
      for i, cell in enumerate(row['bins']):
        changed = row['layer'] == 3 and i == 10
        self.assertEqual(cell['metrics']['max_abs']['max'], 1 if changed else 0)
    self.assertEqual(head['largest_channel'], 5)
    self.assertEqual(head['metrics']['max_abs'], 1)
    self.assertEqual(abs(head['channels'][5]['delta']), 1)
    self.assertEqual(
        {
            row['bins'][0]['metrics']['max_abs']['max']
            for row in prefill['rows']
        },
        {0},
    )

  def test_full_context_range_is_fast(self):
    """A full 4096-position context answers well under a second."""
    index = _random_capture(self.tmp_path, processed=4096)
    context = _context_id(index, 'terminal')

    started = time.perf_counter()
    for kind in ('key', 'value'):
      response = _range(
          index,
          {
              'turn': '1',
              'context': context,
              'start': '0',
              'end': '4096',
              'bins': '1024',
              'kind': kind,
              'head': 'max',
          },
      )
      self.assertLen(response['rows'], 15)
      self.assertTrue(all(row['status'] == 'ok' for row in response['rows']))
    elapsed = time.perf_counter() - started

    self.assertLess(elapsed, 1.0)

  def test_a_tensor_outside_the_layout_is_listed_not_compared(self):
    """KV whose shape does not fit its layout is listed with its shape."""
    stored = self.index.turn(1)['sides']['target']['kv'][1]['layers'][0]['key']
    stored['layout'] = None

    context = kv_analysis.kv_analysis(self.index, 1)['contexts'][1]

    row = context['layers'][0]
    self.assertEqual(row['status'], 'layout_unavailable')
    self.assertEqual(
        row['token_structure']['target']['status'], 'layout_unavailable'
    )
    # The reason is the target side's.
    layout = metrics.reason(
        'layout_unavailable',
        kv_analysis.REASONS['layout_unavailable'],
        'target',
    )
    self.assertEqual(row['reasons'], [layout])
    self.assertEqual(row['token_structure']['target']['reasons'], [layout])
    self.assertEqual(row['token_structure']['ref']['status'], 'ok')
    self.assertEqual(row['position_count'], 5)

  def test_a_snapshot_is_judged_by_the_rule_its_tensors_are_read_by(self):
    """snapshot_reason: the first tensor's problem when none can be read."""
    sides = self.index.turn(1)['sides']
    pre, post = sides['target']['kv'][0], sides['target']['kv'][1]
    self.assertIsNone(kv_analysis.snapshot_reason(post, 'target'))
    self.assertEqual(
        kv_analysis.snapshot_reason(pre, 'target'),
        metrics.reason('empty', kv_analysis.REASONS['empty'], 'target'),
    )
    for layer in post['layers']:
      for kind in kv_analysis.KINDS:
        layer[kind]['layout'] = None
    self.assertEqual(
        kv_analysis.snapshot_reason(post, 'target'),
        metrics.reason(
            'layout_unavailable',
            kv_analysis.REASONS['layout_unavailable'],
            'target',
        ),
    )
    self.assertEqual(
        kv_analysis.snapshot_reason({**post, 'layers': []}, 'ref'),
        metrics.reason(
            kv_analysis.TENSOR_MISSING,
            'This snapshot stores no KV tensor.',
            'ref',
        ),
    )

  def test_responses_are_strict_json(self):
    """Every response encodes as JSON without NaN or numpy values."""
    alignment = {'alignment': {'start': 0, 'pairs': [[0, 0], [1, None]]}}
    head = {
        'turn': '1',
        'context_id': _context_id(self.index, 'prefill_post'),
        'layer': '0',
        'kind': 'key',
        'position': '0',
    }
    for response in (
        kv_analysis.kv_analysis(self.index, 1),
        _range(
            self.index,
            _query(self.index, start=0, end=8, head='mean'),
            alignment,
        ),
        _heads(self.index, head, alignment),
    ):
      json.dumps(response, allow_nan=False)


if __name__ == '__main__':
  test_main.main()
