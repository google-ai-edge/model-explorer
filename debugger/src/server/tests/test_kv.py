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

"""Tests of analysis/kv.py over small hand-computed captures and a real one."""

import json
import math
import pathlib

from model_debugger_server import capture_index, errors
from model_debugger_server.analysis import kv
import numpy as np
import pytest
import safetensors
from safetensors import numpy as safetensors_numpy

REAL_TURN = pathlib.Path('/tmp/model_debugger_test_turn1')
K_LAYOUT = ['batch', 'kv_head', 'sequence', 'head_dim']
V_LAYOUT = ['batch', 'kv_head', 'head_dim', 'sequence']
SIGNATURES = {
    'prefill_pre': ('prefill_8', 'pre'),
    'prefill_post': ('prefill_8', 'post'),
    'terminal': ('decode', 'post'),
}

# Every Ref vector is [1, 2, 2, 0] (norm 3). Target differs from Ref only at
# layer 0, KV head 1, position 3, where it is [1, 2, 2, 4] (norm 5):
# cosine 9 / 15 = 0.6, delta norm 4, relative L2 4 / 3, RMSE 2, max abs 4.
HEADS, CAPACITY, CHANNELS = 2, 8, 4
BASE = np.array([1, 2, 2, 0], dtype=np.int8)


def _logical(processed: int, padding: int) -> np.ndarray:
  """[kv_head, sequence, head_dim]; rows past `processed` are padding."""
  values = np.full((HEADS, CAPACITY, CHANNELS), padding, dtype=np.int8)
  values[:, :processed] = BASE
  return values


def _side_info(sha256: str) -> dict:
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
    directory: pathlib.Path, moment: str, step: int, layers: dict
) -> tuple[list[dict], dict]:
  """Writes K and V of each layer in the runner's layouts.

  Args:
    directory: the side's turn directory.
    moment: the snapshot's moment.
    step: the snapshot's step label.
    layers: layer -> logical [kv_head, sequence, head_dim] int values.

  Returns:
    (the index's layer entries, {tensor key: file}).
  """
  signature, edge = SIGNATURES[moment]
  entries, files = [], {}
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
      files[key] = file
      entry[kind] = {
          'file': file,
          'dtype': 'I8',
          'shape': list(raw.shape),
          'layout': layout,
      }
    entries.append(entry)
  return entries, files


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
    snapshots, forwards = [], []
    for moment, step, processed in (
        ('prefill_pre', 6, 0),
        ('prefill_post', 6, 5),
        ('terminal', 7, terminal),
    ):
      padding = 0 if side == 'ref' else 100
      layer_0 = _logical(processed, padding)
      if side == 'target' and processed > 3:
        layer_0[1, 3] = [1, 2, 2, 4]
      layers, files = _write_kv(
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
      signature = SIGNATURES[moment][0]
      forward = next(
          (
              row
              for row in forwards
              if (row['signature'], row['step']) == (signature, step)
          ),
          None,
      )
      if forward is None:
        forward = {
            'id': len(forwards),
            'signature': signature,
            'phase': 'decode' if signature == 'decode' else 'prefill',
            'step': step,
            'tensors': {},
        }
        forwards.append(forward)
      forward['tensors'].update(files)
    turn['sides'][side] = {
        'dir': f'{side}/turn-1',
        'forwards': forwards,
        'kv': snapshots,
        'input': [],
        'output': [],
        'files': [],
    }
  index = {'format_version': 1, 'sides': sides, 'turns': [turn]}
  (root / 'index.json').write_text(json.dumps(index))
  return capture_index.CaptureIndex.load(root)


@pytest.fixture(name='index')
def _index(tmp_path) -> capture_index.CaptureIndex:
  return _write_capture(tmp_path)


def _context_id(index, moment: str, turn: int = 1) -> str:
  analysis = kv.kv_analysis(index, {'mode': 'metadata', 'turn': str(turn)})
  return next(c['id'] for c in analysis['contexts'] if c['moment'] == moment)


def _query(index, moment='prefill_post', **values) -> dict:
  query = {'turn': '1', 'context': _context_id(index, moment)}
  query.update({key: str(value) for key, value in values.items()})
  return query


def _row(response: dict, layer: int, kind: str = 'key') -> dict:
  return next(
      row
      for row in response['rows']
      if row['layer'] == layer and row['kind'] == kind
  )


def test_metadata_lists_each_moment_with_its_rows(index):
  """Metadata lists each moment with its rows, structure and snapshots."""
  analysis = kv.kv_analysis(index, {'mode': 'metadata', 'turn': '1'})

  contexts = {c['moment']: c for c in analysis['contexts']}
  assert list(contexts) == ['prefill_pre', 'prefill_post', 'terminal']
  post = contexts['prefill_post']
  assert post['id'].startswith('snapshot-') and len(post['id']) == 33
  assert post['turn'] == 1 and post['phase'] == 'prefill'
  assert post['step'] == 6 and post['source'] == 'snapshot'
  assert post['runtime'] == 'LiteRT-LM'
  assert post['processed_token_count'] == 5
  assert post['label'] == 'Turn 1 · After Prefill · 5 processed tokens'
  assert [(row['layer'], row['kind']) for row in post['layers']] == [
      (0, 'key'),
      (0, 'value'),
      (1, 'key'),
      (1, 'value'),
  ]
  row = post['layers'][1]
  assert row['status'] == 'ok'
  assert row['head_count'] == 2
  assert row['position_start'] == 0 and row['position_count'] == 5
  assert row['shape'] == [1, 2, 5, 4]
  assert row['canonical_layout'] == K_LAYOUT
  assert row['cells'] == [] and row['token_metrics'] == []
  assert 'no quantization scales' in row['comparison_basis']
  assert row['token_structure']['target'] == {
      'status': 'ok',
      'shape': [2, 4],
      'dtype': 'I8',
      'head_count': 2,
      'channel_count': 4,
      'batch_count': 1,
      'position_start': 0,
      'position_count': 5,
  }
  snapshot = post['snapshots']['ref'][0]
  assert snapshot['signature'] == 'prefill_8'
  assert snapshot['forward_id'] == 0 and snapshot['step'] == 6
  assert snapshot['edge'] == 'post' and snapshot['moment'] == 'prefill_post'
  assert contexts['terminal']['phase'] == 'decode'
  assert contexts['terminal']['snapshots']['target'][0]['forward_id'] == 1


def test_metadata_reports_an_empty_snapshot(index):
  """A snapshot of no processed positions is empty, not compared."""
  analysis = kv.kv_analysis(index, {'mode': 'metadata', 'turn': '1'})

  pre = analysis['contexts'][0]
  assert pre['moment'] == 'prefill_pre'
  row = pre['layers'][0]
  assert row['status'] == 'empty'
  assert row['position_count'] == 0 and row['head_count'] == 2
  assert row['token_structure']['ref']['status'] == 'empty'
  assert 'comparison_basis' not in row


def test_metadata_refuses_other_modes_and_unknown_turns(index):
  """Only mode=metadata is served, for a captured turn."""
  with pytest.raises(ValueError):
    kv.kv_analysis(index, {'turn': '1'})
  with pytest.raises(ValueError):
    kv.kv_analysis(index, {'mode': 'metadata', 'turn': 'x'})
  with pytest.raises(errors.NotFound):
    kv.kv_analysis(index, {'mode': 'metadata', 'turn': '2'})


def test_different_models_are_not_comparable(tmp_path):
  """Different model files are not comparable."""
  index = _write_capture(tmp_path, target_sha256='b' * 64)

  context = kv.kv_analysis(index, {'mode': 'metadata', 'turn': '1'})[
      'contexts'
  ][1]
  assert {row['status'] for row in context['layers']} == {'model_mismatch'}


def test_range_metrics_per_head(index):
  """One KV head's metrics match the hand-computed values."""
  response = kv.kv_range(
      index, _query(index, start=0, end=5, bins=5, head=1), None
  )

  assert response['context_id'] == _context_id(index, 'prefill_post')
  assert (response['start'], response['end']) == (0, 5)
  assert response['bins'] == [{'start': i, 'end': i + 1} for i in range(5)]
  assert response['kind'] == 'key' and response['head'] == 1
  row = _row(response, 0)
  assert row['status'] == 'ok'
  changed = row['bins'][3]['metrics']
  assert changed['relative_l2']['max'] == pytest.approx(4 / 3)
  assert changed['cosine_similarity']['min'] == pytest.approx(0.6)
  assert changed['cosine_distance']['max'] == pytest.approx(0.4)
  assert changed['max_abs']['mean'] == pytest.approx(4)
  assert changed['max_abs']['valid_count'] == 1
  same = row['bins'][0]['metrics']
  assert same['relative_l2']['max'] == 0
  assert same['cosine_similarity']['max'] == pytest.approx(1)
  assert _row(response, 1)['bins'][3]['metrics']['max_abs']['max'] == 0


def test_range_aggregates_heads(index):
  """Heads aggregate by min, max, mean and over a subset."""

  def metrics(head, heads=None):
    values = {'start': 3, 'end': 4, 'bins': 1, 'head': head}
    if heads is not None:
      values['heads'] = json.dumps(heads)
    response = kv.kv_range(index, _query(index, **values), None)
    return _row(response, 0)['bins'][0]['metrics']

  assert metrics('max')['relative_l2']['max'] == pytest.approx(4 / 3)
  assert metrics('max')['cosine_similarity']['max'] == pytest.approx(1)
  assert metrics('min')['cosine_similarity']['min'] == pytest.approx(0.6)
  assert metrics('min')['relative_l2']['min'] == 0
  assert metrics('mean')['relative_l2']['mean'] == pytest.approx(2 / 3)
  assert metrics('max', [0])['relative_l2']['max'] == 0
  assert metrics('max', [1, 5])['relative_l2']['max'] == pytest.approx(4 / 3)


def test_range_bin_statistics(index):
  """A bin keeps finite min, max, mean, their positions and a count."""
  response = kv.kv_range(
      index, _query(index, start=0, end=5, bins=1, head='max'), None
  )

  stat = _row(response, 0)['bins'][0]['metrics']['relative_l2']
  assert stat == {
      'min': 0.0,
      'max': pytest.approx(4 / 3),
      'mean': pytest.approx(4 / 15),
      'min_position': 0,
      'max_position': 3,
      'valid_count': 5,
  }


def test_range_value_layout_matches_key(index):
  """The transposed value layout reads the same logical values."""
  key = kv.kv_range(index, _query(index, start=0, end=5, bins=5), None)
  value = kv.kv_range(
      index, _query(index, start=0, end=5, bins=5, kind='value'), None
  )

  assert value['kind'] == 'value'
  assert _row(value, 0, 'value')['bins'] == _row(key, 0)['bins']


def test_range_reads_only_processed_positions(index):
  """Positions past the processed ones are never compared."""
  # Target's padding (100) past position 5 would differ from Ref's (0).
  response = kv.kv_range(index, _query(index, start=0, end=8, bins=8), None)

  row = _row(response, 1)
  assert row['status'] == 'ok'
  assert [b['metrics']['max_abs']['valid_count'] for b in row['bins']] == [
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
  ]
  assert row['bins'][6]['metrics']['max_abs']['max'] is None


def test_range_outside_processed_positions(index):
  """A range of no processed positions says so."""
  response = kv.kv_range(index, _query(index, start=5, end=8), None)

  assert _row(response, 0)['status'] == 'position_unavailable'
  assert len(response['bins']) == 3


def test_range_refuses_bad_requests(index):
  """Malformed range requests are refused."""
  for values in (
      {'start': 2, 'end': 2},
      {'start': 0, 'end': 5, 'bins': 0},
      {'start': 0, 'end': 5, 'bins': 1025},
      {'start': 0, 'end': 5, 'kind': 'both'},
      {'start': 0, 'end': 5, 'head': 'median'},
      {'start': 0, 'end': 5, 'head': 1, 'heads': '[0]'},
      {'start': 0, 'end': 5, 'heads': '[0, 0]'},
      {'start': 0, 'end': 5, 'formula': 'max_abs > 1'},
  ):
    with pytest.raises(ValueError):
      kv.kv_range(index, _query(index, **values), None)
  with pytest.raises(errors.NotFound):
    kv.kv_range(
        index,
        {'turn': '1', 'context': 'snapshot-x', 'start': '0', 'end': '1'},
        None,
    )


def test_range_head_outside_tensor(index):
  """A KV head outside the tensor is unavailable."""
  response = kv.kv_range(index, _query(index, start=0, end=5, head=2), None)

  assert _row(response, 0)['status'] == 'head_unavailable'


def test_processed_counts_must_match_unless_aligned(tmp_path):
  """Different processed counts compare only through an alignment."""
  index = _write_capture(tmp_path, target_terminal=6)
  query = _query(index, 'terminal', start=0, end=7, bins=7)

  plain = kv.kv_range(index, query, None)
  pairs = [[i, i] for i in range(6)] + [[6, None]]
  aligned = kv.kv_range(
      index, query, {'alignment': {'start': 0, 'pairs': pairs}}
  )

  assert _row(plain, 0)['status'] == 'logical_range_mismatch'
  row = _row(aligned, 0)
  assert row['status'] == 'ok'
  assert row['bins'][3]['metrics']['relative_l2']['max'] == pytest.approx(4 / 3)
  assert row['bins'][6]['metrics']['relative_l2']['valid_count'] == 0


def test_alignment_maps_columns_to_side_positions(index):
  """An alignment maps each display column to a position per side."""
  # Column 10 compares Ref position 0 with Target position 1 (equal), and
  # column 13 Ref position 2 with Target position 3 (changed).
  alignment = {'start': 10, 'pairs': [[0, 1], [None, 2], [1, None], [2, 3]]}

  response = kv.kv_range(
      index,
      _query(index, start=10, end=14, bins=4, head=1),
      {'alignment': alignment},
  )
  head = kv.kv_head(
      index,
      {
          'turn': '1',
          'context_id': _context_id(index, 'prefill_post'),
          'layer': '0',
          'kind': 'key',
          'position': '13',
          'head': '1',
      },
      {'alignment': alignment},
  )

  bins = _row(response, 0)['bins']
  assert bins[0]['metrics']['relative_l2']['max'] == 0
  assert bins[3]['metrics']['relative_l2']['max'] == pytest.approx(4 / 3)
  assert bins[3]['metrics']['relative_l2']['max_position'] == 13
  assert [b['metrics']['max_abs']['valid_count'] for b in bins] == [1, 0, 0, 1]
  assert head['selection']['positions'] == {'ref': 2, 'target': 3}
  assert [c['target'] for c in head['channels']] == [1, 2, 2, 4]


def test_alignment_is_validated(index):
  """Malformed alignments are refused."""
  query = _query(index, start=0, end=2)
  for alignment in (
      {'start': -1, 'pairs': []},
      {'start': 0, 'pairs': [[None, None]]},
      {'start': 0, 'pairs': [[1, 1], [1, 2]]},
      {'start': 0, 'pairs': [[0]]},
  ):
    with pytest.raises(ValueError):
      kv.kv_range(index, query, {'alignment': alignment})


def test_selection_whole_token_metrics(index):
  """A one-position selection carries whole-token metrics."""
  response = kv.kv_selection(index, _query(index, start=3, end=4), None)

  assert response['context_id'] == _context_id(index, 'prefill_post')
  assert (response['start'], response['end']) == (3, 4)
  assert len(response['rows']) == 4
  row = _row(response, 0)
  # Both heads together: Ref norm √18, Target norm √34, dot 18.
  token = row['token_metrics'][0]
  assert token['position'] == 3 and token['status'] == 'ok'
  assert token['metrics']['cosine_similarity'] == pytest.approx(
      math.sqrt(18 / 34)
  )
  assert token['metrics']['relative_l2'] == pytest.approx(4 / math.sqrt(18))
  assert token['metrics']['rmse'] == pytest.approx(math.sqrt(2))
  assert token['metrics']['max_abs'] == 4
  assert row['metrics']['max_abs'] == {
      'value': 4.0,
      'position': 3,
      'valid_count': 1,
      'status': 'ok',
  }


def test_selection_keeps_the_worst_value(index):
  """A range selection keeps each metric's worst value and position."""
  response = kv.kv_selection(
      index, _query(index, start=0, end=5, layer=0), None
  )

  assert [(row['layer'], row['kind']) for row in response['rows']] == [
      (0, 'key'),
      (0, 'value'),
  ]
  metrics = _row(response, 0)['metrics']
  assert 'token_metrics' not in _row(response, 0)
  assert metrics['cosine_similarity']['value'] == pytest.approx(
      math.sqrt(18 / 34)
  )
  assert metrics['cosine_similarity']['position'] == 3
  assert metrics['relative_l2']['position'] == 3
  assert metrics['rmse']['valid_count'] == 5


def test_selection_of_an_empty_snapshot(index):
  """An empty snapshot's selection reports why nothing was compared."""
  response = kv.kv_selection(
      index, _query(index, 'prefill_pre', start=0, end=1), None
  )

  row = _row(response, 0)
  assert row['status'] == 'empty'
  assert row['metrics']['rmse']['status'] == 'empty'
  assert row['metrics']['rmse']['value'] is None


def test_head_channels_and_metrics(index):
  """kv-head echoes the selection and compares the channels."""
  query = {
      'turn': '1',
      'context_id': _context_id(index, 'terminal'),
      'layer': '0',
      'kind': 'value',
      'position': '3',
      'head': '1',
  }

  head = kv.kv_head(index, query, None)

  assert head['selection'] == {
      'turn': 1,
      'context_id': query['context_id'],
      'layer': 0,
      'kind': 'value',
      'position': 3,
      'head': 1,
      'batch': 0,
  }
  assert head['status'] == 'ok'
  assert head['batch_count'] == 1 and head['channel_count'] == 4
  assert head['shape'] == [4]
  assert head['metrics'] == {
      'cosine_similarity': pytest.approx(0.6),
      'cosine_distance': pytest.approx(0.4),
      'relative_l2': pytest.approx(4 / 3),
      'mean_abs': 1.0,
      'rmse': 2.0,
      'max_abs': 4.0,
  }
  assert set(head['metric_status'].values()) == {'ok'}
  assert head['channels'][3] == {
      'channel': 3,
      'ref': 0,
      'target': 4,
      'delta': 4.0,
      'abs_delta': 4.0,
      'status': 'ok',
  }
  assert head['largest_channel'] == 3
  source = head['sources']['ref']
  assert source['values_mode'] == 'stored'
  assert source['source_dtype'] == source['comparison_dtype'] == 'I8'
  assert source['source_shape'] == [1, 2, 4, 8]
  assert source['comparison_shape'] == [4]
  assert source['layout'] == V_LAYOUT and source['axis_order'] == [0, 1, 3, 2]
  assert source['dequantization'] is None
  assert head['calculation_dtype'] == 'float64'
  assert head['delta_definition'] == 'Target - Reference'
  assert 'no quantization scales' in head['comparison_basis']


def test_head_outside_processed_positions(index):
  """kv-head reports positions that were not processed."""
  query = {
      'turn': '1',
      'context_id': _context_id(index, 'prefill_post'),
      'layer': '0',
      'kind': 'key',
      'position': '6',
      'head': '0',
  }

  head = kv.kv_head(index, query, None)

  assert head['status'] == 'position_unavailable'
  assert head['channels'] == []
  assert head['metric_status']['rmse'] == 'position_unavailable'
  with pytest.raises(ValueError):
    kv.kv_head(index, {**query, 'position': '0', 'head': '2'}, None)


# The real turn: Gemma, CPU, I8 KV of layers 0..14, prefill 16 positions,
# terminal 24.


def _real_capture(root: pathlib.Path, perturb: str | None = None):
  """Indexes the real turn for both sides; Target may change one file."""
  shapes = {}
  for path in REAL_TURN.glob('*kv_cache*.safetensors'):
    with safetensors.safe_open(path, framework='numpy') as tensor_file:
      (key,) = tensor_file.keys()
      shapes[path.name] = tensor_file.get_slice(key).get_shape()
  for side in ('ref', 'target'):
    directory = root / side / 'turn-1'
    directory.mkdir(parents=True)
    for name in shapes:
      if side == 'target' and name == perturb:
        with safetensors.safe_open(
            REAL_TURN / name, framework='numpy'
        ) as tensor_file:
          (key,) = tensor_file.keys()
          value = tensor_file.get_tensor(key)
          metadata = tensor_file.metadata()
        # Position 10, channel 5 of the only KV head moves by one.
        value[0, 0, 10, 5] += 1 if value[0, 0, 10, 5] < 127 else -1
        safetensors_numpy.save_file({key: value}, directory / name, metadata)
      else:
        (directory / name).symlink_to(REAL_TURN / name)

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
      'forwards': [],
      'kv': [
          snapshot('prefill_pre', 'prefill_128', 'pre', 17, 0),
          snapshot('prefill_post', 'prefill_128', 'post', 17, 16),
          snapshot('terminal', 'decode', 'post', 24, 24),
      ],
  }
  index = {
      'format_version': 1,
      'sides': {'ref': _side_info('c' * 64), 'target': _side_info('c' * 64)},
      'turns': [{
          'n': 1,
          'prompt': 'What is 2 + 2?',
          'sides': {
              side: {**side_entry, 'dir': f'{side}/turn-1'}
              for side in ('ref', 'target')
          },
      }],
  }
  (root / 'index.json').write_text(json.dumps(index))
  return capture_index.CaptureIndex.load(root)


needs_real_turn = pytest.mark.skipif(
    not REAL_TURN.is_dir(), reason='the real captured turn is absent'
)


@needs_real_turn
def test_real_turn_compared_with_itself(tmp_path):
  """The real turn compared with itself differs nowhere."""
  index = _real_capture(tmp_path)

  contexts = kv.kv_analysis(index, {'mode': 'metadata', 'turn': '1'})[
      'contexts'
  ]
  moments = {c['moment']: c for c in contexts}
  terminal = moments['terminal']
  assert terminal['processed_token_count'] == 24
  assert len(terminal['layers']) == 30
  assert {row['status'] for row in terminal['layers']} == {'ok'}
  assert {row['head_count'] for row in terminal['layers']} == {1}
  channels = {
      row['token_structure']['ref']['channel_count']
      for row in terminal['layers']
  }
  assert channels == {256, 512}
  assert {row['status'] for row in moments['prefill_pre']['layers']} == {
      'empty'
  }
  for kind in ('key', 'value'):
    response = kv.kv_range(
        index,
        _query(index, 'terminal', start=0, end=24, bins=24, kind=kind),
        None,
    )
    for row in response['rows']:
      assert row['status'] == 'ok', row
      for cell in row['bins']:
        assert cell['metrics']['max_abs']['max'] == 0
        assert cell['metrics']['relative_l2']['max'] == 0
        assert cell['metrics']['cosine_similarity']['min'] == pytest.approx(1)
  selection = kv.kv_selection(
      index, _query(index, 'prefill_post', start=0, end=16), None
  )
  assert {row['metrics']['max_abs']['value'] for row in selection['rows']} == {
      0
  }


@needs_real_turn
def test_real_turn_with_one_changed_value(tmp_path):
  """One changed stored value shows at its layer and position only."""
  index = _real_capture(
      tmp_path, perturb='decode_post_kv_cache_k_3_step_24.safetensors'
  )

  response = kv.kv_range(
      index, _query(index, 'terminal', start=0, end=24, bins=24), None
  )
  head = kv.kv_head(
      index,
      {
          'turn': '1',
          'context_id': _context_id(index, 'terminal'),
          'layer': '3',
          'kind': 'key',
          'position': '10',
          'head': '0',
      },
      None,
  )
  prefill = kv.kv_range(
      index, _query(index, 'prefill_post', start=0, end=16, bins=1), None
  )

  for row in response['rows']:
    for i, cell in enumerate(row['bins']):
      changed = row['layer'] == 3 and i == 10
      assert cell['metrics']['max_abs']['max'] == (1 if changed else 0)
  assert head['largest_channel'] == 5
  assert head['metrics']['max_abs'] == 1
  assert abs(head['channels'][5]['delta']) == 1
  assert {
      row['bins'][0]['metrics']['max_abs']['max'] for row in prefill['rows']
  } == {0}


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
        'forwards': [],
        'kv': [{
            'moment': 'terminal',
            'step': 9,
            'processed': processed,
            'layers': layers,
        }],
    }
  index = {
      'format_version': 1,
      'sides': {'ref': _side_info('d' * 64), 'target': _side_info('d' * 64)},
      'turns': [{'n': 1, 'prompt': 'p', 'sides': side_entries}],
  }
  (root / 'index.json').write_text(json.dumps(index))
  return capture_index.CaptureIndex.load(root)


def test_full_context_range_is_fast(tmp_path):
  """A full 4096-position context answers for both key and value."""
  index = _random_capture(tmp_path, processed=4096)
  context = _context_id(index, 'terminal')

  for kind in ('key', 'value'):
    response = kv.kv_range(
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
        None,
    )
    assert len(response['rows']) == 15
    assert all(row['status'] == 'ok' for row in response['rows'])


def test_a_tensor_outside_the_layout_is_listed_not_compared(index):
  """KV whose shape does not fit its layout is listed with its own shape."""
  stored = index.turn(1)['sides']['target']['kv'][1]['layers'][0]['key']
  stored['layout'] = None

  context = kv.kv_analysis(index, {'mode': 'metadata', 'turn': '1'})[
      'contexts'
  ][1]

  row = context['layers'][0]
  assert row['status'] == 'layout_unavailable'
  assert row['token_structure']['target']['status'] == 'layout_unavailable'
  assert row['token_structure']['ref']['status'] == 'ok'
  assert row['position_count'] == 5


def test_responses_are_strict_json(index):
  """Every response encodes as JSON without NaN or numpy values."""
  alignment = {'alignment': {'start': 0, 'pairs': [[0, 0], [1, None]]}}
  head = {
      'turn': '1',
      'context_id': _context_id(index, 'prefill_post'),
      'layer': '0',
      'kind': 'key',
      'position': '0',
      'head': '0',
  }
  for response in (
      kv.kv_analysis(index, {'mode': 'metadata', 'turn': '1'}),
      kv.kv_range(index, _query(index, start=0, end=8, head='mean'), alignment),
      kv.kv_selection(index, _query(index, start=0, end=1), None),
      kv.kv_selection(
          index, _query(index, 'prefill_pre', start=0, end=2), None
      ),
      kv.kv_head(index, head, alignment),
  ):
    json.dumps(response, allow_nan=False)


def test_corrupt_safetensors_row_reports_unavailable_without_failing_other_rows(
    index,
):
  """A corrupt safetensors file marks only its row unavailable."""
  stored = index.turn(1)['sides']['target']['kv'][1]['layers'][0]['key']
  index.path('target', 1, stored['file']).write_bytes(b'corrupt')

  ranged = kv.kv_range(index, _query(index, start=0, end=5), None)
  assert _row(ranged, 0, 'key')['status'] == 'unavailable'
  assert _row(ranged, 1, 'key')['status'] == 'ok'

  selected = kv.kv_selection(index, _query(index, start=0, end=5), None)
  assert _row(selected, 0, 'key')['status'] == 'unavailable'
  assert _row(selected, 1, 'key')['status'] == 'ok'

  head = kv.kv_head(
      index,
      {
          'turn': '1',
          'context_id': _context_id(index, 'prefill_post'),
          'layer': '0',
          'kind': 'key',
          'position': '0',
          'head': '0',
      },
      None,
  )
  assert head['status'] == 'unavailable'
  assert len(head['channels']) == CHANNELS
