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

"""Tests of token-analysis, final-logits and final-logits-vector.

Each test writes a small capture in the stored format (index.json plus one
safetensors file per tensor, named and keyed as the runner names them) and
checks the responses against values computed by hand.
"""

import hashlib
import json
import math
import pathlib

from model_debugger_server import capture_index, errors
from model_debugger_server.analysis import tokens
import numpy as np
import pytest
from safetensors import numpy as safetensors_numpy

MODEL = 'a' * 64
START = 2
STOP = 106
REAL_TURN = pathlib.Path('/tmp/model_debugger_test_turn1')


def _fixed(model: str = MODEL, **options) -> dict:
  """A side's fixed info; `options` replace the greedy chat options."""
  chat_options = {'max_completion_tokens': 32, 'top_k': 1, **options}
  chat_options = {
      key: value for key, value in chat_options.items() if value is not None
  }
  return {
      'model': {'sha256': model, 'name': 'model.litertlm', 'size': 1},
      'runtime': {'name': 'LiteRT-LM', 'version': '0' * 40},
      'host': {},
      'device_id': 'local',
      'device_name': 'This computer',
      'load': {'device': 'CPU', 'context_length': 64, 'runtime_options': {}},
      'chat': {'messages': [], 'options': chat_options},
      'start_token': START,
      'stop_tokens': [[STOP]],
  }


def _save(path: pathlib.Path, key: str, value: np.ndarray, step: int) -> None:
  safetensors_numpy.save_file(
      {key: value},
      str(path),
      metadata={'signature': 'decode', 'step': str(step)},
  )


def _write_side_turn(
    root: pathlib.Path,
    side: str,
    n: int,
    inputs: list[tuple[int, str | None, str]],
    logits: list[list[float]],
    first_position: int = 0,
    outputs: list[int] | None = None,
    texts: dict[int, str] | None = None,
    activations: list[list[float]] | None = None,
) -> dict:
  """Writes one side's decode files of a turn and returns its index entry.

  Args:
    root: The capture directory.
    side: 'ref' or 'target'.
    n: The turn.
    inputs: `(id, text, origin)` of every token the turn fed, from
      `first_position` on.
    logits: Each decode step's logits.
    first_position: Position of the first input token.
    outputs: The generated ids (default: each logits' argmax).
    texts: Released text of generated ids.
    activations: Each decode step's activations (default: the logits' first four
      values).

  Returns:
    The side's entry of the turn in index.json.
  """
  directory = root / side / f'turn-{n}'
  directory.mkdir(parents=True)
  outputs = outputs or [int(np.argmax(row)) for row in logits]
  activations = activations or [row[:4] for row in logits]
  texts = texts or {}
  last_input = first_position + len(inputs) - 1
  first_step = last_input + 1
  input_entries = [
      {
          'id': identity,
          'text': text,
          'position': first_position + i,
          'origin': origin,
      }
      for i, (identity, text, origin) in enumerate(inputs)
  ]
  forwards = [{
      'id': 0,
      'signature': 'prefill_128',
      'phase': 'prefill',
      'step': first_step,
      'input_positions': [first_position, last_input - 1],
      'tensors': {},
  }]
  output_entries, files = [], []
  for k, row in enumerate(logits):
    step = first_step + k
    names = {
        'post_logits': f'decode_post_logits_step_{step}.safetensors',
        'post_activations': f'decode_post_activations_step_{step}.safetensors',
    }
    _save(
        directory / names['post_logits'],
        'post_logits',
        np.array(row, np.float32).reshape(1, 1, -1),
        step,
    )
    _save(
        directory / names['post_activations'],
        'post_activations',
        np.array(activations[k], np.float32).reshape(1, 1, -1),
        step,
    )
    forwards.append({
        'id': k + 1,
        'signature': 'decode',
        'phase': 'decode',
        'step': step,
        'input_position': step - 1,
        'input_positions': [step - 1],
        'tensors': names,
        'argmax': int(np.argmax(row)),
    })
    output_entries.append({
        'id': outputs[k],
        'text': texts.get(outputs[k]),
        'position': step,
        'forward': k + 1,
        'released': True,
    })
    for name in names.values():
      blob = (directory / name).read_bytes()
      files.append({
          'path': name,
          'size': len(blob),
          'sha256': hashlib.sha256(blob).hexdigest(),
      })
  return {
      'dir': f'{side}/turn-{n}',
      'completion': {
          'content': ''.join(texts.get(i, '') for i in outputs),
          'reasoning_content': '',
          'finish_reason': 'length',
          'usage': {
              'prompt_tokens': len(inputs),
              'completion_tokens': len(outputs),
              'total_tokens': len(inputs) + len(outputs),
          },
          'prompt_tokens': [],
          'context_token_count': first_step + len(outputs),
      },
      'stop_reason': 'max_output_tokens',
      'input': input_entries,
      'output': output_entries,
      'forwards': forwards,
      'kv': [],
      'files': files,
  }


def _index(
    root: pathlib.Path, turns: list[dict[str, dict]], **fixed: dict
) -> capture_index.CaptureIndex:
  """Writes index.json for the turns' side entries and loads it."""
  sides = {side: fixed.get(side, _fixed()) for side in ('ref', 'target')}
  index = {
      'format_version': 1,
      'sides': sides,
      'turns': [
          {'n': n, 'prompt': 'prompt', 'sides': entries}
          for n, entries in enumerate(turns, start=1)
      ],
  }
  (root / 'index.json').write_text(json.dumps(index))
  return capture_index.CaptureIndex.load(root)


PROMPT = [(START, None, 'start'), (5, 'Hi', 'prompt'), (3, '!', 'prompt')]
TEXTS = {0: 'zero', 1: 'one', 4: 'four'}
# Three decode steps over a vocabulary of six; argmax 4, 1, 0.
LOGITS = [
    [0.0, 1.0, 0.5, -1.0, 3.0, 2.0],
    [0.2, 2.5, 0.1, 0.0, 1.0, -0.5],
    [4.0, 1.0, 0.5, 0.25, 2.0, 1.5],
]


def _same_turn(root: pathlib.Path, target_logits=None, **target) -> dict:
  return {
      'ref': _write_side_turn(root, 'ref', 1, PROMPT, LOGITS, texts=TEXTS),
      'target': _write_side_turn(
          root,
          'target',
          1,
          PROMPT,
          target_logits or LOGITS,
          texts=TEXTS,
          **target,
      ),
  }


def _softmax(values: list[float]) -> np.ndarray:
  exp = np.exp(np.array(values, np.float32).astype(np.float64))
  return exp / exp.sum()


def test_identical_sides_have_zero_metrics(tmp_path):
  """One turn against itself: every metric is zero."""
  index = _index(tmp_path, [_same_turn(tmp_path)])
  response = tokens.token_analysis(
      index, {'turn': 1, 'pairs': [{'ref': 0, 'target': 0}]}
  )
  assert response['turn'] == 1
  (pair,) = response['pairs']
  assert (pair['ref_step'], pair['target_step']) == (0, 0)
  assert set(pair['metrics']) == {
      'kl',
      'js',
      'relative_l2',
      'cosine_distance',
      'norm_ratio',
      'max_abs_error',
  }
  assert pair['metrics']['kl'] == {'value': 0.0, 'reason': None}
  assert pair['metrics']['js'] == {'value': 0.0, 'reason': None}
  assert pair['metrics']['relative_l2'] == {'value': 0.0, 'reason': None}
  assert pair['metrics']['norm_ratio']['value'] == pytest.approx(1.0)
  assert pair['metrics']['max_abs_error'] == {'value': 0.0, 'reason': None}
  for transform in ('raw', 'centered'):
    assert pair['final_logits'][transform] == {
        'rmse': {'value': 0.0, 'reason': None},
        'max_abs': {'value': 0.0, 'reason': None},
        'max_abs_index': 0,
    }
  distribution = pair['distribution']
  assert distribution['compatible'] is True
  assert distribution['reason'] is None
  assert distribution['context'] == 'same'
  p = _softmax(LOGITS[0])
  ref = distribution['ref']
  assert ref['size'] == 6
  assert ref['source_forward_id'] == 1
  assert ref['selected_id'] == 4
  assert ref['selected_rank'] == 1
  assert ref['selected_probability'] == pytest.approx(p[4])
  assert ref['margin'] == pytest.approx(p[4] - p[5])
  assert ref['entropy'] == pytest.approx(-np.sum(p * np.log2(p)))
  assert ref['reason'] is None
  assert ref['logits_metadata'] == {
      'source_shape': [1, 1, 6],
      'source_dtype': 'float32',
      'comparison_shape': [6],
      'vocab_identity': f'{MODEL}:6',
  }
  assert distribution['target'] == ref
  rows = distribution['rows']
  # The sampled id first, then by probability; every id of a small vocab.
  assert [row['id'] for row in rows] == [4, 5, 1, 2, 0, 3]
  # Labels are texts the chat produced: released, prompt, else none.
  assert [row['label'] for row in rows] == [
      'four',
      'Hi',
      'one',
      None,
      'zero',
      '!',
  ]
  assert rows[0]['ref'] == {
      'probability': pytest.approx(p[4]),
      'rank': 1,
      'logit': 3.0,
      'selected': True,
  }
  assert rows[1]['ref']['selected'] is False
  assert all(row['delta'] == 0.0 for row in rows)


def test_sampling_distribution_reconstructs_greedy(tmp_path):
  """Greedy sampling puts all probability on the argmax."""
  index = _index(tmp_path, [_same_turn(tmp_path)])
  (pair,) = tokens.token_analysis(
      index, {'turn': 1, 'pairs': [{'ref': 2, 'target': 2}]}
  )['pairs']
  sampling = pair['sampling_distribution']
  assert sampling['context'] == 'same'
  assert sampling['vocabulary_compatible'] is True
  assert sampling['rows_complete'] is True
  assert sampling['reason'] is None
  for side in ('ref', 'target'):
    value = sampling[side]
    assert value['basis'] == 'reconstructed_greedy'
    assert value['candidate_count'] == 1
    assert value['selected_id'] == 0
    assert (value['entropy'], value['selected_probability']) == (0.0, 1.0)
    assert (value['selected_rank'], value['margin']) == (1, 1.0)
    assert value['sampler'] == {
        'temperature': None,
        'top_k': 1,
        'top_p': None,
        'seed': None,
    }
  assert sampling['rows'] == [{
      'id': 0,
      'label': 'zero',
      'ref': {'probability': 1.0, 'rank': 1, 'selected': True},
      'target': {'probability': 1.0, 'rank': 1, 'selected': True},
  }]


def test_sampling_distribution_needs_greedy_sampling(tmp_path):
  """Other samplers have no reconstructed distribution."""
  index = _index(
      tmp_path,
      [_same_turn(tmp_path)],
      target=_fixed(top_k=40, temperature=0.8, seed=7),
  )
  (pair,) = tokens.token_analysis(
      index, {'turn': 1, 'pairs': [{'ref': 0, 'target': 0}]}
  )['pairs']
  sampling = pair['sampling_distribution']
  target = sampling['target']
  assert target['basis'] is None
  assert target['candidate_count'] is None
  assert target['entropy'] is None
  assert target['sampler'] == {
      'temperature': 0.8,
      'top_k': 40,
      'top_p': None,
      'seed': 7,
  }
  assert 'only greedy sampling' in target['reason']
  assert sampling['reason'] == 'target: ' + target['reason']
  # The ref side still has its own row.
  assert sampling['rows'] == [{
      'id': 4,
      'label': 'four',
      'ref': {'probability': 1.0, 'rank': 1, 'selected': True},
      'target': None,
  }]
  # Logits comparison does not depend on the sampler.
  assert pair['distribution']['compatible'] is True


def test_sampling_distribution_refuses_speculative_decoding(tmp_path):
  """Speculative decoding has no reconstructed distribution."""
  index = _index(
      tmp_path,
      [_same_turn(tmp_path)],
      ref=_fixed(runtime_options={'speculative_decoding': True}),
  )
  (pair,) = tokens.token_analysis(
      index, {'turn': 1, 'pairs': [{'ref': 0, 'target': 0}]}
  )['pairs']
  assert (
      'speculative decoding' in pair['sampling_distribution']['ref']['reason']
  )


def test_sampling_distribution_needs_the_argmax(tmp_path):
  """A sampled id other than the argmax is not greedy."""
  # The target released id 5 although its logits' argmax is 4.
  turn = _same_turn(tmp_path, outputs=[5, 1, 0])
  index = _index(tmp_path, [turn])
  (pair,) = tokens.token_analysis(
      index, {'turn': 1, 'pairs': [{'ref': 0, 'target': 0}]}
  )['pairs']
  target = pair['sampling_distribution']['target']
  assert target['basis'] is None
  assert target['reason'] == (
      'Greedy reconstruction does not match the sampled token and source logits'
  )
  assert pair['distribution']['target']['selected_id'] == 5
  assert pair['distribution']['target']['selected_rank'] == 2


def test_perturbed_target_by_hand(tmp_path):
  """One changed logit, every metric by hand."""
  target_logits = [list(LOGITS[0]), *LOGITS[1:]]
  target_logits[0][5] = 2.5
  index = _index(tmp_path, [_same_turn(tmp_path, target_logits)])
  (pair,) = tokens.token_analysis(
      index, {'turn': 1, 'pairs': [{'ref': 0, 'target': 0}]}
  )['pairs']
  p, q = _softmax(LOGITS[0]), _softmax(target_logits[0])
  values = pair['metrics']
  assert values['kl']['value'] == pytest.approx(np.sum(p * np.log(p / q)))
  m = (p + q) / 2
  js = 0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m))
  assert values['js']['value'] == pytest.approx(js)
  # Activations are the logits' first four values: unchanged.
  assert values['relative_l2']['value'] == 0.0
  raw = pair['final_logits']['raw']
  assert raw['max_abs'] == {'value': 0.5, 'reason': None}
  assert raw['max_abs_index'] == 5
  assert raw['rmse']['value'] == pytest.approx(math.sqrt(0.25 / 6))
  centered = pair['final_logits']['centered']
  # Centering subtracts each side's mean: 5.5/6 and 6/6.
  delta = np.array(target_logits[0]) - 1.0 - (np.array(LOGITS[0]) - 5.5 / 6)
  assert centered['rmse']['value'] == pytest.approx(
      math.sqrt(np.mean(delta**2))
  )
  assert centered['max_abs_index'] == 5
  row = next(row for row in pair['distribution']['rows'] if row['id'] == 5)
  assert row['delta'] == pytest.approx((q[5] - p[5]) * 100)
  assert row['target']['rank'] == 2


def test_activation_metrics_by_hand(tmp_path):
  """Activation metrics of two orthogonal vectors, by hand."""
  rest = [[1.0] * 4, [1.0] * 4]
  turn = {
      'ref': _write_side_turn(
          tmp_path, 'ref', 1, PROMPT, LOGITS, activations=[[0, 4, 0, 0], *rest]
      ),
      'target': _write_side_turn(
          tmp_path,
          'target',
          1,
          PROMPT,
          LOGITS,
          activations=[[2, 0, 0, 0], *rest],
      ),
  }
  index = _index(tmp_path, [turn])
  (pair,) = tokens.token_analysis(
      index, {'turn': 1, 'pairs': [{'ref': 0, 'target': 0}]}
  )['pairs']
  values = pair['metrics']
  # ref [0, 4], target [2, 0]: |t - r| / |r| = sqrt(20) / 4, in percent.
  assert values['relative_l2']['value'] == pytest.approx(
      100 * math.sqrt(20) / 4
  )
  assert values['norm_ratio']['value'] == pytest.approx(0.5)
  assert values['cosine_distance']['value'] == pytest.approx(1.0)
  assert values['max_abs_error']['value'] == pytest.approx(4.0)


def test_diverged_generations_are_not_compared(tmp_path):
  """After the sides generate differently, pairs are not compared."""
  target_logits = [list(LOGITS[0]), *LOGITS[1:]]
  target_logits[0][5] = 5.0  # The target generates 5 where ref made 4.
  index = _index(tmp_path, [_same_turn(tmp_path, target_logits)])
  pairs = tokens.token_analysis(
      index,
      {'turn': 1, 'pairs': [{'ref': 0, 'target': 0}, {'ref': 1, 'target': 1}]},
  )['pairs']
  # The first tokens follow the same context: still comparable.
  assert pairs[0]['distribution']['compatible'] is True
  assert pairs[0]['distribution']['context'] == 'same'
  diverged = pairs[1]
  distribution = diverged['distribution']
  assert distribution['compatible'] is False
  assert distribution['context'] == 'different'
  assert distribution['reason'] == tokens.CONTEXT_REASON
  for key in diverged['metrics']:
    assert diverged['metrics'][key] == {
        'value': None,
        'reason': tokens.CONTEXT_REASON,
    }
  assert diverged['final_logits']['raw']['rmse']['reason'] == (
      tokens.CONTEXT_REASON
  )
  # One vocabulary: each side still shows its candidates, without deltas.
  assert len(distribution['rows']) == 6
  assert all(row['delta'] is None for row in distribution['rows'])
  assert distribution['ref']['entropy'] is not None
  assert diverged['sampling_distribution']['context'] == 'different'


def test_different_models_have_no_shared_vocabulary(tmp_path):
  """Different model files share no vocabulary."""
  index = _index(
      tmp_path, [_same_turn(tmp_path)], target=_fixed(model='b' * 64)
  )
  (pair,) = tokens.token_analysis(
      index, {'turn': 1, 'pairs': [{'ref': 0, 'target': 0}]}
  )['pairs']
  distribution = pair['distribution']
  assert distribution['compatible'] is False
  assert distribution['reason'] == tokens.VOCABULARY_REASON
  assert distribution['context'] == 'same'
  assert distribution['rows'] == []
  assert pair['metrics']['kl']['reason'] == tokens.VOCABULARY_REASON
  assert pair['metrics']['relative_l2']['reason'] == (
      'Matching activation coordinate spaces are not established'
  )
  sampling = pair['sampling_distribution']
  assert sampling['vocabulary_compatible'] is False
  assert sampling['rows'] == []
  assert sampling['rows_complete'] is False
  assert sampling['reason'] == tokens.VOCABULARY_REASON


def test_one_side_missing(tmp_path):
  """A pair with one side shows that side alone."""
  index = _index(tmp_path, [_same_turn(tmp_path)])
  (pair,) = tokens.token_analysis(
      index, {'turn': 1, 'pairs': [{'ref': 1, 'target': None}]}
  )['pairs']
  assert (pair['ref_step'], pair['target_step']) == (1, None)
  distribution = pair['distribution']
  assert distribution['compatible'] is False
  assert distribution['context'] is None
  assert distribution['reason'] == 'target: Token not captured on this side'
  assert distribution['target'] == {'reason': 'Token not captured on this side'}
  assert distribution['ref']['selected_id'] == 1
  assert [row['id'] for row in distribution['rows']][:2] == [1, 4]
  assert all(row['target'] is None for row in distribution['rows'])
  assert pair['metrics']['relative_l2']['reason'] == (
      'Paired source forwards not captured'
  )
  assert pair['final_logits']['raw']['max_abs_index'] is None
  sampling = pair['sampling_distribution']
  assert sampling['context'] is None
  assert sampling['ref']['basis'] == 'reconstructed_greedy'
  assert sampling['target']['reason'] == 'Token not captured on this side'
  # A step past the side's output is not captured either.
  (pair,) = tokens.token_analysis(
      index, {'turn': 1, 'pairs': [{'ref': 3, 'target': 0}]}
  )['pairs']
  assert pair['distribution']['ref'] == {
      'reason': 'Token not captured on this side'
  }


def test_context_spans_earlier_turns(tmp_path):
  """The context of a token includes the earlier turns."""
  turn_1 = _same_turn(tmp_path)
  # Turn 2 feeds the same tokens on both sides; only its earlier turn says
  # whether the contexts match.
  carried = [(9, None, 'carried_stop'), (5, 'Hi', 'prompt')]
  turn_2 = {
      side: _write_side_turn(
          tmp_path, side, 2, carried, LOGITS[:1], first_position=6
      )
      for side in ('ref', 'target')
  }
  index = _index(tmp_path, [turn_1, turn_2])
  (pair,) = tokens.token_analysis(
      index, {'turn': 2, 'pairs': [{'ref': 0, 'target': 0}]}
  )['pairs']
  assert pair['distribution']['context'] == 'same'
  assert pair['distribution']['ref']['source_forward_id'] == 1

  other = tmp_path / 'other'
  other.mkdir()
  diverged_1 = _same_turn(other, outputs=[4, 1, 2])
  turn_2 = {
      side: _write_side_turn(
          other, side, 2, carried, LOGITS[:1], first_position=6
      )
      for side in ('ref', 'target')
  }
  index = _index(other, [diverged_1, turn_2])
  (pair,) = tokens.token_analysis(
      index, {'turn': 2, 'pairs': [{'ref': 0, 'target': 0}]}
  )['pairs']
  assert pair['distribution']['context'] == 'different'


def test_tokens_the_runner_did_not_name_prove_no_context(tmp_path):
  """Two unknown ids at one position are not the same token."""
  history = [(None, None, 'history'), (5, 'Hi', 'prompt')]
  turn_2 = {
      side: _write_side_turn(
          tmp_path, side, 2, history, LOGITS[:1], first_position=6
      )
      for side in ('ref', 'target')
  }
  index = _index(tmp_path, [_same_turn(tmp_path), turn_2])
  (pair,) = tokens.token_analysis(
      index, {'turn': 2, 'pairs': [{'ref': 0, 'target': 0}]}
  )['pairs']
  assert pair['distribution']['context'] == 'different'
  assert pair['distribution']['reason'] == tokens.CONTEXT_REASON


def test_unknown_turn_is_not_found(tmp_path):
  """A turn that was not captured is not found."""
  index = _index(tmp_path, [_same_turn(tmp_path)])
  with pytest.raises(errors.NotFound):
    tokens.token_analysis(index, {'turn': 2, 'pairs': []})


@pytest.mark.parametrize(
    'payload',
    [
        {'turn': 0, 'pairs': []},
        {'turn': True, 'pairs': []},
        {'turn': 1},
        {'turn': 1, 'pairs': [{'ref': 0, 'target': 0}] * 129},
        {'turn': 1, 'pairs': [{'ref': 0}]},
        {'turn': 1, 'pairs': [{'ref': -1, 'target': 0}]},
        {'turn': 1, 'pairs': [{'ref': False, 'target': 0}]},
        {'turn': 1, 'pairs': [{'ref': 0, 'target': 0, 'x': 1}]},
    ],
)
def test_token_analysis_refuses_malformed_payloads(tmp_path, payload):
  """Malformed token-analysis requests are refused."""
  index = _index(tmp_path, [_same_turn(tmp_path)])
  with pytest.raises(ValueError):
    tokens.token_analysis(index, payload)


def _final_logits(index, **fields):
  payload = {
      'turn': 1,
      'ref_step': 0,
      'target_step': 0,
      'transform': 'raw',
      'offset': 0,
      'limit': 16,
      'order': 'index',
      'direction': 'ascending',
      **fields,
  }
  return tokens.final_logits(index, payload)


def test_final_logits_pages_and_sorts(tmp_path):
  """Final logits are sorted over the whole vocabulary, then paged."""
  target_logits = [list(LOGITS[0]), *LOGITS[1:]]
  target_logits[0][2] = 3.5
  index = _index(tmp_path, [_same_turn(tmp_path, target_logits)])
  response = _final_logits(index, offset=1, limit=2)
  assert {
      key: response[key]
      for key in (
          'turn',
          'ref_step',
          'target_step',
          'transform',
          'offset',
          'limit',
          'total',
          'compatible',
          'reason',
      )
  } == {
      'turn': 1,
      'ref_step': 0,
      'target_step': 0,
      'transform': 'raw',
      'offset': 1,
      'limit': 2,
      'total': 6,
      'compatible': True,
      'reason': None,
  }
  assert response['rows'] == [
      {
          'index': 1,
          'label': 'one',
          'ref': 1.0,
          'target': 1.0,
          'delta': 0.0,
          'absolute_delta': 0.0,
      },
      {
          'index': 2,
          'label': None,
          'ref': 0.5,
          'target': 3.5,
          'delta': 3.0,
          'absolute_delta': 3.0,
      },
  ]
  assert response['metrics']['max_abs'] == {'value': 3.0, 'reason': None}
  assert response['metrics']['max_abs_index'] == 2
  assert response['ref']['logits_metadata']['comparison_shape'] == [6]

  def indices(**fields):
    return [row['index'] for row in _final_logits(index, **fields)['rows']]

  assert indices(direction='descending', limit=3) == [5, 4, 3]
  assert indices(order='ref', direction='descending') == [4, 5, 1, 2, 0, 3]
  assert indices(order='target') == [3, 0, 1, 5, 4, 2]
  assert indices(order='absolute_delta', direction='descending')[:2] == [2, 0]
  # Equal values keep index order in either direction.
  assert indices(order='delta', direction='descending') == [2, 0, 1, 3, 4, 5]
  assert indices(order='delta', direction='ascending') == [0, 1, 3, 4, 5, 2]
  assert _final_logits(index, offset=6)['rows'] == []


def test_final_logits_puts_non_finite_values_last(tmp_path):
  """Non-finite values sort last and are spelled out."""
  target_logits = [list(LOGITS[0]), *LOGITS[1:]]
  target_logits[0][1] = -math.inf
  index = _index(tmp_path, [_same_turn(tmp_path, target_logits)])
  response = _final_logits(index, order='target', direction='ascending')
  assert [row['index'] for row in response['rows']] == [3, 0, 2, 5, 4, 1]
  assert response['rows'][-1]['target'] == '-Infinity'
  assert response['rows'][-1]['delta'] == '-Infinity'
  assert response['rows'][-1]['absolute_delta'] == 'Infinity'
  assert response['metrics']['rmse']['reason'] == (
      'Logits contain non-finite values; full-vector metrics are unavailable'
  )
  centered = _final_logits(index, transform='centered')
  assert centered['compatible'] is False
  assert centered['target']['reason'] == (
      'Centered logits unavailable: source contains non-finite values'
  )
  assert centered['rows'][0]['target'] is None
  assert centered['rows'][0]['delta'] is None


def test_final_logits_without_a_comparable_pair(tmp_path):
  """Without a shared vocabulary there are no labels or deltas."""
  index = _index(
      tmp_path, [_same_turn(tmp_path)], target=_fixed(model='b' * 64)
  )
  response = _final_logits(index, order='delta')
  assert response['compatible'] is False
  assert response['reason'] == tokens.VOCABULARY_REASON
  # Different vocabularies share no labels and no deltas.
  assert [row['index'] for row in response['rows']] == list(range(6))
  assert all(row['label'] is None for row in response['rows'])
  assert all(row['delta'] is None for row in response['rows'])
  single = _final_logits(index, target_step=None)
  assert single['reason'] == 'target: Token not captured on this side'
  assert single['rows'][4]['label'] == 'four'
  assert single['rows'][4]['target'] is None
  assert single['target'] == {'reason': 'Token not captured on this side'}


@pytest.mark.parametrize(
    'fields',
    [
        {'limit': 0},
        {'limit': 257},
        {'offset': -1},
        {'order': 'probability'},
        {'direction': 'up'},
        {'transform': 'softmax'},
        {'turn': 0},
        {'ref_step': -1},
        {'extra': 1},
    ],
)
def test_final_logits_refuses_malformed_payloads(tmp_path, fields):
  """Malformed final-logits requests are refused."""
  index = _index(tmp_path, [_same_turn(tmp_path)])
  with pytest.raises(ValueError):
    _final_logits(index, **fields)


def test_final_logits_offset_past_the_vocabulary(tmp_path):
  """An offset past the vocabulary is refused."""
  index = _index(tmp_path, [_same_turn(tmp_path)])
  with pytest.raises(ValueError, match='offset exceeds'):
    _final_logits(index, offset=7)


def test_final_logits_vector(tmp_path, monkeypatch):
  """The whole vectors, echoed request, and the size budget."""
  target_logits = [list(LOGITS[0]), *LOGITS[1:]]
  target_logits[0][1] = math.nan
  index = _index(tmp_path, [_same_turn(tmp_path, target_logits)])
  payload = {'turn': 1, 'ref_step': 0, 'target_step': 0, 'transform': 'raw'}
  response = tokens.final_logits_vector(index, payload)
  assert {key: response[key] for key in payload} == payload
  assert response['complete'] is True
  assert response['total'] == 6
  assert response['values']['ref'] == LOGITS[0]
  assert response['values']['target'][1] == 'NaN'
  assert 'rows' not in response and 'offset' not in response
  single = tokens.final_logits_vector(index, {**payload, 'ref_step': None})
  assert single['values']['ref'] is None
  with pytest.raises(ValueError):
    tokens.final_logits_vector(index, {**payload, 'offset': 0})
  monkeypatch.setattr(tokens, 'MAX_VECTOR_VALUES', 5)
  with pytest.raises(ValueError, match='no values were truncated'):
    tokens.final_logits_vector(index, payload)


def _real_side_turn(
    root: pathlib.Path, side: str, source: pathlib.Path
) -> dict:
  """The index entry of the real turn, its files linked into `side`."""
  directory = root / side / 'turn-1'
  directory.mkdir(parents=True)
  for path in source.glob('*.safetensors'):
    (directory / path.name).symlink_to(path)
  completion = json.loads((REAL_TURN / '_completion.json').read_text())
  prompt = completion['model_debugger']['prompt_tokens']
  released = [
      json.loads(line)
      for line in (
          (REAL_TURN / 'generated_tokens.jsonl').read_text().splitlines()
      )
  ]
  first_step = 1 + len(prompt)
  inputs = [{'id': START, 'text': None, 'position': 0, 'origin': 'start'}] + [
      {
          'id': token['id'],
          'text': token['text'],
          'position': i + 1,
          'origin': 'prompt',
      }
      for i, token in enumerate(prompt)
  ]
  forwards = [{
      'id': 0,
      'signature': 'prefill_128',
      'phase': 'prefill',
      'step': first_step,
      'input_positions': [0, first_step - 2],
      'tensors': {},
  }]
  outputs = []
  for k, line in enumerate(released):
    step = first_step + k
    forwards.append({
        'id': k + 1,
        'signature': 'decode',
        'phase': 'decode',
        'step': step,
        'input_position': step - 1,
        'input_positions': [step - 1],
        'tensors': {
            'post_logits': f'decode_post_logits_step_{step}.safetensors',
            'post_activations': (
                f'decode_post_activations_step_{step}.safetensors'
            ),
        },
    })
    outputs.append({
        'id': line['token_ids'][0][0],
        'text': line['texts'][0],
        'position': step,
        'forward': k + 1,
        'released': True,
    })
  return {
      'dir': f'{side}/turn-1',
      'completion': {},
      'stop_reason': 'max_output_tokens',
      'input': inputs,
      'output': outputs,
      'forwards': forwards,
      'kv': [],
      'files': [],
  }


@pytest.mark.skipif(not REAL_TURN.is_dir(), reason='no real capture here')
def test_real_turn_against_itself_and_a_perturbed_copy(tmp_path):
  """The real turn: zero against itself, non-zero against noise."""
  perturbed = tmp_path / 'perturbed'
  perturbed.mkdir()
  for path in REAL_TURN.glob('decode_post_*_step_*.safetensors'):
    if 'kv_cache' in path.name:
      continue
    key = path.name.split('_step_')[0].removeprefix('decode_')
    data = safetensors_numpy.load_file(str(path))[key]
    # Small noise that keeps every argmax.
    noise = np.random.default_rng(0).normal(0, 0.01, data.shape)
    step = int(path.name.split('_step_')[1].split('.')[0])
    _save(perturbed / path.name, key, (data + noise).astype(np.float32), step)

  same_root = tmp_path / 'same'
  same_root.mkdir()
  index = _index(
      same_root,
      [{
          side: _real_side_turn(same_root, side, REAL_TURN)
          for side in ('ref', 'target')
      }],
  )
  steps = list(range(len(index.turn(1)['sides']['ref']['output'])))
  assert len(steps) == 8
  response = tokens.token_analysis(
      index,
      {'turn': 1, 'pairs': [{'ref': step, 'target': step} for step in steps]},
  )
  for pair in response['pairs']:
    for key in ('kl', 'js', 'relative_l2', 'max_abs_error'):
      assert pair['metrics'][key] == {'value': 0.0, 'reason': None}, key
    assert pair['final_logits']['raw']['rmse']['value'] == 0.0
    distribution = pair['distribution']
    assert distribution['context'] == 'same'
    assert distribution['ref'] == distribution['target']
    assert distribution['ref']['size'] == 262144
    assert distribution['ref']['selected_rank'] == 1
    assert distribution['ref']['logits_metadata']['source_shape'] == [
        1,
        1,
        262144,
    ]
    assert len(distribution['rows']) == 64
    sampling = pair['sampling_distribution']
    assert sampling['ref']['basis'] == 'reconstructed_greedy'
    assert sampling['ref'] == sampling['target']
  first = response['pairs'][0]['distribution']['rows'][0]
  assert (first['id'], first['label']) == (236778, '2')
  vector = tokens.final_logits_vector(
      index, {'turn': 1, 'ref_step': 7, 'target_step': 7, 'transform': 'raw'}
  )
  assert vector['values']['ref'] == vector['values']['target']
  assert len(vector['values']['ref']) == 262144

  perturbed_root = tmp_path / 'perturbed_index'
  perturbed_root.mkdir()
  index = _index(
      perturbed_root,
      [{
          'ref': _real_side_turn(perturbed_root, 'ref', REAL_TURN),
          'target': _real_side_turn(perturbed_root, 'target', perturbed),
      }],
  )
  response = tokens.token_analysis(
      index,
      {'turn': 1, 'pairs': [{'ref': step, 'target': step} for step in steps]},
  )
  for pair in response['pairs']:
    assert pair['distribution']['compatible'] is True
    for key in ('kl', 'js', 'relative_l2', 'cosine_distance', 'max_abs_error'):
      assert pair['metrics'][key]['value'] > 0, key
    raw = pair['final_logits']['raw']
    assert 0 < raw['rmse']['value'] < raw['max_abs']['value']
    assert pair['sampling_distribution']['target']['basis'] == (
        'reconstructed_greedy'
    )
  page = tokens.final_logits(
      index,
      {
          'turn': 1,
          'ref_step': 0,
          'target_step': 0,
          'transform': 'centered',
          'offset': 0,
          'limit': 16,
          'order': 'absolute_delta',
          'direction': 'descending',
      },
  )
  assert page['total'] == 262144
  deltas = [row['absolute_delta'] for row in page['rows']]
  assert deltas == sorted(deltas, reverse=True)


def test_corrupt_or_missing_safetensors_reports_per_pair_reason(tmp_path):
  """Corrupt or missing safetensors files degrade per pair without raising."""
  index = _index(tmp_path, [_same_turn(tmp_path)])
  target_forwards = index.turn(1)['sides']['target']['forwards']
  decode_0 = next(f for f in target_forwards if f['phase'] == 'decode')
  index.path('target', 1, decode_0['tensors']['post_logits']).write_bytes(
      b'corrupt'
  )
  index.path('target', 1, decode_0['tensors']['post_activations']).unlink()

  response = tokens.token_analysis(
      index,
      {'turn': 1, 'pairs': [{'ref': 0, 'target': 0}, {'ref': 1, 'target': 1}]},
  )
  first, second = response['pairs']
  assert first['distribution']['compatible'] is False
  assert first['distribution']['reason'] is not None
  assert first['metrics']['relative_l2']['value'] is None
  assert first['metrics']['relative_l2']['reason'] is not None
  assert second['distribution']['compatible'] is True
  assert second['metrics']['relative_l2']['value'] == 0.0

  logits_page = tokens.final_logits(
      index,
      {
          'turn': 1,
          'ref_step': 0,
          'target_step': 0,
          'transform': 'raw',
          'offset': 0,
          'limit': 4,
          'order': 'index',
      },
  )
  assert logits_page['compatible'] is False
  assert logits_page['reason'] is not None
