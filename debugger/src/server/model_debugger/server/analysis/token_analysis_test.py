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

"""Tests of token-analysis and the final-logits routes.

Each test writes a small capture's files (one safetensors file per tensor,
named and keyed as the runner names them), indexes them by hand as
CaptureIndex derives a turn, and checks the responses against values
computed by hand.
"""

import json
import math
import pathlib
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from model_debugger.server.analysis import metrics
from model_debugger.server.analysis import token_analysis
from model_debugger.server.capture import capture_index
from model_debugger.server.testing import capture_fixtures as fixtures
from model_debugger.server.testing import real_capture
from model_debugger.server.testing import test_main
from model_debugger.server.util import errors
import numpy as np
from safetensors import numpy as safetensors_numpy

MODEL = 'a' * 64
START = 2
STOP = 106


def _fixed(model: str = MODEL, **options) -> dict[str, Any]:
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
    rewrite: list[int] | None = None,
) -> dict[str, Any]:
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
      activations: Each decode step's activations (default: the logits' first
        four values).
      rewrite: The ids the turn fed after its decode steps, from the first one's
        position on (default: none).

  Returns:
      The side's entry of the turn, as CaptureIndex derives it.
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
      size = (directory / name).stat().st_size
      files.append({'path': name, 'size': size})
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
          'context_token_count': first_step + len(outputs),
      },
      'stop_reason': 'max_output_tokens',
      'input': input_entries,
      'output': output_entries,
      'rewrite': [
          {'id': id, 'text': None, 'position': last_input + i, 'forward': 0}
          for i, id in enumerate(rewrite or [])
      ],
      'forwards': forwards,
      'kv': [],
      'files': files,
  }


def _index(
    root: pathlib.Path,
    turns: list[dict[str, dict[str, Any]]],
    **fixed: dict[str, Any],
) -> capture_index.CaptureIndex:
  """Returns the capture of the turns' side entries."""
  return capture_index.CaptureIndex(
      root,
      {side: fixed.get(side, _fixed()) for side in ('ref', 'target')},
      [
          {'n': n, 'prompt': 'prompt', 'sides': entries}
          for n, entries in enumerate(turns, start=1)
      ],
  )


PROMPT = [(START, None, 'start'), (5, 'Hi', 'prompt'), (3, '!', 'prompt')]
TEXTS = {0: 'zero', 1: 'one', 4: 'four'}
# Three decode steps over a vocabulary of six; argmax 4, 1, 0.
LOGITS = [
    [0.0, 1.0, 0.5, -1.0, 3.0, 2.0],
    [0.2, 2.5, 0.1, 0.0, 1.0, -0.5],
    [4.0, 1.0, 0.5, 0.25, 2.0, 1.5],
]


def _same_turn(
    root: pathlib.Path, target_logits=None, **target
) -> dict[str, Any]:
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


def _why(code: str, side: str | None = None) -> dict[str, Any]:
  """The reason of `code` in token_analysis.REASONS's words."""
  return metrics.reason(code, token_analysis.REASONS[code], side)


def _vocabulary_why(words: str) -> dict[str, Any]:
  """VOCABULARY_MISMATCH, its words followed by why."""
  code = token_analysis.VOCABULARY_MISMATCH
  return metrics.reason(code, f'{token_analysis.REASONS[code]}: {words}.')


# Why two models whose captures record no tokenizer share no vocabulary.
NO_TOKENIZER = _vocabulary_why(
    'Reference and Target ran different models, and the capture records no '
    'tokenizer of Reference and Target'
)


def _tokenized(model: str, table: str | None) -> dict[str, Any]:
  """A side's fixed info whose model records a tokenizer's table digest."""
  fixed = _fixed(model=model)
  if table is not None:
    fixed['model']['vocabulary'] = {
        'table_sha256': table,
        'size': len(LOGITS[0]),
        'source': 'tokenizer.json',
    }
  return fixed


def _analyze(index, n, pairs):
  return token_analysis.token_analysis(index, n, pairs)


def _final_logits(index, **fields):
  arguments = {
      'n': 1,
      'ref_step': 0,
      'target_step': 0,
      'transform': 'raw',
      'offset': 0,
      'limit': 16,
      'order': 'index',
      'direction': 'ascending',
      **fields,
  }
  return token_analysis.final_logits(index, **arguments)


def _real_side_turn(
    root: pathlib.Path, side: str, source: pathlib.Path
) -> dict[str, Any]:
  """The index entry of the real turn, its files linked into `side`."""
  directory = root / side / 'turn-1'
  directory.mkdir(parents=True)
  for path in source.glob('*.safetensors'):
    (directory / path.name).symlink_to(path)
  real_turn = real_capture.turn('CPU')
  fed = real_turn.completion['fed_tokens']
  # The input: the prefill's rows and the first decode step's, which feeds
  # the last input token.
  first_decode = next(
      i for i, token in enumerate(fed) if token['phase'] == 'decode'
  )
  released = [
      json.loads(line)
      for line in (
          (real_turn.directory / 'generated_tokens.jsonl')
          .read_text()
          .splitlines()
      )
  ]
  first_step = first_decode + 1
  inputs = [
      {
          'id': token['id'],
          'text': token['text'],
          'position': token['position'],
          'origin': 'start' if i == 0 else 'prompt',
      }
      for i, token in enumerate(fed[:first_step])
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
      'rewrite': [],
      'forwards': forwards,
      'kv': [],
      'files': [],
  }


def _edited_capture(root: pathlib.Path, edit) -> capture_index.CaptureIndex:
  """A greedy chat whose reference's step that generates 'a' `edit` changes.

  The turn goes through the capture index as the server adds it, so the
  step is read by the index first.

  Args:
    root: The directory path for creating the capture index.
    edit: A callable applied to the logits of the reference step.

  Returns:
    The created capture index.
  """
  chats = {
      side: fixtures.TinyChat(seed=seed)
      for seed, side in enumerate(capture_index.SIDES)
  }
  index = capture_index.CaptureIndex.create(
      root, {side: fixtures.side_info(chat) for side, chat in chats.items()}
  )
  completions = {}
  for side, chat in chats.items():
    directory = index.turn_directory(side, 1)
    completions[side] = fixtures.write_turn(
        directory, chat.send('Hi there', ['a', 'b'])
    )
    if side == 'ref':
      fixtures.edit_logits(directory, 0, edit)
  index.add_turn(1, 'Hi there', completions)
  return index


class TokenAnalysisTest(parameterized.TestCase):
  """Tests of token-analysis and the final-logits routes."""

  def setUp(self):
    """Gives each test an empty directory of its own."""
    super().setUp()
    self.tmp_path = pathlib.Path(self.create_tempdir().full_path)

  def test_identical_sides_have_zero_metrics(self):
    """One turn against itself: every metric is zero."""
    index = _index(self.tmp_path, [_same_turn(self.tmp_path)])
    (pair,) = _analyze(index, 1, [(0, 0)])
    self.assertEqual((pair['ref_step'], pair['target_step']), (0, 0))
    self.assertEqual(
        set(pair['metrics']),
        {
            'kl',
            'js',
            'relative_l2',
            'cosine_similarity',
            'cosine_distance',
            'norm_ratio',
            'max_abs',
        },
    )
    self.assertEqual(pair['metrics']['kl'], {'value': 0.0, 'reasons': []})
    self.assertEqual(pair['metrics']['js'], {'value': 0.0, 'reasons': []})
    self.assertEqual(
        pair['metrics']['relative_l2'], {'value': 0.0, 'reasons': []}
    )
    np.testing.assert_allclose(
        pair['metrics']['norm_ratio']['value'], 1.0, rtol=1e-6
    )
    self.assertEqual(pair['metrics']['max_abs'], {'value': 0.0, 'reasons': []})
    self.assertEqual(
        pair['metrics']['cosine_similarity'], {'value': 1.0, 'reasons': []}
    )
    for transform in ('raw', 'centered'):
      self.assertEqual(
          pair['final_logits'][transform],
          {
              'rmse': {'value': 0.0, 'reasons': []},
              'max_abs': {'value': 0.0, 'reasons': []},
              'max_abs_index': 0,
          },
      )
    distribution = pair['distribution']
    self.assertIs(distribution['compatible'], True)
    self.assertEqual(distribution['reasons'], [])
    self.assertEqual(distribution['context'], 'same')
    p = _softmax(LOGITS[0])
    ref = distribution['ref']
    self.assertEqual(ref['size'], 6)
    self.assertEqual(ref['source_forward_id'], 1)
    self.assertEqual(ref['selected_id'], 4)
    self.assertEqual(ref['selected_rank'], 1)
    np.testing.assert_allclose(ref['selected_probability'], p[4], rtol=1e-6)
    np.testing.assert_allclose(ref['margin'], p[4] - p[5], rtol=1e-6)
    np.testing.assert_allclose(
        ref['entropy'], -np.sum(p * np.log2(p)), rtol=1e-6
    )
    self.assertEqual(ref['reasons'], [])
    self.assertEqual(
        ref['logits_metadata'],
        {
            'source_shape': [1, 1, 6],
            'source_dtype': 'float32',
            'comparison_shape': [6],
            'vocab_identity': f'{MODEL}:6',
        },
    )
    self.assertEqual(distribution['target'], ref)
    rows = distribution['rows']
    # The sampled id first, then by probability; every id of a small vocab.
    self.assertEqual([row['id'] for row in rows], [4, 5, 1, 2, 0, 3])
    # Labels are texts the chat produced: released, prompt, else none.
    self.assertEqual(
        [row['label'] for row in rows],
        [
            'four',
            'Hi',
            'one',
            None,
            'zero',
            '!',
        ],
    )
    ref = rows[0]['ref']
    np.testing.assert_allclose(ref.pop('probability'), p[4], rtol=1e-6)
    self.assertEqual(
        ref,
        {
            'rank': 1,
            'logit': 3.0,
            'selected': True,
        },
    )
    self.assertIs(rows[1]['ref']['selected'], False)
    self.assertTrue(all(row['delta'] == 0.0 for row in rows))

  def test_sampling_distribution_is_the_logits_softmax(self):
    """A greedy side's distribution is its logits' softmax, every id."""
    index = _index(self.tmp_path, [_same_turn(self.tmp_path)])
    (pair,) = _analyze(index, 1, [(2, 2)])
    sampling = pair['sampling_distribution']
    self.assertEqual(sampling['context'], 'same')
    self.assertIs(sampling['vocabulary_compatible'], True)
    # The rows are the most likely ids, never said to be every id.
    self.assertIs(sampling['rows_complete'], False)
    self.assertEqual(sampling['reasons'], [])
    p = _softmax(LOGITS[2])
    for side in ('ref', 'target'):
      value = sampling[side]
      self.assertEqual(value['basis'], token_analysis.SAMPLING_BASIS)
      self.assertEqual(value['temperature'], 1.0)
      self.assertEqual(value['candidate_count'], 6)
      self.assertEqual(value['selected_id'], 0)
      self.assertEqual(value['selected_rank'], 1)
      np.testing.assert_allclose(value['selected_probability'], p[0], rtol=1e-6)
      np.testing.assert_allclose(value['margin'], p[0] - p[4], rtol=1e-6)
      np.testing.assert_allclose(
          value['entropy'], -np.sum(p * np.log2(p)), rtol=1e-6
      )
      self.assertEqual(value['reasons'], [])
      self.assertEqual(
          value['sampler'],
          {
              'temperature': None,
              'top_k': 1,
              'top_p': None,
              'seed': None,
          },
      )
    rows = sampling['rows']
    # The sampled id first, then by probability: both sides' every id.
    self.assertEqual([row['id'] for row in rows], [0, 4, 5, 1, 2, 3])
    for row in rows:
      for side in ('ref', 'target'):
        np.testing.assert_allclose(
            row[side]['probability'], p[row['id']], rtol=1e-6
        )
    self.assertEqual(
        [row['ref']['selected'] for row in rows],
        [True, False, False, False, False, False],
    )

  def test_sampling_distribution_divides_by_the_temperature(self):
    """softmax(logits / T) at the side's own temperature, any sampler."""
    index = _index(
        self.tmp_path,
        [_same_turn(self.tmp_path)],
        target=_fixed(top_k=40, temperature=0.8, seed=7),
    )
    (pair,) = _analyze(index, 1, [(0, 0)])
    sampling = pair['sampling_distribution']
    target, ref = sampling['target'], sampling['ref']
    self.assertEqual(target['basis'], token_analysis.SAMPLING_BASIS)
    self.assertEqual(target['temperature'], 0.8)
    self.assertEqual(ref['temperature'], 1.0)
    self.assertEqual(
        target['sampler'],
        {
            'temperature': 0.8,
            'top_k': 40,
            'top_p': None,
            'seed': 7,
        },
    )
    self.assertEqual(sampling['reasons'], [])
    exp = np.exp(np.array(LOGITS[0], np.float32).astype(np.float64) / 0.8)
    cooled = exp / exp.sum()
    np.testing.assert_allclose(
        target['selected_probability'], cooled[4], rtol=1e-6
    )
    rows = {row['id']: row for row in sampling['rows']}
    self.assertLen(rows, 6)
    for identity, row in rows.items():
      np.testing.assert_allclose(
          row['target']['probability'], cooled[identity], rtol=1e-6
      )
      np.testing.assert_allclose(
          row['ref']['probability'],
          _softmax(LOGITS[0])[identity],
          rtol=1e-6,
      )
    # The logits' own distribution does not depend on the sampler.
    self.assertIs(pair['distribution']['compatible'], True)

  def test_a_piece_is_the_text_it_decodes_to(self):
    """A word boundary is a space; an ASCII byte piece is its byte."""
    self.assertEqual(token_analysis.piece_text('▁好的'.encode()), ' 好的')
    self.assertEqual(token_analysis.piece_text(b'<0x0A>'), '\n')
    # A byte of a character is not a character of its own.
    self.assertEqual(token_analysis.piece_text(b'<0xE4>'), '<0xE4>')
    self.assertEqual(token_analysis.piece_text(b'<turn|>'), '<turn|>')

  def test_candidates_the_chat_did_not_produce_are_named_by_the_tokenizer(
      self,
  ):
    """The chat's texts first; any other id its tokenizer's piece."""
    turn = mock.Mock(
        spec=['index', '_labels', '_pieces', 'labels'],
    )
    turn.index = mock.Mock()
    turn.index.turns = [{
        'sides': {
            'ref': {
                'input': [],
                'output': [{'id': 1, 'text': 'chat'}],
                'rewrite': [],
            }
        }
    }]
    turn.index.sides = {
        'ref': {'model': {'sha256': 'm', 'vocabulary': {'table_sha256': 'v'}}}
    }
    turn._labels = {}
    turn._pieces = {'v': (b'a', b'b', '▁c'.encode())}.get
    labels = token_analysis._Turn.labels(turn, ['ref'])
    self.assertEqual(labels.get(1), 'chat')
    self.assertEqual(labels.get(2), ' c')
    self.assertIsNone(labels.get(3))
    turn._pieces = None
    self.assertIsNone(token_analysis._Turn.labels(turn, ['ref']).get(2))

  def test_sampling_rows_are_both_sides_top_candidates(self):
    """The union of each side's most likely ids, with both sides' values."""

    class Turn:

      def labels(self, sides):
        del sides
        return {}

    ref_logits = np.array([5.0, 4.0, 3.0, 0.0, 0.0, 0.0])
    target_logits = np.array([0.0, 0.0, 0.0, 3.0, 4.0, 5.0])
    sources, distributions = {}, {}
    for side, logits in (('ref', ref_logits), ('target', target_logits)):
      sources[side] = token_analysis._Logits(logits, 'v', {})
      distributions[side] = metrics.distribution(logits)
    tokens = {
        side: token_analysis._Token(side, identity, {}, None, 'm')
        for side, identity in (('ref', 0), ('target', 5))
    }
    with mock.patch.object(token_analysis, 'CANDIDATES', 2):
      union = token_analysis._candidate_rows(
          Turn(), tokens, sources, distributions, True, union=True
      )
      first = token_analysis._candidate_rows(
          Turn(), tokens, sources, distributions, True
      )
    # The sampled ids, then each side's top 2 that is not sampled.
    self.assertEqual([row['id'] for row in union], [0, 5, 1, 4])
    self.assertEqual([row['id'] for row in first], [0, 5])
    for row in union:
      for side in ('ref', 'target'):
        np.testing.assert_allclose(
            row[side]['probability'],
            distributions[side].p[row['id']],
        )

  @parameterized.named_parameters(
      ('none', None),
      ('zero', 0),
      ('negative', -1.0),
  )
  def test_sampling_temperature_is_one_unless_positive(self, value):
    """A temperature that divides nothing is 1."""
    self.assertEqual(
        token_analysis.sampling_temperature({'temperature': value}), 1.0
    )

  def test_sampling_distribution_ranks_a_sampled_id_below_the_argmax(self):
    """A sampled id other than the argmax keeps its own rank."""
    # The target released id 5 although its logits' argmax is 4.
    turn = _same_turn(self.tmp_path, outputs=[5, 1, 0])
    index = _index(self.tmp_path, [turn])
    (pair,) = _analyze(index, 1, [(0, 0)])
    target = pair['sampling_distribution']['target']
    self.assertEqual(target['basis'], token_analysis.SAMPLING_BASIS)
    self.assertEqual(target['selected_id'], 5)
    self.assertEqual(target['selected_rank'], 2)
    self.assertEqual(target['reasons'], [])

  @parameterized.named_parameters(
      (name, name) for name in fixtures.greedy_edits()
  )
  def test_sampling_reads_the_logits_softmax_rule(self, name):
    """Logits a softmax accepts have a distribution; others say why."""
    index = _edited_capture(self.tmp_path, fixtures.greedy_edits()[name])

    (pair,) = _analyze(index, 1, [(0, 0)])

    ref = pair['sampling_distribution']['ref']
    if name in ('tie', 'masked', 'masked_elsewhere'):
      self.assertEqual(ref['basis'], token_analysis.SAMPLING_BASIS)
      self.assertEqual(ref['reasons'], [])
      self.assertEqual(ref['selected_id'], fixtures.token('a')[0])
    else:
      self.assertIsNone(ref['basis'])
      self.assertEqual(
          ref['reasons'],
          [_why(token_analysis.LOGITS_INVALID, 'ref')],
      )
    self.assertEqual(
        pair['sampling_distribution']['target']['basis'],
        token_analysis.SAMPLING_BASIS,
    )

  def test_perturbed_target_by_hand(self):
    """One changed logit, every metric by hand."""
    target_logits = [list(LOGITS[0]), *LOGITS[1:]]
    target_logits[0][5] = 2.5
    index = _index(self.tmp_path, [_same_turn(self.tmp_path, target_logits)])
    (pair,) = _analyze(index, 1, [(0, 0)])
    p, q = _softmax(LOGITS[0]), _softmax(target_logits[0])
    values = pair['metrics']
    np.testing.assert_allclose(
        values['kl']['value'], np.sum(p * np.log(p / q)), rtol=1e-6
    )
    m = (p + q) / 2
    js = 0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m))
    np.testing.assert_allclose(values['js']['value'], js, rtol=1e-6)
    # Activations are the logits' first four values: unchanged.
    self.assertEqual(values['relative_l2']['value'], 0.0)
    raw = pair['final_logits']['raw']
    self.assertEqual(raw['max_abs'], {'value': 0.5, 'reasons': []})
    self.assertEqual(raw['max_abs_index'], 5)
    np.testing.assert_allclose(
        raw['rmse']['value'], math.sqrt(0.25 / 6), rtol=1e-6
    )
    centered = pair['final_logits']['centered']
    # Centering subtracts each side's mean: 5.5/6 and 6/6.
    delta = np.array(target_logits[0]) - 1.0 - (np.array(LOGITS[0]) - 5.5 / 6)
    np.testing.assert_allclose(
        centered['rmse']['value'], math.sqrt(np.mean(delta**2)), rtol=1e-6
    )
    self.assertEqual(centered['max_abs_index'], 5)
    row = next(row for row in pair['distribution']['rows'] if row['id'] == 5)
    np.testing.assert_allclose(row['delta'], (q[5] - p[5]) * 100, rtol=1e-6)
    self.assertEqual(row['target']['rank'], 2)

  def test_activation_metrics_by_hand(self):
    """Activation metrics of two orthogonal vectors, by hand."""
    rest = [[1.0] * 4, [1.0] * 4]
    turn = {
        'ref': _write_side_turn(
            self.tmp_path,
            'ref',
            1,
            PROMPT,
            LOGITS,
            activations=[[0, 4, 0, 0], *rest],
        ),
        'target': _write_side_turn(
            self.tmp_path,
            'target',
            1,
            PROMPT,
            LOGITS,
            activations=[[2, 0, 0, 0], *rest],
        ),
    }
    index = _index(self.tmp_path, [turn])
    (pair,) = _analyze(index, 1, [(0, 0)])
    values = pair['metrics']
    # ref [0, 4], target [2, 0]: |t - r| / |r| = sqrt(20) / 4, in percent.
    np.testing.assert_allclose(
        values['relative_l2']['value'], 100 * math.sqrt(20) / 4, rtol=1e-6
    )
    np.testing.assert_allclose(values['norm_ratio']['value'], 0.5, rtol=1e-6)
    np.testing.assert_allclose(
        values['cosine_distance']['value'], 1.0, rtol=1e-6
    )
    self.assertEqual(values['cosine_similarity']['value'], 0.0)
    np.testing.assert_allclose(values['max_abs']['value'], 4.0, rtol=1e-6)

  def test_diverged_generations_still_compare(self):
    """After the sides generate differently, pairs compare and say so."""
    target_logits = [list(LOGITS[0]), *LOGITS[1:]]
    target_logits[0][5] = 5.0  # The target generates 5 where ref made 4.
    index = _index(self.tmp_path, [_same_turn(self.tmp_path, target_logits)])
    pairs = _analyze(index, 1, [(0, 0), (1, 1)])
    # The first tokens follow the same context.
    self.assertIs(pairs[0]['distribution']['compatible'], True)
    self.assertEqual(pairs[0]['distribution']['context'], 'same')
    self.assertEqual(pairs[0]['distribution']['notes'], [])
    diverged = pairs[1]
    distribution = diverged['distribution']
    self.assertIs(distribution['compatible'], True)
    self.assertEqual(distribution['context'], 'different')
    self.assertEqual(distribution['reasons'], [])
    self.assertEqual(
        distribution['notes'],
        [
            metrics.reason(
                token_analysis.CONTEXT_DIFFERENT,
                'After the divergence at position 3 the two sides read '
                'different tokens; the logits still compare.',
            )
        ],
    )
    for key in ('kl', 'js', 'cosine_similarity'):
      self.assertIsNotNone(diverged['metrics'][key]['value'], key)
    self.assertIsNotNone(diverged['final_logits']['raw']['rmse']['value'])
    self.assertTrue(
        all(row['delta'] is not None for row in distribution['rows'])
    )
    self.assertEqual(diverged['sampling_distribution']['context'], 'different')
    # The inspection of its whole vectors compares them, and says so.
    page = token_analysis.final_logits(
        index, 1, 1, 1, 'raw', 0, 2, 'index', 'ascending'
    )
    self.assertIs(page['compatible'], True)
    self.assertEqual(page['notes'], distribution['notes'])
    self.assertIsNotNone(page['rows'][0]['delta'])

  def test_different_models_have_no_shared_vocabulary(self):
    """Different model files share no vocabulary."""
    index = _index(
        self.tmp_path,
        [_same_turn(self.tmp_path)],
        target=_fixed(model='b' * 64),
    )
    (pair,) = _analyze(index, 1, [(0, 0)])
    distribution = pair['distribution']
    self.assertIs(distribution['compatible'], False)
    vocabulary = [NO_TOKENIZER]
    self.assertEqual(distribution['reasons'], vocabulary)
    self.assertEqual(distribution['context'], 'same')
    self.assertEqual(distribution['rows'], [])
    self.assertEqual(pair['metrics']['kl']['reasons'], vocabulary)
    self.assertEqual(
        pair['metrics']['relative_l2']['reasons'],
        [_why(token_analysis.ACTIVATION_SPACE_MISMATCH)],
    )
    sampling = pair['sampling_distribution']
    self.assertIs(sampling['vocabulary_compatible'], False)
    self.assertEqual(sampling['rows'], [])
    self.assertIs(sampling['rows_complete'], False)
    self.assertEqual(sampling['reasons'], vocabulary)

  def test_one_tokenizer_in_two_models_is_one_vocabulary(self):
    """Two models whose tokenizers have one table compare their logits."""
    table = 't' * 64
    index = _index(
        self.tmp_path,
        [_same_turn(self.tmp_path)],
        ref=_tokenized('a' * 64, table),
        target=_tokenized('b' * 64, table),
    )
    (pair,) = _analyze(index, 1, [(0, 0)])
    distribution = pair['distribution']
    self.assertIs(distribution['compatible'], True)
    self.assertEqual(distribution['reasons'], [])
    self.assertEqual(pair['metrics']['kl'], {'value': 0.0, 'reasons': []})
    self.assertEqual(
        pair['final_logits']['raw']['rmse'], {'value': 0.0, 'reasons': []}
    )
    for side in ('ref', 'target'):
      self.assertEqual(
          distribution[side]['logits_metadata']['vocab_identity'],
          f'tokenizer:{table}:{len(LOGITS[0])}',
      )
    self.assertTrue(all(row['delta'] == 0.0 for row in distribution['rows']))
    # The hidden states are still each model's own.
    self.assertEqual(
        pair['metrics']['relative_l2']['reasons'],
        [_why(token_analysis.ACTIVATION_SPACE_MISMATCH)],
    )
    self.assertIs(pair['sampling_distribution']['vocabulary_compatible'], True)
    response = _final_logits(index, order='delta')
    self.assertIs(response['compatible'], True)
    self.assertTrue(all(row['delta'] == 0.0 for row in response['rows']))

  def test_two_tokenizers_are_two_vocabularies(self):
    """Tokenizers whose tables differ say so."""
    index = _index(
        self.tmp_path,
        [_same_turn(self.tmp_path)],
        ref=_tokenized('a' * 64, 't' * 64),
        target=_tokenized('b' * 64, 'u' * 64),
    )
    (pair,) = _analyze(index, 1, [(0, 0)])
    self.assertIs(pair['distribution']['compatible'], False)
    self.assertEqual(
        pair['distribution']['reasons'],
        [
            _vocabulary_why(
                'Reference and Target ran different models whose '
                'tokenizers map ids to different pieces'
            )
        ],
    )
    self.assertEqual(
        pair['distribution']['target']['logits_metadata']['vocab_identity'],
        f'tokenizer:{"u" * 64}:{len(LOGITS[0])}',
    )

  def test_a_side_without_a_tokenizer_is_named(self):
    """The words name the side whose capture records no tokenizer."""
    index = _index(
        self.tmp_path,
        [_same_turn(self.tmp_path)],
        ref=_tokenized('a' * 64, 't' * 64),
        target=_tokenized('b' * 64, None),
    )
    (pair,) = _analyze(index, 1, [(0, 0)])
    self.assertEqual(
        pair['distribution']['reasons'],
        [
            _vocabulary_why(
                'Reference and Target ran different models, and the '
                'capture records no tokenizer of Target'
            )
        ],
    )

  def test_one_tokenizer_with_logits_of_two_lengths_says_so(self):
    """One tokenizer, but logits of different lengths: no comparison."""
    longer = [row + [-9.0] for row in LOGITS]
    index = _index(
        self.tmp_path,
        [_same_turn(self.tmp_path, target_logits=longer)],
        ref=_tokenized('a' * 64, 't' * 64),
        target=_tokenized('b' * 64, 't' * 64),
    )
    (pair,) = _analyze(index, 1, [(0, 0)])
    size = len(LOGITS[0])
    self.assertEqual(
        pair['distribution']['reasons'],
        [
            _vocabulary_why(
                'Reference and Target share a tokenizer, but their logits '
                f'hold {size} and {size + 1} values'
            )
        ],
    )

  def test_one_model_file_keeps_its_identity(self):
    """One model file: its SHA-256 is the identity, tokenizer or not."""
    index = _index(
        self.tmp_path,
        [_same_turn(self.tmp_path)],
        ref=_tokenized(MODEL, 't' * 64),
        target=_tokenized(MODEL, 't' * 64),
    )
    (pair,) = _analyze(index, 1, [(0, 0)])
    self.assertEqual(
        pair['distribution']['ref']['logits_metadata']['vocab_identity'],
        f'{MODEL}:{len(LOGITS[0])}',
    )
    other = pathlib.Path(self.create_tempdir().full_path)
    plain = _index(other, [_same_turn(other)])
    self.assertEqual(_analyze(index, 1, [(0, 0)]), _analyze(plain, 1, [(0, 0)]))

  def test_one_side_missing(self):
    """A pair with one side shows that side alone."""
    index = _index(self.tmp_path, [_same_turn(self.tmp_path)])
    (pair,) = _analyze(index, 1, [(1, None)])
    self.assertEqual((pair['ref_step'], pair['target_step']), (1, None))
    distribution = pair['distribution']
    self.assertIs(distribution['compatible'], False)
    self.assertIsNone(distribution['context'])
    missing = [_why(token_analysis.TOKEN_NOT_CAPTURED, 'target')]
    self.assertEqual(distribution['reasons'], missing)
    self.assertEqual(distribution['target'], {'reasons': missing})
    self.assertEqual(distribution['ref']['selected_id'], 1)
    self.assertEqual([row['id'] for row in distribution['rows']][:2], [1, 4])
    self.assertTrue(all(row['target'] is None for row in distribution['rows']))
    self.assertEqual(
        pair['metrics']['relative_l2']['reasons'],
        [_why(token_analysis.FORWARDS_NOT_PAIRED)],
    )
    self.assertIsNone(pair['final_logits']['raw']['max_abs_index'])
    sampling = pair['sampling_distribution']
    self.assertIsNone(sampling['context'])
    self.assertEqual(sampling['ref']['basis'], token_analysis.SAMPLING_BASIS)
    self.assertEqual(sampling['target']['reasons'], missing)
    # A step past the side's output is not captured either.
    (pair,) = _analyze(index, 1, [(3, 0)])
    self.assertEqual(
        pair['distribution']['ref'],
        {'reasons': [_why(token_analysis.TOKEN_NOT_CAPTURED, 'ref')]},
    )

  def test_an_unreadable_file_is_a_reason_of_its_pair(self):
    """A logits file that cannot be opened: that pair's side says why."""
    index = _index(self.tmp_path, [_same_turn(self.tmp_path)])
    forward = index.turn(1)['sides']['target']['forwards'][2]
    index.path('target', 1, forward['tensors']['post_logits']).unlink()

    first, second = _analyze(index, 1, [(0, 0), (1, 1)])

    (why,) = second['distribution']['target']['reasons']
    self.assertEqual((why['code'], why['side']), (metrics.UNREADABLE, 'target'))
    self.assertIsNone(second['metrics']['kl']['value'])
    self.assertEqual(first['metrics']['kl']['value'], 0.0)

  def test_context_spans_earlier_turns(self):
    """The context of a token includes the earlier turns."""
    turn_1 = _same_turn(self.tmp_path)
    # Turn 2 feeds the same tokens on both sides; only its earlier turn says
    # whether the contexts match.
    carried = [(9, None, 'carried_stop'), (5, 'Hi', 'prompt')]
    turn_2 = {
        side: _write_side_turn(
            self.tmp_path, side, 2, carried, LOGITS[:1], first_position=6
        )
        for side in ('ref', 'target')
    }
    index = _index(self.tmp_path, [turn_1, turn_2])
    (pair,) = _analyze(index, 2, [(0, 0)])
    self.assertEqual(pair['distribution']['context'], 'same')
    self.assertEqual(pair['distribution']['ref']['source_forward_id'], 1)

    other = self.tmp_path / 'other'
    other.mkdir()
    diverged_1 = _same_turn(other, outputs=[4, 1, 2])
    turn_2 = {
        side: _write_side_turn(
            other, side, 2, carried, LOGITS[:1], first_position=6
        )
        for side in ('ref', 'target')
    }
    index = _index(other, [diverged_1, turn_2])
    (pair,) = _analyze(index, 2, [(0, 0)])
    self.assertEqual(pair['distribution']['context'], 'different')

  def test_a_rewrite_replaces_the_reply_in_later_contexts(self):
    """A turn's rewrite, not its reply, is the context of the next turn."""
    # The replies differ from their first token on, but both sides
    # rewrite them, from the first decode step's position, as the same
    # tokens.
    turn_1 = _same_turn(self.tmp_path, outputs=[0, 1, 0])
    for entry in turn_1.values():
      entry['rewrite'] = [
          {'id': id, 'text': None, 'position': 2 + i, 'forward': 0}
          for i, id in enumerate([3, 0, 1])
      ]
    carried = [(4, None, 'carried'), (5, 'Hi', 'prompt')]
    turn_2 = {
        side: _write_side_turn(
            self.tmp_path, side, 2, carried, LOGITS[:1], first_position=5
        )
        for side in ('ref', 'target')
    }
    index = _index(self.tmp_path, [turn_1, turn_2])

    (pair,) = _analyze(index, 2, [(0, 0)])

    self.assertEqual(pair['distribution']['context'], 'same')
    self.assertTrue(pair['distribution']['compatible'])

  def test_a_rewrite_is_no_context_of_its_own_turn(self):
    """A turn's own rewrite came after its decode steps."""
    turn_1 = {
        side: _write_side_turn(
            self.tmp_path,
            side,
            1,
            PROMPT,
            LOGITS,
            texts=TEXTS,
            rewrite=[3, id],
        )
        for side, id in (('ref', 0), ('target', 1))
    }
    index = _index(self.tmp_path, [turn_1])

    pairs = _analyze(index, 1, [(k, k) for k in range(3)])

    self.assertEqual(
        [pair['distribution']['context'] for pair in pairs], ['same'] * 3
    )

  def test_a_rewrite_the_runner_did_not_name_leaves_the_context_unknown(
      self,
  ):
    """An unnamed rewrite id leaves the next turn's context unproven."""
    turn_1 = _same_turn(self.tmp_path)
    for entry in turn_1.values():
      entry['rewrite'] = [
          {'id': id, 'text': None, 'position': 2 + i, 'forward': 0}
          for i, id in enumerate([3, None])
      ]
    carried = [(4, None, 'carried'), (5, 'Hi', 'prompt')]
    turn_2 = {
        side: _write_side_turn(
            self.tmp_path, side, 2, carried, LOGITS[:1], first_position=4
        )
        for side in ('ref', 'target')
    }
    index = _index(self.tmp_path, [turn_1, turn_2])

    (pair,) = _analyze(index, 2, [(0, 0)])

    self.assertEqual(pair['distribution']['context'], 'unknown')
    self.assertTrue(pair['distribution']['compatible'])
    self.assertEqual(
        pair['distribution']['notes'],
        [
            metrics.reason(
                token_analysis.CONTEXT_UNKNOWN,
                'The runner did not name the token at position 3 on a '
                'side, so from there on the two sides may read different '
                'tokens; the logits still compare.',
            )
        ],
    )

  def test_a_chat_that_thinks_compares_every_turn(self):
    """Two backends of one chat that thinks have one context each turn."""
    index = fixtures.capture(
        self.tmp_path,
        [
            ('Hi', ['ok'], True, ['hmm']),
            ('More', ['b'], True, ['so']),
            ('Again', ['c'], True),
        ],
        max_completion_tokens=12,
    )

    for n in (1, 2, 3):
      output = index.turn(n)['sides']['ref']['output']
      pairs = _analyze(index, n, [(k, k) for k in range(len(output))])
      self.assertEqual(
          {pair['distribution']['context'] for pair in pairs}, {'same'}
      )
      self.assertTrue(
          all(pair['metrics']['kl']['value'] is not None for pair in pairs)
      )

  def test_tokens_the_runner_did_not_name_leave_the_context_unknown(self):
    """Two unknown ids at one position could be the same token, or not."""
    history = [(None, None, 'history'), (5, 'Hi', 'prompt')]
    turn_2 = {
        side: _write_side_turn(
            self.tmp_path, side, 2, history, LOGITS[:1], first_position=6
        )
        for side in ('ref', 'target')
    }
    index = _index(self.tmp_path, [_same_turn(self.tmp_path), turn_2])
    (pair,) = _analyze(index, 2, [(0, 0)])
    self.assertEqual(pair['distribution']['context'], 'unknown')
    self.assertEqual(pair['distribution']['reasons'], [])
    (note,) = pair['distribution']['notes']
    self.assertEqual(note['code'], token_analysis.CONTEXT_UNKNOWN)
    self.assertIn('at position 6', note['text'])
    self.assertEqual(pair['sampling_distribution']['context'], 'unknown')
    for key in ('kl', 'js', 'cosine_similarity'):
      self.assertIsNotNone(pair['metrics'][key]['value'], key)

  def test_divergence_is_where_the_fed_tokens_first_differ(self):
    """The first position of two ids, or of one side alone; else none."""

    def context(positions, ids):
      return token_analysis.Context(
          np.array(positions, dtype=np.int64),
          np.array(ids, dtype=np.int64),
      )

    unnamed = token_analysis.UNNAMED
    same = context([0, 1, 2], [7, 8, 9])
    self.assertIsNone(token_analysis.divergence(same, same))
    self.assertEqual(
        token_analysis.divergence(same, context([0, 1, 2], [7, 5, 9])),
        (1, token_analysis.DIFFERENT),
    )
    # One side fed more: the first position only it holds.
    self.assertEqual(
        token_analysis.divergence(same, context([0, 1], [7, 8])),
        (2, token_analysis.DIFFERENT),
    )
    self.assertEqual(
        token_analysis.divergence(same, context([0, 2, 3], [7, 9, 9])),
        (1, token_analysis.DIFFERENT),
    )
    # A token not named is unknown, unless two named ids differ anyway.
    self.assertEqual(
        token_analysis.divergence(same, context([0, 1, 2], [7, unnamed, 9])),
        (1, token_analysis.UNKNOWN),
    )
    self.assertEqual(
        token_analysis.divergence(same, context([0, 1, 2], [7, unnamed, 4])),
        (2, token_analysis.DIFFERENT),
    )
    self.assertIsNone(token_analysis.context_reason(None, 'logits'))
    self.assertEqual(
        token_analysis.context_reason(
            (17, token_analysis.DIFFERENT), 'K and V'
        ),
        metrics.reason(
            token_analysis.CONTEXT_DIFFERENT,
            'After the divergence at position 17 the two sides read '
            'different tokens; the K and V still compare.',
        ),
    )

  def test_unknown_turn_is_not_found(self):
    """A turn that was not captured is not found."""
    index = _index(self.tmp_path, [_same_turn(self.tmp_path)])
    with self.assertRaises(errors.NotFound):
      _analyze(index, 2, [])

  def test_final_logits_pages_and_sorts(self):
    """Final logits are sorted over the whole vocabulary, then paged."""
    target_logits = [list(LOGITS[0]), *LOGITS[1:]]
    target_logits[0][2] = 3.5
    index = _index(self.tmp_path, [_same_turn(self.tmp_path, target_logits)])
    response = _final_logits(index, offset=1, limit=2)
    self.assertEqual(
        {
            key: response[key]
            for key in (
                'ref_step',
                'target_step',
                'transform',
                'offset',
                'limit',
                'total',
                'compatible',
                'reasons',
            )
        },
        {
            'ref_step': 0,
            'target_step': 0,
            'transform': 'raw',
            'offset': 1,
            'limit': 2,
            'total': 6,
            'compatible': True,
            'reasons': [],
        },
    )
    self.assertEqual(
        response['rows'],
        [
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
        ],
    )
    self.assertEqual(
        response['metrics']['max_abs'], {'value': 3.0, 'reasons': []}
    )
    self.assertEqual(response['metrics']['max_abs_index'], 2)
    self.assertEqual(
        response['ref']['logits_metadata']['comparison_shape'], [6]
    )

    def indices(**fields):
      return [row['index'] for row in _final_logits(index, **fields)['rows']]

    self.assertEqual(indices(direction='descending', limit=3), [5, 4, 3])
    self.assertEqual(
        indices(order='ref', direction='descending'), [4, 5, 1, 2, 0, 3]
    )
    self.assertEqual(indices(order='target'), [3, 0, 1, 5, 4, 2])
    self.assertEqual(
        indices(order='absolute_delta', direction='descending')[:2], [2, 0]
    )
    # Equal values keep index order in either direction.
    self.assertEqual(
        indices(order='delta', direction='descending'), [2, 0, 1, 3, 4, 5]
    )
    self.assertEqual(
        indices(order='delta', direction='ascending'), [0, 1, 3, 4, 5, 2]
    )
    self.assertEqual(_final_logits(index, offset=6)['rows'], [])

  def test_final_logits_puts_non_finite_values_last(self):
    """Non-finite values sort last and are spelled out."""
    target_logits = [list(LOGITS[0]), *LOGITS[1:]]
    target_logits[0][1] = -math.inf
    index = _index(self.tmp_path, [_same_turn(self.tmp_path, target_logits)])
    response = _final_logits(index, order='target', direction='ascending')
    self.assertEqual(
        [row['index'] for row in response['rows']], [3, 0, 2, 5, 4, 1]
    )
    self.assertEqual(response['rows'][-1]['target'], '-Infinity')
    self.assertEqual(response['rows'][-1]['delta'], '-Infinity')
    self.assertEqual(response['rows'][-1]['absolute_delta'], 'Infinity')
    self.assertEqual(
        response['metrics']['rmse']['reasons'],
        [
            metrics.reason(
                metrics.NON_FINITE,
                'Logits contain non-finite values; full-vector metrics '
                'are unavailable',
            )
        ],
    )
    centered = _final_logits(index, transform='centered')
    self.assertIs(centered['compatible'], False)
    not_centered = [
        metrics.reason(
            metrics.NON_FINITE,
            'Centered logits unavailable: source contains non-finite values',
            'target',
        )
    ]
    self.assertEqual(centered['target']['reasons'], not_centered)
    self.assertEqual(centered['reasons'], not_centered)
    self.assertIsNone(centered['rows'][0]['target'])
    self.assertIsNone(centered['rows'][0]['delta'])

  def test_final_logits_without_a_comparable_pair(self):
    """Without a shared vocabulary there are no labels or deltas."""
    index = _index(
        self.tmp_path,
        [_same_turn(self.tmp_path)],
        target=_fixed(model='b' * 64),
    )
    response = _final_logits(index, order='delta')
    self.assertIs(response['compatible'], False)
    self.assertEqual(response['reasons'], [NO_TOKENIZER])
    # Different vocabularies share no labels and no deltas.
    self.assertEqual([row['index'] for row in response['rows']], list(range(6)))
    self.assertTrue(all(row['label'] is None for row in response['rows']))
    self.assertTrue(all(row['delta'] is None for row in response['rows']))
    single = _final_logits(index, target_step=None)
    missing = [_why(token_analysis.TOKEN_NOT_CAPTURED, 'target')]
    self.assertEqual(single['reasons'], missing)
    self.assertEqual(single['rows'][4]['label'], 'four')
    self.assertIsNone(single['rows'][4]['target'])
    self.assertEqual(single['target'], {'reasons': missing})

  def test_final_logits_offset_past_the_vocabulary(self):
    """An offset past the vocabulary is refused."""
    index = _index(self.tmp_path, [_same_turn(self.tmp_path)])
    with self.assertRaisesRegex(ValueError, 'offset exceeds'):
      _final_logits(index, offset=7)

  def test_final_logits_summary(self):
    """What is known of the vectors, echoed request, and no values."""
    target_logits = [list(LOGITS[0]), *LOGITS[1:]]
    target_logits[0][1] = math.nan
    index = _index(self.tmp_path, [_same_turn(self.tmp_path, target_logits)])
    response = token_analysis.final_logits_summary(index, 1, 0, 0, 'raw')
    self.assertEqual(
        {key: response[key] for key in ('ref_step', 'target_step')},
        {'ref_step': 0, 'target_step': 0},
    )
    self.assertEqual(response['transform'], 'raw')
    self.assertEqual(response['total'], 6)
    self.assertIs(response['compatible'], True)
    for key in ('values', 'rows', 'offset', 'complete'):
      self.assertNotIn(key, response)
    self.assertEqual(response['ref']['size'], 6)
    single = token_analysis.final_logits_summary(index, 1, None, 0, 'raw')
    self.assertIs(single['compatible'], False)
    self.assertEqual(single['total'], 6)

  def test_final_logits_bins(self):
    """The line chart's bins of a range and of the whole vocabulary."""
    target_logits = [list(LOGITS[0]), *LOGITS[1:]]
    target_logits[0][1] = math.nan
    index = _index(self.tmp_path, [_same_turn(self.tmp_path, target_logits)])

    response = token_analysis.final_logits_bins(
        index, 1, 0, 0, 'raw', (1, 5), 2, 'max'
    )

    self.assertEqual(response['range'], {'start': 1, 'end': 5})
    self.assertEqual((response['capacity'], response['reduction']), (2, 'max'))
    # Ref [1.0, 0.5 | -1.0, 3.0]; the target's NaN makes its first bin
    # a gap.
    self.assertEqual(
        response['visible'],
        {
            'start': 1,
            'end': 5,
            'size': 2,
            'ref': [1.0, 3.0],
            'ref_index': [1, 4],
            'target': [None, 3.0],
            'target_index': [None, 4],
        },
    )
    self.assertEqual(
        (response['overview']['start'], response['overview']['end']),
        (0, 6),
    )
    self.assertEqual(response['overview']['ref'], [1.0, 3.0])
    whole = token_analysis.final_logits_bins(
        index, 1, None, 0, 'raw', None, 6, 'mean'
    )
    self.assertEqual(whole['range'], {'start': 0, 'end': 6})
    self.assertEqual(whole['visible']['ref'], [None] * 6)
    self.assertEqual(whole['visible']['target'][0], 0.0)
    # A range past the vocabulary is kept inside it, as the UI keeps it.
    kept = token_analysis.final_logits_bins(
        index, 1, 0, 0, 'raw', (9, 12), 4, 'min'
    )
    self.assertEqual(kept['range'], {'start': 5, 'end': 6})

  def test_final_logits_points(self):
    """The scatter plot's points: only compared, paired elements."""
    target_logits = [list(LOGITS[0]), *LOGITS[1:]]
    target_logits[0][1] = math.nan
    index = _index(self.tmp_path, [_same_turn(self.tmp_path, target_logits)])

    response = token_analysis.final_logits_points(index, 1, 0, 0, 'raw', None)

    # Paired values span -1..3: padded by 5% of 4.
    self.assertEqual(response['extent'], {'start': -1.2, 'end': 3.2})
    self.assertEqual(response['range'], response['extent'])
    self.assertEqual(response['paired_runs'], [[0, 1], [2, 6]])
    self.assertEqual(
        (response['paired'], response['visible'], response['grouped']),
        (5, 5, False),
    )
    self.assertEqual(response['points']['index'], [0, 2, 3, 4, 5])
    self.assertEqual(response['points']['ref'], [0.0, 0.5, -1.0, 3.0, 2.0])
    zoomed = token_analysis.final_logits_points(
        index, 1, 0, 0, 'raw', (0.0, 1.0)
    )
    self.assertEqual(zoomed['range'], {'start': 0.0, 'end': 1.0})
    self.assertEqual(zoomed['visible'], 2)
    # The extent and the paired runs come with the whole extent alone.
    self.assertNotIn('extent', zoomed)
    self.assertNotIn('paired_runs', zoomed)
    # Few points are all drawn, inside the range or not.
    self.assertEqual(zoomed['points']['index'], [0, 2, 3, 4, 5])
    single = token_analysis.final_logits_points(index, 1, None, 0, 'raw', None)
    self.assertEqual(single['extent'], {'start': -1.0, 'end': 1.0})
    self.assertEqual(single['points']['index'], [])
    self.assertEqual(single['paired_runs'], [])

  def test_final_logits_summary_statistics(self):
    """Each vector's statistics, and their difference's when compared."""
    target_logits = [[0.0, 1.5, 0.5, -1.0, 2.0, 2.0], *LOGITS[1:]]
    index = _index(self.tmp_path, [_same_turn(self.tmp_path, target_logits)])

    statistics = token_analysis.final_logits_summary(index, 1, 0, 0, 'raw')[
        'statistics'
    ]

    ref = statistics['ref']
    self.assertEqual(
        (ref['min'], ref['min_position'], ref['max'], ref['max_position']),
        (-1.0, 3, 3.0, 4),
    )
    np.testing.assert_allclose(ref['mean'], 5.5 / 6)
    self.assertEqual(ref['valid_count'], 6)
    # Delta [0, 0.5, 0, 0, -1, 0].
    delta = statistics['delta']
    self.assertEqual((delta['min'], delta['min_position']), (-1.0, 4))
    self.assertEqual((delta['max'], delta['max_position']), (0.5, 1))
    np.testing.assert_allclose(delta['mean'], -0.5 / 6)
    absolute = statistics['abs_delta']
    self.assertEqual((absolute['max'], absolute['max_position']), (1.0, 4))
    self.assertEqual((absolute['min'], absolute['min_position']), (0.0, 0))
    single = token_analysis.final_logits_summary(index, 1, None, 0, 'raw')
    self.assertEqual(single['statistics']['ref']['valid_count'], 0)
    self.assertEqual(single['statistics']['target']['max'], 2.0)
    self.assertEqual(single['statistics']['delta']['valid_count'], 0)

  def test_final_logits_summary_statistics_leave_out_non_finite_values(
      self,
  ):
    """A NaN is neither counted nor compared."""
    target_logits = [list(LOGITS[0]), *LOGITS[1:]]
    target_logits[0][1] = math.nan
    index = _index(self.tmp_path, [_same_turn(self.tmp_path, target_logits)])

    statistics = token_analysis.final_logits_summary(index, 1, 0, 0, 'raw')[
        'statistics'
    ]

    self.assertEqual(statistics['target']['valid_count'], 5)
    self.assertEqual(statistics['delta']['valid_count'], 5)
    self.assertEqual(statistics['abs_delta']['max'], 0.0)

  def _range_rows(self, index, present=None):
    """The stage's rows (0, 0), (1, 1), (2, 2) as token-range reads them."""
    steps = [(0, 0), (1, 1), (2, 2)]
    present = present or [(True, True)] * 3
    pairs = _analyze(index, 1, steps)
    return [
        token_analysis.RangeRow(
            ref=ref,
            target=target,
            steps=key,
            values=token_analysis.range_values(pair),
        )
        for (ref, target), key, pair in zip(present, steps, pairs)
    ]

  def test_token_range_bins_the_pair_metric(self):
    """Final Logits: one series of the pair's metric, binned by column."""
    target_logits = [LOGITS[0], [0.2, 0.5, 0.1, 0.0, 1.0, -0.5], LOGITS[2]]
    index = _index(self.tmp_path, [_same_turn(self.tmp_path, target_logits)])
    rows = self._range_rows(index)
    kl = [
        None if np.isnan(value) else value
        for value in (
            row.values.numbers[token_analysis.RANGE_NUMBERS.index('kl')]
            for row in rows
        )
    ]

    answer = token_analysis.token_range(
        10, rows, None, 'final_logits', 'kl', 10, 13, 2
    )

    self.assertEqual(
        answer['bins'], [{'start': 10, 'end': 11}, {'start': 11, 'end': 13}]
    )
    (series,) = answer['rows']
    self.assertIsNone(series['side'])
    first, rest = series['bins']
    self.assertEqual((first['start'], first['end']), (10, 11))
    # Step 1's tokens differ, so step 2's contexts do; its logits still
    # compare, and are the same.
    self.assertEqual(kl[2], 0.0)
    self.assertEqual(first['max'], kl[0])
    self.assertEqual(rest['max'], kl[1])
    self.assertEqual(rest['max_position'], 11)
    self.assertEqual(rest['valid_count'], 2)
    self.assertAlmostEqual(rest['mean'], kl[1] / 2)
    self.assertEqual(answer['all'], series['bins'])
    self.assertEqual(answer['peak_kl'], {'value': kl[1], 'row': 1, 'step': 1})
    metadata = answer['metadata']
    self.assertEqual(
        metadata['ref'],
        {
            'count': 3,
            'vocabulary_sizes': [6],
            'dtypes': ['float32'],
            'vocabulary_identities': [f'{MODEL}:6'],
        },
    )
    self.assertEqual(
        metadata['output'],
        {'dimensions': ['Position'], 'shape': [3], 'dtype': 'float64'},
    )

  def test_token_range_peak_takes_the_first_divergence_pair(self):
    """The first divergence's pair stands for its row in the peak KL."""
    target_logits = [LOGITS[0], [0.2, 0.5, 0.1, 0.0, 1.0, -0.5], LOGITS[2]]
    index = _index(self.tmp_path, [_same_turn(self.tmp_path, target_logits)])
    rows = self._range_rows(index)
    fork = token_analysis.RangeRow(
        ref=True, target=True, steps=(0, 0), values=rows[0].values
    )

    answer = token_analysis.token_range(
        10, rows, (1, fork), 'final_logits', 'kl', 11, 12, 1
    )

    # Row 1's pair is replaced by (0, 0), whose KL is 0: the peak is
    # none of row 1's.
    self.assertEqual(answer['peak_kl']['row'], 0)
    self.assertEqual(answer['peak_kl']['value'], 0.0)
    self.assertEqual(answer['rows'][0]['bins'][0]['valid_count'], 1)

  def test_token_range_of_each_side(self):
    """Sampling: one series per side, at the rows with its token."""
    index = _index(self.tmp_path, [_same_turn(self.tmp_path)])
    rows = self._range_rows(index, [(True, True), (True, False), (True, True)])

    answer = token_analysis.token_range(
        0, rows, None, 'sampling', 'selected_probability', 0, 3, 1
    )

    ref, target = answer['rows']
    self.assertEqual((ref['side'], target['side']), ('ref', 'target'))
    self.assertEqual(ref['bins'][0]['valid_count'], 3)
    self.assertEqual(target['bins'][0]['valid_count'], 2)
    self.assertEqual(answer['all'][0]['valid_count'], 5)
    # Each step samples its logits' argmax: its softmax's largest value.
    top = [max(_softmax(logits)) for logits in LOGITS]
    np.testing.assert_allclose(
        answer['all'][0]['mean'],
        np.mean(top + [top[0], top[2]]),
        rtol=1e-6,
    )
    metadata = answer['metadata']
    self.assertEqual(
        metadata['target'],
        {
            'vocabulary_sizes': [6],
            'sampler': {
                'temperature': [None],
                'top_k': [1],
                'top_p': [None],
                'seed': [None],
            },
        },
    )
    self.assertEqual(metadata['output']['shape'], [2, 3])

  @absltest.skipUnless(real_capture.AVAILABLE, real_capture.UNAVAILABLE)
  def test_real_turn_against_itself_and_a_perturbed_copy(self):
    """The real turn: zero against itself, non-zero against noise."""
    real_turn = real_capture.turn('CPU').directory
    perturbed = self.tmp_path / 'perturbed'
    perturbed.mkdir()
    for path in real_turn.glob('decode_post_*_step_*.safetensors'):
      if 'kv_cache' in path.name:
        continue
      key = path.name.split('_step_')[0].removeprefix('decode_')
      data = safetensors_numpy.load_file(str(path))[key]
      # Small noise that keeps every argmax.
      noise = np.random.default_rng(0).normal(0, 0.01, data.shape)
      step = int(path.name.split('_step_')[1].split('.')[0])
      _save(
          perturbed / path.name,
          key,
          (data + noise).astype(np.float32),
          step,
      )

    same_root = self.tmp_path / 'same'
    same_root.mkdir()
    index = _index(
        same_root,
        [{
            side: _real_side_turn(same_root, side, real_turn)
            for side in ('ref', 'target')
        }],
    )
    steps = list(range(len(index.turn(1)['sides']['ref']['output'])))
    self.assertLen(steps, 8)
    pairs = _analyze(index, 1, [(step, step) for step in steps])
    for pair in pairs:
      for key in ('kl', 'js', 'relative_l2', 'max_abs'):
        self.assertEqual(
            pair['metrics'][key], {'value': 0.0, 'reasons': []}, key
        )
      self.assertEqual(pair['final_logits']['raw']['rmse']['value'], 0.0)
      distribution = pair['distribution']
      self.assertEqual(distribution['context'], 'same')
      self.assertEqual(distribution['ref'], distribution['target'])
      self.assertEqual(distribution['ref']['size'], 262144)
      self.assertEqual(distribution['ref']['selected_rank'], 1)
      self.assertEqual(
          distribution['ref']['logits_metadata']['source_shape'],
          [
              1,
              1,
              262144,
          ],
      )
      self.assertLen(distribution['rows'], 64)
      sampling = pair['sampling_distribution']
      self.assertEqual(sampling['ref']['basis'], token_analysis.SAMPLING_BASIS)
      self.assertEqual(sampling['ref'], sampling['target'])
      self.assertLen(sampling['rows'], 64)
    first = pairs[0]['distribution']['rows'][0]
    self.assertEqual((first['id'], first['label']), (236778, '2'))
    bins = token_analysis.final_logits_bins(
        index, 1, 7, 7, 'raw', None, 1000, 'max'
    )
    self.assertEqual(bins['total'], 262144)
    self.assertEqual(bins['visible']['ref'], bins['visible']['target'])
    self.assertLen(bins['visible']['ref'], 997)
    points = token_analysis.final_logits_points(index, 1, 7, 7, 'raw', None)
    self.assertEqual(points['paired_runs'], [[0, 262144]])
    self.assertTrue(points['grouped'])
    self.assertEqual(points['points']['ref'], points['points']['target'])

    perturbed_root = self.tmp_path / 'perturbed_index'
    perturbed_root.mkdir()
    index = _index(
        perturbed_root,
        [{
            'ref': _real_side_turn(perturbed_root, 'ref', real_turn),
            'target': _real_side_turn(perturbed_root, 'target', perturbed),
        }],
    )
    pairs = _analyze(index, 1, [(step, step) for step in steps])
    for pair in pairs:
      self.assertIs(pair['distribution']['compatible'], True)
      for key in (
          'kl',
          'js',
          'relative_l2',
          'cosine_distance',
          'max_abs',
      ):
        self.assertGreater(pair['metrics'][key]['value'], 0, key)
      raw = pair['final_logits']['raw']
      self.assertTrue(0 < raw['rmse']['value'] < raw['max_abs']['value'])
      self.assertEqual(
          pair['sampling_distribution']['target']['basis'],
          token_analysis.SAMPLING_BASIS,
      )
    page = token_analysis.final_logits(
        index, 1, 0, 0, 'centered', 0, 16, 'absolute_delta', 'descending'
    )
    self.assertEqual(page['total'], 262144)
    deltas = [row['absolute_delta'] for row in page['rows']]
    self.assertEqual(deltas, sorted(deltas, reverse=True))


def _fed(*tokens):
  """A Context of `(position, id)` tokens; an id None was not named."""
  return token_analysis.Context(
      np.array([position for position, _ in tokens], dtype=np.int64),
      np.array(
          [token_analysis.UNNAMED if id is None else id for _, id in tokens],
          dtype=np.int64,
      ),
  )


class SameContextTest(absltest.TestCase):
  """Whether two sides fed the same tokens, proven or not."""

  def test_the_same_named_tokens(self):
    """Every token named and equal."""
    self.assertIs(
        token_analysis.same_context(_fed((0, 2), (1, 5)), _fed((0, 2), (1, 5))),
        True,
    )

  def test_different_positions_differ_beside_an_unnamed_token(self):
    """One side fed a token more: whatever the unnamed one is."""
    self.assertIs(
        token_analysis.same_context(
            _fed((0, 2), (1, 5)), _fed((0, 2), (1, None), (2, 7))
        ),
        False,
    )

  def test_a_named_difference_differs_beside_an_unnamed_token(self):
    """Two named ids at one position differ: proof enough."""
    self.assertIs(
        token_analysis.same_context(
            _fed((0, 2), (1, 5)), _fed((0, 3), (1, None))
        ),
        False,
    )

  def test_only_an_unnamed_token_could_tell(self):
    """Equal where named; an unnamed token could be either."""
    self.assertIsNone(
        token_analysis.same_context(
            _fed((0, 2), (1, 5)), _fed((0, 2), (1, None))
        )
    )
    self.assertIsNone(
        token_analysis.same_context(
            _fed((0, None), (1, 5)), _fed((0, None), (1, 5))
        )
    )

  def test_nothing_fed_is_the_same(self):
    """Two empty contexts."""
    self.assertIs(token_analysis.same_context(_fed(), _fed()), True)


if __name__ == '__main__':
  test_main.main()
