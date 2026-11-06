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

"""Tests of how a turn's ref and target tokens line up."""

import pathlib
import random

from absl.testing import absltest
from model_debugger.server.analysis import alignment
from model_debugger.server.analysis import token_analysis
from model_debugger.server.capture import capture_index
from model_debugger.server.testing import capture_fixtures as fixtures
from model_debugger.server.testing import test_main


def _tokens(texts, phase=None):
  tokens = [{'text': text, 'step': step} for step, text in enumerate(texts)]
  if phase:
    for token in tokens:
      token['phase'] = phase
  return tokens


def _rows(rows):
  return [(row['ref'], row['target'], row['same']) for row in rows]


def _quadratic_rows(a, b):
  """The rows of one phase by the full table of suffix lengths.

  What the UI computed before the server did: the alignment `_SuffixLcs`
  must reproduce, row for row.

  Args:
    a: The ref side's tokens, compared by equality.
    b: The target side's tokens, likewise.

  Returns:
    The rows in order: `(i, j)` for a pair of equal tokens, else
    `(removed, added)` with None for a side that has no token in the row.
  """
  n, m = len(a), len(b)
  table = [[0] * (m + 1) for _ in range(n + 1)]
  for i in range(n - 1, -1, -1):
    for j in range(m - 1, -1, -1):
      table[i][j] = (
          1 + table[i + 1][j + 1]
          if a[i] == b[j]
          else max(table[i + 1][j], table[i][j + 1])
      )
  rows, removed, added = [], [], []

  def flush():
    for k in range(max(len(removed), len(added))):
      rows.append((
          removed[k] if k < len(removed) else None,
          added[k] if k < len(added) else None,
      ))
    removed.clear()
    added.clear()

  i = j = 0
  while i < n or j < m:
    if i < n and j < m and a[i] == b[j]:
      flush()
      rows.append((i, j))
      i, j = i + 1, j + 1
    elif i < n and (j == m or table[i + 1][j] >= table[i][j + 1]):
      removed.append(i)
      i += 1
    else:
      added.append(j)
      j += 1
  flush()
  return rows


class AlignTest(absltest.TestCase):
  """Rows, sameness, the first divergence and the conclusion."""

  def test_identical_sides_are_the_same_in_every_row(self):
    """Identical sides: one row per step, all the same."""
    tokens = _tokens(['Four', '.', '<turn|>'])

    for kind in alignment.ALIGNMENTS:
      compared = alignment.compare(tokens, list(tokens), kind)

      self.assertEqual(
          _rows(compared['rows']), [(k, k, True) for k in range(3)]
      )
      self.assertIsNone(compared['first_mismatch'])
      self.assertIsNone(compared['first_divergence'])
      self.assertEqual(
          compared['conclusion'],
          {'kind': alignment.SAME, 'step': None, 'prefix': 3},
      )

  def test_a_divergence_mid_way_is_the_first_differing_step(self):
    """Content keeps equal text together; the step pairs the fork."""
    ref, target = _tokens(['a', 'b', 'c']), _tokens(['a', 'x', 'b', 'c'])

    content = alignment.compare(ref, target, 'content')
    steps = alignment.compare(ref, target, 'steps')

    self.assertEqual(
        _rows(content['rows']),
        [(0, 0, True), (None, 1, False), (1, 2, True), (2, 3, True)],
    )
    self.assertEqual(content['first_mismatch'], 1)
    # Row 1 holds the target's step 1 alone; the ref's step 1, 'b', is
    # drawn in row 2.
    self.assertEqual(
        content['first_divergence'],
        {'row': 1, 'ref': 1, 'target': 1, 'filled': 'ref', 'partner': 2},
    )
    self.assertEqual(
        content['conclusion'],
        {'kind': alignment.DIVERGED, 'step': 1, 'prefix': 1},
    )
    self.assertEqual(
        _rows(steps['rows']),
        [(0, 0, True), (1, 1, False), (2, 2, False), (None, 3, False)],
    )
    self.assertEqual(
        steps['first_divergence'],
        {'row': 1, 'ref': 1, 'target': 1, 'filled': None, 'partner': None},
    )
    self.assertEqual(steps['conclusion']['kind'], alignment.DIVERGED)

  def test_thinking_and_answer_are_aligned_apart(self):
    """By content each phase on its own; by step a phase change differs."""
    ref = _tokens(['hm', 'ok', 'Yes'])
    target = _tokens(['hm', 'ok', 'Yes'])
    for token, phase in zip(ref, ['thinking', 'thinking', 'response']):
      token['phase'] = phase
    for token, phase in zip(target, ['thinking', 'response', 'response']):
      token['phase'] = phase

    content = alignment.align(ref, target, 'content')
    steps = alignment.align(ref, target, 'steps')

    # The target's answer 'ok' is not the ref's thought 'ok'.
    self.assertEqual(
        _rows(content),
        [(0, 0, True), (1, None, False), (None, 1, False), (2, 2, True)],
    )
    self.assertEqual(_rows(steps), [(0, 0, True), (1, 1, False), (2, 2, True)])

  def test_a_token_without_text_proves_nothing(self):
    """A missing text: unknown sameness, an unknown conclusion."""
    ref = _tokens(['a', None])
    target = _tokens(['a', '<turn|>'])

    for kind in alignment.ALIGNMENTS:
      compared = alignment.compare(ref, target, kind)

      self.assertIsNone(compared['rows'][-1]['same'])
      self.assertIsNone(compared['first_divergence'])
      self.assertIsNone(compared['first_mismatch'])
      self.assertEqual(
          compared['conclusion'],
          {'kind': alignment.UNKNOWN, 'step': None, 'prefix': 1},
      )
    # Two missing texts are not the same text either.
    self.assertEqual(
        _rows(alignment.align(_tokens([None]), _tokens([None]), 'content')),
        [(0, 0, None)],
    )

  def test_texts_are_compared_not_ids(self):
    """Ids are one tokenizer's: equal text is the same, equal ids not."""
    ref = [
        {'text': 'a', 'step': 0, 'id': 1},
        {'text': None, 'step': 1, 'id': 106},
    ]
    target = [
        {'text': 'a', 'step': 0, 'id': 2},
        {'text': None, 'step': 1, 'id': 106},
    ]

    self.assertEqual(
        _rows(alignment.align(ref, target, 'steps')),
        [(0, 0, True), (1, 1, None)],
    )

  def test_no_divergence_after_rows_of_different_steps(self):
    """Rows before the first difference must pair one step on both sides."""
    ref = _tokens(['a', 'c'])
    target = [{'text': 'a', 'step': 1}, {'text': 'b', 'step': 2}]

    compared = alignment.compare(ref, target, 'content')

    self.assertEqual(compared['first_mismatch'], 1)
    self.assertIsNone(compared['first_divergence'])
    self.assertEqual(compared['conclusion']['kind'], alignment.UNKNOWN)
    # Nor where the other side has no token of that step.
    self.assertIsNone(
        alignment.compare([], _tokens(['a']), 'content')['first_divergence']
    )

  def test_a_side_not_captured(self):
    """The other side's tokens each have a row, of unknown sameness."""
    tokens = _tokens(['a', 'b'])

    compared = alignment.compare(None, tokens, 'content')

    self.assertEqual(
        _rows(compared['rows']), [(None, 0, None), (None, 1, None)]
    )
    self.assertEqual(
        compared['conclusion'],
        {'kind': alignment.NOT_CAPTURED, 'step': None, 'prefix': 0},
    )
    self.assertEqual(
        alignment.compare_prefill(tokens, None)['conclusion'],
        {'kind': alignment.NOT_CAPTURED, 'row': None},
    )

  def test_a_large_alignment_is_aligned(self):
    """2000 differing tokens a side: no limit, every token in one row."""
    ref = _tokens([f'r{k % 50}' for k in range(2000)])
    target = _tokens([f'r{k % 47}' for k in range(2000)])

    rows = alignment.align(ref, target, 'content')

    for side in capture_index.SIDES:
      self.assertEqual(
          [row[side] for row in rows if row[side] is not None],
          list(range(2000)),
      )
    self.assertTrue(any(row['same'] for row in rows))

  def test_content_rows_are_the_longest_common_subsequence_walk(self):
    """Row for row the full table's alignment, on random sequences."""
    rng = random.Random(7)
    for _ in range(300):
      letters = rng.randint(1, 4)
      a = [chr(97 + rng.randrange(letters)) for _ in range(rng.randint(0, 20))]
      b = [chr(97 + rng.randrange(letters)) for _ in range(rng.randint(0, 20))]

      rows = alignment.align(_tokens(a), _tokens(b), 'content')

      # The equal start and end are rows of their own, as the full
      # table also pairs them.
      self.assertEqual(
          [(row['ref'], row['target']) for row in rows],
          _trimmed_quadratic_rows(a, b),
          (a, b),
      )


def _trimmed_quadratic_rows(a, b):
  prefix = 0
  while prefix < min(len(a), len(b)) and a[prefix] == b[prefix]:
    prefix += 1
  suffix = 0
  while (
      suffix < min(len(a), len(b)) - prefix and a[-1 - suffix] == b[-1 - suffix]
  ):
    suffix += 1
  rows = [(k, k) for k in range(prefix)]
  rows += [
      (
          None if one is None else prefix + one,
          None if other is None else prefix + other,
      )
      for one, other in _quadratic_rows(
          a[prefix : len(a) - suffix], b[prefix : len(b) - suffix]
      )
  ]
  rows += [(len(a) - suffix + k, len(b) - suffix + k) for k in range(suffix)]
  return rows


def _input(*parts):
  """Input tokens: (kind, text) pairs."""
  return [
      {'kind': kind, 'text': text, 'step': step}
      for step, (kind, text) in enumerate(parts)
  ]


_PROMPT = _input(
    ('special', '<bos>'),
    ('special', '<|turn>'),
    ('template', 'system'),
    ('template', '\n'),
    ('text', 'Be'),
    ('text', ' brief.'),
    ('special', '<turn|>'),
    ('template', '\n'),
    ('special', '<|turn>'),
    ('template', 'user'),
    ('template', '\n'),
    ('text', 'Hi'),
    ('special', '<turn|>'),
    ('template', '\n'),
    ('special', '<|turn>'),
    ('template', 'model'),
    ('template', '\n'),
)


class PrefillTest(absltest.TestCase):
  """The input's message rows, and whether both sides fed it alike."""

  def test_the_input_is_grouped_by_message(self):
    """System, then user; the reply's prompt closes the user's message."""
    self.assertEqual(
        alignment.messages(_PROMPT),
        [
            {
                'role': 'system',
                'leading': [0, 1, 2, 3],
                'text': [4, 5],
                'trailing': [6, 7],
            },
            {
                'role': 'user',
                'leading': [8, 9, 10],
                'text': [11],
                'trailing': [12, 13, 14, 15, 16],
            },
        ],
    )

  def test_one_token_openers_and_text_after_a_closed_message(self):
    """`<|turn>user` alone opens; text after a close is a message too."""
    tokens = _input(
        ('special', '<|turn>user'),
        ('text', 'Hi'),
        ('special', '<turn|>'),
        ('text', 'stray'),
    )

    self.assertEqual(
        [
            (row['role'], row['leading'], row['text'], row['trailing'])
            for row in alignment.messages(tokens)
        ],
        [('user', [0], [1], [2]), ('unknown', [], [3], [])],
    )

  def test_identical_inputs_are_the_same(self):
    """Same inputs: same, and the messages are over the rows."""
    compared = alignment.compare_prefill(_PROMPT, list(_PROMPT))

    self.assertEqual(
        compared['conclusion'], {'kind': alignment.SAME, 'row': None}
    )
    self.assertEqual(compared['messages'], alignment.messages(_PROMPT))

  def test_a_different_input_names_its_first_row(self):
    """A longer system message: the first differing row."""
    target = _PROMPT[:6] + _input(('text', ' now')) + _PROMPT[6:]
    target = [{**token, 'step': step} for step, token in enumerate(target)]

    compared = alignment.compare_prefill(_PROMPT, target)

    self.assertEqual(
        compared['conclusion'], {'kind': alignment.DIFFERENT, 'row': 6}
    )
    self.assertEqual(compared['messages'][0]['text'], [4, 5, 6])
    self.assertEqual(compared['messages'][0]['trailing'], [7, 8])

  def test_an_input_token_without_text_is_unknown(self):
    """An unnamed token of the input leaves it unknown."""
    target = [dict(token) for token in _PROMPT]
    target[0]['text'] = None

    compared = alignment.compare_prefill(_PROMPT, target)

    self.assertEqual(
        compared['conclusion'], {'kind': alignment.UNKNOWN, 'row': 0}
    )


class StageSummaryTest(absltest.TestCase):
  """Match rates and the stage of the first different row."""

  def test_rates_and_the_first_mismatch(self):
    """The first different stage is marked; an unknown row stops rates."""
    same = [{'ref': 0, 'target': 0, 'same': True}]
    different = same + [{'ref': 1, 'target': 1, 'same': False}]
    unknown = [{'ref': 0, 'target': 0, 'same': None}]

    summaries = alignment.stage_summaries(
        [(same, 1, 1), (different, 2, 2), (same, 1, 1), (unknown, 1, 1)]
    )

    self.assertEqual(
        summaries,
        [
            {
                'match_rate': 1.0,
                'cumulative_match_rate': 1.0,
                'first_mismatch': False,
            },
            {
                'match_rate': 0.5,
                'cumulative_match_rate': 2 / 3,
                'first_mismatch': True,
            },
            {
                'match_rate': 1.0,
                'cumulative_match_rate': 0.75,
                'first_mismatch': False,
            },
            {
                'match_rate': None,
                'cumulative_match_rate': None,
                'first_mismatch': False,
            },
        ],
    )

  def test_a_side_not_captured_has_no_rates(self):
    """Without a side's tokens nothing is proven."""
    rows = [{'ref': None, 'target': 0, 'same': None}]

    (summary,) = alignment.stage_summaries([(rows, None, 1)])

    self.assertEqual(
        summary,
        {
            'match_rate': None,
            'cumulative_match_rate': None,
            'first_mismatch': None,
        },
    )


class HistoryTest(absltest.TestCase):
  """Whether both sides had fed the same tokens before a stage."""

  def setUp(self):
    """An empty directory for the chat's capture."""
    super().setUp()
    self.root = pathlib.Path(self.create_tempdir().full_path)

  def _context(self, index, n):
    """Token analysis's context of the turn's first generated tokens."""
    (pair,) = token_analysis.token_analysis(index, n, [(0, 0)])
    return pair['distribution']['context']

  def test_the_same_history(self):
    """Nothing before the first turn; then the same tokens."""
    index = fixtures.capture(
        self.root, [('Hi', ['a', 'b'], True), ('More', ['c'], True)]
    )

    self.assertEqual(
        alignment.histories(index)[1],
        {'prefill': None, 'decode': alignment.SAME},
    )
    self.assertEqual(
        alignment.histories(index)[2],
        {'prefill': alignment.SAME, 'decode': alignment.SAME},
    )
    self.assertEqual(self._context(index, 2), 'same')
    self.assertEqual(alignment.divergences(index), {1: None, 2: None})

  def test_different_history_lengths_differ(self):
    """A system message on one side only: its history differs."""
    chats = {
        'ref': fixtures.TinyChat('', 8, seed=0),
        'target': fixtures.TinyChat('Be brief.', 8, seed=1),
    }
    index = capture_index.CaptureIndex.create(
        self.root,
        {side: fixtures.side_info(chat) for side, chat in chats.items()},
    )
    for n, prompt in enumerate(('Hi', 'More'), 1):
      completions = {
          side: fixtures.write_turn(
              index.turn_directory(side, n),
              chat.send(prompt, ['a'], True),
          )
          for side, chat in chats.items()
      }
      index.add_turn(n, prompt, completions)

    self.assertEqual(
        alignment.histories(index)[1],
        {'prefill': None, 'decode': alignment.DIFFERENT},
    )
    self.assertEqual(
        alignment.histories(index)[2],
        {'prefill': alignment.DIFFERENT, 'decode': alignment.DIFFERENT},
    )
    # Token analysis says so too; its logits still compare.
    self.assertEqual(self._context(index, 2), 'different')
    # Turn 1's sides differ from their first position on: the system
    # message the target's template renders there.
    diverged = alignment.divergences(index)
    self.assertEqual(diverged[1], diverged[2])
    position, kind = diverged[1]
    self.assertEqual(kind, alignment.DIFFERENT)
    self.assertLess(
        position, index.turn(1)['sides']['ref']['input'][-1]['position']
    )


class MatchRateTest(absltest.TestCase):
  """The share of chosen rows that are the same."""

  def test_rates_over_chosen_rows(self):
    """All rows, or the chosen ones."""
    rows = [{'same': True}, {'same': False}, {'same': True}, {'same': True}]
    self.assertEqual(alignment.match_rate(rows), 0.75)
    self.assertEqual(alignment.match_rate(rows, [0, 2]), 1.0)
    self.assertEqual(alignment.match_rate(rows, [1]), 0.0)

  def test_no_rate_without_rows_or_with_an_unknown_one(self):
    """Nothing to count, or a row that cannot tell: no rate."""
    self.assertIsNone(alignment.match_rate([]))
    self.assertIsNone(alignment.match_rate([{'same': True}], []))
    rows = [{'same': True}, {'same': None}]
    self.assertIsNone(alignment.match_rate(rows))
    self.assertEqual(alignment.match_rate(rows, [0]), 1.0)


class PositionsTest(absltest.TestCase):
  """The columns of a stage's rows."""

  def test_each_row_reads_each_side_at_its_own_position(self):
    """A gap has no position on its side."""
    ref = [{'position': 10}, {'position': 11}]
    target = [{'position': 20}, {'position': 21}, {'position': 22}]
    rows = [
        {'ref': 0, 'target': 0},
        {'ref': None, 'target': 1},
        {'ref': 1, 'target': 2},
    ]
    self.assertEqual(
        alignment.positions(rows, ref, target, 10),
        {'start': 10, 'pairs': [[10, 20], [None, 21], [11, 22]]},
    )


class SuffixLcsTest(absltest.TestCase):
  """What the bit-parallel lengths keep."""

  def test_codes_of_one_sequence_alone_get_no_mask(self):
    """A textless token's own code, say, can match nothing."""
    lengths = alignment._SuffixLcs([1, 2, 3], [2, -1, -2, 3, 4])
    self.assertEqual(set(lengths._matches), {2, 3})
    self.assertEqual(lengths.length(0, 0), 2)


if __name__ == '__main__':
  test_main.main()
