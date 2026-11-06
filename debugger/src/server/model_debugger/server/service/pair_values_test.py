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

"""Tests of each chat's token pairs: made once, waited for, kept."""

import pathlib
import threading
from typing import Any
from unittest import mock

from absl.testing import absltest
from model_debugger.server.analysis import token_analysis
from model_debugger.server.capture import capture_index
from model_debugger.server.service import pair_values
from model_debugger.server.testing import capture_fixtures as fixtures
from model_debugger.server.testing import test_main
import numpy as np

_WORDS = [f' w{k}' for k in range(20)]
_MAKE = token_analysis.token_analysis
_STEPS = [(k, k) for k in range(len(_WORDS))]


def _add_turn(
    index: capture_index.CaptureIndex, chats: dict[str, Any], n: int
) -> None:
  completions = {
      side: fixtures.write_turn(
          index.turn_directory(side, n),
          chat.send(f'Hi {n}', _WORDS, True),
      )
      for side, chat in chats.items()
  }
  index.add_turn(n, f'Hi {n}', completions)


class PairValuesTest(absltest.TestCase):
  """A chat of 21-token replies, the same on both sides."""

  def setUp(self):
    """One turn captured; nothing kept; token_analysis counted."""
    super().setUp()
    root = pathlib.Path(self.create_tempdir().full_path)
    self.chats = {
        side: fixtures.TinyChat('', 24, seed=seed)
        for seed, side in enumerate(capture_index.SIDES)
    }
    self.index = capture_index.CaptureIndex.create(
        root,
        {side: fixtures.side_info(chat) for side, chat in self.chats.items()},
    )
    _add_turn(self.index, self.chats, 1)
    self.pairs = pair_values.PairValues()
    self.made = mock.Mock(wraps=_MAKE)
    self.enter_context(
        mock.patch.object(token_analysis, 'token_analysis', self.made)
    )

  def _made(self) -> list[Any]:
    """Every pair made so far, in the order asked."""
    keys = []
    for call in self.made.call_args_list:
      keys.extend(call.args[2])
    return keys

  def test_a_pair_is_made_once(self):
    """token-range reads kept pairs; token-analysis makes whole ones."""
    alone = token_analysis.range_values(_MAKE(self.index, 1, [(3, 3)])[0])

    first = self.pairs.values(self.index, 1, _STEPS + [(3, 3)])
    again = self.pairs.values(self.index, 1, _STEPS)

    self.assertCountEqual(self._made(), _STEPS)
    np.testing.assert_array_equal(first[3].numbers, alone.numbers)
    self.assertEqual(first[3].described, alone.described)
    np.testing.assert_array_equal(first[-1].numbers, again[3].numbers)
    # Each task made at most CHUNK pairs.
    self.assertTrue(
        all(
            len(call.args[2]) <= pair_values.CHUNK
            for call in self.made.call_args_list
        )
    )
    # Kept pairs hold no whole pair: token-analysis makes it again,
    # and token-range then reads what it kept.
    self.made.reset_mock()
    (pair,) = self.pairs.pairs(self.index, 1, [(3, 3)])
    self.assertEqual(pair['ref_step'], 3)
    self.assertEqual(self._made(), [(3, 3)])

  def test_token_range_reads_what_token_analysis_made(self):
    """Pairs token-analysis made are not made again for token-range."""
    self.pairs.pairs(self.index, 1, _STEPS[:12])
    self.pairs.values(self.index, 1, _STEPS)
    self.assertCountEqual(self._made(), _STEPS)

  def test_a_pair_being_made_is_waited_for(self):
    """Whoever asks while another makes a pair waits for it."""
    release = threading.Event()
    real = _MAKE

    def slow(index, n, steps, pieces=None):
      release.wait()
      return real(index, n, steps, pieces)

    self.made.side_effect = slow
    answers = {}
    maker = threading.Thread(
        target=lambda: answers.update(
            pairs=self.pairs.pairs(self.index, 1, _STEPS[:12])
        )
    )
    maker.start()
    self._wait_until_making(12)
    waiter = threading.Thread(
        target=lambda: answers.update(
            values=self.pairs.values(self.index, 1, _STEPS)
        )
    )
    waiter.start()
    release.set()
    maker.join()
    waiter.join()

    self.assertCountEqual(self._made(), _STEPS)
    self.assertLen(answers['pairs'], 12)
    self.assertLen(answers['values'], len(_STEPS))
    self.assertEqual(
        token_analysis.range_values(answers['pairs'][5]).described,
        answers['values'][5].described,
    )

  def test_a_waiter_makes_again_what_its_maker_failed(self):
    """A maker's failure: the waiter makes the pair itself."""
    release = threading.Event()
    real = _MAKE
    calls = []

    def failing_once(index, n, steps, pieces=None):
      calls.append(steps)
      if len(calls) == 1:
        release.wait()
        raise OSError('unreadable')
      return real(index, n, steps, pieces)

    self.made.side_effect = failing_once
    errors = []

    def make():
      try:
        self.pairs.pairs(self.index, 1, [(0, 0), (1, 1)])
      except OSError as error:
        errors.append(error)

    maker = threading.Thread(target=make)
    maker.start()
    self._wait_until_making(2)
    answers = []
    waiter = threading.Thread(
        target=lambda: answers.extend(
            self.pairs.values(self.index, 1, [(0, 0), (1, 1), (2, 2)])
        )
    )
    waiter.start()
    release.set()
    maker.join()
    waiter.join()

    self.assertLen(errors, 1)
    self.assertLen(answers, 3)
    self.assertEqual(calls[0], [(0, 0), (1, 1)])
    keys = []
    for steps in calls[1:]:
      keys.extend(steps)
    self.assertCountEqual(
        keys,
        [(0, 0), (1, 1), (2, 2)],
    )

  def test_a_waiter_gets_its_own_error(self):
    """When the pair fails again, the waiter's own error is raised."""
    release = threading.Event()

    def failing(index, n, steps, pieces=None):
      del index, n, steps, pieces  # Unused.
      release.wait()
      raise OSError('unreadable')

    self.made.side_effect = failing
    maker = threading.Thread(
        target=lambda: self.assertRaises(
            OSError, self.pairs.pairs, self.index, 1, [(0, 0)]
        )
    )
    maker.start()
    self._wait_until_making(1)
    failures = []

    def wait():
      try:
        self.pairs.values(self.index, 1, [(0, 0)])
      except OSError as error:
        failures.append(error)

    waiter = threading.Thread(target=wait)
    waiter.start()
    release.set()
    maker.join()
    waiter.join()

    self.assertLen(failures, 1)
    self.assertEqual(self.made.call_count, 2)

  def test_a_failure_leaves_nothing_being_made(self):
    """Whatever a task fails at, its pairs are answered and freed."""
    self.made.side_effect = TypeError('a bug')
    with self.assertRaises(TypeError):
      self.pairs.values(self.index, 1, _STEPS)
    self.made.side_effect = None
    self.assertLen(self.pairs.values(self.index, 1, _STEPS), len(_STEPS))

  def test_a_gone_client_starts_nothing_more(self):
    """Its started pairs are made and kept; no more are started."""
    self.enter_context(mock.patch.object(pair_values, 'CHUNK', 2))
    release = threading.Event()
    real = _MAKE

    def slow(index, n, steps, pieces=None):
      release.wait()
      return real(index, n, steps, pieces)

    self.made.side_effect = slow
    with self.assertRaises(pair_values.AbandonedError):
      self.pairs.values(self.index, 1, _STEPS, gone=lambda: True)
    with self.pairs._lock:  # pylint: disable=protected-access
      started = len(self.pairs._turns[1].making)  # pylint: disable=protected-access
    self.assertEqual(started, pair_values.THREADS * 2)

    release.set()
    self.pairs.values(self.index, 1, _STEPS)
    # The next request waited for the started pairs, and made the rest.
    self.assertCountEqual(self._made(), _STEPS)

  def test_a_turn_is_kept_across_new_turns(self):
    """Turn 1's pairs survive turn 2; a turn recorded again does not."""
    self.pairs.values(self.index, 1, _STEPS)
    _add_turn(self.index, self.chats, 2)
    self.pairs.keep_turns(self.index)
    self.made.reset_mock()

    self.pairs.values(self.index, 1, _STEPS)
    self.assertEqual(self._made(), [])
    self.pairs.values(self.index, 2, _STEPS[:3])
    self.assertCountEqual(self._made(), _STEPS[:3])

    # The same turn number, another entry: its pairs are made again.
    snapshot = self.index.snapshot()
    snapshot.turns[1] = dict(snapshot.turns[1])
    self.pairs.keep_turns(snapshot)
    self.made.reset_mock()
    self.pairs.values(snapshot, 1, _STEPS[:3])
    self.pairs.values(snapshot, 2, _STEPS[:3])
    self.assertCountEqual(self._made(), _STEPS[:3])

  def test_size_counts_the_kept_pairs(self):
    """About a hundred and fifty bytes a pair, and the descriptions."""
    empty = self.pairs.size()
    self.pairs.values(self.index, 1, _STEPS)
    per_pair = (self.pairs.size() - empty) / len(_STEPS)
    self.assertBetween(per_pair, 100, 400)

  def _wait_until_making(self, count: int) -> None:
    """Waits until the pairs being made of turn 1 number `count`."""
    for _ in range(500):
      with self.pairs._lock:  # pylint: disable=protected-access
        turn = self.pairs._turns.get(1)  # pylint: disable=protected-access
        if turn is not None and len(turn.making) >= count:
          return
      threading.Event().wait(0.01)
    self.fail('No pairs were being made.')


if __name__ == '__main__':
  test_main.main()
