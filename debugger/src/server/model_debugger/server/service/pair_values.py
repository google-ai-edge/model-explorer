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

"""Each token pair of a chat, made once, and what token-range reads of it kept.

Making a pair (analysis/token_analysis.token_analysis) reads both sides'
logits, tens of milliseconds, so the hundreds of rows of a Decode take
seconds. token-analysis answers with whole pairs, token-range with what it
reads of them (token_analysis.range_values); both ask `PairValues`, which
makes each pair once, by whoever needs it first, on the pair threads
(POOL_THREADS), at most THREADS tasks of one request at a time: as many as
the token-analysis batches the UI keeps in flight.
Anyone else asking for a pair meanwhile waits for it; a pair whose maker
failed is made again by the next to ask, who then gets its own error.

What token-range reads of a pair is kept, a few numbers per pair, for as long
as the chat's view is (service/session_view.SessionViews). A turn is kept
across the turns added after it: its pairs read its files and the tokens fed
before it, which a later turn does not change. A pair is named by its turn
and each side's step, which no alignment changes. A turn the capture dropped
and recorded again is a new entry of the index, whose pairs are made again.

A request whose client is gone (`gone`) starts no more pairs; the pairs it
started are made all the same, and kept, so the next request reads them.
"""

import collections
from collections.abc import Callable, Iterator, Sequence
import concurrent.futures
import threading
from typing import Any

from model_debugger.server.analysis import token_analysis
from model_debugger.server.capture import capture_index
import numpy as np

# How many tasks of one request make pairs at a time: as many as the UI's
# token-analysis batches in flight (src/ui token_analysis_controller.ts,
# ANALYSIS_CONCURRENCY).
THREADS = 4
# The threads that make pairs, for every chat: two requests' worth. Measured
# on 12 cores with four stages' token-range and two turns' token-analysis at
# once: 26.6 s with 4 threads, 16.0 s with 8.
POOL_THREADS = 8
# How many pairs one task of `values` makes: between tasks it sees what
# others made meanwhile.
CHUNK = 8
# How often a waiting request asks whether its client is gone, in seconds.
POLL_SECONDS = 0.2
# About how many bytes a kept pair holds beyond its row of the arrays
# (RANGE_NUMBERS float64 and a description id): its steps, its slot in the
# turn's dict and its row's index. With the arrays, about 230 bytes a pair,
# as tracemalloc measured on a real turn of 424 pairs (B8's Kept: 4,758).
_PAIR_BYTES = 120
# About how many bytes one distinct description (RangeValues.described)
# holds: a few strings and small tuples.
_DESCRIPTION_BYTES = 1024

Steps = tuple[int | None, int | None]
Gone = Callable[[], bool]

_pool: concurrent.futures.ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _threads() -> concurrent.futures.ThreadPoolExecutor:
  """The pair threads, started when first asked for."""
  global _pool
  with _pool_lock:
    if _pool is None:
      _pool = concurrent.futures.ThreadPoolExecutor(
          POOL_THREADS, thread_name_prefix='token-pairs'
      )
    return _pool


def _present() -> bool:
  return False


class AbandonedError(Exception):
  """The request's client is gone: it starts nothing more."""


class _Turn:
  """The pairs of one turn: those kept, and those being made.

  `numbers` and `described` hold one row per kept pair, the row `rows`
  names by the pair's steps.
  """

  def __init__(self, entry: dict[str, Any]) -> None:
    self.entry = entry
    self.rows: dict[Steps, int] = {}
    self.numbers = np.empty(
        (0, len(token_analysis.RANGE_NUMBERS)), dtype=np.float64
    )
    self.described = np.empty(0, dtype=np.int32)
    # The pairs being made, each a future of its `(pair, values)`.
    self.making: dict[Steps, concurrent.futures.Future[Any]] = {}

  def keep(self, key: Steps, numbers: np.ndarray, described: int) -> None:
    """Stores the numbers and description index for the given steps key."""
    if key in self.rows:
      return
    row = len(self.rows)
    if row == len(self.numbers):
      capacity = max(16, 2 * row)
      grown = np.empty((capacity, self.numbers.shape[1]))
      grown[:row] = self.numbers
      ids = np.empty(capacity, dtype=np.int32)
      ids[:row] = self.described
      self.numbers, self.described = grown, ids
    self.numbers[row] = numbers
    self.described[row] = described
    self.rows[key] = row

  def size(self) -> int:
    return (
        self.numbers.nbytes
        + self.described.nbytes
        + len(self.rows) * _PAIR_BYTES
    )


class PairValues:
  """The pairs of one chat's turns: made once, what token-range reads kept.

  Shared by the threads that answer the chat's requests.
  """

  def __init__(self, pieces: token_analysis.Pieces | None = None) -> None:
    """Nothing kept yet.

    Args:
        pieces: The tokenizer pieces of a vocabulary by its digest, that label
          candidates (token_analysis); None: only the chat's.
    """
    self._pieces = pieces
    self._lock = threading.Lock()
    self._turns: dict[int, _Turn] = {}
    # Each distinct RangeValues.described, and its id.
    self._descriptions: list[tuple[Any, ...]] = []
    self._ids: dict[tuple[Any, ...], int] = {}

  def size(self) -> int:
    """About how many bytes the kept values hold."""
    with self._lock:
      return sum(
          turn.size() for turn in self._turns.values()
      ) + _DESCRIPTION_BYTES * len(self._descriptions)

  def keep_turns(self, index: capture_index.CaptureIndex) -> None:
    """Forgets the turns that are no longer the index's, as entries."""
    with self._lock:
      for n in list(self._turns):
        if not (
            n <= len(index.turns) and index.turns[n - 1] is self._turns[n].entry
        ):
          del self._turns[n]

  def pairs(
      self,
      index: capture_index.CaptureIndex,
      n: int,
      steps: Sequence[Steps],
      gone: Gone = _present,
  ) -> list[dict[str, Any]]:
    """The `TokenAnalysisPair` of each pair, as token_analysis makes it.

    Each pair nobody is making is made, all of them in one task; the
    others are waited for.

    Args:
      index: The chat's capture.
      n: The turn.
      steps: `(ref step, target step)` per pair, as token_analysis.
      gone: Whether the request's client is gone.

    Returns:
      The TokenAnalysisPair dictionary for each pair in steps.

    Raises:
      AbandonedError: The client is gone.
      As token_analysis.
    """
    made: dict[Steps, dict[str, Any]] = {}
    pending = list(dict.fromkeys(steps))
    while pending:
      with self._lock:
        turn = self._turn(index, n)
        waiting, claimed = {}, []
        for key in pending:
          future = turn.making.get(key)
          if future is None:
            future = concurrent.futures.Future()
            turn.making[key] = future
            claimed.append((key, future))
          waiting[future] = key
      if claimed:
        self._start(index, n, turn, claimed)
      mine = {future for _, future in claimed}
      pending = []
      for future in _completed(waiting, gone):
        key = waiting[future]
        try:
          made[key] = future.result()[0]
        except Exception:  # pylint: disable=broad-exception-caught
          if future in mine:
            raise
          pending.append(key)  # Its maker failed: make it again.
    return [made[key] for key in steps]

  def values(
      self,
      index: capture_index.CaptureIndex,
      n: int,
      steps: Sequence[Steps],
      gone: Gone = _present,
  ) -> list[token_analysis.RangeValues]:
    """What token-range reads of each pair (token_analysis.range_values).

    Kept pairs are read; the pairs nobody is making are made, CHUNK at a
    time, up to THREADS tasks at once; the others are waited for.

    Args:
      index: The chat's capture.
      n: The turn.
      steps: `(ref step, target step)` per pair, as token_analysis.
      gone: Whether the request's client is gone.

    Returns:
      The RangeValues for each pair in steps.

    Raises:
      AbandonedError: The client is gone.
      As token_analysis.
    """
    found: dict[Steps, token_analysis.RangeValues] = {}
    free = collections.deque(dict.fromkeys(steps))
    waiting: dict[concurrent.futures.Future[Any], Steps] = {}
    mine: set[concurrent.futures.Future[Any]] = set()
    # The futures of each task this request started and is making.
    tasks: list[list[concurrent.futures.Future[Any]]] = []
    while free or waiting:
      tasks = [task for task in tasks if not all(f.done() for f in task)]
      while free and len(tasks) < THREADS:
        with self._lock:
          turn = self._turn(index, n)
          claimed = []
          while free and len(claimed) < CHUNK:
            key = free.popleft()
            row = turn.rows.get(key)
            if row is not None:
              found[key] = self._read(turn, row)
            elif key in turn.making:
              waiting[turn.making[key]] = key
            else:
              future = concurrent.futures.Future()
              turn.making[key] = future
              claimed.append((key, future))
        if claimed:
          for key, future in claimed:
            waiting[future] = key
            mine.add(future)
          tasks.append([future for _, future in claimed])
          self._start(index, n, turn, claimed)
      if not waiting:
        continue
      done, _ = concurrent.futures.wait(
          waiting,
          timeout=POLL_SECONDS,
          return_when=concurrent.futures.FIRST_COMPLETED,
      )
      for future in done:
        key = waiting.pop(future)
        try:
          found[key] = future.result()[1]
        except Exception:  # pylint: disable=broad-exception-caught
          if future in mine:
            raise
          free.append(key)  # Its maker failed: make it again.
      if (free or waiting) and gone():
        raise AbandonedError()
    return [found[key] for key in steps]

  def _turn(self, index: capture_index.CaptureIndex, n: int) -> _Turn:
    """The kept turn `n` of the index, new if its entry is another."""
    entry = index.turn(n)
    turn = self._turns.get(n)
    if turn is None or turn.entry is not entry:
      turn = self._turns[n] = _Turn(entry)
    return turn

  def _read(self, turn: _Turn, row: int) -> token_analysis.RangeValues:
    return token_analysis.RangeValues(
        turn.numbers[row].copy(),
        self._descriptions[turn.described[row]],
    )

  def _start(
      self,
      index: capture_index.CaptureIndex,
      n: int,
      turn: _Turn,
      claimed: list[tuple[Steps, concurrent.futures.Future[Any]]],
  ) -> None:
    """Makes claimed pairs on the pair threads, or answers them failed."""
    try:
      _threads().submit(self._make, index, n, turn, claimed)
    except BaseException as error:
      self._answer(turn, claimed, None, error)
      raise

  def _make(
      self,
      index: capture_index.CaptureIndex,
      n: int,
      turn: _Turn,
      claimed: list[tuple[Steps, concurrent.futures.Future[Any]]],
  ) -> None:
    """Makes claimed pairs, keeps them, and answers their futures."""
    try:
      pairs = token_analysis.token_analysis(
          index, n, [key for key, _ in claimed], self._pieces
      )
      made = [(pair, token_analysis.range_values(pair)) for pair in pairs]
      with self._lock:
        for (key, _), (_, values) in zip(claimed, made, strict=True):
          turn.keep(key, values.numbers, self._id(values.described))
    except BaseException as error:  # pylint: disable=broad-exception-caught
      self._answer(turn, claimed, None, error)
    else:
      self._answer(turn, claimed, made, None)

  def _answer(
      self,
      turn: _Turn,
      claimed: list[tuple[Steps, concurrent.futures.Future[Any]]],
      made: list[tuple[dict[str, Any], token_analysis.RangeValues]] | None,
      failed: BaseException | None,
  ) -> None:
    """Answers every claimed future, which is then no longer being made.

    A failure goes to each of them: the next to ask makes it again.

    Args:
      turn: The turn being updated.
      claimed: The claimed step pairs and their futures.
      made: The computed results, or None if failed.
      failed: The exception if computation failed, or None.
    """
    with self._lock:
      for key, future in claimed:
        if turn.making.get(key) is future:
          del turn.making[key]
    for k, (_, future) in enumerate(claimed):
      if failed is None:
        future.set_result(made[k])
      else:
        future.set_exception(failed)

  def _id(self, described: tuple[Any, ...]) -> int:
    """The id of a description, interned; under the lock."""
    found = self._ids.get(described)
    if found is None:
      found = self._ids[described] = len(self._descriptions)
      self._descriptions.append(described)
    return found


def _completed(
    futures: dict[concurrent.futures.Future[Any], Steps], gone: Gone
) -> Iterator[concurrent.futures.Future[Any]]:
  """Yields each future once done, asking `gone` while some are not.

  Args:
    futures: Dictionary mapping futures to their steps.
    gone: Callable returning whether client is gone.

  Yields:
    Completed futures one by one.

  Raises:
    AbandonedError: The client is gone.
  """
  remaining = set(futures)
  while remaining:
    done, remaining = concurrent.futures.wait(
        remaining,
        timeout=POLL_SECONDS,
        return_when=concurrent.futures.FIRST_COMPLETED,
    )
    yield from done
    if remaining and gone():
      raise AbandonedError()
