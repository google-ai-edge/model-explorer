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

"""How a turn's ref and target tokens line up, and where they part.

A turn's two sides are shown in rows. By content, each phase (the thinking,
then the answer) is aligned on its own and tokens of equal text share a row
(a longest common subsequence); by step, a row holds the tokens of one
generation step. A row names each side's token by its index in that side's
list, or None where the side has none: a row is the same when both sides
have a token, of one text and one phase, and unknown (None) when a token's
text was not captured.

The input: a side's tokens are a list of dicts, in order, each with `text`
(None where the capture names none), `step` (its ordinal in the turn) and
`position` (its context position), and, where it has one, `phase`
('thinking' or 'response'; 'response' without) and `kind` ('text',
'template' or 'special'; only `messages` reads it). A side without tokens is
None: not captured. service/session_view's `CapturedToken`s are such tokens.

Rows compare text and history compares ids, for one reason: what each asks.
A row asks what the two sides generated, and each side's ids are of its own
tokenizer, so a row compares what the reader sees, the text. History
(`histories`) asks whether the two runtimes were fed the same context before
a stage; a runtime is fed ids, and the logits that follow depend on nothing
else, so it compares the id at every position (token_analysis.same_context),
and a token the runner did not name proves nothing either way.
"""

from collections.abc import Sequence
import itertools
import math
import re
from typing import Any

from model_debugger.server.analysis import token_analysis
from model_debugger.server.capture import capture_index

ALIGNMENTS = ('content', 'steps')
SAME = 'same'
DIFFERENT = 'different'
DIVERGED = 'diverged'
UNKNOWN = 'unknown'
NOT_CAPTURED = 'not_captured'

# A position after every token a side feeds.
_END = 2**63 - 1

_PHASES = ('thinking', 'response')
_ROLES = ('system', 'user', 'model')
# A turn opener of one token, such as `<|turn>user`.
_OPENER = re.compile(r'<\|turn>(system|user|model)')


def _phase(token: dict[str, Any]) -> str:
  return token.get('phase') or 'response'


def _same_text(a: dict[str, Any], b: dict[str, Any]) -> bool:
  """Whether two tokens have known, equal text."""
  return a['text'] is not None and a['text'] == b['text']


class _SuffixLcs:
  """The lengths of the longest common subsequences of two suffixes.

  `length(i, j)` is that of `a[i:]` and `b[j:]`. It reads them from the
  bit-parallel rows (Hyyro's) of the reversed sequences: row p, over
  `a[::-1][:p]`, has a zero at bit q exactly where the length grows from
  `b[::-1][:q]` to `b[::-1][:q + 1]`. Only every `step`-th row is kept, and
  the block of rows a length needs is computed again from the row before
  it. With n = len(a) and m = len(b), it holds about 2 * sqrt(n) rows and
  one match mask per code that occurs in both sequences, each of m bits:
  memory grows with (sqrt(n) + the shared codes) * m bits, not with n * m.
  """

  def __init__(self, a: Sequence[int], b: Sequence[int]) -> None:
    self._a = list(reversed(a))
    self._n, self._m = len(a), len(b)
    self._full = (1 << self._m) - 1
    # A code of b alone never matches: it needs no mask (a token without
    # text has a code of its own, which nothing else has).
    wanted = set(a)
    code_positions = {}
    for q, code in enumerate(reversed(b)):
      if code in wanted:
        code_positions.setdefault(code, []).append(q)
    self._matches = {}
    for code, found in code_positions.items():
      bits = bytearray((self._m + 7) // 8)
      for q in found:
        bits[q >> 3] |= 1 << (q & 7)
      self._matches[code] = int.from_bytes(bits, 'little')
    self._step = math.isqrt(self._n) + 1
    self._kept = {}
    row = self._full
    for p in range(self._n + 1):
      if p % self._step == 0:
        self._kept[p] = row
      if p < self._n:
        row = self._next(row, self._a[p])
    self._start = None
    self._rows = []

  def _next(self, row: int, code: int) -> int:
    match = self._matches.get(code)
    if not match:
      return row
    common = row & match
    return ((row + common) | (row - common)) & self._full

  def _row(self, p: int) -> int:
    """Row p; its block also holds row p - 1, the other one a step needs."""
    if self._start is None or not (
        self._start <= p <= self._start + self._step
    ):
      start = max(0, p - 1) // self._step * self._step
      rows = [self._kept[start]]
      for k in range(start, min(start + self._step, self._n)):
        rows.append(self._next(rows[-1], self._a[k]))
      self._start, self._rows = start, rows
    return self._rows[p - self._start]

  def length(self, i: int, j: int) -> int:
    q = self._m - j
    return q - (self._row(self._n - i) & ((1 << q) - 1)).bit_count()


def _common_pairs(
    a: Sequence[int], b: Sequence[int]
) -> list[tuple[int | None, int | None]]:
  """Rows of two sequences of codes, equal codes sharing a row.

  Walks a longest common subsequence from the start: equal heads share a
  row; otherwise the head of `a` goes alone while that keeps the longest
  length, else the head of `b`. The tokens that go alone between two
  shared rows are then paired in order, the longer side's last alone.

  Args:
      a: The first sequence of codes.
      b: The second sequence of codes.

  Returns:
      `(index in a, index in b)` per row, in order; None for a side without
      an element in the row.
  """
  lengths = _SuffixLcs(a, b) if a and b else None
  rows, removed, added = [], [], []

  def flush() -> None:
    for one, other in itertools.zip_longest(removed, added):
      rows.append((one, other))
    removed.clear()
    added.clear()

  i = j = 0
  while i < len(a) or j < len(b):
    if i < len(a) and j < len(b) and a[i] == b[j]:
      flush()
      rows.append((i, j))
      i += 1
      j += 1
    elif i < len(a) and (
        j == len(b) or lengths.length(i + 1, j) >= lengths.length(i, j + 1)
    ):
      removed.append(i)
      i += 1
    else:
      added.append(j)
      j += 1
  flush()
  return rows


def _content_pairs(
    ref: Sequence[dict[str, Any]], target: Sequence[dict[str, Any]]
) -> list[tuple[int | None, int | None]]:
  """Rows by content: each phase on its own, the thinking first."""
  codes = {}
  unnamed = itertools.count(-1, -1)

  def code(token: dict[str, Any]) -> int:
    # A token without text matches nothing, not even another without.
    if token['text'] is None:
      return next(unnamed)
    return codes.setdefault(token['text'], len(codes))

  rows = []
  for phase in _PHASES:
    a = [k for k, token in enumerate(ref) if _phase(token) == phase]
    b = [k for k, token in enumerate(target) if _phase(token) == phase]
    # The equal start and end need no search: long turns often differ
    # in a short stretch.
    prefix = 0
    while prefix < min(len(a), len(b)) and _same_text(
        ref[a[prefix]], target[b[prefix]]
    ):
      prefix += 1
    suffix = 0
    while suffix < min(len(a), len(b)) - prefix and _same_text(
        ref[a[-1 - suffix]], target[b[-1 - suffix]]
    ):
      suffix += 1
    rows += zip(a[:prefix], b[:prefix])
    middle_a = a[prefix : len(a) - suffix]
    middle_b = b[prefix : len(b) - suffix]
    for one, other in _common_pairs(
        [code(ref[k]) for k in middle_a],
        [code(target[k]) for k in middle_b],
    ):
      rows.append((
          None if one is None else middle_a[one],
          None if other is None else middle_b[other],
      ))
    rows += zip(a[len(a) - suffix :], b[len(b) - suffix :])
  return rows


def _step_pairs(
    ref: Sequence[dict[str, Any]], target: Sequence[dict[str, Any]]
) -> list[tuple[int | None, int | None]]:
  """Rows by step: the tokens of one generation step share a row."""
  a = {token['step']: k for k, token in enumerate(ref)}
  b = {token['step']: k for k, token in enumerate(target)}
  return [(a.get(step), b.get(step)) for step in sorted(a.keys() | b.keys())]


def _same(
    ref: Sequence[dict[str, Any]],
    target: Sequence[dict[str, Any]],
    one: int | None,
    other: int | None,
) -> bool | None:
  if (one is not None and ref[one]['text'] is None) or (
      other is not None and target[other]['text'] is None
  ):
    return None
  return (
      one is not None
      and other is not None
      and _same_text(ref[one], target[other])
      and _phase(ref[one]) == _phase(target[other])
  )


def _mismatched(row: dict[str, Any]) -> bool:
  """Whether the two tokens of `row` are known to differ, not just unknown."""
  return row['same'] is not None and not row['same']


def align(
    ref: Sequence[dict[str, Any]] | None,
    target: Sequence[dict[str, Any]] | None,
    alignment: str,
) -> list[dict[str, Any]]:
  """Returns the rows of two sides' tokens.

  Args:
      ref: The reference side's tokens; None when not captured.
      target: The target side's tokens; None when not captured.
      alignment: 'content' or 'steps'.

  Returns:
      `{ref, target, same}` per row, in order; every token of a side is in
      exactly one row. When a side was not captured, the other's tokens
      each have a row, of unknown sameness.
  """
  if ref is None or target is None:
    return [
        {
            'ref': k if ref is not None else None,
            'target': k if ref is None else None,
            'same': None,
        }
        for k in range(len(ref if ref is not None else target or []))
    ]
  pairs = (
      _step_pairs(ref, target)
      if alignment == 'steps'
      else _content_pairs(ref, target)
  )
  return [
      {'ref': one, 'target': other, 'same': _same(ref, target, one, other)}
      for one, other in pairs
  ]


def first_divergence(
    rows: Sequence[dict[str, Any]],
    ref: Sequence[dict[str, Any]] | None,
    target: Sequence[dict[str, Any]] | None,
) -> dict[str, Any] | None:
  """Returns the first generation step where the two sides differ.

  It is the first row that is not the same, when that row is different
  and every row before it holds one step on both sides: both runtimes
  sampled that step after the same generated tokens. A content row may
  hold one side's token alone; the other side's token of that step is then
  `filled` in, and `partner` is the row it is drawn in.

  Args:
      rows: The turn's rows (`align`).
      ref: The reference side's tokens; None when not captured.
      target: The target side's tokens; None when not captured.

  Returns:
      `{row, ref, target, filled, partner}`: the row, each side's token of
      the step, the side filled in ('ref', 'target' or None) and the row
      that side's token is drawn in (None when nothing was filled in).
      None when no row differs, when a token's text is missing first, or
      when the rows before it do not pair equal steps.
  """
  first = next((k for k, row in enumerate(rows) if not row['same']), None)
  if first is None or not _mismatched(rows[first]):
    return None
  if any(
      ref[row['ref']]['step'] != target[row['target']]['step']
      for row in rows[:first]
  ):
    return None
  row = rows[first]
  if row['ref'] is not None and row['target'] is not None:
    if ref[row['ref']]['step'] != target[row['target']]['step']:
      return None
    return {
        'row': first,
        'ref': row['ref'],
        'target': row['target'],
        'filled': None,
        'partner': None,
    }
  step = (ref[row['ref']] if row['ref'] is not None else target[row['target']])[
      'step'
  ]
  filled = 'target' if row['ref'] is not None else 'ref'
  tokens = ref if filled == 'ref' else target
  found = next(
      (k for k, token in enumerate(tokens or []) if token['step'] == step),
      None,
  )
  if found is None:
    return None
  pair = {**row, filled: found}
  return {
      'row': first,
      'ref': pair['ref'],
      'target': pair['target'],
      'filled': filled,
      'partner': next(
          (k for k, other in enumerate(rows) if other[filled] == found), None
      ),
  }


def conclusion(
    rows: Sequence[dict[str, Any]],
    divergence: dict[str, Any] | None,
    ref: Sequence[dict[str, Any]] | None,
    target: Sequence[dict[str, Any]] | None,
) -> dict[str, Any]:
  """Returns what a turn's generations did relative to each other.

  Args:
      rows: The turn's rows (`align`).
      divergence: Its `first_divergence`.
      ref: The reference side's tokens; None when not captured.
      target: The target side's tokens; None when not captured.

  Returns:
      `{kind, step, prefix}`: `kind` is SAME, DIVERGED, UNKNOWN (a token's
      text is missing, or the rows do not prove which step differs first)
      or NOT_CAPTURED; `step` is the target's first differing step, for
      DIVERGED; `prefix` counts the leading rows that are the same.
  """
  if ref is None or target is None:
    return {'kind': NOT_CAPTURED, 'step': None, 'prefix': 0}
  prefix = next((k for k, row in enumerate(rows) if not row['same']), len(rows))
  if prefix == len(rows):
    return {'kind': SAME, 'step': None, 'prefix': prefix}
  if divergence is None:
    return {'kind': UNKNOWN, 'step': None, 'prefix': prefix}
  return {
      'kind': DIVERGED,
      'step': target[divergence['target']]['step'],
      'prefix': prefix,
  }


def compare(
    ref: Sequence[dict[str, Any]] | None,
    target: Sequence[dict[str, Any]] | None,
    alignment: str,
) -> dict[str, Any]:
  """Returns one alignment of a turn's generated tokens.

  Args:
      ref: The reference side's generated tokens; None when not captured.
      target: The target side's generated tokens; None when not captured.
      alignment: 'content' or 'steps'.

  Returns:
      `{rows, first_mismatch, first_divergence, conclusion}`:
      `first_mismatch` is the first row that is different (None when none
      is), the others are as `align`, `first_divergence` and `conclusion`
      return them.
  """
  rows = align(ref, target, alignment)
  divergence = first_divergence(rows, ref, target)
  return {
      'rows': rows,
      'first_mismatch': next(
          (k for k, row in enumerate(rows) if _mismatched(row)), None
      ),
      'first_divergence': divergence,
      'conclusion': conclusion(rows, divergence, ref, target),
  }


def _opener(tokens: Sequence[dict[str, Any]], k: int) -> tuple[str, int] | None:
  """The role a turn opener at `k` opens, and how many tokens it has.

  An opener is `<|turn>user` as one token, or `<|turn>` and then the role
  as a token of its own; neither is text.

  Args:
      tokens: The input tokens.
      k: The index of the token that may open a turn.

  Returns:
      `(role, number of tokens)`, or None when no opener is at `k`.
  """
  if tokens[k].get('kind') == 'text':
    return None
  text = tokens[k]['text'] or ''
  single = _OPENER.fullmatch(text)
  if single:
    return single[1], 1
  after = tokens[k + 1] if k + 1 < len(tokens) else None
  role = ''
  if after is None or after.get('kind') != 'text':
    role = ((after or {}).get('text') or '').strip()
  if text == '<|turn>' and role in _ROLES:
    return role, 2
  return None


def messages(tokens: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
  """Returns input tokens grouped into the messages of the chat template.

  Args:
      tokens: Input tokens, each with its `kind` (`text`, `template` or
        `special`; service/session_view kinds them).

  Returns:
      `{role, leading, text, trailing}` per message, in order, as indices
      of `tokens`: `role` is 'system', 'user', 'model', or 'unknown' for
      text that follows a closed message or tokens before any opener;
      `leading` holds the template and special tokens before the text (the
      start token, the opener, a newline), `trailing` those after it. The
      prompt for the reply (`<|turn>model` and its newline) is the last
      message's `trailing`.
  """
  rows, preamble = [], []
  current = None
  k = 0
  while k < len(tokens):
    opener = _opener(tokens, k)
    if opener:
      role, length = opener
      current = {
          'role': role,
          'leading': preamble + list(range(k, k + length)),
          'text': [],
          'trailing': [],
      }
      preamble = []
      rows.append(current)
      k += length
      continue
    if current is None:
      preamble.append(k)
    elif tokens[k].get('kind') == 'text':
      if current['trailing']:
        current = {
            'role': 'unknown',
            'leading': [],
            'text': [k],
            'trailing': [],
        }
        rows.append(current)
      else:
        current['text'].append(k)
    else:
      current['trailing' if current['text'] else 'leading'].append(k)
    k += 1
  if preamble:
    rows.append(
        {'role': 'unknown', 'leading': preamble, 'text': [], 'trailing': []}
    )
  last = rows[-1] if rows else None
  if (
      len(rows) > 1
      and last['role'] == 'model'
      and not last['text']
      and not last['trailing']
  ):
    rows.pop()
    rows[-1]['trailing'] += last['leading']
  return rows


def compare_prefill(
    ref: Sequence[dict[str, Any]] | None,
    target: Sequence[dict[str, Any]] | None,
) -> dict[str, Any]:
  """Returns a turn's two inputs aligned by content, in message rows.

  Args:
      ref: The reference side's input tokens; None when not captured.
      target: The target side's input tokens; None when not captured.

  Returns:
      `{rows, messages, conclusion}`: `rows` as `align` returns them;
      `messages` as `messages` groups them, by each row's ref token (its
      target token where it has none), as indices of `rows`; `conclusion`
      is `{kind, row}`, `row` the first row that is not the same and
      `kind` SAME, DIFFERENT, UNKNOWN (that row's text is missing) or
      NOT_CAPTURED (a side has no input tokens).
  """
  if not ref or not target:
    return {
        'rows': [],
        'messages': [],
        'conclusion': {'kind': NOT_CAPTURED, 'row': None},
    }
  rows = align(ref, target, 'content')
  first = next((k for k, row in enumerate(rows) if not row['same']), None)
  if first is None:
    kind = SAME
  else:
    kind = UNKNOWN if rows[first]['same'] is None else DIFFERENT
  return {
      'rows': rows,
      'messages': messages([
          ref[row['ref']] if row['ref'] is not None else target[row['target']]
          for row in rows
      ]),
      'conclusion': {'kind': kind, 'row': first},
  }


def stage_summaries(
    stages: Sequence[tuple[Sequence[dict[str, Any]], int | None, int | None]],
) -> list[dict[str, Any]]:
  """Returns how much of each stage of a chat is the same, in order.

  Args:
      stages: Every stage of the chat in order (each turn's Prefill, then its
        Decode): its rows, and how many tokens each side has (None when not
        captured).

  Returns:
      Per stage, `{match_rate, cumulative_match_rate, first_mismatch}`:
      the share of its rows that are the same, of the rows from the first
      stage to it, and whether it holds the chat's first different row.
      A rate is None while a row's sameness is unknown or a side was not
      captured, here or (for the cumulative one) in a stage before; so is
      `first_mismatch` while an unseen difference may come before it.
  """
  total = matched = 0
  complete, diverged = True, False
  summaries = []
  for rows, *counts in stages:
    stage_matched = sum(1 for row in rows if row['same'])
    stage_complete = all(
        count is not None
        and sum(row[side] is not None for row in rows) == count
        for side, count in zip(capture_index.SIDES, counts)
    ) and all(row['same'] is not None for row in rows)
    mismatch = any(_mismatched(row) for row in rows)
    first = mismatch and not diverged
    total += len(rows)
    matched += stage_matched
    complete = complete and stage_complete
    diverged = diverged or mismatch
    summaries.append({
        'match_rate': (
            stage_matched / len(rows) if stage_complete and rows else None
        ),
        'cumulative_match_rate': (
            matched / total if complete and total else None
        ),
        'first_mismatch': (
            True if first else False if stage_complete or diverged else None
        ),
    })
  return summaries


def match_rate(
    rows: Sequence[dict[str, Any]], selected: Sequence[int] | None = None
) -> float | None:
  """Returns the share of a stage's rows that are the same.

  Args:
      rows: The stage's rows (`align`, or `compare_prefill`'s).
      selected: Indices of the rows to count, each once; None for all.

  Returns:
      The share, or None when no row is counted or one counted row's
      sameness is unknown.
  """
  chosen = rows if selected is None else [rows[k] for k in selected]
  if not chosen or any(row['same'] is None for row in chosen):
    return None
  return sum(1 for row in chosen if row['same']) / len(chosen)


def positions(
    rows: Sequence[dict[str, Any]],
    ref: Sequence[dict[str, Any]] | None,
    target: Sequence[dict[str, Any]] | None,
    start: int,
) -> dict[str, Any]:
  """Returns the context positions a stage's rows name, column by column.

  A row is one column of the KV and activation analyses: column
  `start + k` reads row k's ref token at its position and its target token
  at its own, which differ where the histories do.

  Args:
      rows: The stage's rows.
      ref: The ref side's tokens the rows index; None when not captured.
      target: The target side's tokens; None when not captured.
      start: The first column: the ref side's first position of the stage.

  Returns:
      `{start, pairs}`: per row, `[ref position, target position]`, None
      for a side without a token in it.
  """
  return {
      'start': start,
      'pairs': [
          [
              None if row['ref'] is None else ref[row['ref']]['position'],
              None
              if row['target'] is None
              else target[row['target']]['position'],
          ]
          for row in rows
      ],
  }


def histories(
    index: capture_index.CaptureIndex,
) -> dict[int, dict[str, str | None]]:
  """Returns whether both sides had fed the same tokens before each stage.

  Token analysis's rule (`token_analysis.same_context`): the same tokens
  at the same positions, different where the named ids or the positions
  prove it, unknown where only a token the runner did not name could tell.
  Each side's fed tokens are read once, turn by turn.

  Args:
      index: The chat's capture.

  Returns:
      Per read turn, `{prefill, decode}`: before the turn's input, and
      before its first generated token; each SAME, DIFFERENT, UNKNOWN or
      None when neither side had fed anything before.
  """
  fed = {
      side: token_analysis.fed_turns(index, side)
      for side in capture_index.SIDES
  }
  result = {}
  for turn in index.turns:
    n = turn['n']
    sides = turn['sides']
    tokens = {side: next(fed[side]) for side in capture_index.SIDES}
    result[n] = {}
    for stage, key in (('prefill', 'input'), ('decode', 'output')):
      if not all(sides[side][key] for side in capture_index.SIDES):
        result[n][stage] = UNKNOWN
        continue
      contexts = [
          tokens[side].before(sides[side][key][0]['position'])
          for side in capture_index.SIDES
      ]
      same = token_analysis.same_context(*contexts)
      if not any(len(context) for context in contexts):
        result[n][stage] = None
      elif same is None:
        result[n][stage] = UNKNOWN
      else:
        result[n][stage] = SAME if same else DIFFERENT
  return result


def divergences(
    index: capture_index.CaptureIndex,
) -> dict[int, tuple[int, str] | None]:
  """Returns where each read turn's two sides stop reading the same tokens.

  Of everything each side had fed by the end of the turn, its input and
  reply after the earlier turns (token_analysis.fed_turns), as
  token_analysis.divergence finds it: a column of the turn that reads a
  position after it reads different tokens on the two sides (or tokens not
  proven the same), as does one whose two sides sit at different
  positions.

  Args:
      index: The chat's capture.

  Returns:
      Per read turn, `(position, DIFFERENT or UNKNOWN)`, or None where the
      two sides fed the same tokens.
  """
  fed = {
      side: token_analysis.fed_turns(index, side)
      for side in capture_index.SIDES
  }
  result = {}
  for turn in index.turns:
    contexts = [next(fed[side]).before(_END) for side in capture_index.SIDES]
    result[turn['n']] = token_analysis.divergence(*contexts)
  return result
