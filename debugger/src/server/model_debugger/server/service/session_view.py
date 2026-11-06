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

"""The UI's view of a chat's capture: the response of GET /api/session.

Its shape is the UI's `Session` (`src/ui/src/server/contracts/capture.ts`). A
turn's `step_start` is the reference side's: decode token `i` at
`step_start + i`. The target's tokens may sit elsewhere, when the two histories
differ in length, so every token also carries its own side's `position`, and
says which step it is at a column of the analyses (`_steps`): `produced_text`
the step that produced it, which Final Logits and the intermediate tensors read
there, and `written_text` the step that wrote its K and V. A turn the capture
index rejected follows the others, with its prompt and no tokens: its
`rejected` says which side, its code (`turn_rejected`, or `turn_unreadable` as
the capture was read) and the text to show. A generated token of a greedy chat
whose step's logits name no one argmax (capture_index.greedy_choice) carries
`cannot_confirm`: the text to show, and why as a reason of its side
(metrics.reason, of code token_analysis.GREEDY_UNCONFIRMED). Each side's turn
carries what that side was sent, `input`, and why its reply ended, as
`stop_reason` (`eos`, `max_output_tokens` or `context_full`) and as the text
to show, `stop_text`.

Each turn the index read carries its `comparison`, what the server makes of
its two sides (analysis/alignment): whether the sides had fed the same tokens
before its Prefill and before its Decode (`history`), its two inputs aligned
by content and grouped into the messages of the chat template (`prefill`),
and its generated tokens by each alignment the UI offers (`alignments`: by
content and by step), with the first different row, the first differing
step, the turn's conclusion and the stage summaries of the chat up to it. A
row names a token by its index in its side's `input_tokens` or `tokens`, and
each stage's rows come with the context positions they name, column by
column (`positions`), the columns of the KV and activation analyses.
`positions.columns` says, per analysis of the stage, which of those columns
it has, `{start, end}`, and why it has no more (`omitted`, reasons), as the
server decides them, and what the stage's values cannot show (`notes`): the
intermediate tensors have the columns some forward pass produced, and the
layers both sides computed (tensor_analysis.columns); the KV cache the
columns its snapshot of the stage holds; Final Logits and Sampling every
column. Where the two sides read different tokens before a column, after
the turn's divergence (alignment.divergences), the values still compare,
by one rule (token_analysis.context_reason): each analysis that compares
the two sides at a column (`_COMPARED_VALUES`) says from which of its
columns on, and why in words (`context`, `{start, reason}`, null when
none does). Each side's input is grouped into messages on its own too, as
`input_messages`.
A side's turn names its own prompt (`input`) and reply (`output`); turn 1
also carries the messages the chat opened with (`messages`, such as its
system message), which no turn's prompt or reply holds.

Each turn carries its `stages`, `prefill` and `decode`, each with what the
"Add analysis" picker and the stage's panels need, read from the capture
index alone (no tensor is read): each side's `positions` of the stage
(`{start, end}`, end exclusive; null for a rejected turn); its KV snapshot
(`kv`: the `moment` that belongs to the stage, after its Prefill or at the
end of generation, and the `context_id` of kv-analysis, null when no side
has one); and per analysis of the metric catalog (`analyses`) whether it can
be added (`available`) and if not why (`reasons`, metrics.reason, with a
code of `REASONS` or of the analysis that decides). Final Logits and Sampling
serve the stages the token routes do (token_analysis.stage_reason): Final
Logits needs a step's `post_logits`, and so does Sampling, the softmax of
those logits (token_analysis.sampling_temperature); the KV cache a
snapshot at the stage's moment that kv-range and kv-head can read
(kv_analysis.snapshot_reason); intermediate tensors a forward pass that
produced a position of the stage, on each side, that holds a tensor of a
point the Session observes in a layer (tensor_analysis.availability). A
rejected turn's analyses are all unavailable, for its rejection. `batches`
are the forward passes that hold observed tensors, one per turn, stage and
graph (tensor_analysis.batches).

`SessionViews` keeps the conversation and the comparisons of each chat's
capture until a turn is added, so GET /api/session and the analysis routes,
which name rows of the comparison, do not compare again; a view is made once
however many ask for it at a time. A view keeps what its Session observes
(tensor_analysis.Observed) and its batches, and each stage's intermediate
tensors once compared (Compared.tensors; a Decode's at both alignments in
one pass, whichever is asked for), or a cell compared alone
(Compared.cell); those count toward its size as they are added, and the
oldest go first when the views outgrow what is kept. A view whose Session's
model could not be read asks again whenever it is asked for, and is made
again once the model reads. A kept view is of a snapshot of the capture
(CaptureIndex.snapshot): a turn added while it is made or read changes the
next view, not this one. Each chat's token pairs (service/pair_values) are
kept with its view, and across its views, for as long as the chat is kept.
"""

import collections
from collections.abc import Callable
import concurrent.futures
import dataclasses
import json
import re
import threading
from typing import Any

from model_debugger.server.analysis import alignment
from model_debugger.server.analysis import kv_analysis
from model_debugger.server.analysis import metrics
from model_debugger.server.analysis import tensor_analysis
from model_debugger.server.analysis import token_analysis
from model_debugger.server.capture import capture_index
from model_debugger.server.service import pair_values
from model_debugger.server.session import model_server
from model_debugger.server.session import session_store

# A chat-template marker such as `<|turn>`, `<turn|>` or `<|channel>`. A
# plain `<b>` is text: it is a marker only with the bar.
_MARKER = re.compile(r'<\|[^\s<>|]*>|<[^\s<>|]*\|>')
_OPENER = re.compile(r'<\|[^\s<>|]*>')
# Gemma's thinking channel: the reply's tokens from `<|channel>` to
# `<channel|>` are its reasoning, as the runtime's jsonl shows them.
_THINKING = ('<|channel>', '<channel|>')
# Where a token's text comes from when its own source names none: the
# runner's tokenization of the same id in a prompt of this side.
_TEXT_BASIS = 'runner prompt tokens'
# How many chats' comparisons SessionViews keeps.
KEPT_VIEWS = 16
# About how many bytes of views, with their pairs, SessionViews keeps; the
# last one asked for is kept whatever its size.
KEPT_BYTES = 512 * 2**20
# About how many bytes a view holds per token of its conversation, with the
# rows and positions that name it: 400 to 600, measured on synthetic chats of
# up to 40 turns of 2,200 tokens a side (a 107 MB view), and 250 to 300 more
# for the texts of its steps (`_steps`), measured on a synthetic chat and the
# kept E2B captures.
_TOKEN_BYTES = 900
# About how many bytes a stage's intermediate tensors hold per cell, kept:
# 1,600 to 2,200, measured on the kept E2B captures (CPU and GPU).
_CELL_BYTES = 2200
# The KV snapshot of each stage: after its Prefill, at the end of generation.
STAGE_MOMENTS = {'prefill': 'prefill_post', 'decode': 'terminal'}
# What the words of a reason call the KV snapshot of each stage's moment.
_MOMENT_WORDS = {
    'prefill_post': 'after Prefill',
    'terminal': 'at the end of generation',
}
# The analyses a stage offers, by their ids in the metric catalog.
ANALYSES = ('final_logits', 'sampling', 'kv', 'intermediate_tensors')
# What each analysis that compares the two sides at a column compares, as
# the words of its context reason (token_analysis.context_reason) name it.
# Sampling shows each side's own distribution.
_COMPARED_VALUES = {
    'final_logits': 'logits',
    'kv': 'K and V',
    'intermediate_tensors': 'tensors',
}
# The reason codes of this module, and their words as the metric catalog
# lists them.
KV_NOT_CAPTURED = 'kv_not_captured'
KV_NOT_WRITTEN = 'kv_position_not_written'
TURN_REJECTED = 'turn_rejected'
TURN_UNREADABLE = 'turn_unreadable'
REASONS = {
    KV_NOT_CAPTURED: 'Comparable KV positions not captured',
    KV_NOT_WRITTEN: (
        'The KV snapshot of the stage holds no such position: '
        'no step had written it yet.'
    ),
    TURN_REJECTED: 'The capture index rejected the turn as it was added.',
    TURN_UNREADABLE: 'The turn could not be read.',
}
# The words of Final Logits missing from a stage.
_NO_FINAL_LOGITS = 'Final logits not captured'


@dataclasses.dataclass(frozen=True)
class Compared:
  """A chat's conversation and comparisons, made from one capture snapshot.

  Attributes:
      index: The snapshot of the chat's capture they were made from.
      conversation: Each side's `CapturedConversation` of each read turn, by
        (turn, side).
      comparisons: Each read turn's `comparison`, by turn.
      stages: Each read turn's `stages`, by turn.
      view_bytes: About how many bytes the conversation and comparisons hold.
      pairs: The chat's token pairs (pair_values.PairValues), which the chat's
        next view keeps too (SessionViews).
      observed: What the chat's Session observes.
      batches: The UI's batches (tensor_analysis.batches).
  """

  index: capture_index.CaptureIndex
  conversation: dict[tuple[int, str], dict[str, Any]]
  comparisons: dict[int, dict[str, Any]]
  stages: dict[int, dict[str, Any]]
  view_bytes: int
  pairs: pair_values.PairValues = dataclasses.field(
      default_factory=pair_values.PairValues, repr=False, compare=False
  )
  observed: tensor_analysis.Observed = dataclasses.field(
      default_factory=tensor_analysis.nothing, repr=False, compare=False
  )
  batches: list[Any] = dataclasses.field(
      default_factory=list, repr=False, compare=False
  )
  _columns: dict[str, Any] = dataclasses.field(
      default_factory=dict, init=False, repr=False, compare=False
  )
  # The intermediate tensors kept, by key, oldest asked for first; their
  # sizes once made.
  _kept: collections.OrderedDict[Any, Any] = dataclasses.field(
      default_factory=collections.OrderedDict,
      init=False,
      repr=False,
      compare=False,
  )
  _kept_bytes: dict[str, Any] = dataclasses.field(
      default_factory=dict, init=False, repr=False, compare=False
  )
  _listeners: list[Any] = dataclasses.field(
      default_factory=list, init=False, repr=False, compare=False
  )
  _lock: threading.Lock = dataclasses.field(
      default_factory=threading.Lock, init=False, repr=False, compare=False
  )

  @property
  def size(self) -> int:
    """About how many bytes the view holds, but for `pairs`."""
    with self._lock:
      return self.view_bytes + sum(self._kept_bytes.values())

  def on_growth(self, listener: Callable[[], None]) -> None:
    """Calls `listener` whenever intermediate tensors are kept."""
    self._listeners.append(listener)

  def shed(self, excess: int) -> int:
    """Drops the oldest intermediate tensors kept, but the last asked for.

    Args:
        excess: How many bytes to drop, about.

    Returns:
        How many bytes were dropped.
    """
    freed = 0
    with self._lock:
      for key in list(self._kept)[:-1]:
        if freed >= excess:
          break
        if key in self._kept_bytes:  # Not while it is made.
          del self._kept[key]
          freed += self._kept_bytes.pop(key)
    return freed

  def columns(self, n: int, stage: str, kind: str) -> kv_analysis.Alignment:
    """Returns the KV columns of a stage's rows, made once per view.

    Args:
        n: A compared turn.
        stage: 'prefill' or 'decode'.
        kind: The alignment, 'content' or 'steps' ('content' for a Prefill).

    Raises:
        KeyError: The turn, or its stage's alignment, was not compared.
        ValueError: As kv_analysis.Alignment.of.
    """
    key = n, stage, kind
    columns = self._columns.get(key)
    if columns is None:
      comparison = self.comparisons[n]
      rows = (
          comparison['prefill']
          if stage == 'prefill'
          else comparison['alignments'][kind]
      )
      columns = kv_analysis.Alignment.of(rows['positions'])
      self._columns[key] = columns
    return columns

  def tensors(
      self, n: int, stage: str, kind: str | None
  ) -> tensor_analysis.Stage:
    """Returns a stage's intermediate tensors, compared once while kept.

    A Decode's, at an alignment, are compared at the turn's other
    alignment too, in the same pass (tensor_analysis.stages), and both
    kept: the UI asks for both. Whoever asks while they are compared
    waits for them, and gets the error if comparing fails.

    Args:
        n: A compared turn.
        stage: 'prefill' or 'decode'.
        kind: The alignment whose columns they are read at ('content' or
          'steps'), or None: a column is the position on both sides.
    """
    kinds = [kind] + self._other_alignments(n, stage, kind)

    def make(wanted: list[Any]) -> list[Any]:
      return tensor_analysis.stages(
          self.index,
          n,
          stage,
          self.observed,
          [self._alignment(n, stage, k) for k in wanted],
      )

    return self._keep_together(
        [('stage', n, stage, k) for k in kinds],
        lambda keys: make([key[-1] for key in keys]),
        lambda found: _CELL_BYTES * max(1, len(found.cells)),
    )

  def _other_alignments(
      self, n: int, stage: str, kind: str | None
  ) -> list[str]:
    """The turn's other alignments of a Decode whose columns are known."""
    if stage != 'decode' or kind is None:
      return []
    others = []
    for other in self.comparisons[n].get('alignments', {}):
      if other == kind:
        continue
      try:
        self.columns(n, stage, other)
      except (KeyError, ValueError):
        continue
      others.append(other)
    return others

  def cell(
      self,
      n: int,
      stage: str,
      kind: str | None,
      column: int,
      layer: int,
      anchor: str,
  ) -> tensor_analysis.Cell:
    """Returns one cell of a stage's intermediate tensors.

    The kept stage's, if it is compared; else the cell alone, compared
    once while kept: asking for one cell never compares the stage.

    Args:
        n: A compared turn.
        stage: 'prefill' or 'decode'.
        kind: As `tensors`'s.
        column: The cell's column.
        layer: The cell's layer.
        anchor: The cell's anchor.

    Raises:
        ValueError: The stage has no such cell.
    """
    with self._lock:
      whole = self._kept.get(('stage', n, stage, kind))
    if whole is not None and whole.done() and not whole.exception():
      return whole.result().cell(column, layer, anchor)
    return self._keep(
        ('cell', n, stage, kind, column, layer, anchor),
        lambda: tensor_analysis.cell(
            self.index,
            n,
            stage,
            self.observed,
            self._alignment(n, stage, kind),
            column,
            layer,
            anchor,
        ),
        lambda found: _CELL_BYTES,
    )

  def _alignment(
      self, n: int, stage: str, kind: str | None
  ) -> kv_analysis.Alignment | None:
    return None if kind is None else self.columns(n, stage, kind)

  def _keep(
      self,
      key: tuple[Any, ...],
      make: Callable[..., Any],
      size: Callable[..., Any],
  ):
    """Makes what `key` names once while it is kept, and keeps it."""
    return self._keep_together([key], lambda keys: [make()], size)

  def _keep_together(
      self,
      keys: list[tuple[Any, ...]],
      make: Callable[[list[tuple[Any, ...]]], list[Any]],
      size: Callable[..., Any],
  ):
    """Makes what `keys` name together, once each while kept; the first's.

    Args:
      keys: What is asked for first, then what is made with it.
      make: Makes what the keys it is given name, in their order: the first and
        those of the others neither kept nor being made.
      size: The bytes one of them holds.

    Returns:
      The value made for the first key.
    """
    with self._lock:
      making = self._kept.get(keys[0])
      if making is not None:
        self._kept.move_to_end(keys[0])
        mine = {}
      else:
        # The one asked for last: the last to be shed.
        mine = {
            key: concurrent.futures.Future()
            for key in reversed(keys)
            if key not in self._kept
        }
        self._kept.update(mine)
    if not mine:
      return making.result()
    wanted = [key for key in keys if key in mine]
    try:
      found = dict(zip(wanted, make(wanted), strict=True))
    except BaseException as error:
      with self._lock:
        for key, future in mine.items():
          if self._kept.get(key) is future:
            del self._kept[key]
      for future in mine.values():
        future.set_exception(error)
      raise
    with self._lock:
      for key, future in mine.items():
        if self._kept.get(key) is future:
          self._kept_bytes[key] = size(found[key])
    for key, future in mine.items():
      future.set_result(found[key])
    for listener in list(self._listeners):
      listener()
    return found[keys[0]]


def compared(
    index: capture_index.CaptureIndex,
    pairs: pair_values.PairValues | None = None,
    observed: tensor_analysis.Observed | None = None,
) -> Compared:
  """Returns the conversation and comparisons of a chat's capture as it is.

  Args:
      index: The chat's capture.
      pairs: The chat's token pairs so far; None: none yet.
      observed: What the chat's Session observes; None: nothing.
  """
  index = index.snapshot()
  names = {side: _names(index, side) for side in capture_index.SIDES}
  conversation = {}
  for turn in index.turns:
    for side in capture_index.SIDES:
      conversation[(turn['n'], side)] = _conversation(
          index, turn, side, names[side]
      )
  tokens = sum(
      len(side['input_tokens']) + len(side['tokens'])
      for side in conversation.values()
  )
  observed = tensor_analysis.nothing() if observed is None else observed
  return Compared(
      index,
      conversation,
      _comparisons(index, conversation, observed),
      {turn['n']: _stages(index, turn, observed) for turn in index.turns},
      tokens * _TOKEN_BYTES,
      pair_values.PairValues() if pairs is None else pairs,
      observed,
      tensor_analysis.batches(index, observed),
  )


@dataclasses.dataclass
class _Chat:
  """What SessionViews keeps of one chat.

  Attributes:
      index: The capture index the view was asked for.
      pairs: The chat's token pairs, kept across its views.
      view: The chat's last view, or None before one is made.
      making: The view being made, while one is.
  """

  index: capture_index.CaptureIndex
  pairs: pair_values.PairValues
  view: Compared | None = None
  making: concurrent.futures.Future[Any] | None = None

  def size(self) -> int:
    return (0 if self.view is None else self.view.size) + self.pairs.size()


class SessionViews:
  """The `Compared` of each chat, kept until its capture changes."""

  def __init__(
      self,
      kept: int = KEPT_VIEWS,
      kept_bytes: int = KEPT_BYTES,
      observed: Callable[[str], tensor_analysis.Observed] | None = None,
      pieces: token_analysis.Pieces | None = None,
  ) -> None:
    """Keeps the views of the last chats asked for.

    Args:
        kept: At most this many views.
        kept_bytes: About at most this many bytes of views (Compared.size) and
          their chats' pairs (PairValues.size); the last view asked for is kept
          whatever its size.
        observed: What the Session of a record id observes, asked as its view is
          made; None: no Session observes anything.
        pieces: The tokenizer pieces of a vocabulary by its digest, that label
          the chats' candidate tokens (PairValues); None: only the texts each
          chat produced.
    """
    self._observed = observed or (lambda record_id: None)
    self._pieces = pieces
    self._kept = kept
    self._kept_bytes = kept_bytes
    self._lock = threading.Lock()
    # By (record id, chat number).
    self._chats: collections.OrderedDict[tuple[str, int], _Chat] = (
        collections.OrderedDict()
    )
    # Counts forget(): a view made before the last one is not kept.
    self._generation = 0

  def forget(self) -> None:
    """Makes every chat's view again when it is next asked for.

    What a Session observes changed outside its capture: a pair made by
    hand of its models was saved or deleted (mappings).
    """
    with self._lock:
      self._generation += 1
      for chat in self._chats.values():
        chat.view = None

  def compared(
      self, record_id: str, n: int, index: capture_index.CaptureIndex
  ) -> Compared:
    """Returns a chat's `Compared`, made again if its capture changed.

    One view is made at a time per chat: whoever asks while it is made
    waits for it, and gets its error if making it fails.

    Args:
        record_id: The chat's Session.
        n: The chat's number.
        index: The chat's capture.
    """
    key = (record_id, n)
    while True:
      with self._lock:
        chat = self._chats.get(key)
        if chat is None or chat.index is not index:
          chat = _Chat(index, pair_values.PairValues(self._pieces))
          self._chats[key] = chat
        self._chats.move_to_end(key)
        view = chat.view
        if view is not None and _current(chat.index, view, index):
          if not self._unreadable(view):
            self._evict()
            return view
          recheck = view
        else:
          recheck = None
        if recheck is None:
          making = chat.making
          if making is None:
            making = chat.making = concurrent.futures.Future()
            generation = self._generation
            break
      if recheck is not None:
        # Its model could not be read: ask again, cheaply, and make
        # the view again once it reads.
        if self._unreadable_still(record_id):
          return recheck
        with self._lock:
          if chat.view is recheck:
            chat.view = None
        continue
      view = making.result()
      if _current(index, view, index):
        return view
    try:
      view = compared(index, chat.pairs, self._observed(record_id))
    except BaseException as error:
      with self._lock:
        chat.making = None
      making.set_exception(error)
      raise
    chat.pairs.keep_turns(view.index)
    view.on_growth(self._grew)
    with self._lock:
      chat.making = None
      # One made while forget() was called may hold what changed.
      chat.view = view if generation == self._generation else None
      self._evict()
    making.set_result(view)
    return view

  def _unreadable(self, view: Compared) -> bool:
    reason = view.observed.reason
    return (
        reason is not None
        and reason['code'] == tensor_analysis.MODEL_UNREADABLE
    )

  def _unreadable_still(self, record_id: str) -> bool:
    """Whether the Session's model still cannot be read."""
    reason = (self._observed(record_id) or tensor_analysis.nothing()).reason
    return (
        reason is not None
        and reason['code'] == tensor_analysis.MODEL_UNREADABLE
    )

  def _grew(self) -> None:
    with self._lock:
      self._evict()

  def _evict(self) -> None:
    """Drops the chats asked for longest ago, beyond what is kept.

    Under the lock. The last chat asked for is kept whatever its size,
    but for its oldest intermediate tensors (Compared.shed); the pairs and
    the tensors a chat kept since are counted too.
    """
    size = sum(chat.size() for chat in self._chats.values())
    while len(self._chats) > 1 and (
        len(self._chats) > self._kept or size > self._kept_bytes
    ):
      _, dropped = self._chats.popitem(last=False)
      size -= dropped.size()
    if size > self._kept_bytes and self._chats:
      last = next(reversed(self._chats.values()))
      if last.view is not None:
        last.view.shed(size - self._kept_bytes)

  def session(
      self,
      record_id: str,
      n: int,
      index: capture_index.CaptureIndex,
      record: dict[str, Any],
  ) -> dict[str, Any]:
    """Returns the UI's `Session` of a chat (`session_view`), as kept.

    Args:
        record_id: The chat's Session.
        n: The chat's number.
        index: The chat's capture.
        record: The Session's record.
    """
    return _session(
        record,
        session_store.chat(record, n),
        self.compared(record_id, n, index),
    )


def _current(
    asked: capture_index.CaptureIndex,
    view: Compared,
    index: capture_index.CaptureIndex,
) -> bool:
  """Whether a view of `asked` is of `index` as it is now.

  Turns are only appended, and dropped again only when add_turn cannot
  write the index: the capture is unchanged while its last entries are.

  Args:
    asked: The capture index requested for the view.
    view: The Compared view to check.
    index: The current capture index.

  Returns:
    True if the view matches the current index, False otherwise.
  """
  return (
      asked is index
      and _same_entries(view.index.turns, index.turns)
      and _same_entries(view.index.rejected, index.rejected)
  )


def _same_entries(
    kept: list[dict[str, Any]], now: list[dict[str, Any]]
) -> bool:
  return len(kept) == len(now) and (not kept or kept[-1] is now[-1])


def session_view(
    index: capture_index.CaptureIndex,
    record: dict[str, Any],
    chat: dict[str, Any],
) -> dict[str, Any]:
  """Returns the UI's `Session` for a chat's capture.

  Args:
      index: The chat's capture.
      record: The chat's Session's record (name, model, runs).
      chat: The chat, as the record holds it (generation, notice).
  """
  return _session(record, chat, compared(index))


def _session(
    record: dict[str, Any], chat: dict[str, Any], view: Compared
) -> dict[str, Any]:
  """Returns the UI's `Session` of a view, its turns those of its snapshot.

  Of the Session `record`, its chat `chat`: its number (`chat`), the
  settings it runs with (`generation`) and why it is over (`notice`).

  Args:
    record: The session record dictionary.
    chat: The chat metadata dictionary.
    view: The Compared view for the session.

  Returns:
    The session representation dictionary for the UI.
  """
  index = view.index
  conversation, comparisons = view.conversation, view.comparisons
  return {
      'model': record['model'],
      'runs': [
          _run(index.sides[side], record, side) for side in capture_index.SIDES
      ],
      'turns': (
          [
              {
                  **_turn(turn),
                  'comparison': comparisons[turn['n']],
                  'stages': view.stages[turn['n']],
              }
              for turn in index.turns
          ]
          + [_rejected_turn(turn) for turn in index.rejected]
      ),
      'batches': view.batches,
      'notice': model_server.notice(chat, index),
      'name': record['name'],
      'created_at': record['created_at'],
      'capture_id': record['id'],
      'chat': chat['n'],
      'generation': chat['generation'],
      'conversation': list(conversation.values()),
  }


def _comparisons(
    index: capture_index.CaptureIndex,
    conversation: dict[str, Any],
    observed: tensor_analysis.Observed,
) -> dict[int, dict[str, Any]]:
  """Returns each read turn's `comparison`, by turn.

  Args:
      index: The chat's capture.
      conversation: Each side's `CapturedConversation` of each read turn, by
        (turn, side).
      observed: What the chat's Session observes.
  """
  comparisons = {}
  stages = {kind: [] for kind in alignment.ALIGNMENTS}
  histories = alignment.histories(index)
  diverged = alignment.divergences(index)
  for turn in index.turns:
    n = turn['n']
    ref, target = (conversation[n, side] for side in capture_index.SIDES)
    decode_start = _turn(turn)['step_start']
    prefill = alignment.compare_prefill(
        ref['input_tokens'], target['input_tokens']
    )
    prefill['positions'] = alignment.positions(
        prefill['rows'],
        ref['input_tokens'],
        target['input_tokens'],
        decode_start - len(ref['input_tokens']),
    )
    prefill['positions']['columns'] = _columns(
        index, turn, 'prefill', prefill, observed, diverged[n]
    )
    alignments = {}
    for kind in alignment.ALIGNMENTS:
      alignments[kind] = alignment.compare(
          ref['tokens'], target['tokens'], kind
      )
      alignments[kind]['positions'] = alignment.positions(
          alignments[kind]['rows'],
          ref['tokens'],
          target['tokens'],
          decode_start,
      )
      alignments[kind]['positions']['columns'] = _columns(
          index, turn, 'decode', alignments[kind], observed, diverged[n]
      )
    comparisons[n] = {
        'history': histories[n],
        'prefill': prefill,
        'alignments': alignments,
    }
    for kind, comparison in alignments.items():
      stages[kind] += [
          (
              prefill['rows'],
              len(ref['input_tokens']),
              len(target['input_tokens']),
          ),
          (comparison['rows'], len(ref['tokens']), len(target['tokens'])),
      ]
  for kind, chat in stages.items():
    summaries = alignment.stage_summaries(chat)
    for n, prefill, decode in zip(comparisons, summaries[::2], summaries[1::2]):
      comparisons[n]['alignments'][kind]['stages'] = {
          'prefill': prefill,
          'decode': decode,
      }
  return comparisons


def _columns(  # pylint: disable=redefined-outer-name
    index: capture_index.CaptureIndex,
    turn: dict[str, Any],
    stage: str,
    compared: dict[str, Any],
    observed: tensor_analysis.Observed,
    diverged: tuple[int, str] | None,
) -> dict[str, Any]:
  """Which of a stage's columns each analysis has: `positions.columns`.

  Args:
      index: The chat's capture.
      turn: The turn, as the index read it.
      stage: 'prefill' or 'decode'.
      compared: The stage's comparison of one alignment, its `positions` made.
      observed: What the chat's Session observes.
      diverged: Where the turn's two sides stop reading the same tokens
        (alignment.divergences).

  Returns:
      Per analysis id, `{start, end, omitted, notes, context}`.
  """
  positions = compared['positions']
  start, pairs = positions['start'], positions['pairs']
  every = {
      'start': start,
      'end': start + len(pairs),
      'omitted': [],
      'notes': [],
  }
  try:
    columns = kv_analysis.Alignment.of(positions)
  except ValueError:
    columns = None
  tensors = (
      tensor_analysis.columns(index, turn['n'], stage, observed, columns)
      if columns is not None
      else {'start': every['start'], 'end': every['end'], 'omitted': []}
  )
  found = {
      'kv': _kv_columns(turn, stage, start, pairs),
      'intermediate_tensors': {**tensors, 'notes': []},
  }
  if stage == 'decode':
    found['final_logits'] = dict(every)
    found['sampling'] = dict(every)
  first = _first_different(start, pairs, diverged)
  for analysis, shown in found.items():
    values = _COMPARED_VALUES.get(analysis)
    if values and first is not None and first < shown['end']:
      shown['context'] = {
          'start': first,
          'reason': token_analysis.context_reason(diverged, values),
      }
    else:
      shown['context'] = None
  return found


def _first_different(
    start: int, pairs: list[Any], diverged: tuple[int, str] | None
) -> int | None:
  """The first column whose two sides read different tokens, or None.

  A column reads every token before its position on each side: different
  ones where the two sides sit at different positions, or one of them
  after the turn's divergence. Columns after it do too, as the rows
  follow each side's positions in order.

  Args:
      start: The stage's first column.
      pairs: Per column, each side's position, None for none.
      diverged: Where the turn's two sides stop reading the same tokens
        (alignment.divergences).

  Returns:
      The index of the first different column, or None if none.
  """
  if diverged is None:
    return None
  position = diverged[0]
  for k, pair in enumerate(pairs):
    present = [p for p in pair if p is not None]
    if len(set(present)) > 1 or any(p > position for p in present):
      return start + k
  return None


def _kv_columns(
    turn: dict[str, Any], stage: str, start: int, pairs: list[Any]
) -> dict[str, Any]:
  """The columns whose position the stage's KV snapshot holds on a side.

  A snapshot holds the positions processed so far, [0, processed): after
  Prefill not the input's last token, which the first decode step reads
  and writes; at the end of generation not the reply's last token, which
  no step reads. A column no side's snapshot holds is not shown.

  Args:
    turn: The turn dictionary.
    stage: 'prefill' or 'decode'.
    start: The stage's first column index.
    pairs: Per column, each side's position, None for none.

  Returns:
    Per analysis id, `{start, end, omitted, notes}`.
  """
  moment = STAGE_MOMENTS[stage]
  processed = {}
  for side in capture_index.SIDES:
    for snapshot in turn['sides'][side]['kv']:
      if snapshot['moment'] == moment:
        processed[side] = snapshot['processed']
  end = start + len(pairs)
  if not processed:
    return {'start': start, 'end': end, 'omitted': [], 'notes': []}

  def held(pair: list[Any]) -> bool:
    return any(
        position is not None
        and side in processed
        and position < processed[side]
        for side, position in zip(capture_index.SIDES, pair)
    )

  kept = [k for k, pair in enumerate(pairs) if held(pair)]
  if not kept:
    return {'start': start, 'end': end, 'omitted': [], 'notes': []}
  first, last = kept[0], kept[-1] + 1
  left_positions = set()
  for pair in pairs[:first] + pairs[last:]:
    for position in pair:
      if position is not None:
        left_positions.add(position)
  left = sorted(left_positions)
  omitted = []
  if left:
    named = ', '.join(str(p) for p in left)
    noun = 'Position' if len(left) == 1 else 'Positions'
    held_to = max(processed.values()) - 1
    if stage == 'prefill':
      why = 'the first decode step reads it and writes its K and V'
    else:
      why = (
          'it is the reply’s last token, which no step reads, so no '
          'step writes its K and V'
      )
    omitted.append(
        metrics.reason(
            KV_NOT_WRITTEN,
            f'{noun} {named} {"is" if len(left) == 1 else "are"} not '
            f'shown: the KV cache {_MOMENT_WORDS[moment]} holds '
            f'positions 0–{held_to}; {why}.',
        )
    )
  return {
      'start': start + first,
      'end': start + last,
      'omitted': omitted,
      'notes': [],
  }


def _run(
    fixed: dict[str, Any], record: dict[str, Any], side: str
) -> dict[str, Any]:
  """Returns a `CapturedRun`: the record's run with what really ran."""
  run = next(run for run in record['runs'] if run['id'] == side)
  return {
      **run,
      'runtime': fixed['runtime']['name'],
      'backend': fixed['load']['device'],
      'device': fixed['device_id'],
      'contextLength': str(fixed['load']['context_length']),
      'provenance': {
          'runtime_version': fixed['runtime']['version'],
          'device_name': fixed['device_name'],
          'host': fixed['host'],
      },
  }


def _turn(turn: dict[str, Any]) -> dict[str, Any]:
  """Returns a `CapturedTurn` in the reference side's positions.

  The two sides of a turn may sit at different positions when their
  histories differ; the turn takes one origin, the reference side's, and
  each token its own side's position.

  Args:
    turn: The turn dictionary from the capture index.

  Returns:
    A dictionary representing the captured turn.
  """
  ref = turn['sides']['ref']
  return {
      'n': turn['n'],
      'step_start': ref['output'][0]['position'],
  }


def _rejected_turn(turn: dict[str, Any]) -> dict[str, Any]:
  """Returns the `CapturedTurn` of a rejected turn: why, and no tokens."""
  rejected = turn['rejected']
  code = TURN_UNREADABLE if rejected['on_read'] else TURN_REJECTED
  text = capture_index.rejection(turn)
  # The text names the side already: the reason is of the turn.
  why = _unavailable([metrics.reason(code, text)])
  return {
      'n': turn['n'],
      'prompt': turn['prompt'],
      'step_start': None,
      'rejected': {
          'code': code,
          'side': rejected['side'],
          'text': text,
      },
      'stages': {
          stage: {
              'positions': dict.fromkeys(capture_index.SIDES),
              'kv': {'moment': moment, 'context_id': None},
              'analyses': {analysis: why for analysis in ANALYSES},
          }
          for stage, moment in STAGE_MOMENTS.items()
      },
  }


def _stages(
    index: capture_index.CaptureIndex,
    turn: dict[str, Any],
    observed: tensor_analysis.Observed,
) -> dict[str, Any]:
  """Returns a read turn's `stages`, from the capture index alone."""
  sides = turn['sides']
  stages = {}
  for stage, moment in STAGE_MOMENTS.items():
    tokens = 'input' if stage == 'prefill' else 'output'
    snapshots = {}
    for side in capture_index.SIDES:
      for snapshot in sides[side]['kv']:
        if snapshot['moment'] == moment:
          snapshots[side] = snapshot
    stages[stage] = {
        'positions': {
            side: {
                'start': sides[side][tokens][0]['position'],
                'end': sides[side][tokens][-1]['position'] + 1,
            }
            for side in capture_index.SIDES
        },
        'kv': {
            'moment': moment,
            'context_id': (
                kv_analysis.context_id(turn['n'], moment) if snapshots else None
            ),
        },
        'analyses': {
            'final_logits': _final_logits(sides, stage),
            'sampling': _sampling(sides, stage),
            'kv': _kv(snapshots),
            'intermediate_tensors': _tensors(index, turn['n'], stage, observed),
        },
    }
  return stages


def _final_logits(sides: dict[str, Any], stage: str) -> dict[str, Any]:
  """Final Logits: available where a side captured a decode step's logits.

  Of the stages the token routes serve (token_analysis.stage_reason) only.

  Args:
    sides: Dictionary of side information.
    stage: 'prefill' or 'decode'.

  Returns:
    An availability dictionary for Final Logits analysis.
  """
  why = token_analysis.stage_reason(stage)
  if why is not None:
    return _unavailable([why])
  for entry in sides.values():
    for forward in entry['forwards']:
      if forward['phase'] == 'decode' and 'post_logits' in forward['tensors']:
        return _available()
  return _unavailable(
      [metrics.reason(token_analysis.LOGITS_NOT_CAPTURED, _NO_FINAL_LOGITS)]
  )


def _sampling(sides: dict[str, Any], stage: str) -> dict[str, Any]:
  """Sampling: available where Final Logits is.

  A side's Sampling distribution is the softmax of a step's logits
  (token_analysis._sampling_side), so it needs what Final Logits needs.

  Args:
    sides: Dictionary of side information.
    stage: 'prefill' or 'decode'.

  Returns:
    An availability dictionary for Sampling analysis.
  """
  return _final_logits(sides, stage)


def _kv(snapshots: dict[str, dict[str, Any]]) -> dict[str, Any]:
  """The KV cache: available where a side's snapshot can be read.

  By the routes' own rule (kv_analysis.snapshot_reason); a side that has
  none says why.

  Args:
    snapshots: Each side's snapshot of the stage's moment, by side; a side
      without one is absent.

  Returns:
    An availability dictionary for KV cache analysis.
  """
  reasons = []
  for side in capture_index.SIDES:
    if side not in snapshots:
      reasons.append(_reason(KV_NOT_CAPTURED, side))
      continue
    why = kv_analysis.snapshot_reason(snapshots[side], side)
    if why is None:
      return _available()
    reasons.append(why)
  return _unavailable(reasons)


def _tensors(
    index: capture_index.CaptureIndex,
    n: int,
    stage: str,
    observed: tensor_analysis.Observed,
) -> dict[str, Any]:
  """Intermediate tensors: by the routes' own rule (availability)."""
  reasons = tensor_analysis.availability(index, n, stage, observed)
  return _unavailable(reasons) if reasons else _available()


def _reason(code: str, side: str | None = None) -> dict[str, Any]:
  """A reason of the stage, of `code` in the words `REASONS` gives it."""
  return metrics.reason(code, REASONS[code], side)


def _available() -> dict[str, Any]:
  return {'available': True, 'reasons': []}


def _unavailable(reasons: list[dict[str, Any]]) -> dict[str, Any]:
  """An analysis that cannot be added, and why.

  Both sides' reasons, when they say the same, are one reason of both.

  Args:
    reasons: List of reason dictionaries explaining unavailability.

  Returns:
    An unavailability dictionary with the given reasons.
  """
  if len(reasons) == 2 and all(
      reasons[0][key] == reasons[1][key] for key in ('code', 'text')
  ):
    reasons = [{**reasons[0], 'side': None}]
  return {'available': False, 'reasons': reasons}


def _conversation(
    index: capture_index.CaptureIndex,
    turn: dict[str, Any],
    side: str,
    names: dict[int, str],
) -> dict[str, Any]:
  """Returns the `CapturedConversation` of one side of one turn.

  Args:
      index: The chat's capture.
      turn: The turn.
      side: 'ref' or 'target'.
      names: The side's `_names`.
  """
  entry = turn['sides'][side]
  completion = entry['completion']
  stop_ids = {
      token[0]
      for token in index.sides[side]['stop_tokens']
      if isinstance(token, list) and len(token) == 1
  }
  start_token = index.sides[side]['start_token']
  special_ids = stop_ids | ({start_token} if start_token is not None else set())

  # After a rewrite, the input's first token is the one held back after
  # the rewrite's last, and is kinded as following it.
  before = []
  if turn['n'] > 1:
    before = index.turns[turn['n'] - 2]['sides'][side]['rewrite'][-1:]
  kinds = _kinds(before + entry['input'], special_ids)[len(before) :]
  steps = _steps(entry, names)
  input_tokens = [
      _token(token, names, step=step, kind=kind, steps=steps)
      for step, (token, kind) in enumerate(zip(entry['input'], kinds))
  ]
  tokens = []
  thinking = False
  output = entry['output']
  # A stop token of unknown id, as another sampler's is, is the token the
  # next turn carried in at its position: the runner fed it, and named it.
  if turn['n'] < len(index.turns) and output and output[-1]['id'] is None:
    carried = index.turns[turn['n']]['sides'][side]['input'][:1]
    if (
        carried
        and carried[0]['origin'] == 'carried'
        and carried[0]['position'] == output[-1]['position']
        and carried[0]['id'] is not None
    ):
      output = output[:-1] + [{
          **output[-1],
          'id': carried[0]['id'],
          'text': carried[0]['text'],
      }]
  kinds = _kinds(output, special_ids)
  for step, (token, kind) in enumerate(zip(output, kinds)):
    thinking = thinking or token['text'] == _THINKING[0]
    view = _token(token, names, step=step, kind=kind, steps=steps)
    view.update(
        source_forward_id=token['forward'],
        released=token['released'],
        phase='thinking' if thinking else 'response',
    )
    if token.get('stop'):
      view['stop'] = True
    if 'cannot_confirm' in token:
      view['cannot_confirm'] = {
          'text': capture_index.CANNOT_CONFIRM,
          'reason': metrics.reason(
              token_analysis.GREEDY_UNCONFIRMED,
              token['cannot_confirm'],
              side,
          ),
      }
    tokens.append(view)
    thinking = thinking and token['text'] != _THINKING[1]

  conversation = {
      'turn': turn['n'],
      'run': side,
      # What this side was sent, as the capture keeps it.
      'input': capture_index.turn_message(turn, side)['content'],
      'input_tokens': input_tokens,
      'input_token_count': len(input_tokens),
      'input_messages': alignment.messages(input_tokens),
      'output': completion['content'],
      'tokens': tokens,
      'stop_reason': entry['stop_reason'],
      'stop_text': capture_index.ending(index.sides[side], entry),
  }
  if completion['reasoning_content']:
    conversation['thinking'] = completion['reasoning_content']
  if turn['n'] == 1:
    conversation['messages'] = [
        dict(message) for message in index.sides[side]['chat']['messages']
    ]
  return conversation


def _names(index: capture_index.CaptureIndex, side: str) -> dict[int, str]:
  """Returns the text the runner gave each id this side's prompts fed.

  The prompt's tokens first, then those of any rewrite after a reply.

  Args:
    index: The capture index.
    side: The side name ('ref' or 'target').

  Returns:
    Mapping from token ID to text string.
  """
  names = {}
  for turn in index.turns:
    entry = turn['sides'][side]
    tokens = [t for t in entry['input'] if t['origin'] == 'prompt']
    for token in tokens + entry['rewrite']:
      if token['id'] is not None and token['text'] is not None:
        names.setdefault(token['id'], token['text'])
  return names


def _token(
    token: dict[str, Any],
    names: dict[int, str],
    *,
    step: int,
    kind: str,
    steps: dict[int, tuple[str, str]],
) -> dict[str, Any]:
  """Returns a `CapturedToken`, its missing text named by the prompts.

  A token whose id the runner did not name (a sampled chat's stop token, or
  what the runtime fed of its own rendering of earlier turns) has no `id`.
  Its `produced_text` and `written_text` are its side's `_steps`.

  Args:
    token: Raw token dictionary.
    names: Mapping from token ID to prompt text.
    step: Step index of the token.
    kind: Classification kind of the token.
    steps: Mapping from position to (produced_text, written_text).

  Returns:
    A CapturedToken dictionary representation.
  """
  produced, written = steps[token['position']]
  view = {
      'kind': kind,
      'text': token['text'],
      'step': step,
      'position': token['position'],
      'produced_text': produced,
      'written_text': written,
  }
  if token['id'] is not None:
    view['id'] = token['id']
    if token['text'] is None and token['id'] in names:
      view.update(text=names[token['id']], text_basis=_TEXT_BASIS)
  return view


def _steps(
    entry: dict[str, Any], names: dict[int, str]
) -> dict[int, tuple[str, str]]:
  """Says which step each position of a side's turn is, in the UI's words.

  A column of Final Logits and of the intermediate tensors shows the step
  that produced its token (tensor_analysis.produced, the decode step Final
  Logits reads at a Decode column); a column of the KV cache, the step
  that wrote its token's K and V, the one that fed it (capture_index.feeds).

  Args:
      entry: The side's entry of a derived turn.
      names: The side's `_names`.

  Returns:
      Per position of the turn's input and output, the text of the step
      that produced its token and of the step that wrote its K and V. A
      step is named by its phase as well as its label: the input's last
      prefill chunk shares its label with the first decode step.
  """
  shown = {
      token['position']: _shown(token, names)
      for token in entry['input'] + entry['output']
  }
  producers = tensor_analysis.produced(entry)
  feeds = capture_index.feeds(entry)
  found = {}
  for position, token in shown.items():
    producer = producers.get(position)
    if producer is None:
      produced = f'No step of this turn produces {token}'
    elif producer.forward['phase'] == 'decode':
      produced = (
          f'Decode step {producer.forward["step"]} · reads '
          f'{shown[producer.position]} · produces {token}'
      )
    else:
      produced = (
          f'Prefill step {producer.forward["step"]}, row '
          f'{producer.row} · reads {shown[producer.position]} · next '
          f'token {token}'
      )
    feed = feeds.get(position)
    if feed is None:
      written = f'No step of this turn reads {token}'
    elif feed.forward['phase'] == 'decode':
      written = (
          f'Written at decode step {feed.forward["step"]}, when the '
          f'model reads {token}'
      )
    else:
      written = (
          f'Written at prefill step {feed.forward["step"]}, row '
          f'{feed.row}, when the model reads {token}'
      )
    found[position] = (produced, written)
  return found


def _shown(token: dict[str, Any], names: dict[int, str]) -> str:
  """A token as a step's text names it: its text quoted, else its id."""
  text = token['text']
  if text is None and token['id'] is not None:
    text = names.get(token['id'])
  if text is not None:
    return json.dumps(text, ensure_ascii=False)
  if token['id'] is not None:
    return f'token {token["id"]}'
  return 'a token the runner did not name'


def _is_marker(text: str | None) -> bool:
  return text is not None and _MARKER.fullmatch(text) is not None


def _kinds(tokens: list[dict[str, Any]], special_ids: set[int]) -> list[str]:
  """Classifies tokens as the UI groups them into message rows.

  `special`: the start and stop tokens and chat-template markers;
  `template`: what the runtime fed of its own rendering of earlier turns,
  the word after an opening marker (a role, or `thought`), and the newline
  after that word or after a closing marker; `text`: everything else.

  Args:
    tokens: List of token dictionaries to classify.
    special_ids: Set of special token IDs (start, stop, etc.).

  Returns:
    List of classification kind strings ('special', 'template', or 'text').
  """
  kinds = []
  for index, token in enumerate(tokens):
    text = token['text']
    before = tokens[index - 1]['text'] if index else None
    if token['id'] in special_ids or _is_marker(text):
      kinds.append('special')
    elif token.get('origin') == 'history':
      kinds.append('template')
    elif (
        before is not None
        and _OPENER.fullmatch(before)
        and text is not None
        and text.strip().isalpha()
    ):
      kinds.append('template')
    elif (
        text == '\n'
        and index
        and (
            kinds[-1] == 'special'
            or (kinds[-1] == 'template' and before != '\n')
        )
    ):
      kinds.append('template')
    else:
      kinds.append('text')
  return kinds
