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

"""Per-token comparison of ref and target: token-analysis and final logits.

A pair names one generated token on each side by its ordinal in the turn's
`output` (its `step`); the service finds the pair a request names by a row of
its comparison, of a stage of TOKEN_STAGES (stage_reason). Each token
resolves to the decode forward whose logits produced it, and that forward's
`post_logits` and `post_activations` files are what is compared. Two logits
vectors are compared only when they are of one vocabulary (as
vocabulary_identities decides: one model file, or two models whose
tokenizers have one id -> piece table) and of one length; otherwise each
side still shows its own distribution, without deltas. A pair's `context`
says whether both sides fed the same tokens before it: `same`, `different`,
or `unknown` where a token the runner did not name could make the contexts
either (same_context). Values read after different contexts still compare,
as every analysis's do: the pair's `notes` say where the two sides' tokens
first differ (context_reason, the one rule).

Whatever is missing or not compared says why as reasons (metrics.reason),
each of one side or of the pair, with a code of `REASONS`.

token-range bins one metric of a stage's rows by their columns, the
statistics of the Line chart and of the details (analysis/statistics),
with the stage's peak KL; it reads what token-analysis gives of each pair
(`range_values`), which the service keeps (service/pair_values).

A pair's whole logits vectors never go to the UI: final_logits_summary says
what is known of them (their statistics among it), final_logits pages them
sorted, and final_logits_bins and final_logits_points give what the
Inspection's line chart and scatter plot draw of them
(analysis/vector_drawing).
"""

import array
import bisect
from collections.abc import Callable, Iterable, Iterator, Sequence
import dataclasses
import itertools
import re
import types
from typing import Any

from model_debugger.server.analysis import comparability
from model_debugger.server.analysis import metrics
from model_debugger.server.analysis import statistics
from model_debugger.server.analysis import vector_drawing
from model_debugger.server.capture import capture_index
from model_debugger.server.capture import tensor_file
import numpy as np

MAX_PAIRS = 128
MAX_FINAL_LOGITS_ROWS = 256
# The most bins a line chart of final logits asks for.
MAX_LINE_BINS = 8192
TRANSFORMS = ('raw', 'centered')
ORDERS = ('index', 'ref', 'target', 'delta', 'absolute_delta')
DIRECTIONS = ('ascending', 'descending')
# The id of a fed token the runner did not name, in a `Context`.
UNNAMED = -1
# Candidate rows of a distribution: each side's most likely ids.
CANDIDATES = 64
# The stages whose rows the token analyses (token-analysis, final logits,
# Sampling) serve: a Decode's rows name generated tokens, each produced by its
# step's logits. A Prefill feeds its input and samples no token
# (stage_reason).
TOKEN_STAGES = ('decode',)
# The metrics token-range bins, by the analysis that shows them: the metric
# catalog's lists of them.
RANGE_METRICS = {
    'final_logits': ('rmse', 'max_abs', 'kl', 'js'),
    'sampling': ('selected_probability', 'entropy', 'margin'),
}
MAX_RANGE_BINS = 1024
# What a side's Sampling distribution is (_sampling_side): the softmax of its
# whole vocabulary's logits at its temperature.
SAMPLING_BASIS = 'logits_softmax'
# The sampler settings a side's sampling distribution reports.
SAMPLER_KEYS = ('temperature', 'top_k', 'top_p', 'seed')
# The numbers token-range reads of a pair (RangeValues), by name: the pair's
# metrics of RANGE_METRICS['final_logits'], then each side's own of
# RANGE_METRICS['sampling'] as `{side}.{metric}`.
RANGE_NUMBERS = RANGE_METRICS['final_logits'] + tuple(
    f'{side}.{metric}'
    for side, metric in itertools.product(
        capture_index.SIDES, RANGE_METRICS['sampling']
    )
)
_NUMBER = {name: k for k, name in enumerate(RANGE_NUMBERS)}

# What a pair's `context` says: the two sides fed the same tokens before it,
# different ones, or the captures cannot tell (same_context).
SAME, DIFFERENT, UNKNOWN = 'same', 'different', 'unknown'

# The reason codes of this module, and their words as the metric catalog
# lists them; a response's reason carries its own words, which may name more.
CONTEXT_DIFFERENT = 'context_different'
CONTEXT_UNKNOWN = 'context_unknown'
VOCABULARY_MISMATCH = 'vocabulary_mismatch'
LOGITS_NOT_PAIRED = 'logits_not_paired'
TOKEN_NOT_CAPTURED = 'token_not_captured'
LOGITS_NOT_CAPTURED = 'logits_not_captured'
ACTIVATIONS_NOT_CAPTURED = 'activations_not_captured'
LAYOUT_UNSUPPORTED = 'layout_unsupported'
NOT_NUMERIC = 'not_numeric'
LOGITS_INVALID = 'logits_invalid'
SELECTED_OUTSIDE_VOCABULARY = 'selected_outside_vocabulary'
STOP_TOKEN_UNNAMED = 'stop_token_unnamed'
GREEDY_UNCONFIRMED = 'greedy_unconfirmed'
FORWARDS_NOT_PAIRED = 'forwards_not_paired'
ACTIVATION_SPACE_MISMATCH = 'activation_space_mismatch'
DIVERGENCE_INFINITE = 'divergence_infinite'
NOT_DECODE = 'not_decode'
REASONS = types.MappingProxyType({
    CONTEXT_DIFFERENT: (
        'After the divergence the two sides read different '
        'tokens; the values still compare.'
    ),
    CONTEXT_UNKNOWN: (
        'The runner did not name a token a side fed, so the two '
        'sides may read different tokens; the values still compare.'
    ),
    VOCABULARY_MISMATCH: (
        'Matching captured vocabulary IDs and sizes are required'
    ),
    LOGITS_NOT_PAIRED: 'Paired full-vocabulary logits not captured',
    TOKEN_NOT_CAPTURED: 'Token not captured on this side',
    LOGITS_NOT_CAPTURED: 'Full-vocabulary logits not captured',
    ACTIVATIONS_NOT_CAPTURED: 'Last hidden-state activation not captured',
    LAYOUT_UNSUPPORTED: (
        'Token output layout is not a single causal-LM position'
    ),
    NOT_NUMERIC: 'Token output is not a numeric vector',
    LOGITS_INVALID: 'Logits contain invalid values',
    SELECTED_OUTSIDE_VOCABULARY: (
        'Selected token ID is outside captured vocabulary'
    ),
    STOP_TOKEN_UNNAMED: 'The runner does not name the sampled stop token',
    GREEDY_UNCONFIRMED: capture_index.CANNOT_CONFIRM,
    FORWARDS_NOT_PAIRED: 'Paired source forwards not captured',
    ACTIVATION_SPACE_MISMATCH: (
        'Matching activation coordinate spaces are not established'
    ),
    DIVERGENCE_INFINITE: (
        'Divergence is infinite: target assigns zero probability'
    ),
    NOT_DECODE: (
        'Only Decode rows name generated tokens, whose logits '
        'produced them; a Prefill samples none'
    ),
})


def _reason(code: str, side: str | None = None) -> dict[str, Any]:
  """A reason of `code` in the words `REASONS` gives it."""
  return metrics.reason(code, REASONS[code], side)


def stage_reason(stage: str) -> dict[str, Any] | None:
  """Why the token analyses do not serve a stage's rows, or None if they do.

  The one rule of the token routes (service/analysis_requests) and of the
  session's availability (service/session_view).

  Args:
      stage: 'prefill' or 'decode'.

  Returns:
      None if the token analyses serve the stage's rows, else a NOT_DECODE
      reason (metrics.reason).
  """
  if stage in TOKEN_STAGES:
    return None
  return _reason(NOT_DECODE)


class UnavailableError(ValueError):
  """A side's token or vector cannot be had; `reason` says why."""

  def __init__(self, reason: dict[str, Any]) -> None:
    """Initializes the error of `reason` (metrics.reason)."""
    super().__init__(reason['text'])
    self.reason = reason


# The old name, still used by service/ and session/ tests.
Unavailable = UnavailableError


def vocabulary_identities(
    index: capture_index.CaptureIndex,
) -> tuple[dict[str, str], str | None]:
  """Each side's vocabulary identity, and why the two differ.

  The one rule is comparability.vocabularies(): two sides that ran one
  model file share its vocabulary: each identity is the file's SHA-256.
  Two that ran different files (one model in two
  runtimes) share one when both models' tokenizers have one id -> piece
  table, as each side's `model.vocabulary` records it (tap/model_facts):
  each identity is then `tokenizer:<the table's SHA-256>`. Otherwise each
  is its own model's SHA-256. A logits vector's identity is its side's and
  its length, `<identity>:<length>` (the vocabulary size).

  Args:
      index: The chat's capture.

  Returns:
      `(identities, why)`: each side's identity, and why they differ, in
      words, when the model files do and the tokenizers cannot show that
      the vocabularies are one; else None.
  """
  found = comparability.vocabularies(*(
      comparability.Model(
          index.sides[side]['model']['sha256'],
          vocabulary=(index.sides[side]['model'].get('vocabulary') or {}).get(
              'table_sha256'
          ),
      )
      for side in capture_index.SIDES
  ))
  if found.differ == comparability.VOCABULARY_UNKNOWN:
    names = ' and '.join(
        capture_index.SIDE_NAMES[side] for side in found.unknown
    )
    return found.identities, (
        'Reference and Target ran different models, and the capture '
        f'records no tokenizer of {names}'
    )
  if found.differ == comparability.VOCABULARY_DIFFERENT:
    return found.identities, (
        'Reference and Target ran different models whose tokenizers '
        'map ids to different pieces'
    )
  return found.identities, None


@dataclasses.dataclass(frozen=True)
class _Token:
  """One side's token of a pair, resolved to the forward that produced it.

  Attributes:
      side: 'ref' or 'target'.
      id: The token's vocabulary id; None for a sampled chat's stop token, which
        the runner does not name.
      forward: The decode forward whose logits produced the token.
      context: Every token the chat fed before it.
      model: The sha256 of the side's model file.
  """

  side: str
  id: int | None
  forward: dict[str, Any]
  context: 'Context'
  model: str


@dataclasses.dataclass(frozen=True)
class Context:
  """The tokens a side had fed before a position, by position.

  Attributes:
      positions: Their positions, increasing.
      ids: Their ids; UNNAMED where the runner did not name the token.
  """

  positions: np.ndarray
  ids: np.ndarray

  def __len__(self) -> int:
    """How many tokens were fed."""
    return len(self.positions)


class Fed:
  """What one side had fed so far, by position; see `fed_turns`."""

  def __init__(self) -> None:
    """Nothing fed yet."""
    self._positions = array.array('q')
    self._ids = array.array('q')

  def put(self, tokens: Sequence[dict[str, Any]]) -> None:
    """Feeds tokens: each replaces whatever was fed at its position."""
    positions, ids = self._positions, self._ids
    for token in tokens:
      position = token['position']
      token_id = UNNAMED if token['id'] is None else token['id']
      if not positions or position > positions[-1]:
        positions.append(position)
        ids.append(token_id)
        continue
      k = bisect.bisect_left(positions, position)
      if positions[k] == position:
        ids[k] = token_id
      else:
        positions.insert(k, position)
        ids.insert(k, token_id)

  def cut(self, position: int) -> None:
    """Forgets everything fed from `position` on."""
    k = bisect.bisect_left(self._positions, position)
    del self._positions[k:]
    del self._ids[k:]

  def before(self, position: int) -> Context:
    """Returns the tokens fed before `position`, a copy later feeds keep."""
    k = bisect.bisect_left(self._positions, position)
    return Context(
        np.frombuffer(self._positions[:k], dtype=np.int64),
        np.frombuffer(self._ids[:k], dtype=np.int64),
    )


def fed_turns(index: capture_index.CaptureIndex, side: str) -> Iterator[Fed]:
  """Yields what a side had fed by each read turn, in one pass.

  The n-th value holds turns 1..n: each turn's input, then what it
  generated; then, for every turn before turn n, its rewrite, which
  replaces everything from its first position on: after a reply that
  thought, the reply without the thoughts. Turn n's own rewrite came after
  its decode steps, so it is applied only before the next value. A later
  turn's input replaces what an earlier output left at a position (the
  carried stop token is both, with one id). One `Fed` is yielded each
  time, changed in place: read it before asking for the next.

  Args:
      index: The chat's capture.
      side: 'ref' or 'target'.

  Yields:
      The side's `Fed` after each turn, the same object each time.
  """
  fed = Fed()
  for turn in index.turns:
    entry = turn['sides'][side]
    fed.put(entry['input'] + entry['output'])
    yield fed
    rewrite = entry['rewrite']
    if rewrite:
      fed.cut(rewrite[0]['position'])
      fed.put(rewrite)


@dataclasses.dataclass(frozen=True)
class _Logits:
  """One side's logits vector and what the UI is told about it."""

  values: np.ndarray
  vocabulary: str
  summary: dict[str, Any]


# A tokenizer piece that stands for one byte (SentencePiece's byte fallback).
_BYTE_PIECE = re.compile(rb'<0x([0-9A-Fa-f]{2})>')
# The tokenizer pieces of a vocabulary by its table's digest (a side's
# model's `vocabulary.table_sha256`), in id order; None where no model file
# here holds it (model_store.ModelStore.pieces).
Pieces = Callable[[str], Sequence[bytes] | None]


def piece_text(piece: bytes) -> str:
  """The text a tokenizer piece stands for, as Gemma's tokenizer decodes it.

  SentencePiece's word boundary U+2581 is a space; a byte piece is its
  byte when that is ASCII, else stays as written (`<0xE4>`), being part of
  a character; any other piece is its UTF-8 text.

  Args:
      piece: The tokenizer piece's bytes.

  Returns:
      The text the piece stands for.
  """
  found = _BYTE_PIECE.fullmatch(piece)
  if found is not None and int(found.group(1), 16) < 0x80:
    return chr(int(found.group(1), 16))
  return piece.decode('utf-8', errors='replace').replace('\u2581', ' ')


class _Labels:
  """Token texts by id: the chat's own, else the side's tokenizer's."""

  def __init__(
      self, texts: dict[int, str], pieces: Sequence[Sequence[bytes]]
  ) -> None:
    self._texts = texts
    self._pieces = pieces

  def get(self, identity: int) -> str | None:
    """The id's text, or None: neither the chat nor a tokenizer has it."""
    text = self._texts.get(identity)
    if text is not None:
      return text
    for pieces in self._pieces:
      if 0 <= identity < len(pieces):
        return piece_text(pieces[identity])
    return None


class _Turn:
  """The turn a request names, read from the capture index."""

  def __init__(
      self,
      index: capture_index.CaptureIndex,
      n: int,
      pieces: Pieces | None = None,
  ) -> None:
    self.index = index
    self.n = n
    self.sides = index.turn(n)['sides']
    self.vocabularies, self._vocabularies_differ = vocabulary_identities(index)
    self._fed = {}
    self._labels = {}
    self._pieces = pieces

  def vocabulary_reason(self, sources: dict[str, _Logits]) -> dict[str, Any]:
    """Why two sides' logits are of two vocabularies (VOCABULARY_MISMATCH).

    Its words say why, where the model files differ: their tokenizers,
    or, of one tokenizer, the logits' lengths.

    Args:
        sources: Each side's logits.

    Returns:
        The VOCABULARY_MISMATCH reason (metrics.reason).
    """
    why = self._vocabularies_differ
    tokenizer = self.vocabularies['ref'].startswith('tokenizer:')
    if why is None and tokenizer and len(sources) == 2:
      sizes = [sources[side].values.size for side in capture_index.SIDES]
      why = (
          'Reference and Target share a tokenizer, but their logits '
          f'hold {sizes[0]} and {sizes[1]} values'
      )
    if why is None:
      return _reason(VOCABULARY_MISMATCH)
    return metrics.reason(
        VOCABULARY_MISMATCH, f'{REASONS[VOCABULARY_MISMATCH]}: {why}.'
    )

  def token(self, side: str, step: int | None) -> _Token:
    """The `step`-th token the side generated in this turn."""
    output = self.sides[side]['output']
    if step is None or step >= len(output):
      raise UnavailableError(_reason(TOKEN_NOT_CAPTURED))
    token = output[step]
    forward = next(
        forward
        for forward in self.sides[side]['forwards']
        if forward['id'] == token['forward']
    )
    return _Token(
        side=side,
        id=token['id'],
        forward=forward,
        context=self.context(side, token['position']),
        model=self.index.sides[side]['model']['sha256'],
    )

  def context(self, side: str, position: int) -> Context:
    """Every token the side fed before `position`, by turn n."""
    if side not in self._fed:
      fed = fed_turns(self.index, side)
      for _ in range(self.n):
        self._fed[side] = next(fed)
    return self._fed[side].before(position)

  def logits(self, token: _Token) -> _Logits:
    """The logits that produced the token, as one float64 vector."""
    values, metadata = self._vector(token, 'post_logits', LOGITS_NOT_CAPTURED)
    # The side's vocabulary (vocabulary_identities); its size is the
    # logits length.
    vocabulary = f'{self.vocabularies[token.side]}:{values.size}'
    summary = {
        'size': int(values.size),
        'source_forward_id': token.forward['id'],
        'reasons': [],
        'logits_metadata': {**metadata, 'vocab_identity': vocabulary},
    }
    return _Logits(values, vocabulary, summary)

  def activations(self, token: _Token) -> np.ndarray:
    """The final hidden state of the forward that produced the token."""
    return self._vector(token, 'post_activations', ACTIVATIONS_NOT_CAPTURED)[0]

  def _vector(
      self, token: _Token, key: str, missing: str
  ) -> tuple[np.ndarray, dict[str, Any]]:
    """The forward's `key` tensor; `missing` is the code of its absence.

    Args:
        token: The token whose forward holds the tensor.
        key: The tensor's key in the forward's `tensors`.
        missing: The reason code when the forward has no such tensor.

    Returns:
        `(vector, metadata)`: the tensor flattened to one float64 vector,
        and its `source_shape`, `source_dtype` and `comparison_shape`.

    Raises:
        UnavailableError: It was not captured, or is not one numeric vector.
    """
    file = token.forward['tensors'].get(key)
    if file is None:
      raise UnavailableError(_reason(missing))
    path = self.index.path(token.side, self.n, file)
    try:
      tensor = tensor_file.read_tensor(path)
      dtype = tensor_file.widened_dtype(path) or str(tensor.dtype)
    except (OSError, ValueError) as error:
      raise UnavailableError(
          metrics.reason(metrics.UNREADABLE, str(error))
      ) from None
    # A decode forward computes one position: every axis but the last is 1.
    if tensor.ndim == 0 or any(size != 1 for size in tensor.shape[:-1]):
      raise UnavailableError(_reason(LAYOUT_UNSUPPORTED))
    if not tensor.size or tensor.dtype.kind not in 'fiu':
      raise UnavailableError(_reason(NOT_NUMERIC))
    metadata = {
        'source_shape': list(tensor.shape),
        'source_dtype': dtype,
        'comparison_shape': [int(tensor.shape[-1])],
    }
    return tensor.reshape(-1).astype(np.float64), metadata

  def labels(self, sides: list[str]) -> _Labels:
    """Token texts of the sides, by id (earlier sides first).

    The texts the chat itself produced (fed tokens, released tokens);
    any other id's is its piece of the side's model's tokenizer, where
    the model's file is here to read it (`pieces`).

    Args:
        sides: The sides whose texts to use, earlier sides first.

    Returns:
        The token texts by id.
    """
    result = {}
    for side in sides:
      if side not in self._labels:
        texts = {}
        for turn in self.index.turns:
          entry = turn['sides'][side]
          tokens = entry['input'] + entry['output'] + entry['rewrite']
          for token in tokens:
            if token.get('text') is not None:
              texts.setdefault(token['id'], token['text'])
        self._labels[side] = texts
      for identity, text in self._labels[side].items():
        result.setdefault(identity, text)
    found = []
    if self._pieces is not None:
      for side in sides:
        vocabulary = self.index.sides[side]['model'].get('vocabulary') or {}
        digest = vocabulary.get('table_sha256')
        pieces = self._pieces(digest) if digest else None
        if pieces:
          found.append(pieces)
    return _Labels(result, found)


def same_context(ref: Context, target: Context) -> bool | None:
  """Whether two sides had fed the same tokens at the same positions.

  Different where the positions differ, or a position holds two named ids
  that differ; else unknown (None) where a side did not name a token, as
  that token could prove either.

  Args:
      ref: What the ref side had fed.
      target: What the target side had fed.

  Returns:
      True if the same, False if different, None if unknown.
  """
  if not np.array_equal(ref.positions, target.positions):
    return False
  named = (ref.ids != UNNAMED) & (target.ids != UNNAMED)
  if np.any(named & (ref.ids != target.ids)):
    return False
  return True if named.all() else None


def divergence(ref: Context, target: Context) -> tuple[int, str] | None:
  """Where two sides' fed tokens stop being proven the same.

  As same_context decides: where they are different, the first position
  at which the positions differ or a position holds two named ids that
  differ, and DIFFERENT; else, where they are unknown, the first position
  a side did not name, and UNKNOWN; None where they are the same.

  Args:
      ref: What the ref side had fed.
      target: What the target side had fed.

  Returns:
      `(position, DIFFERENT or UNKNOWN)`, or None.
  """
  same = same_context(ref, target)
  if same:
    return None
  n = min(len(ref), len(target))
  positions = (ref.positions[:n], target.positions[:n])
  ids = (ref.ids[:n], target.ids[:n])
  if same is None:
    unnamed = (ids[0] == UNNAMED) | (ids[1] == UNNAMED)
    return int(positions[0][np.flatnonzero(unnamed)[0]]), UNKNOWN
  differ = (positions[0] != positions[1]) | (
      (ids[0] != UNNAMED) & (ids[1] != UNNAMED) & (ids[0] != ids[1])
  )
  found = np.flatnonzero(differ)
  if found.size:
    k = found[0]
    return int(min(positions[0][k], positions[1][k])), DIFFERENT
  longer = ref if len(ref) > n else target
  return int(longer.positions[n]), DIFFERENT


def context_reason(
    found: tuple[int, str] | None, values: str
) -> dict[str, Any] | None:
  """Why values the two sides read after different tokens still compare.

  The one rule of every analysis that compares the two sides at a column
  (Final Logits, the KV cache, the intermediate tensors): values read
  after the two sides' contexts differ are compared all the same, and say
  so with this reason, of code CONTEXT_DIFFERENT or CONTEXT_UNKNOWN.

  Args:
      found: The contexts' `divergence`.
      values: What the analysis compares, as the words name it: 'logits', say.

  Returns:
      The reason, its words naming the position; None for `found` None.
  """
  if found is None:
    return None
  position, kind = found
  if kind == DIFFERENT:
    return metrics.reason(
        CONTEXT_DIFFERENT,
        f'After the divergence at position {position} the two sides read '
        f'different tokens; the {values} still compare.',
    )
  return metrics.reason(
      CONTEXT_UNKNOWN,
      f'The runner did not name the token at position {position} on a '
      'side, so from there on the two sides may read different tokens; '
      f'the {values} still compare.',
  )


def _context(tokens: dict[str, _Token]) -> str | None:
  """SAME, DIFFERENT or UNKNOWN (same_context) before the pair; None unpaired.

  Every context's logits are compared; one not SAME says so (_notes).

  Args:
      tokens: Each side's token of the pair.

  Returns:
      SAME, DIFFERENT or UNKNOWN; None unless both sides have a token.
  """
  if len(tokens) != 2:
    return None
  same = same_context(*(tokens[side].context for side in capture_index.SIDES))
  return UNKNOWN if same is None else SAME if same else DIFFERENT


def _notes(tokens: dict[str, _Token], values: str) -> list[dict[str, Any]]:
  """What a pair's values cannot show: that they read different tokens."""
  if len(tokens) != 2:
    return []
  reason = context_reason(
      divergence(*(tokens[side].context for side in capture_index.SIDES)),
      values,
  )
  return [reason] if reason else []


def _side_reason(side: str, error: ValueError) -> dict[str, Any]:
  """The reason of `side` an UnavailableError (or another ValueError) gives."""
  if isinstance(error, UnavailableError):
    return {**error.reason, 'side': side}
  return metrics.reason(metrics.UNREADABLE, str(error), side)


def _compatibility(
    turn: _Turn,
    sources: dict[str, _Logits],
    reasons: dict[str, dict[str, Any]],
) -> tuple[bool, list[dict[str, Any]]]:
  """Whether two logits vectors share a vocabulary, and so compare.

  Their contexts do not matter (context_reason): logits read after
  different tokens compare too.

  Args:
      turn: The pair's turn.
      sources: The logits of each side that has them.
      reasons: Why a side has no logits, by side.

  Returns:
      `(compatible, reasons)`: `compatible` is true when both sides have
      logits of one vocabulary and length; `reasons` say why they are not
      compatible (none when they are), each side's own.
  """
  both = 'ref' in sources and 'target' in sources
  compatible = (
      both
      and sources['ref'].vocabulary == sources['target'].vocabulary
      and sources['ref'].values.shape == sources['target'].values.shape
  )
  if both:
    why = [] if compatible else [turn.vocabulary_reason(sources)]
  else:
    why = list(reasons.values())
  return compatible, why


def _transformed(
    source: _Logits | None, transform: str
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
  if source is None:
    return None, None
  if transform == 'raw':
    return source.values, None
  return metrics.centered(source.values)


def _selection_reasons(
    selected: int | None, known: bool, side: str
) -> list[dict[str, Any]]:
  """Returns why a side's selected token has no probability, if it has none."""
  if selected is not None:
    return []
  if known:
    return [_reason(SELECTED_OUTSIDE_VOCABULARY, side)]
  return [_reason(STOP_TOKEN_UNNAMED, side)]


def _side_summary(
    source: _Logits | None,
    distribution: metrics.Distribution | None,
    token: _Token | None,
    reason: dict[str, Any] | None,
) -> dict[str, Any]:
  """A side's `TokenDistributionSide`; `reason` is why it has no logits."""
  summary = dict(source.summary) if source else {}
  if distribution is None:
    summary['reasons'] = [reason] if reason else []
    return summary
  known = token.id is not None
  selected = token.id if known and 0 <= token.id < distribution.p.size else None
  summary.update(
      entropy=distribution.entropy,
      selected_id=selected,
      selected_rank=(
          int(distribution.ranks[selected]) if selected is not None else None
      ),
      selected_probability=(
          float(distribution.p[selected]) if selected is not None else None
      ),
      margin=distribution.margin,
      reasons=_selection_reasons(selected, known, token.side),
  )
  return summary


def sampling_temperature(options: dict[str, Any]) -> float:
  """The temperature a side's Sampling distribution divides its logits by.

  The side's sampler temperature when it is a positive finite number; else
  1. Greedy sampling (top_k 1) keeps its temperature too: its distribution
  is the logits' softmax all the same.

  Args:
      options: The side's chat options.

  Returns:
      The temperature.
  """
  value = options.get('temperature')
  if (
      isinstance(value, (int, float))
      and not isinstance(value, bool)
      and np.isfinite(value)
      and value > 0
  ):
    return float(value)
  return 1.0


def _sampling_side(
    turn: _Turn,
    side: str,
    token: _Token | None,
    source: _Logits | None,
    distribution: metrics.Distribution | None,
    reason: dict[str, Any] | None,
) -> tuple[dict[str, Any], metrics.Distribution | None]:
  """One side's Sampling distribution: softmax(logits / T), every id.

  The capture holds the logits before sampling and the sampled id, not the
  sampler's own probabilities; the distribution is the softmax of the
  whole vocabulary's logits at the side's temperature
  (sampling_temperature), before top-k, top-p or any other constraint.

  Args:
      turn: The pair's turn.
      side: 'ref' or 'target'.
      token: The side's token, or None.
      source: Its logits, or None.
      distribution: The logits' softmax at temperature 1, reused at that
        temperature; None without logits.
      reason: Why the side has no token or logits.

  Returns:
      `(summary, distribution)`: the `SamplingDistributionSide`, and the
      distribution, None where the side has none.
  """
  options = turn.index.sides[side]['chat']['options']
  sampler = {key: options.get(key) for key in SAMPLER_KEYS}
  temperature = sampling_temperature(options)
  result = {
      **(source.summary if source else {}),
      'entropy': None,
      'selected_probability': None,
      'margin': None,
      'selected_rank': None,
      'candidate_count': None,
      'basis': None,
      'temperature': None,
      'sampler': sampler,
      'reasons': [],
  }
  if token is None or source is None or distribution is None:
    result['reasons'] = [reason or _reason(LOGITS_NOT_CAPTURED, side)]
    return result, None
  if temperature != 1.0:
    try:
      distribution = metrics.distribution(source.values / temperature)
    except ValueError as error:
      result['reasons'] = [metrics.reason(LOGITS_INVALID, str(error), side)]
      return result, None
  result.update(
      _side_summary(source, distribution, token, None),
      basis=SAMPLING_BASIS,
      temperature=temperature,
      candidate_count=int(distribution.p.size),
      sampler=sampler,
  )
  return result, distribution


def _sampling_distribution(
    turn: _Turn,
    tokens: dict[str, _Token],
    sources: dict[str, _Logits],
    distributions: dict[str, metrics.Distribution],
    reasons: dict[str, dict[str, Any]],
    context: str | None,
) -> dict[str, Any]:
  """The pair's `SamplingDistribution`: each side's, and their candidates.

  The rows (_candidate_rows): the sampled ids, then the union of each
  side's CANDIDATES most likely ids, each with both sides' probability and
  rank, of that side's whole-vocabulary distribution.

  Args:
      turn: The pair's turn.
      tokens: Each side's token.
      sources: Each side's logits.
      distributions: Each side's distribution at temperature 1.
      reasons: Why a side has no token or logits, by side.
      context: The pair's `context` (_context).

  Returns:
      The `SamplingDistribution`.
  """
  sides, sampled = {}, {}
  for side in capture_index.SIDES:
    sides[side], found = _sampling_side(
        turn,
        side,
        tokens.get(side),
        sources.get(side),
        distributions.get(side),
        reasons.get(side),
    )
    if found is not None:
      sampled[side] = found
  vocabulary, why = _compatibility(
      turn, {side: sources[side] for side in sampled}, reasons
  )
  # Ids of two different vocabularies are never merged into one row; a
  # single available side still lists its own candidates.
  mergeable = len(sampled) < 2 or vocabulary
  rows = (
      _candidate_rows(turn, tokens, sources, sampled, vocabulary, union=True)
      if mergeable
      else []
  )
  return {
      **sides,
      'context': context,
      'vocabulary_compatible': vocabulary,
      # The rows are the most likely ids, never every id of a vocabulary.
      'rows_complete': False,
      'reasons': _merged_reasons(sides) if mergeable else why,
      'rows': rows,
  }


def _merged_reasons(sides: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
  """Returns the reasons of every side, in order."""
  reasons = []
  for value in sides.values():
    reasons += value['reasons']
  return reasons


def _activation_metrics(
    turn: _Turn, tokens: dict[str, _Token], context: str | None
) -> dict[str, Any]:
  """Returns the metrics comparing the activations of the two sides' tokens."""
  try:
    if context is None:
      raise UnavailableError(_reason(FORWARDS_NOT_PAIRED))
    if tokens['ref'].model != tokens['target'].model:
      raise UnavailableError(_reason(ACTIVATION_SPACE_MISMATCH))
    vectors = {}
    for side in capture_index.SIDES:
      try:
        vectors[side] = turn.activations(tokens[side])
      except UnavailableError as error:
        raise UnavailableError(_side_reason(side, error)) from None
    return metrics.activation_metrics(vectors['ref'], vectors['target'])
  except UnavailableError as error:
    return {
        key: metrics.missing([error.reason]) for key in metrics.ACTIVATION_KEYS
    }


def _candidate_rows(
    turn: _Turn,
    tokens: dict[str, _Token],
    sources: dict[str, _Logits],
    distributions: dict[str, metrics.Distribution],
    compatible: bool,
    union: bool = False,
) -> list[dict[str, Any]]:
  """Pinned sampled ids, then the most likely ids of either side.

  Args:
      turn: The pair's turn.
      tokens: Each side's token.
      sources: Each side's logits.
      distributions: Each side's distribution; the rows' sides.
      compatible: Whether the two sides share a vocabulary.
      union: Whether the rows are the union of each side's CANDIDATES most
        likely ids, at most twice CANDIDATES besides a sampled id outside them,
        each with both sides' probabilities; else the first CANDIDATES of them.

  Returns:
      The rows, each `{id, label, delta, ref, target}`.
  """
  present = [side for side in capture_index.SIDES if side in distributions]
  selected = {
      side: tokens[side].id
      for side in present
      if tokens[side].id is not None
      and 0 <= tokens[side].id < distributions[side].p.size
  }
  pinned = list(dict.fromkeys(selected.values()))
  pool = set()
  for side in present:
    pool.update(
        int(identity) for identity in distributions[side].order[:CANDIDATES]
    )
  pool -= set(pinned)
  ids = pinned + sorted(
      pool,
      key=lambda identity: (
          -max(float(distributions[side].p[identity]) for side in present),
          identity,
      ),
  )
  labels = turn.labels(present)
  rows = []
  for identity in ids if union else ids[:CANDIDATES]:
    row = {'id': identity, 'label': labels.get(identity), 'delta': None}
    for side in capture_index.SIDES:
      if side not in distributions:
        row[side] = None
        continue
      distribution = distributions[side]
      logit = float(sources[side].values[identity])
      row[side] = {
          'probability': float(distribution.p[identity]),
          'rank': int(distribution.ranks[identity]),
          # Logits a distribution accepts are finite or -Infinity.
          'logit': metrics.json_number(logit),
          'selected': selected.get(side) == identity,
      }
    if compatible:
      row['delta'] = float(
          (
              distributions['target'].p[identity]
              - distributions['ref'].p[identity]
          )
          * 100
      )
    rows.append(row)
  return rows


def _pair(
    turn: _Turn, ref_step: int | None, target_step: int | None
) -> dict[str, Any]:
  """The `TokenAnalysisPair` of one ref token and one target token."""
  steps = {'ref': ref_step, 'target': target_step}
  tokens, sources, distributions, reasons = {}, {}, {}, {}
  for side in capture_index.SIDES:
    try:
      tokens[side] = turn.token(side, steps[side])
      sources[side] = turn.logits(tokens[side])
      try:
        distributions[side] = metrics.distribution(sources[side].values)
      except ValueError as error:
        raise UnavailableError(
            metrics.reason(LOGITS_INVALID, str(error))
        ) from None
    except ValueError as error:
      reasons[side] = _side_reason(side, error)
  context = _context(tokens)
  compatible, why = _compatibility(
      turn, {side: sources[side] for side in distributions}, reasons
  )

  values = {}
  if compatible:
    ref, target = distributions['ref'], distributions['target']
    kl = metrics.kl_divergence(ref, target)
    values['kl'] = (
        metrics.metric(kl)
        if np.isfinite(kl)
        else metrics.missing([_reason(DIVERGENCE_INFINITE)])
    )
    values['js'] = metrics.metric(metrics.js_divergence(ref, target))
  else:
    values['kl'] = values['js'] = metrics.missing(
        why or [_reason(LOGITS_NOT_CAPTURED)]
    )
  values.update(_activation_metrics(turn, tokens, context))

  # Rows are keyed by vocabulary id, so two sides' need one vocabulary.
  rows = []
  if len(distributions) < 2 or compatible:
    rows = _candidate_rows(turn, tokens, sources, distributions, compatible)

  logits_compatible, logits_why = _compatibility(turn, sources, reasons)
  final = {}
  for transform in ('raw', 'centered'):
    ref_values, ref_reason = _transformed(sources.get('ref'), transform)
    target_values, target_reason = _transformed(
        sources.get('target'), transform
    )
    if not logits_compatible:
      final[transform] = metrics.missing_logits_errors(
          logits_why or [_reason(LOGITS_NOT_PAIRED)]
      )
    elif ref_reason or target_reason:
      final[transform] = metrics.missing_logits_errors([
          {**reason, 'side': side}
          for side, reason in (
              ('ref', ref_reason),
              ('target', target_reason),
          )
          if reason
      ])
    else:
      final[transform] = metrics.logits_errors(ref_values, target_values)

  return {
      'ref_step': ref_step,
      'target_step': target_step,
      'metrics': values,
      'final_logits': final,
      'sampling_distribution': _sampling_distribution(
          turn, tokens, sources, distributions, reasons, context
      ),
      'distribution': {
          'compatible': compatible,
          'reasons': [] if compatible else why,
          'context': context if len(distributions) == 2 else None,
          'notes': _notes(tokens, 'logits') if compatible else [],
          **{
              side: _side_summary(
                  sources.get(side),
                  distributions.get(side),
                  tokens.get(side),
                  reasons.get(side),
              )
              for side in capture_index.SIDES
          },
          'rows': rows,
      },
  }


@dataclasses.dataclass(frozen=True)
class _Inspection:
  """Both sides' whole logits vectors of a pair, after one transform."""

  values: dict[str, np.ndarray | None]
  sides: dict[str, dict[str, Any]]
  compatible: bool
  reasons: list[dict[str, Any]]
  notes: list[dict[str, Any]]
  labels: dict[int, str]
  metrics: dict[str, Any]
  total: int


def _inspection(
    turn: _Turn,
    ref_step: int | None,
    target_step: int | None,
    transform: str,
    described: bool = True,
) -> _Inspection:
  """Both sides' vectors of a pair, after `transform`, and what is known.

  Args:
      turn: The pair's turn.
      ref_step: The ref token's ordinal in its `output`, or None.
      target_step: The target token's, or None.
      transform: One of TRANSFORMS.
      described: Whether to measure the pair (`metrics`) and label its
        vocabulary (`labels`); a chart's drawing needs neither, and they are
        empty without.

  Returns:
      The pair's `_Inspection`.
  """
  steps = {'ref': ref_step, 'target': target_step}
  tokens, sources, reasons = {}, {}, {}
  for side in capture_index.SIDES:
    try:
      tokens[side] = turn.token(side, steps[side])
      sources[side] = turn.logits(tokens[side])
    except ValueError as error:
      reasons[side] = _side_reason(side, error)
  vocabulary, why = _compatibility(turn, sources, reasons)
  compatible = vocabulary
  values, sides = {}, {}
  for side in capture_index.SIDES:
    source = sources.get(side)
    values[side], transform_reason = _transformed(source, transform)
    sides[side] = (
        dict(source.summary)
        if source
        else {'reasons': [reasons[side]] if side in reasons else []}
    )
    if transform_reason:
      reasons[side] = {**transform_reason, 'side': side}
      sides[side]['reasons'] = [reasons[side]]
  if compatible and any(values[side] is None for side in capture_index.SIDES):
    compatible = False
    why = list(reasons.values())
  if not compatible:
    why = why or [_reason(LOGITS_NOT_PAIRED)]
  errors, labels = {}, {}
  if described and not compatible:
    errors = metrics.missing_logits_errors(why)
  elif described:
    errors = metrics.logits_errors(values['ref'], values['target'])
  # Two vocabularies may share indices, but not a label: labels come from
  # one vocabulary only.
  if described and (vocabulary or len(sources) == 1):
    labels = turn.labels(list(sources))
  return _Inspection(
      values=values,
      sides=sides,
      compatible=compatible,
      reasons=[] if compatible else why,
      notes=_notes(tokens, 'logits') if compatible else [],
      labels=labels,
      metrics=errors,
      total=max((source.values.size for source in sources.values()), default=0),
  )


def _order(inspection: _Inspection, order: str, direction: str) -> np.ndarray:
  """Vocabulary indices sorted by a value; missing/non-finite values last."""
  indices = np.arange(inspection.total)
  values = np.full(inspection.total, np.nan)
  if order in capture_index.SIDES:
    side = inspection.values[order]
    if side is not None:
      values[: side.size] = side
  elif inspection.compatible:
    with np.errstate(over='ignore', invalid='ignore'):
      values = inspection.values['target'] - inspection.values['ref']
      if order == 'absolute_delta':
        values = np.abs(values)
  finite = np.isfinite(values)
  keys = np.where(finite, values if direction == 'ascending' else -values, 0.0)
  return np.lexsort((indices, keys, ~finite))


def _element(values: np.ndarray | None, index: int) -> float | None:
  if values is None or index >= values.size:
    return None
  return float(values[index])


def token_analysis(
    index: capture_index.CaptureIndex,
    n: int,
    pairs: Sequence[tuple[int | None, int | None]],
    pieces: Pieces | None = None,
) -> list[dict[str, Any]]:
  """Returns the metrics and distributions of token pairs of one turn.

  Args:
      index: The chat's capture.
      n: The turn.
      pairs: `(ref step, target step)` per pair: each a token's ordinal in its
        side's `output`, or None for a side without a token.
      pieces: The tokenizer pieces of a vocabulary, by its digest, that label
        the candidate ids the chat did not produce; None: only the chat's.

  Returns:
      The UI's `TokenAnalysisPair` of each pair, in order.

  Raises:
      NotFound: The turn was not captured.
      Conflict: The capture index rejected the turn.
  """
  view = _Turn(index, n, pieces)
  return [_pair(view, ref, target) for ref, target in pairs]


def final_logits(
    index: capture_index.CaptureIndex,
    n: int,
    ref_step: int | None,
    target_step: int | None,
    transform: str,
    offset: int,
    limit: int,
    order: str,
    direction: str,
) -> dict[str, Any]:
  """Returns one page of a pair's logits, sorted.

  Args:
      index: The chat's capture.
      n: The turn.
      ref_step: The ref token's ordinal in its `output`, or None.
      target_step: The target token's, or None.
      transform: One of TRANSFORMS.
      offset: The first row of the page.
      limit: How many rows, 1..MAX_FINAL_LOGITS_ROWS.
      order: One of ORDERS.
      direction: One of DIRECTIONS.

  Returns:
      The UI's `FinalLogitsResponse` but for the request's turn and row.
      Sorting covers the whole vocabulary; missing and non-finite values
      come last either way, ties by index.

  Raises:
      ValueError: The offset is past the vocabulary.
      NotFound: The turn was not captured.
      Conflict: The capture index rejected the turn.
  """
  data = _inspection(_Turn(index, n), ref_step, target_step, transform)
  if offset > data.total:
    raise ValueError('Final logits offset exceeds the vocabulary length')
  if order == 'index':
    indices = range(data.total)
    if direction == 'descending':
      indices = indices[::-1]
    indices = indices[offset : offset + limit]
  else:
    indices = _order(data, order, direction)[offset : offset + limit]
  rows = []
  for index_value in indices:
    vocabulary_index = int(index_value)
    ref, target = (
        _element(data.values[side], vocabulary_index)
        for side in capture_index.SIDES
    )
    delta = None
    if data.compatible:
      with np.errstate(over='ignore', invalid='ignore'):
        delta = target - ref
    rows.append({
        'index': vocabulary_index,
        'label': data.labels.get(vocabulary_index),
        'ref': metrics.json_number(ref),
        'target': metrics.json_number(target),
        'delta': metrics.json_number(delta),
        'absolute_delta': metrics.json_number(
            abs(delta) if delta is not None else None
        ),
    })
  return {
      'ref_step': ref_step,
      'target_step': target_step,
      'transform': transform,
      'compatible': data.compatible,
      'reasons': data.reasons,
      'notes': data.notes,
      'total': data.total,
      'offset': offset,
      'limit': limit,
      'rows': rows,
      'ref': data.sides['ref'],
      'target': data.sides['target'],
      'metrics': data.metrics,
  }


def _vector_echo(
    data: _Inspection,
    ref_step: int | None,
    target_step: int | None,
    transform: str,
) -> dict[str, Any]:
  """What every answer about a pair's whole vectors repeats."""
  return {
      'ref_step': ref_step,
      'target_step': target_step,
      'transform': transform,
      'compatible': data.compatible,
      'total': data.total,
  }


def _by_index(values: np.ndarray | None, length: int) -> np.ndarray:
  """A side's vector by vocabulary index, `length` long, NaN where missing."""
  result = np.full(length, np.nan)
  if values is not None:
    count = min(values.size, length)
    result[:count] = values[:count]
  return result


def final_logits_summary(
    index: capture_index.CaptureIndex,
    n: int,
    ref_step: int | None,
    target_step: int | None,
    transform: str,
) -> dict[str, Any]:
  """Returns what is known of a pair's two whole logits vectors.

  Args:
      index: The chat's capture.
      n: The turn.
      ref_step: The ref token's ordinal in its `output`, or None.
      target_step: The target token's, or None.
      transform: One of TRANSFORMS.

  Returns:
      The UI's `FinalLogitsSummaryResponse` but for the request's turn and
      row: `statistics` are those of each vector and, when they are
      compared, of `delta` (target - ref) and `abs_delta`
      (analysis/statistics), by vocabulary id.

  Raises:
      NotFound: The turn was not captured.
      Conflict: The capture index rejected the turn.
  """
  data = _inspection(_Turn(index, n), ref_step, target_step, transform)
  return {
      **_vector_echo(data, ref_step, target_step, transform),
      'reasons': data.reasons,
      'notes': data.notes,
      'statistics': _vector_statistics(data),
      'ref': data.sides['ref'],
      'target': data.sides['target'],
      'metrics': data.metrics,
  }


def final_logits_bins(
    index: capture_index.CaptureIndex,
    n: int,
    ref_step: int | None,
    target_step: int | None,
    transform: str,
    elements: tuple[int, int] | None,
    capacity: int,
    reduction: str,
) -> dict[str, Any]:
  """Returns what the Inspection's line chart draws of a pair's logits.

  Args:
      index: The chat's capture.
      n: The turn.
      ref_step: The ref token's ordinal in its `output`, or None.
      target_step: The target token's, or None.
      transform: One of TRANSFORMS.
      elements: The vocabulary range shown, `(start, end)`, kept inside the
        vocabulary as the UI keeps it (vector_drawing.clamped_range);
        None: all of it.
      capacity: How many bins the chart has room for.
      reduction: One of vector_drawing.REDUCTIONS.

  Returns:
      The UI's `FinalLogitsBinsResponse` but for the request's turn and
      row: the range shown (`range`), and the bins of the whole vocabulary
      (`overview`) and of the range (`visible`), as
      vector_drawing.line_bins. A side without logits has only gaps.

  Raises:
      NotFound: The turn was not captured.
      Conflict: The capture index rejected the turn.
  """
  data = _inspection(
      _Turn(index, n), ref_step, target_step, transform, described=False
  )
  length = max(1, data.total)
  start, end = (
      (0, length)
      if elements is None
      else vector_drawing.clamped_range(*elements, data.total)
  )
  ref, target = (
      _by_index(data.values[side], length) for side in capture_index.SIDES
  )
  return {
      **_vector_echo(data, ref_step, target_step, transform),
      'range': {'start': start, 'end': end},
      'capacity': capacity,
      'reduction': reduction,
      'overview': vector_drawing.line_bins(
          ref, target, 0, length, capacity, reduction
      ),
      'visible': vector_drawing.line_bins(
          ref, target, start, end, capacity, reduction
      ),
  }


def final_logits_points(
    index: capture_index.CaptureIndex,
    n: int,
    ref_step: int | None,
    target_step: int | None,
    transform: str,
    values: tuple[float, float] | None,
) -> dict[str, Any]:
  """Returns what the Inspection's scatter plot draws of a pair's logits.

  Only compared vectors are paired: the plot of a pair whose logits are
  not compared has no point.

  Args:
      index: The chat's capture.
      n: The turn.
      ref_step: The ref token's ordinal in its `output`, or None.
      target_step: The target token's, or None.
      transform: One of TRANSFORMS.
      values: The value range shown, `(start, end)`; None: the extent of the
        paired values (vector_drawing.scatter_extent).

  Returns:
      The UI's `FinalLogitsPointsResponse` but for the request's turn and
      row: the `range` shown and the points drawn with their counts
      (vector_drawing.scatter_points, as `points: {index, ref, target}`);
      without `values`, also the `extent` and the paired vocabulary ids as
      runs (`paired_runs`), which no value range changes: the UI keeps
      them from that answer.

  Raises:
      NotFound: The turn was not captured.
      Conflict: The capture index rejected the turn.
  """
  data = _inspection(
      _Turn(index, n), ref_step, target_step, transform, described=False
  )
  length = data.total if data.compatible else 0
  ref, target = (
      _by_index(data.values[side], length) for side in capture_index.SIDES
  )
  whole = {}
  if values is None:
    extent = vector_drawing.scatter_extent(ref, target)
    start, end = extent['start'], extent['end']
    whole = {
        'extent': extent,
        'paired_runs': vector_drawing.paired_runs(ref, target),
    }
  else:
    start, end = values
  drawn = vector_drawing.scatter_points(ref, target, start, end)
  return {
      **_vector_echo(data, ref_step, target_step, transform),
      **whole,
      'range': {'start': start, 'end': end},
      'paired': drawn['paired'],
      'visible': drawn['visible'],
      'grouped': drawn['grouped'],
      'points': {
          'index': drawn['indices'],
          'ref': drawn['ref'],
          'target': drawn['target'],
      },
  }


def _vector_statistics(data: _Inspection) -> dict[str, Any]:
  """The statistics of both vectors, and of their difference if compared."""
  result = {
      key: statistics.empty() for key in ('ref', 'target', 'delta', 'abs_delta')
  }
  for side in capture_index.SIDES:
    if data.values[side] is not None:
      result[side] = statistics.summary(data.values[side])
  if data.compatible:
    with np.errstate(over='ignore', invalid='ignore'):
      delta = data.values['target'] - data.values['ref']
    result['delta'] = statistics.summary(delta)
    result['abs_delta'] = statistics.summary(np.abs(delta))
  return result


@dataclasses.dataclass(frozen=True)
class RangeValues:
  """What token-range reads of one pair (range_values).

  Attributes:
      numbers: The value of each of RANGE_NUMBERS, float64; NaN where the pair
        has none.
      described: Per side, in capture_index.SIDES order, `(logits, size,
        sampler)`: its logits' `(vocabulary size, dtype, vocabulary identity)`,
        the size None unless the logits are one vector, or None without logits;
        its sampling distribution's vocabulary size, as it reports it; its
        sampler settings, one per SAMPLER_KEYS.
  """

  numbers: np.ndarray
  described: tuple[
      tuple[
          tuple[int | None, str | None, str | None] | None,
          int | None,
          tuple[Any, ...],
      ],
      ...,
  ]


def range_values(pair: dict[str, Any]) -> RangeValues:
  """What token-range reads of a `TokenAnalysisPair` (token_analysis).

  Args:
      pair: The pair.

  Returns:
      The pair's `RangeValues`.
  """
  raw = pair['final_logits']['raw']
  sampling = pair['sampling_distribution']
  distribution = pair['distribution']
  numbers = [
      raw['rmse']['value'],
      raw['max_abs']['value'],
      pair['metrics']['kl']['value'],
      pair['metrics']['js']['value'],
  ] + [
      sampling[side].get(metric)
      for side, metric in itertools.product(
          capture_index.SIDES, RANGE_METRICS['sampling']
      )
  ]
  described = []
  for side in capture_index.SIDES:
    metadata = distribution[side].get('logits_metadata')
    logits = None
    if metadata is not None:
      shape = metadata['comparison_shape']
      logits = (
          shape[0] if len(shape) == 1 else None,
          metadata.get('source_dtype'),
          metadata.get('vocab_identity'),
      )
    sampler = sampling[side]['sampler']
    described.append((
        logits,
        sampling[side].get('size'),
        tuple(sampler.get(key) for key in SAMPLER_KEYS),
    ))
  return RangeValues(
      np.array(
          [np.nan if value is None else value for value in numbers],
          dtype=np.float64,
      ),
      tuple(described),
  )


@dataclasses.dataclass(frozen=True)
class RangeRow:
  """One row of a stage as token-range reads it.

  Attributes:
      ref: Whether the row has a ref token.
      target: Whether it has a target token.
      steps: Each side's step, or None.
      values: The `range_values` of its pair.
  """

  ref: bool
  target: bool
  steps: tuple[int | None, int | None]
  values: RangeValues


def token_range(
    first: int,
    rows: Sequence[RangeRow],
    fork: tuple[int, RangeRow] | None,
    analysis: str,
    metric: str,
    start: int,
    end: int,
    bins: int,
) -> dict[str, Any]:
  """Returns one metric of a stage's rows, binned by column.

  Row k is column `first + k`, as kv-range's columns. Final Logits draws
  one series of the pair's metric, at every row; Sampling one per side,
  of the side's own distribution, at the rows that have its token. A
  statistic (analysis/statistics) leaves out a value that is missing or not
  finite.

  Args:
      first: The column of the stage's first row.
      rows: The stage's rows, in order.
      fork: The first divergence's row and pair when it pairs the tokens of one
        generation step instead of its row's; else None.
      analysis: 'final_logits' or 'sampling' (RANGE_METRICS).
      metric: One of RANGE_METRICS[analysis].
      start: The first column of the range.
      end: The column after it; more than `start`.
      bins: How many bins, 1..MAX_RANGE_BINS.

  Returns:
      `{start, end, bins, rows, all, metadata, peak_kl}`: `bins` are the
      bins' columns; `rows` each series' statistics per bin, with its
      `side` (None for the pair's); `all` the statistics per bin of every
      series together; `metadata` what the rows of the range compared
      (`_range_metadata`); `peak_kl` the stage's largest KL, `{value,
      row, step}` (the target's step, or the ref's without one; the
      first row of it), the first divergence's pair standing for its row
      (`fork`), or None without one.
  """
  edges = statistics.edges(start, end, bins)
  bounds = [
      {'start': int(low), 'end': int(high)}
      for low, high in zip(edges[:-1], edges[1:])
  ]
  sides = [None] if analysis == 'final_logits' else list(capture_index.SIDES)
  series = []
  for side in sides:
    number = _NUMBER[metric if side is None else f'{side}.{metric}']
    kept = [
        k for k, row in enumerate(rows) if side is None or getattr(row, side)
    ]
    values = np.array(
        [rows[k].values.numbers[number] for k in kept], dtype=np.float64
    )
    series.append((side, first + np.array(kept, dtype=np.int64), values))

  def binned(columns: np.ndarray, values: np.ndarray) -> list[dict[str, Any]]:
    stats = statistics.binned(values, columns, edges)
    return [{**bound, **stat} for bound, stat in zip(bounds, stats)]

  together = np.concatenate([columns for _, columns, _ in series])
  order = np.argsort(together, kind='stable')
  merged = np.concatenate([values for _, _, values in series])
  peak = None
  for k, row in enumerate(rows):
    pair = fork[1] if fork is not None and fork[0] == k else row
    value = float(pair.values.numbers[_NUMBER['kl']])
    if not np.isnan(value) and (peak is None or value > peak['value']):
      ref_step, target_step = pair.steps
      peak = {
          'value': value,
          'row': k,
          'step': target_step if target_step is not None else ref_step,
      }
  return {
      'start': start,
      'end': end,
      'bins': bounds,
      'rows': [
          {'side': side, 'bins': binned(columns, values)}
          for side, columns, values in series
      ],
      'all': binned(together[order], merged[order]),
      'metadata': _range_metadata(
          rows[max(0, start - first) : max(0, end - first)],
          analysis,
          end - start,
      ),
      'peak_kl': peak,
  }


def _range_metadata(
    rows: Sequence[RangeRow], analysis: str, columns: int
) -> dict[str, Any]:
  """What the rows of a range compared, as the details' Metadata list it.

  Per side, of the rows with its token: for Final Logits, how many have
  logits and the distinct vocabulary sizes, dtypes and vocabulary
  identities they report; for Sampling, the distinct vocabulary sizes and
  sampler settings its distributions report, None where one reports none.
  The output is the metric's, in metrics.CALCULATION_DTYPE: for Final
  Logits one value per row, for Sampling one per side and column.

  Args:
      rows: The rows of the range.
      analysis: 'final_logits' or 'sampling'.
      columns: How many columns the range holds.

  Returns:
      Per side its metadata, and the `output`'s dimensions, shape and dtype.
  """
  result = {}
  for k, side in enumerate(capture_index.SIDES):
    present = [row.values.described[k] for row in rows if getattr(row, side)]
    if analysis == 'final_logits':
      logits = [logits for logits, _, _ in present if logits is not None]
      result[side] = {
          'count': len(logits),
          'vocabulary_sizes': _distinct(size for size, _, _ in logits),
          'dtypes': _distinct(dtype for _, dtype, _ in logits),
          'vocabulary_identities': _distinct(
              identity for _, _, identity in logits
          ),
      }
    else:
      result[side] = {
          'vocabulary_sizes': _distinct(
              _finite(size) for _, size, _ in present
          ),
          'sampler': {
              key: _distinct(_finite(sampler[j]) for _, _, sampler in present)
              for j, key in enumerate(SAMPLER_KEYS)
          },
      }
  result['output'] = (
      {'dimensions': ['Position'], 'shape': [len(rows)]}
      if analysis == 'final_logits'
      else {'dimensions': ['Runtime', 'Position'], 'shape': [2, columns]}
  )
  result['output']['dtype'] = metrics.CALCULATION_DTYPE
  return result


def _distinct(values: Iterable[Any]) -> list[Any]:
  """The distinct values, in the order first seen."""
  return list(dict.fromkeys(values))


def _finite(value: Any) -> int | float | None:
  """A finite number as it is; anything else as None."""
  if type(value) in (int, float) and np.isfinite(value):
    return value
  return None
