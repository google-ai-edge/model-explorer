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

"""Captures in the runner's format, made by a tiny model that does not exist.

`TinyChat` plays the runner's side of one chat and writes each turn's files
exactly as LiteRT-LM's RuntimeDebugger does (names, keys, metadata, dtypes,
layouts, prefill chunking and padding, step labels, the jsonl's rules, the
stop token, the token held back at the end of a turn and fed first by the
next, and Gemma's thinking channel with the rewrite of the reply without it
that ends the turn), only small: a vocabulary of 48, a context of 128,
prefill chunks of 8, two KV layers with a head dimension of 4. The decode
logits' argmax is the token the turn generates. Its Completion reports what
the real runner does, `fed_tokens` among it.

A chat that observes tensors (`observe`) also writes, at every step, the
tensors of TAP_POINTS, as a copy of the tiny model that observes them would
have them captured (tap_model): each layer's output in both signatures, and
in decode layer 1's attention scores, which span the context, and the
decode output `activations` too. A row the forward pass does not feed (a
chunk's padding, a context position after the one fed) holds the side's
`junk`, which the analyses never read.
"""

from collections.abc import Callable, Sequence
import dataclasses
import json
import pathlib
import re
from typing import Any
import zlib

from model_debugger.protocol import runner_pb2
from model_debugger.server.capture import capture_index
from model_debugger.server.tap import tap_scan
import numpy
from safetensors.numpy import load_file
from safetensors.numpy import save
from safetensors.numpy import save_file

PAD, EOS, BOS, TURN_OPEN, TURN_CLOSE, NEWLINE = 0, 1, 2, 3, 4, 5
ROLES = {'system': 6, 'user': 7, 'model': 8}
CHANNEL_OPEN, CHANNEL_CLOSE, THOUGHT = 9, 10, 11
VOCABULARY = 48
CONTEXT_LENGTH = 128
PREFILL_SIZE = 8
LAYERS = 2
# What a PyTorch runner's KV files say of their layout.
PYTORCH_KV_LAYOUT = 'k:bhsd,v:bhsd'
HEAD_DIM = 4
HIDDEN = 6
STOP_TOKENS = ([EOS], [TURN_CLOSE])
# The sampling the tiny model sets, as its Load reports it.
SAMPLER_DEFAULTS = runner_pb2.SamplerDefaults(
    type=runner_pb2.SamplerDefaults.TOP_P, top_k=8, top_p=0.95, temperature=1.0
)
SIGNATURE = f'prefill_{PREFILL_SIZE}'
# The subgraph each signature runs in the tiny model.
SUBGRAPHS = {'decode': 0, SIGNATURE: 1}
# The tiny model's KV length; a capture holds CONTEXT_LENGTH instead.
KV_LENGTH = 1000


def _point(signature, op, name, shape, layer, output=False, output_name=None):
  """A point of the tiny model, as its scan would list it."""
  return tap_scan.Point(
      id=tap_scan.point_id(signature, op, 0),
      signature=signature,
      subgraph=SUBGRAPHS[signature],
      op=op,
      output=0,
      tensor=op + 10,
      tensor_name=(
          f'tiny/layer_{layer}/{name}' if layer is not None else f'tiny/{name}'
      ),
      shape=shape,
      dtype='float32',
      operator='MUL',
      output_name=output_name,
      context_axes=tuple(
          axis for axis, extent in enumerate(shape) if extent == KV_LENGTH
      ),
      reason='',
      layer=layer,
      layer_output=output,
      layer_stem='layer' if layer is not None else '',
  )


# What a chat that observes tensors writes, as the tiny model's scan lists it.
TAP_POINTS = (
    _point(SIGNATURE, 3, 'out', (1, PREFILL_SIZE, HIDDEN), 0, True),
    _point(SIGNATURE, 7, 'out', (1, PREFILL_SIZE, HIDDEN), 1, True),
    _point('decode', 3, 'out', (1, 1, HIDDEN), 0, True),
    _point('decode', 5, 'scores', (1, 1, 2, KV_LENGTH), 1),
    _point('decode', 7, 'out', (1, 1, HIDDEN), 1, True),
    _point('decode', 9, 'final', (1, 1, HIDDEN), None, False, 'activations'),
)
_TEXTS = {
    TURN_OPEN: '<|turn>',
    TURN_CLOSE: '<turn|>',
    NEWLINE: '\n',
    CHANNEL_OPEN: '<|channel>',
    CHANNEL_CLOSE: '<channel|>',
    THOUGHT: 'thought',
}
_TEXTS.update({id: role for role, id in ROLES.items()})


def token(text: str) -> tuple[int, str]:
  """Returns the tiny tokenizer's (id, text) for a word."""
  return 12 + zlib.crc32(text.encode()) % (VOCABULARY - 12), text


def tokenize(text: str) -> list[tuple[int, str]]:
  """Splits text into words, each with its leading space."""
  return [token(word) for word in re.findall(r' ?[^ ]+| ', text)]


def _marked(token_id: int) -> tuple[int, str]:
  return token_id, _TEXTS[token_id]


def _message(role: str, text: str) -> list[tuple[int, str]]:
  return [
      _marked(TURN_OPEN),
      _marked(ROLES[role]),
      _marked(NEWLINE),
      *tokenize(text),
      _marked(TURN_CLOSE),
      _marked(NEWLINE),
  ]


@dataclasses.dataclass
class Turn:
  """What a runner returns for one Send.

  Attributes:
      chunks: The content chunks, one per released token of the content.
      completion: The Completion.
      files: The capture's files by name.
      reasoning_chunks: The reasoning chunks, one per released token of the
        thoughts; they come before the content's.
  """

  chunks: list[str]
  completion: runner_pb2.Completion
  files: dict[str, bytes]
  reasoning_chunks: list[str] = dataclasses.field(default_factory=list)


class TinyChat:
  """One chat of a tiny model, written in the runner's capture format."""

  def __init__(
      self,
      system: str = '',
      max_completion_tokens: int = 8,
      seed: int = 0,
      observe: bool = False,
      junk: float = 0.0,
      pytorch: bool = False,
  ) -> None:
    """Starts a chat.

    Args:
        system: The system message; none when empty.
        max_completion_tokens: Most tokens a reply may have, the stop included.
        seed: Varies logits, activations and KV values but not which token a
          step generates: two chats of different seeds are two backends.
        observe: Whether it writes the tensors of TAP_POINTS too.
        junk: What a row of those that no forward pass feeds holds.
        pytorch: Whether it is written as the PyTorch runner writes it: logits
          and KV cache in BF16, the cache in Hugging Face's layout, as its
          `layout` metadata says, holding the positions processed so far, and
          none before the first.
    """
    self.system = system
    self.pytorch = pytorch
    self.observe = observe
    self.junk = junk
    self.max_completion_tokens = max_completion_tokens
    self.seed = seed
    self._turn = 0
    # The token held back for the next turn to feed first, and where.
    self._held: tuple[int, str] | None = None
    self._position = 0
    # Whether the held token has no embedding: one a decode step
    # generated, which reaches the graph as its id's bits.
    self._held_raw = False
    self._fed: list[runner_pb2.FedToken] = []
    self._keys = numpy.zeros((LAYERS, CONTEXT_LENGTH, HEAD_DIM), numpy.int8)
    self._values = numpy.zeros((LAYERS, HEAD_DIM, CONTEXT_LENGTH), numpy.int8)

  def load_response(self) -> runner_pb2.LoadResponse:
    """Returns what Load answers for the tiny model."""
    return runner_pb2.LoadResponse(
        start_token=BOS,
        stop_tokens=[
            runner_pb2.StopToken(ids=runner_pb2.TokenIds(ids=ids))
            for ids in STOP_TOKENS
        ],
        sampler_defaults=SAMPLER_DEFAULTS,
    )

  def send(
      self,
      prompt: str,
      reply: Sequence[str],
      stop: bool = True,
      thought: Sequence[str] = (),
  ) -> Turn:
    """Generates one turn.

    Args:
        prompt: The user message.
        reply: The words the model generates.
        stop: Whether the reply ends with `<turn|>`. The reply and its stop are
          cut at `max_completion_tokens`, or where they would outgrow the
          context.
        thought: The words the model thinks first, between `<|channel>`,
          `thought`, a newline and `<channel|>`. The turn then rewrites the
          context from its first decode step's position: that position's token
          again, then its content, `<turn|>` and a newline, held back.

    Returns:
        The turn: its files and its completion.
    """
    self._turn += 1
    self._files: dict[str, bytes] = {}
    self._fed = []
    prompt_tokens = []
    if self._turn == 1 and self.system:
      prompt_tokens += _message('system', self.system)
    prompt_tokens += _message('user', prompt)
    prompt_tokens += [_marked(TURN_OPEN), _marked(ROLES['model'])]
    prompt_tokens += [_marked(NEWLINE)]
    thinking = []
    if thought:
      thinking = [
          _marked(CHANNEL_OPEN),
          _marked(THOUGHT),
          _marked(NEWLINE),
      ]
      thinking += [token(word) for word in thought] + [_marked(CHANNEL_CLOSE)]
    generated = thinking + [token(word) for word in reply]
    if stop:
      generated.append(_marked(TURN_CLOSE))
    # The input starts with the token held back.
    bos = [(BOS, '<bos>')] if self._turn == 1 else []
    position, held = self._feed(
        self._position, self._held, bos + prompt_tokens, self._held_raw
    )
    # The reply is cut at either limit, its own or the context's: the
    # context then holds every position up to the last token generated.
    limit = min(self.max_completion_tokens, CONTEXT_LENGTH - position - 1)
    # A stop token is generated but not released.
    stopped = stop and len(generated) <= limit
    generated = generated[:limit]
    if not generated:
      raise ValueError('A turn generates at least one token.')
    released = generated[:-1] if stopped else generated
    first_decode = position
    last_input = held
    fed = held
    for index, (token_id, _) in enumerate(generated):
      self._decode(position, fed, token_id, last=index == len(generated) - 1)
      fed = generated[index]
      position += 1

    lines = []
    for index, (token_id, text) in enumerate(released):
      line = {'token_ids': [[token_id]], 'texts': [text]}
      # The last decode step's token has no score.
      if index < len(generated) - 1:
        line['scores'] = [0.0]
      lines.append(json.dumps(line, separators=(',', ':')) + '\n')
    self._files['generated_tokens.jsonl'] = ''.join(lines).encode()

    total = position + 1
    if total > CONTEXT_LENGTH:
      raise ValueError('The chat outgrows the tiny context.')
    content = released[len(thinking) :]
    reasoning = released[3 : len(thinking) - 1]
    # What the turn holds back: the last token generated, or after a
    # reply that thought, the last of the reply rendered without it.
    self._held, self._held_raw = generated[-1], True
    self._position = position
    if thought:
      history = content + [_marked(TURN_CLOSE), _marked(NEWLINE)]
      self._position, self._held = self._feed(
          first_decode, last_input, history, True
      )
      self._held_raw = False
    context = self._position + 1
    hit_limit = (
        len(generated) >= self.max_completion_tokens or total >= CONTEXT_LENGTH
    )
    completion = runner_pb2.Completion(
        content=''.join(text for _, text in content),
        reasoning_content=''.join(text for _, text in reasoning),
        finish_reason=(
            runner_pb2.FINISH_REASON_LENGTH
            if hit_limit
            else runner_pb2.FINISH_REASON_STOP
        ),
        usage=runner_pb2.Usage(
            prompt_tokens=first_decode + 1,
            completion_tokens=len(generated),
            total_tokens=total,
        ),
        model_debugger=runner_pb2.ModelDebugger(
            turn=self._turn,
            context_token_count=context,
            fed_tokens=self._fed,
        ),
    )
    return Turn(
        chunks=[text for _, text in content],
        reasoning_chunks=[text for _, text in reasoning],
        completion=completion,
        files=self._files,
    )

  def _feed(
      self,
      position: int,
      held: tuple[int, str] | None,
      ids: list[tuple[int, str]],
      raw: bool,
  ) -> tuple[int, tuple[int, str] | None]:
    """Prefills ids from a position as LiteRT-LM does.

    The ids come in chunks; each chunk feeds the token held back before
    it and all but its own last id, which it holds back in turn.

    Args:
        position: Where the first row goes.
        held: The token held back before the first chunk, if any.
        ids: The ids to prefill.
        raw: Whether the first row reaches the graph as its id's bits, not its
          embedding, as LiteRT-LM's prefill writes a pending token without an
          embedding.

    Returns:
        The position after the last row, and the token held back.
    """
    for offset in range(0, len(ids), PREFILL_SIZE):
      chunk = ids[offset : offset + PREFILL_SIZE]
      rows = ([held] if held else []) + chunk[:-1]
      if rows:
        self._prefill(position, rows, len(chunk), raw)
        raw = False
        position += len(rows)
      held = chunk[-1]
    return position, held

  def _write(self, signature: str, step: int, key: str, value) -> None:
    name = f'{signature}_{key}_step_{step}.safetensors'
    metadata = {'signature': signature, 'step': str(step)}
    self._files[name] = save({key: value}, metadata=metadata)

  def _write_kv(
      self, signature: str, step: int, moment: str, processed: int
  ) -> None:
    """Writes the KV cache of every layer as it stands at `moment`."""
    if self.pytorch:
      self._write_pytorch_kv(signature, step, moment, processed)
      return
    for layer in range(LAYERS):
      self._write(
          signature,
          step,
          f'{moment}_kv_cache_k_{layer}',
          self._keys[layer][None, None].copy(),
      )
      self._write(
          signature,
          step,
          f'{moment}_kv_cache_v_{layer}',
          self._values[layer][None, None].copy(),
      )

  def _write_pytorch_kv(
      self, signature: str, step: int, moment: str, processed: int
  ) -> None:
    """Writes the cache as the PyTorch runner does: BF16, bhsd."""
    if processed == 0:
      return
    metadata = {
        'signature': signature,
        'step': str(step),
        'layout': PYTORCH_KV_LAYOUT,
    }
    for layer in range(LAYERS):
      for kind, rows in (
          ('k', self._keys[layer][:processed]),
          ('v', self._values[layer][:, :processed].T),
      ):
        key = f'{moment}_kv_cache_{kind}_{layer}'
        name = f'{signature}_{key}_step_{step}.safetensors'
        self._files[name] = save_bf16(
            key, rows[None, None].astype(numpy.float32), metadata
        )

  def _observed(
      self,
      point: tap_scan.Point,
      position: int,
      token_id: int,
      size: int = HIDDEN,
      head: int = 0,
  ) -> numpy.ndarray:
    """The values of a point at a position: the seed moves them a little."""
    layer = 9 if point.layer is None else point.layer
    base = numpy.random.default_rng([token_id, layer, head]).normal(size=size)
    noise = numpy.random.default_rng([self.seed, position, head]).normal(
        size=size
    )
    return (base + position + 0.001 * self.seed * noise).astype(numpy.float32)

  def _write_tap(
      self, signature: str, step: int, point: tap_scan.Point, values
  ) -> None:
    key = f'post_tap_sg{point.subgraph}_op{point.op}_out{point.output}'
    self._write(signature, step, key, values)

  def _store(self, position: int, token_id: int) -> None:
    """Writes the KV rows of a token at a position."""
    if position >= CONTEXT_LENGTH:
      return
    channels = numpy.arange(HEAD_DIM)
    for layer in range(LAYERS):
      base = (
          token_id * 31 + position * 7 + layer * 13 + channels * 3
      ) % 200 - 100
      noise = numpy.where((position + channels) % 3 == 0, self.seed, 0)
      row = numpy.clip(base + noise, -127, 127).astype(numpy.int8)
      self._keys[layer, position] = row
      self._values[layer, :, position] = -row

  def _prefill(
      self,
      first: int,
      rows: list[tuple[int, str]],
      ids: int,
      raw_first: bool,
  ) -> None:
    """Captures one prefill chunk of `rows`, starting at position `first`."""
    step = first + len(rows) + 1
    positions = numpy.zeros(PREFILL_SIZE, numpy.int32)
    positions[: len(rows)] = numpy.arange(first, first + len(rows))
    mask = numpy.zeros((1, 1, PREFILL_SIZE, CONTEXT_LENGTH), numpy.bool_)
    for row, position in enumerate(positions[: len(rows)]):
      mask[0, 0, row, : position + 1] = True
    padded = [token_id for token_id, _ in rows] + [PAD] * (
        PREFILL_SIZE - len(rows)
    )
    embeddings = numpy.stack(
        [_embedding(token_id, HIDDEN) for token_id in padded]
    )
    per_layer = numpy.stack(
        [_embedding(token_id, 2 * 4).reshape(2, 4) for token_id in padded]
    )
    if raw_first:
      embeddings[0] = 0
      embeddings[0, :1] = numpy.array([rows[0][0]], numpy.int32).view(
          numpy.float32
      )
      per_layer[0] = 0
      per_layer[0, 0, :1] = embeddings[0, :1]
    param = numpy.zeros((1, 1, 1, 7), numpy.int32)
    param[0, 0, 0, :3] = [first, first + ids, first + ids]
    for row, (token_id, text) in enumerate(rows):
      self._fed.append(
          runner_pb2.FedToken(
              position=first + row,
              phase=runner_pb2.FedToken.PREFILL,
              id=token_id,
              text=text,
          )
      )
    self._write(SIGNATURE, step, 'pre_input_pos', positions)
    self._write(SIGNATURE, step, 'pre_mask', mask)
    self._write(SIGNATURE, step, 'pre_embeddings', embeddings[None])
    self._write(SIGNATURE, step, 'pre_per_layer_embeddings', per_layer[None])
    self._write(SIGNATURE, step, 'pre_param_tensor', param)
    self._write_kv(SIGNATURE, step, 'pre', first)
    if self.observe:
      for point in TAP_POINTS:
        if point.signature != SIGNATURE:
          continue
        values = numpy.full(point.shape, self.junk, numpy.float32)
        for row, (token_id, _) in enumerate(rows):
          values[0, row] = self._observed(point, first + row, token_id)
        self._write_tap(SIGNATURE, step, point, values)
    # The chunk writes all its rows, its padding included, from `first`.
    for row, token_id in enumerate(padded):
      self._store(first + row, token_id)
    self._write_kv(SIGNATURE, step, 'post', first + len(rows))

  def _decode(
      self, position: int, fed: tuple[int, str], generates: int, last: bool
  ) -> None:
    """Captures the decode step at `position`, which generates `generates`."""
    step = position + 1
    rng = numpy.random.default_rng([self.seed, position])
    logits = rng.normal(size=VOCABULARY).astype(numpy.float32)
    logits[generates] = logits.max() + 3
    mask = numpy.zeros((1, 1, 1, CONTEXT_LENGTH), numpy.bool_)
    mask[..., : position + 1] = True
    param = numpy.zeros((1, 1, 1, 7), numpy.int32)
    param[0, 0, 0, :3] = [position, position + 1, position + 1]
    self._fed.append(
        runner_pb2.FedToken(
            position=position,
            phase=runner_pb2.FedToken.DECODE,
            id=fed[0],
            text=fed[1],
        )
    )
    self._write(
        'decode',
        step,
        'pre_input_pos',
        numpy.array([position], numpy.int32),
    )
    self._write('decode', step, 'pre_mask', mask)
    self._write(
        'decode',
        step,
        'pre_embeddings',
        _embedding(fed[0], HIDDEN)[None, None],
    )
    self._write(
        'decode',
        step,
        'pre_per_layer_embeddings',
        _embedding(fed[0], 8).reshape(1, 1, 2, 4),
    )
    self._write('decode', step, 'pre_param_tensor', param)
    if self.pytorch:
      name = f'decode_post_logits_step_{step}.safetensors'
      self._files[name] = save_bf16(
          'post_logits',
          logits[None, None],
          {'signature': 'decode', 'step': str(step)},
      )
    else:
      self._write('decode', step, 'post_logits', logits[None, None])
    activations = rng.normal(size=HIDDEN).astype(numpy.float32)
    if self.observe:
      activations = self._observed(TAP_POINTS[-1], position, fed[0])
    self._write('decode', step, 'post_activations', activations[None, None])
    if self.observe:
      for point in TAP_POINTS[2:-1]:
        shape = tuple(
            CONTEXT_LENGTH if axis in point.context_axes else extent
            for axis, extent in enumerate(point.shape)
        )
        values = numpy.full(shape, self.junk, numpy.float32)
        if point.context_axes:
          for head in range(shape[2]):
            values[0, 0, head, : position + 1] = self._observed(
                point, position, fed[0], position + 1, head
            )
        else:
          values[0, 0] = self._observed(point, position, fed[0])
        self._write_tap('decode', step, point, values)
    self._store(position, fed[0])
    # The KV cache is written once, after the last decode step.
    if last:
      self._write_kv('decode', step, 'post', position + 1)


def _embedding(token_id: int, size: int) -> numpy.ndarray:
  return (
      numpy.random.default_rng(token_id).normal(size=size).astype(numpy.float32)
  )


def write_turn(directory: pathlib.Path, turn: Turn) -> dict[str, Any]:
  """Writes a turn's files and returns its `completion_dict`."""
  directory.mkdir(parents=True, exist_ok=True)
  for name, data in turn.files.items():
    (directory / name).write_bytes(data)
  return capture_index.completion_dict(turn.completion)


def edit_logits(directory: pathlib.Path, nth: int, change) -> int:
  """Changes the logits of a turn's `nth` decode step in place.

  Args:
      directory: The turn's directory.
      nth: Which decode step, from 0.
      change: Called with the step's logits, a vector it may change.

  Returns:
      The step's label.
  """
  paths = sorted(
      directory.glob('decode_post_logits_step_*.safetensors'),
      key=lambda path: int(path.stem.rsplit('_', 1)[1]),
  )
  path = paths[nth]
  step = path.stem.rsplit('_', 1)[1]
  logits = load_file(path)['post_logits']
  change(logits[0, 0])
  save_file(
      {'post_logits': logits},
      str(path),
      metadata={'signature': 'decode', 'step': step},
  )
  return int(step)


# The id greedy_edits() ties, sets off or leaves unmasked beside the token a
# step generates; the tiny tokenizer's words start at 12, so it is no word.
OTHER = 3


def greedy_edits() -> dict[str, Callable[[numpy.ndarray], None]]:
  """Edits of the logits of a decode step that generates 'a', by name.

  Each is a case of capture_index.greedy_choice's rule, for edit_logits: the
  capture index's tests and the token analysis' read the same step through
  each, so the two are held to one rule.

  - `tie`: 'a' ties with OTHER, which comes first: greedy's pick is either.
  - `nan`: a NaN elsewhere: no pick can be confirmed.
  - `all_minus_inf`: every logit -inf: no pick can be confirmed.
  - `plus_inf`: 'a' is +inf: it is the pick.
  - `two_plus_inf`: 'a' and OTHER are +inf: the pick is either.
  - `masked`: every logit but 'a' and OTHER -inf: 'a' is still the pick.
  - `plus_inf_elsewhere`: OTHER is +inf: it is the pick, not 'a', which
    the turn still generates (the index does not check the sampling).
  - `masked_elsewhere`: as `masked`, but OTHER above 'a': OTHER is the
    pick, not 'a', likewise.

  Returns:
      Each edit, by its name above.
  """
  a = token('a')[0]

  def mask(logits: numpy.ndarray) -> None:
    kept = logits[[a, OTHER]].copy()
    logits[:] = -numpy.inf
    logits[[a, OTHER]] = kept

  def tie(logits: numpy.ndarray) -> None:
    logits[OTHER] = logits[a]

  def nan(logits: numpy.ndarray) -> None:
    logits[OTHER] = numpy.nan

  def all_minus_inf(logits: numpy.ndarray) -> None:
    logits[:] = -numpy.inf

  def plus_inf(logits: numpy.ndarray) -> None:
    logits[a] = numpy.inf

  def two_plus_inf(logits: numpy.ndarray) -> None:
    logits[[a, OTHER]] = numpy.inf

  def plus_inf_elsewhere(logits: numpy.ndarray) -> None:
    logits[OTHER] = numpy.inf

  def masked_elsewhere(logits: numpy.ndarray) -> None:
    mask(logits)
    logits[OTHER] = logits[a] + 1

  return {
      'tie': tie,
      'nan': nan,
      'all_minus_inf': all_minus_inf,
      'plus_inf': plus_inf,
      'two_plus_inf': two_plus_inf,
      'masked': mask,
      'plus_inf_elsewhere': plus_inf_elsewhere,
      'masked_elsewhere': masked_elsewhere,
  }


def save_bf16(
    key: str, values: numpy.ndarray, metadata: dict[str, str]
) -> bytes:
  """A safetensors file of one BF16 tensor: `values` cut to bfloat16.

  numpy has no bfloat16; its bits are a float32's top half.

  Args:
      key: The tensor's name.
      values: The tensor's values.
      metadata: The file's `__metadata__`.

  Returns:
      The file's bytes.
  """
  bits = (
      numpy.ascontiguousarray(values, numpy.float32).view(numpy.uint32) >> 16
  ).astype('<u2')
  header = json.dumps({
      key: {
          'dtype': 'BF16',
          'shape': list(values.shape),
          'data_offsets': [0, bits.nbytes],
      },
      '__metadata__': metadata,
  }).encode()
  header += b' ' * (-len(header) % 8)
  return len(header).to_bytes(8, 'little') + header + bits.tobytes()


def set_metadata(path: pathlib.Path, **changes: str) -> None:
  """Sets entries of a tensor file's `__metadata__`, its data kept."""
  data = path.read_bytes()
  length = int.from_bytes(data[:8], 'little')
  header = json.loads(data[8 : 8 + length])
  header['__metadata__'] = {**header.get('__metadata__', {}), **changes}
  text = json.dumps(header).encode()
  text += b' ' * (-len(text) % 8)
  path.write_bytes(len(text).to_bytes(8, 'little') + text + data[8 + length :])


def side_info(chat: TinyChat) -> dict[str, Any]:
  """Returns a side's fixed info for `CaptureIndex.create`."""
  return {
      'model': {'sha256': 'ab' * 32, 'name': 'tiny.litertlm', 'size': 1024},
      'runtime': (
          {'name': 'PyTorch', 'version': '2.9.0'}
          if chat.pytorch
          else {'name': 'LiteRT-LM', 'version': 'a0f1e97361155c8b'}
      ),
      'host': {'platform': 'MACOS', 'chip': 'Apple M2 Max'},
      'device_id': 'local',
      'device_name': 'This computer',
      'load': {
          'device': 'CPU',
          'context_length': CONTEXT_LENGTH,
          'runtime_options': {},
      },
      'chat': {
          'messages': (
              [{'role': 'system', 'content': chat.system}]
              if chat.system
              else []
          ),
          'options': {
              'max_completion_tokens': chat.max_completion_tokens,
              'top_k': 1,
          },
      },
      'start_token': BOS,
      'stop_tokens': [list(ids) for ids in STOP_TOKENS],
  }


def capture(
    root: pathlib.Path,
    exchanges: Sequence[tuple[Any, ...]],
    *,
    system: str = '',
    max_completion_tokens: int = 8,
    target_seed: int = 1,
    observe: bool = False,
    pytorch: bool = False,
    top_k: int = 1,
) -> capture_index.CaptureIndex:
  """Makes a chat's capture of both sides, one turn per exchange.

  Args:
      root: The capture directory.
      exchanges: Per turn, (prompt, reply words, whether it ends on a stop), and
        optionally the words thought first.
      system: The system message of both sides.
      max_completion_tokens: The reply limit of both sides.
      target_seed: The target side's seed; the reference side's is 0.
      observe: Whether both sides write the tensors of TAP_POINTS; the rows no
        forward pass feeds hold 0 on the ref side, 1 on the target.
      pytorch: Whether both sides are PyTorch runners' (TinyChat).
      top_k: The top_k both chats sample with; 1 is greedy.

  Returns:
      The capture's index.
  """
  chats = {
      'ref': TinyChat(
          system,
          max_completion_tokens,
          0,
          observe,
          junk=0.0,
          pytorch=pytorch,
      ),
      'target': TinyChat(
          system,
          max_completion_tokens,
          target_seed,
          observe,
          junk=1.0,
          pytorch=pytorch,
      ),
  }
  infos = {side: side_info(chat) for side, chat in chats.items()}
  for info in infos.values():
    info['chat']['options']['top_k'] = top_k
  index = capture_index.CaptureIndex.create(root, infos)
  for n, (prompt, *exchange) in enumerate(exchanges, 1):
    completions = {
        side: write_turn(
            index.turn_directory(side, n), chat.send(prompt, *exchange)
        )
        for side, chat in chats.items()
    }
    index.add_turn(n, prompt, completions)
  return index
