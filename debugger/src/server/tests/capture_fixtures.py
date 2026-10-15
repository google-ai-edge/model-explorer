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
stop token, the previous turn's last token fed first by the next turn, and
Gemma's thinking channel with the rollback that follows it), only small: a
vocabulary of 48, a context of 128, prefill chunks of 8, two KV layers with a
head dimension of 4. The decode logits' argmax is the token the turn
generates.
"""

from collections.abc import Sequence
import dataclasses
import json
import pathlib
import re
import zlib

from model_debugger.v1 import runner_pb2
from model_debugger_server import capture_index
import numpy
from safetensors import numpy as safetensors_numpy

PAD, EOS, BOS, TURN_OPEN, TURN_CLOSE, NEWLINE = 0, 1, 2, 3, 4, 5
ROLES = {'system': 6, 'user': 7, 'model': 8}
CHANNEL_OPEN, CHANNEL_CLOSE, THOUGHT = 9, 10, 11
VOCABULARY = 48
CONTEXT_LENGTH = 128
PREFILL_SIZE = 8
LAYERS = 2
HEAD_DIM = 4
HIDDEN = 6
STOP_TOKENS = ([EOS], [TURN_CLOSE])
SIGNATURE = f'prefill_{PREFILL_SIZE}'
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
  """

  chunks: list[str]
  completion: runner_pb2.Completion
  files: dict[str, bytes]


class TinyChat:
  """One chat of a tiny model, written in the runner's capture format."""

  def __init__(
      self, system: str = '', max_completion_tokens: int = 8, seed: int = 0
  ) -> None:
    """Starts a chat.

    Args:
      system: The system message; none when empty.
      max_completion_tokens: Most tokens a reply may have, the stop included.
      seed: Varies logits, activations and KV values but not which token a step
        generates: two chats of different seeds are two backends.
    """
    self.system = system
    self.max_completion_tokens = max_completion_tokens
    self.seed = seed
    self._turn = 0
    self._carried: tuple[int, str] | None = None
    self._position = 0
    # After a turn that thought: where the runtime rolls back to, the token
    # there, and its rendering of that turn without the thoughts.
    self._rollback: tuple[int, tuple[int, str], list] | None = None
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
        cut at `max_completion_tokens`.
      thought: The words the model thinks first, between `<|channel>`,
        `thought`, a newline and `<channel|>`. The next turn then rolls the
        context back to this turn's last input token and feeds this turn again
        without them: its content, `<turn|>` and a newline (seen with an empty
        content only).
    """
    self._turn += 1
    self._files: dict[str, bytes] = {}
    prompt_tokens = []
    if self._turn == 1 and self.system:
      prompt_tokens += _message('system', self.system)
    prompt_tokens += _message('user', prompt)
    prompt_tokens += [_marked(TURN_OPEN), _marked(ROLES['model'])]
    prompt_tokens += [_marked(NEWLINE)]
    thinking = []
    if thought:
      thinking = [_marked(CHANNEL_OPEN), _marked(THOUGHT), _marked(NEWLINE)]
      thinking += [token(word) for word in thought] + [_marked(CHANNEL_CLOSE)]
    generated = thinking + [token(word) for word in reply]
    if stop:
      generated.append(_marked(TURN_CLOSE))
    # A stop token is generated but not released.
    stopped = stop and len(generated) <= self.max_completion_tokens
    generated = generated[: self.max_completion_tokens]
    if not generated:
      raise ValueError('A turn generates at least one token.')
    released = generated[:-1] if stopped else generated

    # Prefill as LiteRT-LM does: the ids come in chunks; each chunk feeds the
    # token held back before it and all but its own last id, which it holds
    # back in turn. The last one held back is fed by the first decode step.
    # The previous turn's last token is held back from that turn; after a
    # turn that thought, the token the context rolls back to is, and the
    # runtime's rendering of that turn comes before the prompt.
    if self._rollback:
      start, held, history = self._rollback
      segments = [history, prompt_tokens]
    else:
      start, held = self._position, self._carried
      bos = [(BOS, '<bos>')] if self._turn == 1 else []
      segments = [bos + prompt_tokens]
    position = start
    for ids in segments:
      for offset in range(0, len(ids), PREFILL_SIZE):
        chunk = ids[offset : offset + PREFILL_SIZE]
        rows = ([held] if held else []) + chunk[:-1]
        if rows:
          # The token held back from the previous turn reaches the graph as
          # its id's bits, not its embedding: LiteRT-LM's prefill writes a
          # pending token without an embedding that way.
          raw_first = self._turn > 1 and position == start
          self._prefill(position, rows, len(chunk), raw_first)
          position += len(rows)
        held = chunk[-1]
    last_input = held
    fed = held
    for index, (id, _) in enumerate(generated):
      self._decode(position, fed, id, last=index == len(generated) - 1)
      fed = generated[index]
      position += 1

    lines = []
    for index, (id, text) in enumerate(released):
      line = {'token_ids': [[id]], 'texts': [text]}
      # The last decode step's token has no score.
      if index < len(generated) - 1:
        line['scores'] = [0.0]
      lines.append(
          json.dumps(line, ensure_ascii=False, separators=(',', ':')) + '\n'
      )
    self._files['generated_tokens.jsonl'] = ''.join(lines).encode()

    context = position + 1
    if context > CONTEXT_LENGTH:
      raise ValueError('The chat outgrows the tiny context.')
    content = released[len(thinking) :]
    reasoning = released[3 : len(thinking) - 1] if thinking else []
    completion = runner_pb2.Completion(
        content=''.join(text for _, text in content),
        reasoning_content=''.join(text for _, text in reasoning),
        finish_reason=(
            runner_pb2.FINISH_REASON_LENGTH
            if len(generated) >= self.max_completion_tokens
            else runner_pb2.FINISH_REASON_STOP
        ),
        usage=runner_pb2.Usage(
            prompt_tokens=context - len(generated),
            completion_tokens=len(generated),
            total_tokens=context,
        ),
        model_debugger=runner_pb2.ModelDebugger(
            turn=self._turn,
            prompt_tokens=[
                runner_pb2.PromptToken(id=id, text=text)
                for id, text in prompt_tokens
            ],
            context_token_count=context,
        ),
    )
    self._carried = generated[-1]
    self._position = context - 1
    self._rollback = None
    if thought:
      first_decode = context - len(generated) - 1
      history = content + [_marked(TURN_CLOSE), _marked(NEWLINE)]
      self._rollback = (first_decode, last_input, history)
    return Turn(
        chunks=[text for _, text in content],
        completion=completion,
        files=self._files,
    )

  def _write(self, signature: str, step: int, key: str, value) -> None:
    name = f'{signature}_{key}_step_{step}.safetensors'
    metadata = {'signature': signature, 'step': str(step)}
    self._files[name] = safetensors_numpy.save({key: value}, metadata=metadata)

  def _write_kv(self, signature: str, step: int, moment: str) -> None:
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

  def _store(self, position: int, id: int) -> None:
    """Writes the KV rows of a token at a position."""
    if position >= CONTEXT_LENGTH:
      return
    channels = numpy.arange(HEAD_DIM)
    for layer in range(LAYERS):
      base = (id * 31 + position * 7 + layer * 13 + channels * 3) % 200 - 100
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
    step = first + len(rows) + 1
    positions = numpy.zeros(PREFILL_SIZE, numpy.int32)
    positions[: len(rows)] = numpy.arange(first, first + len(rows))
    mask = numpy.zeros((1, 1, PREFILL_SIZE, CONTEXT_LENGTH), numpy.bool_)
    for row, position in enumerate(positions[: len(rows)]):
      mask[0, 0, row, : position + 1] = True
    padded = [id for id, _ in rows] + [PAD] * (PREFILL_SIZE - len(rows))
    embeddings = numpy.stack([_embedding(id, HIDDEN) for id in padded])
    per_layer = numpy.stack(
        [_embedding(id, 2 * 4).reshape(2, 4) for id in padded]
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
    self._write(SIGNATURE, step, 'pre_input_pos', positions)
    self._write(SIGNATURE, step, 'pre_mask', mask)
    self._write(SIGNATURE, step, 'pre_embeddings', embeddings[None])
    self._write(SIGNATURE, step, 'pre_per_layer_embeddings', per_layer[None])
    self._write(SIGNATURE, step, 'pre_param_tensor', param)
    self._write_kv(SIGNATURE, step, 'pre')
    # The chunk writes all its rows, its padding included, from `first`.
    for row, id in enumerate(padded):
      self._store(first + row, id)
    self._write_kv(SIGNATURE, step, 'post')

  def _decode(
      self, position: int, fed: tuple[int, str], generates: int, last: bool
  ) -> None:
    step = position + 1
    rng = numpy.random.default_rng([self.seed, position])
    logits = rng.normal(size=VOCABULARY).astype(numpy.float32)
    logits[generates] = logits.max() + 3
    mask = numpy.zeros((1, 1, 1, CONTEXT_LENGTH), numpy.bool_)
    mask[..., : position + 1] = True
    param = numpy.zeros((1, 1, 1, 7), numpy.int32)
    param[0, 0, 0, :3] = [position, position + 1, position + 1]
    self._write(
        'decode', step, 'pre_input_pos', numpy.array([position], numpy.int32)
    )
    self._write('decode', step, 'pre_mask', mask)
    self._write(
        'decode', step, 'pre_embeddings', _embedding(fed[0], HIDDEN)[None, None]
    )
    self._write(
        'decode',
        step,
        'pre_per_layer_embeddings',
        _embedding(fed[0], 8).reshape(1, 1, 2, 4),
    )
    self._write('decode', step, 'pre_param_tensor', param)
    self._write('decode', step, 'post_logits', logits[None, None])
    activations = rng.normal(size=HIDDEN).astype(numpy.float32)
    self._write('decode', step, 'post_activations', activations[None, None])
    self._store(position, fed[0])
    # The KV cache is written once, after the last decode step.
    if last:
      self._write_kv('decode', step, 'post')


def _embedding(id: int, size: int) -> numpy.ndarray:
  return numpy.random.default_rng(id).normal(size=size).astype(numpy.float32)


def write_turn(directory: pathlib.Path, turn: Turn) -> dict:
  """Writes a turn's files and returns its `completion_dict`."""
  directory.mkdir(parents=True, exist_ok=True)
  for name, data in turn.files.items():
    (directory / name).write_bytes(data)
  return capture_index.completion_dict(turn.completion)


def side_info(chat: TinyChat) -> dict:
  """Returns a side's fixed info for `CaptureIndex.create`."""
  return {
      'model': {'sha256': 'ab' * 32, 'name': 'tiny.litertlm', 'size': 1024},
      'runtime': {'name': 'LiteRT-LM', 'version': 'a0f1e97361155c8b'},
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
    exchanges: Sequence[tuple],
    *,
    system: str = '',
    max_completion_tokens: int = 8,
    target_seed: int = 1,
) -> capture_index.CaptureIndex:
  """Makes a chat's capture of both sides, one turn per exchange.

  Args:
    root: The capture directory.
    exchanges: Per turn, (prompt, reply words, whether it ends on a stop), and
      optionally the words thought first.
    system: The system message of both sides.
    max_completion_tokens: The reply limit of both sides.
    target_seed: The target side's seed; the reference side's is 0.
  """
  chats = {
      'ref': TinyChat(system, max_completion_tokens, seed=0),
      'target': TinyChat(system, max_completion_tokens, seed=target_seed),
  }
  index = capture_index.CaptureIndex.create(
      root, {side: side_info(chat) for side, chat in chats.items()}
  )
  for n, (prompt, *exchange) in enumerate(exchanges, 1):
    completions = {
        side: write_turn(
            index.turn_directory(side, n), chat.send(prompt, *exchange)
        )
        for side, chat in chats.items()
    }
    index.add_turn(n, prompt, completions)
  return index
