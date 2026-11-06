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

"""Gemma 4 in Hugging Face transformers, fed as LiteRT-LM feeds a model.

A turn renders the whole chat with the tokenizer's chat template. The context
the cache holds is kept when the rendering starts with it, the token held
back by the turn before included; otherwise the turn replays the rendering
from position 0 into a new cache. If the turn's input is new from position
`first` and ends at position n - 1:

- one prefill forward pass feeds positions first .. n - 2, if there are any,
  labelled step n;
- decode step n feeds position n - 1, and its logits give the first token;
  decode step k feeds position k - 1, the token step k - 1 generated;
- the last token generated is never fed: it is held back, and fed first next
  turn.

The capture (capture_writer) holds each pass's `pre_input_pos`, each decode
step's `post_logits` (the one row that produced its token) and the KV cache
before and after the prefill and after the last decode step. A model loaded
with `observe: anchors` runs each turn's passes inside pytorch-collector's
collect() (anchor_observer), and its capture holds each pass's four anchors
of every decoder layer too, and the tensor of each module the Load asks for
beside them (`observeModules`: `{module, edge}`, what the module reads,
`in`, or returns, `out`), or says why it does not (`observation.json`).

The first observed turn of a Load on the CPU is captured fully, every
module and operator, and its capture holds the execution graph of the
first pass of each signature it ran (`execution_graph.json.gz`), which only
a full capture can export; every other observed turn, and every one on
another device (MPS), is captured module-only: only the modules its tensors
are taken at, on the device the model runs on, which takes a fraction of
the time and memory. A Load whose first turn did not export the graph (its
capture failed) exports none.

A turn runs under torch.no_grad(), not torch.inference_mode(), which
collect() refuses, whether it observes or not.
"""

# pylint: disable=g-bad-exception-name

from collections.abc import Callable
import dataclasses
import json
import pathlib

from google.protobuf import json_format
from model_debugger.protocol import runner_pb2
from model_debugger.runner_pytorch.llm import anchor_observer
from model_debugger.runner_pytorch.llm import capture_writer
from model_debugger.runner_pytorch.llm import sampler
import torch
import transformers

# What a Load's runtime_options may hold.
LOAD_OPTIONS_SCHEMA = json.dumps({
    'type': 'object',
    'properties': {
        'dtype': {'enum': ['default', 'float32', 'float16', 'bfloat16']},
        'cpuThreads': {'type': 'integer', 'minimum': 1},
        'observe': {'enum': list(anchor_observer.LEVELS)},
        'observeMemoryLimitBytes': {'type': 'integer', 'minimum': 1},
        # Each `{module, edge}`, checked as the model loads
        # (anchor_observer.parse_modules): the schema's keywords are
        # those both runners check, which do not reach into arrays.
        'observeModules': {'type': 'array'},
    },
    'additionalProperties': False,
})
# What a chat's runtime_options may hold: nothing yet.
CHAT_OPTIONS_SCHEMA = json.dumps(
    {'type': 'object', 'properties': {}, 'additionalProperties': False}
)

_ROLES = {
    runner_pb2.ChatMessage.SYSTEM: 'system',
    runner_pb2.ChatMessage.USER: 'user',
    runner_pb2.ChatMessage.ASSISTANT: 'assistant',
}
_PREFILL = runner_pb2.FedToken.PREFILL
_DECODE = runner_pb2.FedToken.DECODE
# The signatures a turn runs, whose execution graphs it may write.
_SIGNATURES = ('prefill', 'decode')


class Cancelled(Exception):
  """The client cancelled the turn."""


class NoRoom(Exception):
  """The context has no room for the turn, or for a chat's start."""

  def __init__(self, message: str, needed: int, left: int) -> None:
    """Says so, with the tokens needed and those the context has left."""
    super().__init__(message)
    self.needed = needed
    self.left = left


def version() -> str:
  """Returns the runtime's version: torch's and transformers'."""
  return f'torch {torch.__version__}, transformers {transformers.__version__}'


class Model:
  """A loaded Gemma 4 model, its tokenizer and what its metadata says."""

  def __init__(
      self,
      directory: pathlib.Path,
      device: str,
      request: runner_pb2.LoadRequest,
  ) -> None:
    """Loads the model in `directory` onto `device`.

    Args:
        directory: The Hugging Face model directory.
        device: The torch device: 'cpu', 'mps' or 'cuda'.
        request: The Load, its options checked against the schema.

    Raises:
        ValueError: The directory holds no Gemma 4 model, or a module
          asked for is not one of its modules.
    """
    options = request.runtime_options.fields
    if 'cpuThreads' in options:
      torch.set_num_threads(int(options['cpuThreads'].number_value))
    dtype = options['dtype'].string_value if 'dtype' in options else 'default'
    # Whether each turn captures the decoder layers' anchors, and within
    # what snapshot budget.
    self.observe = (
        options['observe'].string_value
        if 'observe' in options
        else anchor_observer.NONE
    )
    self.memory_limit_bytes = (
        int(options['observeMemoryLimitBytes'].number_value)
        if 'observeMemoryLimitBytes' in options
        else anchor_observer.DEFAULT_MEMORY_LIMIT_BYTES
    )
    config = transformers.AutoConfig.from_pretrained(directory)
    if config.model_type != 'gemma4':
      raise ValueError(
          f'The model is a {config.model_type}; this runner runs Gemma 4.'
      )
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(directory)
    self.model = (
        transformers.Gemma4ForConditionalGeneration.from_pretrained(
            directory,
            dtype='auto' if dtype == 'default' else getattr(torch, dtype),
            attn_implementation='eager',
        )
        .eval()
        .to(device)
    )
    self.device = device
    # The modules each observed turn captures beside the anchors.
    self.modules = anchor_observer.parse_modules(
        self.model,
        json_format.MessageToDict(options['observeModules'])
        if 'observeModules' in options
        else [],
    )
    # Whether the next observed turn is captured fully, to export the
    # execution graph: the first observed turn of a Load on the CPU.
    self.full_turn_pending = device == 'cpu'
    self.context_length = request.context_length
    generation = transformers.GenerationConfig.from_pretrained(directory)
    stops = generation.eos_token_id
    if stops is None:
      stops = self.tokenizer.eos_token_id
    self.stop_ids = [stops] if isinstance(stops, int) else list(stops)
    self.defaults = sampler.Defaults(
        do_sample=bool(generation.do_sample),
        temperature=generation.temperature,
        top_k=generation.top_k,
        top_p=generation.top_p,
    )
    # The start token is the one the chat template puts first, if any.
    probe = self.render([{'role': 'user', 'content': 'x'}])
    bos = self.tokenizer.bos_token_id
    self.start_token = bos if probe and probe[0] == bos else None

  def info(self) -> runner_pb2.LoadResponse:
    """Returns what the model's metadata says, as Load answers it."""
    response = runner_pb2.LoadResponse(
        stop_tokens=[
            runner_pb2.StopToken(ids=runner_pb2.TokenIds(ids=[token]))
            for token in self.stop_ids
        ],
        sampler_defaults=self.defaults.proto(),
    )
    if self.start_token is not None:
      response.start_token = self.start_token
    return response

  def render(self, messages: list[dict[str, str]]) -> list[int]:
    """Returns the ids of a chat rendered for the model's next reply."""
    text = self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return self.tokenizer(text, add_special_tokens=False)['input_ids']

  def text(self, token_id: int) -> str:
    """Returns one id's text, special tokens included."""
    return self.tokenizer.decode([token_id], skip_special_tokens=False)

  def create_chat(self, request: runner_pb2.CreateChatRequest) -> 'Chat':
    """Returns the chat a CreateChat asks for, its options checked.

    Args:
      request: The CreateChatRequest message containing messages and options.

    Raises:
      ValueError: A message has no role, or a system message is not first.
      NoRoom: The starting messages, rendered as the first turn's prompt for an
        empty user message, leave the reply no room.
    """
    messages = []
    for index, message in enumerate(request.messages):
      if message.role not in _ROLES:
        raise ValueError(f'Message {index} has no role.')
      if message.role == runner_pb2.ChatMessage.SYSTEM and index != 0:
        raise ValueError('A system message may only be the first message.')
      messages.append(
          {'role': _ROLES[message.role], 'content': message.content}
      )
    if messages:
      # Counted as a turn is: its prompt, and one position for a reply.
      token_count = len(self.render(messages + [{'role': 'user', 'content': ''}]))
      if token_count + 1 > self.context_length:
        raise NoRoom(
            f'The starting messages take {token_count} positions in the chat '
            'template and a reply one more; the context holds '
            f'{self.context_length}.',
            token_count + 1,
            self.context_length,
        )
    return Chat(self, messages, request.options)


@dataclasses.dataclass
class _Turn:
  """What one turn fed and generated, as it runs."""

  fed: list[runner_pb2.FedToken] = dataclasses.field(default_factory=list)
  generated: list[int] = dataclasses.field(default_factory=list)
  released: list[int] = dataclasses.field(default_factory=list)
  chunks: list[str] = dataclasses.field(default_factory=list)


class Chat:
  """A chat on a model: its messages and the context its cache holds."""

  def __init__(
      self,
      model: Model,
      messages: list[dict[str, str]],
      options: runner_pb2.ChatOptions,
  ) -> None:
    """Starts a chat with earlier messages, none of them fed yet."""
    self._model = model
    self._messages = messages
    self._max_tokens = options.max_completion_tokens
    self._sampler = sampler.Sampler(options, model.defaults)
    self._cache = None
    # The ids the cache holds, from position 0, and the one held back.
    self._context: list[int] = []
    self._held: int | None = None
    self._turns = 0
    # The turn's observer while it runs, if it observes.
    self._observer: anchor_observer.Observer | None = None

  def send(
      self,
      content: str,
      directory: pathlib.Path,
      on_chunk: Callable[[str], None],
      cancelled: Callable[[], bool],
  ) -> runner_pb2.Completion:
    """Runs one turn, capturing it into `directory`.

    Args:
        content: The user's message.
        directory: Where the turn's capture goes; created.
        on_chunk: Called with each piece of the reply's content.
        cancelled: Whether the client cancelled; asked between passes.

    Returns:
        The turn's completion.

    Raises:
        NoRoom: The context has no room for the turn, before any chunk;
          the chat is as it was.
        Cancelled: The client cancelled.
    """
    model = self._model
    messages = self._messages + [{'role': 'user', 'content': content}]
    ids = model.render(messages)
    token_count = len(ids)
    if token_count + 1 > model.context_length:
      # Counted from what the context holds already, as a turn is.
      used = len(self._context)
      raise NoRoom(
          f'The turn takes {token_count} positions and its reply one more; the '
          f'context holds {model.context_length}.',
          token_count + 1 - used,
          model.context_length - used,
      )
    known = self._context + ([] if self._held is None else [self._held])
    if self._cache is not None and ids[: len(known)] == known:
      first = len(self._context)
    else:
      # The rendering rewrites what the cache holds: replay it all.
      first = 0
      self._cache = transformers.DynamicCache(config=model.model.config)
    capture = capture_writer.CaptureWriter(directory)
    turn = _Turn()
    observer = None
    if model.observe != anchor_observer.NONE:
      full = model.full_turn_pending
      model.full_turn_pending = False
      observer = anchor_observer.Observer(
          model.model, model.memory_limit_bytes, model.modules, full
      )
    self._observer = observer
    try:
      with torch.no_grad():
        if first <= token_count - 2:
          self._prefill(capture, ids, first, token_count)
        finish = self._decode(capture, ids, first, turn, on_chunk, cancelled)
    except BaseException:
      if observer is not None:
        observer.close()
      raise
    finally:
      self._observer = None
    if observer is not None:
      wanted = _SIGNATURES if observer.mode == anchor_observer.FULL else ()
      for found in observer.finish(wanted):
        capture.anchor(found)
      graph = observer.graph
      if graph is not None:
        capture.graph(graph)
      capture.observation({
          'observe': anchor_observer.ANCHORS,
          'capture': observer.mode,
          'modules': [
              f'{module.module}:{module.edge}' for module in model.modules
          ],
          'memory_limit_bytes': observer.memory_limit_bytes,
          'captured': not observer.reason,
          'reason': observer.reason,
          'snapshot_bytes': observer.snapshot_bytes,
          'execution_graph': sorted(graph['steps']) if graph else [],
          'execution_graph_reason': observer.graph_reason,
      })
    last = turn.generated[-1]
    self._context = ids + turn.generated[:-1]
    self._held = last
    reply = ''.join(turn.chunks)
    self._messages = messages + [{'role': 'assistant', 'content': reply}]
    self._turns += 1
    steps = len(turn.generated)
    return runner_pb2.Completion(
        content=reply,
        finish_reason=finish,
        usage=runner_pb2.Usage(
            prompt_tokens=token_count,
            completion_tokens=steps,
            total_tokens=token_count + steps,
        ),
        model_debugger=runner_pb2.ModelDebugger(
            turn=self._turns,
            context_token_count=turn.fed[-1].position + 2,
            fed_tokens=turn.fed,
        ),
    )

  def _forward(
      self, ids: list[int], first: int, signature: str, step: int
  ) -> torch.Tensor:
    """Feeds `ids` at positions from `first`; returns the last logits.

    Observed, when the turn observes, as `signature` step `step`.

    Args:
      ids: The token IDs to feed.
      first: The starting position index.
      signature: The signature name (e.g. 'prefill' or 'decode').
      step: The step number.

    Returns:
      The logits tensor of the last position.
    """
    device = self._model.device

    def run() -> torch.Tensor:
      output = self._model.model(
          input_ids=torch.tensor([ids], device=device),
          position_ids=torch.arange(
              first, first + len(ids), device=device
          ).unsqueeze(0),
          past_key_values=self._cache,
          use_cache=True,
          logits_to_keep=1,
      )
      return output.logits[:, -1:, :]

    if self._observer is None:
      return run()
    return self._observer.forward(run, self._cache, signature, step)

  def _prefill(
      self,
      capture: capture_writer.CaptureWriter,
      ids: list[int],
      first: int,
      token_count: int,
  ) -> None:
    """Feeds positions first .. token_count - 2 in one pass, labelled step token_count."""
    positions = torch.arange(first, token_count - 1, dtype=torch.int32)
    capture.tensor('prefill', token_count, 'pre_input_pos', positions)
    capture.kv_cache('prefill', token_count, 'pre', self._cache)
    self._forward(
        ids[first : token_count - 1], first, 'prefill', token_count
    )
    capture.kv_cache('prefill', token_count, 'post', self._cache)

  def _decode(
      self,
      capture: capture_writer.CaptureWriter,
      ids: list[int],
      first: int,
      turn: _Turn,
      on_chunk: Callable[[str], None],
      cancelled: Callable[[], bool],
  ) -> runner_pb2.FinishReason:
    """Runs the decode steps from the input's last position.

    Args:
      capture: The capture writer to record tensors and KV cache to.
      ids: The sequence of token IDs.
      first: The starting position index.
      turn: The turn tracker for fed tokens.
      on_chunk: Callback invoked when a token chunk is emitted.
      cancelled: Predicate returning True if client cancelled.

    Returns:
      Why the decode steps ended.

    Raises:
      Cancelled: The client cancelled the turn.
    """
    model = self._model
    token_count = len(ids)
    turn.fed = [
        runner_pb2.FedToken(
            position=position,
            phase=_PREFILL,
            id=ids[position],
            text=model.text(ids[position]),
        )
        for position in range(first, token_count - 1)
    ]
    position, token = token_count - 1, ids[token_count - 1]
    emitted = ''
    while True:
      if cancelled():
        raise Cancelled()
      step = position + 1
      turn.fed.append(
          runner_pb2.FedToken(
              position=position,
              phase=_DECODE,
              id=token,
              text=model.text(token),
          )
      )
      capture.tensor(
          'decode',
          step,
          'pre_input_pos',
          torch.tensor([position], dtype=torch.int32),
      )
      logits = self._forward([token], position, 'decode', step)
      capture.tensor('decode', step, 'post_logits', logits[0])
      token = self._sampler.choose(logits)
      turn.generated.append(token)
      if token in model.stop_ids:
        finish = runner_pb2.FINISH_REASON_STOP
        break
      turn.released.append(token)
      capture.released(token, model.text(token))
      text = model.tokenizer.decode(turn.released, skip_special_tokens=True)
      # A token may end inside a character; its text waits for the rest.
      if text.startswith(emitted) and not text.endswith('�'):
        if len(text) > len(emitted):
          turn.chunks.append(text[len(emitted) :])
          on_chunk(turn.chunks[-1])
        emitted = text
      if (
          len(turn.generated) >= self._max_tokens
          or token_count + len(turn.generated) >= model.context_length
      ):
        finish = runner_pb2.FINISH_REASON_LENGTH
        break
      position += 1
    text = model.tokenizer.decode(turn.released, skip_special_tokens=True)
    if text.startswith(emitted) and len(text) > len(emitted):
      turn.chunks.append(text[len(emitted) :])
      on_chunk(turn.chunks[-1])
    capture.kv_cache('decode', position + 1, 'post', self._cache)
    return finish
