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

"""A Session's settings, checked and turned into the requests runners get.

A run (one side of a Session) is saved as the UI sends it: text values, ''
for the runtime default, 'on'/'off' for a switch. Its options become a
LoadRequest. A Chat's generation settings are JSON values by the UI's names
and become a CreateChatRequest. What the runtime takes beyond the protocol's
fixed fields is whatever the runner's RuntimeInfo schemas say, and requests
are validated against them. How the UI is offered these options is
`service/option_catalog.py`'s.

A Chat's sampling is saved as the runner gets it: the sampler and all four of
top_k, top_p, temperature and seed, each the value chosen, the sampler's
default (sampler_default) or what the sampler fixes it to (SAMPLER_FIXED).
The runtime has one sampler, top-k then top-p at a temperature, and takes a
value left unset from the model, so a sampler sent alone would not run as
chosen. What was sent is index.json's.
"""

import json
import math
import re
import types
from typing import Any

import jsonschema
from jsonschema import exceptions as jsonschema_exceptions
from model_debugger.protocol import runner_pb2
from model_debugger.server.capture import capture_index
from model_debugger.server.session import model_store

# The UI's run options beyond backend and contextLength (RuntimeOptions in
# the UI's session_summary.ts). A runtime takes those its load schema names.
RUNTIME_OPTION_KEYS = (
    'cpuThreads',
    'audioBackend',
    'audioCpuThreads',
    'visionBackend',
    'activationDataType',
    'prefillChunkSize',
    'enableYnnpack',
    'speculativeDecoding',
    'parallelFileSectionLoading',
    'singleThreadedExecution',
    'useRingbuffersLocalAttention',
    'gpuDecodeStepsPerSync',
    'gpuWaitForWeightUploads',
    'gpuMetalResidencySet',
    'loraRank',
    'audioLoraRank',
    'dtype',
    'observe',
    'observeMemoryGiB',
)
# The run options a PyTorch runner takes, by their name in its
# runtime_options; the LiteRT-LM runner takes its own by the UI's names.
PYTORCH_OPTION_NAMES = types.MappingProxyType({
    'cpuThreads': 'cpuThreads',
    'dtype': 'dtype',
    'observe': 'observe',
    'observeMemoryGiB': 'observeMemoryLimitBytes',
})
# The run options only a PyTorch runner takes.
_PYTORCH_ONLY = ('dtype', 'observe', 'observeMemoryGiB')
# The dtypes a PyTorch run may load its model in; '' is the model's own.
PYTORCH_DTYPES = ('float32', 'float16', 'bfloat16')
# What a PyTorch run observes when `observe` is this: every decoder layer's
# four anchors, each turn (its runner's anchor_observer); '' observes none.
OBSERVE_ANCHORS = 'anchors'
# The PyTorch runner's Load option the server fills, not the user: the
# modules it observes beside the anchors, each {module, edge}.
OBSERVE_MODULES = 'observeModules'
# The most GiB a PyTorch run's snapshot budget may be (observeMemoryGiB).
MAX_OBSERVE_GIB = 1024
# Run options the UI saves as '', 'on' or 'off'.
_RUN_SWITCHES = frozenset((
    'enableYnnpack',
    'speculativeDecoding',
    'parallelFileSectionLoading',
    'singleThreadedExecution',
    'useRingbuffersLocalAttention',
    'gpuWaitForWeightUploads',
    'gpuMetalResidencySet',
))
_BACKENDS = {'CPU': runner_pb2.CPU, 'GPU': runner_pb2.GPU}
DEFAULT_CONTEXT_LENGTH = 4096
# The shortest context a run may ask for: Gemma 4 on the CPU fails to load
# below about 1535, so no shorter one is offered or taken.
MIN_CONTEXT_LENGTH = 2048
INT32_MAX = 2**31 - 1
# What no run may switch on yet: the capture index pairs one token with each
# decode step, and speculative decoding releases several.
SPECULATIVE_DECODING = (
    'Speculative decoding cannot be enabled for now: the capture index '
    'assumes one token per decode step.'
)

SAMPLER_TYPES = ('greedy', 'topK', 'topP')
# What the UI calls each sampler; `topP`, the records' name, cuts by top_k
# and then by top_p, as LiteRT-LM's does.
SAMPLER_LABELS = {
    'greedy': 'Greedy',
    'topK': 'Top-K',
    'topP': 'Top-K + Top-P',
}
DEFAULT_SAMPLER = 'greedy'
# The sampling values every Chat sends, by the UI's names.
SAMPLING_KEYS = ('topK', 'topP', 'temperature', 'seed')
# The value each sampler sends for a sampling value it does not use: top_k 1
# is greedy, whatever the rest; top_p 1 cuts nothing.
SAMPLER_FIXED = {
    'greedy': {'topK': 1, 'topP': 1.0, 'temperature': 1.0, 'seed': 0},
    'topK': {'topP': 1.0},
    'topP': {},
}
# The value of a sampling value a sampler uses when the Chat sets none; topK
# and topP have none, as the sampler of that name needs it, but as
# MODEL_DEFAULTS gives one.
SAMPLING_DEFAULTS = {'temperature': 1.0, 'seed': 0}
# The sampling values a sampler defaults to the model's own when the Chat
# sets none, by the field of LoadResponse.sampler_defaults that has it, and
# this table's value for a model that reports none (Gemma 4 E2B reports none
# on LiteRT-LM 61dcfc77): Top-K + Top-P always sends a top_k, as LiteRT-LM's
# runs top_k 1, greedy, without one.
MODEL_DEFAULTS = {'topP': {'topK': ('top_k', 64)}}
SYSTEM_PROMPT_LIMIT = 16000
# Generation settings with a number value: (lowest, highest, whole number).
GENERATION_NUMBERS = {
    'temperature': (0, 100, False),
    'topK': (1, INT32_MAX, True),
    'topP': (0, 1, False),
    'seed': (0, INT32_MAX, True),
    'thinkingBudget': (-1, INT32_MAX, True),
    'maxOutputTokens': (1, INT32_MAX, True),
}
# Generation switches, saved as 'on' or 'off', and the chat option each is.
_GENERATION_SWITCHES = {
    'thinking': 'thinking',
    'applyPromptTemplate': 'apply_prompt_template',
    'speculativeDecoding': 'speculative_decoding',
}
# Generation settings that are a chat option of the runtime's own.
GENERATION_RUNTIME_OPTIONS = {
    **_GENERATION_SWITCHES,
    'thinkingBudget': 'thinking_token_budget',
}
# Generation settings every runtime takes: the protocol's fixed fields.
_GENERATION_FIXED = (
    'maxOutputTokens',
    'systemPrompt',
    'samplerType',
    'temperature',
    'topK',
    'topP',
    'seed',
)
# The UI's name of each generation setting with a number value.
LABELS = {
    'temperature': 'Temperature',
    'topK': 'Top K',
    'topP': 'Top P',
    'seed': 'Seed',
    'thinkingBudget': 'Thinking budget',
    'maxOutputTokens': 'Max output tokens',
}


def sampler_default(
    sampler: str, key: str, model_sampling: dict[str, Any] | None
) -> tuple[Any, bool] | None:
  """The value a sampler gives a sampling value it uses when none is set.

  Args:
      sampler: One of SAMPLER_TYPES.
      key: One of SAMPLING_KEYS the sampler does not fix.
      model_sampling: The model's own sampling, as model_store.Model has it
        (sampler_defaults); None if no runner reported it.

  Returns:
      (the value, whether it is the model's), or None if the sampler needs
      the value set.
  """
  if key in MODEL_DEFAULTS.get(sampler, {}):
    field, table = MODEL_DEFAULTS[sampler][key]
    if model_sampling and field in model_sampling:
      return model_sampling[field], True
    return table, False
  if key in SAMPLING_DEFAULTS:
    return SAMPLING_DEFAULTS[key], False
  return None


def check_run_option(key: str, value: str) -> None:
  """Checks the text the UI saves for one run option.

  Only the encoding is checked here; whether the runtime takes the option
  and its value is decided against the runner's schema at Load.

  Args:
      key: One of RUNTIME_OPTION_KEYS.
      value: The saved text.

  Raises:
      ValueError: A switch that is not '', 'on' or 'off', or speculative
        decoding switched on.
  """
  if key in _RUN_SWITCHES and value not in ('', 'on', 'off'):
    raise ValueError(f'{key} must be on, off or the default.')
  if key == 'speculativeDecoding' and value == 'on':
    raise ValueError(SPECULATIVE_DECODING)
  if key == 'dtype' and value not in ('', *PYTORCH_DTYPES):
    raise ValueError(
        'The model dtype must be one of ' + ', '.join(PYTORCH_DTYPES) + '.'
    )
  if key == 'observe' and value not in ('', OBSERVE_ANCHORS):
    raise ValueError(
        f'What a PyTorch run observes must be {OBSERVE_ANCHORS} or none.'
    )
  if key == 'observeMemoryGiB' and not (
      not value
      or (
          value.isascii()
          and value.isdigit()
          and 1 <= int(value) <= MAX_OBSERVE_GIB
      )
  ):
    raise ValueError(
        'The snapshot budget must be a whole number of GiB from 1 to '
        f'{MAX_OBSERVE_GIB}.'
    )


def context_length(run: dict[str, Any]) -> int:
  """Returns the context length a run asks for.

  Args:
    run: A mapping of run configuration options.

  Returns:
    The context length requested by the run, or the default context length.

  Raises:
    ValueError: ContextLength is neither '' nor a whole number from
      MIN_CONTEXT_LENGTH to 2^31-1.
  """
  text = run.get('contextLength', '')
  if not text:
    return DEFAULT_CONTEXT_LENGTH
  if not (text.isascii() and text.isdigit()) or int(text) > INT32_MAX:
    raise ValueError('The context length must be a positive whole number.')
  if int(text) < MIN_CONTEXT_LENGTH:
    raise ValueError(
        f'The context length must be at least {MIN_CONTEXT_LENGTH}: '
        'Gemma 4 on the CPU does not load with a shorter one.'
    )
  return int(text)


def chat_maximums(run: dict[str, Any]) -> dict[str, Any]:
  """The highest value of each Chat setting a side's run bounds.

  That is the reply's length, when the Chat sets one: never more than the
  context length the run asks for. A run saved before MIN_CONTEXT_LENGTH
  may ask for less; its chat is still bounded by what it asks for.

  Args:
      run: A run as SessionStore keeps it.

  Returns:
      The highest value by the UI's name of the setting.
  """
  text = run.get('contextLength', '')
  if text.isascii() and text.isdigit() and 0 < int(text) <= INT32_MAX:
    limit = int(text)
  else:
    limit = DEFAULT_CONTEXT_LENGTH
  return {'maxOutputTokens': limit}


def _schema(text: str) -> dict[str, Any]:
  return json.loads(text) if text else {}


def _properties(schema_text: str) -> dict[str, Any]:
  return _schema(schema_text).get('properties', {})


def _validate(
    instance: dict[str, Any], schema_text: str, what: str, names: dict[str, Any]
) -> None:
  """Validates runtime options against a runner's schema.

  Args:
      instance: The runtime options.
      schema_text: The runner's JSON Schema for them.
      what: 'Session' or 'Chat': whose options they are, as the UI says.
      names: The UI's name of each option, by its runtime name.

  Raises:
      ValueError: The schema refuses them; the message names the UI's option.
  """
  schema = _schema(schema_text)
  error = jsonschema_exceptions.best_match(
      jsonschema.Draft202012Validator(schema).iter_errors(instance)
  )
  if error is None:
    return
  if error.validator == 'additionalProperties':
    unknown = sorted(set(instance) - set(schema.get('properties', {})))
    raise ValueError(
        f"The runner's runtime does not take the {what} option "
        f'{", ".join(names.get(key, key) for key in unknown)}; reset it to '
        'its default.'
    )
  path = list(error.absolute_path)
  where = names.get(path[0], path[0]) if path else 'options'
  raise ValueError(f'{what} option {where}: {error.message}.')


def _load_value(key: str, text: str, schema: dict[str, Any]) -> Any:
  """The JSON value of one run option, by the type the runtime declares."""
  if key in _RUN_SWITCHES:
    return text == 'on'
  if key == 'observeMemoryGiB':
    return int(text) * 2**30
  kind = schema.get('type')
  if kind == 'integer' and re.fullmatch(r'-?\d+', text):
    return int(text)
  if kind == 'number':
    try:
      return float(text)
    except ValueError:
      pass
  return text


def load_request(
    run: dict[str, Any],
    model: model_store.Model,
    info: runner_pb2.RunnerInfo,
    modules: list[dict[str, Any]] | None = None,
) -> runner_pb2.LoadRequest:
  """Returns the LoadRequest of a run.

  Args:
      run: A run as SessionStore keeps it.
      model: The run's model.
      info: The runner's info; its load_options_schema decides what the runtime
        takes.
      modules: The modules a PyTorch run observes beside the anchors, each
        `{module, edge}` (prepared_models' observe_modules()), as its runner
        takes them (OBSERVE_MODULES); none sent if empty.

  Returns:
      The request.

  Raises:
      ValueError: The run asks for something the runner does not take.
  """
  backend = run.get('backend', '')
  if backend not in _BACKENDS:
    raise ValueError('The backend must be CPU or GPU.')
  properties = _properties(info.runtime.load_options_schema)
  if info.runtime.name == runner_pb2.RuntimeInfo.PYTORCH:
    names = PYTORCH_OPTION_NAMES
  else:
    names = {
        key: key for key in RUNTIME_OPTION_KEYS if key not in _PYTORCH_ONLY
    }
  runtime_options = {}
  for key, name in names.items():
    if run.get(key):
      runtime_options[name] = _load_value(
          key, run[key], properties.get(name, {})
      )
  if modules:
    runtime_options[OBSERVE_MODULES] = [dict(entry) for entry in modules]
  _validate(
      runtime_options,
      info.runtime.load_options_schema,
      'Session',
      {name: key for key, name in names.items()},
  )
  request = runner_pb2.LoadRequest(
      sha256=model.sha256,
      device=_BACKENDS[backend],
      context_length=context_length(run),
  )
  request.runtime_options.update(runtime_options)
  return request


def _side_generation(
    config: Any, model_sampling: dict[str, Any] | None = None
) -> dict[str, Any]:
  """Checks one side's generation settings; returns what the side runs with.

  That is what is set, and the sampler and every sampling value besides
  (see SAMPLER_FIXED and sampler_default). A reply's length is what is
  set or nothing: a Chat that sets none ends its reply at a stop token or
  at a full context alone (create_chat_request).

  Args:
      config: The side's settings, as the UI sends them.
      model_sampling: The side's model's own sampling, as sampler_default()
        takes it.

  Returns:
      The generation settings that the side runs with.
  """
  if not isinstance(config, dict):
    raise ValueError('The generation settings are not valid.')
  unknown = (
      set(config)
      - set(_GENERATION_FIXED)
      - set(GENERATION_NUMBERS)
      - set(GENERATION_RUNTIME_OPTIONS)
  )
  if unknown:
    raise ValueError(
        'Unsupported generation settings: ' + ', '.join(sorted(unknown)) + '.'
    )
  clean = {}
  for key, (low, high, whole) in GENERATION_NUMBERS.items():
    number = config.get(key)
    if number is None:
      continue
    if (
        isinstance(number, bool)
        or not isinstance(number, (int, float))
        or not math.isfinite(number)
        or not low <= number <= high
        or (whole and int(number) != number)
    ):
      raise ValueError(f'{LABELS[key]} must be between {low} and {high}.')
    clean[key] = int(number) if whole else number
  for key in _GENERATION_SWITCHES:
    value = config.get(key)
    if value not in (None, '', 'default', 'on', 'off'):
      raise ValueError(f'{key} must be on, off or the model default.')
    if value in ('on', 'off'):
      clean[key] = value
  if clean.get('speculativeDecoding') == 'on':
    raise ValueError(SPECULATIVE_DECODING)
  sampler = config.get('samplerType')
  if sampler not in (None, '', 'default', *SAMPLER_TYPES):
    raise ValueError(
        'The sampler must be one of '
        + ', '.join(SAMPLER_LABELS[s] for s in SAMPLER_TYPES)
        + '.'
    )
  prompt = config.get('systemPrompt') or ''
  if not isinstance(prompt, str) or len(prompt) > SYSTEM_PROMPT_LIMIT:
    raise ValueError(
        f'The system prompt must be text of at most {SYSTEM_PROMPT_LIMIT} '
        'characters.'
    )
  if prompt:
    clean['systemPrompt'] = prompt
  sampler = sampler if sampler in SAMPLER_TYPES else DEFAULT_SAMPLER
  clean['samplerType'] = sampler
  # Every sampling value is set: one the sampler does not use is what the
  # sampler fixes it to, and any other value of it is refused rather than
  # recorded as if it had been applied.
  fixed = SAMPLER_FIXED[sampler]
  name = SAMPLER_LABELS[sampler]
  for key in SAMPLING_KEYS:
    if key in fixed:
      if key in clean and clean[key] != fixed[key]:
        raise ValueError(
            f'The {name} sampler does not take {LABELS[key]}; '
            'reset it to its default or choose another sampler.'
        )
      clean[key] = fixed[key]
    elif key not in clean:
      default = sampler_default(sampler, key, model_sampling)
      if default is None:
        raise ValueError(f'The {name} sampler needs {LABELS[key]}.')
      clean[key] = default[0]
  return clean


def validate_generation(
    value: Any,
    runs: list[Any] | None = None,
    model_sampling: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Checks a Session's chat settings, as the UI sends them.

  Args:
      value: `{ref: {...}, target: {...}}`; a setting that is None, '' or
        'default' is the default.
      runs: The Session's runs, if known: no reply may be longer than its side's
        context length.
      model_sampling: Each side's model's own sampling, by side, as
        sampler_default() takes it; None if not known.

  Returns:
      The settings each side runs with, by the UI's names: those set, the
      sampling values always among them.

  Raises:
      ValueError: A setting is unknown, out of range, or one its side's
        sampler does not take.
  """
  if not isinstance(value, dict) or set(value) != set(capture_index.SIDES):
    raise ValueError('Reference and Target generation settings are required.')
  generation = {
      side: _side_generation(value[side], (model_sampling or {}).get(side))
      for side in capture_index.SIDES
  }
  for run in runs or []:
    side = generation.get(run.get('id'))
    if side is None:
      continue
    for key, limit in chat_maximums(run).items():
      if side.get(key, 0) > limit:
        raise ValueError(
            f'{LABELS[key]} must be at most the context length, {limit}.'
        )
  return generation


def summary(generation: dict[str, Any]) -> str:
  """One side's chat settings in a few words, as a chat is listed.

  The sampler, with its values unless greedy, whether the chat opens with
  a system prompt, then any reply length and thinking set: 'Greedy
  · no system prompt', say.

  Args:
      generation: One side's settings, as `validate_generation` returns them; a
        record's of an older server may lack any.

  Returns:
      A short text summary of the side's chat settings.
  """
  sampler = generation.get('samplerType') or DEFAULT_SAMPLER
  words = SAMPLER_LABELS.get(sampler, sampler)
  if sampler == 'topK':
    words = f'Top-K {generation.get("topK")}'
  elif sampler == 'topP':
    words = f'Top-K {generation.get("topK")} + Top-P {generation.get("topP")}'
  if sampler != DEFAULT_SAMPLER:
    words += f', temperature {generation.get("temperature")}'
  parts = [
      words,
      'system prompt' if generation.get('systemPrompt') else 'no system prompt',
  ]
  if 'maxOutputTokens' in generation:
    parts.append(f'at most {generation["maxOutputTokens"]} tokens')
  if generation.get('thinking') in ('on', 'off'):
    parts.append(f'thinking {generation["thinking"]}')
  return ' · '.join(parts)


def create_chat_request(
    generation: dict[str, Any],
    info: runner_pb2.RunnerInfo,
    context_tokens: int,
    model_sampling: dict[str, Any] | None = None,
) -> runner_pb2.CreateChatRequest:
  """Returns the CreateChatRequest of one side's generation settings.

  A reply is as long as the settings' maxOutputTokens; without one, it
  may fill the context: max_completion_tokens is the side's context
  length, which the protocol requires to be set, and a reply then ends at
  a stop token or at a full context alone, on every runtime.

  The default sampler is greedy, top_k 1, the one deterministic choice: the
  Debugger compares two runs token by token. All four sampling values are
  sent as the settings have them, so the runner takes none from the model;
  one the settings lack (a record saved before it was always set) is the
  default the settings dialog shows, the model's own where it has one.

  Args:
      generation: One side's settings, as `validate_generation` returns them.
      info: The runner's info; its chat_options_schema decides what the runtime
        takes.
      context_tokens: The context length the side loads with (context_length()).
      model_sampling: The side's model's own sampling, as sampler_default()
        takes it; None if not known.

  Returns:
      The request.

  Raises:
      ValueError: The settings ask for something the runner does not take.
  """
  generation = _side_generation(generation, model_sampling)
  request = runner_pb2.CreateChatRequest()
  if 'systemPrompt' in generation:
    request.messages.add(
        role=runner_pb2.ChatMessage.SYSTEM,
        content=generation['systemPrompt'],
    )
  options = request.options
  options.max_completion_tokens = generation.get(
      'maxOutputTokens', context_tokens
  )
  options.top_k = generation['topK']
  options.top_p = generation['topP']
  options.temperature = generation['temperature']
  options.seed = generation['seed']
  runtime_options = {}
  for key, name in GENERATION_RUNTIME_OPTIONS.items():
    if key in generation:
      value = generation[key]
      if key in _GENERATION_SWITCHES:
        value = value == 'on'
      runtime_options[name] = value
  _validate(
      runtime_options,
      info.runtime.chat_options_schema,
      'Chat',
      {name: key for key, name in GENERATION_RUNTIME_OPTIONS.items()},
  )
  options.runtime_options.update(runtime_options)
  return request
