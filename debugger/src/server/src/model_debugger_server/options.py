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

"""The UI's option vocabulary, checked and turned into the protocol's.

A run (one side of a Session) is saved as the UI sends it: text values, ''
for the runtime default, 'on'/'off' for a switch. Its options become a
LoadRequest. A Chat's generation settings are JSON values by the UI's names
and become a CreateChatRequest. What the runtime takes beyond the protocol's
fixed fields is whatever the runner's RuntimeInfo schemas say; this module
never repeats those schemas, it validates against them.
"""

import json
import math
import re
import types
from typing import Any

import jsonschema
from jsonschema import exceptions as jsonschema_exceptions
from model_debugger.v1 import runner_pb2
from model_debugger_server import models

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
)
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
_BACKENDS = types.MappingProxyType(
    {'CPU': runner_pb2.CPU, 'GPU': runner_pb2.GPU}
)
DEFAULT_CONTEXT_LENGTH = 4096
_INT32_MAX = 2**31 - 1

DEFAULT_MAX_OUTPUT_TOKENS = 32
MAX_OUTPUT_TOKENS = 256
SAMPLER_TYPES = ('greedy', 'topK', 'topP')
_SYSTEM_PROMPT_LIMIT = 16000
# Generation settings with a number value: (lowest, highest, whole number).
_GENERATION_NUMBERS = types.MappingProxyType({
    'temperature': (0, 100, False),
    'topK': (1, _INT32_MAX, True),
    'topP': (0, 1, False),
    'seed': (0, _INT32_MAX, True),
    'thinkingBudget': (-1, _INT32_MAX, True),
    'maxOutputTokens': (1, MAX_OUTPUT_TOKENS, True),
})
# Generation switches, saved as 'on' or 'off', and the chat option each is.
_GENERATION_SWITCHES = types.MappingProxyType({
    'thinking': 'thinking',
    'applyPromptTemplate': 'apply_prompt_template',
    'speculativeDecoding': 'speculative_decoding',
})
# Generation settings that are a chat option of the runtime's own.
_GENERATION_RUNTIME_OPTIONS = types.MappingProxyType({
    **_GENERATION_SWITCHES,
    'thinkingBudget': 'thinking_token_budget',
})
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
_LABELS = types.MappingProxyType({
    'temperature': 'Temperature',
    'topK': 'Top K',
    'topP': 'Top P',
    'seed': 'Seed',
    'thinkingBudget': 'Thinking budget',
    'maxOutputTokens': 'Max output tokens',
})


def check_run_option(key: str, value: str) -> None:
  """Checks the text the UI saves for one run option.

  Only the encoding is checked here; whether the runtime takes the option
  and its value is decided against the runner's schema at Load.

  Args:
    key: One of RUNTIME_OPTION_KEYS.
    value: The saved text.

  Raises:
    ValueError: A switch that is not '', 'on' or 'off'.
  """
  if key in _RUN_SWITCHES and value not in ('', 'on', 'off'):
    raise ValueError(f'{key} must be on, off or the default.')


def context_length(run: dict) -> int:
  """Returns the context length a run asks for.

  Raises:
    ValueError: contextLength is neither '' nor a whole number in 1..2^31-1.
  """
  text = run.get('contextLength', '')
  if not text:
    return DEFAULT_CONTEXT_LENGTH
  if (
      not (text.isascii() and text.isdigit())
      or not 1 <= int(text) <= _INT32_MAX
  ):
    raise ValueError('The context length must be a positive whole number.')
  return int(text)


def _schema(text: str) -> dict:
  return json.loads(text) if text else {}


def _properties(schema_text: str) -> dict:
  return _schema(schema_text).get('properties', {})


def _validate(instance: dict, schema_text: str, what: str, names: dict) -> None:
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


def _load_value(key: str, text: str, schema: dict) -> Any:
  """The JSON value of one run option, by the type the runtime declares."""
  if key in _RUN_SWITCHES:
    return text == 'on'
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
    run: dict, model: models.Model, info: runner_pb2.RunnerInfo
) -> runner_pb2.LoadRequest:
  """Returns the LoadRequest of a run.

  Args:
    run: A run as SessionStore keeps it.
    model: The run's model.
    info: The runner's info; its load_options_schema decides what the runtime
      takes.

  Returns:
    The request.

  Raises:
    ValueError: The run asks for something the runner does not take.
  """
  backend = run.get('backend', '')
  if backend not in _BACKENDS:
    raise ValueError('The backend must be CPU or GPU.')
  properties = _properties(info.runtime.load_options_schema)
  runtime_options = {
      key: _load_value(key, run[key], properties.get(key, {}))
      for key in RUNTIME_OPTION_KEYS
      if run.get(key)
  }
  _validate(
      runtime_options,
      info.runtime.load_options_schema,
      'Session',
      {},
  )
  request = runner_pb2.LoadRequest(
      sha256=model.sha256,
      device=_BACKENDS[backend],
      context_length=context_length(run),
  )
  request.runtime_options.update(runtime_options)
  return request


def _side_generation(config: Any) -> dict:
  """Checks one side's generation settings and returns those that are set."""
  if not isinstance(config, dict):
    raise ValueError('The generation settings are not valid.')
  unknown = (
      set(config)
      - set(_GENERATION_FIXED)
      - set(_GENERATION_NUMBERS)
      - set(_GENERATION_RUNTIME_OPTIONS)
  )
  if unknown:
    raise ValueError(
        'Unsupported generation settings: ' + ', '.join(sorted(unknown)) + '.'
    )
  clean = {}
  for key, (low, high, whole) in _GENERATION_NUMBERS.items():
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
      raise ValueError(f'{_LABELS[key]} must be between {low} and {high}.')
    clean[key] = int(number) if whole else number
  for key in _GENERATION_SWITCHES:
    value = config.get(key)
    if value not in (None, '', 'default', 'on', 'off'):
      raise ValueError(f'{key} must be on, off or the model default.')
    if value in ('on', 'off'):
      clean[key] = value
  sampler = config.get('samplerType')
  if sampler not in (None, '', 'default', *SAMPLER_TYPES):
    raise ValueError(
        'The sampler must be one of ' + ', '.join(SAMPLER_TYPES) + '.'
    )
  if sampler in SAMPLER_TYPES:
    clean['samplerType'] = sampler
  prompt = config.get('systemPrompt') or ''
  if not isinstance(prompt, str) or len(prompt) > _SYSTEM_PROMPT_LIMIT:
    raise ValueError(
        f'The system prompt must be text of at most {_SYSTEM_PROMPT_LIMIT} '
        'characters.'
    )
  if prompt:
    clean['systemPrompt'] = prompt
  # What each sampler takes; a setting it would not apply is refused rather
  # than recorded as if it had been.
  sampler = clean.get('samplerType', 'greedy')
  takes = {
      'greedy': (),
      'topK': ('topK', 'temperature'),
      'topP': ('topP', 'temperature'),
  }[sampler]
  for key in ('topK', 'topP', 'temperature'):
    if key in clean and key not in takes:
      raise ValueError(
          f'The {sampler} sampler does not take {_LABELS[key]}; reset it to '
          'its default or choose another sampler.'
      )
  if sampler != 'greedy' and sampler not in clean:
    raise ValueError(f'The {sampler} sampler needs {_LABELS[sampler]}.')
  return clean


def validate_generation(value: Any) -> dict:
  """Checks a Chat's generation settings, as the UI sends them.

  Args:
    value: `{ref: {...}, target: {...}}`; a setting that is None, '' or
      'default' is the default and is left out.

  Returns:
    The settings each side sets, by the UI's names.

  Raises:
    ValueError: A setting is unknown, out of range, or one its side's sampler
      does not take.
  """
  if not isinstance(value, dict) or set(value) != {'ref', 'target'}:
    raise ValueError('Reference and Target generation settings are required.')
  return {side: _side_generation(value[side]) for side in ('ref', 'target')}


def create_chat_request(
    generation: dict, info: runner_pb2.RunnerInfo
) -> runner_pb2.CreateChatRequest:
  """Returns the CreateChatRequest of one side's generation settings.

  The default sampler is greedy, top_k 1, the one deterministic choice: the
  Debugger compares two runs token by token.

  Args:
    generation: One side's settings, as `validate_generation` returns them.
    info: The runner's info; its chat_options_schema decides what the runtime
      takes.

  Returns:
    The request.

  Raises:
    ValueError: The settings ask for something the runner does not take.
  """
  generation = _side_generation(generation)
  request = runner_pb2.CreateChatRequest()
  if 'systemPrompt' in generation:
    request.messages.add(
        role=runner_pb2.ChatMessage.SYSTEM, content=generation['systemPrompt']
    )
  options = request.options
  options.max_completion_tokens = generation.get(
      'maxOutputTokens', DEFAULT_MAX_OUTPUT_TOKENS
  )
  sampler = generation.get('samplerType', 'greedy')
  if sampler == 'greedy':
    options.top_k = 1
  elif sampler == 'topK':
    options.top_k = generation['topK']
  else:
    options.top_p = generation['topP']
  if 'temperature' in generation:
    options.temperature = generation['temperature']
  if 'seed' in generation:
    options.seed = generation['seed']
  runtime_options = {}
  for key, name in _GENERATION_RUNTIME_OPTIONS.items():
    if key in generation:
      value = generation[key]
      runtime_options[name] = (
          value == 'on' if key in _GENERATION_SWITCHES else value
      )
  _validate(
      runtime_options,
      info.runtime.chat_options_schema,
      'Chat',
      {name: key for key, name in _GENERATION_RUNTIME_OPTIONS.items()},
  )
  options.runtime_options.update(runtime_options)
  return request


def supported_options(info: runner_pb2.RunnerInfo) -> list[str]:
  """Returns the UI's run options the runner takes."""
  properties = _properties(info.runtime.load_options_schema)
  return ['backend', 'contextLength'] + [
      key for key in RUNTIME_OPTION_KEYS if key in properties
  ]


def chat_option_keys(info: runner_pb2.RunnerInfo) -> list[str]:
  """Returns the UI's generation settings the runner takes."""
  properties = _properties(info.runtime.chat_options_schema)
  return list(_GENERATION_FIXED) + [
      key
      for key, name in _GENERATION_RUNTIME_OPTIONS.items()
      if name in properties
  ]
