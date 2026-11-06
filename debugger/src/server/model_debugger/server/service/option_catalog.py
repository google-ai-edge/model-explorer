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

"""The options the UI offers a Session and a Chat, described for each device.

This server keeps its own table of LiteRT-LM's options: what each is called,
what it does, its type, range and default, which backend it needs and which
samplers use it. A device's options are that table merged with the schemas
the last runner of a build there reported (RuntimeInfo.load_options_schema
and chat_options_schema): an option that runner does not take is offered as
unavailable, with the reason, and the runner's own range or choices replace
the table's. Before a runner of that build ran on the device the table alone
decides, so that no runner runs just to be asked. A Chat's settings are also
bounded by its side's run (options.chat_maximums): the reply by the context
length; and a sampling value a sampler defaults to the model's own
(options.MODEL_DEFAULTS) is described as the last runner to load the side's
model reported it (model_store). What a runner takes is decided again at
Load and CreateChat, against its schemas, by `session/options.py`.

Each description is a JSON object the UI reads as is (OptionDescription in the
UI's server/contracts/runtime.ts):

    key: The option's name in the Session's runs or the Chat's settings.
    label, help: What the UI shows.
    group: The UI's section of it.
    type: 'choice', 'integer', 'number', 'switch' or 'text'.
    choices: The values of a 'choice' or a 'switch', in the record's
      encoding; '' is the runtime's default.
    choice_labels: Of a 'choice', what the UI shows for each value it does
      not show as is.
    minimum, maximum: A number's range, or a text's longest length.
    default: The value a run gets when it sets none, in the record's encoding;
      None when the runtime decides it.
    requires: {key, value}: the option applies only while that option has that
      value (a CPU-only option needs the CPU text backend); absent otherwise.
    samplers: Of a sampling value, the samplers that use it.
    defaults: Of a sampling value, the default of each sampler whose default
      is not `default`: {value, source}, source saying where it comes from.
    fixed: Of a sampling value, what each other sampler sends for it, as the
      UI shows it.
    available: Whether a run may set it; `reason` says why not.
"""

import dataclasses
import json
from typing import Any

from model_debugger.protocol import runner_pb2
from model_debugger.server.session import options

# The runtimes, as the UI names them.
LITERT_LM = 'LiteRT-LM'
PYTORCH = 'PyTorch'
# What a device offers before a runner there said; a GPU Load fails if the
# runner reports no GPU.
BACKENDS = ('CPU', 'GPU')
# The context lengths offered; a run may ask for any from
# options.MIN_CONTEXT_LENGTH on that the model holds.
CONTEXT_LENGTHS = (2048, 4096, 8192, 16384, 32768)
# What LiteRT-LM's runner takes of options.RUNTIME_OPTION_KEYS, as far as this
# table knows before a runner says.
ENGINE_OPTION_KEYS = (
    'cpuThreads',
    'audioBackend',
    'audioCpuThreads',
    'visionBackend',
    'activationDataType',
    'prefillChunkSize',
    'enableYnnpack',
    'parallelFileSectionLoading',
    'singleThreadedExecution',
    'gpuMetalResidencySet',
    'loraRank',
    'audioLoraRank',
)
_NOT_TAKEN = "LiteRT-LM's runner does not take it."
_SWITCH = ('', 'on', 'off')


@dataclasses.dataclass(frozen=True)
class Option:
  """One option of the table, as the UI is offered it.

  The attributes are those of a description (see the module docstring),
  but `requires` is a (key, value) pair and `unavailable`, if set, is why no
  run may set it, whatever a runner takes.
  """

  key: str
  label: str
  group: str
  type: str
  help: str
  choices: tuple[Any, ...] = ()
  choice_labels: dict[str, Any] | None = None
  minimum: float | None = None
  maximum: float | None = None
  default: Any = None
  requires: tuple[str, str] | None = None
  samplers: tuple[str, ...] = ()
  fixed: dict[str, Any] | None = None
  unavailable: str = ''


def _number(key: str, group: str, help_text: str, **kwargs: Any) -> Option:
  """A Chat's number setting, in the range options.GENERATION_NUMBERS has."""
  low, high, whole = options.GENERATION_NUMBERS[key]
  return Option(
      key,
      options.LABELS[key],
      group,
      'integer' if whole else 'number',
      help_text,
      minimum=low,
      maximum=high,
      **kwargs,
  )


def _sampling(key: str, help_text: str) -> Option:
  """A sampling value: who uses it, its default, what the others fix it to.

  A default that is the model's own is the model's to say (_model_default).

  Args:
    key: The option's identifier.
    help_text: Help string describing the option.

  Returns:
    The configured sampling Option.
  """
  users = tuple(
      sampler
      for sampler in options.SAMPLER_TYPES
      if key not in options.SAMPLER_FIXED[sampler]
  )
  return _number(
      key,
      'Sampling',
      help_text,
      default=options.SAMPLING_DEFAULTS.get(key),
      samplers=users,
      fixed={
          sampler: f'{fixed[key]} · fixed by {options.SAMPLER_LABELS[sampler]}'
          for sampler, fixed in options.SAMPLER_FIXED.items()
          if key in fixed
      },
  )


def _model_default(
    sampler: str, key: str, model_sampling: dict[str, Any] | None
) -> dict[str, Any]:
  """A sampler's default of a value it defaults to the model's own.

  Args:
      sampler: The sampler.
      key: The value, in options.MODEL_DEFAULTS[sampler].
      model_sampling: The model's sampling, as model_store.Model has it; None if
        no runner reported it.

  Returns:
      {value, source}: the value a Chat that sets none gets, and whence.
  """
  value, from_model = options.sampler_default(sampler, key, model_sampling)
  # The value is resolved when the settings are saved, and saved with them.
  if from_model:
    source = "the model's own, saved with these settings"
  elif model_sampling is None:
    source = (
        "the Model Debugger's own, until the model's first start says "
        "the model's; saved with these settings"
    )
  else:
    source = (
        "the Model Debugger's own, as the model has none; saved with "
        'these settings'
    )
  return {'value': value, 'source': source}


# A Session's options, in the order the UI shows them: the fixed ones,
# backend and contextLength, and options.RUNTIME_OPTION_KEYS.
SESSION_OPTIONS = (
    Option(
        'backend',
        'Text backend',
        'Backends',
        'choice',
        'Where the model runs. GPU is offered until a runner on the device '
        'reports none.',
        choices=BACKENDS,
        default='CPU',
    ),
    Option(
        'cpuThreads',
        'Text CPU threads',
        'Backends',
        'integer',
        'Threads of the CPU text backend. Leave blank for the runtime default.',
        minimum=1,
        maximum=options.INT32_MAX,
        requires=('backend', 'CPU'),
    ),
    Option(
        'audioBackend',
        'Audio backend',
        'Backends',
        'choice',
        'Where the audio encoder runs. Leave it for the runtime default.',
        choices=('', *BACKENDS),
    ),
    Option(
        'audioCpuThreads',
        'Audio CPU threads',
        'Backends',
        'integer',
        'Threads of the CPU audio backend. Leave blank for the runtime '
        'default.',
        minimum=1,
        maximum=options.INT32_MAX,
        requires=('audioBackend', 'CPU'),
    ),
    Option(
        'visionBackend',
        'Vision backend',
        'Backends',
        'choice',
        'Where the vision encoder runs. Leave it for the runtime default.',
        choices=('', *BACKENDS),
    ),
    Option(
        'activationDataType',
        'Activation dtype',
        'Precision',
        'choice',
        'Activation data type of the engine (set_activation_data_type). The '
        "default keeps the runtime's; stored weights are not changed.",
        choices=('', 'float32', 'float16', 'int16', 'int8'),
    ),
    Option(
        'prefillChunkSize',
        'Prefill chunk size',
        'Batching',
        'integer',
        'Largest prefill chunk (set_prefill_chunk_size); CPU backend with '
        'dynamic models only. Leave blank for the runtime default.',
        minimum=1,
        maximum=65536,
        requires=('backend', 'CPU'),
    ),
    Option(
        'contextLength',
        'Context length',
        'Context',
        'choice',
        'Most tokens the model holds, prompt and reply together; the model '
        f'must support it. At least {options.MIN_CONTEXT_LENGTH}: Gemma 4 '
        'on the CPU does not load with a shorter one.',
        choices=tuple(str(length) for length in CONTEXT_LENGTHS),
        minimum=options.MIN_CONTEXT_LENGTH,
        maximum=options.INT32_MAX,
        default=str(options.DEFAULT_CONTEXT_LENGTH),
    ),
    Option(
        'enableYnnpack',
        'YNNPACK',
        'Advanced engine settings',
        'switch',
        'Delegate supported operations to YNNPACK before XNNPACK '
        '(set_enable_ynnpack).',
        choices=_SWITCH,
    ),
    Option(
        'speculativeDecoding',
        'Speculative decoding (engine)',
        'Advanced engine settings',
        'switch',
        'Engine-level speculative decoding.',
        choices=_SWITCH,
        unavailable=options.SPECULATIVE_DECODING,
    ),
    Option(
        'parallelFileSectionLoading',
        'Parallel section loading',
        'Advanced engine settings',
        'switch',
        'Load the sections of the .litertlm file in parallel (runtime '
        'default: on).',
        choices=_SWITCH,
    ),
    Option(
        'singleThreadedExecution',
        'Single-threaded execution',
        'Advanced engine settings',
        'switch',
        'Run the engine on a single thread (set_single_threaded_execution).',
        choices=_SWITCH,
    ),
    Option(
        'useRingbuffersLocalAttention',
        'Ring-buffer local attention',
        'Advanced engine settings',
        'switch',
        'Use ring buffers for local attention.',
        choices=_SWITCH,
    ),
    Option(
        'gpuDecodeStepsPerSync',
        'GPU decode steps per sync',
        'Advanced engine settings',
        'integer',
        'Decode steps between GPU synchronizations. GPU backend only.',
        minimum=1,
        maximum=1024,
        requires=('backend', 'GPU'),
    ),
    Option(
        'gpuWaitForWeightUploads',
        'GPU wait for weight uploads',
        'Advanced engine settings',
        'switch',
        'Wait for weight uploads before running. GPU backend only.',
        choices=_SWITCH,
        requires=('backend', 'GPU'),
    ),
    Option(
        'gpuMetalResidencySet',
        'GPU Metal residency set',
        'Advanced engine settings',
        'switch',
        'Keep GPU resources resident through a Metal residency set. GPU '
        'backend only.',
        choices=_SWITCH,
        requires=('backend', 'GPU'),
    ),
    Option(
        'loraRank',
        'LoRA rank',
        'Advanced engine settings',
        'integer',
        'Rank of the text LoRA adapter the model was prepared with.',
        minimum=0,
        maximum=1024,
    ),
    Option(
        'audioLoraRank',
        'Audio LoRA rank',
        'Advanced engine settings',
        'integer',
        'Rank of the audio LoRA adapter.',
        minimum=0,
        maximum=1024,
    ),
)

# A PyTorch run's options, in the order the UI shows them: this server's own
# table of what the PyTorch runner takes (options.PYTORCH_OPTION_NAMES).
PYTORCH_SESSION_OPTIONS = (
    Option(
        'backend',
        'Text backend',
        'Backends',
        'choice',
        'Where the model runs: the CPU, or the GPU (MPS on Apple, CUDA '
        'elsewhere). GPU is offered until the runner reports none.',
        choices=BACKENDS,
        default='CPU',
    ),
    Option(
        'cpuThreads',
        'Text CPU threads',
        'Backends',
        'integer',
        "Threads of PyTorch's CPU backend (torch.set_num_threads). Leave "
        'blank for the runtime default.',
        minimum=1,
        maximum=options.INT32_MAX,
        requires=('backend', 'CPU'),
    ),
    Option(
        'dtype',
        'Model dtype',
        'Precision',
        'choice',
        "The dtype the model's weights are loaded in. The default keeps the "
        "model's own (config.json's).",
        choices=('', *options.PYTORCH_DTYPES),
    ),
    dataclasses.replace(
        next(o for o in SESSION_OPTIONS if o.key == 'contextLength'),
        help=(
            'Most tokens the model holds, prompt and reply together; the '
            f'model must support it. At least {options.MIN_CONTEXT_LENGTH}.'
        ),
    ),
)
_PYTORCH_NOT_TAKEN = 'The PyTorch runner does not take it.'
# What a PyTorch run observes, which the editor's tensor capture shows, not
# its options: the runner's anchor_observer.
PYTORCH_TAP_TEXT = "Captures every layer's 4 anchors in prefill and decode."
# How, which the editor shows on that line's tooltip, not as a paragraph.
PYTORCH_TAP_DETAILS = (
    "Every decoder layer's four anchors (its input, where Attention ends, "
    'where Feed Forward ends, and its output), in every forward pass, with '
    'module hooks on the model as it runs: no copy of the model is written, '
    'and it captures on the CPU and on the GPU (MPS) alike. The execution '
    'graph Graph Diff draws of the model comes from the first turn it '
    'observes on the CPU. A turn that exceeds the snapshot budget completes '
    'without tensors and says why.'
)
PYTORCH_TAP_OPTIONS = (
    Option(
        'observe',
        'Capture the anchors',
        'Tensor capture',
        'choice',
        "Capture every decoder layer's four anchors each turn.",
        choices=('', options.OBSERVE_ANCHORS),
        choice_labels={'': 'Off', options.OBSERVE_ANCHORS: 'Anchors'},
    ),
    Option(
        'observeMemoryGiB',
        'Snapshot budget (GiB)',
        'Tensor capture',
        'integer',
        "collect()'s budget for one turn's snapshots, which it fails past. "
        'Leave blank for the runner default, 8 GiB: a turn of Gemma 4 E2B, '
        '"Name three colors." and a 24-token reply, takes 4.3 GiB.',
        minimum=1,
        maximum=options.MAX_OBSERVE_GIB,
        requires=('observe', options.OBSERVE_ANCHORS),
    ),
)

# A Chat's settings, in the order the UI shows them.
CHAT_OPTIONS = (
    Option(
        'systemPrompt',
        'System prompt',
        'Prompt',
        'text',
        'The system message the chat starts with.',
        maximum=options.SYSTEM_PROMPT_LIMIT,
    ),
    Option(
        'applyPromptTemplate',
        'Prompt template',
        'Prompt',
        'switch',
        "Apply the model's prompt template. Leave it for the model default.",
        choices=_SWITCH,
    ),
    Option(
        'samplerType',
        'Sampler',
        'Sampling',
        'choice',
        'How the next token is chosen. Greedy is the one deterministic '
        'choice, and the Debugger compares two runs token by token. The '
        'runtime has one sampler, top-k then top-p at a temperature: every '
        'sampler sends all four values, those it does not use fixed so that '
        'they change nothing. Top-K + Top-P is that sampler as it is.',
        choices=options.SAMPLER_TYPES,
        choice_labels=options.SAMPLER_LABELS,
        default=options.DEFAULT_SAMPLER,
    ),
    _sampling(
        'temperature',
        'Divides the logits before sampling. Top-K and Top-K + Top-P send '
        f'{options.SAMPLING_DEFAULTS["temperature"]} unless it is set; '
        'greedy sends it too, where it changes nothing.',
    ),
    _sampling(
        'topK',
        'How many of the likeliest tokens are kept. Top-K needs one; '
        "Top-K + Top-P, unless it is set, takes the model's own top_k, or "
        "the Model Debugger's own while the model has said none, and saves "
        'it with the settings. Greedy sends 1.',
    ),
    _sampling(
        'topP',
        'The probability mass Top-K + Top-P draws from, of the top_k '
        'kept; it needs one. Greedy and Top-K send 1, which cuts nothing.',
    ),
    _sampling(
        'seed',
        'Seeds the sampler: the same seed draws the same tokens. Top-K and '
        f'Top-K + Top-P send {options.SAMPLING_DEFAULTS["seed"]} unless it is '
        'set.',
    ),
    Option(
        'thinking',
        'Thinking',
        'Thinking',
        'switch',
        'Let the model think before it answers. Leave it for the model '
        'default.',
        choices=_SWITCH,
    ),
    _number(
        'thinkingBudget',
        'Thinking',
        'Most tokens the model may think for; -1 is no limit. Leave blank '
        'for the model default.',
    ),
    _number(
        'maxOutputTokens',
        'Output',
        'Most tokens a reply may have, thinking included; at most the '
        "side's context length. Leave blank for no limit: the reply ends at "
        'a stop token or when the context is full.',
    ),
    Option(
        'speculativeDecoding',
        'Speculative decoding',
        'Output',
        'switch',
        'Speculative decoding of the chat.',
        choices=_SWITCH,
        unavailable=options.SPECULATIVE_DECODING,
    ),
)


def _properties(schema_text: str) -> dict[str, Any]:
  return json.loads(schema_text).get('properties', {}) if schema_text else {}


def _describe(
    option: Option, taken: dict[str, Any] | None, reason: str
) -> dict[str, Any]:
  """The UI's description of one option.

  Args:
      option: The table's.
      taken: The runner's schema of it ({} if there is none to merge), or None
        if the runner does not take it.
      reason: Why it is unavailable when `taken` is None.

  Returns:
      The description.
  """
  description = {
      'key': option.key,
      'label': option.label,
      'group': option.group,
      'type': option.type,
      'help': option.help,
      'default': option.default,
      'available': True,
  }
  choices = list(option.choices)
  minimum, maximum = option.minimum, option.maximum
  if taken:
    if 'enum' in taken:
      choices = [c for c in choices if c in ('', *taken['enum'])]
    minimum = taken.get('minimum', minimum)
    maximum = taken.get('maximum', maximum)
  if choices:
    description['choices'] = choices
  if option.choice_labels:
    description['choice_labels'] = {
        choice: option.choice_labels[choice]
        for choice in choices
        if choice in option.choice_labels
    }
  if minimum is not None:
    description['minimum'] = minimum
  if maximum is not None:
    description['maximum'] = maximum
  if option.requires:
    key, value = option.requires
    description['requires'] = {'key': key, 'value': value}
  if option.samplers:
    description['samplers'] = list(option.samplers)
  if option.fixed:
    description['fixed'] = dict(option.fixed)
  unavailable = option.unavailable or ('' if taken is not None else reason)
  if unavailable:
    description['available'] = False
    description['reason'] = unavailable
  return description


def _runner(info: runner_pb2.RunnerInfo) -> str:
  """The runner that reported `info`, as a reason names it."""
  version = info.runtime.version[:12] or '(no version)'
  if info.runtime.name == runner_pb2.RuntimeInfo.PYTORCH:
    return f'The PyTorch runner last on this device, {version},'
  return f'The runner last on this device, LiteRT-LM {version},'


def session_options(info: runner_pb2.RunnerInfo | None) -> list[dict[str, Any]]:
  """The descriptions of a Session's options on a device.

  Args:
      info: What the last runner on the device reported, or None.

  Returns:
      One description per option, in SESSION_OPTIONS' order.
  """
  if info is None:
    load = {key: {} for key in ENGINE_OPTION_KEYS}
    reason = _NOT_TAKEN
  else:
    load = _properties(info.runtime.load_options_schema)
    reason = f'{_runner(info)} does not take it.'
  return _session_options(SESSION_OPTIONS, load, {}, reason, info)


def pytorch_session_options(
    info: runner_pb2.RunnerInfo | None,
) -> list[dict[str, Any]]:
  """The descriptions of a PyTorch run's options on this computer.

  Args:
      info: What the last PyTorch runner here reported, or None.

  Returns:
      One description per option, in PYTORCH_SESSION_OPTIONS' order.
  """
  names = options.PYTORCH_OPTION_NAMES
  if info is None:
    load = {name: {} for name in names.values()}
    reason = _PYTORCH_NOT_TAKEN
  else:
    load = _properties(info.runtime.load_options_schema)
    reason = f'{_runner(info)} does not take it.'
  return _session_options(PYTORCH_SESSION_OPTIONS, load, names, reason, info)


def pytorch_tap(info: runner_pb2.RunnerInfo | None) -> dict[str, Any]:
  """What a PyTorch run observes, as the editor's tensor capture shows it.

  Args:
      info: What the last PyTorch runner here reported, or None.

  Returns:
      `{available, reason, kind, text, details, options}`: `kind`
      'anchors', the run observing by its own options
      (PYTORCH_TAP_OPTIONS, described as a Session's options are) rather
      than by chosen points; `available` whether the runner takes them.
  """
  names = options.PYTORCH_OPTION_NAMES
  if info is None:
    load = {name: {} for name in names.values()}
    reason = _PYTORCH_NOT_TAKEN
  else:
    load = _properties(info.runtime.load_options_schema)
    reason = f'{_runner(info)} does not take it.'
  described = []
  for option in PYTORCH_TAP_OPTIONS:
    name = names[option.key]
    if name not in load:
      prop = None
    elif option.key == 'observeMemoryGiB':
      # The budget's schema is in bytes; its range here is in GiB.
      prop = {}
    else:
      prop = load[name]
    described.append(_describe(option, prop, reason))
  available = described[0]['available']
  return {
      'available': available,
      'reason': '' if available else described[0].get('reason', reason),
      'kind': options.OBSERVE_ANCHORS,
      'text': PYTORCH_TAP_TEXT,
      'details': PYTORCH_TAP_DETAILS,
      'options': described,
  }


def _session_options(
    table: tuple[Option, ...],
    load: dict[str, Any],
    names: dict[str, Any],
    reason: str,
    info: runner_pb2.RunnerInfo | None,
) -> list[dict[str, Any]]:
  """Describes a table of a Session's options against a load schema.

  Args:
    table: The options.
    load: The load schema's properties, by the runtime's names.
    names: The runtime's name of each option, if not the option's key.
    reason: Why an option the schema lacks is unavailable.
    info: What the last runner reported, or None.

  Returns:
    A list of described option dictionaries.
  """
  described = []
  for option in table:
    taken = load.get(names.get(option.key, option.key))
    if option.key in ('backend', 'contextLength'):
      # The protocol's own fields: every runner takes them.
      taken = {}
    if (
        option.key == 'backend'
        and info is not None
        and not info.host.HasField('gpu')
    ):
      option = dataclasses.replace(option, choices=('CPU',))
    described.append(_describe(option, taken, reason))
  return described


def chat_options(
    info: runner_pb2.RunnerInfo | None,
    maximums: dict[str, Any] | None = None,
    sampler_defaults: dict[str, Any] | None = None,
    runtime: str = LITERT_LM,
) -> list[dict[str, Any]]:
  """The descriptions of a Chat's settings on a device.

  Args:
      info: What the last runner on the device reported, or None.
      maximums: The highest value of a setting a side's run bounds, by its key
        (options.chat_maximums); None for no run.
      sampler_defaults: The sampling of the side's model, as model_store.Model
        has it; None if no runner reported it, or for no run. A default that is
        the model's own is described from it.
      runtime: LITERT_LM or PYTORCH: whose table decides while `info` is None.

  Returns:
      One description per setting, in CHAT_OPTIONS' order.
  """
  pytorch = runtime == PYTORCH or (
      info is not None and info.runtime.name == runner_pb2.RuntimeInfo.PYTORCH
  )
  if info is None and pytorch:
    chat = {}
    reason = _PYTORCH_NOT_TAKEN
  elif info is None:
    chat = {name: {} for name in options.GENERATION_RUNTIME_OPTIONS.values()}
    reason = _NOT_TAKEN
  else:
    chat = _properties(info.runtime.chat_options_schema)
    reason = f'{_runner(info)} does not take it.'
  described = []
  for option in CHAT_OPTIONS:
    name = options.GENERATION_RUNTIME_OPTIONS.get(option.key)
    # A protocol field (no runtime name): every runner takes it.
    taken = {} if name is None else chat.get(name)
    description = _describe(option, taken, reason)
    for sampler, keys in options.MODEL_DEFAULTS.items():
      if option.key in keys:
        description.setdefault('defaults', {})[sampler] = _model_default(
            sampler, option.key, sampler_defaults
        )
    bound = (maximums or {}).get(option.key)
    if bound is not None:
      description['maximum'] = min(description.get('maximum', bound), bound)
    described.append(description)
  return described


def runtime_options(
    info: runner_pb2.RunnerInfo | None = None,
    maximums: dict[str, Any] | None = None,
    sampler_defaults: dict[str, Any] | None = None,
    runtime: str = LITERT_LM,
) -> dict[str, Any]:
  """The options of a runtime in the UI's RuntimeCapability.

  Args:
      info: What the last runner on the device reported; None for the table
        alone.
      maximums: As chat_options() takes them.
      sampler_defaults: As chat_options() takes them.
      runtime: LITERT_LM or PYTORCH.

  Returns:
      Its `backends`, `session_options` and `chat_options`.
  """
  if runtime == PYTORCH:
    session = pytorch_session_options(info)
  else:
    session = session_options(info)
  backends = next(o for o in session if o['key'] == 'backend')['choices']
  return {
      'backends': backends,
      'session_options': session,
      'chat_options': chat_options(info, maximums, sampler_defaults, runtime),
  }
