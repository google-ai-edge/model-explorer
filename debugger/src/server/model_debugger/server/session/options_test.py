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

"""Tests of the UI's options turned into the protocol's."""

import pathlib

from absl.testing import parameterized
from google.protobuf import json_format
from model_debugger.protocol import runner_pb2
from model_debugger.server.session import model_store
from model_debugger.server.session import options
from model_debugger.server.testing import fake_runner
from model_debugger.server.testing import test_main

# The schemas the LiteRT-LM runner declares (litert_lm_runtime.cc).
LOAD_SCHEMA = """{
  "type": "object",
  "properties": {
    "cpuThreads": {"type": "integer", "minimum": 1},
    "audioCpuThreads": {"type": "integer", "minimum": 1},
    "prefillChunkSize": {"type": "integer", "minimum": 1},
    "loraRank": {"type": "integer", "minimum": 0},
    "audioLoraRank": {"type": "integer", "minimum": 0},
    "enableYnnpack": {"type": "boolean"},
    "parallelFileSectionLoading": {"type": "boolean"},
    "singleThreadedExecution": {"type": "boolean"},
    "gpuMetalResidencySet": {"type": "boolean"},
    "visionBackend": {"enum": ["CPU", "GPU"]},
    "audioBackend": {"enum": ["CPU", "GPU"]},
    "activationDataType": {"enum": ["float32", "float16", "int16", "int8"]}
  },
  "additionalProperties": false
}"""
CHAT_SCHEMA = """{
  "type": "object",
  "properties": {
    "thinking": {"type": "boolean"},
    "thinking_token_budget": {"type": "integer", "minimum": -1},
    "apply_prompt_template": {"type": "boolean"},
    "speculative_decoding": {"type": "boolean"}
  },
  "additionalProperties": false
}"""
INFO = runner_pb2.RunnerInfo(
    runtime=runner_pb2.RuntimeInfo(
        name=runner_pb2.RuntimeInfo.LITERT_LM,
        version='abc',
        load_options_schema=LOAD_SCHEMA,
        chat_options_schema=CHAT_SCHEMA,
    ),
)
# The context length a side loads with.
CONTEXT = 4096
# A model's own sampling, as model_store.Model keeps what a Load reported.
MODEL_SAMPLING = {'type': 'TOP_P', 'top_k': 48, 'top_p': 0.95, 'temperature': 1}
MODEL = model_store.Model(
    artifact='model-aa',
    sha256='aa' * 32,
    name='m.litertlm',
    size=1,
    path=pathlib.Path('/m.litertlm'),
)


def _json(message):
  return json_format.MessageToDict(message, preserving_proto_field_name=True)


class OptionsTest(parameterized.TestCase):
  """Tests of the UI's options turned into the protocol's."""

  def test_load_request_converts_ui_text(self):
    """Load request converts ui text."""
    request = options.load_request(
        {
            'backend': 'GPU',
            'contextLength': '2048',
            'cpuThreads': '4',
            'enableYnnpack': 'off',
            'singleThreadedExecution': 'on',
            'activationDataType': 'float16',
            'loraRank': '',
        },
        MODEL,
        INFO,
    )
    self.assertEqual(
        _json(request),
        {
            'sha256': 'aa' * 32,
            'device': 'GPU',
            'context_length': 2048,
            'runtime_options': {
                'cpuThreads': 4,
                'enableYnnpack': False,
                'singleThreadedExecution': True,
                'activationDataType': 'float16',
            },
        },
    )

  def test_load_request_defaults(self):
    """Load request defaults."""
    request = options.load_request(
        {'backend': 'CPU', 'contextLength': ''}, MODEL, INFO
    )
    self.assertEqual(request.device, runner_pb2.CPU)
    self.assertEqual(request.context_length, options.DEFAULT_CONTEXT_LENGTH)
    self.assertFalse(request.runtime_options.fields)

  def test_a_pytorch_run_that_observes_sends_its_budget_in_bytes(self):
    """The observe options go by the PyTorch runner's names."""
    request = options.load_request(
        {'backend': 'CPU', 'observe': 'anchors', 'observeMemoryGiB': '8'},
        MODEL,
        fake_runner.PYTORCH_INFO,
    )
    self.assertEqual(
        dict(request.runtime_options),
        {'observe': 'anchors', 'observeMemoryLimitBytes': 8 * 2**30},
    )
    # The LiteRT-LM runner is never sent them.
    request = options.load_request(
        {'backend': 'CPU', 'observe': 'anchors'}, MODEL, INFO
    )
    self.assertFalse(request.runtime_options.fields)
    for key, value in (('observe', 'modules'), ('observeMemoryGiB', '0')):
      with self.assertRaises(ValueError):
        options.check_run_option(key, value)

  def test_a_pytorch_run_is_sent_the_modules_the_server_pairs(self):
    """observeModules, the server's: what the LiteRT-LM points pair with."""
    modules = [{'module': 'model.language_model.layers.3.mlp', 'edge': 'out'}]
    request = options.load_request(
        {'backend': 'CPU', 'observe': 'anchors'},
        MODEL,
        fake_runner.PYTORCH_INFO,
        modules,
    )
    self.assertEqual(
        json_format.MessageToDict(request.runtime_options),
        {'observe': 'anchors', options.OBSERVE_MODULES: modules},
    )
    # A runner that takes no such option refuses the Session.
    with self.assertRaisesRegex(ValueError, 'observeModules'):
      options.load_request({'backend': 'CPU'}, MODEL, INFO, modules)

  @parameterized.parameters(
      (
          {'useRingbuffersLocalAttention': 'on'},
          'does not take the Session option useRingbuffersLocalAttention',
      ),
      (
          {'gpuDecodeStepsPerSync': '4'},
          'does not take the Session option gpuDecodeStepsPerSync',
      ),
      ({'cpuThreads': '0'}, 'Session option cpuThreads: 0 is less than'),
      ({'cpuThreads': 'four'}, 'Session option cpuThreads'),
      ({'activationDataType': 'float8'}, 'Session option activationDataType'),
      ({'contextLength': '0'}, 'context length must be at least 2048'),
      ({'contextLength': '2047'}, 'context length must be at least 2048'),
      ({'contextLength': '1e3'}, 'positive whole number'),
      ({'contextLength': str(2**31)}, 'positive whole number'),
      ({'backend': 'NPU'}, 'CPU or GPU'),
  )
  def test_load_request_refuses(self, run, message):
    """Load request refuses."""
    with self.assertRaisesRegex(ValueError, message):
      options.load_request({'backend': 'CPU', **run}, MODEL, INFO)

  def test_create_chat_request_default_is_greedy(self):
    """Greedy, every sampling value sent; the reply may fill the context."""
    request = options.create_chat_request({}, INFO, CONTEXT)
    self.assertEqual(
        _json(request),
        {
            'options': {
                'max_completion_tokens': CONTEXT,
                'top_k': 1,
                'top_p': 1.0,
                'temperature': 1.0,
                'seed': 0,
            }
        },
    )

  def test_create_chat_request_maps_every_setting(self):
    """Create chat request maps every setting."""
    request = options.create_chat_request(
        {
            'systemPrompt': 'Be brief.',
            'maxOutputTokens': 64,
            'samplerType': 'topP',
            'topK': 48,
            'topP': 0.5,
            'temperature': 0.75,
            'seed': 7,
            'thinking': 'on',
            'thinkingBudget': 128,
            'applyPromptTemplate': 'off',
            'speculativeDecoding': 'off',
        },
        INFO,
        CONTEXT,
    )
    self.assertEqual(
        _json(request),
        {
            'messages': [{'role': 'SYSTEM', 'content': 'Be brief.'}],
            'options': {
                'max_completion_tokens': 64,
                'top_k': 48,
                'top_p': 0.5,
                'temperature': 0.75,
                'seed': 7,
                'runtime_options': {
                    'thinking': True,
                    'thinking_token_budget': 128,
                    'apply_prompt_template': False,
                    'speculative_decoding': False,
                },
            },
        },
    )

  def test_create_chat_request_top_k(self):
    """Top K 40 sends a top_p that cuts nothing, not the model's."""
    request = options.create_chat_request(
        {'samplerType': 'topK', 'topK': 40},
        INFO,
        CONTEXT,
    )
    self.assertEqual(
        _json(request.options),
        {
            'max_completion_tokens': CONTEXT,
            'top_k': 40,
            'top_p': 1.0,
            'temperature': 1.0,
            'seed': 0,
        },
    )

  def test_create_chat_request_top_k_top_p(self):
    """Top-K + Top-P sends the top_k and top_p it is given, as given."""
    request = options.create_chat_request(
        {'samplerType': 'topP', 'topK': 20, 'topP': 0.5},
        INFO,
        CONTEXT,
    )
    self.assertEqual(
        _json(request.options),
        {
            'max_completion_tokens': CONTEXT,
            'top_k': 20,
            'top_p': 0.5,
            'temperature': 1.0,
            'seed': 0,
        },
    )

  def test_top_k_top_p_defaults_top_k_to_the_models(self):
    """Its top_k is the model's default, else the table's 64."""
    for sampling, top_k in (
        (MODEL_SAMPLING, 48),
        ({'top_p': 0.95}, 64),
        ({}, 64),
        (None, 64),
    ):
      generation = options.validate_generation(
          {'ref': {'samplerType': 'topP', 'topP': 0.5}, 'target': {}},
          model_sampling={'ref': sampling},
      )
      self.assertEqual(generation['ref']['topK'], top_k)
      self.assertEqual(
          options.create_chat_request(
              generation['ref'],
              INFO,
              CONTEXT,
          ).options.top_k,
          top_k,
      )
    self.assertEqual(
        options.sampler_default('topP', 'topK', MODEL_SAMPLING),
        (48, True),
    )
    self.assertEqual(options.sampler_default('topP', 'topK', {}), (64, False))
    # The other samplers keep what they had: Top-K needs a top_k.
    self.assertIsNone(options.sampler_default('topK', 'topK', {}))

  def test_top_k_top_p_takes_the_users_top_k(self):
    """A top_k the user sets is kept and sent as typed, whatever it is."""
    for top_k in (4, 262144, 2**31 - 1):
      generation = options.validate_generation(
          {
              'ref': {'samplerType': 'topP', 'topK': top_k, 'topP': 0.5},
              'target': {},
          },
          model_sampling={'ref': MODEL_SAMPLING},
      )
      self.assertEqual(generation['ref']['topK'], top_k)
      self.assertEqual(
          options.create_chat_request(
              generation['ref'],
              INFO,
              CONTEXT,
          ).options.top_k,
          top_k,
      )

  def test_create_chat_request_refuses_what_the_runner_lacks(self):
    """Create chat request refuses what the runner lacks."""
    info = runner_pb2.RunnerInfo()
    info.CopyFrom(INFO)
    info.runtime.chat_options_schema = (
        '{"type": "object", "additionalProperties": false}'
    )
    with self.assertRaisesRegex(ValueError, 'Chat option thinkingBudget'):
      options.create_chat_request({'thinkingBudget': 5}, info, CONTEXT)
    self.assertEqual(
        options.create_chat_request({}, info, CONTEXT).options.top_k, 1
    )

  def test_validate_generation_records_what_runs(self):
    """What is set, and every sampling value."""
    self.assertEqual(
        options.validate_generation({
            'ref': {
                'samplerType': 'topP',
                'topP': 0.9,
                'thinking': 'default',
                'systemPrompt': '',
                'seed': 3.0,
                'maxOutputTokens': 1000,
            },
            'target': {'samplerType': None, 'thinkingBudget': -1},
        }),
        {
            'ref': {
                'samplerType': 'topP',
                'topK': 64,
                'topP': 0.9,
                'temperature': 1.0,
                'seed': 3,
                'maxOutputTokens': 1000,
            },
            'target': {
                'samplerType': 'greedy',
                'topK': 1,
                'topP': 1.0,
                'temperature': 1.0,
                'seed': 0,
                'thinkingBudget': -1,
            },
        },
    )

  def test_validate_generation_takes_back_what_it_recorded(self):
    """A recorded setting, sent again, is the same setting."""
    recorded = options.validate_generation(
        {'ref': {'samplerType': 'topK', 'topK': 40}, 'target': {}}
    )
    self.assertEqual(options.validate_generation(recorded), recorded)

  def test_a_reply_is_limited_by_the_context_only(self):
    """No 256 cap: a reply may be as long as its side's context."""
    runs = [
        {'id': 'ref', 'contextLength': '4096'},
        {'id': 'target', 'contextLength': ''},
    ]
    long = {'maxOutputTokens': 4096}
    self.assertEqual(
        options.validate_generation({'ref': long, 'target': long}, runs)['ref'][
            'maxOutputTokens'
        ],
        4096,
    )
    with self.assertRaisesRegex(
        ValueError,
        'Max output tokens must be at most the context length, 2048',
    ):
      options.validate_generation(
          {'ref': {}, 'target': {'maxOutputTokens': 2049}},
          [{'id': 'target', 'contextLength': '2048'}],
      )

  @parameterized.parameters(
      ({'unknown': 1}, 'Unsupported generation settings: unknown'),
      (
          {'maxOutputTokens': 0},
          'Max output tokens must be between 1 and 2147483647',
      ),
      ({'maxOutputTokens': 1.5}, 'Max output tokens'),
      ({'seed': 2**31}, 'Seed must be between 0 and 2147483647'),
      ({'topK': True, 'samplerType': 'topK'}, 'Top K'),
      ({'temperature': float('nan')}, 'Temperature'),
      ({'temperature': '0.5'}, 'Temperature'),
      ({'thinking': 'yes'}, 'thinking must be on, off'),
      ({'samplerType': 'beam'}, 'sampler must be one of'),
      ({'systemPrompt': 'x' * 16001}, 'system prompt'),
      ({'temperature': 0.5}, 'Greedy sampler does not take Temperature'),
      ({'topK': 4}, 'Greedy sampler does not take Top K'),
      (
          {'samplerType': 'topK', 'topK': 4, 'topP': 0.5},
          'The Top-K sampler does not take Top P',
      ),
      ({'samplerType': 'topP'}, r'The Top-K \+ Top-P sampler needs Top P'),
      ({'samplerType': 'topK'}, 'The Top-K sampler needs Top K'),
      ({'speculativeDecoding': 'on'}, 'Speculative decoding cannot be'),
  )
  def test_validate_generation_refuses(self, side, message):
    """Validate generation refuses."""
    with self.assertRaisesRegex(ValueError, message):
      options.validate_generation({'ref': {}, 'target': side})

  def test_validate_generation_needs_both_sides(self):
    """Validate generation needs both sides."""
    with self.assertRaisesRegex(ValueError, 'Reference and Target'):
      options.validate_generation({'ref': {}})
    with self.assertRaisesRegex(ValueError, 'not valid'):
      options.validate_generation({'ref': {}, 'target': []})

  def test_check_run_option(self):
    """Check run option."""
    options.check_run_option('enableYnnpack', 'on')
    options.check_run_option('cpuThreads', 'anything')
    options.check_run_option('speculativeDecoding', 'off')
    with self.assertRaisesRegex(ValueError, 'Speculative decoding cannot'):
      options.check_run_option('speculativeDecoding', 'on')
    with self.assertRaisesRegex(ValueError, 'enableYnnpack must be on, off'):
      options.check_run_option('enableYnnpack', 'true')

  def test_chat_options_are_what_the_litert_lm_runner_takes(self):
    """Every runtime chat option, the LiteRT-LM runner's schema takes."""
    options.create_chat_request(
        {
            'thinking': 'on',
            'applyPromptTemplate': 'off',
            'speculativeDecoding': 'off',
            'thinkingBudget': 8,
        },
        INFO,
        CONTEXT,
    )


if __name__ == '__main__':
  test_main.main()
