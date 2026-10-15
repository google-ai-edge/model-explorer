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

from google.protobuf import json_format
from model_debugger.v1 import runner_pb2
from model_debugger_server import models, options
import pytest

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
    protocol_version='1.0.0',
    runtime=runner_pb2.RuntimeInfo(
        name=runner_pb2.RuntimeInfo.LITERT_LM,
        version='abc',
        load_options_schema=LOAD_SCHEMA,
        chat_options_schema=CHAT_SCHEMA,
    ),
)
MODEL = models.Model(
    artifact='model-aa',
    sha256='aa' * 32,
    name='m.litertlm',
    size=1,
    path=pathlib.Path('/m.litertlm'),
)


def _json(message):
  return json_format.MessageToDict(message, preserving_proto_field_name=True)


def test_load_request_converts_ui_text():
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
  assert _json(request) == {
      'sha256': 'aa' * 32,
      'device': 'GPU',
      'context_length': 2048,
      'runtime_options': {
          'cpuThreads': 4,
          'enableYnnpack': False,
          'singleThreadedExecution': True,
          'activationDataType': 'float16',
      },
  }


def test_load_request_defaults():
  """Load request defaults."""
  request = options.load_request(
      {'backend': 'CPU', 'contextLength': ''}, MODEL, INFO
  )
  assert request.device == runner_pb2.CPU
  assert request.context_length == options.DEFAULT_CONTEXT_LENGTH
  assert not request.runtime_options.fields


@pytest.mark.parametrize(
    ('run', 'message'),
    [
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
        ({'contextLength': '0'}, 'context length'),
        ({'contextLength': '1e3'}, 'context length'),
        ({'backend': 'NPU'}, 'CPU or GPU'),
    ],
)
def test_load_request_refuses(run, message):
  """Load request refuses."""
  with pytest.raises(ValueError, match=message):
    options.load_request({'backend': 'CPU', **run}, MODEL, INFO)


def test_create_chat_request_default_is_greedy():
  """Create chat request default is greedy."""
  request = options.create_chat_request({}, INFO)
  assert _json(request) == {
      'options': {
          'max_completion_tokens': options.DEFAULT_MAX_OUTPUT_TOKENS,
          'top_k': 1,
      }
  }


def test_create_chat_request_maps_every_setting():
  """Create chat request maps every setting."""
  request = options.create_chat_request(
      {
          'systemPrompt': 'Be brief.',
          'maxOutputTokens': 64,
          'samplerType': 'topP',
          'topP': 0.5,
          'temperature': 0.75,
          'seed': 7,
          'thinking': 'on',
          'thinkingBudget': 128,
          'applyPromptTemplate': 'off',
          'speculativeDecoding': 'off',
      },
      INFO,
  )
  assert _json(request) == {
      'messages': [{'role': 'SYSTEM', 'content': 'Be brief.'}],
      'options': {
          'max_completion_tokens': 64,
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
  }


def test_create_chat_request_top_k():
  """Create chat request top k."""
  request = options.create_chat_request(
      {'samplerType': 'topK', 'topK': 40}, INFO
  )
  assert request.options.top_k == 40
  assert not request.options.HasField('top_p')
  assert not request.options.HasField('temperature')


def test_create_chat_request_refuses_what_the_runner_lacks():
  """Create chat request refuses what the runner lacks."""
  info = runner_pb2.RunnerInfo()
  info.CopyFrom(INFO)
  info.runtime.chat_options_schema = (
      '{"type": "object", "additionalProperties": false}'
  )
  with pytest.raises(ValueError, match='Chat option thinkingBudget'):
    options.create_chat_request({'thinkingBudget': 5}, info)
  assert options.create_chat_request({}, info).options.top_k == 1


def test_validate_generation_keeps_what_is_set():
  """Validate generation keeps what is set."""
  assert options.validate_generation({
      'ref': {
          'samplerType': None,
          'thinking': 'default',
          'systemPrompt': '',
          'seed': 3.0,
          'maxOutputTokens': 256,
      },
      'target': {'samplerType': 'greedy', 'thinkingBudget': -1},
  }) == {
      'ref': {'seed': 3, 'maxOutputTokens': 256},
      'target': {'samplerType': 'greedy', 'thinkingBudget': -1},
  }


@pytest.mark.parametrize(
    ('side', 'message'),
    [
        ({'unknown': 1}, 'Unsupported generation settings: unknown'),
        (
            {'maxOutputTokens': 257},
            'Max output tokens must be between 1 and 256',
        ),
        ({'maxOutputTokens': 1.5}, 'Max output tokens'),
        ({'seed': 2**31}, 'Seed must be between 0 and 2147483647'),
        ({'topK': True, 'samplerType': 'topK'}, 'Top K'),
        ({'temperature': float('nan')}, 'Temperature'),
        ({'temperature': '0.5'}, 'Temperature'),
        ({'thinking': 'yes'}, 'thinking must be on, off'),
        ({'samplerType': 'beam'}, 'sampler must be one of'),
        ({'systemPrompt': 'x' * 16001}, 'system prompt'),
        ({'temperature': 0.5}, 'greedy sampler does not take Temperature'),
        ({'topK': 4}, 'greedy sampler does not take Top K'),
        ({'samplerType': 'topK', 'topK': 4, 'topP': 0.5}, 'topK sampler does'),
        ({'samplerType': 'topP'}, 'topP sampler needs Top P'),
    ],
)
def test_validate_generation_refuses(side, message):
  """Validate generation refuses."""
  with pytest.raises(ValueError, match=message):
    options.validate_generation({'ref': {}, 'target': side})


def test_validate_generation_needs_both_sides():
  """Validate generation needs both sides."""
  with pytest.raises(ValueError, match='Reference and Target'):
    options.validate_generation({'ref': {}})
  with pytest.raises(ValueError, match='not valid'):
    options.validate_generation({'ref': {}, 'target': []})


def test_check_run_option():
  """Check run option."""
  options.check_run_option('enableYnnpack', 'on')
  options.check_run_option('cpuThreads', 'anything')
  with pytest.raises(ValueError, match='enableYnnpack must be on, off'):
    options.check_run_option('enableYnnpack', 'true')


def test_option_lists_follow_the_schemas():
  """Option lists follow the schemas."""
  assert options.supported_options(INFO) == [
      'backend',
      'contextLength',
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
  ]
  assert options.chat_option_keys(INFO) == [
      'maxOutputTokens',
      'systemPrompt',
      'samplerType',
      'temperature',
      'topK',
      'topP',
      'seed',
      'thinking',
      'applyPromptTemplate',
      'speculativeDecoding',
      'thinkingBudget',
  ]
  assert options.supported_options(runner_pb2.RunnerInfo()) == [
      'backend',
      'contextLength',
  ]
