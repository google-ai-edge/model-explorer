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

"""Tests of the options the UI is offered."""

import json
from typing import Any

from absl.testing import absltest
from model_debugger.protocol import runner_pb2
from model_debugger.server.service import option_catalog
from model_debugger.server.session import options
from model_debugger.server.testing import fake_runner
from model_debugger.server.testing import test_main


def _by_key(descriptions: list[dict[str, Any]]) -> dict[str, Any]:
  return {description['key']: description for description in descriptions}


def _info(**schemas: dict[str, Any]) -> runner_pb2.RunnerInfo:
  """The fake runner's info, with schemas replaced and no GPU."""
  info = runner_pb2.RunnerInfo()
  info.CopyFrom(fake_runner.DEFAULT_INFO)
  info.host.ClearField('gpu')
  for name, schema in schemas.items():
    setattr(info.runtime, name, json.dumps(schema))
  return info


class OptionCatalogTest(absltest.TestCase):
  """Tests of the options the UI is offered."""

  def test_engine_options_are_what_the_litert_lm_runner_takes(self):
    """The table's engine options are those the runner's schema names."""
    load = json.loads(fake_runner.LOAD_OPTIONS_SCHEMA)['properties']
    self.assertEqual(
        option_catalog.ENGINE_OPTION_KEYS,
        tuple(key for key in options.RUNTIME_OPTION_KEYS if key in load),
    )

  def test_every_option_a_run_or_chat_saves_is_described(self):
    """The table describes each key options.py keeps, and no other."""
    tap = [o['key'] for o in option_catalog.pytorch_tap(None)['options']]
    self.assertCountEqual(
        [o['key'] for o in option_catalog.session_options(None)]
        + ['dtype', *tap],
        ['backend', 'contextLength', *options.RUNTIME_OPTION_KEYS],
    )
    self.assertCountEqual(
        [o['key'] for o in option_catalog.pytorch_session_options(None)] + tap,
        ['backend', 'contextLength', *options.PYTORCH_OPTION_NAMES],
    )

  def test_pytorch_tensor_capture_is_its_runners_option(self):
    """The editor's tensor capture of PyTorch: the text and two options."""
    before = option_catalog.pytorch_tap(None)
    self.assertEqual(before['kind'], 'anchors')
    self.assertTrue(before['available'])
    # One line, the LiteRT-LM side's own; how, in its details.
    self.assertEqual(
        before['text'],
        "Captures every layer's 4 anchors in prefill and decode.",
    )
    self.assertIn("Every decoder layer's four anchors", before['details'])
    self.assertIn('module hooks', before['details'])
    observe, budget = before['options']
    self.assertEqual(observe['choices'], ['', 'anchors'])
    self.assertEqual((budget['minimum'], budget['maximum']), (1, 1024))
    self.assertEqual(budget['requires'], {'key': 'observe', 'value': 'anchors'})
    self.assertTrue(
        option_catalog.pytorch_tap(fake_runner.PYTORCH_INFO)['available']
    )
    older = runner_pb2.RunnerInfo()
    older.CopyFrom(fake_runner.PYTORCH_INFO)
    schema = json.loads(older.runtime.load_options_schema)
    del schema['properties']['observe']
    older.runtime.load_options_schema = json.dumps(schema)
    old = option_catalog.pytorch_tap(older)
    self.assertFalse(old['available'])
    self.assertIn('does not take it', old['reason'])
    self.assertCountEqual(
        [o['key'] for o in option_catalog.chat_options(None)],
        [
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
        ],
    )

  def test_pytorch_options_are_the_server_table_then_the_runner_schema(
      self,
  ):
    """PyTorch's options: this server's table, then its runner's."""
    before = _by_key(option_catalog.pytorch_session_options(None))
    self.assertEqual(before['dtype']['choices'], ['', *options.PYTORCH_DTYPES])
    self.assertTrue(before['cpuThreads']['available'])
    chat = _by_key(
        option_catalog.chat_options(None, runtime=option_catalog.PYTORCH)
    )
    self.assertFalse(chat['thinking']['available'])
    self.assertEqual(
        chat['thinking']['reason'], 'The PyTorch runner does not take it.'
    )
    self.assertTrue(chat['topK']['available'])

    reported = _by_key(
        option_catalog.pytorch_session_options(fake_runner.PYTORCH_INFO)
    )
    self.assertEqual(
        reported['dtype']['choices'], ['', *options.PYTORCH_DTYPES]
    )
    self.assertTrue(reported['cpuThreads']['available'])

  def test_the_table_alone_before_a_runner_ran(self):
    """With no runner reported, the table decides what is offered."""
    session = _by_key(option_catalog.session_options(None))
    self.assertEqual(session['backend']['choices'], ['CPU', 'GPU'])
    self.assertEqual(session['backend']['default'], 'CPU')
    self.assertTrue(session['cpuThreads']['available'])
    self.assertEqual(
        session['cpuThreads']['requires'],
        {'key': 'backend', 'value': 'CPU'},
    )
    self.assertEqual(
        session['audioCpuThreads']['requires'],
        {'key': 'audioBackend', 'value': 'CPU'},
    )
    for key in ('useRingbuffersLocalAttention', 'gpuDecodeStepsPerSync'):
      self.assertEqual(
          (session[key]['available'], session[key]['reason']),
          (False, "LiteRT-LM's runner does not take it."),
      )

  def test_context_lengths_start_at_2048_and_default_to_4096(self):
    """No context length below 2048 is offered; 4096 is the default."""
    context = _by_key(option_catalog.session_options(None))['contextLength']
    self.assertEqual(context['default'], '4096')
    self.assertEqual(context['minimum'], 2048)
    self.assertEqual(min(int(c) for c in context['choices']), 2048)
    self.assertIn(context['default'], context['choices'])

  def test_speculative_decoding_cannot_be_enabled(self):
    """Neither the engine's nor the chat's speculative decoding."""
    for descriptions in (
        option_catalog.session_options(fake_runner.DEFAULT_INFO),
        option_catalog.chat_options(fake_runner.DEFAULT_INFO),
    ):
      speculative = _by_key(descriptions)['speculativeDecoding']
      self.assertFalse(speculative['available'])
      self.assertEqual(speculative['reason'], options.SPECULATIVE_DECODING)

  def test_a_device_offers_what_its_last_runner_reported(self):
    """The runner's schema decides what is taken, and its ranges."""
    info = _info(
        load_options_schema={
            'type': 'object',
            'properties': {
                'cpuThreads': {'type': 'integer', 'minimum': 2},
                'activationDataType': {'enum': ['float32']},
            },
        },
        chat_options_schema={'type': 'object', 'properties': {}},
    )
    capability = option_catalog.runtime_options(info)
    # No GPU reported: CPU only.
    self.assertEqual(capability['backends'], ['CPU'])
    session = _by_key(capability['session_options'])
    self.assertEqual(session['backend']['choices'], ['CPU'])
    self.assertEqual(session['cpuThreads']['minimum'], 2)
    self.assertEqual(session['activationDataType']['choices'], ['', 'float32'])
    self.assertFalse(session['loraRank']['available'])
    self.assertIn(
        'LiteRT-LM a0f1e9736115, does not take it',
        session['loraRank']['reason'],
    )
    self.assertTrue(session['contextLength']['available'])
    chat = _by_key(capability['chat_options'])
    self.assertFalse(chat['thinking']['available'])
    self.assertTrue(chat['topP']['available'])

  def test_a_reply_is_limited_only_by_the_context(self):
    """Max output tokens: no cap; unless set, no limit but the context."""
    output = _by_key(option_catalog.chat_options(None))['maxOutputTokens']
    self.assertEqual(output['maximum'], options.INT32_MAX)
    self.assertIsNone(output.get('default'))

  def test_a_sides_run_bounds_its_reply_by_its_context(self):
    """Each side's description says its own longest reply."""
    for context, maximum in (('8192', 8192), ('', 4096)):
      run = {'id': 'ref', 'contextLength': context}
      described = _by_key(
          option_catalog.chat_options(None, options.chat_maximums(run))
      )
      self.assertEqual(described['maxOutputTokens']['maximum'], maximum)
      # Nothing else is bounded by the run.
      self.assertEqual(
          described['topK'],
          _by_key(option_catalog.chat_options(None))['topK'],
      )

  def test_sampling_values_say_who_uses_them_and_what_others_send(self):
    """Each sampling value's samplers, default and fixed values."""
    chat = _by_key(option_catalog.chat_options(None))
    self.assertEqual(chat['samplerType']['default'], 'greedy')
    self.assertEqual(
        chat['samplerType']['choices'], list(options.SAMPLER_TYPES)
    )
    self.assertEqual(
        chat['samplerType']['choice_labels'],
        {'greedy': 'Greedy', 'topK': 'Top-K', 'topP': 'Top-K + Top-P'},
    )
    self.assertEqual(
        {
            key: (chat[key]['samplers'], chat[key]['default'])
            for key in options.SAMPLING_KEYS
        },
        {
            'topK': (['topK', 'topP'], None),
            'topP': (['topP'], None),
            'temperature': (['topK', 'topP'], 1.0),
            'seed': (['topK', 'topP'], 0),
        },
    )
    self.assertEqual(chat['topK']['fixed'], {'greedy': '1 · fixed by Greedy'})
    self.assertEqual(
        chat['topP']['fixed'],
        {
            'greedy': '1.0 · fixed by Greedy',
            'topK': '1.0 · fixed by Top-K',
        },
    )
    # Only Top-K + Top-P's top_k has a default of its own.
    self.assertEqual(
        [key for key in options.SAMPLING_KEYS if 'defaults' in chat[key]],
        ['topK'],
    )

  def test_top_k_top_p_top_k_default_says_whence(self):
    """Top-K + Top-P's top_k default: the model's, else the table's."""

    def top_k_default(sampling):
      chat = _by_key(option_catalog.chat_options(None, None, sampling))
      return chat['topK']['defaults']

    self.assertEqual(
        top_k_default({'type': 'TOP_P', 'top_k': 48}),
        {
            'topP': {
                'value': 48,
                'source': "the model's own, saved with these settings",
            }
        },
    )
    self.assertEqual(
        top_k_default({}),
        {
            'topP': {
                'value': 64,
                'source': (
                    "the Model Debugger's own, as the model has none; "
                    'saved with these settings'
                ),
            }
        },
    )
    self.assertEqual(
        top_k_default(None),
        {
            'topP': {
                'value': 64,
                'source': (
                    "the Model Debugger's own, until the model's first "
                    "start says the model's; saved with these settings"
                ),
            }
        },
    )


if __name__ == '__main__':
  test_main.main()
