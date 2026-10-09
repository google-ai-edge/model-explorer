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
# ==============================================================================

"""Worker request behavior, independent of optional PyTorch installation."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from model_debugger_runner import pytorch_worker
from runtime_fixture import configured_runtime_root


class PyTorchWorkerTest(unittest.TestCase):

  def test_cancel_is_checked_before_any_runtime_or_model_load(self):
    with tempfile.TemporaryDirectory() as name:
      root = Path(name)
      (root / 'cancel').touch()
      with self.assertRaisesRegex(InterruptedError, 'Cancelled'):
        pytorch_worker.execute(
            {'output': str(root / 'ref')}, lambda *args, **kwargs: None
        )

  def test_explicit_sampler_and_default_greedy(self):
    self.assertFalse(pytorch_worker._settings({})['do_sample'])
    result = pytorch_worker._settings({
        'max_output_tokens': 8,
        'generation': dict(
            temperature=0.7, topK=12, topP=0.8, seed=42, maxOutputTokens=4
        ),
    })
    self.assertEqual(
        result,
        dict(
            temperature=0.7,
            top_k=12,
            top_p=0.8,
            seed=42,
            max_output_tokens=4,
            do_sample=True,
        ),
    )

  def test_unsupported_or_invalid_generation_settings_fail(self):
    for config in (
        {'thinking': 'off'},
        {'thinking': 'on'},
        {'thinkingBudget': 0},
        {'stop': ['END']},
        {'temperature': float('nan')},
        {'topK': 0},
        {'seed': True},
        {'maxOutputTokens': 257},
        {'topP': 1.01},
    ):
      with self.subTest(config=config), self.assertRaises(ValueError):
        pytorch_worker._settings({'generation': config})

  def test_history_and_system_prompt_use_actual_text(self):
    result = pytorch_worker._messages(
        dict(
            prompt='Next',
            generation={'systemPrompt': 'Be brief'},
            messages=[
                {
                    'role': 'user',
                    'content': [{'type': 'text', 'text': 'First'}],
                },
                {'role': 'assistant', 'content': 'Reply'},
            ],
        )
    )
    self.assertEqual(
        result,
        [
            dict(role='system', content='Be brief'),
            dict(role='user', content='First'),
            dict(role='assistant', content='Reply'),
            dict(role='user', content='Next'),
        ],
    )

  def test_nontext_history_and_empty_prompt_fail(self):
    for request in (
        dict(prompt=''),
        dict(prompt=' ', messages=[]),
        dict(
            prompt='Next',
            messages=[
                dict(role='user', content=[{'type': 'image', 'data': 'x'}])
            ],
        ),
    ):
      with self.subTest(request=request), self.assertRaises(ValueError):
        pytorch_worker._messages(request)

  def test_local_chat_template_is_applied_without_duplicate_special_tokens(
      self,
  ):
    calls = []

    class Tokenizer:
      chat_template = 'real template'

      def get_chat_template(self):
        return self.chat_template

      def apply_chat_template(self, messages, **kwargs):
        calls.append(('template', messages, kwargs))
        return '<user>hello<assistant>'

      def __call__(self, text, **kwargs):
        calls.append(('tokenize', text, kwargs))
        return {'input_ids': 'actual tokenizer result'}

    inputs, text, method = pytorch_worker._tokenize(
        Tokenizer(), [dict(role='user', content='hello')]
    )
    self.assertEqual(inputs['input_ids'], 'actual tokenizer result')
    self.assertEqual(text, '<user>hello<assistant>')
    self.assertEqual(method['kind'], 'tokenizer_chat_template')
    self.assertEqual(
        calls[0][2], dict(tokenize=False, add_generation_prompt=True)
    )
    self.assertFalse(calls[1][2]['add_special_tokens'])
    self.assertFalse(calls[1][2]['truncation'])

  def test_plain_fallback_is_defined_and_never_truncates(self):
    class Tokenizer:
      chat_template = None

      def __call__(self, text, **kwargs):
        self.assertions = (text, kwargs)
        return {'input_ids': [1]}

    tokenizer = Tokenizer()
    _, text, method = pytorch_worker._tokenize(
        tokenizer,
        [dict(role='system', content='Brief'), dict(role='user', content='Hi')],
    )
    self.assertEqual(text, 'system: Brief\nuser: Hi\nassistant: ')
    self.assertEqual(method['kind'], 'plain_role_lines_v1')
    self.assertFalse(tokenizer.assertions[1]['truncation'])

  def test_context_limit_respects_model_tokenizer_and_request(self):
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=128))
    tokenizer = SimpleNamespace(model_max_length=256)
    self.assertEqual(pytorch_worker._context_limit(model, tokenizer, {}), 128)
    self.assertEqual(
        pytorch_worker._context_limit(model, tokenizer, {'contextLength': 64}),
        64,
    )
    self.assertEqual(
        pytorch_worker._context_limit(model, tokenizer, {'contextLength': 512}),
        128,
    )
    with self.assertRaises(ValueError):
      pytorch_worker._context_limit(model, tokenizer, {'contextLength': True})
    with self.assertRaisesRegex(ValueError, 'unknown'):
      pytorch_worker._context_limit(
          SimpleNamespace(config=SimpleNamespace()), SimpleNamespace(), {}
      )


@unittest.skipUnless(
    os.environ.get('RUNNER_TEST_RUNTIME_ROOT')
    and importlib.util.find_spec('ai_edge_debugger_pytorch'),
    'Requires RUNNER_TEST_RUNTIME_ROOT and the configured PyTorch capture'
    ' environment',
)
class PyTorchWorkerCaptureTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.capture_root = str(configured_runtime_root())
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig
    from transformers import LlamaForCausalLM
    from transformers import PreTrainedTokenizerFast
    import ai_edge_debugger_pytorch
    from model_debugger_contracts.model_identity import describe_model

    cls.temp = tempfile.TemporaryDirectory()
    cls.root = Path(cls.temp.name)
    cls.model_path = cls.root / 'model'
    vocab = {
        '<pad>': 0,
        '<unk>': 1,
        'user': 2,
        'assistant': 3,
        ':': 4,
        'hello': 5,
        'world': 6,
        '!': 7,
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token='<unk>'))
    backend.pre_tokenizer = Whitespace()
    cls.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token='<unk>',
        pad_token='<pad>',
        model_max_length=32,
        model_input_names=['input_ids', 'attention_mask'],
    )
    cls.tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }} {% endfor"
        ' %}assistant: '
    )
    torch.manual_seed(123)
    cls.model = (
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=8,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=32,
                eos_token_id=None,
                pad_token_id=0,
            )
        )
        .eval()
        .to(torch.bfloat16)
    )
    cls.model.save_pretrained(cls.model_path, safe_serialization=True)
    cls.tokenizer.save_pretrained(cls.model_path)
    cls.files, cls.checksum = describe_model(cls.model_path)

  @classmethod
  def tearDownClass(cls):
    cls.temp.cleanup()

  def request(self, name, **changes):
    directory = self.root / name
    directory.mkdir()
    request = dict(
        output=str(directory),
        model=str(self.model_path),
        model_files=self.files,
        model_sha256=self.checksum,
        pytorch_root=self.capture_root,
        run=dict(backend='CPU', precision='default'),
        operation='generate',
        messages=[],
        prompt='hello world',
        max_output_tokens=3,
    )
    request.update(changes)
    return request

  def raw_generation(self, request):
    root = Path(request['output']) / 'raw/prefill'
    generation = json.loads((root / 'generation.json').read_text())
    forwards = json.loads((root / 'forward_index.json').read_text())['forwards']
    tokens = [
        json.loads(line)
        for line in (root / 'tokens.jsonl').read_text().splitlines()
        if line
    ]
    return root, generation, forwards, tokens

  def test_real_prefill_bf16_and_cached_generation_match_hf(self):
    import torch
    from safetensors import safe_open

    request = self.request('capture', turn=1)
    events = []
    result = pytorch_worker.execute(
        request, lambda kind, **data: events.append(dict(type=kind, **data))
    )
    encoded, _, _ = pytorch_worker._tokenize(
        self.tokenizer, pytorch_worker._messages(request)
    )
    with torch.no_grad():
      expected = self.model.generate(
          **encoded, do_sample=False, max_new_tokens=3, pad_token_id=0
      )
    self.assertEqual(
        [t['id'] for t in result['tokens']], expected[0, -3:].tolist()
    )
    self.assertEqual(
        result['output'],
        ''.join(e['text'] for e in events if e['type'] == 'delta'),
    )
    self.assertEqual(result['effective']['precision'], 'bfloat16')
    self.assertTrue(result['decode_capture'])
    self.assertTrue(result['kv_capture'])
    self.assertEqual(result['published_scope'], 'all_observed_forwards')
    index = json.loads(
        (Path(request['output']) / 'export/capture_index.json').read_text()
    )
    expected_modules = {
        'model.layers.0' + suffix
        for suffix in (
            '',
            '.self_attn',
            '.mlp',
            '.input_layernorm',
            '.post_attention_layernorm',
        )
    }
    self.assertEqual(
        {
            (record['module_path'], record['edge'])
            for record in index['tensors']
        },
        {
            (module, edge)
            for module in expected_modules
            for edge in ('in', 'out')
        },
    )
    self.assertEqual(len(index['tensors']), 30)
    self.assertEqual(
        len([row for row in index['tensors'] if row['forward_id'] == 0]), 10
    )
    self.assertEqual(index['format_version'], 2)
    self.assertEqual(
        {record['forward_id'] for record in index['tensors']}, {0, 1, 2}
    )
    self.assertEqual({record['turn'] for record in index['tensors']}, {1})
    for module in expected_modules:
      for forward_id in range(3):
        self.assertEqual(
            len({
                record['module_call_id']
                for record in index['tensors']
                if record['module_path'] == module
                and record['forward_id'] == forward_id
            }),
            1,
        )
    self.assertEqual(
        {(t['phase'], t['step'], t['layer']) for t in index['tensors']},
        {('prefill', 0, 0), ('decode', 0, 0), ('decode', 1, 0)},
    )
    self.assertEqual({t['edge'] for t in index['tensors']}, {'in', 'out'})
    for record in index['tensors']:
      self.assertNotIn('boundaries', Path(record['path']).parts)
      with safe_open(
          Path(request['output']) / record['path'], framework='pt'
      ) as shard:
        tensor = shard.get_tensor(record['key'])
        self.assertEqual(tensor.dtype, torch.bfloat16)
        self.assertEqual(list(tensor.shape), record['shape'])
    with safe_open(
        Path(request['output']) / 'raw/inputs.safetensors', framework='pt'
    ) as shard:
      self.assertTrue(
          torch.equal(shard.get_tensor('input_ids'), encoded['input_ids'])
      )
    self.assertEqual(result['input_identity'], index['input_identity'])
    self.assertEqual(index['export_scope'], 'all')
    self.assertEqual(index['capture_scope'], 'generation')
    self.assertNotIn('selected_forward_id', index)
    self.assertEqual(
        [row['forward_id'] for row in index['forward_input_proofs']], [0, 1, 2]
    )
    self.assertEqual(
        {row['scope'] for row in index['resources']}, {'boundary', 'kv'}
    )
    raw, generation, forwards, token_records = self.raw_generation(request)
    self.assertEqual(
        [(row['forward_id'], row['phase'], row['step']) for row in forwards],
        [(0, 'prefill', 0), (1, 'decode', 0), (2, 'decode', 1)],
    )
    self.assertEqual({row['turn'] for row in forwards}, {1})
    self.assertEqual(
        [row['pos_offset'] for row in forwards],
        [0, encoded['input_ids'].shape[1], encoded['input_ids'].shape[1] + 1],
    )
    module_rows = [
        json.loads(line)
        for line in (raw / 'manifest.jsonl').read_text().splitlines()
        if line
    ]
    self.assertEqual(len(module_rows), 30)
    self.assertEqual({row['forward_id'] for row in module_rows}, {0, 1, 2})
    self.assertEqual([row['forward_id'] for row in token_records], [0, 1, 2])
    self.assertEqual(
        [row['token_ids'] for row in token_records],
        [[row['id']] for row in result['tokens']],
    )
    self.assertEqual(
        [row['texts'] for row in token_records],
        [[row['text']] for row in result['tokens']],
    )
    self.assertTrue(all(row['scores'] is None for row in token_records))
    self.assertEqual(
        [
            row['consumption'][0]['consumed_by_forward_id']
            for row in token_records
        ],
        [1, 2, None],
    )
    self.assertEqual(
        [row['consumption'][0]['position'] for row in token_records],
        [
            encoded['input_ids'].shape[1],
            encoded['input_ids'].shape[1] + 1,
            None,
        ],
    )
    self.assertEqual(generation['status'], 'completed')
    self.assertEqual(generation['stop_reason'], 'max_output_tokens')
    self.assertEqual(generation['generated_token_count'], 3)
    self.assertEqual(
        generation['processed_token_count'], encoded['input_ids'].shape[1] + 2
    )
    self.assertEqual(
        generation['pending_token_ids'], [result['tokens'][-1]['id']]
    )
    self.assertEqual(result['raw_capture']['forward_count'], 3)
    self.assertEqual(result['raw_capture']['scope'], 'whole_generation')
    self.assertEqual(result['raw_capture']['token_record_count'], 3)
    self.assertEqual(result['raw_capture']['kv_snapshot_count'], 3)
    for key in ('forward_index', 'kv_index', 'tokens', 'generation'):
      self.assertTrue(
          (Path(request['output']) / result['raw_capture'][key]).is_file()
      )
    snapshots = json.loads((raw / 'kv_index.json').read_text())['snapshots']
    self.assertEqual(
        [row['moment'] for row in snapshots],
        ['prefill_pre', 'prefill_post', 'terminal'],
    )
    self.assertEqual(
        [row['state'] for row in snapshots],
        ['not_allocated', 'available', 'available'],
    )
    self.assertEqual(
        [row['state'] for row in result['raw_capture']['kv_snapshots']],
        [row['state'] for row in snapshots],
    )
    self.assertEqual([row['forward_id'] for row in snapshots], [0, 0, 2])
    self.assertEqual(
        snapshots[1]['processed_token_count'], encoded['input_ids'].shape[1]
    )
    self.assertEqual(
        snapshots[2]['processed_token_count'], encoded['input_ids'].shape[1] + 2
    )
    kv_rows = {
        row['key']: row
        for row in (
            json.loads(line)
            for line in (raw / 'kv/manifest.jsonl').read_text().splitlines()
            if line
        )
    }
    self.assertEqual(len(kv_rows), 8)
    for snapshot in snapshots:
      self.assertTrue(snapshot['storage_complete'])
      for layer in snapshot['layers']:
        for resource in layer['tensors']:
          self.assertEqual(resource['storage_status'], 'stored')
          row = kv_rows[resource['key']]
          with safe_open(raw / 'kv' / row['shard'], framework='pt') as shard:
            tensor = shard.get_tensor(row['slot'])
            self.assertEqual(tensor.dtype, torch.bfloat16)
            self.assertEqual(list(tensor.shape), resource['shape'])
            self.assertEqual(tensor.shape[-2], layer['valid_length'])

  def test_output_limit_one_leaves_selected_token_out_of_the_actual_cache(self):
    request = self.request('one-token', max_output_tokens=1)
    result = pytorch_worker.execute(request, lambda *args, **kwargs: None)
    _, generation, forwards, token_records = self.raw_generation(request)
    encoded, _, _ = pytorch_worker._tokenize(
        self.tokenizer, pytorch_worker._messages(request)
    )
    self.assertEqual(len(forwards), 1)
    self.assertEqual(len(token_records), 1)
    self.assertEqual(result['token_count'], 1)
    self.assertEqual(generation['stop_reason'], 'max_output_tokens')
    self.assertEqual(
        generation['processed_token_count'], encoded['input_ids'].shape[1]
    )
    self.assertEqual(
        generation['pending_token_ids'], [result['tokens'][0]['id']]
    )
    self.assertEqual(result['captured_tensors'], 10)

  def test_eos_stops_without_an_extra_forward(self):
    import torch
    from model_debugger_contracts.model_identity import describe_model

    request = self.request('eos')
    encoded, _, _ = pytorch_worker._tokenize(
        self.tokenizer, pytorch_worker._messages(request)
    )
    with torch.no_grad():
      first = int(
          self.model.generate(
              **encoded, do_sample=False, max_new_tokens=1, pad_token_id=0
          )[0, -1]
      )
    eos_model = self.root / 'eos-model'
    previous = self.model.generation_config.eos_token_id
    try:
      self.model.generation_config.eos_token_id = first
      self.model.save_pretrained(eos_model, safe_serialization=True)
    finally:
      self.model.generation_config.eos_token_id = previous
    self.tokenizer.save_pretrained(eos_model)
    request['model'] = str(eos_model)
    request['model_files'], request['model_sha256'] = describe_model(eos_model)
    result = pytorch_worker.execute(request, lambda *args, **kwargs: None)
    _, generation, forwards, token_records = self.raw_generation(request)
    self.assertEqual([token['id'] for token in result['tokens']], [first])
    self.assertEqual(result['stop_reason'], 'eos')
    self.assertEqual(generation['status'], 'completed')
    self.assertEqual(generation['stop_reason'], 'eos')
    self.assertEqual(
        generation['processed_token_count'], encoded['input_ids'].shape[1]
    )
    self.assertEqual(generation['pending_token_ids'], [first])
    self.assertEqual(len(forwards), 1)
    self.assertEqual(len(token_records), 1)

  def test_export_preserves_all_prefill_and_decode_evidence(self):
    import torch
    import ai_edge_debugger_pytorch
    from model_debugger_runner.pytorch_capture import export_capture

    request = self.request('complete-export')
    encoded, _, _ = pytorch_worker._tokenize(
        self.tokenizer, pytorch_worker._messages(request)
    )
    with ai_edge_debugger_pytorch.CaptureRun(
        self.model,
        granularity='sublayer',
        layers='0',
        out_dir=Path(request['output']) / 'raw/prefill',
    ) as capture:
      with capture.forward_context(phase='prefill', step=0, pos_offset=0):
        output = self.model(**encoded, use_cache=True)
      token = int(output.logits[0, -1].argmax())
      capture.observe_tokens([token], forward_id=0)
      prompt_length = encoded['input_ids'].shape[1]
      prepared = self.model.prepare_inputs_for_generation(
          torch.tensor([[token]]),
          past_key_values=output.past_key_values,
          attention_mask=torch.ones((1, prompt_length + 1), dtype=torch.long),
          cache_position=torch.tensor([prompt_length]),
          use_cache=True,
      )
      with capture.forward_context(
          phase='decode', step=0, pos_offset=prompt_length
      ):
        output = self.model(**prepared)
      capture.observe_tokens([int(output.logits[0, -1].argmax())], forward_id=1)
      capture.finish_generation('max_output_tokens')
    index = export_capture(capture, request['output'], 'test-input-identity')
    self.assertEqual(len(index['tensors']), 20)
    self.assertEqual(
        {(row['forward_id'], row['phase']) for row in index['tensors']},
        {(0, 'prefill'), (1, 'decode')},
    )
    self.assertEqual(index['forwards'], capture.forwards)
    self.assertEqual(index['kv_snapshots'], capture.kv_snapshots)
    self.assertEqual(index['token_records'], capture.token_records)
    self.assertEqual(index['generation'], capture.generation)
    self.assertEqual(
        index['generation']['processed_token_count'], prompt_length + 1
    )
    self.assertEqual(
        {row['scope'] for row in index['resources']}, {'boundary', 'kv'}
    )
    saved = json.loads(
        (Path(request['output']) / 'export/capture_index.json').read_text()
    )
    self.assertEqual(saved, index)

  def test_context_overflow_rejects_without_truncating_or_capture(self):
    request = self.request('overflow')
    request['run']['contextLength'] = 4
    with self.assertRaisesRegex(ValueError, 'not truncated'):
      pytorch_worker.execute(request, lambda *args, **kwargs: None)
    self.assertFalse((Path(request['output']) / 'export').exists())

  def test_initialize_does_not_invent_tensor_capture(self):
    request = self.request('initialize')
    request['operation'] = 'initialize'
    result = pytorch_worker.execute(request, lambda *args, **kwargs: None)
    self.assertEqual(result['captured_tensors'], 0)
    self.assertIsNone(result['raw_capture'])
    self.assertFalse(result['kv_capture'])
    self.assertFalse((Path(request['output']) / 'raw').exists())

  def test_cancellation_after_prefill_restores_capture_hooks(self):
    import ai_edge_debugger_pytorch

    original = ai_edge_debugger_pytorch.CaptureRun
    restored = []

    class TrackedCapture(original):

      def __enter__(self):
        self.before = [
            (len(module._forward_pre_hooks), len(module._forward_hooks))
            for module in self.model.modules()
        ]
        return super().__enter__()

      def __exit__(self, *args):
        try:
          return super().__exit__(*args)
        finally:
          restored.append(
              self.before
              == [
                  (len(module._forward_pre_hooks), len(module._forward_hooks))
                  for module in self.model.modules()
              ]
          )

    request = self.request('cancel-after-prefill')
    marker = Path(request['output']).parent / 'cancel'

    def emit(kind, **data):
      if kind == 'delta':
        marker.touch()

    try:
      with patch.object(ai_edge_debugger_pytorch, 'CaptureRun', TrackedCapture):
        with self.assertRaisesRegex(InterruptedError, 'Cancelled'):
          pytorch_worker.execute(request, emit)
      self.assertEqual(restored, [True, True])
      raw, generation, _, _ = self.raw_generation(request)
      self.assertEqual(generation['status'], 'cancelled')
      self.assertEqual(generation['stop_reason'], 'cancelled')
      self.assertNotIn(
          'terminal',
          [
              row['moment']
              for row in json.loads((raw / 'kv_index.json').read_text())[
                  'snapshots'
              ]
          ],
      )
      self.assertFalse((Path(request['output']) / 'export').exists())
    finally:
      marker.unlink(missing_ok=True)

  def test_sampling_error_records_failure_without_publishing_capture(self):
    request = self.request('sampling-error')
    with patch(
        'model_debugger_runner.pytorch_worker._choose_token',
        side_effect=ValueError('bad logits'),
    ):
      with self.assertRaisesRegex(ValueError, 'bad logits'):
        pytorch_worker.execute(request, lambda *args, **kwargs: None)
    raw, generation, forwards, token_records = self.raw_generation(request)
    self.assertEqual(generation['status'], 'failed')
    self.assertEqual(generation['stop_reason'], 'error')
    self.assertEqual(generation['generated_token_count'], 0)
    self.assertEqual(len(forwards), 1)
    self.assertEqual(token_records, [])
    self.assertNotIn(
        'terminal',
        [
            row['moment']
            for row in json.loads((raw / 'kv_index.json').read_text())[
                'snapshots'
            ]
        ],
    )
    self.assertFalse((Path(request['output']) / 'export').exists())


if __name__ == '__main__':
  unittest.main()
