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

"""Resident HF model ownership, real cache-prefix reuse and recovery."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest import mock
import weakref

from model_debugger_runner import pytorch_capture
from model_debugger_runner import pytorch_worker
from model_debugger_runner.pytorch_session import PyTorchResidentRuntime
import test_pytorch_worker as worker_tests


@unittest.skipUnless(
    os.environ.get('RUNNER_TEST_RUNTIME_ROOT')
    and importlib.util.find_spec('ai_edge_debugger_pytorch'),
    'Requires RUNNER_TEST_RUNTIME_ROOT and the configured PyTorch capture'
    ' environment',
)
class PyTorchResidentTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    worker_tests.PyTorchWorkerCaptureTest.setUpClass.__func__(cls)

  @classmethod
  def tearDownClass(cls):
    cls.temp.cleanup()

  def setUp(self):
    self.runtime = PyTorchResidentRuntime()

  def tearDown(self):
    self.runtime.close()
    (self.root / 'cancel').unlink(missing_ok=True)

  def request(self, name, **changes):
    return worker_tests.PyTorchWorkerCaptureTest.request(
        self, self._testMethodName + '-' + name, **changes
    )

  def initialize(self, **changes):
    request = self.request('initialize', operation='initialize', **changes)
    result = self.runtime.execute(request, lambda *a, **k: None)
    return request, result

  def next_turn(self, name, **changes):
    return self.request(
        name,
        messages=deepcopy(self.runtime.messages),
        turn=self.runtime.turn + 1,
        **changes,
    )

  def assert_replay(self, request, result):
    import torch

    encoded, _, _ = pytorch_worker._tokenize(
        self.runtime.tokenizer, pytorch_worker._messages(request)
    )
    with torch.no_grad():
      expected = self.model.generate(
          **encoded,
          do_sample=False,
          max_new_tokens=request['max_output_tokens'],
          pad_token_id=0,
      )
    self.assertEqual(
        [item['id'] for item in result['tokens']],
        expected[0, encoded['input_ids'].shape[1] :].tolist(),
    )
    self.assertEqual(result['context_token_ids'], expected[0].tolist())

  def test_two_turns_reuse_actual_consumed_prefix_and_match_full_replay(self):
    _, initialized = self.initialize()
    loaded = self.runtime.model
    first_request = self.next_turn('one', max_output_tokens=1)
    first = self.runtime.execute(first_request, lambda *a, **k: None)
    self.assert_replay(first_request, first)
    self.assertEqual(first['pending_token_ids'], [first['tokens'][-1]['id']])
    second_request = self.next_turn('two')
    second = self.runtime.execute(second_request, lambda *a, **k: None)
    self.assert_replay(second_request, second)
    self.assertIs(loaded, self.runtime.model)
    self.assertEqual(
        {result['worker_pid'] for result in (initialized, first, second)},
        {os.getpid()},
    )
    self.assertEqual(
        {result['runtime_instance'] for result in (initialized, first, second)},
        {self.runtime.identity},
    )
    self.assertEqual(
        {result['model_load_count'] for result in (initialized, first, second)},
        {1},
    )
    self.assertEqual(second['turn_sequence'], 2)
    self.assertTrue(second['cache_reuse']['reused'])
    self.assertEqual(second['cache_reuse']['reason'], 'exact_consumed_prefix')
    self.assertEqual(
        second['cache_reuse']['reused_token_count'],
        first['processed_token_count'],
    )
    index = json.loads(
        (
            Path(second_request['output']) / 'export/capture_index.json'
        ).read_text()
    )
    first_forward = index['forwards'][0]
    self.assertEqual(
        first_forward['pos_offset'], first['processed_token_count']
    )
    self.assertEqual(
        first_forward['cache_before']['processed_token_count'],
        first['processed_token_count'],
    )
    proof = index['forward_input_proofs'][0]['input_proof']
    prefix = proof['cache_context']
    self.assertEqual(
        prefix['token_ids'],
        first['context_token_ids'][: first['processed_token_count']],
    )
    self.assertEqual(
        prefix['attention_mask'], [1] * first['processed_token_count']
    )
    self.assertEqual(
        proof['tensors']['input_ids']['shape'][1],
        second['cache_reuse']['input_token_count'],
    )
    for row in index['forward_input_proofs']:
      recorded = row['input_proof']
      self.assertEqual(recorded['scope'], 'same_logical_context')
      self.assertEqual(
          recorded['cache_context']['processed_token_count'],
          len(recorded['cache_context']['token_ids']),
      )
      start = recorded['cache_context']['processed_token_count']
      actual_ids = recorded['tensors']['input_ids']['values'][0]
      self.assertEqual(
          actual_ids,
          second['context_token_ids'][start : start + len(actual_ids)],
      )
      forward = index['forwards'][row['forward_id']]
      self.assertEqual(
          forward['cache_after']['processed_token_count'],
          start + len(actual_ids),
      )
      if row['forward_id'] > 0:
        self.assertEqual(len(actual_ids), 1)
    self.assertEqual(
        second['processed_token_count'], len(second['context_token_ids']) - 1
    )
    self.assertEqual(self.runtime.token_history, second['context_token_ids'])

  def test_wrong_actual_cache_length_fails_before_sampling_or_publication(self):
    self.initialize()
    request = self.next_turn('cache-mismatch', max_output_tokens=1)
    verify = PyTorchResidentRuntime._verify_consumed_cache

    # Inject a wrong expected count at the worker boundary, without changing
    # HF or model code.
    def mismatch(output, count):
      verify(output, count + 1)

    with mock.patch.object(
        self.runtime,
        '_verify_consumed_cache',
        autospec=True,
        spec_set=True,
        side_effect=mismatch,
    ):
      with self.assertRaisesRegex(
          ValueError, 'Actual KV cache length .* differs'
      ):
        self.runtime.execute(request, lambda *a, **k: None)
    self.assertIsNotNone(self.runtime.model)
    self.assertIsNone(self.runtime.past)
    self.assertFalse((Path(request['output']) / 'export').exists())
    raw = Path(request['output']) / 'raw/prefill'
    self.assertEqual(
        json.loads((raw / 'generation.json').read_text())['status'], 'failed'
    )
    self.assertEqual((raw / 'tokens.jsonl').read_text(), '')

  def test_template_that_rewrites_prefix_replays_with_explicit_reason(self):
    from model_debugger_contracts.model_identity import describe_model

    path = self.root / 'dynamic-template-model'
    self.model.save_pretrained(path, safe_serialization=True)
    self.tokenizer.save_pretrained(path)
    config = json.loads((path / 'tokenizer_config.json').read_text())
    # Transformers may save the chat template in a separate jinja file.
    template = path / 'chat_template.jinja'
    if template.is_file():
      template.write_text(
          '{% if messages|length > 1 %}! {% endif %}' + template.read_text()
      )
    else:
      config['chat_template'] = (
          '{% if messages|length > 1 %}! {% endif %}' + config['chat_template']
      )
      (path / 'tokenizer_config.json').write_text(json.dumps(config))
    files, checksum = describe_model(path)
    model_fields = dict(
        model=str(path), model_files=files, model_sha256=checksum
    )
    self.initialize(**model_fields)
    self.runtime.execute(
        self.next_turn('one', max_output_tokens=1, **model_fields),
        lambda *a, **k: None,
    )
    second_request = self.next_turn('two', **model_fields)
    result = self.runtime.execute(second_request, lambda *a, **k: None)
    self.assertFalse(result['cache_reuse']['reused'])
    self.assertEqual(
        result['cache_reuse']['reason'], 'serialized_token_prefix_changed'
    )
    self.assertEqual(result['cache_reuse']['reused_token_count'], 0)
    self.assertEqual(result['model_load_count'], 1)
    self.assert_replay(second_request, result)

  def test_hidden_generated_tokens_not_reused_after_text_retokenization(self):
    self.initialize()
    first = self.runtime.execute(self.next_turn('one'), lambda *a, **k: None)
    self.assertIn(0, first['context_token_ids'][:-1])
    request = self.next_turn('two')
    second = self.runtime.execute(request, lambda *a, **k: None)
    self.assertFalse(second['cache_reuse']['reused'])
    self.assertEqual(
        second['cache_reuse']['reason'], 'serialized_token_prefix_changed'
    )
    self.assert_replay(request, second)

  def test_sampling_failure_ends_execution_without_reloading_model(self):
    self.initialize()
    loaded = self.runtime.model
    request = self.next_turn('failure')
    with mock.patch.object(
        pytorch_worker,
        '_choose_token',
        autospec=True,
        spec_set=True,
        side_effect=ValueError('bad logits'),
    ):
      with self.assertRaisesRegex(ValueError, 'bad logits'):
        self.runtime.execute(request, lambda *a, **k: None)
    self.assertEqual(self.runtime.state, 'invalidated')
    self.assertIs(self.runtime.model, loaded)
    self.assertIsNone(self.runtime.past)
    self.assertTrue(
        all(
            not module._forward_hooks and not module._forward_pre_hooks
            for module in loaded.modules()
        )
    )
    with self.assertRaisesRegex(ValueError, 'Initialize again'):
      self.runtime.execute(self.next_turn('retry'), lambda *a, **k: None)
    with self.assertRaisesRegex(ValueError, 'cannot create another Chat'):
      self.runtime.execute(
          self.request('reinitialize', operation='initialize'),
          lambda *a, **k: None,
      )
    self.assertEqual(self.runtime.model_load_count, 1)

  def test_cancel_invalidates_cache_and_preserves_cancelled_raw_evidence(self):
    self.initialize()
    loaded = self.runtime.model
    self.runtime.execute(
        self.next_turn('one', max_output_tokens=1), lambda *a, **k: None
    )
    request = self.next_turn('cancel')
    marker = self.root / 'cancel'

    def emit(kind, **data):
      if kind == 'delta':
        marker.touch()

    with self.assertRaisesRegex(InterruptedError, 'Cancelled'):
      self.runtime.execute(request, emit)
    self.assertIsNone(self.runtime.past)
    self.assertIs(self.runtime.model, loaded)
    self.assertEqual(self.runtime.state, 'chat_closed')
    generation = json.loads(
        (Path(request['output']) / 'raw/prefill/generation.json').read_text()
    )
    self.assertEqual(generation['status'], 'cancelled')
    self.assertFalse((Path(request['output']) / 'export').exists())

  def test_new_chat_reuses_model_with_empty_history_and_close_releases_model(
      self,
  ):
    self.initialize()
    self.runtime.execute(
        self.next_turn('old', max_output_tokens=1), lambda *a, **k: None
    )
    model = self.runtime.model
    self.runtime.reset_chat()
    self.assertIs(self.runtime.model, model)
    self.assertIsNone(self.runtime.past)
    initialized = self.runtime.execute(
        self.request('new', operation='initialize', messages=[]),
        lambda *a, **k: None,
    )
    self.assertEqual(initialized['turn_sequence'], 0)
    self.assertEqual(initialized['model_load_count'], 1)
    self.assertIs(self.runtime.model, model)
    request = self.next_turn('fresh', max_output_tokens=1)
    result = self.runtime.execute(request, lambda *a, **k: None)
    self.assertEqual(result['turn_sequence'], 1)
    self.assertEqual(result['cache_reuse']['reason'], 'empty_cache')
    self.assert_replay(request, result)
    loaded = weakref.ref(self.runtime.model)
    del model
    self.runtime.close()
    self.runtime.close()
    self.assertIsNone(loaded())
    with self.assertRaisesRegex(ValueError, 'Initialize again'):
      self.runtime.execute(self.request('closed'), lambda *a, **k: None)

  def test_stale_saved_history_is_rejected_before_forward(self):
    self.initialize()
    self.runtime.execute(
        self.next_turn('one', max_output_tokens=1), lambda *a, **k: None
    )
    request = self.request('stale', messages=[], turn=2)
    with self.assertRaisesRegex(ValueError, 'history differs'):
      self.runtime.execute(request, lambda *a, **k: None)
    self.assertFalse((Path(request['output']) / 'raw').exists())
    self.assertIsNotNone(self.runtime.model)

  def test_capacity_rejection_preserves_context_and_allows_shorter_input(self):
    from model_debugger_contracts.errors import InputRejected

    self.initialize()
    self.runtime.execute(
        self.next_turn('one', max_output_tokens=1), lambda *a, **k: None
    )
    loaded, past = self.runtime.model, self.runtime.past
    history, tokens = deepcopy(self.runtime.messages), list(
        self.runtime.token_history
    )
    rng = self.runtime.generator.get_state().clone()
    request = self.next_turn(
        'too-long', prompt='hello ' * 1024, operation='preflight'
    )
    with self.assertRaises(InputRejected):
      self.runtime.execute(request, lambda *a, **k: None)
    self.assertEqual(self.runtime.state, 'ready')
    self.assertIs(self.runtime.model, loaded)
    self.assertIs(self.runtime.past, past)
    self.assertEqual(self.runtime.messages, history)
    self.assertEqual(self.runtime.token_history, tokens)
    self.assertTrue(
        self.runtime.torch.equal(rng, self.runtime.generator.get_state())
    )
    self.assertFalse((Path(request['output']) / 'raw').exists())
    result = self.runtime.execute(
        self.next_turn('shorter', max_output_tokens=1), lambda *a, **k: None
    )
    self.assertEqual(result['turn_sequence'], 2)

  def test_export_failure_keeps_successful_text_and_resident_context(self):
    self.initialize()
    request = self.next_turn('missing-debug', max_output_tokens=1)
    with mock.patch.object(
        pytorch_capture,
        'export_capture',
        autospec=True,
        spec_set=True,
        side_effect=OSError('disk error'),
    ):
      result = self.runtime.execute(request, lambda *a, **k: None)
    self.assertEqual(result['generation_status'], 'completed')
    self.assertEqual(
        result['debug_data'], {'status': 'unavailable', 'error': 'disk error'}
    )
    self.assertIsNone(result['captured_tensors'])
    self.assertEqual(self.runtime.messages[-1]['content'], result['output'])
    self.assertEqual(self.runtime.state, 'ready')
    saved = json.loads(
        (Path(request['output']) / 'generation-result.json').read_text()
    )
    self.assertEqual(saved['output'], result['output'])
    next_result = self.runtime.execute(
        self.next_turn('after-debug-error', max_output_tokens=1),
        lambda *a, **k: None,
    )
    self.assertEqual(next_result['turn_sequence'], 2)

  def test_new_chat_initialization_error_does_not_poison_loaded_model(self):
    self.initialize()
    loaded = self.runtime.model
    self.runtime.reset_chat()
    with self.assertRaisesRegex(ValueError, 'Invalid PyTorch top_k'):
      self.runtime.execute(
          self.request(
              'bad-chat', operation='initialize', generation={'topK': 0}
          ),
          lambda *a, **k: None,
      )
    self.assertIs(self.runtime.model, loaded)
    self.assertEqual(self.runtime.state, 'chat_closed')
    result = self.runtime.execute(
        self.request('retry-chat', operation='initialize'), lambda *a, **k: None
    )
    self.assertEqual(result['model_load_count'], 1)
    self.assertEqual(result['turn_sequence'], 0)

  def test_stdio_protocol_keeps_one_process_until_close(self):
    log_path = self.root / 'resident-protocol.log'
    with log_path.open('w') as log:
      process = subprocess.Popen(
          [
              sys.executable,
              '-m',
              'model_debugger_runner.pytorch_worker',
              '--serve',
          ],
          stdin=subprocess.PIPE,
          stdout=log,
          stderr=log,
          text=True,
      )
      try:

        def call(request, rejected=False):
          directory = Path(request['output'])
          directory.rmdir()  # serve_request owns output creation.
          path = directory.with_suffix('.request.json')
          path.write_text(json.dumps(request))
          process.stdin.write(json.dumps({'request': str(path)}) + '\n')
          process.stdin.flush()
          deadline = time.monotonic() + 30
          while time.monotonic() < deadline:
            if (directory / 'result.json').is_file():
              return json.loads((directory / 'result.json').read_text())
            if rejected and (directory / 'terminal.json').is_file():
              return json.loads((directory / 'terminal.json').read_text())
            if process.poll() is not None:
              self.fail(log_path.read_text())
            time.sleep(0.05)
          self.fail('Resident protocol timed out: ' + log_path.read_text())

        initialized = call(
            self.request('init-protocol', operation='initialize')
        )
        first = call(
            self.request('first-protocol', turn=1, max_output_tokens=1)
        )
        history = [
            dict(role='user', content='hello world'),
            dict(role='assistant', content=first['output']),
        ]
        second = call(
            self.request(
                'second-protocol', turn=2, messages=history, max_output_tokens=1
            )
        )
        self.assertEqual(
            {r['worker_pid'] for r in (initialized, first, second)},
            {process.pid},
        )
        self.assertEqual(
            {r['model_load_count'] for r in (initialized, first, second)}, {1}
        )
        self.assertTrue(second['cache_reuse']['reused'])
        rejection = call(
            self.request(
                'preflight-protocol',
                operation='preflight',
                prompt='hello ' * 1024,
            ),
            rejected=True,
        )
        self.assertEqual(rejection['generation_status'], 'rejected')
        self.assertIsNone(process.poll())
        reset = call(self.request('reset-protocol', operation='reset'))
        self.assertTrue(reset['chat_closed'])
        new_chat = call(
            self.request('new-chat-protocol', operation='initialize')
        )
        fresh = call(
            self.request('fresh-protocol', turn=1, max_output_tokens=1)
        )
        self.assertEqual(
            {r['worker_pid'] for r in (initialized, new_chat, fresh)},
            {process.pid},
        )
        self.assertEqual(fresh['model_load_count'], 1)
        self.assertEqual(fresh['turn_sequence'], 1)
        self.assertEqual(fresh['cache_reuse']['reason'], 'empty_cache')
        process.stdin.write(json.dumps({'operation': 'close'}) + '\n')
        process.stdin.flush()
        process.stdin.close()
        self.assertEqual(process.wait(timeout=10), 0)
      finally:
        if process.poll() is None:
          process.kill()
          process.wait(timeout=5)

  def test_runner_worker_reset_and_rejection_keep_the_same_model_process(self):
    from model_debugger_runner.python_workers import PythonWorkers
    from model_debugger_contracts.errors import InputRejected

    workers = PythonWorkers()

    def request(name, **changes):
      value = self.request(name, **changes)
      Path(value['output']).rmdir()
      value['session_id'] = 'resident-test-session'
      value['run'].update(id='ref', runtime='PyTorch')
      return value

    try:
      first = workers.execute(
          request('owner-init', operation='initialize'),
          lambda *a, **k: None,
          lambda: False,
      )
      with self.assertRaises(InputRejected):
        workers.execute(
            request(
                'owner-reject', operation='preflight', prompt='hello ' * 1024
            ),
            lambda *a, **k: None,
            lambda: False,
        )
      workers.reset_session('resident-test-session')
      second = workers.execute(
          request('owner-new', operation='initialize'),
          lambda *a, **k: None,
          lambda: False,
      )
      result = workers.execute(
          request('owner-generate', turn=1, max_output_tokens=1),
          lambda *a, **k: None,
          lambda: False,
      )
      self.assertEqual(
          {r['worker_pid'] for r in (first, second, result)},
          {first['worker_pid']},
      )
      self.assertEqual(result['model_load_count'], 1)
      self.assertEqual(result['turn_sequence'], 1)
    finally:
      workers.close()


if __name__ == '__main__':
  unittest.main()
