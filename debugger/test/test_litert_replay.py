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

"""Evidence validation for the explicit first-Prefill native replay contract."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from model_explorer_debugger.runtime.litert_lm_adapter import (
    replay_text_evidence,
    token_decode_evidence,
    token_replay_evidence,
    validate_token_replay,
)
import numpy as np
from safetensors.numpy import load_file, save_file


class TokenReplayEvidenceTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)
    self.capture = self.root / 'capture'
    self.capture.mkdir()
    self.ids = [2, 4, 5, 6]
    self.positions = np.array([0, 1, 2, 0, 0], dtype=np.int32)
    self.mask = np.zeros((1, 1, 5, 8), dtype=np.bool_)
    self.mask[0, 0, :3, :3] = np.tri(3, dtype=np.bool_)
    self.write()

  def write(self, *, step='4'):
    for name, array in [
        ('pre_input_pos', self.positions),
        ('pre_mask', self.mask),
    ]:
      save_file(
          {name: array},
          str(self.capture / f'{name}.safetensors'),
          metadata={'signature': 'prefill_128', 'step': step},
      )

  def proof(self):
    return token_replay_evidence(
        self.root,
        self.capture,
        self.ids,
        {'token_ids': self.ids, 'source_revision': 'requester-claim'},
        {'library_sha256': 'native-library'},
    )

  def test_actual_inputs_have_raw_references_and_distinct_pending_token(self):
    proof = self.proof()
    self.assertEqual(proof['actual_token_ids'], self.ids)
    self.assertEqual(proof['graph_token_ids'], [2, 4, 5])
    self.assertEqual(proof['graph_valid_length'], 3)
    self.assertEqual(proof['pending_token_id'], 6)
    self.assertEqual(proof['weight_equivalence'], 'unverified')
    self.assertEqual(
        proof['requested_source'], {'source_revision': 'requester-claim'}
    )
    resource = proof['input_resource']
    np.testing.assert_array_equal(
        load_file(self.root / resource['path'])['input_ids'], [self.ids]
    )
    self.assertEqual(
        load_file(self.root / resource['path'])['input_ids'].dtype, np.int32
    )
    digest = proof.pop('proof_sha256')
    self.assertEqual(
        digest,
        hashlib.sha256(
            json.dumps(
                proof, sort_keys=True, separators=(',', ':'), allow_nan=False
            ).encode()
        ).hexdigest(),
    )

  def test_wrong_position_padding_or_valid_position_is_rejected(self):
    for index in (2, 4):
      with self.subTest(index=index):
        self.positions[index] = 9
        self.write()
        with self.assertRaisesRegex(ValueError, 'position IDs'):
          self.proof()
        self.positions[index] = 2 if index == 2 else 0

  def test_attention_mask_must_prove_causality_and_padding(self):
    for row, column in ((0, 2), (3, 0), (0, 7)):
      with self.subTest(row=row, column=column):
        self.mask[0, 0, row, column] = True
        self.write()
        with self.assertRaisesRegex(ValueError, 'causal mask'):
          self.proof()
        self.mask[0, 0, row, column] = False

  def test_wrong_graph_step_missing_and_duplicate_input_rejected(self):
    self.write(step='3')
    with self.assertRaisesRegex(ValueError, 'graph step'):
      self.proof()
    self.write()
    position = self.capture / 'pre_input_pos.safetensors'
    position.rename(self.capture / 'position.missing')
    with self.assertRaisesRegex(ValueError, 'missing actual'):
      self.proof()
    self.write()
    (self.capture / 'duplicate.safetensors').write_bytes(position.read_bytes())
    with self.assertRaisesRegex(ValueError, 'exactly one'):
      self.proof()

  def test_symlink_escape_rejected(self):
    path = self.capture / 'pre_input_pos.safetensors'
    outside = self.root / 'outside.safetensors'
    path.rename(outside)
    path.symlink_to(outside)
    with self.assertRaisesRegex(ValueError, 'escapes capture'):
      self.proof()

  def test_small_scope_and_exact_integer_ids_are_explicit(self):
    for value in (
        None,
        {},
        {'token_ids': [1]},
        {'token_ids': [1, True]},
        {'token_ids': [1, -1]},
        {'token_ids': [1, 2.0]},
        {'token_ids': [1] * 129},
    ):
      with self.subTest(value=value), self.assertRaises(ValueError):
        validate_token_replay(value)
    self.assertEqual(validate_token_replay({'token_ids': self.ids}), self.ids)

  def test_native_admission_readback_cannot_be_replaced_by_request_claim(self):
    with self.assertRaisesRegex(ValueError, 'admitted IDs'):
      token_replay_evidence(
          self.root, self.capture, [2, 4, 5, 7], {'token_ids': self.ids}, {}
      )

  def decode_fixture(self):
    proof = self.proof()
    proof['requested_source']['decode_token_ids'] = [7, 8]
    for step, position in [(4, 3), (5, 4)]:
      mask = np.zeros((1, 1, 1, 8), dtype=np.bool_)
      mask[0, 0, 0, : position + 1] = True
      for key, array in [
          ('pre_input_pos', np.array([position], dtype=np.int32)),
          ('pre_mask', mask),
      ]:
        save_file(
            {key: array},
            str(self.capture / f'decode_{key}_step_{step}.safetensors'),
            metadata={'signature': 'decode', 'step': str(step)},
        )
    return proof

  def test_decode_graph_step_is_distinct_from_consumed_position(self):
    proof = token_decode_evidence(
        self.root, self.capture, self.decode_fixture(), [7, 8]
    )
    rows = proof['decode']['executions']
    self.assertEqual(
        [(r['step'], r['input_position'], r['input_token_id']) for r in rows],
        [(4, 3, 6), (5, 4, 7)],
    )
    self.assertEqual(proof['decode']['pending_token_id'], 8)
    self.assertEqual(proof['decode']['pending_token_position'], 5)
    self.assertEqual(proof['decode']['processed_token_count'], 5)

  def test_decode_rejects_wrong_position_mask_or_output_readback(self):
    proof = self.decode_fixture()
    with self.assertRaisesRegex(ValueError, 'output IDs'):
      token_decode_evidence(self.root, self.capture, proof, [7, 9])
    position = self.capture / 'decode_pre_input_pos_step_4.safetensors'
    save_file(
        {'pre_input_pos': np.array([4], dtype=np.int32)},
        str(position),
        metadata={'signature': 'decode', 'step': '4'},
    )
    with self.assertRaisesRegex(ValueError, 'actual position'):
      token_decode_evidence(self.root, self.capture, proof, [7, 8])
    proof = self.decode_fixture()
    mask = np.ones((1, 1, 1, 8), dtype=np.bool_)
    save_file(
        {'pre_mask': mask},
        str(self.capture / 'decode_pre_mask_step_4.safetensors'),
        metadata={'signature': 'decode', 'step': '4'},
    )
    with self.assertRaisesRegex(ValueError, 'actual mask'):
      token_decode_evidence(self.root, self.capture, proof, [7, 8])

  def test_observed_callback_text_wins_over_tokenizer_projection(self):
    trace = self.root / 'trace.jsonl'
    trace.write_text(
        json.dumps({'token_ids': [[7]], 'texts': ['Hello']})
        + '\n'
        + json.dumps({'token_ids': [[8]], 'texts': ['!']})
        + '\n'
    )

    class Engine:

      def detokenize(self, ids):
        raise AssertionError('Observed native text must not be replaced')

    result = replay_text_evidence(
        Engine(), [7, 8], trace_path=trace, native_library_sha256='a' * 64
    )
    self.assertEqual(result['output'], 'Hello!')
    self.assertEqual([row['text'] for row in result['tokens']], ['Hello', '!'])
    self.assertEqual(result['text_source'], 'native_decode_callback')
    self.assertEqual(
        result['text_evidence']['trace_sha256'],
        hashlib.sha256(trace.read_bytes()).hexdigest(),
    )

  def test_missing_callback_uses_same_native_engine_and_labels_projection(self):
    calls = []

    class Engine:

      def detokenize(self, ids):
        calls.append(ids)
        return ''.join({7: 'Hello', 8: '!'}[token] for token in ids)

    result = replay_text_evidence(Engine(), [7, 8])
    self.assertEqual(result['output'], 'Hello!')
    self.assertEqual(calls, [[7, 8], [7], [8]])
    self.assertEqual(result['text_source'], 'native_tokenizer_detokenize')
    self.assertTrue(
        all(
            row['text_source'] == 'native_tokenizer_detokenize'
            for row in result['tokens']
        )
    )
    self.assertIsNone(result['text_evidence']['trace_sha256'])
    self.assertEqual(
        replay_text_evidence(Engine(), [])['text_source'], 'not_generated'
    )

  def test_callback_ids_must_match_even_when_text_looks_correct(self):
    trace = self.root / 'trace.jsonl'
    trace.write_text(json.dumps({'token_ids': [[9]], 'texts': ['Hello!']}))
    with self.assertRaisesRegex(ValueError, 'callback token IDs'):
      replay_text_evidence(None, [7, 8], trace_path=trace)

  def test_grouped_callback_text_is_not_assigned_to_individual_ids(self):
    trace = self.root / 'trace.jsonl'
    trace.write_text(json.dumps({'token_ids': [[7, 8]], 'texts': ['Hello!']}))

    class Engine:

      def detokenize(self, ids):
        return {7: 'Hello', 8: '!'}[ids[0]]

    result = replay_text_evidence(Engine(), [7, 8], trace_path=trace)
    self.assertEqual(result['output'], 'Hello!')
    self.assertEqual(result['text_source'], 'native_decode_callback')
    self.assertEqual([row['text'] for row in result['tokens']], ['Hello', '!'])
    self.assertTrue(
        all(
            row['text_source'] == 'native_tokenizer_detokenize'
            for row in result['tokens']
        )
    )


if __name__ == '__main__':
  unittest.main()
