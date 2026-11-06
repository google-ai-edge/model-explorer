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

"""Tests of the rules by which two sides' models compare, and the hint."""

from absl.testing import absltest
from model_debugger.server.analysis import comparability
from model_debugger.server.testing import test_main

_TABLE = 'a' * 64
_E2B = dict(
    architecture='gemma4',
    architecture_name='Gemma 4',
    skeleton='gemma-4-E2B-it',
)
_LITERT = comparability.Model(
    'litert', 'LiteRT-LM', 'gemma-4-E2B-it.litertlm', _TABLE, **_E2B
)
_PYTORCH = comparability.Model(
    'pytorch', 'PyTorch', 'gemma-4-E2B-it', _TABLE, **_E2B
)
_QWEN = comparability.Model(
    'qwen', 'LiteRT-LM', 'Qwen3-0.6B.litertlm', 'b' * 64
)
_PENDING = comparability.Model(
    'litert',
    'LiteRT-LM',
    'gemma-4-E2B-it.litertlm',
    _TABLE,
    architecture_unknown=(
        'The architecture of gemma-4-E2B-it.litertlm is known once its '
        'graph is converted.'
    ),
)


class ComparabilityTest(absltest.TestCase):
  """Tests of the rules and the hint."""

  def test_one_file_compares_in_every_analysis(self):
    """One file: one vocabulary, tensors by place, the hint says so."""
    found = comparability.vocabularies(_LITERT, _LITERT)
    self.assertIsNone(found.differ)
    self.assertEqual(found.identities, {'ref': 'litert', 'target': 'litert'})
    self.assertEqual(
        comparability.tensor_pairing(_LITERT, _LITERT),
        comparability.SAME_MODEL,
    )
    self.assertEqual(
        comparability.verdict(_LITERT, _LITERT),
        {
            'code': comparability.SAME_FILE,
            'text': 'Same model file: every analysis compares.',
        },
    )

  def test_two_runtimes_of_one_architecture_pair_by_anchors(self):
    """PyTorch and LiteRT-LM of one skeleton: everything compares."""
    found = comparability.vocabularies(_PYTORCH, _LITERT)
    self.assertIsNone(found.differ)
    self.assertEqual(found.identities['ref'], f'tokenizer:{_TABLE}')
    self.assertEqual(
        comparability.tensor_pairing(_PYTORCH, _LITERT),
        comparability.BY_ANCHORS,
    )
    self.assertEqual(
        comparability.verdict(_PYTORCH, _LITERT),
        {
            'code': comparability.SAME_ARCHITECTURE,
            'text': (
                'Same architecture (Gemma 4 · gemma-4-E2B-it) and the '
                'same tokenizer: the generated text, logits, sampling, KV '
                "caches and every layer's anchors compare."
            ),
        },
    )

  def test_architecture_not_known_yet_says_why(self):
    """A LiteRT-LM model whose graph is not converted: not known yet."""
    self.assertEqual(
        comparability.tensor_pairing(_PYTORCH, _PENDING),
        comparability.NOT_PAIRED,
    )
    found = comparability.verdict(_PYTORCH, _PENDING)
    self.assertEqual(found['code'], comparability.DIFFERENT)
    self.assertEqual(
        found['text'],
        'Different model files with the same tokenizer: the generated '
        'text, logits and sampling compare. KV caches compare only at the '
        'layers both store in one shape. Whether their intermediate '
        'tensors pair is not known yet: The architecture of '
        'gemma-4-E2B-it.litertlm is known once its graph is converted.',
    )

  def test_two_runtimes_of_other_architectures_compare_text(self):
    """A model of no architecture the server has pairs no tensors."""
    qwen = comparability.Model('qwen', 'LiteRT-LM', 'Qwen3', 'b' * 64)
    found = comparability.verdict(_PYTORCH, qwen)
    self.assertEqual(found['code'], comparability.DIFFERENT)
    self.assertTrue(
        found['text'].startswith(
            'Different architectures: the generated text compares. Their '
            'tokenizers map ids to different pieces'
        ),
        found['text'],
    )
    self.assertIn(
        'gemma-4-E2B-it is Gemma 4, Qwen3 is of no architecture the server has',
        found['text'],
    )

  def test_two_files_of_one_runtime_pair_by_hand(self):
    """Two LiteRT-LM files pair their tensors by hand alone."""
    self.assertEqual(
        comparability.tensor_pairing(_LITERT, _QWEN),
        comparability.BY_HAND,
    )
    found = comparability.verdict(_LITERT, _QWEN)
    self.assertTrue(
        found['text'].startswith(
            'Different model files of one runtime: the generated text compares.'
        )
    )
    self.assertIn('pairs made by hand in Graph Diff', found['text'])

  def test_unknown_tokenizer(self):
    """A side whose tokenizer is not read has its own vocabulary."""
    unread = comparability.Model('x', 'PyTorch', 'unread', None, **_E2B)
    found = comparability.vocabularies(unread, _LITERT)
    self.assertEqual(found.differ, comparability.VOCABULARY_UNKNOWN)
    self.assertEqual(found.unknown, ('ref',))
    self.assertEqual(found.identities, {'ref': 'x', 'target': 'litert'})
    self.assertIn(
        'The tokenizer of unread cannot be read',
        comparability.verdict(unread, _LITERT)['text'],
    )


if __name__ == '__main__':
  test_main.main()
