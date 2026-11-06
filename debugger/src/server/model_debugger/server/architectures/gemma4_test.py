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

"""Tests of Gemma 4's definition, on the pinned configs of E2B and E4B."""

import json
import pathlib
from typing import Any

from absl.testing import absltest
from model_debugger.server.architectures import gemma4
from model_debugger.server.architectures import skeleton
from model_debugger.server.testing import test_main

_DATA = pathlib.Path(gemma4.__file__).parent / 'data' / 'gemma4'


def _text_config(checkpoint: str) -> dict[str, Any]:
  path = _DATA / f'{checkpoint}.config.json'
  return json.loads(path.read_text(encoding='utf-8'))['text_config']


class Gemma4Test(absltest.TestCase):
  """Tests of build(): what Gemma 4's config makes."""

  def setUp(self):
    """E2B's text config and skeleton."""
    super().setUp()
    self.config = _text_config('gemma-4-E2B-it')
    self.e2b = gemma4.build(self.config)

  def test_its_skeletons_pass_the_schema(self):
    """E2B's 35 layers and E4B's 42, every rule met."""
    self.assertEqual(skeleton.check(self.e2b), [])
    self.assertLen(self.e2b['layers'], 35)
    e4b = gemma4.build(_text_config('gemma-4-E4B-it'))
    self.assertEqual(skeleton.check(e4b), [])
    self.assertLen(e4b['layers'], 42)

  def test_one_definition_two_inputs_four_anchors(self):
    """Two inputs, the second read by ple; four anchors, three blocks."""
    (definition,) = self.e2b['semantic_graph']
    self.assertEqual(
        [(i['id'], i['source']) for i in definition['inputs']],
        [
            ('input', 'residual'),
            ('per_layer_input', 'external per-layer embedding'),
        ],
    )
    ple = next(n for n in definition['nodes'] if n['id'] == 'ple')
    self.assertEqual(skeleton.sources(ple), ['add_ffn:0', 'per_layer_input:0'])
    self.assertEqual(
        [(a['id'], a['of'], a.get('block')) for a in definition['anchors']],
        [
            ('input', 'input:0', None),
            ('intermediate.1', 'add_attn:0', 'Attention'),
            ('intermediate.2', 'add_ffn:0', 'Feed Forward'),
            ('output', 'layer_scale:0', 'Per-Layer Embedding'),
        ],
    )
    attention = next(n for n in definition['nodes'] if n['id'] == 'attn')
    self.assertEqual(attention['label'], 'Multi-Query Attention')
    self.assertTrue(all(n['namespace'] for n in definition['nodes']))

  def test_a_layer_that_borrows_kv_skips_its_kv_steps(self):
    """E2B's layers 15 on borrow from 13 (sliding) or 14 (full)."""
    first_shared = (
        self.config['num_hidden_layers'] - self.config['num_kv_shared_layers']
    )
    self.assertEqual(first_shared, 15)
    layers = self.e2b['layers']
    self.assertNotIn('from', layers[0])
    self.assertEqual(
        layers[1]['from'], {'input': {'layer': 0, 'anchor': 'output'}}
    )
    self.assertNotIn('kv', layers[14]['from'])
    types = self.config['layer_types']
    for index in range(first_shared, len(layers)):
      want = 14 if types[index] == 'full_attention' else 13
      self.assertEqual(layers[index]['from']['kv'], {'layer': want})
      attention = layers[index]['attrs']['attn']['ops']
      self.assertTrue(attention['k_proj']['skip'])
      self.assertNotIn('skip', attention['q_proj'])
      # A layer that borrows K and V has the double-wide MLP.
      self.assertEqual(
          layers[index]['attrs']['ffn']['ops']['up_proj']['out'],
          2 * self.config['intermediate_size'],
      )
    self.assertEqual(
        layers[0]['attrs']['ffn']['ops']['up_proj']['out'],
        self.config['intermediate_size'],
    )

  def test_each_layer_has_its_own_parameters(self):
    """Its slice of the PLE table; v_norm alone learns no scale."""
    for index, layer in enumerate(self.e2b['layers']):
      ops = layer['attrs']
      self.assertEqual(ops['ple']['ops']['gate']['slice'], index)
      self.assertFalse(ops['attn']['ops']['v_norm']['learned_scale'])
      self.assertTrue(ops['attn']['ops']['q_norm']['learned_scale'])
      self.assertEqual(ops['attn']['ops']['attend']['score_scale'], 1.0)


if __name__ == '__main__':
  test_main.main()
