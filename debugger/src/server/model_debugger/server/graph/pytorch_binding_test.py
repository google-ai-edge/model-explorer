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

"""Tests for pytorch_binding: the PyTorch runner's anchors of Gemma 4."""

import json
from typing import Any

from absl.testing import absltest
from model_debugger.server.architectures import catalog
from model_debugger.server.graph import architecture
from model_debugger.server.graph import pytorch_binding
from model_debugger.server.testing import test_main


def _pinned_config(skeleton_id: str) -> dict[str, Any]:
  (gemma4,) = [a for a in catalog.architectures() if a.id == 'gemma4']
  entry = gemma4.entry(skeleton_id)
  path = gemma4.directory / entry['pinned']['file']
  return json.loads(path.read_text(encoding='utf-8'))


class PyTorchBindingTest(absltest.TestCase):
  """Where the PyTorch runner takes Gemma 4's anchors."""

  def test_e2b_has_four_anchors_in_each_of_its_35_layers(self):
    """Each layer's input and output and the end of each of its blocks."""
    bound = pytorch_binding.anchors(_pinned_config('gemma-4-E2B-it'))
    self.assertEqual(bound.architecture, 'gemma4')
    self.assertEqual(bound.width, 1536)
    self.assertEqual(bound.signatures, ('prefill', 'decode'))
    self.assertLen(bound.anchors, 35 * 4)
    layer_3 = [a for a in bound.anchors if a.layer == 3]
    self.assertEqual(
        [(a.anchor, a.module, a.edge, a.path) for a in layer_3],
        [
            ('input', 'model.language_model.layers.3', 'in', 'args.0'),
            (
                'intermediate.1',
                'model.language_model.layers.3.pre_feedforward_layernorm',
                'in',
                'args.0',
            ),
            (
                'intermediate.2',
                'model.language_model.layers.3.per_layer_input_gate',
                'in',
                'args.0',
            ),
            ('output', 'model.language_model.layers.3', 'out', 'output'),
        ],
    )
    self.assertEqual(layer_3[1].label, 'Attention block output')

  def test_the_anchors_are_the_layer_and_its_blocks_ends(self):
    """The skeleton's anchors: the input, then where each block ends."""
    _, document = catalog.generate(_pinned_config('gemma-4-E2B-it'))
    definition = document['semantic_graph'][document['layers'][0]['def']]
    self.assertEqual(
        [(a['id'], a.get('block')) for a in definition['anchors']],
        [
            ('input', None),
            ('intermediate.1', 'Attention'),
            ('intermediate.2', 'Feed Forward'),
            ('output', 'Per-Layer Embedding'),
        ],
    )
    rules = pytorch_binding.binding('gemma4')['anchors']
    self.assertEqual(list(rules), [a['id'] for a in definition['anchors']])

  def test_litert_lm_graphs_bind_by_their_own_binding_alone(self):
    """architectures() reads gemma4.json, not the PyTorch binding."""
    found = architecture.architectures()
    self.assertEqual([a.id for a in found], ['gemma4'])
    self.assertNotIn('runtime', found[0].binding)

  def test_an_architecture_without_a_pytorch_binding_says_so(self):
    """No binding is a ValueError that names the architecture."""
    with self.assertRaisesRegex(ValueError, 'No PyTorch binding of qwen3'):
      pytorch_binding.binding('qwen3')


if __name__ == '__main__':
  test_main.main()
