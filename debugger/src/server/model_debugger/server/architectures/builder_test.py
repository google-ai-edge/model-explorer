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

"""Tests of what a definition writes a skeleton with."""

from absl.testing import absltest
from model_debugger.server.architectures import builder
from model_debugger.server.testing import test_main


class BuilderTest(absltest.TestCase):
  """Tests of the words a config gives, and of Graph."""

  def test_an_activation_is_its_mathematics(self):
    """gelu_pytorch_tanh is GELU by tanh; an unknown key is refused."""
    self.assertEqual(
        builder.activation({'hidden_activation': 'gelu_pytorch_tanh'}),
        {'kind': 'gelu', 'approximate': 'tanh'},
    )
    with self.assertRaisesRegex(ValueError, 'ACTIVATIONS'):
      builder.activation({'hidden_activation': 'quick_gelu'})

  def test_one_kv_head_is_multi_query_not_grouped(self):
    """MHA, MQA, GQA by the K and V heads; GeGLU, SwiGLU, MLP."""
    self.assertEqual(builder.attention_label(8, 8), 'Multi-Head Attention')
    self.assertEqual(builder.attention_label(8, 1), 'Multi-Query Attention')
    self.assertEqual(builder.attention_label(8, 2), 'Grouped-Query Attention')
    gelu = {'kind': 'gelu', 'approximate': 'tanh'}
    self.assertEqual(builder.ffn_label(gelu, True), 'GeGLU')
    self.assertEqual(builder.ffn_label({'kind': 'silu'}, True), 'SwiGLU')
    self.assertEqual(builder.ffn_label(gelu, False), 'MLP')

  def test_a_proportional_rope_leaves_the_rest_without_position(self):
    """Its rotary and NoPE dims add up to the head's; theta an integer."""
    config = {
        'rope_parameters': {
            'full_attention': {
                'rope_type': 'proportional',
                'partial_rotary_factor': 0.25,
                'rope_theta': 1000000.0,
            },
            'sliding_attention': {
                'rope_type': 'default',
                'rope_theta': 10000.5,
            },
        }
    }
    self.assertEqual(
        builder.rope(config, 'full_attention', 512),
        {
            'variant': 'proportional',
            'theta': 1000000,
            'rotary_dims': 128,
            'nope_dims': 384,
        },
    )
    self.assertEqual(
        builder.rope(config, 'sliding_attention', 256),
        {
            'variant': 'default',
            'theta': 10000.5,
            'rotary_dims': 256,
            'nope_dims': 0,
        },
    )

  def test_a_node_reads_tensors_of_nodes_before_it(self):
    """Edges name `<producer>:<port>`; a later producer is refused."""
    graph = builder.Graph()
    graph.add_input('input', ['B', 'S', 4], 'residual')
    graph.add('norm', 'RMSNorm', ['input'], ['B', 'S', 4])
    graph.add('add', 'Add', ['input', 'norm:0'], ['B', 'S', 4])
    self.assertEqual(
        [
            (e['sourceNodeId'], e['sourceNodeOutputId'])
            for e in graph.nodes[1]['incomingEdges']
        ],
        [('input', '0'), ('norm', '0')],
    )
    self.assertEqual(
        graph.nodes[0]['outputsMetadata'][0]['attrs'],
        [
            {'key': 'tensor_name', 'value': 'norm:0'},
            {'key': 'tensor_shape', 'value': 'B,S,4'},
        ],
    )
    self.assertEqual(graph.nodes[0]['ops'], [{'id': 'norm', 'op': 'RMSNorm'}])
    with self.assertRaisesRegex(KeyError, 'topological'):
      graph.add('mul', 'Mul', ['later'], ['B', 'S', 4])
    with self.assertRaisesRegex(ValueError, 'SEMANTIC_OPS'):
      graph.add('fc', 'Mul', ['add'], ['B', 'S', 4], [('fc', 'Gemm')])

  def test_an_anchor_is_of_a_tensor_with_its_port(self):
    """`of` without a port is of port 0."""
    self.assertEqual(
        builder.anchor('output', 'layer output', 'scale'),
        {'id': 'output', 'semantic': 'layer output', 'of': 'scale:0'},
    )


if __name__ == '__main__':
  test_main.main()
