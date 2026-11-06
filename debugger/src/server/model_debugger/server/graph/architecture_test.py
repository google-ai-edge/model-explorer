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

"""Tests of the architectures: detecting a model's, binding its anchors.

The graphs here are laid out as LiteRT-LM's Gemma 4 graphs are, as the
converter writes them, one residual stream of eight operators per layer.
"""

from typing import Any

from absl.testing import absltest
from model_debugger.server.graph import architecture
from model_debugger.server.graph import model_graph
from model_debugger.server.tap import tap_scan
from model_debugger.server.testing import test_main

_STACK = 'LanguageModel.decode_graph/LanguageModel.transformer_stack'
# One layer: (label, scope under `layer_N/layer_N`, what it reads).
_LAYER = (
    ('stablehlo_composite', '.pre_qkv/pre_attention_norm/composite', ['x']),
    ('fully_connected', '.pre_qkv/attn.pre_qkv/q_einsum/dot', [0]),
    ('add', '.post_qkv/add', ['x', 1]),
    ('stablehlo_composite', '.post_qkv/pre_ffw_norm/composite', [2]),
    ('add', '.post_qkv/add1', [2, 3]),
    ('quantize', '.post_qkv/per_layer_embedding_gate/composite', [4]),
    ('add', '.post_qkv/add2', [4, 5]),
    ('mul', '.post_qkv/{layer}._maybe_apply_skip_scale/mul', [6]),
)


def _graph(
    layers: int,
    width: int,
    partial: bool = False,
    inputs: list[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[tap_scan.Point]]:
  """A Gemma 4 decode graph of `layers` layers; and its points.

  Args:
      layers: How many whole layers.
      width: The hidden width.
      partial: Whether one more layer has only its first operator.
      inputs: The signature input names; Gemma 4's by default.

  Returns:
      The graphs, by signature (only 'decode'); and the points.
  """
  inputs = inputs or ['embeddings', 'per_layer_embeddings']
  nodes, points = [], []
  stream = ('graph_inputs', '0')
  for layer in range(layers + (1 if partial else 0)):
    first = len(nodes)
    steps = _LAYER[:1] if layer == layers else _LAYER
    for label, scope, reads in steps:
      name = f'layer_{layer}'
      namespace = (
          f'{_STACK}/transformer.transformer/{name}/{name}'
          + scope.format(layer=name)
      )
      node_id = str(len(nodes))
      nodes.append({
          'id': node_id,
          'label': label,
          'namespace': namespace,
          'incomingEdges': [
              {
                  'sourceNodeId': (
                      stream[0] if read == 'x' else str(first + read)
                  ),
                  'sourceNodeOutputId': stream[1] if read == 'x' else '0',
                  'targetNodeInputId': str(slot),
              }
              for slot, read in enumerate(reads)
          ],
          'outputsMetadata': [{
              'id': '0',
              'attrs': [
                  {'key': 'tensor_index', 'value': node_id},
                  {'key': 'tensor_name', 'value': namespace},
                  {
                      'key': 'tensor_shape',
                      'value': f'float32[1,1,{width}]',
                  },
              ],
          }],
      })
      points.append(
          tap_scan.Point(
              id=tap_scan.point_id('decode', int(node_id), 0),
              signature='decode',
              subgraph=0,
              op=int(node_id),
              output=0,
              tensor=int(node_id),
              tensor_name=namespace,
              shape=(1, 1, width),
              dtype='float32',
              operator=label.upper(),
              output_name=None,
              context_axes=(),
              reason='',
              layer=layer,
              layer_output=label == 'mul',
          )
      )
    if layer < layers:
      stream = (str(len(nodes) - 1), '0')
  graph_inputs = str(len(nodes))
  for node in nodes:
    for edge in node['incomingEdges']:
      if edge['sourceNodeId'] == 'graph_inputs':
        edge['sourceNodeId'] = graph_inputs
  nodes.append({
      'id': graph_inputs,
      'label': model_graph.GRAPH_INPUTS,
      'namespace': model_graph.GRAPH_INPUTS,
      'incomingEdges': [],
      'outputsMetadata': [
          {
              'id': str(slot),
              'attrs': [
                  {'key': 'tensor_name', 'value': f'decode_{name}:0'},
                  {'key': 'signature_name', 'value': name},
              ],
          }
          for slot, name in enumerate(inputs)
      ],
  })
  return {'decode': {'id': 'decode', 'nodes': nodes}}, points


class ArchitectureTest(absltest.TestCase):
  """Tests of detect() and bind() on Gemma 4 graphs."""

  def _bound(self, graphs, points):
    found = architecture.detect(graphs, {'decode': 'layer'}, points)
    document = found.architecture.skeleton(found.skeleton_id)
    return architecture.bind(
        found.architecture, document, graphs, {'decode': 'layer'}, points
    )['decode']

  def test_knows_gemma_4_by_its_graph_and_size(self):
    """E2B: 35 layers of 1536; E4B: 42 of 2560; another size has none."""
    for layers, width, skeleton_id in (
        (35, 1536, 'gemma-4-E2B-it'),
        (42, 2560, 'gemma-4-E4B-it'),
    ):
      graphs, points = _graph(layers, width)
      found = architecture.detect(graphs, {'decode': 'layer'}, points)
      self.assertEqual(found.architecture.id, 'gemma4')
      self.assertEqual(found.skeleton_id, skeleton_id)
      self.assertEqual(found.text, '')
    graphs, points = _graph(30, 1024)
    found = architecture.detect(graphs, {'decode': 'layer'}, points)
    self.assertEqual(found.architecture.name, 'Gemma 4')
    self.assertIsNone(found.skeleton_id)
    self.assertEqual(
        found.text,
        'No architecture template exists for this model: it is of the '
        'Gemma 4 family, but its decode has 30 layers of width 1024, and '
        'the Gemma 4 templates are gemma-4-E2B-it (35 layers of width '
        '1536), gemma-4-E4B-it (42 layers of width 2560).',
    )
    graphs, points = _graph(35, 1536, inputs=['embeddings'])
    found = architecture.detect(graphs, {'decode': 'layer'}, points)
    self.assertIsNone(found.architecture)
    self.assertIn('none of the architectures', found.text)
    found = architecture.detect(graphs, {}, points)
    self.assertIsNone(found.architecture)

  def test_binds_every_anchor_of_every_layer(self):
    """Input, intermediate.1, intermediate.2, output: each layer's."""
    graphs, points = _graph(35, 1536)
    layers = self._bound(graphs, points)
    self.assertLen(layers, 35)
    first = layers[0]['anchors']
    # Layer 0 reads the signature's input: no operator writes it, and
    # it is checked to be the one the architecture names.
    self.assertEqual(
        first['input'],
        {
            'bound': True,
            'checked': True,
            'node': '280',
            'output': '0',
            'tensor_name': 'decode_embeddings:0',
            'tensor_shape': '',
            'point': None,
            'observable': False,
            'reason': (
                'It is the input embeddings of decode, no '
                "operator's output; it cannot be observed."
            ),
            'layer_output': False,
        },
    )
    self.assertEqual(
        [first[a]['point'] for a in ('intermediate.1', 'intermediate.2')],
        ['decode:2:0', 'decode:4:0'],
    )
    self.assertEqual(first['output']['point'], 'decode:7:0')
    self.assertTrue(first['output']['layer_output'])
    for entry in layers:
      anchors = entry['anchors']
      self.assertTrue(all(a['bound'] for a in anchors.values()))
      # Every one was checked against the skeleton.
      self.assertTrue(all(a['checked'] for a in anchors.values()))
      if entry['layer']:
        before = layers[entry['layer'] - 1]['anchors']['output']
        self.assertEqual(anchors['input']['point'], before['point'])
      self.assertTrue(anchors['output']['layer_output'])

  def test_a_signature_without_a_layer_says_so(self):
    """Prefill computes fewer layers, the last one only in part."""
    graphs, points = _graph(14, 1536, partial=True)
    # The layer count is the decode graph's; here it binds E2B's.
    found = architecture.Detected(
        architecture.architectures()[0], 'gemma-4-E2B-it', ''
    )
    document = found.architecture.skeleton(found.skeleton_id)
    layers = architecture.bind(
        found.architecture, document, graphs, {'decode': 'layer'}, points
    )['decode']
    self.assertTrue(layers[13]['anchors']['output']['bound'])
    partial = layers[14]['anchors']
    self.assertTrue(partial['input']['bound'])
    self.assertEqual(
        partial['intermediate.1'],
        {
            'bound': False,
            'reason': (
                'decode has no operator in scope '
                'layer_14.post_qkv/pre_ffw_norm.'
            ),
        },
    )
    self.assertEqual(
        layers[20]['anchors']['output'],
        {'bound': False, 'reason': 'decode has no operator of layer 20.'},
    )

  def test_a_binding_that_breaks_the_skeleton_is_unbound(self):
    """An Add that is a mul; an input that is not the layer before's."""
    graphs, points = _graph(35, 1536)
    nodes = graphs['decode']['nodes']
    nodes[3 * 8 + 2]['label'] = 'mul'  # Layer 3's attention residual.
    # Layer 5's norm reads layer 4's intermediate.2, not its output.
    nodes[5 * 8]['incomingEdges'][0]['sourceNodeId'] = str(4 * 8 + 4)
    layers = self._bound(graphs, points)
    self.assertEqual(
        layers[3]['anchors']['intermediate.1'],
        {
            'bound': False,
            'reason': (
                'Node 26 is not anchor intermediate.1: node 26 is mul, not add.'
            ),
        },
    )
    # What reads an unbound anchor is bound, but not wholly checked.
    after = layers[3]['anchors']['intermediate.2']
    self.assertTrue(after['bound'])
    self.assertFalse(after['checked'])
    self.assertTrue(layers[3]['anchors']['output']['checked'])
    self.assertEqual(
        layers[5]['anchors']['input'],
        {
            'bound': False,
            'reason': (
                'Node 36 is not anchor input: the skeleton has it '
                "be layer 4's output, node 39."
            ),
        },
    )
    self.assertFalse(layers[5]['anchors']['intermediate.1']['checked'])

  def test_the_first_layers_input_is_the_signature_input(self):
    """Layer 0's input must be the embeddings, not another input."""
    graphs, points = _graph(35, 1536)
    nodes = graphs['decode']['nodes']
    nodes[0]['incomingEdges'][0]['sourceNodeOutputId'] = '1'
    layers = self._bound(graphs, points)
    self.assertEqual(
        layers[0]['anchors']['input'],
        {
            'bound': False,
            'reason': (
                "Node 280 is not anchor input: it is decode's "
                'input per_layer_embeddings, not embeddings.'
            ),
        },
    )
    # An operator's output is no signature input either.
    graphs, points = _graph(35, 1536)
    nodes = graphs['decode']['nodes']
    nodes[0]['incomingEdges'][0]['sourceNodeId'] = '1'
    layers = self._bound(graphs, points)
    self.assertEqual(
        layers[0]['anchors']['input'],
        {
            'bound': False,
            'reason': (
                "Node 1 is not anchor input: it is node 1's "
                "output, not decode's input embeddings."
            ),
        },
    )

  def test_each_block_is_the_operators_between_its_anchors(self):
    """Attention: input to intermediate.1; and so on, by edges alone."""
    graphs, points = _graph(35, 1536)
    nodes = graphs['decode']['nodes']
    # A weight layer 1's query reads, and an operator named in layer 1
    # that reads only the signature's input: neither is between anchors.
    nodes.append({
        'id': 'w',
        'label': model_graph.CONSTANT,
        'namespace': '',
        'incomingEdges': [],
        'outputsMetadata': [{'id': '0', 'attrs': []}],
    })
    nodes[9]['incomingEdges'].append(
        {'sourceNodeId': 'w', 'sourceNodeOutputId': '0'}
    )
    nodes.append({
        'id': 'side',
        'label': 'reshape',
        'namespace': nodes[9]['namespace'],
        'incomingEdges': [{'sourceNodeId': '280', 'sourceNodeOutputId': '1'}],
        'outputsMetadata': [{'id': '0', 'attrs': []}],
    })
    nodes[15]['incomingEdges'].append(
        {'sourceNodeId': 'side', 'sourceNodeOutputId': '0'}
    )
    layers = self._bound(graphs, points)
    self.assertEqual(
        layers[1]['blocks'],
        [
            {
                'block': 'Attention',
                'from': 'input',
                'to': 'intermediate.1',
                'by': 'anchors',
                'nodes': ['8', '9', '10'],
                'reason': '',
            },
            {
                'block': 'Feed Forward',
                'from': 'intermediate.1',
                'to': 'intermediate.2',
                'by': 'anchors',
                'nodes': ['11', '12'],
                'reason': '',
            },
            {
                'block': 'Per-Layer Embedding',
                'from': 'intermediate.2',
                'to': 'output',
                'by': 'anchors',
                'nodes': ['13', '14', '15'],
                'reason': '',
            },
        ],
    )
    # Layer 0 reads the signature's input, not the whole of GraphInputs.
    self.assertEqual(layers[0]['blocks'][0]['nodes'], ['0', '1', '2'])

  def test_a_block_without_its_anchors_is_its_layer(self):
    """An unbound anchor: the layer's operators, and why."""
    graphs, points = _graph(35, 1536)
    graphs['decode']['nodes'][3 * 8 + 2]['label'] = 'mul'
    layers = self._bound(graphs, points)
    attention, ffn, ple = layers[3]['blocks']
    self.assertEqual(attention['by'], 'layer')
    self.assertEqual(attention['nodes'], [str(3 * 8 + i) for i in range(8)])
    self.assertEqual(
        attention['reason'],
        'Attention runs from anchor input to anchor intermediate.1, but '
        'anchor intermediate.1 is not bound: Node 26 is not anchor '
        "intermediate.1: node 26 is mul, not add. The layer's operators "
        'are shown instead.',
    )
    self.assertEqual(ffn['by'], 'layer')
    self.assertEqual(ple['by'], 'anchors')
    # A layer the signature does not compute has no operators to show.
    graphs, points = _graph(14, 1536)
    found = architecture.Detected(
        architecture.architectures()[0], 'gemma-4-E2B-it', ''
    )
    document = found.architecture.skeleton(found.skeleton_id)
    layers = architecture.bind(
        found.architecture, document, graphs, {'decode': 'layer'}, points
    )['decode']
    self.assertEqual(
        layers[20]['blocks'][0],
        {
            'block': 'Attention',
            'from': 'input',
            'to': 'intermediate.1',
            'by': 'layer',
            'nodes': [],
            'reason': (
                'Attention runs from anchor input to anchor '
                'intermediate.1, but anchor input is not bound: decode has '
                'no operator of layer 20.'
            ),
        },
    )

  def test_each_module_outside_the_skeleton_names_its_stages(self):
    """`outside`: each module's stages and why; Gemma 4's head: decode."""
    for known in architecture.architectures():
      # The binding gives stages to the architecture's modules alone.
      self.assertEqual(
          set(known.binding.get('outside', {})),
          {m['id'] for m in known.definition.spec.get('outside', [])},
      )
      for module in known.outside():
        self.assertEqual(
            list(module),
            ['id', 'name', 'text', 'stages', 'stages_evidence'],
        )
        self.assertTrue(module['text'], module['id'])
        self.assertTrue(module['stages'], module['id'])
        self.assertLessEqual(
            set(module['stages']), {'prefill', 'decode'}, module['id']
        )
        self.assertTrue(module['stages_evidence'], module['id'])
    (gemma4,) = [a for a in architecture.architectures() if a.id == 'gemma4']
    self.assertEqual(
        {m['id']: m['stages'] for m in gemma4.outside()},
        {
            'embedding': ['prefill', 'decode'],
            'per_layer_embedding': ['prefill', 'decode'],
            'final_norm': ['decode'],
            'output_head': ['decode'],
        },
    )

  def test_the_architectures_files_are_named_by_a_digest(self):
    """identity(): the same files, the same digest."""
    self.assertRegex(architecture.identity(), '^[0-9a-f]{16}$')
    self.assertEqual(architecture.identity(), architecture.identity())


if __name__ == '__main__':
  test_main.main()
