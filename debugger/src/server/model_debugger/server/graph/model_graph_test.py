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

"""Tests of the signatures' graphs: kept, slimmed, layered, annotated."""

import dataclasses
import itertools
from typing import Any

from absl.testing import absltest
from model_debugger.server.graph import model_graph
from model_debugger.server.tap import tap_scan
from model_debugger.server.testing import fake_converter
from model_debugger.server.testing import tap_fixtures
from model_debugger.server.testing import test_main

_SUBGRAPHS = {name: i for i, name in enumerate(tap_fixtures.SIGNATURES)}


def _node(node_id: str, namespace: str, label: str = 'add') -> dict[str, Any]:
  return {'id': node_id, 'label': label, 'namespace': namespace}


class ModelGraphTest(absltest.TestCase):
  """Tests of model_graph on the fixture model and namespaces of real ones."""

  def setUp(self):
    """The fixture's text model, as the converter would write it."""
    super().setUp()
    self.converted = fake_converter.converted(tap_fixtures.text_model())
    self.graphs = model_graph.signature_graphs(self.converted, _SUBGRAPHS)

  def test_keeps_each_signatures_graph_by_its_key(self):
    """One graph per signature; the node of op i is node i."""
    self.assertEqual(list(self.graphs), list(tap_fixtures.SIGNATURES))
    decode = self.graphs['decode']
    self.assertEqual(decode['id'], 'decode')
    self.assertEqual(
        [n['id'] for n in model_graph.operators(decode)],
        ['0', '1', '2', '3'],
    )
    self.assertEqual(
        model_graph.signature_graphs({'graphs': self.converted}, {'decode': 0}),
        {'decode': decode},
    )
    with self.assertRaisesRegex(ValueError, 'no subgraph 9'):
      model_graph.signature_graphs(self.converted, {'decode': 9})
    with self.assertRaisesRegex(ValueError, '2 graphs'):
      model_graph.signature_graphs(self.converted * 2, _SUBGRAPHS)

  def test_drops_long_text_and_composite_subgraphs(self):
    """Long constants and quantization become a note; no expansion."""
    converted = fake_converter.converted(tap_fixtures.text_model())
    node = converted[0]['subgraphs'][0]['nodes'][0]
    node['subgraphIds'] = ['XlaCallModule_odml.rms_norm.impl_0']
    node['attrs'] = [{'key': 'name', 'value': 'odml.rms_norm'}]
    graph = model_graph.signature_graphs(converted, {'decode': 0})['decode']
    first = graph['nodes'][0]
    self.assertEqual(first['subgraphIds'], [])
    self.assertEqual(model_graph.attrs(first)['name'], 'odml.rms_norm')
    self.assertEqual(
        model_graph.attrs(first['outputsMetadata'][0])['quantization'],
        '(100 characters, not kept)',
    )
    constant = next(
        n for n in graph['nodes'] if n['label'] == model_graph.CONSTANT
    )
    self.assertEqual(
        model_graph.attrs(constant)['__value'],
        '(100 characters, not kept)',
    )
    # The converter's output itself is not changed.
    self.assertEqual(
        node['subgraphIds'], ['XlaCallModule_odml.rms_norm.impl_0']
    )

  def test_names_the_stem_of_a_run_of_layers(self):
    """The component numbered 0, 1, ... that holds most operators."""
    gemma = 'LanguageModel.decode_graph/transformer.transformer'
    qwen = 'Export/m.Qwen3Model_model'
    graphs = {
        'gemma': {
            'nodes': [
                _node('0', 'broadcast_in_dim1'),
                *(
                    _node(str(i + 1), f'{gemma}/layer_{i}/layer_{i}.x/add1')
                    for i in range(3)
                ),
                _node('9', 'select_n1'),
                _node('10', 'GraphInputs', model_graph.GRAPH_INPUTS),
            ]
        },
        'qwen': {
            'nodes': [
                _node(str(i), f'{qwen}/m.Qwen3DecoderLayer_{i}/m.Attn_a')
                for i in range(4)
            ]
        },
        # 1 and 2 without 0: no run.
        'gaps': {'nodes': [_node('0', 'a/b_1'), _node('1', 'a/b_2')]},
        'none': {'nodes': [_node('0', 'a/b')]},
    }
    found = model_graph.layer_stems(graphs)
    # Qwen3 names its layers otherwise: no signature has layers then.
    self.assertEqual(found.stems, {})
    self.assertEqual(
        found.reasons['qwen'],
        'The signatures name their layers differently: gemma layer, '
        'qwen m.Qwen3DecoderLayer.',
    )
    self.assertEqual(found.reasons['gemma'], found.reasons['qwen'])
    self.assertEqual(
        found.reasons['gaps'],
        'No scope of it is numbered 0, 1, ... as layers are.',
    )
    self.assertEqual(found.reasons['none'], found.reasons['gaps'])
    del graphs['qwen']
    found = model_graph.layer_stems(graphs)
    self.assertEqual(found.stems, {'gemma': 'layer'})
    del graphs['gemma']
    found = model_graph.layer_stems(graphs)
    self.assertEqual(found.stems, {})
    graphs['qwen'] = {
        'nodes': [
            _node(str(i), f'{qwen}/m.Qwen3DecoderLayer_{i}/m.Attn_a')
            for i in range(4)
        ]
    }
    found = model_graph.layer_stems(graphs)
    self.assertEqual(found.stems, {'qwen': 'm.Qwen3DecoderLayer'})
    found = model_graph.layer_stems(self.graphs)
    self.assertEqual(
        found.stems, {name: 'layer' for name in tap_fixtures.SIGNATURES}
    )
    self.assertEqual(found.reasons, {})

  def test_the_layers_are_the_repeated_component_not_a_part_of_it(self):
    """64 experts in a layer are not its layers; two runs are ambiguous."""
    stack = 'Model/stack'
    experts = [
        _node(str(i), f'{stack}/layer_{layer}/moe/expert_{e}/mul')
        for i, (layer, e) in enumerate(itertools.product(range(4), range(64)))
    ]
    layers = [
        _node(str(1000 + i), f'{stack}/layer_{i}/attn/add') for i in range(4)
    ]
    found = model_graph.layer_stems({'decode': {'nodes': experts + layers}})
    self.assertEqual(found.stems, {'decode': 'layer'})
    # Experts alone, under one parent: 64 of them hold a quarter of the
    # operators, not the most.
    flat = [_node(str(i), f'{stack}/expert_{i}/mul') for i in range(64)] + [
        _node(str(100 + i), f'{stack}/dense/add{i}') for i in range(192)
    ]
    found = model_graph.layer_stems({'decode': {'nodes': flat}})
    self.assertEqual(found.stems, {})
    self.assertEqual(
        found.reasons['decode'],
        'No numbered scope of it holds 50% of its operators, as layers '
        'do; the most is expert_0..63 under Model/stack, 25% of its '
        'operators.',
    )
    # Two runs that each hold most operators: which is the layers?
    both = [_node(str(i), f'{stack}/block_0/layer_{i}/add') for i in range(4)]
    found = model_graph.layer_stems({'decode': {'nodes': both}})
    self.assertEqual(found.stems, {})
    self.assertEqual(
        found.reasons['decode'],
        'Its layers are ambiguous: layer_0..3 under Model/stack/block_0, '
        '100% of its operators; block_0..0 under Model/stack, 100% of its '
        'operators.',
    )

  def test_a_signature_computes_the_layers_it_has_operators_of(self):
    """operator_layers(): a prefill of 2 layers of 4 lists 0 and 1."""
    stack = 'Model/stack'
    prefill = {
        'nodes': [
            _node('0', 'GraphInputs', model_graph.GRAPH_INPUTS),
            _node('1', f'{stack}/embed/gather'),
            *(
                _node(str(2 + i), f'{stack}/layer_{i}/layer_{i}.x/add')
                for i in (1, 0, 1)
            ),
        ]
    }
    self.assertEqual(model_graph.operator_layers(prefill, 'layer'), (0, 1))
    self.assertEqual(model_graph.operator_layers(prefill, None), ())
    self.assertEqual(
        model_graph.operator_layers(self.graphs['decode'], 'layer'), (0,)
    )

  def test_the_slimming_is_named_by_a_digest_of_its_code(self):
    """slimming_digest(): the same code, the same digest."""
    self.assertRegex(model_graph.slimming_digest(), '^[0-9a-f]{16}$')
    self.assertEqual(
        model_graph.slimming_digest(), model_graph.slimming_digest()
    )

  def test_regroups_each_operator_in_its_own_layer(self):
    """As its outputs' names say, but for one several layers share."""
    scope = 'LanguageModel.decode_graph/transformer'

    def op(node_id, namespace, name, reads=()):
      return {
          'id': node_id,
          'label': 'mul',
          'namespace': f'{scope}/{namespace}',
          'attrs': [],
          'incomingEdges': [
              {
                  'sourceNodeId': source,
                  'sourceNodeOutputId': '0',
                  'targetNodeInputId': str(slot),
              }
              for slot, source in enumerate(reads)
          ],
          'outputsMetadata': [{
              'id': '0',
              'attrs': [{'key': 'tensor_name', 'value': f'{scope}/{name}'}],
          }],
      }

    graph = {
        'id': 'decode',
        'nodes': [
            # The cosine table: layer 0's, read by layers 0 and 1.
            op('0', 'layer_0/layer_0.pre_qkv/cos', 'layer_0/x/cos'),
            op('1', 'layer_0/layer_0.pre_qkv/rope', 'layer_0/x/rope', ['0']),
            # Layer 1's RoPE, which the converter put in layer 0.
            op(
                '2',
                'layer_0/layer_0.pre_qkv/rope/mul1',
                'layer_1/layer_1.pre_qkv/rope/mul;'
                f'{scope}/layer_0/layer_0.pre_qkv/rope/mul',
                ['0'],
            ),
            op('3', 'layer_1/layer_1.attn/q', 'layer_1/attn/q', ['2']),
            # Named by no layer: where the converter put it.
            op('4', 'layer_1/cache', 'StatefulPartitionedCall:1', ['3']),
            # Layer 2's, in layer 1, and read by layers 1 and 2: shared.
            op('5', 'layer_1/shared', 'layer_2/shared', ['3']),
            op('6', 'layer_1/use', 'layer_1/use', ['5']),
            op('7', 'layer_2/use', 'layer_2/use', ['5']),
            {
                'id': '8',
                'label': model_graph.GRAPH_OUTPUTS,
                'namespace': model_graph.GRAPH_OUTPUTS,
                'incomingEdges': [{
                    'sourceNodeId': '7',
                    'sourceNodeOutputId': '0',
                    'targetNodeInputId': '0',
                }],
            },
        ],
    }
    found = model_graph.regroup(graph, 'layer')
    namespaces = {
        node['id']: node['namespace'].removeprefix(scope + '/')
        for node in found['nodes']
    }
    self.assertEqual(namespaces['2'], 'layer_1/layer_1.pre_qkv/rope/mul1')
    self.assertEqual(
        model_graph.attrs(found['nodes'][2])[model_graph.CONVERTER_NAMESPACE],
        f'{scope}/layer_0/layer_0.pre_qkv/rope/mul1',
    )
    moved = [
        node['id']
        for node in found['nodes']
        if model_graph.CONVERTER_NAMESPACE in model_graph.attrs(node)
    ]
    self.assertEqual(moved, ['2'])
    self.assertEqual(namespaces['4'], 'layer_1/cache')
    self.assertEqual(namespaces['5'], 'layer_1/shared')
    self.assertEqual(model_graph.own_layer(graph['nodes'][2], 'layer'), 1)
    self.assertIsNone(model_graph.own_layer(graph['nodes'][4], 'layer'))
    self.assertEqual(
        model_graph.namespace_layer(graph['nodes'][2]['namespace'], 'layer'),
        0,
    )
    # The graph itself is not changed; without a stem nothing moves.
    self.assertEqual(
        graph['nodes'][2]['namespace'],
        f'{scope}/layer_0/layer_0.pre_qkv/rope/mul1',
    )
    self.assertIs(model_graph.regroup(graph, None), graph)

  def test_groups_the_operators_whose_names_name_no_layer(self):
    """An output named by no layer is in the layer its op is grouped in."""
    scope = 'LanguageModel.decode_graph/transformer'

    def op(node_id, namespace, *names):
      return {
          'id': node_id,
          'label': 'add',
          'namespace': namespace,
          'outputsMetadata': [
              {
                  'id': str(slot),
                  'attrs': [{'key': 'tensor_name', 'value': name}],
              }
              for slot, name in enumerate(names)
          ],
      }

    graph = {
        'id': 'decode',
        'nodes': [
            op('0', f'{scope}/layer_0/q', f'{scope}/layer_0/q'),
            # Layer 0's KV cache write, which the converter grouped by
            # the operators it reads.
            op(
                '1',
                f'{scope}/layer_0/StatefulPartitionedCall:1',
                'StatefulPartitionedCall:1',
                'StatefulPartitionedCall:16',
            ),
            # One output named by layer 1, the other by none.
            op(
                '2',
                f'{scope}/layer_1/kv',
                f'{scope}/layer_1/kv',
                'StatefulPartitionedCall:2',
            ),
            # Named by none, and grouped in none.
            op(
                '3',
                'StatefulPartitionedCall:3',
                'StatefulPartitionedCall:3',
            ),
            _node('4', model_graph.GRAPH_OUTPUTS, model_graph.GRAPH_OUTPUTS),
        ],
    }
    self.assertEqual(model_graph.layer_groups(graph, 'layer'), {1: 0, 2: 1})
    self.assertEqual(model_graph.layer_groups(graph, None), {})

  def test_annotates_each_output_with_its_point(self):
    """Point id, observable, reason, layer and context axes, as text."""
    scan = tap_scan.read_model(
        tap_fixtures.text_model(),
        {name: 'layer' for name in tap_fixtures.SIGNATURES},
    )
    points = scan[1]
    graph = model_graph.annotate(self.graphs['decode'], points, 'decode')
    facts = {
        node['id']: model_graph.attrs(node['outputsMetadata'][0])
        for node in model_graph.operators(graph)
    }
    self.assertEqual(facts['0']['point_id'], 'decode:0:0')
    self.assertEqual(facts['0']['observable'], 'true')
    self.assertEqual(facts['0']['layer'], '0')
    self.assertNotIn('reason', facts['0'])
    self.assertEqual(facts['2']['context_axes'], '2')
    self.assertEqual(facts['3']['observable'], 'false')
    self.assertEqual(facts['3']['reason'], 'Its type, string, is not captured.')
    # The graph given is not changed; another signature's are ignored.
    self.assertNotIn(
        'point_id',
        model_graph.attrs(
            self.graphs['decode']['nodes'][0]['outputsMetadata'][0]
        ),
    )
    moved = [
        dataclasses.replace(p, tensor=p.tensor + 1)
        if p.id == 'decode:1:0'
        else p
        for p in points
    ]
    with self.assertRaisesRegex(ValueError, 'not the model'):
      model_graph.annotate(self.graphs['decode'], moved, 'decode')

  def test_a_graph_is_a_collection_of_its_own(self):
    """Model Explorer's collection: a label and the one graph."""
    found = model_graph.collection(
        self.graphs['decode'], 'model · decode', {'runs': 'always'}
    )
    self.assertEqual(found['label'], 'model · decode')
    self.assertEqual(found['graphs'][0]['id'], 'decode')
    self.assertEqual(
        found['graphs'][0]['groupNodeAttributes'], {'': {'runs': 'always'}}
    )


if __name__ == '__main__':
  test_main.main()
