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

"""Tests for pytorch_graph: a PyTorch execution graph, folded and bound."""

import json
from typing import Any

from absl.testing import absltest
from model_debugger.server.architectures import catalog
from model_debugger.server.graph import architecture
from model_debugger.server.graph import model_graph
from model_debugger.server.graph import pytorch_binding
from model_debugger.server.graph import pytorch_graph
from model_debugger.server.testing import pytorch_graph_fixtures
from model_debugger.server.testing import test_main

_LAYER = 'model/language_model/layers/3'


def _pinned_config() -> dict[str, Any]:
  (gemma4,) = [a for a in catalog.architectures() if a.id == 'gemma4']
  entry = gemma4.entry('gemma-4-E2B-it')
  path = gemma4.directory / entry['pinned']['file']
  return json.loads(path.read_text(encoding='utf-8'))


def _folded(signature: str = 'decode') -> dict[str, Any]:
  exported = pytorch_graph.read(pytorch_graph_fixtures.document())
  rules = pytorch_binding.binding('gemma4')['fold']
  return pytorch_graph.fold(exported.graphs[signature], exported.modules, rules)


def _node(graph: dict[str, Any], namespace: str) -> dict[str, Any]:
  (found,) = [n for n in graph['nodes'] if n['namespace'] == namespace]
  return found


def _bound() -> dict[str, Any]:
  """E2B's skeleton bound to both folded graphs, by the PyTorch binding."""
  config = _pinned_config()
  definition, _ = catalog.generate(config)
  known = architecture.Architecture(
      definition, pytorch_binding.binding('gemma4')
  )
  document = known.skeleton('gemma-4-E2B-it')
  graphs = {name: _folded(name) for name in ('prefill', 'decode')}
  stems = model_graph.layer_stems(
      graphs, segments=True, operators_of=pytorch_graph.operators
  ).stems
  bound = pytorch_binding.anchors(config)
  points = {}
  for anchor in bound.anchors:
    for signature in bound.signatures:
      points[signature, anchor.layer, anchor.anchor] = anchor
  return architecture.bind(
      known,
      document,
      graphs,
      stems,
      (),
      {key: _Point(key, a) for key, a in points.items()},
      pytorch_graph.operators,
  )


class _Point:
  """What bind() reads of a point: its id and facts."""

  def __init__(self, key, anchor):
    self.id = pytorch_binding.point_id(key[0], anchor)
    self.selectable = True
    self.reason = ''
    self.layer_output = anchor.anchor == 'output'
    self.tensor_name = f'{anchor.module} ({anchor.edge} {anchor.path})'


class ReadTest(absltest.TestCase):
  """The runner's document, checked."""

  def test_it_is_read_with_its_steps_and_modules(self):
    """Each signature's graph, its step, each module's class."""
    found = pytorch_graph.read(pytorch_graph_fixtures.document(layers=2))
    self.assertEqual(list(found.graphs), ['prefill', 'decode'])
    self.assertEqual(found.steps, {'prefill': 13, 'decode': 14})
    self.assertEqual(
        found.modules['model.language_model.layers.1.input_layernorm'],
        'Gemma4RMSNorm',
    )

  def test_what_the_runner_does_not_write_is_refused(self):
    """Another format, or graphs without their steps."""
    document = pytorch_graph_fixtures.document(layers=1)
    with self.assertRaisesRegex(ValueError, 'format 1'):
      pytorch_graph.read({**document, 'format': 2})
    with self.assertRaisesRegex(ValueError, 'without its step'):
      pytorch_graph.read({**document, 'steps': {'prefill': 13}})
    with self.assertRaisesRegex(ValueError, 'no `graphs`'):
      pytorch_graph.read({'format': 1})


class FoldTest(absltest.TestCase):
  """Each module LiteRT-LM runs as one operator is one node."""

  def test_a_norm_is_one_node_of_its_module(self):
    """Its nine operators, one node in its module's namespace."""
    graph = _folded()
    norm = _node(graph, f'{_LAYER}/input_layernorm')
    self.assertEqual(norm['label'], 'Gemma4RMSNorm')
    attrs = model_graph.attrs(norm)
    self.assertEqual(
        attrs['module'], 'model.language_model.layers.3.input_layernorm'
    )
    self.assertEqual(attrs['runs_as'], 'odml.rms_norm')
    self.assertTrue(attrs['folded'].startswith('9 ATen operators'))
    self.assertIn('param: ', attrs['weight'])
    # One input, the layer's: what the layer before returned.
    (edge,) = norm['incomingEdges']
    before = next(n for n in graph['nodes'] if n['id'] == edge['sourceNodeId'])
    self.assertEqual(before['label'], 'aten.mul_')
    self.assertEqual(before['namespace'], 'model/language_model/layers/2')
    self.assertLen(norm['outputsMetadata'], 1)

  def test_projections_and_the_activation_fold_and_are_read_folded(self):
    """q_proj is one Linear, reading the folded norm's output."""
    graph = _folded()
    norm = _node(graph, f'{_LAYER}/input_layernorm')
    q_proj = _node(graph, f'{_LAYER}/self_attn/q_proj')
    self.assertEqual(q_proj['label'], 'Linear')
    self.assertEqual(model_graph.attrs(q_proj)['runs_as'], 'fully_connected')
    self.assertEqual(
        [
            (e['sourceNodeId'], e['sourceNodeOutputId'])
            for e in q_proj['incomingEdges']
        ],
        [(norm['id'], '0')],
    )
    act = _node(graph, f'{_LAYER}/mlp/act_fn')
    self.assertEqual(act['label'], 'GELUTanh')
    self.assertEqual(model_graph.attrs(act)['runs_as'], 'gelu')

  def test_everything_else_stays_an_aten_operator(self):
    """RoPE, attention and the residual adds are not folded."""
    graph = _folded()
    attention = [
        n['label']
        for n in graph['nodes']
        if n['namespace'] == f'{_LAYER}/self_attn'
    ]
    self.assertIn('aten.neg', attention)
    self.assertIn('aten._softmax', attention)
    self.assertEqual(attention.count('aten.bmm'), 2)
    direct = [n['label'] for n in graph['nodes'] if n['namespace'] == _LAYER]
    self.assertEqual(
        direct,
        ['aten.add', 'aten.add', 'aten.mul', 'aten.add', 'aten.mul_'],
    )

  def test_a_layer_is_its_folded_count(self):
    """Fourteen modules fold: 133 operators become 14 nodes."""
    exported = pytorch_graph.read(pytorch_graph_fixtures.document())
    raw = [
        n
        for n in exported.graphs['decode']['nodes']
        if n['namespace'].startswith(f'{_LAYER}/') or n['namespace'] == _LAYER
    ]
    graph = _folded()
    folded = [
        n
        for n in graph['nodes']
        if n['namespace'].startswith(f'{_LAYER}/') or n['namespace'] == _LAYER
    ]
    kinds = [n['label'] for n in folded if 'runs_as' in model_graph.attrs(n)]
    self.assertEqual(kinds.count('Gemma4RMSNorm'), 8)
    self.assertEqual(kinds.count('Linear'), 9)
    self.assertEqual(kinds.count('GELUTanh'), 2)
    self.assertEqual(
        len(raw) - len(folded),
        8 * (9 - 1) + 9 * (4 - 1),
    )

  def test_a_namespace_off_the_module_path_is_not_folded(self):
    """A call the collector groups elsewhere keeps its operators."""
    document = pytorch_graph_fixtures.document(layers=1)
    decode = document['graphs'][1]
    for node in decode['nodes']:
      if node['namespace'].endswith('layers/0/input_layernorm'):
        node['namespace'] = 'elsewhere/input_layernorm'
    exported = pytorch_graph.read(document)
    graph = pytorch_graph.fold(
        exported.graphs['decode'],
        exported.modules,
        pytorch_binding.binding('gemma4')['fold'],
    )
    labels = [
        n['label']
        for n in graph['nodes']
        if n['namespace'] == 'elsewhere/input_layernorm'
    ]
    self.assertLen(labels, 9)


class BindTest(absltest.TestCase):
  """E2B's skeleton bound to the folded graphs by the PyTorch binding."""

  @classmethod
  def setUpClass(cls):
    """Binds once: every test reads the same bindings."""
    super().setUpClass()
    cls.bound = _bound()

  def test_layers_are_named_by_the_module_path(self):
    """The stem `layers/`, each layer's scope `layers/<N>`."""
    graphs = {'decode': _folded()}
    stems = model_graph.layer_stems(
        graphs, segments=True, operators_of=pytorch_graph.operators
    )
    self.assertEqual(stems.stems, {'decode': 'layers/'})
    self.assertEqual(
        model_graph.operator_layers(graphs['decode'], 'layers/'),
        tuple(range(35)),
    )
    self.assertEqual(model_graph.layer_component('layers/', 3), 'layers/3')
    self.assertEqual(model_graph.layer_component('layer', 3), 'layer_3')

  def test_every_anchor_is_bound_and_checked(self):
    """Each anchor of each layer, its point the runner's of it."""
    layer = self.bound['decode'][3]
    self.assertEqual(layer['scope'], 'layers/3')
    anchors = layer['anchors']
    self.assertEqual(
        {a: b['bound'] for a, b in anchors.items()},
        dict.fromkeys(
            ['input', 'intermediate.1', 'intermediate.2', 'output'], True
        ),
    )
    for anchor in anchors.values():
      self.assertTrue(anchor['checked'], anchor)
    self.assertEqual(
        anchors['intermediate.1']['point'],
        'decode:model.language_model.layers.3.pre_feedforward_layernorm:in',
    )
    self.assertTrue(anchors['output']['layer_output'])
    # The layer's input is the layer before's output, the same tensor.
    before = self.bound['decode'][2]['anchors']['output']
    self.assertEqual(
        (anchors['input']['node'], anchors['input']['output']),
        (before['node'], before['output']),
    )
    first = self.bound['prefill'][0]['anchors']['input']
    self.assertTrue(first['bound'])
    self.assertFalse(first['checked'])

  def test_attention_runs_from_the_input_to_the_residual_add(self):
    """Its region: the norm to the add, the layer before's not."""
    decode = _folded()
    by_id = {n['id']: n for n in decode['nodes']}
    layer = self.bound['decode'][3]
    attention = next(b for b in layer['blocks'] if b['block'] == 'Attention')
    self.assertEqual(attention['by'], 'anchors')
    nodes = [by_id[i] for i in attention['nodes']]
    self.assertEqual(nodes[0]['namespace'], f'{_LAYER}/input_layernorm')
    self.assertEqual(
        nodes[-1]['id'], layer['anchors']['intermediate.1']['node']
    )
    self.assertNotIn(layer['anchors']['input']['node'], attention['nodes'])
    self.assertIn(
        _node(decode, f'{_LAYER}/self_attn/o_proj')['id'],
        attention['nodes'],
    )
    # What the attention reads from outside is not of it: the KV cache.
    self.assertFalse(
        [n for n in nodes if n['namespace'] == pytorch_graph.INPUTS]
    )
    for block in layer['blocks']:
      self.assertEqual(block['by'], 'anchors', block)

  def test_annotate_gives_anchors_their_points_and_others_why_not(self):
    """An anchor's output carries its point; any other output, why not."""
    decode = pytorch_graph.annotate(
        _folded(), self.bound['decode'], 'not an anchor'
    )
    by_id = {n['id']: n for n in decode['nodes']}
    output = self.bound['decode'][3]['anchors']['output']
    facts = model_graph.attrs(by_id[output['node']]['outputsMetadata'][0])
    self.assertEqual(
        facts['point_id'], 'decode:model.language_model.layers.3:out'
    )
    self.assertEqual(facts['layer'], '3')
    self.assertEqual(facts['layer_output'], 'true')
    norm = _node(decode, f'{_LAYER}/input_layernorm')
    facts = model_graph.attrs(norm['outputsMetadata'][0])
    self.assertNotIn('point_id', facts)
    self.assertEqual(facts['observable'], 'false')
    self.assertEqual(facts['reason'], 'not an anchor')

  def test_annotate_gives_the_operators_modules_their_points(self):
    """What a module returns, or what it reads, carries its point."""
    modules = {
        (each.module, each.edge): (
            pytorch_binding.module_point_id('decode', each.module, each.edge),
            each.layer,
        )
        for each in pytorch_binding.op_modules(_pinned_config())
    }
    decode = pytorch_graph.annotate(
        _folded(), self.bound['decode'], 'not an anchor', modules
    )
    by_id = {n['id']: n for n in decode['nodes']}
    layer = 'model.language_model.layers.3'
    q_proj = _node(decode, f'{_LAYER}/self_attn/q_proj')
    facts = model_graph.attrs(q_proj['outputsMetadata'][0])
    self.assertEqual(facts['point_id'], f'decode:{layer}.self_attn.q_proj:out')
    self.assertEqual(facts['observable'], 'true')
    self.assertEqual(facts['layer'], '3')
    down = _node(decode, f'{_LAYER}/mlp/down_proj')
    (edge,) = [
        e for e in down['incomingEdges'] if e['targetNodeInputId'] == '0'
    ]
    source = by_id[edge['sourceNodeId']]
    self.assertEqual(source['label'], 'aten.mul')
    facts = model_graph.attrs(
        next(
            o
            for o in source['outputsMetadata']
            if o['id'] == edge['sourceNodeOutputId']
        )
    )
    self.assertEqual(facts['point_id'], f'decode:{layer}.mlp.down_proj:in')
    # A KV-shared layer has no k_proj to observe.
    self.assertNotIn(
        ('model.language_model.layers.20.self_attn.k_proj', 'out'),
        modules,
    )
    self.assertIn(
        ('model.language_model.layers.14.self_attn.k_proj', 'out'),
        modules,
    )


if __name__ == '__main__':
  test_main.main()
