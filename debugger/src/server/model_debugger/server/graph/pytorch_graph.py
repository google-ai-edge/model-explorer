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

"""A PyTorch model's execution graph, as the Graph Diff draws it.

The PyTorch runner exports, from a turn it observed with pytorch-collector's
collect(), the graph of the first forward pass of each signature (prefill,
decode) as the collector's to_model_explorer() draws it (the runner's
execution_graph): one node per ATen operator call, its id the call's,
grouped under its module path (`model/language_model/layers/3/self_attn`,
a module called twice in one call numbered `#0`, `#1`), with the call's
overload and module as attributes `op` and `module`; a parameter or buffer
it reads is an attribute of the node, not a node; what the pass read from
outside (its root inputs, the KV cache an earlier pass wrote) is a node in
namespace `inputs`, what it returned one in `outputs`. And the class of
each module those operators ran in, by module path (`modules`).

What the server draws of it (fold()): whatever LiteRT-LM runs as one
operator is one node, not expanded, as a LiteRT-LM graph's fused operators
are not: each call of a module whose class the architecture's PyTorch
binding names (`fold`: Gemma 4's Gemma4RMSNorm, LiteRT's odml.rms_norm;
Linear, its fully_connected; GELUTanh, its gelu) is one node, in the
namespace of the call (its module path), labelled by its class, with the
module path, the class, what LiteRT-LM runs it as and the operators it
folds as attributes, and the parameters they read. Its inputs are the
tensors its operators read from outside it, in the order they are first
read; its outputs those read outside it, each with its shape. Every other
operator stays an ATen operator (RoPE's multiplications and
concatenations, reshapes, the softmax, the matmuls, the residual adds).

Gemma 4 E2B's decode pass is 5272 nodes as the collector draws it, 2535 so
folded; its prefill pass 5307, 2570.
"""

import collections
from collections.abc import Mapping
import dataclasses
from typing import Any

FORMAT = 1
# The collector's namespaces of the nodes that are no operator.
INPUTS = 'inputs'
OUTPUTS = 'outputs'
# The attributes of a collector's operator node.
MODULE = 'module'
# The module of an operator called by the root's forward itself.
_ROOT = '<root>'
# The attributes fold() gives a node it folds.
MODULE_CLASS = 'module_class'
RUNS_AS = 'runs_as'
FOLDED = 'folded'
_PARAMETER = ('param: ', 'buffer: ')


@dataclasses.dataclass(frozen=True)
class Exported:
  """The execution graph a PyTorch runner's turn exported.

  Attributes:
      graphs: Per signature, its graph as the collector draws it.
      steps: Per signature, the step label of the pass it is of.
      modules: Each module's class, by module path.
      seconds: How long the runner took to export it.
  """

  graphs: dict[str, dict[str, Any]]
  steps: dict[str, int]
  modules: dict[str, str]
  seconds: float


def read(document: object) -> Exported:
  """The execution graph of a runner's `execution_graph.json.gz`, checked.

  Args:
      document: The file's JSON, parsed.

  Returns:
      Its graphs by signature, their steps, the modules' classes and the
      seconds the export took.

  Raises:
      ValueError: It is not what the PyTorch runner writes.
  """
  if not isinstance(document, dict) or document.get('format') != FORMAT:
    raise ValueError(
        f'The execution graph is not of format {FORMAT}, the one the '
        'server reads.'
    )
  graphs = document.get('graphs')
  steps = document.get('steps')
  modules = document.get('modules')
  if not (
      isinstance(graphs, list)
      and isinstance(steps, dict)
      and isinstance(modules, dict)
  ):
    raise ValueError(
        'The execution graph has no `graphs`, `steps` and `modules`.'
    )
  found = {}
  for graph in graphs:
    if not (
        isinstance(graph, dict)
        and isinstance(graph.get('id'), str)
        and isinstance(graph.get('nodes'), list)
    ):
      raise ValueError('A graph of the execution graph has no id or nodes.')
    if graph['id'] in found or not isinstance(steps.get(graph['id']), int):
      raise ValueError(
          f'The execution graph holds {graph["id"]} twice, or without its step.'
      )
    found[graph['id']] = {'id': graph['id'], 'nodes': graph['nodes']}
  if not all(
      isinstance(key, str) and isinstance(value, str)
      for key, value in modules.items()
  ):
    raise ValueError('The modules of the execution graph are not classes.')
  seconds = document.get('seconds', 0.0)
  return Exported(
      found,
      {name: steps[name] for name in found},
      dict(modules),
      float(seconds) if isinstance(seconds, (int, float)) else 0.0,
  )


def operators(graph: dict[str, Any]) -> list[dict[str, Any]]:
  """The nodes of a graph's operators, folded or not, in their order."""
  return [
      node
      for node in graph['nodes']
      if node.get('namespace') not in (INPUTS, OUTPUTS)
  ]


def folded_modules(graph: dict[str, Any]) -> dict[str, str]:
  """The module of each folded node of a graph, and the node's namespace.

  A module called more than once in the pass (its namespace numbered
  `#k`) is left out: which call a capture of the module holds is not
  certain.

  Args:
      graph: A signature's graph, folded (fold()).

  Returns:
      Per module path, the namespace of its node, in the graph's order.
  """
  found, twice = {}, set()
  for node in graph['nodes']:
    attrs = _attrs(node)
    if MODULE_CLASS not in attrs:
      continue
    module = attrs[MODULE]
    if '#' in node.get('namespace', '') or module in found:
      twice.add(module)
    found.setdefault(module, node.get('namespace', ''))
  return {
      module: namespace
      for module, namespace in found.items()
      if module not in twice
  }


def _attrs(node: dict[str, Any]) -> dict[str, str]:
  return {attr['key']: attr['value'] for attr in node.get('attrs', [])}


def _group(
    node: dict[str, Any],
    modules: Mapping[str, str],
    rules: Mapping[str, dict[str, Any]],
) -> str:
  """The namespace of the call a node folds into, or ''.

  The outermost module the node's operator ran in whose class a rule
  names; its call's namespace is the node's namespace up to that module,
  whose segments name the module path's, each numbered `#k` if called more
  than once. A namespace that does not follow the module path folds
  nothing.

  Args:
      node: A node of the graph as the collector draws it.
      modules: Each module's class, by module path.
      rules: Per class to fold, its rule.

  Returns:
      The namespace, or '' if the node folds into no call.
  """
  if node.get('namespace') in (INPUTS, OUTPUTS):
    return ''
  module = _attrs(node).get(MODULE, '')
  if not module or module == _ROOT:
    return ''
  parts = module.split('.')
  for depth in range(1, len(parts) + 1):
    if modules.get('.'.join(parts[:depth])) in rules:
      break
  else:
    return ''
  segments = node.get('namespace', '').split('/')
  named = [segment.split('#')[0] for segment in segments[:depth]]
  if named != parts[:depth]:
    return ''
  return '/'.join(segments[:depth])


def fold(
    graph: dict[str, Any],
    modules: Mapping[str, str],
    rules: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
  """A signature's graph, each call of a module LiteRT-LM fuses one node.

  Args:
      graph: The signature's graph as the collector draws it.
      modules: Each module's class, by module path.
      rules: Per class to fold, `{runs_as}`: what LiteRT-LM runs it as.

  Returns:
      The graph folded; `graph` itself is not changed.
  """
  nodes = graph['nodes']
  groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
  group_of = {}
  for node in nodes:
    namespace = _group(node, modules, rules)
    if namespace:
      groups[namespace].append(node)
      group_of[node['id']] = namespace
  # Per group, the tensors it reads from outside, and those read outside
  # it: (node, output) -> its port.
  inputs: dict[str, list[tuple[str, str]]] = {}
  outputs: dict[str, dict[tuple[str, str], int]] = {}
  inbound: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
  for node in nodes:
    mine = group_of.get(node['id'])
    for edge in node.get('incomingEdges', []):
      source = edge['sourceNodeId'], edge['sourceNodeOutputId']
      theirs = group_of.get(source[0])
      if theirs and theirs != mine:
        inbound[theirs].add(source)
  order = {node['id']: index for index, node in enumerate(nodes)}
  for namespace, members in groups.items():
    ids = {member['id'] for member in members}
    found = []
    for member in members:
      for edge in member.get('incomingEdges', []):
        source = edge['sourceNodeId'], edge['sourceNodeOutputId']
        if source[0] not in ids and source not in found:
          found.append(source)
    inputs[namespace] = found
    outputs[namespace] = {
        source: port
        for port, source in enumerate(
            sorted(
                inbound[namespace],
                key=lambda s: (order[s[0]], int(s[1])),
            )
        )
    }
  # Where each tensor a group's node wrote is read from now.
  moved = {}
  for namespace, ports in outputs.items():
    folded_id = groups[namespace][0]['id']
    for source, port in ports.items():
      moved[source] = (folded_id, str(port))

  def edge_from(source: tuple[str, str], target: str) -> dict[str, Any]:
    node, output = moved.get(source, source)
    return {
        'sourceNodeId': node,
        'sourceNodeOutputId': output,
        'targetNodeInputId': target,
    }

  served = []
  for node in nodes:
    namespace = group_of.get(node['id'])
    if namespace is None:
      served.append({
          **node,
          'incomingEdges': [
              edge_from(
                  (
                      incoming_edge['sourceNodeId'],
                      incoming_edge['sourceNodeOutputId'],
                  ),
                  incoming_edge['targetNodeInputId'],
              )
              for incoming_edge in node.get('incomingEdges', [])
          ],
      })
      continue
    members = groups[namespace]
    if node is not members[0]:
      continue
    served.append(
        _folded(
            namespace,
            members,
            modules,
            rules,
            [
                edge_from(source, str(index))
                for index, source in enumerate(inputs[namespace])
            ],
            outputs[namespace],
        )
    )
  return {**graph, 'nodes': served}


def _folded(
    namespace: str,
    members: list[dict[str, Any]],
    modules: Mapping[str, str],
    rules: Mapping[str, dict[str, Any]],
    incoming: list[dict[str, Any]],
    ports: Mapping[tuple[str, str], int],
) -> dict[str, Any]:
  """The one node of a call that is folded."""
  first = _attrs(members[0])
  parts = first[MODULE].split('.')[: namespace.count('/') + 1]
  module = '.'.join(parts)
  kind = modules[module]
  by_id = {member['id']: member for member in members}
  parameters = {}
  for member in members:
    for attr in member.get('attrs', []):
      if attr['value'].startswith(_PARAMETER):
        named = attr['value'].split(' ')[1]
        key = named.rsplit('.', 1)[-1]
        if parameters.get(key, attr['value']) != attr['value']:
          key = named
        parameters[key] = attr['value']
  labels = collections.Counter(member['label'] for member in members)
  outputs = []
  for (node, output), port in sorted(ports.items(), key=lambda p: p[1]):
    entry = next(
        (
            output_meta
            for output_meta in by_id[node].get('outputsMetadata', [])
            if output_meta['id'] == output
        ),
        {},
    )
    outputs.append({**entry, 'id': str(port)})
  return {
      'id': members[0]['id'],
      'label': kind,
      'namespace': namespace,
      'attrs': [
          {'key': MODULE, 'value': module},
          {'key': MODULE_CLASS, 'value': kind},
          {'key': RUNS_AS, 'value': rules[kind]['runs_as']},
          {
              'key': FOLDED,
              'value': (
                  f'{len(members)} ATen operators, not expanded: '
                  + ', '.join(
                      f'{label} x{count}' if count > 1 else label
                      for label, count in labels.items()
                  )
              ),
          },
          *({'key': key, 'value': value} for key, value in parameters.items()),
      ],
      'incomingEdges': incoming,
      'outputsMetadata': outputs,
  }


def annotate(
    graph: dict[str, Any],
    layers: list[dict[str, Any]],
    reason: str,
    modules: Mapping[tuple[str, str], tuple[str, int]] | None = None,
) -> dict[str, Any]:
  """A signature's graph, each anchor's output with its point's facts.

  As model_graph.annotate() gives a LiteRT-LM graph's points: `point_id`,
  `observable`, and when they apply `reason`, `layer` and `layer_output`,
  of the anchor bound to each output (architecture.bind()); of a tensor
  two anchors are (a layer's output, the next layer's input), the first.
  The tensor of each module of `modules` too, its point the one a Session
  that pairs it with a LiteRT-LM point, or a pair made by hand, observes
  (tensor_analysis.pytorch_side()): what a folded module's node returns
  (`out`), its one output, or what it reads (`in`), its first input; of a
  tensor two are, the first anchor's, else the first node's in the
  graph's order, `out` before `in`. Every other output is not observable,
  for `reason`.

  Args:
      graph: The signature's graph, folded (fold()).
      layers: Per layer, its anchors' bindings in the signature's graph
        (architecture.bind()'s of the signature).
      reason: Why an output no anchor is bound to has no point.
      modules: Per (module path, edge), its point id and layer (None outside the
        layers).

  Returns:
      The graph annotated; `graph` itself is not changed.
  """
  facts: dict[tuple[str, str], list[dict[str, Any]]] = {}
  for entry in layers:
    for binding in entry['anchors'].values():
      if not binding.get('bound') or not binding.get('point'):
        continue
      found = [
          ('point_id', binding['point']),
          ('observable', 'true' if binding['observable'] else 'false'),
      ]
      if binding.get('reason'):
        found.append(('reason', binding['reason']))
      found.append(('layer', str(entry['layer'])))
      if binding.get('layer_output'):
        found.append(('layer_output', 'true'))
      facts.setdefault(
          (binding['node'], binding['output']),
          [{'key': key, 'value': value} for key, value in found],
      )
  for node in graph['nodes']:
    attrs = _attrs(node)
    if MODULE_CLASS not in attrs:
      continue
    for edge in ('out', 'in'):
      found = (modules or {}).get((attrs[MODULE], edge))
      if found is None:
        continue
      if edge == 'out':
        outputs = node.get('outputsMetadata', [])
        if len(outputs) != 1:
          continue
        tensor = (node['id'], outputs[0]['id'])
      else:
        first = [
            incoming_edge
            for incoming_edge in node.get('incomingEdges', [])
            if incoming_edge['targetNodeInputId'] == '0'
        ]
        if not first:
          continue
        tensor = (
            first[0]['sourceNodeId'],
            first[0]['sourceNodeOutputId'],
        )
      point_id, layer = found
      facts.setdefault(
          tensor,
          [
              {'key': 'point_id', 'value': point_id},
              {'key': 'observable', 'value': 'true'},
              *(
                  [{'key': 'layer', 'value': str(layer)}]
                  if layer is not None
                  else []
              ),
          ],
      )
  other = [
      {'key': 'observable', 'value': 'false'},
      {'key': 'reason', 'value': reason},
  ]
  return {
      **graph,
      'nodes': [
          {
              **node,
              'outputsMetadata': [
                  {
                      **entry,
                      'attrs': (
                          entry.get('attrs', [])
                          + facts.get((node['id'], entry['id']), other)
                      ),
                  }
                  for entry in node.get('outputsMetadata', [])
              ],
          }
          for node in graph['nodes']
      ],
  }
