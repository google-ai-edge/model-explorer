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

"""How a LiteRT-LM graph binds to the architectures, and where its anchors are.

What a model is, its architecture and skeletons, is the architectures
package's (catalog); this is how a LiteRT-LM model's graph, as Model
Explorer's converter writes it, binds to one: runtime facts, kept apart
from the model's. An architecture's binding (`bindings/<id>.json`, of the
catalog's architecture `<id>`) says how a model's graph is recognized as of
the family (`detect`); per anchor of the skeleton, the scope of the
LiteRT-LM graph whose operator reads it (`read_by`) or writes it
(`written_by`), `{layer}` standing for the layer's scope component; the
operator each elementwise semantic operator is (`operators`); the signature
input an anchor with no `from` is (`signature_inputs`); why a port is bound
to nothing (`ports`); and the stages each module outside the skeleton runs
in (`outside`). Only Gemma 4 is bound now. A binding of another runtime
(`runtime`, such as the PyTorch runner's `<id>_pytorch.json`) is not a
LiteRT-LM graph's: architectures() leaves it out (pytorch_binding reads it).
bind() binds a PyTorch model's execution graph (pytorch_graph) by such a
binding's rules all the same: its anchors' `read_by` and `written_by` name
module paths, `{layer}` standing for `layers/<N>`, and a `written_by` that
names an `operator` is the one operator of that label directly in the scope,
in no module under it.

A model's skeleton (detect()) is the family's one whose layer count and
hidden width are its decode graph's. A model of no known family, or of no
known size, has none, and says why.

Binding (bind()) places each anchor of each layer in each signature's
graph: the tensor the rule names, as its node and output, and its tap point.
A binding is checked against the skeleton, not trusted: an anchor of an Add
or a Mul of the skeleton must be written by that operator in the graph,
reading what the skeleton's node reads up to the anchor before it (the
residual stream); a layer's input must be the tensor the skeleton's `from`
names (the layer before's output); and an anchor the skeleton gives no
`from` for, such as the first layer's input, must be the signature input
the architecture names for it (`signature_inputs`), if it names one. An
anchor not found, or found but failing a check, is unbound, with the
reason; so is every anchor of a layer a signature does not compute (prefill
computes only the layers whose KV cache it fills). A bound anchor says
whether it was checked (`checked`): it is not when there was nothing to check
it by, or what it was to be checked against is not bound.

A skeleton's input ports are bound only where an anchor is of them (the
layer's `input`): a port with no anchor, such as Gemma 4's
`per_layer_input`, is bound to nothing, and the architecture says why, per
port (`ports`), so that no runtime tensor is named for it that is not it.

Each block of a layer (the skeleton's anchors name the block that ends at
them, `block`) is a region of each signature's graph, found from its bound
anchors and the graph's edges alone (_region()): the operators on a path
from the tensor of the anchor the block reads (the anchor before it, `from`)
to the node of the anchor it ends at (`to`), that node included and the
first anchor's not, which belongs to the block before. What the region reads
from outside it (the anchor before, the weights and constants, the
per-layer input's slice, another layer's KV cache, the RoPE and mask tables
every layer shares) is not of it: the execution graph shows it as the
region's inputs. A block one of whose anchors is not bound, or with no path
between them, is the layer's operators instead, by the regrouped
namespaces, with the reason (`by` 'layer').

An architecture names the modules of the model that are not in its
decoder skeleton (catalog's `outside`), each with why it is not drawn; its
binding says the stages each runs in (`stages`, of 'prefill' and
'decode'), with why (`stages_evidence`): Gemma 4's final norm and output
head run only in LiteRT-LM's decode signatures, for its prefill signatures
produce no logits (Architecture.outside() has both).

The architectures' and the bindings' files are named by identity(): what is
bound by others is not reused.
"""

from collections.abc import Callable, Iterable, Mapping
import dataclasses
import hashlib
import json
import pathlib
from typing import Any

from model_debugger.server.architectures import catalog
from model_debugger.server.architectures import skeleton
from model_debugger.server.graph import model_graph
from model_debugger.server.tap import tap_scan

_DIRECTORY = pathlib.Path(__file__).parent / 'bindings'
NO_TEMPLATE = 'No architecture template exists for this model'
# The runtime whose graphs these bindings bind; a binding names its own.
LITERT_LM = 'LiteRT-LM'


@dataclasses.dataclass(frozen=True)
class Architecture:
  """A model family: what it is, and how a LiteRT-LM graph binds to it.

  Attributes:
      definition: The architecture (catalog).
      binding: Its binding's file, whole (`bindings/<id>.json`).
  """

  definition: catalog.Architecture
  binding: dict[str, Any]

  @property
  def id(self) -> str:
    """Its id, the architecture's."""
    return self.definition.id

  @property
  def name(self) -> str:
    """What the user knows it by."""
    return self.definition.name

  def skeleton(self, skeleton_id: str) -> dict[str, Any]:
    """One of its skeletons, checked (skeleton.load())."""
    return self.definition.skeleton(skeleton_id)

  def outside(self) -> list[dict[str, Any]]:
    """Each module not in the skeleton: why, and the stages it runs in.

    Returns:
        Per module the architecture names, `{id, name, text, stages,
        stages_evidence}`: the architecture's, and the binding's.
    """
    stages = self.binding.get('outside', {})
    return [
        {**module, **stages[module['id']]}
        for module in self.definition.spec.get('outside', [])
    ]


@dataclasses.dataclass(frozen=True)
class Detected:
  """A model's architecture, or why it has none.

  Attributes:
      architecture: Its family's adapter, if a family is known.
      skeleton_id: Its skeleton, if one fits it.
      text: Why it has no skeleton; '' if it has one.
  """

  architecture: Architecture | None
  skeleton_id: str | None
  text: str


def identity() -> str:
  """A digest of the architectures' files and their bindings' rules."""
  digest = hashlib.sha256(catalog.identity().encode() + b'\0')
  for path in sorted(_DIRECTORY.glob('*.json')):
    digest.update(path.name.encode() + b'\0' + path.read_bytes())
  return digest.hexdigest()[:16]


def architectures() -> list[Architecture]:
  """Every architecture a LiteRT-LM graph can be bound to.

  Returns:
      Per LiteRT-LM binding, in file name order, its architecture.

  Raises:
      ValueError: A binding is of an architecture the catalog lacks.
  """
  known = {a.id: a for a in catalog.architectures()}
  found = []
  for path in sorted(_DIRECTORY.glob('*.json')):
    binding = json.loads(path.read_text(encoding='utf-8'))
    if binding.get('runtime', LITERT_LM) != LITERT_LM:
      continue
    definition = known.get(binding['architecture'])
    if definition is None or definition.id != path.stem:
      raise ValueError(
          f'The binding {path.name} is of architecture '
          f'{binding["architecture"]!r}, which is not its name or not '
          'known.'
      )
    found.append(Architecture(definition, binding))
  return found


def detect(
    graphs: Mapping[str, dict[str, Any]],
    layer_stems: Mapping[str, str],
    points: Iterable[tap_scan.Point],
) -> Detected:
  """A model's architecture and skeleton, by its graphs and points.

  Args:
      graphs: Per signature, its graph (model_graph.signature_graphs()).
      layer_stems: Per signature, its layers' stem.
      points: The model's points, with their layers.

  Returns:
      The architecture and skeleton found; or, with no skeleton, why not.
  """
  points = list(points)
  for known in architectures():
    rule = known.binding['detect']
    signature = rule['signature']
    graph = graphs.get(signature)
    if graph is None or layer_stems.get(signature) != rule['layer_stem']:
      continue
    if rule['signature_input'] not in _input_names(graph):
      continue
    layers = {
        p.layer
        for p in points
        if p.signature == signature and p.layer is not None
    }
    widths = {
        p.shape[-1]
        for p in points
        if p.signature == signature and p.layer_output and p.shape
    }
    for entry in known.definition.spec['skeletons']:
      found = known.skeleton(entry['id'])
      width = found['semantic_graph'][0]['inputs'][0]['shape'][-1]
      if len(found['layers']) == len(layers) and widths == {width}:
        return Detected(known, entry['id'], '')
    sizes = ', '.join(
        f'{entry["id"]} ({_size(known.skeleton(entry["id"]))})'
        for entry in known.definition.spec['skeletons']
    )
    return Detected(
        known,
        None,
        f'{NO_TEMPLATE}: it is of the {known.name} family, but its '
        f'{signature} has {len(layers)} layers of width '
        f'{"/".join(str(w) for w in sorted(widths)) or "unknown"}, and '
        f'the {known.name} templates are {sizes}.',
    )
  known = ', '.join(a.name for a in architectures())
  return Detected(
      None,
      None,
      f'{NO_TEMPLATE}: its graph is of none of the architectures the '
      f'server has templates of ({known}).',
  )


def _size(document: dict[str, Any]) -> str:
  width = document['semantic_graph'][0]['inputs'][0]['shape'][-1]
  return f'{len(document["layers"])} layers of width {width}'


def _input_names(graph: dict[str, Any]) -> set[str]:
  """The signature input names of a graph's GraphInputs node."""
  names = set()
  for node in graph['nodes']:
    if node['label'] == model_graph.GRAPH_INPUTS:
      for entry in node.get('outputsMetadata', []):
        names.add(model_graph.attrs(entry).get('signature_name', ''))
  return names


def bind(
    known: Architecture,
    document: dict[str, Any],
    graphs: Mapping[str, dict[str, Any]],
    layer_stems: Mapping[str, str],
    points: Iterable[tap_scan.Point],
    anchor_points: Mapping[tuple[str, int, str], tap_scan.Point] | None = None,
    operators_of: Callable[
        [dict[str, Any]], list[dict[str, Any]]
    ] = model_graph.operators,
) -> dict[str, list[dict[str, Any]]]:
  """Where each anchor of each layer is in each signature's graph.

  Args:
      known: The model's architecture.
      document: Its skeleton.
      graphs: Per signature, its graph.
      layer_stems: Per signature, its layers' stem.
      points: The model's points, each the point of the tensor it taps.
      anchor_points: The point of each anchor, by signature, layer and anchor,
        of a runtime whose points are its anchors (the PyTorch runner's), in
        place of the point of the anchor's tensor.
      operators_of: The nodes of a graph's operators.

  Returns:
      Per signature, per layer of the skeleton, `{layer, scope, anchors,
      blocks}`, `scope` the layer's scope component in the graph
      (model_graph.layer_component(), None without a stem): per anchor,
      `{bound: true, checked, node, output, tensor_name, tensor_shape,
      point, observable, reason, layer_output}`, `checked` whether the
      binding was checked against the skeleton, `point` the tap point's id
      (None for a signature input, which is no operator's output); or
      `{bound: false, reason}`. Per block, in the skeleton's order,
      `{block, from, to, by, nodes, reason}` (_region()).
  """
  by_id = {p.id: p for p in points}
  rules = known.binding['anchors']
  found = {}
  for signature, graph in graphs.items():
    stem = layer_stems.get(signature)
    nodes = {node['id']: node for node in graph['nodes']}
    ops = operators_of(graph)
    layers = []
    for index, layer in enumerate(document['layers']):
      definition = document['semantic_graph'][layer['def']]
      component = model_graph.layer_component(stem, index) if stem else None
      present = component is not None and any(
          f'/{component}/' in f'/{node.get("namespace", "")}/' for node in ops
      )
      anchors = {}
      for anchor in definition['anchors']:
        if not present:
          anchors[anchor['id']] = _unbound(
              f'{signature} has no operator of layer {index}.'
          )
          continue
        rule = rules.get(anchor['id'])
        if rule is None:
          anchors[anchor['id']] = _unbound(
              f'{known.name} has no rule for anchor {anchor["id"]}.'
          )
          continue
        anchors[anchor['id']] = _find(rule, component, signature, ops, nodes)
      layers.append({'layer': index, 'anchors': anchors})
    _check(known, document, layers, nodes, signature)
    edges = _Edges(graph)
    found[signature] = []
    for entry in layers:
      layer = document['layers'][entry['layer']]
      definition = document['semantic_graph'][layer['def']]
      blocks = _blocks(definition)
      component = (
          model_graph.layer_component(stem, entry['layer']) if stem else None
      )
      found[signature].append({
          'layer': entry['layer'],
          'scope': component,
          'anchors': {
              anchor_id: _described(
                  binding,
                  signature,
                  nodes,
                  by_id,
                  anchor_points,
                  (signature, entry['layer'], anchor_id),
              )
              for anchor_id, binding in entry['anchors'].items()
          },
          'blocks': [
              _region(block, entry['anchors'], edges, component, ops)
              for block in blocks
          ],
      })
  return found


@dataclasses.dataclass(frozen=True)
class _Block:
  """A block of a layer's skeleton and the anchors that bound it.

  Attributes:
      name: The block, as the skeleton's anchors and namespaces name it.
      start: The anchor whose tensor it reads; None if it reads no one anchor's.
      end: The anchor it ends at.
      reason: Why it has no start; '' if it has one.
  """

  name: str
  start: str | None
  end: str
  reason: str


def _blocks(definition: dict[str, Any]) -> list[_Block]:
  """Each block of a layer's definition: the anchors it runs between.

  A block ends at the anchor that names it (`block`) and starts at the
  anchor whose tensor its stages read from outside it.

  Args:
      definition: The layer's definition, of the skeleton's semantic graph.

  Returns:
      Each block, in the order of the anchors that end them.
  """
  tensors = {}
  for anchor in definition['anchors']:
    of = anchor['of'] if ':' in anchor['of'] else anchor['of'] + ':0'
    tensors[of] = anchor['id']
  found = []
  for anchor in definition['anchors']:
    name = anchor.get('block')
    if not name:
      continue
    stages = [n for n in definition['nodes'] if n['namespace'] == name]
    inside = {n['id'] for n in stages}
    outside = set()
    for stage in stages:
      for t in skeleton.sources(stage):
        if t.partition(':')[0] not in inside and t in tensors:
          outside.add(tensors[t])
    read = sorted(outside)
    if len(read) == 1:
      found.append(_Block(name, read[0], anchor['id'], ''))
    else:
      found.append(
          _Block(
              name,
              None,
              anchor['id'],
              f'Its stages read {len(read)} anchors from outside it, not one.',
          )
      )
  return found


class _Edges:
  """Who reads what in a graph: per tensor and per node, its readers."""

  def __init__(self, graph: dict[str, Any]):
    self.nodes = {node['id']: node for node in graph['nodes']}
    self.order = {node['id']: i for i, node in enumerate(graph['nodes'])}
    self.readers: dict[tuple[str, str], list[str]] = {}
    self.consumers: dict[str, set[str]] = {}
    for node in graph['nodes']:
      for edge in node.get('incomingEdges', []):
        source = edge['sourceNodeId'], edge['sourceNodeOutputId']
        self.readers.setdefault(source, []).append(node['id'])
        self.consumers.setdefault(source[0], set()).add(node['id'])


def _region(
    block: _Block,
    anchors: Mapping[str, dict[str, Any]],
    edges: _Edges,
    component: str | None,
    ops: list[dict[str, Any]],
) -> dict[str, Any]:
  """A block's operators in a signature's graph, and how they were found.

  By its anchors (`by` 'anchors'): every operator on a path from the
  start anchor's tensor to the end anchor's node, that node included, the
  start's not. A block whose anchors are not both bound, or have no path
  between them, is its layer's operators by the regrouped namespaces
  (`by` 'layer'), and says why in `reason`.

  Args:
      block: The block.
      anchors: The layer's anchors' bindings, by anchor id.
      edges: Who reads what in the signature's graph.
      component: The layer's scope component; None without a stem.
      ops: The nodes of the graph's operators.

  Returns:
      `{block, from, to, by, nodes, reason}`, `nodes` in graph order.
  """
  found = {'block': block.name, 'from': block.start, 'to': block.end}
  start = anchors.get(block.start) if block.start else None
  end = anchors.get(block.end)
  why = ''
  if block.start is None:
    why = block.reason
  else:
    for anchor_id, binding in ((block.start, start), (block.end, end)):
      if not binding or not binding.get('bound'):
        why = (
            f'anchor {anchor_id} is not bound: '
            f'{(binding or {}).get("reason", "")}'
        )
        break
  if not why:
    nodes = _between((start['node'], start['output']), end['node'], edges)
    if nodes:
      return {**found, 'by': 'anchors', 'nodes': nodes, 'reason': ''}
    why = (
        f'no operator path runs from anchor {block.start} (node '
        f'{start["node"]}:{start["output"]}) to anchor {block.end} '
        f'(node {end["node"]}).'
    )
  layer = (
      [node['id'] for node in ops if _in_scope(node, component)]
      if component
      else []
  )
  return {
      **found,
      'by': 'layer',
      'nodes': layer,
      'reason': (
          f'{block.name} runs from anchor {block.start} to anchor '
          f'{block.end}, but {why}'
          + (" The layer's operators are shown instead." if layer else '')
      ),
  }


def _between(start: tuple[str, str], end: str, edges: _Edges) -> list[str]:
  """The nodes on a path from a tensor to a node, the node included.

  Forward from the tensor's readers, never past the end node; then back
  from the end node, within what was reached.

  Args:
      start: The tensor, as (node id, output id).
      end: The end node's id.
      edges: Who reads what in the graph.

  Returns:
      The nodes' ids in graph order; empty if no path reaches the end node.
  """
  reached = set()
  stack = list(edges.readers.get(start, []))
  while stack:
    node = stack.pop()
    if node in reached:
      continue
    reached.add(node)
    if node != end:
      stack.extend(edges.consumers.get(node, ()))
  if end not in reached:
    return []
  region = set()
  stack = [end]
  while stack:
    node = stack.pop()
    if node in region or node not in reached:
      continue
    region.add(node)
    stack.extend(
        edge['sourceNodeId']
        for edge in edges.nodes[node].get('incomingEdges', [])
    )
  return sorted(region, key=edges.order.__getitem__)


def _unbound(reason: str) -> dict[str, Any]:
  return {'bound': False, 'reason': reason}


def _in_scope(node: dict[str, Any], scope: str) -> bool:
  return f'/{scope}/' in f'/{node.get("namespace", "")}/'


def _find(
    rule: dict[str, Any],
    component: str,
    signature: str,
    ops: list[dict[str, Any]],
    nodes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
  """The tensor an anchor's rule names in one layer: (node, output)."""
  if 'read_by' in rule:
    scope = rule['read_by'].format(layer=component)
    inside = [node for node in ops if _in_scope(node, scope)]
    if not inside:
      return _unbound(f'{signature} has no operator in scope {scope}.')
    ids = {node['id'] for node in inside}
    read = [
        edge
        for edge in inside[0]['incomingEdges']
        if edge['sourceNodeId'] not in ids
        and nodes[edge['sourceNodeId']]['label'] != model_graph.CONSTANT
    ]
    if len(read) != 1:
      return _unbound(
          f'The first operator of {scope} in {signature} (node '
          f'{inside[0]["id"]}) reads {len(read)} tensors from outside '
          'it, not one.'
      )
    return {
        'bound': True,
        'node': read[0]['sourceNodeId'],
        'output': read[0]['sourceNodeOutputId'],
    }
  scope = rule['written_by'].format(layer=component)
  inside = [node for node in ops if _in_scope(node, scope)]
  if not inside:
    return _unbound(f'{signature} has no operator in scope {scope}.')
  if 'operator' in rule:
    inside = [
        node
        for node in inside
        if node['label'] == rule['operator']
        and f'/{node.get("namespace", "")}'.endswith(f'/{scope}')
    ]
    if len(inside) != 1:
      return _unbound(
          f'{signature} has {len(inside)} {rule["operator"]} operators '
          f'directly in scope {scope}, not one.'
      )
  if len(inside) != 1:
    return _unbound(
        f'{signature} has {len(inside)} operators in scope {scope}, not one.'
    )
  return {'bound': True, 'node': inside[0]['id'], 'output': '0'}


def _check(
    known: Architecture,
    document: dict[str, Any],
    layers: list[dict[str, Any]],
    nodes: dict[str, dict[str, Any]],
    signature: str,
) -> None:
  """Checks each bound anchor against the skeleton's structure, in place.

  One that breaks it is unbound; every other gets `checked`.

  Args:
      known: The model's architecture.
      document: Its skeleton.
      layers: Per layer, `{layer, anchors}`, its anchors' bindings; changed.
      nodes: The signature's graph's nodes, by id.
      signature: The signature's key.
  """
  names = known.binding['operators']
  inputs = known.binding.get('signature_inputs', {})
  for index, entry in enumerate(layers):
    layer = document['layers'][index]
    definition = document['semantic_graph'][layer['def']]
    anchors = entry['anchors']
    sources = layer.get('from') or {}
    for anchor in definition['anchors']:
      anchor_id = anchor['id']
      binding = anchors.get(anchor_id)
      if not binding or not binding['bound']:
        continue
      path = _residual_path(definition, anchor)
      if path is not None:
        steps, before = path
        reason, unchecked = _walk(
            steps, before, binding, anchors.get(before), names, nodes
        )
      elif 'anchor' in sources.get(anchor_id, {}):
        reason, unchecked = _from(binding, sources[anchor_id], layers)
      elif anchor_id in inputs:
        reason, unchecked = _signature_input(
            binding, inputs[anchor_id], signature, nodes
        )
      else:
        reason, unchecked = (
            '',
            'The skeleton gives nothing to check it by.',
        )
      if reason:
        anchors[anchor_id] = _unbound(
            f'Node {binding["node"]} is not anchor {anchor_id}: {reason}'
        )
      else:
        binding['checked'] = not unchecked


def _from(
    binding: dict[str, Any],
    source: dict[str, Any],
    layers: list[dict[str, Any]],
) -> tuple[str, str]:
  """Whether an anchor is the one the skeleton's `from` names it to be.

  Args:
      binding: The anchor's binding.
      source: The skeleton's `from` of it: `{layer, anchor}`.
      layers: Per layer, `{layer, anchors}`, its anchors' bindings.

  Returns:
      Why it is not, or ''; and why it was not checked, or ''.
  """
  theirs = layers[source['layer']]['anchors'].get(source['anchor'])
  named = f"layer {source['layer']}'s {source['anchor']}"
  if not theirs or not theirs.get('bound'):
    return '', f'The skeleton has it be {named}, which is not bound.'
  if (binding['node'], binding['output']) != (
      theirs['node'],
      theirs['output'],
  ):
    return (
        f'the skeleton has it be {named}, node {theirs["node"]}.',
        '',
    )
  return '', ''


def _signature_input(
    binding: dict[str, Any],
    name: str,
    signature: str,
    nodes: dict[str, dict[str, Any]],
) -> tuple[str, str]:
  """Whether an anchor is the signature input the architecture names.

  Args:
      binding: The anchor's binding.
      name: The signature input the architecture names for it.
      signature: The signature's key.
      nodes: The signature's graph's nodes, by id.

  Returns:
      Why it is not, or ''; and why it was not checked, or ''.
  """
  node = nodes[binding['node']]
  if node['label'] != model_graph.GRAPH_INPUTS:
    return (
        f"it is node {node['id']}'s output, not {signature}'s input {name}.",
        '',
    )
  entry = next(
      (
          e
          for e in node.get('outputsMetadata', [])
          if e['id'] == binding['output']
      ),
      {},
  )
  found = model_graph.attrs(entry).get('signature_name', '')
  if found != name:
    return f"it is {signature}'s input {found}, not {name}.", ''
  return '', ''


def _residual_path(
    definition: dict[str, Any], anchor: dict[str, Any]
) -> tuple[list[str], str] | None:
  """The semantic operators from an anchor back to the anchor before it.

  Followed through elementwise nodes only, from the anchor's node to the
  first that reads another anchor's tensor.

  Args:
      definition: The layer's definition, of the skeleton's semantic graph.
      anchor: The anchor.

  Returns:
      The operators, the anchor's own first, and that other anchor's id;
      None when the anchor's node is not elementwise.
  """
  by_id = {node['id']: node for node in definition['nodes']}
  tensors = {}
  for other in definition['anchors']:
    of = other['of'] if ':' in other['of'] else other['of'] + ':0'
    tensors[of] = other['id']
  node = by_id.get(anchor['of'].partition(':')[0])
  steps = []
  while node is not None:
    if skeleton.attr(node, 'category') != 'elementwise':
      return None
    steps.append(node['ops'][0]['op'])
    read = skeleton.sources(node)
    before = [tensors[t] for t in read if t in tensors]
    if before:
      return steps, before[0]
    elementwise = [
        by_id[t.partition(':')[0]]
        for t in read
        if t.partition(':')[0] in by_id
        and skeleton.attr(by_id[t.partition(':')[0]], 'category')
        == 'elementwise'
    ]
    node = elementwise[0] if len(elementwise) == 1 else None
  return None


def _walk(
    steps: list[str],
    before: str,
    binding: dict[str, Any],
    previous: dict[str, Any] | None,
    names: Mapping[str, str],
    nodes: dict[str, dict[str, Any]],
) -> tuple[str, str]:
  """Whether a bound anchor's graph runs the skeleton's residual steps.

  Args:
      steps: The semantic operators from the anchor back (_residual_path()).
      before: The anchor before it, which the last step must read.
      binding: The anchor's binding.
      previous: The binding of the anchor before it, if any.
      names: The graph's operator of each semantic operator.
      nodes: The signature's graph's nodes, by id.

  Returns:
      Why it does not, or ''; and why it was not wholly checked, or ''.
  """
  node = nodes[binding['node']]
  for i, step in enumerate(steps):
    want = names.get(step)
    if node['label'] != want:
      return f'node {node["id"]} is {node["label"]}, not {want}.', ''
    read = [
        (edge['sourceNodeId'], edge['sourceNodeOutputId'])
        for edge in node['incomingEdges']
    ]
    if i == len(steps) - 1:
      if not previous or not previous.get('bound'):
        return '', (
            f"Its operators are the skeleton's, but {before}, which "
            'it must read, is not bound.'
        )
      if (previous['node'], previous['output']) not in read:
        return (
            (
                f'node {node["id"]} does not read {before} (node '
                f'{previous["node"]}).'
            ),
            '',
        )
      return '', ''
    following = names.get(steps[i + 1])
    producers = [
        nodes[source]
        for source, _ in read
        if nodes[source]['label'] == following
    ]
    if len(producers) != 1:
      return (
          (
              f'node {node["id"]} reads {len(producers)} {following} '
              'operators, not one.'
          ),
          '',
      )
    node = producers[0]
  return '', ''


def _described(
    binding: dict[str, Any],
    signature: str,
    nodes: dict[str, dict[str, Any]],
    points: Mapping[str, tap_scan.Point],
    anchor_points: Mapping[tuple[str, int, str], tap_scan.Point] | None,
    anchor: tuple[str, int, str],
) -> dict[str, Any]:
  """A binding with its tensor's name and shape, and its tap point.

  The point is that of the anchor's tensor, or, given `anchor_points` (a
  runtime whose points are its anchors), the anchor's own.

  Args:
      binding: The anchor's binding.
      signature: The signature's key.
      nodes: The signature's graph's nodes, by id.
      points: The model's points, by id.
      anchor_points: The point of each anchor, by signature, layer and anchor;
        None to use the point of the anchor's tensor.
      anchor: The anchor's (signature, layer, anchor id).

  Returns:
      `binding` itself if unbound; else `{bound, checked, node, output,
      tensor_name, tensor_shape, point, observable, reason, layer_output}`.
  """
  if not binding['bound']:
    return binding
  node = nodes[binding['node']]
  entry = next(
      (
          e
          for e in node.get('outputsMetadata', [])
          if e['id'] == binding['output']
      ),
      {},
  )
  found = model_graph.attrs(entry)
  described = {
      'bound': True,
      'checked': binding['checked'],
      'node': binding['node'],
      'output': binding['output'],
      'tensor_name': found.get('tensor_name', ''),
      'tensor_shape': found.get('tensor_shape', ''),
  }
  if anchor_points is not None:
    point = anchor_points.get(anchor)
    if point is None:
      return {
          **described,
          'point': None,
          'observable': False,
          'reason': 'The runtime has no point of this anchor.',
          'layer_output': False,
      }
    return {
        **described,
        'tensor_name': described['tensor_name'] or point.tensor_name,
        'point': point.id,
        'observable': point.selectable,
        'reason': point.reason,
        'layer_output': point.layer_output,
    }
  if node['label'] == model_graph.GRAPH_INPUTS:
    name = found.get('signature_name', '')
    return {
        **described,
        'point': None,
        'observable': False,
        'reason': (
            f"It is the input {name} of {signature}, no operator's "
            'output; it cannot be observed.'
        ),
        'layer_output': False,
    }
  point = points.get(
      tap_scan.point_id(signature, int(binding['node']), int(binding['output']))
  )
  if point is None:
    return {
        **described,
        'point': None,
        'observable': False,
        'reason': 'The model has no tap point of this output.',
        'layer_output': False,
    }
  return {
      **described,
      'point': point.id,
      'observable': point.selectable,
      'reason': point.reason,
      'layer_output': point.layer_output,
  }
