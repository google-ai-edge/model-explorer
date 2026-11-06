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

"""The graphs of a model's text signatures, as Model Explorer draws them.

Model Explorer's converter (graph_converter) turns the prefill and decode
section of a .litertlm model into Model Explorer's graph JSON: one graph of
the section, one subgraph per TFLite subgraph, each a list of nodes. A node
of an operator has the operator's index in its subgraph as its id, and an
output's metadata, by the output's slot, carries the tensor's index and name
(`tensor_index`, `tensor_name`): the node of a tap point `<signature>:<op>:
<output>` (tap_scan) is node `<op>` of the subgraph the signature runs, and
its output `<output>`. The converter adds nodes of its own after the
operators: `GraphInputs`, `GraphOutputs` and one `pseudo_const` per constant
tensor read.

What the server keeps of it (signature_graphs()): the subgraph of each
signature, by its key, and nothing else: not the decompositions of the
composite operators, which are not expanded (a composite's `subgraphIds`
are emptied; its `decomposition` attribute still names its own), not the
other sections. Text of more than TEXT_LIMIT characters of a constant's
value or of a tensor's quantization (per-channel scales take most of a
graph's bytes) is replaced by a note of its length. That slimmed graph is
what is kept; its code is named by slimming_digest(), so a graph kept by
other code is not reused. Each output of an operator carries the server's
facts of its point (annotate()), added as a graph is read, so that they are
always the server's of now.

Model Explorer groups the nodes by their namespaces. The converter picks one
of a fused tensor's names for an operator's namespace, and so names some
operators of one layer by another's scope: Gemma 4's RoPE of layer 1's
query, say, which reads the cosine table layer 0 computed for every layer,
is in layer 0's scope. The graph the server serves groups each operator in
the layer its outputs' own names place it in, as the intermediate tensors'
analysis places their points (regroup()): the layer component of its
namespace is renumbered, and the converter's namespace kept beside, as the
node's attribute `converter_namespace`. An operator whose outputs are read
by operators of several layers is shared, and stays where the converter
grouped it (the cosine table, in layer 0); so does one whose outputs' names
place it in no layer, or in several.

An output whose name places it in no layer is in the layer the graph as it
is served groups its operator in (layer_groups()), so that the intermediate
tensors' analysis and the graph place every point alike: Gemma 4's layer 0
writes its KV cache to tensors named `StatefulPartitionedCall:1` and the
like, which name no scope, and the converter groups those operators in layer
0, by the scope of the operators they read.

A model's layers are named by a scope component `<stem>_<N>` that repeats
(layer_stems()): `layer_N` in Gemma 4, `...Qwen3DecoderLayer_N` in Qwen3. A
PyTorch model's graph (pytorch_graph) names them by a module path, a
segment `<N>` under a segment `<stem>`, as `layers/N`: its stem is written
`<stem>/` (`layers/`), and a layer's scope is `<stem>/<N>`
(layer_component()).
The stem is read from the converter's namespaces, and each tensor's layer is
then read by it from the tensor's own first name (tap_scan.layer_of), not
from the namespace, unless that name places it in none (layer_groups()):
the converter picks one of a fused tensor's names for its namespace, and so
places some operators of one layer in another. A run of numbered scopes is
a model's layers only if it is its repeated component: it holds most of the
signature's operators, and every signature names it alike; a graph that has
no such run, or more than one, has no layers, and says why.
"""

import collections
from collections.abc import Callable, Iterable, Mapping
import dataclasses
import hashlib
import re
from typing import Any

from model_debugger.server.tap import tap_scan

# The converter's own nodes; every other node is an operator.
GRAPH_INPUTS = 'GraphInputs'
GRAPH_OUTPUTS = 'GraphOutputs'
CONSTANT = 'pseudo_const'
AUXILIARY = (GRAPH_INPUTS, GRAPH_OUTPUTS, CONSTANT)
# Longer text of these attributes is not kept.
TEXT_LIMIT = 64
_LONG_NODE_ATTRS = ('__value',)
_LONG_TENSOR_ATTRS = ('quantization',)
# The attributes annotate() adds to an operator's output.
POINT_ID = 'point_id'
OBSERVABLE = 'observable'
REASON = 'reason'
LAYER = 'layer'
LAYER_OUTPUT = 'layer_output'
CONTEXT_AXES = 'context_axes'
# The attribute regroup() keeps a moved operator's namespace in.
CONVERTER_NAMESPACE = 'converter_namespace'
_COMPONENT = re.compile(r'^(.+)_(\d+)$')
# The share of a signature's operators its layers' scopes hold, at least.
MIN_LAYER_SHARE = 0.5


def signature_graphs(
    converted: list[dict[str, Any]] | dict[str, Any],
    subgraphs: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
  """Each signature's graph of the converter's output, slimmed.

  Args:
      converted: The converter's JSON of one TFLite model: a list of graphs, or
        {"graphs": [...]}, holding one graph.
      subgraphs: The subgraph each signature runs, by its key.

  Returns:
      Per signature, in the order of `subgraphs`, its graph: `{id, nodes}`,
      `id` the signature's key.

  Raises:
      ValueError: The output is not of one model, or lacks a subgraph.
  """
  graphs = converted['graphs'] if isinstance(converted, dict) else converted
  if len(graphs) != 1:
    raise ValueError(
        f'The converter wrote {len(graphs)} graphs of one TFLite model.'
    )
  by_index = {
      subgraph.get('subgraphIndex'): subgraph
      for subgraph in graphs[0]['subgraphs']
  }
  found = {}
  for name, index in subgraphs.items():
    subgraph = by_index.get(index)
    if subgraph is None:
      raise ValueError(
          f'The converter wrote no subgraph {index}, which {name} runs.'
      )
    found[name] = {
        'id': name,
        'nodes': [_slim(node) for node in subgraph['nodes']],
    }
  return found


def _slim(node: dict[str, Any]) -> dict[str, Any]:
  """A node without its long text and with no subgraphs to expand."""
  slim = dict(node)
  slim['attrs'] = _short(node.get('attrs', []), _LONG_NODE_ATTRS)
  for part in ('inputsMetadata', 'outputsMetadata'):
    slim[part] = [
        {
            **entry,
            'attrs': _short(entry.get('attrs', []), _LONG_TENSOR_ATTRS),
        }
        for entry in node.get(part, [])
    ]
  if slim.get('subgraphIds'):
    slim['subgraphIds'] = []
  return slim


def _short(
    attributes: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
  """Returns `attributes` with the long values of `keys` replaced by a note."""
  return [_shortened(attr, keys) for attr in attributes]


def _shortened(attr: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
  """Returns `attr`, or a copy noting its value's length if it is too long."""
  if attr['key'] not in keys or len(attr['value']) <= TEXT_LIMIT:
    return attr
  return {**attr, 'value': f'({len(attr["value"])} characters, not kept)'}


_SLIMMING_VERSION = 1


def slimming_digest() -> str:
  """A digest of the slimming schema and limits before a graph is kept.

  Returns:
      The digest's first 16 hex digits.
  """
  digest = hashlib.sha256(
      repr((
          _SLIMMING_VERSION,
          TEXT_LIMIT,
          _LONG_NODE_ATTRS,
          _LONG_TENSOR_ATTRS,
      )).encode()
  )
  return digest.hexdigest()[:16]


def operators(graph: dict[str, Any]) -> list[dict[str, Any]]:
  """The nodes of a graph's operators, in their order."""
  return [node for node in graph['nodes'] if node['label'] not in AUXILIARY]


def attrs(entry: dict[str, Any]) -> dict[str, str]:
  """A node's or an output's attributes, by key."""
  return {attr['key']: attr['value'] for attr in entry.get('attrs', [])}


@dataclasses.dataclass(frozen=True)
class LayerStems:
  """The stems a model's signatures name their layers by, and why not.

  Attributes:
      stems: Per signature that has layers, their stem, such as 'layer'.
      reasons: Per signature that has none, why.
  """

  stems: dict[str, str]
  reasons: dict[str, str]


def layer_component(stem: str, layer: int) -> str:
  """The scope component a stem names a layer by: `layer_3`, `layers/3`."""
  return f'{stem}{layer}' if stem.endswith('/') else f'{stem}_{layer}'


def layer_stems(
    graphs: Mapping[str, dict[str, Any]],
    segments: bool = False,
    operators_of: (
        Callable[[dict[str, Any]], list[dict[str, Any]]] | None
    ) = None,
) -> LayerStems:
  """The stem each signature's graph names its layers by, if any.

  A namespace component `<stem>_<N>` is a candidate when, under one parent
  scope, its numbers run 0, 1, ... with none missing. It names the
  signature's layers when its scopes hold at least MIN_LAYER_SHARE of the
  signature's operators: the model's repeated component, not one repeated
  within a part of it (the experts of a layer, say). A signature with no
  such candidate has no layers; one with more than one is ambiguous, and
  has none either. Every signature that has layers must name them by the
  same stem; if they do not, none has layers.

  Args:
      graphs: Per signature, its graph (signature_graphs()).
      segments: Whether layers are numbered by a path segment of their own,
        `<stem>/<N>`, as a PyTorch model's module paths are, not by a component
        `<stem>_<N>`; the stems found end in '/'.
      operators_of: The nodes of a graph's operators; operators() by default.

  Returns:
      Per signature, its layers' stem, or why it has none.
  """
  stems, reasons = {}, {}
  for name, graph in graphs.items():
    found, why = _stem((operators_of or operators)(graph), segments)
    if found is None:
      reasons[name] = why
    else:
      stems[name] = found
  if len(set(stems.values())) > 1:
    named = ', '.join(f'{name} {stem}' for name, stem in stems.items())
    why = f'The signatures name their layers differently: {named}.'
    return LayerStems({}, {**reasons, **dict.fromkeys(stems, why)})
  return LayerStems(stems, reasons)


def _stem(ops: list[dict[str, Any]], segments: bool) -> tuple[str | None, str]:
  """The stem of a signature's layers, or None and why it has none."""
  if not ops:
    return None, 'It has no operators.'
  numbers = collections.defaultdict(set)
  holders = collections.defaultdict(set)
  for index, node in enumerate(ops):
    parts = node.get('namespace', '').split('/')
    for depth, part in enumerate(parts):
      if segments:
        if depth and part.isdigit():
          key = '/'.join(parts[: depth - 1]), parts[depth - 1] + '/'
          numbers[key].add(int(part))
          holders[key].add(index)
        continue
      match = _COMPONENT.match(part)
      if match:
        key = '/'.join(parts[:depth]), match.group(1)
        numbers[key].add(int(match.group(2)))
        holders[key].add(index)
  runs = sorted(
      (
          (len(holders[key]) / len(ops), key)
          for key, found in numbers.items()
          if found == set(range(len(found)))
      ),
      reverse=True,
  )
  if not runs:
    return None, 'No scope of it is numbered 0, 1, ... as layers are.'
  layered = [(share, key) for share, key in runs if share >= MIN_LAYER_SHARE]
  if len(layered) > 1:
    named = '; '.join(_run(share, key, numbers) for share, key in layered)
    return None, f'Its layers are ambiguous: {named}.'
  if not layered:
    share, key = runs[0]
    return None, (
        f'No numbered scope of it holds {MIN_LAYER_SHARE:.0%} of its '
        'operators, as layers do; the most is '
        f'{_run(share, key, numbers)}.'
    )
  return layered[0][1][1], ''


def _run(
    share: float, key: tuple[str, str], numbers: dict[tuple[str, str], set[int]]
) -> str:
  parent, stem = key
  where = f' under {parent}' if parent else ''
  last = max(numbers[key])
  return (
      f'{layer_component(stem, 0)}..{last}{where}, {share:.0%} of its operators'
  )


def namespace_layer(namespace: str, stem: str) -> int | None:
  """The layer a namespace groups a node in: its first of the stem's."""
  parts = namespace.split('/')
  if stem.endswith('/'):
    for part, number in zip(parts, parts[1:]):
      if part == stem[:-1] and number.isdigit():
        return int(number)
    return None
  for part in parts:
    match = _COMPONENT.match(part)
    if match and match.group(1) == stem:
      return int(match.group(2))
  return None


def own_layer(node: dict[str, Any], stem: str) -> int | None:
  """The layer an operator's outputs' own names place it in, if one.

  tap_scan.layer_of() of each output's tensor name, as its point's layer;
  None if they place it in none, or in more than one.

  Args:
      node: The operator's node.
      stem: Its signature's layers' stem.

  Returns:
      The layer, or None.
  """
  layers = {
      tap_scan.layer_of(attrs(entry).get('tensor_name', ''), stem)
      for entry in node.get('outputsMetadata', [])
  }
  return layers.pop() if len(layers) == 1 else None


def regroup(graph: dict[str, Any], stem: str | None) -> dict[str, Any]:
  """A signature's graph, each operator grouped in its own layer.

  An operator whose namespace groups it in layer M, but whose outputs'
  names place it in layer N (own_layer()), is grouped in layer N: each
  component `<stem>_<M>` of its namespace becomes `<stem>_<N>`, and one
  `<stem>_<M>.<rest>` becomes `<stem>_<N>.<rest>`; its namespace before is
  kept as CONVERTER_NAMESPACE. One whose outputs operators of several
  layers read (each reader in its own layer) is shared, and is not moved.

  Args:
      graph: The signature's graph (signature_graphs()).
      stem: Its layers' stem (layer_stems()); None moves nothing.

  Returns:
      The graph regrouped; `graph` itself is not changed.
  """
  if not stem:
    return graph
  ops = operators(graph)
  own = {node['id']: own_layer(node, stem) for node in ops}
  readers: dict[str, set[int]] = collections.defaultdict(set)
  for node in ops:
    for edge in node.get('incomingEdges', []):
      if own[node['id']] is not None:
        readers[edge['sourceNodeId']].add(own[node['id']])
  nodes = []
  for node in graph['nodes']:
    target = own.get(node['id'])
    namespace = node.get('namespace', '')
    current = namespace_layer(namespace, stem)
    if (
        target is None
        or current is None
        or current == target
        or len(readers[node['id']]) > 1
    ):
      nodes.append(node)
      continue
    before, after = f'{stem}_{current}', f'{stem}_{target}'
    parts = [
        after + part[len(before) :]
        if part == before or part.startswith(before + '.')
        else part
        for part in namespace.split('/')
    ]
    nodes.append({
        **node,
        'namespace': '/'.join(parts),
        'attrs': (
            node.get('attrs', [])
            + [{'key': CONVERTER_NAMESPACE, 'value': namespace}]
        ),
    })
  return {**graph, 'nodes': nodes}


def operator_layers(graph: dict[str, Any], stem: str | None) -> tuple[int, ...]:
  """The layers a graph has operators of, ascending.

  The layers its operators' namespaces group them in (namespace_layer()):
  the layers the signature computes. A prefill signature computes only
  the layers whose KV cache it fills.

  Args:
      graph: A signature's graph as it is served (regroup()).
      stem: Its layers' stem (layer_stems()); None has none.

  Returns:
      The layers, ascending; empty without a stem.
  """
  if not stem:
    return ()
  found = {
      namespace_layer(node.get('namespace', ''), stem)
      for node in operators(graph)
  }
  return tuple(sorted(found - {None}))


def layer_groups(graph: dict[str, Any], stem: str | None) -> dict[int, int]:
  """The layers a graph groups the operators in whose names place none.

  Of each operator one of whose outputs' names places it in no layer
  (tap_scan.layer_of()), the layer its namespace groups it in, if one: the
  layer of that output's point (tap_scan.read_model()).

  Args:
      graph: A signature's graph as it is served (regroup()).
      stem: Its layers' stem (layer_stems()); None groups nothing.

  Returns:
      Per such operator, by its index in its subgraph, its layer.
  """
  if not stem:
    return {}
  found = {}
  for node in operators(graph):
    layer = namespace_layer(node.get('namespace', ''), stem)
    if layer is not None and any(
        tap_scan.layer_of(attrs(entry).get('tensor_name', ''), stem) is None
        for entry in node.get('outputsMetadata', [])
    ):
      found[int(node['id'])] = layer
  return found


def annotate(
    graph: dict[str, Any], points: Iterable[tap_scan.Point], signature: str
) -> dict[str, Any]:
  """A signature's graph with each operator output's point facts.

  Each output of an operator that is a point gets, as string attributes
  (Model Explorer's), `point_id`, `observable` ('true' or 'false'), and
  when they apply `reason` (why it cannot be observed), `layer`,
  `layer_output` ('true') and `context_axes` (axes of the model's KV
  length, comma-separated).

  Args:
      graph: The signature's graph (signature_graphs()).
      points: The model's points; those of other signatures are ignored.
      signature: The signature's key.

  Returns:
      The graph annotated; `graph` itself is not changed.

  Raises:
      ValueError: An output's tensor is not its point's: the graph is not
        the model's.
  """
  mine = {
      (point.op, point.output): point
      for point in points
      if point.signature == signature
  }
  nodes = []
  for node in graph['nodes']:
    if node['label'] in AUXILIARY:
      nodes.append(node)
      continue
    outputs = []
    for entry in node.get('outputsMetadata', []):
      point = mine.get((int(node['id']), int(entry['id'])))
      if point is None:
        outputs.append(entry)
        continue
      found = attrs(entry)
      if found.get('tensor_index') != str(point.tensor):
        raise ValueError(
            f'Output {entry["id"]} of node {node["id"]} of '
            f'{signature} is tensor {found.get("tensor_index")}, '
            f'not {point.tensor} as {point.id} is: the graph is not '
            "the model's."
        )
      outputs.append(
          {**entry, 'attrs': entry.get('attrs', []) + point_facts(point)}
      )
    nodes.append({**node, 'outputsMetadata': outputs})
  return {**graph, 'nodes': nodes}


def point_facts(point: tap_scan.Point) -> list[dict[str, Any]]:
  """A point's facts, as annotate() gives them an output's attributes."""
  facts = [
      (POINT_ID, point.id),
      (OBSERVABLE, 'true' if point.selectable else 'false'),
  ]
  if point.reason:
    facts.append((REASON, point.reason))
  if point.layer is not None:
    facts.append((LAYER, str(point.layer)))
  if point.layer_output:
    facts.append((LAYER_OUTPUT, 'true'))
  if point.context_axes:
    facts.append(
        (CONTEXT_AXES, ','.join(str(axis) for axis in point.context_axes))
    )
  return [{'key': key, 'value': value} for key, value in facts]


def collection(
    graph: dict[str, Any], label: str, attributes: Mapping[str, str]
) -> dict[str, Any]:
  """A graph as a Model Explorer graph collection of its own.

  Args:
      graph: A signature's graph.
      label: The collection's label.
      attributes: What the graph's root group shows, as strings.

  Returns:
      `{label, graphs}`, the one graph with `groupNodeAttributes`.
  """
  return {
      'label': label,
      'graphs': [{
          **graph,
          'groupNodeAttributes': {'': dict(attributes)},
      }],
  }
