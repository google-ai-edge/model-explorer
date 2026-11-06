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

"""The PyTorch module each LiteRT-LM tap point chosen is, where it is certain.

Two runtimes are paired by places: a layer and an anchor of the skeleton
(tensor_analysis). A Session that pairs a LiteRT-LM model with the PyTorch
runner's may choose more points of its LiteRT-LM model (the editor's
advanced picker); each is paired with the PyTorch module of the same
semantic operator of the same layer, where that is certain, and that module
is then captured on the PyTorch side too (the runner's observeModules).

A semantic operator is one inside a layer's blocks, as the skeleton names
it: `<node>/<op>` for an operator of a module node (`attn/q_proj`,
`ffn/act`, `ple/gate_proj`), or a norm's node id (`norm.attn.in`). Its place
is (layer, that id). Both runtimes' bindings say where it is (`ops`): the
LiteRT-LM binding the tap point of each layer that is its output, the one
whose tensor's name is directly in its scope (`{layer}` standing for the
layer's scope component) and of its operator; the PyTorch binding the module
whose input (argument 0) or output it is, and its shape after [batch,
rows].

A point chosen is paired (find()) when it is certain: it is in a layer of
the skeleton; it is the output of exactly one semantic operator the
LiteRT-LM binding maps, by its scope and its operator (another tensor of
that scope, such as an int8 projection's QUANTIZE or FULLY_CONNECTED, is
not); the PyTorch binding maps that operator to a module; the layer has
that module (a KV-shared layer, which the skeleton marks `skip`, has no
k_proj, v_proj, k_norm or v_norm); and the two shapes, after [batch,
tokens] and without axes of one, are the same, or the same elements with
one of them flat (q_proj's [8, 256] after LiteRT-LM's reshape, [2048] as
the Linear returns it). Any other point chosen says why not: PyTorch has
no counterpart for it, and only the anchors are captured there.
"""

from collections.abc import Iterable, Mapping
import dataclasses
import math
from typing import Any

from model_debugger.server.architectures import skeleton
from model_debugger.server.graph import model_graph
from model_debugger.server.tap import tap_scan

# What every reason a point is not paired for ends with.
ONLY_ANCHORS = 'only the anchors are captured there'


@dataclasses.dataclass(frozen=True)
class Module:
  """The PyTorch module a semantic operator's tensor is taken at.

  Attributes:
      module: Its path, `{layer}` filled in.
      edge: 'out' for what it returns, 'in' for its argument 0.
      path: Its path in pytorch-collector's record, dotted.
      dims: Its shape after [batch, rows].
  """

  module: str
  edge: str
  path: str
  dims: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class Counterpart:
  """A LiteRT-LM point chosen, and the PyTorch module paired with it.

  Attributes:
      point_id: The point.
      layer: Its layer, if it has one.
      op: Its semantic operator, the place's id; '' if not paired.
      label: What the user knows the operator by; '' if not paired.
      module: The PyTorch module paired with it; None if not paired.
      reason: Why it is not paired, as the user reads it; '' if it is.
  """

  point_id: str
  layer: int | None
  op: str = ''
  label: str = ''
  module: Module | None = None
  reason: str = ''


def place_labels(
    document: dict[str, Any], ops: Iterable[str] = ()
) -> dict[str, str]:
  """Each anchor and semantic operator of a skeleton's layers, in order.

  The anchors as skeleton.anchor_label() names them; each operator of
  `ops` as its block and its name (`Attention q_proj`, `Feed Forward
  Pre-RMSNorm`), at its place in the layer's nodes: the layer's input
  first, each node's operators in the skeleton's order, then the anchor
  the node's output is, if any.

  Args:
      document: The skeleton.
      ops: The semantic operators to label: those a binding maps.

  Returns:
      Per anchor id and operator key, its label, in the layers' order.
  """
  wanted = set(ops)
  labels = {}
  for layer in document['layers']:
    definition = document['semantic_graph'][layer['def']]
    anchors = definition.get('anchors', [])
    of = {
        anchor.get('of', '').split(':')[0]: anchor
        for anchor in anchors
        if anchor.get('of')
    }
    for anchor in anchors:
      if anchor.get('of', '').split(':')[0] not in _node_ids(definition):
        labels.setdefault(
            anchor['id'], skeleton.anchor_label(definition, anchor)
        )
    attrs = layer.get('attrs', {})
    for node in definition.get('nodes', []):
      names = list(attrs.get(node['id'], {}).get('ops', {}))
      for name in names or [node['id']]:
        key = node['id'] if name == node['id'] else f'{node["id"]}/{name}'
        if key in wanted:
          text = name if key != node['id'] else node.get('label', name)
          labels.setdefault(key, f'{node.get("namespace", "")} {text}'.strip())
      anchor = of.get(node['id'])
      if anchor is not None:
        labels.setdefault(
            anchor['id'], skeleton.anchor_label(definition, anchor)
        )
  return labels


def _node_ids(definition: dict[str, Any]) -> set[str]:
  return {node['id'] for node in definition.get('nodes', [])}


def directly_in(point: tap_scan.Point, scope: str) -> bool:
  """Whether a point's tensor is directly in a scope of its layer.

  Its first name without its last component ends with the scope, `{layer}`
  standing for the layer's scope component (`layer_3`).

  Args:
      point: The point, placed in a layer.
      scope: The scope, `{layer}` standing for the layer's scope component.

  Returns:
      Whether it is; False for a point in no layer.
  """
  if point.layer is None or not point.layer_stem:
    return False
  component = model_graph.layer_component(point.layer_stem, point.layer)
  name = point.tensor_name.split(';')[0]
  parent = f'/{name}'.rsplit('/', 1)[0]
  return parent.endswith('/' + scope.format(layer=component))


def find(
    chosen: Iterable[tap_scan.Point],
    points: Iterable[tap_scan.Point],
    litert_ops: Mapping[str, dict[str, Any]],
    pytorch_ops: Mapping[str, dict[str, Any]],
    document: dict[str, Any],
) -> list[Counterpart]:
  """The PyTorch module paired with each LiteRT-LM point chosen, or why not.

  Args:
      chosen: The points chosen, as the model's scan lists them, placed in
        layers.
      points: Every point of the scan, to name the one a semantic operator's
        output is where another of its scope was chosen.
      litert_ops: The LiteRT-LM binding's `ops`.
      pytorch_ops: The PyTorch binding's `ops`.
      document: The PyTorch model's skeleton.

  Returns:
      One per point chosen, in their order.
  """
  every = list(points)
  labels = place_labels(document, litert_ops)
  found = []
  for point in chosen:
    op, why = _operator(point, every, litert_ops, labels)
    if op is None:
      found.append(_unpaired(point, why))
      continue
    module, why = _module(point, op, labels[op], pytorch_ops, document)
    if module is None:
      found.append(_unpaired(point, why))
      continue
    found.append(Counterpart(point.id, point.layer, op, labels[op], module, ''))
  return found


def _operator(
    point: tap_scan.Point,
    points: list[tap_scan.Point],
    rules: Mapping[str, dict[str, Any]],
    labels: Mapping[str, str],
) -> tuple[str | None, str]:
  """The semantic operator a point is the output of, or why none."""
  if point.layer is None:
    return None, 'it is in no decoder layer'
  scoped = [
      op for op, rule in rules.items() if directly_in(point, rule['scope'])
  ]
  if not scoped:
    return None, (
        'its tensor is in no scope the binding maps to a semantic '
        'operator of the layer (its projections, norms and activations)'
    )
  if len(scoped) > 1:
    return None, (
        'its tensor is in the scopes of '
        + ', '.join(labels.get(op, op) for op in scoped)
        + ': which one it is is not certain'
    )
  op = scoped[0]
  rule = rules[op]
  if point.operator == rule['operator']:
    return op, ''
  output = [
      other.id
      for other in points
      if other.signature == point.signature
      and other.layer == point.layer
      and other.operator == rule['operator']
      and directly_in(other, rule['scope'])
  ]
  where = f' ({output[0]})' if len(output) == 1 else ''
  return None, (
      f'it is a {point.operator} ({point.dtype}) inside '
      f'{labels.get(op, op)}, not its output, the {rule["operator"]}{where}'
  )


def _module(
    point: tap_scan.Point,
    op: str,
    label: str,
    rules: Mapping[str, dict[str, Any]],
    document: dict[str, Any],
) -> tuple[Module | None, str]:
  """The PyTorch module of a semantic operator of the point's layer."""
  module, why = module_of(point.layer, op, label, rules, document)
  if module is None:
    return None, why
  if not compatible(point.shape, module.dims):
    side = 'returns' if module.edge == 'out' else 'reads'
    return None, (
        f'its shape {list(point.shape)} is not that of what '
        f'{module.module} {side}, [batch, rows, '
        f'{", ".join(map(str, module.dims))}]'
    )
  return module, ''


def module_of(
    layer: int,
    op: str,
    label: str,
    rules: Mapping[str, dict[str, Any]],
    document: dict[str, Any],
) -> tuple[Module | None, str]:
  """The PyTorch module of a semantic operator of a layer, or why none.

  Args:
      layer: The layer.
      op: The semantic operator, as the bindings' `ops` name it.
      label: What the user knows the operator by.
      rules: The PyTorch binding's `ops`.
      document: The PyTorch model's skeleton.

  Returns:
      `(module, '')` if the layer has the operator's module, else
      `(None, why)`.
  """
  rule = rules.get(op)
  if rule is None:
    return None, f'the PyTorch binding maps no module of {label}'
  if layer >= len(document['layers']):
    return None, (
        f'the PyTorch model has {len(document["layers"])} layers, not '
        f'layer {layer}'
    )
  attrs = document['layers'][layer].get('attrs', {})
  node, _, name = op.partition('/')
  found = attrs.get(node, {}).get('ops', {}).get(name or node)
  if found is None:
    return None, f"the PyTorch model's layer {layer} has no {label}"
  if found.get('skip'):
    return None, (
        f'layer {layer} shares the keys and values of an earlier '
        f'layer (KV-shared): its PyTorch attention has no {name or node}'
    )
  try:
    dims = _dims(rule['shape'], attrs, document)
  except (KeyError, TypeError, ValueError) as error:
    return None, f'the shape of {label} is not known: {error}'
  return (
      Module(
          module=rule['module'].format(layer=layer),
          edge=rule['edge'],
          path=rule['path'],
          dims=dims,
      ),
      '',
  )


def _dims(
    shape: str, attrs: dict[str, Any], document: dict[str, Any]
) -> tuple[int, ...]:
  """A PyTorch binding's shape rule, in a layer: its dims after [B, S]."""
  if shape == 'width':
    return (int(document['semantic_graph'][0]['inputs'][0]['shape'][-1]),)
  op, _, form = shape.partition(':')
  node, _, name = op.partition('/')
  out = attrs[node]['ops'][name]['out']
  if isinstance(out, int):
    return (out,)
  heads, dim = int(out['heads']), int(out['dim'])
  if form == 'heads':
    return (heads, dim)
  if form:
    raise ValueError(f'{shape} names no form {form}')
  return (heads * dim,)


def compatible(shape: Iterable[int], dims: Iterable[int]) -> bool:
  """Whether a LiteRT-LM point's shape holds what a module's dims do.

  The point's shape after its first two axes, batch and tokens, and the
  module's dims, each without axes of one: the same, or the same elements
  with one of them flat.

  Args:
      shape: The LiteRT-LM point's shape.
      dims: The module's dims after [batch, rows].

  Returns:
      Whether they match; False for a shape of fewer than two axes or of
      batch other than one.
  """
  shape = tuple(shape)
  if len(shape) < 2 or shape[0] != 1:
    return False
  ours = tuple(size for size in shape[2:] if size != 1)
  theirs = tuple(size for size in dims if size != 1)
  if ours == theirs:
    return True
  return math.prod(ours) == math.prod(theirs) and (
      len(ours) <= 1 or len(theirs) <= 1
  )


def _unpaired(point: tap_scan.Point, why: str) -> Counterpart:
  where = ', '.join(
      part
      for part in (
          f'layer {point.layer}' if point.layer is not None else '',
          point.operator,
      )
      if part
  )
  return Counterpart(
      point.id,
      point.layer,
      reason=(
          f'PyTorch has no counterpart for {point.id}'
          f'{f" ({where})" if where else ""}: {why}; {ONLY_ANCHORS}.'
      ),
  )
