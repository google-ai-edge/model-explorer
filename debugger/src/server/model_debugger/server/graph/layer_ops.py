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

"""The semantic operators inside a decoder layer, captured in every layer.

A Session may capture, beside the anchors, semantic operators of its
architecture's decoder layer, as the skeleton names them (`attn/q_proj`,
`ffn/act`, `norm.attn.in`): each one chosen in every layer that has it, on
both sides (session_store's `tap_ops`). Of a LiteRT-LM model, an operator's
tensor in a layer is the one tap point its binding's `ops` rule names, as
counterparts finds a point's operator: the point directly in the rule's
scope, of the rule's operator (an int8 projection's float DEQUANTIZE, never
its QUANTIZE or FULLY_CONNECTED); of a PyTorch model, the module its
binding's `ops` rule names, where the layer has the operator
(counterparts.module_of()). Each is then placed at (layer, operator), the
place both runtimes pair by (tensor_analysis).

An operator is missing from a layer for a reason the skeleton says (a
KV-shared layer, `skip`, has no k_proj, v_proj, k_norm or v_norm: it reuses
an earlier layer's keys and values, its `from.kv`), or because the side's
graph or model has no tensor of it there; missing() says which, by ranges
of layers.
"""

import collections
from collections.abc import Iterable, Mapping
import dataclasses
import math
from typing import Any

from model_debugger.server.graph import architecture
from model_debugger.server.graph import counterparts
from model_debugger.server.graph import pytorch_binding
from model_debugger.server.tap import tap_scan

# The signature a LiteRT-LM model decodes one token with: a capture's size
# per generated token is that of its tensors.
DECODE = 'decode'
# Bytes per element of the dtypes a side stores; one not listed counts as
# float32.
_ITEMSIZE = {
    'float64': 8,
    'float32': 4,
    'float16': 2,
    'bfloat16': 2,
    'int64': 8,
    'int32': 4,
    'int16': 2,
    'int8': 1,
    'uint8': 1,
    'bool': 1,
}


@dataclasses.dataclass(frozen=True)
class Op:
  """A semantic operator of a decoder layer.

  Attributes:
      id: As the bindings' `ops` name it: `<node>/<op>`, or a norm's node.
      block: The block it is in, the skeleton's namespace (`Attention`).
      name: Its name in the block: the operator's (`q_proj`), or the norm node's
        label (`Pre-RMSNorm`).
      kind: What it computes (`Linear`, `RMSNorm`, `GELU`, `Mul`).
      label: Its place's label (counterparts.place_labels()).
  """

  id: str
  block: str
  name: str
  kind: str
  label: str


def known() -> frozenset[str]:
  """Every operator any architecture's binding, of either runtime, maps."""
  found = set()
  for each in architecture.architectures():
    found.update(each.binding.get('ops', {}))
    try:
      found.update(pytorch_binding.binding(each.id).get('ops', {}))
    except ValueError:
      continue
  return frozenset(found)


def operators(document: dict[str, Any], mapped: Iterable[str]) -> list[Op]:
  """The operators of a skeleton's layers that a binding maps, in order.

  In the layers' order: each node's operators as the skeleton lists them.

  Args:
      document: The skeleton.
      mapped: The operators the bindings map.

  Returns:
      The operators, each once, in the layers' order.
  """
  wanted = set(mapped)
  labels = counterparts.place_labels(document, wanted)
  found = {}
  for layer in document['layers']:
    definition = document['semantic_graph'][layer['def']]
    for node in definition.get('nodes', []):
      for each in node.get('ops') or [{'id': node['id'], 'op': ''}]:
        own = each['id'] == node['id']
        key = node['id'] if own else f'{node["id"]}/{each["id"]}'
        if key not in wanted or key in found:
          continue
        found[key] = Op(
            id=key,
            block=node.get('namespace', ''),
            name=node.get('label', key) if own else each['id'],
            kind=each.get('op', ''),
            label=labels.get(key, key),
        )
  return list(found.values())


def litert_points(
    points: Iterable[tap_scan.Point],
    rules: Mapping[str, dict[str, Any]],
    ops: Iterable[str],
    signatures: Iterable[str],
) -> dict[str, dict[str, dict[int, tap_scan.Point]]]:
  """Each operator's point in each layer of each signature, where certain.

  Args:
      points: A LiteRT-LM model's scan's points, placed in layers.
      rules: Its binding's `ops`.
      ops: The operators wanted.
      signatures: The signatures captured.

  Returns:
      Per operator, per signature, per layer, the one point that can be
      observed directly in the rule's scope, of its operator; a layer
      that has two is left out, for which one it is is not certain.
  """
  wanted = [op for op in ops if op in rules]
  allowed = set(signatures)
  found = {
      op: collections.defaultdict(lambda: collections.defaultdict(list))
      for op in wanted
  }
  for point in points:
    if (
        point.layer is None
        or point.signature not in allowed
        or not point.selectable
    ):
      continue
    for op in wanted:
      rule = rules[op]
      if point.operator == rule['operator'] and counterparts.directly_in(
          point, rule['scope']
      ):
        found[op][point.signature][point.layer].append(point)
  return {
      op: {
          signature: {
              layer: matches[0]
              for layer, matches in sorted(layers.items())
              if len(matches) == 1
          }
          for signature, layers in by_signature.items()
      }
      for op, by_signature in found.items()
  }


def pytorch_modules(
    document: dict[str, Any],
    rules: Mapping[str, dict[str, Any]],
    op: str,
    label: str,
) -> tuple[dict[int, counterparts.Module], dict[int, str]]:
  """An operator's PyTorch module in each layer, or why a layer has none.

  Args:
      document: The PyTorch model's skeleton.
      rules: The PyTorch binding's `ops`.
      op: The operator.
      label: What the user knows the operator by.

  Returns:
      `(modules, reasons)`, each by layer.
  """
  modules, reasons = {}, {}
  for layer in range(len(document['layers'])):
    module, why = counterparts.module_of(layer, op, label, rules, document)
    if module is None:
      reasons[layer] = why
    else:
      modules[layer] = module
  return modules, reasons


def missing(
    document: dict[str, Any], op: str, have: Iterable[int], where: str
) -> list[str]:
  """Why a side lacks an operator in some layers, one line per reason.

  Args:
      document: The skeleton.
      op: The operator.
      have: The layers the side has it in.
      where: What lacks it otherwise, as the line says it (`the LiteRT-LM graph
        has no tensor of it`).

  Returns:
      `not in layers 15–34 (they reuse layer 13/14's K and V)` for the
      layers the skeleton marks KV-shared; `not in layers …: <where>` for
      any other; [] if none lacks it.
  """
  present = set(have)
  gone = [
      index for index in range(len(document['layers'])) if index not in present
  ]
  shared = [index for index in gone if _skipped(document, index, op)]
  other = [index for index in gone if index not in shared]
  lines = []
  if shared:
    sources = sorted(
        {
            document['layers'][index].get('from', {}).get('kv', {}).get('layer')
            for index in shared
        }
        - {None}
    )
    whose = (
        f"layer {'/'.join(map(str, sources))}'s"
        if sources
        else "an earlier layer's"
    )
    lines.append(f'not in {layer_ranges(shared)} (they reuse {whose} K and V)')
  if other:
    lines.append(f'not in {layer_ranges(other)}: {where}')
  return lines


def _skipped(document: dict[str, Any], layer: int, op: str) -> bool:
  node, _, name = op.partition('/')
  attrs = document['layers'][layer].get('attrs', {})
  found = attrs.get(node, {}).get('ops', {}).get(name or node)
  return bool(found and found.get('skip'))


def layer_ranges(layers: Iterable[int]) -> str:
  """Layers as the user reads them: `layer 3`, `layers 0–13, 15`."""
  ordered = sorted(set(layers))
  runs = []
  for layer in ordered:
    if runs and layer == runs[-1][1] + 1:
      runs[-1][1] = layer
    else:
      runs.append([layer, layer])
  text = ', '.join(
      str(first) if first == last else f'{first}–{last}' for first, last in runs
  )
  return f'layer {text}' if len(ordered) == 1 else f'layers {text}'


def itemsize(dtype: str) -> int:
  """Bytes per element of a dtype; 4 for one not known."""
  return _ITEMSIZE.get(dtype, 4)


def token_bytes(point: tap_scan.Point) -> int:
  """Bytes a LiteRT-LM point holds per token: its shape after [1, rows]."""
  return math.prod(point.shape[2:]) * itemsize(point.dtype)


def module_bytes(module: counterparts.Module, dtype: str) -> int:
  """Bytes a PyTorch module's tensor holds per token, its dims."""
  return math.prod(module.dims) * itemsize(dtype)
