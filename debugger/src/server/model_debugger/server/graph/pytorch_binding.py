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

"""How the PyTorch runner's model binds to an architecture.

An architecture's PyTorch binding, `bindings/<id>_pytorch.json`, names for
each anchor of the skeleton's layers the module position the PyTorch runner
takes it at: the module (`{layer}` standing for the layer's index), whether
it is what the module reads (`in`) or returns (`out`), and its path in
pytorch-collector's record; the runner's signatures (`signatures`), every
one of which runs every layer; and the stages each module outside the
skeleton runs in (`outside`). The same binding says where each anchor is
in the runner's execution graph, and what of it is folded (architecture,
pytorch_graph). The runner writes every anchor of every layer
of every forward pass it observes (capture_index.anchor_key), so a PyTorch
model's anchors are bound by the binding alone, with no graph: anchors()
places them by the skeleton the model's own config.json makes
(catalog.generate()). The binding's `ops` name, per semantic operator of a
layer, the module the runner takes its tensor at when the server asks for
it (counterparts; op_modules() lists every layer's).
"""

import dataclasses
import json
import pathlib
from typing import Any

from model_debugger.server.architectures import catalog
from model_debugger.server.architectures import skeleton

_DIRECTORY = pathlib.Path(__file__).parent / 'bindings'
RUNTIME = 'PyTorch'


@dataclasses.dataclass(frozen=True)
class Anchor:
  """Where the PyTorch runner takes one anchor of one layer.

  Attributes:
      layer: The layer.
      anchor: The anchor's id, the skeleton's.
      label: What it is, as skeleton.anchor_label() names it.
      module: The module path.
      edge: 'in' or 'out'.
      path: Its path in pytorch-collector's record, dotted.
  """

  layer: int
  anchor: str
  label: str
  module: str
  edge: str
  path: str


@dataclasses.dataclass(frozen=True)
class Bound:
  """A PyTorch model's architecture and where its anchors are.

  Attributes:
      architecture: The architecture's id.
      width: The width of the layers' input, the skeleton's.
      signatures: The runner's signatures, each running every layer.
      anchors: Per layer, its anchors in the skeleton's order.
  """

  architecture: str
  width: int
  signatures: tuple[str, ...]
  anchors: tuple[Anchor, ...]


def point_id(signature: str, anchor: Anchor) -> str:
  """The id of an anchor's point in a signature: its module and edge."""
  return module_point_id(signature, anchor.module, anchor.edge)


def module_point_id(signature: str, module: str, edge: str) -> str:
  """The id of a module's point in a signature, as an anchor's is."""
  return f'{signature}:{module}:{edge}'


@dataclasses.dataclass(frozen=True)
class OpModule:
  """Where the PyTorch runner can take a semantic operator of a layer.

  Attributes:
      layer: The layer.
      op: The semantic operator (the binding's `ops` key).
      module: The module path.
      edge: 'in' or 'out'.
  """

  layer: int
  op: str
  module: str
  edge: str


def op_modules(config: dict[str, Any]) -> list[OpModule]:
  """Every semantic operator's module of every layer the model has.

  By the binding's `ops`, of the skeleton the model's config.json makes:
  an operator the skeleton marks `skip` in a layer (a KV-shared layer's
  k_proj) has none there.

  Args:
      config: The model directory's config.json, parsed.

  Returns:
      Per layer in order, each operator's module, in the binding's order.

  Raises:
      ValueError: No architecture defines the model, or it has no PyTorch
        binding.
  """
  known, document = catalog.generate(config)
  rules = binding(known.id).get('ops', {})
  found = []
  for index, layer in enumerate(document['layers']):
    attrs = layer.get('attrs', {})
    for op, rule in rules.items():
      node, _, name = op.partition('/')
      facts = attrs.get(node, {}).get('ops', {}).get(name or node)
      if facts is None or facts.get('skip'):
        continue
      found.append(
          OpModule(index, op, rule['module'].format(layer=index), rule['edge'])
      )
  return found


def binding(architecture_id: str) -> dict[str, Any]:
  """An architecture's PyTorch binding, whole.

  Args:
      architecture_id: The architecture's id.

  Returns:
      The binding's file, parsed.

  Raises:
      ValueError: It has none, or the file is of another architecture or
        runtime.
  """
  path = _DIRECTORY / f'{architecture_id}_pytorch.json'
  if not path.is_file():
    raise ValueError(f'No PyTorch binding of {architecture_id} exists.')
  found = json.loads(path.read_text(encoding='utf-8'))
  if (
      found.get('architecture') != architecture_id
      or found.get('runtime') != RUNTIME
  ):
    raise ValueError(
        f'{path.name} is not the PyTorch binding of {architecture_id}.'
    )
  return found


def anchors(config: dict[str, Any]) -> Bound:
  """Where the anchors of the model of a config.json are.

  Args:
      config: The model directory's config.json, parsed.

  Returns:
      The architecture, its width, its signatures and each layer's anchors.

  Raises:
      ValueError: No architecture defines the model, or it has no PyTorch
        binding, or no rule for an anchor of its skeleton.
  """
  known, document = catalog.generate(config)
  rules = binding(known.id)
  found = []
  for index, layer in enumerate(document['layers']):
    definition = document['semantic_graph'][layer['def']]
    for anchor in definition['anchors']:
      rule = rules['anchors'].get(anchor['id'])
      if rule is None:
        raise ValueError(
            f'The PyTorch binding of {known.name} has no rule for '
            f'anchor {anchor["id"]}.'
        )
      found.append(
          Anchor(
              layer=index,
              anchor=anchor['id'],
              label=skeleton.anchor_label(definition, anchor),
              module=rule['module'].format(layer=index),
              edge=rule['edge'],
              path=rule['path'],
          )
      )
  width = document['semantic_graph'][0]['inputs'][0]['shape'][-1]
  return Bound(
      architecture=known.id,
      width=width,
      signatures=tuple(rules['signatures']),
      anchors=tuple(found),
  )
