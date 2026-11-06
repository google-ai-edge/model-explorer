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

"""The semantic skeleton schema: its vocabulary, its form and its checks.

A skeleton is what a model is, and nothing of how a runtime runs it: its
decoder's architecture, made from a Hugging Face config.json by the
architecture's definition (gemma4.build()). It is a document
`{semantic_ops, semantic_graph, layers}`, model_arch_explore's schema
(mkjson.py, from which this package was ported; see README.md).

- `semantic_ops`: the vocabulary of semantic operators (SEMANTIC_OPS), each
  with what it means, sent with the data so that a reader of the raw JSON
  need not read the code.
- `semantic_graph`: the layer definitions, an array; a definition is
  `{kind, inputs, outputs, nodes, anchors}`, structure only, no parameter.
  Its inputs are its ports, `{id, shape, source, label}`: Gemma 4's
  decoder has two, the residual stream `input` and the `per_layer_input`
  its per-layer embedding stage multiplies by (source `external per-layer
  embedding`, made outside the skeleton), which an edge reads as
  `<id>:0`.
  Its nodes are Model Explorer GraphNodes, `{id, label, namespace,
  incomingEdges, outputsMetadata, attrs, ops}`: `namespace` is the node's
  block, `outputsMetadata` holds each tensor once (`tensor_name`
  `<node>:<port>`, `tensor_shape`), `attrs` the node's `category`, and
  `ops` the semantic operators the stage runs, in order. Its anchors,
  `{id, semantic, of, block}`, are the tensors a comparison may be attached
  to: `of` is the tensor (`<node>:<port>`, or a definition input's id),
  `block` the name of the stretch of nodes from the anchor before it.
- `layers`: one entry per layer, `{def, attrs, from}`: the definition it
  uses, its parameters keyed by node id (and, under `ops`, by semantic
  operator id), and where a layer input comes from: `{input: {layer,
  anchor}}`, and `{kv: {layer}}` for a layer that borrows another's K and V.

Where it differs from Model Explorer's graph, and why:

- No `metrics`, and a tensor has no `dtype`: either would weld a run to the
  architecture, which is what the schema keeps apart. The precision is the
  runtime's choice; written here, it would claim the model has one, which
  every run of another contradicts.
- `anchors`, the points a comparison may be attached to, which Model
  Explorer has no notion of.
- A definition's inputs are a list of their own, not Placeholder nodes: an
  input is no computation, so it takes no node's place, as ONNX's
  GraphProto keeps node and input apart. `source` tells the residual
  stream from an input made elsewhere.
- No list of edges: an edge is on its reader (`incomingEdges`), as in Model
  Explorer, whose Graph is `{id, nodes}` alone.
- A tensor is kept once, on the outputs of the node that makes it; a reader
  names it, `<producer>:<port>`, and finds its shape there. One tensor read
  three times is one tensor and three edges; a shape copied on each edge
  could drift three ways.

Two kinds of name, and no third: an `id` names a producer (a node, or a
definition input), a tensor's name is always `<producer>:<port>`, the port
never left out (anchors' `of` too).

The semantic graph and an execution graph are aligned only by anchor; no
semantic operator stands for a runtime operator. check() runs every rule of
the schema, mkjson.py's check() and check_anchor_criterion(), in English.
"""

import json
import pathlib
import types
from typing import Any

# The vocabulary of semantic_ops[].op. A word belongs here iff every
# implementation would agree "this step is that". The cross-framework
# standards (ONNX, StableHLO, TFLite) are interchange formats of
# implementations, so their words are implementation words; the only common
# language of the semantic level is the literature's names. A closed set: a
# new word must meet that test, and say where it comes from. The meanings
# are English because they go out with the JSON.
SEMANTIC_OPS = {
    'Linear': (
        'Projection by a learned matrix. Not FullyConnected \u2014 that is '
        "TFLite's word; not Gemm or MatMul \u2014 those demote a projection "
        'to a matrix multiply.'
    ),
    'RMSNorm': (
        "Root-mean-square normalization. The paper's name; PyTorch, JAX and "
        'LiteRT all use it.'
    ),
    'RoPE': "Rotary position embedding. The paper's name.",
    'Attention': (
        'Attention proper \u2014 score, softmax, weighted sum. Not SDPA: '
        'gemma-4 sets score_scale = 1.0, so "scaled" would contradict the '
        'data.'
    ),
    'GELU': (
        "Gaussian error linear unit. The paper's name, all caps \u2014 an "
        'acronym is spelled the way it is spelled, like RoPE and RMSNorm.'
    ),
    'Add': 'Elementwise addition. Mathematics, not an implementation.',
    'Mul': 'Elementwise multiplication. Same.',
}

# A block is named by the display name of the category of its one node that
# is not structure: the word for "what this stretch does". Not the node's
# label, which says the level below, which algorithm (Multi-Query Attention,
# GeGLU): copying it would make "an Attention box in the Attention block".
CATEGORY_LABELS = types.MappingProxyType({
    'attention': 'Attention',
    'ffn': 'Feed Forward',
    'per_layer_embedding': 'Per-Layer Embedding',
})
# The categories of a block's nodes that are its structure, not what it does.
STRUCTURAL = ('norm', 'elementwise')


def dumps(document: dict[str, Any]) -> str:
  """A skeleton as its file holds it: compact JSON, UTF-8, no newline."""
  return json.dumps(document, ensure_ascii=False, separators=(',', ':'))


def load(path: pathlib.Path) -> dict[str, Any]:
  """A skeleton from its file, checked.

  Args:
      path: The skeleton's JSON file.

  Returns:
      The skeleton, parsed.

  Raises:
      ValueError: It breaks a rule of the schema; the text lists each.
  """
  found = json.loads(path.read_text(encoding='utf-8'))
  errors = check(found)
  if errors:
    raise ValueError(
        f'The skeleton {path.name} is not valid: ' + '; '.join(errors)
    )
  return found


def attr(
    entry: dict[str, Any], key: str, default: str | None = None
) -> str | None:
  """One of a Model Explorer node's or output's `{key, value}` attributes."""
  for pair in entry.get('attrs') or []:
    if pair['key'] == key:
      return pair['value']
  return default


def sources(node: dict[str, Any]) -> list[str]:
  """The tensors a node reads, as `<node>:<port>`."""
  return [
      f'{edge["sourceNodeId"]}:{edge.get("sourceNodeOutputId", "0")}'
      for edge in node.get('incomingEdges') or []
  ]


def anchor_label(definition: dict[str, Any], anchor: dict[str, Any]) -> str:
  """What an anchor of a layer definition is, by the layer's blocks.

  The layer's output names the block it ends too (`Layer output
  (Per-Layer Embedding block)`); an anchor inside the layer is its block's
  output (`Attention block output`); the layer's input is `Layer input`.
  One that is none of these is named by its `semantic`. The one name of
  an anchor wherever it is shown (tensor_analysis.anchor_labels(),
  counterparts.place_labels(), pytorch_binding.anchors()).

  Args:
      definition: The layer definition the anchor is of.
      anchor: The anchor, `{id, semantic, of, block?}`.

  Returns:
      The anchor's label.
  """
  block = anchor.get('block')
  if anchor['id'] in {output['id'] for output in definition.get('outputs', [])}:
    return f'Layer output ({block} block)' if block else 'Layer output'
  if block:
    return f'{block} block output'
  of = anchor.get('of', '').partition(':')[0]
  if of in {entry['id'] for entry in definition.get('inputs', [])}:
    return 'Layer input'
  return anchor.get('semantic', anchor['id'])


def blocks(
    nodes: list[dict[str, Any]], anchors: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
  """Each anchor's block: the nodes from the anchor before it to it.

  Following each node's output 0 downstream, the first node an anchor is
  of owns it; a block is named by the category of its one node that is
  not structure (mkjson.py's blocks_of()).

  Why something derived is stored in the skeleton (anchors' `block`,
  nodes' `namespace`): the derivation was once written in Python and in
  JavaScript, two renderers that had to agree, and drifted. Stored, it is
  read in one place, and check() derives it again and compares.

  Args:
      nodes: A layer definition's nodes, in topological order.
      anchors: The definition's anchors, `{id, semantic, of, ...}`.

  Returns:
      Per anchor that owns nodes, `{label, nodes}`, nodes in order.

  Raises:
      ValueError: A block has no or several nodes to name it by, or one
        of a category without a name.
  """
  by_id = {node['id']: node for node in nodes}
  owner = {
      anchor['of'].partition(':')[0]: anchor['id']
      for anchor in anchors
      if anchor['of'].endswith(':0')
  }
  readers = {}
  for node in nodes:
    for tensor in sources(node):
      readers.setdefault(tensor, []).append(node['id'])
  owned: dict[str, list[str]] = {}
  for node in nodes:
    current, seen = node['id'], set()
    while current and current not in seen:
      seen.add(current)
      if current in owner:
        owned.setdefault(owner[current], []).append(node['id'])
        break
      current = (readers.get(current + ':0') or [None])[0]
  found = {}
  for anchor_id, ids in owned.items():
    main = [i for i in ids if attr(by_id[i], 'category') not in STRUCTURAL]
    if len(main) != 1:
      raise ValueError(
          f'The block of anchor {anchor_id!r} has {len(main)} nodes '
          f'that are not structure ({main}); none names it.'
      )
    category = attr(by_id[main[0]], 'category')
    if category not in CATEGORY_LABELS:
      raise ValueError(f'The category {category!r} names no block.')
    found[anchor_id] = {'label': CATEGORY_LABELS[category], 'nodes': ids}
  return found


def check(document: dict[str, Any]) -> list[str]:
  """Every rule of the schema a skeleton breaks; [] if none.

  mkjson.py's check(), which runs check_anchor_criterion() too, rule for
  rule. Where mkjson.py stops at the first block it cannot name
  (blocks_of() exits), check() says so and goes on with the next
  definition.

  Args:
      document: The skeleton, parsed.

  Returns:
      The errors, one string each; [] if none.
  """
  errors = []
  for index, definition in enumerate(document['semantic_graph']):
    errors += _check_definition(f'defs[{index}]', definition)
  errors += check_anchor_criterion(document)
  errors += _check_layers(document)
  return errors


def _check_definition(where: str, definition: dict[str, Any]) -> list[str]:
  """Returns what is wrong with `definition`, each error prefixed `where`."""
  errors = []
  outputs: dict[str, int] = {}
  for item in definition['inputs']:
    if item['id'] in outputs:
      errors.append(f'{where}: input id {item["id"]!r} repeats.')
    outputs[item['id']] = 1  # A definition input is one output.
  for node in definition['nodes']:
    if node['id'] in outputs:
      errors.append(f'{where}: node id {node["id"]!r} repeats.')
    for tensor in sources(node):
      source, _, port = tensor.rpartition(':')
      if source not in outputs:
        errors.append(
            f'{where}: {node["id"]!r} reads {source!r}, which does '
            'not exist or comes after it (nodes are in topological '
            'order).'
        )
      elif int(port) >= outputs[source]:
        errors.append(
            f'{where}: {node["id"]!r} reads {tensor!r}, but '
            f'{source!r} has {outputs[source]} outputs.'
        )
    slots = [edge.get('targetNodeInputId') for edge in node['incomingEdges']]
    if len(set(slots)) != len(slots) or None in slots:
      errors.append(
          f'{where}: the input slots of {node["id"]!r} repeat or are '
          f'missing: {slots}.'
      )
    if not node['incomingEdges']:
      errors.append(
          f'{where}: {node["id"]!r} reads nothing (a definition input '
          'belongs in inputs).'
      )
    outputs[node['id']] = len(node['outputsMetadata'])
  read = set()
  for node in definition['nodes']:
    for tensor in sources(node):
      read.add(tensor.rsplit(':', 1)[0])
  anchored = {a['of'].rsplit(':', 1)[0] for a in definition['anchors']}
  for node in definition['nodes']:
    if node['id'] not in read and node['id'] not in anchored:
      errors.append(
          f'{where}: {node["id"]!r} is read by nothing and is no '
          'anchor: a dead node.'
      )
  for item in definition['inputs']:
    if item['id'] not in read and item['id'] not in anchored:
      errors.append(
          f'{where}: input {item["id"]!r} is read by nothing and is no '
          'anchor: a dead input.'
      )
  # The per-layer embedding stage multiplies by a tensor from outside the
  # layer (Gemma4TextDecoderLayer.forward's per_layer_input): it reads
  # exactly one input that is not the residual stream. mkjson.py at
  # 7b4f6db lacked that edge, and no structural rule stopped it: a graph
  # one edge short is a valid DAG all the same.
  external = {
      item['id']
      for item in definition['inputs']
      if item.get('source') != 'residual'
  }
  for node in definition['nodes']:
    if attr(node, 'category') != 'per_layer_embedding':
      continue
    found = [t for t in sources(node) if t.rsplit(':', 1)[0] in external]
    if len(found) != 1:
      errors.append(
          f'{where}: {node["id"]!r} is a per-layer embedding, which '
          'reads one per-layer input from outside the layer; it reads '
          f'{found}.'
      )
  try:
    want = blocks(definition['nodes'], definition['anchors'])
  except ValueError as error:
    return errors + [f'{where}: {error}']
  got = {a['id']: a['block'] for a in definition['anchors'] if 'block' in a}
  for anchor_id in sorted(set(want) | set(got)):
    if want.get(anchor_id, {}).get('label') != got.get(anchor_id):
      errors.append(
          f'{where}: anchor {anchor_id!r} names block '
          f'{got.get(anchor_id)!r}, but its nodes make it '
          f'{want.get(anchor_id, {}).get("label")!r}.'
      )
  namespaces = {n['id']: n.get('namespace') for n in definition['nodes']}
  for block in want.values():
    for node_id in block['nodes']:
      if namespaces.get(node_id) != block['label']:
        errors.append(
            f'{where}: the namespace of {node_id!r} is '
            f'{namespaces.get(node_id)!r}, but it is in block '
            f'{block["label"]!r}.'
        )
  stray = [i for i, namespace in namespaces.items() if not namespace]
  if stray:
    errors.append(f'{where}: these nodes are in no block: {stray}.')
  for node in definition['nodes']:
    for op in node.get('ops', []):
      if op['op'] not in SEMANTIC_OPS:
        errors.append(
            f'{where}: {node["id"]}.{op["id"]} runs {op["op"]!r}, '
            'which semantic_ops does not list.'
        )
  inputs = {item['id'] for item in definition['inputs']}
  anchor_ids = {a['id'] for a in definition['anchors']}
  for item in definition.get('outputs', []):
    if item['id'] not in anchor_ids:
      errors.append(
          f'{where}: outputs[{item["id"]!r}] has no anchor of its '
          'name; anchors[].of says which node port it is.'
      )
  for anchor in definition['anchors']:
    node_id, _, port = anchor['of'].partition(':')
    if node_id in inputs:
      continue
    if not port:
      errors.append(
          f'{where}: anchor {anchor["id"]!r} is of {anchor["of"]!r}, '
          'which lacks its port.'
      )
    elif node_id not in outputs:
      errors.append(
          f'{where}: anchor {anchor["id"]!r} is of {anchor["of"]!r}, '
          'which does not exist.'
      )
    elif int(port) >= outputs[node_id]:
      errors.append(
          f'{where}: anchor {anchor["id"]!r} is of {anchor["of"]!r}, '
          'an output its node does not have.'
      )
  if not definition['anchors']:
    errors.append(f'{where}: it has no anchor.')
  return errors


def _check_layers(document: dict[str, Any]) -> list[str]:
  """Returns what is wrong with the layers of `document`."""
  errors = []
  layers = document['layers']
  kv_steps = ('k_proj', 'v_proj', 'k_norm', 'v_norm', 'k_rope')
  for index, layer in enumerate(layers):
    where = f'layers[{index}]'
    definition = document['semantic_graph'][layer['def']]
    # What a layer may bind follows from its definition's inputs, not
    # from a list of names: `from` says where an input of this layer
    # comes from, a definition input (at another layer's anchor) or
    # `kv` (the K and V of another layer, which are inside attention,
    # neither input nor anchor, so only the layer is named). Every one
    # is optional: layer 0's input is made outside the skeleton, and a
    # layer that computes its own K and V has no kv.
    allowed = {
        item['id']
        for item in definition['inputs']
        if item.get('source') != 'tokens'
    } | {'kv'}
    bound = set(layer.get('from', {}))
    if not bound <= allowed:
      errors.append(
          f'{where} binds inputs that do not exist: {sorted(bound - allowed)}.'
      )
    # Both rotary and NoPE dims are kept, to be read; so they must add
    # up to the layer's head_dim.
    ops = ((layer.get('attrs') or {}).get('attn') or {}).get('ops') or {}
    dim = ((ops.get('q_proj') or {}).get('out') or {}).get('dim')
    for rope in ('q_rope', 'k_rope'):
      found = ops.get(rope) or {}
      if (
          'rotary_dims' in found
          and found['rotary_dims'] + found['nope_dims'] != dim
      ):
        errors.append(
            f'{where}.{rope}: rotary {found["rotary_dims"]} + nope '
            f'{found["nope_dims"]} is not head_dim {dim}.'
        )
    # from.kv is there iff attention's five K and V steps skip: two
    # places say one thing, and unchecked they would drift.
    skipped = {step for step in kv_steps if (ops.get(step) or {}).get('skip')}
    want = set(kv_steps) if 'kv' in layer.get('from', {}) else set()
    if skipped != want:
      errors.append(
          f'{where}: from.kv and the steps skipped disagree: from.kv '
          f'{"is" if want else "is not"} there, {sorted(skipped)} '
          'skip.'
      )
    for key, value in layer.get('from', {}).items():
      if 'layer' not in value:
        continue  # Made outside the skeleton.
      if not 0 <= value['layer'] < len(layers):
        errors.append(
            f'{where}.from.{key} names layer {value["layer"]}, '
            'which does not exist.'
        )
      elif value['layer'] >= index:
        errors.append(
            f'{where}.from.{key} names layer {value["layer"]}, not '
            'an earlier one.'
        )
  return errors


def check_anchor_criterion(document: dict[str, Any]) -> list[str]:
  """Every anchor that no runtime would be bound to keep; [] if none.

  An anchor stands if, in some layer of its definition, its tensor has
  two or more readers inside the layer and in other layers, or is read
  across a layer boundary; or, of the last definition, it is the model's
  output (mkjson.py's check_anchor_criterion()). The criterion is
  builder.anchor()'s, run: written only in a comment, it would drift.
  Readers are counted per layer, as instances, not per definition: an
  anchor is the definition's, its readers each layer's, and it stands if
  one layer meets a criterion.

  Args:
      document: The skeleton, parsed.

  Returns:
      The errors, one string per anchor that stands by no criterion.
  """
  definitions, layers = document['semantic_graph'], document['layers']
  errors = []
  for index, definition in enumerate(definitions):
    mine = [j for j, layer in enumerate(layers) if layer['def'] == index]
    for anchor in definition['anchors']:
      inside = sum(
          sources(node).count(anchor['of']) for node in definition['nodes']
      )
      stands = False
      for j in mine:
        across = 0
        for layer in layers:
          across += sum(
              1
              for value in (layer.get('from') or {}).values()
              if value.get('layer') == j and value.get('anchor') == anchor['id']
          )
        if inside + across >= 2 or across:
          stands = True
          break
      if not stands and index == len(definitions) - 1 and inside == 0:
        stands = True  # The model's output.
      if not stands:
        errors.append(
            f'defs[{index}] anchor {anchor["id"]!r} meets none of the '
            f'three criteria ({inside} readers inside the layer, '
            'none across a layer boundary, not the model output).'
        )
  return errors
