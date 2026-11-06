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

"""What an architecture's definition writes a skeleton with.

A definition (gemma4) builds each layer definition as a Graph, node by
node in forward order, names its anchors with anchor(), and words its nodes
by what a Hugging Face config.json says: the activation (activation()), the
attention's and the FFN's algorithm (attention_label(), ffn_label()), the
RoPE of a layer type (rope()). Shapes and the allocation per layer come from
the config; the connections are written by hand, for the data flow of
forward() cannot be inferred from the config or the module tree, only read
in the modeling code.
"""

import types
from typing import Any

from model_debugger.server.architectures import skeleton

# The symbolic batch and sequence dimensions of a shape.
BATCH_DIM, SEQ_DIM = 'B', 'S'
B, S = BATCH_DIM, SEQ_DIM

# What a node is shown as. Saying it is the data's business: kept in each
# renderer, it is kept once per renderer, and those copies drifted once.
LABELS = types.MappingProxyType({
    'RMSNorm': 'RMSNorm',
    'Attention': 'Attention',
    'FeedForward': 'Feed Forward',
    'PerLayerEmbedding': 'Per-Layer Embedding',
    'Add': 'Add',
    'Mul': 'Mul',
})

# The category of a stage. A stage, not an operator: operators differ from
# one implementation to another, and do not go into the architecture.
CATEGORIES = types.MappingProxyType({
    'RMSNorm': 'norm',
    'Attention': 'attention',
    'FeedForward': 'ffn',
    'PerLayerEmbedding': 'per_layer_embedding',
    'Add': 'elementwise',
    'Mul': 'elementwise',
})

# Hugging Face's ACT2FN keys are not architecture: the "pytorch" of
# "gelu_pytorch_tanh" names torch.nn.functional.gelu(approximate="tanh"), an
# implementation; as with dtype, a framework's spelling stays out. What
# stays is the mathematics: GELU by its tanh approximation or exact by erf.
# The two differ numerically, a difference across runtimes that is no
# matter of precision but a choice of implementation, and unsaid, it would
# mix into the diff. Split as ONNX's Gelu (approximate in {none, tanh}), as
# LiteRT and JAX split it.
ACTIVATIONS = types.MappingProxyType({
    'gelu': {'kind': 'gelu', 'approximate': 'none'},
    'gelu_pytorch_tanh': {'kind': 'gelu', 'approximate': 'tanh'},
    'gelu_new': {'kind': 'gelu', 'approximate': 'tanh'},
    'gelu_fast': {'kind': 'gelu', 'approximate': 'tanh'},
    'silu': {'kind': 'silu'},
    'swish': {'kind': 'silu'},
    'relu': {'kind': 'relu'},
})


def activation(config: dict[str, Any]) -> dict[str, Any]:
  """The activation a text config names, as mathematics.

  Args:
      config: The model's text config; its `hidden_activation` is read.

  Returns:
      The activation's entry in ACTIVATIONS, e.g. `{'kind': 'silu'}`.

  Raises:
      ValueError: ACTIVATIONS does not know it: add it there, rather than
        pass a framework's key into the architecture as it is.
  """
  name = config['hidden_activation']
  if name not in ACTIVATIONS:
    raise ValueError(
        f'Unknown activation {name!r}: add it to ACTIVATIONS rather than '
        "pass the framework's key into the architecture as it is."
    )
  return ACTIVATIONS[name]


def attention_label(heads: int, kv_heads: int) -> str:
  """The attention's algorithm, written out: an abbreviation asks the reader.

  kv_heads == 1 is Multi-Query, not Grouped-Query: GQA is 1 < kv_heads <
  heads. Gemma 4 E2B's num_key_value_heads is 1, and its
  num_global_key_value_heads None (global layers take the same), so all
  35 layers are MQA, as the text of Raschka's gallery card says too
  (where GQA is the filter's broad class: MQA is often taken for GQA of
  one group).

  Args:
      heads: The number of query heads.
      kv_heads: The number of key and value heads.

  Returns:
      'Multi-Head Attention', 'Multi-Query Attention' or 'Grouped-Query
      Attention'.
  """
  if kv_heads == heads:
    return 'Multi-Head Attention'
  if kv_heads == 1:
    return 'Multi-Query Attention'
  return 'Grouped-Query Attention'


def ffn_label(act: dict[str, Any], gated: bool) -> str:
  """The FFN's algorithm: gated with GELU is GeGLU, with SiLU SwiGLU."""
  if not gated:
    return 'MLP'
  return {'gelu': 'GeGLU', 'silu': 'SwiGLU'}.get(act['kind'], 'GLU')


def rope(
    config: dict[str, Any], layer_type: str, head_dim: int
) -> dict[str, Any]:
  """The RoPE parameters of a layer type: its variant, theta and dims.

  That it is RoPE the semantic operator says (op 'RoPE'); this says only
  the layer's parameters. A proportional RoPE rotates the first
  partial_rotary_factor of the head and leaves the rest without position
  (NoPE); note that it reads global_head_dim, which config.json does not
  show. The default rotates the whole head. theta is an integer when it
  is one: 10000.0 in a config is Python's and JSON's way of writing, and a
  RoPE base is an integer notion. rotary_dims + nope_dims is the head's
  dim; both are kept, and skeleton.check() asserts it.

  Args:
      config: The model's text config; its `rope_parameters` are read.
      layer_type: The key in `rope_parameters`, e.g. 'full_attention'.
      head_dim: The layer's head dimension.

  Returns:
      `{variant, theta, rotary_dims, nope_dims}`.
  """
  parameters = config['rope_parameters'][layer_type]
  kind = parameters['rope_type']
  if kind == 'proportional':
    rotary = 2 * int(
        parameters.get('partial_rotary_factor', 1.0) * head_dim // 2
    )
  else:
    rotary = head_dim  # compute_default_rope_parameters: all of it.
  theta = parameters['rope_theta']
  return {
      'variant': kind,
      'theta': int(theta) if float(theta).is_integer() else theta,
      'rotary_dims': rotary,
      'nope_dims': head_dim - rotary,
  }


def anchor(anchor_id: str, semantic: str, of: str) -> dict[str, Any]:
  """A point a comparison may be attached to.

  Its id is the name that holds across runtimes: two runtimes name their
  nodes differently, and their anchors alike, which is all an anchor is
  for. `of` is a tensor's name, `<producer>:<port>`, the port never left
  out (':0' is added if it is).

  The criterion is "no runtime can optimize it away", which is any of:

  1. Two or more readers within one layer instance: fused, it would be
     computed twice (intermediate.1 and .2 have two each inside a layer).
  2. It crosses a layer boundary: fusion happens within a layer, the
     layer is the unit of reuse, and its output must be materialized
     (`output`, the next layer's `input`).
  3. It is the model's output (logits).

  Readers are counted per layer instance, not per definition nor inside
  it: `output` has no reader inside its definition, and summed over the
  definition's layers it would have many, but per instance it has one,
  and it stands by criterion 2, not 1. An anchor is the definition's, its
  readers each layer's, so some layers meet a criterion that others do
  not; harmless, for an anchor allows a comparison, it does not require
  the tensor be written. skeleton.check_anchor_criterion() runs it.

  Args:
      anchor_id: The anchor's id, the name that holds across runtimes.
      semantic: What the anchored tensor is.
      of: The tensor's name, `<producer>` or `<producer>:<port>`.

  Returns:
      The anchor, `{id, semantic, of}`, `of` with its port.
  """
  if ':' not in of:
    of += ':0'
  return {'id': anchor_id, 'semantic': semantic, 'of': of}


class Graph:
  """A layer definition, built node by node in forward order.

  Shapes are given where a node is added, but a shape-preserving stage
  may take its input's (like()): working a shape out again is one more
  chance to get it wrong.
  """

  def __init__(self):
    """An empty definition: no input, no node."""
    self.nodes: list[dict[str, Any]] = []
    # The definition's inputs, not nodes: ONNX's GraphProto keeps node
    # and input apart too.
    self.inputs: list[dict[str, Any]] = []
    self._by_id: dict[str, dict[str, Any]] = {}
    # Each producer's outputs, `{name, shape}`. A node holds its shapes
    # as Model Explorer writes them (a string attribute of
    # outputsMetadata); building needs them to work out the shapes
    # downstream, so they are kept here as well, and not in the JSON.
    self._shapes: dict[str, list[dict[str, Any]]] = {}

  def add_input(
      self, input_id: str, shape: list[int | str], source: str, **extra
  ) -> str:
    """Adds a definition input, which takes no node's place.

    "It comes from the layer before" is no computation.

    Args:
        input_id: Its id.
        shape: Its shape.
        source: Where it comes from.
        **extra: Further fields of the input, kept as they are.

    Returns:
        Its id.
    """
    self.inputs.append(
        {'id': input_id, 'shape': shape, 'source': source, **extra}
    )
    self._by_id[input_id] = {'id': input_id}
    self._shapes[input_id] = [{'name': input_id + ':0', 'shape': shape}]
    return input_id

  def add(
      self,
      node_id: str,
      op: str,
      reads: list[str],
      shape: list[int | str],
      steps: list[tuple[str, str]] | None = None,
      label: str | None = None,
  ) -> str:
    """Adds a node: a stage, a Model Explorer GraphNode.

    The node is `{id, label, namespace, incomingEdges, outputsMetadata,
    attrs, ops}`. Its category has no field of Model Explorer's, so it
    is an attribute (whose values are strings); its namespace is filled
    in later with its block's name, by which Model Explorer folds a
    group, which is what a block is. `ops` is the schema's own: Model
    Explorer ignores it. There is no opType: the category says what
    kind of stage it is and `ops` what it does inside, and the field
    between them would say nothing more, and the same word twice on a
    leaf.

    Args:
        node_id: Its id.
        op: Its kind (LABELS, CATEGORIES); for a leaf, its semantic operator
          too.
        reads: What it reads: a producer, or `<producer>:<port>` to say which
          output of a producer of several.
        shape: Its output 0's shape.
        steps: The semantic operators that happen inside the stage, in order,
          `(id, op)`; by default one, of the node's id and op (a
          leaf: a norm, an add, a mul). Their parameters are not here: they
            differ per layer, in layers[].attrs[node].ops[step].
        label: What it is shown as; LABELS[op] by default.

    Returns:
        Its id.

    Raises:
        ValueError: A step's operator is not in skeleton.SEMANTIC_OPS.
        KeyError: It reads a producer not added before it.
    """
    outputs = [{'name': node_id + ':0', 'shape': shape}]
    steps = steps or [(node_id, op)]
    for _, step in steps:
      if step not in skeleton.SEMANTIC_OPS:
        raise ValueError(f'{step!r} is not in SEMANTIC_OPS.')
    node = {
        'id': node_id,
        'label': label or LABELS[op],
        'namespace': '',
        'incomingEdges': [
            {
                'sourceNodeId': tensor.partition(':')[0],
                'sourceNodeOutputId': tensor.rpartition(':')[2],
                'targetNodeInputId': str(slot),
            }
            for slot, tensor in enumerate(self._tensor(r) for r in reads)
        ],
        'outputsMetadata': [
            {
                'id': str(port),
                'attrs': [
                    {'key': 'tensor_name', 'value': output['name']},
                    {
                        'key': 'tensor_shape',
                        'value': ','.join(str(d) for d in output['shape']),
                    },
                ],
            }
            for port, output in enumerate(outputs)
        ],
        'attrs': [{'key': 'category', 'value': CATEGORIES[op]}],
        'ops': [{'id': step_id, 'op': step} for step_id, step in steps],
    }
    self.nodes.append(node)
    self._by_id[node_id] = node
    self._shapes[node_id] = outputs
    return node_id

  def output(self, ref: str) -> dict[str, Any]:
    """The output `{name, shape}` a producer or `<producer>:<port>` is.

    Args:
        ref: A producer, or `<producer>:<port>`; port 0 if none.

    Returns:
        The output, `{name, shape}`.

    Raises:
        KeyError: The producer is not added yet: nodes go in
          topological order.
    """
    producer, _, port = ref.partition(':')
    if producer not in self._by_id:
      raise KeyError(
          f'Input {ref!r} is not defined yet (nodes go in topological order).'
      )
    return self._shapes[producer][int(port or 0)]

  def _tensor(self, ref: str) -> str:
    """An input is a tensor's name: its producer and port.

    Its shape is the producer's to say; an edge once held a copy of the
    producer's whole output, shape and all, and check() needed a rule
    that the two did not drift, which showed the copy carried nothing.

    Args:
        ref: A producer, or `<producer>:<port>`; port 0 if none.

    Returns:
        The tensor's name, `<producer>:<port>`.
    """
    self.output(ref)  # Not there, or not in topological order: raise.
    producer, _, port = ref.partition(':')
    return f'{producer}:{int(port or 0)}'
