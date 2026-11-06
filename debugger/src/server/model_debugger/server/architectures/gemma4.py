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

"""Gemma 4's architecture: its decoder stack's skeleton, from its config.

build() makes the skeleton of a Gemma 4 text model (Hugging Face model_type
gemma4_text) from its text config: one layer definition, the decoder, that
every layer uses, and each layer's parameters. Shapes and the allocation
per layer are the config's; the connections are written here, read from
the modeling code:

    transformers 5.14.1  models/gemma4/modeling_gemma4.py
      Gemma4TextDecoderLayer.forward   L1405   (PLE multiplies by the
                                                per_layer_input from
                                                outside at L1456)
      Gemma4TextAttention.forward      L1235   (is_kv_shared_layer branch
                                                at L1257)
      Gemma4TextMLP.__init__           L1074   (double-wide at L1079)
      Gemma4TextModel.forward          L1653   (the PLE chain L1744/L1788,
                                                sliced per layer L1723)
      Gemma4ForCausalLM.forward        L1843   (softcap at L1896)
    modeling_rope_utils.py             L187    (_compute_proportional_rope_
                                                parameters)

The skeleton holds the decoder stack alone. The embedding (the main
embedding and the PLE chain) and the output (final norm, LM head, softcap)
are not in it: each is a graph of its own, used once, sharing nothing with
the decoder, and what the skeleton answers is "from which step do these
layers differ". The cost: no anchor at the embedding's output, the PLE
table or the logits, so the inputs at the two ends of the stack have no
`from` (their producers are outside the skeleton): layer 0's input, and
each layer's per_layer_input, which ple reads as its input 1 and whose
slice is attrs.ple.ops.gate.slice. (The embedding and output definitions
were dropped from mkjson.py once nothing called them.)
"""

from typing import Any

from model_debugger.server.architectures import builder
from model_debugger.server.architectures import skeleton

B, S = builder.BATCH_DIM, builder.SEQ_DIM

# The semantic operators that happen inside each stage, in order: the
# semantic level, the steps every implementation would agree on. How many
# operators a runtime makes of them, and which it fuses, is the execution
# graph's business. (By contrast: RoPE is one step here, two operators in
# LiteRT, Sub and Add; q/k/v_norm are three steps here, and LiteRT's probe
# found none.)
_ATTENTION = [
    ('q_proj', 'Linear'),
    ('k_proj', 'Linear'),
    ('v_proj', 'Linear'),
    ('q_norm', 'RMSNorm'),
    ('k_norm', 'RMSNorm'),
    ('v_norm', 'RMSNorm'),
    ('q_rope', 'RoPE'),
    ('k_rope', 'RoPE'),
    ('attend', 'Attention'),
    ('o_proj', 'Linear'),
]
_FFN = [
    ('gate_proj', 'Linear'),
    ('up_proj', 'Linear'),
    ('act', 'GELU'),
    ('gate', 'Mul'),
    ('down_proj', 'Linear'),
]
_PLE = [
    ('gate_proj', 'Linear'),
    ('act', 'GELU'),
    ('gate', 'Mul'),
    ('proj', 'Linear'),
]


def decoder_definition(config: dict[str, Any]) -> dict[str, Any]:
  """The architecture graph of one decoder layer.

  Its nodes are stages, not operators: operators differ per
  implementation (one layer is 32 operators in PyTorch, 100 in LiteRT,
  with RoPE split in two), and putting them in the architecture would
  take one implementation for the model. A field stays in the
  architecture iff no runtime could change it.

  Every layer uses this one definition. Gemma 4's decoder layers come in
  four configurations (sliding or full attention, computing or borrowing
  K and V), and their graph is one: the same 12 nodes, the same residual
  topology, the same anchors. What differs is a few numbers (head_dim,
  RoPE, window, the FFN's width), which are that layer's, not another
  graph, so they are in layers[].attrs, a row per layer. Four definitions
  for the four configurations were once four identical graphs, which a
  reader had to compare field by field to see how they differed.

  KV is not in the graph; the PLE input is. Reading and writing the KV
  cache happens inside attention, and a stage does not spread out its
  insides: k_cache.in/out as definition inputs and anchors once put half
  of attention's inside at the layer's level, which is operator
  granularity. PLE differs: in forward, the per-layer embedding stage
  multiplies by a tensor handed in from outside the layer
  (`hidden_states * per_layer_input`, L1456), a dependency of the stage,
  not its inside. mkjson.py at 7b4f6db dropped it with ple.in, which left
  the residual stream as the only input and ple seeming to read add_ffn
  alone, which is wrong (the source review correction of 2026-09-06
  restored that edge). So the definition has two inputs: the residual
  stream `input`, and PLE's `per_layer_input` (ple's input 1), whose
  producer is outside the skeleton (the embedding's PLE chain,
  L1744/L1788, sliced per layer, L1723), so it has no `from`. Which
  layer's K and V a layer borrows and which slice it takes cannot be
  derived from elsewhere: they are those stages' attributes, in
  layers[].from.kv and layers[].attrs.ple.ops.gate.slice.

  No parameter stays on the definition (see layer_attrs()): it says only
  the structure, the nodes, how they connect, the anchors and the names.
  The sandwich norms are nodes, not folded into attrs. The reason once
  given for folding them, that the fan-out criterion cannot cut there,
  governs anchors (which tensors must be materialized), not how nodes are
  cut; and the other, that LiteRT fuses the post-norm, puts a runtime's
  fact into the architecture (and that fact was wrong). The architecture
  says what the model is; whether a runtime fuses is the execution
  graph's. Gemma 4 has five per layer: on both sides of attention and of
  the FFN, and one after PLE.

  Args:
      config: The model's text config.

  Returns:
      `{kind, inputs, outputs, nodes, anchors}`.
  """
  hidden = config['hidden_size']
  per_layer = config['hidden_size_per_layer_input']
  heads = config['num_attention_heads']
  kv_heads = config['num_key_value_heads']

  graph = builder.Graph()
  # The block's input. Its name says its role, not where it comes from:
  # once called resid.0, it lied for 1 layer in 35, for the first
  # decoder's input is the embedding's output, no residual sum. Where it
  # comes from is `source`'s to say, which cannot be wrong. It is a port,
  # not an anchor: the layer before declared the same tensor, and a second
  # declaration would have a diff count it twice.
  graph.add_input('input', [B, S, hidden], 'residual')
  # The second input: the slice of per-layer embedding PLE multiplies by
  # (forward L1456). A port too, not an anchor: ple is its one reader in
  # the layer, and it crosses no layer (each layer has its own slice), so
  # it meets none of the three criteria; to be compared, it must be at
  # the embedding that makes it.
  graph.add_input(
      'per_layer_input',
      [B, S, per_layer],
      'external per-layer embedding',
      label='Per-layer input',
  )

  graph.add(
      'norm.attn.in',
      'RMSNorm',
      ['input'],
      [B, S, hidden],
      label='Pre-RMSNorm',
  )
  graph.add(
      'attn',
      'Attention',
      ['norm.attn.in'],
      [B, S, hidden],
      _ATTENTION,
      label=builder.attention_label(heads, kv_heads),
  )
  graph.add(
      'norm.attn.out',
      'RMSNorm',
      ['attn'],
      [B, S, hidden],
      label='Post-RMSNorm',
  )
  graph.add('add_attn', 'Add', ['input', 'norm.attn.out'], [B, S, hidden])

  graph.add(
      'norm.ffn.in',
      'RMSNorm',
      ['add_attn'],
      [B, S, hidden],
      label='Pre-RMSNorm',
  )
  graph.add(
      'ffn',
      'FeedForward',
      ['norm.ffn.in'],
      [B, S, hidden],
      _FFN,
      label=builder.ffn_label(builder.activation(config), True),
  )
  graph.add(
      'norm.ffn.out',
      'RMSNorm',
      ['ffn'],
      [B, S, hidden],
      label='Post-RMSNorm',
  )
  graph.add('add_ffn', 'Add', ['add_attn', 'norm.ffn.out'], [B, S, hidden])

  # ple reads two: the residual stream (input 0, where gate and projection
  # start) and per_layer_input (input 1, multiplied in after the gate).
  graph.add(
      'ple',
      'PerLayerEmbedding',
      ['add_ffn', 'per_layer_input'],
      [B, S, hidden],
      _PLE,
  )
  graph.add(
      'norm.ple.out',
      'RMSNorm',
      ['ple'],
      [B, S, hidden],
      label='Post-RMSNorm',
  )
  # Its only reader is layer_scale, so it may be fused into an FMA: it
  # means something but cannot hold a diff, so it is no anchor.
  graph.add('add_ple', 'Add', ['add_ffn', 'norm.ple.out'], [B, S, hidden])
  # A learned per-layer scalar, not 1: the layer's output is not the PLE
  # sum.
  graph.add('layer_scale', 'Mul', ['add_ple'], [B, S, hidden])

  # Four anchors, all on the residual trunk. Not resid.N: resid.3 is the
  # output of the layer_scale multiply, no residual sum, the reason
  # resid.0 went. K and V were anchors once (they do cross layers), which
  # took attention's inside for the layer's interface. `input` is an
  # anchor too: it is the same tensor as the layer before's output, but
  # whether this layer can hold a diff should not depend on whether the
  # layer before declared it; layer 0's input comes from outside the
  # skeleton (the embedding is not here), and unnamed it could not be
  # compared. layers[].from says which output an input is bound to.
  anchors = [
      builder.anchor('input', 'layer input', 'input'),
      builder.anchor('intermediate.1', 'residual after attention', 'add_attn'),
      builder.anchor('intermediate.2', 'residual after FFN', 'add_ffn'),
      builder.anchor('output', 'layer output', 'layer_scale'),
  ]
  # The outputs are listed, as the inputs are (ONNX's GraphProto and
  # Model Explorer both have two lists). Unlisted, which output is the
  # trunk's could only be found backwards from layers[].from, "who is
  # bound to another's input", which the last layer is not; and a copy of
  # that search once named the wrong port and found None, unnoticed. The
  # id is that of the anchor of the same name, which says which port.
  outputs = [{'id': 'output', 'shape': [B, S, hidden]}]
  # Each block becomes its nodes' namespace (Model Explorer folds slash
  # paths into groups, which is what a block is); the anchors keep only
  # its name, for each node's namespace says whose it is. `input` has no
  # block before it.
  by_id = {node['id']: node for node in graph.nodes}
  for anchor_id, block in skeleton.blocks(graph.nodes, anchors).items():
    next(a for a in anchors if a['id'] == anchor_id)['block'] = block['label']
    for node_id in block['nodes']:
      by_id[node_id]['namespace'] = block['label']
  return {
      'kind': 'decoder',
      'inputs': graph.inputs,
      'outputs': outputs,
      'nodes': graph.nodes,
      'anchors': anchors,
  }


def layer_attrs(
    config: dict[str, Any], index: int, span: str, kv_from: int | None
) -> dict[str, Any]:
  """The parameters of each node and semantic operator of one layer.

  Keyed by node id, then by operator id. All of them are here, none on
  the definition. Some are the same in every layer (eps, activation) and
  are repeated in each all the same: a layer's parameters are then in one
  place, and a renderer need not merge two.

  A layer that borrows K and V does not compute them: its five K and V
  steps (k_proj, v_proj, k_norm, v_norm, k_rope) skip, for K and V come
  from another layer's cache, normalized and rotated already. Why it lacks
  those steps is not said here but by layers[].from.kv, and
  skeleton.check() asserts the two agree.

  Args:
      config: The model's text config.
      index: The layer's index.
      span: 'full' or 'sliding' attention.
      kv_from: The layer whose K and V it borrows; None if its own.

  Returns:
      The layer's parameters, keyed by node id, then by operator id.
  """
  own = kv_from is None
  epsilon, activation = config['rms_norm_eps'], builder.activation(config)
  layer_type = 'full_attention' if span == 'full' else 'sliding_attention'
  head_dim = config['global_head_dim'] if span == 'full' else config['head_dim']
  hidden = config['hidden_size']
  per_layer = config['hidden_size_per_layer_input']
  heads = config['num_attention_heads']
  kv_heads = config['num_key_value_heads']
  intermediate = config['intermediate_size'] * (
      2 if config['use_double_wide_mlp'] and not own else 1
  )
  rope = builder.rope(config, layer_type, head_dim)
  skip = {} if own else {'skip': True}
  # RMSNorm is two steps: normalize, then multiply by a learned gain per
  # channel. The second is optional (Gemma4RMSNorm(with_scale=...)): seven
  # of the eight norms have it, v_norm alone does not, so all eight say
  # learned_scale, rather than encode "unsaid means yes".
  rms = {'eps': epsilon, 'learned_scale': True}
  return {
      'norm.attn.in': {'ops': {'norm.attn.in': {**rms}}},
      'attn': {
          'ops': {
              # out: {heads, dim} for a multi-head projection, a number
              # for a flat one. out = heads * dim follows, so it is not
              # kept.
              'q_proj': {'out': {'heads': heads, 'dim': head_dim}},
              'k_proj': {'out': {'heads': kv_heads, 'dim': head_dim}, **skip},
              'v_proj': {'out': {'heads': kv_heads, 'dim': head_dim}, **skip},
              'q_norm': {**rms},
              'k_norm': {**rms, **skip},
              # v_norm is the one norm of the eight with learned_scale
              # False: a gain would pass through the weighted sum and be
              # absorbed whole by o_proj right after (a column scaling),
              # so learning it would be redundant. The normalization
              # stays: it bounds the range of the V written to the cache.
              'v_norm': {**rms, 'learned_scale': False, **skip},
              'q_rope': rope,
              'k_rope': {**rope, **skip},
              # score_scale 1.0: Gemma 4 does not divide by sqrt(d) (the
              # textbook's 1/sqrt(head_dim)). The scale is folded into
              # q_norm's learned gain: RoPE between them is a rotation,
              # rope(c*q) = c*rope(q), so the scalar passes through to the
              # scores as it is.
              'attend': {
                  'span': (
                      {'kind': 'sliding', 'window': config['sliding_window']}
                      if span == 'sliding'
                      else {'kind': 'global'}
                  ),
                  'score_scale': 1.0,
              },
              'o_proj': {'out': hidden},
          }
      },
      'norm.attn.out': {'ops': {'norm.attn.out': {**rms}}},
      'add_attn': {'ops': {'add_attn': {}}},
      'norm.ffn.in': {'ops': {'norm.ffn.in': {**rms}}},
      'ffn': {
          'ops': {
              'gate_proj': {'out': intermediate},
              'up_proj': {'out': intermediate},
              'act': activation,
              'gate': {},
              'down_proj': {'out': hidden},
          }
      },
      'norm.ffn.out': {'ops': {'norm.ffn.out': {**rms}}},
      'add_ffn': {'ops': {'add_ffn': {}}},
      'ple': {
          'ops': {
              'gate_proj': {'out': per_layer},
              'act': activation,
              # It multiplies by per_layer_input, slice `index` of the PLE
              # table. The embedding makes the table (not in this
              # skeleton); which slice is this layer's fact, and cannot
              # be derived.
              'gate': {'slice': index},
              'proj': {'out': hidden},
          }
      },
      'norm.ple.out': {'ops': {'norm.ple.out': {**rms}}},
      'add_ple': {'ops': {'add_ple': {}}},
      # A learned per-layer scalar, not 1.
      'layer_scale': {'ops': {'layer_scale': {'learned_scalar': True}}},
  }


def build(config: dict[str, Any]) -> dict[str, Any]:
  """The skeleton of a Gemma 4 text model, from its text config.

  Args:
      config: The text config (config.json's `text_config`).

  Returns:
      `{semantic_ops, semantic_graph, layers}`: the vocabulary, sent with
      the data so that a reader of the raw JSON need not read the code to
      know what RoPE means; the one decoder definition; and per layer
      `{def, attrs, from}`.
  """
  count, types = config['num_hidden_layers'], config['layer_types']
  first_shared = count - config.get('num_kv_shared_layers', 0)
  computed = types[:first_shared]
  # Per layer type, the last layer that computes its own K and V: the one
  # a layer of that type that borrows them borrows from.
  owner = {
      t: len(computed) - 1 - computed[::-1].index(t)
      for t in set(types)
      if t in computed
  }
  layers = []
  for index in range(count):
    span = 'full' if types[index] == 'full_attention' else 'sliding'
    borrows = index >= first_shared > 0
    # Each cross-layer input is another layer's anchor, named, not just
    # a layer number. Layer 0's residual input comes from outside the
    # skeleton. per_layer_input is never in `from`: its producer is
    # outside (the embedding makes the PLE table), and its slice is in
    # attrs.ple. kv is there only for a layer that borrows, and has no
    # anchor (K and V are inside attention, no named tensor), so it has
    # a field fewer than input: `from` entries come in two shapes.
    sources = {'input': {'layer': index - 1, 'anchor': 'output'}}
    sources = sources if index else {}
    if borrows:
      sources['kv'] = {'layer': owner[types[index]]}
    # Four configurations, one definition: the graph is one, and the
    # few numbers that differ are the layer's own.
    entry = {
        'def': 0,
        'attrs': layer_attrs(
            config, index, span, owner[types[index]] if borrows else None
        ),
    }
    if sources:
      entry['from'] = sources
    layers.append(entry)
  return {
      'semantic_ops': skeleton.SEMANTIC_OPS,
      'semantic_graph': [decoder_definition(config)],
      'layers': layers,
  }
