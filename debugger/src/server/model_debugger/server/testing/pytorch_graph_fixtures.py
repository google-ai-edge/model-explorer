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

"""An execution graph of Gemma 4, as the PyTorch runner exports it.

What pytorch-collector's to_model_explorer() draws of a forward pass of
Gemma4ForConditionalGeneration (transformers 5.17.0), written out operator
by operator as the real export of Gemma 4 E2B has them, for as many layers
as asked: the embedding and its scale, the per-layer inputs and the RoPE
tables before the layers; per layer its RMSNorms (nine ATen operators
each), its projections (four each: the weight's transpose, two views and
the matmul), RoPE of the query and key, the KV cache (a decode pass appends
to the one an earlier pass wrote, an input of its graph), attention, the
feed-forward network, the per-layer input and the layer scale, in place;
then the final norm and the output head. The class of each module, as the
runner names them. Shapes are E2B's.

Usage:
  document = pytorch_graph_fixtures.document(layers=35)
  data = pytorch_graph_fixtures.gzipped(document)
"""

import gzip
import json
from typing import Any

FILE = 'execution_graph.json.gz'
WIDTH = 1536
_TEXT = 'model.language_model'


class _Graph:
  """One forward pass's graph, its operators appended in order."""

  def __init__(self, signature: str, rows: int) -> None:
    self.signature = signature
    self.rows = rows
    self.nodes: list[dict[str, Any]] = []
    self.modules: dict[str, str] = {}
    self._next = 100

  def source(self, identifier: str, shape: str) -> tuple[str, str]:
    """A node of what the pass read from outside, under `inputs`."""
    self.nodes.append({
        'id': identifier,
        'label': identifier,
        'namespace': 'inputs',
        'attrs': [],
        'incomingEdges': [],
        'outputsMetadata': [_output(0, shape)],
    })
    return identifier, '0'

  def op(
      self,
      op: str,
      module: str,
      reads: list[tuple[str, str]],
      shape: str = '',
      attrs: tuple[tuple[str, str], ...] = (),
  ) -> tuple[str, str]:
    """Appends one operator call of `module`; returns its output."""
    self._next += 1
    identifier = str(self._next)
    label = op.rsplit('.', 1)[0] if op.count('.') >= 2 else op
    self.nodes.append({
        'id': identifier,
        'label': label,
        'namespace': module.replace('.', '/'),
        'attrs': [
            {'key': 'op', 'value': op},
            {'key': 'module', 'value': module},
            *({'key': k, 'value': v} for k, v in attrs),
        ],
        'incomingEdges': [
            {
                'sourceNodeId': node,
                'sourceNodeOutputId': output,
                'targetNodeInputId': str(index),
            }
            for index, (node, output) in enumerate(reads)
        ],
        'outputsMetadata': [
            _output(0, shape or f'bfloat16[1, {self.rows}, {WIDTH}]')
        ],
    })
    return identifier, '0'

  def module(self, path: str, kind: str) -> str:
    self.modules[path] = kind
    return path

  def rms_norm(self, path: str, x: tuple[str, str]) -> tuple[str, str]:
    """A Gemma4RMSNorm's nine operators."""
    self.module(path, 'Gemma4RMSNorm')
    f = self.op('aten._to_copy.default', path, [x])
    squared = self.op('aten.pow.Tensor_Scalar', path, [f])
    mean = self.op('aten.mean.dim', path, [squared])
    eps = self.op('aten.add.Tensor', path, [mean])
    rsqrt = self.op('aten.pow.Tensor_Scalar', path, [eps])
    scaled = self.op('aten.mul.Tensor', path, [f, rsqrt])
    weight = self.op(
        'aten._to_copy.default',
        path,
        [],
        attrs=(('args.0', f'param: {path}.weight bfloat16[{WIDTH}]'),),
    )
    weighted = self.op('aten.mul.Tensor', path, [scaled, weight])
    return self.op('aten._to_copy.default', path, [weighted])

  def linear(self, path: str, x: tuple[str, str]) -> tuple[str, str]:
    """A Linear's four operators."""
    self.module(path, 'Linear')
    weight = self.op(
        'aten.t.default',
        path,
        [],
        attrs=(('args.0', f'param: {path}.weight bfloat16[{WIDTH}, 2048]'),),
    )
    flat = self.op('aten.view.default', path, [x])
    product = self.op('aten.mm.default', path, [flat, weight])
    return self.op('aten._unsafe_view.default', path, [product])

  def gelu(self, path: str, x: tuple[str, str]) -> tuple[str, str]:
    self.module(path, 'GELUTanh')
    return self.op('aten.gelu.default', path, [x])

  def rope(
      self,
      path: str,
      x: tuple[str, str],
      cos: tuple[str, str],
      sin: tuple[str, str],
  ) -> tuple[str, str]:
    """RoPE of a query or key: ATen operators of the attention's own."""
    c = self.op('aten.unsqueeze.default', path, [cos])
    s = self.op('aten.unsqueeze.default', path, [sin])
    straight = self.op('aten.mul.Tensor', path, [x, c])
    first = self.op('aten.slice.Tensor', path, [x])
    second = self.op('aten.slice.Tensor', path, [x])
    negated = self.op('aten.neg.default', path, [second])
    rotated = self.op('aten.cat.default', path, [negated, first])
    turned = self.op('aten.mul.Tensor', path, [rotated, s])
    summed = self.op('aten.add.Tensor', path, [straight, turned])
    return self.op('aten.transpose.int', path, [summed])

  def layer(
      self,
      index: int,
      x: tuple[str, str],
      per_layer: tuple[str, str],
      cos: tuple[str, str],
      sin: tuple[str, str],
      mask: tuple[str, str],
  ) -> tuple[str, str]:
    """One decoder layer; returns its output."""
    path = self.module(f'{_TEXT}.layers.{index}', 'Gemma4TextDecoderLayer')
    attn = self.module(f'{path}.self_attn', 'Gemma4TextAttention')
    normed = self.rms_norm(f'{path}.input_layernorm', x)
    q = self.op(
        'aten.view.default', attn, [self.linear(f'{attn}.q_proj', normed)]
    )
    q = self.rope(attn, self.rms_norm(f'{attn}.q_norm', q), cos, sin)
    k = self.op(
        'aten.view.default', attn, [self.linear(f'{attn}.k_proj', normed)]
    )
    v = self.op(
        'aten.view.default', attn, [self.linear(f'{attn}.v_proj', normed)]
    )
    k = self.rope(attn, self.rms_norm(f'{attn}.k_norm', k), cos, sin)
    v = self.op(
        'aten.transpose.int', attn, [self.rms_norm(f'{attn}.v_norm', v)]
    )
    if self.signature == 'decode':
      cache = f'kwargs.past_key_values.layers.{index}'
      k = self.op(
          'aten.cat.default',
          attn,
          [self.source(f'{cache}.keys', 'bfloat16[1, 1, 12, 256]'), k],
      )
      v = self.op(
          'aten.cat.default',
          attn,
          [self.source(f'{cache}.values', 'bfloat16[1, 1, 12, 256]'), v],
      )
    keys = self.op(
        'aten.view.default',
        attn,
        [self.op('aten.expand.default', attn, [k])],
    )
    values = self.op(
        'aten.view.default',
        attn,
        [self.op('aten.expand.default', attn, [v])],
    )
    scores = self.op(
        'aten.bmm.default',
        attn,
        [
            self.op('aten.view.default', attn, [q]),
            self.op('aten.transpose.int', attn, [keys]),
        ],
    )
    masked = self.op('aten.add.Tensor', attn, [scores, mask])
    weights = self.op('aten._softmax.default', attn, [masked])
    mixed = self.op('aten.bmm.default', attn, [weights, values])
    flat = self.op(
        'aten.view.default',
        attn,
        [self.op('aten.transpose.int', attn, [mixed])],
    )
    attended = self.rms_norm(
        f'{path}.post_attention_layernorm',
        self.linear(f'{attn}.o_proj', flat),
    )
    residual = self.op('aten.add.Tensor', path, [x, attended])
    mlp = self.module(f'{path}.mlp', 'Gemma4TextMLP')
    normed = self.rms_norm(f'{path}.pre_feedforward_layernorm', residual)
    gate = self.gelu(f'{mlp}.act_fn', self.linear(f'{mlp}.gate_proj', normed))
    up = self.linear(f'{mlp}.up_proj', normed)
    fed = self.linear(
        f'{mlp}.down_proj', self.op('aten.mul.Tensor', mlp, [gate, up])
    )
    residual = self.op(
        'aten.add.Tensor',
        path,
        [
            residual,
            self.rms_norm(f'{path}.post_feedforward_layernorm', fed),
        ],
    )
    gated = self.gelu(
        f'{path}.act_fn',
        self.linear(f'{path}.per_layer_input_gate', residual),
    )
    mixed = self.op('aten.mul.Tensor', path, [gated, per_layer])
    projected = self.rms_norm(
        f'{path}.post_per_layer_input_norm',
        self.linear(f'{path}.per_layer_projection', mixed),
    )
    summed = self.op('aten.add.Tensor', path, [residual, projected])
    return self.op(
        'aten.mul_.Tensor',
        path,
        [summed],
        attrs=(('other', f'buffer: {path}.layer_scalar bfloat16[1]'),),
    )


def _output(index: int, shape: str) -> dict[str, Any]:
  return {
      'id': str(index),
      'attrs': [{'key': 'tensor_shape', 'value': shape}],
  }


def graph(signature: str, layers: int) -> tuple[dict[str, Any], dict[str, str]]:
  """One signature's graph and its modules' classes."""
  rows = 12 if signature == 'prefill' else 1
  g = _Graph(signature, rows)
  g.module('', 'Gemma4ForConditionalGeneration')
  g.module('model', 'Gemma4Model')
  g.module(_TEXT, 'Gemma4TextModel')
  g.module(f'{_TEXT}.layers', 'ModuleList')
  ids = g.source('kwargs.input_ids', f'int64[1, {rows}]')
  embed = g.module(f'{_TEXT}.embed_tokens', 'Gemma4TextScaledWordEmbedding')
  x = g.op(
      'aten.mul.Tensor', embed, [g.op('aten.embedding.default', embed, [ids])]
  )
  per_layer = g.op(
      'aten.view.default',
      _TEXT,
      [g.linear(f'{_TEXT}.per_layer_model_projection', x)],
  )
  rotary = g.module(f'{_TEXT}.rotary_emb', 'Gemma4TextRotaryEmbedding')
  cos = g.op('aten.cos.default', rotary, [ids])
  sin = g.op('aten.sin.default', rotary, [ids])
  mask = g.op('aten.where.self', _TEXT, [ids])
  for index in range(layers):
    mine = g.op('aten.select.int', _TEXT, [per_layer])
    x = g.layer(index, x, mine, cos, sin, mask)
  head = g.linear('lm_head', g.rms_norm(f'{_TEXT}.norm', x))
  g.nodes.append({
      'id': 'output.logits',
      'label': 'output.logits',
      'namespace': 'outputs',
      'attrs': [],
      'incomingEdges': [{
          'sourceNodeId': head[0],
          'sourceNodeOutputId': head[1],
          'targetNodeInputId': '0',
      }],
      'outputsMetadata': [],
  })
  return {'id': signature, 'nodes': g.nodes}, g.modules


def document(
    layers: int = 35, signatures: tuple[str, ...] = ('prefill', 'decode')
) -> dict[str, Any]:
  """The document a turn's execution_graph.json.gz holds."""
  graphs, modules = [], {}
  for signature in signatures:
    found, classes = graph(signature, layers)
    graphs.append(found)
    modules.update(classes)
  return {
      'format': 1,
      'hide_unused': True,
      'graphs': graphs,
      'steps': {s: 13 if s == 'prefill' else 14 for s in signatures},
      'modules': modules,
      'seconds': 0.1,
  }


def gzipped(found: dict[str, Any]) -> bytes:
  """The document as the runner writes it."""
  return gzip.compress(json.dumps(found).encode('utf-8'), mtime=0)
