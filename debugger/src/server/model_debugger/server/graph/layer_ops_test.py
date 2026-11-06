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

"""The operators inside Gemma 4 E2B's layers, of both runtimes.

The points are named as gemma-4-E2B-it.litertlm names them (its scan).
"""

import json
from typing import Any

from absl.testing import absltest
from model_debugger.server.architectures import catalog
from model_debugger.server.graph import architecture
from model_debugger.server.graph import layer_ops
from model_debugger.server.graph import pytorch_binding
from model_debugger.server.tap import tap_scan
from model_debugger.server.testing import test_main

_STACK = (
    'LanguageModel.decode_graph/LanguageModel.transformer_stack/'
    'transformer.transformer/layer_{0}/layer_{0}.'
)
_PRE = 'pre_qkv/attn.pre_qkv/attn._pre_attention_fn/'


def _point(
    op: int,
    layer: int,
    name: str,
    operator: str,
    shape: tuple[int, ...],
    dtype: str = 'float32',
    signature: str = 'decode',
) -> tap_scan.Point:
  """A point of layer `layer`, its name after `layer_<N>.`."""
  return tap_scan.Point(
      id=f'{signature}:{op}:0',
      signature=signature,
      subgraph=0,
      op=op,
      output=0,
      tensor=op,
      tensor_name=_STACK.format(layer) + name,
      shape=shape,
      dtype=dtype,
      operator=operator,
      output_name=None,
      context_axes=(),
      reason='',
      layer=layer,
      layer_stem='layer',
  )


def _document() -> dict[str, Any]:
  """The skeleton of E2B's pinned config.json."""
  (gemma4,) = [a for a in catalog.architectures() if a.id == 'gemma4']
  entry = gemma4.entry('gemma-4-E2B-it')
  path = gemma4.directory / entry['pinned']['file']
  _, document = catalog.generate(json.loads(path.read_text('utf-8')))
  return document


def _rules() -> dict[str, Any]:
  (known,) = [a for a in architecture.architectures() if a.id == 'gemma4']
  return known.binding['ops']


class LayerOpsTest(absltest.TestCase):
  """The operators of E2B, its LiteRT-LM points and PyTorch modules."""

  def test_the_operators_are_the_skeletons_by_block(self):
    """Each block's, in its order, labelled as the places are."""
    ops = layer_ops.operators(_document(), _rules())

    self.assertEqual(
        [(op.block, op.name, op.kind) for op in ops][:9],
        [
            ('Attention', 'Pre-RMSNorm', 'RMSNorm'),
            ('Attention', 'q_proj', 'Linear'),
            ('Attention', 'k_proj', 'Linear'),
            ('Attention', 'v_proj', 'Linear'),
            ('Attention', 'q_norm', 'RMSNorm'),
            ('Attention', 'k_norm', 'RMSNorm'),
            ('Attention', 'v_norm', 'RMSNorm'),
            ('Attention', 'o_proj', 'Linear'),
            ('Attention', 'Post-RMSNorm', 'RMSNorm'),
        ],
    )
    self.assertEqual(
        [op.id for op in ops][9:],
        [
            'norm.ffn.in',
            'ffn/gate_proj',
            'ffn/up_proj',
            'ffn/act',
            'ffn/gate',
            'ffn/down_proj',
            'norm.ffn.out',
            'ple/gate_proj',
            'ple/act',
            'ple/gate',
            'ple/proj',
            'norm.ple.out',
        ],
    )
    self.assertEqual(ops[1].label, 'Attention q_proj')
    # RoPE and the attention are no module's output: not offered.
    self.assertNotIn('attn/q_rope', {op.id for op in ops})

  def test_a_litert_point_is_the_float_output_in_its_scope(self):
    """The DEQUANTIZE, not the QUANTIZE or the int8 FULLY_CONNECTED."""
    points = [
        _point(
            238,
            3,
            _PRE + 'q_einsum/composite',
            'QUANTIZE',
            (1, 1, 1536),
            'int8',
        ),
        _point(
            239,
            3,
            _PRE + 'q_einsum/reshape',
            'FULLY_CONNECTED',
            (1, 1, 2048),
            'int8',
        ),
        _point(
            241,
            3,
            _PRE + 'q_einsum/composite1',
            'DEQUANTIZE',
            (1, 1, 8, 256),
        ),
        _point(
            41,
            0,
            _PRE + 'q_einsum/composite1',
            'DEQUANTIZE',
            (1, 1, 8, 256),
        ),
        _point(
            9,
            0,
            _PRE + 'q_einsum/composite1',
            'DEQUANTIZE',
            (1, 128, 8, 256),
            signature='prefill_128',
        ),
        _point(
            3,
            0,
            _PRE + 'q_einsum/composite1',
            'DEQUANTIZE',
            (1, 4, 8, 256),
            signature='verify',
        ),
    ]

    found = layer_ops.litert_points(
        points, _rules(), ['attn/q_proj'], ['decode', 'prefill_128']
    )

    self.assertEqual(
        {
            signature: {layer: p.id for layer, p in by_layer.items()}
            for signature, by_layer in found['attn/q_proj'].items()
        },
        {
            'decode': {0: 'decode:41:0', 3: 'decode:241:0'},
            'prefill_128': {0: 'prefill_128:9:0'},
        },
    )
    # 8 × 256 float32 per token.
    self.assertEqual(
        layer_ops.token_bytes(found['attn/q_proj']['decode'][3]), 8192
    )

  def test_two_points_of_one_operator_in_a_layer_are_not_certain(self):
    """Both are left out: which one is the operator is not certain."""
    points = [
        _point(1, 0, 'post_qkv/mlp/mul', 'MUL', (1, 1, 6144)),
        _point(2, 0, 'post_qkv/mlp/mul1', 'MUL', (1, 1, 6144)),
    ]

    found = layer_ops.litert_points(points, _rules(), ['ffn/gate'], ['decode'])

    self.assertEqual(found, {'ffn/gate': {'decode': {}}})

  def test_a_kv_shared_layer_has_no_k_proj_and_says_whose_it_reuses(self):
    """Layers 15-34 reuse 13/14's K and V; sizes are dims x dtype."""
    document = _document()
    rules = pytorch_binding.binding('gemma4')['ops']

    modules, reasons = layer_ops.pytorch_modules(
        document, rules, 'attn/k_proj', 'Attention k_proj'
    )

    self.assertEqual(sorted(modules), list(range(15)))
    self.assertEqual(
        modules[2].module, 'model.language_model.layers.2.self_attn.k_proj'
    )
    self.assertIn('KV-shared', reasons[15])
    self.assertEqual(
        layer_ops.missing(document, 'attn/k_proj', modules, 'none there'),
        ["not in layers 15–34 (they reuse layer 13/14's K and V)"],
    )
    self.assertEqual(
        layer_ops.missing(document, 'attn/q_proj', [0, 1, 7], 'none'),
        ['not in layers 2–6, 8–34: none'],
    )
    # gate_proj in bfloat16: 15 layers of 6144, 20 double wide.
    gate, _ = layer_ops.pytorch_modules(
        document, rules, 'ffn/gate_proj', 'Feed Forward gate_proj'
    )
    self.assertEqual(
        sum(layer_ops.module_bytes(m, 'bfloat16') for m in gate.values()),
        (15 * 6144 + 20 * 12288) * 2,
    )

  def test_known_is_every_operator_the_bindings_map(self):
    """What a Session may choose."""
    self.assertIn('attn/q_proj', layer_ops.known())
    self.assertIn('norm.ple.out', layer_ops.known())
    self.assertNotIn('attn/attend', layer_ops.known())


if __name__ == '__main__':
  test_main.main()
