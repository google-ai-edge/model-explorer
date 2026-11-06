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

"""LiteRT-LM points of Gemma 4 E2B paired with the PyTorch runner's modules.

The points are named as gemma-4-E2B-it.litertlm names them (its scan).
"""

import json
from typing import Any

from absl.testing import absltest
from model_debugger.server.architectures import catalog
from model_debugger.server.graph import architecture
from model_debugger.server.graph import counterparts
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


_NORM = 'STABLEHLO_COMPOSITE odml.rms_norm'
# Layer 3's points of decode, as the real model's scan has them.
_LAYER_3 = (
    _point(
        238, 3, _PRE + 'q_einsum/composite', 'QUANTIZE', (1, 1, 1536), 'int8'
    ),
    _point(
        239,
        3,
        _PRE + 'q_einsum/reshape',
        'FULLY_CONNECTED',
        (1, 1, 2048),
        'int8',
    ),
    _point(241, 3, _PRE + 'q_einsum/composite1', 'DEQUANTIZE', (1, 1, 8, 256)),
    _point(244, 3, _PRE + 'k_einsum/composite', 'DEQUANTIZE', (1, 1, 1, 256)),
    _point(248, 3, _PRE + 'query_norm/composite', _NORM, (1, 1, 8, 256)),
    _point(249, 3, _PRE + 'key_norm/composite', _NORM, (1, 1, 1, 256)),
    _point(252, 3, _PRE + 'maybe_rope/split1', 'SLICE', (1, 1, 8, 128)),
    _point(
        278,
        3,
        'post_qkv/attn.post_qkv/attn_vec_einsum/composite1',
        'DEQUANTIZE',
        (1, 1, 1536),
    ),
    _point(281, 3, 'post_qkv/pre_ffw_norm/composite', _NORM, (1, 1, 1536)),
    _point(
        287, 3, 'post_qkv/mlp/activation_fn/composite', 'GELU', (1, 1, 6144)
    ),
    _point(288, 3, 'post_qkv/mlp/mul', 'MUL', (1, 1, 6144)),
    _point(
        291, 3, 'post_qkv/mlp/linear/composite1', 'DEQUANTIZE', (1, 1, 1536)
    ),
    _point(298, 3, 'post_qkv/mul', 'MUL', (1, 1, 256)),
)


def _document() -> dict[str, Any]:
  """The skeleton the PyTorch runner's E2B config.json makes."""
  (gemma4,) = [a for a in catalog.architectures() if a.id == 'gemma4']
  entry = gemma4.entry('gemma-4-E2B-it')
  path = gemma4.directory / entry['pinned']['file']
  _, document = catalog.generate(json.loads(path.read_text('utf-8')))
  return document


def _find(chosen, points=_LAYER_3):
  (known,) = [a for a in architecture.architectures() if a.id == 'gemma4']
  return counterparts.find(
      chosen,
      points,
      known.binding['ops'],
      pytorch_binding.binding('gemma4')['ops'],
      _document(),
  )


def _by_id(point_id: str) -> tap_scan.Point:
  return next(point for point in _LAYER_3 if point.id == point_id)


class CounterpartsTest(absltest.TestCase):
  """Which PyTorch module each LiteRT-LM point chosen is."""

  def test_each_semantic_operator_is_its_module(self):
    """The output of each operator, its module, edge and dims."""
    ids = (
        'decode:241:0',
        'decode:244:0',
        'decode:248:0',
        'decode:249:0',
        'decode:278:0',
        'decode:281:0',
        'decode:287:0',
        'decode:288:0',
        'decode:291:0',
        'decode:298:0',
    )
    found = _find([_by_id(i) for i in ids])
    self.assertEqual([c.reason for c in found], [''] * len(ids))
    layer = 'model.language_model.layers.3'
    self.assertEqual(
        [(c.op, c.module.module, c.module.edge, c.module.dims) for c in found],
        [
            ('attn/q_proj', f'{layer}.self_attn.q_proj', 'out', (2048,)),
            ('attn/k_proj', f'{layer}.self_attn.k_proj', 'out', (256,)),
            ('attn/q_norm', f'{layer}.self_attn.q_norm', 'out', (8, 256)),
            ('attn/k_norm', f'{layer}.self_attn.k_norm', 'out', (1, 256)),
            ('attn/o_proj', f'{layer}.self_attn.o_proj', 'out', (1536,)),
            (
                'norm.ffn.in',
                f'{layer}.pre_feedforward_layernorm',
                'out',
                (1536,),
            ),
            ('ffn/act', f'{layer}.mlp.act_fn', 'out', (6144,)),
            ('ffn/gate', f'{layer}.mlp.down_proj', 'in', (6144,)),
            ('ffn/down_proj', f'{layer}.mlp.down_proj', 'out', (1536,)),
            ('ple/gate', f'{layer}.per_layer_projection', 'in', (256,)),
        ],
    )
    self.assertEqual(found[0].label, 'Attention q_proj')
    self.assertEqual(found[5].label, 'Feed Forward Pre-RMSNorm')
    self.assertEqual(found[0].module.path, 'output')
    self.assertEqual(found[7].module.path, 'args.0')

  def test_another_tensor_of_the_scope_names_the_output(self):
    """An int8 FULLY_CONNECTED inside q_proj is not its output."""
    (found,) = _find([_by_id('decode:239:0')])
    self.assertIsNone(found.module)
    self.assertEqual(
        found.reason,
        'PyTorch has no counterpart for decode:239:0 (layer 3, '
        'FULLY_CONNECTED): it is a FULLY_CONNECTED (int8) inside '
        'Attention q_proj, not its output, the DEQUANTIZE (decode:241:0); '
        'only the anchors are captured there.',
    )

  def test_an_operator_no_module_returns_is_not_paired(self):
    """RoPE's tensors are inside the attention, no module's."""
    (found,) = _find([_by_id('decode:252:0')])
    self.assertIn('in no scope the binding maps', found.reason)
    self.assertTrue(found.reason.endswith(counterparts.ONLY_ANCHORS + '.'))

  def test_a_kv_shared_layer_has_no_k_proj(self):
    """Layer 20 reads layer 13's keys: no k_proj on PyTorch."""
    point = _point(
        1320, 20, _PRE + 'k_einsum/composite', 'DEQUANTIZE', (1, 1, 1, 256)
    )
    (found,) = _find([point], [point])
    self.assertIn(
        'layer 20 shares the keys and values of an earlier layer '
        '(KV-shared): its PyTorch attention has no k_proj',
        found.reason,
    )

  def test_a_shape_of_other_elements_is_not_paired(self):
    """Shapes are compared after [batch, tokens], axes of one dropped."""
    point = _point(
        283,
        3,
        'post_qkv/mlp/gating_einsum1/composite1',
        'DEQUANTIZE',
        (1, 1, 12288),
    )
    (found,) = _find([point], [point])
    self.assertIn(
        'its shape [1, 1, 12288] is not that of what '
        'model.language_model.layers.3.mlp.gate_proj returns, [batch, '
        'rows, 6144]',
        found.reason,
    )

  def test_a_prefill_point_is_paired_by_its_rows(self):
    """A prefill chunk's tokens are its axis 1."""
    point = _point(
        300,
        3,
        _PRE + 'q_einsum/composite1',
        'DEQUANTIZE',
        (1, 128, 8, 256),
        signature='prefill_128',
    )
    (found,) = _find([point], [point])
    self.assertEqual(found.op, 'attn/q_proj')

  def test_compatible_shapes(self):
    """The same dims, or the same elements with one side flat."""
    self.assertTrue(counterparts.compatible((1, 1, 8, 256), (2048,)))
    self.assertTrue(counterparts.compatible((1, 4, 1, 256), (1, 256)))
    self.assertFalse(counterparts.compatible((1, 1, 8, 256), (16, 128)))
    self.assertFalse(counterparts.compatible((2, 1, 2048), (2048,)))

  def test_the_places_are_in_the_layers_order(self):
    """The anchors among the operators, each where its node ends."""
    (known,) = [a for a in architecture.architectures() if a.id == 'gemma4']
    labels = counterparts.place_labels(_document(), known.binding['ops'])
    self.assertEqual(
        list(labels)[:12],
        [
            'input',
            'norm.attn.in',
            'attn/q_proj',
            'attn/k_proj',
            'attn/v_proj',
            'attn/q_norm',
            'attn/k_norm',
            'attn/v_norm',
            'attn/o_proj',
            'norm.attn.out',
            'intermediate.1',
            'norm.ffn.in',
        ],
    )
    self.assertEqual(list(labels)[-2:], ['norm.ple.out', 'output'])
    self.assertLen(labels, 4 + len(known.binding['ops']))

  def test_both_bindings_map_the_same_operators(self):
    """Each operator the LiteRT-LM binding maps has a PyTorch module."""
    (known,) = [a for a in architecture.architectures() if a.id == 'gemma4']
    self.assertEqual(
        list(known.binding['ops']),
        list(pytorch_binding.binding('gemma4')['ops']),
    )


if __name__ == '__main__':
  test_main.main()
