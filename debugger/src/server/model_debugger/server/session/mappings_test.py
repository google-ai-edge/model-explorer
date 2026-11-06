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

"""Tests of the tensor pairs made by hand, per pair of models."""

import dataclasses
import pathlib
from typing import Any

from absl.testing import absltest
from model_debugger.server.session import mappings
from model_debugger.server.session import workspace
from model_debugger.server.testing import test_main
from model_debugger.server.util import errors
from model_debugger.server.util import json_file

_A = 'a' * 64
_B = 'b' * 64
_LITERT = mappings.Tensor('decode', '412', '0', 'decode:412:0')
_PYTORCH = mappings.Tensor(
    'decode',
    '5903',
    '0',
    'decode:model.language_model.layers.3.self_attn.o_proj:in',
)


@dataclasses.dataclass(frozen=True)
class _Model:
  sha256: str
  name: str


class _Graphs:
  """A model's graph outputs, as ModelGraphs.tensor() serves them."""

  def __init__(self, outputs: dict[str, Any]) -> None:
    self.outputs = outputs

  def tensor(self, model, signature, node, output) -> dict[str, Any]:
    try:
      return self.outputs[model.sha256, signature, node, output]
    except KeyError:
      raise errors.NotFound(
          f'The {signature} graph of {model.name} has no output '
          f'{output} of node {node}.'
      ) from None


class MappingStoreTest(absltest.TestCase):
  """Tests of keeping the pairs of two models."""

  def setUp(self):
    """An empty workspace's store."""
    super().setUp()
    root = pathlib.Path(self.create_tempdir().full_path)
    self.space = workspace.Workspace(root / 'ws')
    self.addCleanup(self.space.close)
    self.store = mappings.MappingStore(self.space)

  def test_a_pair_is_kept_for_the_two_models_in_either_order(self):
    """One file per pair of models; the other order reads it turned."""
    self.assertEqual(self.store.pairs(_B, _A), [])

    saved = self.store.add(_B, _A, _LITERT, _PYTORCH, 'o_proj input')

    self.assertEqual(self.store.pairs(_B, _A), [saved])
    (turned,) = self.store.pairs(_A, _B)
    self.assertEqual((turned.ref, turned.target), (_PYTORCH, _LITERT))
    self.assertEqual(turned.label, 'o_proj input')
    stored = json_file.read_json(self.space.mappings / f'{_A}_{_B}.json')
    self.assertEqual(stored['format'], mappings.FORMAT)
    # The order of the Session that saved the first pair.
    self.assertEqual((stored['ref'], stored['target']), (_B, _A))
    self.assertEqual(
        stored['pairs'][0]['ref'],
        {
            'signature': 'decode',
            'node': '412',
            'output': '0',
            'point': 'decode:412:0',
        },
    )
    self.assertEqual(stored['pairs'][0]['created'], saved.created)
    # A Session of the other order adds to the same file, its pairs
    # stored in the file's order.
    other = mappings.Tensor('decode', '413', '0', 'decode:413:0')
    self.store.add(_A, _B, _PYTORCH, other)
    stored = json_file.read_json(self.space.mappings / f'{_A}_{_B}.json')
    self.assertEqual(stored['pairs'][1]['ref']['point'], 'decode:413:0')
    self.assertEqual(self.store.revision, 2)

  def test_duplicates_one_model_and_long_labels_are_refused(self):
    """Nothing is written for any of them."""
    self.store.add(_A, _B, _LITERT, _PYTORCH)
    with self.assertRaisesRegex(errors.Conflict, 'paired already'):
      self.store.add(_A, _B, _LITERT, _PYTORCH)
    with self.assertRaisesRegex(errors.Conflict, 'paired already'):
      self.store.add(_B, _A, _PYTORCH, _LITERT)
    with self.assertRaisesRegex(ValueError, 'same model'):
      self.store.add(_A, _A, _LITERT, _LITERT)
    with self.assertRaisesRegex(ValueError, 'at most 200'):
      self.store.add(
          _A,
          _B,
          _LITERT,
          dataclasses.replace(_PYTORCH, point='x'),
          'l' * 201,
      )
    self.assertLen(self.store.pairs(_A, _B), 1)
    self.assertEqual(self.store.pairs(_A, _A), [])

  def test_a_pair_is_deleted_by_its_id(self):
    """The last one deleted takes the file with it."""
    saved = self.store.add(_A, _B, _LITERT, _PYTORCH)
    with self.assertRaises(errors.NotFound):
      self.store.delete(_A, _B, 'no-such-pair')

    deleted = self.store.delete(_B, _A, saved.id)

    self.assertEqual(deleted.id, saved.id)
    self.assertEqual(self.store.pairs(_A, _B), [])
    self.assertFalse((self.space.mappings / f'{_A}_{_B}.json').exists())

  def test_the_pairs_of_a_model_deleted_go_with_it(self):
    """Either model gone takes the file; two models kept keep theirs."""
    third = 'c' * 64
    self.store.add(_A, _B, _LITERT, _PYTORCH)
    self.store.add(_A, third, _LITERT, _PYTORCH)
    self.store.add(_B, third, _LITERT, _PYTORCH)
    revision = self.store.revision

    removed = self.store.collect({_A, _B})

    self.assertEqual(removed, [f'{_A}_{third}.json', f'{_B}_{third}.json'])
    self.assertLen(self.store.pairs(_A, _B), 1)
    self.assertEqual(self.store.pairs(_A, third), [])
    self.assertEqual(
        [path.name for path in self.space.mappings.iterdir()],
        [f'{_A}_{_B}.json'],
    )
    self.assertEqual(self.store.revision, revision + 1)
    self.assertEqual(self.store.collect({_A, _B}), [])
    self.assertEqual(self.store.revision, revision + 1)

  def test_a_file_of_another_format_is_not_read(self):
    """It says so; it is not overwritten."""
    json_file.write_json(self.space.mappings / f'{_A}_{_B}.json', {'format': 0})
    with self.assertRaisesRegex(ValueError, 'another format'):
      self.store.pairs(_A, _B)


class CheckTest(absltest.TestCase):
  """Tests of checking a pair's two tensors against their graphs."""

  def setUp(self):
    """A LiteRT-LM model and a PyTorch one, each with graph outputs."""
    super().setUp()
    self.models = {
        'ref': _Model(_A, 'gemma-4-E2B-it.litertlm'),
        'target': _Model(_B, 'gemma-4-E2B-it'),
    }
    self.graphs = _Graphs({
        (_A, 'decode', '412', '0'): {
            'point_id': 'decode:412:0',
            'observable': 'true',
            'layer': '3',
            'tensor_name': 'layer_3/attn.post_qkv/attn_vec_einsum/reshape',
            'tensor_shape': 'float32[1,1,8,256]',
        },
        (_A, 'decode', '7', '0'): {
            'point_id': 'decode:7:0',
            'observable': 'false',
            'reason': 'it is a constant',
            'tensor_shape': 'float32[1536]',
        },
        (_A, 'decode', '9', '0'): {
            'point_id': 'decode:9:0',
            'observable': 'true',
            'tensor_shape': 'float32[1, 1, 1536]',
        },
        (_B, 'decode', '5903', '0'): {
            'point_id': _PYTORCH.point,
            'observable': 'true',
            'layer': '3',
            'tensor_shape': 'float32[1, 1, 2048]',
        },
        (_B, 'decode', '6000', '0'): {
            'point_id': 'decode:model.language_model.norm:out',
            'observable': 'true',
            'tensor_shape': 'float32[1, 1, 1536]',
        },
        (_B, 'decode', '1', '0'): {
            'observable': 'false',
            'reason': 'The PyTorch runner captures modules, not ATen operators',
            'tensor_shape': 'float32[1, 1, 2048]',
        },
    })

  def _check(self, ref, target):
    return mappings.check(
        self.graphs,
        self.models,
        {
            'ref': mappings.requested(
                {'signature': 'decode', 'node': ref, 'output': '0'}, 'ref'
            ),
            'target': mappings.requested(
                {'signature': 'decode', 'node': target, 'output': '0'},
                'target',
            ),
        },
    )

  def test_two_outputs_that_hold_the_same_values_are_a_pair(self):
    """[8, 256] by heads against [2048] flat, both in layer 3."""
    found = self._check('412', '5903')

    self.assertEqual(found['ref'].tensor, _LITERT)
    self.assertEqual(found['target'].tensor, _PYTORCH)
    self.assertEqual(
        mappings.default_label(found),
        'attn_vec_einsum/reshape ↔ self_attn.o_proj input',
    )
    self.assertEqual(found['ref'].layer, 3)
    self.assertEqual(found['target'].shape, (1, 1, 2048))

  def test_tensors_that_cannot_pair_are_refused_with_the_reason(self):
    """Missing, not capturable, other values, in no layer."""
    cases = {
        ('404', '5903'): 'has no output 0 of node 404',
        ('7', '5903'): 'cannot be captured: it is a constant',
        ('412', '1'): 'captures modules, not ATen operators',
        ('9', '5903'): 'do not hold the same values per position',
        ('9', '6000'): 'Neither tensor is in a decoder layer',
    }
    for (ref, target), text in cases.items():
      with self.subTest(ref=ref, target=target):
        with self.assertRaisesRegex(ValueError, text):
          self._check(ref, target)
    with self.assertRaisesRegex(ValueError, 'Name the Target tensor'):
      mappings.requested({'signature': 'decode'}, 'target')
    same = dict(self.models, target=self.models['ref'])
    with self.assertRaisesRegex(ValueError, 'same model'):
      mappings.check(self.graphs, same, {})

  def test_shapes_compare_per_position(self):
    """Without axes of one, flat on one side, a prefill's rows dropped."""
    self.assertTrue(mappings.compatible((1, 1, 8, 256), (1, 1, 2048)))
    self.assertTrue(mappings.compatible((1, 128, 8, 256), (1, 9, 2048)))
    self.assertTrue(mappings.compatible((1, 1, 4), (4,)))
    self.assertFalse(mappings.compatible((1, 1, 8, 256), (1, 1, 1024)))
    self.assertFalse(mappings.compatible((1, 1, 4, 6), (1, 1, 6, 4)))
    self.assertEqual(mappings.shape_of('float32[1, 1, 2048]'), (1, 1, 2048))
    self.assertEqual(mappings.shape_of('bfloat16[]'), ())
    self.assertIsNone(mappings.shape_of(''))


if __name__ == '__main__':
  test_main.main()
