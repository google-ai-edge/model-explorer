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

"""Tests of the skeleton schema's checks, on the skeletons the server has."""

import copy
import hashlib

from absl.testing import absltest
from model_debugger.server.architectures import catalog
from model_debugger.server.architectures import skeleton
from model_debugger.server.testing import test_main


class SkeletonTest(absltest.TestCase):
  """Tests of check(), on Gemma 4 E2B's and E4B's skeletons."""

  def setUp(self):
    """Gemma 4 E2B's skeleton."""
    super().setUp()
    (self.gemma4,) = catalog.architectures()
    self.e2b = self.gemma4.skeleton('gemma-4-E2B-it')

  def test_the_skeletons_are_the_generators_and_pass(self):
    """Each file is the one its entry names, by SHA-256, and valid."""
    for entry in self.gemma4.spec['skeletons']:
      path = self.gemma4.directory / entry['file']
      self.assertEqual(
          hashlib.sha256(path.read_bytes()).hexdigest(), entry['sha256']
      )
      found = skeleton.load(path)
      self.assertEqual(skeleton.check(found), [])
      # What the file holds is what dumps() writes.
      self.assertEqual(skeleton.dumps(found).encode(), path.read_bytes())
      self.assertEqual(found['semantic_ops'], skeleton.SEMANTIC_OPS)
      definition = found['semantic_graph'][0]
      self.assertLen(definition['nodes'], 12)
      self.assertEqual(
          [a['id'] for a in definition['anchors']],
          ['input', 'intermediate.1', 'intermediate.2', 'output'],
      )
    e4b = self.gemma4.skeleton('gemma-4-E4B-it')
    self.assertLen(self.e2b['layers'], 35)
    self.assertLen(e4b['layers'], 42)
    self.assertEqual(self.e2b['layers'][15]['from']['kv'], {'layer': 13})

  def test_names_each_anchor_by_the_block_it_ends(self):
    """The layer's input, each block's output, the layer's output."""
    definition = self.e2b['semantic_graph'][0]
    self.assertEqual(
        [
            skeleton.anchor_label(definition, anchor)
            for anchor in definition['anchors']
        ],
        [
            'Layer input',
            'Attention block output',
            'Feed Forward block output',
            'Layer output (Per-Layer Embedding block)',
        ],
    )

  def test_names_each_block_by_its_one_node_of_substance(self):
    """Attention, Feed Forward, Per-Layer Embedding, as the anchors say."""
    definition = self.e2b['semantic_graph'][0]
    found = skeleton.blocks(definition['nodes'], definition['anchors'])
    self.assertEqual(
        {k: v['label'] for k, v in found.items()},
        {
            'intermediate.1': 'Attention',
            'intermediate.2': 'Feed Forward',
            'output': 'Per-Layer Embedding',
        },
    )
    self.assertEqual(
        found['intermediate.1']['nodes'],
        ['norm.attn.in', 'attn', 'norm.attn.out', 'add_attn'],
    )

  def test_finds_what_breaks_the_schema(self):
    """A dead node, a wrong block, KV skips without from.kv, a bad from."""
    broken = copy.deepcopy(self.e2b)
    definition = broken['semantic_graph'][0]
    definition['anchors'][1]['block'] = 'Feed Forward'
    definition['nodes'][0]['ops'][0]['op'] = 'FullyConnected'
    # The vocabulary is SEMANTIC_OPS: a skeleton's copy cannot widen it.
    broken['semantic_ops']['FullyConnected'] = "TFLite's word."
    del broken['layers'][15]['from']['kv']
    broken['layers'][3]['from']['input']['layer'] = 7
    errors = skeleton.check(broken)
    self.assertLen(errors, 4, errors)
    self.assertIn("anchor 'intermediate.1' names block", errors[0])
    self.assertIn('FullyConnected', errors[1])
    self.assertIn('layers[3].from.input names layer 7', errors[2])
    self.assertIn('layers[15]: from.kv and the steps skipped', errors[3])
    dead = copy.deepcopy(self.e2b)
    dead['semantic_graph'][0]['anchors'].pop()
    dead['semantic_graph'][0]['outputs'] = []
    self.assertTrue(
        any('dead node' in e for e in skeleton.check(dead)),
        skeleton.check(dead),
    )

  def test_the_per_layer_embedding_reads_its_per_layer_input(self):
    """Two ports; ple reads the second as its input 1, or it is wrong."""
    definition = self.e2b['semantic_graph'][0]
    self.assertEqual(
        definition['inputs'],
        [
            {
                'id': 'input',
                'shape': ['B', 'S', 1536],
                'source': 'residual',
            },
            {
                'id': 'per_layer_input',
                'shape': ['B', 'S', 256],
                'source': 'external per-layer embedding',
                'label': 'Per-layer input',
            },
        ],
    )
    ple = next(n for n in definition['nodes'] if n['id'] == 'ple')
    self.assertEqual(
        [
            (e['sourceNodeId'], e['targetNodeInputId'])
            for e in ple['incomingEdges']
        ],
        [('add_ffn', '0'), ('per_layer_input', '1')],
    )
    # mkjson.py at 7b4f6db wrote ple without it.
    unwired = copy.deepcopy(self.e2b)
    ple = next(
        n for n in unwired['semantic_graph'][0]['nodes'] if n['id'] == 'ple'
    )
    ple['incomingEdges'].pop()
    errors = skeleton.check(unwired)
    self.assertLen(errors, 2, errors)
    self.assertIn("input 'per_layer_input' is read by nothing", errors[0])
    self.assertIn("'ple' is a per-layer embedding", errors[1])
    unwired['semantic_graph'][0]['inputs'].pop()
    self.assertEqual(
        skeleton.check(unwired),
        [
            "defs[0]: 'ple' is a per-layer embedding, which reads one "
            'per-layer input from outside the layer; it reads [].'
        ],
    )
    twice = copy.deepcopy(self.e2b)
    twice['semantic_graph'][0]['inputs'].append(
        twice['semantic_graph'][0]['inputs'][1]
    )
    self.assertIn(
        "defs[0]: input id 'per_layer_input' repeats.",
        skeleton.check(twice),
    )

  def test_an_anchor_must_be_kept_by_every_runtime(self):
    """One read once inside a layer, never across, is no anchor."""
    loose = copy.deepcopy(self.e2b)
    loose['semantic_graph'][0]['anchors'].insert(
        1, {'id': 'mid', 'semantic': 'x', 'of': 'norm.attn.in:0'}
    )
    self.assertTrue(
        any(
            "anchor 'mid' meets none" in e
            for e in skeleton.check_anchor_criterion(loose)
        )
    )

  def test_finds_bad_edges_and_wrong_node_namespaces(self):
    """Unknown sources, out-of-range ports, and mismatched block namespaces."""
    unknown = copy.deepcopy(self.e2b)
    unknown['semantic_graph'][0]['nodes'][0]['incomingEdges'].append(
        {'sourceNodeId': 'missing_node', 'targetNodeInputId': '1'}
    )
    self.assertTrue(
        any(
            "reads 'missing_node', which does not exist" in e
            for e in skeleton.check(unknown)
        ),
        skeleton.check(unknown),
    )

    bad_port = copy.deepcopy(self.e2b)
    bad_port['semantic_graph'][0]['nodes'][0]['incomingEdges'][0][
        'sourceNodeOutputId'
    ] = '3'
    self.assertTrue(
        any(
            "reads 'input:3', but 'input' has 1 outputs" in e
            for e in skeleton.check(bad_port)
        ),
        skeleton.check(bad_port),
    )

    wrong_ns = copy.deepcopy(self.e2b)
    wrong_ns['semantic_graph'][0]['nodes'][0]['namespace'] = 'Feed Forward'
    self.assertTrue(
        any(
            "the namespace of 'norm.attn.in' is 'Feed Forward', but it is in "
            "block 'Attention'" in e
            for e in skeleton.check(wrong_ns)
        ),
        skeleton.check(wrong_ns),
    )


if __name__ == '__main__':
  test_main.main()
