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

"""Tests of the Session editor's hint of two models."""

import shutil
from unittest import mock

from absl.testing import absltest
from model_debugger.server.analysis import comparability
from model_debugger.server.architectures import catalog
from model_debugger.server.session import model_comparison
from model_debugger.server.session import model_graphs
from model_debugger.server.tap import model_facts
from model_debugger.server.testing import anchor_fixtures
from model_debugger.server.testing import test_main
from model_debugger.server.testing import world as world_lib
from model_debugger.server.util import errors

_TABLE = {'table_sha256': 'a' * 64, 'size': 4, 'source': 'test'}


class ModelComparisonTest(absltest.TestCase):
  """A LiteRT-LM model and Gemma 4 E2B's PyTorch directory."""

  def setUp(self):
    """A world that keeps graphs; both models of one tokenizer."""
    super().setUp()
    self.world = world_lib.create(self, graphs=True)
    directory = self.world.workspace.hf / 'gemma-4-E2B-it'
    directory.mkdir()
    (gemma4,) = [a for a in catalog.architectures() if a.id == 'gemma4']
    pinned = gemma4.entry('gemma-4-E2B-it')['pinned']['file']
    shutil.copyfile(gemma4.directory / pinned, directory / 'config.json')
    self.pytorch = self.world.models.add_download_directory(directory)
    patch = mock.patch.object(
        model_facts,
        'described',
        return_value={'vocabulary': _TABLE, 'kv_quantization': {}},
    )
    patch.start()
    self.addCleanup(patch.stop)
    self.comparison = model_comparison.ModelComparison(
        self.world.models, self.world.graphs
    )

  def test_one_file_shares_its_capture(self):
    """One LiteRT-LM file: every analysis, one capture of both sides."""
    artifact = self.world.model.artifact
    found = self.comparison.compare(artifact, artifact)
    self.assertEqual(found['code'], comparability.SAME_FILE)
    self.assertEqual(found['label'], self.world.model.name)
    self.assertEqual(found['capture_shared'], model_comparison.SHARED_CAPTURE)

  def test_a_graph_not_converted_leaves_the_architecture_unknown(self):
    """The PyTorch model's is its config's; the LiteRT-LM one's, not yet."""
    self.assertEqual(
        self.world.graphs.known_architecture(self.pytorch),
        (
            {
                'id': 'gemma4',
                'name': 'Gemma 4',
                'skeleton': 'gemma-4-E2B-it',
            },
            '',
        ),
    )
    found = self.comparison.compare(
        self.pytorch.artifact, self.world.model.artifact
    )
    self.assertEqual(found['code'], comparability.DIFFERENT)
    self.assertEqual(
        found['label'], f'gemma-4-E2B-it vs {self.world.model.name}'
    )
    self.assertEqual(found['capture_shared'], '')
    self.assertIn('the generated text, logits and sampling', found['text'])
    self.assertIn(
        f'The architecture of {self.world.model.name} is found in its '
        'graph, which is converted as a Session of it first turns on.',
        found['text'],
    )

  def test_a_converted_graph_of_gemma4_pairs_by_anchors(self):
    """Its graph binds Gemma 4: the anchors compare."""
    with (
        mock.patch.object(
            self.world.graphs,
            '_state',
            return_value=model_graphs.State(model_graphs.COMPLETED),
        ),
        mock.patch.object(
            self.world.graphs,
            'architecture',
            return_value=anchor_fixtures.architecture(),
        ),
    ):
      found = self.comparison.compare(
          self.pytorch.artifact, self.world.model.artifact
      )
    self.assertEqual(found['code'], comparability.SAME_ARCHITECTURE)
    self.assertTrue(
        found['text'].startswith(
            'Same architecture (Gemma 4) and the same tokenizer: the '
            "generated text, logits, sampling and every layer's anchors "
            'compare.'
        ),
        found['text'],
    )

  def test_an_unknown_model_is_not_found(self):
    """A model not registered is refused."""
    with self.assertRaises(errors.NotFound):
      self.comparison.compare('model-nothing', self.pytorch.artifact)


if __name__ == '__main__':
  test_main.main()
