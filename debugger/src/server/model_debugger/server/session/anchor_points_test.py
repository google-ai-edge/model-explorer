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

"""Tests of the anchors a Session captures by default."""

import pathlib
from typing import Any

from absl.testing import absltest
from model_debugger.server.session import anchor_points
from model_debugger.server.session import model_graphs
from model_debugger.server.session import model_store
from model_debugger.server.session import tap_scans
from model_debugger.server.session import workspace
from model_debugger.server.testing import anchor_fixtures
from model_debugger.server.testing import tap_fixtures
from model_debugger.server.testing import test_main


class FakeGraphs:
  """The graphs of models, in the state given."""

  def __init__(
      self, state: model_graphs.State, answer: dict[str, Any] | None = None
  ):
    """Graphs in `state`, whose architecture is `answer` (Gemma 4's)."""
    self.state_now = state
    self.answer = answer or anchor_fixtures.architecture()

  def state(self, artifact: str) -> model_graphs.State:
    """The state given, of any model."""
    del artifact
    return self.state_now

  def architecture(self, model: model_store.Model) -> dict[str, Any]:
    """The answer given, of any model."""
    del model
    return self.answer


class AnchorPointsTest(absltest.TestCase):
  """Tests of find(), over the fixture model."""

  def setUp(self):
    """The fixture model, registered, and its scans."""
    super().setUp()
    root = pathlib.Path(self.create_tempdir().full_path)
    opened = workspace.Workspace(root / 'ws')
    self.addCleanup(opened.close)
    tap_fixtures.write_model(opened.hf / 'tap.litertlm')
    models = model_store.ModelStore(opened)
    self.model = models.add_download(opened.hf / 'tap.litertlm')
    self.scans = tap_scans.TapScans(models)
    self.addCleanup(self.scans.close)

  def test_every_layers_anchors_of_every_signature_that_runs(self):
    """Each point once; what is skipped says why, by its places."""
    found = anchor_points.find(
        FakeGraphs(model_graphs.State(model_graphs.COMPLETED)),
        self.scans,
        self.model,
    )

    self.assertEqual(found.status, anchor_points.FOUND)
    # In the model's order; a layer's output, the next one's input, once.
    self.assertEqual(
        found.point_ids,
        (
            'decode:0:0',
            'decode:1:0',
            'decode:2:0',
            'prefill_16:0:0',
            'prefill_16:1:0',
            'prefill_16:2:0',
        ),
    )
    self.assertIn('6 tensor outputs of 2 signatures', found.text)
    decode, prefill, verify = found.signatures
    self.assertEqual(decode.layers, (0, 1))
    self.assertEqual(
        decode.text,
        'decode: 3 tensor outputs, the anchors of 2 layers; 2 anchors skipped.',
    )
    self.assertEqual(
        decode.skipped,
        (
            (
                'Layer 0 input: It is the input embeddings of decode, no '
                "operator's output; it cannot be observed."
            ),
            'Layer 1 intermediate.2: Its type, string, is not a number.',
        ),
    )
    self.assertEqual(prefill.layers, (0,))
    self.assertEqual(
        prefill.skipped[1],
        'Layer 1 every anchor: prefill_16 has no operator of layer 1.',
    )
    self.assertEqual(verify.point_ids, ())
    self.assertIn('speculative decoding', verify.text)

  def test_none_until_the_graph_is_converted(self):
    """Its graph not converted, the anchors are not known yet."""
    for state, says in (
        (model_graphs.State(model_graphs.IDLE), 'first turns on'),
        (
            model_graphs.State(model_graphs.FAILED, error='It crashed.'),
            'could not be converted: It crashed.',
        ),
    ):
      found = anchor_points.find(FakeGraphs(state), self.scans, self.model)
      self.assertEqual(found.status, anchor_points.PENDING)
      self.assertIn(says, found.text)
      self.assertEqual(found.point_ids, ())

  def test_a_model_of_no_architecture_has_none_and_says_why(self):
    """Such as Qwen: the server has no template of it."""
    answer = {
        **anchor_fixtures.architecture(),
        'architecture': None,
        'text': 'No architecture template exists for this model.',
        'bindings': {},
    }
    found = anchor_points.find(
        FakeGraphs(model_graphs.State(model_graphs.COMPLETED), answer),
        self.scans,
        self.model,
    )

    self.assertEqual(found.status, anchor_points.NONE)
    self.assertEqual(
        found.text,
        'tap.litertlm has no default anchors: No architecture template '
        'exists for this model.',
    )
    self.assertEqual(
        anchor_points.find(None, self.scans, self.model).status,
        anchor_points.NONE,
    )


if __name__ == '__main__':
  test_main.main()
