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

"""What two models compare in, as the Session editor says before a run.

The editor's hint under its two models (GET /api/models/compare): the
verdict the analyses' own rules give of the two (analysis/comparability),
read from what the server already has of each model, and converting or
running nothing: its file's SHA-256 and runtime (model_store), its
tokenizer's table (tap/model_facts, as a side's capture records it too, or
as kept with a model imported without its file: Model.described()),
and its architecture as the analyses take it
(model_graphs.ModelGraphs.known_architecture()): a LiteRT-LM model's is
found in its graph, which is converted as a Session of it first turns on, so
until then the hint says it is not known yet. With it, what the listing
calls the two (session_store.models_label()), which the editor suggests a
name by, and whether the sides share one tensor capture: two LiteRT-LM
sides of one file load one copy of it, so they observe the same tensors.
"""

from typing import Any

from model_debugger.server.analysis import comparability
from model_debugger.server.session import model_graphs
from model_debugger.server.session import model_store
from model_debugger.server.session import session_store

# What the editor says of two LiteRT-LM sides of one model file.
SHARED_CAPTURE = (
    'Both sides load this file, and one copy of it observes the same '
    'tensors on both: its anchors and outputs chosen are set once for both.'
)


class ModelComparison:
  """What two registered models compare in."""

  def __init__(
      self,
      models: model_store.ModelStore,
      graphs: model_graphs.ModelGraphs | None,
  ) -> None:
    """Reads the models given, and their graphs if the server keeps any.

    Args:
        models: The registered models.
        graphs: Their graphs; None: no architecture is known.
    """
    self._models = models
    self._graphs = graphs

  def facts(self, model: model_store.Model) -> comparability.Model:
    """What a model is, as the analyses' rules compare it."""
    vocabulary = model.described().get('vocabulary')
    architecture, unknown = (
        None,
        (
            'This server keeps no graphs, so the architecture of '
            f'{model.name} is not known.'
        ),
    )
    if self._graphs is not None:
      architecture, unknown = self._graphs.known_architecture(model)
    return comparability.Model(
        model.sha256,
        model.runtime,
        model.name,
        (vocabulary or {}).get('table_sha256'),
        architecture=(architecture or {}).get('id'),
        architecture_name=(architecture or {}).get('name', ''),
        skeleton=(architecture or {}).get('skeleton'),
        architecture_unknown=unknown,
    )

  def compare(self, ref: str, target: str) -> dict[str, Any]:
    """The editor's hint of two models.

    Args:
        ref: The Reference side's model, by artifact.
        target: The Target side's.

    Returns:
        `{code, text, label, capture_shared}`: the verdict
        (comparability.verdict()), the two models' label, and what the
        editor says of one tensor capture shared, '' if the sides do not
        share one.

    Raises:
        NotFound: A model is not registered.
    """
    models = self._models.get(ref), self._models.get(target)
    shared = (
        models[0].artifact == models[1].artifact
        and models[0].runtime == model_store.LITERT_LM
    )
    return {
        **comparability.verdict(*(self.facts(model) for model in models)),
        'label': session_store.models_label(*models),
        'capture_shared': SHARED_CAPTURE if shared else '',
    }
