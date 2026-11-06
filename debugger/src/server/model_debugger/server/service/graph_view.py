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

"""A model's graph, as the Graph Diff view reads it.

The routes only read: a graph is converted as a Session of its model is
initialized (model_graphs), never on a request.

GET /api/artifacts/{id}/graph answers where the graph is: `status` ('idle',
'converting', 'completed', 'failed', or 'unavailable' with the `text` why:
no turn of the PyTorch model exported its graph yet, and how to get one);
`error` once failed; and once completed its `signatures` (each `{name,
layers, layer_ids}`, `layers` how many layers its points are in,
`layer_ids` the layers it has operators of, the ones it computes: a prefill
computes only the layers whose KV cache it fills), the `layer_stems` its
layers are named by, and `architecture_text`, why it has no architecture
('' if it has one). The architecture never fails the answer.

GET /api/artifacts/{id}/graph/{signature} answers one signature's graph as
a Model Explorer graph collection of its own (model_graph.collection()):
the converter's graph, slimmed, each operator grouped in its own layer
(model_graph.regroup()) and each operator output carrying its point. A
PyTorch model's is the execution graph its runner exported, each module
LiteRT-LM runs as one operator folded into one node (pytorch_graph.fold()),
each anchor's output carrying its point and every other output why it has
none.

GET /api/artifacts/{id}/architecture answers the model's architecture
(model_graphs.ModelGraphs.architecture()): its skeleton, each anchor of
each layer in each signature's graph, each block's region of that graph
(the operators between its anchors), and the modules not in the skeleton.
"""

from typing import Any

from model_debugger.server.graph import model_graph
from model_debugger.server.session import model_graphs
from model_debugger.server.session import model_store
from model_debugger.server.tap import structure_copy

_RUNS = {
    'always': 'runs in every chat',
    'long_prompts': 'runs only for long prompts',
    'speculative_decoding': 'runs only with speculative decoding',
    'unknown': 'LiteRT-LM is not known to run it',
}


def status(
    state: model_graphs.State,
    facts: model_graphs.Facts | None,
    architecture: dict[str, Any] | None,
) -> dict[str, Any]:
  """GET /api/artifacts/{id}/graph's answer.

  Args:
    state: Where the model's graph is.
    facts: What the server says of it, once completed.
    architecture: Its architecture answer, once completed.

  Returns:
    The graph status, signatures and architecture details as a dictionary.
  """
  view = {'status': state.status}
  if state.error:
    view['error'] = state.error
  if state.text:
    view['text'] = state.text
  if state.status != model_graphs.COMPLETED:
    return view
  return {
      **view,
      'signatures': [
          {
              'name': entry['name'],
              'layers': facts.layers[entry['name']],
              'layer_ids': list(facts.layer_ids[entry['name']]),
          }
          for entry in state.manifest['signatures']
      ],
      'layer_stems': facts.layer_stems.stems,
      'architecture_text': (architecture or {}).get('text', ''),
  }


def graph(
    graphs: model_graphs.ModelGraphs,
    model: model_store.Model,
    signature: str,
) -> dict[str, Any]:
  """GET /api/artifacts/{id}/graph/{signature}'s answer."""
  found = graphs.graph(model, signature)
  manifest = graphs.state(model.artifact).manifest
  entry = next(e for e in manifest['signatures'] if e['name'] == signature)
  if model.runtime == model_store.PYTORCH:
    source = entry.get('source', {})
    return model_graph.collection(
        found,
        f'{model.name} · {signature}',
        {
            'signature': signature,
            'runs': _RUNS.get(entry['runs'], entry['runs']),
            'model sha256': model.sha256,
            'exported': (
                "pytorch-collector's to_model_explorer(), by the PyTorch "
                f'runner: the {signature} pass of step {entry["step"]}, '
                f'turn {source.get("turn", "?")} of Session '
                f'{source.get("session", "?")}'
            ),
            'folded': (
                'Each module LiteRT-LM runs as one operator is one node, '
                'labelled by its class, its ATen operators not expanded '
                '(attribute folded); every other operator is an ATen '
                'operator. Parameters are attributes of the node that '
                'reads them.'
            ),
            'grouped': (
                'Each operator is under the module path it ran in, as '
                'pytorch-collector records it; what the pass read from '
                'outside is under inputs.'
            ),
        },
    )
  return model_graph.collection(
      found,
      f'{model.name} · {signature}',
      {
          'signature': signature,
          'runs': _RUNS.get(entry['runs'], entry['runs']),
          'model sha256': model.sha256,
          'converter': manifest['identity'],
          'not shown': (
              'Composite operators are not expanded; constant values and '
              f'quantization longer than {model_graph.TEXT_LIMIT} '
              'characters are not kept, nor the values of constants of '
              f'more than {structure_copy.KEPT_BYTES} bytes, the weights, '
              'which the converter is not given.'
          ),
          'grouped': (
              'Each operator is in the layer its outputs are named by; '
              'one the converter grouped in another layer says where, as '
              f'{model_graph.CONVERTER_NAMESPACE}. One that several layers '
              'read stays where the converter grouped it.'
          ),
      },
  )
