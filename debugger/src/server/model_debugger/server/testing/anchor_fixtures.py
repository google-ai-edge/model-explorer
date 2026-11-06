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

"""The fixture model's anchors, as a Gemma 4 graph would bind them.

What model_graphs.ModelGraphs.architecture() answers of tap_fixtures'
model if its graph were of Gemma 4: a skeleton of two layers of the four
anchors, and per signature, per layer, each anchor's binding.
"""

from typing import Any

ANCHORS = ('input', 'intermediate.1', 'intermediate.2', 'output')


def skeleton(layers: int = 2) -> dict[str, Any]:
  """A skeleton of `layers` layers of the four anchors."""
  return {
      'layers': [{'def': 'layer'}] * layers,
      'semantic_graph': {
          'layer': {'anchors': [{'id': anchor} for anchor in ANCHORS]}
      },
  }


def bound(point: str | None, reason: str = '') -> dict[str, Any]:
  """An anchor bound to a point; one that cannot be observed says why."""
  return {
      'bound': True,
      'point': point,
      'observable': point is not None and not reason,
      'reason': reason,
  }


def unbound(reason: str) -> dict[str, Any]:
  """An anchor not bound, and why."""
  return {'bound': False, 'reason': reason}


def bindings() -> dict[str, Any]:
  """The fixture model's anchors, as a Gemma 4 graph would bind them.

  decode computes both layers: layer 0's input is the embeddings, its
  output layer 1's input; layer 1's intermediate.2 is a string. prefill_16
  computes layer 0 alone; verify, which never runs, both.

  Returns:
    A mapping of signature name to per-layer anchor bindings.
  """
  embeddings = (
      "It is the input embeddings of {}, no operator's output; it cannot "
      'be observed.'
  )
  return {
      'decode': [
          {
              'layer': 0,
              'anchors': {
                  'input': bound(None, embeddings.format('decode')),
                  'intermediate.1': bound('decode:0:0'),
                  'intermediate.2': bound('decode:2:0'),
                  'output': bound('decode:1:0'),
              },
          },
          {
              'layer': 1,
              'anchors': {
                  'input': bound('decode:1:0'),
                  'intermediate.1': bound('decode:0:0'),
                  'intermediate.2': bound(
                      'decode:3:0', 'Its type, string, is not a number.'
                  ),
                  'output': bound('decode:1:0'),
              },
          },
      ],
      'prefill_16': [
          {
              'layer': 0,
              'anchors': {
                  'input': bound(None, embeddings.format('prefill_16')),
                  'intermediate.1': bound('prefill_16:0:0'),
                  'intermediate.2': bound('prefill_16:2:0'),
                  'output': bound('prefill_16:1:0'),
              },
          },
          {
              'layer': 1,
              'anchors': {
                  anchor: unbound('prefill_16 has no operator of layer 1.')
                  for anchor in ANCHORS
              },
          },
      ],
      'verify': [
          {
              'layer': 0,
              'anchors': {anchor: bound('verify:0:0') for anchor in ANCHORS},
          },
      ],
  }


def architecture(found: dict[str, Any] | None = None) -> dict[str, Any]:
  """ModelGraphs.architecture()'s answer for the fixture model."""
  return {
      'architecture': {'id': 'gemma4', 'name': 'Gemma 4'},
      'family': 'Gemma 4',
      'text': '',
      'error': '',
      'skeleton': skeleton(),
      'bindings': bindings() if found is None else found,
  }
