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

"""The anchors a LiteRT-LM model's Session captures by default.

A Session that captures the anchors (session_store's `tap_anchors`) observes,
of each LiteRT-LM model its runs load, the tap point of every anchor of every
decoder layer of every signature its graph binds them in
(model_graphs.ModelGraphs.architecture(): `bindings[signature][layer]
.anchors[anchor].point`), as one set of points: a layer's input is most
often the output of the layer before, one point. An anchor is skipped, with
the binding's reason, when it is not bound (a prefill signature computes
only the layers whose KV cache it fills) or bound to what cannot be observed
(the first layer's input is the signature's input embeddings, no operator's
output). A signature that runs only with speculative decoding, which cannot
be enabled, is skipped whole: its anchors would capture nothing.

The anchors are found in the model's graph, which is converted as a Session
of the model is first initialized (model_graphs): until then they are not
known (PENDING). A model of no architecture the server has, or whose
architecture cannot be bound, has none (NONE), and says why.
"""

import collections
import dataclasses
from typing import Any

from model_debugger.server.analysis import tensor_analysis
from model_debugger.server.session import model_graphs as model_graphs_lib
from model_debugger.server.session import model_store
from model_debugger.server.session import tap_scans
from model_debugger.server.tap import tap_scan

FOUND = 'found'
PENDING = 'pending'
NONE = 'none'
# The signatures whose anchors are skipped whole, and why.
_SKIPPED_RUNS = {
    tap_scan.RUNS_SPECULATIVE: (
        'it runs only with speculative decoding, which cannot be enabled '
        'for now, so its anchors would capture nothing.'
    ),
}


@dataclasses.dataclass(frozen=True)
class Signature:
  """The anchors of one signature.

  Attributes:
      name: The signature.
      point_ids: Its points captured, in the model's order.
      layers: The layers they are anchors of, increasing.
      skipped: Each anchor skipped, as the user reads it: the places, then the
        reason, one line per reason.
      text: What it captures, as the user reads it.
  """

  name: str
  point_ids: tuple[str, ...]
  layers: tuple[int, ...]
  skipped: tuple[str, ...]
  text: str


@dataclasses.dataclass(frozen=True)
class Anchors:
  """The anchors a model's Session captures by default.

  Attributes:
      status: FOUND, PENDING (its graph is not converted) or NONE.
      text: What is captured, or why nothing is, as the user reads it.
      point_ids: Every signature's points, each once, in the model's order.
      signatures: Each signature's, in the model's order; () unless FOUND.
  """

  status: str
  text: str
  point_ids: tuple[str, ...] = ()
  signatures: tuple[Signature, ...] = ()


def find(
    graphs: model_graphs_lib.ModelGraphs | None,
    scans: tap_scans.TapScans,
    model: model_store.Model,
) -> Anchors:
  """The anchors a Session of a LiteRT-LM model captures by default.

  It converts nothing: a model whose graph is not kept is PENDING.

  Args:
    graphs: The models' graphs; None when the server keeps none.
    scans: The models' scans; the model is scanned now if none is kept.
    model: A LiteRT-LM model.

  Returns:
    The default anchors and signature metadata for the model.

  Raises:
    ValueError, OSError: The model cannot be scanned.
  """
  if graphs is None:
    return Anchors(NONE, 'This server keeps no graphs, so no anchors.')
  state = graphs.state(model.artifact)
  if state.status != model_graphs_lib.COMPLETED:
    return Anchors(PENDING, _pending_text(model, state))
  found = graphs.architecture(model)
  known = found.get('architecture')
  if known is None:
    return Anchors(
        NONE,
        f'{model.name} has no default anchors: '
        f'{found.get("error") or found.get("text")}',
    )
  scan = scans.scan(model)
  order = {point.id: index for index, point in enumerate(scan.points)}
  anchors = list(tensor_analysis.anchor_labels(found['skeleton']))
  runs = {signature.name: signature.runs for signature in scan.signatures}
  signatures = []
  for name in sorted(found['bindings'], key=_signature_order(scan)):
    skipped_whole = _SKIPPED_RUNS.get(runs.get(name, ''))
    if skipped_whole is not None:
      signatures.append(
          Signature(name, (), (), (), f'{name}: skipped; {skipped_whole}')
      )
      continue
    signatures.append(_signature(name, found['bindings'][name], anchors, order))
  unique_point_ids = set()
  for signature in signatures:
    unique_point_ids.update(signature.point_ids)
  point_ids = sorted(
      unique_point_ids,
      key=order.__getitem__,
  )
  captured = [s for s in signatures if s.point_ids]
  text = (
      f"Every decoder layer's anchors ({', '.join(anchors)}), as "
      f"{known['name']}'s graph binds them: {len(point_ids)} tensor "
      f'output{_plural(len(point_ids))} of {len(captured)} '
      f'signature{_plural(len(captured))}, captured at every step their '
      'signature runs. A layer whose input is the output of the layer '
      'before captures that tensor once.'
  )
  return Anchors(FOUND, text, tuple(point_ids), tuple(signatures))


def _pending_text(
    model: model_store.Model, state: model_graphs_lib.State
) -> str:
  """Why a model's anchors are not known: its graph is not converted."""
  where = {
      model_graphs_lib.IDLE: 'is converted as the Model Server first turns on',
      model_graphs_lib.CONVERTING: 'is being converted',
      model_graphs_lib.FAILED: f'could not be converted: {state.error}',
  }.get(state.status, state.text)
  return (
      f"{model.name}'s anchors are found in its graph, which {where}. "
      "If the server has the model's architecture (Gemma 4), every decoder "
      "layer's anchors are captured then; if not, no tensor is unless one "
      'is chosen.'
  )


def _signature_order(scan: tap_scan.Scan) -> Any:
  """Sorts signatures in the model's order; unknown ones last, by name."""
  order = {signature.name: i for i, signature in enumerate(scan.signatures)}
  return lambda name: (order.get(name, len(order)), name)


def _signature(
    name: str,
    layers: list[dict[str, Any]],
    anchors: list[str],
    order: dict[str, int],
) -> Signature:
  """One signature's anchors: its points, and the anchors it skips."""
  points = set()
  of_layers = set()
  # Reason -> layer -> the anchors skipped for it, in the skeleton's order.
  skipped = collections.defaultdict(lambda: collections.defaultdict(list))
  for entry in layers:
    for anchor, bound in entry['anchors'].items():
      point = bound.get('point') if bound.get('bound') else None
      if point is not None and bound.get('observable') and point in order:
        points.add(point)
        of_layers.add(entry['layer'])
        continue
      reason = bound.get('reason') or (
          'The model has no tap point of this output.'
      )
      skipped[reason][entry['layer']].append(anchor)
  lines = tuple(
      f'{_places(by_layer, anchors)}: {reason}'
      for reason, by_layer in skipped.items()
  )
  count = 0
  for by_layer in skipped.values():
    for a in by_layer.values():
      count += len(a)
  ordered = tuple(sorted(points, key=order.__getitem__))
  text = (
      f'{name}: {len(ordered)} tensor output{_plural(len(ordered))}, the '
      f'anchors of {len(of_layers)} layer{_plural(len(of_layers))}'
  )
  if count:
    text += f'; {count} anchor{_plural(count)} skipped'
  return Signature(name, ordered, tuple(sorted(of_layers)), lines, text + '.')


def _places(by_layer: dict[int, list[str]], anchors: list[str]) -> str:
  """Where the anchors skipped for one reason are, as the user reads it."""
  parts = []
  for layer, skipped in sorted(by_layer.items()):
    which = (
        'every anchor'
        if set(skipped) >= set(anchors)
        else ', '.join(sorted(skipped, key=_rank(anchors)))
    )
    parts.append(f'layer {layer} {which}')
  text = '; '.join(parts)
  return text[:1].upper() + text[1:]


def _rank(anchors: list[str]):
  return lambda anchor: (
      anchors.index(anchor) if anchor in anchors else len(anchors),
      anchor,
  )


def _plural(count: int) -> str:
  return '' if count == 1 else 's'
