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

"""Intermediate tensors: position-analysis, position-range, position-tensor.

A Session observes points of its models (tap_scan): operator outputs, each
of one signature, which the runtime captures at every step that signature
runs (tap_model, capture_index). What a side observes is its model's scan of
the points chosen (`Side`, `Observed`); where they are is in the capture
index: each forward pass holds its signature's points, each under the key
the model's scan gives it (tap_model.capture_key, the one source of the
keys).

Places. A point's place is its layer and its anchor, both read from the
model (tap_scan.Point): `output` for its layer's output, of whichever
signature, and otherwise its own id. A point in no layer has no place. A
Session that captures the anchors places the points its model's graph binds
to an anchor at that anchor instead, as two runtimes are (below). Two sides
that load one model have the same places, and a place is the same tensor on
both (`pairing`); points of two models of one runtime that differ are not
paired here.

Two sides of different runtimes (PyTorch and LiteRT-LM) of one architecture
are paired by the architecture's anchors instead: a place is a layer and an
anchor of the skeleton (`input`, `intermediate.1`, `intermediate.2`,
`output` of Gemma 4), each side's point there the one its runtime binds to
it (Side.places): of LiteRT-LM, the points observed that the model's graph
binds to an anchor (architecture.bind(), anchor_places()); of PyTorch, the
anchors its runner writes of every layer (pytorch_side()). A PyTorch side is
placed by its anchors whatever the other side is, so two PyTorch sides of
one model pair by them too. A PyTorch point has no fixed shape: its token
axis is ROWS, as long as the forward pass's rows, which a decode step's one
and a prefill's the positions it fed. Values of two runtimes are compared
in float64 whatever dtype each stores; one runtime's two sides are compared
only when both store one.

Beside the anchors, a LiteRT-LM point chosen of two runtimes' Session is
placed at the semantic operator it is the output of, (layer, `attn/q_proj`)
say, where the PyTorch module of that operator is certain
(graph/counterparts), and the PyTorch side observes that module there too
(its runner's `post_module_<module>_<edge>`, capture_index.module_key); its
values are compared flat when one side's is (q_proj's [8, 256] after
LiteRT-LM's reshape, [2048] as the Linear returns it). Every other point
chosen has no place, and says why (`Observed.notes`, the analysis'
`unpaired`).

Pairs made by hand (session/mappings), of two different models: each is a
place of its own, (layer, `manual:<pair id>`) labelled by the pair, its
point on each side the tensor the pair names in its signature
(prepared_models places them); the layer is the ref point's, else the
target's. Two different models of one runtime pair by those places alone.
A PyTorch module such a pair names that the side's runner was not asked
for when the Session started (its Load's observeModules) says so where it
has no tensor: it is asked for from the next start.

Columns. A stage's columns are those of its rows in the server's comparison
(kv_analysis.Alignment), each read at its own position on each side; without
an alignment, a column is the position on both sides. At a column, each
side's value of a place is the tensor of the point of that place in the
signature of the forward pass the side's position is read from (`produced`,
the one place that decides it): the forward pass that produced the token at
the position, which is the one that fed the position before it, of that turn
before any rewrite of the context (capture_index.feeds). It is the step
Final Logits shows at the same column (token_analysis): at a Decode column
p, the decode step that reads token p - 1 and produces token p; at the
first, the step that reads the input's last token. At a Prefill column p, it
is the row of the prefill chunk that fed token p - 1, whose output is the
model's view of position p; the turn's first input token, which no forward
pass of the turn produced, has none. The value is a decode step's whole
tensor, or one row of a prefill chunk's, along its token axis, the one axis
of the chunk's row count in the shape the model gives the point (a tensor
without exactly one is not split by position). Along each context axis
(Point.context_axes, the context length in a capture) only the positions the
context holds once that forward pass has fed its own, 0 to that one, are
read: the rest is never written for this step. The two sides' values are
compared as stored, in float64, by metrics.compare, when both store one
dtype (`cell`s).

Which tensors a stage has is one rule (`_held`), read from the capture index
alone: a side holds a place where a forward pass that produced a position of
the stage holds its point's tensor, split by position. A stage has intermediate
tensors when both sides hold a place (`availability`), and its `batches` are
the graphs of the forward passes that hold those places; its places are those
either side holds a tensor of, and it has a cell of each at each column.

Every place, cell, row and bin that has no value says why, as reasons
(metrics.reason) of one side or of the pair, of the codes of `REASONS`, of
metrics.REASONS, or `DTYPE_MISMATCH`.

Reading. A stage reads each file once where it can: a side's whole tensors
are kept, least recently used first out, up to `READ_BYTES`; one cell reads
only its part of each side's file. A file's header is checked against the
shape the model gives its point before its data is read. A stage's columns
under several alignments are compared in one pass (`stages`): a cell whose
sources a cell of another alignment has too, which up to the turn's
divergence is every cell, is compared once, and every file read once.
"""

import collections
from collections.abc import Iterable
import dataclasses
import itertools
import math
import pathlib
import re
import types
from typing import Any

from model_debugger.server.analysis import comparability
from model_debugger.server.analysis import kv_analysis
from model_debugger.server.analysis import metrics
from model_debugger.server.analysis import statistics
from model_debugger.server.architectures import skeleton
from model_debugger.server.capture import capture_index
from model_debugger.server.capture import tensor_file
from model_debugger.server.graph import counterparts
from model_debugger.server.graph import pytorch_binding
from model_debugger.server.tap import tap_model
from model_debugger.server.tap import tap_scan
import numpy as np

# The anchor of a layer's output; other anchors are point ids.
OUTPUT = 'output'
OUTPUT_LABEL = 'Layer output'
# What a cell holds of metrics.compare, by the metric catalog's ids.
METRICS = ('cosine_similarity', 'cosine_distance', 'relative_l2', 'max_abs')
REDUCTIONS = ('min', 'max', 'mean')
MAX_BINS = kv_analysis.MAX_BINS
# The most elements position-tensor answers of one tensor.
MAX_ELEMENTS = 262_144
# About how many bytes of whole tensors a stage keeps of each side as it
# reads them.
READ_BYTES = 64 * 2**20
# The most reasons a row or a bin lists.
MAX_REASONS = 4
LABEL = 'Intermediate Tensors'
PROVENANCE = {
    'kind': 'captured',
    'label': 'Tensors both runners captured, compared by the server',
}
# The stages, and which of a side's turn's tokens are each one's positions.
STAGE_TOKENS = {'prefill': 'input', 'decode': 'output'}

# The reason codes of this module, and their words as the metric catalog
# lists them; a response's reason carries its own words, which may name more.
NOT_OBSERVED = 'tensors_not_observed'
MODEL_UNREADABLE = 'tensor_model_unreadable'
NOT_PAIRED = 'tensors_not_paired'
NO_LAYER = 'tensor_no_layer'
GRAPH_NOT_RUN = 'tensor_graph_not_run'
NOT_IN_STAGE = 'tensor_not_in_stage'
NOT_CAPTURED = 'tensor_not_captured'
POSITION_MISSING = 'tensor_position_missing'
POSITION_NOT_PRODUCED = 'tensor_position_not_produced'
OTHER_GRAPH = 'tensor_other_graph'
SHAPE_UNEXPECTED = 'tensor_shape_unexpected'
NO_TOKEN_AXIS = 'tensor_no_token_axis'
PLACE_MISSING = 'tensor_place_missing'
TOO_LARGE = 'tensor_too_large'
POINT_NOT_OBSERVED = 'tensor_point_not_observed'
SIDE_NOT_OBSERVED = 'tensor_side_not_observed'
NO_ANCHOR = 'tensor_no_anchor'
OTHER_ARCHITECTURE = 'tensor_other_architecture'
NO_COUNTERPART = 'tensor_no_counterpart'
MANUAL_NOT_OBSERVED = 'tensor_manual_not_observed'
LAYERS_ONE_SIDE = 'tensor_layers_one_side'
REASONS = types.MappingProxyType({
    NOT_OBSERVED: 'This Session observes no tensors.',
    MODEL_UNREADABLE: 'The points a model observes cannot be read from it.',
    NOT_PAIRED: (
        'Ref and Target load different models; their tensors are not paired.'
    ),
    NO_LAYER: 'No point observed is in a transformer layer.',
    GRAPH_NOT_RUN: 'The graph of the point did not run in this turn.',
    NOT_IN_STAGE: 'The graph of the point produced no position of this stage.',
    NOT_CAPTURED: 'The capture holds no tensor of the point at this step.',
    POSITION_MISSING: 'The side has no token at this column.',
    POSITION_NOT_PRODUCED: (
        'No forward pass of this turn produced the token at the position.'
    ),
    OTHER_GRAPH: (
        'Another graph produced the position; it holds no such tensor.'
    ),
    SHAPE_UNEXPECTED: (
        'The captured tensor is not of the shape the model gives it.'
    ),
    NO_TOKEN_AXIS: (
        'The tensor has no one axis of the prefill chunk’s '
        'tokens; it is not split by position.'
    ),
    PLACE_MISSING: 'The layer has none of the sync points chosen.',
    TOO_LARGE: f'A tensor of more than {MAX_ELEMENTS} elements is not listed.',
    POINT_NOT_OBSERVED: (
        'This Session does not observe the point; choose it '
        'in the Session editor to compare it.'
    ),
    SIDE_NOT_OBSERVED: 'One side observes no tensors.',
    NO_ANCHOR: (
        'No point the side observes is an anchor of its architecture; '
        'the two runtimes are paired by anchor alone.'
    ),
    OTHER_ARCHITECTURE: (
        'The two runtimes run models of different or unknown '
        'architectures; their tensors are not paired.'
    ),
    NO_COUNTERPART: (
        'PyTorch has no counterpart for a LiteRT-LM point chosen; '
        'only the anchors are captured there.'
    ),
    MANUAL_NOT_OBSERVED: (
        'A tensor pair made by hand names a tensor this '
        'Session does not capture.'
    ),
    LAYERS_ONE_SIDE: (
        'Only one side computed these layers in this stage; '
        'there is nothing to compare, and they are not shown.'
    ),
})
# What the words of a reason call a stage and a side.
STAGE_LABELS = {'prefill': 'Prefill', 'decode': 'Decode'}
SIDE_LABELS = {'ref': 'Reference', 'target': 'Target'}
# The token axis of a point whose forward passes feed as many rows as they
# do (a PyTorch point's): as long as each pass's rows.
ROWS = -1
LITERT_LM = 'LiteRT-LM'
PYTORCH = 'PyTorch'
# The two sides store different dtypes: kv_analysis's code, in its words.
DTYPE_MISMATCH = 'dtype_mismatch'
# What the anchor of a place of a pair made by hand starts with.
MANUAL = 'manual:'


def manual_anchor(pair_id: str) -> str:
  """The anchor of the place of a pair made by hand."""
  return MANUAL + pair_id


def is_manual(anchor: str) -> bool:
  """Whether an anchor is the place of a pair made by hand."""
  return anchor.startswith(MANUAL)


@dataclasses.dataclass(frozen=True)
class Side:
  """What one side observes.

  Attributes:
      model_sha256: SHA-256 of the model the points are of (the source, not a
        copy that observes them).
      points: The points observed, as the model's scan lists them; a PyTorch
        side's as pytorch_side() makes them.
      runtime: LITERT_LM or PYTORCH.
      architecture: The id of its model's architecture, if it has one.
      places: Per point id, the places (layer, anchor of the architecture) its
        runtime binds it to: what two runtimes are paired by.
      labels: Per anchor of the architecture, in its order, its label.
      keys: Per point id, its key in the capture, where it is not the one
        tap_model gives it (a PyTorch point's).
      anchors: The architecture's anchors, in its order: the places a Session
        captures by default, not an operator chosen inside the layers.
  """

  model_sha256: str
  points: tuple[tap_scan.Point, ...]
  runtime: str = LITERT_LM
  architecture: str | None = None
  places: dict[str, tuple[tuple[int, str], ...]] = dataclasses.field(
      default_factory=dict
  )
  labels: dict[str, str] = dataclasses.field(default_factory=dict)
  keys: dict[str, str] = dataclasses.field(default_factory=dict)
  anchors: tuple[str, ...] = ()


def pytorch_side(
    model_sha256: str,
    bound: pytorch_binding.Bound,
    dtype: str,
    modules: Iterable[counterparts.Counterpart] = (),
    labels: dict[str, str] | None = None,
) -> Side:
  """What a PyTorch side that observes its anchors observes.

  One point per anchor of every layer per signature of the runner, keyed
  as its runner writes it (capture_index.anchor_key), of shape [1, ROWS,
  width]; and one per module paired with a LiteRT-LM point of the other
  side per signature, placed at that point's semantic operator, keyed
  capture_index.module_key, of shape [1, ROWS, *its dims].

  Args:
      model_sha256: SHA-256 of the model.
      bound: Where the model's anchors are (pytorch_binding.anchors()).
      dtype: The dtype it stores, as far as known; '' if not.
      modules: The LiteRT-LM points paired with a module (counterparts.find());
        each module once, whichever points.
      labels: Each place's label, in the layers' order, if not the anchors'
        alone.

  Returns:
      The side, its points placed and keyed.
  """
  points, places, keys = [], {}, {}
  for index, anchor in enumerate(bound.anchors):
    for signature in bound.signatures:
      point_id = pytorch_binding.point_id(signature, anchor)
      reads = 'reads' if anchor.edge == 'in' else 'returns'
      points.append(
          tap_scan.Point(
              id=point_id,
              signature=signature,
              subgraph=0,
              op=index,
              output=0,
              tensor=-1,
              tensor_name=f'{anchor.module} ({reads} {anchor.path})',
              shape=(1, ROWS, bound.width),
              dtype=dtype,
              operator='',
              output_name=None,
              context_axes=(),
              reason='',
              layer=anchor.layer,
              layer_output=anchor.anchor == OUTPUT,
          )
      )
      places[point_id] = ((anchor.layer, anchor.anchor),)
      keys[point_id] = capture_index.anchor_key(anchor.layer, anchor.anchor)
  seen = set()
  for paired in modules:
    module = paired.module
    if module is None or (module.module, module.edge) in seen:
      continue
    seen.add((module.module, module.edge))
    for signature in bound.signatures:
      point = module_point(
          signature,
          module.module,
          module.edge,
          module.dims,
          dtype,
          layer=paired.layer,
          op=len(bound.anchors) + len(seen),
          path=module.path,
      )
      points.append(point)
      places[point.id] = ((paired.layer, paired.op),)
      keys[point.id] = capture_index.module_key(module.module, module.edge)
  if labels is None:
    labels = {}
    for anchor in bound.anchors:
      labels.setdefault(anchor.anchor, anchor.label)
  return Side(
      model_sha256,
      tuple(points),
      runtime=PYTORCH,
      architecture=bound.architecture,
      places=places,
      labels=labels,
      keys=keys,
      anchors=tuple(dict.fromkeys(anchor.anchor for anchor in bound.anchors)),
  )


def module_point(
    signature: str,
    module: str,
    edge: str,
    dims: Iterable[int],
    dtype: str,
    *,
    layer: int | None,
    op: int,
    path: str = '',
) -> tap_scan.Point:
  """The point of what a PyTorch module returns or reads in a signature.

  Of shape [1, ROWS, *dims], its id pytorch_binding.module_point_id();
  its key in the capture is capture_index.module_key().

  Args:
      signature: The runner's signature.
      module: The module's path.
      edge: 'out' for what it returns, 'in' for its argument 0.
      dims: Its shape after [batch, rows].
      dtype: The dtype it stores, as far as known; '' if not.
      layer: Its layer, if it has one.
      op: Its order among the side's points.
      path: Its path in pytorch-collector's record; by its edge if ''.

  Returns:
      The point.
  """
  reads = 'reads' if edge == 'in' else 'returns'
  path = path or ('args.0' if edge == 'in' else 'output')
  return tap_scan.Point(
      id=pytorch_binding.module_point_id(signature, module, edge),
      signature=signature,
      subgraph=0,
      op=op,
      output=0,
      tensor=-1,
      tensor_name=f'{module} ({reads} {path})',
      shape=(1, ROWS, *dims),
      dtype=dtype,
      operator='',
      output_name=None,
      context_axes=(),
      reason='',
      layer=layer,
  )


def anchor_places(
    bindings: dict[str, list[dict[str, Any]]], point_ids: set[str]
) -> dict[str, tuple[tuple[int, str], ...]]:
  """The places of a LiteRT-LM model's points that its graph binds.

  Args:
      bindings: Per signature, per layer, its anchors' bindings
        (architecture.bind()).
      point_ids: The points observed.

  Returns:
      Per point observed that a bound anchor is, every place it is: the
      input of a layer is often the output of the one before.
  """
  found = collections.defaultdict(list)
  for layers in bindings.values():
    for entry in layers:
      for anchor, bound in entry['anchors'].items():
        point = bound.get('point') if bound.get('bound') else None
        if point in point_ids:
          found[point].append((entry['layer'], anchor))
  return {point: tuple(places) for point, places in found.items()}


def anchor_labels(document: dict[str, Any]) -> dict[str, str]:
  """Each anchor of a skeleton's layers, in order, by what it is.

  As skeleton.anchor_label() names it: by the block of the layer it ends.

  Args:
      document: The skeleton's document, with its `layers` and `semantic_graph`.

  Returns:
      Per anchor id, its label: that of the first layer that has it.
  """
  labels = {}
  for layer in document['layers']:
    definition = document['semantic_graph'][layer['def']]
    for anchor in definition['anchors']:
      labels.setdefault(anchor['id'], skeleton.anchor_label(definition, anchor))
  return labels


@dataclasses.dataclass(frozen=True)
class Observed:
  """What a Session observes: each side's points, or why it has none.

  Attributes:
      sides: Per side, its points; empty with `reason`.
      reason: Why nothing can be compared (metrics.reason), or None.
      notes: Why each point chosen that is placed nowhere is not (NO_COUNTERPART
        reasons, one per point).
  """

  sides: dict[str, Side]
  reason: dict[str, Any] | None = None
  notes: tuple[dict[str, Any], ...] = ()


def nothing(code: str = NOT_OBSERVED, text: str | None = None) -> Observed:
  """What a Session that observes nothing comparable observes."""
  return Observed({}, metrics.reason(code, text or REASONS[code]))


def anchor_of(point: tap_scan.Point) -> str:
  """A point's anchor: OUTPUT for its layer's output, else its id."""
  return OUTPUT if point.layer_output else point.id


def label(point: tap_scan.Point) -> str:
  """What the user knows a point's anchor by, from its tensor's name.

  The name's scope inside its layer, without the layer's number: the
  scope up to the layer's component, `<stem>_<N>` by the stem the model's
  graph names its layers by (Point.layer_stem), is dropped, and so is that
  component as a prefix of a later one (`layer_3.attn` is `attn`).

  Args:
      point: The point.

  Returns:
      OUTPUT_LABEL for its layer's output; else its short scope and id.
  """
  if point.layer_output:
    return OUTPUT_LABEL
  name = point.tensor_name.split(';')[0]
  if point.layer_stem:
    component = re.escape(point.layer_stem) + r'_\d+'
    name = re.sub(rf'^(?:.*?/)?{component}/', '', name, count=1)
    name = re.sub(rf'(^|/){component}\.', r'\1', name)
  short = '/'.join(name.split('/')[-2:])
  return f'{short} · {point.id}'


@dataclasses.dataclass(frozen=True)
class Layout:
  """The places of a Session's points: what both sides observe alike.

  Attributes:
      points: Per side, per place (layer, anchor), the point of each signature
        that has one.
      anchors: The anchors, OUTPUT first, then by their points' order.
      labels: Each anchor's label.
      layers: The layers that have a place, increasing.
      reason: Why no place can be compared, or None.
      keys: Each point's key in the capture's files, by point id.
      cross: Whether the sides are of two runtimes, paired by the architecture's
        anchors.
      point_places: Per point id, its places.
      graphless: The ids of the points no graph node is of (PyTorch's).
      notes: Why each point chosen that is placed nowhere is not.
      defaults: The anchors of `anchors` that are an architecture's
        (Side.anchors), not an operator chosen inside the layers.
  """

  points: dict[str, dict[tuple[int, str], dict[str, tap_scan.Point]]]
  anchors: tuple[str, ...]
  labels: dict[str, str]
  layers: tuple[int, ...]
  reason: dict[str, Any] | None
  keys: dict[str, str] = dataclasses.field(default_factory=dict)
  cross: bool = False
  point_places: dict[str, list[tuple[int, str]]] = dataclasses.field(
      default_factory=dict
  )
  graphless: frozenset[str] = frozenset()
  notes: tuple[dict[str, Any], ...] = ()
  defaults: frozenset[str] = frozenset()

  def place_points(self) -> list[tap_scan.Point]:
    """The points that have a place, in the model's order.

    The ref side's, which are the target's of one model; of two
    runtimes, the ref side's and then the target's.

    Returns:
        The points, each once.
    """
    found = {}
    for side in capture_index.SIDES if self.cross else ('ref',):
      mine = {}
      for signatures in self.points.get(side, {}).values():
        mine.update({point.id: point for point in signatures.values()})
      for point in sorted(
          mine.values(), key=lambda p: (p.subgraph, p.op, p.output)
      ):
        found.setdefault(point.id, point)
    return list(found.values())

  def places_of(self, point: tap_scan.Point) -> list[tuple[int, str]]:
    """A point's places."""
    return self.point_places.get(point.id, [])


def layout(observed: Observed) -> Layout:
  """The places of what a Session observes, and whether they pair."""
  empty = Layout({}, (), {}, (), None)
  if observed.reason is not None:
    return dataclasses.replace(empty, reason=observed.reason)
  sides = observed.sides
  for side in capture_index.SIDES:
    if not sides[side].points:
      return dataclasses.replace(
          empty,
          reason=metrics.reason(
              SIDE_NOT_OBSERVED,
              f'{capture_index.SIDE_NAMES[side]} observes no tensors.',
              side,
          ),
      )
  # The one rule of how two sides' tensors pair (comparability).
  pairing = comparability.tensor_pairing(*(
      comparability.Model(
          sides[side].model_sha256,
          sides[side].runtime,
          architecture=sides[side].architecture,
      )
      for side in capture_index.SIDES
  ))
  cross = pairing in (comparability.BY_ANCHORS, comparability.NOT_PAIRED)
  if cross:
    found = {side: sides[side].architecture for side in sides}
    if pairing == comparability.NOT_PAIRED:
      return dataclasses.replace(
          empty,
          reason=metrics.reason(
              OTHER_ARCHITECTURE,
              f'Ref is a {sides["ref"].runtime} model of architecture '
              f'{found["ref"] or "unknown"}, Target a '
              f'{sides["target"].runtime} model of architecture '
              f'{found["target"] or "unknown"}; two runtimes are '
              'paired by the anchors of one architecture.',
          ),
      )
  by_hand = False
  if pairing == comparability.BY_HAND:
    # Two models of one runtime pair by the pairs made by hand alone.
    by_hand = any(
        is_manual(anchor)
        for _, anchor in itertools.chain.from_iterable(
            sides['ref'].places.values()
        )
    )
    if not by_hand:
      return dataclasses.replace(empty, reason=_reason(NOT_PAIRED))
  points, point_places = {}, {}
  for side, observes in sides.items():
    places = collections.defaultdict(dict)
    by_anchor = cross or by_hand or observes.runtime == PYTORCH
    for point in observes.points:
      # A point its graph binds to an anchor is placed there; of one
      # runtime, any other is placed by its layer.
      mine = list(observes.places.get(point.id, ()))
      if by_hand:
        mine = [place for place in mine if is_manual(place[1])]
      if not mine and not by_anchor and point.layer is not None:
        mine = [(point.layer, anchor_of(point))]
      for place in mine:
        places[place][point.signature] = point
      point_places.setdefault(point.id, mine)
    points[side] = dict(places)
    if cross and not points[side]:
      return dataclasses.replace(
          empty,
          reason=metrics.reason(
              NO_ANCHOR,
              f'{capture_index.SIDE_NAMES[side]} observes no anchor of '
              'the layers of its architecture: capture the anchors '
              '(input, intermediate.1, intermediate.2, output), as a '
              'Session does by default, or choose the points the '
              'architecture binds to them; two runtimes are paired by '
              'anchor alone.',
              side,
          ),
      )
  if not points['ref'] and not points['target']:
    return dataclasses.replace(empty, reason=_reason(NO_LAYER))
  candidates = []
  for places in points.values():
    for signatures in places.values():
      candidates.extend(signatures.values())
  ordered = sorted(
      candidates,
      key=lambda p: (p.subgraph, p.op, p.output),
  )
  labels = {}
  for observes in sides.values():
    for anchor, text in observes.labels.items():
      labels.setdefault(anchor, text)
  anchors = {}
  for point in ordered:
    for _, anchor in point_places.get(point.id, []):
      anchors.setdefault(anchor, labels.get(anchor) or label(point))
  if labels:
    # The architecture's anchors, in its order.
    order = sorted(
        anchors,
        key=lambda a: (
            list(labels).index(a) if a in labels else len(labels),
            list(anchors).index(a),
        ),
    )
  else:
    order = sorted(anchors, key=lambda a: (a != OUTPUT, list(anchors).index(a)))
  layers = sorted(
      {layer for layer, _ in itertools.chain.from_iterable(points.values())}
  )
  keys = {}
  for observes in sides.values():
    for point in observes.points:
      keys[point.id] = observes.keys.get(point.id) or (
          tap_model.capture_key(point)
      )
  return Layout(
      points=points,
      anchors=tuple(order),
      labels=anchors,
      layers=tuple(layers),
      reason=None,
      keys=keys,
      cross=cross,
      point_places=point_places,
      graphless=frozenset(
          point.id
          for point in itertools.chain.from_iterable(
              observes.points
              for observes in sides.values()
              if observes.runtime == PYTORCH
          )
      ),
      notes=observed.notes,
      defaults=frozenset(
          anchor
          for anchor in itertools.chain.from_iterable(
              observes.anchors for observes in sides.values()
          )
          if anchor in anchors
      ),
  )


# Where each side's positions are read from.


def produced(entry: dict[str, Any]) -> dict[int, capture_index.Feed]:
  """The forward pass each position of a side's turn is read from.

  The one place that decides which forward pass a column reads, for every
  answer here (availability, batches, cells) and for the step the session
  view names at it: the one that produced the token at the position, which
  fed the position before it (capture_index.feeds, not a rewrite's). A
  decode column p is thus the step that reads p - 1 and produces p, the
  step Final Logits shows at p; a prefill column p the chunk's row that fed
  p - 1. The turn's first input token, which no forward pass of the turn
  produced, has none.

  Args:
      entry: A side's entry of the turn in the index.

  Returns:
      Per position, the forward pass that fed the position before it, and
      its row of that one.
  """
  return {
      position + 1: feed
      for position, feed in capture_index.feeds(entry).items()
  }


def _stage_positions(entry: dict[str, Any], stage_name: str) -> list[int]:
  return [token['position'] for token in entry[STAGE_TOKENS[stage_name]]]


def _expected(
    point: tap_scan.Point, context_length: int, rows: int = 1
) -> tuple[int, ...]:
  """The shape of a point's tensor in a capture at a context length.

  A ROWS axis is as long as the forward pass's `rows`.

  Args:
      point: The point.
      context_length: The context length the side loaded.
      rows: The forward pass's rows.

  Returns:
      The shape.
  """
  extents = []
  for axis, extent in enumerate(point.shape):
    if axis in point.context_axes:
      extents.append(context_length)
    elif extent == ROWS:
      extents.append(rows)
    else:
      extents.append(extent)
  return tuple(extents)


def _token_axis(
    point: tap_scan.Point, rows: int, context_length: int
) -> int | None:
  """A prefill tensor's axis of the chunk's rows, if it has exactly one.

  A point's ROWS axis, if it has one.

  Args:
      point: The point.
      rows: The chunk's rows.
      context_length: The context length the side loaded.

  Returns:
      The axis, or None.
  """
  if ROWS in point.shape:
    return point.shape.index(ROWS)
  axes = [
      axis
      for axis, extent in enumerate(_expected(point, context_length))
      if extent == rows and axis not in point.context_axes
  ]
  return axes[0] if len(axes) == 1 else None


def _unusable(
    forward: dict[str, Any],
    point: tap_scan.Point,
    key: str,
    context_length: int,
) -> str | None:
  """Why a forward pass gives no value of a point by position, or None.

  OTHER_GRAPH, NOT_CAPTURED or NO_TOKEN_AXIS; from the index and the model
  alone.

  Args:
      forward: The forward pass's entry in the index.
      point: The point.
      key: The point's key in the capture.
      context_length: The context length the side loaded.

  Returns:
      The code of why, or None when it gives one.
  """
  if forward['signature'] != point.signature:
    return OTHER_GRAPH
  if key not in forward['tensors']:
    return NOT_CAPTURED
  if forward['phase'] == 'prefill' and (
      _token_axis(point, forward['rows'], context_length) is None
  ):
    return NO_TOKEN_AXIS
  return None


@dataclasses.dataclass(frozen=True)
class _Held:
  """What a side holds of a stage's places (`_held`).

  Attributes:
      usable: Per place, the forward passes that produced a position of the
        stage and hold its point's tensor, split by position, in order.
      captured: The places some such forward pass holds a tensor of, split by
        position or not.
      ran: The signatures that ran in the turn.
      in_stage: The signatures of the forward passes that produced the stage's
        positions.
      no_axis: The ids of the points held without a token axis.
      uncaptured: Why the side's runner captured no tensors in the turn, as it
        says (capture_index's `observation`); '' if it did not say.
  """

  usable: dict[tuple[int, str], list[dict[str, Any]]]
  captured: set[tuple[int, str]]
  ran: set[str]
  in_stage: set[str]
  no_axis: set[str]
  uncaptured: str = ''


def _held(
    index: capture_index.CaptureIndex,
    turn: dict[str, Any],
    stage_name: str,
    places: Layout,
) -> dict[str, _Held]:
  """What each side holds of a stage's places.

  The one rule of which tensors a stage has, which availability, batches
  and the stage's places follow.

  Args:
      index: The chat's capture.
      turn: The turn's entry in the index.
      stage_name: 'prefill' or 'decode'.
      places: The Session's places.

  Returns:
      Per side, what it holds.
  """
  found = {}
  for side in capture_index.SIDES:
    entry = turn['sides'][side]
    context_length = index.sides[side]['load']['context_length']
    producers = produced(entry)
    forwards = {}
    for position in _stage_positions(entry, stage_name):
      if position in producers:
        forward = producers[position].forward
        forwards.setdefault(forward['id'], forward)
    usable = collections.defaultdict(list)
    captured, no_axis = set(), set()
    for forward in forwards.values():
      for place, signatures in places.points[side].items():
        point = signatures.get(forward['signature'])
        if point is None:
          continue
        why = _unusable(forward, point, places.keys[point.id], context_length)
        if why is None:
          usable[place].append(forward)
          captured.add(place)
        elif why == NO_TOKEN_AXIS:
          captured.add(place)
          no_axis.add(point.id)
    found[side] = _Held(
        usable=dict(usable),
        captured=captured,
        ran={forward['signature'] for forward in entry['forwards']},
        in_stage={f['signature'] for f in forwards.values()},
        no_axis=no_axis,
        uncaptured=_uncaptured(entry),
    )
  return found


def asked_modules(load: dict[str, Any]) -> set[str] | None:
  """The keys of the modules a side's Load asked its runner for.

  capture_index.module_key() of each `{module, edge}` of its runtime
  options' observeModules; None for a side whose runner was asked for
  none (a LiteRT-LM side's, or a PyTorch side's that observes only the
  anchors).

  Args:
      load: The side's Load, as the index keeps it.

  Returns:
      The keys, or None.
  """
  modules = (load.get('runtime_options') or {}).get('observeModules')
  if not isinstance(modules, list):
    return None
  return {
      capture_index.module_key(entry.get('module', ''), entry.get('edge', ''))
      for entry in modules
      if isinstance(entry, dict)
  }


def not_asked(key: str, asked: set[str] | None) -> str:
  """Why a module's tensor is not in a capture: not asked for; or ''.

  A module key (capture_index.MODULE_PREFIX) that the side's Load did not
  ask for (asked_modules()): it was chosen, or paired by hand, after the
  Session started, and is asked for from the next start.

  Args:
      key: The tensor's key in the capture.
      asked: The keys the side's Load asked for (asked_modules()), or None.

  Returns:
      Why, or '' when the key is no module's or was asked for.
  """
  if not key.startswith(capture_index.MODULE_PREFIX):
    return ''
  if asked is not None and key in asked:
    return ''
  module, _, edge = key[len(capture_index.MODULE_PREFIX) :].rpartition('_')
  what = 'what it reads' if edge == 'in' else 'what it returns'
  return (
      f'The PyTorch runner was not asked for {module}, {what}, when this '
      'Session started; a Session of these models started from now on '
      'captures it.'
  )


def _uncaptured(entry: dict[str, Any]) -> str:
  """Why a side's runner says it captured no tensors in a turn, or ''."""
  observation = entry.get('observation') or {}
  if observation.get('captured', True):
    return ''
  return (
      'The PyTorch runner captured no tensors in this turn: '
      f'{observation.get("reason") or "it did not say why"}.'
  )


def _common(held: dict[str, _Held]) -> list[tuple[int, str]]:
  """The places both sides hold a tensor of, split by position."""
  return sorted(set(held['ref'].usable) & set(held['target'].usable))


# Availability and batches, from the capture index alone.


def availability(
    index: capture_index.CaptureIndex,
    n: int,
    stage_name: str,
    observed: Observed,
) -> list[dict[str, Any]]:
  """Why a stage of a turn has no intermediate tensors; [] if it has.

  It has when both sides hold a place (`_held`). No tensor is read.

  Args:
      index: The chat's capture.
      n: A turn the index read.
      stage_name: 'prefill' or 'decode'.
      observed: What the Session observes.

  Returns:
      The reasons (metrics.reason), or [] when it has tensors.
  """
  places = layout(observed)
  if places.reason is not None:
    return [places.reason]
  return _unavailable(places, _held(index, index.turn(n), stage_name, places))


def _unavailable(
    places: Layout, held: dict[str, _Held]
) -> list[dict[str, Any]]:
  """Why both sides hold no place in common (`availability`); [] if so."""
  if _common(held):
    return []
  reasons = []
  for side in capture_index.SIDES:
    other = held['target' if side == 'ref' else 'ref']
    mine = held[side]
    # The side's points of the places it lacks; of every place, when the
    # other side lacks them all too.
    lacking = []
    for place, signatures in places.points[side].items():
      if place not in mine.usable and (
          place in other.usable or not other.usable
      ):
        lacking.extend(signatures.values())
    if mine.usable and not lacking:
      continue
    reasons.append(_why_none(lacking, mine, side))
  return _merged(reasons)


def _why_none(
    points: list[tap_scan.Point], held: _Held, side: str
) -> dict[str, Any]:
  """Why none of a side's points has a tensor in a stage."""
  if not points:
    return _reason(NO_LAYER, side)
  if held.uncaptured:
    return metrics.reason(NOT_CAPTURED, held.uncaptured, side)
  groups = collections.defaultdict(list)
  # A point of two places (a layer's output, the next one's input) once.
  unique = {point.id: point for point in points}.values()
  for point in sorted(unique, key=lambda p: p.id):
    if point.signature not in held.ran:
      groups[GRAPH_NOT_RUN].append(point)
    elif point.signature not in held.in_stage:
      groups[NOT_IN_STAGE].append(point)
    elif point.id in held.no_axis:
      groups[NO_TOKEN_AXIS].append(point)
    else:
      groups[NOT_CAPTURED].append(point)
  parts = []
  for code, words in (
      (GRAPH_NOT_RUN, 'did not run in this turn'),
      (NOT_IN_STAGE, 'produced no position of this stage'),
      (NO_TOKEN_AXIS, 'holds them with no one axis of its chunk’s tokens'),
      (NOT_CAPTURED, 'left no tensor of them in the capture'),
  ):
    chosen = groups.get(code)
    if chosen:
      graphs = ', '.join(sorted({p.signature for p in chosen}))
      ids = ', '.join(p.id for p in chosen)
      parts.append((code, f'{ids}: {graphs} {words}'))
  code = parts[0][0]
  return metrics.reason(code, '; '.join(text for _, text in parts) + '.', side)


def batches(
    index: capture_index.CaptureIndex, observed: Observed
) -> list[dict[str, Any]]:
  """The forward passes that hold observed tensors, as the UI's batches.

  One per turn, stage and signature of the reference side's forward passes
  that hold the places both sides hold (`_held`), in turn, stage and step
  order: a stage has batches exactly when it has tensors (`availability`).

  Args:
      index: The chat's capture.
      observed: What the Session observes.

  Returns:
      The batches, each `{batch, turn, phase, index, step, signature,
      graph, comparison_basis}`; [] when nothing can be compared.
  """
  places = layout(observed)
  if places.reason is not None:
    return []
  found = []
  for turn in index.turns:
    of_turn = 0
    for stage_name in STAGE_TOKENS:
      held = _held(index, turn, stage_name, places)
      forwards = {}
      for place in _common(held):
        forwards.update(
            {forward['id']: forward for forward in held['ref'].usable[place]}
        )
      steps = collections.defaultdict(list)
      for _, forward in sorted(forwards.items()):
        steps[forward['signature']].append(forward['step'])
      for signature, held_steps in sorted(
          steps.items(), key=lambda named: named[1][0]
      ):
        found.append({
            'batch': len(found),
            'turn': turn['n'],
            'phase': stage_name,
            'index': of_turn,
            'step': held_steps[0],
            'signature': signature,
            'graph': signature,
            'comparison_basis': (
                f'{len(held_steps)} {signature} step'
                f'{"" if len(held_steps) == 1 else "s"} holding '
                'observed tensors, compared as stored in float64'
            ),
        })
        of_turn += 1
  return found


# Reading a side's value of a place at a position.


@dataclasses.dataclass(frozen=True)
class Source:
  """Where a side's value of a place at a column is.

  Attributes:
      position: The side's position of the column.
      point: The point of the place in the signature that produced it.
      forward: The forward pass that produced it (`produced`).
      row: Which of the forward pass's rows produced it.
      read: The position that row fed, the one before `position`.
      file: The tensor's file name in the turn's directory.
      directory: The side's directory of the turn.
      context_length: The context length the side loaded.
      axis: A prefill chunk's axis of its rows; None for a decode step.
  """

  position: int
  point: tap_scan.Point
  forward: dict[str, Any]
  row: int
  read: int
  file: str
  directory: pathlib.Path
  context_length: int
  axis: int | None


class _Reader:
  """Reads one side's tensors.

  With a budget, whole tensors are kept, least recently used first out, up
  to about that many bytes (the last one read whatever its size); without,
  only the part of a file a value needs is read. Headers are kept. One
  reader reads one directory: files are known by name.
  """

  def __init__(self, budget: int = 0) -> None:
    self._budget = budget
    self._kept: collections.OrderedDict[str, np.ndarray] = (
        collections.OrderedDict()
    )
    self._bytes = 0
    self._headers: dict[str, tensor_file.TensorHeader] = {}

  def header(self, source: Source) -> tensor_file.TensorHeader:
    found = self._headers.get(source.file)
    if found is None:
      found = tensor_file.read_header(source.directory / source.file)
      self._headers[source.file] = found
    return found

  def vector(self, source: Source) -> np.ndarray:
    """A side's value at a position: its tensor's part for that position.

    Args:
        source: Where the value is.

    Returns:
        The part of its tensor for the position.

    Raises:
        _MissingError: The header gives the tensor another shape than the
          model does.
        OSError, ValueError: The file cannot be read.
    """
    point = source.point
    expected = _expected(point, source.context_length, source.forward['rows'])
    shape = tuple(self.header(source).shape)
    if shape != expected:
      raise _MissingError(
          SHAPE_UNEXPECTED,
          f'{point.id} at step {source.forward["step"]} is '
          f'{list(shape)}; the model makes it {list(expected)}.',
      )
    part = [slice(None)] * len(expected)
    if source.axis is not None:
      part[source.axis] = slice(source.row, source.row + 1)
    for axis in point.context_axes:
      part[axis] = slice(0, source.read + 1)
    if self._budget:
      values = self._tensor(source)[tuple(part)]
    else:
      values = tensor_file.read_slice(
          source.directory / source.file, tuple(part)
      )
    # A decode step's tensor and a prefill chunk's row of the same place
    # differ only by axes of one: neither keeps them.
    return np.atleast_1d(np.squeeze(values))

  def _tensor(self, source: Source) -> np.ndarray:
    values = self._kept.get(source.file)
    if values is not None:
      self._kept.move_to_end(source.file)
      return values
    values = tensor_file.read_tensor(source.directory / source.file)
    self._kept[source.file] = values
    self._bytes += values.nbytes
    while len(self._kept) > 1 and self._bytes > self._budget:
      _, dropped = self._kept.popitem(last=False)
      self._bytes -= dropped.nbytes
    return values


class _MissingError(Exception):
  """A side's tensor at a position cannot be read as a value; why."""

  def __init__(self, code: str, text: str) -> None:
    super().__init__(text)
    self.code = code
    self.text = text


@dataclasses.dataclass
class Cell:
  """One place at one column: both sides' values, compared.

  Attributes:
      column: The column.
      layer: The place's layer.
      anchor: The place's anchor.
      sources: Per side, where its value is; a side without is absent.
      shape: The compared values' shape, where both sides have one.
      metrics: Per metric of METRICS, its value or None.
      status: metrics.OK, or the code of why there is no comparison.
      reasons: Why some or all of it is missing.
      dtypes: Per side whose file was read, the dtype it stores.
  """

  column: int
  layer: int
  anchor: str
  sources: dict[str, Source]
  shape: list[int] | None
  metrics: dict[str, float | None]
  status: str
  reasons: list[dict[str, Any]]
  dtypes: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Stage:
  """The intermediate tensors of one stage of one turn, both sides.

  Attributes:
      turn: The turn.
      start: The first column.
      end: The column after the last.
      places: The places observed.
      cells: Each place either side holds a tensor of (`_held`) at every column,
        by column, then layer, then anchor.
      reasons: Why the stage has no tensors (availability), or [].
      unheld: Per id of a point of a place no side holds, why.
      columns: Per column, the forward pass each side reads it from
        (_columns()).
      omitted: Why columns or layers either side has are not the stage's
        (_produced_columns(), _one_sided_layers()), or [].
  """

  turn: int
  start: int
  end: int
  places: Layout
  cells: list[Cell]
  reasons: list[dict[str, Any]]
  unheld: dict[str, list[dict[str, Any]]] = dataclasses.field(
      default_factory=dict
  )
  columns: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  omitted: list[dict[str, Any]] = dataclasses.field(default_factory=list)

  def cell(self, column: int, layer: int, anchor: str) -> Cell:
    """The cell of a place at a column.

    Args:
        column: The column.
        layer: The place's layer.
        anchor: The place's anchor.

    Returns:
        The cell.

    Raises:
        ValueError: The stage has no such cell.
    """
    found = _cells_by_key(self).get((column, layer, anchor))
    if found is None:
      raise ValueError(_no_cell(column, layer, anchor))
    return found


def _cells_by_key(found: Stage) -> dict[tuple[int, int, str], Cell]:
  return {(c.column, c.layer, c.anchor): c for c in found.cells}


def _no_cell(column: int, layer: int, anchor: str) -> str:
  return f'Column {column} has no tensor of layer {layer}, {anchor}.'


@dataclasses.dataclass(frozen=True)
class _Plan:
  """Where every cell of a stage is, from the capture index alone."""

  turn: int
  start: int
  end: int
  places: Layout
  reasons: list[dict[str, Any]]
  order: list[tuple[int, str]]
  unheld: dict[str, list[dict[str, Any]]]
  sides: dict[str, dict[str, Any]]
  alignment: kv_analysis.Alignment | None
  omitted: list[dict[str, Any]] = dataclasses.field(default_factory=list)

  def positions(self, column: int) -> dict[str, int | None]:
    if self.alignment is None:
      return dict.fromkeys(capture_index.SIDES, column)
    return self.alignment.positions(column)

  def locate(self, column: int, place: tuple[int, str]) -> Cell:
    """A cell's sources, or why a side has none; not compared yet."""
    sources, why = {}, []
    at = self.positions(column)
    for side in capture_index.SIDES:
      found = _locate(self.sides[side], self.places, place, at[side])
      if isinstance(found, Source):
        sources[side] = found
      else:
        why.append({**found, 'side': side})
    return Cell(
        column=column,
        layer=place[0],
        anchor=place[1],
        sources=sources,
        shape=None,
        metrics=dict.fromkeys(METRICS),
        status='unavailable',
        reasons=_merged(why),
    )


def _plan(
    index: capture_index.CaptureIndex,
    n: int,
    stage_name: str,
    observed: Observed,
    alignment: kv_analysis.Alignment | None,
) -> _Plan:
  """Returns which tensors of each side to read for one stage of turn `n`."""
  turn = index.turn(n)
  entries = turn['sides']
  if alignment is None:
    positions = _stage_positions(entries['ref'], stage_name)
    start, end = positions[0], positions[-1] + 1
  else:
    start, end = alignment.start, alignment.end()
  places = layout(observed)
  sides = {}
  for side in capture_index.SIDES:
    entry = entries[side]
    sides[side] = {
        'name': side,
        'produced': produced(entry),
        'mine': set(_stage_positions(entry, stage_name)),
        'directory': index.root / entry['dir'],
        'context_length': index.sides[side]['load']['context_length'],
        'first_input': entry['input'][0]['position'],
        'uncaptured': _uncaptured(entry),
        'asked': asked_modules(index.sides[side]['load']),
    }
  start, end, omitted = _produced_columns(start, end, sides, alignment)
  if places.reason is not None:
    return _Plan(
        n,
        start,
        end,
        places,
        [places.reason],
        [],
        {},
        sides,
        alignment,
        omitted,
    )
  held = _held(index, turn, stage_name, places)
  reasons = _unavailable(places, held)
  listed = set(held['ref'].captured) | set(held['target'].captured)
  one_sided = _one_sided_layers(index, stage_name, places, held)
  if one_sided:
    omitted = [*omitted, one_sided[1]]
    listed = {place for place in listed if place[0] not in one_sided[0]}
  order = sorted(listed, key=lambda p: (p[0], places.anchors.index(p[1])))
  unheld = {}
  for point in places.place_points():
    if any(place in listed for place in places.places_of(point)):
      continue
    unheld[point.id] = _merged(
        [_why_none([point], held[side], side) for side in capture_index.SIDES]
    )
  return _Plan(
      n,
      start,
      end,
      places,
      reasons,
      order,
      unheld,
      sides,
      alignment,
      omitted,
  )


def columns(
    index: capture_index.CaptureIndex,
    n: int,
    stage_name: str,
    observed: Observed,
    alignment: kv_analysis.Alignment | None,
) -> dict[str, Any]:
  """The columns a stage's intermediate tensors have, from the index alone.

  Those of `stage` and position-analysis, which read no tensor to say so.

  Args:
      index: The chat's capture.
      n: A turn the index read.
      stage_name: 'prefill' or 'decode'.
      observed: What the Session observes.
      alignment: As `stage`'s.

  Returns:
      `{start, end, omitted}`: the columns, end exclusive, and why columns
      or layers a side has are not the stage's (Stage.omitted).
  """
  plan = _plan(index, n, stage_name, observed, alignment)
  return {'start': plan.start, 'end': plan.end, 'omitted': plan.omitted}


def _producer(
    info: dict[str, Any], position: int | None
) -> capture_index.Feed | None:
  """The forward pass a side reads a column at its position from, or None.

  None where the side has no token at the column, or no forward pass of
  the turn produced its token (produced()).

  Args:
      info: The side's entry of the stage's plan (`_Plan.sides`).
      position: The side's position of the column, or None.

  Returns:
      The forward pass and its row, or None.
  """
  if position is None or position not in info['mine']:
    return None
  return info['produced'].get(position)


def _produced_columns(
    start: int,
    end: int,
    sides: dict[str, dict[str, Any]],
    alignment: kv_analysis.Alignment | None,
) -> tuple[int, int, list[dict[str, Any]]]:
  """A stage's columns that some forward pass produced, and why not more.

  A column no forward pass produced on either side (the turn's first
  input token's, in a Prefill) holds no tensor on any side: it is not a
  column of the stage. Such columns are only ever the stage's first.

  Args:
      start: The stage's first column.
      end: The column after the stage's last.
      sides: Per side, its entry of the stage's plan (`_Plan.sides`).
      alignment: As `stage`'s.

  Returns:
      (start, end) of the columns kept, and the reasons (POSITION_NOT_
      PRODUCED) of the columns left out, [] when none is.
  """

  def at(column: int) -> dict[str, int | None]:
    if alignment is None:
      return dict.fromkeys(capture_index.SIDES, column)
    return alignment.positions(column)

  def made(column: int) -> bool:
    positions = at(column)
    return any(
        _producer(sides[side], positions[side]) is not None
        for side in capture_index.SIDES
    )

  first = start
  while first < end and not made(first):
    first += 1
  if first == start or first == end:
    return start, end, []
  left = []
  for column in range(start, first):
    left += [
        position for position in at(column).values() if position is not None
    ]
  shown = sorted(set(left)) or [start]
  named = ', '.join(str(p) for p in shown)
  # One position on each side, when the sides' turns start apart.
  said = (
      f'Position {named} is not shown: it holds'
      if len(shown) == 1
      else f'Positions {named} are not shown: each holds'
  )
  return (
      first,
      end,
      [
          metrics.reason(
              POSITION_NOT_PRODUCED,
              f'{said} the turn’s first input token, which no forward pass '
              'of this turn produced.',
          )
      ],
  )


def _one_sided_layers(
    index: capture_index.CaptureIndex,
    stage_name: str,
    places: Layout,
    held: dict[str, _Held],
) -> tuple[set[int], dict[str, Any]] | None:
  """The layers only one side computed in a stage, and why they are left out.

  A layer one side holds tensors of in the stage and the other side's
  graphs of the stage have no point of at all (a runtime whose graph of
  the stage does not run that layer, as LiteRT-LM's prefill of Gemma 4
  E2B, which stops before the layers that share their KV cache) has
  nothing to compare: it is not a layer of the stage. A layer the other
  side's graph has a point of but holds no tensor of (one not captured,
  say) is kept, its cells saying why; so is a layer of a pair made by
  hand, with its own reasons.

  Args:
      index: The chat's capture.
      stage_name: 'prefill' or 'decode'.
      places: The Session's places.
      held: What each side holds of the stage's places (`_held`).

  Returns:
      (the layers, the reason), or None when there are none, or a side
      holds no layer.
  """
  layers = {
      side: {
          layer
          for layer, anchor in held[side].captured
          if not is_manual(anchor)
      }
      for side in capture_index.SIDES
  }
  manual = set()
  for side in capture_index.SIDES:
    manual.update(
        layer for layer, anchor in held[side].captured if is_manual(anchor)
    )
  if not all(layers.values()):
    return None
  lone = (layers['ref'] ^ layers['target']) - manual

  def in_graphs(side: str, layer: int) -> bool:
    # Whether a graph of the side that produced the stage has a point of
    # the layer.
    return any(
        set(signatures) & held[side].in_stage
        for (at, _), signatures in places.points[side].items()
        if at == layer
    )

  lone = {
      layer
      for layer in lone
      if not in_graphs(
          'target' if layer in layers['ref'] else 'ref',
          layer,
      )
  }
  if not lone:
    return None
  holder = 'ref' if lone <= layers['ref'] else 'target'
  if not lone <= layers[holder]:
    holder = None
  named = _layer_span(sorted(lone))
  stage_label = STAGE_LABELS[stage_name]
  if holder is None:
    text = (
        f'Layers {named} are not shown: in this {stage_label} only one '
        'side holds tensors of each.'
    )
  else:
    other = 'target' if holder == 'ref' else 'ref'
    signatures = ', '.join(sorted(held[other].in_stage)) or 'none'
    text = (
        f'{"Layer" if len(lone) == 1 else "Layers"} {named} '
        f'{"is" if len(lone) == 1 else "are"} not shown: the '
        f'{_side_label(index, other)} has no such layer in its '
        f'{stage_label} graph ({signatures}); only the '
        f'{_side_label(index, holder)} computed '
        f'{"it" if len(lone) == 1 else "them"}, so there is nothing to '
        'compare.'
    )
  return lone, metrics.reason(LAYERS_ONE_SIDE, text)


def _layer_span(layers: list[int]) -> str:
  """Layers as runs: `15–34`, `3, 5–7`."""
  runs = []
  for layer in layers:
    if runs and layer == runs[-1][1] + 1:
      runs[-1][1] = layer
    else:
      runs.append([layer, layer])
  return ', '.join(
      str(low) if low == high else f'{low}–{high}' for low, high in runs
  )


def _side_label(index: capture_index.CaptureIndex, side: str) -> str:
  """`Reference (PyTorch)`, as the UI calls a side and its runtime."""
  runtime = (index.sides[side].get('runtime') or {}).get('name')
  name = SIDE_LABELS[side]
  return f'{name} ({runtime})' if runtime else name


def _columns(plan: _Plan) -> list[dict[str, Any]]:
  """Per column of a stage, the forward pass each side reads it from.

  The one `_locate` reads a column's cells from (produced()): of each
  side, `{signature, step}`, or None where the side has no token at the
  column or no forward pass of the turn produced it.

  Args:
      plan: The stage's plan.

  Returns:
      Per column, `{position, ref, target}`.
  """
  found = []
  for column in range(plan.start, plan.end):
    at = plan.positions(column)
    entry = {'position': column}
    for side in capture_index.SIDES:
      producer = _producer(plan.sides[side], at[side])
      entry[side] = None
      if producer is not None:
        entry[side] = {
            'signature': producer.forward['signature'],
            'step': producer.forward['step'],
        }
    found.append(entry)
  return found


def _locate(
    side: dict[str, Any],
    places: Layout,
    place: tuple[int, str],
    position: int | None,
) -> Source | dict[str, Any]:
  """Where a side's value of a place at its position is, or why it has none.

  Args:
      side: The side's entry of the stage's plan (`_Plan.sides`).
      places: The Session's places.
      place: The (layer, anchor).
      position: The side's position of the column, or None.

  Returns:
      The Source, or a reason (metrics.reason, without its side).
  """
  if position is None or position not in side['mine']:
    return _reason(POSITION_MISSING)
  producer = side['produced'].get(position)
  if producer is None:
    if position == side['first_input']:
      return metrics.reason(
          POSITION_NOT_PRODUCED,
          f'Position {position} holds the turn’s first input token; no '
          'forward pass of this turn produced it.',
      )
    return _reason(POSITION_NOT_PRODUCED)
  forward = producer.forward
  signature = forward['signature']
  point = places.points[side['name']].get(place, {}).get(signature)
  if point is None:
    return metrics.reason(
        OTHER_GRAPH,
        f'{signature} step {forward["step"]} produced position '
        f'{position}; it holds no tensor of this sync point.',
    )
  key = places.keys[point.id]
  why = _unusable(forward, point, key, side['context_length'])
  if why == NOT_CAPTURED:
    return metrics.reason(
        NOT_CAPTURED,
        side['uncaptured']
        or not_asked(key, side['asked'])
        or (
            f'The capture holds no {point.id} at {signature} step '
            f'{forward["step"]}.'
        ),
    )
  if why == NO_TOKEN_AXIS:
    shape = list(_expected(point, side['context_length'], forward['rows']))
    return metrics.reason(
        NO_TOKEN_AXIS,
        f'{point.id} is {shape}; {signature} feeds {forward["rows"]} '
        'rows a step.',
    )
  return Source(
      position=position,
      point=point,
      forward=forward,
      row=producer.row,
      read=producer.position,
      file=forward['tensors'][key],
      directory=side['directory'],
      context_length=side['context_length'],
      axis=(
          _token_axis(point, forward['rows'], side['context_length'])
          if forward['phase'] == 'prefill'
          else None
      ),
  )


def stage(
    index: capture_index.CaptureIndex,
    n: int,
    stage_name: str,
    observed: Observed,
    alignment: kv_analysis.Alignment | None,
) -> Stage:
  """Reads and compares every place of a stage at every column.

  Args:
      index: The chat's capture.
      n: A turn the index read.
      stage_name: 'prefill' or 'decode'.
      observed: What the Session observes.
      alignment: The stage's columns and each side's position of each; None: a
        column is the position on both sides, the ref side's positions of the
        stage the columns.

  Returns:
      The stage, its cells compared.
  """
  return stages(index, n, stage_name, observed, [alignment])[0]


def stages(
    index: capture_index.CaptureIndex,
    n: int,
    stage_name: str,
    observed: Observed,
    alignments: list[kv_analysis.Alignment | None],
) -> list[Stage]:
  """Reads and compares every place of a stage under each alignment at once.

  Each stage is the one `stage` returns of its alignment. A cell is
  compared once for every alignment whose cell reads the same sources,
  and each file is read once for all of them.

  Args:
      index: The chat's capture.
      n: A turn the index read.
      stage_name: 'prefill' or 'decode'.
      observed: What the Session observes.
      alignments: The alignments, each as `stage`'s.

  Returns:
      The stage under each alignment, in their order, its cells compared.
  """
  found, same = [], {}
  for alignment in alignments:
    plan = _plan(index, n, stage_name, observed, alignment)
    one = Stage(
        n,
        plan.start,
        plan.end,
        plan.places,
        [],
        plan.reasons,
        plan.unheld,
        _columns(plan),
        plan.omitted,
    )
    found.append(one)
    if plan.places.reason is not None:
      continue
    one.cells = [
        plan.locate(column, place)
        for column, place in itertools.product(
            range(plan.start, plan.end), plan.order
        )
    ]
    rank = {place: i for i, place in enumerate(plan.order)}
    for located_cell in one.cells:
      if located_cell.sources:
        same.setdefault(_read_as(located_cell), []).append(
            (located_cell, rank[located_cell.layer, located_cell.anchor])
        )
  if not same:
    return found
  readers = {side: _Reader(READ_BYTES) for side in capture_index.SIDES}
  cross = found[0].places.cross

  def reading_order(
      cells: list[tuple[Cell, int]],
  ) -> tuple[tuple[int, ...], int, int]:
    # Each file's cells one after another: it is read once.
    located, rank = cells[0]
    forwards = tuple(
        located.sources[side].forward['id'] if side in located.sources else -1
        for side in capture_index.SIDES
    )
    return forwards, rank, located.column

  for cells in sorted(same.values(), key=reading_order):
    (first, _), *others = cells
    located = len(first.reasons)
    _compare(first, readers, cross)
    for other, _ in others:
      _compared_as(first, located, other)
  return found


def _read_as(located_cell: Cell) -> tuple[tuple[str, str, str, int, int], ...]:
  """What a located cell reads: two cells of one key compare the same."""
  return tuple(
      (side, source.file, source.point.id, source.row, source.read)
      for side, source in located_cell.sources.items()
  )


def _compared_as(first: Cell, located: int, other: Cell) -> None:
  """Gives `other`, located as `first`, what comparing `first` found.

  Args:
      first: A cell compared.
      located: How many reasons `first` had before it was compared.
      other: A cell of the same sources (_read_as), not compared yet. Its own
        reasons are why a side has no source; comparing, which read the same
        values, adds what it added to `first`'s.
  """
  other.shape = None if first.shape is None else list(first.shape)
  other.metrics = dict(first.metrics)
  other.status = first.status
  other.dtypes = dict(first.dtypes)
  other.reasons.extend(dict(r) for r in first.reasons[located:])


def cell(
    index: capture_index.CaptureIndex,
    n: int,
    stage_name: str,
    observed: Observed,
    alignment: kv_analysis.Alignment | None,
    column: int,
    layer: int,
    anchor: str,
) -> Cell:
  """Reads and compares one place of a stage at one column, alone.

  Args:
      index: The chat's capture.
      n: A turn the index read.
      stage_name: 'prefill' or 'decode'.
      observed: What the Session observes.
      alignment: As `stage`'s.
      column: The cell's column.
      layer: The cell's layer.
      anchor: The cell's anchor.

  Returns:
      The cell, compared.

  Raises:
      ValueError: The stage has no such cell.
  """
  plan = _plan(index, n, stage_name, observed, alignment)
  if not plan.start <= column < plan.end or (layer, anchor) not in (plan.order):
    raise ValueError(_no_cell(column, layer, anchor))
  found = plan.locate(column, (layer, anchor))
  return _compare(
      found,
      {side: _Reader() for side in capture_index.SIDES},
      plan.places.cross,
  )


def _compare(
    located_cell: Cell, readers: dict[str, _Reader], cross: bool = False
) -> Cell:
  """Reads a located cell's values and compares them, in place.

  Of two runtimes (`cross`), whatever dtype each side stores.

  Args:
      located_cell: The cell, its sources located.
      readers: Per side, the reader of its tensors.
      cross: Whether the sides are of two runtimes.

  Returns:
      `located_cell`, compared.
  """
  values = {}
  for side, source in located_cell.sources.items():
    try:
      located_cell.dtypes[side] = readers[side].header(source).dtype
      values[side] = readers[side].vector(source)
    except _MissingError as missing:
      located_cell.reasons.append(
          metrics.reason(missing.code, missing.text, side)
      )
    except (OSError, ValueError) as error:
      located_cell.reasons.append(
          metrics.reason(metrics.UNREADABLE, str(error), side)
      )
  if len(values) != 2:
    return located_cell
  ref, target = values['ref'], values['target']
  if not cross and located_cell.dtypes['ref'] != located_cell.dtypes['target']:
    located_cell.status = DTYPE_MISMATCH
    located_cell.reasons.append(
        metrics.reason(
            DTYPE_MISMATCH,
            f'Ref stores {located_cell.dtypes["ref"]} and Target stores '
            f'{located_cell.dtypes["target"]}.',
        )
    )
    return located_cell
  if (
      cross
      and ref.shape != target.shape
      and ref.size == target.size
      and min(ref.ndim, target.ndim) == 1
  ):
    # A semantic operator's tensor flat on one side, by heads on the
    # other: the same elements in the same order (counterparts).
    ref, target = ref.reshape(-1), target.reshape(-1)
  if ref.shape != target.shape:
    located_cell.status = metrics.SHAPE_MISMATCH
    located_cell.reasons.append(
        metrics.reason(
            metrics.SHAPE_MISMATCH,
            f'Ref is {list(ref.shape)} and Target {list(target.shape)}.',
        )
    )
    return located_cell
  located_cell.shape = list(ref.shape)
  compared = metrics.compare(ref, target, keys=METRICS)
  located_cell.status = metrics.OK
  for key in METRICS:
    value = float(compared.values[key])
    code = str(compared.status[key])
    located_cell.metrics[key] = value if code == metrics.OK else None
    if code != metrics.OK:
      reason = metrics.reason(code, metrics.REASONS[code])
      if reason not in located_cell.reasons:
        located_cell.reasons.append(reason)
  return located_cell


# The answers.


def analysis(found: Stage) -> dict[str, Any]:
  """position-analysis: a stage's places and every cell's metrics.

  Args:
      found: The stage, its cells compared.

  Returns:
      `{label, start, end, layers, anchors, points, pairing, rows, columns,
      not_observed, unpaired, provenance, reasons}`: the layers and anchors of
      the cells that hold a tensor (of every place, when none does), anchors
      as `{id, label, default}`, `default` whether it is one of an
      architecture's (Layout.defaults), not an operator chosen inside the
      layers; `points` each point that has a place, with its graph,
      name, shape and dtype, its node in the model's graph (`graph_node`:
      `{graph, node, output}`, the signature's graph, the operator's node and
      the output's id there, model_graph's), how many cells hold it and, when
      none does, why (`reasons`); `rows` the cells, each `{position, layer,
      anchor, status, reasons, shape, metrics, points}`, a metric as `{value,
      reasons}`, a cell without a value saying why, `points` per side that
      has
      one the id of the point its value is of; `columns` per column the
      forward pass each side reads it from, `{position, ref, target}`, a
      side's `{signature, step}` or None (_columns()); `not_observed` why a
      point of the model that `points` does not list has no cell
      (POINT_NOT_OBSERVED); `unpaired` why each point chosen of a LiteRT-LM
      side that is placed nowhere is not (NO_COUNTERPART); `omitted` why
      columns or layers a side has are not the stage's (Stage.omitted).
  """
  places = found.places
  held = collections.Counter()
  for located in found.cells:
    held.update({source.point.id for source in located.sources.values()})
  holding = [located for located in found.cells if located.sources]
  layers = _shown_layers(found)
  anchors = [
      anchor
      for anchor in places.anchors
      if not holding or any(located.anchor == anchor for located in holding)
  ]
  return {
      'label': LABEL,
      'start': found.start,
      'end': found.end,
      'layers': layers,
      'anchors': [
          {
              'id': anchor,
              'label': places.labels[anchor],
              'default': anchor in places.defaults,
          }
          for anchor in anchors
      ],
      'points': [
          {
              'id': point.id,
              'layer': point.layer,
              'anchor': places.places_of(point)[0][1],
              'places': [
                  {'layer': layer, 'anchor': anchor}
                  for layer, anchor in places.places_of(point)
              ],
              'signature': point.signature,
              'tensor_name': point.tensor_name,
              'shape': list(point.shape),
              'context_axes': list(point.context_axes),
              'dtype': point.dtype,
              'graph_node': _graph_node(point, places),
              'cells': held[point.id],
              'reasons': found.unheld.get(point.id, []),
          }
          for point in places.place_points()
      ],
      'pairing': _pairing(places),
      'rows': [_row(located) for located in found.cells],
      'columns': found.columns,
      'not_observed': _reason(POINT_NOT_OBSERVED),
      'unpaired': list(places.notes),
      'provenance': PROVENANCE,
      'reasons': found.reasons,
      'omitted': found.omitted,
  }


def _shown_layers(found: Stage) -> list[int]:
  """The layers of the stage's cells holding a tensor; else every place's."""
  holding = {cell.layer for cell in found.cells if cell.sources}
  return sorted(holding) or list(found.places.layers)


def _pairing(places: Layout) -> dict[str, Any]:
  """Whether the two sides' points pair: `{status, reasons}`."""
  if places.reason is None:
    return {'status': 'paired', 'reasons': []}
  status = 'not_paired' if places.reason['code'] == NOT_PAIRED else 'none'
  return {'status': status, 'reasons': [places.reason]}


def _graph_node(point: tap_scan.Point, places: Layout) -> dict[str, str] | None:
  """Returns where `point` is in its graph, or None if it has no graph."""
  if point.id in places.graphless:
    return None
  return {
      'graph': point.signature,
      'node': str(point.op),
      'output': str(point.output),
  }


def _row(located: Cell) -> dict[str, Any]:
  return {
      'position': located.column,
      'layer': located.layer,
      'anchor': located.anchor,
      'status': located.status,
      'reasons': located.reasons,
      'shape': located.shape,
      'metrics': {
          key: (
              metrics.metric(value)
              if value is not None
              else metrics.missing(located.reasons)
          )
          for key, value in located.metrics.items()
      },
      'points': {
          side: source.point.id for side, source in located.sources.items()
      },
  }


def tensor_range(
    found: Stage,
    start: int,
    end: int,
    bins: int,
    anchors: list[str] | None,
    reduction: str,
    layers: list[int] | None,
) -> dict[str, Any]:
  """position-range: each layer's metrics over columns, binned.

  At each column, a layer's value of a metric is that of the one anchor
  chosen, or the `reduction` (min, max or mean) of the finite values of
  the anchors chosen; the bins are the statistics of those values
  (analysis/statistics), as kv-range's. `totals` are of the whole range,
  one bin, per layer.

  Args:
      found: The stage.
      start: The first column; within the stage's.
      end: The column after the last; more than `start`.
      bins: How many bins, 1..MAX_BINS.
      anchors: The anchors chosen, of the stage's places; None for every one.
      reduction: How several anchors' values combine.
      layers: The layers wanted, of the stage's places; None for those
        position-analysis lists.

  Returns:
      `{start, end, anchors, reduction, layers, bins, rows, totals,
      metadata}`: rows and totals as kv-range's rows, `{layer, kind,
      status, reasons, bins}`, each bin `{start, end, metrics}`, a metric
      a statistic of its values (statistics) with, for a min or max
      reduction, the anchor of the extremum at its position, and why the
      columns of the bin that have no value have none (`reasons`, at most
      MAX_REASONS); `metadata` each side's stored dtype.
  """
  places = found.places
  chosen = list(places.anchors) if anchors is None else list(anchors)
  wanted = _shown_layers(found) if layers is None else list(layers)
  edges = statistics.edges(start, end, bins)
  bounds = [
      {'start': int(low), 'end': int(high)}
      for low, high in zip(edges[:-1], edges[1:])
  ]
  whole = np.array([start, end])
  by_layer = collections.defaultdict(lambda: collections.defaultdict(dict))
  for located in found.cells:
    if start <= located.column < end and located.anchor in chosen:
      by_layer[located.layer][located.column][located.anchor] = located
  rows, totals = [], []
  for layer in wanted:
    layer_columns = by_layer.get(layer, {})
    row = {'layer': layer, 'kind': LABEL, 'status': 'ok', 'reasons': []}
    if not layer_columns:
      missing = found.reasons or _layer_reasons(found, layer, chosen)
      row.update(status='unavailable', reasons=missing)
    positions = np.array(sorted(layer_columns), dtype=np.int64)
    stats, total, extremes, why = {}, {}, {}, {}
    for key in METRICS:
      values, which = _reduced(layer_columns, positions, key, chosen, reduction)
      stats[key] = statistics.binned(values, positions, edges)
      total[key] = statistics.binned(values, positions, whole)
      extremes[key] = dict(zip(positions.tolist(), which))
      why[key] = _missing_by_column(layer_columns, positions, values, key)
    if layer_columns and all(
        s['valid_count'] == 0
        for s in itertools.chain.from_iterable(stats.values())
    ):
      row['status'] = 'unavailable'
      row['reasons'] = _cell_reasons(layer_columns)
    rows.append({
        **row,
        'bins': _bins(bounds, stats, extremes, reduction, why),
    })
    totals.append({
        **row,
        'bins': _bins(
            [{'start': start, 'end': end}],
            total,
            extremes,
            reduction,
            why,
        ),
    })
  return {
      'start': start,
      'end': end,
      'anchors': chosen,
      'reduction': reduction,
      'layers': wanted,
      'bins': bounds,
      'rows': rows,
      'totals': totals,
      'metadata': _range_metadata(found, start, end, chosen, wanted),
  }


def _reduced(layer_columns, positions, key, chosen, reduction):
  """One value per column of a metric, and the anchor it is of."""
  values, which = [], []
  for column in positions.tolist():
    found = [
        (cell.metrics[key], anchor)
        for anchor, cell in layer_columns[column].items()
        if cell.metrics[key] is not None and math.isfinite(cell.metrics[key])
    ]
    found.sort(key=lambda item: chosen.index(item[1]))
    if not found:
      values.append(math.nan)
      which.append(None)
    elif reduction == 'mean' and len(found) > 1:
      values.append(float(np.mean([value for value, _ in found])))
      which.append(None)
    else:
      pick = min if reduction == 'min' else max
      value, anchor = (
          found[0] if len(found) == 1 else pick(found, key=lambda item: item[0])
      )
      values.append(value)
      which.append(anchor)
  return np.array(values, dtype=np.float64), which


def _missing_by_column(
    layer_columns, positions, values, key
) -> dict[int, list[dict[str, Any]]]:
  """Why each column without a value of a metric has none: its cells'."""
  found = {}
  for column, value in zip(positions.tolist(), values.tolist()):
    if math.isfinite(value):
      continue
    reasons = []
    for located in layer_columns[column].values():
      if located.metrics[key] is not None:
        continue
      for reason in located.reasons:
        if reason not in reasons:
          reasons.append(reason)
    found[column] = reasons
  return found


def _bins(bounds, stats, extremes, reduction, why):
  """Returns each bin of `bounds` with its metrics' statistics."""
  result = []
  for i, bound in enumerate(bounds):
    bin_metrics = {}
    for key in METRICS:
      summary = dict(stats[key][i])
      if reduction in ('min', 'max'):
        for end in ('min', 'max'):
          at = summary[f'{end}_position']
          summary[f'{end}_anchor'] = (
              None if at is None else extremes[key].get(at)
          )
      reasons = []
      for column, given in why[key].items():
        if bound['start'] <= column < bound['end']:
          for reason in given:
            if reason not in reasons:
              reasons.append(reason)
      summary['reasons'] = reasons[:MAX_REASONS]
      bin_metrics[key] = summary
    result.append({**bound, 'metrics': bin_metrics})
  return result


def _layer_reasons(
    found: Stage, layer: int, chosen: list[str]
) -> list[dict[str, Any]]:
  """Why a layer has no cell of the anchors chosen in a stage."""
  has = {cell.anchor for cell in found.cells if cell.layer == layer}
  if has and not has & set(chosen):
    return [_reason(PLACE_MISSING)]
  return [
      metrics.reason(
          NOT_IN_STAGE,
          f'No graph that produced this stage holds layer {layer} of the '
          'sync points chosen.',
      )
  ]


def _cell_reasons(layer_columns) -> list[dict[str, Any]]:
  reasons = []
  for anchors in layer_columns.values():
    for located in anchors.values():
      for reason in located.reasons:
        if reason not in reasons:
          reasons.append(reason)
  return reasons[:MAX_REASONS]


def _range_metadata(found, start, end, chosen, layers) -> dict[str, Any]:
  """What position-range compared: each side's input, and the output.

  A side's dtype is the one its files store, as position-tensor's.

  Args:
      found: The stage.
      start: The first column.
      end: The column after the last.
      chosen: The anchors chosen.
      layers: The layers chosen.

  Returns:
      Per side and `output`, `{dimensions, shape, dtype}`.
  """
  result = {}
  for side in capture_index.SIDES:
    shapes = collections.Counter()
    dtypes = set()
    for located in found.cells:
      if (
          side in located.dtypes
          and start <= located.column < end
          and located.anchor in chosen
          and located.layer in layers
      ):
        dtypes.add(located.dtypes[side])
        if located.shape is not None:
          shapes[math.prod(located.shape)] += 1
    sizes = sorted(shapes)
    result[side] = {
        'dimensions': ['Token position', 'Layer', 'Element'],
        'shape': [
            end - start,
            len(layers),
            sizes[0] if len(sizes) == 1 else (sizes or None),
        ],
        'dtype': (
            sorted(dtypes)[0] if len(dtypes) == 1 else (sorted(dtypes) or None)
        ),
    }
  result['output'] = {
      'dimensions': ['Token position', 'Layer'],
      'shape': [end - start, len(layers)],
      'dtype': metrics.CALCULATION_DTYPE,
  }
  return result


def tensor(found: Cell) -> dict[str, Any]:
  """position-tensor: one place at one column, both sides, element by element.

  Args:
      found: The cell (`cell`, or one of a `Stage`).

  Returns:
      `{position, layer, anchor, reference_position, target_position,
      reference, target, comparable, channels, statistics, metrics, status,
      reasons, provenance}`: each side's source (its point, graph, the
      step that produced the column and its row, the position that row
      `reads`, stored dtype and shapes) or None; `channels` each element's
      `{channel, ref, target, delta, abs_delta, status}`, a side's value
      null past its own elements, delta being Target - Reference where
      both have one and the cell compares; `statistics` of each of those,
      by element (statistics.summary).
  """
  reasons = list(found.reasons)
  values, described = {}, {}
  for side in capture_index.SIDES:
    source = found.sources.get(side)
    if source is None:
      described[side] = None
      continue
    reader = _Reader()
    try:
      vector = reader.vector(source)
      dtype = reader.header(source).dtype
    except (_MissingError, OSError, ValueError):
      # The cell's reasons say why already.
      described[side] = None
      continue
    values[side] = vector
    described[side] = {
        'point': source.point.id,
        'signature': source.point.signature,
        'step': source.forward['step'],
        'row': source.row,
        'reads': source.read,
        'tensor_name': source.point.tensor_name,
        'shape': list(vector.shape),
        'source_shape': [
            source.forward['rows'] if extent == ROWS else extent
            for extent in source.point.shape
        ],
        'dtype': dtype,
        'comparison_dtype': metrics.CALCULATION_DTYPE,
    }
  size = max((v.size for v in values.values()), default=0)
  result = {
      'position': found.column,
      'layer': found.layer,
      'anchor': found.anchor,
      'reference_position': (
          found.sources['ref'].position if 'ref' in found.sources else None
      ),
      'target_position': (
          found.sources['target'].position
          if 'target' in found.sources
          else None
      ),
      'reference': described['ref'],
      'target': described['target'],
      'comparable': found.status == metrics.OK,
      'channels': [],
      'statistics': {
          key: statistics.empty()
          for key in ('ref', 'target', 'delta', 'abs_delta')
      },
      'metrics': _row(found)['metrics'],
      'status': found.status,
      'reasons': reasons,
      'provenance': PROVENANCE,
  }
  if size > MAX_ELEMENTS:
    result['status'] = TOO_LARGE
    result['reasons'] = reasons + [_reason(TOO_LARGE)]
    return result
  flat = {side: v.reshape(-1).astype(np.float64) for side, v in values.items()}
  comparable = result['comparable']
  padded = {}
  for side in capture_index.SIDES:
    given = flat.get(side, np.empty(0))
    padded[side] = np.concatenate([given, np.full(size - given.size, np.nan)])
  ref, target = padded['ref'], padded['target']
  with np.errstate(invalid='ignore', over='ignore'):
    delta = target - ref if comparable else np.full(size, np.nan)
  finite_delta = np.isfinite(delta)
  lengths = {side: flat[side].size if side in flat else 0 for side in padded}
  channels = []
  for i in range(size):
    d = float(delta[i]) if finite_delta[i] else None
    channels.append({
        'channel': i,
        'ref': _number(ref[i]) if i < lengths['ref'] else None,
        'target': _number(target[i]) if i < lengths['target'] else None,
        'delta': d,
        'abs_delta': None if d is None else abs(d),
        'status': 'available' if d is not None else 'unavailable',
    })
  result['channels'] = channels
  positions = np.arange(size)
  result['statistics'] = {
      'ref': statistics.summary(ref, positions),
      'target': statistics.summary(target, positions),
      'delta': statistics.summary(delta, positions),
      'abs_delta': statistics.summary(np.abs(delta), positions),
  }
  return result


def _number(value: float) -> float | str | None:
  """A JSON-safe element: NaN and infinities spelled as JavaScript does."""
  return metrics.json_number(float(value))


def _reason(code: str, side: str | None = None) -> dict[str, Any]:
  """A reason of `code` in the words `REASONS` gives it."""
  return metrics.reason(code, REASONS[code], side)


def _merged(reasons: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Both sides' reasons, when they say the same, as one of both."""
  if len(reasons) == 2 and all(
      reasons[0][key] == reasons[1][key] for key in ('code', 'text')
  ):
    return [{**reasons[0], 'side': None}]
  return reasons
