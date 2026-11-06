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

"""What else a Session may see inside a layer: the editor's operator list.

One list for both sides (GET /api/models/ops): the semantic operators of
the decoder layer of the two models' architecture, as its skeleton names
and groups them by block, that a side's binding maps (graph/layer_ops).
Ticking one captures it in every layer that has it, on both sides, beside
the anchors (session_store's `tap_ops`). Per operator the list says where
a side lacks it and why, by ranges of layers (`not in layers 15–34 (they
reuse layer 13/14's K and V)`), and what it adds to the capture per
generated token: of a LiteRT-LM model the bytes of its decode signature's
tensors of every layer, of a PyTorch model its modules' (their dims, in
the dtype the model stores), both sides together; and the total of those
ticked. A LiteRT-LM model observes at most tap_model.MAX_POINTS tensors,
its anchors among them: operators beyond that are refused, and say so.

A LiteRT-LM model's layers are known from its graph, which is converted as
a Session of it first turns on: until then the list is not known
(PENDING). A model of no architecture the server has a template of has
none (NONE), and says why. It converts and runs nothing.
"""

from collections.abc import Iterable
import dataclasses
from typing import Any

from model_debugger.server.architectures import catalog
from model_debugger.server.graph import layer_ops
from model_debugger.server.graph import pytorch_binding
from model_debugger.server.session import anchor_points
from model_debugger.server.session import model_graphs as model_graphs_lib
from model_debugger.server.session import model_store
from model_debugger.server.session import prepared_models
from model_debugger.server.session import tap_scans
from model_debugger.server.tap import tap_model
from model_debugger.server.util import errors

FOUND = anchor_points.FOUND
PENDING = anchor_points.PENDING
NONE = anchor_points.NONE
TITLE = 'Inside each layer'
DESCRIPTION = (
    'Tick an operator to capture it in every layer that has it, on both '
    'sides, beside the anchors; Intermediate Tensors compares it there. '
    'Sizes are the extra capture per generated token, both sides together.'
)
_SIDE_NAMES = {'ref': 'Reference', 'target': 'Target'}


@dataclasses.dataclass(frozen=True)
class _Side:
  """What one model has of the operators inside its layers.

  Attributes:
      model: The model.
      architecture: Its architecture's name.
      architecture_id: Its id.
      document: Its skeleton.
      rules: Its binding's `ops`.
      layers: Per operator mapped, the layers it is captured in (of a LiteRT-LM
        model, in decode).
      bytes: Per operator mapped, bytes per generated token.
      missing: Per operator mapped, why it is not in some layers.
      points: Per operator, its points (LiteRT-LM).
      anchors: The anchors the model captures (LiteRT-LM).
      prefill: The layers its prefill computes an operator of (LiteRT-LM); None
        if it has no prefill signature captured.
  """

  model: model_store.Model
  architecture: str
  architecture_id: str
  document: dict[str, Any]
  rules: dict[str, Any]
  layers: dict[str, set[int]]
  bytes: dict[str, int]
  missing: dict[str, list[str]]
  points: dict[str, set[str]] = dataclasses.field(default_factory=dict)
  anchors: frozenset[str] = frozenset()
  prefill: set[int] | None = None


class OpMenu:
  """The editor's list of operators inside the layers of two models."""

  def __init__(
      self,
      models: model_store.ModelStore,
      graphs: model_graphs_lib.ModelGraphs | None,
      scans: tap_scans.TapScans,
  ) -> None:
    """Reads the models given, their graphs and their scans."""
    self._models = models
    self._graphs = graphs
    self._scans = scans

  def menu(
      self, ref: str, target: str, chosen: Iterable[str] = ()
  ) -> dict[str, Any]:
    """The list for a Session of two models, with the operators ticked.

    Args:
        ref: The Reference side's model, by artifact.
        target: The Target side's.
        chosen: The operators ticked.

    Returns:
        `{status, text, title, description, blocks, notes, total,
        refusal}`: `status` FOUND, PENDING or NONE, `text` why when not
        FOUND; `blocks` in the skeleton's order, each `{name, ops}`, an
        operator `{id, name, kind, label, available, reason, size,
        size_detail}`; `notes` what holds of every operator; `total`
        `{bytes, text}` of those ticked; `refusal` why those ticked
        cannot be captured, '' if they can.

    Raises:
        NotFound: A model is not registered.
        ValueError, OSError: A model cannot be read.
    """
    runs = {
        'ref': self._models.get(ref),
        'target': self._models.get(target),
    }
    sides, statuses, reasons = {}, set(), []
    for model in runs.values():
      if model.artifact in sides or model.artifact in statuses:
        continue
      side, status, why = self._side(model)
      if side is None:
        statuses.update((status, model.artifact))
        reasons.append(why)
      else:
        sides[model.artifact] = side
    if reasons:
      return _empty(NONE if NONE in statuses else PENDING, ' '.join(reasons))
    ids = {side.architecture_id for side in sides.values()}
    if len(ids) > 1:
      named = ' and '.join(
          f'{side.model.name} is {side.architecture}' for side in sides.values()
      )
      return _empty(
          NONE,
          f'{named}: the operators inside a layer are paired within '
          'one architecture.',
      )
    first = next(iter(sides.values()))
    mapped = set().union(*(side.rules for side in sides.values()))
    ops = layer_ops.operators(first.document, mapped)
    wanted = set(chosen)
    blocks = {}
    total = dict.fromkeys(runs, 0)
    for op in ops:
      entry = self._entry(op, runs, sides)
      blocks.setdefault(op.block, []).append(entry)
      if op.id in wanted and entry['available']:
        for name, model in runs.items():
          total[name] += sides[model.artifact].bytes.get(op.id, 0)
    return {
        'status': FOUND,
        'text': '',
        'title': TITLE,
        'description': DESCRIPTION,
        'blocks': [
            {'name': name, 'ops': entries} for name, entries in blocks.items()
        ],
        'notes': _notes(sides.values()),
        'total': {
            'bytes': sum(total.values()),
            'text': _total_text(total, sides.values(), wanted),
        },
        'refusal': _refusal(sides.values(), wanted),
    }

  def _entry(
      self,
      op: layer_ops.Op,
      runs: dict[str, model_store.Model],
      sides: dict[str, _Side],
  ) -> dict[str, Any]:
    """One operator of the list."""
    distinct = list(sides.values())
    runtimes = {side.model.runtime for side in distinct}
    lines = {}
    for side in distinct:
      if op.id not in side.rules:
        mine = [f'the {side.model.runtime} binding maps no tensor of it']
      else:
        mine = side.missing.get(op.id, [])
      for line in mine:
        who = side.model.runtime if len(runtimes) > 1 else side.model.name
        lines.setdefault(line, []).append(who)
    reason = '; '.join(
        line if len(whose) == len(distinct) else f'{", ".join(whose)}: {line}'
        for line, whose in lines.items()
    )
    available = all(side.layers.get(op.id) for side in distinct)
    per_run = {
        name: sides[model.artifact].bytes.get(op.id, 0)
        for name, model in runs.items()
    }
    if available:
      size_detail = (
          ' · '.join(
              f'{_SIDE_NAMES[name]} {size_text(count)}'
              for name, count in per_run.items()
          )
          + ' per generated token'
      )
    else:
      size_detail = ''
    return {
        'id': op.id,
        'name': op.name,
        'kind': op.kind,
        'label': op.label,
        'available': available,
        'reason': reason,
        'size': size_text(sum(per_run.values())) if available else '',
        'size_detail': size_detail,
    }

  def _side(self, model: model_store.Model) -> tuple[_Side | None, str, str]:
    """A model's operators, or (None, PENDING or NONE, why)."""
    if model.runtime == model_store.PYTORCH:
      return self._pytorch(model)
    return self._litert(model)

  def _pytorch(self, model: model_store.Model) -> tuple[_Side | None, str, str]:
    """Resolves operators for a PyTorch model."""
    config = model.config()
    try:
      definition, document = catalog.generate(config)
      rules = pytorch_binding.binding(definition.id).get('ops', {})
    except ValueError as error:
      return (
          None,
          NONE,
          f'{model.name} has no architecture template: {errors.message(error)}',
      )
    dtype = prepared_models.pytorch_dtype(config)
    layers, sizes, missing = {}, {}, {}
    labels = layer_ops.operators(document, rules)
    for op in labels:
      modules, _ = layer_ops.pytorch_modules(document, rules, op.id, op.label)
      layers[op.id] = set(modules)
      sizes[op.id] = sum(
          layer_ops.module_bytes(module, dtype) for module in modules.values()
      )
      missing[op.id] = layer_ops.missing(
          document, op.id, modules, 'PyTorch has no module of it there'
      )
    return (
        _Side(
            model,
            definition.name,
            definition.id,
            document,
            rules,
            layers,
            sizes,
            missing,
        ),
        FOUND,
        '',
    )

  def _litert(self, model: model_store.Model) -> tuple[_Side | None, str, str]:
    """Resolves operators for a LiteRT model."""
    if self._graphs is None:
      return (
          None,
          NONE,
          (
              'This server keeps no graphs, so the operators inside the '
              'layers are not known.'
          ),
      )
    state = self._graphs.state(model.artifact)
    if state.status != model_graphs_lib.COMPLETED:
      where = {
          model_graphs_lib.CONVERTING: 'is being converted',
          model_graphs_lib.FAILED: f'could not be converted: {state.error}',
      }.get(state.status, 'is converted as the Model Server first turns on')
      return (
          None,
          PENDING,
          (
              f"The operators inside {model.name}'s layers are found in "
              f'its graph, which {where}.'
          ),
      )
    found = self._graphs.architecture(model)
    known = found.get('architecture')
    if known is None:
      return (
          None,
          NONE,
          (
              f'{model.name} has no architecture template: '
              f'{found.get("error") or found.get("text")}'
          ),
      )
    document = found['skeleton']
    rules = prepared_models.litert_rules(known['id'])
    layers = self._graphs.layers(model)
    scan = self._scans.scan(model, layers.stems, layers.groups)
    signatures = prepared_models.captured_signatures(scan)
    by_op = layer_ops.litert_points(scan.points, rules, rules, signatures)
    decoded, sizes, missing, points = {}, {}, {}, {}
    prefill = set()
    for op, by_signature in by_op.items():
      decode = by_signature.get(layer_ops.DECODE, {})
      decoded[op] = set(decode)
      sizes[op] = sum(layer_ops.token_bytes(point) for point in decode.values())
      missing[op] = layer_ops.missing(
          document,
          op,
          decode,
          'its LiteRT-LM graph has no tensor of it there',
      )
      op_points = set()
      for by_layer in by_signature.values():
        for point in by_layer.values():
          op_points.add(point.id)
      points[op] = op_points
      for signature, by_layer in by_signature.items():
        if signature != layer_ops.DECODE:
          prefill.update(by_layer)
    anchors = anchor_points.find(self._graphs, self._scans, model)
    return (
        _Side(
            model,
            known['name'],
            known['id'],
            document,
            rules,
            decoded,
            sizes,
            missing,
            points,
            frozenset(anchors.point_ids),
            prefill if len(signatures) > 1 else None,
        ),
        FOUND,
        '',
    )


def _empty(status: str, text: str) -> dict[str, Any]:
  """The list when it is not known."""
  return {
      'status': status,
      'text': text,
      'title': TITLE,
      'description': DESCRIPTION,
      'blocks': [],
      'notes': [],
      'total': {'bytes': 0, 'text': ''},
      'refusal': '',
  }


def _notes(sides: Iterable[_Side]) -> list[str]:
  """What holds of every operator: a LiteRT-LM prefill's fewer layers."""
  notes = []
  for side in sides:
    if side.prefill is None:
      continue
    every = set().union(*side.layers.values())
    rest = every - side.prefill
    if side.prefill and rest:
      notes.append(
          f'In prefill, {side.model.name} computes only '
          f'{layer_ops.layer_ranges(side.prefill)}: the operators of '
          f'{layer_ops.layer_ranges(rest)} are captured in decode '
          'alone.'
      )
  return notes


def _total_text(
    total: dict[str, int], sides: Iterable[_Side], wanted: set[str]
) -> str:
  """What the operators ticked add, per generated token, as said."""
  if not wanted:
    return 'No operator inside the layers is ticked: only the anchors.'
  text = (
      f'Ticked: {len(wanted)} operator{"" if len(wanted) == 1 else "s"}, '
      f'{size_text(sum(total.values()))} more per generated token ('
      + ' · '.join(
          f'{_SIDE_NAMES[name]} {size_text(count)}'
          for name, count in total.items()
      )
      + ').'
  )
  for side in sides:
    if side.model.runtime != model_store.LITERT_LM:
      continue
    count = len(_ticked_points(side, wanted) - side.anchors)
    text += (
        f' {side.model.name}: {count:,} more tensor output'
        f'{"" if count == 1 else "s"} beside its {len(side.anchors):,} '
        'anchors.'
    )
  return text


def _ticked_points(side: _Side, wanted: set[str]) -> set[str]:
  """The points of the operators ticked of a LiteRT-LM model."""
  found = set()
  for op in wanted:
    found |= side.points.get(op, set())
  return found


def _refusal(sides: Iterable[_Side], wanted: set[str]) -> str:
  """Why the operators ticked cannot be captured; '' if they can."""
  for side in sides:
    if side.model.runtime != model_store.LITERT_LM:
      continue
    count = len(_ticked_points(side, wanted) - side.anchors)
    anchors = len(side.anchors)
    if count + anchors > tap_model.MAX_POINTS:
      return (
          f'{side.model.name} would observe {count + anchors:,} tensor '
          f'outputs ({anchors:,} anchors and {count:,} of the operators '
          'ticked); a copy of it observes at most '
          f'{tap_model.MAX_POINTS:,}. Tick fewer operators.'
      )
  return ''


def size_text(size: int) -> str:
  """Bytes as the user reads them: '96 KiB', '1.3 MiB'."""
  if size >= 1024**2:
    return f'{size / 1024**2:.1f} MiB'
  if size >= 1024:
    return f'{size / 1024:.0f} KiB'
  return f'{size} B'
