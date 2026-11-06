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

"""The copies of models that observe the tensors Sessions observe (tap).

A Session that observes tensors (session_store.tap_points) is prepared as
its Model Server starts: the start's own job first finds the anchors the
Session captures, by default every decoder layer's of each LiteRT-LM model
(anchor_points), once the start converted the models' graphs, and keeps
them in the record (SessionStore.set_anchor_points()); then it writes and
checks, for each model its runs load, a copy that observes the anchors and
the points chosen for it (tap_model.write), then each side loads the copy
instead of the model (prepare_start(), capture_model()). A model whose
points are all
outputs of their signatures already is loaded as it is: the runtime captures
those without a copy. A copy is written in the workspace,
`prepared/<key>/model.litertlm` with `prepared.json` beside it, by the
model's SHA-256, the points and the code that writes copies
(tap_model.identity()): every Session that chooses the same points of the
same model, and every later start, uses the one copy, which only a runner
that lacks it is sent. A copy is deterministic, so one written again, after
its directory was deleted, has the same SHA-256, and a runner that kept it
is not sent it again either.

A copy lives as long as an existing Session uses it: one whose record's
tap_points name that model and those points (_referenced()); no
count of users is kept beside the records, to drift from them. A Session
thus has one copy of each model it observes tensors of, from its start
until it is deleted. collect() deletes every other copy: when a Session is
deleted (the service calls it then), as the server starts (a copy left by a
crash, or one older code wrote, whose key no Session names now), and when a
start that used a copy ends. A copy a start writes or loads is in use
meanwhile (_using), and is not deleted until it ends: a start holds its
copies until its sides loaded them (loaded()).

What a Session observes, its points as each model's scan lists them, is
what the analyses of its intermediate tensors read (observed()); the scan
places them in layers as the model's graph does
(model_graphs.ModelGraphs.layers()). A Session that started has its
models' graphs, since a start fails without them; a model whose graph is
not kept, such as an old Session's recorded before graphs were, has no
layers: the analyses say where its graph is. They never convert it; a
Session's initialize does.

The operators a Session chose inside the layers (session_store.tap_ops())
are found as its anchors are, in the same step (find_anchors()): of each
LiteRT-LM model, the tap point of each operator in each layer of each
signature the anchors are captured in (graph/layer_ops), kept in the
record beside the anchors and observed with them by the same copy; each is
placed at (layer, operator) (observed()). A PyTorch run that observes
observes, beside its anchors, each operator's module of every layer that
has it, placed there too, which its runner is asked for at its Load
(observe_modules()).

One copy is written at a time; a preparation that would write another
waits, and says so.
"""

import collections
from collections.abc import Callable, Iterable, Iterator
import contextlib
import dataclasses
import hashlib
import json
import os
import re
import shutil
import threading
import time
from typing import Any
import uuid

from absl import logging
from model_debugger.server.analysis import metrics
from model_debugger.server.analysis import tensor_analysis
from model_debugger.server.architectures import catalog
from model_debugger.server.capture import capture_index
from model_debugger.server.graph import architecture as architecture_lib
from model_debugger.server.graph import counterparts
from model_debugger.server.graph import layer_ops
from model_debugger.server.graph import pytorch_binding
from model_debugger.server.session import anchor_points
from model_debugger.server.session import jobs as jobs_lib
from model_debugger.server.session import mappings as mappings_lib
from model_debugger.server.session import model_graphs as model_graphs_lib
from model_debugger.server.session import model_store
from model_debugger.server.session import options as options_lib
from model_debugger.server.session import session_store
from model_debugger.server.session import tap_scans
from model_debugger.server.session import workspace
from model_debugger.server.tap import tap_model
from model_debugger.server.tap import tap_scan
from model_debugger.server.util import clock
from model_debugger.server.util import errors
from model_debugger.server.util import json_file

_MODEL = 'model.litertlm'
_MANIFEST = 'prepared.json'
# What reading a side's points fails with: its model gone or unreadable, its
# graph not converted.
_UNREADABLE = (errors.NotFound, errors.Conflict, ValueError, OSError)
# Disk a prepare leaves free, beyond the copy's size.
_FREE_MARGIN_BYTES = 256 * 1024**2
# How often a preparation that waits for another looks whether its start
# was turned off.
_WAIT_SECONDS = 0.25
NOT_PREPARED = (
    'The tensor capture is not prepared; create a new Session to prepare it.'
)
ANCHORS_NEEDED = (
    'The anchors the tensor capture observes are found, and it is prepared, '
    'when the Model Server turns on.'
)
FAILED_TEXT = 'Preparing the tensor capture failed: '
WAITING = 'waiting for another tensor capture to be prepared'
# A record's tap_preparation state, as the listing shows it.
NONE = 'none'
NEEDED = 'needed'
RUNNING = 'preparing'
READY = 'prepared'


@dataclasses.dataclass(frozen=True)
class Prepared:
  """A copy of a model that observes points.

  Attributes:
      key: What names it: of the model's SHA-256, the points, and the code that
        wrote it.
      model: The copy, as a model to load.
      manifest: Its prepared.json: the source, the points and their outputs,
        what the check compared.
  """

  key: str
  model: model_store.Model
  manifest: dict[str, Any]


class _StoppedError(Exception):
  """The start a preparation runs in was turned off."""


def key(model_sha256: str, point_ids: list[str]) -> str:
  """The key of the copy of a model that observes points, in any order.

  It names the code that writes copies too (tap_model.identity()): a copy
  an older server wrote is not found, and is written anew.

  Args:
    model_sha256: The SHA-256 hash of the model.
    point_ids: The list of point IDs observed.

  Returns:
    The hex digest SHA-256 key for the copy.
  """
  name = json.dumps(
      [tap_model.identity(), model_sha256, sorted(point_ids)],
      separators=(',', ':'),
  )
  return hashlib.sha256(name.encode()).hexdigest()


def needs_copy(scan: tap_scan.Scan, point_ids: list[str]) -> bool:
  """Whether observing points takes a copy: one is no output already.

  Args:
    scan: The scan of the model.
    point_ids: The list of point IDs to check.

  Returns:
    True if any point is not already an output, False otherwise.

  Raises:
    ValueError: The model has no such point.
  """
  return any(scan.point(i).output_name is None for i in point_ids)


def size_text(size: int) -> str:
  """A size as the user reads it, such as '2.4 GiB' or '3.0 MiB'."""
  if size >= 1024**3 // 10:
    return f'{size / 1024**3:.1f} GiB'
  return f'{size / 1024**2:.1f} MiB'


class PreparedModels:
  """The workspace's copies, and the preparations that write them.

  Locks: _writing is taken before _lock, never after; collect() takes
  _lock alone and, under it, the SessionStore's and the ModelStore's, which
  never wait for this class's.
  """

  def __init__(
      self,
      workspace_: workspace.Workspace,
      models: model_store.ModelStore,
      sessions: session_store.SessionStore,
      jobs: jobs_lib.Jobs,
      scans: tap_scans.TapScans,
      graphs: model_graphs_lib.ModelGraphs | None = None,
      mappings: mappings_lib.MappingStore | None = None,
  ) -> None:
    """Holds the copies in the workspace; prepares nothing yet.

    Deletes the copies no Session uses (collect()): the server starts.

    Args:
        workspace_: The workspace.
        models: The registered models.
        sessions: The Session records.
        jobs: The jobs preparations report to.
        scans: The scans of the models' tensors.
        graphs: The models' graphs, whose layers place the points observed; None
          places none in a layer.
        mappings: The tensor pairs made by hand, which a Session of two models
          places too; None places none.
    """
    self._directory = workspace_.prepared
    self._mappings = mappings
    self._models = models
    self._sessions = sessions
    self._jobs = jobs
    self._scans = scans
    self._graphs = graphs
    self._lock = threading.Lock()
    # Held while a copy is written: one at a time.
    self._writing = threading.Lock()
    # Record id -> the jobs preparing it now: its start's.
    self._running: dict[str, list[str]] = {}
    # Key -> how many starts use its copy now; collect()
    # deletes none of these.
    self._using: collections.Counter[str] = collections.Counter()
    # A start's job -> the keys it holds until its sides loaded them.
    self._starts: dict[str, list[str]] = {}
    self.collect()

  @property
  def mappings(self) -> mappings_lib.MappingStore | None:
    """The tensor pairs made by hand, which this places; None if none."""
    return self._mappings

  # ---- The copies.

  def find(
      self, model: model_store.Model, point_ids: list[str]
  ) -> Prepared | None:
    """The copy of a model that observes points, if it is prepared.

    A copy whose file is gone, or is not the size and time it was written
    with, is not.

    Args:
      model: The model to find a copy of.
      point_ids: The list of point IDs observed.

    Returns:
      The Prepared copy if found and valid, None otherwise.
    """
    directory = self._directory / key(model.sha256, point_ids)
    try:
      manifest = json_file.read_json(directory / _MANIFEST)
      stat = (directory / _MODEL).stat()
    except (OSError, ValueError):
      return None
    if (stat.st_size, stat.st_mtime_ns) != (
        manifest.get('size'),
        manifest.get('mtime_ns'),
    ):
      return None
    return Prepared(
        key=directory.name,
        model=model_store.Model(
            artifact='prepared-' + manifest['sha256'][:24],
            sha256=manifest['sha256'],
            name=manifest['name'],
            size=manifest['size'],
            path=directory / _MODEL,
            sampler_defaults=model.sampler_defaults,
        ),
        manifest=manifest,
    )

  def disk_note(self, model: model_store.Model) -> str:
    """What preparing a model costs on disk, as the editor says it.

    Args:
      model: The model the editor chooses tensors of.

    Returns:
      A human-readable explanation of disk usage for prepared copies.
    """
    return (
        'Starting the Session writes a copy of '
        f'{model.name} that outputs the tensors observed, about '
        f'{size_text(model.size)}, unless each is an output already. '
        'A Session keeps its copy until it is deleted, and Sessions that '
        'observe the same tensors share one.'
    )

  def collect(self) -> list[str]:
    """Deletes every copy no existing Session, start or preparation uses.

    A copy an existing Session uses is one its record's points name
    (_referenced()). Each copy deleted is first renamed to a `.pending-`
    directory, under the lock, then removed: a preparation that comes
    meanwhile finds no copy, and writes it anew.

    Returns:
      The keys of the copies deleted.
    """
    trash = []
    with self._lock:
      kept = self._referenced() | set(self._using)
      for directory in sorted(self._directory.iterdir()):
        if (
            directory.name.startswith(workspace.PENDING)
            or directory.name in kept
            or not directory.is_dir()
        ):
          continue
        gone = self._directory / (workspace.PENDING + uuid.uuid4().hex)
        directory.rename(gone)
        trash.append((directory.name, gone))
    for name, gone in trash:
      logging.info('Deleting the copy %s: no Session uses it', name)
      shutil.rmtree(gone, ignore_errors=True)
    return [name for name, _ in trash]

  def _referenced(self) -> set[str]:
    """The keys of the copies the existing Sessions use.

    Read from the records' points: a record whose model is no longer
    registered uses no copy of it, since none of its runs can load it.

    Returns:
      A set of copy keys referenced by existing sessions.
    """
    found = set()
    for record in self._sessions.list():
      found.update(self._keys(record))
    return found

  def _keys(self, record: dict[str, Any]) -> list[str]:
    """The keys of the copies a record's points name, needed or not."""
    found = []
    for artifact, ids in session_store.tap_points(record).items():
      try:
        found.append(key(self._models.get(artifact).sha256, ids))
      except errors.NotFound:
        continue
    return found

  def _hold(self, keys: list[str]) -> None:
    """Marks copies in use: collect() deletes none until _release()."""
    with self._lock:
      self._using.update(keys)

  def _release(self, keys: list[str]) -> None:
    """Ends a _hold(); deletes what no Session uses any more."""
    with self._lock:
      self._using.subtract(keys)
      for name in keys:
        if self._using[name] <= 0:
          del self._using[name]
    self.collect()

  def loaded(self, job_id: str) -> None:
    """Lets go of the copies a start held: its sides loaded them.

    Called on the start's thread once both sides loaded, or the start
    failed or was turned off; a start that held none lets go of nothing.

    Args:
        job_id: The start's job, as prepare_start() was given it.
    """
    with self._lock:
      keys = self._starts.pop(job_id, None)
    if keys is not None:
      self._release(keys)

  def capture_model(
      self, record: dict[str, Any], run: dict[str, Any]
  ) -> model_store.Model:
    """The model a run of a Session loads: its copy, if it takes one.

    Args:
        record: The Session's record.
        run: One of its runs.

    Returns:
        The run's model, or its copy that observes the Session's points.

    Raises:
        NotFound: The run's model is not registered.
        Conflict: The copy is not prepared.
        ValueError: The model cannot be observed, or lacks a point.
    """
    model = self._models.get(run.get('artifact', ''))
    points = session_store.tap_points(record).get(model.artifact)
    if not points or not needs_copy(self._scans.scan(model), points):
      return model
    found = self.find(model, points)
    if found is None:
      raise errors.Conflict(NOT_PREPARED)
    return found.model

  def observed(self, record: dict[str, Any]) -> tensor_analysis.Observed:
    """What a Session observes: each run's model and points.

    A LiteRT-LM model's points are those chosen of its scan, scanned
    now if no scan is kept (tap_scans), by the layers its graph names
    (model_graphs.ModelGraphs.layers()). A PyTorch run that observes
    (its `observe` option) observes every layer's anchors, placed by
    the skeleton its config.json makes and its PyTorch binding
    (pytorch_binding); beside a PyTorch run, or in a Session that
    captures the anchors, a LiteRT-LM model's points are placed by the
    anchors its graph binds them to (ModelGraphs.architecture()), for
    two runtimes pair by anchor, and one runtime's anchors are shown as
    what they are. Beside a PyTorch run that observes, each other
    LiteRT-LM point chosen is placed at the semantic operator it is the
    output of, where the PyTorch module of that operator is certain, and
    the PyTorch run observes that module too (counterparts, pairing());
    any other says why not (`notes`). Of two different models, each
    pair made by hand of them (mappings) whose two tensors the sides
    observe is placed too, at a place of its own, and a PyTorch side
    observes the module it names; one whose tensor a side does not
    observe says why (`notes`). It never raises: a model gone, one
    whose points cannot be read, or one whose graph is not kept
    (converting, failed, or never converted) is a reason.

    Args:
      record: The Session's record.

    Returns:
      The Observed specification of points and models for each run.
    """
    return self._observe(record)[0]

  def _observe(
      self, record: dict[str, Any]
  ) -> tuple[tensor_analysis.Observed, list['_ByHand']]:
    """observed(), and where it placed each pair made by hand."""
    points = session_store.tap_points(record)
    pytorch = [
        run
        for run in record['runs']
        if run.get('runtime') == model_store.PYTORCH
    ]
    observing = [
        run
        for run in pytorch
        if run.get('observe') == options_lib.OBSERVE_ANCHORS
    ]
    if not points and not observing:
      return tensor_analysis.nothing(), []
    sides, models = {}, {}
    paired: list[counterparts.Counterpart] = []
    for run in record['runs']:
      if run.get('runtime') == model_store.PYTORCH:
        continue
      try:
        model = self._models.get(run.get('artifact', ''))
        stems = groups = None
        if self._graphs is not None:
          layers = self._graphs.layers(model)
          if layers.stems is None:
            return (
                tensor_analysis.nothing(
                    tensor_analysis.MODEL_UNREADABLE,
                    f'The layers of the {run["id"]} side come from '
                    f"its model's graph, which {layers.reason}.",
                ),
                [],
            )
          stems, groups = layers.stems, layers.groups
        scan = self._scans.scan(model, stems, groups)
        chosen = tuple(
            scan.point(point_id) for point_id in points.get(model.artifact, [])
        )
        side = tensor_analysis.Side(model.sha256, chosen)
        by_anchor = pytorch or session_store.captures_anchors(record)
        if by_anchor and chosen and self._graphs is not None:
          side = self._anchored(model, side)
          side = self._op_placed(
              model, side, scan, session_store.tap_ops(record)
          )
        if observing and chosen and self._graphs is not None:
          side, found = self._paired(record, model, side, scan, observing[0])
          paired += found
      except _UNREADABLE as error:
        return (
            tensor_analysis.nothing(
                tensor_analysis.MODEL_UNREADABLE,
                f'The points the {run["id"]} side observes cannot be '
                f'read from its model: {errors.message(error)}',
            ),
            [],
        )
      sides[run['id']] = side
      models[run['id']] = model
    for run in pytorch:
      try:
        model = self._models.get(run.get('artifact', ''))
        sides[run['id']] = _pytorch_side(
            model, run, paired, session_store.tap_ops(record)
        )
        models[run['id']] = model
      except _UNREADABLE as error:
        return (
            tensor_analysis.nothing(
                tensor_analysis.MODEL_UNREADABLE,
                f'The points the {run["id"]} side observes cannot be '
                f'read from its model: {errors.message(error)}',
            ),
            [],
        )
    placed = self._by_hand(record, sides, models)
    manual = set()
    for each in placed:
      if each.place is not None:
        for point in each.points.values():
          manual.add(point)
    notes = [
        metrics.reason(tensor_analysis.NO_COUNTERPART, found.reason)
        for found in paired
        if found.module is None and found.point_id not in manual
    ]
    notes += [
        metrics.reason(tensor_analysis.MANUAL_NOT_OBSERVED, each.text)
        for each in placed
        if each.place is None
    ]
    return (
        tensor_analysis.Observed(
            {side: sides[side] for side in capture_index.SIDES},
            notes=tuple(notes),
        ),
        placed,
    )

  # ---- Pairs made by hand.

  def _by_hand(
      self,
      record: dict[str, Any],
      sides: dict[str, tensor_analysis.Side],
      models: dict[str, model_store.Model],
  ) -> list['_ByHand']:
    """Places each pair made by hand of the Session's two models.

    In `sides`, in place: a pair whose two tensors the sides observe is
    placed at (layer, manual anchor) on both, labelled by the pair; a
    PyTorch side that observes observes the module a pair names, which
    its runner is then asked for (observe_modules()). None of one model.

    Args:
      record: The Session's record.
      sides: The dictionary of sides mapped to their analysis Side.
      models: The dictionary of models for each side.

    Returns:
      Each pair, where it is placed, or why not.
    """
    if (
        self._mappings is None
        or self._graphs is None
        or set(models) != set(capture_index.SIDES)
        or models['ref'].sha256 == models['target'].sha256
    ):
      return []
    runs = {run['id']: run for run in record['runs']}
    placed = []
    for pair in self._mappings.pairs(
        models['ref'].sha256, models['target'].sha256
    ):
      found, why = {}, []
      for name, tensor in (('ref', pair.ref), ('target', pair.target)):
        point, text = self._hand_point(
            sides[name], models[name], runs.get(name, {}), tensor
        )
        if point is None:
          why.append(f'{mappings_lib.SIDE_NAMES[name]} {text}')
        else:
          found[name] = point
      if why:
        placed.append(
            _ByHand(
                pair,
                None,
                _label(pair, found),
                {n: p.id for n, p in found.items()},
                f'{_label(pair, found)}: ' + '; '.join(why) + '.',
            )
        )
        continue
      layer = found['ref'].layer
      if layer is None:
        layer = found['target'].layer
      label = _label(pair, found)
      if layer is None:
        placed.append(
            _ByHand(
                pair,
                None,
                label,
                {n: p.id for n, p in found.items()},
                f'{label}: neither tensor is in a decoder layer.',
            )
        )
        continue
      place = (layer, tensor_analysis.manual_anchor(pair.id))
      for name, point in found.items():
        side = sides[name]
        points = side.points
        keys = dict(side.keys)
        if point.id not in {each.id for each in side.points}:
          points = (*points, point)
          module, _, edge = point.id.split(':', 1)[1].rpartition(':')
          keys[point.id] = capture_index.module_key(module, edge)
        places = dict(side.places)
        places[point.id] = (*places.get(point.id, ()), place)
        sides[name] = dataclasses.replace(
            side,
            points=points,
            places=places,
            labels={**side.labels, place[1]: label},
            keys=keys,
        )
      placed.append(
          _ByHand(
              pair,
              place,
              label,
              {name: point.id for name, point in found.items()},
              '',
          )
      )
    return placed

  def _hand_point(
      self,
      side: tensor_analysis.Side,
      model: model_store.Model,
      run: dict[str, Any],
      tensor: mappings_lib.Tensor,
  ) -> tuple[tap_scan.Point | None, str]:
    """A side's point of a tensor a pair names, or why it has none."""
    mine = {point.id: point for point in side.points}
    if tensor.point in mine:
      return mine[tensor.point], ''
    if model.runtime != model_store.PYTORCH:
      return None, (
          f'{tensor.point} is not observed in this Session: add it in '
          'Tensor capture (the Session editor) to compare it'
      )
    if run.get('observe') != options_lib.OBSERVE_ANCHORS:
      return None, (
          'PyTorch captures no tensors in this Session: turn its tensor '
          'capture on in the Session editor'
      )
    try:
      facts = self._graphs.tensor(
          model, tensor.signature, tensor.node, tensor.output
      )
    except (errors.NotFound, errors.Conflict) as error:
      return None, errors.message(error).rstrip('.')
    shape = mappings_lib.shape_of(facts.get('tensor_shape', ''))
    _, module_edge = tensor.point.split(':', 1)
    module, _, edge = module_edge.rpartition(':')
    if (
        shape is None
        or facts.get('point_id') != tensor.point
        or (edge not in ('in', 'out'))
    ):
      return None, (
          f'{tensor.point} is no longer what node {tensor.node}:'
          f'{tensor.output} of its {tensor.signature} graph captures'
      )
    layer = facts.get('layer')
    dtype = next((point.dtype for point in side.points), '')
    return (
        tensor_analysis.module_point(
            tensor.signature,
            module,
            edge,
            shape[2:] if len(shape) >= 2 else shape,
            dtype,
            layer=int(layer) if layer and layer.isdigit() else None,
            op=len(side.points),
        ),
        '',
    )

  def manual_pairs(
      self,
      record: dict[str, Any],
      asked: dict[str, set[str] | None] | None = None,
  ) -> dict[str, Any]:
    """The pairs made by hand of a Session's two models, as it has them.

    Args:
        record: The Session's record.
        asked: Per side, the modules its Load asked its runner for
          (tensor_analysis.asked_modules()), once the Session started; None
          before.

    Returns:
        `{status, text, pairs}`: `status` 'pairs', 'identity'
        (one model: none is needed, `text` says so) or 'none' (`text`
        why none can be made); per pair `{id, label, created, ref,
        target, place, text}`, `ref` and `target` its tensors
        `{signature, node, output, point}` as this Session's sides are,
        `place` `{layer, anchor}` where Intermediate Tensors has it, or
        None with `text` why; `text` also says when a PyTorch module it
        names is captured from the next start only.
    """
    models, why = self._pair_models(record)
    if why:
      return {
          'status': 'none' if models is None else 'identity',
          'text': why,
          'pairs': [],
      }
    observed, placed = self._observe(record)
    by_id = {each.pair.id: each for each in placed}
    found = []
    for pair in self._mappings.pairs(
        models['ref'].sha256, models['target'].sha256
    ):
      each = by_id.get(pair.id)
      if each is None:
        # The Session observes nothing comparable: what of each
        # tensor it lacks, as if it observed nothing else.
        runs = {run['id']: run for run in record['runs']}
        why = []
        for name, tensor in (
            ('ref', pair.ref),
            ('target', pair.target),
        ):
          _, lacking = self._hand_point(
              tensor_analysis.Side(models[name].sha256, ()),
              models[name],
              runs.get(name, {}),
              tensor,
          )
          if lacking:
            why.append(f'{mappings_lib.SIDE_NAMES[name]} {lacking}')
        reason = (observed.reason or {}).get('text', '')
        text = f'{_label(pair, {})}: ' + (
            '; '.join(why) + '.' if why else reason
        )
      else:
        text = each.text
      if each is not None and each.place is not None and asked:
        # A module it names that the side's runner was not asked
        # for as the Session started.
        text = ' '.join(
            tensor_analysis.not_asked(
                observed.sides[name].keys.get(point, ''),
                asked.get(name),
            )
            for name, point in each.points.items()
            if observed.sides[name]
            .keys.get(point, '')
            .startswith(capture_index.MODULE_PREFIX)
        ).strip()
      found.append({
          **pair.to_json(),
          'label': each.label if each is not None else pair.label,
          'place': (
              {'layer': each.place[0], 'anchor': each.place[1]}
              if each is not None and each.place is not None
              else None
          ),
          'text': text,
      })
    return {
        'status': 'pairs',
        'text': '',
        'pairs': found,
    }

  def _pair_models(
      self, record: dict[str, Any]
  ) -> tuple[dict[str, model_store.Model] | None, str]:
    """A Session's two models, and why no pair can be made; '' if one can.

    The models are None when the server keeps no pairs or a model is
    gone; they are one model when its sides load one (IDENTITY).

    Args:
      record: The Session's record.

    Returns:
      A tuple of (models dict or None, reason string).
    """
    if self._mappings is None or self._graphs is None:
      return None, 'This server keeps no tensor pairs.'
    runs = {run['id']: run for run in record['runs']}
    models = {}
    for name in capture_index.SIDES:
      run = runs.get(name, runs.get('ref'))
      try:
        models[name] = self._models.get((run or {}).get('artifact', ''))
      except errors.NotFound as error:
        return None, errors.message(error)
    if models['ref'].sha256 == models['target'].sha256:
      return models, mappings_lib.IDENTITY
    return models, ''

  def save_pair(
      self,
      record: dict[str, Any],
      ref: object,
      target: object,
      label: object = None,
      asked: dict[str, set[str] | None] | None = None,
  ) -> dict[str, Any]:
    """Saves a pair made by hand of a Session's two tensors.

    Args:
        record: The Session's record.
        ref: The Reference's tensor, `{signature, node, output}`.
        target: The Target's tensor.
        label: What the user names the pair, if anything; without, it is named
          by its two tensors (mappings.default_label()).
        asked: As manual_pairs().

    Returns:
        manual_pairs() after it is saved.

    Raises:
        ValueError: The tensors are refused (mappings.check()), the
          sides load one model, or the server keeps no pairs.
        Conflict: The models pair them already, by hand or by
          themselves, or a model's graph is not converted.
    """
    models, why = self._pair_models(record)
    if why:
      raise ValueError(why)
    checked = mappings_lib.check(
        self._graphs,
        models,
        {
            'ref': mappings_lib.requested(ref, 'ref'),
            'target': mappings_lib.requested(target, 'target'),
        },
    )
    automatic = self._automatic(models, checked)
    if automatic:
      raise errors.Conflict(automatic)
    label = label.strip() if isinstance(label, str) else ''
    self._mappings.add(
        models['ref'].sha256,
        models['target'].sha256,
        checked['ref'].tensor,
        checked['target'].tensor,
        label or mappings_lib.default_label(checked),
    )
    return self.manual_pairs(record, asked)

  def delete_pair(
      self,
      record: dict[str, Any],
      pair_id: str,
      asked: dict[str, set[str] | None] | None = None,
  ) -> dict[str, Any]:
    """Deletes a pair made by hand of a Session's two models.

    Args:
      record: The Session's record.
      pair_id: The ID of the pair to delete.
      asked: Optional mapping of asked points.

    Returns:
      manual_pairs() after it is deleted.

    Raises:
      NotFound: The models have no such pair.
    """
    models, why = self._pair_models(record)
    if models is None or why:
      raise errors.NotFound(why or 'These models have no pairs.')
    self._mappings.delete(
        models['ref'].sha256, models['target'].sha256, pair_id
    )
    return self.manual_pairs(record, asked)

  def _automatic(
      self,
      models: dict[str, model_store.Model],
      checked: dict[str, mappings_lib.Checked],
  ) -> str:
    """Why a pair of a LiteRT-LM and a PyTorch tensor is paired already.

    By the anchors (the LiteRT-LM point is bound to an anchor whose
    PyTorch point is the other) or by its semantic operator
    (counterparts: the PyTorch module of the point's operator); ''
    otherwise, and for any other two models.

    Args:
      models: The models dictionary keyed by side.
      checked: The checked mappings for the tensors.

    Returns:
      A reason string if already paired, or '' otherwise.
    """
    runtimes = {name: models[name].runtime for name in models}
    if set(runtimes.values()) != {
        model_store.LITERT_LM,
        model_store.PYTORCH,
    }:
      return ''
    litert = next(n for n in runtimes if runtimes[n] == model_store.LITERT_LM)
    other = 'target' if litert == 'ref' else 'ref'
    point = checked[litert].tensor.point
    theirs = checked[other].tensor
    try:
      found = self.counterparts(models[litert], models[other], [point])
    except (ValueError, OSError, errors.NotFound):
      return ''
    if found['status'] != anchor_points.FOUND or not found['points']:
      return ''
    each = found['points'][0]
    if each['paired'] == 'module':
      automatic = pytorch_binding.module_point_id(
          theirs.signature, each['module'], each['edge']
      )
    elif each['paired'] == 'anchor':
      try:
        bound = pytorch_binding.anchors(models[other].config())
      except (ValueError, OSError):
        return ''
      automatic = next(
          (
              pytorch_binding.point_id(theirs.signature, anchor)
              for anchor in bound.anchors
              if anchor.layer == each['layer']
              and anchor.anchor == each['place']
          ),
          '',
      )
    else:
      return ''
    if automatic != theirs.point:
      return ''
    return (
        f'{point} and {theirs.point} are paired already, without a '
        f'pair made by hand: {each["text"]}'
    )

  def observe_modules(
      self, record: dict[str, Any], run: dict[str, Any]
  ) -> list[dict[str, Any]]:
    """The modules a PyTorch run observes beside the anchors.

    Those paired with the LiteRT-LM points the Session chose
    (observed()), each `{module, edge}` once, as its runner's Load
    takes them (observeModules); [] for a run of another runtime, one
    that observes nothing, or when none is paired. It never raises.

    Args:
      record: The Session's record.
      run: The PyTorch run dictionary.

    Returns:
      A list of dictionaries with module and edge specifications.
    """
    if (
        run.get('runtime') != model_store.PYTORCH
        or run.get('observe') != options_lib.OBSERVE_ANCHORS
    ):
      return []
    side = self.observed(record).sides.get(run['id'])
    if side is None:
      return []
    found = {}
    for point in side.points:
      side_key = side.keys.get(point.id, '')
      if not side_key.startswith(capture_index.MODULE_PREFIX):
        continue
      module, _, edge = point.id.split(':', 1)[1].rpartition(':')
      found.setdefault((module, edge), {'module': module, 'edge': edge})
    return list(found.values())

  def _paired(
      self,
      record: dict[str, Any],
      model: model_store.Model,
      side: tensor_analysis.Side,
      scan: tap_scan.Scan,
      run: dict[str, Any],
  ) -> tuple[tensor_analysis.Side, list[counterparts.Counterpart]]:
    """A LiteRT-LM side's points chosen, paired with PyTorch modules.

    The points the Session chose of the model (not its anchors), other
    than those placed by an anchor, each paired with the module of the
    PyTorch run's model that is the same semantic operator of the same
    layer (counterparts.find()), and placed there; the side's labels
    those of its skeleton's places then. None is paired when the two
    models are not of one architecture, which says so itself
    (tensor_analysis.layout()).

    Args:
      record: The Session's record.
      model: The LiteRT-LM model.
      side: The LiteRT-LM side analysis.
      scan: The scan of the model.
      run: The PyTorch run dictionary.

    Returns:
      A tuple of (updated Side, list of Counterpart pairs).

    Raises:
      ValueError, OSError, NotFound: The PyTorch model is gone, its
        config.json cannot be read, or no architecture defines it.
    """
    wanted = session_store.chosen_points(record).get(model.artifact, [])
    return self._pair(
        model,
        side,
        scan,
        self._models.get(run.get('artifact', '')),
        [p for p in side.points if p.id in set(wanted)],
    )

  def _pair(
      self,
      model: model_store.Model,
      side: tensor_analysis.Side,
      scan: tap_scan.Scan,
      pytorch: model_store.Model,
      chosen: list[tap_scan.Point],
  ) -> tuple[tensor_analysis.Side, list[counterparts.Counterpart]]:
    """_paired()'s pairing of the points `chosen` of an anchored side."""
    if side.architecture is None:
      return side, []
    known = next(
        (
            a
            for a in architecture_lib.architectures()
            if a.id == side.architecture
        ),
        None,
    )
    config = pytorch.config()
    definition, document = catalog.generate(config)
    if known is None or definition.id != side.architecture:
      return side, []
    ops = known.binding.get('ops', {})
    paired = counterparts.find(
        [point for point in chosen if point.id not in side.places],
        scan.points,
        ops,
        pytorch_binding.binding(definition.id).get('ops', {}),
        document,
    )
    places = dict(side.places)
    for each in paired:
      if each.module is not None:
        places[each.point_id] = ((each.layer, each.op),)
    skeleton = self._graphs.architecture(model)['skeleton']
    return (
        dataclasses.replace(
            side,
            places=places,
            labels=counterparts.place_labels(skeleton, ops),
        ),
        paired,
    )

  def counterparts(
      self,
      model: model_store.Model,
      pytorch: model_store.Model,
      point_ids: list[str],
  ) -> dict[str, Any]:
    """What the editor shows of points chosen beside a PyTorch run.

    Each point of a LiteRT-LM model chosen, as a Session of it and of the
    PyTorch model pairs it (observed()): with an anchor, with the PyTorch
    module of its semantic operator, or with nothing, and why. It never
    converts the model's graph: until it is, they are not known.

    Args:
        model: The LiteRT-LM model.
        pytorch: The PyTorch model of the other side.
        point_ids: The points chosen.

    Returns:
        `{status, text, points}`: `status` FOUND, PENDING (its graph is
        not converted) or NONE (no architecture pairs them), `text` why
        when not FOUND; per point `{id, paired, layer, place, label,
        module, edge, text}`, `paired` 'anchor', 'module', 'manual' (a
        pair made by hand of the two models names it) or '' (not
        paired), `text` as the user reads it; once FOUND,
        `pytorch_text`, what the PyTorch side captures for them beside
        its anchors, as its box in the editor says it.

    Raises:
        ValueError, OSError: A model cannot be read, or has no such point.
    """
    if self._graphs is None:
      return {
          'status': anchor_points.NONE,
          'text': 'This server keeps no graphs, so no pairing.',
          'points': [],
      }
    state = self._graphs.state(model.artifact)
    if state.status != model_graphs_lib.COMPLETED:
      return {
          'status': anchor_points.PENDING,
          'text': (
              f'Which PyTorch module each point of {model.name} is '
              "found in the model's graph, which is converted as the "
              'Model Server first turns on.'
          ),
          'points': [],
      }
    layers = self._graphs.layers(model)
    scan = self._scans.scan(model, layers.stems, layers.groups)
    chosen = [scan.point(point_id) for point_id in point_ids]
    side = self._anchored(
        model, tensor_analysis.Side(model.sha256, tuple(chosen))
    )
    if side.architecture is None:
      return {
          'status': anchor_points.NONE,
          'text': (
              f'{model.name} is of no architecture the server has: '
              'its points are not paired with PyTorch modules.'
          ),
          'points': [],
      }
    side, paired = self._pair(model, side, scan, pytorch, chosen)
    by_id = {found.point_id: found for found in paired}
    labels = side.labels
    by_hand = {}
    if self._mappings is not None:
      for pair in self._mappings.pairs(model.sha256, pytorch.sha256):
        by_hand.setdefault(pair.ref.point, pair)
    found = []
    for point in chosen:
      places = side.places.get(point.id, ())
      each = by_id.get(point.id)
      if each is not None and each.module is not None:
        side_text = (
            'what it returns' if each.module.edge == 'out' else 'what it reads'
        )
        found.append({
            'id': point.id,
            'paired': 'module',
            'layer': each.layer,
            'place': each.op,
            'label': each.label,
            'module': each.module.module,
            'edge': each.module.edge,
            'text': (
                f'Layer {each.layer} {each.label}: paired with '
                f'PyTorch {each.module.module}, {side_text}.'
            ),
        })
      elif places and each is None:
        layer, anchor = places[0]
        found.append({
            'id': point.id,
            'paired': 'anchor',
            'layer': layer,
            'place': anchor,
            'label': labels.get(anchor, anchor),
            'module': '',
            'edge': '',
            'text': (
                f'Layer {layer} {labels.get(anchor, anchor)}: an '
                'anchor, paired by the anchors.'
            ),
        })
      elif point.id in by_hand:
        pair = by_hand[point.id]
        named = f' ({pair.label})' if pair.label else ''
        found.append({
            'id': point.id,
            'paired': 'manual',
            'layer': point.layer,
            'place': tensor_analysis.manual_anchor(pair.id),
            'label': pair.label,
            'module': '',
            'edge': '',
            'text': f'Paired by hand with PyTorch {pair.target.point}{named}.',
        })
      else:
        found.append({
            'id': point.id,
            'paired': '',
            'layer': point.layer,
            'place': '',
            'label': '',
            'module': '',
            'edge': '',
            'text': each.reason if each is not None else '',
        })
    return {
        'status': anchor_points.FOUND,
        'text': '',
        'points': found,
        'pytorch_text': _pytorch_text(model, pytorch, found),
    }

  def _anchored(
      self, model: model_store.Model, side: tensor_analysis.Side
  ) -> tensor_analysis.Side:
    """A LiteRT-LM side, its points placed by the anchors they are."""
    found = self._graphs.architecture(model)
    known = found.get('architecture')
    if known is None:
      return side
    labels = tensor_analysis.anchor_labels(found['skeleton'])
    return dataclasses.replace(
        side,
        architecture=known['id'],
        places=tensor_analysis.anchor_places(
            found['bindings'], {point.id for point in side.points}
        ),
        labels=labels,
        anchors=tuple(labels),
    )

  def _op_placed(
      self,
      model: model_store.Model,
      side: tensor_analysis.Side,
      scan: tap_scan.Scan,
      ops: list[str],
  ) -> tensor_analysis.Side:
    """An anchored LiteRT-LM side, the operators' points placed too.

    Each point observed that is an operator chosen's in its layer
    (layer_ops.litert_points()) is placed at (layer, operator); the
    side's labels are then its skeleton's places'.

    Args:
      model: The model.
      side: The side to place operator points on.
      scan: The scan of the model.
      ops: The list of chosen operator names.

    Returns:
      A replacement Side with operator points placed and labels updated.
    """
    if not ops or side.architecture is None:
      return side
    rules = litert_rules(side.architecture)
    observed = {point.id for point in side.points}
    found = layer_ops.litert_points(
        scan.points,
        rules,
        ops,
        {point.signature for point in side.points},
    )
    places = dict(side.places)
    for op, by_signature in found.items():
      for by_layer in by_signature.values():
        for layer, point in by_layer.items():
          if point.id in observed:
            places[point.id] = (
                *places.get(point.id, ()),
                (layer, op),
            )
    skeleton = self._graphs.architecture(model)['skeleton']
    return dataclasses.replace(
        side,
        places=places,
        labels=counterparts.place_labels(skeleton, rules),
    )

  def view(self, record: dict[str, Any]) -> dict[str, Any]:
    """A record's preparation, as the listing shows it.

    It reads no model: a model none of whose points needs a copy shows
    READY once its scan is kept (tap_scans), NEEDED until then. A
    record whose anchors are not found yet is NEEDED, though it may turn
    out to observe nothing.

    Args:
      record: The Session's record.

    Returns:
      `{state, text}`: NONE when it observes no tensors, RUNNING while a
      preparation runs, READY once every copy is, NEEDED otherwise;
      `text` says so to the user.
    """
    points = session_store.tap_points(record)
    finding = session_store.captures_anchors(
        record
    ) and not session_store.anchors_found(record)
    if not points and not finding:
      return {'state': NONE, 'text': ''}
    with self._lock:
      running = list(self._running.get(record['id'], []))
    if running:
      progress = self._jobs.brief(running[-1])['progress']
      text = 'Preparing the tensor capture'
      return {
          'state': RUNNING,
          'text': f'{text} · {progress}' if progress else text,
      }
    if not finding and self._ready(points):
      return {'state': READY, 'text': 'The tensor capture is prepared.'}
    if finding:
      return {'state': NEEDED, 'text': ANCHORS_NEEDED}
    return {
        'state': NEEDED,
        'text': (
            'The tensor capture is prepared when the Model Server turns on.'
        ),
    }

  def _ready(self, points: dict[str, list[str]]) -> bool:
    """Whether every model of a record's points loads without a write."""
    for artifact, ids in points.items():
      try:
        model = self._models.get(artifact)
      except errors.NotFound:
        return False
      scan = self._scans.kept(model)
      try:
        if scan is not None and not needs_copy(scan, ids):
          continue
      except ValueError:
        return False
      if self.find(model, ids) is None:
        return False
    return True

  # ---- Preparing.

  def prepare_start(
      self, record: dict[str, Any], job_id: str, aborted: Callable[[], bool]
  ) -> dict[str, Any] | None:
    """Prepares the copies a starting Session's runs load, in its job.

    Called on the start's thread before any runner launches, once its
    models' graphs are converted; its progress goes to the start's job,
    as what the listing shows of the preparation. The anchors it
    captures are found first, and kept in its record (find_anchors()).
    The copies are held from now until loaded() is called with the job:
    none is deleted while the sides load it.

    Args:
        record: The Session's record.
        job_id: The start's job.
        aborted: Whether the start was turned off meanwhile.

    Returns:
        The record as prepared, with the anchors found; None if the start
        was turned off before every copy was written.

    Raises:
        ValueError: A copy cannot be written, or the anchors cannot be
          found; the text starts with FAILED_TEXT.
        NotFound: A model is not registered.
    """
    try:
      record = self.find_anchors(record, job_id)
    except (ValueError, OSError) as error:
      raise ValueError(FAILED_TEXT + errors.message(error)) from error
    points = session_store.tap_points(record)
    if not points:
      return record
    keys = self._keys(record)
    with self._lock:
      self._using.update(keys)
      self._starts[job_id] = keys
    try:
      self._prepare_all(record['id'], points, job_id, aborted)
    except _StoppedError:
      return None
    except (ValueError, OSError) as error:
      raise ValueError(FAILED_TEXT + errors.message(error)) from error
    return record

  def find_anchors(
      self, record: dict[str, Any], job_id: str | None = None
  ) -> dict[str, Any]:
    """Finds the anchors a record captures, and keeps them in it.

    Of each LiteRT-LM model of a side that captures its anchors
    (session_store.anchor_sides()), as anchor_points.find() says;
    a model whose graph is not converted has none yet, and is found
    again as the record is next prepared. A record that ran keeps those
    it ran with (SessionStore.set_anchor_points()).

    Args:
        record: The Session's record.
        job_id: The job told what was found, if any.

    Returns:
        The record, with the anchors found; the record as it is when it
        captures none.

    Raises:
        NotFound: A model is not registered.
        ValueError, OSError: A model cannot be scanned.
    """
    if not session_store.captures_anchors(record):
      return record
    found, ops = {}, {}
    chosen = session_store.tap_ops(record)
    pending = False
    for artifact in sorted(session_store.anchor_artifacts(record)):
      model = self._models.get(artifact)
      anchors = anchor_points.find(self._graphs, self._scans, model)
      if job_id is not None:
        self._jobs.emit(
            job_id,
            'progress',
            message=f'{model.name}: {_anchor_progress(anchors)}',
        )
      pending = pending or anchors.status == anchor_points.PENDING
      found[artifact] = list(anchors.point_ids)
      if chosen and anchors.status == anchor_points.FOUND:
        ops[artifact] = self.op_points(model, chosen)
        if job_id is not None and ops[artifact]:
          count = len(ops[artifact])
          self._jobs.emit(
              job_id,
              'progress',
              message=(
                  f'{model.name}: the operators chosen inside the '
                  f'layers are {count} tensor output'
                  f'{"" if count == 1 else "s"}'
              ),
          )
    if pending:
      return record
    return self._sessions.set_anchor_points(record['id'], found, ops)

  def op_points(
      self, model: model_store.Model, ops: Iterable[str]
  ) -> list[str]:
    """The points of operators inside the layers of a LiteRT-LM model.

    Each operator's point in every layer of every signature whose
    anchors are captured (anchor_points: not one that runs only with
    speculative decoding), as its graph's architecture binds it
    (layer_ops.litert_points()), in the model's order; [] for a model
    of no architecture. Its graph must be converted.

    Args:
      model: The LiteRT-LM model.
      ops: The operator names to find points for.

    Returns:
      A list of point IDs in the model's scan order.
    """
    known = self._graphs.architecture(model).get('architecture')
    if known is None:
      return []
    layers = self._graphs.layers(model)
    scan = self._scans.scan(model, layers.stems, layers.groups)
    found = layer_ops.litert_points(
        scan.points,
        litert_rules(known['id']),
        ops,
        captured_signatures(scan),
    )
    ids = set()
    for by_signature in found.values():
      for by_layer in by_signature.values():
        for point in by_layer.values():
          ids.add(point.id)
    return [point.id for point in scan.points if point.id in ids]

  def _prepare_all(
      self,
      record_id: str,
      points: dict[str, list[str]],
      job_id: str,
      aborted: Callable[[], bool],
  ) -> None:
    """Writes each copy a record's points need; the record shows it."""
    with self._shown(record_id, job_id):
      for artifact, ids in points.items():
        self._prepare(job_id, self._models.get(artifact), ids, aborted)

  @contextlib.contextmanager
  def _shown(self, record_id: str, job_id: str) -> Iterator[None]:
    """Shows a record as being prepared by a job meanwhile (view())."""
    with self._lock:
      self._running.setdefault(record_id, []).append(job_id)
    try:
      yield
    finally:
      with self._lock:
        running = self._running[record_id]
        running.remove(job_id)
        if not running:
          del self._running[record_id]

  def _prepare(
      self,
      job_id: str,
      model: model_store.Model,
      point_ids: list[str],
      aborted: Callable[[], bool],
  ) -> None:
    """Writes the copy of a model that observes points, if it needs one.

    The copy is held meanwhile (_hold()): collect() deletes it, if no
    Session uses it, only once this ends.

    Args:
      job_id: The ID of the preparation job.
      model: The model to prepare.
      point_ids: The points to observe in the copy.
      aborted: A callback returning True if preparation should abort.

    Raises:
      _StoppedError: `aborted` said so between two steps.
    """
    name = key(model.sha256, point_ids)
    self._hold([name])
    try:
      self._prepare_held(job_id, model, point_ids, aborted)
    finally:
      self._release([name])

  def _prepare_held(
      self,
      job_id: str,
      model: model_store.Model,
      point_ids: list[str],
      aborted: Callable[[], bool],
  ) -> None:
    """_prepare(), with the copy held."""

    def progress(message: str) -> None:
      if aborted():
        raise _StoppedError()
      self._jobs.emit(job_id, 'progress', message=f'{model.name}: {message}')

    if self._scans.kept(model) is None:
      progress('Reading the model')
    scan = self._scans.scan(model)
    tap_model.check_points(scan, point_ids)
    if not needs_copy(scan, point_ids):
      progress('each tensor chosen is an output already; it loads as is')
      return
    if self.find(model, point_ids) is not None:
      progress('its copy is prepared already')
      return
    if not self._writing.acquire(blocking=False):
      progress(WAITING)
      while not self._writing.acquire(timeout=_WAIT_SECONDS):
        if aborted():
          raise _StoppedError()
    try:
      if self.find(model, point_ids) is not None:
        progress('its copy is prepared already')
        return
      self._write(model, scan, point_ids, progress)
    finally:
      self._writing.release()
    progress('its copy is prepared')

  def _write(
      self,
      model: model_store.Model,
      scan: tap_scan.Scan,
      point_ids: list[str],
      progress: Callable[[str], None],
  ) -> None:
    """Writes, checks and keeps a copy; called with _writing held."""
    free = shutil.disk_usage(self._directory).free
    if free < model.size + _FREE_MARGIN_BYTES:
      raise ValueError(
          f'There is not enough disk space for a copy of {model.name}: '
          f'it takes {size_text(model.size)}, and '
          f'{size_text(free)} is free.'
      )
    name = key(model.sha256, point_ids)
    pending = self._directory / (workspace.PENDING + uuid.uuid4().hex)
    pending.mkdir()
    try:
      started = time.monotonic()
      copy = tap_model.write(
          model.path,
          scan,
          point_ids,
          pending / _MODEL,
          model.sha256,
          progress,
      )
      stat = (pending / _MODEL).stat()
      json_file.write_json(
          pending / _MANIFEST,
          _manifest(model, name, copy, stat, started),
      )
      final = self._directory / name
      # One left of a copy no longer valid (find() said so).
      shutil.rmtree(final, ignore_errors=True)
      pending.rename(final)
    finally:
      shutil.rmtree(pending, ignore_errors=True)


def _manifest(
    model: model_store.Model,
    name: str,
    copy: tap_model.Copy,
    stat: os.stat_result,
    started: float,
) -> dict[str, Any]:
  """What prepared.json keeps of a copy."""
  count = len(copy.observed)
  return {
      'format_version': tap_model.FORMAT_VERSION,
      'identity': tap_model.identity(),
      'key': name,
      'name': (
          f'{model.name} (tapped, {count} point{"" if count == 1 else "s"})'
      ),
      'sha256': copy.sha256,
      'size': stat.st_size,
      'mtime_ns': stat.st_mtime_ns,
      'source': {
          'artifact': model.artifact,
          'sha256': model.sha256,
          'name': model.name,
          'size': model.size,
          'section_sha256': copy.source_sha256,
      },
      'points': [
          {
              'id': item.point.id,
              'signature': item.point.signature,
              'subgraph': item.point.subgraph,
              'op': item.point.op,
              'output': item.point.output,
              'tensor': item.point.tensor,
              'tensor_name': item.point.tensor_name,
              'shape': list(item.point.shape),
              'dtype': item.point.dtype,
              'context_axes': list(item.point.context_axes),
              'output_name': item.output_name,
              'capture_key': item.capture_key(),
              'added': item.added,
          }
          for item in copy.observed
      ],
      'checked': copy.checked,
      'created_at': clock.now(),
      'seconds': round(time.monotonic() - started, 1),
  }


@dataclasses.dataclass(frozen=True)
class _ByHand:
  """Where a Session places a pair made by hand.

  Attributes:
      pair: The pair, as the Session's sides are.
      place: Its place (layer, manual anchor); None if not placed.
      label: Its label: the user's, or what its tensors are.
      points: Per side that observes its tensor, that point's id.
      text: Why it is not placed, as the user reads it; '' if it is.
  """

  pair: mappings_lib.Pair
  place: tuple[int | None, str] | None
  label: str
  points: dict[str, str]
  text: str


def _pytorch_text(
    model: model_store.Model,
    pytorch: model_store.Model,
    found: list[dict[str, Any]],
) -> str:
  """What a PyTorch side captures for the points chosen beside it."""
  modules = sum(1 for point in found if point['paired'] in ('module', 'manual'))
  unpaired = sum(1 for point in found if not point['paired'])
  text = (
      f'Beside its anchors, {pytorch.name} also captures '
      f'{modules} module{"" if modules == 1 else "s"} paired with the '
      f'outputs chosen of {model.name}'
  )
  if unpaired:
    text += (
        f'; {unpaired} output{"" if unpaired == 1 else "s"} chosen '
        f'{"has" if unpaired == 1 else "have"} no module paired with '
        f'{"it" if unpaired == 1 else "them"}'
    )
  return text + '.'


def _label(pair: mappings_lib.Pair, points: dict[str, tap_scan.Point]) -> str:
  """A pair's label: the user's, else its two tensors as the user knows them.

  A LiteRT-LM tensor's scope and name, a PyTorch module's path in its layer
  and what of it.

  Args:
    pair: The pair mapping.
    points: A mapping from side name to Point.

  Returns:
    The formatted label string for the pair.
  """
  if pair.label:
    return pair.label

  def short(name: str) -> str:
    point = points.get(name)
    tensor = pair.ref if name == 'ref' else pair.target
    if point is None:
      return tensor.point
    if point.tensor < 0:
      module = point.tensor_name.split(' (')[0]
      inside = re.sub(r'^.*?\blayers\.\d+\.?', '', module) or 'layer'
      return f'{inside} {"input" if point.id.endswith(":in") else "output"}'
    return tensor_analysis.label(point).split(' · ')[0]

  return f'{short("ref")} ↔ {short("target")}'


def litert_rules(architecture_id: str) -> dict[str, Any]:
  """The LiteRT-LM binding's `ops` of an architecture; {} if it has none."""
  return next(
      (
          known.binding.get('ops', {})
          for known in architecture_lib.architectures()
          if known.id == architecture_id
      ),
      {},
  )


def captured_signatures(scan: tap_scan.Scan) -> list[str]:
  """The signatures a model's anchors and operators are captured in."""
  return [
      signature.name
      for signature in scan.signatures
      if signature.runs != tap_scan.RUNS_SPECULATIVE
  ]


def pytorch_dtype(
    config: dict[str, Any], run: dict[str, Any] | None = None
) -> str:
  """The dtype a PyTorch run stores: its option's, else its model's; ''."""
  return (run or {}).get('dtype') or str(
      catalog.text_config(config).get('dtype')
      or config.get('dtype')
      or config.get('torch_dtype')
      or ''
  )


def _pytorch_side(
    model: model_store.Model,
    run: dict[str, Any],
    paired: Iterable[counterparts.Counterpart] = (),
    ops: Iterable[str] = (),
) -> tensor_analysis.Side:
  """What a PyTorch run observes: every layer's anchors, or nothing.

  And beside the anchors each module paired with a LiteRT-LM point of
  the other side (`paired`), and each operator chosen inside the layers
  (`ops`) in every layer that has it (layer_ops.pytorch_modules()), its
  places labelled as its skeleton's are.

  Args:
    model: The PyTorch model.
    run: The run dictionary.
    paired: Counterpart pairings from the other side.
    ops: Chosen operator names.

  Returns:
    A Side object representing the observed PyTorch points and anchors.

  Raises:
    ValueError, OSError: Its config.json cannot be read, or no
      architecture with a PyTorch binding defines its model.
  """
  if run.get('observe') != options_lib.OBSERVE_ANCHORS:
    return tensor_analysis.Side(
        model.sha256, (), runtime=tensor_analysis.PYTORCH
    )
  config = model.config()
  dtype = pytorch_dtype(config, run)
  modules = [found for found in paired if found.module is not None]
  ops = list(ops)
  labels = None
  if modules or ops:
    definition, document = catalog.generate(config)
    rules = pytorch_binding.binding(definition.id).get('ops', {})
    labels = counterparts.place_labels(document, rules)
    for op in ops:
      label = labels.get(op, op)
      found, _ = layer_ops.pytorch_modules(document, rules, op, label)
      modules += [
          counterparts.Counterpart('', layer, op, label, module)
          for layer, module in found.items()
      ]
  return tensor_analysis.pytorch_side(
      model.sha256, pytorch_binding.anchors(config), dtype, modules, labels
  )


def _anchor_progress(anchors: anchor_points.Anchors) -> str:
  """What finding a model's anchors found, as a progress line."""
  if anchors.status != anchor_points.FOUND:
    return anchors.text
  count = len(anchors.point_ids)
  return (
      f'its anchors are {count} tensor output{"" if count == 1 else "s"}: '
      + '; '.join(s.text.rstrip('.') for s in anchors.signatures)
  )
