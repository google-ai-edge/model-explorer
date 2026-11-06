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

"""The graphs of the models, converted once per model and kept on disk.

A model's graph is that of its text model's prefill and decode section, one
graph per signature, as Model Explorer's converter writes it, slimmed
(graph/model_graph). It is converted only as a Session is initialized: in
its Model Server's initialize job, first, before the tensor capture is
prepared and before that Session's runners launch (prepare_start()). Nothing
else converts one: what reads a graph, or the layers it names, says where
it is when it is not there. A model whose graph is kept is not converted
again. Another Session's runners may be running on this computer meanwhile:
the rule orders a Session's own start, not the others', and the converter's
memory (about 0.7 GB for Gemma 4 E2B, 0.9 GB for E4B) is taken beside
theirs.

The converter runs in a child process (graph/graph_converter), one
conversion at a time, for at most `timeout` seconds; a start that waits for
another's says for which, and whose. A start that is turned off stops its
conversion, and its child is killed; so does close(), which waits for the
child to end. A Session cannot start without its models' graphs: a
conversion that fails (the converter failed, crashed, was killed, took too
long, or is not available on this platform) fails the start, as any failed
start does: its job fails with the reason, the Session ends with no turn
(model_server), and no runner launches.

A graph is kept in the workspace, `graphs/<sha256>/`, by the SHA-256 of the
model file: `manifest.json` and one `signature-<i>.json.gz` per signature,
the slimmed graphs the child wrote. The manifest names what wrote them
(identity(): the converter, what it was given, a digest of the slimming code
and this layout); a graph another wrote is converted anew. A graph is kept
as long as its model (collect()): small, about 0.1 MB per signature, it goes
when the model is deleted (model_store.ModelStore.collect()). What the server
says of a graph beyond, each output's point and its facts, the operators
regrouped by their layers, the layers' stems and the architecture, is worked
out as the graph is read, and kept in memory: it is always that of the
server's code of now. For Gemma 4 E2B, reading its four graphs takes about
0.04 s, annotating its decode graph 0.02 s, and the scan by its layers 0.8 s
once, which the analyses of its tensors need anyway.

A conversion that failed is kept too, in the same directory without a
manifest: `failure.json` (the reason and the identity it failed with) and
the tail of the converter's log, `converter.log`. Nothing converts it again
on its own, and the Session whose start it failed has ended, as any Session
whose start failed; a later start, of another Session of the model, tries it
once more; a start that waited for the conversion that failed fails with it.
A new identity converts anew.

What the graph knows of the model is the source of its layers: the stem of
each signature's layer scopes (model_graph.layer_stems()), by which the
intermediate tensors' analysis places each point in its layer; and, of
each operator that has an output whose name names no layer, the layer the
graph as it is served groups it in (model_graph.layer_groups()), which is
that output's layer: the analysis and the graph place every point alike.
That dependency goes through layers() alone (prepared_models.observed()
calls it): a started Session's models always have their graphs, and what a
model with no graph has, an old Session's recorded before graphs were, is
decided there. Its architecture is the skeleton the server has of the
model's family and size (architectures/catalog), and each anchor's place in
each signature's graph as it is served (regrouped), by the family's
LiteRT-LM binding (graph/architecture); it is bound when first asked for,
and kept in memory by the graph and the architectures' and bindings' files.

A PyTorch model's graph is not converted: it is the execution graph its
runner exports of a turn it observed on the CPU (pytorch-collector's
collect() and to_model_explorer(), the runner's execution_graph), the
graph of the first forward pass of each signature, prefill and decode, in
the turn's capture (capture_index.EXECUTION_GRAPH). Once the turn is
recorded, its Model Server hands the file here (keep_exported()), and the
graph of each signature the model has none of is kept, in the same
directory, by the SHA-256 the runner names the model directory by: the
first turn so observed gives both, a later one only one the first did not
run; a turn of a model whose graphs are kept leaves them as they are. The
manifest names the runner's format and this layout (pytorch_identity()) and
where each graph came from (Session, turn, side); `modules.json` beside the
graphs holds each module's class. A model none of whose turns exported its
graph (its runs observed nothing, or ran on the GPU, where collect()
captures nothing) has none, and says how to get one (NO_PYTORCH_GRAPH). As
it is read, each signature's graph is folded as the Graph Diff draws it
(pytorch_graph.fold(), by the architecture's PyTorch binding: each module
LiteRT-LM runs as one operator is one node), its layers named by the
module path, `layers/<N>` (model_graph.layer_stems()); its architecture is
the skeleton its config.json makes, bound to those graphs by the same
binding (architecture.bind()), each anchor the PyTorch runner's point of
it; and the anchors' outputs carry their points, as do the tensors of the
modules of the semantic operators a Session may pair with LiteRT-LM points
(pytorch_binding.op_modules()) and of every other folded node's module,
which a pair made by hand may name (session/mappings), what it returns and
what it reads; every other output, an ATen operator's, why it is not
observed (pytorch_graph.annotate()). Gemma 4 E2B's two graphs take
about 0.04 s to fold.
"""

from collections.abc import Callable, Collection
import dataclasses
import gzip
import json
import pathlib
import shutil
import threading
import time
from typing import Any
import uuid

from absl import logging
from model_debugger.server.analysis import tensor_analysis
from model_debugger.server.architectures import catalog
from model_debugger.server.graph import architecture as architecture_lib
from model_debugger.server.graph import graph_converter
from model_debugger.server.graph import model_graph
from model_debugger.server.graph import pytorch_binding
from model_debugger.server.graph import pytorch_graph
from model_debugger.server.session import jobs as jobs_lib
from model_debugger.server.session import model_store
from model_debugger.server.session import tap_scans
from model_debugger.server.session import workspace
from model_debugger.server.tap import tap_scan
from model_debugger.server.util import clock
from model_debugger.server.util import errors
from model_debugger.server.util import json_file

IDLE = 'idle'
CONVERTING = 'converting'
COMPLETED = 'completed'
FAILED = 'failed'
# A PyTorch model none of whose turns exported its graph.
UNAVAILABLE = 'unavailable'
NO_PYTORCH_GRAPH = (
    'The execution graph of this PyTorch model is not kept yet: a turn of a '
    'PyTorch run on the CPU that observes the anchors exports it (its '
    "runner's pytorch-collector captures on the CPU only). Run a Session "
    'with such a side, one turn.'
)
# Why an output of a PyTorch graph that is no anchor has no point.
NOT_AN_ANCHOR = (
    'The PyTorch runner captures modules, not ATen operators: the four '
    'anchors of each decoder layer, and what a module LiteRT-LM runs as one '
    'operator (a folded node) returns or reads; this tensor is none of them.'
)
# The seconds one conversion may take: E4B's takes about 5 s.
DEFAULT_TIMEOUT_SECONDS = 600.0
# The layout of a kept graph: its manifest and files.
_LAYOUT = 2
_MANIFEST = 'manifest.json'
_FAILURE = 'failure.json'
_LOG = 'converter.log'
# The last bytes of a failed conversion's log that are kept.
_LOG_KEPT = 64 * 1024
_CONVERTED = 'converted.json'
# The layout of a PyTorch model's kept graph; its modules' classes.
_PYTORCH_LAYOUT = 1
_MODULES = 'modules.json'
# How often a conversion that waits for another looks whether it is stopped.
_WAIT_SECONDS = 0.25
# How long close() waits for a conversion to end once told to stop; it
# stops within _WAIT_SECONDS, its child is killed then.
_CLOSE_SECONDS = 30.0


def identity() -> str:
  """What names the graphs kept: the converter, its slimming, the layout."""
  return f'{graph_converter.identity()}, layout {_LAYOUT}'


def pytorch_identity() -> str:
  """What names a PyTorch model's kept graphs: the runner's format, layout."""
  return (
      "the PyTorch runner's execution graph, format "
      f'{pytorch_graph.FORMAT}, layout {_PYTORCH_LAYOUT}'
  )


def _identity(model: model_store.Model) -> str:
  if model.runtime == model_store.PYTORCH:
    return pytorch_identity()
  return identity()


@dataclasses.dataclass(frozen=True)
class State:
  """Where the graph of a model is.

  Attributes:
      status: IDLE (not converted), CONVERTING, COMPLETED, FAILED, or
        UNAVAILABLE (no graph is converted for the model).
      error: Why it FAILED.
      text: Why it is UNAVAILABLE.
      identity: What it FAILED with (identity()); '' otherwise.
      waiting: While CONVERTING, what it waits for, if for another
        conversion: whose graph, for which Session.
      manifest: The kept graph's manifest, once COMPLETED.
  """

  status: str
  error: str = ''
  text: str = ''
  identity: str = ''
  waiting: str = ''
  manifest: dict[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class Layers:
  """The layers a model's graph names its points by, or why it names none.

  Attributes:
      stems: Per signature that has layers, their stem (model_graph.LayerStems);
        None when the graph is not there.
      groups: Per signature that has layers, the layer of each operator whose
        outputs' names may name none (model_graph.layer_groups()); None when the
        graph is not there.
      reason: Why the graph is not there, as the end of a sentence about it ('is
        being converted', say); '' when it is.
  """

  stems: dict[str, str] | None
  groups: dict[str, dict[int, int]] | None = None
  reason: str = ''


@dataclasses.dataclass(frozen=True)
class Facts:
  """What the server says of a kept graph, worked out as it is read.

  Attributes:
      layer_stems: The stems of each signature's layers, or why none.
      layer_groups: Per signature that has layers, the layer of each operator
        whose outputs' names may name none, as the graph served groups it
        (model_graph.layer_groups()).
      layers: Per signature, how many layers its points are in.
      layer_ids: Per signature, the layers it has operators of, as the graph
        served groups them (model_graph.operator_layers()): the layers it
        computes.
  """

  layer_stems: model_graph.LayerStems
  layer_groups: dict[str, dict[int, int]]
  layers: dict[str, int]
  layer_ids: dict[str, tuple[int, ...]]


class ModelGraphs:
  """Every model's graph, by its SHA-256, in the workspace."""

  def __init__(
      self,
      workspace_: workspace.Workspace,
      models: model_store.ModelStore,
      scans: tap_scans.TapScans,
      jobs: jobs_lib.Jobs,
      timeout: float = DEFAULT_TIMEOUT_SECONDS,
      convert: graph_converter.Converter = graph_converter.convert,
  ) -> None:
    """Holds the graphs in the workspace; converts nothing yet.

    Args:
        workspace_: The workspace.
        models: The registered models.
        scans: The scans of their tensors.
        jobs: The jobs a start's conversion reports to.
        timeout: The seconds one conversion may take.
        convert: Converts a TFLite model's signatures to their graphs
          (graph_converter.convert()).
    """
    self._directory = workspace_.graphs
    self._models = models
    self._scans = scans
    self._jobs = jobs
    self._timeout = timeout
    self._convert = convert
    self._lock = threading.Lock()
    # Held while a graph is converted: one at a time.
    self._converting = threading.Lock()
    # What the conversion under way is, for those that wait: whose
    # graph, for which Session; and that model's SHA-256.
    self._holder = ''
    self._holder_sha256 = ''
    # SHA-256 -> the state of a conversion under way or waiting.
    self._states: dict[str, State] = {}
    # SHA-256 -> a failure of this server run, kept on disk too, and
    # when it failed (time.monotonic()).
    self._failures: dict[str, State] = {}
    self._failed_at: dict[str, float] = {}
    # (SHA-256, created_at) -> the facts of the kept graph.
    self._facts: dict[tuple[str, str], Facts] = {}
    # (SHA-256, created_at, architectures' identity) -> its answer.
    self._architectures: dict[tuple[str, str, str], dict[str, Any]] = {}
    # (SHA-256, created_at) -> a PyTorch model's graphs, folded.
    self._folded: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    # Held while a PyTorch model's exported graph is kept: one at a time.
    self._keeping = threading.Lock()
    # SHA-256 -> why the last graph a turn exported could not be kept.
    self._keep_failures: dict[str, str] = {}
    self._closed = False

  # ---- What is kept.

  def state(self, artifact: str) -> State:
    """Where the graph of a model is now.

    Args:
      artifact: Identifier or path of the model artifact.

    Returns:
      Current graph state for the model.

    Raises:
      NotFound: No such model.
    """
    return self._state(self._models.get(artifact))

  def _state(self, model: model_store.Model) -> State:
    """Returns the current graph state of a resolved model."""
    if model.runtime == model_store.PYTORCH:
      manifest = self._manifest(model)
      if manifest is not None:
        return State(COMPLETED, manifest=manifest)
      with self._lock:
        failed = self._keep_failures.get(model.sha256)
      text = NO_PYTORCH_GRAPH
      if failed:
        text += f' The last one a turn exported was not kept: {failed}'
      return State(UNAVAILABLE, text=text)
    with self._lock:
      running = self._states.get(model.sha256)
      failed = self._failures.get(model.sha256)
    if running is not None:
      return running
    manifest = self._manifest(model)
    if manifest is not None:
      return State(COMPLETED, manifest=manifest)
    if failed is not None and failed.identity == identity():
      return failed
    return self._failure(model) or State(IDLE)

  def _manifest(self, model: model_store.Model) -> dict[str, Any] | None:
    """The manifest of a model's kept graph, if it is whole and current."""
    directory = self._directory / model.sha256
    try:
      manifest = json_file.read_json(directory / _MANIFEST)
    except (OSError, ValueError):
      return None
    if manifest.get('identity') != _identity(model):
      return None
    for signature in manifest.get('signatures', []):
      if not (directory / signature['file']).is_file():
        return None
    return manifest

  def _failure(self, model: model_store.Model) -> State | None:
    """A failure an earlier server run kept, if of this identity."""
    try:
      kept = json_file.read_json(self._directory / model.sha256 / _FAILURE)
    except (OSError, ValueError):
      return None
    if kept.get('identity') != identity():
      return None
    return State(FAILED, error=kept['error'], identity=kept['identity'])

  def layers(self, model: model_store.Model) -> Layers:
    """The layers of a model's points, as its graph names them.

    The one place the intermediate tensors' layers depend on the graph.
    A started Session's models have their graphs; a model whose graph
    is not kept (an old Session's, recorded before graphs were) has
    none (stems None), and the reason says where its graph is. It never
    converts.

    Args:
      model: The model to get layers for.

    Returns:
      Layers describing layer stems and layer groups, or reason if missing.
    """
    state = self._state(model)
    if state.status == COMPLETED:
      facts = self._read(model, state.manifest)
      return Layers(facts.layer_stems.stems, facts.layer_groups)
    return Layers(
        None,
        reason={
            IDLE: (
                'is not converted: it is converted when a Session of '
                'the model is initialized'
            ),
            CONVERTING: 'is being converted; ask again once it is',
            FAILED: f'could not be converted: {state.error}',
            UNAVAILABLE: (
                'is not kept: a turn of a PyTorch run on the CPU that '
                'observes the anchors exports it'
            ),
        }[state.status],
    )

  def facts(self, model: model_store.Model) -> Facts:
    """What the server says of a model's kept graph.

    Args:
      model: The model to get facts for.

    Returns:
      Facts describing stems, groups, counts, and operator layers.

    Raises:
      Conflict: The model's graph is not converted.
    """
    return self._read(model, self._require(model))

  def _read(self, model: model_store.Model, manifest: dict[str, Any]) -> Facts:
    """The facts of a kept graph: worked out once, then kept in memory."""
    key = model.sha256, manifest['created_at']
    with self._lock:
      kept = self._facts.get(key)
    if kept is not None:
      return kept
    graphs = self._graphs(model, manifest)
    if model.runtime == model_store.PYTORCH:
      found = _pytorch_facts(graphs)
      with self._lock:
        self._facts[key] = found
      return found
    stems = model_graph.layer_stems(graphs)
    served = {
        name: model_graph.regroup(graph, stems.stems.get(name))
        for name, graph in graphs.items()
    }
    groups = {
        name: model_graph.layer_groups(graph, stem)
        for name, graph in served.items()
        if (stem := stems.stems.get(name))
    }
    points = self._scans.scan(model, stems.stems, groups).points
    found = Facts(
        stems,
        groups,
        {
            entry['name']: len({
                p.layer
                for p in points
                if p.signature == entry['name'] and p.layer is not None
            })
            for entry in manifest['signatures']
        },
        {
            name: model_graph.operator_layers(graph, stems.stems.get(name))
            for name, graph in served.items()
        },
    )
    with self._lock:
      self._facts[key] = found
    return found

  def _points(
      self, model: model_store.Model, facts: Facts
  ) -> tuple[tap_scan.Point, ...]:
    """The model's points, in the layers its graph places them in."""
    return self._scans.scan(
        model, facts.layer_stems.stems, facts.layer_groups
    ).points

  def _graphs(
      self, model: model_store.Model, manifest: dict[str, Any]
  ) -> dict[str, dict[str, Any]]:
    """Every signature's kept graph, as the child wrote it.

    A PyTorch model's, as the runner exported it, folded: worked out
    once, then kept in memory.

    Args:
      model: The model.
      manifest: Graph manifest dictionary.

    Returns:
      Mapping from signature name to graph dictionary.
    """
    if model.runtime != model_store.PYTORCH:
      return {
          entry['name']: self._kept(model, entry)
          for entry in manifest['signatures']
      }
    key = model.sha256, manifest['created_at']
    with self._lock:
      kept = self._folded.get(key)
    if kept is not None:
      return kept
    modules = json_file.read_json(self._directory / model.sha256 / _MODULES)
    rules = _pytorch_binding(model)['fold']
    found = {
        entry['name']: pytorch_graph.fold(
            self._kept(model, entry), modules, rules
        )
        for entry in manifest['signatures']
    }
    with self._lock:
      self._folded[key] = found
    return found

  def _kept(
      self, model: model_store.Model, entry: dict[str, Any]
  ) -> dict[str, Any]:
    path = self._directory / model.sha256 / entry['file']
    with gzip.open(path, 'rt', encoding='utf-8') as file:
      return json.load(file)

  def graph(self, model: model_store.Model, signature: str) -> dict[str, Any]:
    """A signature's graph, each operator output with its point's facts.

    Regrouped and annotated now (model_graph.regroup(), annotate()), by
    the graph's layers and the points as the model's scan lists them
    with those layers: each operator is grouped in the layer its point
    is in, but for one several layers share; a point whose name names no
    layer is in its operator's group's.

    Args:
      model: The model.
      signature: Signature name.

    Returns:
      Annotated graph dictionary for the requested signature.

    Raises:
      Conflict: The model's graph is not converted.
      NotFound: It has no such signature.
    """
    manifest = self._require(model)
    for entry in manifest['signatures']:
      if entry['name'] == signature and (model.runtime == model_store.PYTORCH):
        bindings = self.architecture(model)['bindings']
        graph = self._graphs(model, manifest)[signature]
        stem = self._read(model, manifest).layer_stems.stems.get(signature)
        return pytorch_graph.annotate(
            graph,
            bindings.get(signature, []),
            NOT_AN_ANCHOR,
            _module_points(model, signature, graph, stem),
        )
      if entry['name'] == signature:
        facts = self._read(model, manifest)
        return model_graph.annotate(
            model_graph.regroup(
                self._kept(model, entry),
                facts.layer_stems.stems.get(signature),
            ),
            self._points(model, facts),
            signature,
        )
    raise errors.NotFound(
        f'The model has no signature {signature}; it has '
        f'{", ".join(e["name"] for e in manifest["signatures"])}.'
    )

  def tensor(
      self,
      model: model_store.Model,
      signature: str,
      node: str,
      output: str,
  ) -> dict[str, str]:
    """One operator output of a signature's graph, as graph() serves it.

    Args:
      model: The model.
      signature: Signature name.
      node: Operator node identifier.
      output: Output identifier.

    Returns:
      Its attributes there, by key: `point_id`, `observable`, `layer`
      and `reason` where they apply, `tensor_shape`.

    Raises:
      Conflict: The model's graph is not converted.
      NotFound: It has no such signature, node or output.
    """
    for entry in self.graph(model, signature)['nodes']:
      if entry['id'] != node:
        continue
      for found in entry.get('outputsMetadata', []):
        if found['id'] == output:
          return model_graph.attrs(found)
    raise errors.NotFound(
        f'The {signature} graph of {model.name} has no output {output} '
        f'of node {node}.'
    )

  def _require(self, model: model_store.Model) -> dict[str, Any]:
    state = self._state(model)
    if state.status != COMPLETED:
      why = {
          IDLE: 'it is converted when a Session of it is initialized',
          CONVERTING: 'it is being converted',
          FAILED: f'converting it failed: {state.error}',
          UNAVAILABLE: state.text.rstrip('.'),
      }[state.status]
      raise errors.Conflict(f'The graph of {model.name} is not ready: {why}.')
    return state.manifest

  def architecture(self, model: model_store.Model) -> dict[str, Any]:
    """A model's architecture and each anchor's place in its graphs.

    It does not raise for the architecture: one that cannot be bound
    says why in `error`.

    Args:
      model: The model.

    Returns:
        `{architecture, family, text, error, skeleton, bindings}`:
        `architecture` the family's `{id, name, skeleton, generator,
        config, skeleton_sha256, anchors, ports, outside}` (`generator`
        and `config` where the skeleton comes from: this repository's
        generator and pinned config.json, by their paths in it, with
        where the definition was ported from (catalog's
        Architecture.provenance()); `anchors` and `ports` its binding's
        rules, `ports` saying why each input port no anchor is of is
        bound to nothing, `outside` each module of the model not in the
        skeleton,
        `{id, name, text, stages, stages_evidence}`: why it is not
        drawn, and the stages it runs in and why), None with
        `text` saying why when the server has no skeleton of the model,
        or with `error` when binding it failed; `skeleton` the skeleton
        itself; `bindings` per signature, per layer, per anchor, and
        per block its region of the graph (architecture.bind()).

    Raises:
        Conflict: The model's graph is not converted.
    """
    manifest = self._require(model)
    key = model.sha256, manifest['created_at'], architecture_lib.identity()
    with self._lock:
      kept = self._architectures.get(key)
    if kept is not None:
      return kept
    try:
      answer = self._bound(model, manifest)
    except Exception as error:  # pylint: disable=broad-exception-caught
      if not isinstance(error, (ValueError, KeyError, OSError)):
        logging.exception('Binding the architecture of %s failed', model.name)
      answer = {
          'architecture': None,
          'family': None,
          'text': '',
          'error': (
              'The architecture could not be bound to the graph: '
              f'{errors.message(error)}'
          ),
          'skeleton': None,
          'bindings': {},
      }
    with self._lock:
      self._architectures[key] = answer
    return answer

  def known_architecture(
      self, model: model_store.Model
  ) -> tuple[dict[str, Any] | None, str]:
    """A model's architecture as far as it is known; converts nothing.

    As the analyses take it: a LiteRT-LM model's, the one its converted
    graph binds to (architecture()); a PyTorch model's, the one its
    config.json is of (catalog), whose skeleton is the one of its layer
    count and width.

    Args:
      model: The model.

    Returns:
        `(architecture, unknown)`: `{id, name, skeleton}` (`skeleton`
        None when none fits), or None; `unknown` why it is not known
        yet, as a sentence, '' when it is known, or known to be none.
    """
    if model.runtime == model_store.PYTORCH:
      try:
        config = model.config()
      except (OSError, ValueError) as error:
        return None, (
            f'The config.json of {model.name} cannot be read: '
            f'{errors.message(error)}.'
        )
      try:
        definition, document = catalog.generate(config)
      except ValueError:
        return None, ''
      return {
          'id': definition.id,
          'name': definition.name,
          'skeleton': _pytorch_skeleton(definition, document),
      }, ''
    state = self._state(model)
    if state.status == FAILED:
      return None, (
          f'The architecture of {model.name} is found in its graph, '
          f'which failed to convert: {state.error}'
      )
    if state.status != COMPLETED:
      return None, (
          f'The architecture of {model.name} is found in its graph, '
          'which is converted as a Session of it first turns on.'
      )
    found = self.architecture(model).get('architecture')
    if found is None:
      return None, ''
    return {
        'id': found['id'],
        'name': found['name'],
        'skeleton': found.get('skeleton'),
    }, ''

  def _bound(
      self, model: model_store.Model, manifest: dict[str, Any]
  ) -> dict[str, Any]:
    """The architecture, bound to the graphs as they are served."""
    if model.runtime == model_store.PYTORCH:
      return self._bound_pytorch(model, manifest)
    facts = self._read(model, manifest)
    stems = facts.layer_stems.stems
    graphs = {
        name: model_graph.regroup(graph, stems.get(name))
        for name, graph in self._graphs(model, manifest).items()
    }
    points = self._points(model, facts)
    found = architecture_lib.detect(graphs, stems, points)
    if found.skeleton_id is None:
      return {
          'architecture': None,
          'family': found.architecture.name if found.architecture else None,
          'text': found.text,
          'error': '',
          'skeleton': None,
          'bindings': {},
      }
    known = found.architecture
    document = known.skeleton(found.skeleton_id)
    entry = known.definition.entry(found.skeleton_id)
    return {
        'architecture': {
            'id': known.id,
            'name': known.name,
            'skeleton': found.skeleton_id,
            **known.definition.provenance(found.skeleton_id),
            'skeleton_sha256': entry['sha256'],
            'anchors': known.binding['anchors'],
            'ports': known.binding.get('ports', {}),
            'outside': known.outside(),
        },
        'family': known.name,
        'text': '',
        'error': '',
        'skeleton': document,
        'bindings': architecture_lib.bind(
            known, document, graphs, stems, points
        ),
    }

  def _bound_pytorch(
      self, model: model_store.Model, manifest: dict[str, Any]
  ) -> dict[str, Any]:
    """A PyTorch model's architecture, bound to its execution graphs.

    The skeleton is the family's whose layer count and width are those
    its config.json makes; each anchor's point the runner's of it.

    Args:
      model: The model.
      manifest: Graph manifest dictionary.

    Returns:
      Architecture and binding dictionary for the PyTorch model.
    """
    config = model.config()
    definition, document = catalog.generate(config)
    known = architecture_lib.Architecture(
        definition, pytorch_binding.binding(definition.id)
    )
    width = document['semantic_graph'][0]['inputs'][0]['shape'][-1]
    skeleton_id = _pytorch_skeleton(definition, document)
    if skeleton_id is None:
      return {
          'architecture': None,
          'family': known.name,
          'text': (
              f'{architecture_lib.NO_TEMPLATE}: it is of the '
              f'{known.name} family, but none of its templates has '
              f'{len(document["layers"])} layers of width {width}.'
          ),
          'error': '',
          'skeleton': None,
          'bindings': {},
      }
    side = tensor_analysis.pytorch_side(
        model.sha256, pytorch_binding.anchors(config), ''
    )
    anchor_points = {
        (point.signature, *side.places[point.id][0]): point
        for point in side.points
    }
    skeleton = known.skeleton(skeleton_id)
    entry = definition.entry(skeleton_id)
    return {
        'architecture': {
            'id': known.id,
            'name': known.name,
            'skeleton': skeleton_id,
            **definition.provenance(skeleton_id),
            'skeleton_sha256': entry['sha256'],
            'anchors': known.binding['anchors'],
            'ports': known.binding.get('ports', {}),
            'outside': known.outside(),
        },
        'family': known.name,
        'text': '',
        'error': '',
        'skeleton': skeleton,
        'bindings': architecture_lib.bind(
            known,
            skeleton,
            self._graphs(model, manifest),
            self._read(model, manifest).layer_stems.stems,
            (),
            anchor_points,
            pytorch_graph.operators,
        ),
    }

  # ---- A PyTorch model's exported graph.

  def exported(
      self, artifact: str, path: pathlib.Path, source: dict[str, Any]
  ) -> list[str]:
    """Keeps the graph a turn of a PyTorch model exported; never raises.

    A file that cannot be kept is logged, and its reason said where the
    model's graph is (state()).

    Args:
        artifact: The model.
        path: The turn's execution_graph.json.gz.
        source: Where it is from: `{session, chat, turn, side}`.

    Returns:
        The signatures it kept now (keep_exported()); [] if none.
    """
    try:
      model = self._models.get(artifact)
    except errors.NotFound:
      return []
    try:
      kept = self.keep_exported(model, path, source)
    except Exception as error:  # pylint: disable=broad-exception-caught
      if not isinstance(error, (ValueError, OSError, KeyError)):
        logging.exception(
            'Keeping the execution graph of %s failed', model.name
        )
      with self._lock:
        self._keep_failures[model.sha256] = errors.message(error)
      return []
    with self._lock:
      self._keep_failures.pop(model.sha256, None)
    return kept

  def keep_exported(
      self, model: model_store.Model, path: pathlib.Path, source: dict[str, Any]
  ) -> list[str]:
    """Keeps each signature's graph a PyTorch model has none of.

    Args:
        model: The PyTorch model.
        path: The execution_graph.json.gz a turn of it exported.
        source: Where it is from: `{session, chat, turn, side}`.

    Returns:
        The signatures kept now, in the file's order; [] if the model
        had them all.

    Raises:
        ValueError: The model is not a PyTorch model's, or the file is
          not what its runner writes.
        OSError: It cannot be read, or the graph not written.
    """
    if model.runtime != model_store.PYTORCH:
      raise ValueError(f'{model.name} is not a PyTorch model.')
    started = time.monotonic()
    with self._keeping:
      manifest = self._manifest(model)
      kept = {} if manifest is None else manifest['signatures']
      have = {entry['name'] for entry in kept}
      with gzip.open(path, 'rt', encoding='utf-8') as file:
        exported = pytorch_graph.read(json.load(file))
      new = [name for name in exported.graphs if name not in have]
      if not new:
        return []
      directory = self._directory / model.sha256
      pending = self._directory / (workspace.PENDING + uuid.uuid4().hex)
      pending.mkdir()
      try:
        signatures = []
        for entry in kept:
          shutil.copyfile(directory / entry['file'], pending / entry['file'])
          signatures.append(entry)
        modules = (
            {}
            if manifest is None
            else json_file.read_json(directory / _MODULES)
        )
        modules.update(exported.modules)
        json_file.write_json(pending / _MODULES, modules)
        for name in new:
          graph = exported.graphs[name]
          body = json.dumps(graph, separators=(',', ':')).encode()
          file_name = f'signature-{len(signatures)}.json.gz'
          with open(pending / file_name, 'wb') as file:
            file.write(gzip.compress(body, 6))
          signatures.append({
              'name': name,
              'runs': 'always',
              'step': exported.steps[name],
              'file': file_name,
              'operators': len(pytorch_graph.operators(graph)),
              'nodes': len(graph['nodes']),
              'bytes': len(body),
              'gz_bytes': (pending / file_name).stat().st_size,
              'source': dict(source),
          })
        json_file.write_json(
            pending / _MANIFEST,
            {
                'identity': pytorch_identity(),
                'runtime': model_store.PYTORCH,
                'model': {
                    'sha256': model.sha256,
                    'name': model.name,
                    'size': model.size,
                },
                'signatures': signatures,
                'seconds': {
                    'export': exported.seconds,
                    'total': round(time.monotonic() - started, 1),
                },
                'created_at': clock.now(),
            },
        )
        self._replace(model, pending)
      finally:
        shutil.rmtree(pending, ignore_errors=True)
    return new

  # ---- Converting.

  def prepare_start(
      self, record: dict[str, Any], job_id: str, aborted: Callable[[], bool]
  ) -> bool:
    """Converts the graphs of a starting Session's models, in its job.

    Called on the start's thread before anything else of the start, so
    before its runners launch. A model whose graph is kept says
    nothing. One whose graph cannot be converted fails the start: the
    Session cannot start without its graph, and that Session has ended,
    as any Session whose start failed. A start of another Session
    converts a model whose
    conversion failed before it began once more; one that failed while
    it waited for that conversion, another start's, fails it too, with
    that reason, and is not converted again.

    Args:
        record: The Session's record.
        job_id: The start's job.
        aborted: Whether the start was turned off meanwhile.

    Returns:
        False if the start was turned off, or the server closed,
        meanwhile; the conversion under way is stopped then.

    Raises:
        RuntimeError: A model's graph could not be converted; the text
          says which, and why.
    """
    began = time.monotonic()

    def stopped() -> bool:
      return self._closed or aborted()

    seen = set()
    for run in record.get('runs', []):
      try:
        model = self._models.get(run.get('artifact', ''))
      except errors.NotFound:
        continue  # The start itself says so.
      if model.sha256 in seen:
        continue
      seen.add(model.sha256)

      def progress(message: str, name: str = model.name) -> None:
        self._jobs.emit(job_id, 'progress', message=f'{name}: {message}')

      if self._state(model).status in (COMPLETED, UNAVAILABLE):
        continue
      try:
        manifest = self._converted(
            model,
            stopped,
            progress,
            record.get('name') or record['id'],
            began,
        )
      except graph_converter.Stopped:
        return False
      except Exception as error:
        if not isinstance(error, (ValueError, OSError, RuntimeError)):
          logging.exception('Converting the graph of %s failed', model.name)
        raise RuntimeError(
            f'The graph of {model.name} could not be converted, and '
            'the Session cannot start without it: '
            f'{errors.message(error)}'
        ) from error
      progress(f'its graph is converted, in {manifest["seconds"]["total"]} s')
    return True

  def _converted(
      self,
      model: model_store.Model,
      stopped: Callable[[], bool],
      progress: Callable[[str], None],
      session: str,
      began: float | None = None,
  ) -> dict[str, Any]:
    """Converts a model's graph, one at a time; returns its manifest.

    Waits for the conversion under way, if any, saying for which; each
    takes at most the timeout, so the wait ends.

    Args:
        model: The model.
        stopped: Whether to stop.
        progress: Called with what is being done.
        session: For whom, as a waiting start names it.
        began: When the start began (time.monotonic()): a conversion of the
          model that failed since, while it waited, is not tried again. None
          tries it whenever it failed.

    Returns:
        The converted model's graph manifest dictionary.

    Raises:
        graph_converter.Stopped: `stopped` said so.
        ValueError, OSError, RuntimeError: It cannot be converted, or
          failed while the start waited.
    """
    waited = None
    while not self._converting.acquire(timeout=_WAIT_SECONDS):
      if stopped():
        with self._lock:
          if self._holder_sha256 != model.sha256:
            self._states.pop(model.sha256, None)
        raise graph_converter.Stopped()
      with self._lock:
        holder = self._holder
        if holder and holder != waited and self._holder_sha256 != model.sha256:
          self._states[model.sha256] = State(CONVERTING, waiting=holder)
      if holder and holder != waited:
        waited = holder
        progress(f'waiting for {holder} to be converted')
    try:
      manifest = self._manifest(model)
      if manifest is not None:
        return manifest
      with self._lock:
        failed = self._failures.get(model.sha256)
        at = self._failed_at.get(model.sha256)
      if (
          failed is not None
          and failed.identity == identity()
          and began is not None
          and at is not None
          and at >= began
      ):
        raise RuntimeError(
            f'{failed.error} (it failed while this start waited for it)'
        )
      with self._lock:
        self._failures.pop(model.sha256, None)
        self._failed_at.pop(model.sha256, None)
        self._holder = f'the graph of {model.name}, for {session}'
        self._holder_sha256 = model.sha256
        self._states[model.sha256] = State(CONVERTING)
      try:
        return self._write(model, stopped, progress)
      except graph_converter.Stopped:
        raise
      except Exception as error:
        with self._lock:
          self._failures[model.sha256] = State(
              FAILED, error=errors.message(error), identity=identity()
          )
          self._failed_at[model.sha256] = time.monotonic()
        raise
    finally:
      with self._lock:
        if self._holder_sha256 == model.sha256:
          self._holder = self._holder_sha256 = ''
        self._states.pop(model.sha256, None)
      self._converting.release()

  def _write(
      self,
      model: model_store.Model,
      stopped: Callable[[], bool],
      progress: Callable[[str], None],
  ) -> dict[str, Any]:
    """Converts and keeps a model's graph; called converting alone.

    A failure is kept, with the tail of the converter's log (_fail()).

    Args:
      model: The model to convert.
      stopped: Whether conversion was requested to stop.
      progress: Progress reporting callback.

    Returns:
      The converted graph manifest dictionary.
    """
    started = time.monotonic()
    pending = self._directory / (workspace.PENDING + uuid.uuid4().hex)
    pending.mkdir()
    converted = pending / _CONVERTED
    try:
      plain = self._scans.scan(model)
      subgraphs = {s.name: s.subgraph for s in plain.signatures}
      progress("Converting its graph with Model Explorer's converter")
      converting = time.monotonic()
      self._convert(model.path, subgraphs, converted, stopped, self._timeout)
      seconds = time.monotonic() - converting
      graphs = json_file.read_json(converted)
      converted.unlink()
      signatures = []
      for index, signature in enumerate(plain.signatures):
        graph = graphs[signature.name]
        body = json.dumps(graph, separators=(',', ':')).encode()
        name = f'signature-{index}.json.gz'
        with open(pending / name, 'wb') as file:
          file.write(gzip.compress(body, 6))
        signatures.append({
            'name': signature.name,
            'subgraph': signature.subgraph,
            'runs': signature.runs,
            'file': name,
            'operators': len(model_graph.operators(graph)),
            'nodes': len(graph['nodes']),
            'bytes': len(body),
            'gz_bytes': (pending / name).stat().st_size,
        })
      manifest = {
          'identity': identity(),
          'model': {
              'sha256': model.sha256,
              'name': model.name,
              'size': model.size,
          },
          'section_sha256': plain.section_sha256,
          'signatures': signatures,
          'seconds': {
              'convert': round(seconds, 1),
              'total': round(time.monotonic() - started, 1),
          },
          'created_at': clock.now(),
      }
      json_file.write_json(pending / _MANIFEST, manifest)
      self._replace(model, pending)
      return manifest
    except graph_converter.Stopped:
      raise
    except Exception as error:
      self._fail(model, error, graph_converter.log_path(converted))
      raise
    finally:
      shutil.rmtree(pending, ignore_errors=True)

  def _replace(self, model: model_store.Model, pending: pathlib.Path) -> None:
    """Makes a pending directory the model's: its graph or its failure."""
    final = self._directory / model.sha256
    # One another identity wrote (_manifest() said so), or a failure.
    shutil.rmtree(final, ignore_errors=True)
    pending.rename(final)

  def _fail(
      self, model: model_store.Model, error: Exception, log: pathlib.Path
  ) -> None:
    """Keeps a failure: its reason, identity and the log's tail."""
    pending = self._directory / (workspace.PENDING + uuid.uuid4().hex)
    try:
      pending.mkdir()
      json_file.write_json(
          pending / _FAILURE,
          {
              'identity': identity(),
              'error': errors.message(error),
              'model': {'sha256': model.sha256, 'name': model.name},
              'failed_at': clock.now(),
          },
      )
      if log.is_file():
        with open(log, 'rb') as file:
          file.seek(max(0, log.stat().st_size - _LOG_KEPT))
          (pending / _LOG).write_bytes(file.read())
      self._replace(model, pending)
    except OSError:
      logging.exception(
          'Keeping the failure of the graph of %s failed', model.name
      )
    finally:
      shutil.rmtree(pending, ignore_errors=True)

  def collect(self, kept: Collection[str]) -> None:
    """Deletes the graphs, and failures, of models no longer kept.

    Args:
        kept: The SHA-256 of every model kept.
    """
    with self._lock:
      for directory in sorted(self._directory.iterdir()):
        if (
            directory.name.startswith(workspace.PENDING)
            or directory.name in kept
            or directory.name == self._holder_sha256
        ):
          continue
        gone = self._directory / (workspace.PENDING + uuid.uuid4().hex)
        directory.rename(gone)
        self._failures.pop(directory.name, None)
        self._failed_at.pop(directory.name, None)
        self._keep_failures.pop(directory.name, None)
        shutil.rmtree(gone, ignore_errors=True)

  def close(self) -> None:
    """Stops a conversion under way, and waits for its child to end.

    Every conversion after stops at once.
    """
    self._closed = True
    if self._converting.acquire(timeout=_CLOSE_SECONDS):
      self._converting.release()
    else:
      logging.warning('A graph conversion did not stop in %s s', _CLOSE_SECONDS)


def _module_points(
    model: model_store.Model,
    signature: str,
    graph: dict[str, Any],
    stem: str | None,
) -> dict[tuple[str, str], tuple[str, int | None]]:
  """The modules of a PyTorch graph the runner can capture, and their points.

  Per (module, edge), the point a Session that observes it captures in
  the signature: each semantic operator's module (pytorch_binding
  .op_modules()), which a Session pairs with LiteRT-LM points, and every
  other folded node's module, what it returns and what it reads, which a
  pair made by hand may name (session/mappings); each with its layer, by
  the rules or its namespace (None outside the layers). {} for a module
  of a model whose config.json binds none.

  Args:
    model: The PyTorch model.
    signature: Signature name.
    graph: The folded graph dictionary.
    stem: Optional layer stem string.

  Returns:
    Mapping of (module, edge) to (point_id, layer).
  """
  try:
    found = pytorch_binding.op_modules(model.config())
  except (ValueError, OSError):
    return {}
  points = {
      (each.module, each.edge): (
          pytorch_binding.module_point_id(signature, each.module, each.edge),
          each.layer,
      )
      for each in found
  }
  for module, namespace in pytorch_graph.folded_modules(graph).items():
    layer = model_graph.namespace_layer(namespace, stem) if stem else None
    for edge in ('out', 'in'):
      points.setdefault(
          (module, edge),
          (
              pytorch_binding.module_point_id(signature, module, edge),
              layer,
          ),
      )
  return points


def _pytorch_skeleton(
    definition: catalog.Architecture, document: dict[str, Any]
) -> str | None:
  """The pinned skeleton of the layer count and width a config makes."""
  width = document['semantic_graph'][0]['inputs'][0]['shape'][-1]
  return next(
      (
          entry['id']
          for entry in definition.spec['skeletons']
          if (pinned := definition.skeleton(entry['id']))
          and len(pinned['layers']) == len(document['layers'])
          and pinned['semantic_graph'][0]['inputs'][0]['shape'][-1] == width
      ),
      None,
  )


def _pytorch_binding(model: model_store.Model) -> dict[str, Any]:
  """The PyTorch binding of a PyTorch model's architecture.

  Args:
    model: The PyTorch model.

  Returns:
    The binding rules dictionary.

  Raises:
    ValueError, OSError: Its config.json cannot be read, or no
      architecture with a PyTorch binding defines its model.
  """
  config = model.config()
  definition, _ = catalog.generate(config)
  return pytorch_binding.binding(definition.id)


def _pytorch_facts(graphs: dict[str, dict[str, Any]]) -> Facts:
  """What the server says of a PyTorch model's graphs, folded.

  Its layers are named by the module path, `layers/<N>`; none is grouped
  by another, and each signature computes every layer it has operators
  of.

  Args:
    graphs: Mapping from signature name to graph dictionary.

  Returns:
    Facts describing stems, groups, counts, and operator layers.
  """
  stems = model_graph.layer_stems(
      graphs, segments=True, operators_of=pytorch_graph.operators
  )
  layer_ids = {
      name: model_graph.operator_layers(graph, stems.stems.get(name))
      for name, graph in graphs.items()
  }
  return Facts(
      stems,
      {name: {} for name in stems.stems},
      {name: len(ids) for name, ids in layer_ids.items()},
      layer_ids,
  )
