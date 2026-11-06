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

"""Each decoder layer's four anchors, and modules asked for, by collect().

A turn that observes runs its forward passes inside one
pytorch-collector `collect()` under one snapshot budget, in one of two
modes (`Observer(full=...)`):

- module-only (MODULES), `collect(model, modules=[...])`: hooks only on
  the modules the anchors are taken at (each decoder layer
  `model.language_model.layers.<i>`, its `pre_feedforward_layernorm` and
  its `per_layer_input_gate`) and on the modules the Load asks for beside
  them (`Module`), and no operator observed. It snapshots on the device the
  model runs on (the CPU, MPS), each selected module's whole inputs and
  output: a layer's arguments carry the KV cache, which is snapshotted again
  at each layer (Gemma 4 E2B: 0.46 GB of a turn of "Name three colors.").
- full (FULL), `collect(model)`: every module's inputs and outputs and every
  operator's, on the CPU only; the one mode the execution graph can be
  exported of (execution_graph), which needs the operators.

Once the turn is over, only these are kept of its records, per forward
pass: the four anchors of each decoder layer, which end the blocks of the
layer as the skeleton names them (transformers 5.17.0,
Gemma4TextDecoderLayer.forward),

- `input`: what the layer reads, its argument 0 (hidden_states);
- `intermediate.1`, where Attention ends: what `pre_feedforward_layernorm`
  reads, the residual after attention;
- `intermediate.2`, where Feed Forward ends: what `per_layer_input_gate`
  reads, the residual after the feed-forward network;
- `output`, where Per-Layer Embedding ends: what the layer returns, after
  `layer_scalar`;

and of each module asked for, what it reads (argument 0) or returns. The
collector clones each at its hook, so each is the tensor as it was there;
nothing else of the capture is written. Both modes record the same module
tensors, bit for bit.

A forward pass whose capture fails, the budget exceeded or (full) a tensor
on another device, is run again without it from the KV cache it started
from (the cache's layers rebind their tensors as they grow, so the ones
held before the pass are unchanged), and the rest of the turn runs
unobserved: the turn completes with no tensors and says why
(`Observer.reason`). The collector's own failure is told apart from the
model's by that second run: a model that fails again fails the turn.

A full turn whose capture held can also export its execution graph
(execution_graph): of the first pass of each signature asked for, the graph
the collector draws of it, before the capture is let go.
"""

from collections.abc import Callable, Collection, Sequence
import dataclasses
from typing import Any, NamedTuple, TypeVar

from model_debugger.runner_pytorch.llm import execution_graph
import pytorch_collector
import torch

# What a Load's `observe` option names: no tensors, or the anchors.
NONE = 'none'
ANCHORS = 'anchors'
LEVELS = (NONE, ANCHORS)
# How a turn's capture observes: every module and operator, or the
# selected modules alone.
FULL = 'full'
MODULES = 'modules'
# The snapshot budget of a turn when the Load sets none: a turn of Gemma 4
# E2B on the CPU, "Name three colors." and its 24 tokens of reply (25 forward
# passes), snapshots 4.26 GiB fully, 0.46 GB module-only.
DEFAULT_MEMORY_LIMIT_BYTES = 8 * 2**30
# The scope of the decoder layers in Gemma4ForConditionalGeneration.
LAYERS = 'model.language_model.layers'
# Where in a module's record its tensor is, by edge: what it reads is its
# argument 0, what it returns its output.
EDGES = {'in': ('args', 0), 'out': ('output',)}

_T = TypeVar('_T')


class _WantedKey(NamedTuple):
  """Lookup key for a wanted module edge and nested selection path."""

  module: str
  edge: str
  path: tuple[str | int, ...]


@dataclasses.dataclass(frozen=True)
class Anchor:
  """Where in a layer an anchor's tensor is taken.

  Attributes:
      name: The anchor, as the architecture names it.
      module: The module under the layer, '' for the layer itself.
      edge: 'in' for what the module reads, 'out' for what it returns.
      path: The tensor's path in the collector's record.
  """

  name: str
  module: str
  edge: str
  path: tuple[str | int, ...]


@dataclasses.dataclass(frozen=True)
class Module:
  """A module the Load asks for beside the anchors, and which tensor.

  Attributes:
      module: Its path, as the model's named_modules() names it.
      edge: 'in' for its argument 0, 'out' for what it returns.
  """

  module: str
  edge: str

  @property
  def path(self) -> tuple[str | int, ...]:
    """The tensor's path in the collector's record."""
    return EDGES[self.edge]


# The anchors in layer order.
GEMMA4_ANCHORS = (
    Anchor('input', '', 'in', ('args', 0)),
    Anchor('intermediate.1', 'pre_feedforward_layernorm', 'in', ('args', 0)),
    Anchor('intermediate.2', 'per_layer_input_gate', 'in', ('args', 0)),
    Anchor('output', '', 'out', ('output',)),
)


def module_path(layer: int, anchor: Anchor) -> str:
  """The module path an anchor of a layer is taken at."""
  path = f'{LAYERS}.{layer}'
  return f'{path}.{anchor.module}' if anchor.module else path


def capture_key(layer: int, anchor: str) -> str:
  """The key of an anchor's tensor in a capture file."""
  return f'post_anchor_{layer}_{anchor}'


def module_key(module: str, edge: str) -> str:
  """The key of a module's tensor asked for in a capture file."""
  return f'post_module_{module}_{edge}'


def parse_modules(model: torch.nn.Module, value: object) -> tuple[Module, ...]:
  """The modules a Load's `observeModules` asks for, checked.

  Args:
    model: The model loaded.
    value: The option as JSON: a list of `{"module": <path>, "edge": "in" |
      "out"}`.

  Returns:
    The modules asked for, checked.

  Raises:
    ValueError: It is not such a list, or a module is not one of the model's.
  """
  if not isinstance(value, list):
    raise ValueError('observeModules must be a list.')
  known = dict(model.named_modules())
  found = []
  for entry in value:
    if (
        not isinstance(entry, dict)
        or set(entry) != {'module', 'edge'}
        or entry['edge'] not in EDGES
        or not isinstance(entry['module'], str)
    ):
      raise ValueError(
          'Each of observeModules must be {"module": <path>, "edge": '
          f'"in" or "out"}}, not {entry!r}.'
      )
    if entry['module'] not in known:
      raise ValueError(
          f'observeModules: the model has no module {entry["module"]}.'
      )
    found.append(Module(entry['module'], entry['edge']))
  return tuple(dict.fromkeys(found))


@dataclasses.dataclass(frozen=True)
class Tensor:
  """One anchor's tensor, or one module's asked for, of one forward pass.

  Attributes:
      signature: 'prefill' or 'decode'.
      step: The forward pass's step label.
      layer: The decoder layer of an anchor; None for a module's.
      anchor: The anchor; '' for a module's.
      module: The module path it was taken at.
      edge: 'in' or 'out'.
      path: Its path in the collector's record, dotted.
      tensor: The collector's clone, on the device it was taken on.
  """

  signature: str
  step: int
  layer: int | None
  anchor: str
  module: str
  edge: str
  path: str
  tensor: torch.Tensor

  @property
  def key(self) -> str:
    """Its key in a capture file."""
    if self.anchor:
      return capture_key(self.layer, self.anchor)
    return module_key(self.module, self.edge)


class _CacheState:
  """A DynamicCache's attributes and its layers', to put back as they were.

  Its layers grow by rebinding `keys` and `values` (torch.cat, slicing)
  and an int `cumulative_length`, never in place: the tensors held here
  stay as they were.
  """

  def __init__(self, cache: object) -> None:
    """Holds what `cache` and its layers are now."""
    self._cache = cache
    self._attributes = dict(vars(cache))
    self._layers = [dict(vars(layer)) for layer in cache.layers]

  def restore(self) -> None:
    """Puts the cache and its layers back as they were held."""
    vars(self._cache).clear()
    vars(self._cache).update(self._attributes)
    for layer, attributes in zip(self._cache.layers, self._layers):
      vars(layer).clear()
      vars(layer).update(attributes)


class Observer:
  """One turn's anchors and modules: its forward passes in one collect()."""

  def __init__(
      self,
      model: torch.nn.Module,
      memory_limit_bytes: int,
      modules: Sequence[Module] = (),
      full: bool = False,
  ) -> None:
    """Starts the capture of a turn.

    Args:
        model: The Gemma4ForConditionalGeneration every pass calls.
        memory_limit_bytes: collect()'s snapshot budget for the turn.
        modules: The modules asked for beside the anchors, checked
          (parse_modules()).
        full: Whether to capture every module and operator (FULL), as exporting
          the execution graph needs, on the CPU only; else only the selected
          modules (MODULES), on any device.
    """
    self._model = model
    self.memory_limit_bytes = memory_limit_bytes
    self.mode = FULL if full else MODULES
    self._passes: list[tuple[str, int]] = []
    self._capture = None
    # Why the turn has no tensors; '' while it may have them.
    self.reason = ''
    self.snapshot_bytes = 0
    # The execution graph finish() exported, if asked for one; why it
    # could not, if it could not.
    self.graph: dict[str, Any] | None = None
    self.graph_reason = ''
    # Per module record (module, edge, path), what it is kept as: the
    # anchor of a layer, the module asked for, or both.
    self._wanted: dict[
        _WantedKey,
        list[tuple[int, Anchor] | Module],
    ] = {}
    layers = len(model.model.language_model.layers)
    for layer in range(layers):
      for anchor in GEMMA4_ANCHORS:
        key = _WantedKey(module_path(layer, anchor), anchor.edge, anchor.path)
        self._wanted.setdefault(key, []).append((layer, anchor))
    for module in dict.fromkeys(modules):
      key = _WantedKey(module.module, module.edge, module.path)
      self._wanted.setdefault(key, []).append(module)
    try:
      if full:
        capture = pytorch_collector.collect(
            model, memory_limit_bytes=memory_limit_bytes
        )
      else:
        capture = pytorch_collector.collect(
            model,
            memory_limit_bytes=memory_limit_bytes,
            modules=sorted({key.module for key in self._wanted}),
        )
      capture.__enter__()
    except Exception as error:  # pylint: disable=broad-exception-caught
      # The collector's refusal, any.
      self.reason = f'collect() did not start: {_described(error)}'
      return
    self._capture = capture

  def forward(
      self, run: Callable[[], _T], cache: object, signature: str, step: int
  ) -> _T:
    """Runs one forward pass, observed while the capture holds.

    Args:
        run: Runs the pass.
        cache: The DynamicCache the pass extends.
        signature: 'prefill' or 'decode'.
        step: The pass's step label.

    Returns:
        What `run` returns.
    """
    if self._capture is None:
      return run()
    before = _CacheState(cache)
    try:
      result = run()
    except Exception as error:  # pylint: disable=broad-exception-caught
      # The collector's or the model's.
      self._fail(f'{signature} step {step}: {_described(error)}')
      before.restore()
      return run()
    self._passes.append((signature, step))
    return result

  def finish(self, graph_signatures: Collection[str] = ()) -> list[Tensor]:
    """Ends the capture; returns each pass's anchors and modules.

    Of a full capture that held, it exports the execution graph of the
    first pass of each of `graph_signatures` the turn ran, as `graph`
    (None if it ran none of them), or says why it could not, as
    `graph_reason`; the tensors do not depend on it. A module-only
    capture exports none.

    Args:
        graph_signatures: The signatures whose graphs to export.

    Returns:
        The tensors, by pass, then the anchors by layer and anchor, then
        the modules by path and edge; [] with `reason` once the capture
        failed.
    """
    capture, self._capture = self._capture, None
    if capture is None:
      return []
    try:
      capture.__exit__(None, None, None)
    except Exception as error:  # pylint: disable=broad-exception-caught
      # The collector's verdict, any.
      self.reason = f'collect() failed: {_described(error)}'
      return []
    if graph_signatures and self.mode == FULL:
      try:
        self.graph = execution_graph.document(
            capture, self._passes, self._model, graph_signatures
        )
      except Exception as error:  # pylint: disable=broad-exception-caught
        # The collector's export, any.
        self.graph_reason = (
            f'The execution graph was not exported: {_described(error)}'
        )
    found = {}
    for record in capture.records:
      self.snapshot_bytes += record.tensor.nbytes
      if record.op is not None:
        continue
      matches = self._wanted.get(
          _WantedKey(record.module, record.edge, record.path), ()
      )
      for match in matches:
        if isinstance(match, Module):
          layer, name = None, ''
          order = (1, match.module, match.edge)
        else:
          layer, anchor = match
          name = anchor.name
          order = (0, layer, GEMMA4_ANCHORS.index(anchor))
        key = (record.forward_id, order)
        if key in found:
          self.reason = (
              f'{record.module} ran twice in forward pass '
              f'{record.forward_id}; its tensor is not one tensor.'
          )
          return []
        signature, step = self._passes[record.forward_id]
        found[key] = Tensor(
            signature=signature,
            step=step,
            layer=layer,
            anchor=name,
            module=record.module,
            edge=record.edge,
            path='.'.join(str(part) for part in record.path),
            tensor=record.tensor,
        )
    expected = len(self._passes) * sum(map(len, self._wanted.values()))
    if len(found) != expected:
      self.reason = (
          f'The capture holds {len(found)} of the {expected} anchor '
          'and module tensors of the turn.'
      )
      return []
    return [found[key] for key in sorted(found)]

  def close(self) -> None:
    """Ends the capture of a turn that failed, its hooks removed."""
    capture, self._capture = self._capture, None
    if capture is None:
      return
    try:
      capture.__exit__(None, None, None)
    except Exception:  # pylint: disable=broad-exception-caught
      # The turn's own failure is what it reports.
      pass

  def _fail(self, reason: str) -> None:
    """Ends a capture that failed: the rest of the turn is unobserved."""
    self.reason = f'collect() failed at {reason}'
    self.close()


def _described(error: BaseException) -> str:
  """An exception as the reason names it."""
  return f'{type(error).__name__}: {error}'
