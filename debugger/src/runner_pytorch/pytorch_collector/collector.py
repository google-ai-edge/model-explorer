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

"""Module and operator capture with scope-owned resources and atomic results."""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import sys
import threading
import types
from typing import Any, Literal

import torch
from torch import nn
from torch.nn.modules import module as module_runtime
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

from ._dispatch import CaptureMode
from ._provenance import describe_arguments
from ._provenance import Edge
from ._provenance import Op
from ._provenance import Provenance
from ._snapshots import Snapshotter
from ._snapshots import walk

# Formal snapshot-data budget: ~2.1x the measured 489.5 MiB cached Gemma
# four-forward capture. This does not limit total process memory.
DEFAULT_MEMORY_LIMIT_BYTES = 1 << 30
_CALL_CODE = nn.Module._call_impl.__code__  # pylint: disable=protected-access
_INNER_CODE = next(
    value
    for value in _CALL_CODE.co_consts
    if isinstance(value, types.CodeType) and value.co_name == "inner"
)
_OWNERS: dict[int, Capture] = {}
_OWNERS_LOCK = threading.RLock()
_HOOK_MAPS = (
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_backward_pre_hooks",
    "_backward_hooks",
)
_GLOBAL_HOOK_MAPS = (
    "_global_forward_pre_hooks",
    "_global_forward_hooks",
    "_global_backward_pre_hooks",
    "_global_backward_hooks",
)


class CaptureError(RuntimeError):
  """The scope cannot deliver a reliable capture."""


@dataclasses.dataclass(frozen=True, slots=True)
class Record:
  forward_id: int
  module: str
  op: str | None
  call_id: int
  module_call_id: int
  edge: Literal["in", "out"]
  path: tuple[str | int, ...]
  tensor: torch.Tensor


@dataclasses.dataclass(slots=True)
class _Call:
  module: nn.Module
  frame: types.FrameType
  forward_id: int
  call_id: int
  scope: tuple[tuple[str, int], ...]
  inputs_seen: bool = False


def _hook_fingerprint(
    module: nn.Module,
) -> tuple[tuple[tuple[Any, int], ...], ...]:
  """Returns a fingerprint of registered hooks on the module."""
  return tuple(
      tuple((key, id(value)) for key, value in getattr(module, name).items())
      for name in _HOOK_MAPS
  )


def _structure(module: nn.Module) -> tuple[Any, ...]:
  """Returns a structural fingerprint of the module."""
  forward = module.forward
  params = []
  # pylint: disable=protected-access
  for name, param in module._parameters.items():
    if param is None:
      params.append((name, None))
    else:
      param_info = (
          id(param),
          param.untyped_storage()._cdata,
          param.dtype,
          param.device,
          tuple(param.shape),
          tuple(param.stride()),
          param.storage_offset(),
      )
      params.append((name, param_info))
  modules = tuple((name, id(child)) for name, child in module._modules.items())
  # pylint: enable=protected-access
  return (
      modules,
      tuple(params),
      id(getattr(forward, "__func__", forward)),
      id(getattr(forward, "__self__", None)),
  )


def _supported_call(module: nn.Module) -> bool:
  """Returns whether the module's __call__ is supported for capture."""
  if type(module).__call__ is nn.Module.__call__:
    return True
  # Transformers 5.17.0 delegates this shared layer wrapper unchanged to
  # nn.Module outside its checkpoint branch. Inspect already-loaded classes;
  # Transformers remains an optional caller dependency, not a runtime import.
  transformers = sys.modules.get("transformers")
  # pylint: disable=line-too-long
  layers = sys.modules.get("transformers.modeling_layers")
  # pylint: enable=line-too-long
  layer_type = getattr(layers, "GradientCheckpointingLayer", None)
  # pylint: disable=line-too-long
  if getattr(transformers, "__version__", None) != "5.17.0" or layer_type is None:
    return False
  # pylint: enable=line-too-long
  return (
      isinstance(module, layer_type)
      and type(module).__call__ is layer_type.__call__
      and getattr(super(layer_type, module).__call__, "__func__", None)
      is nn.Module.__call__
      and not (module.gradient_checkpointing and module.training)
  )


def _select(
    model: nn.Module, modules: Iterable[str | nn.Module]
) -> tuple[tuple[str, nn.Module], ...]:
  """Resolve module paths or objects to ``(path, module)`` in ``named_modules()`` order.

  A path may name any appearance of a shared module; the result, like every
  record, uses the path of its first appearance.

  Args:
    model: The root PyTorch model.
    modules: Iterable of module paths or module instances to select.

  Returns:
    A tuple of (path, module) pairs in named_modules order.
  """
  if isinstance(modules, (str, nn.Module)) or not isinstance(modules, Iterable):
    raise TypeError("modules must be an iterable of module paths or modules")
  requested = list(modules)
  if any(not isinstance(item, (str, nn.Module)) for item in requested):
    raise TypeError(
        "modules must contain module paths (str) or torch.nn.Module objects"
    )
  if not requested:
    raise ValueError(
        "modules must name at least one module; omit it to capture every module"
    )
  by_path = dict(model.named_modules(remove_duplicate=False))
  unknown = [
      item
      for item in requested
      if isinstance(item, str) and item not in by_path
  ]
  if unknown:
    raise ValueError(f"Unknown module paths: {unknown}")
  registered = {id(module) for module in by_path.values()}
  foreign = [
      type(item).__name__
      for item in requested
      if isinstance(item, nn.Module) and id(item) not in registered
  ]
  if foreign:
    raise ValueError(f"Modules not registered in the model: {foreign}")
  wanted = {
      id(by_path[item]) if isinstance(item, str) else id(item)
      for item in requested
  }
  return tuple(
      (path, module)
      for path, module in model.named_modules()
      if id(module) in wanted
  )


def _invocation(module: nn.Module) -> tuple[types.FrameType, bool]:
  """Distinguish a normal post hook from _call_impl's exception cleanup.

  The pinned PyTorch eager implementation runs normal hooks in ``inner``.
  The emergency always_call path runs directly in ``_call_impl``. An ambient
  exception from a caller's except block is therefore not mistaken for failure.

  Args:
    module: The module being invoked.

  Returns:
    A tuple of (call_frame, is_normal_hook).

  Raises:
    CaptureError: If no PyTorch eager call frame is found.
  """
  frame = sys._getframe(1)  # pylint: disable=protected-access
  normal = False
  try:
    while frame is not None:
      if frame.f_locals.get("self") is module:
        if frame.f_code is _INNER_CODE:
          normal = True
        if frame.f_code is _CALL_CODE:
          return frame, normal
      frame = frame.f_back
  finally:
    del frame
  raise CaptureError(
      "Unsupported module invocation: no PyTorch eager call frame"
  )


class Capture:
  """One-use scope for module and dispatcher-visible operator snapshots.

  With selected modules, only their inputs and outputs are snapshotted and no
  operator is observed; ``ops``, ``edges`` and ``outputs`` are then not
  recorded.
  """

  def __init__(
      self,
      model: nn.Module,
      *,
      memory_limit_bytes: int,
      modules: Iterable[str | nn.Module] | None = None,
  ) -> None:
    if not isinstance(model, nn.Module):
      raise TypeError("model must be a torch.nn.Module")
    if isinstance(memory_limit_bytes, bool) or not isinstance(
        memory_limit_bytes, int
    ):
      raise TypeError("memory_limit_bytes must be an integer")
    if memory_limit_bytes < 0:
      raise ValueError("memory_limit_bytes must be nonnegative")
    selection = None if modules is None else _select(model, modules)
    self._model: nn.Module | None = model
    self._limit = memory_limit_bytes
    # None: every module and operator. Otherwise the selected module paths;
    # the root is still hooked, without snapshots, to number its forwards.
    self._selected_paths = (
        None if selection is None else tuple(path for path, _ in selection)
    )
    self._selection = (
        () if selection is None else tuple(module for _, module in selection)
    )
    self._selected: frozenset[int] | None = None
    self._state = "new"
    self._failure: str | None = None
    self._thread: int | None = None
    self._modules: tuple[tuple[str, nn.Module], ...] = ()
    self._paths: dict[int, str] = {}
    self._structures: dict[int, tuple[Any, ...]] = {}
    self._hooks: dict[int, tuple[Any, ...]] = {}
    self._handles: list[Any] = []
    self._calls: list[_Call] = []
    self._snapshots: Snapshotter | None = None
    self._provenance: Provenance | None = None
    self._pending: list[Record] = []
    self._pending_ops: list[Op] = []
    self._pending_edges: list[Edge] = []
    self._pending_outputs: list[Edge] = []
    self._published: tuple[Record, ...] | None = None
    self._published_ops: tuple[Op, ...] = ()
    self._published_edges: tuple[Edge, ...] = ()
    self._published_outputs: tuple[Edge, ...] = ()
    self._next_forward = 0
    self._next_call = 0
    self._grad = None
    self._mode: CaptureMode | None = None

  @property
  def modules(self) -> tuple[str, ...] | None:
    """The selected module paths, or None when every module and operator is captured."""
    return self._selected_paths

  @property
  def records(self) -> tuple[Record, ...]:
    if self._state != "complete":
      raise CaptureError(
          "records are readable only after successful scope exit"
      )
    assert self._published is not None
    return self._published

  @property
  def ops(self) -> tuple[Op, ...]:
    """Every operator call inside root forwards, with or without snapshots."""
    if self._state != "complete":
      raise CaptureError("ops are readable only after successful scope exit")
    self._require_operators("ops")
    return self._published_ops

  @property
  def edges(self) -> tuple[Edge, ...]:
    """Exactly one source for each tensor input of each operator call."""
    if self._state != "complete":
      raise CaptureError("edges are readable only after successful scope exit")
    self._require_operators("edges")
    return self._published_edges

  @property
  def outputs(self) -> tuple[Edge, ...]:
    """The source of each tensor a root forward returned; ``call_id`` is None."""
    if self._state != "complete":
      raise CaptureError(
          "outputs are readable only after successful scope exit"
      )
    self._require_operators("outputs")
    return self._published_outputs

  def _observes(self, module: nn.Module) -> bool:
    """Whether the module's inputs and outputs are snapshotted."""
    return self._selected is None or id(module) in self._selected

  def _require_operators(self, name: str) -> None:
    if self._selected_paths is not None:
      raise CaptureError(
          f"{name} are not recorded when collect() selects modules"
      )

  def __enter__(self) -> Capture:
    if self._state != "new":
      raise CaptureError("a capture scope can only be entered once")
    self._thread = threading.get_ident()
    try:
      self._check_global_state()
      assert self._model is not None
      self._modules = tuple(self._model.named_modules())
      self._paths = {id(module): path for path, module in self._modules}
      if self._selected_paths is not None:
        if (
            tuple(self._paths.get(id(m)) for m in self._selection)
            != self._selected_paths
        ):
          raise CaptureError(
              "the selected modules changed between collect() and scope entry"
          )
        self._selected = frozenset(id(m) for m in self._selection)
      with _OWNERS_LOCK:
        if any(id(module) in _OWNERS for _, module in self._modules):
          raise CaptureError(
              "target modules are already owned by another capture"
          )
        for _, module in self._modules:
          _OWNERS[id(module)] = self
      # Only operator data flow is verified on CPU; module boundaries
      # alone are device-independent and snapshot on the observed device.
      self._snapshots = Snapshotter(
          self._model.parameters(),
          self._limit,
          cpu_only=self._selected is None,
      )
      if self._selected is None:
        self._provenance = Provenance(
            self._model.named_parameters(remove_duplicate=False),
            self._model.named_buffers(remove_duplicate=False),
        )
      self._structures = {id(m): _structure(m) for _, m in self._modules}
      self._grad = torch.no_grad()
      self._grad.__enter__()
      for _, module in self._modules:
        self._check_eager(module)
        if not self._observes(module) and module is not self._model:
          continue
        self._handles.append(
            module.register_forward_pre_hook(
                self._begin,
                prepend=True,
                with_kwargs=True,
            )
        )
        self._handles.append(
            module.register_forward_pre_hook(
                self._inputs,
                with_kwargs=True,
            )
        )
        self._handles.append(
            module.register_forward_hook(
                self._output,
                with_kwargs=True,
                always_call=True,
            )
        )
      self._hooks = {id(m): _hook_fingerprint(m) for _, m in self._modules}
      if self._selected is None:
        mode = CaptureMode(self)
        mode.__enter__()
        self._mode = mode
      self._state = "active"
      return self
    except BaseException:
      self._mark_failed("capture setup failed")
      self._cleanup()
      self._state = "failed"
      raise

  def __exit__(self, exc_type, exc_value, traceback) -> bool:
    if self._state != "active":
      raise CaptureError("only an active capture scope can be exited")
    # Dispatch and grad modes belong to the entering thread. A foreign
    # exit poisons this capture but leaves it active for its owner to clean
    # up, including when an exception or earlier failure is already present.
    self._check_thread()
    validation_error = None
    if exc_type is not None:
      self._mark_failed("scope exited with an exception")
    elif self._failure is None:
      try:
        self._verify_all()
        if self._calls:
          self._fail("a module call did not complete")
      except BaseException as error:  # pylint: disable=broad-exception-caught
        self._mark_failed("scope validation failed")
        validation_error = error
    cleanup_error = self._cleanup()
    if cleanup_error is not None:
      self._mark_failed("capture cleanup failed")
    if exc_type is not None:
      self._state = "failed"
      return False  # Never replace the caller's original exception.
    if self._failure is not None:
      self._state = "failed"
      if validation_error is not None:
        raise validation_error
      if cleanup_error is not None:
        raise CaptureError(self._failure) from cleanup_error
      raise CaptureError(self._failure)
    self._published = tuple(self._pending)
    self._published_ops = tuple(self._pending_ops)
    self._published_edges = tuple(self._pending_edges)
    self._published_outputs = tuple(self._pending_outputs)
    self._pending.clear()
    self._pending_ops.clear()
    self._pending_edges.clear()
    self._pending_outputs.clear()
    self._state = "complete"
    return False

  def _mark_failed(self, message: str) -> None:
    if self._failure is None:
      self._failure = message
    self._pending.clear()
    self._pending_ops.clear()
    self._pending_edges.clear()
    self._pending_outputs.clear()
    if self._snapshots is not None:
      self._snapshots.clear()

  def _fail(self, message: str) -> None:
    self._mark_failed(message)
    raise CaptureError(message)

  def _check_thread(self) -> None:
    if threading.get_ident() != self._thread:
      self._fail("capture must run on its entering Python thread")

  def _check_global_state(self) -> None:
    if any(getattr(module_runtime, name) for name in _GLOBAL_HOOK_MAPS):
      raise CaptureError("global module hooks are not supported")
    if torch.is_inference_mode_enabled():
      raise CaptureError("inference_mode is not supported; use no_grad")
    if torch.jit.is_tracing() or torch.jit.is_scripting():
      raise CaptureError("only eager execution is supported")

  def _check_eager(self, module: nn.Module) -> None:
    # pylint: disable=protected-access
    if module._compiled_call_impl is not None:
      self._fail("compiled module calls are not supported")
    if not _supported_call(module):
      self._fail("custom __call__ implementations are not supported")
    if getattr(module._call_impl, "__func__", None) is not nn.Module._call_impl:
      self._fail("custom _call_impl implementations are not supported")
    # pylint: enable=protected-access

  def _verify_module(self, module: nn.Module) -> None:
    self._check_eager(module)
    if _structure(module) != self._structures[id(module)]:
      self._fail(
          "module topology, parameters or forward binding changed during"
          " capture"
      )
    if _hook_fingerprint(module) != self._hooks[id(module)]:
      self._fail("module hook configuration or order changed during capture")

  def _verify_all(self) -> None:
    self._check_global_state()
    for _, module in self._modules:
      self._verify_module(module)

  def _guard(self, module: nn.Module) -> None:
    """Verifies thread affinity and module state before hook execution."""
    self._check_thread()
    if self._failure is not None:
      raise CaptureError(self._failure)
    self._check_global_state()
    self._verify_module(module)
    # Aborted child BaseExceptions skip PyTorch's always_call hooks. Detect a
    # retained call whose Python frame has already unwound, before pairing.
    live: set[int] = set()
    frame = sys._getframe(1)  # pylint: disable=protected-access
    try:
      while frame is not None:
        live.add(id(frame))
        frame = frame.f_back
    finally:
      del frame
    if any(id(call.frame) not in live for call in self._calls):
      self._fail("module call pairing failed after an interrupted call")

  def _begin(self, module: nn.Module, args, kwargs) -> None:
    """Pre-hook to register module invocation and scope."""
    del args, kwargs  # Unused.
    try:
      self._guard(module)
      if module is self._model:
        if self._calls:
          self._fail("recursive root calls are not supported")
        self._verify_all()
        forward_id = self._next_forward
        self._next_forward += 1
      elif self._calls:
        forward_id = self._calls[0].forward_id
      else:
        return  # A direct child call outside the root is outside scope.
      frame, normal = _invocation(module)
      if not normal:
        self._fail("module entry is outside the supported eager path")
      enclosing = self._calls[-1].scope if self._calls else ()
      scope = enclosing + ((self._paths[id(module)], self._next_call),)
      self._calls.append(
          _Call(module, frame, forward_id, self._next_call, scope)
      )
      self._next_call += 1
    except BaseException:
      self._mark_failed("module entry failed")
      raise

  def _current_call(self, module: nn.Module, frame: types.FrameType) -> _Call:
    if (
        not self._calls
        or self._calls[-1].module is not module
        or self._calls[-1].frame is not frame
    ):
      self._fail("module input/output call pairing is inconsistent")
    return self._calls[-1]

  def _record(
      self,
      call: _Call,
      edge: Literal["in", "out"],
      value,
      path: tuple[Any, ...] | None = None,
      *,
      op: str | None = None,
      call_id: int | None = None,
  ) -> None:
    """Snapshot a value under a path, or tensors that were already walked."""
    assert self._snapshots is not None
    # Snapshot helpers must not reenter this collector, lower collectors, or
    # a third-party mode that could change clone/detach semantics. The mode
    # stack itself is unchanged and all model operations still use it.
    # pylint: disable=protected-access
    with torch._C.DisableTorchFunction(), torch._C._DisableTorchDispatch():
      # pylint: enable=protected-access
      owned = (
          self._snapshots.copy(value)
          if path is None
          else self._snapshots.snapshot(value, path)
      )
      for tensor_path, tensor in owned:
        self._pending.append(
            Record(
                call.forward_id,
                self._paths[id(call.module)],
                op,
                call.call_id if call_id is None else call_id,
                call.call_id,
                edge,
                tensor_path,
                tensor,
            )
        )

  def _operation(self, func, args, kwargs):
    """Intercepts operator dispatches to record inputs and outputs."""
    snapshots = self._snapshots
    if not self._calls:
      if snapshots is None or self._failure is not None:
        return func(*args, **kwargs)
      # Between root calls, keep held parameter copies' mutation state
      # accurate without recording or identifying new copies outside root.
      token = snapshots.prepare_operation(
          func, args, kwargs, track_copies=False
      )
      try:
        output = func(*args, **kwargs)
      except BaseException:
        # Mutable operators can write some targets and then fail.
        # Invalidate even when the caller catches this outside-root error.
        snapshots.finish_operation(token, None)
        raise
      snapshots.finish_operation(token, output)
      try:
        assert self._provenance is not None
        self._provenance.outside_root(output)
      except BaseException:
        self._mark_failed("data-flow tracking failed outside a root call")
        raise
      return output
    try:
      call = self._calls[-1]
      self._guard(call.module)
      call_id = self._next_call
      self._next_call += 1
      op = str(func)
      assert snapshots is not None
      # A lower third-party mode can replace clone/cast's return value.
      # Only our own transparent modes are known to preserve these rules.
      track_copies = all(
          type(mode) is CaptureMode  # pylint: disable=unidiomatic-typecheck
          for mode in _get_current_dispatch_mode_stack()
      )
      token = snapshots.prepare_operation(
          func, args, kwargs, track_copies=track_copies
      )
      provenance = self._provenance
      assert provenance is not None
      # Shapes and sources are read before execution: in-place operators
      # change both for the very tensors they receive.
      # One traversal serves both the sources and the snapshots.
      received = [*walk(args, ("args",)), *walk(kwargs, ("kwargs",))]
      inputs, edges = provenance.inputs(call.forward_id, call_id, received)
      self._record(call, "in", received, op=op, call_id=call_id)
      output = func(*args, **kwargs)
      snapshots.finish_operation(token, output)
      returned = list(walk(output, ("output",)))
      outputs = provenance.outputs(
          call.forward_id, call_id, returned, func, args, kwargs
      )
      self._record(call, "out", returned, op=op, call_id=call_id)
      self._pending_ops.append(
          Op(
              call.forward_id,
              call_id,
              self._paths[id(call.module)],
              call.call_id,
              call.scope,
              op,
              tuple(inputs),
              tuple(outputs),
              describe_arguments(func, args, kwargs),
          )
      )
      self._pending_edges.extend(edges)
      return output
    except BaseException:
      self._mark_failed("operator capture or execution failed")
      raise

  def _inputs(self, module: nn.Module, args, kwargs) -> None:
    """Pre-hook to record module inputs."""
    try:
      self._guard(module)
      if not self._calls:
        return
      frame, _ = _invocation(module)
      call = self._current_call(module, frame)
      if call.inputs_seen:
        self._fail("module inputs were observed more than once")
      if self._observes(module):
        if module is self._model and self._selected is None:
          assert self._provenance is not None
          self._provenance.root_inputs(call.forward_id, args, kwargs)
        self._record(call, "in", args, ("args",))
        self._record(call, "in", kwargs, ("kwargs",))
      call.inputs_seen = True
    except BaseException:
      self._mark_failed("module input capture failed")
      raise

  def _output(self, module: nn.Module, args, kwargs, output) -> None:
    """Post-hook to record module outputs."""
    del args, kwargs  # Unused.
    try:
      frame, normal = _invocation(module)
      if not normal:
        # Never mutate the hook registry or throw over an original
        # exception while _call_impl is iterating emergency hooks.
        if module is self._model or self._calls:
          self._mark_failed("a model call raised an exception")
          self._calls.clear()
        return
      self._guard(module)
      if not self._calls:
        return
      call = self._current_call(module, frame)
      if not call.inputs_seen:
        self._fail("module output was observed before its inputs")
      if self._observes(module):
        self._record(call, "out", output, ("output",))
      self._calls.pop()
      if module is self._model:
        if self._selected is None:
          assert self._provenance is not None
          self._pending_outputs.extend(
              self._provenance.root_outputs(call.forward_id, output)
          )
        self._verify_all()
    except BaseException:
      self._mark_failed("module output capture failed")
      raise

  def _cleanup(self) -> BaseException | None:
    """Removes all hooks and cleans up global state."""
    first_error = None
    for handle in reversed(self._handles):
      try:
        handle.remove()
      except BaseException as error:  # pylint: disable=broad-exception-caught
        if first_error is None:
          first_error = error
    self._handles.clear()
    self._calls.clear()
    if self._mode is not None:
      try:
        if not self._mode.close():
          self._mark_failed("dispatch mode nesting changed during capture")
      except BaseException as error:  # pylint: disable=broad-exception-caught
        if first_error is None:
          first_error = error
      self._mode = None
    if self._grad is not None:
      try:
        self._grad.__exit__(None, None, None)
      except BaseException as error:  # pylint: disable=broad-exception-caught
        if first_error is None:
          first_error = error
      self._grad = None
    with _OWNERS_LOCK:
      for _, module in self._modules:
        if _OWNERS.get(id(module)) is self:
          del _OWNERS[id(module)]
    self._snapshots = None
    self._provenance = None
    self._selection = ()
    self._modules = ()
    self._paths.clear()
    self._structures.clear()
    self._hooks.clear()
    self._model = None
    return first_error


def collect(
    model: nn.Module,
    *,
    memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES,
    modules: Iterable[str | nn.Module] | None = None,
) -> Capture:
  """Observe registered modules and visible operators over root calls.

  ``modules`` selects module paths (as in ``named_modules()``) or module
  objects: only their inputs and outputs are snapshotted, on any device, and
  no operator is observed, so ``ops``, ``edges`` and ``outputs`` raise and
  ``to_model_explorer`` refuses the capture. Unknown paths raise ValueError.
  Results become readable only after the entire scope exits successfully.

  Args:
    model: The root PyTorch module to observe.
    memory_limit_bytes: Budget in bytes for snapshot data.
    modules: Optional iterable of module paths or module objects to select.

  Returns:
    A Capture context manager for the model.
  """
  return Capture(model, memory_limit_bytes=memory_limit_bytes, modules=modules)
