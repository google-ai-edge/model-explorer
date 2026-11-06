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

"""Value traversal and owned tensor snapshots, independent of capture records."""

from collections.abc import Iterable, Iterator, Mapping
import dataclasses
from typing import Any
import weakref

import torch
from torch import Tensor
from torch.multiprocessing.reductions import StorageWeakRef

from ._cache import cache_fields

Path = tuple[str | int, ...]


def _validate_tensor(value: Tensor, cpu_only: bool = True) -> None:
  if type(value) not in (Tensor, torch.nn.Parameter):
    raise TypeError("Tensor subclasses are not supported")
  if cpu_only and value.device.type != "cpu":
    raise ValueError("Snapshots support CPU tensors only")
  if value.layout != torch.strided or value.is_quantized or value.is_nested:
    raise TypeError("Snapshots support ordinary dense tensors only")


def walk(
    value: Any, path: Path, *, cpu_only: bool = True
) -> Iterator[tuple[Path, Tensor]]:
  """Walks every supported tensor with its path, in traversal order.

  The original tensor objects are yielded, registered parameters included;
  callers decide what to copy or skip. Scalars and configuration values
  yield nothing; unknown types, tensor kinds and cyclic containers raise.
  ``cpu_only=False`` also accepts dense tensors on other devices.

  Args:
    value: The root object to traverse.
    path: The traversal prefix path to prepend to each yielded item.
    cpu_only: If True, only accepts CPU tensors.

  Returns:
    An iterator of (path, tensor) pairs for each supported tensor in
    traversal order.
  """
  if not isinstance(path, tuple) or any(
      type(part) not in (str, int) for part in path
  ):
    raise TypeError("path must be a tuple of strings and integers")
  active_containers: set[int] = set()

  def visit(item: Any, item_path: Path) -> Iterator[tuple[Path, Tensor]]:
    if isinstance(item, Tensor):
      _validate_tensor(item, cpu_only)
      yield item_path, item
      return
    if item is None or isinstance(
        item,
        (
            bool,
            int,
            float,
            complex,
            str,
            bytes,
            torch.dtype,
            torch.device,
            torch.layout,
            torch.memory_format,
            torch.qscheme,
        ),
    ):
      return
    fields = None if type(item) in (tuple, list, dict) else cache_fields(item)
    if fields is not None or isinstance(item, (tuple, list, Mapping)):
      identity = id(item)
      if identity in active_containers:
        raise ValueError(f"Cyclic container at {item_path!r}")
      active_containers.add(identity)
      try:
        if fields is not None:
          entries = fields
        elif isinstance(item, Mapping):
          entries = item.items()
        else:
          entries = enumerate(item)
        for key, child in entries:
          if type(key) not in (str, int):
            raise TypeError(
                f"Mapping keys must be strings or integers at {item_path!r}"
            )
          yield from visit(child, item_path + (key,))
      finally:
        active_containers.remove(identity)
      return
    raise TypeError(
        f"Unsupported value of type {type(item).__name__} at {item_path!r}"
    )

  return visit(value, path)


def _byte_bounds(value: Tensor) -> tuple[int, int] | None:
  """A bounding interval, not a claim that every byte in it is occupied."""
  if value.numel() == 0:
    return None
  first = last = value.storage_offset()
  for size, stride in zip(value.shape, value.stride()):
    extent = (size - 1) * stride
    first += min(0, extent)
    last += max(0, extent)
  element_size = value.element_size()
  return first * element_size, (last + 1) * element_size


def _covers_bounds(value: Tensor) -> bool:
  """Prove a parameter's bounding interval has no unowned gaps."""
  extent = 1
  for stride, size in sorted(
      (abs(stride), size)
      for size, stride in zip(value.shape, value.stride())
      if size > 1
  ):
    if stride > extent:
      return False
    extent += (size - 1) * stride
  return True


def _view_signature(value: Tensor) -> tuple[Any, ...]:
  # A permutation of dimensions (or insertion/removal of size-one axes)
  # visits the same elements, even for a parameter with a strided footprint.
  return (
      value.dtype,
      value.storage_offset(),
      tuple(
          sorted(
              (size, stride)
              for size, stride in zip(value.shape, value.stride())
              if size != 1
          )
      ),
  )


@dataclasses.dataclass(frozen=True)
class _Region:
  bounds: tuple[int, int] | None
  covers_bounds: bool
  signature: tuple[Any, ...]


def _region(value: Tensor) -> _Region:
  return _Region(
      _byte_bounds(value), _covers_bounds(value), _view_signature(value)
  )


def _matches(value: Tensor, region: _Region) -> bool:
  if _view_signature(value) == region.signature:
    return True
  bounds = _byte_bounds(value)
  return (
      region.covers_bounds
      and bounds is not None
      and region.bounds is not None
      and region.bounds[0] <= bounds[0]
      and bounds[1] <= region.bounds[1]
  )


def _ordinary_tensor(value: Any) -> bool:
  return (
      type(value) in (Tensor, torch.nn.Parameter)
      and value.layout == torch.strided
      and not value.is_quantized
      and not value.is_nested
  )


def _ordinary_cpu_tensor(value: Any) -> bool:
  return _ordinary_tensor(value) and value.device.type == "cpu"


@dataclasses.dataclass(frozen=True)
class _ParameterRegion:
  # Keep the storage alive: an allocator must not reuse its identity while
  # this region is used to recognize aliases. _cdata identifies the backing
  # storage, unlike Tensor.data_ptr(), which includes the view's offset.
  storage: torch.UntypedStorage
  region: _Region


@dataclasses.dataclass(frozen=True)
class _DerivedRegion:
  # The C++ storage weak reference survives Python wrapper replacement and
  # detects expiration without retaining the tensor's allocation.
  storage: StorageWeakRef
  region: _Region


@dataclasses.dataclass(frozen=True)
class _WriteTarget:
  tensor: weakref.ReferenceType[Tensor]
  storage: StorageWeakRef


@dataclasses.dataclass(frozen=True)
class _OperationToken:
  parameter_copy: bool
  writes: tuple[_WriteTarget, ...]


class Snapshotter:
  """Clone supported values, excluding registered parameters and proven aliases.

  Parameter storage is identified at construction. Independent copies and
  aliases whose occupied bytes cannot be proved to belong to a parameter are
  retained. Only observed aten.clone.default and aten._to_copy.default carry
  parameter identity into new storage; schema-declared writes invalidate
  derived identity without removing registered parameter identity.

  ``cpu_only=False`` accepts parameters and values on any device and clones
  each snapshot on the device it was observed on. Copy and write tracking
  (prepare_operation, finish_operation) stays CPU only; module-only captures,
  the only users of other devices, never call it.
  """

  def __init__(
      self,
      parameters: Iterable[Tensor],
      memory_limit_bytes: int,
      *,
      cpu_only: bool = True,
  ):
    if not isinstance(memory_limit_bytes, int) or isinstance(
        memory_limit_bytes, bool
    ):
      raise TypeError("memory_limit_bytes must be an integer")
    if memory_limit_bytes < 0:
      raise ValueError("memory_limit_bytes must be non-negative")
    self._memory_limit_bytes = memory_limit_bytes
    self._cpu_only = cpu_only
    self._bytes_used = 0
    self._parameters: dict[int, Tensor] = {}
    self._parameter_regions: dict[int, list[_ParameterRegion]] = {}
    self._derived_regions: dict[int, _DerivedRegion] = {}
    self._prune_limit = 128
    for parameter in parameters:
      if not isinstance(parameter, Tensor):
        raise TypeError("parameters must contain tensors")
      _validate_tensor(parameter, cpu_only)
      if id(parameter) in self._parameters:
        continue
      self._parameters[id(parameter)] = parameter
      storage = parameter.untyped_storage()
      self._parameter_regions.setdefault(storage._cdata, []).append(
          _ParameterRegion(storage, _region(parameter))
      )

  @property
  def bytes_used(self) -> int:
    return self._bytes_used

  def clear(self) -> None:
    """Release derived identity and reset accounting; retain parameter sources."""
    self._bytes_used = 0
    self._derived_regions.clear()
    self._prune_limit = 128

  def is_parameter(self, value: Tensor) -> bool:
    """Whether the value is a registered parameter or a proven representation.

    Unsupported values return False so identity-only observation outside
    root calls does not impose the capture tensor restrictions there.

    Args:
      value: The tensor to check.

    Returns:
      True if the value represents a registered parameter or alias.
    """
    if not (
        _ordinary_cpu_tensor(value)
        if self._cpu_only
        else _ordinary_tensor(value)
    ):
      return False
    if self._parameters.get(id(value)) is value:
      return True
    storage_id = value.untyped_storage()._cdata  # pylint: disable=protected-access
    regions = self._parameter_regions.get(storage_id, ())
    for region in regions:
      if _matches(value, region.region):
        return True
    derived = self._derived_regions.get(storage_id)
    if derived is None:
      return False
    if derived.storage.expired():
      del self._derived_regions[storage_id]
      return False
    return _matches(value, derived.region)

  def prepare_operation(
      self,
      func: Any,
      args: tuple[Any, ...],
      kwargs: Mapping[str, Any],
      *,
      track_copies: bool = True,
  ) -> _OperationToken:
    """Read identity before the caller executes an ATen overload once.

    With track_copies=False, only mutation invalidation is prepared. No
    tensor operations run here; schema write targets may be tensor lists.

    Args:
      func: The ATen operator being executed.
      args: Positional arguments passed to func.
      kwargs: Keyword arguments passed to func.
      track_copies: Whether to track copy operations.

    Returns:
      An operation token containing copy and write tracking info.
    """
    parameter_copy = False
    writes: list[_WriteTarget] = []
    seen_tensors: set[int] = set()
    active_containers: set[int] = set()

    def add_write(value: Any) -> None:
      if _ordinary_cpu_tensor(value):
        if id(value) not in seen_tensors:
          seen_tensors.add(id(value))
          writes.append(
              _WriteTarget(
                  weakref.ref(value), StorageWeakRef(value.untyped_storage())
              )
          )
      elif (
          isinstance(value, (tuple, list, Mapping))
          and id(value) not in active_containers
      ):
        active_containers.add(id(value))
        try:
          for item in value.values() if isinstance(value, Mapping) else value:
            add_write(item)
        finally:
          active_containers.remove(id(value))

    copy_operation = (
        func is torch.ops.aten.clone.default
        or func is torch.ops.aten._to_copy.default  # pylint: disable=protected-access
    )
    for index, argument in enumerate(func._schema.arguments):  # pylint: disable=protected-access
      if argument.name in kwargs:
        value = kwargs[argument.name]
      elif not argument.kwarg_only and index < len(args):
        value = args[index]
      else:
        continue
      if track_copies and copy_operation and argument.name == "self":
        parameter_copy = self.is_parameter(value)
      if (
          self._derived_regions
          and argument.alias_info is not None
          and argument.alias_info.is_write
      ):
        add_write(value)
    return _OperationToken(parameter_copy, tuple(writes))

  def finish_operation(self, token: _OperationToken, output: Any) -> None:
    """Invalidate written derived storage, then register supported copy output."""
    for target in token.writes:
      self._derived_regions.pop(target.storage.cdata, None)
      # set_/resize_ can change the write target's storage. Invalidate
      # both identities conservatively, including writes through aliases.
      tensor = target.tensor()
      if _ordinary_cpu_tensor(tensor):
        self._derived_regions.pop(
            tensor.untyped_storage()._cdata, None  # pylint: disable=protected-access
        )
    if token.parameter_copy and _ordinary_cpu_tensor(output):
      storage = StorageWeakRef(output.untyped_storage())
      self._derived_regions[storage.cdata] = _DerivedRegion(
          storage, _region(output)
      )
      if len(self._derived_regions) > self._prune_limit:
        self._derived_regions = {
            key: value
            for key, value in self._derived_regions.items()
            if not value.storage.expired()
        }
        self._prune_limit = max(128, 2 * len(self._derived_regions))

  def snapshot(self, value: Any, path: Path) -> list[tuple[Path, Tensor]]:
    # walk is consumed lazily, so a memory failure still precedes the
    # validation of values that follow it in traversal order.
    return self.copy(walk(value, path, cpu_only=self._cpu_only))

  def copy(
      self, tensors: Iterable[tuple[Path, Tensor]]
  ) -> list[tuple[Path, Tensor]]:
    """Own the walked tensors that are not parameters, all or none of them."""
    snapshots: list[tuple[Path, Tensor]] = []
    initial_bytes = self._bytes_used
    try:
      for item_path, item in tensors:
        if self.is_parameter(item):
          continue
        size_bytes = item.numel() * item.element_size()
        if size_bytes > self._memory_limit_bytes - self._bytes_used:
          raise MemoryError(f"Snapshot memory limit exceeded at {item_path!r}")
        snapshots.append((item_path, item.detach().clone()))
        self._bytes_used += size_bytes
    except BaseException:
      # Nothing from this call is returned if traversal or cloning fails.
      # Previous successful snapshots remain owned by the caller.
      snapshots.clear()
      self._bytes_used = initial_bytes
      raise
    return snapshots
