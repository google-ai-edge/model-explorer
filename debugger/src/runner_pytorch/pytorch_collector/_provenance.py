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

"""Operator call table and tensor-identity data flow, independent of snapshots."""

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Literal, NamedTuple

import torch
from torch import Tensor
from torch.utils.weak import WeakTensorKeyDictionary

from ._snapshots import _ordinary_cpu_tensor
from ._snapshots import Path
from ._snapshots import walk

EdgeKind = Literal["op", "input", "param", "buffer", "external", "constant"]
_TEXT_TYPES = (
    bool,
    int,
    float,
    torch.dtype,
    torch.device,
    torch.layout,
    torch.memory_format,
)
_MAX_SEQUENCE = 8


# These are created once or more per operator call inside the model's forward;
# a NamedTuple costs about a third of a frozen dataclass to build.
class Port(NamedTuple):
  path: Path
  shape: tuple[int, ...]
  dtype: torch.dtype


class Op(NamedTuple):
  forward_id: int
  call_id: int
  module: str
  module_call_id: int
  scope: tuple[tuple[str, int], ...]
  op: str
  inputs: tuple[Port, ...]
  outputs: tuple[Port, ...]
  arguments: tuple[tuple[str, str], ...]


class Edge(NamedTuple):
  """Data flow edge describing the provenance of an operator tensor input."""

  forward_id: int
  call_id: (
      int | None
  )  # None: the reader is the root forward's caller, not an operator.
  path: Path
  kind: EdgeKind
  source_forward_id: int | None = None
  source_call_id: int | None = None
  source_path: Path | None = None
  name: str | None = None
  # Where the reading forward received an earlier forward's operator output,
  # if it did.
  input_path: Path | None = None


class _Producer(NamedTuple):
  kind: Literal["op", "input", "external"]
  forward_id: int | None = None
  call_id: int | None = None
  path: Path | None = None


_EXTERNAL = _Producer("external")


def _port(path: Path, tensor: Tensor) -> Port:
  return Port(path, tuple(tensor.shape), tensor.dtype)


def _loose_tensors(value: Any) -> Iterator[Tensor]:
  """Ordinary tensors of a value outside root calls, where nothing is rejected."""
  if _ordinary_cpu_tensor(value):
    yield value
  elif isinstance(value, (tuple, list, Mapping)):
    for item in value.values() if isinstance(value, Mapping) else value:
      yield from _loose_tensors(item)


def _text(value: Any) -> str | None:
  """A bounded description of one non-tensor argument; None for tensor-bearing values."""
  if isinstance(value, Tensor):
    return None
  if value is None or type(value) in _TEXT_TYPES:
    return str(value)
  if type(value) in (tuple, list):
    if any(isinstance(item, Tensor) for item in value):
      return None
    if len(value) <= _MAX_SEQUENCE and all(
        type(item) in (bool, int) for item in value
    ):
      return str(list(value))
  return f"<{type(value).__name__}>"


def _schema_arguments(func: Any) -> tuple[tuple[int, str, bool, bool], ...]:
  """(position, name, keyword only, written) per schema argument, read once per overload."""
  cached = _SCHEMAS.get(func)
  if cached is None:
    # pylint: disable=protected-access
    cached = _SCHEMAS[func] = tuple(
        (
            index,
            argument.name,
            argument.kwarg_only,
            argument.alias_info is not None and argument.alias_info.is_write,
        )
        for index, argument in enumerate(func._schema.arguments)
    )
  return cached


_SCHEMAS: dict[Any, tuple[tuple[int, str, bool, bool], ...]] = {}


def describe_arguments(
    func: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[tuple[str, str], ...]:
  """Describes non-tensor arguments of an operator call for inspection."""
  described = []
  for index, name, keyword_only, _ in _schema_arguments(func):
    if name in kwargs:
      value = kwargs[name]
    elif not keyword_only and index < len(args):
      value = args[index]
    else:
      continue  # Schema defaults were not supplied by this call.
    text = _text(value)
    if text is not None:
      described.append((name, text))
  return tuple(described)


def _written(
    func: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> list[tuple[Path, Tensor]]:
  """Tensors bound to schema arguments the operator declares it writes."""
  written: list[tuple[Path, Tensor]] = []
  for index, name, keyword_only, is_write in _schema_arguments(func):
    if not is_write:
      continue
    if name in kwargs:
      written.extend(walk(kwargs[name], ("kwargs", name)))
    elif not keyword_only and index < len(args):
      written.extend(walk(args[index], ("args", index)))
  return written


class Provenance:
  """Attribute every operator tensor input to the call that produced that object.

  Identity is the tensor object. Weak keys neither extend activation lifetime
  nor survive the tensor, so a reused id() cannot produce a false source. The
  table lives as long as the capture: a later forward finds earlier producers.
  """

  def __init__(
      self,
      parameters: Iterable[tuple[str, Tensor]] = (),
      buffers: Iterable[tuple[str, Tensor]] = (),
  ) -> None:
    self._producers: WeakTensorKeyDictionary = WeakTensorKeyDictionary()
    self._passed: WeakTensorKeyDictionary = WeakTensorKeyDictionary()
    # The model owns these tensors for the whole scope. Holding them keeps
    # id() stable; a shared tensor keeps the first name it was registered under.
    self._registered: dict[int, tuple[Tensor, EdgeKind, str]] = {}
    for kind, named in (("param", parameters), ("buffer", buffers)):
      for name, tensor in named:
        self._registered.setdefault(id(tensor), (tensor, kind, name))

  def root_inputs(
      self, forward_id: int, args: tuple[Any, ...], kwargs: Mapping[str, Any]
  ) -> None:
    for value, prefix in ((args, ("args",)), (kwargs, ("kwargs",))):
      for path, tensor in walk(value, prefix):
        known = self._producers.get(tensor)
        # A cache tensor passed back in keeps the operator that wrote it,
        # and the path it came back under.
        self._passed[tensor] = (forward_id, path)
        if known is None or known.kind != "op":
          self._producers[tensor] = _Producer("input", forward_id, None, path)

  def root_outputs(self, forward_id: int, output: Any) -> list[Edge]:
    """The source of every tensor a root forward returned, under its output path."""
    return [
        self._edge(forward_id, None, path, tensor)
        for path, tensor in walk(output, ("output",))
    ]

  def outside_root(self, output: Any) -> None:
    for tensor in _loose_tensors(output):
      self._producers[tensor] = _EXTERNAL

  def inputs(
      self,
      forward_id: int,
      call_id: int,
      tensors: list[tuple[Path, Tensor]],
  ) -> tuple[list[Port], list[Edge]]:
    """Ports and sources of the already walked arguments of one call."""
    return (
        [_port(path, tensor) for path, tensor in tensors],
        [
            self._edge(forward_id, call_id, path, tensor)
            for path, tensor in tensors
        ],
    )

  def outputs(
      self,
      forward_id: int,
      call_id: int,
      returned: list[tuple[Path, Tensor]],
      func: Any = None,
      args: tuple[Any, ...] = (),
      kwargs: Mapping[str, Any] | None = None,
  ) -> list[Port]:
    """Makes this call the producer of what it returned and of what it wrote.

    A written argument that is not returned, such as the list of an in-place
    _foreach operator, is listed as an output under its argument path, so
    every operator source names one of its producer's output ports.

    Args:
      forward_id: The identifier of the root forward invocation.
      call_id: The identifier of the operator call.
      returned: The walked tensor outputs returned by the operator.
      func: The operator function or overload.
      args: Positional arguments passed to the operator.
      kwargs: Keyword arguments passed to the operator.

    Returns:
      The output ports produced or written by this call.
    """
    written = [] if func is None else _written(func, args, kwargs or {})
    if not written:
      for path, tensor in returned:
        self._producers[tensor] = _Producer("op", forward_id, call_id, path)
      return [_port(path, tensor) for path, tensor in returned]
    produced = {id(tensor) for _, tensor in returned}
    targets = {id(tensor) for _, tensor in written}
    ports = []
    for path, tensor in (
        *returned,
        *(item for item in written if id(item[1]) not in produced),
    ):
      ports.append(_port(path, tensor))
      producer = _Producer("op", forward_id, call_id, path)
      self._producers[tensor] = producer
      # A write through a view changes its base: later reads of the base
      # depend on this call. Other existing views of that base keep their
      # earlier producer; they are not indexed.
      # pylint: disable=protected-access
      if (
          id(tensor) in targets
          and tensor._base is not None
          and _ordinary_cpu_tensor(tensor._base)
      ):
        self._producers[tensor._base] = producer
    return ports

  def _edge(
      self, forward_id: int, call_id: int | None, path: Path, tensor: Tensor
  ) -> Edge:
    """Returns the provenance edge describing the producer of a tensor."""
    producer = self._producers.get(tensor)
    # A buffer written by an observed operator reports that write instead.
    if producer is None or producer.kind != "op":
      registered = self._registered.get(id(tensor))
      if registered is not None and registered[0] is tensor:
        return Edge(
            forward_id, call_id, path, registered[1], name=registered[2]
        )
    if producer is None:
      return Edge(forward_id, call_id, path, "constant")
    if producer.kind == "external":
      return Edge(forward_id, call_id, path, "external")
    input_path = None
    if producer.kind == "op" and producer.forward_id != forward_id:
      passed = self._passed.get(tensor)
      if passed is not None and passed[0] == forward_id:
        input_path = passed[1]
    return Edge(
        forward_id,
        call_id,
        path,
        producer.kind,
        producer.forward_id,
        producer.call_id,
        producer.path,
        input_path=input_path,
    )
