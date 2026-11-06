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

"""One completed capture as a graph collection Model Explorer reads natively."""

from collections.abc import Iterable, Mapping
import functools
from typing import Any

import torch

from ._provenance import Edge
from ._provenance import Op
from ._provenance import Port
from .collector import Capture


def _path_text(path: tuple[Any, ...]) -> str:
  return ".".join(str(part) for part in path)


def _shape_text(port: Port) -> str:
  return f"{str(port.dtype).removeprefix('torch.')}{list(port.shape)}"


def _label(op: str) -> str:
  # "aten.add.Tensor" -> "aten.add": the overload is kept as a node attribute.
  return op.rsplit(".", 1)[0] if op.count(".") >= 2 else op


def _attribute(key: str, value: Any) -> dict[str, str]:
  return {"key": key, "value": str(value)}


def _output(identifier: int, port: Port) -> dict[str, Any]:
  return {
      "id": str(identifier),
      "attrs": [_attribute("tensor_shape", _shape_text(port))],
  }


def _require_operators(capture: Capture, caller: str) -> None:
  if capture.modules is not None:
    raise ValueError(
        f"{caller} needs every operator, but this capture selected"
        f" {len(capture.modules)} modules and recorded no operators; collect"
        " without modules to build a graph"
    )


@functools.cache
def _mutates(op: str) -> bool:
  """Whether the overload writes to an argument; an unknown one is assumed to."""
  try:
    namespace, name, overload = op.split(".")
    target = getattr(getattr(getattr(torch.ops, namespace), name), overload)
    return target._schema.is_mutable  # pylint: disable=protected-access
  except (ValueError, AttributeError):
    return True


def unused_ops(capture: Capture) -> dict[int, tuple[int, ...]]:
  """Per forward, the operator calls whose results reach nothing that was kept.

  Kept are the tensors a root forward returned and every call that writes to an
  argument: a write through a view is not attributed to the other views of its
  base, so such a call can matter without a reader. A call read only by a later
  forward is used when that reader is.

  Args:
    capture: The completed capture whose operators and edges to analyze.

  Returns:
    Mapping from forward ID to a tuple of unused call IDs.
  """
  _require_operators(capture, "unused_ops")
  sources: dict[int, list[int]] = {}
  for edge in capture.edges:
    if edge.kind == "op":
      sources.setdefault(edge.call_id, []).append(edge.source_call_id)
  pending = [
      edge.source_call_id for edge in capture.outputs if edge.kind == "op"
  ]
  pending += [op.call_id for op in capture.ops if _mutates(op.op)]
  used: set[int] = set()
  while pending:
    call_id = pending.pop()
    if call_id not in used:
      used.add(call_id)
      pending.extend(sources.get(call_id, ()))
  unused: dict[int, list[int]] = {}
  for op in capture.ops:
    if op.call_id not in used:
      unused.setdefault(op.forward_id, []).append(op.call_id)
  return {forward_id: tuple(calls) for forward_id, calls in unused.items()}


def _namespaces(ops: list[Op]) -> dict[int, str]:
  """Namespace of each module call: its path, numbered when it repeats.

  A module path called more than once inside the same enclosing call gets
  "#0", "#1", ... so the operators of separate calls are not merged.

  Args:
    ops: The list of captured operators.

  Returns:
    Mapping from module call ID to its formatted namespace string.
  """
  siblings: dict[tuple[int | None, str], list[int]] = {}
  for op in ops:
    enclosing = None
    for path, call_id in op.scope:
      calls = siblings.setdefault((enclosing, path), [])
      if call_id not in calls:
        calls.append(call_id)
      enclosing = call_id
  result: dict[int, str] = {}
  for op in ops:
    enclosing_id, enclosing_path, namespace = None, "", ""
    for path, call_id in op.scope:
      if call_id not in result:
        relative = (
            path[len(enclosing_path) :].lstrip(".")
            if path.startswith(enclosing_path)
            else path
        )
        segments = [
            part.replace("/", "\\/") for part in relative.split(".") if part
        ]
        calls = siblings[(enclosing_id, path)]
        if segments and len(calls) > 1:
          segments[-1] += f"#{calls.index(call_id)}"
        result[call_id] = "/".join(
            part for part in (namespace, *segments) if part
        )
      enclosing_id, enclosing_path, namespace = call_id, path, result[call_id]
  return result


def _forward_graph(
    forward_id: int,
    ops: list[Op],
    edges: list[Edge],
    outputs: list[Edge],
    all_ops: dict[int, Op],
    names: Mapping[int, str],
    hide_unused: bool,
) -> dict[str, Any]:
  """Builds a Model Explorer graph dict for a single forward pass."""
  namespaces = _namespaces(ops)
  sources: dict[str, dict[str, Any]] = (
      {}
  )  # Nodes standing for what this forward did not compute.
  nodes = []
  incoming: dict[int, list[Edge]] = {}
  for edge in edges:
    incoming.setdefault(edge.call_id, []).append(edge)

  def source_node(
      identifier: str,
      label: str,
      port: Port | None,
      attrs: list[dict[str, Any]],
  ) -> str | None:
    if port is None:
      # Only an operator that reads a tensor makes it a node: what is merely
      # passed through to the output took no part in this forward.
      return identifier if identifier in sources else None
    sources.setdefault(
        identifier,
        {
            "id": identifier,
            "label": label,
            "namespace": "inputs",
            "attrs": attrs,
            "incomingEdges": [],
            "outputsMetadata": [_output(0, port)],
        },
    )
    return identifier

  def source(edge: Edge, port: Port | None) -> tuple[str, str] | None:
    """The node and output port an edge comes from, or None when it has no node."""
    if edge.kind == "op" and edge.source_forward_id == forward_id:
      producer = all_ops[edge.source_call_id]
      return str(producer.call_id), str(
          [p.path for p in producer.outputs].index(edge.source_path)
      )
    if edge.kind in ("op", "input"):
      # What this forward did not compute is an input of its graph,
      # whoever made it.
      attrs = []
      if edge.source_forward_id != forward_id:
        attrs.append(_attribute("source_graph", names[edge.source_forward_id]))
      if edge.kind == "op":
        attrs.append(_attribute("source_node", edge.source_call_id))
        attrs.append(_attribute("source_output", _path_text(edge.source_path)))
      if edge.kind == "input" and edge.source_forward_id == forward_id:
        identifier = _path_text(edge.source_path)
      elif edge.input_path is not None:
        identifier = _path_text(edge.input_path)
      else:
        # Kept by the model between forwards rather than passed in.
        made_by = edge.source_call_id if edge.kind == "op" else "input"
        identifier = f"kept:{names[edge.source_forward_id]}:{made_by}:{_path_text(edge.source_path)}"
      found = source_node(identifier, identifier, port, attrs)
      return None if found is None else (found, "0")
    if edge.kind == "external" and port is not None:
      return (
          source_node(
              f"external:{edge.call_id}:{_path_text(edge.path)}",
              "external",
              port,
              [_attribute("source", "operator outside root calls")],
          ),
          "0",
      )
    return None

  for op in ops:
    attrs = [
        _attribute("op", op.op),
        _attribute("module", op.module or "<root>"),
        *(_attribute(name, text) for name, text in op.arguments),
    ]
    links = []
    ports = {port.path: (index, port) for index, port in enumerate(op.inputs)}
    for edge in incoming.get(op.call_id, ()):
      index, port = ports[edge.path]
      found = source(edge, port)
      if found is None:
        # Parameters, buffers and constants are facts about this input, not
        # nodes.
        value = edge.kind if edge.name is None else f"{edge.kind}: {edge.name}"
        attrs.append(
            _attribute(_path_text(edge.path), f"{value} {_shape_text(port)}")
        )
        continue
      links.append({
          "sourceNodeId": found[0],
          "sourceNodeOutputId": found[1],
          "targetNodeInputId": str(index),
      })
    nodes.append({
        "id": str(op.call_id),
        "label": _label(op.op),
        "namespace": namespaces[op.module_call_id],
        "attrs": attrs,
        "incomingEdges": links,
        "outputsMetadata": [
            _output(index, port) for index, port in enumerate(op.outputs)
        ],
    })
  returned = []
  for edge in outputs:
    found = source(edge, None)
    if found is None and hide_unused:
      continue
    identifier = _path_text(edge.path)
    attrs: list[dict[str, Any]] = []
    incoming_edges: list[dict[str, Any]] = []
    if found:
      incoming_edges.append({
          "sourceNodeId": found[0],
          "sourceNodeOutputId": found[1],
          "targetNodeInputId": "0",
      })
    else:
      source_target = (
          None
          if edge.source_path is None
          else _path_text(edge.input_path or edge.source_path)
      )
      parts = [
          str(part)
          for part in (edge.kind, edge.name, source_target)
          if part is not None
      ]
      attrs.append(_attribute("source", " ".join(parts)))

    returned.append({
        "id": identifier,
        "label": identifier,
        "namespace": "outputs",
        "attrs": attrs,
        "incomingEdges": incoming_edges,
        "outputsMetadata": [],
    })
  return {
      "id": names[forward_id],
      "nodes": [*sources.values(), *nodes, *returned],
  }


def to_model_explorer(
    capture: Capture,
    forwards: Iterable[int] | None = None,
    *,
    names: Mapping[int, str] | None = None,
    label: str = "pytorch-collector",
    hide_unused: bool = False,
) -> dict[str, Any]:
  """Return ``{"label", "graphs"}`` with one graph per root forward; nothing is written.

  ``json.dump`` the result to a ``.json`` file and open it with Model Explorer.
  ``names`` maps a forward id to the graph name the caller knows it by, such as
  ``{0: "prefill", 1: "decode_1"}``; the collector cannot tell what a forward
  means, so unnamed forwards are ``forward_N``. Each operator call is a node
  under its module path, with its ``call_id`` as id. Whatever a forward reads
  but did not compute is a node under ``inputs``, with the path it was passed
  under as id: root inputs, and operator outputs of an earlier forward such as
  a KV cache, which also name their ``source_graph`` and ``source_node``. What
  the forward returned is a node under ``outputs``, with its output path as id.
  ``hide_unused`` leaves out the calls ``unused_ops`` reports, such as an index
  that a mask function receives and ignores, and outputs that no node of the
  graph feeds, such as a cache field returned as it came; records, ops and edges
  keep them. A capture that selected modules has no operators and is refused.

  Args:
    capture: The completed capture to convert into Model Explorer graphs.
    forwards: The root forwards to include in the exported graph collection.
    names: Mapping from forward ID to custom graph name.
    label: Label for the exported graph collection.
    hide_unused: Whether to filter out unused operator calls and pass-through
      outputs.

  Returns:
    A dict structured as a Model Explorer graph collection.
  """
  _require_operators(capture, "to_model_explorer")
  all_ops = {op.call_id: op for op in capture.ops}
  available = sorted({op.forward_id for op in capture.ops})
  selected = available if forwards is None else list(forwards)
  unknown = [
      forward_id for forward_id in selected if forward_id not in available
  ]
  if unknown:
    raise ValueError(
        f"No operators were captured for forwards {unknown}; available:"
        f" {available}"
    )
  unused = sorted(set(names or {}) - set(available))
  if unused:
    raise ValueError(
        f"Names were given for forwards {unused}; available: {available}"
    )
  resolved = {
      forward_id: (names or {}).get(forward_id, f"forward_{forward_id}")
      for forward_id in available
  }
  if any(
      not isinstance(name, str) or not name or "/" in name
      for name in resolved.values()
  ):
    raise ValueError("Forward names must be non-empty strings without '/'")
  if len(set(resolved.values())) != len(resolved):
    raise ValueError(
        f"Forward names must be unique: {sorted(resolved.values())}"
    )
  hidden = set()
  if hide_unused:
    for calls in unused_ops(capture).values():
      hidden.update(calls)
  return {
      "label": label,
      "graphs": [
          _forward_graph(
              forward_id,
              [
                  op
                  for op in capture.ops
                  if op.forward_id == forward_id and op.call_id not in hidden
              ],
              [
                  edge
                  for edge in capture.edges
                  if edge.forward_id == forward_id
                  and edge.call_id not in hidden
              ],
              [
                  edge
                  for edge in capture.outputs
                  if edge.forward_id == forward_id
              ],
              all_ops,
              resolved,
              hide_unused,
          )
          for forward_id in selected
      ],
  }
