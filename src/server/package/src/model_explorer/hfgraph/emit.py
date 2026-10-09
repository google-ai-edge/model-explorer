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
# ==============================================================================

"""Stage F: LogicGraph → Google Model Explorer GraphCollection.

An op's path is its namespace.
"""

from __future__ import annotations

import collections
from typing import Any

from . import prune
from . import schema

COLORS = {
    "pruned": ("#e0e0e0", "#9e9e9e"),
    "guarded": ("#fff3b0", "#c9a400"),
    "uncovered": ("#f5f5f5", "#9e9e9e"),
    "data_dependent": ("#e1bee7", "#8e24aa"),
    "conflict": ("#ffcdd2", "#c62828"),
    "covered": ("#e8f5e9", "#66bb6a"),
}
CAT_COLOR = {
    "linear": "#dbe9ff",
    "scaled_dot_product_attention": "#ffe0b2",
    "embedding": "#c8e6c9",
}
COND_CHARS = 80


class Emitter:
  """Emits Model Explorer GraphCollection JSON from skeleton and traces.

  Attributes:
    skeleton: Skeleton representing static model structure.
    coverage: Coverage analysis across execution paths.
    iterations: Mapping of loop paths to iteration counts.
    templates: Pre-indexed function and module templates from skeleton.
    status: Mapping from region path to PathCoverage records.
  """

  def __init__(
      self,
      skeleton: schema.Skeleton,
      coverage: schema.Coverage,
      iterations: dict[str, int],
  ):
    self.skeleton = skeleton
    self.coverage = coverage
    self.iterations = iterations
    self.templates = prune.Templates(skeleton)
    self.status = {row.path: row for row in coverage.paths}
    self._render_cache: dict[str, str] = {}

  # ---- path -> namespace
  def render(self, path: str) -> str:
    """Render a region path into a human-readable Model Explorer namespace."""
    if path in self._render_cache:
      return self._render_cache[path]
    segments = path.split("/") if path else []
    rendered = []
    for i, segment in enumerate(segments):
      if "#" in segment and "@" in segment:
        region_path = "/".join(segments[: i + 1])
        branch = self.templates.branch_at(region_path)
        branch_id, label = segment.split("#")
        cond = branch.cond_src[:COND_CHARS] if branch else ""
        text = (
            f"{branch_id} {cond} ▸ {label}"
            if label != "body"
            else f"{branch_id} {cond}"
        )
        n_iterations = self.iterations.get(region_path)
        if n_iterations:
          text += f" ×{n_iterations}"
        rendered.append(text)
      else:
        rendered.append(segment)
    namespace = "/".join(rendered)
    self._render_cache[path] = namespace
    return namespace

  # ---- one graph
  def graph(
      self,
      gid: str,
      ops: list[schema.Op],
      edges: list[schema.Edge],
      shapes_by_id: dict[str, list[str]] | None = None,
  ) -> dict[str, Any]:
    """Build a Model Explorer graph dict from ops and edges."""
    nodes = []
    by_dst = collections.defaultdict(list)
    for edge in edges:
      by_dst[edge.dst].append(edge)
    for op in ops:
      namespace = self.render(op.path)
      attrs = [
          {"key": "aten", "value": op.aten},
          {"key": "path", "value": op.path},
          {"key": "source", "value": f"{op.file}:{op.line}" if op.file else ""},
      ]
      if op.loop_iter is not None:
        attrs.append({"key": "loop_iter", "value": str(op.loop_iter)})
      attrs += [{"key": key, "value": value} for key, value in op.attrs.items()]
      if shapes_by_id and op.id in shapes_by_id:
        attrs.append(
            {"key": "shapes_by_trace", "value": "; ".join(shapes_by_id[op.id])}
        )
      node = {
          "id": op.id,
          "label": op.label,
          "namespace": namespace,
          "attrs": attrs,
          "incomingEdges": [
              {
                  "sourceNodeId": edge.src,
                  "sourceNodeOutputId": str(edge.src_out),
                  "targetNodeInputId": str(edge.dst_in),
              }
              for edge in by_dst.get(op.id, [])
          ],
          "outputsMetadata": [
              {
                  "id": str(i),
                  "attrs": [{
                      "key": "tensor_shape",
                      "value": f"{tensor.dtype}{tensor.shape}",
                  }],
              }
              for i, tensor in enumerate(op.outputs)
          ],
      }
      base = op.aten.replace("aten.", "")
      if base in CAT_COLOR:
        node["style"] = {"backgroundColor": CAT_COLOR[base]}
      nodes.append(node)
    present_prefixes = set()
    for namespace in {node["namespace"] for node in nodes}:
      parts = namespace.split("/")
      for i in range(len(parts)):
        present_prefixes.add("/".join(parts[: i + 1]))
    # placeholders for every static region path that has no ops in this graph
    for row in self.coverage.paths:
      namespace = self.render(row.path)
      if namespace in present_prefixes:
        continue
      background, border = COLORS.get(row.status, ("#eeeeee", "#9e9e9e"))
      label = row.status if row.status != "covered" else "taken (no tensor ops)"
      nodes.append({
          "id": f"ph:{gid}:{row.path}",
          "label": label,
          "namespace": namespace,
          "attrs": [
              {"key": "status", "value": row.status},
              {"key": "path", "value": row.path},
              {"key": "reason", "value": row.reason},
              {"key": "needs", "value": row.needs or ""},
              {"key": "evidence", "value": row.evidence},
          ],
          "incomingEdges": [],
          "outputsMetadata": [],
          "style": {"backgroundColor": background, "borderColor": border},
      })
      present_prefixes.add(namespace)
    # group attributes for every region group, info node for single-child groups
    group_attrs = {}
    namespace_to_path = {}
    for node in nodes:
      node_path = next(
          (a["value"] for a in node["attrs"] if a["key"] == "path"),
          node["namespace"],
      )
      segments = node_path.split("/")
      for i, segment in enumerate(segments):
        if "#" in segment:
          region_path = "/".join(segments[: i + 1])
          namespace_to_path[self.render(region_path)] = region_path
    for namespace, region_path in namespace_to_path.items():
      row = self.status.get(region_path)
      branch = self.templates.branch_at(region_path)
      group_attrs[namespace] = {
          "path": region_path,
          "condition": branch.cond_src if branch else "",
          "kind": branch.cond_kind if branch else "",
          "status": row.status if row else "executed",
          "source": f"{branch.file}:{branch.line_start}" if branch else "",
          "evidence": (row.evidence if row else "")[:200],
          "needs": row.needs if row and row.needs else "",
      }
    children = collections.Counter(node["namespace"] for node in nodes)
    for namespace, n_children in list(children.items()):
      if n_children == 1 and namespace in group_attrs:
        attrs = group_attrs[namespace]
        nodes.append({
            "id": f"info:{gid}:{namespace}",
            "label": "ℹ " + attrs["condition"][:COND_CHARS],
            "namespace": namespace,
            "attrs": [
                {"key": key, "value": value} for key, value in attrs.items()
            ],
            "incomingEdges": [],
            "outputsMetadata": [],
            "style": {"backgroundColor": "#ffffff", "borderColor": "#bdbdbd"},
        })
    return {"id": gid, "nodes": nodes, "groupNodeAttributes": group_attrs}

  def emit(self, traces: list[schema.Trace], primary: str) -> dict[str, Any]:
    """Emit the full GraphCollection with merged graph and individual traces.

    Args:
      traces: List of execution traces to emit.
      primary: The trace ID considered primary.

    Returns:
      Model Explorer GraphCollection dictionary.
    """
    primary_trace = next(
        (t for t in traces if t.trace.id == primary), traces[0]
    )
    ordered = [primary_trace] + [t for t in traces if t is not primary_trace]
    merged_ops, shapes = {}, collections.defaultdict(list)
    for trace in ordered:
      for op in trace.ops:
        shapes[op.id].append(
            f"{trace.trace.id}: "
            + ", ".join(f"{t.dtype}{t.shape}" for t in op.outputs)
        )
        merged_ops.setdefault(op.id, op)
    merged_edges, seen = [], set()
    for trace in ordered:
      for edge in trace.edges:
        key = (edge.src, edge.src_out, edge.dst, edge.dst_in)
        if (
            key not in seen
            and edge.src in merged_ops
            and edge.dst in merged_ops
        ):
          seen.add(key)
          merged_edges.append(edge)
    graphs = [
        self.graph(
            "merged",
            list(merged_ops.values()),
            merged_edges,
            {op_id: s for op_id, s in shapes.items() if len(s) > 1},
        )
    ]
    for trace in traces:
      graphs.append(self.graph(trace.trace.id, trace.ops, trace.edges))
    return {
        "label": self.skeleton.meta.hf_id.replace("/", "_"),
        "graphs": graphs,
    }


def check(
    graph_collection: dict[str, Any], coverage: schema.Coverage
) -> list[str]:
  """Validate that every static region path has a group in the merged graph.

  Args:
    graph_collection: Emitted Model Explorer GraphCollection dictionary.
    coverage: Coverage report the collection was emitted from.

  Returns:
    A list of problem descriptions encountered during self-check.
  """
  problems = []
  merged_graph = graph_collection["graphs"][0]
  paths_present = set()
  for node in merged_graph["nodes"]:
    node_path = next(
        (a["value"] for a in node["attrs"] if a["key"] == "path"), None
    )
    if node_path:
      segments = node_path.split("/")
      for i in range(1, len(segments) + 1):
        paths_present.add("/".join(segments[:i]))
  placeholder_paths = {
      node["id"].split(":", 2)[2]
      for node in merged_graph["nodes"]
      if node["id"].startswith("ph:")
  }
  for row in coverage.paths:
    if row.path not in paths_present:
      problems.append(
          f"region path missing from the graph: {row.path} ({row.status})"
      )
    elif (
        row.status != "covered"
        and row.path not in placeholder_paths
        and not any(p.startswith(row.path + "/") for p in paths_present)
    ):
      problems.append(
          f"{row.status} region has neither ops nor a placeholder: {row.path}"
      )
  return problems
